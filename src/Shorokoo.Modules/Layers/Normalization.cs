using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Modules.Initializers;
using Shorokoo.Onnx;
using static Shorokoo.Globals;

namespace Shorokoo.Modules.Layers;

/// <summary>BatchNorm running-mean state (module-owned, updated via StateUpdate; initialized to 0).</summary>
[StateInitializer(Ownership = StateOwnership.ModuleOwned)]
public static partial class BatchNormRunningMeanInit
{
    public static Tensor<float32> Inline(Vector<int64> shape)
        => Globals.TensorFill(shape, 0.0f);
}

/// <summary>BatchNorm running-variance state (module-owned, updated via StateUpdate; initialized to 1).</summary>
[StateInitializer(Ownership = StateOwnership.ModuleOwned)]
public static partial class BatchNormRunningVarInit
{
    public static Tensor<float32> Inline(Vector<int64> shape)
        => Globals.TensorFill(shape, 1.0f);
}

/// <summary>
/// Rank-generic batch normalization over <c>[N, C, D1..Dn]</c> input (ranks 2–5:
/// <c>[N,C]</c>, <c>[N,C,L]</c>, <c>[N,C,H,W]</c>, <c>[N,C,D,H,W]</c>). The channel
/// is axis 1; statistics and parameters are per-channel and the rank-dependent
/// reduction set <c>{0} ∪ {2..rank-1}</c> and broadcast shape <c>[1, C, 1, …, 1]</c>
/// are derived from the input's runtime shape in-graph (the same
/// <c>Range</c>-from-rank idiom as <see cref="LayerNorm"/>), so a single module
/// covers PyTorch's per-dim <c>BatchNorm1d/2d/3d</c> contracts.
/// <list type="bullet">
///   <item>training (<c>training = true</c>): normalizes with <b>batch</b>
///         statistics (biased variance) and EMA-updates the running stats through
///         <see cref="Globals.StateUpdate"/> (ONNX momentum convention:
///         <c>running = running * momentum + batch * (1 - momentum)</c>).</item>
///   <item>eval (<c>training = false</c>): normalizes with the <b>running</b>
///         statistics when <c>trackRunningStats = true</c>, or with the eval
///         <b>batch</b> statistics when <c>trackRunningStats = false</c> (PyTorch
///         <c>track_running_stats=False</c> semantics); the StateUpdate value is
///         gated to the unchanged running stats, so eval passes leave the state
///         untouched.</item>
/// </list>
/// The affine transform <c>y = gamma * x̂ + beta</c> is applied when
/// <c>affine = true</c>; when <c>affine = false</c> the normalized <c>x̂</c> is
/// returned directly. gamma (<see cref="Ones"/>) and beta (<see cref="Zeros"/>)
/// are always created as trainable parameters (so the trainable-param struct shape
/// is independent of the bits, mirroring <see cref="Linear"/>'s always-present
/// bias); they simply receive zero gradient on the unselected branch when
/// <c>affine = false</c>. Both toggles are realized with the
/// build-both-branches-then-<c>IfElse</c>-select idiom. The running statistics are
/// module-owned state (<see cref="BatchNormRunningMeanInit"/> /
/// <see cref="BatchNormRunningVarInit"/>), which the TrainingRig threads as model
/// state, not trainable params.
/// Note: graphs containing StateUpdate links execute through the training
/// pipeline (TrainingRig), not the plain inference executor.
/// </summary>
[Module]
public partial class BatchNorm
{
    public static Tensor<float32> Inline(
        Tensor<float32> x,
        [Hyper(0.9f)] Scalar<float32> momentum,            // ONNX/Keras sense (weights the retained running stat)
        [Hyper(1e-5f)] Scalar<float32> epsilon,
        [Hyper] Scalar<bit> training,                      // true = batch stats + EMA; false = running/batch stats per trackRunningStats
        [Hyper] Scalar<bit> affine,                        // true = learnable gamma, beta; false = identity scale/shift
        [Hyper] Scalar<bit> trackRunningStats)             // true = eval uses running stats; false = eval uses batch stats
    {
        var shape = x.ShapeTensor();
        Scalar<int64> rank = shape.ShapeTensor()[0];
        Scalar<int64> numChannels = shape[1];

        var runningMean = BatchNormRunningMeanInit.Init([numChannels]).Vec();
        var runningVar = BatchNormRunningVarInit.Init([numChannels]).Vec();
        var weight = Ones.Init([numChannels]).Vec();
        var bias = Zeros.Init([numChannels]).Vec();

        var one = Scalar(1L);

        // Reduction set {0} ∪ {2..rank-1} (batch + every spatial axis, skipping the
        // channel axis 1), built in-graph from the runtime rank.
        Vector<int64> reduceAxes = [Scalar(0L)];
        reduceAxes = reduceAxes.Concat(((Tensor<int64>)OnnxOp.Range(Scalar(2L), rank, one)).Vec());

        // Per-channel broadcast shape [1, C, 1, ..., 1]: a leading 1, the channel
        // count, then (rank - 2) trailing ones (empty for rank-2 [N, C]).
        Vector<int64> paramShape = [one, numChannels];
        paramShape = paramShape.Concat(VectorFill(rank - 2L, 1L));

        // Biased batch statistics, kept at the broadcast shape [1, C, 1, ...].
        var batchMean = x.Reduce(ReduceKind.Mean, reduceAxes, keepDims: true);
        var diff = x - batchMean;
        var batchVar = (diff * diff).Reduce(ReduceKind.Mean, reduceAxes, keepDims: true);

        var rMeanBatch = batchMean;
        var rVarBatch = batchVar;
        var rMeanRunning = runningMean.Reshape(paramShape);
        var rVarRunning = runningVar.Reshape(paramShape);

        // Eval normalizer statistics: running stats when tracked, else the eval
        // batch stats (PyTorch track_running_stats=False).
        var evalMean = trackRunningStats.IfElse(rMeanRunning, rMeanBatch);
        var evalVar = trackRunningStats.IfElse(rVarRunning, rVarBatch);

        // Normalize (pre-affine x̂) for the batch (train) and eval paths, then pick
        // the path with the train/eval bit before applying the affine.
        var trainXHat = diff / (batchVar + epsilon).Sqrt();
        var evalXHat = (x - evalMean) / (evalVar + epsilon).Sqrt();
        var xHat = training.IfElse(trainXHat, evalXHat);

        // Affine apply, gated so gamma/beta are bypassed when affine = false.
        var rW = weight.Reshape(paramShape);
        var rB = bias.Reshape(paramShape);
        var y = affine.IfElse(rW * xHat + rB, xHat);

        // Running-statistics EMA update, applied only in training mode.
        var batchMeanVec = batchMean.Reshape([numChannels]).Vec();
        var batchVarVec = batchVar.Reshape([numChannels]).Vec();
        var emaMean = runningMean * momentum + batchMeanVec * (1f - momentum);
        var emaVar = runningVar * momentum + batchVarVec * (1f - momentum);
        Globals.StateUpdate(runningMean, training.IfElse(emaMean, runningMean));
        Globals.StateUpdate(runningVar, training.IfElse(emaVar, runningVar));

        return y;
    }
}

/// <summary>
/// Batch normalization over <c>[N, C, H, W]</c> (NCHW) input — a thin alias for the
/// rank-generic <see cref="BatchNorm"/> with <c>affine</c> and
/// <c>trackRunningStats</c> defaulted on, preserving the 4-argument
/// <c>(momentum, epsilon, training, x)</c> call shape. See <see cref="BatchNorm"/>
/// for the full toggle surface and semantics.
/// </summary>
[Module]
public partial class BatchNorm2d
{
    public static Tensor<float32> Inline(
        Tensor<float32> x,
        [Hyper] Scalar<float32> momentum,
        [Hyper] Scalar<float32> epsilon,
        [Hyper] Scalar<bit> training)
        => BatchNorm.Call(momentum, epsilon, training, true, true, x);
}

/// <summary>
/// Batch normalization over <c>[N, C]</c> or <c>[N, C, L]</c> input — a thin alias
/// for the rank-generic <see cref="BatchNorm"/> with <c>affine</c> and
/// <c>trackRunningStats</c> defaulted on, preserving the 4-argument
/// <c>(momentum, epsilon, training, x)</c> call shape. Both the rank-2 <c>[N, C]</c>
/// (MLP/tabular) and the rank-3 <c>[N, C, L]</c> forms are supported (the latter via
/// <see cref="BatchNorm"/>'s rank inference). See <see cref="BatchNorm"/> for the
/// full toggle surface and semantics.
/// </summary>
[Module]
public partial class BatchNorm1d
{
    public static Tensor<float32> Inline(
        Tensor<float32> x,
        [Hyper] Scalar<float32> momentum,
        [Hyper] Scalar<float32> epsilon,
        [Hyper] Scalar<bit> training)
        => BatchNorm.Call(momentum, epsilon, training, true, true, x);
}

/// <summary>
/// Batch normalization over <c>[N, C, D, H, W]</c> (NCDHW) input — a thin alias for
/// the rank-generic <see cref="BatchNorm"/> with <c>affine</c> and
/// <c>trackRunningStats</c> defaulted on, preserving the 4-argument
/// <c>(momentum, epsilon, training, x)</c> call shape. See <see cref="BatchNorm"/>
/// for the full toggle surface and semantics.
/// </summary>
[Module]
public partial class BatchNorm3d
{
    public static Tensor<float32> Inline(
        Tensor<float32> x,
        [Hyper] Scalar<float32> momentum,
        [Hyper] Scalar<float32> epsilon,
        [Hyper] Scalar<bit> training)
        => BatchNorm.Call(momentum, epsilon, training, true, true, x);
}

/// <summary>
/// Layer normalization over the last <c>normalizedDims</c> dimensions
/// (built in-graph from elementwise/reduce ops so <c>epsilon</c> can be a
/// hyperparameter; the ONNX LayerNormalization op only takes a static epsilon attribute).
/// gamma/beta are trainable, shaped like the normalized trailing dimensions
/// (<see cref="Ones"/> / <see cref="Zeros"/>), broadcast over the leading dims.
/// </summary>
[Module]
public partial class LayerNorm
{
    public static Tensor<float32> Inline(
        Tensor<float32> x,
        [Hyper] Scalar<int64> normalizedDims,
        [Hyper] Scalar<float32> epsilon)
    {
        var shape = x.ShapeTensor();
        Scalar<int64> rank = shape.ShapeTensor()[0];
        var start = rank - normalizedDims;

        // axes = [rank - normalizedDims, ..., rank - 1]
        var axes = ((Tensor<int64>)OnnxOp.Range(start, rank, Scalar(1L))).Vec();

        var mean = x.Reduce(ReduceKind.Mean, axes, keepDims: true);
        var diff = x - mean;
        var variance = (diff * diff).Reduce(ReduceKind.Mean, axes, keepDims: true);
        var xHat = diff / (variance + epsilon).Sqrt();

        // gamma/beta over the trailing normalized dims (broadcasts over leading dims).
        var paramShape = shape.Slice(start, rank);
        var weight = Ones.Init(paramShape);
        var bias = Zeros.Init(paramShape);

        return xHat * weight + bias;
    }
}

/// <summary>
/// Shared in-graph helpers for the channel-second (NCHW…) feature normalizers
/// (<see cref="GroupNorm"/> / <see cref="InstanceNorm"/>). Keeps the per-channel
/// affine-broadcast shape construction in one place so the two modules stay
/// numerically consistent.
/// </summary>
internal static class FeatureNormShapes
{
    /// <summary>
    /// Per-channel affine broadcast shape <c>[1, C, 1, …, 1]</c> sized to the
    /// runtime <paramref name="rank"/> (a leading 1, the channel count, then
    /// <c>rank - 2</c> trailing ones — empty for rank-2). This is the proven
    /// <see cref="BatchNorm"/> idiom (<c>[1, C] ++ VectorFill(rank - 2, 1)</c>),
    /// generalizing the rank-4 <c>[1, C, 1, 1]</c> reshape to arbitrary rank.
    /// </summary>
    public static Vector<int64> ChannelBroadcast(Scalar<int64> numChannels, Scalar<int64> rank)
    {
        Vector<int64> shape = [Scalar(1L), numChannels];
        return shape.Concat(VectorFill(rank - 2L, 1L));
    }
}

/// <summary>
/// Rank-generic group normalization over <c>[N, C, D1..Dn]</c> input (any rank
/// ≥ 3: <c>[N,C,L]</c>, <c>[N,C,H,W]</c>, <c>[N,C,D,H,W]</c>, …). The channel is
/// axis 1; channels are split into <c>numGroups</c> groups and each
/// (sample, group) slice — the group's channels together with <b>every</b>
/// spatial position — is normalized to zero mean / unit variance (biased
/// variance), then optionally scaled and shifted by per-channel trainable
/// gamma/beta. Built in-graph (reshape to <c>[N, G, -1]</c>, which flattens the
/// group's channels and all spatial axes into one axis so reducing axis 2 covers
/// the whole per-(sample, group) region at any rank; normalize; reshape back) so
/// <c>epsilon</c> and <c>numGroups</c> can be hypers (the ONNX
/// <c>GroupNormalization</c> op takes both as static attributes).
/// <para>
/// The affine transform <c>y = gamma * x̂ + beta</c> is applied when
/// <c>affine = true</c> (the PyTorch/Keras/Flax default); when
/// <c>affine = false</c> the normalized <c>x̂</c> is returned directly. gamma
/// (<see cref="Ones"/>) and beta (<see cref="Zeros"/>) are always created as
/// trainable parameters (so the trainable-param struct shape is independent of
/// the bit, mirroring <see cref="Linear"/>'s always-present bias) and reshaped to
/// the rank-generic broadcast shape <c>[1, C, 1, …, 1]</c>; they simply receive
/// zero gradient on the unselected branch when <c>affine = false</c>. The toggle
/// is realized with the build-both-branches-then-<c>IfElse</c>-select idiom.
/// </para>
/// <para>
/// <c>numGroups = 1</c> recovers LayerNorm-over-CHW and <c>numGroups = C</c>
/// recovers <see cref="InstanceNorm"/> (the Wu &amp; He 2018 spectrum). <c>C</c>
/// must be divisible by <c>numGroups</c>, else the <c>[N, G, -1]</c> reshape
/// fails at concretization.
/// </para>
/// </summary>
[Module]
public partial class GroupNorm
{
    public static Tensor<float32> Inline(
        Tensor<float32> x,
        [Hyper] Scalar<int64> numGroups,   // C must be divisible by numGroups
        [Hyper] Scalar<bit> affine,        // true = learnable gamma, beta; false = identity scale/shift
        [Hyper] Scalar<float32> epsilon)
    {
        var shape = x.ShapeTensor();
        Scalar<int64> rank = shape.ShapeTensor()[0];
        Scalar<int64> n = shape[0];
        Scalar<int64> c = shape[1];

        // Reshape to [N, G, -1] flattens (C/G channels)·(all spatial) into one
        // axis, so reducing axis 2 covers the whole per-(sample, group) region at
        // ANY rank.
        var xg = x.Reshape([n, numGroups, Scalar(-1L)]);
        Vector<int64> groupAxis = [Scalar(2L)];
        var mean = xg.Reduce(ReduceKind.Mean, groupAxis, keepDims: true);
        var diff = xg - mean;
        var variance = (diff * diff).Reduce(ReduceKind.Mean, groupAxis, keepDims: true);
        var xHat = (diff / (variance + epsilon).Sqrt()).Reshape(shape);

        // Per-channel gamma/beta broadcast as [1, C, 1, …, 1] sized to runtime rank.
        var affineShape = FeatureNormShapes.ChannelBroadcast(c, rank);
        var weight = Ones.Init([c]).Vec().Reshape(affineShape);
        var bias = Zeros.Init([c]).Vec().Reshape(affineShape);
        return affine.IfElse(xHat * weight + bias, xHat);
    }
}

/// <summary>
/// Rank-generic instance normalization over <c>[N, C, D1..Dn]</c> input (any rank
/// ≥ 3: <c>[N,C,L]</c>, <c>[N,C,H,W]</c>, <c>[N,C,D,H,W]</c>, …). The channel is
/// axis 1; each (sample, channel) slice is normalized over <b>all</b> of its
/// spatial extent (axes <c>2..rank-1</c>, derived from the runtime rank with the
/// same <c>Range</c>-from-rank idiom as <see cref="LayerNorm"/>) to zero mean /
/// unit variance (biased variance), then optionally scaled and shifted by
/// per-channel trainable gamma/beta. Built in-graph so <c>epsilon</c> can be a
/// hyperparameter (the ONNX <c>InstanceNormalization</c> op takes a static epsilon
/// attribute and requires mandatory affine inputs).
/// <para>
/// The affine transform <c>y = gamma * x̂ + beta</c> is applied when
/// <c>affine = true</c>; when <c>affine = false</c> the normalized <c>x̂</c> is
/// returned directly. <b>Affine defaults off in the aliases</b>
/// (<see cref="InstanceNorm1d"/>/<see cref="InstanceNorm2d"/>/<see cref="InstanceNorm3d"/>),
/// matching PyTorch's <c>affine=False</c> InstanceNorm default — the opposite of
/// <see cref="GroupNorm"/>'s affine-on default — because the canonical
/// (style-transfer) use case normalizes without a learnable affine. gamma
/// (<see cref="Ones"/>) and beta (<see cref="Zeros"/>) are always created as
/// trainable parameters and reshaped to the rank-generic broadcast shape
/// <c>[1, C, 1, …, 1]</c>; they receive zero gradient on the unselected branch
/// when <c>affine = false</c>. The toggle uses the
/// build-both-branches-then-<c>IfElse</c>-select idiom.
/// </para>
/// <para>
/// InstanceNorm carries <b>no</b> running-stats / momentum machinery: its
/// statistics are per-instance and identical at train and eval time, so it runs
/// on the plain inference pipeline. Batch/running-stat normalization is
/// <see cref="BatchNorm"/>'s job.
/// </para>
/// </summary>
[Module]
public partial class InstanceNorm
{
    public static Tensor<float32> Inline(
        Tensor<float32> x,
        [Hyper] Scalar<bit> affine,        // true = learnable gamma, beta; false = identity scale/shift
        [Hyper] Scalar<float32> epsilon)
    {
        var shape = x.ShapeTensor();
        Scalar<int64> rank = shape.ShapeTensor()[0];
        Scalar<int64> numChannels = shape[1];

        // spatial axes = [2 .. rank)  (per-(sample, channel) over all spatial dims).
        var spatialAxes = ((Tensor<int64>)OnnxOp.Range(Scalar(2L), rank, Scalar(1L))).Vec();

        var mean = x.Reduce(ReduceKind.Mean, spatialAxes, keepDims: true);
        var diff = x - mean;
        var variance = (diff * diff).Reduce(ReduceKind.Mean, spatialAxes, keepDims: true);
        var xHat = diff / (variance + epsilon).Sqrt();

        // Per-channel gamma/beta broadcast as [1, C, 1, …, 1] sized to runtime rank.
        var affineShape = FeatureNormShapes.ChannelBroadcast(numChannels, rank);
        var weight = Ones.Init([numChannels]).Vec().Reshape(affineShape);
        var bias = Zeros.Init([numChannels]).Vec().Reshape(affineShape);
        return affine.IfElse(xHat * weight + bias, xHat);
    }
}

/// <summary>
/// Instance normalization over <c>[N, C, L]</c> input — a thin alias for the
/// rank-generic <see cref="InstanceNorm"/> with <c>affine</c> defaulted <b>off</b>
/// (PyTorch's <c>InstanceNorm1d</c> default), preserving the 2-argument
/// <c>(epsilon, x)</c> call shape. See <see cref="InstanceNorm"/> for the affine
/// toggle and semantics; pass <c>affine = true</c> to the generic module for a
/// learnable affine.
/// </summary>
[Module]
public partial class InstanceNorm1d
{
    public static Tensor<float32> Inline(
        Tensor<float32> x,
        [Hyper] Scalar<float32> epsilon)
        => InstanceNorm.Call(false, epsilon, x);
}

/// <summary>
/// Instance normalization over <c>[N, C, H, W]</c> (NCHW) input — a thin alias for
/// the rank-generic <see cref="InstanceNorm"/> with <c>affine</c> defaulted
/// <b>off</b> (PyTorch's <c>InstanceNorm2d</c> default), preserving the
/// 2-argument <c>(epsilon, x)</c> call shape. <b>Note:</b> this changes the
/// previous rank-4-only <c>InstanceNorm2d</c>'s behavior, which hardcoded the
/// affine <b>on</b>; the affine is now off by default (a deliberate PyTorch-parity
/// fix). Pass <c>affine = true</c> to the generic <see cref="InstanceNorm"/> for a
/// learnable affine.
/// </summary>
[Module]
public partial class InstanceNorm2d
{
    public static Tensor<float32> Inline(
        Tensor<float32> x,
        [Hyper] Scalar<float32> epsilon)
        => InstanceNorm.Call(false, epsilon, x);
}

/// <summary>
/// Instance normalization over <c>[N, C, D, H, W]</c> (NCDHW) input — a thin alias
/// for the rank-generic <see cref="InstanceNorm"/> with <c>affine</c> defaulted
/// <b>off</b> (PyTorch's <c>InstanceNorm3d</c> default), preserving the
/// 2-argument <c>(epsilon, x)</c> call shape. See <see cref="InstanceNorm"/> for
/// the affine toggle and semantics; pass <c>affine = true</c> to the generic
/// module for a learnable affine.
/// </summary>
[Module]
public partial class InstanceNorm3d
{
    public static Tensor<float32> Inline(
        Tensor<float32> x,
        [Hyper] Scalar<float32> epsilon)
        => InstanceNorm.Call(false, epsilon, x);
}

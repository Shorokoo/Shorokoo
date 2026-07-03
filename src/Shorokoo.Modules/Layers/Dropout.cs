using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using static Shorokoo.Globals;

namespace Shorokoo.Modules.Layers;

/// <summary>
/// Builds Dropout keep/scale masks from a plain uniform draw (<see cref="Globals.RandomUniform"/>)
/// instead of the ONNX <c>Dropout</c> op. The draw site is id-bearing, so when an
/// <see cref="RngConfig"/> is bound at concretization the mask rides the keyed deterministic
/// RNG — its stream key folds from the runtime master along this site's ModelId path, giving
/// every dropout layer instance its own decorrelated, reproducible, backend-independent
/// stream. Without a bound config the draw takes the ONNX random-op fallback (fresh values
/// per execution, non-reproducible). One draw per mask element; survivors scale to
/// <c>1/(1−ratio)</c>, drops to 0; eval (<c>training = false</c>) is all-ones (identity).
/// </summary>
internal static class DropoutMasking
{
    public static Tensor<float32> Mask(Vector<int64> maskShape, Scalar<float32> ratio, Scalar<bit> training)
    {
        // Per-step mask freshness needs nothing here: the RNG system owns the drawBase.
        // Concretization injects one model-global execution counter (state, +1 per step
        // under the training rig; frozen at 0 in one-shot inference) and wires it into
        // every feed — see FastInjectRngDrawCounter.
        //
        // Seed 42 is honored only by the unkeyed ONNX fallback (RandomUniformLike), keeping
        // that path deterministic per session — the pre-keyed-RNG behavior. The keyed path
        // derives its stream from the RngConfig masters and ignores this attribute.
        var u = RandomUniform(maskShape, low: 0f, high: 1f, seed: 42f);

        var keep = (u >= ratio).Cast<float32>();                        // 1 if kept (prob 1−ratio), else 0
        var scaled = keep / (Scalar(1f) - ratio);                       // survivors → 1/(1−ratio)
        return training.IfElse(scaled, TensorFill(maskShape, 1.0f));    // eval → all ones (identity)
    }
}

/// <summary>
/// Dropout layer wrapping the ONNX Dropout op, whose <c>ratio</c> and
/// <c>training_mode</c> are runtime tensor inputs — so both map directly onto
/// hyperparameters. In training mode each element is zeroed with probability
/// <c>ratio</c> and survivors are scaled by <c>1/(1-ratio)</c>;
/// in eval mode the layer is the identity. The mask seed is fixed (42) so
/// builds are deterministic. The gradient path through a wired training flag
/// uses the forward mask (plumbed automatically by autograd).
/// </summary>
[Module]
public partial class Dropout
{
    public static Tensor<float32> Inline(
        Tensor<float32> x,
        [Hyper] Scalar<float32> ratio,
        [Hyper] Scalar<bit> training)
    {
        return x * DropoutMasking.Mask(x.ShapeTensor(), ratio, training);
    }
}

/// <summary>
/// Channel-wise (spatial) dropout — PyTorch <c>nn.Dropout1d/2d/3d</c> / Keras
/// <c>SpatialDropout1D/2D/3D</c> (Tompson et al. 2015) — over <c>[N, C, D1..Dn]</c>
/// input (channel = axis 1, any rank ≥ 2). Unlike the elementwise
/// <see cref="Dropout"/>, a single Bernoulli draw is taken per <c>(sample, channel)</c>
/// pair and broadcast over <b>every</b> spatial position, so in training mode an
/// entire feature map <c>x[n, c, …]</c> is zeroed or rescaled <i>as a unit</i>
/// (survivors scaled by <c>1/(1-ratio)</c>, inverted dropout); in eval mode
/// (<c>training = false</c>) the layer is the exact identity. The mask seed is fixed
/// (42) so builds are deterministic. Dropping whole channels (rather than scattered
/// elements) actually regularizes strongly-correlated conv feature maps, which plain
/// dropout fails to do. The rank is read in-graph, so one module covers the 1-D
/// (<c>[N,C,L]</c>), 2-D (<c>[N,C,H,W]</c>) and 3-D (<c>[N,C,D,H,W]</c>) cases; the
/// thin <see cref="Dropout1d"/> / <see cref="Dropout2d"/> / <see cref="Dropout3d"/>
/// aliases name the per-rank forms. On rank-2 <c>[N, C]</c> (no spatial axis) this
/// degenerates to elementwise <see cref="Dropout"/>, which is the correct behavior.
/// </summary>
[Module]
public partial class SpatialDropout
{
    public static Tensor<float32> Inline(
        Tensor<float32> x,                 // [N, C, D1..Dn] (channel = axis 1)
        [Hyper] Scalar<float32> ratio,     // p — fraction of CHANNELS dropped
        [Hyper] Scalar<bit> training)      // true = drop+rescale; false = identity
    {
        var shape = x.ShapeTensor();
        Scalar<int64> rank = shape.ShapeTensor()[0];
        Scalar<int64> n = shape[0];
        Scalar<int64> c = shape[1];

        // Channel-broadcast mask shape [N, C, 1, …, 1] (N, C, then rank-2 ones;
        // the ones-run is empty for rank-2 [N, C]). Same [.. , C] ++ VectorFill
        // idiom as BatchNorm's broadcast shape, but per-sample (leading N, not 1).
        Vector<int64> maskShape = [n, c];
        maskShape = maskShape.Concat(VectorFill(rank - 2L, 1L));

        // Mask of that shape (survivors -> 1/(1-ratio), drops -> 0), one draw per (sample,
        // channel); eval (training=false) => identity => all ones. Broadcast over spatial dims.
        return x * DropoutMasking.Mask(maskShape, ratio, training);
    }
}

/// <summary>
/// Channel-wise dropout over <c>[N, C, L]</c> (1-D) input — a thin alias for the
/// rank-generic <see cref="SpatialDropout"/> (PyTorch's <c>nn.Dropout1d</c>). See
/// <see cref="SpatialDropout"/> for the full semantics.
/// </summary>
[Module]
public partial class Dropout1d
{
    public static Tensor<float32> Inline(
        Tensor<float32> x,
        [Hyper] Scalar<float32> ratio,
        [Hyper] Scalar<bit> training)
        => SpatialDropout.Call(ratio, training, x);
}

/// <summary>
/// Channel-wise dropout over <c>[N, C, H, W]</c> (2-D) input — a thin alias for the
/// rank-generic <see cref="SpatialDropout"/> (PyTorch's <c>nn.Dropout2d</c>). See
/// <see cref="SpatialDropout"/> for the full semantics.
/// </summary>
[Module]
public partial class Dropout2d
{
    public static Tensor<float32> Inline(
        Tensor<float32> x,
        [Hyper] Scalar<float32> ratio,
        [Hyper] Scalar<bit> training)
        => SpatialDropout.Call(ratio, training, x);
}

/// <summary>
/// Channel-wise dropout over <c>[N, C, D, H, W]</c> (3-D) input — a thin alias for the
/// rank-generic <see cref="SpatialDropout"/> (PyTorch's <c>nn.Dropout3d</c>). See
/// <see cref="SpatialDropout"/> for the full semantics.
/// </summary>
[Module]
public partial class Dropout3d
{
    public static Tensor<float32> Inline(
        Tensor<float32> x,
        [Hyper] Scalar<float32> ratio,
        [Hyper] Scalar<bit> training)
        => SpatialDropout.Call(ratio, training, x);
}

/// <summary>
/// SELU-paired dropout — PyTorch <c>nn.AlphaDropout</c> (Klambauer et al. 2017,
/// "Self-Normalizing Neural Networks"). In training mode each element is, with
/// probability <c>ratio</c>, set to SELU's negative saturation value
/// <c>α' = −λα ≈ −1.7581</c> (<b>not</b> zeroed), then the whole tensor is
/// affine-renormalized <c>out = a·x' + b</c> with <c>a = (q + α'²·q·p)^(−1/2)</c> and
/// <c>b = −a·p·α'</c> (keep prob <c>q = 1−ratio</c>), so that a zero-mean / unit-variance
/// input keeps its mean and variance — the self-normalizing property that plain dropout
/// (which zeros, shifting the moments off the SELU support) destroys. The moment
/// preservation holds <b>in expectation</b> over the mask (a population property), not as
/// a per-realization identity. In eval mode (<c>training = false</c>) the layer is the
/// <b>exact</b> identity — gated explicitly, because the affine is not the identity and the
/// eval mask returns all ones. The mask seed is fixed (42) so builds are deterministic;
/// <c>α'</c> is a baked literal while <c>a</c>, <c>b</c> are computed in-graph from the
/// runtime <c>ratio</c>.
/// </summary>
[Module]
public partial class AlphaDropout
{
    public static Tensor<float32> Inline(
        Tensor<float32> x,
        [Hyper] Scalar<float32> ratio,     // p — drop probability
        [Hyper] Scalar<bit> training)      // true = alpha-dropout; false = identity
    {
        // Baked SELU negative-saturation value α' = −λα (PyTorch's fused constant).
        Scalar<float32> alphaP = Scalar(-1.7580993408473766f);
        Scalar<float32> q = Scalar(1f) - ratio;                       // keep prob 1 − p

        // Affine renorm constants (computed in-graph from the runtime ratio):
        //   a = (q + α'²·q·p)^(−1/2),  b = −a·(p·α')
        Scalar<float32> a = (q + alphaP * alphaP * q * ratio).Sqrt().Reciprocal();
        Scalar<float32> b = -a * (ratio * alphaP);

        // RAW 0/1 keep mask, elementwise (full input shape). The in-graph mask rescales
        // survivors to 1/(1−p) and drops to 0; multiplying back by q recovers d ∈ {0,1}.
        // (Eval: r = 1 everywhere ⇒ d = q, unused — the training gate restores identity.)
        var r = DropoutMasking.Mask(x.ShapeTensor(), ratio, training);
        Tensor<float32> d = r * q;

        // Kept → x, dropped → α'; then the affine renorm a·x' + b.
        Tensor<float32> xPrime = x * d + alphaP * (1f - d);
        Tensor<float32> affineOut = xPrime * a + b;

        // Eval ⇒ exact identity (the eval mask is all ones, so the affine is NOT the
        // identity); train ⇒ the affine output.
        return training.IfElse(affineOut, x);
    }
}

/// <summary>
/// Channel-wise SELU-paired dropout — PyTorch <c>nn.FeatureAlphaDropout</c>
/// (Klambauer et al. 2017 + Tompson et al. 2015) — over <c>[N, C, D1..Dn]</c> input
/// (channel = axis 1, any rank ≥ 2). Like <see cref="AlphaDropout"/> it drops to SELU's
/// negative saturation value <c>α' = −λα ≈ −1.7581</c> and applies the <b>same</b> affine
/// renorm (<c>a</c>, <c>b</c> as in <see cref="AlphaDropout"/>), but a single Bernoulli
/// draw is taken per <c>(sample, channel)</c> and broadcast over <b>every</b> spatial
/// position — so an entire feature map is dropped to <c>α'</c> as a unit. It reuses the
/// <c>[N, C, 1, …, 1]</c> channel-broadcast mask shape (the spatial-dropout idiom), so one
/// rank-generic module covers the 1-D / 2-D / 3-D cases; on rank-2 <c>[N, C]</c> (no
/// spatial axis) it degenerates to elementwise <see cref="AlphaDropout"/>, which is the
/// correct behavior. The moment preservation holds <b>in expectation</b> over the mask. In
/// eval mode (<c>training = false</c>) the layer is the <b>exact</b> identity (gated
/// explicitly). The mask seed is fixed (42) so builds are deterministic; <c>α'</c> is a
/// baked literal while <c>a</c>, <c>b</c> are computed in-graph from the runtime
/// <c>ratio</c>.
/// </summary>
[Module]
public partial class FeatureAlphaDropout
{
    public static Tensor<float32> Inline(
        Tensor<float32> x,                 // [N, C, D1..Dn] (channel = axis 1)
        [Hyper] Scalar<float32> ratio,     // p — fraction of CHANNELS dropped
        [Hyper] Scalar<bit> training)      // true = alpha-dropout; false = identity
    {
        // Baked SELU negative-saturation value α' = −λα (PyTorch's fused constant).
        Scalar<float32> alphaP = Scalar(-1.7580993408473766f);
        Scalar<float32> q = Scalar(1f) - ratio;                       // keep prob 1 − p

        // Affine renorm constants (computed in-graph from the runtime ratio):
        //   a = (q + α'²·q·p)^(−1/2),  b = −a·(p·α')
        Scalar<float32> a = (q + alphaP * alphaP * q * ratio).Sqrt().Reciprocal();
        Scalar<float32> b = -a * (ratio * alphaP);

        // Channel-broadcast mask shape [N, C, 1, …, 1] (N, C, then rank-2 ones; the
        // ones-run is empty for rank-2 [N, C]) — the spatial-dropout shape idiom.
        var shape = x.ShapeTensor();
        Scalar<int64> rank = shape.ShapeTensor()[0];
        Scalar<int64> n = shape[0];
        Scalar<int64> c = shape[1];
        Vector<int64> maskShape = [n, c];
        maskShape = maskShape.Concat(VectorFill(rank - 2L, 1L));

        // RAW 0/1 keep mask, one draw per (sample, channel), broadcast over the spatial
        // axes. The in-graph mask rescales survivors to 1/(1−p) and drops to 0; multiplying
        // back by q recovers d ∈ {0,1}. (Eval: r = 1 ⇒ d = q, unused — the gate restores
        // identity.)
        var r = DropoutMasking.Mask(maskShape, ratio, training);
        Tensor<float32> d = r * q;

        // Kept → x, dropped → α'; then the affine renorm a·x' + b. The [N,C,1,…,1] mask d
        // broadcasts over the spatial axes in x·d exactly as a spatial-dropout mask does.
        Tensor<float32> xPrime = x * d + alphaP * (1f - d);
        Tensor<float32> affineOut = xPrime * a + b;

        // Eval ⇒ exact identity; train ⇒ the affine output.
        return training.IfElse(affineOut, x);
    }
}

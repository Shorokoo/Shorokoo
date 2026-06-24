using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules.Initializers;
using Shorokoo.Modules.Layers;

namespace Shorokoo.Tests.Modules;

// ---------------------------------------------------------------------------
// Self-checking [Module]s for the modern normalization / parametric-activation
// layers (RMSNorm, PReLU, GatedLinear.GLU). Each returns a single Scalar<bit>
// so AutoTest.AdvancedTestGraph treats it as a self-checking graph (the bit
// must be true), keeping the xUnit tests one-liners. Mirrors the style of
// NNLibraryTestModules.cs: the gain (Ones) is deterministic at init, so the
// reference closed forms reproduce the layer output exactly.
// ---------------------------------------------------------------------------

/// <summary>RMSNorm over the last dim must produce ~unit-RMS rows (mean(y²) ≈ 1) at init (gain=1).</summary>
[Module]
public partial class RMSNormNormalizes
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var y = RMSNorm.Call(Scalar(1L), Scalar(1e-5f), x);
        Vector<int64> lastAxis = [Scalar(1L)];
        var ms = (y * y).Reduce(ReduceKind.Mean, lastAxis, keepDims: false);
        var pen = (ms - Scalar(1f)).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return pen < Scalar(1e-2f);
    }
}

/// <summary>RMSNorm must equal the manual closed form x / sqrt(mean(x²) + eps) * 1 (gain=1 at init).</summary>
[Module]
public partial class RMSNormMatchesManual
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var eps = Scalar(1e-5f);
        var y = RMSNorm.Call(Scalar(1L), eps, x);

        Vector<int64> lastAxis = [Scalar(1L)];
        var ms = (x * x).Reduce(ReduceKind.Mean, lastAxis, keepDims: true);
        var yRef = x / (ms + eps).Sqrt();

        var diff = (y - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return diff < Scalar(1e-3f) * (Scalar(1f) + yRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar());
    }
}

/// <summary>PReLU at init (alpha=0.25) must equal the Where-based closed form (x>0).Where(x, 0.25·x).</summary>
[Module]
public partial class PReLUClosedForm
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var y = PReLU.Call(x);
        var zeros = x * Scalar(0f);
        var yRef = (x > zeros).Where(x, x * Scalar(0.25f));

        var pen = (y - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return pen < Scalar(1e-5f);
    }
}

/// <summary>
/// PReLUChannelwise at init (every per-channel slope = 0.25) must equal the
/// relu-built closed form <c>relu(x) − a·relu(−x)</c> where <c>a</c> is a
/// manually-built <c>[1, C, 1, …, 1]</c> constant filled with 0.25 (matching the
/// PReLUAlphaInit default), at BOTH rank 4 (<c>[N,C,H,W]</c>) and rank 2
/// (<c>[N,C]</c>). The reference slope is built rank-generically (the BatchNorm
/// <c>[1, C] ++ VectorFill(rank−2, 1)</c> idiom) so the one module covers both
/// ranks driven from the [Fact], exercising the in-graph <c>C</c> read and the
/// <c>[C] → [1, C, 1, …]</c> broadcast. (Because the init is uniform 0.25 this
/// closed form matches the *shared* PReLU too — it proves the relu form + the
/// rank-generic broadcast, NOT that the slope is per-channel; that discrimination
/// is the rig param-shape / divergence checks in NormActTrainingCoverageTests.)
/// Uses the Within(...) ok-counting idiom: a tight relative-L1 tolerance per
/// rank, threshold &gt; (N−1) where N = 2 closed-form ok-bits (rank-4 + rank-2).
/// </summary>
[Module]
public partial class NNPReLUChannelwiseClosedForm
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, C, H, W] (rank 4)
    {
        // Rank-4 closed form against a hand-built [1, C, 1, 1] 0.25 slope.
        var rank4Pen = RelClosedFormPen(x);

        // Rank-2 [N, C] closed form: collapse the spatial dims to a [N, C] view so
        // the SAME relu-built reference (with a [1, C] 0.25 slope) is exercised at
        // rank 2 — covering the rank-generic [1, C] ++ VectorFill(rank−2, 1) tail.
        var n = x.DimTensor(0);
        Scalar<int64> c = x.ShapeTensor()[1];
        var x2 = x.Reshape([n, c, Scalar(-1L)]).Reduce(ReduceKind.Sum, [Scalar(2L)], keepDims: false);  // [N, C]
        var rank2Pen = RelClosedFormPen(x2);

        var ok = Within(rank4Pen, 1e-5f) + Within(rank2Pen, 1e-5f);
        return ok > Scalar(1L);   // both closed-form ok-bits (N − 1 = 2 − 1 = 1)
    }

    /// <summary>Relative-L1 distance between PReLUChannelwise.Call(t) and the relu-built
    /// closed form using a hand-built [1, C, 1, …, 1]-broadcast 0.25 slope (the init).</summary>
    private static Scalar<float32> RelClosedFormPen(Tensor<float32> t)
    {
        var y = PReLUChannelwise.Call(t);

        // Hand-built [1, C, 1, …, 1] slope filled with 0.25 (the PReLUAlphaInit default),
        // built rank-generically exactly like the implementation's broadcast shape.
        var shape = t.ShapeTensor();
        Scalar<int64> rank = shape.ShapeTensor()[0];
        Scalar<int64> numChannels = shape[1];
        Vector<int64> bcast = [Scalar(1L), numChannels];
        bcast = bcast.Concat(VectorFill(rank - Scalar(2L), 1L));
        var a = Globals.TensorFill(bcast, 0.25f);

        var yRef = t.Relu() - a * (t * Scalar(-1f)).Relu();

        var diff = (y - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var scale = Scalar(1f) + yRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return diff / scale;
    }

    /// <summary>1 if dist ≤ bound, else 0 (the NaN-safe ok-counting idiom from NNLibraryTestModules).</summary>
    private static Scalar<int64> Within(Scalar<float32> dist, float bound)
        => (dist <= Scalar(bound)).IfElse(Scalar(1L), Scalar(0L));
}

/// <summary>GLU over the last dim must equal a manual split's a · sigmoid(b).</summary>
[Module]
public partial class GLUMatchesManual
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var y = GatedLinear.GLU(x, axis: -1);

        var halves = x.Split(numOutputs: 2, axis: -1);
        var yRef = halves[0] * halves[1].Sigmoid();

        var pen = (y - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return pen < Scalar(1e-5f);
    }
}

/// <summary>
/// Design §7-1 (the core check): the new param-free <c>[Module] GLU</c> (baked
/// <c>dim = -1</c>) must equal an INDEPENDENT hand-split reference
/// <c>a · sigmoid(b)</c>, where <c>[a, b]</c> is <c>x</c> split in half along the
/// LAST axis. This is the sibling of <see cref="GLUMatchesManual"/> but driving
/// <see cref="GLU.Call"/> (the module forwarder + its codegen) rather than the
/// <see cref="GatedLinear.GLU"/> static helper — proving the module produces the
/// same math as the closed form and that the baked axis halves the last dim.
/// Rank-generic last-axis split: the same module is driven at multiple ranks from
/// the [Fact] (e.g. <c>[2, 6]</c> and <c>[2, 3, 4]</c>).
/// </summary>
[Module]
public partial class GLUModuleMatchesManual
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // last-axis size even
    {
        var y = GLU.Call(x);                              // new module, dim = -1 baked

        var halves = x.Split(numOutputs: 2, axis: -1);    // independent hand split
        var yRef = halves[0] * halves[1].Sigmoid();       // a · sigmoid(b)

        var pen = (y - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var scale = Scalar(1f) + yRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return pen < Scalar(1e-5f) * scale;
    }
}

/// <summary>
/// Design §7-1 (faithfulness): the <c>[Module] GLU.Call(x)</c> forwarder must equal
/// the underlying static helper <c>GatedLinear.GLU(x, -1)</c> BIT-FOR-BIT (the module
/// is a thin delegate that bakes <c>dim = -1</c>, so there is no rounding difference).
/// </summary>
[Module]
public partial class GLUModuleEqualsHelper
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // last-axis size even
    {
        var y = GLU.Call(x);                              // module (baked dim = -1)
        var yRef = GatedLinear.GLU(x, axis: -1);          // static helper (same axis)

        var pen = (y - yRef).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return pen == Scalar(0f);                         // bit-for-bit identical
    }
}

/// <summary>
/// Design §7-2 (output shape): the <c>[Module] GLU</c> halves the LAST axis. For an
/// <c>[N, …, 2H]</c> input the output is <c>[N, …, H]</c> (the leading dims are
/// preserved, the last is divided by two). Asserted rank-generically off
/// <see cref="Tensor{T}.ShapeTensor"/>: every leading dim is unchanged and the last
/// dim equals <c>inputLast / 2</c>. Driven at multiple ranks from the [Fact].
/// </summary>
[Module]
public partial class GLUModuleHalvesLastAxis
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // [N, …, 2H], last-axis even
    {
        var y = GLU.Call(x);

        var xShape = x.ShapeTensor();
        var yShape = y.ShapeTensor();
        Scalar<int64> rank = xShape.ShapeTensor()[0];
        Scalar<int64> last = rank - Scalar(1L);

        // Same rank in and out (no axis added/dropped).
        var sameRank = yShape.ShapeTensor()[0] == rank;

        // Last axis halved: y[last] == x[last] / 2.
        var lastHalved = yShape[last] == xShape[last] / Scalar(2L);

        // All leading dims (0 .. rank-2) unchanged: Σ |y[i] − x[i]| over the
        // leading prefix is zero. (Slice off the last entry of each shape vector.)
        var xLead = xShape.Slice(Scalar(0L), last);
        var yLead = yShape.Slice(Scalar(0L), last);
        var leadOk = (xLead - yLead).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar() == Scalar(0L);

        return sameRank & lastHalved & leadOk;
    }
}

// ---------------------------------------------------------------------------
// LocalResponseNorm (LRN) self-checking [Module]s. The [Module]
// LocalResponseNorm is a hand-rolled PRIMITIVE graph (Pad axis-1 +
// unrolled channel-window Slice-sum of x², then x·(k+(α/5)·sum)^(−β)) with
// size baked to 5; LRNHelper.Lrn wraps the native OnnxOp.Lrn op with an
// arbitrary size. See local-response-norm/design.md §7.
// ---------------------------------------------------------------------------

/// <summary>
/// Design §7-1 (the load-bearing parity anchor): the hand-rolled primitive
/// <c>[Module] LocalResponseNorm</c> MUST equal the native ONNX <c>LRN</c> op
/// (via <see cref="LRNHelper.Lrn{T}"/> with the matching <c>size=5</c>/params)
/// for the SAME hyperparameters. This validates the primitive forward graph
/// against ORT's <c>LRN</c> kernel — the single most important LRN check.
/// Asserted as a relative-L1 distance over a small <c>[1,5,2,2]</c> input with
/// distinct values, at the ONNX/PyTorch defaults <c>α=1e-4, β=0.75, k=1</c>.
/// </summary>
[Module]
public partial class NNLocalResponseNormMatchesOp
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // [1, 5, 2, 2]
    {
        var y = LocalResponseNorm.Call(Scalar(1e-4f), Scalar(0.75f), Scalar(1f), x);  // primitive module, size=5 baked
        var yRef = LRNHelper.Lrn(x, size: 5, alpha: 1e-4f, beta: 0.75f, k: 1f);       // native OnnxOp.Lrn

        var pen = (y - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var scale = Scalar(1f) + yRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return pen < Scalar(1e-4f) * scale;
    }
}

/// <summary>
/// Design §7-1 (closed-form value): hand-compute LRN for a <c>[1,5,2,2]</c>
/// input (5 channels, window <c>size=5</c>) from the SAME primitives the formula
/// names — <c>Pad</c> axis-1 by <c>[2,2]</c> then an unrolled channel-window
/// <c>Slice</c>-sum of <c>x²</c>, <c>pool = k + (α/5)·Σ a_{c'}²</c>,
/// <c>y = x·pool^(−β)</c> — and assert the <c>[Module] LocalResponseNorm</c>
/// output equals it. This exercises an EDGE channel (channel 0's window is
/// clamped to <c>{0,1,2}</c>) and an INTERIOR channel (channel 2 sees the full
/// <c>{0..4}</c>), confirming the actual values + the <c>α/size</c> divisor +
/// the <c>k</c>-additive (PyTorch <c>k</c> / ONNX <c>bias</c>) mapping, not just
/// module ≡ op. Non-default <c>α=2e-4, β=0.5, k=2</c> so the params bite.
/// </summary>
[Module]
public partial class NNLocalResponseNormClosedForm
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // [1, 5, 2, 2]
    {
        var a = Scalar(2e-4f); var b = Scalar(0.5f); var k = Scalar(2f);
        var y = LocalResponseNorm.Call(a, b, k, x);       // primitive module, size=5 baked

        // Hand reference: pad axis-1 by [2,2], unrolled channel Slice-sum of x² over size=5.
        var x2 = x * x;
        var padded = (Tensor<float32>)OnnxOp.Pad(x2, Vector(2L, 2L), null,
            axes: Vector(1L), mode: PadMode.Constant);
        var sum = (Tensor<float32>)OnnxOp.Slice(padded, Vector(0L), Vector(5L), Vector(1L));
        for (long i = 1; i < 5; i++)
            sum = sum + (Tensor<float32>)OnnxOp.Slice(padded, Vector(i), Vector(5L + i), Vector(1L));
        var pool = k + (a / Scalar(5f)) * sum;
        var yRef = x * (Tensor<float32>)OnnxOp.Pow(pool, -b);

        var pen = (y - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var scale = Scalar(1f) + yRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return pen < Scalar(1e-4f) * scale;
    }
}

/// <summary>
/// Design §7-3 (alpha/beta/k coverage — the hypers are LIVE, not baked): drive
/// the <c>[Module] LocalResponseNorm</c> with NON-default <c>α=1e-3, β=0.5, k=2</c>
/// and assert it equals <see cref="LRNHelper.Lrn{T}"/> (the native op) with the
/// SAME params. If <c>α/β/k</c> were silently baked to their <c>[Hyper(…)]</c>
/// defaults the two would diverge — so this confirms the float hypers reach the
/// primitive graph. Tightened relative-L1 on a <c>[1,5,2,2]</c> input.
/// </summary>
[Module]
public partial class NNLocalResponseNormHypersLive
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // [1, 5, 2, 2]
    {
        var a = Scalar(1e-3f); var b = Scalar(0.5f); var k = Scalar(2f);
        var y = LocalResponseNorm.Call(a, b, k, x);                       // module with non-default hypers
        var yRef = LRNHelper.Lrn(x, size: 5, alpha: 1e-3f, beta: 0.5f, k: 2f);  // native op, same params

        var pen = (y - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var scale = Scalar(1f) + yRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return pen < Scalar(1e-4f) * scale;
    }
}

/// <summary>
/// Design §7-3 (the arbitrary-<c>size</c> helper path the module — baked
/// <c>size=5</c> — lacks): <see cref="LRNHelper.Lrn{T}"/> with a NON-baked
/// <c>size=3</c> must match a hand reference built from the same primitives
/// (pad axis-1 by the floor/ceil split <c>[leftHalf, rightHalf] = [1, 1]</c>,
/// Slice-sum 3 channel windows, <c>pool = k + (α/3)·Σ</c>, <c>y = x·pool^(−β)</c>)
/// with non-default <c>α=3e-4, β=0.6, k=1.5</c> on a <c>[1,5,2,2]</c> input. This
/// pins that the helper exposes window widths other than the module's baked 5 and
/// the narrower (<c>{c−1,c,c+1}</c>) clamped window — distinct values from the
/// size-5 closed form, so it genuinely exercises the <c>size</c> argument.
/// <para>
/// NOTE: an even <c>size</c> (e.g. 4, the asymmetric <c>[1,2]</c> floor/ceil
/// case) is intentionally NOT tested here: although the ONNX <c>LRN</c> spec and
/// Shorokoo's own autodiff rule both handle even sizes, the ONNX Runtime CPU
/// <c>LRN</c> kernel asserts <c>size % 2 == 1</c> and refuses to even build a
/// session for an even window — an external-backend limitation (NOT a Shorokoo
/// product bug), so the arbitrary-size path is covered with an odd non-default
/// width instead.
/// </para>
/// </summary>
[Module]
public partial class NNLrnHelperArbitrarySizeClosedForm
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // [1, 5, 2, 2]
    {
        const long size = 3;
        const long leftHalf = (size - 1) / 2;        // 1
        const long rightHalf = size - 1 - leftHalf;  // 1
        var a = 3e-4f; var b = 0.6f; var k = 1.5f;

        var y = LRNHelper.Lrn(x, size: size, alpha: a, beta: b, k: k);  // native op, non-baked size=3

        // Hand reference with the floor/ceil window pads = [leftHalf, rightHalf].
        var x2 = x * x;
        var padded = (Tensor<float32>)OnnxOp.Pad(x2, Vector(leftHalf, rightHalf), null,
            axes: Vector(1L), mode: PadMode.Constant);
        const long cDim = 5L;   // C = 5 (known here)
        var sum = (Tensor<float32>)OnnxOp.Slice(padded, Vector(0L), Vector(cDim), Vector(1L));
        for (long i = 1; i < size; i++)
            sum = sum + (Tensor<float32>)OnnxOp.Slice(padded, Vector(i), Vector(cDim + i), Vector(1L));
        var pool = Scalar(k) + (Scalar(a) / Scalar((float)size)) * sum;
        var yRef = x * (Tensor<float32>)OnnxOp.Pow(pool, Scalar(-b));

        var pen = (y - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var scale = Scalar(1f) + yRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return pen < Scalar(1e-4f) * scale;
    }
}

// ---------------------------------------------------------------------------
// Training-rig models (no hypers; layer hypers fixed via Model(...) so the
// model graphs satisfy the rig's inputs-only contract). Each ends in a scalar
// reduction so it can be trained against an L2 target.
// ---------------------------------------------------------------------------

/// <summary>RMSNorm(last dim) followed by per-row mean: [N, F] → [N].</summary>
[Module]
public partial class NormActRMSNormModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var y = RMSNorm.Model(Scalar(1L), Scalar(1e-5f)).Call(input);
        Vector<int64> lastAxis = [Scalar(1L)];
        return y.Reduce(ReduceKind.Mean, lastAxis, keepDims: false);
    }
}

/// <summary>PReLU followed by per-row mean: [N, F] → [N].</summary>
[Module]
public partial class NormActPReLUModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var y = PReLU.Model().Call(input);
        Vector<int64> lastAxis = [Scalar(1L)];
        return y.Reduce(ReduceKind.Mean, lastAxis, keepDims: false);
    }
}

/// <summary>
/// Design §7-4 (param-free rig smoke): GLU is PARAM-FREE, so there is no learnable
/// slope to move — front it with a tiny trainable scalar pre-weight <c>w</c> (the
/// <see cref="NNInstanceNormAffineFalseParamModel"/> trick) so SOMETHING moves while
/// the gradient flows THROUGH the differentiable Split/Sigmoid/Mul gate. Shape:
/// <c>[N, 2H] → (scale by w) → GLU → [N, H] → per-row mean → [N]</c>. The single
/// trainable param is the upstream scalar <c>w</c>; if it moves after one TrainStep,
/// a gradient flowed back through GLU.
/// </summary>
[Module]
public partial class NormActGLUModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)   // [N, 2H]
    {
        var w = InitScalarWeight.Init(Vector(1L));
        var y = GLU.Model().Call(input * w);                      // [N, H]
        Vector<int64> lastAxis = [Scalar(1L)];
        return y.Reduce(ReduceKind.Mean, lastAxis, keepDims: false);
    }
}

/// <summary>
/// PReLUChannelwise followed by a per-row mean: <c>[N, C] → [N]</c> (channels on
/// axis 1). The single trainable param is the per-channel slope, which materializes
/// to <c>[C]</c> (C elements) — the load-bearing discriminator vs the shared PReLU's
/// <c>[1]</c> slope. Used by the rig param-shape and per-channel-divergence checks.
/// </summary>
[Module]
public partial class NormActPReLUChannelwiseModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)   // [N, C]
    {
        var y = PReLUChannelwise.Model().Call(input);
        Vector<int64> lastAxis = [Scalar(1L)];
        return y.Reduce(ReduceKind.Mean, lastAxis, keepDims: false);
    }
}

/// <summary>
/// The SHARED <see cref="PReLU"/> sibling of <see cref="NormActPReLUChannelwiseModel"/>
/// (identical wrapper: per-row mean over a <c>[N, C]</c> input). Its single trainable
/// slope materializes to <c>[1]</c> (1 element), regardless of C — the contrast that
/// proves PReLUChannelwise's slope is per-channel (<c>[C]</c>), not shared.
/// </summary>
[Module]
public partial class NormActPReLUSharedSlopeModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)   // [N, C]
    {
        var y = PReLU.Model().Call(input);
        Vector<int64> lastAxis = [Scalar(1L)];
        return y.Reduce(ReduceKind.Mean, lastAxis, keepDims: false);
    }
}

/// <summary>
/// Design §7-4 (param-free rig smoke): LocalResponseNorm is PARAM-FREE (a fixed
/// pointwise rescale), so — exactly like <see cref="NormActGLUModel"/> /
/// <see cref="NNInstanceNormAffineFalseParamModel"/> — front it with a tiny
/// trainable scalar pre-weight <c>w</c> so SOMETHING moves while the gradient
/// flows THROUGH the differentiable LRN primitive graph (Pad/Slice/Pow/Mul/Add).
/// Shape: <c>[N, C, H, W] → (scale by w) → LocalResponseNorm → [N, C, H, W] →
/// per-sample mean → [N]</c> (collapse axes 1/2/3 to a scalar per row). The single
/// trainable param is the upstream scalar <c>w</c>; if it moves after one TrainStep,
/// a finite gradient flowed back through the primitive LRN graph — confirming
/// differentiability. Layer hypers fixed to the ONNX defaults via <c>Model(…)</c>.
/// </summary>
[Module]
public partial class NormActLRNModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)   // [N, C, H, W]
    {
        var w = InitScalarWeight.Init(Vector(1L));
        var y = LocalResponseNorm.Model(Scalar(1e-4f), Scalar(0.75f), Scalar(1f)).Call(input * w);
        Vector<int64> reduceAxes = [Scalar(1L), Scalar(2L), Scalar(3L)];
        return y.Reduce(ReduceKind.Mean, reduceAxes, keepDims: false);
    }
}

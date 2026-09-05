using Shorokoo.Modules.Initializers;
using Shorokoo.Modules.Layers;

namespace Shorokoo.Tests.Modules;

// ---------------------------------------------------------------------------
// Self-checking [Module]s for the Transformer / Attention stack
// (Shorokoo.Modules.Layers.Attention). Each module returns a single
// Scalar<bit> so AutoTest.AdvancedTestGraph treats it as a self-checking graph
// (the bit must be true), keeping the xUnit tests one-liners.
//
// Value checks pin frozen forward-value goldens (self-generated at
// master-seed-0). The former hand-built MHA reference re-created the projection
// weights via second Init calls and was retired with keyed per-parameter init
// (call sites do not share weights). Weight-free properties (RoPE identities,
// causal masking) are still asserted relationally.
// ---------------------------------------------------------------------------

/// <summary>
/// ScaledDotProductAttention on the fixed [1,1,3,2] input (used as q = k = v) must match the
/// frozen reference. The old check re-ran softmax(qkᵀ/sqrt(d))·v by hand (a tautology); the
/// reference is now the op's own frozen output. Output [1,1,3,2]=6.
/// </summary>
[Module]
public partial class AttnSdpaForwardGolden
{
    public static Scalar<bit> Inline(Tensor<float32> qkv)   // [1, 1, L, d]
    {
        var y = Attention.ScaledDotProductAttention(qkv, qkv, qkv);

        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-0.06533041f, 0.47590908f, 0.07712328f, 0.2375093f, -0.17771891f, 0.43166658f);

        var diff = (y.Reshape([Scalar(-1L)]) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>
/// Causal SDPA: position 0 may attend only to key 0, so the first output row must
/// equal value[..., 0, :] exactly (the softmax over a single unmasked logit is 1).
/// q = k = v = the [1, 1, L, d] input.
/// </summary>
[Module]
public partial class AttnSdpaCausalMasksFuture
{
    public static Scalar<bit> Inline(Tensor<float32> qkv)   // [1, 1, L, d]
    {
        var y = Attention.ScaledDotProductAttention(qkv, qkv, qkv, causal: true);

        // First query row of the output and first value row, both [1, 1, 1, d].
        var firstOut = y.Slice(Vector(0L), Vector(1L), axes: Vector(2L));
        var firstVal = qkv.Slice(Vector(0L), Vector(1L), axes: Vector(2L));

        var diff = (firstOut - firstVal).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return diff < Scalar(1e-4f);
    }
}

/// <summary>
/// MultiHeadAttention self-attention (embedDim 4, numHeads 2, no bias, non-causal) on the fixed
/// [1,3,4] input at MasterSeed=0 must match the frozen reference. The old check re-ran the whole
/// projection / SDPA / recombine by hand (a tautology); the reference is now the layer's own
/// frozen forward output.
/// </summary>
[Module]
public partial class MhaForwardGolden
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // [N, L, 4]
    {
        var y = MultiHeadAttention.Model(Scalar(4L), Scalar(2L), Scalar(false), Scalar(false)).Call(x, x, x);   // [1,3,4] = 12

        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.24785332f, 0.22260496f, -0.3109769f, 0.009348283f, 0.19827217f, 0.054472778f, -0.21742551f, -0.08049695f, 0.060026057f, 0.1277156f, -0.03093447f, 0.20230557f);

        var diff = (y.Reshape([Scalar(-1L)]) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

// ---------------------------------------------------------------------------
// Self-checking [Module]s for the queryChunks memory lever. queryChunks is a
// build-time C# int, so each module loops over the counts it covers in C# and
// ANDs the per-count bits — one module, a table of cases.
// ---------------------------------------------------------------------------

/// <summary>
/// Chunking the query axis is exact: for every chunk count, causal and not, the output
/// must equal the dense path. L = 8 with counts 2/3/8/9 covers an even split, an uneven one
/// (3 -> 2+3+3), one row per chunk, and more chunks than rows (empty chunks).
/// </summary>
[Module]
public partial class AttnChunkedMatchesDense
{
    public static Scalar<bit> Inline(Tensor<float32> qkv)   // [N, H, L, d]
    {
        int[] counts = [2, 3, 8, 9];
        bool[] causals = [false, true];

        var ok = Scalar(true);
        foreach (var causal in causals)
        {
            var dense = Attention.ScaledDotProductAttention(qkv, qkv, qkv, causal: causal);
            foreach (var c in counts)
            {
                var chunked = Attention.ScaledDotProductAttention(qkv, qkv, qkv, causal: causal, queryChunks: c);
                ok = ok & ((chunked - dense).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-5f));
            }
        }
        return ok;
    }
}

/// <summary>
/// Chunking with an additiveMask must match the dense path for every mask shape the dense
/// path accepts: Lq-tall masks are sliced per chunk, ones that already broadcast over the
/// query axis (rank-1 [Lk], [1, Lk], [N, 1, 1, Lk] padding) are handed over whole.
/// </summary>
[Module]
public partial class AttnChunkedMatchesDenseWithMask
{
    public static Scalar<bit> Inline(Tensor<float32> qkv)   // [N, H, L, d]
    {
        int[] counts = [2, 3];
        var lq = qkv.DimTensor(-2);
        var lk = qkv.DimTensor(-2);
        var bias = VectorRange(0L, lk, 1L).Cast<float32>() * -0.25f;              // [Lk]

        Tensor<float32>[] masks =
        [
            Attention.CausalMask(lq, lk),                                         // [Lq, Lk]
            bias,                                                                 // [Lk]
            bias.Unsqueeze(0L),                                                   // [1, Lk]
            bias.Unsqueeze(Vector(0L, 1L, 2L)),                                   // [1, 1, 1, Lk]
            Attention.CausalMask(lq, lk).Unsqueeze(Vector(0L, 1L)),               // [1, 1, Lq, Lk]
        ];

        var ok = Scalar(true);
        foreach (var mask in masks)
        {
            var dense = Attention.ScaledDotProductAttention(qkv, qkv, qkv, additiveMask: mask);
            foreach (var c in counts)
            {
                var chunked = Attention.ScaledDotProductAttention(qkv, qkv, qkv, additiveMask: mask, queryChunks: c);
                ok = ok & ((chunked - dense).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-5f));
            }
        }
        return ok;
    }
}

/// <summary>
/// Chunking must not disturb gradients: d(sum y^2)/d(qkv) through the per-chunk Slice /
/// Concat must equal the dense gradient. A finite loss and a moved parameter would not
/// catch a dropped or misordered chunk; this does.
/// </summary>
[Module]
public partial class AttnChunkedGradientMatchesDense
{
    public static Scalar<bit> Inline(Tensor<float32> qkv)   // [N, H, L, d]
    {
        int[] counts = [2, 3];
        bool[] causals = [false, true];

        var ok = Scalar(true);
        foreach (var causal in causals)
        {
            var dense = AutoGrad(Attention.ScaledDotProductAttention(qkv, qkv, qkv, causal: causal), qkv);
            foreach (var c in counts)
            {
                var chunked = AutoGrad(
                    Attention.ScaledDotProductAttention(qkv, qkv, qkv, causal: causal, queryChunks: c), qkv);
                ok = ok & ((chunked - dense).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-4f));
            }
        }
        return ok;

        static Tensor<float32> AutoGrad(Tensor<float32> y, Tensor<float32> wrt)
            => (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(
                wrt, (y * y).Reduce(ReduceKind.Sum, keepDims: false).Scalar());
    }
}

/// <summary>
/// A single query row still matches dense when chunked, with and without a mask: every
/// chunk but one is empty, which is the degenerate end of the Slice / Concat path.
/// </summary>
[Module]
public partial class AttnChunkedSingleQueryRow
{
    public static Scalar<bit> Inline(Tensor<float32> qkv)   // [N, H, L, d]
    {
        int[] counts = [2, 4];
        var q = qkv.Slice(Vector(0L), Vector(1L), axes: Vector(-2L));             // [N, H, 1, d]
        var mask = (VectorRange(0L, qkv.DimTensor(-2), 1L).Cast<float32>() * -0.25f).Unsqueeze(0L);

        var ok = Scalar(true);
        foreach (var c in counts)
        {
            ok = ok & Matches(Attention.ScaledDotProductAttention(q, qkv, qkv, causal: true),
                              Attention.ScaledDotProductAttention(q, qkv, qkv, causal: true, queryChunks: c));
            ok = ok & Matches(Attention.ScaledDotProductAttention(q, qkv, qkv, additiveMask: mask),
                              Attention.ScaledDotProductAttention(q, qkv, qkv, additiveMask: mask, queryChunks: c));
        }
        return ok;

        static Scalar<bit> Matches(Tensor<float32> a, Tensor<float32> b)
            => (a - b).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-5f);
    }
}

/// <summary>
/// CausalMask's queryOffset shifts the rows to absolute positions: the offset-o mask of
/// Lq rows must equal rows [o, o + Lq) of the unshifted (o + Lq)-row mask. Checked for
/// o = 1 and o = 2 against an L-row query and L-column key.
/// </summary>
[Module]
public partial class AttnCausalMaskQueryOffset
{
    public static Scalar<bit> Inline(Tensor<float32> qkv)   // [N, H, L, d]
    {
        long[] offsets = [1L, 2L];
        var l = qkv.DimTensor(-2);

        var ok = Scalar(true);
        foreach (var o in offsets)
        {
            var shifted = Attention.CausalMask(l - Scalar(o), l, Scalar(o));                 // [L - o, L]
            var rows = Attention.CausalMask(l, l).Slice(Vector(o), l.Unsqueeze(), axes: Vector(0L));
            ok = ok & ((shifted - rows).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-5f));
        }
        return ok;
    }
}

// Causal attention over a [N, H, L, d] input with all three projections trainable —
// the realistic shape, in which the P @ V gradient keeps the forward softmax output
// alive as well as the recompute. A model whose value is a raw input instead makes
// attention look half as expensive as it is. Mean-pooled to [N, H, d] for a target.
// Dense and queryChunks: 4 forms, differing only in that argument.

/// <summary>Dense form: the baseline for the graph-shape pin.</summary>
[Module]
public partial class SdpaMeanPoolModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)   // [N, H, L, d]
        => AttentionTestGraphs.MeanPooledAttention(input, queryChunks: 1);
}

/// <summary>Chunked form: drives both the graph-shape pin and the training-rig smoke test.</summary>
[Module]
public partial class ChunkedSdpaMeanPoolModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)   // [N, H, L, d]
        => AttentionTestGraphs.MeanPooledAttention(input, queryChunks: 4);
}

internal static class AttentionTestGraphs
{
    internal static Tensor<float32> MeanPooledAttention(Tensor<float32> input, int queryChunks)
    {
        var d = input.DimTensor(-1);
        var q = input.MatMul(Shorokoo.Modules.Initializers.XavierUniform.Init([d, d]));
        var k = input.MatMul(Shorokoo.Modules.Initializers.XavierUniform.Init([d, d]));
        var v = input.MatMul(Shorokoo.Modules.Initializers.XavierUniform.Init([d, d]));
        var y = Attention.ScaledDotProductAttention(q, k, v, causal: true, queryChunks: queryChunks);
        Vector<int64> seqAxis = [Scalar(2L)];
        return y.Reduce(ReduceKind.Mean, seqAxis, keepDims: false);
    }
}

// ---------------------------------------------------------------------------
// Self-checking [Module]s for RoPE (Attention.ApplyRoPE). Each returns a single
// Scalar<bit> (must be true) so AutoTest.AdvancedTestGraph treats it as a
// self-checking graph. Inputs are [N, H, L, d] (d EVEN) with per-element-distinct
// values so the rotation is non-trivial.
// ---------------------------------------------------------------------------

/// <summary>
/// RoPE at sequence position 0 is the identity: mθ = 0 ⇒ cos = 1, sin = 0, so
/// ApplyRoPE(x)[..., 0, :] must equal x[..., 0, :] exactly. Slices the first
/// sequence row (axis -2) of both and asserts they match.
/// </summary>
[Module]
public partial class RoPEPositionZeroIsIdentity
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // [N, H, L, d], d EVEN
    {
        var y = Attention.ApplyRoPE(x);

        // First sequence row (axis -2 == axis 2 for rank-4 input): [N, H, 1, d].
        var firstOut = y.Slice(Vector(0L), Vector(1L), axes: Vector(2L));
        var firstIn = x.Slice(Vector(0L), Vector(1L), axes: Vector(2L));

        var diff = (firstOut - firstIn).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return diff < Scalar(1e-5f);
    }
}

/// <summary>
/// RoPE is an orthogonal rotation, so it preserves each position's vector norm:
/// ‖RoPE(x)[..., i, :]‖² == ‖x[..., i, :]‖² for every i. Reduces sum-of-squares
/// over the last axis (keepDims) and asserts element-wise equality within a
/// relative tolerance.
/// </summary>
[Module]
public partial class RoPEPreservesNorm
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // [N, H, L, d], d EVEN
    {
        var y = Attention.ApplyRoPE(x);

        Vector<int64> lastAxis = [Scalar(-1L)];
        var ssOut = (y * y).Reduce(ReduceKind.Sum, lastAxis, keepDims: true);   // [N, H, L, 1]
        var ssIn = (x * x).Reduce(ReduceKind.Sum, lastAxis, keepDims: true);    // [N, H, L, 1]

        // Element-wise relative-tolerance check, collapsed to a single bit:
        // max |ssOut - ssIn| over all positions must be small relative to ‖x‖².
        var absDiff = (ssOut - ssIn).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var refMag = ssIn.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return absDiff < Scalar(1e-3f) * (Scalar(1f) + refMag);
    }
}

/// <summary>
/// Closed-form RoPE at sequence position 1 for d = 4, at the default theta 10000 and at an
/// explicit theta 100. The half-split
/// (GPT-NeoX) layout pairs dim j with dim j + d/2, i.e. (0,2) and (1,3). The inverse
/// frequencies are θ0 = theta^0 = 1 and θ1 = theta^{-2/4} = theta^{-0.5}, so at
/// position m = 1 the angles are exactly 1 rad and theta^{-0.5} rad - 0.01 at the default
/// theta 10000, 0.1 at theta 100. With
/// x[...,1,:] = [x0, x1, x2, x3] and rotateHalf(x) = concat(-x2', x1') the output row is
/// <code>
///   [ x0·cosθ0 - x2·sinθ0,   x1·cosθ1 - x3·sinθ1,
///     x2·cosθ0 + x0·sinθ0,   x3·cosθ1 + x1·sinθ1 ]
/// </code>
/// We rebuild that row in-graph using the SAME Cos()/Sin() ops on the Scalar angle
/// constants 1f and theta1, pinning the rotate-half pairing + frequency formula + sign
/// convention exactly, and run it at both thetas so the parameter is pinned too.
/// </summary>
[Module]
public partial class RoPEClosedFormPositionOne
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // [1, 1, 2, 4]  (L>=2, d==4)
    {
        // theta 10000 gives θ1 = 0.01 rad; theta 100 gives θ1 = 100^-0.5 = 0.1 rad.
        return Matches(x, Attention.ApplyRoPE(x), 0.01f)
             & Matches(x, Attention.ApplyRoPE(x, theta: 100), 0.1f);
    }

    private static Scalar<bit> Matches(Tensor<float32> x, Tensor<float32> y, float theta1)
    {
        // Position-1 output row: [1, 1, 1, 4].
        var outRow = y.Slice(Vector(1L), Vector(2L), axes: Vector(2L));

        // Position-1 input row, then its four scalar components along the head dim.
        var inRow = x.Slice(Vector(1L), Vector(2L), axes: Vector(2L));   // [1, 1, 1, 4]
        var x0 = inRow.Slice(Vector(0L), Vector(1L), axes: Vector(-1L));
        var x1 = inRow.Slice(Vector(1L), Vector(2L), axes: Vector(-1L));
        var x2 = inRow.Slice(Vector(2L), Vector(3L), axes: Vector(-1L));
        var x3 = inRow.Slice(Vector(3L), Vector(4L), axes: Vector(-1L));

        // θ0 = 1 rad (dims 0,2), θ1 (dims 1,3); same Cos/Sin ops as the impl.
        var c0 = Scalar(1f).Cos();
        var s0 = Scalar(1f).Sin();
        var c1 = Scalar(theta1).Cos();
        var s1 = Scalar(theta1).Sin();

        var e0 = x0 * c0 - x2 * s0;
        var e1 = x1 * c1 - x3 * s1;
        var e2 = x2 * c0 + x0 * s0;
        var e3 = x3 * c1 + x1 * s1;
        var expected = e0.Concat(-1L, e1, e2, e3);   // [1, 1, 1, 4]

        var diff = (outRow - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var refMag = expected.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return diff < Scalar(1e-3f) * (Scalar(1f) + refMag);
    }
}

// ---------------------------------------------------------------------------
// Self-checking [Module]s for TransformerDecoderLayer. The decoder composes
// LayerNorm + two MultiHeadAttention sublayers + a GELU FFN; a full closed-form
// re-derivation would just re-implement that composition, so instead these run the
// layer end-to-end and compare the output against an inlined frozen golden
// reference (self-generated at master-seed-0 init). The FFN weight training path
// is covered by the TransformerDecoderLayer training-rig smoke test.
// ---------------------------------------------------------------------------

/// <summary>
/// TransformerDecoderLayer output shape: tgt [N, Lt, E] + memory [N, Lm, E] with
/// Lt != Lm must produce [N, Lt, E]. Asserts each output dim matches the expected
/// (N, Lt, E) in-graph via DimTensor.
/// </summary>
[Module]
public partial class DecoderLayerShapeCheck
{
    public static Scalar<bit> Inline(
        Tensor<float32> tgt,        // [N, Lt, E]
        Tensor<float32> memory)     // [N, Lm, E]
    {
        var y = TransformerDecoderLayer.Call(Scalar(4L), Scalar(2L), Scalar(8L), Scalar(false), tgt, memory);

        var okN = y.DimTensor(0) == tgt.DimTensor(0);
        var okL = y.DimTensor(1) == tgt.DimTensor(1);   // Lt, NOT Lm
        var okE = y.DimTensor(2) == tgt.DimTensor(2);
        return okN & okL & okE;
    }
}

/// <summary>
/// TransformerDecoderLayer (embedDim 4, numHeads 2, ffnDim 8, NO bias) runs end-to-end on the fixed
/// tgt [1,3,4] + memory [1,5,4] (Lt != Lm exercises distinct-k/v cross-attention; self-attn causal,
/// cross-attn non-causal) at MasterSeed=0 and must match the frozen reference. The former manual
/// sublayer re-derivation re-created the projection weights via second Init calls and was retired
/// with keyed per-parameter init; now a frozen forward-value golden (self-generated). Output [1,3,4]=12.
/// </summary>
[Module]
public partial class DecoderLayerNoBiasGolden
{
    public static Scalar<bit> Inline(
        Tensor<float32> tgt,        // [N, Lt, 4]
        Tensor<float32> memory)     // [N, Lm, 4]
    {
        var y = TransformerDecoderLayer.Call(Scalar(4L), Scalar(2L), Scalar(8L), Scalar(false), tgt, memory);

        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.8060598f, -0.5871042f, -0.3277368f, 1.1288095f, 1.1136469f, -0.96690774f, -0.011617601f, 1.2854031f, 0.103560954f, 1.1655895f, 1.099887f, -0.5216064f);

        var diff = (y.Reshape([Scalar(-1L)]) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>
/// Same as <see cref="DecoderLayerNoBiasGolden"/> but useBias = true (biases are Zeros-init,
/// so this exercises the useBias.IfElse(true) branch and must produce the same output — the frozen
/// reference below equals the no-bias one — Zeros-init biases are constant, hence keying-independent).
/// The former manual sublayer re-derivation was retired with keyed per-parameter init; now a
/// frozen forward-value golden (self-generated).
/// </summary>
[Module]
public partial class DecoderLayerWithBiasGolden
{
    public static Scalar<bit> Inline(
        Tensor<float32> tgt,        // [N, Lt, 4]
        Tensor<float32> memory)     // [N, Lm, 4]
    {
        var y = TransformerDecoderLayer.Call(Scalar(4L), Scalar(2L), Scalar(8L), Scalar(true), tgt, memory);

        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated; zero biases ⇒ == no-bias).
        var reference = Vector(0.8060598f, -0.5871042f, -0.3277368f, 1.1288095f, 1.1136469f, -0.96690774f, -0.011617601f, 1.2854031f, 0.103560954f, 1.1655895f, 1.099887f, -0.5216064f);

        var diff = (y.Reshape([Scalar(-1L)]) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

// ---------------------------------------------------------------------------
// Training-rig model (no hypers; layer hypers fixed via Model(...) so the model
// graph satisfies the rig's inputs-only contract). Wraps TransformerEncoderLayer
// and reduces the [N, L, E] output to a small [N, E] tensor.
// ---------------------------------------------------------------------------

/// <summary>
/// TransformerEncoderLayer (embedDim 4, numHeads 2, ffnDim 8, with bias) over a
/// [N, L, 4] input, mean-pooled over the sequence to [N, 4] for a small training target.
/// </summary>
[Module]
public partial class TransformerEncoderMeanPoolModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var y = TransformerEncoderLayer.Model(Scalar(4L), Scalar(2L), Scalar(8L), Scalar(true)).Call(input);
        Vector<int64> seqAxis = [Scalar(1L)];
        return y.Reduce(ReduceKind.Mean, seqAxis, keepDims: false);
    }
}

/// <summary>
/// TransformerDecoderLayer (embedDim 4, numHeads 2, ffnDim 8, with bias) over a
/// target [N, Lt, 4] and memory [N, Lm, 4], mean-pooled over the target sequence to
/// [N, 4]. Two graph inputs (tgt, memory) for the training-rig smoke test.
/// </summary>
[Module]
public partial class TransformerDecoderMeanPoolModel
{
    public static Tensor<float32> Inline(Tensor<float32> tgt, Tensor<float32> memory)
    {
        var y = TransformerDecoderLayer.Model(Scalar(4L), Scalar(2L), Scalar(8L), Scalar(true)).Call(tgt, memory);
        Vector<int64> seqAxis = [Scalar(1L)];
        return y.Reduce(ReduceKind.Mean, seqAxis, keepDims: false);
    }
}

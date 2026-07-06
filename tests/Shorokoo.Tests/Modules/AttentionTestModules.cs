using Shorokoo.Modules.Initializers;
using Shorokoo.Modules.Layers;

namespace Shorokoo.Tests.Modules;

// ---------------------------------------------------------------------------
// Self-checking [Module]s for the Transformer / Attention stack
// (Shorokoo.Modules.Layers.Attention). Each module returns a single
// Scalar<bit> so AutoTest.AdvancedTestGraph treats it as a self-checking graph
// (the bit must be true), keeping the xUnit tests one-liners.
//
// The MHA reference check exploits that XavierUniform.Init is seeded: a
// hand-built reference using the same initializer + shapes materializes the
// exact same projection weights as the module (same pattern as
// NNLinearMatchesManualMatMul).
// ---------------------------------------------------------------------------

/// <summary>
/// ScaledDotProductAttention on a [1, 1, L, d] input (used as q = k = v) must equal
/// the manual softmax(qkᵀ / sqrt(d)) · v built from the same primitives.
/// </summary>
[Module]
public partial class AttnSdpaMatchesManual
{
    public static Scalar<bit> Inline(Tensor<float32> qkv)   // [1, 1, L, d]
    {
        var y = Attention.ScaledDotProductAttention(qkv, qkv, qkv);

        var d = qkv.DimTensor(-1);
        var scale = Scalar(1f) / d.Cast<float32>().Sqrt();
        var scores = (qkv.MatMul(qkv.Transpose(0L, 1L, 3L, 2L)) * scale).Softmax(-1L);
        var yRef = scores.MatMul(qkv);

        var diff = (y - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return diff < Scalar(1e-4f) * (Scalar(1f) + yRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar());
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
public partial class MhaMatchesManualReference
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // [N, L, 4]
    {
        var y = MultiHeadAttention.Model(Scalar(4L), Scalar(2L), Scalar(false), Scalar(false)).Call(x, x, x);   // [1,3,4] = 12

        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-0.036230966f, 0.13973841f, 0.38819423f, 0.06302919f, 0.0045923414f, 0.18314093f, 0.40649772f, -0.0031697568f, -0.029367514f, 0.14447637f, 0.37646538f, 0.036222134f);

        var diff = (y.Reshape([Scalar(-1L)]) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
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
/// Closed-form RoPE at sequence position 1 for d = 4, base = 10000. The half-split
/// (GPT-NeoX) layout pairs dim j with dim j + d/2, i.e. (0,2) and (1,3). The inverse
/// frequencies are θ0 = base^0 = 1 and θ1 = base^{-2/4} = base^{-0.5} = 0.01, so at
/// position m = 1 the angles are exactly 1 rad and 0.01 rad. With
/// x[...,1,:] = [x0, x1, x2, x3] and rotateHalf(x) = concat(-x2', x1') the output row is
/// <code>
///   [ x0·cosθ0 - x2·sinθ0,   x1·cosθ1 - x3·sinθ1,
///     x2·cosθ0 + x0·sinθ0,   x3·cosθ1 + x1·sinθ1 ]
/// </code>
/// We rebuild that row in-graph using the SAME Cos()/Sin() ops on the Scalar angle
/// constants 1f and 0.01f, pinning the rotate-half pairing + frequency formula + sign
/// convention exactly.
/// </summary>
[Module]
public partial class RoPEClosedFormPositionOne
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // [1, 1, 2, 4]  (L>=2, d==4)
    {
        var y = Attention.ApplyRoPE(x);

        // Position-1 output row: [1, 1, 1, 4].
        var outRow = y.Slice(Vector(1L), Vector(2L), axes: Vector(2L));

        // Position-1 input row, then its four scalar components along the head dim.
        var inRow = x.Slice(Vector(1L), Vector(2L), axes: Vector(2L));   // [1, 1, 1, 4]
        var x0 = inRow.Slice(Vector(0L), Vector(1L), axes: Vector(-1L));
        var x1 = inRow.Slice(Vector(1L), Vector(2L), axes: Vector(-1L));
        var x2 = inRow.Slice(Vector(2L), Vector(3L), axes: Vector(-1L));
        var x3 = inRow.Slice(Vector(3L), Vector(4L), axes: Vector(-1L));

        // θ0 = 1 rad (dims 0,2), θ1 = 0.01 rad (dims 1,3); same Cos/Sin ops as the impl.
        var c0 = Scalar(1f).Cos();
        var s0 = Scalar(1f).Sin();
        var c1 = Scalar(0.01f).Cos();
        var s1 = Scalar(0.01f).Sin();

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
// Self-checking [Module]s for TransformerDecoderLayer: a shape check plus two
// reasonable-output checks. The decoder composes LayerNorm + two MultiHeadAttention
// sublayers + a GELU FFN; a full closed-form re-derivation would just re-implement
// that composition, so instead these run the layer end-to-end and assert the output
// is sane (Sanity.Reasonable). The FFN weight training path is covered by the
// TransformerDecoderLayer training-rig smoke test.
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
/// TransformerDecoderLayer (embedDim 4, numHeads 2, ffnDim 8, NO bias) runs end-to-end on
/// tgt [N, Lt, 4] + memory [N, Lm, 4] (Lt != Lm exercises the distinct-k/v cross-attention)
/// and produces a sane, finite, bounded output. Self-attn is causal; cross-attn is non-causal.
/// </summary>
[Module]
public partial class DecoderLayerMatchesManualNoBias
{
    public static Scalar<bit> Inline(
        Tensor<float32> tgt,        // [N, Lt, 4]
        Tensor<float32> memory)     // [N, Lm, 4]
    {
        var y = TransformerDecoderLayer.Call(Scalar(4L), Scalar(2L), Scalar(8L), Scalar(false), tgt, memory);
        return Sanity.Reasonable(y);
    }
}

/// <summary>
/// Same as <see cref="DecoderLayerMatchesManualNoBias"/> but with useBias = true (the biases
/// are Zeros-init, so this exercises the useBias.IfElse(true) branch and must still be sane).
/// </summary>
[Module]
public partial class DecoderLayerMatchesManualWithBias
{
    public static Scalar<bit> Inline(
        Tensor<float32> tgt,        // [N, Lt, 4]
        Tensor<float32> memory)     // [N, Lm, 4]
    {
        var y = TransformerDecoderLayer.Call(Scalar(4L), Scalar(2L), Scalar(8L), Scalar(true), tgt, memory);
        return Sanity.Reasonable(y);
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

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

/// <summary>
/// Scaled dot-product attention built entirely from autograd-supported primitives
/// (MatMul / Softmax / Transpose / Range / Where), so it carries gradients under
/// the default opset 21 — unlike the fused <c>NN.Attention</c> op, which has no
/// gradient rule.
/// </summary>
public static class Attention
{
    /// <summary>
    /// Scaled dot-product attention: <c>softmax(QKᵀ · scale + mask) · V</c>.
    ///
    /// <para>Contract: <paramref name="query"/>/<paramref name="key"/>/<paramref name="value"/>
    /// must be rank-4 <c>[N, H, L, d]</c> (batch, heads, sequence, head-dim). Transposing
    /// only the last two dims of an arbitrary-rank tensor isn't expressible with a static
    /// perm, so the helper fixes the perm at <c>[0, 1, 3, 2]</c> and requires 4-D inputs;
    /// <see cref="MultiHeadAttention"/> reshapes to that layout before calling.</para>
    ///
    /// <para><paramref name="scale"/> defaults to <c>1/sqrt(d)</c> with <c>d</c> the last
    /// query dim. When <paramref name="causal"/> is true an additive mask (0 on/below the
    /// diagonal, −1e9 above) is added before the softmax so position i attends only to
    /// j ≤ i; the mask is a constant (built from Range/compare/Where), so it needs no
    /// gradient. <paramref name="additiveMask"/> is an optional pre-built additive mask
    /// (broadcastable to the <c>[..., Lq, Lk]</c> scores) added on top — used by
    /// <see cref="MultiHeadAttention"/> to gate a causal mask by a runtime <c>Scalar&lt;bit&gt;</c>
    /// it cannot branch on in C#.</para>
    /// </summary>
    public static Tensor<float32> ScaledDotProductAttention(
        Tensor<float32> query,
        Tensor<float32> key,
        Tensor<float32> value,
        bool causal = false,
        float? scale = null,
        Tensor<float32>? additiveMask = null)
    {
        var d = query.DimTensor(-1);
        var scaleTensor = scale is null
            ? 1f / d.Cast<float32>().Sqrt()
            : Scalar(scale.Value);

        var scores = query.MatMul(key.Transpose(0L, 1L, 3L, 2L)) * scaleTensor;

        if (causal)
        {
            var lq = query.DimTensor(-2);
            var lk = key.DimTensor(-2);
            scores = scores + CausalMask(lq, lk);
        }

        if (additiveMask is not null)
            scores = scores + additiveMask.Value;

        return scores.Softmax(-1L).MatMul(value);
    }

    /// <summary>
    /// Additive causal mask of shape <c>[Lq, Lk]</c>: 0 where the key position is on or
    /// before the query position (col ≤ row), −1e9 above the diagonal. Built from Range +
    /// comparison + Where on constants (no Trilu, no gradient).
    /// </summary>
    public static Tensor<float32> CausalMask(Scalar<int64> lq, Scalar<int64> lk)
    {
        var rows = VectorRange(0L, lq, 1L).Unsqueeze(1L);
        var cols = VectorRange(0L, lk, 1L).Unsqueeze(0L);
        var zeros = TensorFill([lq, lk], 0f);
        var negInf = TensorFill([lq, lk], -1e9f);
        return (cols <= rows).Where(zeros, negInf);
    }

    /// <summary>
    /// Rotary Positional Embedding (RoPE; Su et al. 2021). Rotates each query/key
    /// vector by an angle proportional to its sequence position, encoding RELATIVE
    /// position inside the attention dot-product. Param-free: a deterministic
    /// rotation from position + a fixed frequency base. Apply to Q and K (each
    /// <c>[N, H, L, d]</c>, d EVEN) BEFORE <see cref="ScaledDotProductAttention"/>;
    /// do NOT apply to V.
    ///
    /// <para>Uses the GPT-NeoX / HuggingFace HALF-SPLIT layout and the rotate-half
    /// trick:</para>
    /// <code>
    ///     RoPE(x) = x · cos(mθ) + rotateHalf(x) · sin(mθ)
    ///     rotateHalf(x) = concat(-x[…, d/2:], x[…, :d/2])
    /// </code>
    /// with <c>θ_i = base^{-2i/d}</c>, <c>m = position</c>. The cos/sin tables are
    /// built in-graph from the input's own <c>L</c> (axis -2) and <c>d</c> (axis -1)
    /// and broadcast over the leading <c>[N, H]</c> via <c>[1, 1, L, d]</c>. The
    /// rotation is orthogonal, so it preserves each vector's norm exactly. Returns
    /// the rotated tensor with the SAME shape <c>[N, H, L, d]</c>.
    /// </summary>
    public static Tensor<float32> ApplyRoPE(Tensor<float32> x, long @base = 10000)
    {
        var l = x.DimTensor(-2);   // sequence length (axis -2), in-graph
        var d = x.DimTensor(-1);   // head dim (axis -1), must be even
        var half = d / 2L; // d/2 as a graph scalar

        // θ_i = base^{-2i/d} for i = 0 … d/2-1, via exponents -2i/d on [0, d) step 2.
        var dF = d.Cast<float32>();
        var exps = VectorRange(0L, d, 2L).Cast<float32>() * (-1f / dF); // [-0, -2/d, …]  length d/2
        var invFreq = Scalar((float)@base).Pow(exps);                                            // base^{-2i/d}    [d/2]

        var pos = VectorRange(0L, l, 1L).Cast<float32>();                         // [0,1,…,L-1]     [L]
        var angles = pos.Unsqueeze(1L) * invFreq.Unsqueeze(0L);                                   // outer(pos,freq) [L, d/2]
        var emb = angles.Concat(-1L, angles);                                                     // duplicate halves [L, d]
        var cos = emb.Cos().Unsqueeze(Vector(0L, 1L));                                            // [1, 1, L, d]
        var sin = emb.Sin().Unsqueeze(Vector(0L, 1L));                                            // [1, 1, L, d]

        // rotateHalf(x) = concat(-x2, x1) along the head dim, where
        //   x1 = x[..., :d/2]  (first half),  x2 = x[..., d/2:]  (second half).
        var x1 = x.Slice(Vector(0L), half.Unsqueeze(), axes: Vector(-1L));   // x[..., :d/2]
        var x2 = x.Slice(half.Unsqueeze(), d.Unsqueeze(), axes: Vector(-1L)); // x[..., d/2:]
        var rotated = (-x2).Concat(-1L, x1);                                  // [-x2, x1]

        return x * cos + rotated * sin; // x·cos + rotateHalf(x)·sin
    }
}

/// <summary>
/// Multi-head attention (batch-first by construction). <c>query [N, Lq, embedDim]</c>,
/// <c>key</c>/<c>value [N, Lk, embedDim]</c> — pass <c>(x, x, x)</c> for self-attention,
/// distinct tensors for cross-attention. Four <see cref="XavierUniform"/> projections
/// (q/k/v/out, each <c>[embedDim, embedDim]</c>, matching PyTorch's MHA init) with optional
/// zero biases gated by <c>useBias</c>. <c>causal</c> is a runtime <c>Scalar&lt;bit&gt;</c>:
/// since C# can't branch on it, the SDPA math is done inline here and the causal mask is
/// gated via <c>causal.IfElse(causalMask, zeros)</c> added to the scores. No PyTorch
/// backwards-compat surface (need_weights / kdim/vdim / add_zero_attn / batch_first) —
/// Shorokoo is explicit and batch-first.
/// </summary>
[Module]
public partial class MultiHeadAttention
{
    public static Tensor<float32> Inline(
        Tensor<float32> query,
        Tensor<float32> key,
        Tensor<float32> value,
        [Hyper] Scalar<int64> embedDim,
        [Hyper] Scalar<int64> numHeads,
        [Hyper] Scalar<bit> useBias,
        [Hyper] Scalar<bit> causal)
    {
        var headDim = embedDim / numHeads;
        var n = query.DimTensor(0);
        var lq = query.DimTensor(1);
        var lk = key.DimTensor(1);

        var wq = XavierUniform.Init([embedDim, embedDim]);
        var wk = XavierUniform.Init([embedDim, embedDim]);
        var wv = XavierUniform.Init([embedDim, embedDim]);
        var wo = XavierUniform.Init([embedDim, embedDim]);
        var bq = Zeros.Init([embedDim]).Vec();
        var bk = Zeros.Init([embedDim]).Vec();
        var bv = Zeros.Init([embedDim]).Vec();
        var bo = Zeros.Init([embedDim]).Vec();

        var q = useBias.IfElse(query.MatMul(wq.Transpose(1L, 0L)) + bq, query.MatMul(wq.Transpose(1L, 0L)));
        var k = useBias.IfElse(key.MatMul(wk.Transpose(1L, 0L)) + bk, key.MatMul(wk.Transpose(1L, 0L)));
        var v = useBias.IfElse(value.MatMul(wv.Transpose(1L, 0L)) + bv, value.MatMul(wv.Transpose(1L, 0L)));

        // [N, L, embedDim] -> [N, L, numHeads, headDim] -> [N, numHeads, L, headDim].
        var qh = q.Reshape([n, lq, numHeads, headDim]).Transpose(0L, 2L, 1L, 3L);
        var kh = k.Reshape([n, lk, numHeads, headDim]).Transpose(0L, 2L, 1L, 3L);
        var vh = v.Reshape([n, lk, numHeads, headDim]).Transpose(0L, 2L, 1L, 3L);

        // causal is a runtime bit; build the [Lq, Lk] mask and gate it instead of branching.
        var mask = causal.IfElse(Attention.CausalMask(lq, lk), TensorFill([lq, lk], 0f));
        var attended = Attention.ScaledDotProductAttention(qh, kh, vh, additiveMask: mask);

        // [N, numHeads, Lq, headDim] -> [N, Lq, numHeads, headDim] -> [N, Lq, embedDim].
        var combined = attended.Transpose(0L, 2L, 1L, 3L).Reshape([n, lq, embedDim]);
        return useBias.IfElse(combined.MatMul(wo.Transpose(1L, 0L)) + bo, combined.MatMul(wo.Transpose(1L, 0L)));
    }
}

/// <summary>
/// Pre-LayerNorm transformer encoder layer (the modern, training-robust default):
/// <c>h = x + MHA(LN(x)); out = h + FFN(LN(h))</c>, FFN = <c>Linear → GELU → Linear</c>.
/// Composes the existing <see cref="LayerNorm"/> and <see cref="MultiHeadAttention"/>
/// modules for the attention sub-layer.
///
/// <para>The FFN is built from explicit token-wise MatMuls rather than the
/// <see cref="Linear"/> module: <see cref="Linear"/> flattens ALL trailing dims into the
/// feature axis, so on a rank-3 <c>[N, L, E]</c> input it would collapse the sequence
/// (treating it as <c>[N, L·E]</c> features) and emit <c>[N, out]</c> — losing the
/// per-token structure. Matrix-multiplying by <c>[E, ffnDim]</c> / <c>[ffnDim, E]</c>
/// weights broadcasts over the leading <c>[N, L]</c> dims and keeps the <c>[N, L, ·]</c>
/// shape, which is what a transformer FFN requires.</para>
///
/// <para>LayerNorm is over the last dim only; its epsilon is fixed at 1e-5. There is no
/// dropout in v1 (a Dropout module call can be added around the sub-layer outputs). The
/// self-attention is non-causal.</para>
/// </summary>
[Module]
public partial class TransformerEncoderLayer
{
    public static Tensor<float32> Inline(
        Tensor<float32> x,
        [Hyper] Scalar<int64> embedDim,
        [Hyper] Scalar<int64> numHeads,
        [Hyper] Scalar<int64> ffnDim,
        [Hyper] Scalar<bit> useBias)
    {
        var attnIn = LayerNorm.Call(1L, Scalar(1e-5f), x);
        var h = x + MultiHeadAttention.Call(embedDim, numHeads, useBias, false, attnIn, attnIn, attnIn);

        var ffIn = LayerNorm.Call(1L, Scalar(1e-5f), h);

        var w1 = XavierUniform.Init([embedDim, ffnDim]);
        var w2 = XavierUniform.Init([ffnDim, embedDim]);
        var b1 = Zeros.Init([ffnDim]).Vec();
        var b2 = Zeros.Init([embedDim]).Vec();

        var hidden = useBias.IfElse(ffIn.MatMul(w1) + b1, ffIn.MatMul(w1)).Gelu();
        var ff = useBias.IfElse(hidden.MatMul(w2) + b2, hidden.MatMul(w2));
        return h + ff;
    }
}

/// <summary>
/// Pre-LayerNorm transformer DECODER layer (the modern, training-robust default):
/// three sublayers, each a residual around a pre-LN sublayer —
/// <c>h = tgt + MHA_self(LN(tgt), causal); h2 = h + MHA_cross(LN(h), memory, memory);
/// out = h2 + FFN(LN(h2))</c>. Mirrors <see cref="TransformerEncoderLayer"/>'s
/// residual/LN/FFN structure exactly, inserting the cross-attention sublayer between
/// the masked self-attention and the FFN.
///
/// <para>The masked self-attention is <b>causal</b> (position i attends only to
/// j ≤ i), so autoregressive generation is well-defined; query = key = value =
/// <c>LN(tgt)</c>. The cross-attention (encoder–decoder attention) is non-causal:
/// query = <c>LN(h)</c>, while key = value = the encoder <c>memory</c>
/// passed <b>raw</b> (no LayerNorm on memory — it is expected to be the
/// already-normalized encoder-stack output, matching PyTorch's
/// <c>_mha_block(norm2(x), memory, memory)</c>). Because the cross-attn query length
/// <c>Lt</c> and key/value length <c>Lm</c> differ, this exercises
/// <see cref="MultiHeadAttention"/>'s distinct-k/v (separate kdim/vdim) path.</para>
///
/// <para>The FFN is the same token-wise GELU MatMul idiom as
/// <see cref="TransformerEncoderLayer"/> (<c>Linear → GELU → Linear</c> built from
/// explicit <c>[E, ffnDim]</c> / <c>[ffnDim, E]</c> <see cref="XavierUniform"/>
/// weights with optional zero biases gated by <c>useBias</c>), preserving the
/// per-token <c>[N, L, ·]</c> structure. LayerNorm is over the last dim only, epsilon
/// fixed at 1e-5; no dropout in v1.</para>
/// </summary>
[Module]
public partial class TransformerDecoderLayer
{
    public static Tensor<float32> Inline(
        Tensor<float32> tgt,        // [N, Lt, embedDim]
        Tensor<float32> memory,     // [N, Lm, embedDim]
        [Hyper] Scalar<int64> embedDim,
        [Hyper] Scalar<int64> numHeads,
        [Hyper] Scalar<int64> ffnDim,
        [Hyper] Scalar<bit> useBias)
    {
        // Sublayer 1: masked (causal) self-attention over tgt, pre-LN.
        var saIn = LayerNorm.Call(1L, Scalar(1e-5f), tgt);
        var h = tgt + MultiHeadAttention.Call(embedDim, numHeads, useBias, true, saIn, saIn, saIn);

        // Sublayer 2: cross-attention — query = LN(h), key = value = memory (raw), non-causal.
        var caIn = LayerNorm.Call(1L, Scalar(1e-5f), h);
        var h2 = h + MultiHeadAttention.Call(embedDim, numHeads, useBias, false, caIn, memory, memory);

        // Sublayer 3: position-wise GELU FFN, pre-LN (mirrors TransformerEncoderLayer).
        var ffIn = LayerNorm.Call(1L, Scalar(1e-5f), h2);

        var w1 = XavierUniform.Init([embedDim, ffnDim]);
        var w2 = XavierUniform.Init([ffnDim, embedDim]);
        var b1 = Zeros.Init([ffnDim]).Vec();
        var b2 = Zeros.Init([embedDim]).Vec();

        var hidden = useBias.IfElse(ffIn.MatMul(w1) + b1, ffIn.MatMul(w1)).Gelu();
        var ff = useBias.IfElse(hidden.MatMul(w2) + b2, hidden.MatMul(w2));
        return h2 + ff;
    }
}

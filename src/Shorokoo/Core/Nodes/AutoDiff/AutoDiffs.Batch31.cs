using System;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Globals;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.AutoDiff
{
    /// <summary>
    /// Gradient entries for the opset 22-26 op batch (Attention, RMSNormalization,
    /// RotaryEmbedding @23; TensorScatter, Swish @24; BitCast, CumProd @26).
    /// Real gradients for Swish, CumProd and RMSNormalization via the [AutoDiff]
    /// reflection pattern; AD003 guards (DeformConv pattern — registered in
    /// <c>RegisterVariadicGradientOps</c> so the engine's unregistered-op path
    /// check doesn't fire first) for Attention, RotaryEmbedding and TensorScatter,
    /// whose adjoints are not implemented. BitCast is registered as a ZERO-class
    /// <c>NullInputGradient</c> (bitwise reinterpretation is non-differentiable).
    /// </summary>
    internal static partial class AutoDiffs
    {
        // ===== Swish =====
        // y = x * sigmoid(alpha * x), alpha default 1.
        // Let s = sigmoid(alpha * x).
        // dy/dx = s + alpha * x * s * (1 - s).

        [AutoDiff(SWISH)]
        public static Variable?[] Swish<T>(Tensor<T> x, Tensor<T> grad, float? alpha) where T : IVarType
        {
            var a = TypedConst(alpha ?? 1.0f, x);
            var one = TypedConst(1.0f, x);
            Tensor<T> s = OnnxOp.Sigmoid(a * x);
            var deriv = s + a * x * s * (one - s);
            return [grad * deriv];
        }

        // ===== CumProd =====
        //
        // Forward: y = CumProd(x, axis, exclusive, reverse).
        // Each y_j is the product of the x_k in its window, so dy_j/dx_i = y_j / x_i
        // whenever x_i lies in y_j's window. Summing over the consumers of x_i gives
        // the reverse-window cumulative sum (same window-flip as the CumSum gradient):
        //   grad_x = CumSum(grad * y, axis, exclusive=same, reverse=!reverse) / x
        //
        // CAVEAT (division by zero — same shape as the ReduceProd gradient caveat):
        // the (grad*y)/x_i form is undefined where x_i == 0. With a single zero in a
        // window the zero element's true gradient is the (nonzero) product of the
        // other elements but evaluates 0/0 = NaN; with two or more zeros every true
        // gradient is 0 yet the zero slots still evaluate 0/0 = NaN. The exact
        // zero-safe form needs exclusive prefix AND suffix products per element —
        // not worth the graph blow-up for a measure-zero input set. The NaN is
        // deliberately left to surface (a loud poisoned update) rather than masked
        // with Where(x==0, 0, ...), which would silently produce a WRONG (zero)
        // gradient in the single-zero case. Keep inputs away from exact zeros when
        // training through CumProd.

        [AutoDiff(CUM_PROD)]
        public static Variable?[] CumProd<T1, T2>(Tensor<T1> x, Tensor<T2> axis, Tensor<T1> grad, bool? exclusive, bool? reverse)
            where T1 : IVarType
            where T2 : IVarType
        {
            var effectiveExclusive = exclusive ?? false;
            var effectiveReverse = reverse ?? false;
            Tensor<T1> y = OnnxOp.CumProd(x, axis, exclusive: effectiveExclusive, reverse: effectiveReverse);
            Tensor<T1> summed = OnnxOp.CumSum(grad * y, axis,
                exclusive: effectiveExclusive, reverse: !effectiveReverse);
            return [summed / x, null];
        }

        // ===== RMSNormalization =====
        //
        // Forward: y = xHat * scale where xHat = x * invRms and
        //   invRms = 1 / sqrt(mean(x^2, suffix axes from `axis`) + epsilon).
        // RMSNorm has no mean-centering, so the LayerNorm backward loses its
        // mean(g) term; with g = grad * scale (broadcast):
        //   dx     = invRms * (g - x * invRms^2 * mean(g * x, axes, keepdims))
        //   dscale = ReverseBroadcast(grad * xHat, scale shape)
        // invRms is recomputed from x, like the BatchNorm/LayerNorm gradients do.
        // stash_type only selects the internal computation precision (no effect on
        // the gradient form under the float32-only engine).

        [AutoDiff(RMS_NORMALIZATION)]
        public static Variable?[] RMSNormalization<T>(Tensor<T> x, Tensor<T> scale, Tensor<T> grad,
            long? axis, float? epsilon, long? stashType) where T : IVarType
        {
            _ = stashType;
            var axisAttr = axis ?? -1;
            var epsConst = TypedConst(epsilon ?? 1e-5f, x);

            // Runtime axis arithmetic (same pattern as LayerNormalizationGradient):
            // handles negative AND positive axis without a statically-known rank.
            var xShape = OnnxOp.Shape(x);
            var xRankScalar = OnnxOp.Squeeze(OnnxOp.Shape(xShape), Vector(0L));
            var effectiveAxisScalar = axisAttr >= 0
                ? (Variable)Scalar(axisAttr)
                : OnnxOp.Add(xRankScalar, Scalar(axisAttr));
            var reduceAxes = OnnxOp.Range(effectiveAxisScalar, xRankScalar, Scalar(1L));

            Tensor<T> meanSq = OnnxOp.ReduceMean(x * x, reduceAxes, keepdims: true);
            Tensor<T> invRms = OnnxOp.Reciprocal(OnnxOp.Sqrt(meanSq + epsConst));
            var xHat = x * invRms;

            var gradScaled = grad * scale;
            Tensor<T> meanGradX = OnnxOp.ReduceMean(gradScaled * x, reduceAxes, keepdims: true);
            var dx = invRms * (gradScaled - x * invRms * invRms * meanGradX);

            var dScale = ReverseBroadcast(grad * xHat, scale.DShape);
            return [dx, dScale];
        }

        // ===== Attention (AD003 guard) =====
        //
        // The fused scaled-dot-product-attention backward (through softmax(QK^T·scale)V
        // with masks, causal windows, GQA head mapping, softcap and the KV-cache
        // present_* outputs) is not implemented. Throw AD003 instead of silently
        // returning null grads (DeformConv pattern).

        internal static Variable?[] AttentionGradient(
            Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            _ = inputs;
            _ = outputGrads;
            _ = attributes;
            throw new AutoDiffNotSupportedException(ErrorCodes.AD003, ATTENTION,
                "the Attention gradient (backward through the fused softmax(Q·K^T·scale)·V "
                + "kernel, including attn_mask, causal windows, GQA head mapping, softcap "
                + "and the KV-cache present_key/present_value outputs) is not implemented — "
                + "training through it would silently freeze the parameters behind it. This "
                + "is an implementation limitation, not a mathematical one. Compose the "
                + "attention from MatMul/Softmax primitives when it must be trained "
                + "end-to-end, or detach the Attention op from the loss path.");
        }

        // ===== RotaryEmbedding (AD003 guard) =====
        //
        // The rotation adjoint (inverse pairwise mix honoring position_ids gathering,
        // interleaved layout and partial rotary_embedding_dim slicing) is not
        // implemented; fail loudly instead of freezing upstream parameters.

        internal static Variable?[] RotaryEmbeddingGradient(
            Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            _ = inputs;
            _ = outputGrads;
            _ = attributes;
            throw new AutoDiffNotSupportedException(ErrorCodes.AD003, ROTARY_EMBEDDING,
                "the RotaryEmbedding gradient (the inverse pairwise rotation honoring "
                + "position_ids cache gathering, the interleaved layout and partial "
                + "rotary_embedding_dim slicing) is not implemented — training through it "
                + "would silently freeze the parameters behind it. This is an "
                + "implementation limitation, not a mathematical one. Compose the rotation "
                + "from Mul/Add/Slice/Concat primitives when it must be trained "
                + "end-to-end, or detach the RotaryEmbedding op from the loss path.");
        }

        // ===== TensorScatter (AD003 guard) =====
        //
        // The windowed-write adjoint (route dPresent back to past_cache outside the
        // written window and to update inside it, per-batch via write_indices, for
        // both linear and circular modes) is not implemented; fail loudly.

        internal static Variable?[] TensorScatterGradient(
            Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            _ = inputs;
            _ = outputGrads;
            _ = attributes;
            throw new AutoDiffNotSupportedException(ErrorCodes.AD003, TENSOR_SCATTER,
                "the TensorScatter gradient (routing the present_cache gradient back to "
                + "past_cache outside the written window and to update inside it, at the "
                + "per-batch write_indices offsets for both linear and circular modes) is "
                + "not implemented — training through it would silently freeze the "
                + "parameters behind it. This is an implementation limitation, not a "
                + "mathematical one. Use ScatterND/ScatterElements when the cache update "
                + "must be trained through, or detach TensorScatter from the loss path.");
        }
    }
}

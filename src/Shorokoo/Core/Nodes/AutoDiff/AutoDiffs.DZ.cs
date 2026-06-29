using System.Collections.Generic;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Globals;
using System.Reflection;
using System.Linq;
using System;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.AutoDiff
{
    internal static partial class AutoDiffs
    {
        /// <summary>
        /// Helper to create a scalar constant of a specific DType and cast to Tensor&lt;T&gt;.
        /// </summary>
        private static Tensor<T> TypedConst<T>(float value, Tensor<T> likeThis) where T : IVarType
            => OnnxOp.Cast(Scalar(value), saturate: null, to: likeThis.Type);

        /// <summary>
        /// Expands the upstream gradient back to the original data shape for reduction ops.
        /// Handles keepdims=true (direct Expand), keepdims=false (Unsqueeze then Expand),
        /// and axes=null (all-axes reduction, scalar grad).
        /// </summary>
        private static Tensor<T1> ExpandGradToOriginalShape<T1, T2>(
            Tensor<T1> grad, Tensor<T1> data, Tensor<T2>? axes, bool? keepdims)
            where T1 : IVarType
            where T2 : IVarType
        {
            var originalShape = data.DShape;
            var effectiveKeepdims = keepdims ?? true;

            if (!axes.HasValue)
                return OnnxOp.Expand(grad, originalShape);

            if (effectiveKeepdims)
                return OnnxOp.Expand(grad, originalShape);

            Tensor<T1> unsqueezedGrad = OnnxOp.Unsqueeze(grad, axes);
            return OnnxOp.Expand(unsqueezedGrad, originalShape);
        }

        // ===== Arithmetic Operations =====

        [AutoDiff(SUB)]
        public static Variable?[] Sub<T>(Tensor<T> a, Tensor<T> b, Tensor<T> grad) where T : IVarType
        {
            // d(a - b)/da = 1, d(a - b)/db = -1
            var aGrad = ReverseBroadcast(grad, a.DShape);
            var bGrad = ReverseBroadcast(-grad, b.DShape);

            return [aGrad, bGrad];
        }

        [AutoDiff(DIV)]
        public static Variable?[] Div<T>(Tensor<T> a, Tensor<T> b, Tensor<T> grad) where T : IVarType
        {
            // d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
            var aGrad = ReverseBroadcast(grad / b, a.DShape);
            var bGrad = ReverseBroadcast(-(grad * a) / (b * b), b.DShape);

            return [aGrad, bGrad];
        }

        [AutoDiff(NEG)]
        public static Variable?[] Neg<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // d(-x)/dx = -1
            return [-grad];
        }

        [AutoDiff(POW)]
        public static Variable?[] Pow<T>(Tensor<T> x, Tensor<T> y, Tensor<T> grad) where T : IVarType
        {
            // d(x^y)/dx = y * x^(y-1) * grad
            // d(x^y)/dy = x^y * ln(x) * grad
            var one = TypedConst(1.0f, x);
            Tensor<T> powResult = OnnxOp.Pow(x, y);     // x^y
            Tensor<T> powMinus1 = OnnxOp.Pow(x, y - one); // x^(y-1)
            var xGrad = ReverseBroadcast(grad * y * powMinus1, x.DShape);
            var yGrad = ReverseBroadcast(grad * powResult * x.Ln(), y.DShape);

            return [xGrad, yGrad];
        }

        // ===== Math Functions =====

        [AutoDiff(EXP)]
        public static Variable?[] Exp<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // d(exp(x))/dx = exp(x)
            return [grad * x.Exp()];
        }

        [AutoDiff(LOG)]
        public static Variable?[] Log<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // d(ln(x))/dx = 1/x → gradient = grad / x
            return [grad / x];
        }

        [AutoDiff(SQRT)]
        public static Variable?[] Sqrt<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // d(sqrt(x))/dx = 1/(2*sqrt(x))
            var two = TypedConst(2.0f, x);
            return [grad / (two * x.Sqrt())];
        }

        // ===== Activation Functions =====

        [AutoDiff(RELU)]
        public static Variable?[] Relu<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // d(relu(x))/dx = 1 if x > 0, 0 otherwise
            var zero = TypedConst(0.0f, x);
            var mask = x > zero;
            return [OnnxOp.Where(mask, grad, zero)];
        }

        [AutoDiff(SIGMOID)]
        public static Variable?[] Sigmoid<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
            var sig = x.Sigmoid();
            var one = TypedConst(1.0f, x);
            return [grad * sig * (one - sig)];
        }

        [AutoDiff(TANH)]
        public static Variable?[] Tanh<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // d(tanh(x))/dx = 1 - tanh(x)^2
            var tanhX = x.Tanh();
            var one = TypedConst(1.0f, x);
            return [grad * (one - tanhX * tanhX)];
        }

        // ===== Reduction Operations =====

        [AutoDiff(REDUCE_SUM)]
        public static Variable?[] ReduceSum<T1, T2>(Tensor<T1> data, Tensor<T2>? axes, Tensor<T1> grad, bool? keepdims, bool? noopWithEmptyAxes) 
            where T1 : IVarType
            where T2 : IVarType
        {
            // Gradient of ReduceSum: broadcast grad back to original shape
            var expandedGrad = ExpandGradToOriginalShape(grad, data, axes, keepdims);
            return [expandedGrad, null];
        }

        [AutoDiff(REDUCE_MEAN)]
        public static Variable?[] ReduceMean<T1, T2>(Tensor<T1> data, Tensor<T2>? axes, Tensor<T1> grad, bool? keepdims, bool? noopWithEmptyAxes) 
            where T1 : IVarType
            where T2 : IVarType
        {
            // Gradient of ReduceMean: broadcast grad / N back to original shape
            var expandedGrad = ExpandGradToOriginalShape(grad, data, axes, keepdims);

            // Count the number of elements that were reduced
            Tensor<T1> reducedCountTyped;
            if (!axes.HasValue)
            {
                // All axes reduced — N is the total number of elements = product of all dims
                Tensor<int64> fullShape = OnnxOp.Shape(data);
                Tensor<int64> totalSize = OnnxOp.ReduceProd(fullShape, keepdims: false);
                reducedCountTyped = OnnxOp.Cast(totalSize, saturate: null, to: data.Type);
            }
            else
            {
                Tensor<int64> reducedShape = OnnxOp.Shape(data);
                Tensor<int64> gatheredDims = OnnxOp.Gather(reducedShape, axes, axis: 0);
                Tensor<int64> reducedCount = OnnxOp.ReduceProd(gatheredDims, keepdims: false);
                reducedCountTyped = OnnxOp.Cast(reducedCount, saturate: null, to: data.Type);
            }

            return [expandedGrad / reducedCountTyped, null];
        }

        // ===== Shape Operations =====

        [AutoDiff(RESHAPE)]
        public static Variable?[] Reshape<T1, T2>(Tensor<T1> input, Tensor<T2> shape, Tensor<T1> grad, bool? allowzero) 
            where T1 : IVarType
            where T2 : IVarType
        {
            // Gradient of Reshape: reshape grad back to original shape
            var originalShape = input.DShape;
            return [OnnxOp.Reshape(grad, originalShape, allowZero: false), null];
        }

        [AutoDiff(TRANSPOSE)]
        public static Variable?[] Transpose<T>(Tensor<T> data, Tensor<T> grad, long[]? perm) where T : IVarType
        {
            // Gradient of Transpose: transpose with inverse permutation
            if (perm == null || perm.Length == 0)
            {
                // Default: reverse all axes; inverse of reverse is reverse.
                // Use OnnxOp.Transpose with null perm to signal "reverse all dims" per ONNX spec.
                return [OnnxOp.Transpose(grad, null)];
            }

            // Compute inverse permutation
            var inversePerm = new long[perm.Length];
            for (int i = 0; i < perm.Length; i++)
                inversePerm[perm[i]] = i;

            return [grad.Transpose(inversePerm)];
        }

        // ===== Matrix Operations =====

        [AutoDiff(MATMUL)]
        public static Variable?[] MatMul<T>(Tensor<T> a, Tensor<T> b, Tensor<T> grad) where T : IVarType
        {
            // For 2D: d(A@B)/dA = grad @ B^T, d(A@B)/dB = A^T @ grad
            // For batched matmul, transpose the last two dims
            var bTransposed = TransposeLastTwoDims(b);
            var aTransposed = TransposeLastTwoDims(a);

            Tensor<T> aGrad = OnnxOp.MatMul(grad, bTransposed);
            Tensor<T> bGrad = OnnxOp.MatMul(aTransposed, grad);

            // Handle broadcasting: reduce to original shapes
            aGrad = ReverseBroadcast(aGrad, a.DShape);
            bGrad = ReverseBroadcast(bGrad, b.DShape);

            return [aGrad, bGrad];
        }

        /// <summary>
        /// Transpose the last two dimensions of a tensor for matmul gradient computation.
        /// </summary>
        private static Tensor<T> TransposeLastTwoDims<T>(Tensor<T> tensor) where T : IVarType
        {
            var rank = tensor.Rank;
            if (rank != null && rank >= 2)
            {
                var perm = new long[rank.Value];
                for (int i = 0; i < rank.Value; i++)
                    perm[i] = i;
                perm[rank.Value - 2] = rank.Value - 1;
                perm[rank.Value - 1] = rank.Value - 2;
                return tensor.Transpose(perm);
            }

            // Rank is statically unknown (the common case for computed intermediates such
            // as Reshape/MatMul results, whose Rank is often null even though it is >= 2).
            // Transposing with a null perm reverses ALL axes, which equals "swap the last
            // two dims" only for rank <= 2; for a 3-D+ batched matmul it moves a batch dim
            // into the contraction position and the lowered MatMul then fails in ORT with
            // "operand cannot broadcast on dim 0". Do a rank-agnostic last-two-dims
            // transpose instead: collapse all leading dims into a single batch dim, swap
            // the (now fixed) last two dims with a 3-D perm, then restore the original
            // leading dims. The shapes are read at runtime via Shape, so this needs no
            // static rank. Operands always have rank >= 2 here — a matmul never contracts a
            // statically-rank-<2 operand, so that degenerate case does not reach this point.
            Tensor<int64> lastTwo = OnnxOp.Shape(tensor, start: -2);                      // [M, N]
            Tensor<int64> collapsedShape = OnnxOp.Concat([Vector(-1L), lastTwo], axis: 0); // [-1, M, N]
            Tensor<T> collapsed = OnnxOp.Reshape(tensor, collapsedShape, allowZero: false); // (B', M, N)
            var swapped = collapsed.Transpose(0L, 2L, 1L);                                      // (B', N, M)

            Tensor<int64> leading = OnnxOp.Shape(tensor, end: -2);                         // [d0 .. d_{r-3}]
            Tensor<int64> mDim = OnnxOp.Shape(tensor, start: -2, end: -1);                 // [M]
            Tensor<int64> nDim = OnnxOp.Shape(tensor, start: -1);                          // [N]
            Tensor<int64> restoredShape = OnnxOp.Concat([leading, nDim, mDim], axis: 0);   // [..., N, M]
            return OnnxOp.Reshape(swapped, restoredShape, allowZero: false);
        }

        // ===== Identity Operation =====

        [AutoDiff(IDENTITY)]
        public static Variable?[] Identity<T>(Tensor<T> input, Tensor<T> grad) where T : IVarType
        {
            // Identity gradient: pass through unchanged
            return [grad];
        }

        // ===== Softmax =====

        [AutoDiff(SOFTMAX)]
        public static Variable?[] Softmax<T>(Tensor<T> input, Tensor<T> grad, long? axis) where T : IVarType
        {
            // d(softmax(x))/dx_i = softmax(x)_i * (delta_ij - softmax(x)_j)
            // Efficient form: grad_input = softmax * (grad - sum(grad * softmax, axis=axis, keepdims=true))
            var softmaxOutput = input.Softmax(axis);
            var gradTimesSoftmax = grad * softmaxOutput;
            var effectiveAxis = axis ?? -1;
            var sumGradSoftmax = gradTimesSoftmax.Reduce(ReduceKind.Sum, axes: Vector(effectiveAxis), keepDims: true);
            return [softmaxOutput * (grad - sumGradSoftmax)];
        }

        // ===== ReduceProd =====

        [AutoDiff(REDUCE_PROD)]
        public static Variable?[] ReduceProd<T1, T2>(Tensor<T1> data, Tensor<T2>? axes, Tensor<T1> grad, bool? keepdims, bool? noopWithEmptyAxes) 
            where T1 : IVarType
            where T2 : IVarType
        {
            // d(∏x_i)/dx_j = ∏(x_i, i≠j) = (∏x_i) / x_j
            //
            // CAVEAT (division by zero): the prod/x_j form is undefined when x_j == 0.
            //   - exactly one zero in the reduced group: prod == 0, so prod/x_j = 0/0 = NaN
            //     for the zero element (its true gradient is the product of the OTHER
            //     elements, which is nonzero) and 0 for every other element (correct).
            //   - two or more zeros: every element's true gradient is 0; the zero elements
            //     still evaluate 0/0 = NaN.
            // The exact zero-safe formula needs prefix/suffix (exclusive) products along the
            // reduced axes, which the op set only expresses via per-axis CumSum-style
            // unrolling — not worth the graph blow-up for a measure-zero input set. The NaN
            // is deliberately left to surface (a loud poisoned update) rather than masked
            // with Where(x==0, 0, ...), which would silently produce a WRONG (zero) gradient
            // in the single-zero case. Keep inputs away from exact zeros when training
            // through ReduceProd.
            var originalShape = data.DShape;

            // Compute product along axes
            Tensor<T1> prod = OnnxOp.ReduceProd(data, axes, keepdims: true, noopWithEmptyAxes: noopWithEmptyAxes);
            var expandedGrad = ExpandGradToOriginalShape(grad, data, axes, keepdims);

            // Gradient: prod / x_i * grad (broadcast prod to match data shape)
            Tensor<T1> expandedProd = OnnxOp.Expand(prod, originalShape);
            return [expandedGrad * expandedProd / data, null];
        }

        // ===== ReduceSumSquare =====

        [AutoDiff(REDUCE_SUM_SQUARE)]
        public static Variable?[] ReduceSumSquare<T1, T2>(Tensor<T1> data, Tensor<T2>? axes, Tensor<T1> grad, bool? keepdims, bool? noopWithEmptyAxes) 
            where T1 : IVarType
            where T2 : IVarType
        {
            // ReduceSumSquare(x) = sum(x²) → d/dx_i = 2·x_i
            var two = TypedConst(2.0f, data);
            var expandedGrad = ExpandGradToOriginalShape(grad, data, axes, keepdims);
            return [two * data * expandedGrad, null];
        }

        // ===== ReduceLogSumExp =====

        [AutoDiff(REDUCE_LOG_SUM_EXP)]
        public static Variable?[] ReduceLogSumExp<T1, T2>(Tensor<T1> data, Tensor<T2>? axes, Tensor<T1> grad, bool? keepdims, bool? noopWithEmptyAxes) 
            where T1 : IVarType
            where T2 : IVarType
        {
            // ReduceLogSumExp(x) = log(sum(exp(x)))
            // d/dx_i = exp(x_i) / sum(exp(x_j)) = softmax(x)_i
            var originalShape = data.DShape;

            // Compute softmax-like term: exp(x_i) / sum(exp(x_j)) along the reduction axes
            var expData = data.Exp();
            Tensor<T1> sumExp = OnnxOp.ReduceSum(expData, axes, keepdims: true, noopWithEmptyAxes: noopWithEmptyAxes);
            var softmaxLike = expData / OnnxOp.Expand(sumExp, originalShape);

            var expandedGrad = ExpandGradToOriginalShape(grad, data, axes, keepdims);
            return [softmaxLike * expandedGrad, null];
        }

        // ===== ReduceL1 =====

        [AutoDiff(REDUCE_L1)]
        public static Variable?[] ReduceL1<T1, T2>(Tensor<T1> data, Tensor<T2>? axes, Tensor<T1> grad, bool? keepdims, bool? noopWithEmptyAxes) 
            where T1 : IVarType
            where T2 : IVarType
        {
            // ReduceL1(x) = sum(|x|) → d/dx_i = sign(x_i)
            var expandedGrad = ExpandGradToOriginalShape(grad, data, axes, keepdims);
            return [OnnxOp.Sign(data) * expandedGrad, null];
        }

        // ===== ReduceL2 =====

        [AutoDiff(REDUCE_L2)]
        public static Variable?[] ReduceL2<T1, T2>(Tensor<T1> data, Tensor<T2>? axes, Tensor<T1> grad, bool? keepdims, bool? noopWithEmptyAxes) 
            where T1 : IVarType
            where T2 : IVarType
        {
            // ReduceL2(x) = sqrt(sum(x²)) → d/dx_i = x_i / sqrt(sum(x²)) = x_i / L2
            var originalShape = data.DShape;

            // Compute L2 norm with keepdims=true for broadcasting
            Tensor<T1> l2 = OnnxOp.ReduceL2(data, axes, keepdims: true, noopWithEmptyAxes: noopWithEmptyAxes);
            var expandedGrad = ExpandGradToOriginalShape(grad, data, axes, keepdims);

            Tensor<T1> expandedL2 = OnnxOp.Expand(l2, originalShape);
            return [data / expandedL2 * expandedGrad, null];
        }

        // ===== ReduceLogSum =====

        [AutoDiff(REDUCE_LOG_SUM)]
        public static Variable?[] ReduceLogSum<T1, T2>(Tensor<T1> data, Tensor<T2>? axes, Tensor<T1> grad, bool? keepdims, bool? noopWithEmptyAxes) 
            where T1 : IVarType
            where T2 : IVarType
        {
            // ReduceLogSum(x) = log(sum(x)) → d/dx_i = 1/sum(x)
            var originalShape = data.DShape;

            // Compute sum with keepdims=true for broadcasting
            Tensor<T1> sumData = OnnxOp.ReduceSum(data, axes, keepdims: true, noopWithEmptyAxes: noopWithEmptyAxes);
            var expandedGrad = ExpandGradToOriginalShape(grad, data, axes, keepdims);

            Tensor<T1> expandedSum = OnnxOp.Expand(sumData, originalShape);
            return [expandedGrad / expandedSum, null];
        }

        // ===== ReduceMax =====

        [AutoDiff(REDUCE_MAX)]
        public static Variable?[] ReduceMax<T1, T2>(Tensor<T1> data, Tensor<T2>? axes, Tensor<T1> grad, bool? keepdims, bool? noopWithEmptyAxes) 
            where T1 : IVarType
            where T2 : IVarType
        {
            // ReduceMax(x) → gradient flows only to the max element(s).
            // If there are ties, gradient is shared equally among all max elements.
            var originalShape = data.DShape;

            // Compute max with keepdims=true for broadcasting
            Tensor<T1> maxVal = OnnxOp.ReduceMax(data, axes, keepdims: true, noopWithEmptyAxes: noopWithEmptyAxes);
            Tensor<T1> expandedMax = OnnxOp.Expand(maxVal, originalShape);

            // Create mask where data equals max; normalize by tie count
            Tensor<T1> mask = OnnxOp.Cast(OnnxOp.Equal(data, expandedMax), saturate: null, to: data.Type);
            Tensor<T1> maskSum = OnnxOp.ReduceSum(mask, axes, keepdims: true, noopWithEmptyAxes: noopWithEmptyAxes);
            Tensor<T1> expandedMaskSum = OnnxOp.Expand(maskSum, originalShape);

            var expandedGrad = ExpandGradToOriginalShape(grad, data, axes, keepdims);
            return [mask / expandedMaskSum * expandedGrad, null];
        }

        // ===== ReduceMin =====

        [AutoDiff(REDUCE_MIN)]
        public static Variable?[] ReduceMin<T1, T2>(Tensor<T1> data, Tensor<T2>? axes, Tensor<T1> grad, bool? keepdims, bool? noopWithEmptyAxes) 
            where T1 : IVarType
            where T2 : IVarType
        {
            // ReduceMin(x) → gradient flows only to the min element(s).
            // If there are ties, gradient is shared equally among all min elements.
            var originalShape = data.DShape;

            // Compute min with keepdims=true for broadcasting
            Tensor<T1> minVal = OnnxOp.ReduceMin(data, axes, keepdims: true, noopWithEmptyAxes: noopWithEmptyAxes);
            Tensor<T1> expandedMin = OnnxOp.Expand(minVal, originalShape);

            // Create mask where data equals min; normalize by tie count
            Tensor<T1> mask = OnnxOp.Cast(OnnxOp.Equal(data, expandedMin), saturate: null, to: data.Type);
            Tensor<T1> maskSum = OnnxOp.ReduceSum(mask, axes, keepdims: true, noopWithEmptyAxes: noopWithEmptyAxes);
            Tensor<T1> expandedMaskSum = OnnxOp.Expand(maskSum, originalShape);

            var expandedGrad = ExpandGradToOriginalShape(grad, data, axes, keepdims);
            return [mask / expandedMaskSum * expandedGrad, null];
        }

        // ===== CumSum =====

        [AutoDiff(CUM_SUM)]
        public static Variable?[] CumSum<T1, T2>(Tensor<T1> x, Tensor<T2> axis, Tensor<T1> grad, bool? exclusive, bool? reverse)
            where T1 : IVarType
            where T2 : IVarType
        {
            // CumSum(x, axis, exclusive, reverse) → y
            // Gradient: CumSum(grad, axis, exclusive=same, reverse=!reverse)
            var effectiveExclusive = exclusive ?? false;
            var effectiveReverse = reverse ?? false;
            Tensor<T1> gradX = OnnxOp.CumSum(grad, axis, exclusive: effectiveExclusive, reverse: !effectiveReverse);
            return [gradX, null];
        }
    }
}


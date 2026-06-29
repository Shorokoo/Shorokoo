using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Globals;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.AutoDiff
{
    internal static partial class AutoDiffs
    {
        // ===== GlobalAveragePool =====

        [AutoDiff(GLOBAL_AVERAGE_POOL)]
        public static Variable?[] GlobalAveragePool<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // GlobalAveragePool averages over spatial dimensions (dims 2, 3, ...)
            // Output shape: [N, C, 1, 1, ...]
            // Gradient: grad / spatial_size, expanded to input shape
            var inputShape = x.DShape;

            // Compute spatial size = product of dims from index 2 onwards (H * W * ...)
            Tensor<int64> spatialShape = OnnxOp.Shape(x, start: 2);
            Tensor<int64> spatialSize = OnnxOp.ReduceProd(spatialShape, keepdims: false);
            Tensor<T> spatialSizeT = OnnxOp.Cast(spatialSize, saturate: null, to: x.Type);

            Tensor<T> expandedGrad = OnnxOp.Expand(grad, inputShape);
            return [expandedGrad / spatialSizeT];
        }

        // ===== GlobalMaxPool =====

        [AutoDiff(GLOBAL_MAX_POOL)]
        public static Variable?[] GlobalMaxPool<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // GlobalMaxPool takes max over spatial dimensions
            // Gradient flows only to the element(s) that achieved the maximum
            var inputShape = x.DShape;

            // Recompute the max values [N, C, 1, 1, ...]
            Tensor<T> maxVal = OnnxOp.GlobalMaxPool(x);
            Tensor<T> expandedMax = OnnxOp.Expand(maxVal, inputShape);

            // Create mask: 1 where x == max, 0 elsewhere
            Tensor<T> mask = OnnxOp.Cast(OnnxOp.Equal(x, expandedMax), saturate: null, to: x.Type);

            // Normalize by number of max elements (handle ties):
            // GlobalAveragePool(mask) gives the mean of mask per channel, multiplied by
            // spatialSize gives the count of max elements per channel.
            Tensor<int64> spatialShape = OnnxOp.Shape(x, start: 2);
            Tensor<int64> spatialSize = OnnxOp.ReduceProd(spatialShape, keepdims: false);
            Tensor<T> spatialSizeT = OnnxOp.Cast(spatialSize, saturate: null, to: x.Type);
            Tensor<T> maskSum = OnnxOp.GlobalAveragePool(mask) * spatialSizeT;
            Tensor<T> expandedMaskSum = OnnxOp.Expand(maskSum, inputShape);

            Tensor<T> expandedGrad = OnnxOp.Expand(grad, inputShape);
            return [mask / expandedMaskSum * expandedGrad];
        }

        // ===== GlobalLpPool =====

        [AutoDiff(GLOBAL_LP_POOL)]
        public static Variable?[] GlobalLpPool<T>(Tensor<T> x, Tensor<T> grad, long? p) where T : IVarType
        {
            // GlobalLpPool: Y = (Σ|X_i|^p)^(1/p) over spatial dimensions (dims 2, 3, ...)
            // Output shape: [N, C, 1, 1, ...]
            //
            // Gradient derivation:
            //   Let S = Σ|X_i|^p, so Y = S^(1/p)
            //   dY/dX_i = (1/p) * S^(1/p - 1) * p * |X_i|^(p-1) * sign(X_i)
            //           = S^(1/p - 1) * |X_i|^(p-1) * sign(X_i)
            //           = Y^(1-p) * |X_i|^(p-1) * sign(X_i)
            //
            //   dX = grad * Y^(1-p) * |X|^(p-1) * sign(X)
            var pVal = p ?? 2L;
            var inputShape = x.DShape;

            // Recompute forward: Y = GlobalLpPool(X, p) → [N, C, 1, 1, ...]
            Tensor<T> y = OnnxOp.GlobalLpPool(x, pVal);

            Tensor<T> absX = OnnxOp.Abs(x);
            Tensor<T> signX = OnnxOp.Sign(x);

            // Compute |X|^(p-1) and Y^(1-p)
            var pMinus1 = TypedConst((float)(pVal - 1), x);
            var oneMinusP = TypedConst((float)(1 - pVal), x);
            Tensor<T> absXPm1 = OnnxOp.Pow(absX, pMinus1);
            Tensor<T> yPow = OnnxOp.Pow(y, oneMinusP);

            // Expand pooled-shape tensors to input shape for element-wise multiply
            Tensor<T> expandedYPow = OnnxOp.Expand(yPow, inputShape);
            Tensor<T> expandedGrad = OnnxOp.Expand(grad, inputShape);

            return [expandedGrad * expandedYPow * absXPm1 * signX];
        }
    }
}

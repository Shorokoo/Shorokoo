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
        public static IVariable?[] GlobalAveragePool<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // GlobalAveragePool averages over spatial dimensions (dims 2, 3, ...)
            // Output shape: [N, C, 1, 1, ...]
            // Gradient: grad / spatial_size, expanded to input shape
            var inputShape = x.DShape;

            // Compute spatial size = product of dims from index 2 onwards (H * W * ...)
            var spatialShape = (Tensor<int64>)OnnxOp.Shape(x, start: 2);
            var spatialSize = (Tensor<int64>)OnnxOp.ReduceProd(spatialShape, keepdims: false);
            var spatialSizeT = (Tensor<T>)OnnxOp.Cast(spatialSize, saturate: null, to: x.Type);

            var expandedGrad = (Tensor<T>)OnnxOp.Expand(grad, inputShape);
            return [expandedGrad / spatialSizeT];
        }

        // ===== GlobalMaxPool =====

        [AutoDiff(GLOBAL_MAX_POOL)]
        public static IVariable?[] GlobalMaxPool<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // GlobalMaxPool takes max over spatial dimensions
            // Gradient flows only to the element(s) that achieved the maximum
            var inputShape = x.DShape;

            // Recompute the max values [N, C, 1, 1, ...]
            var maxVal = (Tensor<T>)OnnxOp.GlobalMaxPool(x);
            var expandedMax = (Tensor<T>)OnnxOp.Expand(maxVal, inputShape);

            // Create mask: 1 where x == max, 0 elsewhere
            var mask = (Tensor<T>)OnnxOp.Cast(OnnxOp.Equal(x, expandedMax), saturate: null, to: x.Type);

            // Normalize by number of max elements (handle ties):
            // GlobalAveragePool(mask) gives the mean of mask per channel, multiplied by
            // spatialSize gives the count of max elements per channel.
            var spatialShape = (Tensor<int64>)OnnxOp.Shape(x, start: 2);
            var spatialSize = (Tensor<int64>)OnnxOp.ReduceProd(spatialShape, keepdims: false);
            var spatialSizeT = (Tensor<T>)OnnxOp.Cast(spatialSize, saturate: null, to: x.Type);
            var maskSum = (Tensor<T>)OnnxOp.GlobalAveragePool(mask) * spatialSizeT;
            var expandedMaskSum = (Tensor<T>)OnnxOp.Expand(maskSum, inputShape);

            var expandedGrad = (Tensor<T>)OnnxOp.Expand(grad, inputShape);
            return [mask / expandedMaskSum * expandedGrad];
        }

        // ===== GlobalLpPool =====

        [AutoDiff(GLOBAL_LP_POOL)]
        public static IVariable?[] GlobalLpPool<T>(Tensor<T> x, Tensor<T> grad, long? p) where T : IVarType
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
            var y = (Tensor<T>)OnnxOp.GlobalLpPool(x, pVal);

            var absX = (Tensor<T>)OnnxOp.Abs(x);
            var signX = (Tensor<T>)OnnxOp.Sign(x);

            // Compute |X|^(p-1) and Y^(1-p)
            var pMinus1 = TypedConst((float)(pVal - 1), x);
            var oneMinusP = TypedConst((float)(1 - pVal), x);
            var absXPm1 = (Tensor<T>)OnnxOp.Pow(absX, pMinus1);
            var yPow = (Tensor<T>)OnnxOp.Pow(y, oneMinusP);

            // Expand pooled-shape tensors to input shape for element-wise multiply
            var expandedYPow = (Tensor<T>)OnnxOp.Expand(yPow, inputShape);
            var expandedGrad = (Tensor<T>)OnnxOp.Expand(grad, inputShape);

            return [expandedGrad * expandedYPow * absXPm1 * signX];
        }
    }
}

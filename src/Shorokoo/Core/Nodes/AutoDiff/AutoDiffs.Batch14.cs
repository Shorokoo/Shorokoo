using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Globals;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.AutoDiff
{
    internal static partial class AutoDiffs
    {
        // ===== Compress =====
        //
        // Forward: output = Compress(input, condition, axis)
        //   If axis is null: flattens input, selects elements where condition is True → 1D output
        //   If axis is set: selects slices along axis where condition is True → same rank output
        //
        // Gradient:
        //   dInput: scatter output gradient back to positions where condition was True
        //   dCondition: null (boolean, not differentiable)
        //
        // No-axis case:
        //   1. Find True positions via NonZero(condition) → [K]
        //   2. Create 1D zeros of flattened input size
        //   3. ScatterElements grad at True positions
        //   4. Reshape back to input shape
        //
        // With-axis case:
        //   1. Find True positions via NonZero(condition) → [K]
        //   2. Build index tensor [1,...,-1,...,1] to reshape indices for axis broadcasting
        //   3. Expand indices to match grad shape
        //   4. Create zeros of input shape, ScatterElements grad back along axis

        [AutoDiff(COMPRESS)]
        public static Variable?[] Compress<T1>(
            Tensor<T1> input, Tensor<bit> condition,
            Tensor<T1> grad,
            long? axis)
            where T1 : IVarType
        {
            var inputShape = input.DShape;

            // Find positions where condition is True
            Tensor<int64> truePositions = OnnxOp.NonZero(condition); // [1, K]
            Tensor<int64> indices = OnnxOp.Reshape(truePositions, Vector(-1L), allowZero: false); // [K]

            var zero = TypedConst(0f, input);

            if (axis is null)
            {
                // No-axis case: input was flattened, grad is 1D [K]
                // Create 1D zeros of flattened input size
                Tensor<int64> flatSize = OnnxOp.ReduceProd(inputShape, keepdims: false);
                Tensor<int64> flatShape = OnnxOp.Reshape(flatSize, Vector(1L), allowZero: false);
                Tensor<T1> zeros = OnnxOp.Expand(zero, flatShape);

                // Scatter gradient at true positions
                Tensor<T1> dFlat = OnnxOp.ScatterElements(zeros, indices, grad, axis: 0);

                // Reshape back to input shape
                Tensor<T1> dInput = OnnxOp.Reshape(dFlat, inputShape, allowZero: false);
                return [dInput, null];
            }
            else
            {
                // Axis case: scatter grad back along the specified axis
                var effectiveAxis = axis.Value;

                // Build reshape target [1,...,-1,...,1] with -1 at axis position
                Tensor<int64> rank = OnnxOp.Shape(inputShape); // [rank_value]
                Tensor<int64> ones = OnnxOp.Expand(Scalar(1L), rank);
                Tensor<int64> axisIdx = OnnxOp.Reshape(Scalar(effectiveAxis), Vector(1L), allowZero: false);
                Tensor<int64> negOne = OnnxOp.Reshape(Scalar(-1L), Vector(1L), allowZero: false);
                Tensor<int64> reshapeTarget = OnnxOp.ScatterElements(ones, axisIdx, negOne, axis: 0);

                // Reshape indices to [1,...,K,...,1] and expand to grad shape
                Tensor<int64> indicesReshaped = OnnxOp.Reshape(indices, reshapeTarget, allowZero: false);
                var gradShape = grad.DShape;
                Tensor<int64> indicesExpanded = OnnxOp.Expand(indicesReshaped, gradShape);

                // Create zeros of input shape and scatter grad back
                Tensor<T1> zeros = OnnxOp.Expand(zero, inputShape);
                Tensor<T1> dInput = OnnxOp.ScatterElements(zeros, indicesExpanded, grad, axis: effectiveAxis);

                return [dInput, null];
            }
        }
    }
}

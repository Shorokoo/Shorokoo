using System;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Globals;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.AutoDiff
{
    internal static partial class AutoDiffs
    {
        // ===== TopK =====
        //
        // Forward: (Values, Indices) = TopK(X, K, axis, largest, sorted)
        //   Values: the top K values along the given axis
        //   Indices: the int64 indices of those values in the original tensor
        //
        // Gradient: only the Values output is differentiable (Indices are int64).
        //   dX = ScatterElements(zeros_like(X), Indices, dValues, axis=axis, reduction=Add)
        //
        // The gradient scatters the upstream gradient back to the original positions
        // using the indices from the forward pass. Positions not selected by TopK
        // receive zero gradient. If the same element appears at multiple positions
        // (ties), gradients accumulate via Add reduction.

        internal static Variable?[] TopKGradient(Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            var x = inputs[0]!;            // X: [..., N, ...] (N along axis)
            var k = inputs[1]!;            // K: [1] int64
            var dValues = outputGrads[0]!; // gradient of Values output: [..., K, ...]
            // outputGrads[1] is null (Indices are int64, no gradient)

            var axis = attributes.GetAttributeObj(AttrAxis) as long? ?? -1;
            var largest = attributes.GetAttributeObj(AttrLargest) as bool? ?? true;
            var sorted = attributes.GetAttributeObj(AttrSorted) as bool? ?? true;

            // Recompute TopK forward to retrieve the indices, with the forward node's actual
            // `sorted` attribute: if the forward ran with sorted=0, recomputing with sorted=1
            // could order the recovered indices differently from the forward Values output,
            // mis-permuting the gradient within the top-k set.
            var (_, indices) = OnnxOp.TopK(x, k, axis, largest, sorted: sorted);

            // Create zeros tensor with the same shape and type as X
            var xShape = OnnxOp.Shape(x);
            var zeros = OnnxOp.Expand(
                OnnxOp.Cast(Scalar(0.0f), saturate: null, to: x.Type),
                xShape);

            // Scatter dValues back to original positions along the specified axis
            var dX = OnnxOp.ScatterElements(
                zeros, indices, dValues,
                axis: axis, reduction: ScatterNDReduction.Add);

            return [dX, null]; // no gradient for K
        }
    }
}

using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Globals;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.AutoDiff
{
    internal static partial class AutoDiffs
    {
        // ===== Unique =====
        //
        // Forward: (Y, indices, inverse_indices, counts) = Unique(X, axis, sorted)
        //   No axis: flattens X to 1D, finds unique values → Y is 1D.
        //     inverse_indices: 1D tensor of length numel(X), where X_flat[i] = Y[inverse_indices[i]]
        //   With axis: finds unique slices along the specified axis → Y has same rank as X.
        //     inverse_indices: 1D tensor of length X.shape[axis], where X[...,i,...] = Y[...,inverse_indices[i],...]
        //
        // Gradient: only Y (output 0) is differentiable.
        //   indices, inverse_indices, counts are int64 → no gradient.
        //
        //   The gradient uses inverse_indices to route dL/dY back to dL/dX:
        //     dL/dX[i] = dL/dY[inverse_indices[i]]
        //   This is a Gather operation: dL/dX = Gather(dL/dY, inverse_indices, axis=0)
        //
        //   No axis: dL/dX_flat = Gather(dL/dY, inverse_indices, axis=0), then reshape to X's shape.
        //   With axis: dL/dX = Gather(dL/dY, inverse_indices, axis=axis).

        internal static Variable?[] UniqueGradient(Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            var x = inputs[0]!;
            var dY = outputGrads[0]!; // gradient of Y (unique values)
            // outputGrads[1..3] are null (indices, inverse_indices, counts are int64)

            var axis = attributes.GetAttributeObj(AttrAxis) as long?;
            var sorted = attributes.GetAttributeObj(AttrSorted) as bool? ?? true;

            // Recompute forward to get inverse_indices
            var (_, _, inverseIndices, _) = OnnxOp.Unique(x, axis: axis, sorted: sorted);

            if (axis is null)
            {
                // No-axis case: X was flattened to 1D before finding unique values.
                // inverse_indices has length numel(X), mapping each flat element to its Y position.
                // dX_flat = Gather(dY, inverse_indices, axis=0)
                var dXFlat = OnnxOp.Gather(dY, inverseIndices, axis: 0);
                // Reshape back to original X shape
                var dX = OnnxOp.Reshape(dXFlat, OnnxOp.Shape(x), allowZero: false);
                return [dX];
            }
            else
            {
                // Axis case: unique slices along the specified axis.
                // inverse_indices has length X.shape[axis], mapping each slice to its Y position.
                // dX = Gather(dY, inverse_indices, axis=axis)
                var dX = OnnxOp.Gather(dY, inverseIndices, axis: axis.Value);
                return [dX];
            }
        }
    }
}

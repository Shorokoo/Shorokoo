using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Globals;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.AutoDiff
{
    internal static partial class AutoDiffs
    {
        // ===== MaxUnpool (variadic registration) =====
        //
        // Forward: MaxUnpool(X, I, output_shape?) → Y
        //   Y is a zero-filled tensor of the output shape.
        //   For each element X[n,c,h,w], Y[n,c,I[n,c,h,w]] = X[n,c,h,w]
        //   where I[n,c,h,w] is a flat index into the output spatial dimensions.
        //
        // Gradient:
        //   Since MaxUnpool scatters X to positions I in Y, the gradient gathers
        //   from dL/dY at positions I:
        //     dL/dX[n,c,h,w] = dL/dY[n,c,I[n,c,h,w]]
        //   This is GatherElements on flattened spatial dimensions.
        //
        //   dL/dI = null (int64 indices, not differentiable)
        //   dL/doutput_shape = null (int64, not differentiable)

        internal static Variable?[] MaxUnpoolGradient(Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            var x = inputs[0]!;        // X: pooled tensor [N, C, H_pool, W_pool, ...]
            var indices = inputs[1]!;  // I: indices [N, C, H_pool, W_pool, ...]
            var grad = outputGrads[0]!; // output gradient [N, C, H_out, W_out, ...]

            // Flatten spatial dimensions (dims 2+) to a single dimension
            // Reshape to [N, C, -1] for both grad and indices
            var xShape = OnnxOp.Shape(x);
            var ncDims = OnnxOp.Slice(xShape, Globals.Vector(0L), Globals.Vector(2L));
            var negOne = Globals.Vector(-1L);

            // Flatten output gradient: [N, C, outSpatialFlat]
            var gradFlatShape = OnnxOp.Concat([
                OnnxOp.Slice(OnnxOp.Shape(grad), Globals.Vector(0L), Globals.Vector(2L)),
                negOne], axis: 0);
            var gradFlat = OnnxOp.Reshape(grad, gradFlatShape, allowZero: false);

            // Flatten indices: [N, C, poolSpatialFlat]
            var indicesFlatShape = OnnxOp.Concat([ncDims, negOne], axis: 0);
            var indicesFlat = OnnxOp.Reshape(indices, indicesFlatShape, allowZero: false);

            // Gather from flattened gradient at index positions
            var result = OnnxOp.GatherElements(gradFlat, indicesFlat, axis: 2);

            // Reshape back to X shape
            var gradX = OnnxOp.Reshape(result, xShape, allowZero: false);

            // Return: gradient for X, null for I (indices)
            return [gradX, null];
        }
    }
}

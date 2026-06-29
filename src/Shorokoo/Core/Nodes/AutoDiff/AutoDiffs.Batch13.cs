using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Globals;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.AutoDiff
{
    internal static partial class AutoDiffs
    {
        // ===== CenterCropPad =====
        //
        // Forward: output = CenterCropPad(input, target_shape, axes)
        //   Centers the input along each specified axis, then either:
        //   - Crops if input dim > target dim (extracts center region)
        //   - Pads with zeros if input dim < target dim (adds zeros around center)
        //
        // Gradient: CenterCropPad is self-inverse w.r.t. shapes.
        //   dInput = CenterCropPad(dOutput, Shape(input), axes)
        //
        // If forward cropped (input larger → output smaller), the gradient pads
        // zeros back to original size. If forward padded (input smaller → output
        // larger), the gradient crops to original size. In both cases, applying
        // CenterCropPad with the original input shape reverses the operation.

        [AutoDiff(CENTER_CROP_PAD)]
        public static Variable?[] CenterCropPad<T1, T2>(
            Tensor<T1> input, Tensor<T2> shape,
            Tensor<T1> grad,
            long[]? axes)
            where T1 : IVarType
            where T2 : IVarType
        {
            var inputShape = input.DShape;

            if (axes is null)
            {
                // No axes specified: shape covers all dimensions
                Tensor<T1> dInput = OnnxOp.CenterCropPad(grad, inputShape, axes: null);
                return [dInput, null];
            }
            else
            {
                // Axes specified: extract only the sizes for those axes from the original shape
                var axesTensor = Vector(axes);
                Tensor<int64> originalAxesSizes = OnnxOp.Gather(inputShape, axesTensor, axis: 0);
                Tensor<T1> dInput = OnnxOp.CenterCropPad(grad, originalAxesSizes, axes);
                return [dInput, null];
            }
        }
    }
}

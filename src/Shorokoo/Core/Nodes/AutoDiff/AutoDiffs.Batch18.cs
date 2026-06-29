using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Globals;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.AutoDiff
{
    internal static partial class AutoDiffs
    {
        // ===== LpPool (variadic registration) =====
        //
        // Forward: Y_i = (Σ_{j ∈ W_i} |X_j|^p)^(1/p) for each pooling window W_i
        //
        // Gradient derivation:
        //   dL/dX_j = Σ_{i: j ∈ W_i} dL/dY_i · ∂Y_i/∂X_j
        //
        //   ∂Y_i/∂X_j = Y_i^(1-p) · |X_j|^(p-1) · sign(X_j)
        //
        //   dL/dX_j = |X_j|^(p-1) · sign(X_j) · Σ_{i: j ∈ W_i} dL/dY_i · Y_i^(1-p)
        //           = |X_j|^(p-1) · sign(X_j) · scatter_back(dL/dY · Y^(1-p))
        //
        // The scatter_back is implemented via ConvTranspose with a ones kernel,
        // the same approach used in AveragePool gradient but without the 1/kernel_size factor.

        internal static Variable?[] LpPoolGradient(Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            var x = inputs[0]!;
            var grad = outputGrads[0]!;

            // Read attributes
            var kernelShape = attributes.GetAttributeObj("kernel_shape") as long[] ?? [1, 1];
            var pads = attributes.GetAttributeObj("pads") as long[];
            var strides = attributes.GetAttributeObj("strides") as long[] ?? [1, 1];
            var p = attributes.GetAttributeObj("p") as long? ?? 2L;

            // Recompute forward: Y = LpPool(X, ...)
            var y = OnnxOp.LpPool(x, autoPad: null, ceilMode: null, dilations: null,
                kernelShape: kernelShape, p: p, pads: pads, strides: strides);

            // Compute element-wise factors: |X|^(p-1) and sign(X)
            var absX = OnnxOp.Abs(x);
            var signX = OnnxOp.Sign(x);
            var pMinus1Const = OnnxOp.Cast(Globals.Scalar((float)(p - 1)), saturate: null, to: x.Type);
            var oneMinusPConst = OnnxOp.Cast(Globals.Scalar((float)(1 - p)), saturate: null, to: x.Type);
            var absXPm1 = OnnxOp.Pow(absX, pMinus1Const);  // |X|^(p-1)
            var yPow = OnnxOp.Pow(y, oneMinusPConst);       // Y^(1-p)

            // Weight in output space: dL/dY * Y^(1-p)
            var weight = OnnxOp.Mul(grad, yPow);

            // Scatter-back via ConvTranspose with ones kernel
            // Kernel shape: [C, 1, kH, kW] for depthwise operation
            var gradShape = OnnxOp.Shape(weight);
            var channels = OnnxOp.Slice(gradShape, Globals.Vector(1L), Globals.Vector(2L));
            var oneVec = Globals.Vector(1L);
            var spatialKernelShape = Globals.Vector(kernelShape);
            var fullKernelShape = OnnxOp.Concat([channels, oneVec, spatialKernelShape], axis: 0);

            var onesKernel = OnnxOp.Expand(
                OnnxOp.Cast(Globals.Scalar(1.0f), saturate: null, to: x.Type),
                fullKernelShape);

            var scattered = NodeBuilder.BuildNodeSingleOut(OpCodes.CONV_TRANSPOSE, [weight, onesKernel, null], [
                (OnnxOpAttributeNames.AttrAutoPad, (AutoPad?)AutoPad.NotSet),
                (OnnxOpAttributeNames.AttrDilations, (long[]?)null),
                (OnnxOpAttributeNames.AttrGroup, (long?)null),
                (OnnxOpAttributeNames.AttrKernelShape, kernelShape),
                (OnnxOpAttributeNames.AttrOutputPadding, (long[]?)null),
                (OnnxOpAttributeNames.AttrOutputShape, (long[]?)null),
                (OnnxOpAttributeNames.AttrPads, pads),
                (OnnxOpAttributeNames.AttrStrides, strides)]);

            // Reshape to match input shape
            scattered = OnnxOp.Reshape(scattered, OnnxOp.Shape(x), allowZero: false);

            // Final gradient: |X|^(p-1) * sign(X) * scatter_back(dY * Y^(1-p))
            var gradX = OnnxOp.Mul(OnnxOp.Mul(absXPm1, signX), scattered);

            return [gradX];
        }
    }
}

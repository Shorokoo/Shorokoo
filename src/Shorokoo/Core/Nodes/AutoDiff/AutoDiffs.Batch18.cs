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
        // The scatter_back is a Col2Im of the weights, each repeated over its window's
        // kernel_size positions, onto the padded input the windows run over: along each axis it
        // starts at the start padding and spans (out - 1) * stride + (k - 1) * dilation + 1
        // elements — every window, including one that ceil_mode keeps past the end padding — or
        // the input and its start padding if that is longer. The input is then sliced out of it.
        // SAME_UPPER / SAME_LOWER take their start padding from the input's and the output's
        // lengths, floor or ceil of half of (out - 1) * stride + (k - 1) * dilation + 1 - in.

        internal static Variable?[] LpPoolGradient(Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            var x = inputs[0]!;
            var grad = outputGrads[0]!;

            var kernelShape = attributes.GetAttributeObj("kernel_shape") as long[]
                ?? throw new InvalidOperationException("LpPool gradient: kernel_shape attribute is required.");
            int n = kernelShape.Length;
            var autoPad = attributes.GetAttributeObj("auto_pad") as AutoPad?;
            var ceilMode = attributes.GetAttributeObj("ceil_mode") as bool?;
            var dilations = attributes.GetAttributeObj("dilations") as long[] ?? [.. Enumerable.Repeat(1L, n)];
            var pads = attributes.GetAttributeObj("pads") as long[] ?? new long[2 * n];
            var strides = attributes.GetAttributeObj("strides") as long[] ?? [.. Enumerable.Repeat(1L, n)];
            var p = attributes.GetAttributeObj("p") as long? ?? 2L;

            var y = OnnxOp.LpPool(x, autoPad, ceilMode, dilations, kernelShape, p, pads, strides);

            var absXPm1 = OnnxOp.Pow(OnnxOp.Abs(x), OnnxOp.Cast(Globals.Scalar((float)(p - 1)), saturate: null, to: x.Type));
            var weight = OnnxOp.Mul(grad, OnnxOp.Pow(y, OnnxOp.Cast(Globals.Scalar((float)(1 - p)), saturate: null, to: x.Type)));

            long kernelSize = kernelShape.Aggregate(1L, (a, k) => a * k);
            long[] extent = [.. Enumerable.Range(0, n).Select(a => (kernelShape[a] - 1) * dilations[a] + 1)];
            var steps = Globals.Vector(strides);
            var one = Globals.Vector([.. Enumerable.Repeat(1L, n)]);
            var inLength = OnnxOp.Shape(x, start: 2L);
            var outLength = OnnxOp.Shape(weight, start: 2L);
            var covered = OnnxOp.Add(OnnxOp.Mul(OnnxOp.Sub(outLength, one), steps), Globals.Vector(extent));

            Variable begin = Globals.Vector(pads[..n]);
            if (autoPad is AutoPad.SameUpper or AutoPad.SameLower)
            {
                var total = OnnxOp.Max(OnnxOp.Sub(covered, inLength), Globals.Vector(new long[n]));
                if (autoPad is AutoPad.SameLower) total = OnnxOp.Add(total, one);
                begin = OnnxOp.Div(total, Globals.Vector([.. Enumerable.Repeat(2L, n)]));
            }
            var imageShape = OnnxOp.Max(covered, OnnxOp.Add(begin, inLength));

            var gradShape = OnnxOp.Shape(weight);
            var batch = OnnxOp.Slice(gradShape, Globals.Vector(0L), Globals.Vector(1L));
            var channels = OnnxOp.Slice(gradShape, Globals.Vector(1L), Globals.Vector(2L));
            var blocks = OnnxOp.ReduceProd(outLength, keepdims: true);
            var kernelSizeVec = Globals.Vector(kernelSize);
            var repeated = OnnxOp.Expand(
                OnnxOp.Reshape(weight, OnnxOp.Concat([batch, channels, Globals.Vector(1L), blocks], axis: 0), allowZero: false),
                OnnxOp.Concat([batch, channels, kernelSizeVec, blocks], axis: 0));
            var columns = OnnxOp.Reshape(repeated,
                OnnxOp.Concat([batch, OnnxOp.Mul(channels, kernelSizeVec), blocks], axis: 0), allowZero: false);
            var folded = OnnxOp.Col2Im(columns, imageShape, Globals.Vector(kernelShape), dilations, new long[2 * n], strides);
            var scattered = OnnxOp.Slice(folded, begin, OnnxOp.Add(begin, inLength),
                Globals.Vector([.. Enumerable.Range(2, n).Select(a => (long)a)]), null);

            return [OnnxOp.Mul(OnnxOp.Mul(absXPm1, OnnxOp.Sign(x)), scattered)];
        }
    }
}

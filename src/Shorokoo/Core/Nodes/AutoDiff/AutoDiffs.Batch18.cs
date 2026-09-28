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
        // lengths, floor or ceil of half of (out - 1) * stride + (k - 1) * dilation + 1 - in,
        // which is negative for some lengths when the stride exceeds the kernel's extent: the
        // first window then starts past the input's first element (SameStartPadding,
        // UnpaddedFold).

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

            bool isSame = autoPad is AutoPad.SameUpper or AutoPad.SameLower;
            var begin = isSame
                ? SameStartPadding(OnnxOp.Sub(covered, inLength), strides, autoPad is AutoPad.SameLower)
                : (Variable)Globals.Vector(pads[..n]);
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
            var scattered = UnpaddedFold(folded, begin, inLength, n, isSame && SameMayCrop(kernelShape, dilations, strides));

            return [OnnxOp.Mul(OnnxOp.Mul(absXPm1, OnnxOp.Sign(x)), scattered)];
        }

        /// <summary>The start padding of a <c>SAME_UPPER</c> (<paramref name="lower"/> false) or
        /// <c>SAME_LOWER</c> pool along each spatial axis: <c>floor(total / 2)</c> or
        /// <c>ceil(total / 2)</c> of its <paramref name="total"/> padding, which is at least
        /// <c>1 - stride</c> and so negative for some lengths when the stride exceeds the kernel's
        /// extent — taken as a truncating division of the total raised by twice the stride.</summary>
        private static Variable SameStartPadding(Variable total, long[] strides, bool lower)
            => OnnxOp.Sub(
                OnnxOp.Div(OnnxOp.Add(total, Globals.Vector([.. strides.Select(s => 2 * s + (lower ? 1 : 0))])),
                    Globals.Vector([.. Enumerable.Repeat(2L, strides.Length)])),
                Globals.Vector(strides));

        /// <summary>Whether a <c>SAME</c> pool's padding can be negative along some axis: a stride
        /// above the dilated kernel's extent.</summary>
        private static bool SameMayCrop(long[] kernelShape, long[] dilations, long[] strides)
            => Enumerable.Range(0, kernelShape.Length).Any(a => strides[a] > (kernelShape[a] - 1) * dilations[a] + 1);

        /// <summary>The input's positions of a fold over its padded extent, in which input position
        /// <c>i</c> sits at <c>i + begin</c> along each spatial axis. With
        /// <paramref name="mayCrop"/>, <paramref name="begin"/> may be negative: the positions a
        /// negative one leaves before the fold are in no window, and take zeros.</summary>
        private static Variable UnpaddedFold(Variable folded, Variable begin, Variable inLength, int n, bool mayCrop)
        {
            var axes = Globals.Vector([.. Enumerable.Range(2, n).Select(a => (long)a)]);
            if (!mayCrop)
                return OnnxOp.Slice(folded, begin, OnnxOp.Add(begin, inLength), axes, null);
            var none = OnnxOp.Mul(begin, Globals.Scalar(0L));
            var ahead = OnnxOp.Max(OnnxOp.Neg(begin), none);
            var start = OnnxOp.Max(begin, none);
            var padded = OnnxOp.Pad(folded, OnnxOp.Concat([Globals.Vector(0L, 0L), ahead, Globals.Vector(0L, 0L), none], axis: 0),
                OnnxOp.CastLike(Globals.Scalar(0f), folded, null));
            return OnnxOp.Slice(padded, start, OnnxOp.Add(start, inLength), axes, null);
        }
    }
}

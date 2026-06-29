using System;
using System.Linq;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Globals;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.AutoDiff
{
    internal static partial class AutoDiffs
    {
        // ===== Gemm =====

        [AutoDiff(GEMM)]
        public static Variable?[] Gemm<T>(
            Tensor<T> a, Tensor<T> b, Tensor<T>? c,
            Tensor<T> grad, float? alpha, float? beta, long? transA, long? transB)
            where T : IVarType
        {
            // Gemm: Y = alpha * (transA ? A^T : A) @ (transB ? B^T : B) + beta * C
            // Let A' = transA ? A^T : A, B' = transB ? B^T : B
            // dY/dA' = alpha * grad @ B'^T
            // dY/dB' = alpha * A'^T @ grad
            // dY/dC = beta * grad (with reverse broadcast)
            var effectiveAlpha = alpha ?? 1.0f;
            var effectiveBeta = beta ?? 1.0f;
            var effectiveTransA = transA ?? 0;
            var effectiveTransB = transB ?? 0;

            var alphaConst = TypedConst(effectiveAlpha, (Tensor<T>)grad);

            // Compute gradient w.r.t. A
            Tensor<T> gradA;
            if (effectiveTransA != 0)
            {
                // A was transposed: Y = alpha * A^T @ B' + beta*C
                // dL/dA^T = alpha * grad @ B'^T, so dL/dA = (dL/dA^T)^T = alpha * B' @ grad^T
                if (effectiveTransB != 0)
                    gradA = alphaConst * OnnxOp.MatMul(
                        OnnxOp.Transpose(b, perm: [1, 0]),
                        OnnxOp.Transpose(grad, perm: [1, 0]));
                else
                    gradA = alphaConst * OnnxOp.MatMul(
                        b,
                        OnnxOp.Transpose(grad, perm: [1, 0]));
            }
            else
            {
                // A was not transposed: dL/dA = alpha * grad @ B'^T
                if (effectiveTransB != 0)
                    gradA = alphaConst * OnnxOp.MatMul(grad, b);
                else
                    gradA = alphaConst * OnnxOp.MatMul(
                        grad,
                        OnnxOp.Transpose(b, perm: [1, 0]));
            }

            // Compute gradient w.r.t. B
            Tensor<T> gradB;
            if (effectiveTransB != 0)
            {
                // B was transposed: dL/dB^T = alpha * A'^T @ grad, dL/dB = (dL/dB^T)^T
                if (effectiveTransA != 0)
                    gradB = alphaConst * OnnxOp.MatMul(
                        OnnxOp.Transpose(grad, perm: [1, 0]),
                        OnnxOp.Transpose(a, perm: [1, 0]));
                else
                    gradB = alphaConst * OnnxOp.MatMul(
                        OnnxOp.Transpose(grad, perm: [1, 0]),
                        a);
            }
            else
            {
                // B was not transposed: dL/dB = alpha * A'^T @ grad
                if (effectiveTransA != 0)
                    gradB = alphaConst * OnnxOp.MatMul(a, grad);
                else
                    gradB = alphaConst * OnnxOp.MatMul(
                        OnnxOp.Transpose(a, perm: [1, 0]),
                        grad);
            }

            // Compute gradient w.r.t. C
            Variable? gradC = null;
            if (c is not null)
            {
                var betaConst = TypedConst(effectiveBeta, (Tensor<T>)grad);
                gradC = ReverseBroadcast(betaConst * grad, c.Value.DShape);
            }

            return [gradA, gradB, gradC];
        }

        // ===== InstanceNormalization =====

        [AutoDiff(INSTANCE_NORMALIZATION)]
        public static Variable?[] InstanceNormalization<T>(
            Tensor<T> x, Tensor<T> scale, Tensor<T> bias,
            Tensor<T> grad, float? epsilon) where T : IVarType
        {
            // InstanceNormalization: y = scale * (x - mean(x)) / sqrt(var(x) + eps) + bias
            // Normalize per instance (N) and per channel (C) over spatial dims [H, W, ...]
            var effectiveEps = epsilon ?? 1e-5f;
            var epsConst = TypedConst(effectiveEps, x);

            // Get input shape components
            var xShape = x.DShape;                                                    // [N, C, ...]
            var xRank = OnnxOp.Shape(xShape);                                         // [1] containing rank
            Tensor<int64> cVec = OnnxOp.Slice(xShape, Vector(1L), Vector(2L));   // [C]

            // Compute reduction axes for spatial dims: [2, 3, ..., rank-1]
            var xRankScalar = OnnxOp.Squeeze(xRank, Vector(0L));
            var spatialAxes = OnnxOp.Range(Scalar(2L), xRankScalar, Scalar(1L));

            // Compute mean and variance per instance per channel
            Tensor<T> mean = OnnxOp.ReduceMean(x, spatialAxes, keepdims: true);
            var xCentered = x - mean;
            Tensor<T> variance = OnnxOp.ReduceMean(xCentered * xCentered, spatialAxes, keepdims: true);
            Tensor<T> invStd = OnnxOp.Reciprocal(OnnxOp.Sqrt(variance + epsConst));

            // Normalized values
            var xHat = xCentered * invStd;

            // Build broadcast shape [1, C, 1, 1, ...] for scale/bias
            Tensor<int64> onesShape = OnnxOp.Expand(Scalar(1L), xRank);
            Tensor<int64> scatterIdx = OnnxOp.Reshape(Vector(1L), Vector(1L, 1L), allowZero: false);
            Tensor<int64> broadcastShape = OnnxOp.ScatterND(onesShape, scatterIdx, cVec);
            Tensor<T> scaleBC = OnnxOp.Reshape(scale, broadcastShape, allowZero: false);

            // Build reduce axes for dscale/dbias: [0, 2, 3, ..., rank-1] (all except channel dim 1)
            var allAxes = OnnxOp.Range(Scalar(0L), xRankScalar, Scalar(1L));
            Tensor<int64> axis0 = OnnxOp.Slice(allAxes, Vector(0L), Vector(1L));
            Tensor<int64> axesSuffix = OnnxOp.Slice(allAxes, Vector(2L), xRank);
            Tensor<int64> channelReduceAxes = OnnxOp.Concat([axis0, axesSuffix], axis: 0);

            // dscale = sum(grad * x_hat, axes=[0, 2, 3, ...]) → [C]
            Tensor<T> gradScale = OnnxOp.ReduceSum(grad * xHat, channelReduceAxes, keepdims: false);

            // dbias = sum(grad, axes=[0, 2, 3, ...]) → [C]
            Tensor<T> gradBias = OnnxOp.ReduceSum(grad, channelReduceAxes, keepdims: false);

            // dx: use instance normalization backward formula
            // dx = invStd * (grad_scaled - mean(grad_scaled) - x_centered * mean(grad_scaled * x_centered) * invStd²)
            var gradScaled = grad * scaleBC;
            Tensor<T> meanGrad = OnnxOp.ReduceMean(gradScaled, spatialAxes, keepdims: true);
            Tensor<T> meanGradXc = OnnxOp.ReduceMean(gradScaled * xCentered, spatialAxes, keepdims: true);
            var gradX = invStd * (gradScaled - meanGrad - xCentered * meanGradXc * invStd * invStd);

            return [gradX, gradScale, gradBias];
        }

        // ===== SpaceToDepth =====

        [AutoDiff(SPACE_TO_DEPTH)]
        public static Variable?[] SpaceToDepth<T>(
            Tensor<T> input, Tensor<T> grad, long? blocksize)
            where T : IVarType
        {
            // SpaceToDepth: [N, C, H, W] → [N, C*r², H/r, W/r]
            // Gradient: DepthToSpace(grad) which is the inverse transform
            var r = blocksize ?? 2;
            var originalShape = input.DShape;

            // Get grad shape components: [N, C*r², H/r, W/r]
            var gradShape = grad.DShape;
            Tensor<int64> nVec = OnnxOp.Slice(gradShape, Vector(0L), Vector(1L));
            Tensor<int64> cr2Vec = OnnxOp.Slice(gradShape, Vector(1L), Vector(2L));
            Tensor<int64> hrVec = OnnxOp.Slice(gradShape, Vector(2L), Vector(3L));
            Tensor<int64> wrVec = OnnxOp.Slice(gradShape, Vector(3L), Vector(4L));
            var rVec = Vector(r);
            var r2Vec = Vector(r * r);
            var cVec = cr2Vec / r2Vec;

            // Step 1: Reshape grad [N, C*r², H/r, W/r] → [N, r, r, C, H/r, W/r] (DCR mode)
            Tensor<int64> intermediateShape = OnnxOp.Concat(
                [nVec, rVec, rVec, cVec, hrVec, wrVec], axis: 0);
            Tensor<T> reshaped = OnnxOp.Reshape(grad, intermediateShape, allowZero: false);

            // Step 2: Transpose [N, r, r, C, H/r, W/r] → [N, C, H/r, r, W/r, r]
            Tensor<T> transposed = OnnxOp.Transpose(reshaped, perm: [0, 3, 4, 1, 5, 2]);

            // Step 3: Reshape to original input shape [N, C, H, W]
            Tensor<T> result = OnnxOp.Reshape(transposed, originalShape, allowZero: false);

            return [result];
        }

        // ===== Trilu =====

        [AutoDiff(TRILU)]
        public static Variable?[] Trilu<T1>(
            Tensor<T1> input, Tensor<int64>? k, Tensor<T1> grad, long? upper)
            where T1 : IVarType
        {
            // Trilu: extracts upper or lower triangular part of a matrix
            // Gradient: apply the same triangular mask to the upstream gradient
            var effectiveUpper = upper ?? 1;

            Variable gradResult;
            if (k is not null)
                gradResult = OnnxOp.Trilu(grad, k, upper: effectiveUpper);
            else
                gradResult = OnnxOp.Trilu(grad, null, upper: effectiveUpper);

            return [gradResult, null];
        }

        // ===== LpNormalization =====

        [AutoDiff(LP_NORMALIZATION)]
        public static Variable?[] LpNormalization<T>(
            Tensor<T> input, Tensor<T> grad, long? axis, long? p)
            where T : IVarType
        {
            // LpNormalization: y = x / ||x||_p along axis
            // For p=2 (default): y = x / ||x||_2
            //   dy/dx = (grad - y * sum(grad * y, axis, keepdims=true)) / ||x||_2
            // For p=1: y = x / ||x||_1
            //   dy/dx_j = (grad_j - sign(x_j) * sum(grad * y, axis, keepdims=true)) / ||x||_1
            var effectiveP = p ?? 2;
            var effectiveAxis = axis ?? -1;

            var axesTensor = Vector(effectiveAxis);

            if (effectiveP == 2)
            {
                // L2 normalization gradient
                Tensor<T> norm = OnnxOp.ReduceL2(input, axesTensor, keepdims: true);
                var epsConst = TypedConst(1e-12f, input);
                Tensor<T> safeNorm = OnnxOp.Max(norm, epsConst);
                var y = input / safeNorm;
                Tensor<T> dotProduct = OnnxOp.ReduceSum(grad * y, axesTensor, keepdims: true);
                var gradInput = (grad - y * dotProduct) / safeNorm;
                return [gradInput];
            }
            else // p == 1
            {
                // L1 normalization gradient
                // y = x / ||x||_1, so dy_j/dx_i = (δ_ij - sign(x_i) * y_j) / ||x||_1
                // dL/dx_i = sum_j(dL/dy_j * dy_j/dx_i) = (dL/dy_i - sign(x_i) * sum(dL/dy * y)) / ||x||_1
                Tensor<T> norm = OnnxOp.ReduceL1(input, axesTensor, keepdims: true);
                var epsConst = TypedConst(1e-12f, input);
                Tensor<T> safeNorm = OnnxOp.Max(norm, epsConst);
                Tensor<T> signX = OnnxOp.Sign(input);
                var y = input / safeNorm;
                Tensor<T> dotGradY = OnnxOp.ReduceSum(grad * y, axesTensor, keepdims: true);
                var gradInput = (grad - signX * dotGradY) / safeNorm;
                return [gradInput];
            }
        }

        // ===== Conv (variadic registration) =====

        internal static Variable?[] ConvGradient(Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            // Conv(x, w, b) → y
            // dx = ConvTranspose(grad, w) with matching parameters
            // dw = Conv(x, grad) with strides/dilations swapped (see below)
            // db = ReduceSum(grad, axes=[0, 2, 3, ...])
            var x = inputs[0]!;
            var w = inputs[1]!;
            var b = inputs.Length > 2 ? inputs[2] : null;
            var grad = outputGrads[0]!;

            // Read attributes with defaults
            var dilations = attributes.GetAttributeObj("dilations") as long[] ?? [1, 1];
            var group = attributes.GetAttributeObj("group") as long? ?? 1L;
            var kernelShape = attributes.GetAttributeObj("kernel_shape") as long[];
            var pads = attributes.GetAttributeObj("pads") as long[];
            var strides = attributes.GetAttributeObj("strides") as long[] ?? [1, 1];

            // dx: Use ConvTranspose with matching parameters.
            // output_padding = strides - 1 ensures the ConvTranspose spatial output is >= x's
            // spatial size. This is needed because ConvTranspose with stride > 1 can produce
            // output that is 1 pixel smaller than the original Conv input. We then slice to
            // match x's exact shape.
            var outputPadding = strides.Select(s => s - 1).ToArray();
            var gradX = NodeBuilder.BuildNodeSingleOut(OpCodes.CONV_TRANSPOSE, [grad, w, null], [
                (OnnxOpAttributeNames.AttrAutoPad, (AutoPad?)AutoPad.NotSet),
                (OnnxOpAttributeNames.AttrDilations, dilations),
                (OnnxOpAttributeNames.AttrGroup, group),
                (OnnxOpAttributeNames.AttrKernelShape, kernelShape),
                (OnnxOpAttributeNames.AttrOutputPadding, (long[]?)outputPadding),
                (OnnxOpAttributeNames.AttrOutputShape, (long[]?)null),
                (OnnxOpAttributeNames.AttrPads, pads),
                (OnnxOpAttributeNames.AttrStrides, strides)]);

            // Slice to match x's exact shape (ConvTranspose with output_padding may produce
            // slightly larger spatial output than needed when the original input had odd spatial dims)
            var xShape = OnnxOp.Shape(x);
            var xRank = OnnxOp.Shape(xShape);
            var zeros = OnnxOp.ConstantOfShape(xRank, Globals.TensorData(1, 0L));
            gradX = OnnxOp.Slice(gradX, zeros, xShape);

            // dw: weight gradient. For a 2-D convolution the weight gradient is itself a
            // convolution of the input by the output-gradient:
            //   dw[oc,ic,kh,kw] = Σ_{n,oh,ow} grad[n,oc,oh,ow] · x[n,ic, oh·stride + kh·dilation − pad]
            // This is expressed as Conv(x^T, grad^T) with stride/dilation swapped, then
            // transposed back, where the reduction axis (the conv's "in-channels") is the
            // batch N and the conv "kernel" is the full grad map:
            //   A = transpose(x,    perm [1,0,2,3]) → [Cin,  N, H,  W ]
            //   B = transpose(grad, perm [1,0,2,3]) → [Cout, N, OH, OW]
            //   Conv(A, B, strides=dilations, dilations=strides, pads=pads) → [Cin, Cout, KH', KW']
            //   dw = transpose(…, [1,0,2,3]) sliced to w's [Cout, Cin, KH, KW].
            // Grouped convolutions apply the same trick per channel group (the group count is
            // a static attribute, so the per-group loop is unrolled at graph-build time) and
            // concatenate the per-group [Cout/g, Cin/g, KH', KW'] pieces along the output-
            // channel axis — matching w's grouped [Cout, Cin/g, KH, KW] layout. See
            // GroupedConvWeightGradient.
            Variable? gradW = GroupedConvWeightGradient(
                image: x, kernel: grad, group: group,
                strides: strides, dilations: dilations, pads: pads);

            // Slice to w's exact spatial size: the gradient-conv can overrun KH/KW by up to
            // stride−1 when the forward output didn't evenly tile the input (mirrors dx's slice).
            {
                var wShape = OnnxOp.Shape(w);
                var wRank = OnnxOp.Shape(wShape);
                var wZeros = OnnxOp.ConstantOfShape(wRank, Globals.TensorData(1, 0L));
                gradW = OnnxOp.Slice(gradW, wZeros, wShape);
            }

            // db = ReduceSum(grad, axes=[0, 2, 3, ...]) → [outChannels]
            Variable? gradB = null;
            if (b is not null)
            {
                var gradShape = OnnxOp.Shape(grad);
                var gradRank = OnnxOp.Shape(gradShape);
                var gradRankScalar = OnnxOp.Squeeze(gradRank, Globals.Vector(0L));
                var allAxes = OnnxOp.Range(Globals.Scalar(0L), gradRankScalar, Globals.Scalar(1L));
                var axis0 = OnnxOp.Slice(allAxes, Globals.Vector(0L), Globals.Vector(1L));
                var axesSuffix = OnnxOp.Slice(allAxes, Globals.Vector(2L), gradRank);
                var reduceAxes = OnnxOp.Concat([axis0, axesSuffix], axis: 0);
                gradB = OnnxOp.ReduceSum(grad, reduceAxes, keepdims: false);
            }

            return [gradX, gradW, gradB];
        }

        // ===== ConvTranspose (variadic registration) =====

        internal static Variable?[] ConvTransposeGradient(Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            // ConvTranspose(x, w, b) → y, with w laid out [Cin, Cout/group, KH, KW] and
            //   y[n, co, j] = Σ_{ci, i, k : i·stride + k·dilation − pad_begin = j} x[n, ci, i] · w[ci, co, k]
            // dx = Conv(grad, w) with matching parameters (the adjoint of ConvTranspose is Conv)
            // dw = Conv(grad^T, x^T) with strides/dilations swapped (see below)
            // db = ReduceSum(grad, axes=[0, 2, 3, ...])
            var x = inputs[0]!;
            var w = inputs[1]!;
            var b = inputs.Length > 2 ? inputs[2] : null;
            var grad = outputGrads[0]!;

            // Read attributes with defaults
            var dilations = attributes.GetAttributeObj("dilations") as long[] ?? [1, 1];
            var group = attributes.GetAttributeObj("group") as long? ?? 1L;
            var kernelShape = attributes.GetAttributeObj("kernel_shape") as long[];
            var pads = attributes.GetAttributeObj("pads") as long[];
            var strides = attributes.GetAttributeObj("strides") as long[] ?? [1, 1];

            // dx: Use Conv with matching parameters
            var gradX = NodeBuilder.BuildNodeSingleOut(OpCodes.CONV, [grad, w, null], [
                (OnnxOpAttributeNames.AttrAutoPad, (AutoPad?)AutoPad.NotSet),
                (OnnxOpAttributeNames.AttrDilations, dilations),
                (OnnxOpAttributeNames.AttrGroup, group),
                (OnnxOpAttributeNames.AttrKernelShape, kernelShape),
                (OnnxOpAttributeNames.AttrPads, pads),
                (OnnxOpAttributeNames.AttrStrides, strides)]);

            // Reshape to ensure output matches input x's shape
            gradX = OnnxOp.Reshape(gradX, OnnxOp.Shape(x), allowZero: false);

            // dw: weight gradient. From the forward formula above:
            //   dw[ci, co, kh, kw] = Σ_{n, ih, iw} x[n, ci, ih, iw] · grad[n, co, ih·stride + kh·dilation − pad]
            // This is the same Conv-as-weight-gradient trick used in ConvGradient, with the
            // roles of x and grad swapped: treat grad as the conv "image" and x as the conv
            // "kernel", with strides/dilations swapped:
            //   A = transpose(grad, [1,0,2,3]) → [Cout, N, OH, OW]
            //   B = transpose(x,    [1,0,2,3]) → [Cin,  N, H,  W ]
            //   Conv(A, B, strides=dilations, dilations=strides, pads=pads) → [Cout, Cin, KH', KW']
            //   dw = transpose(…, [1,0,2,3]) sliced to w's [Cin, Cout, KH, KW].
            // The slice trims the up-to-floor(output_padding/dilation) kernel-position overrun
            // introduced when the forward used output_padding. Grouped deconvolutions apply
            // the trick per channel group and concatenate the per-group [Cin/g, Cout/g, …]
            // pieces along axis 0 — matching w's grouped [Cin, Cout/g, KH, KW] layout. See
            // GroupedConvWeightGradient (image/kernel roles swapped vs. Conv's call).
            Variable? gradW = GroupedConvWeightGradient(
                image: grad, kernel: x, group: group,
                strides: strides, dilations: dilations, pads: pads);

            {
                var wShape = OnnxOp.Shape(w);
                var wRank = OnnxOp.Shape(wShape);
                var wZeros = OnnxOp.ConstantOfShape(wRank, Globals.TensorData(1, 0L));
                gradW = OnnxOp.Slice(gradW, wZeros, wShape);
            }

            // db = ReduceSum(grad, axes=[0, 2, 3, ...])
            Variable? gradB = null;
            if (b is not null)
            {
                var gradShape = OnnxOp.Shape(grad);
                var gradRank = OnnxOp.Shape(gradShape);
                var gradRankScalar = OnnxOp.Squeeze(gradRank, Globals.Vector(0L));
                var allAxes = OnnxOp.Range(Globals.Scalar(0L), gradRankScalar, Globals.Scalar(1L));
                var axis0 = OnnxOp.Slice(allAxes, Globals.Vector(0L), Globals.Vector(1L));
                var axesSuffix = OnnxOp.Slice(allAxes, Globals.Vector(2L), gradRank);
                var reduceAxes = OnnxOp.Concat([axis0, axesSuffix], axis: 0);
                gradB = OnnxOp.ReduceSum(grad, reduceAxes, keepdims: false);
            }

            return [gradX, gradW, gradB];
        }

        // ===== Conv / ConvTranspose shared weight-gradient machinery =====
        //
        // The swapped-roles trick: treating `image`'s channel axis as the conv batch and
        // `kernel`'s channel axis as the conv filter count, with strides/dilations swapped,
        // a single Conv computes Σ_{n, spatial} image[n, ci, …] · kernel[n, co, …] over every
        // kernel offset — i.e. the weight gradient.
        //   For Conv:          image = x,    kernel = dy → [Cout, Cin, KH', KW']
        //   For ConvTranspose: image = dy,   kernel = x  → [Cin, Cout, KH', KW']
        // (after the trailing [1,0,2,3] transpose; KH'/KW' may overrun the true kernel size —
        // callers slice down to w's shape). 2-D only, mirroring the dx implementations.

        private static Variable ConvWeightGradientViaSwappedRoles(
            Variable image, Variable kernel, long[] strides, long[] dilations, long[]? pads)
        {
            var imageT = OnnxOp.Transpose(image, [1L, 0L, 2L, 3L]);
            var kernelT = OnnxOp.Transpose(kernel, [1L, 0L, 2L, 3L]);
            // kernel_shape = null → inferred from kernel^T's spatial extent (it is dynamic).
            var dwConv = OnnxOp.Conv(imageT, kernelT, null!, AutoPad.NotSet,
                dilations: strides, group: 1L, kernelShape: null!, pads: pads, strides: dilations);
            return OnnxOp.Transpose(dwConv, [1L, 0L, 2L, 3L]);
        }

        // Grouped weight gradient: group is a static attribute, so the per-group loop is
        // unrolled at graph-construction time. Each group g slices its channel band out of
        // both operands (channel counts are runtime values — Shape/Div arithmetic), applies
        // the swapped-roles Conv, and the per-group pieces are concatenated along axis 0,
        // which is the kernel's group-blocked first axis for Conv ([Cout, Cin/g, …]) and
        // ConvTranspose ([Cin, Cout/g, …]) alike. Note: for large group counts (e.g.
        // depthwise convs over many channels) this emits 'group' Conv nodes.

        private static Variable GroupedConvWeightGradient(
            Variable image, Variable kernel, long group,
            long[] strides, long[] dilations, long[]? pads)
        {
            if (group <= 1L)
                return ConvWeightGradientViaSwappedRoles(image, kernel, strides, dilations, pads);

            var imagePerGroup = OnnxOp.Div(
                OnnxOp.Slice(OnnxOp.Shape(image), Globals.Vector(1L), Globals.Vector(2L)),
                Globals.Vector(group));
            var kernelPerGroup = OnnxOp.Div(
                OnnxOp.Slice(OnnxOp.Shape(kernel), Globals.Vector(1L), Globals.Vector(2L)),
                Globals.Vector(group));

            var parts = new Variable[group];
            for (long g = 0; g < group; g++)
            {
                var imageG = OnnxOp.Slice(image,
                    OnnxOp.Mul(imagePerGroup, Globals.Vector(g)),
                    OnnxOp.Mul(imagePerGroup, Globals.Vector(g + 1)),
                    Globals.Vector(1L));
                var kernelG = OnnxOp.Slice(kernel,
                    OnnxOp.Mul(kernelPerGroup, Globals.Vector(g)),
                    OnnxOp.Mul(kernelPerGroup, Globals.Vector(g + 1)),
                    Globals.Vector(1L));
                parts[g] = ConvWeightGradientViaSwappedRoles(imageG, kernelG, strides, dilations, pads);
            }
            return OnnxOp.Concat(parts, axis: 0);
        }

        // ===== AveragePool (variadic registration) =====
        //
        // AveragePool(x) → y splits x into windows of shape kernel_shape and emits the
        // (possibly-weighted) average of each window. Gradient distributes each window's
        // grad/divisor back to every input position covered by that window. Overlapping
        // windows (stride < kernel) sum their contributions on the input side.
        //
        // We express this with Col2Im: it's the natural inverse of im2col, accepts the
        // exact (kernel_shape, pads, strides, dilations) tuple we have, and accumulates
        // overlapping window contributions on output. No depthwise convolution needed,
        // and no static channel count needed — Col2Im derives C from input_dim_1 /
        // prod(block_shape) at runtime.
        //
        // Pipeline (attribute-driven, one graph shape per attribute combination):
        //
        //   1. Compute the per-window divisor. With count_include_pad=true (or no pads
        //      anywhere) this is a static 1/kernel_size scalar. With count_include_pad=
        //      false AND at least one non-zero pad, we recover the per-window valid count
        //      by running AveragePool(ones_like(x), count_include_pad=true, … same other
        //      attrs …) which yields valid_count/kernel_size at every output position;
        //      we then divide grad by that (the *kernel_size factor cancels).
        //
        //   2. Reshape the scaled grad [N, C, *output_spatial] into Col2Im's input
        //      layout [N, C * kernel_size, L] where L = prod(output_spatial). Done in
        //      three sub-steps using -1 / Concat-on-Shape so the channel count C never
        //      needs to be known statically:
        //        Reshape grad → [N, C, 1, L]
        //        Expand to     [N, C, kernel_size, L]
        //        Reshape to    [N, C * kernel_size, L]
        //
        //   3. Col2Im(col, image_shape=Shape(x)[2:], block_shape=kernel_shape,
        //             dilations, pads, strides) → [N, C, *input_spatial]. Overlapping
        //      windows naturally accumulate via Col2Im's fold semantics.
        //
        // auto_pad: only NotSet/null is supported here. SAME_UPPER/SAME_LOWER need the
        // forward AvgPool's resolved explicit pads (computed from spatial shape) which
        // are not preserved on the node. ceil_mode doesn't affect this gradient — the
        // window count is taken from grad's actual shape.

        internal static Variable?[] AveragePoolGradient(Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            var x = inputs[0]!;
            var grad = outputGrads[0]!;

            // === Decode attributes ===
            var kernelShape = (attributes.GetAttributeObj("kernel_shape") as long[])
                ?? throw new InvalidOperationException(
                    "AveragePool gradient: kernel_shape attribute is required.");
            var nDims = kernelShape.Length;

            var pads = attributes.GetAttributeObj("pads") as long[] ?? new long[2 * nDims];
            var strides = attributes.GetAttributeObj("strides") as long[]
                ?? Enumerable.Repeat(1L, nDims).ToArray();
            var dilations = attributes.GetAttributeObj("dilations") as long[]
                ?? Enumerable.Repeat(1L, nDims).ToArray();
            var countIncludePad = attributes.GetAttributeObj("count_include_pad") is bool cip && cip;
            var autoPad = attributes.GetAttributeObj("auto_pad") as AutoPad?;
            var isSameMode = autoPad is AutoPad.SameUpper or AutoPad.SameLower;
            var sameLower = autoPad is AutoPad.SameLower;

            // ceil_mode changes the output window count in a way neither the broadcast fast
            // path nor the Col2Im fold models (partial trailing windows). The old behavior was
            // a loud-but-cryptic runtime shape failure; fail with a clear error instead.
            var ceilModeAttr = attributes.GetAttributeObj("ceil_mode");
            if (ceilModeAttr is true || (ceilModeAttr is long cmLong && cmLong != 0))
                throw new AutoDiffNotSupportedException(ErrorCodes.AD003, AVERAGE_POOL,
                    "the gradient does not support ceil_mode=1 (the Col2Im-based fold assumes "
                    + "floor-mode window counts). This is an implementation limitation, not a "
                    + "mathematical one — use ceil_mode=0, or pad the input so the windows tile "
                    + "exactly.");

            long kernelSize = 1L;
            foreach (var k in kernelShape) kernelSize *= k;
            var hasPads = pads.Any(p => p != 0);
            var hasOverlap = !strides.SequenceEqual(kernelShape);
            var hasDilation = dilations.Any(d => d != 1);

            // === Fast path: stride==kernel, no pads, no dilations, NotSet/Valid ===
            // Non-overlapping windows means each input position is touched by exactly
            // one output window, so the gradient is a pure broadcast: replicate each
            // pooled-grad value across its kernel block. Uses only Reshape/Tile/Mul/Cast,
            // which keeps the gradient subgraph executable by every backend (ORT, CS
            // codegen, QuickExecutionEngine) — Col2Im is not in the QEE op set. SAME
            // modes can't take this path because the effective pads aren't known
            // statically (they depend on the runtime input spatial shape).
            if (!hasOverlap && !hasPads && !hasDilation && !isSameMode)
            {
                var scaleFast = OnnxOp.Cast(Globals.Scalar(1.0f / kernelSize), saturate: null, to: x.Type);
                var scaledFast = OnnxOp.Mul(grad, scaleFast);

                // Unsqueeze: insert a size-1 axis after each spatial axis of grad.
                //   grad rank = 2 + nDims, target rank = 2 + 2*nDims.
                //   axes (in the post-unsqueeze coords) = [3, 5, 7, …, 1 + 2*nDims].
                var unsqueezeAxes = new long[nDims];
                for (int i = 0; i < nDims; i++) unsqueezeAxes[i] = 3 + 2 * i;
                var unsqueezed = OnnxOp.Unsqueeze(scaledFast, Globals.Vector(unsqueezeAxes));

                // Tile by [1, 1, 1, kH, 1, kW, …] — replicate the inserted axes.
                var repeats = new long[2 + 2 * nDims];
                for (int i = 0; i < repeats.Length; i++) repeats[i] = 1;
                for (int i = 0; i < nDims; i++) repeats[3 + 2 * i] = kernelShape[i];
                var tiled = OnnxOp.Tile(unsqueezed, Globals.Vector(repeats));

                // Reshape back to input shape. With stride==kernel + no-pad + no-dilation,
                // total elements match exactly (Hp*kH = Hin, etc.).
                var gradXFast = OnnxOp.Reshape(tiled, OnnxOp.Shape(x), allowZero: false);
                return [gradXFast];
            }

            // === SAME mode: resolve auto_pad to dynamic explicit pads ===
            // ONNX AvgPool with SAME_UPPER/SAME_LOWER chooses pads at runtime so the
            // output spatial dims equal ceil(input / stride). For each spatial dim:
            //   pad_total = max(0, (output - 1) * stride + (kernel - 1) * dilation + 1 - input)
            //   SAME_UPPER: pad_begin = pad_total / 2,  pad_end = pad_total - pad_begin
            //   SAME_LOWER: pad_begin = (pad_total + 1) / 2, pad_end = pad_total - pad_begin
            // We can't pass these to Col2Im's static `pads` attribute, so instead we
            // inflate Col2Im's dynamic `image_shape` input to absorb them and Slice the
            // result down to x's spatial shape at the end.
            Variable[]? padBeginVars = null;
            Variable? paddedImageShape = null;
            if (isSameMode)
            {
                var xShapeSame = OnnxOp.Shape(x);
                var gradShapeSame = OnnxOp.Shape(grad);
                padBeginVars = new Variable[nDims];
                var imageShapeParts = new Variable[nDims];

                for (int d = 0; d < nDims; d++)
                {
                    var idx = Globals.Scalar(2L + d);
                    Tensor<int64> outDim = OnnxOp.Gather(gradShapeSame, idx, axis: 0);
                    Tensor<int64> inDim = OnnxOp.Gather(xShapeSame, idx, axis: 0);
                    var padTotalCandidate = (outDim - Globals.Scalar(1L)) * Globals.Scalar(strides[d])
                                          + Globals.Scalar((kernelShape[d] - 1) * dilations[d])
                                          + Globals.Scalar(1L) - inDim;
                    Tensor<int64> padTotal = OnnxOp.Max(
                        (Variable)Globals.Scalar(0L),
                        padTotalCandidate);

                    Tensor<int64> padBegin = sameLower
                        ? (padTotal + Globals.Scalar(1L)) / Globals.Scalar(2L)
                        : padTotal / Globals.Scalar(2L);
                    // Store the Immutable* graph value so later reads can downcast it.
                    padBeginVars[d] = (Variable)padBegin;

                    imageShapeParts[d] = OnnxOp.Reshape(
                        inDim + padTotal,
                        Globals.Vector(1L),
                        allowZero: false);
                }
                paddedImageShape = OnnxOp.Concat(imageShapeParts, axis: 0);
            }

            // === Step 1: scale grad by per-window divisor.
            // For SAME mode with count_include_pad=false the per-window divisor still
            // varies (windows near the edges touch padded zeros that aren't counted),
            // so we use the same AvgPool(ones, count_include_pad=true) trick — passing
            // the original auto_pad/pads so the divisor tensor has the right shape.
            Variable scaledGrad;
            if (countIncludePad || (!hasPads && !isSameMode))
            {
                // Constant divisor kernel_size everywhere.
                var scale = OnnxOp.Cast(Globals.Scalar(1.0f / kernelSize), saturate: null, to: x.Type);
                scaledGrad = OnnxOp.Mul(grad, scale);
            }
            else
            {
                // count_include_pad=false with non-zero pads or SAME mode: per-window
                // valid count via a re-forwarded AvgPool on ones.
                var onesLikeX = OnnxOp.Expand(
                    OnnxOp.Cast(Globals.Scalar(1.0f), saturate: null, to: x.Type),
                    OnnxOp.Shape(x));
                var countAvg = OnnxOp.AveragePool(
                    onesLikeX,
                    autoPad: autoPad,
                    ceilMode: attributes.GetAttributeObj("ceil_mode") as bool?,
                    countIncludePad: true,
                    dilations: dilations,
                    kernelShape: kernelShape,
                    pads: pads,
                    strides: strides);
                // countAvg = valid_count / kernel_size  ⇒  grad / valid_count
                //                                       = (grad / kernel_size) / countAvg
                var scale = OnnxOp.Cast(Globals.Scalar(1.0f / kernelSize), saturate: null, to: x.Type);
                scaledGrad = OnnxOp.Div(OnnxOp.Mul(grad, scale), countAvg);
            }

            // === Step 2: reshape grad to Col2Im input layout [N, C * kernelSize, L] ===
            // grad has shape [N, C, *output_spatial]; we don't know C statically, so we
            // build the target shape from Shape(grad).
            var gradShape = OnnxOp.Shape(scaledGrad);
            var nVec = OnnxOp.Slice(gradShape, Globals.Vector(0L), Globals.Vector(1L));
            var cVec = OnnxOp.Slice(gradShape, Globals.Vector(1L), Globals.Vector(2L));
            var outputSpatial = OnnxOp.Slice(gradShape, Globals.Vector(2L), Globals.Vector(2L + (long)nDims));
            var L = OnnxOp.ReduceProd(outputSpatial, keepdims: true);   // [1] = product(output_spatial)
            var oneVec = Globals.Vector(1L);
            var kernelSizeVec = Globals.Vector(kernelSize);

            // 2a. grad → [N, C, 1, L]
            var fourDShape = OnnxOp.Concat([nVec, cVec, oneVec, L], axis: 0);
            var reshaped4D = OnnxOp.Reshape(scaledGrad, fourDShape, allowZero: false);

            // 2b. Expand to [N, C, kernel_size, L]  (broadcasting via the size-1 axis)
            var expandShape = OnnxOp.Concat([nVec, cVec, kernelSizeVec, L], axis: 0);
            var expanded = OnnxOp.Expand(reshaped4D, expandShape);

            // 2c. Reshape to [N, C * kernel_size, L]
            var cTimesB = OnnxOp.Mul(cVec, kernelSizeVec);
            var colShape = OnnxOp.Concat([nVec, cTimesB, L], axis: 0);
            var col = OnnxOp.Reshape(expanded, colShape, allowZero: false);

            // === Step 3: Col2Im to fold contributions back into input shape ===
            // For NotSet/Valid: image_shape = x's spatial, pads = the static `pads` attr.
            // For SAME_*: image_shape = x_spatial + pad_begin + pad_end (dynamic, absorbs
            //             the dynamic padding), pads attr = zeros. We then Slice off the
            //             padding edges to recover x's exact spatial extent.
            var blockShape = Globals.Vector(kernelShape);
            Variable imageShape;
            long[] col2imPads;
            if (isSameMode)
            {
                imageShape = paddedImageShape!;
                col2imPads = new long[2 * nDims];   // zeros — padding absorbed by image_shape
            }
            else
            {
                imageShape = OnnxOp.Slice(OnnxOp.Shape(x), Globals.Vector(2L), Globals.Vector(2L + (long)nDims));
                col2imPads = pads;
            }
            var folded = OnnxOp.Col2Im(col, imageShape, blockShape, dilations, col2imPads, strides);

            // === Step 4: for SAME mode, slice off the padding edges ===
            if (!isSameMode) return [folded];

            // Build dynamic starts/ends. starts[d] = pad_begin[d];
            // ends[d] = pad_begin[d] + x_size[d] = padded_size[d] - pad_end[d]. We use
            // (padded - pad_end) because the latter is a plain Sub we already have. But
            // start + x_size is simpler — compute that.
            var xShapeFinal = OnnxOp.Shape(x);
            var startsParts = new Variable[nDims];
            var endsParts = new Variable[nDims];
            var axesArray = new long[nDims];
            for (int d = 0; d < nDims; d++)
            {
                Tensor<int64> xSizeD = OnnxOp.Gather(xShapeFinal, Globals.Scalar(2L + d), axis: 0);
                Tensor<int64> pb = (Variable)padBeginVars![d];
                startsParts[d] = OnnxOp.Reshape(pb, Globals.Vector(1L), allowZero: false);
                endsParts[d] = OnnxOp.Reshape(pb + xSizeD, Globals.Vector(1L), allowZero: false);
                axesArray[d] = 2L + d;
            }
            var startsVec = OnnxOp.Concat(startsParts, axis: 0);
            var endsVec = OnnxOp.Concat(endsParts, axis: 0);
            var gradX = OnnxOp.Slice(folded, startsVec, endsVec, Globals.Vector(axesArray), null);
            return [gradX];
        }

        // ===== MaxPool (variadic registration) =====
        //
        // MaxPool(x) → y [, indices]. The gradient routes each pooled-output gradient to the
        // input element that won its window. We recompute the forward pass with the Indices
        // output enabled — per the ONNX spec (storage_order=0) each index is the row-major
        // flat offset of the winning element into the WHOLE input tensor — then scatter the
        // output gradient onto a flat zeros tensor with Add reduction and reshape back to
        // x's shape. Because the indices come from the forward op itself, every attribute
        // combination the forward supports (overlapping strides, pads, ceil_mode, dilations,
        // SAME_* auto_pad) is handled exactly. Ties route to the first max element (the
        // index the forward op reports), matching the standard subgradient convention.
        //
        // storage_order=1 (column-major within each spatial slice) would need an index
        // remap and is not supported.

        internal static Variable?[] MaxPoolGradient(Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            var x = inputs[0]!;
            // grad of y. May be null when only the forward Indices output was consumed
            // downstream (int64 — non-differentiable): no gradient flows to x then.
            var grad = outputGrads[0];
            if (grad is null) return [null];

            // Read attributes
            var autoPad = attributes.GetAttributeObj("auto_pad") as AutoPad?;
            var ceilMode = attributes.GetAttributeObj("ceil_mode") as bool?;
            var dilations = attributes.GetAttributeObj("dilations") as long[];
            var kernelShape = attributes.GetAttributeObj("kernel_shape") as long[];
            var padsList = attributes.GetAttributeObj("pads") as long[];
            var storageOrder = attributes.GetAttributeObj("storage_order") as long?;
            var strides = attributes.GetAttributeObj("strides") as long[];

            if ((storageOrder ?? 0L) != 0L)
                throw new AutoDiffNotSupportedException(ErrorCodes.AD003, MAX_POOL,
                    "the gradient is only implemented for storage_order=0 (row-major indices); "
                    + "storage_order=1 would need a column-major index remap. This is an "
                    + "implementation limitation, not a mathematical one.");

            // Recompute the forward pass with the Indices output to learn each window's argmax.
            var (_, indices) = OnnxOp.MaxPoolWithIndices(x,
                autoPad: autoPad, ceilMode: ceilMode,
                dilations: dilations, kernelShape: kernelShape,
                pads: padsList, storageOrder: storageOrder, strides: strides);

            // Flat zeros covering the whole input tensor: [numel(x)].
            var xShape = OnnxOp.Shape(x);
            var totalSize = OnnxOp.ReduceProd(xShape, keepdims: true);   // [1]
            var zerosFlat = OnnxOp.Expand(
                OnnxOp.Cast(Globals.Scalar(0.0f), saturate: null, to: x.Type),
                totalSize);

            // Scatter each output-position gradient onto its argmax's flat offset. Add
            // reduction accumulates when overlapping windows share the same argmax.
            var gradFlat = OnnxOp.Reshape(grad, Globals.Vector(-1L), allowZero: false);
            var idxFlat = OnnxOp.Reshape(indices, Globals.Vector(-1L), allowZero: false);
            var scattered = OnnxOp.ScatterElements(zerosFlat, idxFlat, gradFlat,
                axis: 0, reduction: ScatterNDReduction.Add);

            var gradX = OnnxOp.Reshape(scattered, xShape, allowZero: false);

            return [gradX];
        }
    }
}

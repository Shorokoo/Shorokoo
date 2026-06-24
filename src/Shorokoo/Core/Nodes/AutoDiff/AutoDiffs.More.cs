using System;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Globals;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.AutoDiff
{
    internal static partial class AutoDiffs
    {
        // ===== Abs =====

        [AutoDiff(ABS)]
        public static IVariable?[] Abs<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // d|x|/dx = sign(x)
            return [grad * (Tensor<T>)OnnxOp.Sign(x)];
        }

        // ===== Reciprocal =====

        [AutoDiff(RECIPROCAL)]
        public static IVariable?[] Reciprocal<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // d(1/x)/dx = -1/x²
            return [-(grad / (x * x))];
        }

        // ===== LeakyRelu =====

        [AutoDiff(LEAKY_RELU)]
        public static IVariable?[] LeakyRelu<T>(Tensor<T> x, Tensor<T> grad, float? alpha) where T : IVarType
        {
            // d(leaky_relu(x))/dx = 1 if x > 0, alpha if x <= 0
            var effectiveAlpha = alpha ?? 0.01f;
            var zero = TypedConst(0.0f, x);
            var alphaConst = TypedConst(effectiveAlpha, x);
            var one = TypedConst(1.0f, x);
            var mask = x > zero;
            var slope = (Tensor<T>)OnnxOp.Where(mask, one, alphaConst);
            return [grad * slope];
        }

        // ===== Gelu =====

        [AutoDiff(GELU)]
        public static IVariable?[] Gelu<T>(Tensor<T> x, Tensor<T> grad, GeluApproximate? approximate) where T : IVarType
        {
            var half = TypedConst(0.5f, x);
            var one = TypedConst(1.0f, x);

            if (approximate == GeluApproximate.Tanh)
            {
                // approximate="tanh": gelu(x) = 0.5x(1 + tanh(u)), u = √(2/π)(x + 0.044715x³)
                // gelu'(x) = 0.5(1 + tanh(u)) + 0.5x(1 − tanh²(u))·√(2/π)(1 + 3·0.044715x²)
                // (matches the tanh-approximation FORWARD — using the exact-erf derivative
                // here would silently mismatch the forward by up to ~1e-3).
                var c = TypedConst(0.7978845608028654f, x);   // sqrt(2/pi)
                var k = TypedConst(0.044715f, x);
                var three = TypedConst(3.0f, x);
                var xSq = x * x;
                var u = c * (x + k * xSq * x);
                var t = (Tensor<T>)OnnxOp.Tanh(u);
                var sechSq = one - t * t;
                var du = c * (one + three * k * xSq);
                return [grad * (half * (one + t) + half * x * sechSq * du)];
            }

            // approximate="none" (default): gelu(x) = x * Φ(x), Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
            // gelu'(x) = Φ(x) + x * φ(x) where φ(x) = exp(-x²/2) / sqrt(2π)
            var sqrt2 = TypedConst(MathF.Sqrt(2.0f), x);
            var invSqrt2Pi = TypedConst(1.0f / MathF.Sqrt(2.0f * MathF.PI), x);
            var negHalf = TypedConst(-0.5f, x);

            var cdf = half * (one + (Tensor<T>)OnnxOp.Erf(x / sqrt2));
            var pdf = (x * x * negHalf).Exp() * invSqrt2Pi;

            return [grad * (cdf + x * pdf)];
        }

        // ===== Elu =====

        [AutoDiff(ELU)]
        public static IVariable?[] Elu<T>(Tensor<T> x, Tensor<T> grad, float? alpha) where T : IVarType
        {
            // elu(x) = x if x > 0, alpha * (exp(x) - 1) if x <= 0
            // elu'(x) = 1 if x > 0, alpha * exp(x) if x <= 0
            var effectiveAlpha = alpha ?? 1.0f;
            var zero = TypedConst(0.0f, x);
            var one = TypedConst(1.0f, x);
            var alphaConst = TypedConst(effectiveAlpha, x);
            var mask = x > zero;
            var eluGrad = (Tensor<T>)OnnxOp.Where(mask, one, alphaConst * x.Exp());
            return [grad * eluGrad];
        }

        // ===== Selu =====

        [AutoDiff(SELU)]
        public static IVariable?[] Selu<T>(Tensor<T> x, Tensor<T> grad, float? alpha, float? gamma) where T : IVarType
        {
            // selu(x) = gamma * (x if x > 0, alpha * (exp(x) - 1) if x <= 0)
            // selu'(x) = gamma if x > 0, gamma * alpha * exp(x) if x <= 0
            var effectiveAlpha = alpha ?? 1.67326319217681884765625f;
            var effectiveGamma = gamma ?? 1.0507010221481323242187f;
            var zero = TypedConst(0.0f, x);
            var one = TypedConst(1.0f, x);
            var alphaConst = TypedConst(effectiveAlpha, x);
            var gammaConst = TypedConst(effectiveGamma, x);
            var mask = x > zero;
            var seluGrad = gammaConst * (Tensor<T>)OnnxOp.Where(mask, one, alphaConst * x.Exp());
            return [grad * seluGrad];
        }

        // ===== Celu =====

        [AutoDiff(CELU)]
        public static IVariable?[] Celu<T>(Tensor<T> x, Tensor<T> grad, float? alpha) where T : IVarType
        {
            // celu(x) = max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
            // celu'(x) = 1 if x > 0, exp(x/alpha) if x <= 0
            var effectiveAlpha = alpha ?? 1.0f;
            var zero = TypedConst(0.0f, x);
            var one = TypedConst(1.0f, x);
            var alphaConst = TypedConst(effectiveAlpha, x);
            var mask = x > zero;
            var celuGrad = (Tensor<T>)OnnxOp.Where(mask, one, (x / alphaConst).Exp());
            return [grad * celuGrad];
        }

        // ===== Flatten =====

        [AutoDiff(FLATTEN)]
        public static IVariable?[] Flatten<T>(Tensor<T> input, Tensor<T> grad, long? axis) where T : IVarType
        {
            // Gradient of Flatten: reshape grad back to original shape
            var originalShape = input.DShape;
            return [(Tensor<T>)OnnxOp.Reshape(grad, originalShape, allowZero: false)];
        }

        // ===== Squeeze =====

        [AutoDiff(SQUEEZE)]
        public static IVariable?[] Squeeze<T1, T2>(Tensor<T1> data, Tensor<T2> axes, Tensor<T1> grad)
            where T1 : IVarType
            where T2 : IVarType
        {
            // Gradient of Squeeze: reshape grad back to original shape
            var originalShape = data.DShape;
            return [(Tensor<T1>)OnnxOp.Reshape(grad, originalShape, allowZero: false), null];
        }

        // ===== Unsqueeze =====

        [AutoDiff(UNSQUEEZE)]
        public static IVariable?[] Unsqueeze<T1, T2>(Tensor<T1> data, Tensor<T2> axes, Tensor<T1> grad)
            where T1 : IVarType
            where T2 : IVarType
        {
            // Gradient of Unsqueeze: reshape grad back to original shape
            var originalShape = data.DShape;
            return [(Tensor<T1>)OnnxOp.Reshape(grad, originalShape, allowZero: false), null];
        }

        // ===== Expand =====

        [AutoDiff(EXPAND)]
        public static IVariable?[] Expand<T1, T2>(Tensor<T1> input, Tensor<T2> shape, Tensor<T1> grad)
            where T1 : IVarType
            where T2 : IVarType
        {
            // Gradient of Expand: reverse broadcast back to original shape
            var originalShape = input.DShape;
            return [ReverseBroadcast(grad, originalShape), null];
        }

        // ===== Clip =====

        [AutoDiff(CLIP)]
        public static IVariable?[] Clip<T>(Tensor<T> input, Tensor<T> min, Tensor<T> max, Tensor<T> grad) where T : IVarType
        {
            // d(clip(x, min, max))/dx = 1 where min <= x <= max, 0 otherwise
            // Apply mask in two steps: zero out where x < min, then zero out where x > max
            // Note: min and max are non-optional in this framework's Clip API
            var zero = TypedConst(0.0f, input);
            var result = grad;
            if (min is not null)
                result = (Tensor<T>)OnnxOp.Where(input < min, zero, result);
            if (max is not null)
                result = (Tensor<T>)OnnxOp.Where(input > max, zero, result);
            return [result, null, null];
        }

        // ===== Where =====

        [AutoDiff(WHERE)]
        public static IVariable?[] Where<T>(Tensor<bit> condition, Tensor<T> x, Tensor<T> y, Tensor<T> grad) where T : IVarType
        {
            // Gradient flows to x where condition is true, to y where condition is false
            var zero = TypedConst(0.0f, x);
            var xGrad = ReverseBroadcast((Tensor<T>)OnnxOp.Where(condition, grad, zero), x.DShape);
            var yGrad = ReverseBroadcast((Tensor<T>)OnnxOp.Where(condition, zero, grad), y.DShape);
            return [null, xGrad, yGrad];
        }

        // ===== Erf =====

        [AutoDiff(ERF)]
        public static IVariable?[] Erf<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // d(erf(x))/dx = 2/√π · exp(-x²)
            var twoOverSqrtPi = TypedConst(2.0f / MathF.Sqrt(MathF.PI), x);
            var negOne = TypedConst(-1.0f, x);
            return [grad * twoOverSqrtPi * (negOne * x * x).Exp()];
        }

        // ===== Sign =====

        [AutoDiff(SIGN)]
        public static IVariable?[] Sign<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // sign(x) is not differentiable; gradient is zero everywhere
            var zero = TypedConst(0.0f, x);
            return [zero * grad];
        }

        // ===== Ceil =====

        [AutoDiff(CEIL)]
        public static IVariable?[] Ceil<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // ceil(x) is piecewise constant; gradient is zero
            var zero = TypedConst(0.0f, x);
            return [zero * grad];
        }

        // ===== Floor =====

        [AutoDiff(FLOOR)]
        public static IVariable?[] Floor<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // floor(x) is piecewise constant; gradient is zero
            var zero = TypedConst(0.0f, x);
            return [zero * grad];
        }

        // ===== Pad (constant mode) =====

        [AutoDiff(PAD)]
        public static IVariable?[] Pad<T1, T2, T3>(
            Tensor<T1> data, Tensor<T2> pads, Tensor<T1> constantValue, Tensor<T3> axes,
            Tensor<T1> grad, PadMode? mode)
            where T1 : IVarType
            where T2 : IVarType
            where T3 : IVarType
        {
            // For constant padding, gradient of data = slice of grad that removes the padded
            // regions. Non-constant modes (reflect/edge/wrap) re-emit border values, so their
            // adjoint must ALSO scatter-add the border grads back onto interior positions —
            // reusing the constant-mode slice for them silently mis-attributes those grads.
            if (mode is not null && mode != PadMode.Constant)
                throw new AutoDiffNotSupportedException(ErrorCodes.AD003, PAD,
                    $"the gradient is only implemented for mode='constant' (got '{mode}'): the "
                    + "reflect/edge/wrap adjoints (slice + scatter-add of the border grads) are "
                    + "not implemented. This is an implementation limitation, not a mathematical "
                    + "one.");

            var inputShape = data.DShape;
            var padsCast = (Tensor<int64>)OnnxOp.Cast(pads, saturate: null, to: DType.Int64);

            if (axes is null)
            {
                // pads = [begin_0, ..., begin_N, end_0, ..., end_N]
                var numDimsShape = inputShape.DShape; // Vector<int64> of shape [1], value = number of dims
                var begins = (Tensor<int64>)OnnxOp.Slice(padsCast, Vector(0L), numDimsShape);
                var ends = begins + inputShape;
                return [(Tensor<T1>)OnnxOp.Slice(grad, begins, ends), null, null, null];
            }
            else
            {
                // pads only describes specified axes
                var axesCast = (Tensor<int64>)OnnxOp.Cast(axes, saturate: null, to: DType.Int64);
                var numAxesShape = axesCast.DShape; // [1] containing number of axes
                var begins = (Tensor<int64>)OnnxOp.Slice(padsCast, Vector(0L), numAxesShape);
                var dimSizes = (Tensor<int64>)OnnxOp.Gather(inputShape, axesCast, axis: 0);
                var ends = begins + dimSizes;
                return [(Tensor<T1>)OnnxOp.Slice(grad, begins, ends, axesCast), null, null, null];
            }
        }

        // ===== Slice =====

        [AutoDiff(SLICE)]
        public static IVariable?[] Slice<T1, T2>(
            Tensor<T1> data, Tensor<T2> starts, Tensor<T2> ends, Tensor<T2> axes, Tensor<T2> steps,
            Tensor<T1> grad)
            where T1 : IVarType
            where T2 : IVarType
        {
            // Slice extracts a region of data. Gradient = pad grad with zeros back to original shape.
            // For step=1 (default): use Pad to place gradient back at the slice position.
            var inputShape = data.DShape;
            var startsCast = (Tensor<int64>)OnnxOp.Cast(starts, saturate: null, to: DType.Int64);
            var endsCast = (Tensor<int64>)OnnxOp.Cast(ends, saturate: null, to: DType.Int64);

            if (steps is not null)
            {
                // steps is a runtime input whose values aren't known at gradient-build time,
                // so whenever it's wired build the gradient with the exact general adjoint:
                // run the SAME Slice over a tensor of flat element offsets (Range reshaped to
                // data's shape) to recover precisely which positions the forward selected —
                // including strided, negative-step, negative-start/end and clamped configs —
                // then scatter the gradient onto those flat offsets and reshape back.
                var stepsCast = (Tensor<int64>)OnnxOp.Cast(steps, saturate: null, to: DType.Int64);
                var axesCastOpt = axes is null
                    ? null
                    : (Tensor<int64>)OnnxOp.Cast(axes, saturate: null, to: DType.Int64);

                var flatCount = OnnxOp.ReduceProd(inputShape, keepdims: true);          // [1]
                var flatRange = OnnxOp.Range(
                    Scalar(0L), OnnxOp.Squeeze(flatCount, Vector(0L)), Scalar(1L));      // [numel]
                var positions = OnnxOp.Reshape(flatRange, inputShape, allowZero: false); // data-shaped int64

                var slicedPositions = OnnxOp.Slice(
                    positions, startsCast, endsCast, axesCastOpt, stepsCast);            // mirrors grad's shape
                var scatterIndices = OnnxOp.Unsqueeze(
                    OnnxOp.Reshape(slicedPositions, Vector(-1L), allowZero: false),
                    Vector(-1L));                                                        // [M, 1]
                var gradFlat = OnnxOp.Reshape(grad, Vector(-1L), allowZero: false);      // [M]

                var zerosFlat = OnnxOp.Expand(TypedConst(0.0f, data), flatCount);        // [numel]
                var scattered = OnnxOp.ScatterND(
                    zerosFlat, scatterIndices, gradFlat, ScatterNDReduction.Add);
                var gradData = OnnxOp.Reshape(scattered, inputShape, allowZero: false);
                return [(Tensor<T1>)gradData, null, null, null, null];
            }

            if (axes is null)
            {
                // All axes sliced: pads = [start_0, ..., start_N, (dim_0 - end_0), ..., (dim_N - end_N)]
                var clampedStarts = (Tensor<int64>)OnnxOp.Max(startsCast, (Tensor<int64>)OnnxOp.Expand(Scalar(0L), startsCast.DShape));
                var clampedEnds = (Tensor<int64>)OnnxOp.Min(endsCast, inputShape);
                var endPads = inputShape - clampedEnds;
                var pads = (Tensor<int64>)OnnxOp.Concat([clampedStarts, endPads], axis: 0);
                return [(Tensor<T1>)OnnxOp.Pad(grad, pads, mode: PadMode.Constant), null, null, null, null];
            }
            else
            {
                // Only specified axes are sliced
                var axesCast = (Tensor<int64>)OnnxOp.Cast(axes, saturate: null, to: DType.Int64);
                var dimSizes = (Tensor<int64>)OnnxOp.Gather(inputShape, axesCast, axis: 0);
                var clampedStarts = (Tensor<int64>)OnnxOp.Max(startsCast, (Tensor<int64>)OnnxOp.Expand(Scalar(0L), startsCast.DShape));
                var clampedEnds = (Tensor<int64>)OnnxOp.Min(endsCast, dimSizes);
                var endPads = dimSizes - clampedEnds;
                var pads = (Tensor<int64>)OnnxOp.Concat([clampedStarts, endPads], axis: 0);
                return [(Tensor<T1>)OnnxOp.Pad(grad, pads, axes: axesCast, mode: PadMode.Constant), null, null, null, null];
            }
        }

        // ===== Gather =====

        [AutoDiff(GATHER)]
        public static IVariable?[] Gather<T1, T2>(Tensor<T1> data, Tensor<T2> indices, Tensor<T1> grad, long? axis)
            where T1 : IVarType
            where T2 : IVarType
        {

            // Gather(data, indices, axis=a): y = data[indices] along axis a
            // Gradient w.r.t. data: scatter grad back to the original positions using ScatterND.
            var effectiveAxis = axis ?? 0;
            var dataShape = data.DShape;

            // Create zeros with same shape and type as data
            var zero = TypedConst(0.0f, data);
            var zeros = (Tensor<T1>)OnnxOp.Expand(zero, dataShape);

            // Flatten indices to 1D for ScatterND and add trailing index-depth dimension
            var indicesInt = (Tensor<int64>)OnnxOp.Cast(indices, saturate: null, to: DType.Int64);
            var flatIndices = (Tensor<int64>)OnnxOp.Reshape(indicesInt, Vector(-1L), allowZero: false);
            var scatterIndices = (Tensor<int64>)OnnxOp.Unsqueeze(flatIndices, Vector(-1L)); // [M, 1]

            if (effectiveAxis == 0)
            {
                // Always flatten grad's leading index dimensions to [M_total, ...]: this aligns
                // grad with scatterIndices (which was unconditionally flattened to [M_total, 1]
                // above). For 1D indices the reshape is a no-op; for multi-dim indices it
                // correctly collapses the leading index dims. Using an unconditional reshape
                // also avoids depending on indices.Rank being statically known — Reshape-built
                // indices commonly have a null Rank even though their actual rank is > 1.
                var tailShape = (Tensor<int64>)OnnxOp.Shape(data, start: 1);
                var newShape = (Tensor<int64>)OnnxOp.Concat([Vector(-1L), tailShape], axis: 0);
                grad = (Tensor<T1>)OnnxOp.Reshape(grad, newShape, allowZero: false);

                var result = (Tensor<T1>)OnnxOp.ScatterND(zeros, scatterIndices, grad, ScatterNDReduction.Add);
                return [result, null];
            }

            // Non-zero axis: ScatterND scatters along the LEADING dim, so move the target
            // axis to the front, scatter, then move it back.
            var dataRank = data.Rank;
            long normalizedAxis;
            if (effectiveAxis >= 0)
                normalizedAxis = effectiveAxis;
            else if (dataRank is not null)
                normalizedAxis = effectiveAxis + dataRank.Value;
            else
                throw new System.NotSupportedException(
                    "Gather gradient: a negative axis requires a statically-known data rank; pass a non-negative axis.");

            var indicesRankVal = indices.Rank ?? 1;

            if (dataRank is null)
            {
                // Rank is statically unknown (common for computed intermediates such as
                // Expand/Reshape/MatMul results). A rank-length transpose perm can't be
                // built, so collapse `data` to 3-D [L, K, R] around the (static, non-negative)
                // target axis — L = product of the dims before the axis, K = the axis dim,
                // R = the rest — move the K dim to the front with a fixed [1,0,2] perm,
                // ScatterND, then move it back and restore the original shape. Only the
                // statically-known axis is needed, no rank. `scatterIndices` is already
                // flattened to [M_total, 1], so the same collapse handles 1-D and multi-dim
                // gather indices alike (M_total = product of all the gathered index dims).
                var dataShapeVec = data.DShape;
                var beforeDims = (Tensor<int64>)OnnxOp.Shape(data, start: 0, end: normalizedAxis);                // non-empty (axis >= 1)
                var axisDim = (Tensor<int64>)OnnxOp.Shape(data, start: normalizedAxis, end: normalizedAxis + 1);  // [K]
                var lead = (Tensor<int64>)OnnxOp.ReduceProd(beforeDims, Vector(0L), keepdims: true, noopWithEmptyAxes: false); // [L]
                var gatheredCount = (Tensor<int64>)OnnxOp.Shape(scatterIndices, start: 0, end: 1);                // [M_total]

                // The trailing [-1] lets Reshape infer R, so the (possibly empty) after-axis
                // dims never need their own product.
                var data3DShape = (Tensor<int64>)OnnxOp.Concat([lead, axisDim, Vector(-1L)], axis: 0);         // [L, K, -1]
                var grad3DShape = (Tensor<int64>)OnnxOp.Concat([lead, gatheredCount, Vector(-1L)], axis: 0);   // [L, M_total, -1]

                var zeros3D = (Tensor<T1>)OnnxOp.Reshape(zeros, data3DShape, allowZero: false);   // (L, K, R)
                var grad3D = (Tensor<T1>)OnnxOp.Reshape(grad, grad3DShape, allowZero: false);     // (L, M_total, R)

                var zeros3DKFront = (Tensor<T1>)zeros3D.Transpose(1L, 0L, 2L); // (K, L, R)
                var grad3DKFront = (Tensor<T1>)grad3D.Transpose(1L, 0L, 2L);   // (M_total, L, R)
                var scattered3D = (Tensor<T1>)OnnxOp.ScatterND(zeros3DKFront, scatterIndices, grad3DKFront, ScatterNDReduction.Add); // (K, L, R)
                var scattered3DRestored = (Tensor<T1>)scattered3D.Transpose(1L, 0L, 2L); // (L, K, R)
                return [(Tensor<T1>)OnnxOp.Reshape(scattered3DRestored, dataShapeVec, allowZero: false), null];
            }

            // perm: [axis, 0, 1, ..., axis-1, axis+1, ..., rank-1]
            var perm = new long[dataRank.Value];
            perm[0] = normalizedAxis;
            var idx = 1;
            for (var i = 0; i < dataRank.Value; i++)
            {
                if (i != normalizedAxis)
                    perm[idx++] = i;
            }

            var inversePerm = new long[perm.Length];
            for (var i = 0; i < perm.Length; i++)
                inversePerm[perm[i]] = i;

            var zerosT = zeros.Transpose(perm);

            // For 1D/scalar indices: grad has same rank as data
            if (indicesRankVal <= 1)
            {
                var gradT = grad.Transpose(perm);
                var scattered = (Tensor<T1>)OnnxOp.ScatterND(zerosT, scatterIndices, gradT, ScatterNDReduction.Add);
                return [scattered.Transpose(inversePerm), null];
            }

            // Multi-dim indices with non-zero axis:
            // Build a grad permutation that moves the Q index dims (at positions a..a+Q-1) to front
            var gradRank = dataRank.Value + indicesRankVal - 1;
            var gradPerm = new long[gradRank];
            var gIdx = 0;
            for (var i = 0; i < indicesRankVal; i++)
                gradPerm[gIdx++] = normalizedAxis + i;
            for (var i = 0; i < gradRank; i++)
            {
                if (i < normalizedAxis || i >= normalizedAxis + indicesRankVal)
                    gradPerm[gIdx++] = i;
            }

            var gradT2 = grad.Transpose(gradPerm);

            // Flatten the leading index dims of grad to [M_total, remaining_dims...]
            var tailShape2 = (Tensor<int64>)OnnxOp.Shape(zerosT, start: 1);
            var newShape2 = (Tensor<int64>)OnnxOp.Concat([Vector(-1L), tailShape2], axis: 0);
            gradT2 = (Tensor<T1>)OnnxOp.Reshape(gradT2, newShape2, allowZero: false);

            var scattered2 = (Tensor<T1>)OnnxOp.ScatterND(zerosT, scatterIndices, gradT2, ScatterNDReduction.Add);
            return [scattered2.Transpose(inversePerm), null];
        }

        // ===== GatherND =====

        [AutoDiff(GATHER_ND)]
        public static IVariable?[] GatherND<T1, T2>(Tensor<T1> data, Tensor<T2> indices, Tensor<T1> grad, long? batchDims)
            where T1 : IVarType
            where T2 : IVarType
        {
            // GatherND(data, indices) selects slices from data using multi-dimensional indices.
            // Gradient w.r.t. data: scatter grad back to original positions using ScatterND with Add reduction.
            var effectiveBatchDims = batchDims ?? 0;
            if (effectiveBatchDims != 0)
                throw new AutoDiffNotSupportedException(ErrorCodes.AD003, GATHER_ND,
                    $"the gradient is only implemented for batch_dims=0 (got {effectiveBatchDims}): "
                    + "the batched ScatterND adjoint is not implemented. This is an implementation "
                    + "limitation, not a mathematical one.");

            var dataShape = data.DShape;
            var zero = TypedConst(0.0f, data);
            var zeros = (Tensor<T1>)OnnxOp.Expand(zero, dataShape);

            var indicesInt = (Tensor<int64>)OnnxOp.Cast(indices, saturate: null, to: DType.Int64);
            var result = (Tensor<T1>)OnnxOp.ScatterND(zeros, indicesInt, grad, ScatterNDReduction.Add);
            return [result, null];
        }

        // ===== Tile =====

        [AutoDiff(TILE)]
        public static IVariable?[] Tile<T1, T2>(Tensor<T1> input, Tensor<T2> repeats, Tensor<T1> grad)
            where T1 : IVarType
            where T2 : IVarType
        {
            // Tile(input, repeats): output[i*d+j] = input[j] for each repeat dimension
            // Gradient: reshape grad to interleave repeat and data dims, then sum over repeat dims.
            // grad shape: [r0*d0, r1*d1, ..., rn*dn]
            // Reshape to: [r0, d0, r1, d1, ..., rn, dn]
            // ReduceSum over axes [0, 2, 4, ...] → [d0, d1, ..., dn]
            var dataShape = input.DShape;
            var repeatsInt = (Tensor<int64>)OnnxOp.Cast(repeats, saturate: null, to: DType.Int64);

            // Build interleaved shape: [r0, d0, r1, d1, ...]
            var repeatsUnsqueezed = (Tensor<int64>)OnnxOp.Unsqueeze(repeatsInt, Vector(1L));
            var shapeUnsqueezed = (Tensor<int64>)OnnxOp.Unsqueeze(dataShape, Vector(1L));
            var interleaved = (Tensor<int64>)OnnxOp.Concat([repeatsUnsqueezed, shapeUnsqueezed], axis: 1);
            var interleavedFlat = (Tensor<int64>)OnnxOp.Reshape(interleaved, Vector(-1L), allowZero: false);

            // Reshape grad to interleaved shape
            var reshaped = (Tensor<T1>)OnnxOp.Reshape(grad, interleavedFlat, allowZero: false);

            // Sum over even-indexed axes [0, 2, 4, ..., 2*(n-1)] using dynamic range
            var dataRankScalar = dataShape.DShape.Squeeze();
            var twoN = dataRankScalar + dataRankScalar;
            var evenAxes = VectorRange(Scalar(0L), twoN, Scalar(2L));
            var reduced = (Tensor<T1>)OnnxOp.ReduceSum(reshaped, evenAxes, keepdims: false);

            return [reduced, null];
        }

        // ===== ScatterND =====

        [AutoDiff(SCATTER_ND)]
        public static IVariable?[] ScatterND<T1, T2>(
            Tensor<T1> data, Tensor<T2> indices, Tensor<T1> updates,
            Tensor<T1> grad, ScatterNDReduction? reduction)
            where T1 : IVarType
            where T2 : IVarType
        {
            // ScatterND(data, indices, updates, reduction) → output
            // Gradient depends on reduction mode. mul/min/max reductions have data-dependent
            // adjoints (product-quotient and argmin/argmax routing respectively) that are not
            // implemented — falling into the 'none' branch for them would be silently wrong.
            var effectiveReduction = reduction ?? ScatterNDReduction.None;
            if (effectiveReduction is ScatterNDReduction.Mul or ScatterNDReduction.Min or ScatterNDReduction.Max)
                throw new AutoDiffNotSupportedException(ErrorCodes.AD003, SCATTER_ND,
                    $"the gradient is only implemented for reduction='none'/'add' (got "
                    + $"'{effectiveReduction}'). This is an implementation limitation, not a "
                    + "mathematical one.");

            var indicesInt = (Tensor<int64>)OnnxOp.Cast(indices, saturate: null, to: DType.Int64);

            // grad_updates = GatherND(grad, indices) always
            var gradUpdates = (Tensor<T1>)OnnxOp.GatherND(grad, indicesInt);

            Tensor<T1> gradData;
            if (effectiveReduction == ScatterNDReduction.Add)
            {
                // "add": data values pass through entirely, updates are added
                // grad_data = grad (unchanged)
                gradData = grad;
            }
            else
            {
                // "none": data values at indices are replaced
                // grad_data = grad with positions at indices zeroed out
                var zero = TypedConst(0.0f, data);
                var zeroUpdates = (Tensor<T1>)OnnxOp.Expand(zero, updates.DShape);
                gradData = (Tensor<T1>)OnnxOp.ScatterND(grad, indicesInt, zeroUpdates, ScatterNDReduction.None);
            }

            return [gradData, null, gradUpdates];
        }

        // ===== DepthToSpace =====

        [AutoDiff(DEPTH_TO_SPACE)]
        public static IVariable?[] DepthToSpace<T>(
            Tensor<T> input, Tensor<T> grad, long? blocksize, DepthColumnRowMode? mode)
            where T : IVarType
        {
            // DepthToSpace: [N, C*r², H, W] → [N, C, H*r, W*r]
            // Gradient: inverse transform (SpaceToDepth)
            // SpaceToDepth: [N, C, H*r, W*r] → [N, C*r², H, W]
            var r = blocksize ?? 2;
            var effectiveMode = mode ?? DepthColumnRowMode.DCR;
            var originalShape = input.DShape;

            // Get grad shape components: [N, C, H*r, W*r]
            var gradShape = grad.DShape;
            var nVec = (Tensor<int64>)OnnxOp.Slice(gradShape, Vector(0L), Vector(1L));
            var cVec = (Tensor<int64>)OnnxOp.Slice(gradShape, Vector(1L), Vector(2L));
            var hrVec = (Tensor<int64>)OnnxOp.Slice(gradShape, Vector(2L), Vector(3L));
            var wrVec = (Tensor<int64>)OnnxOp.Slice(gradShape, Vector(3L), Vector(4L));
            var rVec = Vector(r);
            var hVec = hrVec / rVec;
            var wVec = wrVec / rVec;

            // Step 1: Reshape grad [N, C, H*r, W*r] → [N, C, H, r, W, r]
            var intermediateShape = (Tensor<int64>)OnnxOp.Concat(
                [nVec, cVec, hVec, rVec, wVec, rVec], axis: 0);
            var reshaped = (Tensor<T>)OnnxOp.Reshape(grad, intermediateShape, allowZero: false);

            // Step 2: Transpose to rearrange back to depth-channel layout
            long[] perm;
            if (effectiveMode == DepthColumnRowMode.DCR)
                // [N, C, H, r, W, r] → [N, r, r, C, H, W]: perm = [0, 3, 5, 1, 2, 4]
                perm = [0, 3, 5, 1, 2, 4];
            else
                // CRD: [N, C, H, r, W, r] → [N, C, r, r, H, W]: perm = [0, 1, 3, 5, 2, 4]
                perm = [0, 1, 3, 5, 2, 4];

            var transposed = (Tensor<T>)OnnxOp.Transpose(reshaped, perm: perm);

            // Step 3: Reshape to original input shape [N, C*r², H, W]
            var result = (Tensor<T>)OnnxOp.Reshape(transposed, originalShape, allowZero: false);

            return [result];
        }

        // ===== Mod =====

        [AutoDiff(MOD)]
        public static IVariable?[] Mod<T>(Tensor<T> a, Tensor<T> b, Tensor<T> grad, bool? fmod) where T : IVarType
        {
            // Float Mod is piecewise linear in `a` with slope 1 (a.e.) and piecewise linear
            // in `b` with slope -q (a.e.), where q is the quotient the op rounds:
            //   fmod=1 (C fmod):    Mod(a, b) = a - trunc(a/b)·b → d/da = 1, d/db = -trunc(a/b)
            //   fmod=0 (numpy mod): Mod(a, b) = a - floor(a/b)·b → d/da = 1, d/db = -floor(a/b)
            // This is the PyTorch convention for fmod/remainder. Both quotients come out of
            // the dtype-preserving identity q = (a - Mod(a, b, fmod)) / b, which matches
            // trunc or floor according to the same attribute the forward node carries.
            //
            // Integer Mod stays at zero: the output is integer-valued (piecewise constant),
            // so the a.e. derivative is 0. Note that FastProcessAutoGrad instantiates T from
            // the float32 stand-ins it synthesizes (per-tensor dtype is stripped during Fast
            // conversion), so int-typed Mod nodes only reach the zero branch when a future
            // engine pins T to the host dtype; under the current engine an int Mod's float
            // pseudo-gradient dies at the next integer-op ZERO entry upstream anyway.
            if (!typeof(FloatLike).IsAssignableFrom(typeof(T)))
            {
                // Shape-exact zeros for both inputs (grad has the broadcast OUTPUT shape).
                return [a - a, b - b];
            }

            var q = (a - (Tensor<T>)OnnxOp.Mod(a, b, fmod)) / b;
            var aGrad = ReverseBroadcast(grad, a.DShape);
            var bGrad = ReverseBroadcast(-(grad * q), b.DShape);
            return [aGrad, bGrad];
        }

        // ===== GroupNormalization =====

        [AutoDiff(GROUP_NORMALIZATION)]
        public static IVariable?[] GroupNormalization<T>(
            Tensor<T> x, Tensor<T> scale, Tensor<T> bias,
            Tensor<T> grad, float? epsilon, long? numGroups, long? stashType) where T : IVarType
        {
            // GroupNorm: reshape [N, C, *spatial] → [N, G, C/G, *spatial], normalize over (C/G, *spatial),
            // then reshape back and apply scale/bias.
            // y = scale * (x_hat) + bias, where x_hat = (x_grouped - mean) / sqrt(var + eps) per group.
            //
            // Gradients:
            //   dx: needs group-level statistics
            //   dscale: sum(grad * x_hat) over batch and spatial dims → [C]
            //   dbias: sum(grad) over batch and spatial dims → [C]
            var effectiveEps = epsilon ?? 1e-5f;
            var effectiveGroups = numGroups ?? 1;
            var epsConst = TypedConst(effectiveEps, x);

            // Get input shape components
            var xShape = x.DShape;                                                    // [N, C, ...]
            var xRank = OnnxOp.Shape(xShape);                                         // [1] containing rank
            var nVec = (Tensor<int64>)OnnxOp.Slice(xShape, Vector(0L), Vector(1L));   // [N]
            var cVec = (Tensor<int64>)OnnxOp.Slice(xShape, Vector(1L), Vector(2L));   // [C]
            var spatialShape = (Tensor<int64>)OnnxOp.Slice(xShape, Vector(2L), xRank);// [H, W, ...]

            var gVec = Vector(effectiveGroups);
            var cpgVec = cVec / gVec;  // channels per group

            // Reshape x to [N, G, C/G, *spatial] for group normalization computation
            var groupedShape = (Tensor<int64>)OnnxOp.Concat(
                [nVec, gVec, cpgVec, spatialShape], axis: 0);
            var xGrouped = (Tensor<T>)OnnxOp.Reshape(x, groupedShape, allowZero: false);

            // Compute group mean and variance over axes [2, 3, ..., rank_grouped-1] (C/G and spatial dims)
            var groupedRankScalar = OnnxOp.Squeeze(OnnxOp.Shape(groupedShape), Vector(0L));
            var reduceAxes = OnnxOp.Range(Scalar(2L), groupedRankScalar, Scalar(1L));

            var groupMean = (Tensor<T>)OnnxOp.ReduceMean(xGrouped, reduceAxes, keepdims: true);
            var xCentered = xGrouped - groupMean;
            var groupVar = (Tensor<T>)OnnxOp.ReduceMean(xCentered * xCentered, reduceAxes, keepdims: true);
            var invStd = (Tensor<T>)OnnxOp.Reciprocal((Tensor<T>)OnnxOp.Sqrt(groupVar + epsConst));

            // x_hat (normalized) in grouped shape
            var xHat = xCentered * invStd;

            // Reshape x_hat back to [N, C, *spatial] for scale/bias computation
            var xHatFlat = (Tensor<T>)OnnxOp.Reshape(xHat, xShape, allowZero: false);

            // Build broadcast shape [1, C, 1, 1, ...] for scale/bias
            var onesShape = (Tensor<int64>)OnnxOp.Expand(Scalar(1L), xRank);
            var scatterIdx = (Tensor<int64>)OnnxOp.Reshape(Vector(1L), Vector(1L, 1L), allowZero: false);
            var broadcastShape = (Tensor<int64>)OnnxOp.ScatterND(onesShape, scatterIdx, cVec);
            var scaleBC = (Tensor<T>)OnnxOp.Reshape(scale, broadcastShape, allowZero: false);

            // Build reduce axes for dscale/dbias: [0, 2, 3, ..., rank-1] (all except channel dim 1)
            var xRankScalar = OnnxOp.Squeeze(xRank, Vector(0L));
            var allAxes = OnnxOp.Range(Scalar(0L), xRankScalar, Scalar(1L));
            var axis0 = (Tensor<int64>)OnnxOp.Slice(allAxes, Vector(0L), Vector(1L));
            var axesSuffix = (Tensor<int64>)OnnxOp.Slice(allAxes, Vector(2L), xRank);
            var channelReduceAxes = (Tensor<int64>)OnnxOp.Concat([axis0, axesSuffix], axis: 0);

            // dscale = sum(grad * x_hat, axes=[0, 2, 3, ...]) → [C]
            var gradScale = (Tensor<T>)OnnxOp.ReduceSum(grad * xHatFlat, channelReduceAxes, keepdims: false);

            // dbias = sum(grad, axes=[0, 2, 3, ...]) → [C]
            var gradBias = (Tensor<T>)OnnxOp.ReduceSum(grad, channelReduceAxes, keepdims: false);

            // dx: reshape grad to grouped shape and compute using group statistics
            var gradGrouped = (Tensor<T>)OnnxOp.Reshape(grad * scaleBC, groupedShape, allowZero: false);

            // dx_grouped = invStd * (gradGrouped - mean(gradGrouped) - x_centered * mean(gradGrouped * x_centered) / (var + eps))
            var meanGrad = (Tensor<T>)OnnxOp.ReduceMean(gradGrouped, reduceAxes, keepdims: true);
            var meanGradXc = (Tensor<T>)OnnxOp.ReduceMean(gradGrouped * xCentered, reduceAxes, keepdims: true);
            var dxGrouped = invStd * (gradGrouped - meanGrad - xCentered * meanGradXc * invStd * invStd);

            // Reshape dx back to original shape
            var gradX = (Tensor<T>)OnnxOp.Reshape(dxGrouped, xShape, allowZero: false);

            return [gradX, gradScale, gradBias];
        }

        // ===== GatherElements =====

        [AutoDiff(GATHER_ELEMENTS)]
        public static IVariable?[] GatherElements<T1, T2>(
            Tensor<T1> data, Tensor<T2> indices, Tensor<T1> grad, long? axis)
            where T1 : IVarType
            where T2 : IVarType
        {
            // GatherElements(data, indices, axis): output[i][j][k] = data[i][indices[i][j][k]][k] for axis=1
            // Gradient w.r.t. data: scatter grad values back to original positions using ScatterElements
            var effectiveAxis = axis ?? 0;
            var dataShape = data.DShape;

            // Create zeros with same shape and type as data
            var zero = TypedConst(0.0f, data);
            var zeros = (Tensor<T1>)OnnxOp.Expand(zero, dataShape);

            // Scatter grad values back to data positions using Add reduction (handles duplicates)
            var indicesInt = (Tensor<int64>)OnnxOp.Cast(indices, saturate: null, to: DType.Int64);
            var result = (Tensor<T1>)OnnxOp.ScatterElements(zeros, indicesInt, grad,
                axis: effectiveAxis, reduction: ScatterNDReduction.Add);
            return [result, null];
        }

        // ===== ScatterElements =====

        [AutoDiff(SCATTER_ELEMENTS)]
        public static IVariable?[] ScatterElements<T1, T2>(
            Tensor<T1> data, Tensor<T2> indices, Tensor<T1> updates,
            Tensor<T1> grad, long? axis, ScatterNDReduction? reduction)
            where T1 : IVarType
            where T2 : IVarType
        {
            // ScatterElements(data, indices, updates, axis, reduction) → output
            // Gradient depends on reduction mode. Same mul/min/max gap as ScatterND: their
            // data-dependent adjoints are not implemented, and the 'none' branch would be
            // silently wrong for them.
            var effectiveReduction = reduction ?? ScatterNDReduction.None;
            if (effectiveReduction is ScatterNDReduction.Mul or ScatterNDReduction.Min or ScatterNDReduction.Max)
                throw new AutoDiffNotSupportedException(ErrorCodes.AD003, SCATTER_ELEMENTS,
                    $"the gradient is only implemented for reduction='none'/'add' (got "
                    + $"'{effectiveReduction}'). This is an implementation limitation, not a "
                    + "mathematical one.");
            var effectiveAxis = axis ?? 0;

            var indicesInt = (Tensor<int64>)OnnxOp.Cast(indices, saturate: null, to: DType.Int64);

            // grad_updates = GatherElements(grad, indices, axis) always
            var gradUpdates = (Tensor<T1>)OnnxOp.GatherElements(grad, indicesInt, axis: effectiveAxis);

            Tensor<T1> gradData;
            if (effectiveReduction == ScatterNDReduction.Add)
            {
                // "add": data values pass through entirely, updates are added
                // grad_data = grad (unchanged)
                gradData = grad;
            }
            else
            {
                // "none": data values at scatter positions are replaced
                // grad_data = grad with positions at indices zeroed out
                var zero = TypedConst(0.0f, data);
                var zeroUpdates = (Tensor<T1>)OnnxOp.Expand(zero, updates.DShape);
                gradData = (Tensor<T1>)OnnxOp.ScatterElements(grad, indicesInt, zeroUpdates,
                    axis: effectiveAxis, reduction: ScatterNDReduction.None);
            }

            return [gradData, null, gradUpdates];
        }
    }
}

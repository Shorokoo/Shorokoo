using System;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Globals;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.AutoDiff
{
    /// <summary>
    /// Gradient implementations for ONNX activation, normalization, and shape-preserving ops
    /// whose derivative is well-defined on float inputs (HardSigmoid, HardSwish, Mish,
    /// Softplus, Softsign, ThresholdedRelu, Shrink, LogSoftmax, PRelu,
    /// MeanVarianceNormalization). All entries use the [AutoDiff] reflection pattern: one
    /// output tensor and Tensor&lt;T&gt; float inputs only.
    /// </summary>
    internal static partial class AutoDiffs
    {
        // ===== HardSigmoid =====
        // y = max(0, min(1, alpha*x + beta))
        // dy/dx = alpha when 0 < alpha*x + beta < 1, else 0.

        [AutoDiff(HARD_SIGMOID)]
        public static Variable?[] HardSigmoid<T>(Tensor<T> x, Tensor<T> grad, float? alpha, float? beta) where T : IVarType
        {
            var effAlpha = alpha ?? 0.2f;
            var effBeta = beta ?? 0.5f;
            var alphaConst = TypedConst(effAlpha, x);
            var betaConst = TypedConst(effBeta, x);
            var zero = TypedConst(0.0f, x);
            var one = TypedConst(1.0f, x);
            var t = alphaConst * x + betaConst;
            Tensor<bit> inRange = OnnxOp.And(OnnxOp.Greater(t, zero), OnnxOp.Less(t, one));
            Tensor<T> slope = OnnxOp.Where(inRange, alphaConst, zero);
            return [grad * slope];
        }

        // ===== HardSwish =====
        // y = x * max(0, min(1, x/6 + 0.5))  (ONNX fixes alpha=1/6, beta=0.5)
        // Let s = max(0, min(1, x/6 + 0.5)).
        // dy/dx = s + x * ds/dx = s + (x/6) when -3 < x < 3, else s.

        [AutoDiff(HARD_SWISH)]
        public static Variable?[] HardSwish<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            var oneSixth = TypedConst(1.0f / 6.0f, x);
            var half = TypedConst(0.5f, x);
            var zero = TypedConst(0.0f, x);
            var one = TypedConst(1.0f, x);
            var t = oneSixth * x + half;
            Tensor<T> s = OnnxOp.Min(OnnxOp.Max(t, zero), one);
            Tensor<bit> inRange = OnnxOp.And(OnnxOp.Greater(t, zero), OnnxOp.Less(t, one));
            Tensor<T> deriv = OnnxOp.Where(inRange, s + x * oneSixth, s);
            return [grad * deriv];
        }

        // ===== Mish =====
        // y = x * tanh(softplus(x))
        // Let sp = softplus(x), th = tanh(sp), sig = sigmoid(x).
        // dy/dx = th + x * (1 - th^2) * sig.

        [AutoDiff(MISH)]
        public static Variable?[] Mish<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            var one = TypedConst(1.0f, x);
            Tensor<T> sp = OnnxOp.Softplus(x);
            Tensor<T> th = OnnxOp.Tanh(sp);
            Tensor<T> sig = OnnxOp.Sigmoid(x);
            var deriv = th + x * (one - th * th) * sig;
            return [grad * deriv];
        }

        // ===== Softplus =====
        // y = log(1 + exp(x))
        // dy/dx = sigmoid(x).

        [AutoDiff(SOFTPLUS)]
        public static Variable?[] Softplus<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            Tensor<T> sig = OnnxOp.Sigmoid(x);
            return [grad * sig];
        }

        // ===== Softsign =====
        // y = x / (1 + |x|)
        // dy/dx = 1 / (1 + |x|)^2.

        [AutoDiff(SOFTSIGN)]
        public static Variable?[] Softsign<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            var one = TypedConst(1.0f, x);
            var denom = one + OnnxOp.Abs(x);
            return [grad / (denom * denom)];
        }

        // ===== ThresholdedRelu =====
        // y = x if x > alpha else 0
        // dy/dx = 1 if x > alpha else 0.

        [AutoDiff(THRESHOLDED_RELU)]
        public static Variable?[] ThresholdedRelu<T>(Tensor<T> x, Tensor<T> grad, float? alpha) where T : IVarType
        {
            var effAlpha = alpha ?? 1.0f;
            var alphaConst = TypedConst(effAlpha, x);
            var zero = TypedConst(0.0f, x);
            var one = TypedConst(1.0f, x);
            var mask = x > alphaConst;
            Tensor<T> slope = OnnxOp.Where(mask, one, zero);
            return [grad * slope];
        }

        // ===== Shrink =====
        // y = x - bias if x > lambd; x + bias if x < -lambd; else 0
        // dy/dx = 1 outside [-lambd, lambd], else 0. (The piecewise constant -bias / +bias
        // contributions vanish under differentiation.)

        [AutoDiff(SHRINK)]
        public static Variable?[] Shrink<T>(Tensor<T> x, Tensor<T> grad, float? bias, float? lambd) where T : IVarType
        {
            _ = bias; // |bias| only shifts the piecewise-constant offset; it has no gradient effect.
            var effLambd = lambd ?? 0.5f;
            var lambdConst = TypedConst(effLambd, x);
            var zero = TypedConst(0.0f, x);
            var one = TypedConst(1.0f, x);
            Tensor<bit> outside = OnnxOp.Abs(x) > lambdConst;
            Tensor<T> slope = OnnxOp.Where(outside, one, zero);
            return [grad * slope];
        }

        // ===== LogSoftmax =====
        // y_i = x_i - log(sum_k exp(x_k))
        // dy_i/dx_j = δ_ij - softmax(x)_j
        // dx_j = grad_j - softmax(x)_j * sum_i(grad_i)

        [AutoDiff(LOG_SOFTMAX)]
        public static Variable?[] LogSoftmax<T>(Tensor<T> x, Tensor<T> grad, long? axis) where T : IVarType
        {
            var effAxis = axis ?? -1;
            var sm = x.Softmax(effAxis);
            var sumGrad = grad.Reduce(ReduceKind.Sum, axes: Vector(effAxis), keepDims: true);
            return [grad - sm * sumGrad];
        }

        // ===== PRelu =====
        // y = max(0, x) + slope * min(0, x)
        // dy/dx = 1 if x > 0 else slope
        // dy/dslope = 0 if x > 0 else x  (sum-reduced over broadcast dims for the slope gradient)

        [AutoDiff(P_RELU)]
        public static Variable?[] PRelu<T>(Tensor<T> x, Tensor<T> slope, Tensor<T> grad) where T : IVarType
        {
            var zero = TypedConst(0.0f, x);
            var one = TypedConst(1.0f, x);
            var positive = x > zero;
            Tensor<T> dxFactor = OnnxOp.Where(positive, one, slope);
            Tensor<T> dslopeFactor = OnnxOp.Where(positive, zero, x);
            var dx = ReverseBroadcast(grad * dxFactor, x.DShape);
            var dslope = ReverseBroadcast(grad * dslopeFactor, slope.DShape);
            return [dx, dslope];
        }

        // ===== MeanVarianceNormalization =====
        // y = (x - mean(x, axes)) / sqrt(var(x, axes) + eps)
        // Identical to LayerNormalization without a learnable scale/bias. Uses the standard
        // batch-norm-style backward formula reduced over `axes` (default [0, 2, 3] when
        // unspecified, per the ONNX spec).

        [AutoDiff(MEAN_VARIANCE_NORMALIZATION)]
        public static Variable?[] MeanVarianceNormalization<T>(Tensor<T> x, Tensor<T> grad, long[]? axes) where T : IVarType
        {
            // ONNX default axes are [0, 2, 3] (over batch and spatial dims of an NCHW tensor).
            var effectiveAxes = axes ?? new long[] { 0, 2, 3 };
            var axesVec = Vector(effectiveAxes);
            var eps = TypedConst(1e-9f, x);

            Tensor<T> mean = OnnxOp.ReduceMean(x, axesVec, keepdims: true);
            var xCentered = x - mean;
            Tensor<T> variance = OnnxOp.ReduceMean(xCentered * xCentered, axesVec, keepdims: true);
            Tensor<T> invStd = OnnxOp.Reciprocal(OnnxOp.Sqrt(variance + eps));

            // N = product of the dim sizes over the reduction axes, in T's dtype.
            var xShape = x.DShape;
            Tensor<int64> axisSizes = OnnxOp.Gather(xShape, Vector(effectiveAxes), axis: 0);
            Tensor<int64> nLong = OnnxOp.ReduceProd(axisSizes, axes: Vector(0L), keepdims: false, noopWithEmptyAxes: false);
            Tensor<T> n = OnnxOp.Cast(nLong, saturate: null, to: x.Type);

            Tensor<T> meanGrad = OnnxOp.ReduceMean(grad, axesVec, keepdims: true);
            Tensor<T> meanGradXc = OnnxOp.ReduceMean(grad * xCentered, axesVec, keepdims: true);
            var dx = invStd * (grad - meanGrad - xCentered * meanGradXc * invStd * invStd);
            _ = n; // Reduce*Mean already divides by N; kept here only for parity with the textbook formula.
            return [dx];
        }
    }
}

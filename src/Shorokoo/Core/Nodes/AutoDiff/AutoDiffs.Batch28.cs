using System;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Globals;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.AutoDiff
{
    /// <summary>
    /// Non-differentiable gradient stubs for ops whose mathematical derivative is zero
    /// almost everywhere (piecewise-constant, index-only, random, comparison, bitwise,
    /// boolean output). Registering them ensures the autograd engine treats them as a
    /// proper "gradient is zero" rather than throwing <c>NotImplementedException</c>.
    /// </summary>
    internal static partial class AutoDiffs
    {
        // ===== Round =====
        // Piecewise constant: gradient is zero everywhere.

        [AutoDiff(ROUND)]
        public static Variable?[] Round<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            var zero = TypedConst(0.0f, x);
            return [zero * grad];
        }

        // ===== Hardmax =====
        // One-hot output is piecewise constant; gradient is zero (PyTorch convention).

        [AutoDiff(HARDMAX)]
        public static Variable?[] Hardmax<T>(Tensor<T> x, Tensor<T> grad, long? axis) where T : IVarType
        {
            var zero = TypedConst(0.0f, x);
            return [zero * grad];
        }

        // ===== Bernoulli =====
        // Stochastic boolean sample; gradient is zero (the sampling is the
        // discrete-output equivalent of round).

        [AutoDiff(BERNOULLI)]
        public static Variable? Bernoulli<TIn, TOut>(Tensor<TIn> x, Tensor<TOut> grad, DType? dtype, float? seed)
            where TIn : IVarType
            where TOut : IVarType
        {
            // grad is in the dtype of the output (often bit/integer); we can't easily
            // produce a typed zero for the input dtype here, so we simply break the
            // gradient chain — the input contributes no gradient to a downstream loss.
            return null;
        }

        // ===== EyeLike =====
        // Output is purely structural (identity-like) — no gradient flows to the
        // template input.

        [AutoDiff(EYE_LIKE)]
        public static Variable? EyeLike<TIn, TOut>(Tensor<TIn> input, Tensor<TOut> grad, DType? dtype, long? k)
            where TIn : IVarType
            where TOut : IVarType
        {
            return null;
        }

        // ===== Shape, Size =====
        // Integer outputs derived from the input's shape — non-differentiable.

        [AutoDiff(SHAPE)]
        public static Variable? Shape<TIn, TGrad>(Tensor<TIn> input, Tensor<TGrad> grad, long? end, long? start)
            where TIn : IVarType
            where TGrad : IVarType
            => null;

        [AutoDiff(SIZE)]
        public static Variable? Size<TIn, TGrad>(Tensor<TIn> input, Tensor<TGrad> grad)
            where TIn : IVarType
            where TGrad : IVarType
            => null;

        // ===== Range =====
        // Integer index range from non-differentiable scalar inputs.

        [AutoDiff(RANGE)]
        public static Variable?[] Range<T>(Tensor<T> start, Tensor<T> limit, Tensor<T> delta, Tensor<T> grad) where T : IVarType
            => [null, null, null];

        // ===== NonZero =====
        // Index output: non-differentiable.

        [AutoDiff(NON_ZERO)]
        public static Variable? NonZero<TIn, TGrad>(Tensor<TIn> input, Tensor<TGrad> grad)
            where TIn : IVarType
            where TGrad : IVarType
            => null;

        // ===== ArgMax, ArgMin =====
        // Integer index outputs: non-differentiable.

        [AutoDiff(ARG_MAX)]
        public static Variable? ArgMax<TIn, TGrad>(Tensor<TIn> input, Tensor<TGrad> grad,
            long? axis, bool? keepdims, bool? selectLastIndex)
            where TIn : IVarType
            where TGrad : IVarType
            => null;

        [AutoDiff(ARG_MIN)]
        public static Variable? ArgMin<TIn, TGrad>(Tensor<TIn> input, Tensor<TGrad> grad,
            long? axis, bool? keepdims, bool? selectLastIndex)
            where TIn : IVarType
            where TGrad : IVarType
            => null;

        // ===== Bitwise integer ops =====
        // Integer-only ops with no meaningful derivative.

        [AutoDiff(BITWISE_AND)]
        public static Variable?[] BitwiseAnd<T>(Tensor<T> a, Tensor<T> b, Tensor<T> grad) where T : IVarType
            => [null, null];

        [AutoDiff(BITWISE_OR)]
        public static Variable?[] BitwiseOr<T>(Tensor<T> a, Tensor<T> b, Tensor<T> grad) where T : IVarType
            => [null, null];

        [AutoDiff(BITWISE_XOR)]
        public static Variable?[] BitwiseXor<T>(Tensor<T> a, Tensor<T> b, Tensor<T> grad) where T : IVarType
            => [null, null];

        [AutoDiff(BITWISE_NOT)]
        public static Variable? BitwiseNot<T>(Tensor<T> a, Tensor<T> grad) where T : IVarType
            => null;

        [AutoDiff(BIT_SHIFT)]
        public static Variable?[] BitShift<T>(Tensor<T> a, Tensor<T> b, Tensor<T> grad,
            BitShiftDirection? direction) where T : IVarType
            => [null, null];

        // ===== Boolean ops =====
        // Boolean inputs and outputs: no float gradient. We keep the grad parameter
        // generic so the autograd engine can pass whatever dtype it's accumulated
        // upstream without a CheckValue failure on Tensor<bit> coercion.

        [AutoDiff(AND)]
        public static Variable?[] And<TGrad>(Tensor<bit> a, Tensor<bit> b, Tensor<TGrad> grad) where TGrad : IVarType
            => [null, null];

        [AutoDiff(OR)]
        public static Variable?[] Or<TGrad>(Tensor<bit> a, Tensor<bit> b, Tensor<TGrad> grad) where TGrad : IVarType
            => [null, null];

        [AutoDiff(XOR)]
        public static Variable?[] Xor<TGrad>(Tensor<bit> a, Tensor<bit> b, Tensor<TGrad> grad) where TGrad : IVarType
            => [null, null];

        [AutoDiff(NOT)]
        public static Variable? Not<TGrad>(Tensor<bit> a, Tensor<TGrad> grad) where TGrad : IVarType
            => null;

        // ===== Comparison ops =====
        // Bool output, gradient does not propagate.

        [AutoDiff(EQUAL)]
        public static Variable?[] Equal<T, TGrad>(Tensor<T> a, Tensor<T> b, Tensor<TGrad> grad)
            where T : IVarType where TGrad : IVarType
            => [null, null];

        [AutoDiff(GREATER)]
        public static Variable?[] Greater<T, TGrad>(Tensor<T> a, Tensor<T> b, Tensor<TGrad> grad)
            where T : IVarType where TGrad : IVarType
            => [null, null];

        [AutoDiff(GREATER_OR_EQUAL)]
        public static Variable?[] GreaterOrEqual<T, TGrad>(Tensor<T> a, Tensor<T> b, Tensor<TGrad> grad)
            where T : IVarType where TGrad : IVarType
            => [null, null];

        [AutoDiff(LESS)]
        public static Variable?[] Less<T, TGrad>(Tensor<T> a, Tensor<T> b, Tensor<TGrad> grad)
            where T : IVarType where TGrad : IVarType
            => [null, null];

        [AutoDiff(LESS_OR_EQUAL)]
        public static Variable?[] LessOrEqual<T, TGrad>(Tensor<T> a, Tensor<T> b, Tensor<TGrad> grad)
            where T : IVarType where TGrad : IVarType
            => [null, null];

        // ===== IsInf, IsNaN =====
        // Boolean classifiers: non-differentiable.

        [AutoDiff(IS_INF)]
        public static Variable? IsInf<T, TGrad>(Tensor<T> input, Tensor<TGrad> grad)
            where T : IVarType where TGrad : IVarType
            => null;

        [AutoDiff(IS_NAN)]
        public static Variable? IsNaN<T, TGrad>(Tensor<T> input, Tensor<TGrad> grad)
            where T : IVarType where TGrad : IVarType
            => null;

        // ===== Random ops =====
        // Sampling has no continuous derivative w.r.t. the (non-existent) input
        // through the standard reparameterization-free path.

        [AutoDiff(RANDOM_NORMAL_LIKE)]
        public static Variable? RandomNormalLike<TIn, TOut>(
            Tensor<TIn> input, Tensor<TOut> grad,
            DType? dtype, float? mean, float? scale, float? seed)
            where TIn : IVarType
            where TOut : IVarType
            => null;

        [AutoDiff(RANDOM_UNIFORM_LIKE)]
        public static Variable? RandomUniformLike<TIn, TOut>(
            Tensor<TIn> input, Tensor<TOut> grad,
            DType? dtype, float? high, float? low, float? seed)
            where TIn : IVarType
            where TOut : IVarType
            => null;

        // RANDOM_NORMAL/RANDOM_UNIFORM have no inputs; SHRK_RANDOM_NORMAL/UNIFORM take
        // hyperparam-derived inputs that the autograd engine never differentiates
        // through. They don't need [AutoDiff] entries because no reachable autograd
        // path passes through them.

        // ===== Multinomial =====
        // Discrete-sample output: non-differentiable w.r.t. logits in the standard path.

        [AutoDiff(MULTINOMIAL)]
        public static Variable? Multinomial<TIn, TOut>(
            Tensor<TIn> input, Tensor<TOut> grad,
            DType? dtype, float? seed)
            where TIn : IVarType
            where TOut : IVarType
            => null;

        // ===== OneHot =====
        // Categorical encoding: indices are integer (non-diff). Values are scalar
        // (off_value, on_value) — their gradient could in principle flow back, but
        // typical use treats them as constants. We stop the gradient here.

        [AutoDiff(ONE_HOT)]
        public static Variable?[] OneHot<TIdx, TDepth, TVal>(
            Tensor<TIdx> indices, Tensor<TDepth> depth, Tensor<TVal> values,
            Tensor<TVal> grad, long? axis)
            where TIdx : IVarType
            where TDepth : IVarType
            where TVal : IVarType
            => [null, null, null];

        // ===== BlackmanWindow =====
        // All-attribute window generator with one integer input (size) — non-diff.

        [AutoDiff(BLACKMAN_WINDOW)]
        public static Variable? BlackmanWindow<T>(
            Tensor<int64> size, Tensor<T> grad, DType? outputDatatype, bool? periodic)
            where T : IVarType
            => null;

        // Note: DET is already handled by the variadic DetGradient registration in
        // AutoDiffs.AC.cs/Batch7.cs (which implements the closed-form cofactor-based
        // gradient). Variadic registrations overwrite [AutoDiff] reflection entries
        // in GetGradientOps, so adding a DET stub here would be a no-op — see the
        // RegisterVariadicGradientOps tail in AC.cs.
    }
}

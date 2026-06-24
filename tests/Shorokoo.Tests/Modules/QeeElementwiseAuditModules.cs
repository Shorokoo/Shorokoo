namespace Shorokoo.Tests.Modules
{
    // ===================================================================
    //  Self-checking VALUE-audit modules for the Phase 4 QEE-A2 batch
    //  (elementwise / comparison / logical / bitwise family, ONNX opset
    //  21). Unlike the A1 shape-audit modules, these compare the audited
    //  ops' computed VALUES against spec-expected constants (within a
    //  small tolerance for float ops) and return a single Scalar<bit>.
    //
    //  Driven two ways by QeeElementwiseAuditTests: AdvancedTestGraph
    //  validates the expected values against real ONNX Runtime execution,
    //  and the QeeSelfCheck bit-check validates that QuickExecutionEngine
    //  computes the same concrete values (every op in the comparison
    //  chain — Sub/Abs/Greater/Cast/Reduce/Less — propagates concrete
    //  data, so a wrong or missing QEE value flips the bit or leaves it
    //  uncomputed, failing the test either way).
    // ===================================================================

    /// <summary>Trig / hyperbolic / exp / log / sqrt / reciprocal unary float ops.
    /// Inputs: xs = [0.5, -0.25, 0.75] (trig + atanh domain), xp = [1, 2, 4]
    /// (acosh / log / sqrt domain).</summary>
    [Module]
    public partial class QeeTrigExpLogValueAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> xs, Tensor<float32> xp)
        {
            var mismatch =
                FloatMismatch(xs.Acos(), Vector(1.047198f, 1.823477f, 0.7227343f)) +
                FloatMismatch(xs.Asin(), Vector(0.5235988f, -0.2526802f, 0.8480621f)) +
                FloatMismatch(xs.Atan(), Vector(0.4636476f, -0.2449787f, 0.6435011f)) +
                FloatMismatch(xs.Atanh(), Vector(0.5493062f, -0.2554128f, 0.972955f)) +
                FloatMismatch(xs.Cos(), Vector(0.8775826f, 0.9689124f, 0.7316889f)) +
                FloatMismatch(xs.Sin(), Vector(0.4794255f, -0.247404f, 0.6816388f)) +
                FloatMismatch(xs.Tan(), Vector(0.5463025f, -0.2553419f, 0.9315965f)) +
                FloatMismatch(xs.Cosh(), Vector(1.127626f, 1.031413f, 1.294683f)) +
                FloatMismatch(xs.Sinh(), Vector(0.5210953f, -0.2526123f, 0.8223167f)) +
                FloatMismatch(xs.Tanh(), Vector(0.4621172f, -0.2449187f, 0.6351489f)) +
                FloatMismatch(xs.Exp(), Vector(1.648721f, 0.7788008f, 2.117f)) +
                FloatMismatch(xp.Acosh(), Vector(0f, 1.316958f, 2.063437f)) +
                FloatMismatch(xp.Asinh(), Vector(0.8813736f, 1.443635f, 2.094712f)) +
                FloatMismatch(xp.Ln(), Vector(0f, 0.6931472f, 1.386294f)) +
                FloatMismatch(xp.Sqrt(), Vector(1f, 1.414214f, 2f)) +
                FloatMismatch(xp.Reciprocal(), Vector(1f, 0.5f, 0.25f));
            return mismatch < Scalar(1L);
        }

        // NaN-safe: Not(<= tol) counts a NaN diff as a mismatch; a plain "> tol" would pass it (IEEE).
        private static Scalar<int64> FloatMismatch(Tensor<float32> actual, Vector<float32> expected)
            => ((Tensor<bit>)OnnxOp.Not((actual - expected).Abs() <= Scalar(1e-3f))).Cast<int64>()
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

    /// <summary>Rounding / sign / sigmoid-family unary float ops, incl. Round's
    /// half-to-even ties. Input x = [-1.5, -0.5, 0.5, 1.5, 2.5].</summary>
    [Module]
    public partial class QeeUnaryRoundingValueAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var mismatch =
                FloatMismatch(x.Abs(), Vector(1.5f, 0.5f, 0.5f, 1.5f, 2.5f)) +
                FloatMismatch(x.Ceiling(), Vector(-1f, 0f, 1f, 2f, 3f)) +
                FloatMismatch(x.Floor(), Vector(-2f, -1f, 0f, 1f, 2f)) +
                // Round is half-to-even: -1.5 → -2, ±0.5 → 0, 1.5 → 2, 2.5 → 2 (not 3).
                FloatMismatch(x.Round(), Vector(-2f, 0f, 0f, 2f, 2f)) +
                FloatMismatch(-x, Vector(1.5f, 0.5f, -0.5f, -1.5f, -2.5f)) +
                FloatMismatch(x.Sign(), Vector(-1f, -1f, 1f, 1f, 1f)) +
                FloatMismatch(x.Erf(), Vector(-0.9661052f, -0.5204999f, 0.5204999f, 0.9661052f, 0.999593f)) +
                FloatMismatch(x.Sigmoid(), Vector(0.1824255f, 0.3775407f, 0.6224594f, 0.8175745f, 0.9241418f)) +
                FloatMismatch(x.Mish(), Vector(-0.2980998f, -0.2207438f, 0.3752452f, 1.403378f, 2.471392f)) +
                FloatMismatch(x.Softplus(), Vector(0.2014133f, 0.474077f, 0.974077f, 1.701413f, 2.57889f)) +
                FloatMismatch(x.Softsign(), Vector(-0.6f, -0.3333333f, 0.3333333f, 0.6f, 0.7142857f)) +
                FloatMismatch(x.HardSwish(), Vector(-0.375f, -0.2083333f, 0.2916667f, 1.125f, 2.291667f)) +
                FloatMismatch(x.Relu(), Vector(0f, 0f, 0.5f, 1.5f, 2.5f));
            return mismatch < Scalar(1L);
        }

        // NaN-safe: Not(<= tol) counts a NaN diff as a mismatch; a plain "> tol" would pass it (IEEE).
        private static Scalar<int64> FloatMismatch(Tensor<float32> actual, Vector<float32> expected)
            => ((Tensor<bit>)OnnxOp.Not((actual - expected).Abs() <= Scalar(1e-3f))).Cast<int64>()
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

    /// <summary>Parametrized activations: Celu / Elu / Gelu (approximate none vs tanh —
    /// checked at ±2.7 where the two formulas differ by ~4.7e-4, against a 1.5e-4
    /// tolerance) / HardSigmoid / LeakyRelu / PRelu / Selu / Shrink / ThresholdedRelu /
    /// Clip (min/max as INPUTS). Inputs: xa = [-2, -0.5, 0, 0.5, 2], xg = [-2.7, -1, 0.5, 2.7].</summary>
    [Module]
    public partial class QeeActivationValueAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> xa, Tensor<float32> xg)
        {
            var geluNoneExpected = Vector(-0.009360829f, -0.1586553f, 0.3457312f, 2.690639f);
            var geluTanhExpected = Vector(-0.008887595f, -0.158808f, 0.345714f, 2.691113f);
            var mismatch =
                FloatMismatch(xa.Celu(alpha: 2f), Vector(-1.264241f, -0.4423984f, 0f, 0.5f, 2f)) +
                FloatMismatch(xa.Elu(), Vector(-0.8646647f, -0.3934693f, 0f, 0.5f, 2f)) +
                FloatMismatch(xa.Elu(alpha: 2f), Vector(-1.729329f, -0.7869387f, 0f, 0.5f, 2f)) +
                FloatMismatch(xg.Gelu(GeluApproximate.None), geluNoneExpected, tolerance: 1.5e-4f) +
                FloatMismatch(xg.Gelu(GeluApproximate.Tanh), geluTanhExpected, tolerance: 1.5e-4f) +
                // attr left unset → spec default "none".
                FloatMismatch((Tensor<float32>)OnnxOp.Gelu(xg), geluNoneExpected, tolerance: 1.5e-4f) +
                FloatMismatch(xa.HardSigmoid(), Vector(0.1f, 0.4f, 0.5f, 0.6f, 0.9f)) +
                FloatMismatch(xa.HardSigmoid(alpha: 0.5f, beta: 0.6f), Vector(0f, 0.35f, 0.6f, 0.85f, 1f)) +
                FloatMismatch(xa.LeakyRelu(alpha: 0.1f), Vector(-0.2f, -0.05f, 0f, 0.5f, 2f)) +
                // PRelu slope [1] broadcasts unidirectionally to xa's [5].
                FloatMismatch((Tensor<float32>)OnnxOp.PRelu(xa, Vector(0.25f)), Vector(-0.5f, -0.125f, 0f, 0.5f, 2f)) +
                FloatMismatch(xa.Selu(), Vector(-1.520167f, -0.6917582f, 0f, 0.5253505f, 2.101402f)) +
                FloatMismatch(xa.Shrink(), Vector(-2f, 0f, 0f, 0f, 2f)) +
                FloatMismatch(xa.Shrink(bias: 0.5f, lambd: 1f), Vector(-1.5f, 0f, 0f, 0f, 1.5f)) +
                FloatMismatch(xa.ThresholdedRelu(), Vector(0f, 0f, 0f, 0f, 2f)) +
                FloatMismatch((Tensor<float32>)OnnxOp.Clip(xa, Scalar(-1f), Scalar(1f)), Vector(-1f, -0.5f, 0f, 0.5f, 1f));
            return mismatch < Scalar(1L);
        }

        // NaN-safe: Not(<= tol) counts a NaN diff as a mismatch; a plain "> tol" would pass it (IEEE).
        private static Scalar<int64> FloatMismatch(Tensor<float32> actual, Vector<float32> expected, float tolerance = 1e-3f)
            => ((Tensor<bit>)OnnxOp.Not((actual - expected).Abs() <= Scalar(tolerance))).Cast<int64>()
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

    /// <summary>Binary + variadic arithmetic: Add / Sub / Mul / Div (float and trunc-toward-
    /// zero int) / Pow / Mod (fmod=0 sign-of-DIVISOR int default, fmod=1 sign-of-dividend
    /// for int and float) / Min / Max / Sum / Mean (3 inputs with a broadcast scalar).
    /// Inputs: af = [7.5, -5.5, 9.25], bf = [2, 3, -4], ai = [7, -7, 9], bi = [2, 2, -4].</summary>
    [Module]
    public partial class QeeBinaryArithValueAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> af, Tensor<float32> bf, Tensor<int64> ai, Tensor<int64> bi)
        {
            var mismatch =
                FloatMismatch(af + bf, Vector(9.5f, -2.5f, 5.25f)) +
                FloatMismatch(af - bf, Vector(5.5f, -8.5f, 13.25f)) +
                FloatMismatch(af * bf, Vector(15f, -16.5f, -37f)) +
                FloatMismatch(af / bf, Vector(3.75f, -1.833333f, -2.3125f)) +
                // Integer division truncates toward zero: -7/2 → -3, 9/-4 → -2.
                IntMismatch(ai / bi, Vector(3L, -3L, -2L)) +
                FloatMismatch(af.Pow(Vector(2f, 1f, 0.5f)), Vector(56.25f, -5.5f, 3.041381f)) +
                // fmod unset → 0 → numpy.mod (sign of divisor): [-7 mod 2, 9 mod -4] = [1, -3].
                IntMismatch((Tensor<int64>)OnnxOp.Mod(ai, bi), Vector(1L, 1L, -3L)) +
                // fmod=1 → C fmod (sign of dividend).
                IntMismatch((Tensor<int64>)OnnxOp.Mod(ai, bi, fmod: true), Vector(1L, -1L, 1L)) +
                FloatMismatch((Tensor<float32>)OnnxOp.Mod(af, bf, fmod: true), Vector(1.5f, -2.5f, 1.25f)) +
                FloatMismatch((Tensor<float32>)OnnxOp.Min(af, bf, Scalar(0f)), Vector(0f, -5.5f, -4f)) +
                FloatMismatch((Tensor<float32>)OnnxOp.Max(af, bf, Scalar(0f)), Vector(7.5f, 3f, 9.25f)) +
                FloatMismatch((Tensor<float32>)OnnxOp.Sum(af, bf, Scalar(1f)), Vector(10.5f, -1.5f, 6.25f)) +
                FloatMismatch((Tensor<float32>)OnnxOp.Mean(af, bf, Scalar(1.5f)), Vector(3.666667f, -0.3333333f, 2.25f)) +
                IntMismatch((Tensor<int64>)OnnxOp.Min(ai, bi), Vector(2L, -7L, -4L)) +
                IntMismatch((Tensor<int64>)OnnxOp.Max(ai, bi), Vector(7L, 2L, 9L));
            return mismatch < Scalar(1L);
        }

        // NaN-safe: Not(<= tol) counts a NaN diff as a mismatch; a plain "> tol" would pass it (IEEE).
        private static Scalar<int64> FloatMismatch(Tensor<float32> actual, Vector<float32> expected)
            => ((Tensor<bit>)OnnxOp.Not((actual - expected).Abs() <= Scalar(1e-3f))).Cast<int64>()
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        private static Scalar<int64> IntMismatch(Tensor<int64> actual, Vector<int64> expected)
            => (actual - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

    /// <summary>Comparisons (float, int and — for Equal — bool inputs, all → bool output)
    /// and the bool-only logical ops And / Or / Xor / Not. Inputs: cf = [1, 2, 3],
    /// cg = [2, 2, 2], p = [T, F, T, F], q = [T, T, F, F].</summary>
    [Module]
    public partial class QeeCompareLogicValueAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> cf, Tensor<float32> cg, Tensor<bit> p, Tensor<bit> q)
        {
            var ci = cf.Cast<int64>();
            var di = cg.Cast<int64>();
            var mismatch =
                BoolMismatch(cf == cg, Vector(0L, 1L, 0L)) +
                BoolMismatch(cf > cg, Vector(0L, 0L, 1L)) +
                BoolMismatch(cf >= cg, Vector(0L, 1L, 1L)) +
                BoolMismatch(cf < cg, Vector(1L, 0L, 0L)) +
                BoolMismatch(cf <= cg, Vector(1L, 1L, 0L)) +
                BoolMismatch(ci == di, Vector(0L, 1L, 0L)) +
                BoolMismatch(ci > di, Vector(0L, 0L, 1L)) +
                BoolMismatch(ci >= di, Vector(0L, 1L, 1L)) +
                BoolMismatch(ci < di, Vector(1L, 0L, 0L)) +
                BoolMismatch(ci <= di, Vector(1L, 1L, 0L)) +
                // Equal supports bool tensors since opset 19.
                BoolMismatch((Tensor<bit>)OnnxOp.Equal(p, q), Vector(1L, 0L, 0L, 1L)) +
                BoolMismatch((Tensor<bit>)OnnxOp.And(p, q), Vector(1L, 0L, 0L, 0L)) +
                BoolMismatch((Tensor<bit>)OnnxOp.Or(p, q), Vector(1L, 1L, 1L, 0L)) +
                BoolMismatch((Tensor<bit>)OnnxOp.Xor(p, q), Vector(0L, 1L, 1L, 0L)) +
                BoolMismatch((Tensor<bit>)OnnxOp.Not(p), Vector(0L, 1L, 0L, 1L));
            return mismatch < Scalar(1L);
        }

        private static Scalar<int64> BoolMismatch(Tensor<bit> actual, Vector<int64> expected)
            => (actual.Cast<int64>() - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

    /// <summary>Bitwise And / Or / Xor / Not and BitShift LEFT / RIGHT on uint32 (the
    /// ops are unsigned-only in-framework; BitwiseNot must mask the complement to the
    /// 32-bit width: ~12 → 4294967283). Inputs: ba = [12, 10, 15], bb = [10, 5, 3].</summary>
    [Module]
    public partial class QeeBitwiseValueAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<int64> ba, Tensor<int64> bb)
        {
            var ua = ba.Cast<uint32>();
            var ub = bb.Cast<uint32>();
            var mismatch =
                IntMismatch(((Tensor<uint32>)OnnxOp.BitwiseAnd(ua, ub)).Cast<int64>(), Vector(8L, 0L, 3L)) +
                IntMismatch(((Tensor<uint32>)OnnxOp.BitwiseOr(ua, ub)).Cast<int64>(), Vector(14L, 15L, 15L)) +
                IntMismatch(((Tensor<uint32>)OnnxOp.BitwiseXor(ua, ub)).Cast<int64>(), Vector(6L, 15L, 12L)) +
                IntMismatch(((Tensor<uint32>)OnnxOp.BitwiseNot(ua)).Cast<int64>(),
                    Vector(4294967283L, 4294967285L, 4294967280L)) +
                IntMismatch(((Tensor<uint32>)OnnxOp.BitShift(ua, ub, BitShiftDirection.Left)).Cast<int64>(),
                    Vector(12288L, 320L, 120L)) +
                IntMismatch(((Tensor<uint32>)OnnxOp.BitShift(ua, ub, BitShiftDirection.Right)).Cast<int64>(),
                    Vector(0L, 0L, 1L));
            return mismatch < Scalar(1L);
        }

        private static Scalar<int64> IntMismatch(Tensor<int64> actual, Vector<int64> expected)
            => (actual - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

    /// <summary>Misc elementwise: IsInf (detect_positive / detect_negative variants over
    /// [inf, -inf, NaN, 2] built via division), IsNaN, Cast (float→int64 trunc-toward-zero,
    /// int64→float, float→bool), CastLike, Where (float / int / bool 3-way broadcast
    /// select), Expand. Inputs: wv = [1, -1, 0, 2], wd = [0, 0, 0, 1], pw = [T, F, T, F].</summary>
    [Module]
    public partial class QeeMiscElementwiseValueAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> wv, Tensor<float32> wd, Tensor<bit> pw)
        {
            var q = wv / wd; // [inf, -inf, NaN, 2]
            var scaled = wv * Scalar(1.7f); // [1.7, -1.7, 0, 3.4]
            var castI = scaled.Cast<int64>(); // trunc toward zero → [1, -1, 0, 3]
            var mismatch =
                BoolMismatch((Tensor<bit>)OnnxOp.IsInf(q), Vector(1L, 1L, 0L, 0L)) +
                BoolMismatch((Tensor<bit>)OnnxOp.IsInf(q, detectNegative: false, detectPositive: true), Vector(1L, 0L, 0L, 0L)) +
                BoolMismatch((Tensor<bit>)OnnxOp.IsInf(q, detectNegative: true, detectPositive: false), Vector(0L, 1L, 0L, 0L)) +
                BoolMismatch((Tensor<bit>)OnnxOp.IsNaN(q), Vector(0L, 0L, 1L, 0L)) +
                IntMismatch(castI, Vector(1L, -1L, 0L, 3L)) +
                FloatMismatch(castI.Cast<float32>(), Vector(1f, -1f, 0f, 3f)) +
                BoolMismatch(wv.Cast<bit>(), Vector(1L, 1L, 0L, 1L)) +
                IntMismatch((Tensor<int64>)OnnxOp.CastLike(scaled, castI, saturate: null), Vector(1L, -1L, 0L, 3L)) +
                FloatMismatch((Tensor<float32>)OnnxOp.Where(pw, wv, wd), Vector(1f, 0f, 0f, 1f)) +
                IntMismatch((Tensor<int64>)OnnxOp.Where(pw, wv.Cast<int64>(), wd.Cast<int64>()), Vector(1L, 0L, 0L, 1L)) +
                FloatMismatch((Tensor<float32>)OnnxOp.Expand(wv, Vector(2L, 4L)),
                    (Tensor<float32>)OnnxOp.Reshape(Vector(1f, -1f, 0f, 2f, 1f, -1f, 0f, 2f), Vector(2L, 4L), allowZero: false));
            return mismatch < Scalar(1L);
        }

        // NaN-safe: Not(<= tol) counts a NaN diff as a mismatch; a plain "> tol" would pass it (IEEE).
        private static Scalar<int64> FloatMismatch(Tensor<float32> actual, Tensor<float32> expected)
            => ((Tensor<bit>)OnnxOp.Not((actual - expected).Abs() <= Scalar(1e-3f))).Cast<int64>()
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        private static Scalar<int64> IntMismatch(Tensor<int64> actual, Vector<int64> expected)
            => (actual - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        private static Scalar<int64> BoolMismatch(Tensor<bit> actual, Vector<int64> expected)
            => (actual.Cast<int64>() - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

    /// <summary>Where with BOOL then/else values: <c>Where([T,F,T,F], p, Not(p))</c> →
    /// all-true. ONNX Runtime's CPU EP has no bool-typed Where kernel, so this module is
    /// driven only by the QeeSelfCheck pass (QEE's BoolWhere path). Input pw = [T, F, T, F].</summary>
    [Module]
    public partial class QeeWhereBoolValueAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<bit> pw)
        {
            var whereB = (Tensor<bit>)OnnxOp.Where(pw, pw, OnnxOp.Not(pw));
            var mismatch = (whereB.Cast<int64>() - Vector(1L, 1L, 1L, 1L)).Abs()
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            return mismatch < Scalar(1L);
        }
    }

    /// <summary>Full reverse Slice (starts=-1, ends=INT_MIN, steps=-1): the spec clamps
    /// a negative-step exclusive `ends` to −1 so the slice runs backward THROUGH index 0
    /// ([1,2,3] → [3,2,1]). Pinned in AD-B3: the QEE Slice kernel used to clamp the
    /// (dim-shifted) ends to 0, dropping the first element and mis-shaping the output to
    /// [2]; ORT computed the spec value. Discovered while implementing reverse-direction
    /// recurrent gradients (which now use a Gather-based flip instead).</summary>
    [Module]
    public partial class QeeSliceReverseValueAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var reversed = (Tensor<float32>)OnnxOp.Slice(x,
                starts: Vector(-1L), ends: Vector(long.MinValue), axes: Vector(0L), steps: Vector(-1L));
            var diff = (reversed - Vector(3f, 2f, 1f).Tensor()).Abs();
            return diff.Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-6f);
        }
    }

    // ===================================================================
    //  Bug-pin modules (Phase 7 documentation sweep): wrapper-API value
    //  bugs found in Scalar.cs / Vector.cs. Each module self-checks the
    //  value the public wrapper SHOULD produce, so the check stays false
    //  (test stays failing) until the wrapper is fixed.
    // ===================================================================

    /// <summary>BUG PIN: <c>Scalar&lt;T&gt;.operator &lt;&lt;(Scalar&lt;T&gt;, PrimitiveParam)</c>
    /// (Scalar.cs) delegates to <c>operator &gt;&gt;</c>, so a left shift by a primitive
    /// constant actually right-shifts. Checks <c>(uint32)4 &lt;&lt; 2 == 16</c>; with the
    /// bug the graph computes <c>4 &gt;&gt; 2 = 1</c>. Input a = 4.</summary>
    [Module]
    public partial class ScalarShiftLeftPrimitiveBugPinCheck
    {
        public static Scalar<bit> Inline(Scalar<int64> a)
        {
            var shifted = a.Cast<uint32>() << 2L; // PrimitiveParam overload under test
            var mismatch = (shifted.Cast<int64>() - Scalar(16L)).Abs().Scalar();
            return mismatch < Scalar(1L);
        }
    }

    /// <summary>BUG PIN: <c>Scalar&lt;T&gt;.Min/Max(params Scalar&lt;T&gt;[] others)</c> (Scalar.cs)
    /// call <c>base.Min()</c>/<c>base.Max()</c> with no arguments, silently ignoring
    /// <c>others</c> — <c>a.Min(b)</c> returns <c>a</c>. Checks <c>5.Min(2) == 2</c> and
    /// <c>2.Max(5) == 5</c>. Inputs a = 5, b = 2.</summary>
    [Module]
    public partial class ScalarMinMaxOthersBugPinCheck
    {
        public static Scalar<bit> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var mismatch =
                FloatMismatch(a.Min(b), b) +
                FloatMismatch(b.Max(a), a);
            return mismatch < Scalar(1L);
        }

        // NaN-safe: Not(<= tol) counts a NaN diff as a mismatch; a plain "> tol" would pass it (IEEE).
        private static Scalar<int64> FloatMismatch(Scalar<float32> actual, Scalar<float32> expected)
            => ((actual - expected).Abs() <= Scalar(1e-3f)).IfElse(Scalar(0L), Scalar(1L));
    }

    /// <summary>BUG PIN: <c>Vector&lt;T&gt;.Min/Max(params Tensor&lt;T&gt;[] others)</c> (Vector.cs)
    /// likewise ignore <c>others</c>. Checks <c>[1,5].Min([3,2]) == [1,2]</c> and
    /// <c>[1,5].Max([3,2]) == [3,5]</c>. Inputs xs = [1,5], ys = [3,2].</summary>
    [Module]
    public partial class VectorMinMaxOthersBugPinCheck
    {
        public static Scalar<bit> Inline(Vector<float32> xs, Vector<float32> ys)
        {
            var mismatch =
                FloatMismatch(xs.Min(ys), Vector(1f, 2f)) +
                FloatMismatch(xs.Max(ys), Vector(3f, 5f));
            return mismatch < Scalar(1L);
        }

        // NaN-safe: Not(<= tol) counts a NaN diff as a mismatch; a plain "> tol" would pass it (IEEE).
        private static Scalar<int64> FloatMismatch(Tensor<float32> actual, Vector<float32> expected)
            => ((Tensor<bit>)OnnxOp.Not((actual - expected).Abs() <= Scalar(1e-3f))).Cast<int64>()
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }
}

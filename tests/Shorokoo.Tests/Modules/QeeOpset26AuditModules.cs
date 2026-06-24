namespace Shorokoo.Tests.Modules
{
    // Coverage audit modules for the decomposable members of the opset 22-26 op
    // batch. Under Shorokoo's single-opset-21 export, Swish (@24) and
    // RMSNormalization (@23) lower inline to opset-21 primitives, so their value
    // audits run on real graphs (same self-checking Scalar<bit> + value-assertion
    // style as the other Qee*Audit modules; all expected values are HAND-COMPUTED).
    // The non-decomposable ops (Attention, RotaryEmbedding, TensorScatter, BitCast,
    // CumProd) have no opset-21 equivalent and throw NotImplementedException from
    // their OnnxOp entry point, so QeeOpset26AuditTests asserts that authoring throw
    // directly rather than driving a module.

    /// <summary>Swish VALUES: y = x * sigmoid(alpha * x) with the default alpha (1)
    /// and an explicit alpha (2). Input x = [-2,-1,0,1,2]. Swish lowers to
    /// Mul/Sigmoid, so the graph loads anywhere; QEE-only here only to match the
    /// audit-module style (the value math is backend-independent).</summary>
    [Module]
    public partial class QeeSwishValueAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var mismatch =
                // x*sigmoid(x): sigmoid(-2)=0.119203, sigmoid(-1)=0.268941, ...
                FloatMismatch(NN.Swish(x), Vector(-0.238406f, -0.268941f, 0f, 0.731059f, 1.761594f)) +
                // x*sigmoid(2x): sigmoid(-4)=0.0179862, sigmoid(-2)=0.119203, ...
                FloatMismatch(NN.Swish(x, alpha: 2.0f), Vector(-0.035972f, -0.119203f, 0f, 0.880797f, 1.964028f));
            return mismatch < Scalar(1L);
        }

        // NaN-safe: Not(<= tol) counts a NaN diff as a mismatch; a plain "> tol" would pass it (IEEE).
        private static Scalar<int64> FloatMismatch(Tensor<float32> actual, Vector<float32> expected)
            => ((Tensor<bit>)OnnxOp.Not((actual - expected).Abs() <= Scalar(1e-3f))).Cast<int64>()
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

    /// <summary>RMSNormalization VALUES (small-vector closed forms):
    /// 1-D x = [1,2,3,4] (runtime input), scale = [2,0.5,1,1], default axis −1 →
    /// rms = sqrt(7.5+1e-5) = 2.738615, y = x/rms*scale;
    /// 2-D X = [[1,2],[3,4]], scale = [1,2], explicit positive axis 1 → per-row rms
    /// sqrt(2.5)/sqrt(12.5);
    /// epsilon path: x = [3,4], epsilon = 0.5 → rms = sqrt(12.5+0.5) = sqrt(13).</summary>
    [Module]
    public partial class QeeRmsNormValueAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var y1 = NN.RMSNormalization(x, Vector(2f, 0.5f, 1f, 1f).Tensor());

            var x2 = Vector(1f, 2f, 3f, 4f).Reshape(Vector(2L, 2L));
            var y2 = NN.RMSNormalization(x2, Vector(1f, 2f).Tensor(), axis: 1L);

            var xe = Vector(3f, 4f).Tensor();
            var ye = NN.RMSNormalization(xe, Vector(1f, 1f).Tensor(), epsilon: 0.5f);

            var mismatch =
                FloatMismatch(y1, Vector(0.730296f, 0.365148f, 1.095444f, 1.460593f)) +
                FloatMismatch(Flat(y2), Vector(0.632456f, 2.529822f, 0.848528f, 2.262742f)) +
                FloatMismatch(ye, Vector(0.83205f, 1.1094f));
            return mismatch < Scalar(1L);
        }

        private static Tensor<float32> Flat(Tensor<float32> t) => t.Reshape(Vector(-1L));

        // NaN-safe: Not(<= tol) counts a NaN diff as a mismatch; a plain "> tol" would pass it (IEEE).
        private static Scalar<int64> FloatMismatch(Tensor<float32> actual, Vector<float32> expected)
            => ((Tensor<bit>)OnnxOp.Not((actual - expected).Abs() <= Scalar(1e-3f))).Cast<int64>()
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }
}

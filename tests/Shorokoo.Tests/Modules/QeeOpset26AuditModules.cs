using static Shorokoo.Tests.Modules.QeeAuditVerdicts;
using static Shorokoo.Tests.Modules.TensorScatterCases;

namespace Shorokoo.Tests.Modules
{
    // Coverage audit modules for the decomposable members of the opset 22-26 op
    // batch. Under Shorokoo's single-opset-21 export, Swish (@24) and
    // RMSNormalization (@23) lower inline to opset-21 primitives and TensorScatter
    // (@24) through its registered lowering, so their value audits run on real
    // graphs (same self-checking Scalar<bit> + value-assertion style as the other
    // Qee*Audit modules; all expected values are HAND-COMPUTED, the two spec-example
    // ones straight off the ONNX TensorScatter page). The rest (Attention,
    // RotaryEmbedding, BitCast, CumProd) have no opset-21 equivalent and throw
    // NotImplementedException from their OnnxOp entry point, so QeeOpset26AuditTests
    // asserts that authoring throw directly rather than driving a module.

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
    }

    internal static class TensorScatterCases
    {
        internal static Tensor<float32> Scatter(Tensor<float32> past, Tensor<float32> update,
            Vector<int64>? writeIndices = null, long? axis = null, TensorScatterMode? mode = null)
            => (Tensor<float32>)OnnxOp.TensorScatter(past, update,
                writeIndices is { } w ? (Variable)w : null, axis, mode);

        internal static Scalar<int64> Mismatch(Tensor<float32> actual, Vector<float32> expected)
            => FloatMismatch(actual.Reshape(Vector(-1L)), expected);
    }

    /// <summary>TensorScatter VALUES. cache = [[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]]]
    /// (batch 2, sequence 3, 2 trailing) and wide = [[1,2,3,4],[5,6,7,8]] (rank 2, so axis −1
    /// is the sequence axis). Updates are 100-series for batch 0 and 200-series for batch 1,
    /// so every written element names where it came from.</summary>
    [Module]
    public partial class QeeTensorScatterValueAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> cache, Tensor<float32> wide)
        {
            var one = Vector(100f, 101f, 200f, 201f).Reshape(Vector(2L, 1L, 2L));
            var two = Vector(100f, 101f, 102f, 103f, 200f, 201f, 202f, 203f).Reshape(Vector(2L, 2L, 2L));
            var pair = Vector(10f, 11f, 20f, 21f).Reshape(Vector(2L, 2L));

            var mismatch =
                Mismatch(Scatter(cache, one, Vector(0L, 2L)),
                    Vector(100f, 101f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 200f, 201f)) +
                Mismatch(Scatter(cache, two, Vector(1L, 0L)),
                    Vector(1f, 2f, 100f, 101f, 102f, 103f, 200f, 201f, 202f, 203f, 11f, 12f)) +
                Mismatch(Scatter(cache, one),
                    Vector(100f, 101f, 3f, 4f, 5f, 6f, 200f, 201f, 9f, 10f, 11f, 12f)) +
                Mismatch(Scatter(cache, one, Vector(2L, 0L), axis: 1L),
                    Vector(1f, 2f, 3f, 4f, 100f, 101f, 200f, 201f, 9f, 10f, 11f, 12f)) +
                Mismatch(Scatter(cache, two, Vector(2L, 1L), mode: TensorScatterMode.Circular),
                    Vector(102f, 103f, 3f, 4f, 100f, 101f, 7f, 8f, 200f, 201f, 202f, 203f)) +
                Mismatch(Scatter(cache, two, Vector(3L, 0L), mode: TensorScatterMode.Circular),
                    Vector(100f, 101f, 102f, 103f, 5f, 6f, 200f, 201f, 202f, 203f, 11f, 12f)) +
                Mismatch(Scatter(wide, pair, Vector(1L, 2L), axis: -1L),
                    Vector(1f, 10f, 11f, 4f, 5f, 6f, 20f, 21f)) +
                Mismatch(Scatter(wide, pair, Vector(3L, 2L), axis: -1L, mode: TensorScatterMode.Circular),
                    Vector(11f, 2f, 3f, 10f, 5f, 6f, 20f, 21f));
            return mismatch < Scalar(1L);
        }

    }

    /// <summary>TensorScatter VALUES against the two worked examples on the ONNX
    /// TensorScatter page (`test_tensorscatter` and `test_tensorscatter_circular`):
    /// past = [2,1,4,5], default axis −2, write_indices [1,2] linear and [1,3] circular. Rank 4
    /// with a size-1 dimension between the batch and the sequence axis and a trailing one after
    /// it, which the compact cases do not reach.</summary>
    [Module]
    public partial class QeeTensorScatterSpecExampleAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> past)
        {
            var one = Vector(5f, 5f, 5f, 5f, 5f, 1f, 1f, 1f, 1f, 1f).Reshape(Vector(2L, 1L, 1L, 5L));
            var two = Vector(5f, 5f, 5f, 5f, 5f, 6f, 6f, 6f, 6f, 6f,
                             1f, 1f, 1f, 1f, 1f, 2f, 2f, 2f, 2f, 2f).Reshape(Vector(2L, 1L, 2L, 5L));

            var mismatch =
                Mismatch(
                    Scatter(past, one, Vector(1L, 2L)),
                    Vector(1f, 2f, 3f, 4f, 5f, 5f, 5f, 5f, 5f, 5f, 8f, 7f, 6f, 5f, 4f, 4f, 3f, 2f, 1f, 0f,
                           1f, 2f, 3f, 4f, 5f, 5f, 6f, 7f, 8f, 9f, 1f, 1f, 1f, 1f, 1f, 4f, 3f, 2f, 1f, 0f)) +
                Mismatch(
                    Scatter(past, two, Vector(1L, 3L),
                        mode: TensorScatterMode.Circular),
                    Vector(1f, 2f, 3f, 4f, 5f, 5f, 5f, 5f, 5f, 5f, 6f, 6f, 6f, 6f, 6f, 4f, 3f, 2f, 1f, 0f,
                           2f, 2f, 2f, 2f, 2f, 5f, 6f, 7f, 8f, 9f, 8f, 7f, 6f, 5f, 4f, 1f, 1f, 1f, 1f, 1f));
            return mismatch < Scalar(1L);
        }
    }
}

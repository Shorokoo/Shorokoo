namespace Shorokoo.Tests.Modules
{
    // Phase-4 follow-up audit modules for TopK and Unique (both were missed by the
    // family batches; the operator-support doc generation surfaced the gap). Same
    // self-checking Scalar<bit> + value-assertion style as the other Qee*Audit modules.

    /// <summary>TopK VALUES: largest (default) and smallest, tie → lower index, k input,
    /// negative axis, middle axis. Unique flatten-form VALUES: sorted (default) and
    /// first-occurrence (sorted=0), all four outputs. Input x = [2,1,1,3,4,3] reshaped
    /// where needed.</summary>
    [Module]
    public partial class QeeTopKUniqueValueAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            // x = [3,1,4,1,5,9,2,6] as [2,4]
            var (top2, top2Idx) = NN.TopK(x, Scalar(2L), axis: -1);
            var (small2, small2Idx) = NN.TopK(x, Scalar(2L), axis: -1, largest: false);

            // Middle axis: x3 = [1..8] as [2,2,2], k=1 along axis 1.
            var x3 = Vector(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f).Reshape(Vector(2L, 2L, 2L));
            var (mid, midIdx) = NN.TopK(x3, Scalar(1L), axis: 1);

            var u = Vector(2f, 1f, 1f, 3f, 4f, 3f);
            var (yS, idxS, invS, cntS) = OnnxOp.Unique(u);
            var (yF, idxF, invF, cntF) = OnnxOp.Unique(u, sorted: false);

            var mismatch =
                FloatMismatch(Flat(top2), Vector(4f, 3f, 9f, 6f)) +
                IntMismatch(FlatI(top2Idx), Vector(2L, 0L, 1L, 3L)) +
                FloatMismatch(Flat(small2), Vector(1f, 1f, 2f, 5f)) +
                IntMismatch(FlatI(small2Idx), Vector(1L, 3L, 2L, 0L)) +
                FloatMismatch(Flat(mid), Vector(3f, 4f, 7f, 8f)) +
                IntMismatch(FlatI(midIdx), Vector(1L, 1L, 1L, 1L)) +
                FloatMismatch(Flat((Tensor<float32>)yS), Vector(1f, 2f, 3f, 4f)) +
                IntMismatch(FlatI((Tensor<int64>)idxS), Vector(1L, 0L, 3L, 4L)) +
                IntMismatch(FlatI((Tensor<int64>)invS), Vector(1L, 0L, 0L, 2L, 3L, 2L)) +
                IntMismatch(FlatI((Tensor<int64>)cntS), Vector(2L, 1L, 2L, 1L)) +
                FloatMismatch(Flat((Tensor<float32>)yF), Vector(2f, 1f, 3f, 4f)) +
                IntMismatch(FlatI((Tensor<int64>)idxF), Vector(0L, 1L, 3L, 4L)) +
                IntMismatch(FlatI((Tensor<int64>)invF), Vector(0L, 1L, 1L, 2L, 3L, 2L)) +
                IntMismatch(FlatI((Tensor<int64>)cntF), Vector(1L, 2L, 2L, 1L));
            return mismatch < Scalar(1L);
        }

        private static Tensor<float32> Flat(Tensor<float32> t) => t.Reshape(Vector(-1L));
        private static Tensor<int64> FlatI(Tensor<int64> t) => t.Reshape(Vector(-1L));

        // NaN-safe: Not(<= tol) counts a NaN diff as a mismatch; a plain "> tol" would pass it (IEEE).
        private static Scalar<int64> FloatMismatch(Tensor<float32> actual, Vector<float32> expected)
            => ((Tensor<bit>)OnnxOp.Not((actual - expected).Abs() <= Scalar(1e-3f))).Cast<int64>()
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        private static Scalar<int64> IntMismatch(Tensor<int64> actual, Vector<int64> expected)
            => (actual - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

    /// <summary>Unique axis-form SHAPES (data-dependent extent — checked under real ORT
    /// execution only; QEE legitimately reports rank-only): x [3,2] with duplicate rows
    /// along axis 0 → y [2,2], inverse_indices [3]. TopK output shape with k from a
    /// computed (non-constant) scalar.</summary>
    [Module]
    public partial class QeeTopKUniqueShapeAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            // x = [[1,2],[1,2],[3,4]] (rows 0 and 1 identical)
            var (y, _, inv, _) = OnnxOp.Unique(x, axis: 0);

            var k = x.DimTensor(1); // k = 2, computed in-graph
            var (vals, _) = NN.TopK(x, k, axis: 0);

            var mismatch =
                IntMismatch(((Tensor<float32>)y).ShapeTensor(), Vector(2L, 2L)) +
                IntMismatch(((Tensor<int64>)inv).ShapeTensor(), Vector(3L)) +
                IntMismatch(vals.ShapeTensor(), Vector(2L, 2L));
            return mismatch < Scalar(1L);
        }

        private static Scalar<int64> IntMismatch(Tensor<int64> actual, Vector<int64> expected)
            => (actual - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }
}

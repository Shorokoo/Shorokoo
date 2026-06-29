namespace Shorokoo.Tests.Modules
{
    // ===================================================================
    //  Self-checking VALUE-audit modules for the Phase 4 QEE-A3 batch
    //  (reductions + shape/data-movement family, ONNX opset 21). Like the
    //  A2 modules, these compare the audited ops' computed VALUES (and,
    //  via ShapeTensor, their inferred SHAPES) against spec-expected
    //  constants and return a single Scalar<bit>.
    //
    //  Driven two ways by QeeReductionShapeAuditTests: AdvancedTestGraph
    //  validates the expected values against real ONNX Runtime execution,
    //  and the QeeSelfCheck bit-check validates that QuickExecutionEngine
    //  computes the same concrete values (every op in the comparison
    //  chain propagates concrete data, so a wrong or missing QEE value
    //  flips the bit or leaves it uncomputed, failing the test).
    // ===================================================================

    /// <summary>All 10 Reduce* ops: axes as INPUT (positive + negative), keepdims 0/1,
    /// axes absent (reduce ALL dims), noop_with_empty_axes with absent AND empty axes
    /// (identity), float + exact-int paths (incl. integer-truncating ReduceMean).
    /// Inputs: xf = [[1,2,3],[4,5,6]] (float32), xi = [[1,-2,3],[4,5,-6]] (int64).</summary>
    [Module]
    public partial class QeeReduceValueAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> xf, Tensor<int64> xi)
        {
            var mismatch =
                FloatMismatch(NN.Reduce(ReduceKind.Sum, xf, Vector(1L), keepDims: false, noOp: null), Vector(6f, 15f)) +
                FloatMismatch(NN.Reduce(ReduceKind.Sum, xf, Vector(-2L), keepDims: false, noOp: null), Vector(5f, 7f, 9f)) +
                ShapeMismatch(NN.Reduce(ReduceKind.Sum, xf, Vector(1L), keepDims: true, noOp: null), Vector(2L, 1L)) +
                FloatMismatch(Flat(NN.Reduce(ReduceKind.Sum, xf, Vector(1L), keepDims: true, noOp: null)), Vector(6f, 15f)) +
                // axes absent + noop unset → reduce over ALL dims.
                FloatMismatch(NN.Reduce(ReduceKind.Sum, xf, null, keepDims: false, noOp: null), Vector(21f)) +
                // noop_with_empty_axes=1 with absent axes → identity.
                FloatMismatch(Flat(NN.Reduce(ReduceKind.Sum, xf, null, keepDims: true, noOp: true)), Vector(1f, 2f, 3f, 4f, 5f, 6f)) +
                // noop_with_empty_axes=1 with an EMPTY axes input tensor → identity.
                FloatMismatch(Flat(NN.Reduce(ReduceKind.Sum, xf, EmptyVector<int64>(), keepDims: false, noOp: true)), Vector(1f, 2f, 3f, 4f, 5f, 6f)) +
                IntMismatch(NN.Reduce(ReduceKind.L1, xi, Vector(1L), keepDims: false, noOp: null), Vector(6L, 15L)) +
                FloatMismatch(NN.Reduce(ReduceKind.L2, xf, Vector(0L), keepDims: false, noOp: null), Vector(4.123106f, 5.385165f, 6.708204f)) +
                FloatMismatch(NN.Reduce(ReduceKind.LogSum, xf, Vector(1L), keepDims: false, noOp: null), Vector(1.791759f, 2.708050f)) +
                FloatMismatch(NN.Reduce(ReduceKind.LogSumExp, xf, Vector(1L), keepDims: false, noOp: null), Vector(3.407606f, 6.407606f)) +
                IntMismatch(NN.Reduce(ReduceKind.Max, xi, Vector(0L), keepDims: false, noOp: null), Vector(4L, 5L, 3L)) +
                IntMismatch(NN.Reduce(ReduceKind.Min, xi, Vector(-1L), keepDims: false, noOp: null), Vector(-2L, -6L)) +
                FloatMismatch(NN.Reduce(ReduceKind.Mean, xf, Vector(1L), keepDims: false, noOp: null), Vector(2f, 5f)) +
                // Integer mean truncates: (1-2+3)/3 = 0, (4+5-6)/3 = 1.
                IntMismatch(NN.Reduce(ReduceKind.Mean, xi, Vector(1L), keepDims: false, noOp: null), Vector(0L, 1L)) +
                FloatMismatch(NN.Reduce(ReduceKind.Prod, xf, Vector(1L), keepDims: false, noOp: null), Vector(6f, 120f)) +
                FloatMismatch(NN.Reduce(ReduceKind.SumSquare, xf, Vector(1L), keepDims: false, noOp: null), Vector(14f, 77f));
            return mismatch < Scalar(1L);
        }

        private static Tensor<float32> Flat(Tensor<float32> t) => t.Reshape(Vector(-1L));

        // NaN-safe: Not(<= tol) counts a NaN diff as a mismatch; a plain "> tol" would pass it (IEEE).
        private static Scalar<int64> FloatMismatch(Tensor<float32> actual, Vector<float32> expected)
            => ((Tensor<bit>)OnnxOp.Not((actual - expected).Abs() <= Scalar(1e-3f))).Cast<int64>()
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        private static Scalar<int64> IntMismatch(Tensor<int64> actual, Vector<int64> expected)
            => (actual - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

    /// <summary>ArgMax / ArgMin (axis incl. negative, keepdims, select_last_index ties,
    /// int64 output, float + int inputs) and CumSum (exclusive / reverse / both, axis as
    /// scalar INPUT incl. negative, float + int). Input a = [[1,3,3],[2,0,2]].</summary>
    [Module]
    public partial class QeeArgCumSumValueAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> a)
        {
            var ai = a.Cast<int64>();
            var c = Vector(1f, 2f, 3f, 4f);
            var mismatch =
                IntMismatch(a.ArgMax(1), Vector(1L, 0L)) +
                // select_last_index resolves the [.,3,3] and [2,.,2] ties to the LAST index.
                IntMismatch(a.ArgMax(1, keepdims: false, selectLastIndex: true), Vector(2L, 2L)) +
                ShapeMismatch(a.ArgMax(1, keepdims: true), Vector(2L, 1L)) +
                IntMismatch(a.ArgMin(0), Vector(0L, 1L, 1L)) +
                IntMismatch(a.ArgMin(-2), Vector(0L, 1L, 1L)) +
                IntMismatch(ai.ArgMax(1, keepdims: false, selectLastIndex: true), Vector(2L, 2L)) +
                IntMismatch(ai.ArgMin(1), Vector(0L, 1L)) +
                FloatMismatch(c.CumSum(Scalar(0L)), Vector(1f, 3f, 6f, 10f)) +
                FloatMismatch(c.CumSum(Scalar(0L), exclusive: true), Vector(0f, 1f, 3f, 6f)) +
                FloatMismatch(c.CumSum(Scalar(0L), reverse: true), Vector(10f, 9f, 7f, 4f)) +
                FloatMismatch(c.CumSum(Scalar(0L), exclusive: true, reverse: true), Vector(9f, 7f, 4f, 0f)) +
                FloatMismatch(Flat(a.CumSum(Scalar(-1L))), Vector(1f, 4f, 7f, 2f, 2f, 4f)) +
                IntMismatch(FlatI(ai.CumSum(Scalar(0L))), Vector(1L, 3L, 3L, 3L, 3L, 5L));
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

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

    /// <summary>Reshape (0-dims, -1, allowzero on an empty tensor), Flatten (negative axis,
    /// axis 0, value pass-through), Squeeze (axes input incl. negative + absent), Unsqueeze
    /// (negative axes), Transpose (explicit perm + default reverse, with values), Identity,
    /// Shape (start/end incl. negative + out-of-range clamping), Size.
    /// Input r = [2,3,4] float32 with values 0..23.</summary>
    [Module]
    public partial class QeeReshapeFamilyValueAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> r)
        {
            var flat24 = Vector(0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f,
                12f, 13f, 14f, 15f, 16f, 17f, 18f, 19f, 20f, 21f, 22f, 23f);
            var sq = r.Reshape(Vector(2L, 1L, 3L, 1L, 4L));
            var t2 = Vector(1f, 2f, 3f, 4f, 5f, 6f).Reshape(Vector(2L, 3L));
            var tr3 = (Tensor<float32>)OnnxOp.Transpose(r); // default perm = reverse dims
            var empty0 = Flat(r).Slice(Vector(0L), Vector(0L));
            var mismatch =
                ShapeMismatch(r.Reshape(Vector(4L, 6L)), Vector(4L, 6L)) +
                FloatMismatch(Flat(r.Reshape(Vector(4L, 6L))), flat24) +
                ShapeMismatch(r.Reshape(Vector(2L, -1L)), Vector(2L, 12L)) +
                // 0 copies the input dim (allowzero unset).
                ShapeMismatch(r.Reshape(Vector(0L, -1L)), Vector(2L, 12L)) +
                // allowzero=1: 0 is a real zero dim.
                ShapeMismatch((Tensor<float32>)OnnxOp.Reshape(empty0, Vector(0L, 4L), allowZero: true), Vector(0L, 4L)) +
                ShapeMismatch(r.Flatten(2), Vector(6L, 4L)) +
                FloatMismatch(Flat(r.Flatten(2)), flat24) +
                ShapeMismatch(r.Flatten(-2), Vector(2L, 12L)) +
                ShapeMismatch(r.Flatten(0), Vector(1L, 24L)) +
                ShapeMismatch(sq.Squeeze(Vector(1L, -2L)), Vector(2L, 3L, 4L)) +
                FloatMismatch(Flat(sq.Squeeze(Vector(1L, -2L))), flat24) +
                ShapeMismatch(sq.Squeeze(), Vector(2L, 3L, 4L)) +
                ShapeMismatch(r.Unsqueeze(Vector(0L, -1L)), Vector(1L, 2L, 3L, 4L, 1L)) +
                FloatMismatch(Flat(r.Unsqueeze(Vector(0L, -1L))), flat24) +
                FloatMismatch(Flat(t2.Transpose(1, 0)), Vector(1f, 4f, 2f, 5f, 3f, 6f)) +
                ShapeMismatch(tr3, Vector(4L, 3L, 2L)) +
                FloatMismatch(Flat(tr3), Vector(0f, 12f, 4f, 16f, 8f, 20f, 1f, 13f, 5f, 17f, 9f, 21f,
                    2f, 14f, 6f, 18f, 10f, 22f, 3f, 15f, 7f, 19f, 11f, 23f)) +
                FloatMismatch(Flat(NN.Identity(t2)), Vector(1f, 2f, 3f, 4f, 5f, 6f)) +
                IntMismatch(r.ShapeTensor(), Vector(2L, 3L, 4L)) +
                IntMismatch(r.ShapeTensor(1), Vector(3L, 4L)) +
                IntMismatch(r.ShapeTensor(0, -1), Vector(2L, 3L)) +
                IntMismatch(r.ShapeTensor(-2), Vector(3L, 4L)) +
                // end beyond the rank clamps to the rank.
                IntMismatch(r.ShapeTensor(1, 100), Vector(3L, 4L)) +
                IntMismatch(r.SizeTensor(), Vector(24L));
            return mismatch < Scalar(1L);
        }

        private static Tensor<float32> Flat(Tensor<float32> t) => t.Reshape(Vector(-1L));

        // NaN-safe: Not(<= tol) counts a NaN diff as a mismatch; a plain "> tol" would pass it (IEEE).
        private static Scalar<int64> FloatMismatch(Tensor<float32> actual, Vector<float32> expected)
            => ((Tensor<bit>)OnnxOp.Not((actual - expected).Abs() <= Scalar(1e-3f))).Cast<int64>()
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        private static Scalar<int64> IntMismatch(Tensor<int64> actual, Vector<int64> expected)
            => (actual - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

    /// <summary>Slice (negative steps — full reverse via huge-negative end, start clamped
    /// from a huge positive for step&lt;0, end clamping for step&gt;0, negative starts),
    /// Gather (axis 0/1, negative + 2-D indices), GatherElements (axis 0/1, negative
    /// indices), GatherND (plain + batch_dims=1), Compress (axis mode, condition shorter
    /// than the dim, flatten mode). Input g = [[0,1,2,3],[10,11,12,13],[20,21,22,23]].</summary>
    [Module]
    public partial class QeeSliceGatherValueAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> g)
        {
            var d22 = Vector(1f, 2f, 3f, 4f).Reshape(Vector(2L, 2L));
            var b3 = Vector(0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f).Reshape(Vector(2L, 2L, 2L));
            var revRows = g.Slice(Vector(-1L), Vector(-1000000L), Vector(0L), Vector(-1L));
            var mismatch =
                FloatMismatch(Flat(g.Slice(Vector(1L, 1L), Vector(3L, 3L))), Vector(11f, 12f, 21f, 22f)) +
                // step −1 from the last row through index 0 (end −1000000 clamps to the −1 boundary).
                ShapeMismatch(revRows, Vector(3L, 4L)) +
                FloatMismatch(Flat(revRows), Vector(20f, 21f, 22f, 23f, 10f, 11f, 12f, 13f, 0f, 1f, 2f, 3f)) +
                // step −2: start 1000 clamps to dim−1=3 → columns 3, 1.
                FloatMismatch(Flat(g.Slice(Vector(1000L), Vector(0L), Vector(1L), Vector(-2L))),
                    Vector(3f, 1f, 13f, 11f, 23f, 21f)) +
                // step +1: end 1000 clamps to the dim.
                FloatMismatch(Flat(g.Slice(Vector(2L), Vector(1000L), Vector(1L))), Vector(2f, 3f, 12f, 13f, 22f, 23f)) +
                // negative start with step 2 → columns 1, 3.
                FloatMismatch(Flat(g.Slice(Vector(-3L), Vector(4L), Vector(1L), Vector(2L))),
                    Vector(1f, 3f, 11f, 13f, 21f, 23f)) +
                FloatMismatch(Flat(g.Gather(Vector(2L, 0L), 0)), Vector(20f, 21f, 22f, 23f, 0f, 1f, 2f, 3f)) +
                FloatMismatch(Flat(g.Gather(Vector(-1L), 1)), Vector(3f, 13f, 23f)) +
                ShapeMismatch(g.Gather(Vector(0L, 1L, 1L, 2L).Reshape(Vector(2L, 2L)), 0), Vector(2L, 2L, 4L)) +
                FloatMismatch(Flat(g.Gather(Vector(0L, 1L, 1L, 2L).Reshape(Vector(2L, 2L)), 0)),
                    Vector(0f, 1f, 2f, 3f, 10f, 11f, 12f, 13f, 10f, 11f, 12f, 13f, 20f, 21f, 22f, 23f)) +
                FloatMismatch(Flat((Tensor<float32>)OnnxOp.GatherElements(
                    d22, Vector(0L, 0L, 1L, 0L).Reshape(Vector(2L, 2L)), 1)), Vector(1f, 1f, 4f, 3f)) +
                FloatMismatch(Flat((Tensor<float32>)OnnxOp.GatherElements(
                    d22, Vector(-1L, 0L).Reshape(Vector(1L, 2L)), 0)), Vector(3f, 2f)) +
                FloatMismatch(d22.GatherND(Vector(0L, 0L, 1L, 1L).Reshape(Vector(2L, 2L)), 0), Vector(1f, 4f)) +
                // batch_dims=1: per-batch row select → [[2,3],[4,5]].
                FloatMismatch(Flat(b3.GatherND(Vector(1L, 0L).Reshape(Vector(2L, 1L)), 1)), Vector(2f, 3f, 4f, 5f)) +
                ShapeMismatch(g.Compress(Vector(false, true, true), 0), Vector(2L, 4L)) +
                FloatMismatch(Flat(g.Compress(Vector(false, true, true), 0)),
                    Vector(10f, 11f, 12f, 13f, 20f, 21f, 22f, 23f)) +
                // condition shorter than the dim: the missing tail counts as false.
                ShapeMismatch(g.Compress(Vector(true), 1), Vector(3L, 1L)) +
                FloatMismatch(Flat(g.Compress(Vector(true), 1)), Vector(0f, 10f, 20f)) +
                FloatMismatch(d22.Compress(Vector(true, false, false, true)), Vector(1f, 4f));
            return mismatch < Scalar(1L);
        }

        private static Tensor<float32> Flat(Tensor<float32> t) => t.Reshape(Vector(-1L));

        // NaN-safe: Not(<= tol) counts a NaN diff as a mismatch; a plain "> tol" would pass it (IEEE).
        private static Scalar<int64> FloatMismatch(Tensor<float32> actual, Vector<float32> expected)
            => ((Tensor<bit>)OnnxOp.Not((actual - expected).Abs() <= Scalar(1e-3f))).Cast<int64>()
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

    /// <summary>ScatterElements (reduction none/add/mul/min/max, negative indices, axis 1),
    /// ScatterND (none + add, slice updates), Pad (constant with custom value, NEGATIVE
    /// pads = cropping, edge, reflect, wrap, axes input with negative axis).
    /// Input m = [[1,2,3],[4,5,6]].</summary>
    [Module]
    public partial class QeeScatterPadValueAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> m)
        {
            var v5 = Vector(1f, 2f, 3f, 4f, 5f);
            var v4 = Vector(1f, 2f, 3f, 4f);
            var d22 = Vector(1f, 2f, 3f, 4f).Reshape(Vector(2L, 2L));
            var snIdx = Vector(3L, 1L).Reshape(Vector(2L, 1L));
            var mismatch =
                FloatMismatch((Tensor<float32>)OnnxOp.ScatterElements(v5, Vector(1L, 3L), Vector(10f, 20f)),
                    Vector(1f, 10f, 3f, 20f, 5f)) +
                FloatMismatch((Tensor<float32>)OnnxOp.ScatterElements(v5, Vector(1L, 3L), Vector(10f, 20f),
                    axis: 0, reduction: ScatterNDReduction.Add), Vector(1f, 12f, 3f, 24f, 5f)) +
                FloatMismatch((Tensor<float32>)OnnxOp.ScatterElements(v5, Vector(1L, 3L), Vector(10f, 20f),
                    axis: 0, reduction: ScatterNDReduction.Mul), Vector(1f, 20f, 3f, 80f, 5f)) +
                FloatMismatch((Tensor<float32>)OnnxOp.ScatterElements(v5, Vector(1L, 3L), Vector(0f, 20f),
                    axis: 0, reduction: ScatterNDReduction.Min), Vector(1f, 0f, 3f, 4f, 5f)) +
                FloatMismatch((Tensor<float32>)OnnxOp.ScatterElements(v5, Vector(1L, 3L), Vector(0f, 20f),
                    axis: 0, reduction: ScatterNDReduction.Max), Vector(1f, 2f, 3f, 20f, 5f)) +
                FloatMismatch((Tensor<float32>)OnnxOp.ScatterElements(v5, Vector(-1L), Vector(99f)),
                    Vector(1f, 2f, 3f, 4f, 99f)) +
                FloatMismatch(Flat((Tensor<float32>)OnnxOp.ScatterElements(d22,
                    Vector(1L, 0L).Reshape(Vector(2L, 1L)), Vector(9f, 8f).Reshape(Vector(2L, 1L)), axis: 1)),
                    Vector(1f, 9f, 8f, 4f)) +
                FloatMismatch(v4.ScatterND(snIdx, Vector(30f, 10f)), Vector(1f, 10f, 3f, 30f)) +
                FloatMismatch(v4.ScatterND(snIdx, Vector(30f, 10f), ScatterNDReduction.Add), Vector(1f, 12f, 3f, 34f)) +
                // k < rank: each index tuple addresses a whole [2]-slice.
                FloatMismatch(Flat(d22.ScatterND(Vector(1L).Reshape(Vector(1L, 1L)),
                    Vector(9f, 8f).Reshape(Vector(1L, 2L)))), Vector(1f, 2f, 9f, 8f)) +
                ShapeMismatch(m.Pad(PadMode.Constant, Vector(0L, 1L, 0L, 1L), Scalar(9f)), Vector(2L, 5L)) +
                FloatMismatch(Flat(m.Pad(PadMode.Constant, Vector(0L, 1L, 0L, 1L), Scalar(9f))),
                    Vector(9f, 1f, 2f, 3f, 9f, 9f, 4f, 5f, 6f, 9f)) +
                // Negative pads CROP: drop the first column.
                ShapeMismatch(m.Pad(PadMode.Constant, Vector(0L, -1L, 0L, 0L)), Vector(2L, 2L)) +
                FloatMismatch(Flat(m.Pad(PadMode.Constant, Vector(0L, -1L, 0L, 0L))), Vector(2f, 3f, 5f, 6f)) +
                FloatMismatch(Flat(m.Pad(PadMode.Edge, Vector(1L, 0L, 0L, 0L))),
                    Vector(1f, 2f, 3f, 1f, 2f, 3f, 4f, 5f, 6f)) +
                FloatMismatch(Flat(m.Pad(PadMode.Reflect, Vector(0L, 2L, 0L, 0L))),
                    Vector(3f, 2f, 1f, 2f, 3f, 6f, 5f, 4f, 5f, 6f)) +
                FloatMismatch(Flat(m.Pad(PadMode.Wrap, Vector(0L, 1L, 0L, 1L))),
                    Vector(3f, 1f, 2f, 3f, 1f, 6f, 4f, 5f, 6f, 4f)) +
                // axes input (negative axis −1 → pad only the last dim).
                FloatMismatch(Flat(m.Pad(PadMode.Constant, Vector(1L, 1L), Scalar(7f), Vector(-1L))),
                    Vector(7f, 1f, 2f, 3f, 7f, 7f, 4f, 5f, 6f, 7f));
            return mismatch < Scalar(1L);
        }

        private static Tensor<float32> Flat(Tensor<float32> t) => t.Reshape(Vector(-1L));

        // NaN-safe: Not(<= tol) counts a NaN diff as a mismatch; a plain "> tol" would pass it (IEEE).
        private static Scalar<int64> FloatMismatch(Tensor<float32> actual, Vector<float32> expected)
            => ((Tensor<bit>)OnnxOp.Not((actual - expected).Abs() <= Scalar(1e-3f))).Cast<int64>()
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

    /// <summary>Split (num_outputs with UNEVEN last chunk [3,3,1], explicit split input,
    /// axis 1), Concat (axis 0, negative axis, 1-D int), Tile, DepthToSpace (DCR vs CRD —
    /// same shape, different values), SpaceToDepth. Input v7 = [1..7].</summary>
    [Module]
    public partial class QeeSplitConcatTileSpaceValueAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> v7)
        {
            var sp3 = v7.Split(3);
            var spE = v7.Split(new long[] { 2, 5 });
            var m24 = Vector(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f).Reshape(Vector(2L, 4L));
            var sp2 = m24.Split(2, axis: 1);
            var t2a = Vector(1f, 2f, 3f, 4f).Reshape(Vector(2L, 2L));
            var t2b = Vector(5f, 6f).Reshape(Vector(1L, 2L));
            var t2c = Vector(7f, 8f).Reshape(Vector(2L, 1L));
            var d = Vector(0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f)
                .Reshape(Vector(1L, 8L, 1L, 2L));
            var s = Vector(0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f)
                .Reshape(Vector(1L, 1L, 4L, 4L));
            var dcr = d.DepthToSpace(2, DepthColumnRowMode.DCR);
            var crd = d.DepthToSpace(2, DepthColumnRowMode.CRD);
            var s2d = (Tensor<float32>)OnnxOp.SpaceToDepth(s, 2);
            var mismatch =
                // 7 into 3 chunks: ceil(7/3)=3 → [3,3,1] (LAST chunk smaller).
                FloatMismatch(sp3[0], Vector(1f, 2f, 3f)) +
                FloatMismatch(sp3[1], Vector(4f, 5f, 6f)) +
                FloatMismatch(sp3[2], Vector(7f)) +
                ShapeMismatch(sp3[2], Vector(1L)) +
                FloatMismatch(spE[0], Vector(1f, 2f)) +
                FloatMismatch(spE[1], Vector(3f, 4f, 5f, 6f, 7f)) +
                FloatMismatch(Flat(sp2[0]), Vector(1f, 2f, 5f, 6f)) +
                FloatMismatch(Flat(sp2[1]), Vector(3f, 4f, 7f, 8f)) +
                ShapeMismatch(t2a.Concat(0, t2b), Vector(3L, 2L)) +
                FloatMismatch(Flat(t2a.Concat(0, t2b)), Vector(1f, 2f, 3f, 4f, 5f, 6f)) +
                FloatMismatch(Flat(t2a.Concat(-1, t2c)), Vector(1f, 2f, 7f, 3f, 4f, 8f)) +
                IntMismatch(Vector(1L, 2L).Concat(0, Vector(3L)), Vector(1L, 2L, 3L)) +
                ShapeMismatch(t2a.Tile(Vector(2L, 2L)), Vector(4L, 4L)) +
                FloatMismatch(Flat(t2a.Tile(Vector(2L, 2L))),
                    Vector(1f, 2f, 1f, 2f, 3f, 4f, 3f, 4f, 1f, 2f, 1f, 2f, 3f, 4f, 3f, 4f)) +
                ShapeMismatch(dcr, Vector(1L, 2L, 2L, 4L)) +
                FloatMismatch(Flat(dcr),
                    Vector(0f, 4f, 1f, 5f, 8f, 12f, 9f, 13f, 2f, 6f, 3f, 7f, 10f, 14f, 11f, 15f)) +
                ShapeMismatch(crd, Vector(1L, 2L, 2L, 4L)) +
                FloatMismatch(Flat(crd),
                    Vector(0f, 2f, 1f, 3f, 4f, 6f, 5f, 7f, 8f, 10f, 9f, 11f, 12f, 14f, 13f, 15f)) +
                ShapeMismatch(s2d, Vector(1L, 4L, 2L, 2L)) +
                FloatMismatch(Flat(s2d),
                    Vector(0f, 2f, 8f, 10f, 1f, 3f, 9f, 11f, 4f, 6f, 12f, 14f, 5f, 7f, 13f, 15f));
            return mismatch < Scalar(1L);
        }

        private static Tensor<float32> Flat(Tensor<float32> t) => t.Reshape(Vector(-1L));

        // NaN-safe: Not(<= tol) counts a NaN diff as a mismatch; a plain "> tol" would pass it (IEEE).
        private static Scalar<int64> FloatMismatch(Tensor<float32> actual, Vector<float32> expected)
            => ((Tensor<bit>)OnnxOp.Not((actual - expected).Abs() <= Scalar(1e-3f))).Cast<int64>()
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        private static Scalar<int64> IntMismatch(Tensor<int64> actual, Vector<int64> expected)
            => (actual - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

    /// <summary>OneHot (default axis −1, axis 0 and negative axis −2, custom off/on
    /// values, negative index −2 → +depth, out-of-range index → all-off row, int values
    /// dtype), Trilu (upper default, k input ±1, lower), NonZero (int64 [rank, n] output
    /// with coordinate values). Input idx = [1, 3, −2, 5] (int64).</summary>
    [Module]
    public partial class QeeOneHotTriluNonZeroValueAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<int64> idx)
        {
            var idx3 = Vector(0L, 2L, 1L);
            var t9 = Vector(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f).Reshape(Vector(3L, 3L));
            var oh = NN.OneHot(idx, Scalar(4L), Vector(0f, 1f));
            var nz1 = NN.NonZero(Vector(5f, 0f, 7f).Tensor());
            var nz2 = NN.NonZero(Vector(0f, 3f, 0f, 2f).Reshape(Vector(2L, 2L)));
            var mismatch =
                ShapeMismatch(oh, Vector(4L, 4L)) +
                // 1 → row 1 on; 3 → row 3 on; −2 → +4 = 2 on; 5 outside [−4, 3] → all-off.
                FloatMismatch(Flat(oh),
                    Vector(0f, 1f, 0f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 0f)) +
                FloatMismatch(Flat(NN.OneHot(idx3.Tensor(), Scalar(3L), Vector(7f, 9f), 0)),
                    Vector(9f, 7f, 7f, 7f, 7f, 9f, 7f, 9f, 7f)) +
                // axis −2 on output rank 2 ≡ axis 0.
                FloatMismatch(Flat(NN.OneHot(idx3.Tensor(), Scalar(3L), Vector(7f, 9f), -2)),
                    Vector(9f, 7f, 7f, 7f, 7f, 9f, 7f, 9f, 7f)) +
                IntMismatch(FlatI(NN.OneHot(idx3.Tensor(), Scalar(3L), Vector(0L, 1L))),
                    Vector(1L, 0L, 0L, 0L, 0L, 1L, 0L, 1L, 0L)) +
                FloatMismatch(Flat((Tensor<float32>)OnnxOp.Trilu(t9)),
                    Vector(1f, 2f, 3f, 0f, 5f, 6f, 0f, 0f, 9f)) +
                FloatMismatch(Flat((Tensor<float32>)OnnxOp.Trilu(t9, Scalar(1L), 1L)),
                    Vector(0f, 2f, 3f, 0f, 0f, 6f, 0f, 0f, 0f)) +
                FloatMismatch(Flat((Tensor<float32>)OnnxOp.Trilu(t9, null, 0L)),
                    Vector(1f, 0f, 0f, 4f, 5f, 0f, 7f, 8f, 9f)) +
                FloatMismatch(Flat((Tensor<float32>)OnnxOp.Trilu(t9, Scalar(-1L), 0L)),
                    Vector(0f, 0f, 0f, 4f, 0f, 0f, 7f, 8f, 0f)) +
                ShapeMismatch(nz1, Vector(1L, 2L)) +
                IntMismatch(FlatI(nz1), Vector(0L, 2L)) +
                ShapeMismatch(nz2, Vector(2L, 2L)) +
                // nonzeros at (0,1) and (1,1) → rows [0,1] and [1,1].
                IntMismatch(FlatI(nz2), Vector(0L, 1L, 1L, 1L));
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

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }
}

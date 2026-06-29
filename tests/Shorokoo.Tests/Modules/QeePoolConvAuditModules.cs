namespace Shorokoo.Tests.Modules
{
    // ===================================================================
    //  Self-checking shape-audit modules for the Phase 4 QEE-A1 batch
    //  (pooling & convolution family, ONNX opset 21). Each module runs a
    //  group of related ops under the attribute combinations audited in
    //  the batch (dilations / ceil_mode / auto_pad / asymmetric pads /
    //  group / output_padding / output_shape) and compares every result's
    //  ShapeTensor() against the spec-expected constant dims, returning a
    //  single Scalar<bit> so AutoTest treats the graph as self-checking.
    //
    //  Driven two ways by QeePoolConvAuditTests: AdvancedTestGraph
    //  validates the expected dims against real ONNX Runtime execution,
    //  and the QEE bit-check validates that QuickExecutionEngine's shape
    //  inference reproduces the same dims (the Shape op reads the QEE
    //  inferred shape, so a wrong inferred dim flips the bit to false).
    // ===================================================================

    /// <summary>MaxPool shape audit: dilations, ceil_mode (incl. the last-window clamp),
    /// asymmetric pads, SAME_LOWER auto_pad, and the optional Indices output. Input x is
    /// expected as [1,1,10,10].</summary>
    [Module]
    public partial class QeeMaxPoolShapeAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            // dilations: effective kernel (3-1)*2+1 = 5 → 10-5+1 = 6.
            var dil = NN.MaxPool(x, ceilMode: false, dilations: [2L, 2L], kernelShape: [3L, 3L],
                pads: [0L, 0L, 0L, 0L], storageOrder: 0L, strides: [1L, 1L]);
            // ceil_mode: ceil((10-2)/3)+1 = 4 (floor would give 3).
            var ceil = NN.MaxPool(x, ceilMode: true, dilations: [1L, 1L], kernelShape: [2L, 2L],
                pads: [0L, 0L, 0L, 0L], storageOrder: 0L, strides: [3L, 3L]);
            // ceil_mode clamp: ceil((10+2-2)/4)+1 = 4, but (4-1)*4 ≥ 10+1 → last window starts
            // in the end padding, so the output clamps to 3.
            var ceilClamp = NN.MaxPool(x, ceilMode: true, dilations: [1L, 1L], kernelShape: [2L, 2L],
                pads: [1L, 1L, 1L, 1L], storageOrder: 0L, strides: [4L, 4L]);
            // asymmetric pads [hBegin,wBegin,hEnd,wEnd] = [1,0,2,0]: H (10+3-3)/2+1 = 6, W (10-3)/2+1 = 4.
            var asym = NN.MaxPool(x, ceilMode: false, dilations: [1L, 1L], kernelShape: [3L, 3L],
                pads: [1L, 0L, 2L, 0L], storageOrder: 0L, strides: [2L, 2L]);
            // SAME_LOWER: ceil(10/2) = 5 (pads must stay unset — ORT rejects auto_pad + pads).
            var sameLower = NN.MaxPool(x, ceilMode: false, dilations: [1L, 1L], kernelShape: [3L, 3L],
                pads: null, storageOrder: 0L, strides: [2L, 2L], autoPad: AutoPad.SameLower);
            // Optional Indices output (int64): both outputs are [1,1,5,5].
            var (pooled, indices) = NN.MaxPoolWithIndices(x, ceilMode: false,
                kernelShape: [2L, 2L], pads: [0L, 0L, 0L, 0L], strides: [2L, 2L]);

            var mismatch =
                ShapeMismatch(dil, Vector(1L, 1L, 6L, 6L)) +
                ShapeMismatch(ceil, Vector(1L, 1L, 4L, 4L)) +
                ShapeMismatch(ceilClamp, Vector(1L, 1L, 3L, 3L)) +
                ShapeMismatch(asym, Vector(1L, 1L, 6L, 4L)) +
                ShapeMismatch(sameLower, Vector(1L, 1L, 5L, 5L)) +
                ShapeMismatch(pooled, Vector(1L, 1L, 5L, 5L)) +
                ShapeMismatch(indices, Vector(1L, 1L, 5L, 5L));
            return mismatch < Scalar(1L);
        }

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

    /// <summary>AveragePool shape audit: dilations (since opset 19), ceil_mode with
    /// count_include_pad (incl. the clamp), SAME_UPPER and VALID auto_pad. Input x is
    /// expected as [1,1,10,10].</summary>
    [Module]
    public partial class QeeAveragePoolShapeAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            // dilations: 10 - ((3-1)*2+1) + 1 = 6.
            var dil = (Tensor<float32>)OnnxOp.AveragePool(x, autoPad: AutoPad.NotSet, ceilMode: false,
                countIncludePad: false, dilations: [2L, 2L], kernelShape: [3L, 3L],
                pads: [0L, 0L, 0L, 0L], strides: [1L, 1L]);
            // ceil_mode + count_include_pad + clamp: ceil((10+2-2)/4)+1 = 4 → clamp → 3.
            var ceilClamp = (Tensor<float32>)OnnxOp.AveragePool(x, autoPad: AutoPad.NotSet, ceilMode: true,
                countIncludePad: true, dilations: [1L, 1L], kernelShape: [2L, 2L],
                pads: [1L, 1L, 1L, 1L], strides: [4L, 4L]);
            // SAME_UPPER: ceil(10/3) = 4.
            var sameUpper = (Tensor<float32>)OnnxOp.AveragePool(x, autoPad: AutoPad.SameUpper, ceilMode: false,
                countIncludePad: false, dilations: null, kernelShape: [4L, 4L],
                pads: null, strides: [3L, 3L]);
            // VALID: (10-4)/2+1 = 4.
            var valid = (Tensor<float32>)OnnxOp.AveragePool(x, autoPad: AutoPad.Valid, ceilMode: false,
                countIncludePad: false, dilations: null, kernelShape: [4L, 4L],
                pads: null, strides: [2L, 2L]);

            var mismatch =
                ShapeMismatch(dil, Vector(1L, 1L, 6L, 6L)) +
                ShapeMismatch(ceilClamp, Vector(1L, 1L, 3L, 3L)) +
                ShapeMismatch(sameUpper, Vector(1L, 1L, 4L, 4L)) +
                ShapeMismatch(valid, Vector(1L, 1L, 4L, 4L));
            return mismatch < Scalar(1L);
        }

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

    /// <summary>LpPool (ceil_mode clamp + dilations) and the three global pools
    /// (GlobalAveragePool / GlobalMaxPool / GlobalLpPool → all spatial dims collapse
    /// to 1). Input x is expected as [1,3,10,10].</summary>
    [Module]
    public partial class QeeLpPoolGlobalPoolShapeAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            // ceil_mode + clamp: ceil((10+2-2)/4)+1 = 4 → (4-1)*4 ≥ 10+1 → 3.
            var lpCeilClamp = (Tensor<float32>)OnnxOp.LpPool(x, autoPad: AutoPad.NotSet, ceilMode: true,
                dilations: [1L, 1L], kernelShape: [2L, 2L], p: 3L,
                pads: [1L, 1L, 1L, 1L], strides: [4L, 4L]);
            // dilations: (10 - ((3-1)*2+1))/2 + 1 = 3.
            var lpDil = (Tensor<float32>)OnnxOp.LpPool(x, autoPad: AutoPad.NotSet, ceilMode: false,
                dilations: [2L, 2L], kernelShape: [3L, 3L], p: 2L,
                pads: [0L, 0L, 0L, 0L], strides: [2L, 2L]);
            var gap = NN.GlobalAveragePool(x);
            var gmp = NN.GlobalMaxPool(x);
            var glp = NN.GlobalLpPool(x, p: 1L);

            var mismatch =
                ShapeMismatch(lpCeilClamp, Vector(1L, 3L, 3L, 3L)) +
                ShapeMismatch(lpDil, Vector(1L, 3L, 3L, 3L)) +
                ShapeMismatch(gap, Vector(1L, 3L, 1L, 1L)) +
                ShapeMismatch(gmp, Vector(1L, 3L, 1L, 1L)) +
                ShapeMismatch(glp, Vector(1L, 3L, 1L, 1L));
            return mismatch < Scalar(1L);
        }

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

    /// <summary>Conv shape audit: group (output channels = W.shape[0]), dilations,
    /// SAME_UPPER and VALID auto_pad. Input x is expected as [1,4,9,9].</summary>
    [Module]
    public partial class QeeConvShapeAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var wGrouped = InitSimple.Init([Scalar(6L), Scalar(2L), Scalar(3L), Scalar(3L)]);
            var bGrouped = InitSimple.Init([Scalar(6L)]).Vec();
            var w = InitSimple.Init([Scalar(2L), Scalar(4L), Scalar(3L), Scalar(3L)]);
            var b = InitSimple.Init([Scalar(2L)]).Vec();

            // group=2 with W [6,2,3,3]: out channels 6; spatial (9+2-3)/2+1 = 5.
            var grouped = NN.Conv(x, wGrouped, bGrouped, AutoPad.NotSet,
                dilations: [1L, 1L], group: 2L, kernelShape: [3L, 3L],
                pads: [1L, 1L, 1L, 1L], strides: [2L, 2L]);
            // dilations: 9 - ((3-1)*2+1) + 1 = 5.
            var dil = NN.Conv(x, w, b, AutoPad.NotSet,
                dilations: [2L, 2L], group: 1L, kernelShape: [3L, 3L],
                pads: [0L, 0L, 0L, 0L], strides: [1L, 1L]);
            // SAME_UPPER: ceil(9/2) = 5.
            var same = NN.Conv(x, w, b, AutoPad.SameUpper,
                dilations: [1L, 1L], group: 1L, kernelShape: [3L, 3L],
                pads: null, strides: [2L, 2L]);
            // VALID: 9-3+1 = 7.
            var valid = NN.Conv(x, w, b, AutoPad.Valid,
                dilations: [1L, 1L], group: 1L, kernelShape: [3L, 3L],
                pads: null, strides: [1L, 1L]);

            var mismatch =
                ShapeMismatch(grouped, Vector(1L, 6L, 5L, 5L)) +
                ShapeMismatch(dil, Vector(1L, 2L, 5L, 5L)) +
                ShapeMismatch(same, Vector(1L, 2L, 5L, 5L)) +
                ShapeMismatch(valid, Vector(1L, 2L, 7L, 7L));
            return mismatch < Scalar(1L);
        }

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

    /// <summary>ConvTranspose shape audit: output_padding + asymmetric-capable pads,
    /// dilations, the output_shape attribute override, SAME_UPPER auto_pad, and group
    /// (output channels = W.shape[1] * group). Input x is expected as [1,2,5,5].</summary>
    [Module]
    public partial class QeeConvTransposeShapeAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var w = InitSimple.Init([Scalar(2L), Scalar(3L), Scalar(3L), Scalar(3L)]);
            var b = InitSimple.Init([Scalar(3L)]).Vec();
            var wGrouped = InitSimple.Init([Scalar(2L), Scalar(3L), Scalar(2L), Scalar(2L)]);
            var bGrouped = InitSimple.Init([Scalar(6L)]).Vec();

            // stride 2, pads 1/1, output_padding 1: 2*(5-1) + 1 + 3 - 1 - 1 = 10.
            var basePlusOutPad = NN.ConvTranspose(x, w, b, AutoPad.NotSet,
                dilations: [1L, 1L], group: 1L, kernelShape: [3L, 3L],
                outputPadding: [1L, 1L], outputShape: null,
                pads: [1L, 1L, 1L, 1L], strides: [2L, 2L]);
            // dilations: (5-1) + ((3-1)*2+1) = 9.
            var dil = NN.ConvTranspose(x, w, b, AutoPad.NotSet,
                dilations: [2L, 2L], group: 1L, kernelShape: [3L, 3L],
                outputPadding: [0L, 0L], outputShape: null,
                pads: [0L, 0L, 0L, 0L], strides: [1L, 1L]);
            // output_shape attribute overrides the arithmetic (default would be 11): → 10.
            var outShape = NN.ConvTranspose(x, w, b, AutoPad.NotSet,
                dilations: [1L, 1L], group: 1L, kernelShape: [3L, 3L],
                outputPadding: [0L, 0L], outputShape: [10L, 10L],
                pads: null, strides: [2L, 2L]);
            // SAME_UPPER: output = input * stride = 10 (pads unset — ORT rejects auto_pad + pads).
            var same = NN.ConvTranspose(x, w, b, AutoPad.SameUpper,
                dilations: [1L, 1L], group: 1L, kernelShape: [3L, 3L],
                outputPadding: [0L, 0L], outputShape: null,
                pads: null, strides: [2L, 2L]);
            // group=2 with W [2,3,2,2]: out channels 3*2 = 6; spatial (5-1) + 2 = 6.
            var grouped = NN.ConvTranspose(x, wGrouped, bGrouped, AutoPad.NotSet,
                dilations: [1L, 1L], group: 2L, kernelShape: [2L, 2L],
                outputPadding: [0L, 0L], outputShape: null,
                pads: [0L, 0L, 0L, 0L], strides: [1L, 1L]);

            var mismatch =
                ShapeMismatch(basePlusOutPad, Vector(1L, 3L, 10L, 10L)) +
                ShapeMismatch(dil, Vector(1L, 3L, 9L, 9L)) +
                ShapeMismatch(outShape, Vector(1L, 3L, 10L, 10L)) +
                ShapeMismatch(same, Vector(1L, 3L, 10L, 10L)) +
                ShapeMismatch(grouped, Vector(1L, 6L, 6L, 6L));
            return mismatch < Scalar(1L);
        }

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

    /// <summary>ConvInteger (int32 output dtype) and QLinearConv (output dtype follows
    /// y_zero_point) shape audit: dilations, SAME_LOWER / SAME_UPPER auto_pad, asymmetric
    /// pads. Inputs: x [1,1,7,7] int8, w [1,1,3,3] int8, plus the quantization scalars.</summary>
    [Module]
    public partial class QeeQuantizedConvShapeAuditCheck
    {
        public static Scalar<bit> Inline(
            Tensor<int8> x, Tensor<int8> w, Scalar<int8> xZp, Scalar<int8> wZp,
            Scalar<float32> xScale, Scalar<float32> wScale, Scalar<float32> yScale, Scalar<int8> yZp)
        {
            // ConvInteger with dilations: 7 - ((3-1)*2+1) + 1 = 3; output dtype int32.
            var ciDil = (Tensor<int32>)OnnxOp.ConvInteger(x, w, xZp, wZp,
                autoPad: AutoPad.NotSet, dilations: [2L, 2L], group: 1L,
                kernelShape: [3L, 3L], pads: [0L, 0L, 0L, 0L], strides: [1L, 1L]);
            // ConvInteger SAME_LOWER: ceil(7/2) = 4 (pads unset — ORT rejects auto_pad + pads).
            var ciSame = (Tensor<int32>)OnnxOp.ConvInteger(x, w, xZp, wZp,
                autoPad: AutoPad.SameLower, dilations: [1L, 1L], group: 1L,
                kernelShape: [3L, 3L], pads: null, strides: [2L, 2L]);
            // QLinearConv SAME_UPPER: ceil(7/2) = 4; output dtype int8 (y_zero_point's type).
            var qlcSame = (Tensor<int8>)OnnxOp.QLinearConv(x, xScale, xZp, w, wScale, wZp, yScale, yZp,
                b: null, autoPad: AutoPad.SameUpper, dilations: null, group: 1L,
                kernelShape: [3L, 3L], pads: null, strides: [2L, 2L]);
            // QLinearConv asymmetric pads [2,0,0,1]: H 7+2+0-3+1 = 7, W 7+0+1-3+1 = 6.
            var qlcAsym = (Tensor<int8>)OnnxOp.QLinearConv(x, xScale, xZp, w, wScale, wZp, yScale, yZp,
                b: null, autoPad: AutoPad.NotSet, dilations: [1L, 1L], group: 1L,
                kernelShape: [3L, 3L], pads: [2L, 0L, 0L, 1L], strides: [1L, 1L]);

            var mismatch =
                ShapeMismatch(ciDil, Vector(1L, 1L, 3L, 3L)) +
                ShapeMismatch(ciSame, Vector(1L, 1L, 4L, 4L)) +
                ShapeMismatch(qlcSame, Vector(1L, 1L, 4L, 4L)) +
                ShapeMismatch(qlcAsym, Vector(1L, 1L, 7L, 6L));
            return mismatch < Scalar(1L);
        }

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

    /// <summary>MaxRoiPool shape audit: output is [num_rois, C, *pooled_shape]. Inputs:
    /// x [1,2,8,8], rois [2,5].</summary>
    [Module]
    public partial class QeeMaxRoiPoolShapeAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x, Tensor<float32> rois)
        {
            var roi = (Tensor<float32>)OnnxOp.MaxRoiPool(x, rois,
                pooledShape: [3L, 4L], spatialScale: 1f);
            var mismatch = (roi.ShapeTensor() - Vector(2L, 2L, 3L, 4L)).Abs()
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            return mismatch < Scalar(1L);
        }
    }

    /// <summary>DeformConv shape audit: asymmetric pads + stride; batch from X, channels
    /// from W.shape[0]. Inputs: x [1,1,4,4], w [1,1,2,2], offset [1,8,3,2], b [1].
    /// The optional mask input stays unbound (it is optional per spec).</summary>
    [Module]
    public partial class QeeDeformConvShapeAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x, Tensor<float32> w, Tensor<float32> offset, Vector<float32> b)
        {
            // H: (4+1+1-2)/2+1 = 3; W: (4-2)/2+1 = 2.
            var dc = (Tensor<float32>)OnnxOp.DeformConv(x, w, offset, b, mask: null,
                dilations: [1L, 1L], group: 1L, kernelShape: [2L, 2L],
                offsetGroup: 1L, pads: [1L, 0L, 1L, 0L], strides: [2L, 2L]);
            var mismatch = (dc.ShapeTensor() - Vector(1L, 1L, 3L, 2L)).Abs()
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            return mismatch < Scalar(1L);
        }
    }

    /// <summary>MaxUnpool shape audit: the inverse-pool formula with pads, and the optional
    /// output_shape input overriding the formula outright (the formula would give [1,1,4,4]).
    /// Inputs: pooled [1,1,2,2], indices [1,1,2,2] int64 (flat spatial indices that stay
    /// in-bounds for the smaller [1,1,2,4] formula output too).</summary>
    [Module]
    public partial class QeeMaxUnpoolShapeAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> pooled, Tensor<int64> indices)
        {
            // H: 2*(2-1)+2-1-1 = 2; W: 2*(2-1)+2 = 4.
            var formula = (Tensor<float32>)OnnxOp.MaxUnpool(pooled, indices,
                kernelShape: [2L, 2L], pads: [1L, 0L, 1L, 0L], strides: [2L, 2L]);
            // output_shape input overrides the formula (default would be [1,1,4,4]).
            var overridden = (Tensor<float32>)OnnxOp.MaxUnpool(pooled, indices,
                kernelShape: [2L, 2L], strides: [2L, 2L],
                outputShape: Vector(1L, 1L, 5L, 5L));

            var mismatch =
                ShapeMismatch(formula, Vector(1L, 1L, 2L, 4L)) +
                ShapeMismatch(overridden, Vector(1L, 1L, 5L, 5L));
            return mismatch < Scalar(1L);
        }

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }
}

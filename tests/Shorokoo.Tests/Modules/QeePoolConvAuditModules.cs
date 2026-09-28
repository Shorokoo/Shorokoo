using static Shorokoo.Tests.Modules.QeeAuditVerdicts;

namespace Shorokoo.Tests.Modules
{
    // ===================================================================
    //  Self-checking shape-audit modules for the QEE audit of the pooling
    //  & convolution family (ONNX opset 21). Each module runs a
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
    }

    /// <summary>Convolution variants on non-trivial weights: 1-D with asymmetric pads + stride +
    /// dilation, 2-D SAME_LOWER grouped and asymmetric-pad strided, 3-D SAME_UPPER; ConvTranspose
    /// 1-D SAME_LOWER, 2-D grouped with asymmetric pads + dilations + output_padding, 2-D with an
    /// output_shape leaving an odd total padding, 3-D strided; DeformConv with a mask, two offset
    /// groups, two groups and dilations. Inputs: x1 [2,2,9], x2 [1,4,7,6], x3 [1,2,5,4,4], and k,
    /// 1024 values the weights, offsets and mask are cut from.</summary>
    [Module]
    public partial class QeeConvVariantsShapeAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x1, Tensor<float32> x2, Tensor<float32> x3, Tensor<float32> k)
        {
            // (9+2+1-5)/2+1 = 4.
            var conv1 = NN.Conv(x1, Cut(k, 0, 3, 2, 3), Cut(k, 20, 3).Vec(), AutoPad.NotSet,
                dilations: [2L], group: 1L, kernelShape: [3L], pads: [2L, 1L], strides: [2L]);
            // SAME_LOWER: ceil(7/2) = 4, ceil(6/2) = 3.
            var convSame = NN.Conv(x2, Cut(k, 30, 4, 2, 3, 3), Cut(k, 110, 4).Vec(), AutoPad.SameLower,
                dilations: [1L, 1L], group: 2L, kernelShape: [3L, 3L], pads: null, strides: [2L, 2L]);
            // pads [1,0,0,2]: H 7+1-2+1 = 7, W (6+2-3)/2+1 = 3.
            var convAsym = NN.Conv(x2, Cut(k, 120, 2, 4, 2, 3), Cut(k, 170, 2).Vec(), AutoPad.NotSet,
                dilations: [1L, 1L], group: 1L, kernelShape: [2L, 3L], pads: [1L, 0L, 0L, 2L], strides: [1L, 2L]);
            // SAME_UPPER: [5, ceil(4/2), 4].
            var conv3 = NN.Conv(x3, Cut(k, 180, 2, 2, 3, 3, 2), Cut(k, 260, 2).Vec(), AutoPad.SameUpper,
                dilations: [1L, 1L, 1L], group: 1L, kernelShape: [3L, 3L, 2L], pads: null, strides: [1L, 2L, 1L]);
            // SAME_LOWER: 9*2 = 18.
            var convT1 = NN.ConvTranspose(x1, Cut(k, 270, 2, 3, 3), Cut(k, 290, 3).Vec(), AutoPad.SameLower,
                dilations: [1L], group: 1L, kernelShape: [3L], outputPadding: [0L], outputShape: null,
                pads: null, strides: [2L]);
            // H (7-1)*2+1+3-1-2 = 13, W (6-1)*3+2+3-0-1 = 19.
            var convTAsym = NN.ConvTranspose(x2, Cut(k, 300, 4, 2, 3, 2), Cut(k, 350, 4).Vec(), AutoPad.NotSet,
                dilations: [1L, 2L], group: 2L, kernelShape: [3L, 2L], outputPadding: [1L, 2L], outputShape: null,
                pads: [1L, 0L, 2L, 1L], strides: [2L, 3L]);
            // Full extent [15, 13]; output_shape [14, 13] leaves a total padding of 1 on H.
            var convTShape = NN.ConvTranspose(x2, Cut(k, 360, 4, 3, 3, 3), Cut(k, 470, 3).Vec(), AutoPad.NotSet,
                dilations: [1L, 1L], group: 1L, kernelShape: [3L, 3L], outputPadding: [0L, 0L], outputShape: [14L, 13L],
                pads: null, strides: [2L, 2L]);
            // (5-1)*2+2-2 = 8, (4-1)*2+2-2 = 6.
            var convT3 = NN.ConvTranspose(x3, Cut(k, 480, 2, 1, 2, 2, 2), Cut(k, 500, 1).Vec(), AutoPad.NotSet,
                dilations: [1L, 1L, 1L], group: 1L, kernelShape: [2L, 2L, 2L], outputPadding: [0L, 0L, 0L], outputShape: null,
                pads: [1L, 1L, 1L, 1L, 1L, 1L], strides: [2L, 2L, 2L]);
            // H (7+1-3)+1 = 6, W (6+1-2)+1 = 6.
            var deform = (Tensor<float32>)OnnxOp.DeformConv(x2, Cut(k, 0, 2, 2, 2, 2), Cut(k, 100, 1, 16, 6, 6) * Scalar(2f),
                Cut(k, 40, 2).Vec(), (Cut(k, 700, 1, 8, 6, 6) + Scalar(3f)) / Scalar(6f),
                dilations: [2L, 1L], group: 2L, kernelShape: [2L, 2L], offsetGroup: 2L,
                pads: [1L, 0L, 0L, 1L], strides: [1L, 1L]);

            var mismatch =
                ShapeMismatch(conv1, Vector(2L, 3L, 4L)) +
                ShapeMismatch(convSame, Vector(1L, 4L, 4L, 3L)) +
                ShapeMismatch(convAsym, Vector(1L, 2L, 7L, 3L)) +
                ShapeMismatch(conv3, Vector(1L, 2L, 5L, 2L, 4L)) +
                ShapeMismatch(convT1, Vector(2L, 3L, 18L)) +
                ShapeMismatch(convTAsym, Vector(1L, 4L, 13L, 19L)) +
                ShapeMismatch(convTShape, Vector(1L, 3L, 14L, 13L)) +
                ShapeMismatch(convT3, Vector(1L, 1L, 8L, 6L, 6L)) +
                ShapeMismatch(deform, Vector(1L, 2L, 6L, 6L));
            return mismatch < Scalar(1L);
        }

        private static Tensor<float32> Cut(Tensor<float32> k, long start, params long[] shape)
            => k.Slice(Vector(start), Vector(start + shape.Aggregate(1L, (a, b) => a * b))).Reshape(Vector(shape));
    }

    /// <summary>Pooling variants: MaxPool 1-D with ceil_mode + dilations + pads, 2-D with
    /// storage_order 1 and its Indices over asymmetric pads, 3-D SAME_UPPER; AveragePool 2-D with
    /// count_include_pad over asymmetric pads + ceil_mode, 1-D SAME_LOWER, 3-D dilated; LpPool
    /// p=1 SAME_UPPER and p=3 over asymmetric pads; the global pools in 1-D and 3-D; MaxRoiPool
    /// with a spatial_scale and fractional regions. Inputs: x1 [1,2,11], x2 [1,2,9,8],
    /// x3 [1,1,5,6,4].</summary>
    [Module]
    public partial class QeePoolVariantsShapeAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x1, Tensor<float32> x2, Tensor<float32> x3)
        {
            // ceil((11+2-5)/2)+1 = 5.
            var max1 = NN.MaxPool(x1, ceilMode: true, dilations: [2L], kernelShape: [3L],
                pads: [1L, 1L], storageOrder: 0L, strides: [2L]);
            // H (9+1-3)/2+1 = 4, W (8+1-2)/2+1 = 4.
            var (max2, indices) = OnnxOp.MaxPoolWithIndices(x2, autoPad: AutoPad.NotSet, ceilMode: false,
                dilations: [1L, 1L], kernelShape: [3L, 2L], pads: [1L, 0L, 0L, 1L], storageOrder: 1L, strides: [2L, 2L]);
            // SAME_UPPER: [ceil(5/2), ceil(6/2), ceil(4/2)].
            var max3 = (Tensor<float32>)OnnxOp.MaxPool(x3, autoPad: AutoPad.SameUpper, kernelShape: [2L, 2L, 2L],
                strides: [2L, 2L, 2L]);
            // H ceil((9+2-3)/2)+1 = 5, W ceil((8+2-3)/2)+1 = 5.
            var avgPad = (Tensor<float32>)OnnxOp.AveragePool(x2, autoPad: AutoPad.NotSet, ceilMode: true,
                countIncludePad: true, dilations: [1L, 1L], kernelShape: [3L, 3L], pads: [2L, 1L, 0L, 1L], strides: [2L, 2L]);
            // SAME_LOWER: ceil(11/3) = 4.
            var avg1 = (Tensor<float32>)OnnxOp.AveragePool(x1, autoPad: AutoPad.SameLower, ceilMode: false,
                countIncludePad: false, dilations: null, kernelShape: [4L], pads: null, strides: [3L]);
            // [5-2+1, 6-3+1, 4-2+1].
            var avg3 = (Tensor<float32>)OnnxOp.AveragePool(x3, autoPad: AutoPad.NotSet, ceilMode: false,
                countIncludePad: false, dilations: [1L, 2L, 1L], kernelShape: [2L, 2L, 2L],
                pads: [0L, 0L, 0L, 0L, 0L, 0L], strides: [1L, 1L, 1L]);
            // SAME_UPPER: [ceil(9/1), ceil(8/2)].
            var lpSame = (Tensor<float32>)OnnxOp.LpPool(x2, autoPad: AutoPad.SameUpper, ceilMode: false,
                dilations: null, kernelShape: [2L, 3L], p: 1L, pads: null, strides: [1L, 2L]);
            // 11+0+2-3+1 = 11.
            var lpAsym = (Tensor<float32>)OnnxOp.LpPool(x1, autoPad: AutoPad.NotSet, ceilMode: false,
                dilations: [1L], kernelShape: [3L], p: 3L, pads: [0L, 2L], strides: [1L]);
            var gmp = (Tensor<float32>)OnnxOp.GlobalMaxPool(x3);
            var gap = (Tensor<float32>)OnnxOp.GlobalAveragePool(x1);
            var glp = (Tensor<float32>)OnnxOp.GlobalLpPool(x3, p: 3L);
            var rois = Vector(0f, 0.6f, 1.4f, 13f, 15f, 0f, 3f, 2.5f, 7f, 9f, 0f, -2f, 0f, 4f, 20f).Reshape(Vector(3L, 5L));
            var roiPool = (Tensor<float32>)OnnxOp.MaxRoiPool(x2, rois, pooledShape: [2L, 3L], spatialScale: 0.5f);

            var mismatch =
                ShapeMismatch(max1, Vector(1L, 2L, 5L)) +
                ShapeMismatch((Tensor<float32>)max2, Vector(1L, 2L, 4L, 4L)) +
                ShapeMismatch((Tensor<int64>)indices, Vector(1L, 2L, 4L, 4L)) +
                ShapeMismatch(max3, Vector(1L, 1L, 3L, 3L, 2L)) +
                ShapeMismatch(avgPad, Vector(1L, 2L, 5L, 5L)) +
                ShapeMismatch(avg1, Vector(1L, 2L, 4L)) +
                ShapeMismatch(avg3, Vector(1L, 1L, 4L, 4L, 3L)) +
                ShapeMismatch(lpSame, Vector(1L, 2L, 9L, 4L)) +
                ShapeMismatch(lpAsym, Vector(1L, 2L, 11L)) +
                ShapeMismatch(gmp, Vector(1L, 1L, 1L, 1L, 1L)) +
                ShapeMismatch(gap, Vector(1L, 2L, 1L)) +
                ShapeMismatch(glp, Vector(1L, 1L, 1L, 1L, 1L)) +
                ShapeMismatch(roiPool, Vector(3L, 2L, 2L, 3L));
            return mismatch < Scalar(1L);
        }
    }

    /// <summary>SAME_UPPER then SAME_LOWER MaxPool with dilations: the padding follows the dilated
    /// kernel, so the output keeps ceil(in/stride) elements. Input x is [1,1,10].</summary>
    [Module]
    public partial class SameDilatedMaxPoolValues
    {
        public static Tensor<float32> Inline(Tensor<float32> x)
            => ((Tensor<float32>)OnnxOp.MaxPool(x, AutoPad.SameUpper, false, [2L], [3L], null, 0L, [2L]))
                .Concat(2L, (Tensor<float32>)OnnxOp.MaxPool(x, AutoPad.SameLower, false, [2L], [3L], null, 0L, [2L]));
    }

    /// <summary>As <see cref="SameDilatedMaxPoolValues"/>, for LpPool (p = 2).</summary>
    [Module]
    public partial class SameDilatedLpPoolValues
    {
        public static Tensor<float32> Inline(Tensor<float32> x)
            => ((Tensor<float32>)OnnxOp.LpPool(x, AutoPad.SameUpper, false, [2L], [3L], 2L, null, [2L]))
                .Concat(2L, (Tensor<float32>)OnnxOp.LpPool(x, AutoPad.SameLower, false, [2L], [3L], 2L, null, [2L]));
    }

    /// <summary>As <see cref="SameDilatedMaxPoolValues"/>, for AveragePool (count_include_pad = 0).</summary>
    [Module]
    public partial class SameDilatedAveragePoolValues
    {
        public static Tensor<float32> Inline(Tensor<float32> x)
            => ((Tensor<float32>)OnnxOp.AveragePool(x, AutoPad.SameUpper, false, false, [2L], [3L], null, [2L]))
                .Concat(2L, (Tensor<float32>)OnnxOp.AveragePool(x, AutoPad.SameLower, false, false, [2L], [3L], null, [2L]));
    }

    /// <summary>SAME-padded dilated pools over an odd length: MaxPool (SAME_UPPER, stride 2),
    /// AveragePool (SAME_LOWER, count_include_pad = 1, stride 2) and MaxPool (SAME_LOWER, stride 1).
    /// Input x is [1,1,9].</summary>
    [Module]
    public partial class SameDilatedPoolOddLengthValues
    {
        public static Tensor<float32> Inline(Tensor<float32> x)
            => ((Tensor<float32>)OnnxOp.MaxPool(x, AutoPad.SameUpper, false, [2L], [3L], null, 0L, [2L]))
                .Concat(2L, (Tensor<float32>)OnnxOp.AveragePool(x, AutoPad.SameLower, false, true, [2L], [3L], null, [2L]))
                .Concat(2L, (Tensor<float32>)OnnxOp.MaxPool(x, AutoPad.SameLower, false, [2L], [3L], null, 0L, [1L]));
    }

    /// <summary>SAME_LOWER MaxPool with dilations [2,2], kernel [3,2], strides [2,2]: values then
    /// indices. Input x is [1,1,5,6].</summary>
    [Module]
    public partial class SameDilatedMaxPoolWithIndicesValues
    {
        public static Tensor<float32> Inline(Tensor<float32> x)
        {
            var (y, indices) = OnnxOp.MaxPoolWithIndices(x, AutoPad.SameLower, false, [2L, 2L], [3L, 2L], null, 0L, [2L, 2L]);
            return ((Tensor<float32>)y).Concat(1L, ((Tensor<int64>)indices).Cast<float32>());
        }
    }

    /// <summary>Kernel 2, dilation 3, explicit pads [2,1] as large as the kernel: MaxPool,
    /// AveragePool (count_include_pad 0, then 1), LpPool (p = 2); then a ceil_mode MaxPool
    /// (kernel 2, stride 3) whose end pad of 2 reaches its kernel. Input x is [1,1,9].</summary>
    [Module]
    public partial class PadsReachingTheKernelPoolValues
    {
        public static Tensor<float32> Inline(Tensor<float32> x)
            => ((Tensor<float32>)OnnxOp.MaxPool(x, AutoPad.NotSet, false, [3L], [2L], [2L, 1L], 0L, [1L]))
                .Concat(2L, (Tensor<float32>)OnnxOp.AveragePool(x, AutoPad.NotSet, false, false, [3L], [2L], [2L, 1L], [1L]))
                .Concat(2L, (Tensor<float32>)OnnxOp.AveragePool(x, AutoPad.NotSet, false, true, [3L], [2L], [2L, 1L], [1L]))
                .Concat(2L, (Tensor<float32>)OnnxOp.LpPool(x, AutoPad.NotSet, false, [3L], [2L], 2L, [2L, 1L], [1L]))
                .Concat(2L, (Tensor<float32>)OnnxOp.MaxPool(x, AutoPad.NotSet, true, [1L], [2L], [0L, 2L], 0L, [3L]));
    }

    /// <summary>SAME pools whose dilation needs padding as large as the kernel: MaxPool
    /// (SAME_UPPER, k 2, d 3, stride 2), AveragePool (SAME_LOWER, count_include_pad 0, k 2, d 3),
    /// AveragePool (SAME_UPPER, count_include_pad 1, k 3, d 3, stride 2), LpPool (SAME_LOWER,
    /// p = 2, k 2, d 4, stride 2). Input x is [1,1,9].</summary>
    [Module]
    public partial class SameWideDilationPoolValues
    {
        public static Tensor<float32> Inline(Tensor<float32> x)
            => ((Tensor<float32>)OnnxOp.MaxPool(x, AutoPad.SameUpper, false, [3L], [2L], null, 0L, [2L]))
                .Concat(2L, (Tensor<float32>)OnnxOp.AveragePool(x, AutoPad.SameLower, false, false, [3L], [2L], null, [1L]))
                .Concat(2L, (Tensor<float32>)OnnxOp.AveragePool(x, AutoPad.SameUpper, false, true, [3L], [3L], null, [2L]))
                .Concat(2L, (Tensor<float32>)OnnxOp.LpPool(x, AutoPad.SameLower, false, [4L], [2L], 2L, null, [2L]));
    }

    /// <summary>MaxPool with indices, kernel [2,2], strides [2,2], whose padding reaches the
    /// kernel: explicit pads [2,0,1,2] with dilations [3,2], storage_order 0 then 1, and SAME_LOWER
    /// with dilations [3,3]. Each call's values then indices, flattened. Input x is [1,2,4,4].</summary>
    [Module]
    public partial class PadsReachingTheKernelMaxPoolWithIndicesValues
    {
        public static Tensor<float32> Inline(Tensor<float32> x)
        {
            var (y0, i0) = OnnxOp.MaxPoolWithIndices(x, AutoPad.NotSet, false, [3L, 2L], [2L, 2L], [2L, 0L, 1L, 2L], 0L, [2L, 2L]);
            var (y1, i1) = OnnxOp.MaxPoolWithIndices(x, AutoPad.NotSet, false, [3L, 2L], [2L, 2L], [2L, 0L, 1L, 2L], 1L, [2L, 2L]);
            var (y2, i2) = OnnxOp.MaxPoolWithIndices(x, AutoPad.SameLower, false, [3L, 3L], [2L, 2L], null, 0L, [2L, 2L]);
            var flat = OnnxOp.Constant([-1L]);
            return (Tensor<float32>)OnnxOp.Concat([
                OnnxOp.Reshape(y0, flat, false), OnnxOp.Reshape(((Tensor<int64>)i0).Cast<float32>(), flat, false),
                OnnxOp.Reshape(y1, flat, false), OnnxOp.Reshape(((Tensor<int64>)i1).Cast<float32>(), flat, false),
                OnnxOp.Reshape(y2, flat, false), OnnxOp.Reshape(((Tensor<int64>)i2).Cast<float32>(), flat, false)], 0L);
        }
    }

    /// <summary>SAME pools whose padding is negative: MaxPool k 2 d 2 s 4 (SAME_UPPER, SAME_LOWER),
    /// AveragePool count_include_pad 0 k 2 s 5 (SAME_UPPER, SAME_LOWER), LpPool p 2 k 1 s 3
    /// (SAME_UPPER), MaxPool k 2 s 4 (SAME_LOWER), MaxPool k 1 s 3 (SAME_UPPER). Input x is [1,1,8].</summary>
    [Module]
    public partial class NegativeSamePaddingPoolValues
    {
        public static Tensor<float32> Inline(Tensor<float32> x)
            => ((Tensor<float32>)OnnxOp.MaxPool(x, AutoPad.SameUpper, false, [2L], [2L], null, 0L, [4L]))
                .Concat(2L, (Tensor<float32>)OnnxOp.MaxPool(x, AutoPad.SameLower, false, [2L], [2L], null, 0L, [4L]))
                .Concat(2L, (Tensor<float32>)OnnxOp.AveragePool(x, AutoPad.SameUpper, false, false, null, [2L], null, [5L]))
                .Concat(2L, (Tensor<float32>)OnnxOp.AveragePool(x, AutoPad.SameLower, false, false, null, [2L], null, [5L]))
                .Concat(2L, (Tensor<float32>)OnnxOp.LpPool(x, AutoPad.SameUpper, false, null, [1L], 2L, null, [3L]))
                .Concat(2L, (Tensor<float32>)OnnxOp.MaxPool(x, AutoPad.SameLower, false, null, [2L], null, 0L, [4L]))
                .Concat(2L, (Tensor<float32>)OnnxOp.MaxPool(x, AutoPad.SameUpper, false, null, [1L], null, 0L, [3L]));
    }

    /// <summary>MaxPool values then indices, SAME_UPPER k [1,2] s [3,4] and SAME_LOWER k [2,1] s [3,3], whose
    /// padding is negative along an axis, each flattened. Input x is [1,2,5,6].</summary>
    [Module]
    public partial class NegativeSamePaddingMaxPoolIndicesValues
    {
        public static Tensor<float32> Inline(Tensor<float32> x)
        {
            Tensor<float32> Pooled(AutoPad same, long[] kernel, long[] strides)
            {
                var (y, indices) = OnnxOp.MaxPoolWithIndices(x, same, false, null, kernel, null, 0L, strides);
                return ((Tensor<float32>)y).Reshape(Vector(-1L)).Concat(0L, ((Tensor<int64>)indices).Cast<float32>().Reshape(Vector(-1L)));
            }
            return Pooled(AutoPad.SameUpper, [1L, 2L], [3L, 4L]).Concat(0L, Pooled(AutoPad.SameLower, [2L, 1L], [3L, 3L]));
        }
    }

    /// <summary>ceil_mode pools whose pads reach the kernel: AveragePool count_include_pad 1 and
    /// LpPool p 2 (k 2, s 3, pads [0,4]), MaxPool (k 3, d 2, s 2, pads [3,1]), MaxPool (k 2, d 2,
    /// s 3, pads [2,2]), AveragePool count_include_pad 0 (k 3, s 2, pads [1,3]). Input x is [1,1,9].</summary>
    [Module]
    public partial class CeilModePadsReachingTheKernelPoolValues
    {
        public static Tensor<float32> Inline(Tensor<float32> x)
            => ((Tensor<float32>)OnnxOp.AveragePool(x, AutoPad.NotSet, true, true, null, [2L], [0L, 4L], [3L]))
                .Concat(2L, (Tensor<float32>)OnnxOp.LpPool(x, AutoPad.NotSet, true, null, [2L], 2L, [0L, 4L], [3L]))
                .Concat(2L, (Tensor<float32>)OnnxOp.MaxPool(x, AutoPad.NotSet, true, [2L], [3L], [3L, 1L], 0L, [2L]))
                .Concat(2L, (Tensor<float32>)OnnxOp.MaxPool(x, AutoPad.NotSet, true, [2L], [2L], [2L, 2L], 0L, [3L]))
                .Concat(2L, (Tensor<float32>)OnnxOp.AveragePool(x, AutoPad.NotSet, true, false, null, [3L], [1L, 3L], [2L]));
    }

    /// <summary>MaxPool (k 2, d 3, pads [2,1]) whose pads reach the kernel over int8, uint8 (x + 4)
    /// and float16, each cast back. Input x is [1,1,9].</summary>
    [Module]
    public partial class PadsReachingTheKernelMaxPoolElementTypeValues
    {
        public static Tensor<float32> Inline(Tensor<float32> x)
        {
            Tensor<float32> Pool<T>(Tensor<T> v) where T : IVarType
                => ((Tensor<T>)OnnxOp.MaxPool(v, AutoPad.NotSet, false, [3L], [2L], [2L, 1L], 0L, [1L])).Cast<float32>();
            return Pool(x.Cast<int8>()).Concat(2L, Pool((x + Scalar(4f)).Cast<uint8>())).Concat(2L, Pool(x.Cast<float16>()));
        }
    }

    /// <summary>MaxPool over three spatial axes: k [2,2,2], d [2,2,1], s [1,2,1], pads [2,0,1,0,2,1];
    /// then AveragePool count_include_pad 1, SAME_LOWER, k [2,2,2], d [2,2,1], s [2,1,2].
    /// Input x is [1,2,3,4,3].</summary>
    [Module]
    public partial class PadsReachingTheKernelThreeAxisPoolValues
    {
        public static Tensor<float32> Inline(Tensor<float32> x)
        {
            var flat = OnnxOp.Constant([-1L]);
            return (Tensor<float32>)OnnxOp.Concat([
                OnnxOp.Reshape(OnnxOp.MaxPool(x, AutoPad.NotSet, false, [2L, 2L, 1L], [2L, 2L, 2L], [2L, 0L, 1L, 0L, 2L, 1L], 0L, [1L, 2L, 1L]), flat, false),
                OnnxOp.Reshape(OnnxOp.AveragePool(x, AutoPad.SameLower, false, true, [2L, 2L, 1L], [2L, 2L, 2L], null, [2L, 1L, 2L]), flat, false)], 0L);
        }
    }

    /// <summary>MaxPool indices of windows whose every value is the element type's lowest: uint8
    /// and int8 (x - 128) with k 2, then uint8 with k 2, d 2 and pads [2,2] reaching the kernel.
    /// Input x is [1,1,5].</summary>
    [Module]
    public partial class LowestValueWindowMaxPoolIndicesValues
    {
        public static Tensor<float32> Inline(Tensor<float32> x)
        {
            var u8 = x.Cast<uint8>();
            var (_, plain) = OnnxOp.MaxPoolWithIndices(u8, AutoPad.NotSet, false, null, [2L], null, 0L, null);
            var (_, signed) = OnnxOp.MaxPoolWithIndices((x - Scalar(128f)).Cast<int8>(), AutoPad.NotSet, false, null, [2L], null, 0L, null);
            var (_, padded) = OnnxOp.MaxPoolWithIndices(u8, AutoPad.NotSet, false, [2L], [2L], [2L, 2L], 0L, null);
            return ((Tensor<int64>)plain).Cast<float32>().Concat(2L, ((Tensor<int64>)signed).Cast<float32>())
                .Concat(2L, ((Tensor<int64>)padded).Cast<float32>());
        }
    }

    /// <summary>Whether an int8 and a uint8 MaxPool whose indices are read give the values of the same pool
    /// without them, over pads reaching the kernel (k 1, pads [1,1]) that leave whole windows in the padding.
    /// Input x is [1,1,n].</summary>
    [Module]
    public partial class IndexedSmallIntegerMaxPoolMatchesThePlainPool
    {
        public static Tensor<bit> Inline(Tensor<float32> x)
        {
            Tensor<bit> Agree<T>(Tensor<T> v) where T : IVarType
            {
                var (y, indices) = OnnxOp.MaxPoolWithIndices(v, AutoPad.NotSet, false, null, [1L], [1L, 1L], 0L, null);
                var plain = OnnxOp.MaxPool(v, AutoPad.NotSet, false, null, [1L], [1L, 1L], 0L, null);
                return ((Tensor<bit>)OnnxOp.Equal(OnnxOp.Cast(y, null, DType.Int64), OnnxOp.Cast(plain, null, DType.Int64)))
                    .Concat(2L, (Tensor<bit>)OnnxOp.GreaterOrEqual(indices, OnnxOp.Constant(-1000L)));
            }
            return Agree(x.Cast<int8>()).Concat(2L, Agree(x.Cast<uint8>()));
        }
    }

    /// <summary>Whether a uint8 MaxPool (k 2, d 2, pads [2,2] reaching the kernel) gives every window
    /// its value and the position of its first maximum. Input x is [0,0,5,0,0] as [1,1,5].</summary>
    [Module]
    public partial class PaddedMaxPoolIndicesPointAtTheirValues
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var (y, indices) = OnnxOp.MaxPoolWithIndices(x.Cast<uint8>(), AutoPad.NotSet, false, [2L], [2L], [2L, 2L], 0L, null);
            var values = OnnxOp.Equal(OnnxOp.Reshape(((Tensor<uint8>)y).Cast<int64>(), Vector(-1L), false), Vector(0L, 0L, 5L, 0L, 5L, 0L, 0L));
            var positions = OnnxOp.Equal(OnnxOp.Reshape(indices, Vector(-1L), false), Vector(0L, 1L, 2L, 1L, 2L, 3L, 4L));
            var valid = (Tensor<bit>)OnnxOp.And(values, positions);
            return valid.Cast<int32>().Reduce(ReduceKind.Min, keepDims: false).Scalar() > Scalar(0);
        }
    }

    /// <summary>MaxPool values and indices over x [1,2,2,3] holding windows whose maximum is at or
    /// below the element type's lowest finite value, against y and indices: k [2,2] with
    /// storage_order 0 and 1, then k [1,2], d [1,2] and pads [0,2,0,2] reaching the kernel, their
    /// outputs flattened and concatenated.</summary>
    [Module]
    public partial class LowestValueWindowMaxPoolCheck
    {
        public static Tensor<bit> Inline<T>(Tensor<T> x, Tensor<T> y, Tensor<int64> indices) where T : IVarType
        {
            var (y0, i0) = OnnxOp.MaxPoolWithIndices(x, AutoPad.NotSet, false, null, [2L, 2L], null, 0L, null);
            var (y1, i1) = OnnxOp.MaxPoolWithIndices(x, AutoPad.NotSet, false, null, [2L, 2L], null, 1L, null);
            var (y2, i2) = OnnxOp.MaxPoolWithIndices(x, AutoPad.NotSet, false, [1L, 2L], [1L, 2L], [0L, 2L, 0L, 2L], 0L, null);
            Variable Flat(Variable v) => OnnxOp.Cast(OnnxOp.Reshape(v, Vector(-1L), false), null, DType.Float64);
            var values = OnnxOp.Concat([Flat(y0), Flat(y1), Flat(y2)], 0L);
            var positions = OnnxOp.Concat([Flat(i0), Flat(i1), Flat(i2)], 0L);
            return (Tensor<bit>)OnnxOp.Concat([OnnxOp.Equal(values, Flat(y)), OnnxOp.Equal(positions, Flat(indices))], 0L);
        }
    }

    /// <summary>The values of a MaxPool without indices, k 2, over x [1,1,n], against y.</summary>
    [Module]
    public partial class MaxPoolValuesCheck
    {
        public static Tensor<bit> Inline<T>(Tensor<T> x, Tensor<T> y) where T : IVarType
            => (Tensor<bit>)OnnxOp.Equal(OnnxOp.Cast(OnnxOp.MaxPool(x, AutoPad.NotSet, false, null, [2L], null, null, null), null, DType.Float64),
                OnnxOp.Cast(OnnxOp.Reshape(y, Vector(1L, 1L, -1L), false), null, DType.Float64));
    }

    /// <summary>ConvTranspose SAME_UPPER then SAME_LOWER, bias 0.5, whose stride exceeds the kernel's extent plus the
    /// output padding: w [1] with s 3 and s 4, then kernel_shape [2], w [1,10], s 4, output_padding 1, each flattened.
    /// Input x is [1,1,3].</summary>
    [Module]
    public partial class ConvTransposeSameStridedPastTheKernelValues
    {
        public static Tensor<float32> Inline(Tensor<float32> x)
        {
            var w1 = Vector(1f).Reshape(Vector(1L, 1L, 1L));
            var w2 = Vector(1f, 10f).Reshape(Vector(1L, 1L, 2L));
            Tensor<float32> T(AutoPad same, Tensor<float32> w, long[]? kernel, long stride, long[]? outputPadding)
                => ((Tensor<float32>)OnnxOp.ConvTranspose(x, w, Vector(0.5f), same, null, 1L, kernel, outputPadding, null, null, [stride]))
                    .Reshape(Vector(-1L));
            return T(AutoPad.SameUpper, w1, null, 3L, null).Concat(0L, T(AutoPad.SameLower, w1, null, 3L, null))
                .Concat(0L, T(AutoPad.SameUpper, w1, null, 4L, null)).Concat(0L, T(AutoPad.SameLower, w1, null, 4L, null))
                .Concat(0L, T(AutoPad.SameUpper, w2, [2L], 4L, [1L])).Concat(0L, T(AutoPad.SameLower, w2, [2L], 4L, [1L]));
        }
    }

    /// <summary>ConvTranspose whose output_shape exceeds the full extent by a whole stride.
    /// x and w are [1,1,2,2], b is [1].</summary>
    [Module]
    public partial class ConvTransposeOversizedOutputShapeValues
    {
        public static Tensor<float32> Inline(Tensor<float32> x, Tensor<float32> w, Tensor<float32> b)
            => (Tensor<float32>)OnnxOp.ConvTranspose(x, w, b, AutoPad.NotSet, null, 1L, [2L, 2L], null, [6L, 6L], null, [2L, 2L]);
    }

    /// <summary>The ONNX output_shape example: output_shape [10,8] one past the full extent [9,7] zero-extends
    /// the end. x is [1,1,3,3], w is [1,1,3,3].</summary>
    [Module]
    public partial class ConvTransposeOutputShapeOnePastTheFullExtentValues
    {
        public static Tensor<float32> Inline(Tensor<float32> x, Tensor<float32> w)
            => (Tensor<float32>)OnnxOp.ConvTranspose(x, w, Vector(0f), AutoPad.NotSet, null, 1L, null, null, [10L, 8L], null, [3L, 2L]);
    }
}

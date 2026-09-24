using static Shorokoo.Tests.Modules.QeeAuditVerdicts;
using static Shorokoo.Tests.Modules.RecurrentAuditVerdicts;

namespace Shorokoo.Tests.Modules
{
    // ===================================================================
    //  Self-checking shape/value-audit modules for the QEE audit of the
    //  image/geometry, random/generator, and recurrent families (ONNX
    //  opset 21). Same pattern as QeePoolConvAuditModules.cs: each
    //  module runs a group of related ops under the attribute combos
    //  audited in the batch and compares every result's ShapeTensor()
    //  (and, where values are deterministic, the values) against the
    //  spec-expected constants, returning a single Scalar<bit>.
    //
    //  Driven two ways by QeeImageRandomRnnAuditTests: AdvancedTestGraph
    //  validates the expectations against real ONNX Runtime execution,
    //  and QeeAudit strict-QEE validates that QuickExecutionEngine reproduces
    //  them (Shape ops read the QEE-inferred shape, so a wrong inferred
    //  dim flips the bit to false). Modules whose checks are not
    //  QEE-computable (data-dependent NMS shapes, nondeterministic
    //  random values) are driven by AdvancedTestGraph and/or direct
    //  RuntimeTensor inspection only — see the test file.
    // ===================================================================

    /// <summary>Resize shape audit: scales-vs-sizes inputs, axes subsets (incl. a negative
    /// axis), keep_aspect_ratio_policy not_larger/not_smaller, nearest/linear/cubic modes,
    /// cubic_coeff_a, exclude_outside, antialias, and tf_crop_and_resize with a roi input +
    /// extrapolation_value. Input x is expected as [1,1,8,8].</summary>
    [Module]
    public partial class QeeResizeShapeAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            // scales over every dim: floor(8*0.5)=4, floor(8*2)=16.
            var scalesAll = (Tensor<float32>)OnnxOp.Resize(x, roi: null,
                scales: Vector(1f, 1f, 0.5f, 2f), sizes: null,
                antialias: null, axes: null, coordinateTransformationMode: null,
                cubicCoeffA: null, excludeOutside: null, extrapolationValue: null,
                keepAspectRatioPolicy: null, mode: ResizeMode.Nearest, nearestMode: null);
            // sizes over every dim (linear + align_corners): exactly [1,1,3,5].
            var sizesAll = (Tensor<float32>)OnnxOp.Resize(x, roi: null,
                scales: null, sizes: Vector(1L, 1L, 3L, 5L),
                antialias: null, axes: null,
                coordinateTransformationMode: CoordinateTransformationMode.Align_corners,
                cubicCoeffA: null, excludeOutside: null, extrapolationValue: null,
                keepAspectRatioPolicy: null, mode: ResizeMode.Linear, nearestMode: null);
            // axes subset with sizes (nearest + floor nearest_mode): only dims 2,3 change.
            var axesSizes = (Tensor<float32>)OnnxOp.Resize(x, roi: null,
                scales: null, sizes: Vector(4L, 4L),
                antialias: null, axes: [2L, 3L], coordinateTransformationMode: null,
                cubicCoeffA: null, excludeOutside: null, extrapolationValue: null,
                keepAspectRatioPolicy: null, mode: ResizeMode.Nearest,
                nearestMode: NearestMode.Floor);
            // single-axis scales: axes [3] → floor(8*0.5) = 4 (negative axes are covered by
            // the QEE-only module below — ORT 1.25.1's Resize kernel rejects them).
            var axesScales = (Tensor<float32>)OnnxOp.Resize(x, roi: null,
                scales: Vector(0.5f), sizes: null,
                antialias: null, axes: [3L], coordinateTransformationMode: null,
                cubicCoeffA: null, excludeOutside: null, extrapolationValue: null,
                keepAspectRatioPolicy: null, mode: ResizeMode.Nearest, nearestMode: null);
            // not_larger: uniform scale = min(4/8, 6/8) = 0.5 → round(0.5*8) = 4 on both axes.
            var notLarger = (Tensor<float32>)OnnxOp.Resize(x, roi: null,
                scales: null, sizes: Vector(4L, 6L),
                antialias: null, axes: [2L, 3L], coordinateTransformationMode: null,
                cubicCoeffA: null, excludeOutside: null, extrapolationValue: null,
                keepAspectRatioPolicy: KeepAspectRatioPolicy.not_larger,
                mode: ResizeMode.Nearest, nearestMode: null);
            // not_smaller: uniform scale = max(4/8, 6/8) = 0.75 → round(0.75*8) = 6.
            var notSmaller = (Tensor<float32>)OnnxOp.Resize(x, roi: null,
                scales: null, sizes: Vector(4L, 6L),
                antialias: null, axes: [2L, 3L], coordinateTransformationMode: null,
                cubicCoeffA: null, excludeOutside: null, extrapolationValue: null,
                keepAspectRatioPolicy: KeepAspectRatioPolicy.not_smaller,
                mode: ResizeMode.Nearest, nearestMode: null);
            // cubic upscale with cubic_coeff_a + exclude_outside.
            var cubic = (Tensor<float32>)OnnxOp.Resize(x, roi: null,
                scales: Vector(1f, 1f, 2f, 2f), sizes: null,
                antialias: null, axes: null, coordinateTransformationMode: null,
                cubicCoeffA: -0.5f, excludeOutside: true, extrapolationValue: null,
                keepAspectRatioPolicy: null, mode: ResizeMode.Cubic, nearestMode: null);
            // antialiased linear downscale.
            var anti = (Tensor<float32>)OnnxOp.Resize(x, roi: null,
                scales: Vector(1f, 1f, 0.5f, 0.5f), sizes: null,
                antialias: true, axes: null, coordinateTransformationMode: null,
                cubicCoeffA: null, excludeOutside: null, extrapolationValue: null,
                keepAspectRatioPolicy: null, mode: ResizeMode.Linear, nearestMode: null);
            // tf_crop_and_resize: roi [start*4, end*4] + extrapolation_value; scales halve H/W.
            var cropResize = (Tensor<float32>)OnnxOp.Resize(x,
                roi: Vector(0f, 0f, 0f, 0f, 1f, 1f, 0.5f, 0.5f),
                scales: Vector(1f, 1f, 0.5f, 0.5f), sizes: null,
                antialias: null, axes: null,
                coordinateTransformationMode: CoordinateTransformationMode.Tf_crop_and_resize,
                cubicCoeffA: null, excludeOutside: null, extrapolationValue: 3f,
                keepAspectRatioPolicy: null, mode: ResizeMode.Linear, nearestMode: null);

            var mismatch =
                ShapeMismatch(scalesAll, Vector(1L, 1L, 4L, 16L)) +
                ShapeMismatch(sizesAll, Vector(1L, 1L, 3L, 5L)) +
                ShapeMismatch(axesSizes, Vector(1L, 1L, 4L, 4L)) +
                ShapeMismatch(axesScales, Vector(1L, 1L, 8L, 4L)) +
                ShapeMismatch(notLarger, Vector(1L, 1L, 4L, 4L)) +
                ShapeMismatch(notSmaller, Vector(1L, 1L, 6L, 6L)) +
                ShapeMismatch(cubic, Vector(1L, 1L, 16L, 16L)) +
                ShapeMismatch(anti, Vector(1L, 1L, 4L, 4L)) +
                ShapeMismatch(cropResize, Vector(1L, 1L, 4L, 4L));
            return mismatch < Scalar(1L);
        }
    }

    /// <summary>Resize over every coordinate_transformation_mode and nearest_mode: nearest with
    /// round_prefer_ceil / ceil / round_prefer_floor under half_pixel_symmetric / asymmetric /
    /// pytorch_half_pixel, linear antialiased under align_corners and a one-pixel axis under
    /// pytorch_half_pixel, cubic antialiased and cubic with exclude_outside + cubic_coeff_a,
    /// cubic with a not_smaller aspect policy, and tf_crop_and_resize with nearest and with a
    /// single-axis roi. Input x is expected as [1,2,5,7].</summary>
    [Module]
    public partial class QeeResizeModesShapeAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            // floor(5*1.7) = 8, floor(7*0.6) = 4.
            var nearCeil = Resize(x, Vector(1f, 1f, 1.7f, 0.6f), null, ResizeMode.Nearest,
                CoordinateTransformationMode.Half_pixel_symmetric, NearestMode.Round_prefer_ceil);
            // floor(5*0.6) = 3, floor(7*2.5) = 17.
            var nearAsym = Resize(x, Vector(1f, 1f, 0.6f, 2.5f), null, ResizeMode.Nearest,
                CoordinateTransformationMode.Asymmetric, NearestMode.Ceil);
            var nearTorch = Resize(x, null, Vector(1L, 2L, 9L, 3L), ResizeMode.Nearest,
                CoordinateTransformationMode.Pytorch_half_pixel, NearestMode.Round_prefer_floor);
            var linearAnti = (Tensor<float32>)OnnxOp.Resize(x, roi: null, scales: null, sizes: Vector(1L, 2L, 3L, 4L),
                antialias: true, axes: null, coordinateTransformationMode: CoordinateTransformationMode.Align_corners,
                cubicCoeffA: null, excludeOutside: null, extrapolationValue: null,
                keepAspectRatioPolicy: null, mode: ResizeMode.Linear, nearestMode: null);
            var linearOnePixel = Resize(x, null, Vector(1L, 2L, 1L, 10L), ResizeMode.Linear,
                CoordinateTransformationMode.Pytorch_half_pixel, null);
            // floor(5*0.6) = 3, floor(7*0.4) = 2.
            var cubicAnti = (Tensor<float32>)OnnxOp.Resize(x, roi: null, scales: Vector(1f, 1f, 0.6f, 0.4f), sizes: null,
                antialias: true, axes: null, coordinateTransformationMode: null,
                cubicCoeffA: null, excludeOutside: null, extrapolationValue: null,
                keepAspectRatioPolicy: null, mode: ResizeMode.Cubic, nearestMode: null);
            // floor(5*1.5) = 7, floor(7*1.3) = 9.
            var cubicExclude = (Tensor<float32>)OnnxOp.Resize(x, roi: null, scales: Vector(1f, 1f, 1.5f, 1.3f), sizes: null,
                antialias: null, axes: null, coordinateTransformationMode: CoordinateTransformationMode.Asymmetric,
                cubicCoeffA: -0.5f, excludeOutside: true, extrapolationValue: null,
                keepAspectRatioPolicy: null, mode: ResizeMode.Cubic, nearestMode: null);
            // not_smaller: max(3/5, 14/7) = 2 → [10, 14].
            var cubicPolicy = (Tensor<float32>)OnnxOp.Resize(x, roi: null, scales: null, sizes: Vector(3L, 14L),
                antialias: null, axes: [2L, 3L], coordinateTransformationMode: CoordinateTransformationMode.Half_pixel_symmetric,
                cubicCoeffA: null, excludeOutside: null, extrapolationValue: null,
                keepAspectRatioPolicy: KeepAspectRatioPolicy.not_smaller, mode: ResizeMode.Cubic, nearestMode: null);
            // floor(5*1.4) = 7, floor(7*0.6) = 4.
            var cropNearest = (Tensor<float32>)OnnxOp.Resize(x, roi: Vector(0f, 0f, 0.1f, -0.2f, 1f, 1f, 0.8f, 1.3f),
                scales: Vector(1f, 1f, 1.4f, 0.6f), sizes: null, antialias: null, axes: null,
                coordinateTransformationMode: CoordinateTransformationMode.Tf_crop_and_resize,
                cubicCoeffA: null, excludeOutside: null, extrapolationValue: -5f,
                keepAspectRatioPolicy: null, mode: ResizeMode.Nearest, nearestMode: NearestMode.Floor);
            // floor(7*1.5) = 10 on the last axis only.
            var cropAxis = (Tensor<float32>)OnnxOp.Resize(x, roi: Vector(0.1f, 1.2f), scales: Vector(1.5f), sizes: null,
                antialias: null, axes: [3L], coordinateTransformationMode: CoordinateTransformationMode.Tf_crop_and_resize,
                cubicCoeffA: null, excludeOutside: null, extrapolationValue: 2f,
                keepAspectRatioPolicy: null, mode: ResizeMode.Cubic, nearestMode: null);

            var mismatch =
                ShapeMismatch(nearCeil, Vector(1L, 2L, 8L, 4L)) +
                ShapeMismatch(nearAsym, Vector(1L, 2L, 3L, 17L)) +
                ShapeMismatch(nearTorch, Vector(1L, 2L, 9L, 3L)) +
                ShapeMismatch(linearAnti, Vector(1L, 2L, 3L, 4L)) +
                ShapeMismatch(linearOnePixel, Vector(1L, 2L, 1L, 10L)) +
                ShapeMismatch(cubicAnti, Vector(1L, 2L, 3L, 2L)) +
                ShapeMismatch(cubicExclude, Vector(1L, 2L, 7L, 9L)) +
                ShapeMismatch(cubicPolicy, Vector(1L, 2L, 10L, 14L)) +
                ShapeMismatch(cropNearest, Vector(1L, 2L, 7L, 4L)) +
                ShapeMismatch(cropAxis, Vector(1L, 2L, 5L, 10L));
            return mismatch < Scalar(1L);
        }

        private static Tensor<float32> Resize(Tensor<float32> x, Vector<float32>? scales, Vector<int64>? sizes,
            ResizeMode mode, CoordinateTransformationMode transform, NearestMode? nearest)
            => (Tensor<float32>)OnnxOp.Resize(x, roi: null, scales: scales, sizes: sizes,
                antialias: null, axes: null, coordinateTransformationMode: transform,
                cubicCoeffA: null, excludeOutside: null, extrapolationValue: null,
                keepAspectRatioPolicy: null, mode: mode, nearestMode: nearest);
    }

    /// <summary>Sampling and rearrangement variants: GridSample cubic with reflection padding,
    /// nearest with zeros padding under align_corners, linear with border padding, on a rotated
    /// and scaled AffineGrid that reaches outside the image; RoiAlign avg with a sampling_ratio
    /// and spatial_scale under output_half_pixel, and max with an adaptive sampling grid;
    /// DepthToSpace in DCR and CRD modes and SpaceToDepth back; CenterCropPad cropping one axis
    /// while padding another; Col2Im with dilations, asymmetric pads and strides. Inputs:
    /// x [1,2,5,6], d [1,8,2,3], cols [1,12,12].</summary>
    [Module]
    public partial class QeeSamplingVariantsShapeAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x, Tensor<float32> d, Tensor<float32> cols)
        {
            var theta = Vector(1.1f, -0.4f, 0.1f, 0.5f, 1.2f, -0.2f).Reshape(Vector(1L, 2L, 3L));
            var grid = (Tensor<float32>)OnnxOp.AffineGrid(theta, Vector(1L, 2L, 4L, 7L), alignCorners: false);
            var gridCorners = (Tensor<float32>)OnnxOp.AffineGrid(theta, Vector(1L, 2L, 3L, 5L), alignCorners: true);
            var cubic = (Tensor<float32>)OnnxOp.GridSample(x, grid, alignCorners: false,
                mode: GridSampleMode.Cubic, paddingMode: GridSamplePaddingMode.Reflection);
            var nearest = (Tensor<float32>)OnnxOp.GridSample(x, grid, alignCorners: true,
                mode: GridSampleMode.Nearest, paddingMode: GridSamplePaddingMode.Zeros);
            var linear = (Tensor<float32>)OnnxOp.GridSample(x, grid, alignCorners: false,
                mode: GridSampleMode.Linear, paddingMode: GridSamplePaddingMode.Border);
            var rois = Vector(0.5f, 1f, 9f, 7.5f, 2.2f, 0.3f, 4.9f, 3.1f, -1f, 2f, 5f, 12f).Reshape(Vector(3L, 4L));
            var batch = Vector(0L, 0L, 0L);
            var roiAvg = (Tensor<float32>)OnnxOp.RoiAlign(x, rois, batch,
                coordinateTransformationMode: RoiAlignTransformationMode.Output_half_pixel,
                mode: RoiAlignMode.Avg, outputHeight: 2, outputWidth: 3, samplingRatio: 2, spatialScale: 0.5f);
            var roiMax = (Tensor<float32>)OnnxOp.RoiAlign(x, rois, batch,
                coordinateTransformationMode: RoiAlignTransformationMode.Half_pixel,
                mode: RoiAlignMode.Max, outputHeight: 3, outputWidth: 2, samplingRatio: 0, spatialScale: 1f);
            var dcr = (Tensor<float32>)OnnxOp.DepthToSpace(d, 2L, DepthColumnRowMode.DCR);
            var crd = (Tensor<float32>)OnnxOp.DepthToSpace(d, 2L, DepthColumnRowMode.CRD);
            var back = (Tensor<float32>)OnnxOp.SpaceToDepth(crd, 2L);
            var cropPad = (Tensor<float32>)OnnxOp.CenterCropPad(x, Vector(4L, 9L), axes: [2L, 3L]);
            var image = (Tensor<float32>)OnnxOp.Col2Im(cols, Vector(5L, 6L), Vector(2L, 3L),
                dilations: [2L, 1L], pads: [1L, 0L, 0L, 1L], strides: [1L, 2L]);

            var mismatch =
                ShapeMismatch(gridCorners, Vector(1L, 3L, 5L, 2L)) +
                ShapeMismatch(cubic, Vector(1L, 2L, 4L, 7L)) +
                ShapeMismatch(nearest, Vector(1L, 2L, 4L, 7L)) +
                ShapeMismatch(linear, Vector(1L, 2L, 4L, 7L)) +
                ShapeMismatch(roiAvg, Vector(3L, 2L, 2L, 3L)) +
                ShapeMismatch(roiMax, Vector(3L, 2L, 3L, 2L)) +
                ShapeMismatch(dcr, Vector(1L, 2L, 4L, 6L)) +
                ShapeMismatch(crd, Vector(1L, 2L, 4L, 6L)) +
                ShapeMismatch(back, Vector(1L, 8L, 2L, 3L)) +
                ShapeMismatch(cropPad, Vector(1L, 2L, 4L, 9L)) +
                ShapeMismatch(image, Vector(1L, 2L, 5L, 6L));
            return mismatch < Scalar(1L);
        }
    }

    /// <summary>Resize with NEGATIVE axes (spec opset 18+: counted from the back) — QEE-only:
    /// ONNX Runtime 1.25.1's Resize kernel rejects negative axes ("Scale value should be
    /// greater than 0"), so this module is driven through QeeAudit strict-QEE without ORT.
    /// Input x is expected as [1,1,8,8].</summary>
    [Module]
    public partial class QeeResizeNegativeAxesAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            // axes [-1] = dim 3 → floor(8*0.5) = 4.
            var negScales = (Tensor<float32>)OnnxOp.Resize(x, roi: null,
                scales: Vector(0.5f), sizes: null,
                antialias: null, axes: [-1L], coordinateTransformationMode: null,
                cubicCoeffA: null, excludeOutside: null, extrapolationValue: null,
                keepAspectRatioPolicy: null, mode: ResizeMode.Nearest, nearestMode: null);
            // axes [-2, -1] = dims 2, 3 with sizes.
            var negSizes = (Tensor<float32>)OnnxOp.Resize(x, roi: null,
                scales: null, sizes: Vector(3L, 5L),
                antialias: null, axes: [-2L, -1L], coordinateTransformationMode: null,
                cubicCoeffA: null, excludeOutside: null, extrapolationValue: null,
                keepAspectRatioPolicy: null, mode: ResizeMode.Nearest, nearestMode: null);

            var mismatch =
                ShapeMismatch(negScales, Vector(1L, 1L, 8L, 4L)) +
                ShapeMismatch(negSizes, Vector(1L, 1L, 3L, 5L));
            return mismatch < Scalar(1L);
        }
    }

    /// <summary>Upsample (deprecated — exported to ORT via the LowerUpsampleToResize
    /// lowering; floor(in*scale) shape), AffineGrid 2-D ([N,H,W,2] from theta [N,2,3]) and
    /// GridSample 4-D under linear/zeros and nearest/border attribute combos. Input x is
    /// expected as [1,2,4,4].</summary>
    [Module]
    public partial class QeeUpsampleAffineGridSampleAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            // floor semantics: [1, floor(2*2), floor(4*1.5), floor(4*2)] = [1,4,6,8].
            var up = (Tensor<float32>)OnnxOp.Upsample(x, Vector(1f, 2f, 1.5f, 2f),
                mode: ResizeMode.Nearest);

            // Identity affine theta [1,2,3]; size [N=1, C=2, H=3, W=5] → grid [1,3,5,2].
            var theta = OnnxOp.Reshape(Vector(1f, 0f, 0f, 0f, 1f, 0f), Vector(1L, 2L, 3L),
                allowZero: false);
            var grid = (Tensor<float32>)OnnxOp.AffineGrid(theta, Vector(1L, 2L, 3L, 5L),
                alignCorners: false);
            // GridSample: batch/channels from X, spatial dims from the grid → [1,2,3,5].
            var sampledLinear = (Tensor<float32>)OnnxOp.GridSample(x, grid,
                alignCorners: false, mode: GridSampleMode.Linear,
                paddingMode: GridSamplePaddingMode.Zeros);
            var sampledNearest = (Tensor<float32>)OnnxOp.GridSample(x, grid,
                alignCorners: true, mode: GridSampleMode.Nearest,
                paddingMode: GridSamplePaddingMode.Border);

            var mismatch =
                ShapeMismatch(up, Vector(1L, 4L, 6L, 8L)) +
                ShapeMismatch(grid, Vector(1L, 3L, 5L, 2L)) +
                ShapeMismatch(sampledLinear, Vector(1L, 2L, 3L, 5L)) +
                ShapeMismatch(sampledNearest, Vector(1L, 2L, 3L, 5L));
            return mismatch < Scalar(1L);
        }
    }

    /// <summary>AffineGrid 3-D (theta [N,3,4], size [N,C,D,H,W] → grid [N,D,H,W,3]) and the
    /// 5-D GridSample over it. Input x5 is expected as [1,1,3,4,4].</summary>
    [Module]
    public partial class QeeAffineGridSample5DAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x5)
        {
            var theta3d = OnnxOp.Reshape(
                Vector(1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f),
                Vector(1L, 3L, 4L), allowZero: false);
            var grid5 = (Tensor<float32>)OnnxOp.AffineGrid(theta3d, Vector(1L, 1L, 2L, 3L, 4L),
                alignCorners: true);
            var sampled5 = (Tensor<float32>)OnnxOp.GridSample(x5, grid5,
                alignCorners: true, mode: GridSampleMode.Linear,
                paddingMode: GridSamplePaddingMode.Zeros);

            var mismatch =
                ShapeMismatch(grid5, Vector(1L, 2L, 3L, 4L, 3L)) +
                ShapeMismatch(sampled5, Vector(1L, 1L, 2L, 3L, 4L));
            return mismatch < Scalar(1L);
        }
    }

    /// <summary>RoiAlign shape audit: output is [num_rois, C, output_height, output_width]
    /// under both modes / coordinate_transformation_modes / sampling_ratio / spatial_scale.
    /// Inputs: x [1,2,8,8], rois [3,4], batchIdx [3] int64.</summary>
    [Module]
    public partial class QeeRoiAlignShapeAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x, Tensor<float32> rois, Tensor<int64> batchIdx)
        {
            var avg = (Tensor<float32>)OnnxOp.RoiAlign(x, rois, batchIdx,
                coordinateTransformationMode: RoiAlignTransformationMode.Half_pixel,
                mode: RoiAlignMode.Avg, outputHeight: 3, outputWidth: 4,
                samplingRatio: 2, spatialScale: 1f);
            var max = (Tensor<float32>)OnnxOp.RoiAlign(x, rois, batchIdx,
                coordinateTransformationMode: RoiAlignTransformationMode.Output_half_pixel,
                mode: RoiAlignMode.Max, outputHeight: 2, outputWidth: 2,
                samplingRatio: 1, spatialScale: 0.5f);

            var mismatch =
                ShapeMismatch(avg, Vector(3L, 2L, 3L, 4L)) +
                ShapeMismatch(max, Vector(3L, 2L, 2L, 2L));
            return mismatch < Scalar(1L);
        }
    }

    /// <summary>NonMaxSuppression against real ORT execution: the [n,3] output shape with
    /// the data-dependent n (2 with max_output_boxes_per_class=2, 1 with 1, and 1 under
    /// center_point_box=1). QEE infers n as unknown, so this module is ORT-only; the QEE
    /// degradation (rank 2, no guessed dims, MaxShape cap) is asserted by direct
    /// RuntimeTensor inspection in the test file. Inputs: boxes [1,4,4] corner format whose
    /// first two boxes overlap heavily, scores [1,1,4].</summary>
    [Module]
    public partial class QeeNmsOrtShapeAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> boxes, Tensor<float32> scores)
        {
            // 4 boxes; boxes 0/1 overlap (IoU > 0.5), boxes 2/3 overlap → 2 survivors.
            var nms2 = (Tensor<int64>)OnnxOp.NonMaxSuppression(boxes, scores,
                Scalar(2L), Scalar(0.5f), Scalar(0.0f), centerPointBox: false);
            // Cap of one box per class → [1,3].
            var nms1 = (Tensor<int64>)OnnxOp.NonMaxSuppression(boxes, scores,
                Scalar(1L), Scalar(0.5f), Scalar(0.0f), centerPointBox: false);
            // center_point_box=1 ([cx,cy,w,h] interpretation), cap 1 → [1,3].
            var nmsCenter = (Tensor<int64>)OnnxOp.NonMaxSuppression(boxes, scores,
                Scalar(1L), Scalar(0.5f), Scalar(0.0f), centerPointBox: true);

            var mismatch =
                ShapeMismatch(nms2, Vector(2L, 3L)) +
                ShapeMismatch(nms1, Vector(1L, 3L)) +
                ShapeMismatch(nmsCenter, Vector(1L, 3L));
            return mismatch < Scalar(1L);
        }
    }

    /// <summary>NonMaxSuppression with the max_output_boxes_per_class input ABSENT — the
    /// spec default is 0 = "no output", so the result is exactly [0,3] and QEE can pin the
    /// full shape (this branch IS QeeAudit strict-QEE-able, unlike the data-dependent ones).</summary>
    [Module]
    public partial class QeeNmsEmptyAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> boxes, Tensor<float32> scores)
        {
            var nmsEmpty = (Tensor<int64>)OnnxOp.NonMaxSuppression(boxes, scores);
            var mismatch = (nmsEmpty.ShapeTensor() - Vector(0L, 3L)).Abs()
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            return mismatch < Scalar(1L);
        }
    }

    /// <summary>NonMaxSuppression output fed to direct RuntimeTensor inspection (rank-2
    /// int64, unknown shape, MaxShape = [batches*classes*min(spatial, max_boxes), 3]).</summary>
    [Module]
    public partial class QeeNmsRankOnlyCheck
    {
        public static Tensor<int64> Inline(Tensor<float32> boxes, Tensor<float32> scores)
            => (Tensor<int64>)OnnxOp.NonMaxSuppression(boxes, scores,
                Scalar(2L), Scalar(0.5f), Scalar(0.0f), centerPointBox: false);
    }

    /// <summary>Col2Im (block_shape from an INPUT: [1, C*prod(block), L] → [N,C,*image])
    /// and CenterCropPad (crop offset (in-out)/2, zero-pad begin (out-in)/2, axes subset
    /// with a negative axis) — shapes AND values, ORT-validated. Inputs: cols [1,8,12],
    /// img [3,5] with values 1..15 row-major.</summary>
    [Module]
    public partial class QeeCol2ImCenterCropPadAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> cols, Tensor<float32> img)
        {
            // [1, 8, 12] with block [2,2] → C = 8/4 = 2; image [4,5] (L = 3*4 = 12).
            var c2i = (Tensor<float32>)OnnxOp.Col2Im(cols, Vector(4L, 5L), Vector(2L, 2L),
                dilations: [1L, 1L], pads: [0L, 0L, 0L, 0L], strides: [1L, 1L]);

            // All axes: rows crop 3→2 (offset (3-2)/2 = 0 → rows 0..1), cols pad 5→7.
            // Σ out = Σ rows 0..1 of img = 55 (zero padding adds nothing).
            var cropPad = (Tensor<float32>)OnnxOp.CenterCropPad(img, Vector(2L, 7L), axes: null);
            // Negative axis subset: axes [-1], cols crop 5→4 (offset 0 → cols 0..3) → Σ = 90.
            var cropAxes = (Tensor<float32>)OnnxOp.CenterCropPad(img, Vector(4L), axes: [-1L]);
            // Pad rows 3→5: begin pad (5-3)/2 = 1 → out row 0 is zeros, out row 1 = in row 0.
            var padRows = (Tensor<float32>)OnnxOp.CenterCropPad(img, Vector(5L, 5L), axes: null);
            var padRow0 = (Tensor<float32>)OnnxOp.Gather(padRows, Scalar(0L), axis: 0);
            var padRow1 = (Tensor<float32>)OnnxOp.Gather(padRows, Scalar(1L), axis: 0);

            var shapeMismatch =
                ShapeMismatch(c2i, Vector(1L, 2L, 4L, 5L)) +
                ShapeMismatch(cropPad, Vector(2L, 7L)) +
                ShapeMismatch(cropAxes, Vector(3L, 4L)) +
                ShapeMismatch(padRows, Vector(5L, 5L));
            var valuesOk =
                ((Sum(cropPad) - Scalar(55f)).Abs() < Scalar(1e-4f)) &
                ((Sum(cropAxes) - Scalar(90f)).Abs() < Scalar(1e-4f)) &
                ((Sum(padRows) - Scalar(120f)).Abs() < Scalar(1e-4f)) &
                (Sum(padRow0).Abs() < Scalar(1e-4f)) &
                ((Sum(padRow1) - Scalar(15f)).Abs() < Scalar(1e-4f));
            return (shapeMismatch < Scalar(1L)) & valuesOk;
        }

        private static Scalar<float32> Sum(Tensor<float32> t)
            => t.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

    /// <summary>Random-generator family shape/dtype audit (values are nondeterministic and
    /// never enter the bit): RandomNormal/RandomUniform (shape + dtype attrs, default
    /// float32), the Like-variants (shape from input; dtype from input unless overridden),
    /// Bernoulli (optional dtype), Multinomial (sample_size; DEFAULT dtype int32), plus the
    /// deterministic EyeLike k-offset value path. Inputs: probs [2,3] in [0,1],
    /// logits [2,4].</summary>
    [Module]
    public partial class QeeRandomFamilyAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> probs, Tensor<float32> logits)
        {
            var rn = (Tensor<float32>)OnnxOp.RandomNormal([2L, 2L], mean: 0f, scale: 1f,
                dtype: DType.Float32, seed: 5f);
            // dtype omitted → defaults to float32 (validated by the QEE dtype pass).
            var ru = (Tensor<float32>)OnnxOp.RandomUniform([3L], high: 1f, low: 0f,
                dtype: null, seed: 3f);
            var rnl = (Tensor<float32>)OnnxOp.RandomNormalLike(probs, mean: 0f, scale: 1f,
                dtype: null, seed: 1f);
            // dtype override on the Like-variant, then the same op inheriting the input dtype.
            var rul = (Tensor<float64>)OnnxOp.RandomUniformLike(probs, high: 1f, low: 0f,
                dtype: DType.Float64, seed: 2f);
            var rulLike = (Tensor<float32>)OnnxOp.RandomUniformLike(probs, high: 1f, low: 0f,
                dtype: null, seed: 2f);
            var bern = (Tensor<float32>)OnnxOp.Bernoulli(probs, dtype: null, seed: 7f);
            var bernInt = (Tensor<int32>)OnnxOp.Bernoulli(probs, dtype: DType.Int32, seed: 7f);
            // No dtype → int32 per spec (the node-def default branch used to be unresolvable).
            var mult = (Tensor<int32>)OnnxOp.Multinomial(logits, dtype: null, sampleSize: 5L, seed: 11f);

            // EyeLike is deterministic: ones on diagonal k. [2,4] k=1 → 2 ones; k=-1 → 1 one.
            var eyeUp = (Tensor<float32>)OnnxOp.EyeLike(logits, dtype: null, k: 1);
            var eyeDown = (Tensor<int64>)OnnxOp.EyeLike(logits, dtype: DType.Int64, k: -1);

            var mismatch =
                ShapeMismatch(rn, Vector(2L, 2L)) +
                ShapeMismatch(ru, Vector(3L)) +
                ShapeMismatch(rnl, Vector(2L, 3L)) +
                ShapeMismatch(rul, Vector(2L, 3L)) +
                ShapeMismatch(rulLike, Vector(2L, 3L)) +
                ShapeMismatch(bern, Vector(2L, 3L)) +
                ShapeMismatch(bernInt, Vector(2L, 3L)) +
                ShapeMismatch(mult, Vector(2L, 5L)) +
                ShapeMismatch(eyeUp, Vector(2L, 4L)) +
                ShapeMismatch(eyeDown, Vector(2L, 4L));
            var eyeValuesOk =
                ((eyeUp.Reduce(ReduceKind.Sum, keepDims: false).Scalar() - Scalar(2f)).Abs() < Scalar(1e-5f)) &
                ((eyeDown.Reduce(ReduceKind.Sum, keepDims: false).Scalar() - Scalar(1L)).Abs() < Scalar(1L));
            return (mismatch < Scalar(1L)) & eyeValuesOk;
        }
    }

    /// <summary>Dropout: outside training (training_mode false) and at ratio 0 it is the identity
    /// with an all-true mask; in training at ratio 0.5 (values random, so shape-checked, and every
    /// element either dropped or scaled by 2) the mask is bool of the input's shape. Input
    /// x = [1, 2, 3, 4].</summary>
    [Module]
    public partial class QeeDropoutAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var (off, offMask) = OnnxOp.Dropout(x, Scalar(0.5f), Scalar(false));
            var (none, noneMask) = OnnxOp.Dropout(x, Scalar(0f), Scalar(true));
            var (on, onMask) = OnnxOp.Dropout(x, Scalar(0.5f), Scalar(true), seed: 3L);
            var kept = (Tensor<float32>)on;
            var droppedOrScaled = ((Tensor<bit>)OnnxOp.Or(OnnxOp.Equal(kept, Scalar(0f)), OnnxOp.Equal(kept, x * Scalar(2f))))
                .Cast<int64>().Reduce(ReduceKind.Min, keepDims: false).Scalar();
            var mismatch =
                FloatMismatch((Tensor<float32>)off, Vector(1f, 2f, 3f, 4f)) +
                IntMismatch(((Tensor<bit>)offMask!).Cast<int64>(), Vector(1L, 1L, 1L, 1L)) +
                FloatMismatch((Tensor<float32>)none, Vector(1f, 2f, 3f, 4f)) +
                IntMismatch(((Tensor<bit>)noneMask!).Cast<int64>(), Vector(1L, 1L, 1L, 1L)) +
                ShapeMismatch(kept, Vector(4L)) +
                ShapeMismatch((Tensor<bit>)onMask!, Vector(4L)) +
                IntMismatch1(droppedOrScaled, 1L);
            return mismatch < Scalar(1L);
        }
    }

    /// <summary>Shorokoo's keyed generator, which is deterministic integer arithmetic (Threefry-2x32
    /// over uint32, packed into uint64) and so must match on every backend value for value: the
    /// standard uniform at 20 and 13 rounds, the dense uniform over [−2, 3), the standard normal,
    /// raw uint32 and uint64 bits, and a batch of key splits. Input x = zeros [3, 5] (its shape
    /// is the draws').</summary>
    [Module]
    public partial class QeeKeyedRngValueAuditCheck
    {
        public static (Tensor<float32>, Tensor<float32>, Tensor<float32>, Tensor<float32>, Tensor<uint32>, Tensor<uint64>, Tensor<uint64>)
            Inline(Tensor<float32> x)
        {
            var shape = x.ShapeTensor();
            var key = Scalar(0x9E3779B97F4A7C15UL);
            var substream = Scalar(5UL);
            return (
                Shorokoo.Core.Rng.RuntimeRng.StandardUniform(shape, key, substream),
                Shorokoo.Core.Rng.RuntimeRng.StandardUniform(shape, key, substream, Shorokoo.Core.Rng.Threefry2x32.Rounds13),
                Shorokoo.Core.Rng.RuntimeRng.Uniform(shape, key, substream, Scalar(-2f), Scalar(3f)),
                Shorokoo.Core.Rng.RuntimeRng.StandardNormal(shape, key, substream),
                Shorokoo.Core.Rng.RuntimeRng.BitsU32(shape, key, substream),
                Shorokoo.Core.Rng.RuntimeRng.BitsU64(shape, key, substream),
                Shorokoo.Core.Rng.RuntimeRng.BatchSplitKeys(Vector(1UL, 0xFFFFFFFFFFFFFFFFUL, 1UL << 63), Vector(0UL, 7UL, 0xFFFFFFFFUL)));
        }
    }

    /// <summary>Seeded determinism (ORT-only — QEE never computes random values): two
    /// RandomNormal / RandomUniform nodes with identical seed + distribution params must
    /// produce identical streams per spec.</summary>
    [Module]
    public partial class QeeRandomSeededDeterminismCheck
    {
        public static Scalar<bit> Inline()
        {
            var rn1 = (Tensor<float32>)OnnxOp.RandomNormal([2L, 3L], mean: 0f, scale: 1f,
                dtype: DType.Float32, seed: 42f);
            var rn2 = (Tensor<float32>)OnnxOp.RandomNormal([2L, 3L], mean: 0f, scale: 1f,
                dtype: DType.Float32, seed: 42f);
            var ru1 = (Tensor<float32>)OnnxOp.RandomUniform([2L, 3L], high: 1f, low: 0f,
                dtype: DType.Float32, seed: 7f);
            var ru2 = (Tensor<float32>)OnnxOp.RandomUniform([2L, 3L], high: 1f, low: 0f,
                dtype: DType.Float32, seed: 7f);
            var diff = (rn1 - rn2).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar()
                     + (ru1 - ru2).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            return diff < Scalar(1e-6f);
        }
    }

    /// <summary>Range (count = max(ceil((limit-start)/delta), 0) — int, negative-delta,
    /// empty, and float variants, value-checked) and ConstantOfShape (the value attribute
    /// defines dtype + fill; int64 and bool fills here, float32-zero default covered by
    /// GenericConstantOfShapeLayer).</summary>
    [Module]
    public partial class QeeRangeConstantOfShapeAuditCheck
    {
        public static Scalar<bit> Inline()
        {
            var rInt = (Tensor<int64>)OnnxOp.Range(Scalar(0L), Scalar(7L), Scalar(2L));
            var rNeg = (Tensor<int64>)OnnxOp.Range(Scalar(5L), Scalar(-5L), Scalar(-3L));
            var rEmpty = (Tensor<int64>)OnnxOp.Range(Scalar(3L), Scalar(3L), Scalar(1L));
            var rFloat = (Tensor<float32>)OnnxOp.Range(Scalar(0.5f), Scalar(2f), Scalar(0.5f));

            var cosInt = (Tensor<int64>)OnnxOp.ConstantOfShape(Vector(2L, 3L),
                TensorData(DType.Int64, [1L], 5L).MoveToAttribute());
            var cosBool = (Tensor<bit>)OnnxOp.ConstantOfShape(Vector(2L, 2L),
                TensorData(DType.Bool, [1L], true).MoveToAttribute());

            var shapeMismatch =
                ShapeMismatch(rInt, Vector(4L)) +
                ShapeMismatch(rNeg, Vector(4L)) +
                ShapeMismatch(rEmpty, Vector(0L)) +
                ShapeMismatch(rFloat, Vector(3L)) +
                ShapeMismatch(cosInt, Vector(2L, 3L)) +
                ShapeMismatch(cosBool, Vector(2L, 2L));
            var valuesOk =
                ((rInt - Vector(0L, 2L, 4L, 6L)).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar() < Scalar(1L)) &
                ((rNeg - Vector(5L, 2L, -1L, -4L)).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar() < Scalar(1L)) &
                ((rFloat - Vector(0.5f, 1f, 1.5f)).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar() < Scalar(1e-5f)) &
                ((cosInt.Reduce(ReduceKind.Sum, keepDims: false).Scalar() - Scalar(30L)).Abs() < Scalar(1L)) &
                ((cosBool.Cast<int64>().Reduce(ReduceKind.Sum, keepDims: false).Scalar() - Scalar(4L)).Abs() < Scalar(1L));
            return (shapeMismatch < Scalar(1L)) & valuesOk;
        }
    }

    /// <summary>Constant value_string / value_strings branches — output inspected directly
    /// through QEE (string tensors don't fit the bit-check arithmetic).</summary>
    [Module]
    public partial class QeeConstantStringCheck
    {
        public static (Tensor<utf8>, Tensor<utf8>) Inline()
        {
            var cs = (Tensor<utf8>)OnnxOp.Constant("hello");
            var css = (Tensor<utf8>)OnnxOp.Constant((string[])["a", "b", "c"]);
            return (cs, css);
        }
    }

    // ===================================================================
    //  Recurrent family: shape-only audits (QEE computes no recurrent
    //  values). All-ones weights via InitSimple keep ORT happy while the
    //  bit only reads ShapeTensors.
    // ===================================================================

    /// <summary>RNN shape audit (ORT-validated — ORT's CPU kernels mandate the hidden_size
    /// attribute, so it is always set here; hidden-size inference from W/R is covered by the
    /// QEE-only module below): forward and bidirectional (num_directions = 2, with B +
    /// activations). Input x is expected as [4,2,3] (seq, batch, input).</summary>
    [Module]
    public partial class QeeRnnShapeAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var w1 = InitSimple.Init([Scalar(1L), Scalar(5L), Scalar(3L)]);
            var r1 = InitSimple.Init([Scalar(1L), Scalar(5L), Scalar(5L)]);
            var (y1, yh1) = OnnxOp.Rnn(x, w1, r1, null, null, null,
                null, null, null, null, RNNDirection.Forward, 5L, false);

            var w2 = InitSimple.Init([Scalar(2L), Scalar(5L), Scalar(3L)]);
            var r2 = InitSimple.Init([Scalar(2L), Scalar(5L), Scalar(5L)]);
            var b2 = InitSimple.Init([Scalar(2L), Scalar(10L)]);
            var (y3, yh3) = OnnxOp.Rnn(x, w2, r2, b2, null, null,
                null, null, ["Tanh", "Tanh"], null, RNNDirection.Bidirectional, 5L, false);

            var mismatch =
                ShapeMismatch((Tensor<float32>)y1, Vector(4L, 1L, 2L, 5L)) +
                ShapeMismatch((Tensor<float32>)yh1, Vector(1L, 2L, 5L)) +
                ShapeMismatch((Tensor<float32>)y3, Vector(4L, 2L, 2L, 5L)) +
                ShapeMismatch((Tensor<float32>)yh3, Vector(2L, 2L, 5L));
            return mismatch < Scalar(1L);
        }
    }

    /// <summary>GRU shape audit (ORT-validated): forward with linear_before_reset and
    /// bidirectional. Input x is expected as [4,2,3].</summary>
    [Module]
    public partial class QeeGruShapeAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var w1 = InitSimple.Init([Scalar(1L), Scalar(15L), Scalar(3L)]);
            var r1 = InitSimple.Init([Scalar(1L), Scalar(15L), Scalar(5L)]);
            var (y1, yh1) = OnnxOp.Gru(x, w1, r1, null, null, null,
                null, null, null, null, GRUDirection.Forward, 5L, false, linearBeforeReset: true);

            var w2 = InitSimple.Init([Scalar(2L), Scalar(15L), Scalar(3L)]);
            var r2 = InitSimple.Init([Scalar(2L), Scalar(15L), Scalar(5L)]);
            var (y2, yh2) = OnnxOp.Gru(x, w2, r2, null, null, null,
                null, null, null, null, GRUDirection.Bidirectional, 5L, false, null);

            var mismatch =
                ShapeMismatch((Tensor<float32>)y1, Vector(4L, 1L, 2L, 5L)) +
                ShapeMismatch((Tensor<float32>)yh1, Vector(1L, 2L, 5L)) +
                ShapeMismatch((Tensor<float32>)y2, Vector(4L, 2L, 2L, 5L)) +
                ShapeMismatch((Tensor<float32>)yh2, Vector(2L, 2L, 5L));
            return mismatch < Scalar(1L);
        }
    }

    /// <summary>QEE-only recurrent variants that ORT's CPU kernels reject: hidden_size
    /// omitted (kernel mandates the attribute; QEE infers hidden from W.shape[1]/gates) and
    /// layout=1 ("Batchwise recurrent operations (layout == 1) are not supported" on CPU —
    /// shape semantics still follow the opset 14+ spec: Y [batch, seq, dirs, hidden],
    /// Y_h/Y_c [batch, dirs, hidden]). Input x is expected as [4,2,3].</summary>
    [Module]
    public partial class QeeRecurrentQeeOnlyShapeAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var wRnn = InitSimple.Init([Scalar(1L), Scalar(5L), Scalar(3L)]);
            var rRnn = InitSimple.Init([Scalar(1L), Scalar(5L), Scalar(5L)]);
            var wGru = InitSimple.Init([Scalar(1L), Scalar(15L), Scalar(3L)]);
            var rGru = InitSimple.Init([Scalar(1L), Scalar(15L), Scalar(5L)]);
            var wLstm = InitSimple.Init([Scalar(1L), Scalar(20L), Scalar(3L)]);
            var rLstm = InitSimple.Init([Scalar(1L), Scalar(20L), Scalar(5L)]);

            // hidden_size attr omitted → inferred from W.shape[1] / gates (1 / 3 / 4).
            var (yRnn, yhRnn) = OnnxOp.Rnn(x, wRnn, rRnn, null, null, null,
                null, null, null, null, RNNDirection.Forward, null, false);
            var (yGru, yhGru) = OnnxOp.Gru(x, wGru, rGru, null, null, null,
                null, null, null, null, GRUDirection.Forward, null, false, null);
            var (yLstm, yhLstm, ycLstm) = OnnxOp.Lstm(x, wLstm, rLstm, null, null, null, null, null,
                null, null, null, null, LSTMDirection.Forward, null, null, false);

            // layout=1 (batch-first): x becomes [batch, seq, input] in-graph.
            var xBatchFirst = (Tensor<float32>)OnnxOp.Transpose(x, [1L, 0L, 2L]);
            var (yRnnL, yhRnnL) = OnnxOp.Rnn(xBatchFirst, wRnn, rRnn, null, null, null,
                null, null, null, null, RNNDirection.Forward, 5L, true);
            var (yGruL, yhGruL) = OnnxOp.Gru(xBatchFirst, wGru, rGru, null, null, null,
                null, null, null, null, GRUDirection.Forward, 5L, true, null);
            var (yLstmL, yhLstmL, ycLstmL) = OnnxOp.Lstm(xBatchFirst, wLstm, rLstm, null, null, null, null, null,
                null, null, null, null, LSTMDirection.Forward, 5L, null, true);

            var mismatch =
                ShapeMismatch((Tensor<float32>)yRnn, Vector(4L, 1L, 2L, 5L)) +
                ShapeMismatch((Tensor<float32>)yhRnn, Vector(1L, 2L, 5L)) +
                ShapeMismatch((Tensor<float32>)yGru, Vector(4L, 1L, 2L, 5L)) +
                ShapeMismatch((Tensor<float32>)yhGru, Vector(1L, 2L, 5L)) +
                ShapeMismatch((Tensor<float32>)yLstm, Vector(4L, 1L, 2L, 5L)) +
                ShapeMismatch((Tensor<float32>)yhLstm, Vector(1L, 2L, 5L)) +
                ShapeMismatch((Tensor<float32>)ycLstm, Vector(1L, 2L, 5L)) +
                ShapeMismatch((Tensor<float32>)yRnnL, Vector(2L, 4L, 1L, 5L)) +
                ShapeMismatch((Tensor<float32>)yhRnnL, Vector(2L, 1L, 5L)) +
                ShapeMismatch((Tensor<float32>)yGruL, Vector(2L, 4L, 1L, 5L)) +
                ShapeMismatch((Tensor<float32>)yhGruL, Vector(2L, 1L, 5L)) +
                ShapeMismatch((Tensor<float32>)yLstmL, Vector(2L, 4L, 1L, 5L)) +
                ShapeMismatch((Tensor<float32>)yhLstmL, Vector(2L, 1L, 5L)) +
                ShapeMismatch((Tensor<float32>)ycLstmL, Vector(2L, 1L, 5L));
            return mismatch < Scalar(1L);
        }
    }

    /// <summary>LSTM shape audit: forward with every optional input wired (B,
    /// sequence_lens, initial_h, initial_c, peephole P) producing (Y, Y_h, Y_c), and
    /// bidirectional with input_forget. Inputs: x [4,2,3], seqLens [2] int32 (= seq
    /// length).</summary>
    [Module]
    public partial class QeeLstmShapeAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x, Tensor<int32> seqLens)
        {
            var w1 = InitSimple.Init([Scalar(1L), Scalar(20L), Scalar(3L)]);
            var r1 = InitSimple.Init([Scalar(1L), Scalar(20L), Scalar(5L)]);
            var b1 = InitSimple.Init([Scalar(1L), Scalar(40L)]);
            var h0 = InitSimple.Init([Scalar(1L), Scalar(2L), Scalar(5L)]);
            var c0 = InitSimple.Init([Scalar(1L), Scalar(2L), Scalar(5L)]);
            var p1 = InitSimple.Init([Scalar(1L), Scalar(15L)]);
            var (y1, yh1, yc1) = OnnxOp.Lstm(x, w1, r1, b1, seqLens, h0, c0, p1,
                null, null, null, null, LSTMDirection.Forward, 5L, null, false);

            var w2 = InitSimple.Init([Scalar(2L), Scalar(20L), Scalar(3L)]);
            var r2 = InitSimple.Init([Scalar(2L), Scalar(20L), Scalar(5L)]);
            var (y2, yh2, yc2) = OnnxOp.Lstm(x, w2, r2, null, null, null, null, null,
                null, null, null, null, LSTMDirection.Bidirectional, 5L, inputForget: true, layout: false);

            var mismatch =
                ShapeMismatch((Tensor<float32>)y1, Vector(4L, 1L, 2L, 5L)) +
                ShapeMismatch((Tensor<float32>)yh1, Vector(1L, 2L, 5L)) +
                ShapeMismatch((Tensor<float32>)yc1, Vector(1L, 2L, 5L)) +
                ShapeMismatch((Tensor<float32>)y2, Vector(4L, 2L, 2L, 5L)) +
                ShapeMismatch((Tensor<float32>)yh2, Vector(2L, 2L, 5L)) +
                ShapeMismatch((Tensor<float32>)yc2, Vector(2L, 2L, 5L));
            return mismatch < Scalar(1L);
        }
    }


    // ===================================================================
    //  Recurrent family: value audits. QEE computes no recurrent values,
    //  so these run on ONNX Runtime (and every value is compared on the
    //  PyTorch backend); the bit checks the identities any correct
    //  implementation satisfies: Y_h is Y's last step (the first, for the
    //  reverse direction) and Y is zero past a sequence's length.
    //  Weights arrive as runtime inputs so nothing is folded away.
    // ===================================================================

    /// <summary>RNN values: forward with B and initial_h; reverse with Relu and clip;
    /// bidirectional with sequence_lens, initial_h and Affine / ScaledTanh alpha-beta lists.
    /// Inputs: x [4,2,3], w [2,5,3], r [2,5,5], b [2,10], h0 [2,2,5], seqLens [4,2].</summary>
    [Module]
    public partial class QeeRnnValueAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x, Tensor<float32> w, Tensor<float32> r,
            Tensor<float32> b, Tensor<float32> h0, Tensor<int32> seqLens)
        {
            var (y1, yh1) = OnnxOp.Rnn(x, First(w), First(r), First(b), null, First(h0),
                null, null, null, null, RNNDirection.Forward, 5L, false);
            var (y2, yh2) = OnnxOp.Rnn(x, First(w), First(r), null, null, null,
                null, null, ["Relu"], 0.8f, RNNDirection.Reverse, 5L, false);
            var (y3, yh3) = OnnxOp.Rnn(x, w, r, b, seqLens, h0,
                [0.7f, 0.9f], [0.2f, 0.5f], ["Affine", "ScaledTanh"], null, RNNDirection.Bidirectional, 5L, false);

            var mismatch =
                ShapeMismatch((Tensor<float32>)y3, Vector(4L, 2L, 2L, 5L)) +
                ShapeMismatch((Tensor<float32>)yh3, Vector(2L, 2L, 5L)) +
                Apart(Step((Tensor<float32>)y1, 3L), (Tensor<float32>)yh1) +
                Apart(Step((Tensor<float32>)y2, 0L), (Tensor<float32>)yh2) +
                PastLength((Tensor<float32>)y3);
            return mismatch < Scalar(1L);
        }
    }

    /// <summary>GRU values: forward with linear_before_reset, B and initial_h; reverse
    /// without it, with clip; bidirectional with sequence_lens and HardSigmoid / Softsign /
    /// Sigmoid / ScaledTanh alpha-beta lists. Inputs: x [4,2,3], w [2,15,3], r [2,15,5],
    /// b [2,30], h0 [2,2,5], seqLens [4,2].</summary>
    [Module]
    public partial class QeeGruValueAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x, Tensor<float32> w, Tensor<float32> r,
            Tensor<float32> b, Tensor<float32> h0, Tensor<int32> seqLens)
        {
            var (y1, yh1) = OnnxOp.Gru(x, First(w), First(r), First(b), null, First(h0),
                null, null, null, null, GRUDirection.Forward, 5L, false, linearBeforeReset: true);
            var (y2, yh2) = OnnxOp.Gru(x, First(w), First(r), First(b), null, null,
                null, null, null, 0.6f, GRUDirection.Reverse, 5L, false, linearBeforeReset: false);
            var (y3, yh3) = OnnxOp.Gru(x, w, r, b, seqLens, h0,
                [0.3f, 0.9f], [0.4f, 1.2f], ["HardSigmoid", "Softsign", "Sigmoid", "ScaledTanh"], null,
                GRUDirection.Bidirectional, 5L, false, linearBeforeReset: true);

            var mismatch =
                ShapeMismatch((Tensor<float32>)y3, Vector(4L, 2L, 2L, 5L)) +
                ShapeMismatch((Tensor<float32>)yh3, Vector(2L, 2L, 5L)) +
                Apart(Step((Tensor<float32>)y1, 3L), (Tensor<float32>)yh1) +
                Apart(Step((Tensor<float32>)y2, 0L), (Tensor<float32>)yh2) +
                PastLength((Tensor<float32>)y3);
            return mismatch < Scalar(1L);
        }
    }

    /// <summary>LSTM values: forward with B, initial_h, initial_c, peepholes and clip;
    /// reverse with input_forget; bidirectional with sequence_lens and a six-entry activation
    /// list (HardSigmoid / Elu / Softplus on the reverse half). Inputs: x [4,2,3],
    /// w [2,20,3], r [2,20,5], b [2,40], h0 [2,2,5], c0 [2,2,5], p [2,15], seqLens [4,2].</summary>
    [Module]
    public partial class QeeLstmValueAuditCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x, Tensor<float32> w, Tensor<float32> r,
            Tensor<float32> b, Tensor<float32> h0, Tensor<float32> c0, Tensor<float32> p, Tensor<int32> seqLens)
        {
            var (y1, yh1, _) = OnnxOp.Lstm(x, First(w), First(r), First(b), null, First(h0), First(c0), First(p),
                null, null, null, 0.9f, LSTMDirection.Forward, 5L, null, false);
            var (y2, yh2, _) = OnnxOp.Lstm(x, First(w), First(r), First(b), null, null, null, null,
                null, null, null, null, LSTMDirection.Reverse, 5L, inputForget: true, layout: false);
            var (y3, yh3, yc3) = OnnxOp.Lstm(x, w, r, b, seqLens, h0, c0, p,
                [0.3f, 0.8f], [0.4f], ["Sigmoid", "Tanh", "Tanh", "HardSigmoid", "Elu", "Softplus"], null,
                LSTMDirection.Bidirectional, 5L, null, false);

            var mismatch =
                ShapeMismatch((Tensor<float32>)y3, Vector(4L, 2L, 2L, 5L)) +
                ShapeMismatch((Tensor<float32>)yh3, Vector(2L, 2L, 5L)) +
                ShapeMismatch((Tensor<float32>)yc3, Vector(2L, 2L, 5L)) +
                Apart(Step((Tensor<float32>)y1, 3L), (Tensor<float32>)yh1) +
                Apart(Step((Tensor<float32>)y2, 0L), (Tensor<float32>)yh2) +
                PastLength((Tensor<float32>)y3);
            return mismatch < Scalar(1L);
        }
    }

    /// <summary>layout=1 (batch-first) against layout=0 on the same data, for bidirectional
    /// RNN, GRU and LSTM with sequence_lens and initial states: Y is layout 0's Y permuted
    /// [seq,dirs,batch,hidden] → [batch,seq,dirs,hidden], and Y_h / Y_c are transposed. ONNX
    /// Runtime's CPU kernels refuse layout=1, so this runs on the PyTorch backend. Inputs as in
    /// <see cref="QeeLstmValueAuditCheck"/>.</summary>
    [Module]
    public partial class QeeRecurrentBatchFirstValueCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x, Tensor<float32> w, Tensor<float32> r,
            Tensor<float32> b, Tensor<float32> h0, Tensor<float32> c0, Tensor<float32> p, Tensor<int32> seqLens)
        {
            var xB = BatchFirst(x);
            var h0B = BatchFirst(h0);
            var c0B = BatchFirst(c0);
            var wRnn = w.Slice(Vector(0L), Vector(5L), axes: Vector(1L));
            var rRnn = r.Slice(Vector(0L), Vector(5L), axes: Vector(1L));
            var bRnn = b.Slice(Vector(0L), Vector(10L), axes: Vector(1L));
            var wGru = w.Slice(Vector(0L), Vector(15L), axes: Vector(1L));
            var rGru = r.Slice(Vector(0L), Vector(15L), axes: Vector(1L));
            var bGru = b.Slice(Vector(0L), Vector(30L), axes: Vector(1L));

            var (yR, yhR) = OnnxOp.Rnn(x, wRnn, rRnn, bRnn, seqLens, h0, null, null, null, null, RNNDirection.Bidirectional, 5L, false);
            var (yRB, yhRB) = OnnxOp.Rnn(xB, wRnn, rRnn, bRnn, seqLens, h0B, null, null, null, null, RNNDirection.Bidirectional, 5L, true);
            var (yG, yhG) = OnnxOp.Gru(x, wGru, rGru, bGru, seqLens, h0, null, null, null, null, GRUDirection.Bidirectional, 5L, false, true);
            var (yGB, yhGB) = OnnxOp.Gru(xB, wGru, rGru, bGru, seqLens, h0B, null, null, null, null, GRUDirection.Bidirectional, 5L, true, true);
            var (yL, yhL, ycL) = OnnxOp.Lstm(x, w, r, b, seqLens, h0, c0, p, null, null, null, null, LSTMDirection.Bidirectional, 5L, null, false);
            var (yLB, yhLB, ycLB) = OnnxOp.Lstm(xB, w, r, b, seqLens, h0B, c0B, p, null, null, null, null, LSTMDirection.Bidirectional, 5L, null, true);

            var mismatch =
                Apart(SeqFirstY((Tensor<float32>)yRB), (Tensor<float32>)yR) +
                Apart(BatchFirst((Tensor<float32>)yhRB), (Tensor<float32>)yhR) +
                Apart(SeqFirstY((Tensor<float32>)yGB), (Tensor<float32>)yG) +
                Apart(BatchFirst((Tensor<float32>)yhGB), (Tensor<float32>)yhG) +
                Apart(SeqFirstY((Tensor<float32>)yLB), (Tensor<float32>)yL) +
                Apart(BatchFirst((Tensor<float32>)yhLB), (Tensor<float32>)yhL) +
                Apart(BatchFirst((Tensor<float32>)ycLB), (Tensor<float32>)ycL) +
                ShapeMismatch((Tensor<float32>)yLB, Vector(2L, 4L, 2L, 5L));
            return mismatch < Scalar(1L);
        }

        private static Tensor<float32> BatchFirst(Tensor<float32> t) => (Tensor<float32>)OnnxOp.Transpose(t, [1L, 0L, 2L]);

        private static Tensor<float32> SeqFirstY(Tensor<float32> y) => (Tensor<float32>)OnnxOp.Transpose(y, [1L, 2L, 0L, 3L]);
    }

    internal static class RecurrentAuditVerdicts
    {
        internal static Tensor<float32> First(Tensor<float32> t)
            => t.Slice(Vector(0L), Vector(1L), axes: Vector(0L));

        internal static Tensor<float32> Step(Tensor<float32> y, long t)
            => y.Slice(Vector(t), Vector(t + 1L), axes: Vector(0L)).Reshape(Vector(-1L));

        internal static Scalar<int64> PastLength(Tensor<float32> y)
            => Apart(y.Slice(Vector(2L, 1L), Vector(4L, 2L), axes: Vector(0L, 2L)), Vector(0f).Tensor());
    }

    /// <summary>Activation arguments ONNX Runtime reads otherwise than the spec: a bidirectional
    /// RNN whose beta list is shorter than its directions ([LeakyRelu, Affine], alpha [0.3, 0.7],
    /// beta [0.2]) or whose one alpha belongs to its second activation, a GRU leaving Affine's and
    /// ThresholdedRelu's alpha to their defaults of 1, and a bidirectional LSTM consuming one alpha
    /// among three activations that take one. The values are ONNX Runtime's, and the PyTorch
    /// backend must agree with them. Inputs: x [4,2,3], w [2,20,3], r [2,20,5], b [2,40].</summary>
    [Module]
    public partial class QeeRecurrentActivationArgumentsValueCheck
    {
        public static (Tensor<float32>, Tensor<float32>, Tensor<float32>, Tensor<float32>) Inline(
            Tensor<float32> x, Tensor<float32> w, Tensor<float32> r, Tensor<float32> b)
        {
            var wRnn = w.Slice(Vector(0L), Vector(5L), axes: Vector(1L));
            var rRnn = r.Slice(Vector(0L), Vector(5L), axes: Vector(1L));
            var bRnn = b.Slice(Vector(0L), Vector(10L), axes: Vector(1L));
            var wGru = w.Slice(Vector(0L, 0L), Vector(1L, 15L), axes: Vector(0L, 1L));
            var rGru = r.Slice(Vector(0L, 0L), Vector(1L, 15L), axes: Vector(0L, 1L));
            var bGru = b.Slice(Vector(0L, 0L), Vector(1L, 30L), axes: Vector(0L, 1L));

            var (y1, _) = OnnxOp.Rnn(x, wRnn, rRnn, bRnn, null, null,
                [0.3f, 0.7f], [0.2f], ["LeakyRelu", "Affine"], null, RNNDirection.Bidirectional, 5L, false);
            var (y2, _) = OnnxOp.Rnn(x, wRnn, rRnn, bRnn, null, null,
                [0.3f], null, ["Tanh", "LeakyRelu"], null, RNNDirection.Bidirectional, 5L, false);
            var (y3, _) = OnnxOp.Gru(x, wGru, rGru, bGru, null, null,
                null, null, ["Affine", "ThresholdedRelu"], null, GRUDirection.Forward, 5L, false);
            var (y4, _, _) = OnnxOp.Lstm(x, w, r, b, null, null, null, null,
                [0.1f], null, ["HardSigmoid", "Tanh", "Elu", "Sigmoid", "Relu", "LeakyRelu"], null, LSTMDirection.Bidirectional, 5L, null, false);
            return ((Tensor<float32>)y1, (Tensor<float32>)y2, (Tensor<float32>)y3, (Tensor<float32>)y4);
        }
    }

    /// <summary>ImageDecoder values in every pixel format: an RGB PNG, an RGB JPEG and a greyscale
    /// JPEG, each decoded to HWC uint8. ONNX Runtime has no ImageDecoder kernel, so this runs on
    /// the PyTorch backend.</summary>
    [Module]
    public partial class QeeImageDecoderValueCheck
    {
        public static (Tensor<uint8>, Tensor<uint8>, Tensor<uint8>, Tensor<uint8>, Tensor<uint8>, Tensor<uint8>) Inline(
            Vector<uint8> png, Vector<uint8> jpeg, Vector<uint8> greyJpeg)
            => ((Tensor<uint8>)OnnxOp.ImageDecoder(png, pixelFormat: "RGB"),
                (Tensor<uint8>)OnnxOp.ImageDecoder(png, pixelFormat: "BGR"),
                (Tensor<uint8>)OnnxOp.ImageDecoder(png, pixelFormat: "Grayscale"),
                (Tensor<uint8>)OnnxOp.ImageDecoder(jpeg),
                (Tensor<uint8>)OnnxOp.ImageDecoder(jpeg, pixelFormat: "Grayscale"),
                (Tensor<uint8>)OnnxOp.ImageDecoder(greyJpeg, pixelFormat: "BGR"));
    }

    /// <summary>tf_crop_and_resize at scale 1 still crops to the roi: linear over scales, then nearest over sizes.
    /// Input x is [1,1,1,5].</summary>
    [Module]
    public partial class CropAndResizeAtScaleOneValues
    {
        public static Tensor<float32> Inline(Tensor<float32> x)
        {
            var roi = Vector(0f, 0f, 0f, 0.5f, 1f, 1f, 1f, 1.5f);
            var linear = (Tensor<float32>)OnnxOp.Resize(x, roi: roi, scales: Vector(1f, 1f, 1f, 1f), sizes: null,
                antialias: null, axes: null, coordinateTransformationMode: CoordinateTransformationMode.Tf_crop_and_resize,
                cubicCoeffA: null, excludeOutside: null, extrapolationValue: -1f,
                keepAspectRatioPolicy: null, mode: ResizeMode.Linear, nearestMode: null);
            var nearest = (Tensor<float32>)OnnxOp.Resize(x, roi: roi, scales: null, sizes: Vector(1L, 1L, 1L, 5L),
                antialias: null, axes: null, coordinateTransformationMode: CoordinateTransformationMode.Tf_crop_and_resize,
                cubicCoeffA: null, excludeOutside: null, extrapolationValue: -1f,
                keepAspectRatioPolicy: null, mode: ResizeMode.Nearest, nearestMode: null);
            return linear.Concat(3L, nearest);
        }
    }

    /// <summary>1-D Col2Im with pads and a stride. Input cols is [1,3,4].</summary>
    [Module]
    public partial class Col2Im1DPaddedValues
    {
        public static Tensor<float32> Inline(Tensor<float32> cols)
            => (Tensor<float32>)OnnxOp.Col2Im(cols, Vector(8L), Vector(3L), dilations: [1L], pads: [1L, 1L], strides: [2L]);
    }
}

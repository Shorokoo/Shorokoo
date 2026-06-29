namespace Shorokoo.Tests.Modules
{
    // ===================================================================
    //  Self-checking shape/value-audit modules for the Phase 4 QEE-A5
    //  batch (image/geometry, random/generator, and recurrent families,
    //  ONNX opset 21). Same pattern as QeePoolConvAuditModules.cs: each
    //  module runs a group of related ops under the attribute combos
    //  audited in the batch and compares every result's ShapeTensor()
    //  (and, where values are deterministic, the values) against the
    //  spec-expected constants, returning a single Scalar<bit>.
    //
    //  Driven two ways by QeeImageRandomRnnAuditTests: AdvancedTestGraph
    //  validates the expectations against real ONNX Runtime execution,
    //  and QeeSelfCheck validates that QuickExecutionEngine reproduces
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

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

    /// <summary>Resize with NEGATIVE axes (spec opset 18+: counted from the back) — QEE-only:
    /// ONNX Runtime 1.25.1's Resize kernel rejects negative axes ("Scale value should be
    /// greater than 0"), so this module is driven through QeeSelfCheck without ORT.
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

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
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

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
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

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
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

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
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

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

    /// <summary>NonMaxSuppression with the max_output_boxes_per_class input ABSENT — the
    /// spec default is 0 = "no output", so the result is exactly [0,3] and QEE can pin the
    /// full shape (this branch IS QeeSelfCheck-able, unlike the data-dependent ones).</summary>
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

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
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
            // dtype override on the Like-variant.
            var rul = (Tensor<float64>)OnnxOp.RandomUniformLike(probs, high: 1f, low: 0f,
                dtype: DType.Float64, seed: 2f);
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

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
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
                TensorData(DType.Int64, [1L], 5L));
            var cosBool = (Tensor<bit>)OnnxOp.ConstantOfShape(Vector(2L, 2L),
                TensorData(DType.Bool, [1L], true));

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

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

    /// <summary>Constant value_string / value_strings branches — output inspected directly
    /// through QEE (string tensors don't fit the bit-check arithmetic).</summary>
    [Module]
    public partial class QeeConstantStringCheck
    {
        public static (Tensor<@string>, Tensor<@string>) Inline()
        {
            var cs = (Tensor<@string>)OnnxOp.Constant("hello");
            var css = (Tensor<@string>)OnnxOp.Constant(new[] { "a", "b", "c" });
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

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
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

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
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

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
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

        private static Scalar<int64> ShapeMismatch(ITensor t, Vector<int64> expected)
            => (t.TShape - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }

}

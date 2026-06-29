namespace Shorokoo.Tests.Modules
{
    // ===================================================================
    //  Modules that exercise the QEE shape-inference handlers added by
    //  the AutoGrad/QEE op expansion batch (Hardmax/OneHot/etc. file
    //  group under src/Shorokoo/Core/Inference/Ops/). Mirrors the
    //  one-liner pattern in QeeOpsTestModules.cs: each module chains
    //  several related ops so a single Coverage [Fact] driving one
    //  Module class widens QEE coverage across many branches that the
    //  existing AutoGrad and QEE suites never reach.
    //
    //  Each test module focuses on ops that DO have an OnnxOp factory
    //  method — ops like Hardmax / OneHot / IsInf / IsNaN exist only
    //  via ONNX import and have no in-framework constructor, so they
    //  can't be reached through these test graphs.
    // ===================================================================

    /// <summary>SpaceToDepth + DepthToSpace shape transforms — round-trip a 4-D image.</summary>
    [Module]
    public partial class QeeSpaceDepthOpsCheck
    {
        public static (Tensor<float32>, Tensor<float32>) Inline(Tensor<float32> x)
        {
            var sd = (Tensor<float32>)OnnxOp.SpaceToDepth(x, blockSize: 2);
            var ds = (Tensor<float32>)OnnxOp.DepthToSpace(sd, blockSize: 2, mode: DepthColumnRowMode.DCR);
            return (sd, ds);
        }
    }

    /// <summary>CenterCropPad with default axes (every dim) and with an explicit axes attr.</summary>
    [Module]
    public partial class QeeCenterCropPadOpsCheck
    {
        public static (Tensor<float32>, Tensor<float32>) Inline(Tensor<float32> x)
        {
            // No axes → shape input must cover every dim.
            var allAxes = (Tensor<float32>)OnnxOp.CenterCropPad(x, Vector(2L, 2L), axes: null);
            // With axes → only the listed dim resizes.
            var oneAxis = (Tensor<float32>)OnnxOp.CenterCropPad(x, Vector(2L), axes: new long[] { 1L });
            return (allAxes, oneAxis);
        }
    }

    /// <summary>AffineGrid + GridSample — affine sampler emitting [N,H,W,2] grids and resampling X.</summary>
    [Module]
    public partial class QeeAffineGridSampleOpsCheck
    {
        public static (Tensor<float32>, Tensor<float32>) Inline(Tensor<float32> theta, Tensor<float32> x)
        {
            var grid = (Tensor<float32>)OnnxOp.AffineGrid(theta, Vector(1L, 1L, 2L, 2L), alignCorners: false);
            var sampled = (Tensor<float32>)OnnxOp.GridSample(x, grid,
                alignCorners: false, mode: GridSampleMode.Linear, paddingMode: GridSamplePaddingMode.Zeros);
            return (grid, sampled);
        }
    }

    /// <summary>MaxUnpool from pre-computed pooled tensor + indices. The MAX_UNPOOL node
    /// definition types the indices as int64 per the ONNX spec (fixed in the Phase 4 QEE-A1
    /// audit batch — it previously reused X's generic T).</summary>
    [Module]
    public partial class QeeMaxUnpoolOpsCheck
    {
        public static Tensor<float32> Inline(Tensor<float32> pooled, Tensor<int64> indices)
            => (Tensor<float32>)OnnxOp.MaxUnpool(pooled, indices,
                kernelShape: new long[] { 2, 2 },
                pads: new long[] { 0, 0, 0, 0 },
                strides: new long[] { 2, 2 });
    }

    /// <summary>LRN + LpPool + GlobalLpPool — three spatial pooling variants in one module.</summary>
    [Module]
    public partial class QeePoolingVariantsOpsCheck
    {
        public static (Tensor<float32>, Tensor<float32>, Tensor<float32>) Inline(Tensor<float32> x)
        {
            var lrn = (Tensor<float32>)OnnxOp.Lrn(x, alpha: 1e-4f, beta: 0.75f, bias: 1f, size: 3L);
            var lp = (Tensor<float32>)OnnxOp.LpPool(x,
                autoPad: AutoPad.NotSet, ceilMode: false, dilations: null,
                kernelShape: new long[] { 2, 2 }, p: 2L,
                pads: new long[] { 0, 0, 0, 0 }, strides: new long[] { 1, 1 });
            var glp = (Tensor<float32>)OnnxOp.GlobalLpPool(x, p: 2L);
            return (lrn, lp, glp);
        }
    }

    /// <summary>Bernoulli — random sample whose output dtype defaults to the input dtype.</summary>
    [Module]
    public partial class QeeBernoulliOpsCheck
    {
        public static Tensor<float32> Inline(Tensor<float32> x)
            => (Tensor<float32>)OnnxOp.Bernoulli(x, dtype: null, seed: 1f);
    }

    /// <summary>BitShift — integer bit shift in both directions on uint64 (the spec restricts
    /// BitShift to unsigned integer types).</summary>
    [Module]
    public partial class QeeBitShiftOpsCheck
    {
        public static (Tensor<uint64>, Tensor<uint64>) Inline(Tensor<uint64> a, Tensor<uint64> b)
        {
            var shiftL = (Tensor<uint64>)OnnxOp.BitShift(a, b, direction: BitShiftDirection.Left);
            var shiftR = (Tensor<uint64>)OnnxOp.BitShift(a, b, direction: BitShiftDirection.Right);
            return (shiftL, shiftR);
        }
    }

    /// <summary>Compress along an explicit axis + Compress with no axis (flatten path).</summary>
    [Module]
    public partial class QeeCompressOpsCheck
    {
        public static (Tensor<float32>, Tensor<float32>) Inline(Tensor<float32> x, Tensor<bit> cond)
        {
            var alongAxis = (Tensor<float32>)OnnxOp.Compress(x, cond, axis: 0);
            var flattened = (Tensor<float32>)OnnxOp.Compress(x, cond, axis: null);
            return (alongAxis, flattened);
        }
    }

    /// <summary>ReverseSequence with explicit batch_axis/time_axis attrs.</summary>
    [Module]
    public partial class QeeReverseSequenceOpsCheck
    {
        public static Tensor<float32> Inline(Tensor<float32> x, Tensor<int64> seqLens)
            => (Tensor<float32>)OnnxOp.ReverseSequence(x, seqLens, batchAxis: 0, timeAxis: 1);
    }

    /// <summary>CumSum with both reverse modes and exclusive=true to hit each toggle path.</summary>
    [Module]
    public partial class QeeCumSumVariantsOpsCheck
    {
        public static (Tensor<float32>, Tensor<float32>, Tensor<float32>) Inline(Tensor<float32> x)
        {
            var c1 = (Tensor<float32>)OnnxOp.CumSum(x, Scalar(0L), exclusive: false, reverse: false);
            var c2 = (Tensor<float32>)OnnxOp.CumSum(x, Scalar(0L), exclusive: true, reverse: false);
            var c3 = (Tensor<float32>)OnnxOp.CumSum(x, Scalar(0L), exclusive: false, reverse: true);
            return (c1, c2, c3);
        }
    }

    /// <summary>Det + Einsum — scalar determinant of a batched matrix + index-contract Einsum.</summary>
    [Module]
    public partial class QeeDetEinsumOpsCheck
    {
        public static (Tensor<float32>, Tensor<float32>) Inline(Tensor<float32> mat)
        {
            var det = (Tensor<float32>)OnnxOp.Det(mat);
            // Einsum with explicit output: "ij,ji->" computes the trace of A·B.
            var trace = (Tensor<float32>)OnnxOp.Einsum((Variable[])[ mat, mat ], equation: "ij,ji->");
            return (det, trace);
        }
    }

    /// <summary>Einsum implicit-output form ("ij" → "ji"-style permutation).</summary>
    [Module]
    public partial class QeeEinsumImplicitOpsCheck
    {
        public static Tensor<float32> Inline(Tensor<float32> a, Tensor<float32> b)
            // Implicit-output: labels that appear once are summed out in alphabetical order.
            => (Tensor<float32>)OnnxOp.Einsum((Variable[])[ a, b ], equation: "ij,jk");
    }

    /// <summary>Unique with explicit axis (returns 4 outputs).</summary>
    [Module]
    public partial class QeeUniqueOpsCheck
    {
        public static (Tensor<float32>, Tensor<int64>, Tensor<int64>, Tensor<int64>) Inline(Tensor<float32> x)
        {
            var (y, idx, inv, counts) = OnnxOp.Unique(x, axis: 0, sorted: true);
            return ((Tensor<float32>)y, (Tensor<int64>)idx, (Tensor<int64>)inv, (Tensor<int64>)counts);
        }
    }

    /// <summary>Unique with no axis (flattens input first).</summary>
    [Module]
    public partial class QeeUniqueFlatOpsCheck
    {
        public static (Tensor<float32>, Tensor<int64>) Inline(Tensor<float32> x)
        {
            var (y, idx, _, _) = OnnxOp.Unique(x, axis: null, sorted: true);
            return ((Tensor<float32>)y, (Tensor<int64>)idx);
        }
    }

    /// <summary>NonMaxSuppression — IoU-based box filtering with the standard 5-input layout.</summary>
    [Module]
    public partial class QeeNonMaxSuppressionOpsCheck
    {
        public static Tensor<int64> Inline(Tensor<float32> boxes, Tensor<float32> scores)
            => (Tensor<int64>)OnnxOp.NonMaxSuppression(boxes, scores,
                maxOutputBoxesPerClass: Scalar(2L),
                iouThreshold: Scalar(0.5f),
                scoreThreshold: Scalar(0f),
                centerPointBox: false);
    }

    /// <summary>Upsample with explicit scales (the deprecated pre-Resize API).</summary>
    [Module]
    public partial class QeeUpsampleOpsCheck
    {
        public static Tensor<float32> Inline(Tensor<float32> x)
            => (Tensor<float32>)OnnxOp.Upsample(x, Vector(1f, 1f, 2f, 2f), mode: ResizeMode.Nearest);
    }

    /// <summary>Col2Im — inverse of the implicit im2col, reconstructs a feature map.</summary>
    [Module]
    public partial class QeeCol2ImOpsCheck
    {
        public static Tensor<float32> Inline(Tensor<float32> cols)
            => (Tensor<float32>)OnnxOp.Col2Im(cols,
                imageShape: Vector(4L, 4L),
                blockShape: Vector(2L, 2L),
                dilations: new long[] { 1, 1 }, pads: new long[] { 0, 0, 0, 0 },
                strides: new long[] { 2, 2 });
    }

    /// <summary>DeformConv — deformable Conv with offset (and optional mask) inputs.</summary>
    [Module]
    public partial class QeeDeformConvOpsCheck
    {
        public static Tensor<float32> Inline(Tensor<float32> x, Tensor<float32> w, Tensor<float32> offset, Vector<float32> b)
            => (Tensor<float32>)OnnxOp.DeformConv(x, w, offset, b, mask: null,
                dilations: new long[] { 1, 1 }, group: 1, kernelShape: new long[] { 2, 2 },
                offsetGroup: 1, pads: new long[] { 0, 0, 0, 0 }, strides: new long[] { 1, 1 });
    }

    /// <summary>ConvInteger — quantized int8 Conv producing an int32 accumulator output.</summary>
    [Module]
    public partial class QeeConvIntegerOpsCheck
    {
        public static Tensor<int32> Inline(Tensor<int8> x, Tensor<int8> w, Scalar<int8> xZp, Scalar<int8> wZp)
            => (Tensor<int32>)OnnxOp.ConvInteger(x, w, xZp, wZp,
                autoPad: AutoPad.NotSet, dilations: new long[] { 1, 1 }, group: 1,
                kernelShape: new long[] { 2, 2 },
                pads: new long[] { 0, 0, 0, 0 }, strides: new long[] { 1, 1 });
    }

    /// <summary>MatMulInteger — quantized int8 MatMul producing an int32 result.</summary>
    [Module]
    public partial class QeeMatMulIntegerOpsCheck
    {
        public static Tensor<int32> Inline(Tensor<int8> a, Tensor<int8> b, Scalar<int8> aZp, Scalar<int8> bZp)
            => (Tensor<int32>)OnnxOp.MatMulInteger(a, b, aZp, bZp);
    }

    /// <summary>DequantizeLinear — dequantize int8 → float32 with explicit scale + zero point.</summary>
    [Module]
    public partial class QeeDequantizeLinearOpsCheck
    {
        public static Tensor<float32> Inline(Tensor<int8> x, Tensor<float32> scale, Tensor<int8> zp)
            => (Tensor<float32>)OnnxOp.DequantizeLinear(x, scale, zp, axis: 1, blockSize: null);
    }

    /// <summary>DynamicQuantizeLinear — three-output dynamic quant on a float32 scalar.</summary>
    [Module]
    public partial class QeeDynamicQuantizeLinearOpsCheck
    {
        public static (Scalar<uint8>, Scalar<float32>, Scalar<uint8>) Inline(Scalar<float32> x)
        {
            var (y, scale, zp) = OnnxOp.DynamicQuantizeLinear(x);
            return ((Scalar<uint8>)y, (Scalar<float32>)scale, (Scalar<uint8>)zp);
        }
    }

    /// <summary>shrk_Conv — Shorokoo's tensor-attr Conv variant lowered before execution.</summary>
    [Module]
    public partial class QeeShrkConvOpsCheck
    {
        public static Tensor<float32> Inline(Tensor<float32> x, Tensor<float32> w, Vector<float32> b)
            => (Tensor<float32>)InternalOp.Conv(x, w, b, AutoPad.NotSet,
                pads: Vector(0L, 0L, 0L, 0L),
                strides: Vector(1L, 1L),
                dilations: Vector(1L, 1L),
                kernelShape: Vector(2L, 2L),
                group: Scalar(1L));
    }
}

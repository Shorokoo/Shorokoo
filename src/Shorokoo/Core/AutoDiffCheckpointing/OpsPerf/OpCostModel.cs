namespace Shorokoo.Core.AutoDiffCheckpointing.OpsPerf;

/// <summary>
/// The calibrated constants behind every <see cref="IOpPerf"/> estimate, in nanoseconds of
/// ONNX Runtime CPU kernel time on the reference machine (4 threads, ORT 1.26, x64 Linux).
///
/// <para>They were fitted to ORT's profiler over the memory-pass model families (MLP, conv,
/// one- and two-layer encoders, LSTM, dense and chunked attention) at several batch sizes:
/// every kernel's median duration over repeated runs, paired with the node it ran for. The
/// harness is <c>OpsPerfCalibrationTests</c> (Manual) and the fit is
/// <c>tools/opsperf-calibration/fit.py</c>; re-run both after an ORT upgrade or on a machine
/// class that matters and paste the refitted numbers here.</para>
///
/// <para>The MatMul constants below predate the harness's per-tensor feed and are due a refit.
/// The fill they were fitted through was keyed on tensor length alone, so the encoder families'
/// attention ran on denormal operands and their MatMul kernels were profiled at up to 8x their
/// real cost; a flush-controlled A/B put <see cref="MatMulNsPerFlop"/> about 24% high because of
/// it. The pass's chosen strategy and modelled peak were unchanged on all eight families at that
/// margin, so nothing is misbehaving on these numbers — they are simply known-stale until the
/// harness is re-run on the reference machine.</para>
///
/// <para>The shape of the model matters more than the digits. Every kernel pays a fixed
/// <see cref="Launch"/>; a training step has thousands of tiny nodes, so this term is most of
/// their cost and the reason recomputing a chain of small ops is never free. Streaming ops are
/// priced per byte moved (<see cref="StreamNsPerByte"/>), so a large elementwise op costs about
/// what a small MatMul does. MatMul pays an asymptotic rate per FLOP plus a
/// <c>sqrt(FLOPs)</c> ramp that captures the poor efficiency of small products. Conv is
/// priced per FLOP with an im2col/GEMM penalty that grows as the output image shrinks —
/// the weight-gradient conv (kernel the size of the image, 3×3 output) is an order of
/// magnitude less efficient than the forward one.</para>
/// </summary>
internal static class OpCostModel
{
    /// <summary>Fixed cost of running any kernel, however small its tensors.</summary>
    public const double Launch = 4200;

    /// <summary>Contiguous streaming: ns per byte read or written by an elementwise op.</summary>
    public const double StreamNsPerByte = 0.0175;

    /// <summary>Multiplier on the streaming rate for strided copies (Concat, Slice, Pad, Expand, Gather, …).</summary>
    public const double CopyPenalty = 3.3;

    /// <summary>Multiplier on the streamed bytes of a binary op when a non-scalar input is broadcast.</summary>
    public const double BroadcastPenalty = 1.33;

    /// <summary>Transpose ns per byte (in + out) when the innermost axis moves.</summary>
    public const double TransposeInnerNsPerByte = 0.070;

    /// <summary>Transpose ns per byte (in + out) when the innermost axis stays put (block copy).</summary>
    public const double TransposeOuterNsPerByte = 0.035;

    /// <summary>Reduction ns per input byte when the reduced axes are the innermost ones.</summary>
    public const double ReduceInnerNsPerByte = 0.030;

    /// <summary>Reduction ns per input byte when a reduced axis is not innermost (strided accumulation).</summary>
    public const double ReduceOuterNsPerByte = 0.113;

    /// <summary>Softmax ns per element, plus <see cref="SoftmaxNsPerRow"/> per vector along the axis.</summary>
    public const double SoftmaxNsPerElement = 1.38;
    public const double SoftmaxNsPerRow = 82;

    /// <summary>MatMul/Gemm: ns per FLOP asymptotically, and ns per sqrt(FLOP) for the small-product ramp.</summary>
    public const double MatMulNsPerFlop = 0.00363;
    public const double MatMulNsPerSqrtFlop = 9.67;

    /// <summary>Conv: ns per FLOP for a large output image, scaled by (1 + <see cref="ConvSpatialPenalty"/> / output image size).</summary>
    public const double ConvNsPerFlop = 0.0133;
    public const double ConvSpatialPenalty = 47.7;

    /// <summary>LSTM/GRU/RNN kernels: ns per FLOP of their gate products.</summary>
    public const double RnnNsPerFlop = 0.0086;

    /// <summary>
    /// Share of int64 shape-arithmetic nodes ORT still runs at ORT_ENABLE_ALL. The training
    /// step is compiled with static input dims, so ORT folds every one of them at load: none
    /// survive (before static dims, 76% did).
    /// </summary>
    public const double ShapeArithmeticSurvival = 0.0;

    /// <summary>Share of Reshape/Squeeze/Unsqueeze/Flatten nodes ORT still runs as (aliasing) kernels.</summary>
    public const double MetadataSurvival = 0.28;

    /// <summary>A kernel that streams <paramref name="bytes"/>.</summary>
    public static double Stream(double bytes, double multiplier = 1.0)
        => Launch + StreamNsPerByte * multiplier * bytes;

    /// <summary>A kernel that copies <paramref name="bytes"/> with a strided pattern.</summary>
    public static double Copy(double bytes)
        => Stream(bytes, CopyPenalty);

    /// <summary>A MatMul-shaped product of <paramref name="flops"/> floating-point operations.</summary>
    public static double MatMul(double flops)
        => Launch + MatMulNsPerFlop * flops + MatMulNsPerSqrtFlop * System.Math.Sqrt(flops);

    /// <summary>Bytes of every present tensor, summed.</summary>
    public static double BytesOf(TensorShapeInfo?[] shapes)
    {
        double total = 0;
        foreach (var s in shapes)
            if (s is not null) total += s.MemoryBytes;
        return total;
    }

    /// <summary>
    /// Whether the node is int64 shape arithmetic (every output a non-float tensor of at most
    /// 64 bytes), the kind ORT folds away at load when the dims are static.
    /// </summary>
    public static bool IsShapeArithmetic(OpPerfInput input)
    {
        bool any = false;
        foreach (var o in input.OutputShapes)
        {
            if (o is null) continue;
            any = true;
            if (o.DType == DType.Float32 || o.MemoryBytes > 64) return false;
        }
        return any;
    }

    /// <summary>Discounts a shape-arithmetic node's cost by its runtime survival rate.</summary>
    public static double Survival(OpPerfInput input, double cost)
        => IsShapeArithmetic(input) ? ShapeArithmeticSurvival * cost : cost;
}

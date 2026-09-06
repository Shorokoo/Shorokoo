using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;

namespace Shorokoo.Core.AutoDiffCheckpointing.OpsPerf;

/// <summary>
/// Performance estimator for tensor shape manipulation operations. Metadata-only ops
/// (Reshape, Squeeze, …) alias their input and cost a fraction of a launch — ORT still runs
/// most of them as kernels. Shape/Size fold away with static dims. Everything else is a
/// copy priced per byte at the strided-copy rate; Transpose pays more when the innermost axis moves, and a
/// Transpose that only feeds a MatMul on its last two axes is folded into a
/// <c>FusedMatMul</c> flag and costs nothing (when the caller supplies
/// <see cref="OpPerfInput.ConsumerOpCodes"/>).
/// </summary>
internal class TensorManipulationPerf : IOpPerf
{
    public IReadOnlySet<string> SupportedOpCodes { get; } = new HashSet<string>
    {
        RESHAPE, FLATTEN, SQUEEZE, UNSQUEEZE, EXPAND, TRANSPOSE,
        CONCAT, SPLIT, SLICE, PAD, TILE, GATHER, GATHER_ELEMENTS, GATHER_ND,
        SCATTER_ELEMENTS, SCATTER_ND, SHAPE, SIZE, NON_ZERO,
        ONE_HOT, DEPTH_TO_SPACE, SPACE_TO_DEPTH, TRILU,
        COMPRESS, REVERSE_SEQUENCE, UNIQUE, EYE_LIKE,
        CONSTANT_OF_SHAPE, RANGE, CENTER_CROP_PAD, CLIP,
    };

    public OpPerfResult Estimate(OpPerfInput input)
    {
        var opCode = input.OpCode;

        // Shape/Size fold to constants once the dims are static
        if (opCode == SHAPE || opCode == SIZE)
            return OpPerfResult.Zero;

        var outputShape = input.OutputShapes[0];
        if (outputShape is null)
            return OpPerfResult.Zero;

        var outputElements = outputShape.ElementCount;
        var moved = OpCostModel.BytesOf(input.InputShapes) + outputShape.MemoryBytes;

        switch (opCode)
        {
            case RESHAPE:
            case FLATTEN:
            case SQUEEZE:
            case UNSQUEEZE:
            {
                // A view change: the kernel, when ORT keeps it, aliases the input buffer
                return new OpPerfResult
                {
                    ComputeTime = OpCostModel.MetadataSurvival * OpCostModel.Launch,
                    ExtraMemoryBytes = 0,
                    InPlaceBufferReuse = new Dictionary<int, int> { [0] = 0 }
                };
            }

            case EXPAND:
            {
                var inputShape = input.InputShapes[0];
                if (inputShape is not null && inputShape.ElementCount == outputElements)
                {
                    // No actual expansion needed — same size
                    return new OpPerfResult
                    {
                        ComputeTime = OpCostModel.MetadataSurvival * OpCostModel.Launch,
                        ExtraMemoryBytes = 0,
                        InPlaceBufferReuse = new Dictionary<int, int> { [0] = 0 }
                    };
                }
                return new OpPerfResult
                {
                    ComputeTime = OpCostModel.Survival(input, OpCostModel.Copy(moved)),
                    ExtraMemoryBytes = 0,
                };
            }

            case TRANSPOSE:
            {
                var inputShape = input.InputShapes[0];
                if (inputShape is null) return OpPerfResult.Zero;
                if (FusesIntoMatMul(input, inputShape))
                    return OpPerfResult.Zero;
                var rate = InnermostAxisMoves(input, inputShape)
                    ? OpCostModel.TransposeInnerNsPerByte
                    : OpCostModel.TransposeOuterNsPerByte;
                return new OpPerfResult
                {
                    ComputeTime = OpCostModel.Launch + rate * (inputShape.MemoryBytes + outputShape.MemoryBytes),
                    ExtraMemoryBytes = 0,
                };
            }

            case SCATTER_ELEMENTS:
            case SCATTER_ND:
            {
                // Check if output can reuse the data input buffer (first input)
                var canInPlace = !input.InputMustRemainIntact[0]
                    && input.InputShapes[0] is not null
                    && input.InputShapes[0]!.ElementCount == outputElements
                    && input.InputShapes[0]!.DType == outputShape.DType;
                return new OpPerfResult
                {
                    ComputeTime = OpCostModel.Stream(moved, 2.0), // random-access read-modify-write
                    ExtraMemoryBytes = 0,
                    InPlaceBufferReuse = canInPlace ? new Dictionary<int, int> { [0] = 0 } : new Dictionary<int, int>()
                };
            }

            case CLIP:
            {
                var inputShape = input.InputShapes[0];
                if (inputShape is null) return OpPerfResult.Zero;
                var canInPlace = !input.InputMustRemainIntact[0]
                    && inputShape.ElementCount == outputElements
                    && inputShape.DType == outputShape.DType;
                return new OpPerfResult
                {
                    ComputeTime = OpCostModel.Survival(input, OpCostModel.Stream(moved)),
                    ExtraMemoryBytes = 0,
                    InPlaceBufferReuse = canInPlace ? new Dictionary<int, int> { [0] = 0 } : new Dictionary<int, int>()
                };
            }

            default:
            {
                // Concat, Split, Slice, Pad, Tile, Gather*, Range, ConstantOfShape, …: a strided copy
                return new OpPerfResult
                {
                    ComputeTime = OpCostModel.Survival(input, OpCostModel.Copy(moved)),
                    ExtraMemoryBytes = 0,
                };
            }
        }
    }

    private static long[]? Perm(OpPerfInput input)
        => input.Attributes.TryGetValue("perm", out var val) && val is long[] perm ? perm : null;

    /// <summary>Without a perm the axes reverse, which always moves the innermost one.</summary>
    private static bool InnermostAxisMoves(OpPerfInput input, TensorShapeInfo inputShape)
    {
        var rank = inputShape.Shape.Dims.Length;
        var perm = Perm(input);
        return perm is null || perm.Length == 0 || perm[^1] != rank - 1;
    }

    /// <summary>
    /// ORT's MatMulTransposeFusion: a Transpose that swaps only the last two axes and whose
    /// every consumer is a MatMul is absorbed as that MatMul's transA/transB.
    /// </summary>
    private static bool FusesIntoMatMul(OpPerfInput input, TensorShapeInfo inputShape)
    {
        var consumers = input.ConsumerOpCodes;
        if (consumers is null || consumers.Count == 0) return false;
        foreach (var c in consumers)
            if (c != MATMUL) return false;
        var rank = inputShape.Shape.Dims.Length;
        var perm = Perm(input);
        if (rank < 2 || perm is null || perm.Length != rank) return false;
        if (perm[^1] != rank - 2 || perm[^2] != rank - 1) return false;
        for (int i = 0; i < rank - 2; i++)
            if (perm[i] != i) return false;
        return true;
    }
}

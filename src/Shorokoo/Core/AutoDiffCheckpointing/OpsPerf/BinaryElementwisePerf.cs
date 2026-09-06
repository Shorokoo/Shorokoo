using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;

namespace Shorokoo.Core.AutoDiffCheckpointing.OpsPerf;

/// <summary>
/// Performance estimator for elementwise binary (and <c>Where</c>) operations: one kernel
/// launch plus the bytes streamed across every input and the output. Broadcasting a
/// non-scalar input costs extra (<see cref="OpCostModel.BroadcastPenalty"/>); ORT's
/// <c>Where</c> is several times slower per byte than an add. Binary ops can operate in-place
/// on the first input when shapes match and the input is no longer needed.
/// </summary>
internal class BinaryElementwisePerf : IOpPerf
{
    // Multiplier on the streaming rate; 1 is a memory-bound op. Where and Greater were fitted.
    private static readonly Dictionary<string, double> CostMultipliers = new()
    {
        [ADD] = 1.0,
        [SUB] = 1.0,
        [MUL] = 1.0,
        [DIV] = 1.0,
        [MOD] = 2.0,
        [POW] = 3.0,
        [MAX] = 1.0,
        [MIN] = 1.0,
        [MEAN] = 1.0,
        [SUM] = 1.0,
        [AND] = 1.0,
        [OR] = 1.0,
        [XOR] = 1.0,
        [BITWISE_AND] = 1.0,
        [BITWISE_OR] = 1.0,
        [BITWISE_XOR] = 1.0,
        [BIT_SHIFT] = 1.0,
        [EQUAL] = 1.0,
        [GREATER] = 1.45,
        [GREATER_OR_EQUAL] = 1.45,
        [LESS] = 1.45,
        [LESS_OR_EQUAL] = 1.45,
        [WHERE] = 4.3,
    };

    public IReadOnlySet<string> SupportedOpCodes { get; } =
        new HashSet<string>(CostMultipliers.Keys);

    public OpPerfResult Estimate(OpPerfInput input)
    {
        var outputShape = input.OutputShapes[0];
        if (outputShape is null)
            return OpPerfResult.Zero;

        var outputElements = outputShape.ElementCount;
        var bytes = OpCostModel.BytesOf(input.InputShapes) + outputShape.MemoryBytes;
        var multiplier = CostMultipliers.GetValueOrDefault(input.OpCode, 1.0);
        if (IsBroadcast(input.InputShapes, outputElements))
            multiplier *= OpCostModel.BroadcastPenalty;
        var computeTime = OpCostModel.Survival(input, OpCostModel.Stream(bytes, multiplier));

        // Check if in-place is possible on the first input:
        // Same shape, same dtype, and input not needed later
        var canInPlaceFirst = input.InputShapes[0] is not null
            && !input.InputMustRemainIntact[0]
            && input.InputShapes[0]!.ElementCount == outputElements
            && input.InputShapes[0]!.DType == outputShape.DType;

        return new OpPerfResult
        {
            ComputeTime = computeTime,
            ExtraMemoryBytes = 0,
            InPlaceBufferReuse = canInPlaceFirst ? new Dictionary<int, int> { [0] = 0 } : new Dictionary<int, int>()
        };
    }

    /// <summary>A non-scalar input smaller than the output is broadcast along some axis.</summary>
    private static bool IsBroadcast(TensorShapeInfo?[] inputs, long outputElements)
    {
        foreach (var s in inputs)
            if (s is not null && s.ElementCount > 1 && s.ElementCount < outputElements)
                return true;
        return false;
    }
}

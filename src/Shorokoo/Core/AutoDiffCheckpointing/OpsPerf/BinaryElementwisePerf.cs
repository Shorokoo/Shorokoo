using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;

namespace Shorokoo.Core.AutoDiffCheckpointing.OpsPerf;

/// <summary>
/// Performance estimator for elementwise binary operations.
/// These ops apply a function element-by-element across two tensors (with broadcasting).
/// Compute cost is proportional to the output element count.
/// Binary ops can operate in-place on the first input when shapes match
/// and the input is no longer needed.
/// </summary>
internal class BinaryElementwisePerf : IOpPerf
{
    private static readonly Dictionary<string, double> CostMultipliers = new()
    {
        [ADD] = 1.0,
        [SUB] = 1.0,
        [MUL] = 1.0,
        [DIV] = 3.0,
        [MOD] = 4.0,
        [POW] = 5.0,
        [MAX] = 1.0,
        [MIN] = 1.0,
        [MEAN] = 1.5,
        [SUM] = 1.0,
        [AND] = 1.0,
        [OR] = 1.0,
        [XOR] = 1.0,
        [BITWISE_AND] = 1.0,
        [BITWISE_OR] = 1.0,
        [BITWISE_XOR] = 1.0,
        [BIT_SHIFT] = 1.0,
        [EQUAL] = 1.0,
        [GREATER] = 1.0,
        [GREATER_OR_EQUAL] = 1.0,
        [LESS] = 1.0,
        [LESS_OR_EQUAL] = 1.0,
        [WHERE] = 1.0, // Ternary but elementwise
    };

    public IReadOnlySet<string> SupportedOpCodes { get; } =
        new HashSet<string>(CostMultipliers.Keys);

    public OpPerfResult Estimate(OpPerfInput input)
    {
        var outputShape = input.OutputShapes[0];
        if (outputShape is null)
            return OpPerfResult.Zero;

        var outputElements = outputShape.ElementCount;
        var costMultiplier = CostMultipliers.GetValueOrDefault(input.OpCode, 1.0);

        var computeTime = (outputElements / 256.0) * costMultiplier;

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
}

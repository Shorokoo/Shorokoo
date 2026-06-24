using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;

namespace Shorokoo.Core.AutoDiffCheckpointing.OpsPerf;

/// <summary>
/// Performance estimator for reduction operations.
/// These ops reduce one or more dimensions, with cost proportional to
/// the number of input elements (each must be read and accumulated).
/// Reduction ops cannot operate in-place since the output is smaller than the input.
/// </summary>
internal class ReductionPerf : IOpPerf
{
    private static readonly Dictionary<string, double> CostMultipliers = new()
    {
        [REDUCE_SUM] = 1.0,
        [REDUCE_MEAN] = 1.5, // Sum + divide
        [REDUCE_MAX] = 1.0,
        [REDUCE_MIN] = 1.0,
        [REDUCE_PROD] = 1.5,
        [REDUCE_L1] = 1.5, // Abs + sum
        [REDUCE_L2] = 2.5, // Square + sum + sqrt
        [REDUCE_LOG_SUM] = 2.0, // Sum + log
        [REDUCE_LOG_SUM_EXP] = 5.0, // Exp + sum + log
        [REDUCE_SUM_SQUARE] = 2.0, // Square + sum
        [CUM_SUM] = 1.0,
        [SOFTMAX] = 6.0, // Exp + sum + div (multi-pass)
        [HARDMAX] = 2.0, // Max + compare
        [ARG_MAX] = 1.5, // Max with index tracking
        [ARG_MIN] = 1.5,
    };

    public IReadOnlySet<string> SupportedOpCodes { get; } =
        new HashSet<string>(CostMultipliers.Keys);

    public OpPerfResult Estimate(OpPerfInput input)
    {
        var inputShape = input.InputShapes[0];
        if (inputShape is null)
            return OpPerfResult.Zero;

        var inputElements = inputShape.ElementCount;
        var costMultiplier = CostMultipliers.GetValueOrDefault(input.OpCode, 1.0);

        // Compute cost proportional to input elements
        var computeTime = (inputElements / 256.0) * costMultiplier;

        // CumSum can be in-place since output has same shape
        var canInPlace = input.OpCode == CUM_SUM
            && !input.InputMustRemainIntact[0]
            && input.OutputShapes[0] is not null
            && inputShape.ElementCount == input.OutputShapes[0]!.ElementCount
            && inputShape.DType == input.OutputShapes[0]!.DType;

        return new OpPerfResult
        {
            ComputeTime = computeTime,
            ExtraMemoryBytes = 0,
            InPlaceBufferReuse = canInPlace ? new Dictionary<int, int> { [0] = 0 } : new Dictionary<int, int>()
        };
    }
}

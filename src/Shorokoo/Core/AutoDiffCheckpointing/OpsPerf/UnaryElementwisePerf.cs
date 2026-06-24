using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;

namespace Shorokoo.Core.AutoDiffCheckpointing.OpsPerf;

/// <summary>
/// Performance estimator for elementwise unary operations.
/// These ops apply a function to each element independently.
/// Compute cost is proportional to element count with a multiplier
/// depending on the complexity of the operation.
/// Most unary ops can operate in-place when the input is no longer needed.
/// </summary>
internal class UnaryElementwisePerf : IOpPerf
{
    // Relative cost multipliers compared to a simple add (1.0)
    private static readonly Dictionary<string, double> CostMultipliers = new()
    {
        [ABS] = 1.0,
        [NEG] = 1.0,
        [SIGN] = 1.0,
        [CEIL] = 1.0,
        [FLOOR] = 1.0,
        [ROUND] = 1.0,
        [NOT] = 1.0,
        [BITWISE_NOT] = 1.0,
        [IDENTITY] = 0.0, // Zero-copy / no-op
        [RELU] = 1.0,
        [LEAKY_RELU] = 1.5,
        [SELU] = 3.0,
        [ELU] = 3.0,
        [CELU] = 3.0,
        [GELU] = 5.0,
        [SIGMOID] = 4.0,
        [TANH] = 4.0,
        [SQRT] = 3.0,
        [RECIPROCAL] = 3.0,
        [EXP] = 4.0,
        [LOG] = 4.0,
        [SIN] = 4.0,
        [COS] = 4.0,
        [TAN] = 5.0,
        [ASIN] = 5.0,
        [ACOS] = 5.0,
        [ATAN] = 5.0,
        [SINH] = 5.0,
        [COSH] = 5.0,
        [ATANH] = 5.0,
        [ASINH] = 5.0,
        [ACOSH] = 5.0,
        [ERF] = 6.0,
        [CAST] = 1.0,
        [CAST_LIKE] = 1.0,
        [IS_INF] = 1.0,
        [IS_NAN] = 1.0,
        [DROPOUT] = 2.0, // Random gen + comparison + multiply
    };

    public IReadOnlySet<string> SupportedOpCodes { get; } =
        new HashSet<string>(CostMultipliers.Keys);

    public OpPerfResult Estimate(OpPerfInput input)
    {
        var inputShape = input.InputShapes[0];
        if (inputShape is null)
            return OpPerfResult.Zero;

        var elementCount = inputShape.ElementCount;
        var costMultiplier = CostMultipliers.GetValueOrDefault(input.OpCode, 1.0);

        // Compute time: elements / 256 (since 256 adds = 1 unit) × cost multiplier
        var computeTime = (elementCount / 256.0) * costMultiplier;

        // Identity is zero-copy
        if (input.OpCode == IDENTITY)
            return new OpPerfResult
            {
                ComputeTime = 0,
                ExtraMemoryBytes = 0,
                InPlaceBufferReuse = new Dictionary<int, int> { [0] = 0 } // Output reuses input buffer
            };

        // Most unary ops can operate in-place if input is no longer needed
        var canInPlace = !input.InputMustRemainIntact[0]
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

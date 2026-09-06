using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;

namespace Shorokoo.Core.AutoDiffCheckpointing.OpsPerf;

/// <summary>
/// Performance estimator for elementwise unary operations: one kernel launch plus the bytes
/// streamed (input and output), scaled by how much arithmetic the op does per element —
/// memory-bound ops sit at 1, transcendental ones at about 2 (see <see cref="OpCostModel"/>).
/// Most unary ops can operate in-place when the input is no longer needed.
/// </summary>
internal class UnaryElementwisePerf : IOpPerf
{
    // Multiplier on the streaming rate; 1 is a memory-bound op. Relu, Gelu and Erf were fitted
    // (ORT's Relu is not vectorised as well as Add); the rest are grouped with their kind.
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
        [IDENTITY] = 0.0, // Removed by ORT
        [RELU] = 1.3,
        [LEAKY_RELU] = 1.3,
        [SELU] = 2.0,
        [ELU] = 2.0,
        [CELU] = 2.0,
        [GELU] = 2.75,
        [SIGMOID] = 2.0,
        [TANH] = 2.0,
        [SQRT] = 1.5,
        [RECIPROCAL] = 1.5,
        [EXP] = 1.0,
        [LOG] = 1.5,
        [SIN] = 2.0,
        [COS] = 2.0,
        [TAN] = 2.0,
        [ASIN] = 2.0,
        [ACOS] = 2.0,
        [ATAN] = 2.0,
        [SINH] = 2.0,
        [COSH] = 2.0,
        [ATANH] = 2.0,
        [ASINH] = 2.0,
        [ACOSH] = 2.0,
        [ERF] = 2.35,
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

        // Identity is removed by ORT: zero cost, output aliases the input buffer
        if (input.OpCode == IDENTITY)
            return new OpPerfResult
            {
                ComputeTime = 0,
                ExtraMemoryBytes = 0,
                InPlaceBufferReuse = new Dictionary<int, int> { [0] = 0 }
            };

        var outputShape = input.OutputShapes[0];
        var bytes = inputShape.MemoryBytes + (outputShape?.MemoryBytes ?? 0);
        var computeTime = OpCostModel.Survival(input,
            OpCostModel.Stream(bytes, CostMultipliers.GetValueOrDefault(input.OpCode, 1.0)));

        var canInPlace = !input.InputMustRemainIntact[0]
            && outputShape is not null
            && inputShape.ElementCount == outputShape.ElementCount
            && inputShape.DType == outputShape.DType;

        return new OpPerfResult
        {
            ComputeTime = computeTime,
            ExtraMemoryBytes = 0,
            InPlaceBufferReuse = canInPlace ? new Dictionary<int, int> { [0] = 0 } : new Dictionary<int, int>()
        };
    }
}

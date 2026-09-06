using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;

namespace Shorokoo.Core.AutoDiffCheckpointing.OpsPerf;

/// <summary>
/// Performance estimator for reductions and softmax. A reduction streams its input once, at
/// a rate that depends on which axes go: reducing the innermost axes is a contiguous
/// accumulation, while reducing an outer axis while keeping the innermost one is a strided
/// pass about four times slower per byte (see <see cref="OpCostModel"/>). A reduction over no
/// axes (the empty-axes form the gradient rules emit) is a copy. Softmax pays per element and
/// per vector along its axis. Reduction ops cannot operate in-place since the output is smaller
/// than the input; CumSum can.
/// </summary>
internal class ReductionPerf : IOpPerf
{
    // Multiplier on the reduction rate for ops that do more than accumulate.
    private static readonly Dictionary<string, double> CostMultipliers = new()
    {
        [REDUCE_SUM] = 1.0,
        [REDUCE_MEAN] = 1.0,
        [REDUCE_MAX] = 1.0,
        [REDUCE_MIN] = 1.0,
        [REDUCE_PROD] = 1.0,
        [REDUCE_L1] = 1.0,
        [REDUCE_L2] = 1.5,
        [REDUCE_LOG_SUM] = 1.5,
        [REDUCE_LOG_SUM_EXP] = 2.0,
        [REDUCE_SUM_SQUARE] = 1.5,
        [CUM_SUM] = 1.0,
        [SOFTMAX] = 1.0,
        [LOG_SOFTMAX] = 1.0,
        [HARDMAX] = 1.0,
        [ARG_MAX] = 1.5,
        [ARG_MIN] = 1.5,
    };

    public IReadOnlySet<string> SupportedOpCodes { get; } =
        new HashSet<string>(CostMultipliers.Keys);

    public OpPerfResult Estimate(OpPerfInput input)
    {
        var inputShape = input.InputShapes[0];
        if (inputShape is null)
            return OpPerfResult.Zero;
        var outputShape = input.OutputShapes[0];

        double computeTime;
        if (input.OpCode is SOFTMAX or LOG_SOFTMAX or HARDMAX)
        {
            computeTime = EstimateSoftmax(input, inputShape);
        }
        else if (input.OpCode == CUM_SUM || outputShape is null || outputShape.ElementCount == inputShape.ElementCount)
        {
            // Nothing is reduced: one streaming pass
            computeTime = OpCostModel.Stream(inputShape.MemoryBytes + (outputShape?.MemoryBytes ?? 0));
        }
        else
        {
            var rate = ReducesInnermost(inputShape, outputShape)
                ? OpCostModel.ReduceInnerNsPerByte
                : OpCostModel.ReduceOuterNsPerByte;
            var multiplier = CostMultipliers.GetValueOrDefault(input.OpCode, 1.0);
            computeTime = OpCostModel.Launch
                + rate * multiplier * inputShape.MemoryBytes
                + OpCostModel.StreamNsPerByte * outputShape.MemoryBytes;
        }
        computeTime = OpCostModel.Survival(input, computeTime);

        // CumSum can be in-place since output has same shape
        var canInPlace = input.OpCode == CUM_SUM
            && !input.InputMustRemainIntact[0]
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

    private static double EstimateSoftmax(OpPerfInput input, TensorShapeInfo inputShape)
    {
        var dims = inputShape.Shape.Dims;
        long rowLength = 1;
        if (dims.Length > 0)
        {
            var axis = input.Attributes.TryGetValue("axis", out var a) && a is long l ? l : -1;
            if (axis < 0) axis += dims.Length;
            if (axis >= 0 && axis < dims.Length) rowLength = System.Math.Max(1, dims[axis]);
        }
        var elements = inputShape.ElementCount;
        return OpCostModel.Launch
            + OpCostModel.SoftmaxNsPerElement * elements
            + OpCostModel.SoftmaxNsPerRow * (elements / rowLength);
    }

    /// <summary>
    /// Whether the reduced axes are the innermost ones, read off the shapes: with keepdims the
    /// last output dim collapses to 1 (or was 1 already); without, the last dims disappear so
    /// the output's last dim differs from the input's.
    /// </summary>
    private static bool ReducesInnermost(TensorShapeInfo inputShape, TensorShapeInfo outputShape)
    {
        var inDims = inputShape.Shape.Dims;
        var outDims = outputShape.Shape.Dims;
        if (inDims.Length == 0) return true;
        if (outDims.Length == inDims.Length)
            return outDims[^1] == 1 || inDims[^1] == 1;
        return outDims.Length == 0 || outDims[^1] != inDims[^1];
    }
}

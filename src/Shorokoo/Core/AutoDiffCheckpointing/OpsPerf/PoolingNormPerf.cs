using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;

namespace Shorokoo.Core.AutoDiffCheckpointing.OpsPerf;

/// <summary>
/// Performance estimator for pooling and normalization operations, priced as streaming
/// passes over the data (<see cref="OpCostModel.Stream"/>): a pool reads its window per
/// output element, a normalization makes a few passes for statistics and the rescale. Not
/// calibrated directly — none of the reference families pools or batch-normalizes — so these
/// carry the elementwise rate with a pass count.
/// </summary>
internal class PoolingNormPerf : IOpPerf
{
    public IReadOnlySet<string> SupportedOpCodes { get; } = new HashSet<string>
    {
        AVERAGE_POOL, MAX_POOL, LP_POOL,
        GLOBAL_AVERAGE_POOL, GLOBAL_MAX_POOL, GLOBAL_LP_POOL,
        BATCH_NORMALIZATION, INSTANCE_NORMALIZATION, GROUP_NORMALIZATION,
        LP_NORMALIZATION, LRN,
    };

    public OpPerfResult Estimate(OpPerfInput input)
    {
        var opCode = input.OpCode;

        if (opCode is AVERAGE_POOL or MAX_POOL or LP_POOL)
            return EstimatePool(input);

        if (opCode is GLOBAL_AVERAGE_POOL or GLOBAL_MAX_POOL or GLOBAL_LP_POOL)
            return EstimateGlobalPool(input);

        if (opCode is BATCH_NORMALIZATION or INSTANCE_NORMALIZATION
            or GROUP_NORMALIZATION or LP_NORMALIZATION or LRN)
            return EstimateNorm(input);

        return OpPerfResult.Zero;
    }

    private static OpPerfResult EstimatePool(OpPerfInput input)
    {
        var inputShape = input.InputShapes[0];
        var outputShape = input.OutputShapes[0];
        if (inputShape is null || outputShape is null)
            return OpPerfResult.Zero;

        var kernelShape = GetLongsAttr(input, "kernel_shape");
        long kernelVolume = 1;
        if (kernelShape is not null)
        {
            foreach (var k in kernelShape)
                kernelVolume *= k;
        }
        else
        {
            // Default: estimate from input/output spatial ratio
            kernelVolume = 9; // 3×3 default assumption
        }

        // Each output element reads kernelVolume input elements (cache-resident, so cheaper than a stream)
        var computeTime = OpCostModel.Stream(inputShape.MemoryBytes + outputShape.MemoryBytes)
            + OpCostModel.StreamNsPerByte * 0.25 * outputShape.MemoryBytes * System.Math.Max(0, kernelVolume - 1);

        return new OpPerfResult
        {
            ComputeTime = computeTime,
            ExtraMemoryBytes = 0,
        };
    }

    private static OpPerfResult EstimateGlobalPool(OpPerfInput input)
    {
        var inputShape = input.InputShapes[0];
        if (inputShape is null)
            return OpPerfResult.Zero;

        // Global pool reduces all spatial dims — one pass over the input
        var costMultiplier = input.OpCode == GLOBAL_LP_POOL ? 2.0 : 1.0;
        var computeTime = OpCostModel.Stream(inputShape.MemoryBytes, costMultiplier);

        return new OpPerfResult
        {
            ComputeTime = computeTime,
            ExtraMemoryBytes = 0,
        };
    }

    private static OpPerfResult EstimateNorm(OpPerfInput input)
    {
        var inputShape = input.InputShapes[0];
        var outputShape = input.OutputShapes[0];
        if (inputShape is null || outputShape is null)
            return OpPerfResult.Zero;

        // Mean, variance and normalize: about three passes over the input plus the output write
        var computeTime = OpCostModel.Stream(3.0 * inputShape.MemoryBytes + outputShape.MemoryBytes);

        // BatchNorm can be in-place
        var canInPlace = !input.InputMustRemainIntact[0]
            && inputShape.ElementCount == outputShape.ElementCount
            && inputShape.DType == outputShape.DType;

        return new OpPerfResult
        {
            ComputeTime = computeTime,
            ExtraMemoryBytes = 0,
            InPlaceBufferReuse = canInPlace ? new Dictionary<int, int> { [0] = 0 } : new Dictionary<int, int>()
        };
    }

    private static long[]? GetLongsAttr(OpPerfInput input, string name)
    {
        if (input.Attributes.TryGetValue(name, out var val) && val is long[] arr)
            return arr;
        return null;
    }
}

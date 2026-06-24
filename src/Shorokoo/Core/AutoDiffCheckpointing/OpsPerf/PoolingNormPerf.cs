using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;

namespace Shorokoo.Core.AutoDiffCheckpointing.OpsPerf;

/// <summary>
/// Performance estimator for pooling and normalization operations.
/// Pooling ops reduce spatial dimensions by applying a function over a window.
/// Normalization ops compute statistics and normalize activations.
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

        // Each output element requires reading kernelVolume input elements
        var outputElements = outputShape.ElementCount;
        var computeTime = outputElements * kernelVolume / 256.0;

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

        // Global pool reduces all spatial dims — cost is proportional to input elements
        var costMultiplier = input.OpCode == GLOBAL_LP_POOL ? 2.0 : 1.0;
        var computeTime = inputShape.ElementCount / 256.0 * costMultiplier;

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

        var elements = inputShape.ElementCount;

        // Normalization typically requires:
        // Pass 1: Compute mean (sum + divide)
        // Pass 2: Compute variance (subtract mean, square, sum, divide)
        // Pass 3: Normalize (subtract mean, divide by std, scale, shift)
        // ~5 passes over data
        var computeTime = elements / 256.0 * 5.0;

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

using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;

namespace Shorokoo.Core.AutoDiffCheckpointing.OpsPerf;

/// <summary>
/// Performance estimator for miscellaneous operations that don't fit into other categories.
/// Priced as streaming passes over the output (<see cref="OpCostModel.Stream"/>) with a
/// multiplier for the work per element; Det is a batched O(n³) product.
/// </summary>
internal class MiscPerf : IOpPerf
{
    public IReadOnlySet<string> SupportedOpCodes { get; } = new HashSet<string>
    {
        RESIZE, UPSAMPLE, DET, TOPK, NON_MAX_SUPPRESSION,
        BERNOULLI, RANDOM_NORMAL, RANDOM_NORMAL_LIKE,
        RANDOM_UNIFORM, RANDOM_UNIFORM_LIKE, MULTINOMIAL,
        CONSTANT, CONSTANT_OF_SHAPE,
    };

    public OpPerfResult Estimate(OpPerfInput input)
    {
        var opCode = input.OpCode;

        // Constants have zero compute cost
        if (opCode is CONSTANT or CONSTANT_OF_SHAPE)
            return OpPerfResult.Zero;

        var outputShape = input.OutputShapes[0];
        if (outputShape is null)
            return OpPerfResult.Zero;

        var moved = OpCostModel.BytesOf(input.InputShapes) + outputShape.MemoryBytes;

        return opCode switch
        {
            RESIZE or UPSAMPLE => new OpPerfResult
            {
                ComputeTime = OpCostModel.Stream(moved, 3.0), // Interpolation
                ExtraMemoryBytes = 0,
            },
            DET => EstimateDet(input),
            TOPK => new OpPerfResult
            {
                ComputeTime = OpCostModel.Stream(moved, 10.0), // Partial sort
                ExtraMemoryBytes = 0,
            },
            NON_MAX_SUPPRESSION => new OpPerfResult
            {
                ComputeTime = OpCostModel.Stream(moved, 20.0), // Sorting + IoU computation
                ExtraMemoryBytes = 0,
            },
            // Random generators
            BERNOULLI or RANDOM_NORMAL or RANDOM_NORMAL_LIKE
                or RANDOM_UNIFORM or RANDOM_UNIFORM_LIKE or MULTINOMIAL => new OpPerfResult
            {
                ComputeTime = OpCostModel.Stream(moved, 3.0),
                ExtraMemoryBytes = 0,
            },
            _ => new OpPerfResult
            {
                ComputeTime = OpCostModel.Stream(moved),
                ExtraMemoryBytes = 0,
            }
        };
    }

    private static OpPerfResult EstimateDet(OpPerfInput input)
    {
        var inputShape = input.InputShapes[0];
        if (inputShape is null)
            return OpPerfResult.Zero;

        var dims = inputShape.Shape.Dims;
        if (dims.Length < 2)
            return OpPerfResult.Zero;

        long n = dims[^1]; // Matrix size
        long batch = 1;
        for (int i = 0; i < dims.Length - 2; i++)
            batch *= dims[i];

        // Determinant via LU decomposition: O(n³) per matrix
        var flops = batch * n * n * n * 2.0;
        return new OpPerfResult
        {
            ComputeTime = OpCostModel.MatMul(flops),
            ExtraMemoryBytes = batch * n * n * 4, // LU workspace
        };
    }
}

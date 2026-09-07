using Shorokoo.Core.AutoDiffCheckpointing;
using Shorokoo.Core.AutoDiffCheckpointing.OpsPerf;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;

namespace Shorokoo.Tests;

/// <summary>
/// Pins the orderings ORT's profiler established for the <c>OpsPerf</c> compute model
/// (<see cref="OpCostModel"/>): launch cost dominates tiny ops, streaming ops scale with bytes,
/// MatMul with FLOPs, and the axis/consumer/broadcast refinements point the right way. Not
/// the constants — those are recalibrated — only the relations the memory-aware pass trades on.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class OpsPerfCostModelTests
{
    private static readonly OpPerfRegistry Registry = new();

    private static TensorShapeInfo T(params long[] dims) => new(new Shape(dims), DType.Float32, null);
    private static TensorShapeInfo I(params long[] dims) => new(new Shape(dims), DType.Int64, null);

    private static double Cost(string op, TensorShapeInfo[] ins, TensorShapeInfo[] outs,
        Dictionary<string, object?>? attrs = null, string[]? consumers = null)
        => Registry.Estimate(new OpPerfInput
        {
            OpCode = op,
            InputShapes = ins,
            OutputShapes = outs,
            InputMustRemainIntact = ins.Select(_ => true).ToArray(),
            Attributes = attrs ?? new Dictionary<string, object?>(),
            ConsumerOpCodes = consumers,
        }).ComputeTime;

    private static double Add(long n, long m) => Cost(ADD, [T(n, m), T(n, m)], [T(n, m)]);
    private static double MatMul(long m, long k, long n) => Cost(MATMUL, [T(m, k), T(k, n)], [T(m, n)]);
    private static double Transpose(long n, long m, long[] perm) => Cost(TRANSPOSE, [T(n, m)], [T(m, n)], new() { ["perm"] = perm });
    private static double Conv(long[] x, long[] w, long[] y) => Cost(CONV, [T(x), T(w), T(w[0])], [T(y)], new() { ["group"] = 1L, ["kernel_shape"] = (long[])[w[2], w[3]] });

    [Fact]
    public void TestLaunchFloorAndScaling()
    {
        Assert.True(Cost(ADD, [T(1), T(1)], [T(1)]) >= OpCostModel.Launch);
        Assert.True(Cost(NEG, [T(1)], [T(1)]) >= OpCostModel.Launch);
        Assert.True(MatMul(16, 16, 16) >= OpCostModel.Launch);
        Assert.True(Add(512, 512) > MatMul(16, 16, 16));
        Assert.True(MatMul(1024, 1024, 1024) > Add(1024, 1024));
        Assert.True(Add(512, 512) > Add(16, 16));
        Assert.InRange(MatMul(1024, 1024, 1024) / MatMul(512, 512, 512), 4.0, 8.0);
        Assert.InRange(Transpose(2048, 2048, [1, 0]) / Transpose(1024, 1024, [1, 0]), 3.0, 4.5);
        Assert.InRange(Add(2048, 2048) / Add(1024, 1024), 3.0, 4.5);
    }

    [Fact]
    public void TestRefinementsPointTheRightWay()
    {
        Assert.True(Cost(TRANSPOSE, [T(4, 256, 64)], [T(4, 64, 256)], new() { ["perm"] = (long[])[0, 2, 1] })
            > Cost(TRANSPOSE, [T(4, 256, 64)], [T(256, 4, 64)], new() { ["perm"] = (long[])[1, 0, 2] }));
        Assert.Equal(0, Cost(TRANSPOSE, [T(512, 256)], [T(256, 512)], new() { ["perm"] = (long[])[1, 0] }, [MATMUL]));
        Assert.True(Cost(TRANSPOSE, [T(512, 256)], [T(256, 512)], new() { ["perm"] = (long[])[1, 0] }, [MATMUL, ADD]) > 0);
        Assert.True(Cost(REDUCE_SUM, [T(512, 512), I(1)], [T(1, 512)]) > Cost(REDUCE_SUM, [T(512, 512), I(1)], [T(512, 1)]));
        Assert.True(Cost(WHERE, [T(512, 512), T(512, 512), T(512, 512)], [T(512, 512)]) > Add(512, 512));
        Assert.True(Cost(ADD, [T(512, 512), T(512)], [T(512, 512)]) > Cost(ADD, [T(512, 512), T()], [T(512, 512)]));
        Assert.True(Cost(SOFTMAX, [T(8, 4, 128, 128)], [T(8, 4, 128, 128)], new() { ["axis"] = -1L }) > Cost(EXP, [T(8, 4, 128, 128)], [T(8, 4, 128, 128)]));
        Assert.True(Cost(GELU, [T(1024, 1024)], [T(1024, 1024)]) > Cost(RELU, [T(1024, 1024)], [T(1024, 1024)]));
        Assert.True(Cost(CONCAT, [T(256, 512), T(256, 512)], [T(512, 512)]) > Add(512, 512));
        Assert.True(Conv([32, 8, 64, 64], [32, 8, 64, 64], [32, 32, 3, 3]) > Conv([8, 32, 64, 64], [32, 32, 3, 3], [8, 32, 64, 64]));
    }

    [Fact]
    public void TestFreeAndFoldedOps()
    {
        Assert.Equal(0, Cost(IDENTITY, [T(1024, 1024)], [T(1024, 1024)]));
        Assert.Equal(0, Cost(SHAPE, [T(1024, 1024)], [I(2)]));
        Assert.Equal(0, Cost(CONSTANT, [], [T(1024)]));
        Assert.Equal(0, Cost(ADD, [I(2), I(2)], [I(2)]));
        Assert.True(Cost(RESHAPE, [T(1024, 1024), I(2)], [T(1048576)]) < OpCostModel.Launch);
        Assert.True(Cost(RESHAPE, [T(1024, 1024), I(2)], [T(1048576)]) < Add(1024, 1024));
    }
}

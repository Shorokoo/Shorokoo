using System;
using System.Linq;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Rng;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>Emits the in-graph runtime uniform draw at the input's shape (fixed key/drawBase).</summary>
[Module]
public partial class RtUniformDraw
{
    public static Tensor<float32> Inline(Tensor<float32> x)
        => RuntimeRng.StandardUniform(x.ShapeTensor(), Scalar(123L), Scalar(456L), Scalar(0L));
}

/// <summary>Emits the in-graph runtime normal draw at the input's shape (fixed key/drawBase).</summary>
[Module]
public partial class RtNormalDraw
{
    public static Tensor<float32> Inline(Tensor<float32> x)
        => RuntimeRng.StandardNormal(x.ShapeTensor(), Scalar(7L), Scalar(9L), Scalar(0L));
}

/// <summary>Emits a plain <c>Globals.RandomUniform</c> draw — routed through the SHRK_RANDOM
/// lowering (<c>FastLowerRandomOps</c>), i.e. the in-graph counter-based path, not ONNX's
/// RandomUniformLike.</summary>
[Module]
public partial class RtLoweredUniform
{
    public static Tensor<float32> Inline(Tensor<float32> x)
        => RandomUniform(x.ShapeTensor(), 0f, 1f);
}

/// <summary>
/// Coverage for the in-graph counter-based runtime RNG (<see cref="RuntimeRng"/>): the ONNX-op
/// Threefry subgraph must reproduce the host generator (<see cref="Threefry2x32"/>) bit-for-bit
/// — proving the novel integer-op PRNG is correct and execution-provider-independent — and
/// produce well-distributed draws.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngRuntimeTests
{
    private static float[] RunDraw<TModule>(long rows, long cols)
    {
        var g = (FastComputationGraph)typeof(TModule)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var input = TensorData([rows, cols], Enumerable.Repeat(0f, (int)(rows * cols)).ToArray());
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([input])).ToConcreteModel();
        var outputs = ComputeContext.Default.Execute(concrete, input);
        return outputs[0].ToTensorData().As<float32>().AccessMemory().ToArray();
    }

    // Host reference for the runtime scheme: element i -> counter (i, drawBase);
    // uniform = low 24 bits of x0 * 2^-24.
    private static float HostUniform(long i, uint k0, uint k1, uint drawBase)
    {
        var (x0, _) = Threefry2x32.Bijection((uint)i, drawBase, k0, k1);
        return (x0 & 0x00FFFFFFu) * (1.0f / 16777216.0f);
    }

    [Fact]
    public void TestInGraphUniformMatchesHostBitExact()
    {
        var vals = RunDraw<RtUniformDraw>(4, 4);
        Assert.Equal(16, vals.Length);
        for (long i = 0; i < 16; i++)
            Assert.Equal(HostUniform(i, 123, 456, 0), vals[i]);
    }

    [Fact]
    public void TestInGraphUniformIsInRangeAndSpread()
    {
        var vals = RunDraw<RtUniformDraw>(8, 8);
        Assert.All(vals, v => Assert.InRange(v, 0.0f, 0.99999997f));
        Assert.InRange(vals.Average(), 0.4f, 0.6f);
    }

    [Fact]
    public void TestLoweredRandomUniformIsDeterministicAndInRange()
    {
        // A plain Globals.RandomUniform draw now lowers to the in-graph counter-based RNG, so
        // it is bit-reproducible across executions (the old ONNX RandomUniformLike advanced its
        // own state per Run and would differ). Two runs must be identical, in range, and spread.
        var a = RunDraw<RtLoweredUniform>(8, 8);
        var b = RunDraw<RtLoweredUniform>(8, 8);
        Assert.Equal(64, a.Length);
        Assert.Equal(a, b);                                        // deterministic / portable
        Assert.All(a, v => Assert.InRange(v, 0.0f, 0.99999997f));
        Assert.InRange(a.Average(), 0.3f, 0.7f);   // 64-sample mean; loose (the point is determinism + range)
    }

    [Fact]
    public void TestNoConfigModelDrawsKeyedThreefryUnderTheDefaultIdentity()
    {
        // "No config" means the DEFAULT deterministic identity (master seed 0), never the
        // ONNX random fallback: a concrete model built without any RngConfig carries the
        // default identity, and its feed draws are bit-exactly the host fold of the
        // default runtime master along the feed's ModelId — reconstructible offline.
        var g = (FastComputationGraph)typeof(RtLoweredUniform)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var input = TensorData([4L, 4L], Enumerable.Repeat(0f, 16).ToArray());
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([input])).ToConcreteModel();

        Assert.NotNull(concrete.TryGetRngKeyVector());   // the default identity, recorded

        var vals = ComputeContext.Default.Execute(concrete, input)[0]
            .ToTensorData().As<float32>().AccessMemory().ToArray();
        var (k0, k1) = RngConfig.Default.FoldRunKey([1]);   // the feed's site is slot 1
        for (long i = 0; i < 16; i++)
            Assert.Equal(HostUniform(i, k0, k1, 0), vals[i]);
    }

    [Fact]
    public void TestRngConfigRebindsInPlaceWithoutGraphChange()
    {
        // Re-binding is re-initialization scoped to keys: it replaces the identity carrier
        // and re-runs the key initializers (each SHRK_RNG_KEY_PARAM entity's value re-materializes
        // in place — key initializers are pure in the identity, so this is always safe).
        // No node is added or removed and no feed is touched; parameter values would be
        // untouched too (this model has none to re-key).
        var g = (FastComputationGraph)typeof(RtLoweredUniform)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var input = TensorData([4L, 4L], Enumerable.Repeat(0f, 16).ToArray());
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([input]))
            .ToConcreteModel(new RngConfig { MasterSeed = 1 });

        float[] Run() => ComputeContext.Default.Execute(concrete, input)[0]
            .ToTensorData().As<float32>().AccessMemory().ToArray();

        int nodeCount = concrete.Nodes.Count;
        Assert.Contains(concrete.Nodes, n => n.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM);
        var underSeed1 = Run();

        concrete.ApplyRngConfig(new RngConfig { MasterSeed = 2 });
        Assert.Equal(nodeCount, concrete.Nodes.Count);   // re-binding replaces one node
        Assert.Contains(concrete.Nodes, n => n.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM);
        var underSeed2 = Run();
        Assert.NotEqual(underSeed1, underSeed2);         // new master -> new stream

        concrete.ApplyRngConfig(new RngConfig { MasterSeed = 1 });
        var underSeed1Again = Run();
        Assert.Equal(underSeed1, underSeed1Again);       // re-binding is exact, not approximate
    }

    [Fact]
    public void TestSplitDerivesChildKeyFromParentKeyInput()
    {
        // SHRK_RNG_SPLIT folds its parent key input with the index — bit-exact with the host
        // bijection. The split function is the versioned in-graph form of the key tree's
        // derivation primitive (the lowering itself derives keys host-side from the carrier).
        var parentKey = Vector(1L, 2L);
        var split = Shorokoo.Core.Nodes.NodeDefinitions.InternalOp.RngSplit(
            parentKey, Scalar(7L), Shorokoo.Core.Rng.RngAlgorithms.Default);
        var g = new FastComputationGraph([], [split]);

        var childWords = ComputeContext.Default.Execute(g)[0]
            .ToTensorData().As<int64>().AccessMemory().ToArray();
        var (x0, x1) = Threefry2x32.Bijection(7u, 0u, 1u, 2u);
        Assert.Equal((long[])[x0, x1], childWords);
    }

    [Fact]
    public void TestInGraphNormalHasStandardMoments()
    {
        var vals = RunDraw<RtNormalDraw>(40, 40);
        double mean = vals.Average();
        double variance = vals.Select(v => (v - mean) * (v - mean)).Average();
        Assert.InRange(mean, -0.1, 0.1);
        Assert.InRange(variance, 0.85, 1.15);
    }
}

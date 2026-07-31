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

/// <summary>A trainable weight plus a runtime RNG feed: the feed forces the framework to inject
/// the <c>RngExecutionCounter</c> as model state, while the draw is zeroed so the model's output
/// is exactly the linear transform. Used to exercise safetensors export/import of a model that
/// carries the execution counter.</summary>
[Module]
public partial class RtFcWithRngFeed
{
    public static Tensor<float32> Inline(Tensor<float32> input, [Hyper] Scalar<int64> numOutFeatures)
    {
        var numInFeatures = input.ShapeTensor()[-1L];
        var weights = Shorokoo.Tests.Modules.InitSimple.Init([numOutFeatures, numInFeatures]);
        var y = input.MatMul(weights.Transpose(1, 0));
        return y + RandomUniform(y.ShapeTensor(), 0f, 1f) * Scalar(0f);
    }
}

/// <summary>Emits the in-graph raw-bits draws (U8/U16/U32/U64) at the input's shape.</summary>
[Module] public partial class RtBitsU8Draw  { public static Tensor<uint8>  Inline(Tensor<float32> x) => RuntimeRng.BitsU8 (x.ShapeTensor(), Scalar(111L), Scalar(222L), Scalar(0L)); }
[Module] public partial class RtBitsU16Draw { public static Tensor<uint16> Inline(Tensor<float32> x) => RuntimeRng.BitsU16(x.ShapeTensor(), Scalar(111L), Scalar(222L), Scalar(0L)); }
[Module] public partial class RtBitsU32Draw { public static Tensor<uint32> Inline(Tensor<float32> x) => RuntimeRng.BitsU32(x.ShapeTensor(), Scalar(111L), Scalar(222L), Scalar(0L)); }
[Module] public partial class RtBitsU64Draw { public static Tensor<uint64> Inline(Tensor<float32> x) => RuntimeRng.BitsU64(x.ShapeTensor(), Scalar(111L), Scalar(222L), Scalar(0L)); }

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
        var g = ((ComputationGraph)typeof(TModule)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
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

    // Host reference for the raw-bits scheme: element i draws one generator word pair; the
    // narrow widths take the low bits of x0, U32 the whole word, U64 = x0 | (x1 << 32).
    private static ulong HostBits(long i, int width, uint k0, uint k1, uint drawBase)
    {
        var (x0, x1) = Threefry2x32.Bijection((uint)i, drawBase, k0, k1);
        return width switch
        {
            8 => (byte)x0,
            16 => (ushort)x0,
            32 => x0,
            64 => x0 | ((ulong)x1 << 32),
            _ => throw new ArgumentOutOfRangeException(nameof(width)),
        };
    }

    private static TensorData RunDrawRaw<TModule>(long rows, long cols)
    {
        var g = ((ComputationGraph)typeof(TModule)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var input = TensorData([rows, cols], Enumerable.Repeat(0f, (int)(rows * cols)).ToArray());
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([input])).ToConcreteModel();
        return ComputeContext.Default.Execute(concrete, input)[0].ToTensorData();
    }

    [Fact]
    public void TestInGraphBitsMatchHostBitExact()
    {
        var u8 = RunDrawRaw<RtBitsU8Draw>(4, 4);
        Assert.Equal(DType.UInt8, u8.DType);
        var u8v = u8.As<uint8>().AccessMemory().ToArray();
        for (long i = 0; i < 16; i++) Assert.Equal((byte)HostBits(i, 8, 111, 222, 0), u8v[i]);

        var u16 = RunDrawRaw<RtBitsU16Draw>(4, 4);
        Assert.Equal(DType.UInt16, u16.DType);
        var u16v = u16.As<uint16>().AccessMemory().ToArray();
        for (long i = 0; i < 16; i++) Assert.Equal((ushort)HostBits(i, 16, 111, 222, 0), u16v[i]);

        var u32 = RunDrawRaw<RtBitsU32Draw>(4, 4);
        Assert.Equal(DType.UInt32, u32.DType);
        var u32v = u32.As<uint32>().AccessMemory().ToArray();
        for (long i = 0; i < 16; i++) Assert.Equal((uint)HostBits(i, 32, 111, 222, 0), u32v[i]);

        var u64 = RunDrawRaw<RtBitsU64Draw>(4, 4);
        Assert.Equal(DType.UInt64, u64.DType);
        var u64v = u64.As<uint64>().AccessMemory().ToArray();
        for (long i = 0; i < 16; i++) Assert.Equal(HostBits(i, 64, 111, 222, 0), u64v[i]);
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
        var g = ((ComputationGraph)typeof(RtLoweredUniform)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var input = TensorData([4L, 4L], Enumerable.Repeat(0f, 16).ToArray());
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([input])).ToConcreteModel();

        Assert.NotNull(concrete.TryGetRngSeed());   // the default identity, recorded

        var vals = ComputeContext.Default.Execute(concrete, input)[0]
            .ToTensorData().As<float32>().AccessMemory().ToArray();
        var (k0, k1) = RngConfig.Default.FoldRunKey([1]);   // the feed's site is slot 1
        for (long i = 0; i < 16; i++)
            Assert.Equal(HostUniform(i, k0, k1, 0), vals[i]);
    }

    [Fact]
    public void TestRngConfigRebindsInPlaceWithoutGraphChange()
    {
        // Re-binding is the RngSeed parameter's re-initialization: it replaces that one
        // parameter's value, and every draw's key — a split chain rooted at the parameter —
        // re-derives from it. No node is added or removed and no feed is touched; parameter
        // values would be untouched too (this model has none to re-key).
        var g = ((ComputationGraph)typeof(RtLoweredUniform)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
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
        var g = new InternalComputationGraph([], [split]);

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

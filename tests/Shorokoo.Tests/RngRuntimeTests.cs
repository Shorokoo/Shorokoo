using System;
using System.Linq;
using Shorokoo.Core.Rng;
using Shorokoo.Runtime;
using static Shorokoo.Tests.RngDrawRunners;

namespace Shorokoo.Tests;

/// <summary>Emits the in-graph runtime uniform draw at the input's shape (fixed key/substreamIndex).</summary>
[Module]
public partial class RtUniformDraw
{
    public static Tensor<float32> Inline(Tensor<float32> x)
        => RuntimeRng.StandardUniform(x.ShapeTensor(), Scalar(123UL | (456UL << 32)), Scalar(0UL));
}

/// <summary>Emits the in-graph runtime normal draw at the input's shape (fixed key/substreamIndex).</summary>
[Module]
public partial class RtNormalDraw
{
    public static Tensor<float32> Inline(Tensor<float32> x)
        => RuntimeRng.StandardNormal(x.ShapeTensor(), Scalar(7UL | (9UL << 32)), Scalar(0UL));
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
/// is exactly the linear transform.</summary>
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
[Module] public partial class RtBitsU8Draw  { public static Tensor<uint8>  Inline(Tensor<float32> x) => RuntimeRng.BitsU8 (x.ShapeTensor(), Scalar(111UL | (222UL << 32)), Scalar(0UL)); }
[Module] public partial class RtBitsU16Draw { public static Tensor<uint16> Inline(Tensor<float32> x) => RuntimeRng.BitsU16(x.ShapeTensor(), Scalar(111UL | (222UL << 32)), Scalar(0UL)); }
[Module] public partial class RtBitsU32Draw { public static Tensor<uint32> Inline(Tensor<float32> x) => RuntimeRng.BitsU32(x.ShapeTensor(), Scalar(111UL | (222UL << 32)), Scalar(0UL)); }
[Module] public partial class RtBitsU64Draw { public static Tensor<uint64> Inline(Tensor<float32> x) => RuntimeRng.BitsU64(x.ShapeTensor(), Scalar(111UL | (222UL << 32)), Scalar(0UL)); }

/// <summary>A plain <c>Globals.RandomBits</c> feed — routed through the SHRK_RANDOM_BITS
/// lowering (id-bearing keyed draw), i.e. the public runtime raw-bits path.</summary>
[Module] public partial class RtLoweredBits   { public static Tensor<uint32> Inline(Tensor<float32> x) => RandomBits<uint32>(x.ShapeTensor()); }
[Module] public partial class RtLoweredBits64 { public static Tensor<uint64> Inline(Tensor<float32> x) => RandomBits<uint64>(x.ShapeTensor()); }

/// <summary>
/// The in-graph counter-based runtime RNG (<see cref="RuntimeRng"/>): the ONNX-op Threefry
/// subgraph must reproduce the host generator (<see cref="Threefry2x32"/>) bit-for-bit —
/// execution-provider-independent — and produce well-distributed draws.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngRuntimeTests
{
    private const ulong BitsKey = 111UL | (222UL << 32);
    private const ulong UniformKey = 123UL | (456UL << 32);

    // Host reference for the runtime scheme: substreamIndex folds into the key, element i
    // indexes the whole counter; uniform = low 24 bits of x0 * 2^-24.
    private static float HostUniform(long i, ulong key, ulong substreamIndex)
        => RngTestOracle.DrawUniform(key, substreamIndex, i);

    // Host reference for the raw-bits scheme: the narrow widths take the low bits of x0, U32
    // the whole word, U64 = x0 | (x1 << 32).
    private static ulong HostBits(long i, int width, ulong key, ulong substreamIndex)
        => RngTestOracle.DrawBits(key, substreamIndex, i, width);

    [Fact]
    public void TestInGraphBitsAndUniformDrawsMatchTheHostGeneratorBitExactly()
    {
        var u8 = RunDrawRaw<RtBitsU8Draw>(4, 4);
        Assert.Equal(DType.UInt8, u8.DType);
        var u8v = u8.As<uint8>().AccessMemory().ToArray();
        for (long i = 0; i < 16; i++) Assert.Equal((byte)HostBits(i, 8, BitsKey, 0), u8v[i]);

        var u16 = RunDrawRaw<RtBitsU16Draw>(4, 4);
        Assert.Equal(DType.UInt16, u16.DType);
        var u16v = u16.As<uint16>().AccessMemory().ToArray();
        for (long i = 0; i < 16; i++) Assert.Equal((ushort)HostBits(i, 16, BitsKey, 0), u16v[i]);

        var u32 = RunDrawRaw<RtBitsU32Draw>(4, 4);
        Assert.Equal(DType.UInt32, u32.DType);
        var u32v = u32.As<uint32>().AccessMemory().ToArray();
        for (long i = 0; i < 16; i++) Assert.Equal((uint)HostBits(i, 32, BitsKey, 0), u32v[i]);

        var u64 = RunDrawRaw<RtBitsU64Draw>(4, 4);
        Assert.Equal(DType.UInt64, u64.DType);
        var u64v = u64.As<uint64>().AccessMemory().ToArray();
        for (long i = 0; i < 16; i++) Assert.Equal(HostBits(i, 64, BitsKey, 0), u64v[i]);

        var vals = RunDraw<RtUniformDraw>(4, 4);
        Assert.Equal(16, vals.Length);
        for (long i = 0; i < 16; i++) Assert.Equal(HostUniform(i, UniformKey, 0), vals[i]);
    }

    [Fact]
    public void TestNarrowBitsDrawsPackLanesLowFirstIntoTheGeneratorWordAndSliceTheTail()
    {
        var words = RunDrawRaw<RtBitsU32Draw>(4, 4).As<uint32>().AccessMemory().ToArray();
        var u8 = RunDrawRaw<RtBitsU8Draw>(4, 4).As<uint8>().AccessMemory().ToArray();
        var u16 = RunDrawRaw<RtBitsU16Draw>(4, 4).As<uint16>().AccessMemory().ToArray();

        for (int j = 0; j < 4; j++)
            Assert.Equal(words[j], u8[4 * j] | ((uint)u8[4 * j + 1] << 8)
                                             | ((uint)u8[4 * j + 2] << 16) | ((uint)u8[4 * j + 3] << 24));
        for (int j = 0; j < 8; j++)
            Assert.Equal(words[j], u16[2 * j] | ((uint)u16[2 * j + 1] << 16));

        Assert.Equal(u8.Take(5).ToArray(), RunDrawRaw<RtBitsU8Draw>(1, 5).As<uint8>().AccessMemory().ToArray());
        Assert.Equal(u16.Take(5).ToArray(), RunDrawRaw<RtBitsU16Draw>(1, 5).As<uint16>().AccessMemory().ToArray());
    }

    [Fact]
    public void TestInGraphUniformIsInRangeAndSpreadAndNormalHasStandardMoments()
    {
        var uniform = RunDraw<RtUniformDraw>(8, 8);
        Assert.All(uniform, v => Assert.InRange(v, 0.0f, 0.99999997f));
        Assert.InRange(uniform.Average(), 0.4f, 0.6f);

        var normal = RunDraw<RtNormalDraw>(40, 40);
        double mean = normal.Average();
        double variance = normal.Select(v => (v - mean) * (v - mean)).Average();
        Assert.InRange(mean, -0.1, 0.1);
        Assert.InRange(variance, 0.85, 1.15);
    }

    [Fact]
    public void TestLoweredFeedsAreDeterministicAndKeyedUnderTheDefaultIdentity()
    {
        // A plain Globals.RandomUniform draw lowers to the in-graph counter-based RNG, so it is
        // bit-reproducible across executions (ONNX RandomUniformLike advanced state per Run).
        var a = RunDraw<RtLoweredUniform>(8, 8);
        Assert.Equal(64, a.Length);
        Assert.Equal(a, RunDraw<RtLoweredUniform>(8, 8));
        Assert.All(a, v => Assert.InRange(v, 0.0f, 0.99999997f));
        Assert.InRange(a.Average(), 0.3f, 0.7f);

        // "No config" means the DEFAULT deterministic identity (master seed 0), never the ONNX
        // random fallback: the draws are bit-exactly the host fold of the default runtime
        // master along the feed's ModelId (slot 1) — reconstructible offline.
        var g = ((ComputationGraph)typeof(RtLoweredUniform)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var input = TensorData([4L, 4L], Enumerable.Repeat(0f, 16).ToArray());
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([input])).ToConcreteModel();
        Assert.NotNull(concrete.TryGetRngSeed());
        var defaultKey = RngTestOracle.RunKey(RngConfig.Default, [1]);
        var vals = ComputeContext.Default.Execute(concrete, input)[0]
            .ToTensorData().As<float32>().AccessMemory().ToArray();
        for (long i = 0; i < 16; i++) Assert.Equal(HostUniform(i, defaultKey, 0), vals[i]);

        var bits = RunDrawRaw<RtLoweredBits>(4, 4);
        Assert.Equal(DType.UInt32, bits.DType);
        var bv = bits.As<uint32>().AccessMemory().ToArray();
        Assert.Equal(bv, RunDrawRaw<RtLoweredBits>(4, 4).As<uint32>().AccessMemory().ToArray());
        for (long i = 0; i < 16; i++) Assert.Equal((uint)HostBits(i, 32, defaultKey, 0), bv[i]);

        // The U64 path (unsigned BitShift + BitwiseOr above the int64 range) must survive the
        // full public feed -> keyed draw -> width-specialized function call.
        var bits64 = RunDrawRaw<RtLoweredBits64>(4, 4);
        Assert.Equal(DType.UInt64, bits64.DType);
        var b64 = bits64.As<uint64>().AccessMemory().ToArray();
        for (long i = 0; i < 16; i++) Assert.Equal(HostBits(i, 64, defaultKey, 0), b64[i]);
    }

    [Fact]
    public void TestBitsFeedIsLabelledAsSuchAndRebindingReplacesOnlyTheIdentityValue()
    {
        // A bits feed must be classified and described as a "bits feed", not silently as the
        // "normal feed" default the RngStreamKind switches fall through to.
        var bg = ((ComputationGraph)typeof(RtLoweredBits)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var bitsInput = TensorData([4L, 4L], Enumerable.Repeat(0f, 16).ToArray());
        var report = bg.ToConcreteArchitecture(bg.FromOrderedInputs([bitsInput])).GetRngStreamReport();
        var bitsStreams = report.Streams.Where(s => s.Kind == RngStreamKind.BitsFeed).ToList();
        Assert.NotEmpty(bitsStreams);
        Assert.Contains("bits feed", report.ToString());
        Assert.Contains("bits feed", report.EmitPinSkeleton());
        Assert.DoesNotContain("normal feed", report.ToString());

        // Re-binding is the RngSeed parameter's re-initialization: every draw's key — a split
        // chain rooted at that parameter — re-derives from it, with no node added or removed.
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
        Assert.Equal(nodeCount, concrete.Nodes.Count);
        Assert.Contains(concrete.Nodes, n => n.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM);
        Assert.NotEqual(underSeed1, Run());

        concrete.ApplyRngConfig(new RngConfig { MasterSeed = 1 });
        Assert.Equal(underSeed1, Run());   // re-binding is exact, not approximate
    }

    [Fact]
    public void TestSplitIndexAndDrawPositionUseTheWholeSixtyFourBitRange()
    {
        static ulong Split(ulong k, ulong index)
        {
            var node = InternalOp.RngSplit(Scalar(k), Scalar(index), RngAlgorithms.Default);
            return ComputeContext.Default.Execute(new InternalComputationGraph([], [node]))[0]
                .ToTensorData().As<uint64>().AccessMemory().ToArray()[0];
        }

        // SHRK_RNG_SPLIT folds its parent key input with the index, bit-exact with the host
        // bijection (the split function is the versioned in-graph derivation primitive).
        var (px0, px1) = Threefry2x32.Bijection(7u, 0u, 1u, 2u);
        Assert.Equal(px0 | ((ulong)px1 << 32), Split(1UL | (2UL << 32), 7UL));

        // Distinct indices give distinct children over the ENTIRE range. The first two pairs
        // ALIAS under a 32-bit index (same low word, different high word); the third is in the
        // top half, where a signed reading would go negative. key's high bit is set too.
        const ulong key = 0x8000_0000_0000_0001UL;
        (ulong a, ulong b)[] pairs =
        [
            (7UL, 7UL + (1UL << 32)),
            (0UL, 1UL << 32),
            (0xFFFF_FFFF_FFFF_FFFEUL, 0xFFFF_FFFEUL),
        ];
        foreach (var (ia, ib) in pairs)
        {
            var a = Split(key, ia);
            var b = Split(key, ib);
            Assert.Equal(RngTestOracle.FoldKey(key, ia), a);
            Assert.Equal(RngTestOracle.FoldKey(key, ib), b);
            Assert.NotEqual(a, b);
        }

        // substreamIndex is the execution counter; under a 32-bit counter word, execution 2^32
        // repeated execution 0's draw exactly.
        const ulong drawKey = 0xDEAD_BEEF_FEED_FACEUL;
        static float[] Draw(ulong substreamIndex)
        {
            var g = new InternalComputationGraph([],
                [RuntimeRng.StandardUniform(Vector(4L), Scalar(drawKey), Scalar(substreamIndex))]);
            return ComputeContext.Default.Execute(g)[0]
                .ToTensorData().As<float32>().AccessMemory().ToArray();
        }

        var atZero = Draw(0);
        var atTwoPow32 = Draw(1UL << 32);
        var atTop = Draw(0xFFFF_FFFF_FFFF_FFFFUL);
        Assert.NotEqual(atZero, atTwoPow32);
        Assert.NotEqual(atZero, atTop);
        for (long i = 0; i < 4; i++)
        {
            Assert.Equal(RngTestOracle.DrawUniform(drawKey, 0, i), atZero[i]);
            Assert.Equal(RngTestOracle.DrawUniform(drawKey, 1UL << 32, i), atTwoPow32[i]);
            Assert.Equal(RngTestOracle.DrawUniform(drawKey, 0xFFFF_FFFF_FFFF_FFFFUL, i), atTop[i]);
        }
    }
}

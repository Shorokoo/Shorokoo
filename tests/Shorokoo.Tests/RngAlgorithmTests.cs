using System;
using System.Linq;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Rng;
using Shorokoo.Modules.Layers;
using Shorokoo.Runtime;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Tests.RngDrawRunners;

namespace Shorokoo.Tests;

internal static class RngDrawRunners
{
    internal static TensorData RunDrawRaw<TModule>(long rows, long cols)
    {
        var g = ((ComputationGraph)typeof(TModule)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var input = TensorData([rows, cols], Enumerable.Repeat(0f, (int)(rows * cols)).ToArray());
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([input])).ToConcreteModel();
        return ComputeContext.Default.Execute(concrete, input)[0].ToTensorData();
    }

    internal static float[] RunDraw<TModule>(long rows, long cols)
        => RunDrawRaw<TModule>(rows, cols).As<float32>().AccessMemory().ToArray();
}

/// <summary>Keyed uniform draw at the input's shape under a literal key, substreamIndex 0.</summary>
[Module]
public partial class RngKeyedUniformDraw
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var key = Scalar(123UL | (456UL << 32));
        return (Tensor<float32>)InternalOp.RngUniform(
            key, Scalar(0UL), x.ShapeTensor(), Scalar(0f), Scalar(1f), RngAlgorithms.Default);
    }
}

/// <summary>Splits a literal key at index 5, then draws uniform under the child key.</summary>
[Module]
public partial class RngSplitThenDraw
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var parent = Scalar(7UL | (9UL << 32));
        var child = InternalOp.RngSplit(parent, Scalar(5UL), RngAlgorithms.Default);
        return (Tensor<float32>)InternalOp.RngUniform(
            child, Scalar(0UL), x.ShapeTensor(), Scalar(0f), Scalar(1f), RngAlgorithms.Default);
    }
}

/// <summary>Keyed normal draw at the input's shape under a literal key.</summary>
[Module]
public partial class RngKeyedNormalDraw
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var key = Scalar(11UL | (13UL << 32));
        return (Tensor<float32>)InternalOp.RngNormal(
            key, Scalar(0UL), x.ShapeTensor(), Scalar(0f), Scalar(1f), RngAlgorithms.Default);
    }
}

/// <summary>Keyed raw-bits draw (U32) at the input's shape under a literal key, substreamIndex 0.</summary>
[Module]
public partial class RngKeyedBitsDraw
{
    public static Tensor<uint32> Inline(Tensor<float32> x)
    {
        var key = Scalar(123UL | (456UL << 32));
        return (Tensor<uint32>)InternalOp.RngBits(
            key, Scalar(0UL), x.ShapeTensor(), DType.UInt32, RngAlgorithms.Default);
    }
}

/// <summary>A single Linear whose weight is drawn by a random initializer — so its init values
/// change when the RNG algorithm changes.</summary>
[Module]
public partial class SwitchInitLinear
{
    public static Tensor<float32> Inline(Tensor<float32> x)
        => Linear.Model(Scalar(4L), Scalar(false)).Call(x);
}

/// <summary>
/// The named-algorithm keyed RNG operators (SHRK_RNG_SPLIT / UNIFORM / NORMAL / BITS) and their
/// ONNX lowering: each op lowers at export to a call of the algorithm's <b>non-inlined</b>
/// function (an ONNX local FunctionProto tagged with RngAlgorithm / RngFunctionKind metadata),
/// and the executed values reproduce the host Threefry generator bit-for-bit — including
/// through an index-based key split.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngAlgorithmTests
{
    // Host reference: substreamIndex folds into the key, element i indexes the whole counter.
    // A plain uniform draw is the dense arbitrary-range draw over [0,1), so the dense oracle is
    // the reference — an independent host rebuild, not a call back into the graph.
    private static float HostUniform(long i, ulong key)
        => RngDenseUniformOracle.Draw(key, 0, i, 0f, 1f);

    [Fact]
    public void TestDenseNormalOracleDecodesEveryGoldenDrawExactly()
    {
        foreach ((ulong draw, uint bits) in RngDenseNormalGolden.Pairs)
            Assert.Equal(bits, (uint)RngDenseNormalOracle.SampleBits(draw));
    }

    [Fact]
    public void TestDenseNormalOracleReachesBothSignedZeros()
    {
        Assert.Equal(0x0000_0000U, (uint)RngDenseNormalOracle.SampleBits(0UL));
        Assert.Equal(0x8000_0000U, (uint)RngDenseNormalOracle.SampleBits(1UL << 63));
    }

    [Fact]
    public void TestDenseNormalOracleIsMonotoneMirroredAndFinite()
    {
        ulong mask = (1UL << 63) - 1;
        uint previous = 0;
        foreach ((ulong draw, uint _) in RngDenseNormalGolden.Pairs)
        {
            ulong code = draw & mask;
            uint magnitude = (uint)RngDenseNormalOracle.MagnitudeBits(code);
            Assert.True((magnitude >> 23) < 0xFF);
            Assert.Equal(magnitude | 0x8000_0000U, (uint)RngDenseNormalOracle.SampleBits(draw | (1UL << 63)));
            Assert.Equal(magnitude, (uint)RngDenseNormalOracle.SampleBits(code));
        }
        foreach (ulong code in new ulong[] { 0, 1, 1UL << 20, 1UL << 40, 1UL << 62, mask })
        {
            uint magnitude = (uint)RngDenseNormalOracle.MagnitudeBits(code);
            Assert.True(magnitude >= previous);
            previous = magnitude;
        }
    }

    [Fact]
    public void TestKeyedUniformBitsAndSplitThenDrawMatchTheHostGeneratorBitExactly()
    {
        const ulong key = 123UL | (456UL << 32);

        var uniform = RunDraw<RngKeyedUniformDraw>(4, 4);
        Assert.Equal(16, uniform.Length);
        for (long i = 0; i < 16; i++) Assert.Equal(HostUniform(i, key), uniform[i]);

        var bits = RunDrawRaw<RngKeyedBitsDraw>(4, 4).As<uint32>().AccessMemory().ToArray();
        Assert.Equal(16, bits.Length);
        for (long i = 0; i < 16; i++) Assert.Equal((uint)RngTestOracle.DrawBits(key, 0, i, 32), bits[i]);

        // Child key = Bijection(counter: 5, key) — the split — then the draw under the child.
        var (ck0, ck1) = Threefry2x32.Bijection(5u, 0u, 7u, 9u);
        var childKey = ck0 | ((ulong)ck1 << 32);
        var split = RunDraw<RngSplitThenDraw>(4, 4);
        Assert.Equal(16, split.Length);
        for (long i = 0; i < 16; i++) Assert.Equal(HostUniform(i, childKey), split[i]);
    }

    [Fact]
    public void TestKeyedNormalHasStandardMoments()
    {
        var vals = RunDraw<RngKeyedNormalDraw>(40, 40);
        double mean = vals.Average();
        double variance = vals.Select(v => (v - mean) * (v - mean)).Average();
        Assert.InRange(mean, -0.1, 0.1);
        Assert.InRange(variance, 0.85, 1.15);
    }

    [Fact]
    public void TestGetFunctionValidatesTheBitsWidthAndRejectsUnknownAlgorithmsForEveryKind()
    {
        // bits requires a supported uint width...
        Assert.Throws<NotSupportedException>(
            () => RngAlgorithms.GetFunction(RngAlgorithms.Default, RngAlgorithms.KindBits, DType.Float32));
        Assert.Throws<NotSupportedException>(
            () => RngAlgorithms.GetFunction(RngAlgorithms.Default, RngAlgorithms.KindBits, null));
        // ...and a bitsDtype must not be supplied for a non-bits kind.
        Assert.Throws<ArgumentException>(
            () => RngAlgorithms.GetFunction(RngAlgorithms.Default, RngAlgorithms.KindUniform, DType.UInt32));

        // The split kinds remap the name to the default (the key tree is algorithm-independent);
        // that remap must never launder an unrecognized name into a valid one.
        foreach (var kind in (string[])[
            RngAlgorithms.KindSplit, RngAlgorithms.KindSplitBatch,
            RngAlgorithms.KindUniform, RngAlgorithms.KindNormal])
        {
            var ex = Assert.Throws<NotSupportedException>(
                () => RngAlgorithms.GetFunction("Threefry4x64-Ziggurat.v9", kind));
            Assert.Contains("Threefry4x64-Ziggurat.v9", ex.Message);
        }
    }

    [Fact]
    public void TestRngFunctionsExportNonInlinedWithMetadata()
    {
        var g = ((ComputationGraph)typeof(RngSplitThenDraw)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var input = TensorData([2L, 2L], 0f, 0f, 0f, 0f);
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([input])).ToConcreteModel();

        var proto = FastOnnxModelBuilder.BuildOnnxModel(concrete);

        // RNG is graph-only (#136): the split no longer constant-folds host-side. Both the split
        // AND the draw survive as calls of the algorithm's NON-INLINED functions — local
        // FunctionProtos tagged with the algorithm name and function kind.
        var rngFns = proto.Functions.Where(f => f.Name.Contains("ShrkRng_")).ToArray();
        Assert.Equal(2, rngFns.Length);
        foreach (var fn in rngFns)
        {
            var algo = fn.MetadataProps.FirstOrDefault(p => p.Key == Function.IRRngAlgorithmParamName)?.Value;
            var kind = fn.MetadataProps.FirstOrDefault(p => p.Key == Function.IRRngFunctionKindParamName)?.Value;
            Assert.Equal(RngAlgorithms.Default, algo);
            Assert.Contains(kind, (string[])[RngAlgorithms.KindSplit, RngAlgorithms.KindUniform, RngAlgorithms.KindNormal]);
        }

        // The main graph CALLS both the split and the draw (Functions-domain call nodes), not
        // their spliced bodies; no raw SHRK_RNG_SPLIT opcode survives.
        var callOps = proto.Graph.Nodes.Where(n => n.Domain == "Functions").Select(n => n.OpType).ToArray();
        Assert.Contains(callOps, op => op.Contains("uniform"));
        Assert.Contains(callOps, op => op.Contains("split"));
        Assert.DoesNotContain(proto.Graph.Nodes, n => n.OpType.Contains("RngSplit"));
    }
}

/// <summary>
/// Switching the configured <see cref="RngAlgorithm"/> between the default 20-round Threefry
/// draw and the reduced 13-round variant: it must change the numbers drawn (runtime feeds and
/// parameter init alike), stay deterministic per algorithm, export the selected algorithm's
/// tagged function, and — because the key tree is algorithm-independent — leave every stream's
/// resolved key untouched.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngAlgorithmSwitchTests
{
    private static readonly RngConfig Rounds20 = new() { MasterSeed = 5, Algorithm = RngAlgorithm.Threefry2x32 };
    private static readonly RngConfig Rounds13 = new() { MasterSeed = 5, Algorithm = RngAlgorithm.Threefry2x32Rounds13 };

    private static InternalComputationGraph FeedModel(RngConfig cfg)
    {
        var g = ((ComputationGraph)typeof(RtLoweredUniform)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var input = TensorData([4L, 4L], Enumerable.Repeat(0f, 16).ToArray());
        return g.ToConcreteArchitecture(g.FromOrderedInputs([input])).ToConcreteModel(cfg);
    }

    private static float[] RunFeed(InternalComputationGraph concrete)
    {
        var input = TensorData([4L, 4L], Enumerable.Repeat(0f, 16).ToArray());
        return ComputeContext.Default.Execute(concrete, input)[0]
            .ToTensorData().As<float32>().AccessMemory().ToArray();
    }

    /// <summary>The feed's resolved stream key, derived from the graph's bound RngSeed identity
    /// — exactly what the feed's in-graph split chain derives at execution.</summary>
    private static ulong ResolvedKey(InternalComputationGraph concrete)
    {
        var feed = concrete.Nodes.Single(n => n.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM);
        var path = feed.Attributes.GetIntsVal(ShrkAttrLocalModelId)!;
        return RngTestOracle.RunKey(RngRuntimeIdentity.Decode(concrete.TryGetRngSeed()!), path);
    }

    private static string BoundAlgorithm(InternalComputationGraph concrete)
        => RngAlgorithms.NameOf(
            RngRuntimeIdentity.Decode(concrete.TryGetRngSeed()!).Algorithm!.Value);

    [Fact]
    public void TestRuntimeFeedDrawSwitchesWithAlgorithmStaysDeterministicAndKeepsTheStreamKey()
    {
        var concrete20 = FeedModel(Rounds20);
        var concrete13 = FeedModel(Rounds13);

        var draws20 = RunFeed(concrete20);
        var draws13 = RunFeed(concrete13);

        Assert.Equal(draws20, RunFeed(concrete20));   // deterministic per algorithm
        Assert.Equal(draws13, RunFeed(concrete13));
        Assert.NotEqual(draws20, draws13);            // same stream, different bit generator

        // The resolved stream key is identical across algorithms (the key tree is fixed); only
        // the carrier's algorithm tag differs.
        var key20 = ResolvedKey(concrete20);
        var key13 = ResolvedKey(concrete13);
        Assert.Equal(key20, key13);
        Assert.Equal(RngAlgorithms.Threefry2x32BoxMullerV1, BoundAlgorithm(concrete20));
        Assert.Equal(RngAlgorithms.Threefry2x32x13BoxMullerV1, BoundAlgorithm(concrete13));

        // Bit-exact against the host generator at each algorithm's round count (substreamIndex 0
        // — the injected counter is baked at 0 in one-shot inference).
        for (long i = 0; i < 16; i++)
        {
            Assert.Equal(RngDenseUniformOracle.Draw(key20, 0, i, 0f, 1f, Threefry2x32.Rounds), draws20[i]);
            Assert.Equal(RngDenseUniformOracle.Draw(key13, 0, i, 0f, 1f, Threefry2x32.Rounds13), draws13[i]);
        }
    }

    [Fact]
    public void TestInitDrawsSwitchWithAlgorithmAndATamperedIdentityAlgorithmFailsLoudly()
    {
        var ig = ((ComputationGraph)typeof(SwitchInitLinear)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var initInput = TensorData([1L, 3L], 0.1f, 0.2f, 0.3f);
        var initArch = ig.ToConcreteArchitecture(ig.FromOrderedInputs([initInput]));

        float[] Weight(RngConfig cfg) =>
            initArch.InitializeTrainableParams(rngConfig: cfg).ModelParams[0]
                .ToTensorData<float32>().AccessMemory().ToArray();

        var w20 = Weight(Rounds20);
        var w13 = Weight(Rounds13);
        Assert.Equal(w20, Weight(Rounds20));   // deterministic per algorithm
        Assert.Equal(w13, Weight(Rounds13));
        Assert.NotEqual(w20, w13);             // init noise honors the switched algorithm

        // SwitchInitLinear has no runtime feeds, so no RngSeed exists to tamper with; use a
        // model that carries one — no-config init reads its algorithm.
        var g = ((ComputationGraph)typeof(RtLoweredUniform)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([TensorData([4L, 4L], new float[16])]));
        arch.ApplyRngConfig(Rounds20);

        const ulong unknownId = 9999;
        var identity = arch.TryGetRngSeed()!;
        identity[RngRuntimeIdentity.AlgorithmIdIndex] = unknownId;
        var seedNode = arch.Nodes.Single(n =>
            n.IdentifierTemplate == Shorokoo.Core.Nodes.Processors.Fast
                .FastWireRngKeyDerivation.RngSeedIdentifierTemplate);
        seedNode.Attributes = seedNode.Attributes.SetAttributes(
            (ShrkAttrTensorData, (object?)Shorokoo.TensorData.Create(
                new Shape(identity.Length), DType.UInt64,
                Shorokoo.Core.Utils.OnnxUtils.CreateTensorValue(new Shape(identity.Length), identity))));

        var ex = Assert.Throws<NotSupportedException>(() => arch.InitializeTrainableParams());
        Assert.Contains(unknownId.ToString(), ex.Message);
        // The escape hatch: an explicit config bypasses the identity decode.
        Assert.NotEmpty(arch.InitializeTrainableParams(rngConfig: Rounds20).ModelParams);
    }

    [Fact]
    public void TestExportTagsTheSelectedAlgorithmFunction()
    {
        static (string name, string algo) UniformFn(RngConfig cfg)
        {
            var proto = FastOnnxModelBuilder.BuildOnnxModel(FeedModel(cfg));
            var fn = proto.Functions.Single(f => f.Name.Contains("ShrkRng_") && f.Name.Contains("uniform"));
            return (fn.Name, fn.MetadataProps.First(p => p.Key == Function.IRRngAlgorithmParamName).Value);
        }

        var (name20, algo20) = UniformFn(Rounds20);
        var (name13, algo13) = UniformFn(Rounds13);

        Assert.Equal(RngAlgorithms.Threefry2x32BoxMullerV1, algo20);
        Assert.Equal(RngAlgorithms.Threefry2x32x13BoxMullerV1, algo13);
        Assert.Contains("Threefry2x32_13", name13);
        Assert.DoesNotContain("Threefry2x32_13", name20);
        Assert.NotEqual(name20, name13);
    }
}

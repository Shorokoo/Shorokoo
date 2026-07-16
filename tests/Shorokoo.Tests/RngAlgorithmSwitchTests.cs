using System;
using System.Linq;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Rng;
using Shorokoo.Modules.Layers;
using Shorokoo.Runtime;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;

namespace Shorokoo.Tests;

/// <summary>A single Linear whose weight is drawn by a random initializer — so its init values
/// change when the RNG algorithm changes.</summary>
[Module]
public partial class SwitchInitLinear
{
    public static Tensor<float32> Inline(Tensor<float32> x)
        => Linear.Model(Scalar(4L), Scalar(false)).Call(x);
}

/// <summary>
/// Validates switching the configured <see cref="RngAlgorithm"/> between the default 20-round
/// Threefry draw and the reduced 13-round variant. Switching must: change the numbers drawn
/// (runtime feeds and parameter init alike), stay deterministic per algorithm, export the
/// selected algorithm's tagged function, and — because the key tree is algorithm-independent —
/// leave every stream's resolved key untouched (only the draw changes, not which stream is which).
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngAlgorithmSwitchTests
{
    private static readonly RngConfig Rounds20 = new() { MasterSeed = 5, Algorithm = RngAlgorithm.Threefry2x32 };
    private static readonly RngConfig Rounds13 = new() { MasterSeed = 5, Algorithm = RngAlgorithm.Threefry2x32Rounds13 };

    private static FastComputationGraph FeedModel(RngConfig cfg)
    {
        var g = (FastComputationGraph)typeof(RtLoweredUniform)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var input = TensorData([4L, 4L], Enumerable.Repeat(0f, 16).ToArray());
        return g.ToConcreteArchitecture(g.FromOrderedInputs([input])).ToConcreteModel(cfg);
    }

    private static float[] RunFeed(FastComputationGraph concrete)
    {
        var input = TensorData([4L, 4L], Enumerable.Repeat(0f, 16).ToArray());
        return ComputeContext.Default.Execute(concrete, input)[0]
            .ToTensorData().As<float32>().AccessMemory().ToArray();
    }

    /// <summary>The feed's resolved stream key, derived from the graph's carrier — exactly
    /// what the lowering derives when it emits the keyed draw.</summary>
    private static long[] ResolvedKey(FastComputationGraph concrete)
    {
        var feed = concrete.Nodes.Single(n => n.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM);
        var path = feed.Attributes.GetIntsVal(ShrkAttrLocalModelId)!;
        var decoded = RngConfig.FromKeyVector(concrete.TryGetRngKeyVector()!.Value.keyVector);
        var (k0, k1) = decoded.FoldRunKey(path);
        return [k0, k1];
    }

    private static string BoundAlgorithm(FastComputationGraph concrete)
        => concrete.TryGetRngKeyVector()!.Value.algorithm;

    [Fact]
    public void TestRuntimeFeedDrawSwitchesWithAlgorithmAndStaysDeterministic()
    {
        var concrete20 = FeedModel(Rounds20);
        var concrete13 = FeedModel(Rounds13);

        var draws20 = RunFeed(concrete20);
        var draws13 = RunFeed(concrete13);

        // Deterministic per algorithm (re-execute is identical).
        Assert.Equal(draws20, RunFeed(concrete20));
        Assert.Equal(draws13, RunFeed(concrete13));
        // Same stream key, different bit generator -> different draws.
        Assert.NotEqual(draws20, draws13);

        // Bit-exact against the host generator at each algorithm's round count, using the
        // feed's actually-resolved key (drawBase 0 — the injected counter is baked at 0 in
        // one-shot inference).
        var key20 = ResolvedKey(concrete20);
        var key13 = ResolvedKey(concrete13);
        for (long i = 0; i < 16; i++)
        {
            var (h20, _) = Threefry2x32.Bijection((uint)i, 0u, (uint)key20[0], (uint)key20[1], Threefry2x32.Rounds);
            var (h13, _) = Threefry2x32.Bijection((uint)i, 0u, (uint)key13[0], (uint)key13[1], Threefry2x32.Rounds13);
            Assert.Equal((h20 & 0x00FFFFFFu) * (1.0f / 16777216.0f), draws20[i]);
            Assert.Equal((h13 & 0x00FFFFFFu) * (1.0f / 16777216.0f), draws13[i]);
        }
    }

    [Fact]
    public void TestSwitchingAlgorithmDoesNotRekeyTheStream()
    {
        // The resolved stream key is identical across algorithms (the key tree is fixed);
        // only the carrier's algorithm tag differs. Switching draws different numbers from
        // the SAME stream — it never reshuffles which stream is which.
        var concrete20 = FeedModel(Rounds20);
        var concrete13 = FeedModel(Rounds13);

        Assert.Equal(ResolvedKey(concrete20), ResolvedKey(concrete13));
        Assert.Equal(RngAlgorithms.Threefry2x32BoxMullerV1, BoundAlgorithm(concrete20));
        Assert.Equal(RngAlgorithms.Threefry2x32x13BoxMullerV1, BoundAlgorithm(concrete13));
    }

    [Fact]
    public void TestInitDrawsSwitchWithAlgorithm()
    {
        var g = (FastComputationGraph)typeof(SwitchInitLinear)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var input = TensorData([1L, 3L], 0.1f, 0.2f, 0.3f);
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([input]));

        float[] Weight(RngConfig cfg) =>
            arch.InitializeTrainableParams(rngConfig: cfg).ModelParams[0]
                .ToTensorData<float32>().AccessMemory().ToArray();

        var w20 = Weight(Rounds20);
        var w13 = Weight(Rounds13);

        Assert.Equal(w20, Weight(Rounds20));   // deterministic per algorithm
        Assert.Equal(w13, Weight(Rounds13));
        Assert.NotEqual(w20, w13);             // init noise honors the switched algorithm
    }

    [Fact]
    public void TestTamperedCarrierAlgorithmFailsLoudlyAtParamInit()
    {
        var g = (FastComputationGraph)typeof(SwitchInitLinear)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var input = TensorData([1L, 3L], 0.1f, 0.2f, 0.3f);
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([input]));
        arch.ApplyRngConfig(Rounds20);

        // A model file written by a newer framework version: the carrier's recorded algorithm
        // name is one this version does not know.
        const string newerName = "Threefry4x64-Ziggurat.v9";
        var carrier = arch.Nodes.Single(n => n.OpCode == InternalOpCodes.SHRK_RNG_KEY_VECTOR);
        carrier.Attributes = carrier.Attributes.SetAttributes((ShrkAttrRngAlgorithm, newerName));
        Assert.Equal(newerName, arch.TryGetRngKeyVector()!.Value.algorithm);

        // No-config init trusts the carrier as the identity: an unreadable identity must
        // throw — never silently re-initialize under a default algorithm while the carrier
        // keeps reporting the newer name.
        var ex = Assert.Throws<NotSupportedException>(() => arch.InitializeTrainableParams());
        Assert.Contains(newerName, ex.Message);

        // The escape hatch for deliberately re-keying an unreadable file: an explicit config
        // bypasses the carrier decode.
        Assert.NotEmpty(arch.InitializeTrainableParams(rngConfig: Rounds20).ModelParams);
    }

    [Fact]
    public void TestExportTagsTheSelectedAlgorithmFunction()
    {
        static (string name, string algo) UniformFn(RngConfig cfg)
        {
            var proto = FastOnnxModelBuilder.BuildOnnxModel(FeedModel(cfg));
            var fn = proto.Functions.Single(f => f.Name.Contains("ShrkRng_") && f.Name.Contains("uniform"));
            var algo = fn.MetadataProps.First(p => p.Key == Function.IRRngAlgorithmParamName).Value;
            return (fn.Name, algo);
        }

        var (name20, algo20) = UniformFn(Rounds20);
        var (name13, algo13) = UniformFn(Rounds13);

        Assert.Equal(RngAlgorithms.Threefry2x32BoxMullerV1, algo20);
        Assert.Equal(RngAlgorithms.Threefry2x32x13BoxMullerV1, algo13);
        // The two algorithms export distinct, identifiable functions.
        Assert.Contains("Threefry2x32_13", name13);
        Assert.DoesNotContain("Threefry2x32_13", name20);
        Assert.NotEqual(name20, name13);
    }
}

using System;
using System.Linq;
using Shorokoo.Core.Rng;
using Shorokoo.Modules.Layers;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// Two same-shape Linear weights at distinct module paths (so distinct parameters,
/// each KaimingUniform-initialized on a [4,4] weight). Distinct paths — not two bare
/// identical Init calls, which would be a single common-subexpression parameter.
/// </summary>
[Module]
public partial class RngInitTwoLinears
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var y1 = Linear.Call(Scalar(4L), Scalar(false), x);   // weight [4,4]
        var y2 = Linear.Call(Scalar(4L), Scalar(false), y1);  // weight [4,4], distinct path
        return y2;
    }
}

/// <summary>
/// End-to-end coverage for per-parameter initialization RNG (phase 2). Concretizes
/// <see cref="RngInitTwoLinears"/> and initializes it under various
/// <see cref="RngConfig"/>s, asserting the properties the design promises:
/// same-shape parameters now differ, initialization is reproducible for a config,
/// and the master seed re-randomizes everything.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngInitTests
{
    private static FastComputationGraph ConcreteArch()
    {
        var g = RngInitTwoLinears.ComputationGraph;
        var sample = TensorData([4L, 4L], Enumerable.Repeat(1f, 16).ToArray());
        return g.ToConcreteArchitecture(g.FromOrderedInputs([sample]));
    }

    private static float[][] InitWeights(RngConfig? cfg = null)
    {
        var arch = ConcreteArch();
        var pl = arch.InitializeTrainableParams(rngConfig: cfg);
        // Both Linear weights are [4,4] = 16 elements.
        return pl.ModelParams
            .Select(p => p.ToTensorData().As<float32>().AccessMemory().ToArray())
            .Where(v => v.Length == 16)
            .ToArray();
    }

    [Fact]
    public void TestSameShapeParamsAreNotIdentical()
    {
        var w = InitWeights();
        Assert.Equal(2, w.Length);
        // The core bug the design fixes: two same-shape parameters previously received
        // identical values; keyed by their (distinct) canonical names they now differ.
        Assert.False(w[0].SequenceEqual(w[1]));
    }

    [Fact]
    public void TestInitializationIsReproducibleForAConfig()
    {
        var a = InitWeights(new RngConfig { MasterSeed = 123 });
        var b = InitWeights(new RngConfig { MasterSeed = 123 });
        Assert.Equal(a.Length, b.Length);
        for (int i = 0; i < a.Length; i++)
            Assert.Equal(a[i], b[i]);
    }

    [Fact]
    public void TestMasterSeedChangesAllValues()
    {
        var a = InitWeights(new RngConfig { MasterSeed = 1 });
        var b = InitWeights(new RngConfig { MasterSeed = 2 });
        for (int i = 0; i < a.Length; i++)
            Assert.False(a[i].SequenceEqual(b[i]), $"param {i} unchanged across seeds");
    }

    [Fact]
    public void TestKaimingValuesAreFiniteAndInBound()
    {
        // KaimingUniform bound for fanIn=4 is sqrt(6/4) ≈ 1.22474; values stay within it.
        foreach (var v in InitWeights())
            foreach (var x in v)
            {
                Assert.True(float.IsFinite(x));
                Assert.InRange(x, -1.2248f, 1.2248f);
            }
    }

    [Fact]
    public void TestUnmatchedParamsOverrideFailsInitialization()
    {
        // Mirror of the Runtime-side bind check: a Params override that matches no trainable
        // parameter must fail initialization loudly — a silently inactive override is exactly
        // the re-keying hazard explicit seeding exists to prevent.
        var cfg = new RngConfig { MasterSeed = 1 };
        cfg = cfg.Override(RngCollection.Params, [9, 9, 9], 1UL);
        var ex = Assert.Throws<InvalidOperationException>(() => InitWeights(cfg));
        Assert.Contains("matches no trainable parameter", ex.Message);
    }

    [Fact]
    public void TestMatchedParamsOverrideReSeedsExactlyOneParam()
    {
        // Overriding one weight's stream by its ModelId path re-rolls that weight only.
        var baseline = InitWeights(new RngConfig { MasterSeed = 5 });

        var arch = ConcreteArch();
        var firstWeightPath = arch.GetConcreteModelParamInfos().ParamInfos
            .Single(p => p.Shape.Dims.SequenceEqual((long[])[4, 4]) && p.ModelId.Vals[0] == 1)
            .ModelId.Vals.ToArray();
        var cfg = new RngConfig { MasterSeed = 5 };
        cfg = cfg.Override(RngCollection.Params, firstWeightPath, 4242UL);

        var overridden = InitWeights(cfg);
        Assert.False(baseline[0].SequenceEqual(overridden[0]));   // re-seeded
        Assert.Equal(baseline[1], overridden[1]);                 // untouched
    }
}

/// <summary>
/// The init-value derivation pinned to FROZEN constants — the cross-version seed contract.
/// Every other init test is relational (the system compared against itself), so a silent
/// change anywhere in the chain — master → "init" sub-master fold → per-path FoldInitKey →
/// HostRng (counterBase, draw rounds) → uniform transform → Kaiming scaling — would keep
/// them all green while breaking every seed anyone has ever shared. These values were
/// generated once from the implementation that defines the derivation; a red here means
/// "MasterSeed 123 no longer produces the weights it used to" and must never be fixed by
/// regenerating the constants without a deliberate, breaking-change decision.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngInitFrozenDerivationTests
{
    [Fact]
    public void TestInitKeyDerivationIsFrozen()
    {
        // Layer 1: the key derivation alone (fold order, the "init" label, sub-master wiring).
        var cfg = new RngConfig { MasterSeed = 123 };
        Assert.Equal((0x0177f47cu, 0x33e150fcu), cfg.FoldInitKey([1, 1]));
        Assert.Equal((0x3c6c3147u, 0x2a93ecfcu), cfg.FoldInitKey([2, 1]));
    }

    [Fact]
    public void TestInitValuesAreFrozen()
    {
        // Layer 2: the full materialized values (draw composition: counterBase, rounds,
        // uniform transform, ordinal scheme, initializer scaling). REFERENCE: golden.
        float[] expected0 = [0.30900666f, 0.6994194f, 0.08134546f, -0.33324182f, 0.37910292f, 1.0430968f, 1.1605477f, -0.09467988f, 1.2190907f, 0.10304924f, 0.94080746f, -0.7929816f, -0.56461143f, -0.6191869f, 0.02709632f, 0.8262475f];
        float[] expected1 = [-0.17985144f, -0.28237903f, 0.14574882f, 0.9324704f, 0.6254746f, -0.6462647f, 0.06555341f, 0.7868881f, 0.7384043f, -0.21305309f, -0.88358283f, 0.10607847f, -0.1497983f, -1.006346f, -0.48929685f, 0.80844414f];

        var g = RngInitTwoLinears.ComputationGraph;
        var sample = TensorData([4L, 4L], System.Linq.Enumerable.Repeat(1f, 16).ToArray());
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([sample]));
        var pl = arch.InitializeTrainableParams(rngConfig: new RngConfig { MasterSeed = 123 });
        var ws = pl.ModelParams
            .Select(p => p.ToTensorData().As<float32>().AccessMemory().ToArray())
            .Where(v => v.Length == 16).ToArray();

        Assert.Equal(2, ws.Length);
        Assert.Equal(expected0, ws[0]);   // weight at ModelId [1, 1]
        Assert.Equal(expected1, ws[1]);   // weight at ModelId [2, 1]
    }
}

/// <summary>
/// In-place trainable-parameter re-initialization
/// (<c>ReinitializeTrainableParams(rngConfig)</c>): the explicit opt-in that re-runs every
/// trainable parameter's initializer under a new identity and overwrites the values on the
/// concrete model — the params-collection half of "re-binding is re-initialization" (the
/// runtime half being <c>ApplyRngConfig</c>). Bit-exact with a fresh build under the same
/// config; model state and the runtime identity stay untouched; fails loudly when the
/// parameter inventory (the in-memory source architecture) is unavailable.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngReinitializeTests
{
    private static FastComputationGraph ConcreteArch()
    {
        var g = RngInitTwoLinears.ComputationGraph;
        var sample = TensorData([4L, 4L], Enumerable.Repeat(1f, 16).ToArray());
        return g.ToConcreteArchitecture(g.FromOrderedInputs([sample]));
    }

    /// <summary>The model's trainable weights, ordered by parameter identity.</summary>
    private static float[][] TrainableWeights(FastComputationGraph model)
        => model.Nodes
            .Where(n => n.OpCode == Shorokoo.Core.Nodes.NodeDefinitions.InternalOpCodes.MODEL_PARAM_DATA
                        && (n.Attributes.GetBoolVal(OnnxOpAttributeNames.ShrkAttrIsTrainable) ?? false))
            .OrderBy(n => n.IdentifierTemplate, StringComparer.Ordinal)
            .Select(n => n.Attributes.GetTensorVal(OnnxOpAttributeNames.ShrkAttrTensorData)!
                .As<float32>().AccessMemory().ToArray())
            .ToArray();

    [Fact]
    public void TestReinitializeMatchesFreshBuildBitExactly()
    {
        // THE pin: initialize under seed A, re-initialize in place under seed B → values
        // change and match a fresh build under seed B bit-exactly (same code path: keyed
        // host noise per parameter stream).
        var seedA = new RngConfig { MasterSeed = 5 };
        var seedB = new RngConfig { MasterSeed = 6 };

        var model = ConcreteArch().ToConcreteModel(seedA);
        var weightsA = TrainableWeights(model);

        model.ReinitializeTrainableParams(seedB);

        var weightsB = TrainableWeights(model);
        Assert.Equal(weightsA.Length, weightsB.Length);
        for (int i = 0; i < weightsA.Length; i++)
            Assert.False(weightsA[i].SequenceEqual(weightsB[i]), $"param {i} unchanged");

        var freshB = TrainableWeights(ConcreteArch().ToConcreteModel(seedB));
        Assert.Equal(freshB.Length, weightsB.Length);
        for (int i = 0; i < freshB.Length; i++)
            Assert.Equal(freshB[i], weightsB[i]);   // bit-exact
    }

    [Fact]
    public void TestReinitializeHonorsParamsOverridesAndValidatesThem()
    {
        // RngCollection.Params overrides validate against the inventory exactly as at first
        // initialization: a matched override re-seeds exactly that parameter; an unmatched
        // one fails loudly.
        var arch = ConcreteArch();
        var firstWeightPath = arch.GetConcreteModelParamInfos().ParamInfos
            .Single(p => p.Shape.Dims.SequenceEqual((long[])[4, 4]) && p.ModelId.Vals[0] == 1)
            .ModelId.Vals.ToArray();

        var seed = new RngConfig { MasterSeed = 5 };
        var model = ConcreteArch().ToConcreteModel(seed);
        var baseline = TrainableWeights(ConcreteArch().ToConcreteModel(seed));

        model.ReinitializeTrainableParams(
            seed.Override(RngCollection.Params, firstWeightPath, 4242UL));
        var overridden = TrainableWeights(model);
        Assert.False(baseline[0].SequenceEqual(overridden[0]));   // re-seeded
        Assert.Equal(baseline[1], overridden[1]);                 // untouched

        var ex = Assert.Throws<InvalidOperationException>(() =>
            model.ReinitializeTrainableParams(
                seed.Override(RngCollection.Params, [9, 9, 9], 1UL)));
        Assert.Contains("matches no trainable parameter", ex.Message);
    }

    [Fact]
    public void TestReinitializeLeavesRuntimeIdentityAndDrawsUntouched()
    {
        // Re-initialization is the params-collection half only: the runtime identity (the
        // RngSeed parameter) and every feed draw stay exactly as bound — re-keying the
        // runtime side is a separate, equally explicit ApplyRngConfig call.
        var g = RngNormalBothCollections.ComputationGraph;
        var input = TensorData([4L, 4L], Enumerable.Repeat(0f, 16).ToArray());
        var seedA = new RngConfig { MasterSeed = 123 };
        var model = g.ToConcreteArchitecture(g.FromOrderedInputs([input])).ToConcreteModel(seedA);

        float[] Run() => ComputeContext.Default.Execute(model, input)[0]
            .ToTensorData().As<float32>().AccessMemory().ToArray();

        var identityBefore = model.TryGetRngSeed();
        var drawsBefore = Run();   // = the feed draw (the weight term is multiplied by 0)

        model.ReinitializeTrainableParams(new RngConfig { MasterSeed = 456 });

        Assert.Equal(identityBefore, model.TryGetRngSeed());   // identity untouched
        Assert.Equal(drawsBefore, Run());                      // feed draws untouched
    }

    [Fact]
    public void TestReinitializeOnLoadedModelFailsLoudly()
    {
        // The parameter inventory (initializer functions) is in-memory only — a loaded model
        // cannot be re-initialized in place and must say so, never silently no-op.
        var model = ConcreteArch().ToConcreteModel(new RngConfig { MasterSeed = 5 });
        var loaded = Shorokoo.Core.Utils.CompressedFormatUtils.LoadFastGraphFromBinary(
            Shorokoo.Core.Utils.CompressedFormatUtils.SaveFastGraphToBinary(model, compressed: true),
            isCompressed: true);

        var ex = Assert.Throws<InvalidOperationException>(() =>
            loaded.ReinitializeTrainableParams(new RngConfig { MasterSeed = 6 }));
        Assert.Contains("re-initialized in place", ex.Message);
    }
}

/// <summary>
/// The two normal-family consumers in one module: a KaimingNormal-initialized [4,4] weight
/// (host Box–Muller path, run at parameter initialization) and a Globals.RandomNormal feed
/// (in-graph Box–Muller path, lowered to the keyed counter RNG). The weight is kept live via
/// a ×0 term, so with a zero input the module's output equals the feed draw exactly.
/// </summary>
[Module]
public partial class RngNormalBothCollections
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var w = Shorokoo.Modules.Initializers.KaimingNormal.Init([Scalar(4L), Scalar(4L)]);
        var feed = RandomNormal(x.ShapeTensor(), mean: 0.0f, scale: 1.0f);
        return feed + w.Reduce(ReduceKind.Sum, keepDims: false).Scalar() * Scalar(0.0f);
    }
}

/// <summary>
/// The NORMAL-family value derivation pinned to FROZEN constants, for both consumer kinds —
/// the cross-version seed contract for normals (the sibling of
/// <see cref="RngInitFrozenDerivationTests"/>, which pins the uniform family). Every
/// Box–Muller composition variant (cos↔sin, u₁↔u₂, 1−u₁↔u₁, uniform-to-element pairing)
/// yields a perfect N(0,1) distribution, so the moments tests can never detect a composition
/// change: only value pins hold the convention fixed. One Fact covers the host path
/// (parameter initialization: fold → init sub-master → HostRng pair-packed Box–Muller →
/// Kaiming scaling) and the in-graph path (runtime feed: RngSeed-rooted split chain →
/// per-element SHRK lowering → ONNX Ln/Sqrt/Cos kernels), at both round counts. The two paths realize
/// different sequences BY DESIGN (see HostRng) and are pinned independently, never compared.
/// Feed values are asserted at 1e-6 (ORT transcendental kernels may drift in the last ULP;
/// a composition change shifts values by O(1)). A red here means "this seed no longer draws
/// the normals it used to" and must never be fixed by regenerating the constants without a
/// deliberate, breaking-change decision.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngNormalFrozenDerivationTests
{
    private static (float[] init, float[] feed) Run(RngConfig cfg)
    {
        var g = RngNormalBothCollections.ComputationGraph;
        var input = TensorData([4L, 4L], Enumerable.Repeat(0f, 16).ToArray());
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([input]));
        // Filter to the float32 weight before casting: the param list also carries the
        // framework-injected RngExecutionCounter, which is int64 state.
        var init = arch.InitializeTrainableParams(rngConfig: cfg).ModelParams
            .Select(p => p.ToTensorData())
            .Where(t => t.DType == DType.Float32)
            .Select(t => t.As<float32>().AccessMemory().ToArray())
            .Single(v => v.Length == 16);
        var concrete = arch.ToConcreteModel(cfg);
        var feed = ComputeContext.Default.Execute(concrete, input)[0]
            .ToTensorData().As<float32>().AccessMemory().ToArray();
        return (init, feed);
    }

    [Fact]
    public void TestNormalInitAndDrawValuesAreFrozen()
    {
        // REFERENCE: golden — generated once from the implementation that defines the convention.
        float[] init20 = [0.74819666f, -1.3497522f, 0.58009183f, -0.5437696f, -1.1170965f, 0.33587033f, 0.31675094f, 0.19958028f, 0.16639294f, 0.9580838f, 0.2545579f, -0.76778877f, 0.17528465f, 0.9627476f, -0.8868993f, -0.65395457f];
        float[] feed20 = [-0.21420276f, -0.5717528f, 0.46444735f, -1.0332288f, 0.46397528f, 0.84883124f, 0.6769181f, -0.8103971f, 0.25310147f, 0.33421588f, 0.14988664f, 0.105597205f, -0.270022f, 0.26715103f, -0.052951735f, 0.9648315f];
        float[] init13 = [0.5342924f, 0.030172912f, 0.969521f, -0.4301536f, -0.5262921f, 0.34364656f, 0.0136166755f, 0.48939812f, 1.076198f, 0.542235f, -0.5106929f, -0.28768054f, -0.63540673f, 0.03363665f, -0.03083078f, 0.53736115f];
        float[] feed13 = [-2.6632779f, -0.93596685f, 1.2713523f, 0.5301773f, -2.0887053f, 0.17946513f, 0.503763f, 0.6912192f, -2.0152493f, -0.9966242f, -0.23023638f, 0.6537417f, 0.16786636f, -0.09063158f, 0.575317f, 1.555586f];

        var (i20, f20) = Run(new RngConfig { MasterSeed = 123 });
        // Host path: exact, like the uniform pin (pure MathF, no backend kernel involved).
        Assert.Equal(init20, i20);
        // In-graph path: ORT Ln/Sqrt/Cos kernels — tight tolerance, not bit-equality.
        for (int i = 0; i < 16; i++)
            Assert.Equal(feed20[i], f20[i], 1e-6f);

        var (i13, f13) = Run(new RngConfig { MasterSeed = 123, Algorithm = RngAlgorithm.Threefry2x32Rounds13 });
        Assert.Equal(init13, i13);
        for (int i = 0; i < 16; i++)
            Assert.Equal(feed13[i], f13[i], 1e-6f);
    }
}

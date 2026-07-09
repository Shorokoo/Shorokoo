using System;
using System.Linq;
using Shorokoo.Modules.Layers;

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

    [Fact]
    public void TestInitFailsLoudlyOnUnrecognizedCarrierAlgorithm()
    {
        // Regression: an unrecognized carrier algorithm (e.g. one written by a newer
        // Shorokoo version) used to silently substitute the 20-round default here, while
        // the runtime-feed side already failed loudly for the same situation — so a
        // loaded model's weights could initialize under an identity the carrier didn't
        // actually name. Both sides must fail loudly now.
        var arch = ConcreteArch();
        arch.ApplyRngConfig(new RngConfig { MasterSeed = 1 });

        var carrier = arch.Nodes.Single(
            n => n.OpCode == Shorokoo.Core.Nodes.NodeDefinitions.InternalOpCodes.SHRK_RNG_KEY_VECTOR);
        carrier.Attributes = carrier.Attributes.SetAttributes(
            (Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames.ShrkAttrRngAlgorithm,
                (object?)"SomeFutureAlgorithm.v1"));

        var ex = Assert.Throws<InvalidOperationException>(
            () => arch.InitializeTrainableParams());
        Assert.Contains("SomeFutureAlgorithm.v1", ex.Message);
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

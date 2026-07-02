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
}

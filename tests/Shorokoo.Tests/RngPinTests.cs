using System.Linq;
using Shorokoo.Modules.Layers;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>Two Linears created a-then-b, NO pin: a takes the first id slot.</summary>
[Module]
public partial class PinBaselineTwoLinears
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var a = Linear.Model(Scalar(2L), Scalar(false));   // creation order 1 -> id [1]
        var b = Linear.Model(Scalar(3L), Scalar(false));   // creation order 2 -> id [2]
        return a.Call(x).Concat(-1L, b.Call(x));
    }
}

/// <summary>Same creation order a-then-b, but Rng.Pin(b, a): b takes the first id slot.</summary>
[Module]
public partial class PinSwappedTwoLinears
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var a = Linear.Model(Scalar(2L), Scalar(false));
        var b = Linear.Model(Scalar(3L), Scalar(false));
        Rng.Pin(b, a);                                     // pin order defines id order
        return a.Call(x).Concat(-1L, b.Call(x));
    }
}

/// <summary>
/// Rng.Pin reshapes ModelId (hence RNG stream) assignment without touching the graph's
/// dataflow: pinned items take the module-local id slots in pin order, so a pinned module's
/// streams no longer depend on creation position. Verified structurally — the out-features
/// of the param at id path [1, 1] flip from a's (2) to b's (3) under Pin(b, a) — and
/// behaviorally: the pinned module still executes.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngPinTests
{
    private static (long firstParamOutFeatures, float[] output) Probe<TModule>()
    {
        var g = (FastComputationGraph)typeof(TModule)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var input = TensorData([1L, 4L], 0.1f, 0.2f, 0.3f, 0.4f);
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([input]));

        // The weight of the FIRST-id Linear ([1, 1] = sub-model 1's param 1) has shape [out, in].
        var infos = arch.GetConcreteModelParamInfos().ParamInfos;
        var firstWeight = infos.Single(i => i.ModelId.Vals.SequenceEqual(new[] { 1, 1 }));
        var outFeatures = firstWeight.Shape.Dims[0];

        var concrete = arch.ToConcreteModel(RngConfig.Default);
        var output = ComputeContext.Default.Execute(concrete, input)[0]
            .ToTensorData().As<float32>().AccessMemory().ToArray();
        return (outFeatures, output);
    }

    [Fact]
    public void TestPinReordersIdAssignment()
    {
        var (baselineFirst, baselineOut) = Probe<PinBaselineTwoLinears>();
        var (pinnedFirst, pinnedOut) = Probe<PinSwappedTwoLinears>();

        // Baseline: a (out=2) was created first -> id [1]; its weight is [2, in].
        Assert.Equal(2L, baselineFirst);
        // Pinned: Pin(b, a) gives b (out=3) the first slot -> id [1]; its weight is [3, in].
        Assert.Equal(3L, pinnedFirst);

        // Dataflow untouched: both produce [1, 5] outputs and execute fine.
        Assert.Equal(5, baselineOut.Length);
        Assert.Equal(5, pinnedOut.Length);
    }
}

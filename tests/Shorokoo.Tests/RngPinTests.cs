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

/// <summary>Sparse pin: only a is pinned, to slot [2]; b keeps the first free slot (1).</summary>
[Module]
public partial class PinSparseTwoLinears
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var a = Linear.Model(Scalar(2L), Scalar(false));   // creation order 1, pinned to slot 2
        var b = Linear.Model(Scalar(3L), Scalar(false));   // creation order 2 -> first free slot 1
        Rng.Pin(([2], a));
        return a.Call(x).Concat(-1L, b.Call(x));
    }
}

/// <summary>Pins the module INPUT — no id-bearing producer, so the module build must fail.</summary>
[Module]
public partial class PinUnresolvableInput
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var a = Linear.Model(Scalar(2L), Scalar(false));
        Rng.Pin(x);
        return a.Call(x);
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

    [Fact]
    public void TestSparsePinTakesExplicitSlotAndLeavesOthersAlone()
    {
        // Pin(([2], a)) puts a (out=2) at slot 2; unpinned b (out=3) fills the first FREE
        // slot, 1 — a partial sparse pin does not shift unlisted consumers behind the
        // pinned ones the way a partial positional pin would.
        var (firstSlotOutFeatures, output) = Probe<PinSparseTwoLinears>();
        Assert.Equal(3L, firstSlotOutFeatures);
        Assert.Equal(5, output.Length);
    }

    [Fact]
    public void TestUnresolvablePinFailsTheModuleBuild()
    {
        // Pinning something with no RNG stream (here: the module input) must fail the build
        // loudly — a silently inactive pin is exactly the re-keying hazard Pin guards against.
        var ex = Assert.ThrowsAny<Exception>(() =>
            _ = typeof(PinUnresolvableInput).GetProperty("ComputationGraph")!.GetValue(null));
        for (Exception? e = ex; e is not null; e = e.InnerException)
            if (e.Message.Contains("Rng.Pin")) return;
        Assert.Fail($"expected an Rng.Pin build error, got: {ex}");
    }
}

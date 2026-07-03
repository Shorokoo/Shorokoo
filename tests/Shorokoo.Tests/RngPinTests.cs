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
    public void TestRngStreamReportDescribesStreamsAndEmitsPinSkeleton()
    {
        var g = (FastComputationGraph)typeof(PinBaselineTwoLinears)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var input = TensorData([1L, 4L], 0.1f, 0.2f, 0.3f, 0.4f);
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([input]));

        var cfg = new RngConfig { MasterSeed = 3 };
        var report = arch.GetRngStreamReport(cfg);

        // Two Linears, one weight each: two init streams at [1, 1] and [2, 1], named,
        // shaped, and keyed distinctly under the config.
        var inits = report.Streams.Where(s => s.Kind == RngStreamKind.ParamInit).ToList();
        Assert.Equal(2, inits.Count);
        Assert.Equal([1, 1], inits[0].ModelIdPath);
        Assert.Equal([2, 1], inits[1].ModelIdPath);
        Assert.All(inits, s => Assert.Contains("Linear", s.Name));
        Assert.All(inits, s => Assert.NotNull(s.Shape));
        Assert.NotNull(inits[0].KeyWords);
        Assert.NotEqual(inits[0].KeyWords, inits[1].KeyWords);

        // The skeleton lists each stream at its path with the variable left as ?.
        var skeleton = report.EmitPinSkeleton();
        Assert.StartsWith("Rng.Pin(", skeleton);
        Assert.Contains("([1, 1], /*", skeleton);
        Assert.Contains("*/ ?)", skeleton);

        // Without a config, streams are listed but unkeyed.
        Assert.All(arch.GetRngStreamReport().Streams, s => Assert.Null(s.KeyWords));
    }

    [Fact]
    public void TestRngStreamReportShowsLoopFeedWithPrefixKey()
    {
        var g = (FastComputationGraph)typeof(RngRuntimeLoopFeed)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var x = TensorData([8L], new float[8]);
        var steps = TensorData(System.Array.Empty<long>(), 2L);
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([x, steps]));

        var cfg = new RngConfig { MasterSeed = 3 };
        var feed = Assert.Single(arch.GetRngStreamReport(cfg).Streams);

        // The loop-body feed sits at [1, -1, 1]; the reported key is the PREFIX key before
        // the iteration slot (per-iteration keys are split at runtime).
        Assert.Equal(RngStreamKind.UniformFeed, feed.Kind);
        Assert.Equal([1, -1, 1], feed.ModelIdPath);
        Assert.True(feed.KeyIsPrefix);
        var (k0, k1) = cfg.FoldRunKey([1]);
        Assert.Equal([k0, k1], feed.KeyWords);

        // The skeleton elides the -1 slot and flags the loop.
        var skeleton = arch.GetRngStreamReport().EmitPinSkeleton();
        Assert.Contains("([1, 1], /* uniform feed (loop body) */ ?)", skeleton);
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

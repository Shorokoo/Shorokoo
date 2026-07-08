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

/// <summary>Initializer used ONLY by <see cref="PinSurvivesNestedFirstUseBuild"/>, so its Function is
/// guaranteed uncached when that module's body traces — forcing a nested graph build mid-trace.</summary>
[TrainableParamInitializer]
public static partial class PinWipeFreshInit
{
    public static Tensor<float32> Inline(Vector<int64> shape)
    {
        return Globals.TensorFill(shape, 0.5f);
    }
}

/// <summary>
/// Pins recorded BEFORE a nested first-use build must survive it. Building a not-yet-cached
/// sub-module/initializer mid-trace re-enters the graph builder on the same thread; its entry-time
/// pin clearing used to wipe the outer body's already-recorded pins, silently deactivating them
/// (and cache-order-dependently: a warm Function cache hid the loss). Pin(b, a) then first-use
/// a fresh initializer: b must still take the first id slot.
/// </summary>
[Module]
public partial class PinSurvivesNestedFirstUseBuild
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var a = Linear.Model(Scalar(2L), Scalar(false));
        var b = Linear.Model(Scalar(3L), Scalar(false));
        Rng.Pin(b, a);                                    // recorded now — before the nested build
        var w = PinWipeFreshInit.Init([Scalar(4L)]);      // FIRST use: nested initializer body build
        return a.Call(x).Concat(-1L, b.Call(x)) + w.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }
}

/// <summary>Mixes positional and sparse pins in ONE scope (the module body): must fail the build.</summary>
[Module]
public partial class PinMixedFormsOneScope
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var a = Linear.Model(Scalar(2L), Scalar(false));
        var b = Linear.Model(Scalar(3L), Scalar(false));
        Rng.Pin(a);
        Rng.Pin(([1], b));
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
    public void TestSparsePinReservesItsSlotAndUnlistedConsumersFillFreeSlots()
    {
        // Pin(([2], a)) RESERVES slot 2 for a (out=2); unlisted b (out=3) fills the first
        // FREE slot, 1. Note that b MOVED — unpinned it sat at slot 2 — so a sparse pin
        // that relocates an item displaces (re-keys) the unlisted consumer whose slot it
        // takes. Only pinning items to their CURRENT slots (the skeleton workflow) leaves
        // unlisted consumers untouched; a relocation must list every consumer it disturbs.
        // The docs state this reservation rule — this test pins the displacement behavior.
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

        // The skeleton groups by scope (here just the module body) and lists each consumer's
        // local slot with the variable left as ?. The two Linears are top-level slots 1 and 2.
        var skeleton = report.EmitPinSkeleton();
        Assert.Contains("// at the end of Inline:", skeleton);
        Assert.Contains("Rng.Pin(", skeleton);
        Assert.Contains("([1], /*", skeleton);
        Assert.Contains("([2], /*", skeleton);
        Assert.Contains("*/ ?)", skeleton);

        // Without a config, streams are listed but unkeyed.
        Assert.All(arch.GetRngStreamReport().Streams, s => Assert.Null(s.KeyWords));
    }

    [Fact]
    public void TestRngStreamReportShowsRealizedLoopFeedStreams()
    {
        var g = (FastComputationGraph)typeof(RngRuntimeLoopFeed)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var x = TensorData([8L], new float[8]);
        var steps = TensorData(System.Array.Empty<long>(), 2L);
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([x, steps]));

        var cfg = new RngConfig { MasterSeed = 3 };
        var report = arch.GetRngStreamReport(cfg);

        // Three streams: the generator's injected drawBase counter state (RngExecutionCounter —
        // a draw-free zero fill, but it occupies an id slot, so the inventory lists it), plus
        // one REALIZED stream per hinted loop iteration of the feed site [1, -1, 1]. Realized
        // ids are static and carry the exact per-stream key (no -1 placeholders).
        Assert.Equal(3, report.Streams.Count);
        Assert.Contains(report.Streams, s =>
            s.Kind == RngStreamKind.ParamInit && s.Name!.Contains("RngExecutionCounter"));
        var feeds = report.Streams.Where(s => s.Kind == RngStreamKind.UniformFeed).ToList();
        Assert.Equal(2, feeds.Count);
        for (int i = 0; i < 2; i++)
        {
            Assert.Equal([1, i, 1], feeds[i].ModelIdPath);
            Assert.Equal([1, -1, 1], feeds[i].SitePath);
            // Exact stream key: fold(fold(fold(runMaster, 1), i), 1) — identical to the
            // key-table row the lowering emits for iteration i.
            var (k0, k1) = RngConfig.FoldKey(RngConfig.FoldKey(cfg.FoldRunKey([1]), i), 1);
            Assert.Equal([k0, k1], feeds[i].KeyWords);
        }

        // The skeleton still groups the feed under its loop SCOPE [1, -1] with the feed's
        // local slot (1) — pins address sites, not iterations — and lists it once.
        var skeleton = arch.GetRngStreamReport().EmitPinSkeleton();
        Assert.Contains("// inside the loop body at ModelId path [1, -1]:", skeleton);
        Assert.Contains("([1], /* uniform feed */ ?)", skeleton);
    }

    [Fact]
    public void TestPinSkeletonGroupsInLoopParamsLikeFeeds()
    {
        // A param AND a feed inside ONE runtime loop: both consumer kinds carry the same
        // site identity ([1, -1, localSlot]) and the skeleton groups both under the
        // loop-body scope at their local slots. Previously an in-loop param — whose
        // realized ModelId carries no -1 — was mis-slotted to module scope under the
        // loop's own slot (an unusable handle) and its sibling iterations were dropped.
        var g = (FastComputationGraph)typeof(RngRuntimeLoopParamAndFeed)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var x = TensorData([8L], new float[8]);
        var steps = TensorData(System.Array.Empty<long>(), 2L);
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([x, steps]));

        var report = arch.GetRngStreamReport();

        // Realized in-loop param rows carry their site id exactly like realized feed rows.
        var paramRows = report.Streams
            .Where(s => s.Kind == RngStreamKind.ParamInit && !s.FrameworkOwned).ToList();
        Assert.Equal(2, paramRows.Count);
        for (int i = 0; i < 2; i++)
        {
            Assert.Equal([1, i, 1], paramRows[i].ModelIdPath);
            Assert.Equal([1, -1, 1], paramRows[i].SitePath);
        }

        // One pin per scope: the loop body lists the param (local slot 1) and the feed
        // (local slot 2) once each. Module scope has no author-pinnable consumer — the
        // framework-owned RngExecutionCounter is excluded — so no module block is emitted.
        var skeleton = report.EmitPinSkeleton();
        Assert.Contains("// inside the loop body at ModelId path [1, -1]:", skeleton);
        Assert.Contains("([1], /*", skeleton);
        Assert.Contains("InitSimple", skeleton);
        Assert.Contains("([2], /* uniform feed */ ?)", skeleton);
        Assert.DoesNotContain("// at the end of Inline:", skeleton);
        Assert.DoesNotContain("RngExecutionCounter", skeleton);
    }

    [Fact]
    public void TestPinSurvivesNestedFirstUseModuleBuild()
    {
        // Pin(b, a) is recorded, then the body first-uses PinWipeFreshInit — whose Function is
        // uncached (it is referenced nowhere else), so its body graph builds mid-trace. The pin
        // must survive that nested build: b (out=3) takes id slot 1. Before the fix, the nested
        // build's entry-time clear wiped the recorded pins and creation order won (a first).
        var (firstSlotOutFeatures, output) = Probe<PinSurvivesNestedFirstUseBuild>();
        Assert.Equal(3L, firstSlotOutFeatures);
        Assert.Equal(5, output.Length);
    }

    [Fact]
    public void TestMixedFormPinsInOneScopeFailTheModuleBuild()
    {
        // In one scope, sparse reservations shift positional pins off the first id slots
        // (they take the first UNRESERVED slots), silently re-keying the streams the
        // positional pin froze — so a scope pinned both ways is rejected at build. Different
        // scopes may still use different forms (covered by SiblingNestedLoopsPin).
        var ex = Assert.ThrowsAny<Exception>(() =>
            _ = typeof(PinMixedFormsOneScope).GetProperty("ComputationGraph")!.GetValue(null));
        for (Exception? e = ex; e is not null; e = e.InnerException)
            if (e.Message.Contains("cannot be mixed within one scope")) return;
        Assert.Fail($"expected the mixed-form Rng.Pin build error, got: {ex}");
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

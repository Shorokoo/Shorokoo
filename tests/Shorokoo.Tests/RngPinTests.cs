using System;
using System.Linq;
using System.Text;
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
/// Pins recorded BEFORE a nested first-use build must survive it: building a not-yet-cached
/// sub-module/initializer mid-trace re-enters the graph builder on the same thread, and its
/// entry-time pin clearing used to wipe the outer body's already-recorded pins.
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
/// streams no longer depend on creation position. Verified structurally — the out-features of
/// the param at id path [1, 1] flip from a's (2) to b's (3) under Pin(b, a) — and behaviorally:
/// the pinned module still executes. Plus the RNG stream report / pin skeleton those pins are
/// authored against, and the loud failures a pin that could never apply must produce.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngPinTests
{
    private static (long firstParamOutFeatures, float[] output) Probe<TModule>()
    {
        var g = ((ComputationGraph)typeof(TModule)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var input = TensorData([1L, 4L], 0.1f, 0.2f, 0.3f, 0.4f);
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([input]));

        // The weight of the FIRST-id Linear ([1, 1] = sub-model 1's param 1) has shape [out, in].
        var firstWeight = arch.GetConcreteModelParamInfos().ParamInfos
            .Single(i => i.ModelId.Vals.SequenceEqual((int[])[1, 1]));

        var output = ComputeContext.Default.Execute(arch.ToConcreteModel(RngConfig.Default), input)[0]
            .ToTensorData().As<float32>().AccessMemory().ToArray();
        return (firstWeight.Shape.Dims[0], output);
    }

    private static InternalComputationGraph Arch<TModule>(params TensorData[] inputs)
    {
        var g = ((ComputationGraph)typeof(TModule)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        return g.ToConcreteArchitecture(g.FromOrderedInputs([.. inputs]));
    }

    private static string AllMessages(Exception ex)
    {
        var sb = new StringBuilder();
        for (Exception? e = ex; e is not null; e = e.InnerException) sb.AppendLine(e.Message);
        return sb.ToString();
    }

    private static void AssertFailsAnyWith(Action act, string fragment)
        => Assert.Contains(fragment, AllMessages(Assert.ThrowsAny<Exception>(act)));

    private static void AssertFailsWith<TException>(Action act, params string[] fragments)
        where TException : Exception
    {
        var msg = AllMessages(Assert.Throws<TException>(act));
        foreach (var f in fragments) Assert.Contains(f, msg);
    }

    [Fact]
    public void TestPinReordersIdAssignmentReservesSparseSlotsAndSurvivesANestedFirstUseBuild()
    {
        // Baseline: a (out=2) was created first -> id [1]. Pin(b, a) gives b (out=3) that slot.
        var (baselineFirst, baselineOut) = Probe<PinBaselineTwoLinears>();
        var (pinnedFirst, pinnedOut) = Probe<PinSwappedTwoLinears>();
        Assert.Equal(2L, baselineFirst);
        Assert.Equal(3L, pinnedFirst);
        Assert.Equal(5, baselineOut.Length);   // dataflow untouched: both produce [1, 5]
        Assert.Equal(5, pinnedOut.Length);

        // Pin(([2], a)) RESERVES slot 2 for a (out=2); unlisted b (out=3) fills the first FREE
        // slot, 1 — so a sparse pin that RELOCATES an item displaces (re-keys) the unlisted
        // consumer whose slot it takes.
        var (sparseFirst, sparseOut) = Probe<PinSparseTwoLinears>();
        Assert.Equal(3L, sparseFirst);
        Assert.Equal(5, sparseOut.Length);

        // Pin(b, a) recorded, then a first-use of an uncached initializer builds a body graph
        // mid-trace: the pin must survive that nested build.
        var (nestedFirst, nestedOut) = Probe<PinSurvivesNestedFirstUseBuild>();
        Assert.Equal(3L, nestedFirst);
        Assert.Equal(5, nestedOut.Length);
    }

    [Fact]
    public void TestRngStreamReportDescribesStreamsAndResolvesInitFeedAndOverriddenKeys()
    {
        var input = TensorData([1L, 4L], 0.1f, 0.2f, 0.3f, 0.4f);
        var arch = Arch<PinBaselineTwoLinears>(input);
        var cfg = new RngConfig { MasterSeed = 3 };
        var report = arch.GetRngStreamReport(cfg);

        // Two Linears, one weight each: two init streams at [1, 1] and [2, 1], named, shaped,
        // and keyed distinctly under the config.
        var inits = report.Streams.Where(s => s.Kind == RngStreamKind.ParamInit).ToList();
        Assert.Equal(2, inits.Count);
        Assert.Equal([1, 1], inits[0].ModelIdPath);
        Assert.Equal([2, 1], inits[1].ModelIdPath);
        Assert.All(inits, s => Assert.Contains("Linear", s.Name));
        Assert.All(inits, s => Assert.NotNull(s.Shape));
        Assert.NotEqual(inits[0].Key, inits[1].Key);
        // The reported keys are resolved by EXECUTING each stream's in-graph derivation (#136),
        // so pin them against the independent host oracle.
        foreach (var s in inits) Assert.Equal(RngTestOracle.InitKey(cfg, s.ModelIdPath), s.Key);

        // The skeleton groups by scope and lists each consumer's local slot, variable left as ?.
        var skeleton = report.EmitPinSkeleton();
        Assert.Contains("// at the end of Inline:", skeleton);
        Assert.Contains("Rng.Pin(", skeleton);
        Assert.Contains("([1], /*", skeleton);
        Assert.Contains("([2], /*", skeleton);
        Assert.Contains("*/ ?)", skeleton);

        // Without a config, streams are listed but unkeyed.
        Assert.All(arch.GetRngStreamReport().Streams, s => Assert.Null(s.Key));

        // Overriding the FIRST param stream only: its row resolves directly (empty fold path)
        // while the second still needs a folded chain, so the resolver's direct/pending
        // partition and its remap back to the original order are exercised too.
        var ovCfg = cfg.Override(RngCollection.Params, [1, 1], 4242UL);
        var ovInits = Arch<PinBaselineTwoLinears>(input).GetRngStreamReport(ovCfg).Streams
            .Where(s => s.Kind == RngStreamKind.ParamInit).ToList();
        Assert.Equal(2, ovInits.Count);
        Assert.Equal(4242UL, ovInits[0].Key);   // the override seed itself — no fold applied
        Assert.Equal(RngTestOracle.InitKey(ovCfg, ovInits[1].ModelIdPath), ovInits[1].Key);
        Assert.NotEqual(ovInits[0].Key, ovInits[1].Key);

        // A realized (non-loop) runtime feed: its row carries a key resolved through RunKeySpec
        // plus the executed split chain.
        var feedCfg = new RngConfig { MasterSeed = 7 };
        var feedArch = Arch<RtLoweredUniform>(TensorData([4L, 4L], Enumerable.Repeat(0f, 16).ToArray()));
        var feed = Assert.Single(feedArch.GetRngStreamReport(feedCfg).Streams
            .Where(s => s.Kind == RngStreamKind.UniformFeed));
        Assert.NotNull(feed.Key);
        Assert.Equal(RngTestOracle.RunKey(feedCfg, feed.ModelIdPath), feed.Key);
    }

    [Fact]
    public void TestPinSkeletonGroupsLoopScopesForFeedsAndInLoopParamsAlike()
    {
        var x = TensorData([8L], new float[8]);
        var steps = TensorData(Array.Empty<long>(), 2L);

        // Two streams: the injected substreamIndex counter state (RngExecutionCounter — a
        // draw-free zero fill that still occupies an id slot) plus ONE row for the feed site
        // [1, -1, 1]. The -1 iteration slot stays: per-iteration streams derive at runtime from
        // the iteration index, so the realized set is unbounded and no enumeration exists.
        var feedArch = Arch<RngRuntimeLoopFeed>(x, steps);
        var report = feedArch.GetRngStreamReport(new RngConfig { MasterSeed = 3 });
        Assert.Equal(2, report.Streams.Count);
        Assert.Contains(report.Streams, s =>
            s.Kind == RngStreamKind.ParamInit && s.Name!.Contains("RngExecutionCounter"));
        var feed = Assert.Single(report.Streams, s => s.Kind == RngStreamKind.UniformFeed);
        Assert.Equal([1, -1, 1], feed.ModelIdPath);
        Assert.Null(feed.SitePath);   // the site row is its own site
        Assert.Null(feed.Key);        // per-iteration keys are runtime-derived, not listable

        // The skeleton groups the feed under its loop SCOPE [1, -1] at its local slot — pins
        // address sites, not iterations — and lists it once.
        var feedSkeleton = feedArch.GetRngStreamReport().EmitPinSkeleton();
        Assert.Contains("// inside the loop body at ModelId path [1, -1]:", feedSkeleton);
        Assert.Contains("([1], /* uniform feed */ ?)", feedSkeleton);

        // A param AND a feed inside ONE runtime loop: both consumer kinds carry the same site
        // identity ([1, -1, localSlot]) and group under the loop-body scope at their local
        // slots. An in-loop param used to be mis-slotted to module scope under the loop's own
        // slot (an unusable handle) with its sibling iterations dropped.
        var bothReport = Arch<RngRuntimeLoopParamAndFeed>(x, steps).GetRngStreamReport();
        var paramRows = bothReport.Streams
            .Where(s => s.Kind == RngStreamKind.ParamInit && !s.FrameworkOwned).ToList();
        Assert.Equal(2, paramRows.Count);
        for (int i = 0; i < 2; i++)
        {
            Assert.Equal([1, i, 1], paramRows[i].ModelIdPath);
            Assert.Equal([1, -1, 1], paramRows[i].SitePath);
        }

        // Module scope has no author-pinnable consumer — the framework-owned
        // RngExecutionCounter is excluded — so no module block is emitted.
        var bothSkeleton = bothReport.EmitPinSkeleton();
        Assert.Contains("// inside the loop body at ModelId path [1, -1]:", bothSkeleton);
        Assert.Contains("([1], /*", bothSkeleton);
        Assert.Contains("InitSimple", bothSkeleton);
        Assert.Contains("([2], /* uniform feed */ ?)", bothSkeleton);
        Assert.DoesNotContain("// at the end of Inline:", bothSkeleton);
        Assert.DoesNotContain("RngExecutionCounter", bothSkeleton);
    }

    [Fact]
    public void TestPinsThatCouldNeverApplyFailLoudly()
    {
        // Mixed forms in one scope: sparse reservations would shift positional pins off the
        // first id slots, silently re-keying the streams the positional pin froze. (Different
        // scopes may still use different forms — see SiblingNestedLoopsPin.)
        AssertFailsAnyWith(
            () => _ = typeof(PinMixedFormsOneScope).GetProperty("ComputationGraph")!.GetValue(null),
            "cannot be mixed within one scope");
        // Pinning something with no RNG stream (the module input).
        AssertFailsAnyWith(
            () => _ = typeof(PinUnresolvableInput).GetProperty("ComputationGraph")!.GetValue(null),
            "Rng.Pin");
        // ModelId slots are numbered from 1; slot 0 at every level is the RngSeed parameter.
        AssertFailsWith<ArgumentException>(() => Rng.Pin(([0], new object())), "reserved", "RngSeed");
        // Both pin forms need a module build in progress, or they could never be applied.
        AssertFailsWith<InvalidOperationException>(() => Rng.Pin(new object()), "inside a module body");
        AssertFailsWith<InvalidOperationException>(() => Rng.Pin(([1], new object())), "inside a module body");

        // A standalone LoopAPI.Iterate traces in an isolated ModuleBuildContext with no
        // harvester, so a pin recorded there could never be applied either.
        Scalar<int64> counter = Scalar(0L);
        foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
        {
            LoopAPI.Init(counter);
            counter = counter + Scalar(1L);
            AssertFailsWith<InvalidOperationException>(() => Rng.Pin(new object()), "inside a module body");
        }
    }
}

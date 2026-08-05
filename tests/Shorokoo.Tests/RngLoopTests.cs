using System;
using System.Linq;
using System.Text;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.Processors.Fast;
using Shorokoo.Core.Rng;
using Shorokoo.Modules.Initializers;
using Shorokoo.Runtime;
using Shorokoo.Tests.Modules;

namespace Shorokoo.Tests;

/// <summary>
/// Adds <c>steps</c> keyed uniform draws to <c>x</c> inside a RUNTIME loop — the trip count is a
/// graph input, so the loop survives concretization and executes as an ONNX Loop.
/// </summary>
[Module]
public partial class RngRuntimeLoopFeed
{
    public static Tensor<float32> Inline(Tensor<float32> x, Scalar<int64> steps)
    {
        var acc = x;
        foreach (var ctx in LoopAPI.Iterate(steps))
        {
            var u = RandomUniform(x.ShapeTensor(), 0f, 1f);
            acc = acc + u;
            ctx.ContinueWhile(Scalar(true));
        }
        return acc;
    }
}

/// <summary>
/// A trainable param AND a runtime feed inside one RUNTIME loop — the fixture for asserting that
/// the two consumer kinds get identical ModelId treatment. Loop = top slot 1; in the loop body
/// the param takes local slot 1 and the feed local slot 2 (creation order).
/// </summary>
[Module]
public partial class RngRuntimeLoopParamAndFeed
{
    public static Tensor<float32> Inline(Tensor<float32> x, Scalar<int64> steps)
    {
        var acc = x;
        foreach (var ctx in LoopAPI.Iterate(steps))
        {
            var w = InitSimple.Init([Scalar(2L)]);
            var u = RandomUniform(x.ShapeTensor(), 0f, 1f);
            acc = acc + u + w.Reduce(ReduceKind.Sum);
            ctx.ContinueWhile(Scalar(true));
        }
        return acc;
    }
}

/// <summary>
/// An in-loop trainable param whose value DIFFERS per iteration, combined into an
/// ORDER-sensitive recurrence <c>acc = acc*2 + w_i</c>: iteration <c>i</c>'s param is weighted by
/// <c>2^(N-1-i)</c>, so selecting the wrong per-iteration slot — or an empty filler — changes the
/// executed output, which a pure sum could not detect. Multiplying by the exact power of two
/// keeps <c>acc*2</c> rounding-free, so the host recurrence is bit-for-bit (FMA-agnostic).
/// </summary>
[Module]
public partial class RngRuntimeLoopParamRecurrence
{
    public static Tensor<float32> Inline(Tensor<float32> x, Scalar<int64> steps)
    {
        var acc = x;
        foreach (var ctx in LoopAPI.Iterate(steps))
        {
            var w = UniformRange.Init([Scalar(1L)], Scalar(0f), Scalar(1f));
            acc = acc * Scalar(2f) + w.Reduce(ReduceKind.Sum);
            ctx.ContinueWhile(Scalar(true));
        }
        return acc;
    }
}

/// <summary>Same body with a CONSTANT trip count of 2: the loop unrolls at concretization.</summary>
[Module]
public partial class RngUnrolledLoopFeed
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var acc = x;
        foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
        {
            var u = RandomUniform(x.ShapeTensor(), 0f, 1f);
            acc = acc + u;
            ctx.ContinueWhile(Scalar(true));
        }
        return acc;
    }
}

/// <summary>
/// In-loop feed keying via the in-graph derivation chain: a feed under a loop takes the ModelId
/// <c>[loopSlot, -1, feedSlot]</c>, and its key is a split chain rooted at the RngSeed parameter
/// with the <b>runtime iteration index</b> entering as the split counter at the <c>-1</c>
/// position — so iteration <c>i</c> draws from <c>fold(fold(fold(runMaster, loopSlot), i),
/// feedSlot)</c>, bit-exactly reproducible host-side, whether the loop survives to runtime or
/// unrolls into constants.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngLoopTests
{
    private const long N = 8;

    private static readonly float[] XVals = [10f, 20f, 30f, 40f, 50f, 60f, 70f, 80f];

    private static float HostUniform(long e, ulong key)
        => RngTestOracle.DrawUniform(key, substreamIndex: 0, e);

    // The feed's ModelId is [1, -1, 1]: the runtime master folds slot 1, then the iteration
    // index, then the feed's slot under the loop (1).
    private static ulong IterationKey(RngConfig cfg, int i)
        => RngTestOracle.FoldKey(RngTestOracle.FoldKey(RngTestOracle.RunKey(cfg, [1]), (ulong)i), 1);

    /// <summary>x + the per-iteration draws, added in loop order (float order matters).</summary>
    private static float[] HostExpected(RngConfig cfg, int steps, Func<int, ulong>? keyOf = null)
    {
        keyOf ??= i => IterationKey(cfg, i);
        var expected = (float[])XVals.Clone();
        for (int i = 0; i < steps; i++)
            for (long e = 0; e < N; e++)
                expected[e] += HostUniform(e, keyOf(i));
        return expected;
    }

    private static (float[] output, InternalComputationGraph concrete) RunRuntimeLoop(RngConfig cfg, long steps)
    {
        var g = ((ComputationGraph)typeof(RngRuntimeLoopFeed)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var x = TensorData([N], XVals);
        var stepsData = TensorData(Array.Empty<long>(), steps);
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([x, stepsData]))
            .ToConcreteModel(cfg);
        var output = ComputeContext.Default.Execute(concrete, x, stepsData)[0]
            .ToTensorData().As<float32>().AccessMemory().ToArray();
        return (output, concrete);
    }

    private static void AssertFailsWithMessage(Action act, string fragment)
    {
        var sb = new StringBuilder();
        for (Exception? e = Assert.ThrowsAny<Exception>(act); e is not null; e = e.InnerException)
            sb.AppendLine(e.Message);
        Assert.Contains(fragment, sb.ToString());
    }

    [Fact]
    public void TestRuntimeAndUnrolledLoopFeedsDrawTheSamePerIterationStreamsBitExactly()
    {
        var cfg = new RngConfig { MasterSeed = 11 };
        var (output, concrete) = RunRuntimeLoop(cfg, steps: 3);

        // The loop really survived to runtime — otherwise this test proves nothing.
        Assert.Contains(concrete.Nodes, n => n.OpCode == OpCodes.LOOP_OPEN);
        Assert.Equal(HostExpected(cfg, steps: 3), output);

        // Deterministic across executions; re-keyed by a different master.
        Assert.Equal(output, RunRuntimeLoop(cfg, steps: 3).output);
        Assert.NotEqual(output, RunRuntimeLoop(new RngConfig { MasterSeed = 12 }, steps: 3).output);

        // Running FEWER iterations than enumerated draws from exactly the same streams.
        var g = ((ComputationGraph)typeof(RngRuntimeLoopFeed)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var x = TensorData([N], XVals);
        var partial = g.ToConcreteArchitecture(g.FromOrderedInputs([x, TensorData(Array.Empty<long>(), 3L)]))
            .ToConcreteModel(cfg);
        Assert.Equal(HostExpected(cfg, steps: 2),
            ComputeContext.Default.Execute(partial, x, TensorData(Array.Empty<long>(), 2L))[0]
                .ToTensorData().As<float32>().AccessMemory().ToArray());

        // A constant trip count unrolls the loop away by concretization, yet each unrolled copy
        // folds to the same per-iteration key the runtime loop splits at execution.
        var ug = ((ComputationGraph)typeof(RngUnrolledLoopFeed)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var unrolled = ug.ToConcreteArchitecture(ug.FromOrderedInputs([x])).ToConcreteModel(cfg);
        Assert.DoesNotContain(unrolled.Nodes, n => n.OpCode == OpCodes.LOOP_OPEN);
        var unrolledOutput = ComputeContext.Default.Execute(unrolled, x)[0]
            .ToTensorData().As<float32>().AccessMemory().ToArray();
        Assert.Equal(HostExpected(cfg, steps: 2), unrolledOutput);
        Assert.Equal(RunRuntimeLoop(cfg, steps: 2).output, unrolledOutput);
    }

    [Fact]
    public void TestPerIterationOverridesRouteStructurallyAndRebindingRewiresInPlace()
    {
        // Override routing is structural: the site's chain selects the record's key (at its
        // fixed offset in the RngSeedData) when the runtime iteration index matches the
        // record's path, and the folded chain otherwise.
        var cfg = new RngConfig { MasterSeed = 11 };
        var ov = cfg.Override(RngCollection.Runtime, [1, 1, 1], 424242UL);
        Func<int, ulong> ovKeys = i => i == 1 ? RngTestOracle.RunKey(ov, [1, 1, 1]) : IterationKey(ov, i);

        var (output, concrete) = RunRuntimeLoop(ov, steps: 3);
        Assert.Contains(concrete.Nodes, n => n.OpCode == OpCodes.LOOP_OPEN);
        Assert.Equal(HostExpected(ov, steps: 3, ovKeys), output);

        // A re-bind that CHANGES the override set re-runs the wiring pass on the same in-memory
        // model; removing the override restores the master-derived chain bit-exactly.
        var (baseline, plain) = RunRuntimeLoop(cfg, steps: 3);
        var x = TensorData([N], XVals);
        var stepsData = TensorData(Array.Empty<long>(), 3L);
        float[] Run() => ComputeContext.Default.Execute(plain, x, stepsData)[0]
            .ToTensorData().As<float32>().AccessMemory().ToArray();
        plain.ApplyRngConfig(ov);
        Assert.Equal(HostExpected(ov, steps: 3, ovKeys), Run());
        plain.ApplyRngConfig(cfg);
        Assert.Equal(baseline, Run());

        // A loop that executes exactly ONE iteration: the override must still reach the draw (a
        // dispatch bug once dropped it precisely when the key table had a single row).
        var single = cfg.Override(RngCollection.Runtime, [1, 0, 1], 99999UL);
        var (singleOutput, singleConcrete) = RunRuntimeLoop(single, steps: 1);
        Assert.Contains(singleConcrete.Nodes, n => n.OpCode == OpCodes.LOOP_OPEN);
        Assert.Equal(
            HostExpected(single, steps: 1, _ => RngTestOracle.RunKey(single, [1, 0, 1])),
            singleOutput);
    }

    [Fact]
    public void TestZeroTripLoopsRealizeParamsAndFeedsAndPerIterationParamsAreSelectedBitExactly()
    {
        // A trip-count hint of 0 means the loop never runs under the hints. Params — whose
        // values must be enumerated and materialized — realize the single all-zero grid cell as
        // padding. Feeds need no padding: their per-iteration keys derive at runtime.
        var g = ((ComputationGraph)typeof(RngRuntimeLoopParamAndFeed)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var x = TensorData([N], XVals);
        var zeroArch = g.ToConcreteArchitecture(
            g.FromOrderedInputs([x, TensorData(Array.Empty<long>(), 0L)]));

        // Exactly one realized in-loop param, at the padded cell [1, 0, 1] (the other entry is
        // the injected RngExecutionCounter at the next free top slot).
        var paramIds = zeroArch.GetConcreteModelParamInfos().ParamInfos
            .Select(p => p.ModelId.Vals.ToArray()).OrderBy(v => v.Length).ToArray();
        Assert.Equal(2, paramIds.Length);
        Assert.Equal((int[])[2], paramIds[0]);
        Assert.Equal((int[])[1, 0, 1], paramIds[1]);

        // Feed side: the site row [1, -1, 2], iteration slot intact.
        var feedRows = zeroArch.GetRngStreamReport().Streams
            .Where(s => s.Collection == RngCollection.Runtime).ToArray();
        Assert.Single(feedRows);
        Assert.Equal((int[])[1, -1, 2], feedRows[0].ModelIdPath.ToArray());

        // Initialization succeeds and executing the valid iteration count — 0 — draws nothing.
        var zeroConcrete = zeroArch.ToConcreteModel(new RngConfig { MasterSeed = 11 });
        Assert.Equal(XVals, ComputeContext.Default
            .Execute(zeroConcrete, x, TensorData(Array.Empty<long>(), 0L))[0]
            .ToTensorData().As<float32>().AccessMemory().ToArray());

        // The trainable-param analogue of the feed test: the in-loop MODEL_PARAM_ID_REF
        // per-iteration selection path with value-DISCRIMINATING data.
        const int steps = 3;
        var cfg = new RngConfig { MasterSeed = 11 };
        var rg = ((ComputationGraph)typeof(RngRuntimeLoopParamRecurrence)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var stepsData = TensorData(Array.Empty<long>(), (long)steps);
        var arch = rg.ToConcreteArchitecture(rg.FromOrderedInputs([x, stepsData]));

        // The realized in-loop params, ORDERED BY ITERATION SLOT, read straight from the init
        // draw — INDEPENDENTLY of the in-graph selection under test. An in-loop param takes a
        // 3-slot ModelId [loopSlot, iterationIndex, paramSlot].
        var perIter = FastInitializeModelParams.Process(
                arch, ComputeContext.Default, cfg, arch.GetConcreteModelParamInfos())
            .Where(kv => kv.Key.Vals.Length == 3)
            .OrderBy(kv => kv.Key.Vals[1])
            .Select(kv => kv.Value.As<float32>().AccessMemory().ToArray()[0])
            .ToArray();
        Assert.Equal(steps, perIter.Length);
        Assert.Equal(perIter.Length, perIter.Distinct().Count());

        var expected = (float[])XVals.Clone();
        foreach (var w in perIter)
            for (long e = 0; e < N; e++)
                expected[e] = expected[e] * 2f + w;

        var concrete = arch.ToConcreteModel(cfg);
        Assert.Contains(concrete.Nodes, n => n.OpCode == OpCodes.LOOP_OPEN);
        Assert.Equal(expected, ComputeContext.Default.Execute(concrete, x, stepsData)[0]
            .ToTensorData().As<float32>().AccessMemory().ToArray());
    }

    [Fact]
    public void TestMalformedIterationVectorsAndUnmatchedOverridesFailLoudly()
    {
        // A feed site with more iteration slots than its iteration-indices input supplies is a
        // corrupted stream identity, not a zero-trip loop: wiring it anyway would derive keys
        // from indices that never exist at runtime. Concretization itself must throw.
        var g = ((ComputationGraph)typeof(RngRuntimeLoopFeed)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var feed = g.Nodes.Single(n => n.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM);
        var idVals = feed.Attributes.GetIntsVal(OnnxOpAttributeNames.ShrkAttrLocalModelId)!;
        Assert.Contains(-1, idVals);   // guard: the fixture is an in-loop feed as expected
        feed.Attributes = feed.Attributes.SetAttributes(
            (OnnxOpAttributeNames.ShrkAttrLocalModelId,
             (long[])[.. idVals[..^1], -1L, idVals[^1]]));

        var x = TensorData([N], XVals);
        var steps = TensorData(Array.Empty<long>(), 2L);
        AssertFailsWithMessage(
            () => g.ToConcreteArchitecture(g.FromOrderedInputs([x, steps])), "iteration slot");

        // An override that matches no stream of the graph must fail the bind loudly.
        var unmatched = new RngConfig { MasterSeed = 11 }.Override(RngCollection.Runtime, [9, 9, 9], 1UL);
        AssertFailsWithMessage(
            () => RunRuntimeLoop(unmatched, steps: 2), "matches no runtime stream");
    }
}

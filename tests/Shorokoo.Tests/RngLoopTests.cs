using System;
using System.Linq;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Rng;
using Shorokoo.Runtime;
using Shorokoo.Tests.Modules;

namespace Shorokoo.Tests;

/// <summary>
/// Adds <c>steps</c> keyed uniform draws to <c>x</c> inside a RUNTIME loop — the trip count
/// is a graph input, so the loop survives concretization and executes as an ONNX Loop.
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
/// A trainable param AND a runtime feed inside one RUNTIME loop — the fixture for asserting
/// that the two consumer kinds get identical ModelId treatment (realization, padding,
/// reporting, pin-skeleton grouping). Loop = top slot 1; in the loop body the param takes
/// local slot 1 and the feed local slot 2 (creation order).
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
/// In-loop feed keying via the enumerated per-stream key table: a feed under a loop takes the
/// ModelId <c>[loopSlot, -1, feedSlot]</c>; its per-iteration streams are enumerated at
/// concretization (the concreteness contract — static stream set), and at ONNX prep the feed
/// draws from a dense key table with the row selected by the runtime iteration index — so
/// iteration <c>i</c> draws from the stream
/// <c>fold(fold(fold(runMaster, loopSlot), i), feedSlot)</c>, bit-exactly reproducible
/// host-side, whether the loop survives to runtime or unrolls into constants.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngLoopTests
{
    private const long N = 8;

    private static readonly float[] XVals = [10f, 20f, 30f, 40f, 50f, 60f, 70f, 80f];

    /// <summary>Host replica of one keyed uniform draw: element e -> counter (e, drawBase=0).</summary>
    private static float HostUniform(long e, (uint k0, uint k1) key)
    {
        var (x0, _) = Threefry2x32.Bijection((uint)e, 0u, key.k0, key.k1);
        return (x0 & 0x00FFFFFFu) * (1.0f / 16777216.0f);
    }

    /// <summary>x + sum of per-iteration draws, added in loop order (float order matters).</summary>
    private static float[] HostExpected(RngConfig cfg, int steps)
    {
        var expected = (float[])XVals.Clone();
        for (int i = 0; i < steps; i++)
        {
            // Feed ModelId is [1, -1, 1]: the runtime master folds slot 1, then the
            // iteration index, then the feed's slot under the loop (1).
            var key = RngConfig.FoldKey(RngConfig.FoldKey(cfg.FoldRunKey([1]), i), 1);
            for (long e = 0; e < N; e++)
                expected[e] += HostUniform(e, key);
        }
        return expected;
    }

    private static (float[] output, FastComputationGraph concrete) RunRuntimeLoop(RngConfig cfg, long steps)
    {
        var g = (FastComputationGraph)typeof(RngRuntimeLoopFeed)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var x = TensorData([N], XVals);
        var stepsData = TensorData(Array.Empty<long>(), steps);
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([x, stepsData]))
            .ToConcreteModel(cfg);
        var output = ComputeContext.Default.Execute(concrete, x, stepsData)[0]
            .ToTensorData().As<float32>().AccessMemory().ToArray();
        return (output, concrete);
    }

    [Fact]
    public void TestRuntimeLoopFeedDrawsPerIterationStreamsBitExactly()
    {
        var cfg = new RngConfig { MasterSeed = 11 };
        var (output, concrete) = RunRuntimeLoop(cfg, steps: 3);

        // The loop really survived to runtime — otherwise this test proves nothing.
        Assert.Contains(concrete.Nodes, n => n.OpCode == OpCodes.LOOP_OPEN);

        // Every iteration drew from its own stream, and each matches the host fold exactly.
        Assert.Equal(HostExpected(cfg, steps: 3), output);

        // Deterministic across executions; re-keyed by a different master.
        var (again, _) = RunRuntimeLoop(cfg, steps: 3);
        Assert.Equal(output, again);
        var (other, _) = RunRuntimeLoop(new RngConfig { MasterSeed = 12 }, steps: 3);
        Assert.NotEqual(output, other);
    }

    [Fact]
    public void TestUnrolledLoopFoldsToSameKeysAsRuntimeLoop()
    {
        var cfg = new RngConfig { MasterSeed = 11 };

        var g = (FastComputationGraph)typeof(RngUnrolledLoopFeed)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var x = TensorData([N], XVals);
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([x])).ToConcreteModel(cfg);

        // Constant trip count: the loop is gone by concretization…
        Assert.DoesNotContain(concrete.Nodes, n => n.OpCode == OpCodes.LOOP_OPEN);

        // …yet each unrolled copy folds to the same per-iteration key the runtime loop
        // splits at execution: graceful degradation in both directions, bit-for-bit.
        var output = ComputeContext.Default.Execute(concrete, x)[0]
            .ToTensorData().As<float32>().AccessMemory().ToArray();
        Assert.Equal(HostExpected(cfg, steps: 2), output);

        var (runtimeOutput, _) = RunRuntimeLoop(cfg, steps: 2);
        Assert.Equal(runtimeOutput, output);
    }

    [Fact]
    public void TestPerIterationOverrideReSeedsExactlyOneStream()
    {
        // Override ITERATION 1 of the loop feed site [1, -1, 1] by its realized stream path.
        // The lowering resolves every realized stream's key host-side from the carrier
        // (override-aware) into the per-stream key table it selects from by iteration index.
        var cfg = new RngConfig { MasterSeed = 11 };
        cfg = cfg.Override(RngCollection.Runtime, [1, 1, 1], 424242UL);
        var (output, concrete) = RunRuntimeLoop(cfg, steps: 3);
        Assert.Contains(concrete.Nodes, n => n.OpCode == OpCodes.LOOP_OPEN);

        // Iterations 0 and 2 derive as usual; iteration 1 draws from the override — bit-exact.
        var expected = (float[])XVals.Clone();
        for (int i = 0; i < 3; i++)
        {
            var key = i == 1
                ? cfg.FoldRunKey([1, 1, 1])
                : RngConfig.FoldKey(RngConfig.FoldKey(cfg.FoldRunKey([1]), i), 1);
            for (long e = 0; e < N; e++)
                expected[e] += HostUniform(e, key);
        }
        Assert.Equal(expected, output);
    }

    [Fact]
    public void TestSingleIterationLoopOverrideIsHonored()
    {
        // A loop whose enumerated iteration space has exactly ONE stream: the override must
        // reach the draw (regression guard — a representation-dispatch bug once dropped the
        // override precisely when the key table had a single row).
        var cfg = new RngConfig { MasterSeed = 11 };
        cfg = cfg.Override(RngCollection.Runtime, [1, 0, 1], 99999UL);
        var (output, concrete) = RunRuntimeLoop(cfg, steps: 1);
        Assert.Contains(concrete.Nodes, n => n.OpCode == OpCodes.LOOP_OPEN);

        var expected = (float[])XVals.Clone();
        var key = cfg.FoldRunKey([1, 0, 1]);   // the override, not the master derivation
        for (long e = 0; e < N; e++)
            expected[e] += HostUniform(e, key);
        Assert.Equal(expected, output);
    }

    [Fact]
    public void TestExecutingFewerIterationsThanEnumeratedStaysExact()
    {
        // The concreteness contract: the enumerated iteration space is the stream set, and
        // running FEWER iterations than enumerated is valid use — the executed subset draws
        // from exactly the same per-iteration streams. (Running MORE would mint stream ids
        // that did not exist at concretization — invalid use of the concrete artifact.)
        var cfg = new RngConfig { MasterSeed = 11 };
        var g = (FastComputationGraph)typeof(RngRuntimeLoopFeed)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var x = TensorData([N], XVals);
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([x, TensorData(Array.Empty<long>(), 3L)]))
            .ToConcreteModel(cfg);

        var output = ComputeContext.Default.Execute(concrete, x, TensorData(Array.Empty<long>(), 2L))[0]
            .ToTensorData().As<float32>().AccessMemory().ToArray();
        Assert.Equal(HostExpected(cfg, steps: 2), output);
    }

    [Fact]
    public void TestZeroEnumeratedIterationsPadOneCellForParamsAndFeedsAlike()
    {
        // Concretizing with a trip-count hint of 0 means the loop never runs under the hints,
        // yet the concreteness contract requires a static, non-empty consumer set. BOTH
        // ModelId-based consumer kinds resolve this the same way: the single all-zero grid
        // cell is realized as padding — validly derived, never consumed when the executed
        // iteration count is 0 (the only valid use of this concrete artifact).
        var g = (FastComputationGraph)typeof(RngRuntimeLoopParamAndFeed)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var x = TensorData([N], XVals);
        var arch = g.ToConcreteArchitecture(
            g.FromOrderedInputs([x, TensorData(Array.Empty<long>(), 0L)]));

        // Param side: exactly one realized in-loop param, at the padded cell [1, 0, 1]
        // (the second entry is the injected RngExecutionCounter at the next free top slot).
        var paramIds = arch.GetConcreteModelParamInfos().ParamInfos
            .Select(p => p.ModelId.Vals.ToArray()).OrderBy(v => v.Length).ToArray();
        Assert.Equal(2, paramIds.Length);
        Assert.Equal((int[])[2], paramIds[0]);
        Assert.Equal((int[])[1, 0, 1], paramIds[1]);

        // Feed side: exactly one realized runtime stream, at the padded cell [1, 0, 2].
        var feedRows = arch.GetRngStreamReport().Streams
            .Where(s => s.Collection == RngCollection.Runtime).ToArray();
        Assert.Single(feedRows);
        Assert.Equal((int[])[1, 0, 2], feedRows[0].ModelIdPath.ToArray());

        // Initialization succeeds (the padded param materializes like any other) and
        // executing the valid iteration count — 0 — consumes neither padding cell.
        var concrete = arch.ToConcreteModel(new RngConfig { MasterSeed = 11 });
        var output = ComputeContext.Default.Execute(concrete, x, TensorData(Array.Empty<long>(), 0L))[0]
            .ToTensorData().As<float32>().AccessMemory().ToArray();
        Assert.Equal(XVals, output);
    }

    [Fact]
    public void TestUnmatchedRuntimeOverrideFailsTheBind()
    {
        // An override that matches no stream of the graph must fail the bind loudly — a
        // silently inactive override is exactly the re-keying hazard explicit seeding
        // exists to prevent.
        var cfg = new RngConfig { MasterSeed = 11 };
        cfg = cfg.Override(RngCollection.Runtime, [9, 9, 9], 1UL);
        var ex = Assert.ThrowsAny<System.Exception>(() => RunRuntimeLoop(cfg, steps: 2));
        for (System.Exception? e = ex; e is not null; e = e.InnerException)
            if (e.Message.Contains("matches no runtime stream")) return;
        Assert.Fail($"expected the unmatched-override bind error, got: {ex}");
    }
}

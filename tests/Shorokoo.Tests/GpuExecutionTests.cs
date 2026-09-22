using System.Runtime.CompilerServices;
using Microsoft.ML.OnnxRuntime;
using Shorokoo.Core.Backends;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// What the CUDA execution provider actually does, run rather than read: a graph on the card, the
/// device-memory budget and the shrinking arena a resident training loop needs, and the memory
/// statistics and node placement of
/// <see href="https://github.com/Shorokoo/Shorokoo/issues/377">Shorokoo/Shorokoo#377</see>, every
/// CUDA path of which had been reviewed on a machine with no card.
///
/// <para>Excluded from the coverage suite. These need the process-wide default backend to be a
/// GPU one, which the test project deploys under <c>-p:ShorokooGpuTests=true</c>; without it
/// <see cref="CudaFactAttribute"/> skips each of them and says so rather than failing on a box
/// with no card. Run with <c>--filter "Purpose=Hardware"</c>, and with that switch and nothing
/// else — it changes the backend every test in the process discovers.</para>
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Hardware")]
[Collection(DeviceMemoryPeak.Name)]
public class GpuExecutionTests
{
    /// <summary>
    /// Explicitly exercises the CUDA execution provider.
    /// </summary>
    [CudaFact]
    public void CudaProvider_AddTwoFloat32Scalars_ReturnsCorrectSum()
    {
        // [CudaFact] already gates on a GPU backend being loaded; confirm the
        // bound factory really is a CUDA one rather than a silent CPU fallback.
        var backend = DefaultBackend.Instance.GetType().Assembly.GetName().Name ?? "";
        Assert.EndsWith("GPU", backend);

        var result = AddTwoScalars(new ComputeContext(), 2.0f, 3.0f);
        Assert.Equal(5.0f, result);
    }

    /// <summary>
    /// The device-memory surface end to end, on both settings of every knob: the shipped
    /// defaults build and run a session, and so does a budgeted power-of-two arena with per-run
    /// shrinkage on, with the card reporting a reading either way. Each configuration is a
    /// context of its own, so the two legs cannot reach each other and neither leaves anything
    /// behind for the next test.
    /// </summary>
    [CudaFact]
    public void CudaProvider_RunsUnderEveryDeviceMemoryConfigurationAndReportsTheCardsUsage()
    {
        DeviceMemory.ResetPeak();
        try
        {
            Assert.Equal(5.0f, AddTwoScalars(new ComputeContext(), 2.0f, 3.0f));

            var budgeted = new ComputeContext
            {
                DeviceMemory = new DeviceMemorySettings
                {
                    LimitBytes = 2L * 1024 * 1024 * 1024,
                    ArenaExtend = ArenaExtendStrategy.NextPowerOfTwo,
                },
                RunSettings = new RunSettings { ShrinkArenaAfterRun = true },
            };
            Assert.Equal(5.0f, AddTwoScalars(budgeted, 2.0f, 3.0f));

            var reading = DeviceMemory.Sample();
            Assert.NotNull(reading);
            Assert.True(reading!.Value.TotalBytes > 0);
            Assert.True(reading.Value.UsedBytes > 0);
            Assert.Equal(reading.Value.UsedBytes, DeviceMemory.PeakUsedBytes);
        }
        finally
        {
            DeviceMemory.ResetPeak();
        }
    }

    /// <summary>
    /// A tensor moved onto the card is allocated under the owning context's budget, so one that
    /// does not fit fails instead of taking the rest of the device — and what it is holding is
    /// reported against that budget rather than against a session's.
    /// </summary>
    [CudaFact]
    public void CudaProvider_ATensorMovedOntoTheCardIsBoundedByItsContextsDeviceMemoryBudget()
    {
        const long limit = 64L * 1024 * 1024;
        using var budgeted = new ComputeContext
        {
            DeviceMemory = new DeviceMemorySettings { LimitBytes = limit },
        };
        using var uncapped = new ComputeContext();

        Assert.Null(budgeted.ReadTransferArenaStatistics());

        var fits = TensorData([4L * 1024 * 1024], new float[4 * 1024 * 1024]).CopyTo(budgeted);
        Assert.False(fits.IsHostResident);

        var arena = budgeted.ReadTransferArenaStatistics();
        Assert.NotNull(arena);
        Assert.Equal(limit, arena!.Value.LimitBytes);
        Assert.True(arena.Value.InUseBytes >= 16L * 1024 * 1024);

        var tooBig = TensorData([32L * 1024 * 1024], new float[32 * 1024 * 1024]);
        var refused = Assert.ThrowsAny<OnnxRuntimeException>(() => tooBig.CopyTo(budgeted));
        Assert.Contains("BFCArena", refused.Message);

        // The same tensor fits a context that named no ceiling, so what refused it was the budget
        // and not the card.
        var elsewhere = tooBig.CopyTo(uncapped);
        Assert.False(elsewhere.IsHostResident);
        Assert.NotEqual(limit, uncapped.ReadTransferArenaStatistics()!.Value.LimitBytes);
    }

    /// <summary>
    /// The interaction the whole integration turns on, and the one no CPU test can reach: state
    /// left in the provider's own memory across steps, while the arena it lives in is budgeted,
    /// extends by request, and is handed back after every run. The trained result has to be the
    /// same as an ordinary step loop's on the shipped defaults.
    /// </summary>
    [CudaFact]
    public void CudaProvider_AResidentRunTrainsTheSameUnderABudgetedAndShrinkingArena()
    {
        var (input, target) = (TrainingRigHelpers.InBatch(1f, 2f, 3f, 4f),
                               TrainingRigHelpers.TargetBatch(2f, 4f, 6f, 8f));
        var expected = StepLoopWeights(input, target);

        var rig = ScalarRig(new ComputeContext
        {
            DeviceMemory = new DeviceMemorySettings
            {
                LimitBytes = 2L * 1024 * 1024 * 1024,
                ArenaExtend = ArenaExtendStrategy.SameAsRequested,
            },
            RunSettings = new RunSettings { ShrinkArenaAfterRun = true },
        });
        using var run = rig.BeginResidentRun();
        run.Step(input, target);
        run.Step(input, target);
        var published = run.StepToCheckpoint(input, target);

        Assert.Equal(3, published.Step);
        var actual = Weights(published);
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], precision: 4);
    }

    /// <summary>
    /// The nine arena figures off a CUDA allocator, which is what the whole statistics surface
    /// rests on and what no CPU machine can ask for: the allocator is built by name and a box
    /// without a card refuses it, so only the failure path has ever run. And what the arena holds
    /// before the first run — the session's weights, out of this same arena, so run 1's peak is
    /// them plus what the run added and the record has to carry both.
    /// </summary>
    [CudaFact]
    public void CudaProvider_TheDeviceArenaAnswersAndAlreadyHoldsTheWeightsBeforeTheFirstRun()
    {
        using var ctx = new ComputeContext
        {
            Diagnostics = new DiagnosticSettings { CollectRunStatistics = true },
        };
        var compiled = ArenaProbeModels.Weighted(ctx);

        var built = Assert.IsType<ArenaStatistics>(compiled.ReadArenaStatistics());
        Assert.Equal(ArenaProbeModels.WeightBytes, built.MaxInUseBytes);
        Assert.Equal(ArenaProbeModels.WeightBytes, built.InUseBytes);
        Assert.Equal(ArenaProbeModels.WeightBytes, built.MaxAllocSizeBytes);
        Assert.Equal(ArenaProbeModels.WeightBytes, built.TotalAllocatedBytes);
        Assert.Equal(1L, built.AllocationCount);
        Assert.Equal(-1L, built.LimitBytes);
        // The card takes the weight as a block of its own where the host arena reserves it, so
        // these two counts mean different things on the two devices and do not compare.
        Assert.Equal(1L, built.ArenaExtensionCount);
        Assert.Equal(0L, built.ReserveCount);

        compiled.Execute(ArenaProbeModels.WeightedInput());
        var run = Assert.Single(ctx.RunStats.RecentRuns);
        Assert.Equal(ArenaProbeModels.WeightBytes, run.PriorPeakBytes);
        Assert.Equal(MemoryFigureKind.Measured, run.PeakKind);
        Assert.True(run.PeakBytes > run.PriorPeakBytes);
        Assert.True(run.PeakBytes - run.PriorPeakBytes < ArenaProbeModels.WeightBytes / 16);
    }

    /// <summary>
    /// A graph the provider cannot run whole: one output stays on the card and one comes back from
    /// the host, which is what <see cref="SessionOutputPlacement.Mixed"/ > is for, and the crossing
    /// is charged to the pinned host arena rather than to the device one. A graph with no node the
    /// provider can run is <see cref="SessionOutputPlacement.Host"/>, and one it runs whole is
    /// <see cref="SessionOutputPlacement.Device"/>.
    /// </summary>
    [CudaFact]
    public void CudaProvider_OutputPlacementSeparatesADeviceGraphAPartitionedOneAndOneThatFellBack()
    {
        using var ctx = new ComputeContext();

        var partitioned = ArenaProbeModels.Partitioned(ctx);
        Assert.Equal(SessionOutputPlacement.Mixed, partitioned.OutputPlacement);
        Assert.True(partitioned.HasDeviceMemory);

        var pinnedBefore = Assert.IsType<ArenaStatistics>(partitioned.ReadPinnedArenaStatistics());
        Assert.Equal(0L, pinnedBefore.AllocationCount);

        partitioned.Execute(ArenaProbeModels.Square());
        var pinned = Assert.IsType<ArenaStatistics>(partitioned.ReadPinnedArenaStatistics());
        var device = Assert.IsType<ArenaStatistics>(partitioned.ReadArenaStatistics());
        Assert.True(pinned.AllocationCount > 0);
        Assert.True(pinned.MaxInUseBytes > 0);
        Assert.NotEqual(device, pinned);

        var host = ArenaProbeModels.HostOnly(ctx);
        host.Execute(ArenaProbeModels.Square());
        Assert.Equal(SessionOutputPlacement.Host, host.OutputPlacement);
        Assert.False(host.HasDeviceMemory);
        Assert.Equal(0L, Assert.IsType<ArenaStatistics>(host.ReadArenaStatistics()).AllocationCount);

        var onCard = ArenaProbeModels.MatMul(ctx);
        onCard.Execute(ArenaProbeModels.MatMulOperand(8), ArenaProbeModels.MatMulOperand(8));
        Assert.Equal(SessionOutputPlacement.Device, onCard.OutputPlacement);
        Assert.True(onCard.HasDeviceMemory);
    }

    /// <summary>
    /// The node trace of a genuinely partitioned graph: two providers, the fallen-back node named,
    /// and the copy node the runtime inserts at the boundary sitting <i>at</i> the boundary. That
    /// last one is the whole point — the copy carries the highest graph index of the four, so
    /// ordering the trace by index would move exactly the node that marks the fallback to the end,
    /// past the node it feeds.
    /// </summary>
    [CudaFact]
    public void CudaProvider_APartitionedTraceNamesBothProvidersAndKeepsTheCopyAtTheBoundary()
    {
        using var ctx = new ComputeContext
        {
            Diagnostics = new DiagnosticSettings { TraceNodePlacement = true },
        };
        var compiled = ArenaProbeModels.Partitioned(ctx);
        compiled.Execute(ArenaProbeModels.Square());

        var placement = Assert.IsType<NodePlacement>(compiled.ReadNodePlacement());
        Assert.Equal(["CUDAExecutionProvider", "CPUExecutionProvider"],
            placement.Providers.Select(share => share.Provider));

        var onHost = Assert.Single(placement.NodesOn("CPUExecutionProvider"));
        Assert.Equal("Det", onHost.OpType);
        Assert.Equal(placement.Nodes.Count, placement.Providers.Sum(share => share.NodeCount));

        var ran = placement.Nodes.ToList();
        var copy = Assert.Single(ran.Where(node => node.OpType.StartsWith("Memcpy", StringComparison.Ordinal)));
        Assert.Equal("MemcpyToHost", copy.OpType);
        Assert.Equal(ran.Max(node => node.NodeIndex), copy.NodeIndex);
        Assert.True(copy.NodeIndex > onHost.NodeIndex);
        Assert.Equal(ran.IndexOf(onHost) - 1, ran.IndexOf(copy));
        Assert.NotEqual(ran.Count - 1, ran.IndexOf(copy));
    }

    /// <summary>
    /// Shrinkage, asked for and observed: the arena hands blocks back at the end of every run, and
    /// what it is holding afterwards drops below the high-water mark it reached — which is the one
    /// case where <see cref="RunStatistics.ArenaBytes"/> sits under
    /// <see cref="RunStatistics.PeakBytes"/> rather than above it. Nothing had ever seen it
    /// happen; the CPU arena never shrank in testing.
    /// </summary>
    [CudaFact]
    public void CudaProvider_AShrinkingRunHandsBlocksBackAndEndsBelowThePeakItReached()
    {
        static RunStatistics Runs(bool shrink)
        {
            using var ctx = new ComputeContext
            {
                Diagnostics = new DiagnosticSettings { CollectRunStatistics = true },
                RunSettings = new RunSettings { ShrinkArenaAfterRun = shrink },
            };
            var compiled = ArenaProbeModels.MatMul(ctx);
            var operand = ArenaProbeModels.MatMulOperand(512);
            for (int run = 0; run < 3; run++) compiled.Execute(operand, operand);
            return ctx.RunStats;
        }

        var shrinking = Runs(shrink: true);
        Assert.True(shrinking.ArenaShrinkageCount >= 3);
        Assert.True(shrinking.PeakBytes > 0);
        Assert.True(shrinking.ArenaBytes < shrinking.PeakBytes);

        var keeping = Runs(shrink: false);
        Assert.Equal(0L, keeping.ArenaShrinkageCount);
        Assert.Equal(shrinking.PeakBytes, keeping.PeakBytes);
        Assert.True(keeping.ArenaBytes >= keeping.PeakBytes);
        // The blocks figure falls when they go back, so a shrinking context undercounts them.
        Assert.True(keeping.ArenaExtensionCount > shrinking.ArenaExtensionCount);
    }

    /// <summary>
    /// The comparison #198 asked for and no CPU machine can make: what this context's own sessions
    /// took, against what the card reports for every process on it. The session figure is the
    /// smaller of the two and accounts for part of the rise — the rest is the provider's context,
    /// its libraries and whatever else holds the device.
    /// </summary>
    [CudaFact]
    public void CudaProvider_TheContextsRunStatisticsAccountForPartOfWhatTheCardReports()
    {
        DeviceMemory.ResetPeak();
        try
        {
            using var ctx = new ComputeContext
            {
                Diagnostics = new DiagnosticSettings { CollectRunStatistics = true },
            };
            var compiled = ArenaProbeModels.MatMul(ctx);
            var operand = ArenaProbeModels.MatMulOperand(1024);

            var idle = DeviceMemory.Sample();
            Assert.NotNull(idle);
            for (int run = 0; run < 5; run++)
            {
                compiled.Execute(operand, operand);
                DeviceMemory.Sample();
            }

            var stats = ctx.RunStats;
            Assert.Equal(5L, stats.RunCount);
            Assert.True(stats.PeakBytes > 0);
            var device = Assert.IsType<ArenaStatistics>(compiled.ReadArenaStatistics());
            Assert.Equal(stats.PeakBytes, device.MaxInUseBytes);

            var grew = DeviceMemory.PeakUsedBytes - idle!.Value.UsedBytes;
            Assert.True(grew >= stats.PeakBytes);
            Assert.True(DeviceMemory.PeakUsedBytes < idle.Value.TotalBytes);
        }
        finally
        {
            DeviceMemory.ResetPeak();
        }
    }

    /// <summary>
    /// The placement probe hands ONNX Runtime the session as a bare handle, and a compiled graph is
    /// held weakly by its context, so this reads the placement off a graph nothing else holds while
    /// another thread collects and drains finalizers. <b>It means something only in Release</b>:
    /// unoptimized code roots a local to the end of its scope, so the hazard cannot arise in Debug
    /// at all.
    ///
    /// <para>What it pins is the path, not the <c>GC.KeepAlive</c> in it — removing that does not
    /// reproduce a fault, because the probe is reached through a <c>Lazy</c> whose factory closure
    /// holds the session while the factory runs. A change that takes the closure away is what this
    /// would catch.</para>
    /// </summary>
    [CudaFact]
    public void CudaProvider_OutputPlacementReadsFromAGraphHeldOnlyInALocalWhileAnotherThreadCollects()
    {
        using var collecting = new CancellationTokenSource();
        var churn = Task.Run(() =>
        {
            // The finalizers are the hazard, not the collection: a session the collector frees is
            // only released once ~InferenceSession runs. Throttled, because a bare collect loop
            // starves the thread compiling the sessions and buys no extra collections per probe.
            while (!collecting.IsCancellationRequested)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                Thread.Sleep(1);
            }
        });
        try
        {
            for (int round = 0; round < 20; round++)
                Assert.Equal(SessionOutputPlacement.Device, PlacementOfAGraphNothingElseHolds());
        }
        finally
        {
            collecting.Cancel();
            churn.Wait();
        }
    }

    /// <summary>The graph, its context and the session under both are unreachable from the moment
    /// the placement is asked for: every one of them is a local at its last read. Not inlined, so
    /// the caller's frame cannot keep them alive either.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static SessionOutputPlacement PlacementOfAGraphNothingElseHolds()
    {
        var a = InputVector<float32>();
        var b = InputVector<float32>();
        var context = new ComputeContext();
        var compiled = context.Compile(new InternalComputationGraph([a, b], [a * b + a]));
        return compiled.OutputPlacement;
    }

    private static TrainingRig ScalarRig(ComputeContext? runtimeContext = null) => TrainingRig.FromScratch(
        ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, AdamWOptimizer.ComputationGraph,
        [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData([4L], [1f, 2f, 3f, 4f]))],
        new AdamWOptimizerHyperparameters { LearningRate = 0.1f }, runtimeContext: runtimeContext);

    private static float[] StepLoopWeights(TensorDataStruct input, TensorDataStruct target)
    {
        var rig = ScalarRig();
        var ckpt = rig.CreateInitialCheckpoint();
        for (int i = 0; i < 3; i++) ckpt = rig.TrainStep(ckpt, input, target);
        return Weights(ckpt);
    }

    private static float[] Weights(TrainingCheckpoint checkpoint) =>
        TrainingRigHelpers.FlattenStruct(checkpoint.TrainableParams);

    private static float AddTwoScalars(ComputeContext ctx, float left, float right)
    {
        var a = InputScalar<float32>();
        var b = InputScalar<float32>();
        var c = a + b;

        var graph = new InternalComputationGraph([a, b], [c]);
        var results = ctx.Execute(
            (graph),
            TensorData([], left),
            TensorData([], right));

        return results[0].ToTensorData().As<float32>().AccessMemory<float>()[0];
    }
}

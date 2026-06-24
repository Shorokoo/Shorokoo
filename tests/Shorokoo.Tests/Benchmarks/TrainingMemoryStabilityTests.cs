using System.Diagnostics;
using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;
using Shorokoo.Runtime;

namespace Shorokoo.Tests.Benchmarks;

/// <summary>
/// Code-pinned memory-stability gate for the training hot path — the automated
/// half of <c>release-test-plan</c> <c>R-2</c> ("a long-running training loop
/// shows stable memory: no unbounded RSS growth / handle leaks"), the
/// memory-performance check introduced in the
/// <see href="../../../docs/testing/v1.1/release-test-plan.md">v1.1 plan</see>.
/// It drives the same pinned linear scenario as the <c>R-1</c> throughput gate
/// (<see cref="PerfBaselineLinearModel"/>) through thousands of
/// <see cref="TrainingRig.TrainStep"/> calls and asserts the live managed heap
/// does not grow without bound — a per-step reference leak (accumulating
/// checkpoints, tensors, event handlers, undisposed wrappers) would climb
/// roughly linearly with the step count and blow the budget.
///
/// <para>
/// Like the R-1 gate this is deliberately loose so ordinary run-to-run / GC /
/// fresh-container jitter never trips it, while a genuine leak — which grows
/// without bound — still does:
/// </para>
/// <list type="bullet">
/// <item><description>the gate is the <b>live managed heap delta</b> measured
/// with a forced full collection on both ends, so transient per-step garbage is
/// reclaimed and only a true accumulation survives the measurement;</description></item>
/// <item><description>RSS (working set) is measured too, but only as a very
/// generous catastrophe backstop and a diagnostic — the runtime retains
/// committed segments after a GC, so short-window RSS growth is a noisy leak
/// signal and is not the primary gate.</description></item>
/// </list>
/// </summary>
[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class TrainingMemoryStabilityTests
{
    // Pinned scenario geometry — identical to the R-1 throughput gate.
    private static readonly long[] InputShape = [4L, 2L];
    private static readonly long[] TargetShape = [4L, 1L];

    // A long loop preceded by a warm-up so pools / tiered JIT / first-touch
    // allocations have settled before the first measurement. At the steady-state
    // rate the R-1 gate records (several thousand steps/s on this scenario) the
    // whole run is a couple of seconds of CPU.
    private const int WarmupSteps = 1_000;
    private const int MeasuredSteps = 10_000;

    // Budgets. The managed delta of a non-leaking loop is ~0 after a forced
    // collection (the checkpoint is replaced, not accumulated); 16 MiB absorbs
    // JIT / finalizer / fragmentation jitter while a real leak over 10k steps
    // (even ~2 KiB/step) sails past it. The RSS ceiling is a loose
    // pathological-leak backstop only.
    private const long ManagedGrowthBudgetBytes = 16L * 1024 * 1024;
    private const long RssGrowthCeilingBytes = 512L * 1024 * 1024;

    [Fact]
    public void LongTrainingLoopKeepsManagedMemoryStable()
    {
        var baseGraph = PerfBaselineLinearModel.ComputationGraph;
        var exampleInput = TensorData(InputShape, new float[8]);

        var ctx = new ComputeContext();
        var rig = TrainingRig.FromScratch(
            baseGraph, Losses.L2Loss, Optimizers.Adam,
            baseGraph.FromOrderedInputs([exampleInput]),
            new AdamOptimizerHyperparameters { LearningRate = 1e-3f });
        var compiled = ctx.Compile(rig.TrainingStepPureGraph);

        var inputBatch = rig.InputDef.FromOrderedData(
            TensorData(InputShape, new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }));
        var targetBatch = rig.TargetDef.FromOrderedData(
            TensorData(TargetShape, new float[] { 1f, 0f, 1f, 0f }));

        var ckpt = rig.CreateDefaultCheckpoint();
        for (int i = 0; i < WarmupSteps; i++)
            ckpt = rig.TrainStep(ckpt, inputBatch, targetBatch, compiled).Checkpoint;

        long managedBefore = LiveManagedBytes();
        long rssBefore = WorkingSetBytes();

        for (int i = 0; i < MeasuredSteps; i++)
            ckpt = rig.TrainStep(ckpt, inputBatch, targetBatch, compiled).Checkpoint;

        long managedAfter = LiveManagedBytes();
        long rssAfter = WorkingSetBytes();

        long managedGrowth = managedAfter - managedBefore;
        long rssGrowth = rssAfter - rssBefore;

        // Keep the checkpoint reachable past the final measurement so the loop's
        // last result can't be collected before we read the heap.
        Assert.NotNull(ckpt);

        Assert.True(managedGrowth <= ManagedGrowthBudgetBytes,
            $"Training loop leaked managed memory: live heap grew {Mib(managedGrowth)} over " +
            $"{MeasuredSteps} steps ({managedBefore:N0} -> {managedAfter:N0} bytes), exceeding the " +
            $"{Mib(ManagedGrowthBudgetBytes)} budget. A non-leaking loop returns to ~baseline after a " +
            $"forced collection. (RSS moved {Mib(rssGrowth)}.)");

        Assert.True(rssGrowth <= RssGrowthCeilingBytes,
            $"Training loop working set grew {Mib(rssGrowth)} over {MeasuredSteps} steps " +
            $"({rssBefore:N0} -> {rssAfter:N0} bytes), exceeding the {Mib(RssGrowthCeilingBytes)} " +
            $"catastrophe ceiling — likely an unmanaged / handle leak. (Managed heap moved {Mib(managedGrowth)}.)");
    }

    /// <summary>Live managed bytes after a blocking full collection — drops transient garbage.</summary>
    private static long LiveManagedBytes()
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        return GC.GetTotalMemory(forceFullCollection: true);
    }

    private static long WorkingSetBytes()
    {
        using var proc = Process.GetCurrentProcess();
        proc.Refresh();
        return proc.WorkingSet64;
    }

    private static string Mib(long bytes) => $"{bytes / (1024.0 * 1024.0):+0.0;-0.0} MiB";
}

using System.Diagnostics;
using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;
using Shorokoo.Runtime;

namespace Shorokoo.Tests.Benchmarks;

/// <summary>One [1024, 1024] trainable parameter — 4 MiB of weight, 12 MiB of checkpoint state per
/// step once Adam's two moments are counted. Large enough for per-step retention to show up in the
/// working set, small enough to train in well under a second a step.</summary>
[Module]
public partial class MemoryStabilityWideModel
{
    public static Tensor<float32> Inline(Tensor<float32> x)
        => x.MatMul(Shorokoo.Modules.Initializers.XavierUniform.Init([Scalar(1024L), Scalar(1024L)]));
}

/// <summary>
/// Code-pinned memory-stability gate for the training hot path: a long-running
/// training loop must show stable memory, with no unbounded RSS growth and no
/// handle leaks. It drives the same pinned linear scenario as the throughput gate
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
///
/// <para>
/// That shape is blind to native memory a forced collection would have released, so the second
/// gate here forces nothing and measures RSS over a model whose per-step state is large enough to
/// see — see
/// <see cref="TestATrainingLoopDoesNotGrowTheProcessWhenNothingForcesACollection"/>.
/// </para>
/// </summary>
[Trait("Domain", "Training")]
[Trait("Purpose", "Benchmark")]
[Collection(SerialMeasurement.Name)]
public class TrainingMemoryStabilityTests
{
    private static readonly long[] WideInputShape = [4L, 1024L];

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

    // Geometry for the native-growth half below. One [1024, 1024] trainable parameter, so a
    // checkpoint carries 4 MiB of parameter plus two Adam moments — 12 MiB of per-step state, a
    // size the linear scenario above is far too small to expose. Over the measured steps a loop
    // that retains every step's state grows by ~360 MiB; 64 MiB absorbs arena and allocator
    // jitter.
    private const int NativeWarmupSteps = 5;
    private const int NativeMeasuredSteps = 30;
    private const long NativeRssGrowthBudgetBytes = 64L * 1024 * 1024;

    [Fact]
    public void LongTrainingLoopKeepsManagedMemoryStable()
    {
        var baseGraph = PerfBaselineLinearModel.ComputationGraph;
        var exampleInput = TensorData(InputShape, new float[8]);

        var rig = TrainingRig.FromScratch(
            baseGraph, Losses.L2Loss, Optimizers.Adam,
            baseGraph.FromOrderedInputs([exampleInput]),
            new AdamOptimizerHyperparameters { LearningRate = 1e-3f });

        var inputBatch = rig.InputDef.FromOrderedData(
            TensorData(InputShape, (float[])[1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f]));
        var targetBatch = rig.TargetDef.FromOrderedData(
            TensorData(TargetShape, (float[])[1f, 0f, 1f, 0f]));

        var ckpt = rig.CreateInitialCheckpoint();
        for (int i = 0; i < WarmupSteps; i++)
            ckpt = rig.TrainStep(ckpt, inputBatch, targetBatch);

        long managedBefore = LiveManagedBytes();
        long rssBefore = WorkingSetBytes();

        for (int i = 0; i < MeasuredSteps; i++)
            ckpt = rig.TrainStep(ckpt, inputBatch, targetBatch);

        long managedAfter = LiveManagedBytes();
        long rssAfter = WorkingSetBytes();

        long managedGrowth = managedAfter - managedBefore;
        long rssGrowth = rssAfter - rssBefore;

        // Keep the checkpoint reachable past the final measurement so the loop's
        // last result can't be collected before we read the heap.
        Assert.NotNull(ckpt);

        Assert.True(managedGrowth <= ManagedGrowthBudgetBytes);
        Assert.True(rssGrowth <= RssGrowthCeilingBytes);
    }

    /// <summary>
    /// The gate above measures the live managed heap with a forced collection on both ends, which
    /// is blind to the native buffers a checkpoint's tensors own: they are released only when the
    /// wrappers in front of them are collected, and the wrappers are far too small to prompt a
    /// collection on their own. This half therefore forces nothing and measures RSS, so it fails
    /// if the rig ever stops reclaiming state its own steps have superseded. The control loop —
    /// the same steps with an explicit collection each step — is what the other one has to match.
    /// </summary>
    [Fact]
    public void TestATrainingLoopDoesNotGrowTheProcessWhenNothingForcesACollection()
    {
        Assert.True(NativeRssGrowth(collectEachStep: true) <= NativeRssGrowthBudgetBytes);
        Assert.True(NativeRssGrowth(collectEachStep: false) <= NativeRssGrowthBudgetBytes);
    }

    /// <summary>
    /// A resident run (Shorokoo/Shorokoo#325) supersedes its own state every step and releases what
    /// it superseded, rather than leaving it to a finalizer that a loop this light on managed
    /// allocation never provokes. So the same scenario as the gate above, driven through a resident
    /// run and forcing nothing, must hold the process flat. This is also the only place the release
    /// can be seen on a host-only backend, where residency itself is a no-op.
    /// </summary>
    [Fact]
    public void TestAResidentRunDoesNotGrowTheProcessWhenNothingForcesACollection()
        => Assert.True(ResidentRssGrowth() <= NativeRssGrowthBudgetBytes);

    /// <summary>Working-set growth across <see cref="NativeMeasuredSteps"/> resident steps of the
    /// same 12 MiB-per-step rig, measured after a warm-up and with no forced collection.</summary>
    private static long ResidentRssGrowth()
    {
        var (rig, inputBatch, targetBatch) = WideRig();
        using var run = rig.BeginResidentRun();
        for (int i = 0; i < NativeWarmupSteps; i++) run.Step(inputBatch, targetBatch);

        long before = WorkingSetBytes();
        for (int i = 0; i < NativeMeasuredSteps; i++) run.Step(inputBatch, targetBatch);
        long after = WorkingSetBytes();

        // Keep the run reachable past the measurement.
        Assert.Equal(NativeWarmupSteps + NativeMeasuredSteps, run.CurrentStep);
        return after - before;
    }

    /// <summary>The 12 MiB-per-step rig both native-growth measurements drive, and its batches.</summary>
    private static (TrainingRig Rig, TensorDataStruct Input, TensorDataStruct Target) WideRig()
    {
        var graph = MemoryStabilityWideModel.ComputationGraph;
        var exampleInput = TensorData(WideInputShape, new float[4 * 1024]);
        var rig = TrainingRig.FromScratch(
            graph, Losses.L2Loss, Optimizers.Adam,
            graph.FromOrderedInputs([exampleInput]),
            new AdamOptimizerHyperparameters { LearningRate = 1e-3f });
        return (rig,
            rig.InputDef.FromOrderedData(TensorData(WideInputShape, new float[4 * 1024])),
            rig.TargetDef.FromOrderedData(TensorData(WideInputShape, new float[4 * 1024])));
    }

    /// <summary>Working-set growth across <see cref="NativeMeasuredSteps"/> steps of a rig whose
    /// per-step state is 12 MiB, measured after a warm-up so the arena has settled.</summary>
    private static long NativeRssGrowth(bool collectEachStep)
    {
        var (rig, inputBatch, targetBatch) = WideRig();

        var ckpt = rig.CreateInitialCheckpoint();
        for (int i = 0; i < NativeWarmupSteps; i++)
            ckpt = rig.TrainStep(ckpt, inputBatch, targetBatch);

        long before = WorkingSetBytes();
        for (int i = 0; i < NativeMeasuredSteps; i++)
        {
            ckpt = rig.TrainStep(ckpt, inputBatch, targetBatch);
            if (collectEachStep)
            {
                GC.Collect(2, GCCollectionMode.Forced, blocking: true);
                GC.WaitForPendingFinalizers();
            }
        }
        long after = WorkingSetBytes();

        // Keep the final checkpoint reachable past the measurement.
        Assert.NotNull(ckpt);
        return after - before;
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

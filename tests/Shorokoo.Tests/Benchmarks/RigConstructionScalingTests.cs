using System.Diagnostics;
using Shorokoo.Modules.Initializers;

namespace Shorokoo.Tests.Benchmarks;

/// <summary>One trainable <c>[rows, 384]</c> normal-initialized table, reduced to a scalar factor.
/// The small half of the per-element memory measurement.</summary>
[Module]
public partial class RigScalingTableSmall
{
    public const long Rows = 4096L;
    public static Tensor<float32> Inline(Tensor<float32> x)
        => x * NormalDist.Init(Vector(Rows, 384L), Scalar(0f), Scalar(0.02f))
                 .Reduce(ReduceKind.Mean, null, keepDims: false).Scalar();
}

/// <summary>The large half of the per-element memory measurement: four times
/// <see cref="RigScalingTableSmall"/>'s elements in one trainable parameter.</summary>
[Module]
public partial class RigScalingTableLarge
{
    public const long Rows = 16384L;
    public static Tensor<float32> Inline(Tensor<float32> x)
        => x * NormalDist.Init(Vector(Rows, 384L), Scalar(0f), Scalar(0.02f))
                 .Reduce(ReduceKind.Mean, null, keepDims: false).Scalar();
}

/// <summary>2 trainable <c>[384, 384]</c> tables. The two stack sizes exist as distinct module
/// types because a module's <c>ComputationGraph</c> is a cached static, so one parameterized
/// module cannot be built at several sizes in one process.</summary>
[Module]
public partial class RigScalingStack2
{
    public static Tensor<float32> Inline(Tensor<float32> x) => RigScalingStack.Chain(x, 2);
}

/// <summary>12 trainable <c>[384, 384]</c> tables.</summary>
[Module]
public partial class RigScalingStack12
{
    public static Tensor<float32> Inline(Tensor<float32> x) => RigScalingStack.Chain(x, 12);
}

/// <summary>12 trainable tables of 12 different shapes, <c>[4096 + i, 384]</c>: twelve slices,
/// so twelve initialization sessions, which is what gives the retention arm twelve sessions a result
/// could keep alive.</summary>
[Module]
public partial class RigScalingDistinct12
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var acc = x;
        for (long i = 0; i < 12; i++)
            acc *= NormalDist.Init(Vector(4096L + i, 384L), Scalar(0f), Scalar(0.02f))
                     .Reduce(ReduceKind.Mean, null, keepDims: false).Scalar();
        return acc;
    }
}

internal static class RigScalingStack
{
    internal static Tensor<float32> Chain(Tensor<float32> x, int layers)
    {
        var acc = x;
        for (int i = 0; i < layers; i++)
            acc *= NormalDist.Init(Vector(384L, 384L), Scalar(0f), Scalar(0.02f))
                     .Reduce(ReduceKind.Mean, null, keepDims: false).Scalar();
        return acc;
    }
}

/// <summary>
/// Code-pinned scaling gate for the phase that runs every trainable parameter's initializer
/// before any training happens. Three laws — host memory
/// (<see href="https://github.com/Shorokoo/Shorokoo/issues/194">#194</see>), build time
/// (<see href="https://github.com/Shorokoo/Shorokoo/issues/195">#195</see>), and what each
/// per-parameter session retains — each with a cause of its own, so one measurement cannot stand
/// in for another.
///
/// <para><b>Bytes per trainable element.</b> A backend folding the whole input-less
/// initialization graph at session build costs ~4.4 KiB of host working set per parameter
/// ELEMENT, ~1100x the 4 bytes the fp32 parameter occupies, and a draw computed over a whole
/// parameter at once still costs ~400 B, its integer intermediates all live together; a draw
/// computed a chunk of positions at a time costs the parameter itself. Measured as the ADDITIONAL
/// peak initializing the large table needs over the small one, so the fixed process floor cancels
/// and what remains is the per-element law — machine-independent in a way a wall clock is not.
/// Around initialization rather than a whole rig build, whose own fixed peaks — composing the
/// training step — sit above anything a table's draw adds and would flatten the difference
/// to nothing.</para>
///
/// <para><b>Cost per trainable parameter.</b> Initializing every parameter in one session makes
/// construction quadratic in the parameter count, since the backend's session build is
/// superlinear in graph size. Each parameter runs on a session built for its own slice of the
/// initialization graph, shared by the parameters whose slices match, which is at most linear. Measured as
/// a RATIO of per-parameter cost at the two ends of the range, so no absolute time budget is
/// needed and it holds on any machine: linear keeps it near 1, the quadratic law puts
/// it near 6. What a ratio cannot see is a uniform constant-factor slowdown, and between 2 and 12
/// parameters it does not separate mild superlinearity (N^1.3 lands at 1.7) from linear either.
/// It pins the shape that broke, not every way construction could get slower.</para>
///
/// <para><b>Bytes retained.</b> Running the parameters on sessions of their own is only
/// affordable because each result holds only its own bytes; a result that kept its session alive
/// would keep the session's own memory with it, and the blocks its runs let go of where the session
/// keeps them for its next run, and a forced collection could not reclaim either, since the values
/// are genuinely referenced as the rig's initial weights. Measured around
/// initialization alone, not around a whole <see cref="TrainingRig.FromScratch"/>: a rig
/// legitimately retains 100-150 MiB of graphs and state, which is both larger and noisier than the
/// signal. And measured outside the managed heap, where a session and its blocks live: the heap's
/// own commit and decommit around a collection moves tens of MiB either way. The values live there too, each in
/// memory of its own, so what is gated is what is retained beyond their own bytes. Measured over
/// twelve tables of twelve shapes, so twelve sessions: sessions are shared by same-shaped
/// parameters, so it is the number of sessions, and the size of what they computed, that a
/// regression here multiplies. A healthy
/// reading is near zero, so a known native block is read too, to show the instrument sees one.
/// Optimizer-state seeding is the other per-parameter run loop that keeps its outputs, and they
/// hold only their own bytes the same way — but its graph is a fill rather than a draw, so what
/// one of its sessions would keep is small enough to sit inside that noise, and no memory gate
/// discriminates it. It is not pinned here.</para>
///
/// <para>Each budget sits well above the measured behaviour and well below the broken law, so
/// jitter never trips one. The timing points are best-of-<see cref="TimingRuns"/>: a single
/// sample's noise is comparable to the effect at the small end.</para>
/// </summary>
[Trait("Domain", "Training")]
[Trait("Purpose", "Benchmark")]
[Collection(SerialMeasurement.Name)]
public class RigConstructionScalingTests
{
    /// <summary>Measured 14-18 B/element; a whole-parameter draw gives ~400, and a folded graph ~4.4 KiB.</summary>
    private const double MemoryBudgetBytesPerElement = 128.0;

    /// <summary>Measured 0.2-0.4 (shared sessions, drawn side by side, make the deeper stack
    /// cheaper per parameter); the quadratic law gives ~6.</summary>
    private const double MaxPerParameterCostGrowth = 2.0;

    /// <summary>Measured -4 to +5 MiB of native memory beyond the values' own 72 MiB across a
    /// 12-session initialization, Windows and Linux. Results that each keep their session alive
    /// retain 42-59 MiB beyond them, and 443-461 MiB where the sessions also keep what their runs
    /// let go of.</summary>
    private const long RetainedBudgetBytes = 24L * 1024 * 1024;

    /// <summary>The native memory the retention instrument is shown, to prove it reads one.</summary>
    private const int ControlBytes = 64 * 1024 * 1024;

    private const int TimingRuns = 3;
    private const int StackParams = 12;
    private const int SmallStackParams = 2;

    private const long SmallTableElements = RigScalingTableSmall.Rows * 384L;
    private const long LargeTableElements = RigScalingTableLarge.Rows * 384L;

    [Fact]
    public void RigConstructionScalesWithTheModelRatherThanExplodingOnIt()
    {
        // Concretize up front: only initialization is under measurement, and concretizing inside
        // a timed region would put a second, unrelated cost into the ratio.
        var small = Concretize(RigScalingStack2.ComputationGraph);
        var large = Concretize(RigScalingStack12.ComputationGraph);
        // Pays the process's one-time JIT / first-touch cost, in turn as the tables below run, so
        // it cannot set a peak their single-parameter builds do not reach.
        using (Shorokoo.Core.Nodes.Processors.Fast.FastInitializeModelParams.DecideSideBySide(false))
            small.InitializeTrainableParams();

        // Peak working set is monotonic, so the two table builds have to be what raises it. That
        // holds while this class runs in a process of its own, which is how the release workflow
        // invokes it; the assertion below is what catches it if that ever stops being true,
        // rather than letting the arm read zero and pass.
        var smallTable = Concretize(RigScalingTableSmall.ComputationGraph);
        var largeTable = Concretize(RigScalingTableLarge.ComputationGraph);
        long peakBeforeTables = PeakWorkingSetBytes();
        smallTable.InitializeTrainableParams();
        long peakAfterSmallTable = PeakWorkingSetBytes();
        largeTable.InitializeTrainableParams();
        long peakGrowth = PeakWorkingSetBytes() - peakAfterSmallTable;

        // Warmed first, so what the allocators take or hand back on a first run is not counted
        // against the one measured.
        var distinct = Concretize(RigScalingDistinct12.ComputationGraph);
        distinct.InitializeTrainableParams();
        long before = LiveNativeBytes();
        var values = distinct.InitializeTrainableParams();
        long retained = LiveNativeBytes() - before
                        - values.ModelParams.Sum(value => value.ToTensorData().ByteCount);
        long controlSeen = NativeBytesSeenOf(ControlBytes);

        double smallSeconds = BestInitSeconds(small);
        double largeSeconds = BestInitSeconds(large);
        double perParameterCostGrowth =
            (largeSeconds / StackParams) / (smallSeconds / SmallStackParams);

        // Keep the values past the retention measurement: collecting them early is exactly the
        // thing that would make a retention regression invisible.
        Assert.Equal(StackParams, values.ModelParams.Length);

        // Both memory arms are differences that a saturated or reused peak can flatten to zero,
        // and a healthy retention reads near zero by design. Fail on a measurement that
        // established nothing — a flat peak, or an instrument blind to a known native block —
        // rather than divide by it and pass.
        Assert.True(peakAfterSmallTable > peakBeforeTables);
        Assert.True(peakGrowth > 0);
        Assert.True(controlSeen >= ControlBytes * 3L / 4);

        Assert.True(peakGrowth / (double)(LargeTableElements - SmallTableElements)
                    <= MemoryBudgetBytesPerElement);
        Assert.True(retained <= RetainedBudgetBytes);
        Assert.True(perParameterCostGrowth <= MaxPerParameterCostGrowth);
    }

    private static InternalComputationGraph Concretize(ComputationGraph model)
    {
        var g = model.ToInternal();
        return g.ToConcreteArchitecture([TensorData([1L], (float[])[1f])]);
    }

    private static double BestInitSeconds(InternalComputationGraph arch)
    {
        double best = double.MaxValue;
        for (int i = 0; i < TimingRuns; i++)
        {
            var sw = Stopwatch.StartNew();
            arch.InitializeTrainableParams();
            best = Math.Min(best, sw.Elapsed.TotalSeconds);
        }
        return best;
    }

    private static long PeakWorkingSetBytes()
    {
        using var proc = Process.GetCurrentProcess();
        proc.Refresh();
        return proc.PeakWorkingSet64;
    }

    // The positive control for the retention arm, whose healthy reading is near zero.
    private static long NativeBytesSeenOf(int bytes)
    {
        long before = LiveNativeBytes();
        var block = System.Runtime.InteropServices.Marshal.AllocHGlobal(bytes);
        try
        {
            for (int offset = 0; offset < bytes; offset += 4096)
                System.Runtime.InteropServices.Marshal.WriteByte(block, offset, 1);
            return LiveNativeBytes() - before;
        }
        finally
        {
            System.Runtime.InteropServices.Marshal.FreeHGlobal(block);
        }
    }

    // The working set outside the managed heap, whose own commit and decommit around a collection
    // moves tens of MiB either way.
    private static long LiveNativeBytes()
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        using var proc = Process.GetCurrentProcess();
        proc.Refresh();
        return proc.WorkingSet64 - GC.GetGCMemoryInfo().TotalCommittedBytes;
    }
}

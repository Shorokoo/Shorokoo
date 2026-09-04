using System.Diagnostics;
using Shorokoo.Modules.Initializers;
using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;

namespace Shorokoo.Tests.Benchmarks;

/// <summary>One trainable <c>[rows, 384]</c> normal-initialized table, reduced to a scalar factor.
/// The small half of the per-element memory measurement.</summary>
[Module]
public partial class RigScalingTableSmall
{
    public const long Rows = 1024L;
    public static Tensor<float32> Inline(Tensor<float32> x)
        => x * NormalDist.Init(Vector(Rows, 384L), Scalar(0f), Scalar(0.02f))
                 .Reduce(ReduceKind.Mean, null, keepDims: false).Scalar();
}

/// <summary>The large half of the per-element memory measurement: four times
/// <see cref="RigScalingTableSmall"/>'s elements in one trainable parameter.</summary>
[Module]
public partial class RigScalingTableLarge
{
    public const long Rows = 4096L;
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
/// before any training happens. Three laws, because rig construction broke on three separate
/// things (<see href="https://github.com/Shorokoo/Shorokoo/issues/194">#194</see> host memory,
/// <see href="https://github.com/Shorokoo/Shorokoo/issues/195">#195</see> build time, and the
/// per-session retention that fixing #195 introduced) and each has a different cause, so one
/// measurement cannot stand in for another.
///
/// <para><b>Bytes per trainable element.</b> Construction used to cost ~4.4 KiB of host working
/// set per parameter ELEMENT, ~1100x the 4 bytes the fp32 parameter occupies, because the backend
/// folded the whole input-less initialization graph at session build. Measured as the ADDITIONAL
/// peak the large table needs over the small one, so the fixed process floor cancels and what
/// remains is the per-element law — machine-independent in a way a wall clock is not.</para>
///
/// <para><b>Cost per trainable parameter.</b> Initializing every parameter in one session made
/// construction quadratic in the parameter count, since the backend's session build is
/// superlinear in graph size. It runs one session per parameter now, which is linear. Measured as
/// a RATIO of per-parameter cost at the two ends of the range, so no absolute time budget is
/// needed and it holds on any machine: linear keeps it near 1, the quadratic law it replaced puts
/// it near 6. What a ratio cannot see is a uniform constant-factor slowdown, and between 2 and 12
/// parameters it does not separate mild superlinearity (N^1.3 lands at 1.7) from linear either.
/// It pins the shape that broke, not every way construction could get slower.</para>
///
/// <para><b>Bytes retained.</b> One session per parameter is only affordable because each result
/// is copied off its session; a retained result keeps its session's whole arena alive, and a
/// forced collection cannot reclaim it, since the values are genuinely referenced as the rig's
/// initial weights. Measured around initialization alone, not around a whole
/// <see cref="TrainingRig.FromScratch"/>: a rig legitimately retains 100-150 MiB of graphs and
/// state, which is both larger and noisier than the signal. Optimizer-state seeding is the other
/// per-parameter session loop that keeps its outputs, and it takes the same copy — but its graph
/// is a fill rather than a draw, so the arena it would pin is small enough to sit inside that
/// noise, and no memory gate discriminates it. It is not pinned here, and the finding
/// <c>ort-values-are-never-disposed</c> says so.</para>
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
    /// <summary>Measured ~400 B/element over a 0.5% spread; the law this catches is ~4.4 KiB.</summary>
    private const double MemoryBudgetBytesPerElement = 1536.0;

    /// <summary>Measured 0.65-1.14 (linear); the quadratic law it replaced gives ~6.</summary>
    private const double MaxPerParameterCostGrowth = 2.0;

    /// <summary>Measured 10-46 MiB across a 12-parameter initialization; uncopied, 379-481 MiB.</summary>
    private const long RetainedBudgetBytes = 96L * 1024 * 1024;

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
        small.InitializeTrainableParams();   // pays the process's one-time JIT / first-touch cost

        // Peak working set is monotonic, so the two table builds have to be what raises it. That
        // holds while this class runs in a process of its own, which is how the release workflow
        // and CLAUDE.md invoke it; the assertion below is what catches it if that ever stops
        // being true, rather than letting the arm read zero and pass.
        long peakBeforeTables = PeakWorkingSetBytes();
        BuildRig(RigScalingTableSmall.ComputationGraph);
        long peakAfterSmallTable = PeakWorkingSetBytes();
        BuildRig(RigScalingTableLarge.ComputationGraph);
        long peakGrowth = PeakWorkingSetBytes() - peakAfterSmallTable;

        long before = LiveWorkingSetBytes();
        var values = large.InitializeTrainableParams();
        long retained = LiveWorkingSetBytes() - before;

        double smallSeconds = BestInitSeconds(small);
        double largeSeconds = BestInitSeconds(large);
        double perParameterCostGrowth =
            (largeSeconds / StackParams) / (smallSeconds / SmallStackParams);

        // Keep the values past the retention measurement: collecting them early is exactly the
        // thing that would make a retention regression invisible.
        Assert.Equal(StackParams, values.ModelParams.Length);

        // Both memory arms are differences that a saturated or reused peak can flatten to zero.
        // Fail on a measurement that established nothing rather than divide by it and pass.
        Assert.True(peakAfterSmallTable > peakBeforeTables);
        Assert.True(peakGrowth > 0);
        Assert.True(retained > 0);

        Assert.True(peakGrowth / (double)(LargeTableElements - SmallTableElements)
                    <= MemoryBudgetBytesPerElement);
        Assert.True(retained <= RetainedBudgetBytes);
        Assert.True(perParameterCostGrowth <= MaxPerParameterCostGrowth);
    }

    private static InternalComputationGraph Concretize(ComputationGraph model)
    {
        var g = model.ToInternal();
        return g.ToConcreteArchitecture(g.FromOrderedInputs([TensorData([1L], (float[])[1f])]));
    }

    private static TrainingRig BuildRig(ComputationGraph model) =>
        TrainingRig.FromScratch(
            model, L1Loss.ComputationGraph, AdamOptimizer.ComputationGraph,
            [new TensorDataModelParam("x", ModelParamType.InputParam, TensorData([1L], (float[])[1f]))],
            new AdamOptimizerHyperparameters { LearningRate = 0.1f });

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

    /// <summary>Working set after a blocking full collection, so only real retention is counted.</summary>
    private static long LiveWorkingSetBytes()
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        using var proc = Process.GetCurrentProcess();
        proc.Refresh();
        return proc.WorkingSet64;
    }
}

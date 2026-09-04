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

/// <summary>2 trainable <c>[384, 384]</c> tables. The stack sizes exist as four distinct module
/// types because a module's <c>ComputationGraph</c> is a cached static, so one parameterized
/// module cannot be built at several sizes in one process.</summary>
[Module]
public partial class RigScalingStack2
{
    public static Tensor<float32> Inline(Tensor<float32> x) => RigScalingStack.Chain(x, 2);
}

/// <summary>4 trainable <c>[384, 384]</c> tables.</summary>
[Module]
public partial class RigScalingStack4
{
    public static Tensor<float32> Inline(Tensor<float32> x) => RigScalingStack.Chain(x, 4);
}

/// <summary>8 trainable <c>[384, 384]</c> tables.</summary>
[Module]
public partial class RigScalingStack8
{
    public static Tensor<float32> Inline(Tensor<float32> x) => RigScalingStack.Chain(x, 8);
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
/// Code-pinned scaling gate for <see cref="TrainingRig.FromScratch"/> — the phase that runs every
/// trainable parameter's initializer before any training happens. Two laws are pinned, one per
/// axis, because rig construction once broke on both
/// (<see href="https://github.com/Shorokoo/Shorokoo/issues/194">#194</see> host memory,
/// <see href="https://github.com/Shorokoo/Shorokoo/issues/195">#195</see> build time) and the two
/// failures had different causes.
///
/// <para><b>Memory — bytes per trainable element.</b> Construction used to cost ~4.4 KiB of host
/// working set per parameter ELEMENT, about 1100x the 4 bytes the fp32 parameter occupies, so a
/// few-million-parameter model needed tens of GB to build and a GPT-sized embedding died with
/// ORT's bare "bad allocation". The cause was the backend folding the whole input-less
/// initialization graph at session build, materializing every intermediate of every keyed
/// Threefry draw at once. The gate divides the ADDITIONAL peak working set the large table needs
/// over the small one by their element difference, so the fixed process floor cancels and the
/// figure is the per-element law itself — a machine-independent quantity, unlike a wall clock.</para>
///
/// <para><b>Time — cost per additional trainable parameter.</b> The backend's session build is
/// superlinear in graph size, so initializing every parameter in one session made construction
/// grow quadratically in the parameter count (4x per doubling): minutes of pure build for a
/// GPT-sized model, re-paid on every process restart. Initialization now runs one session per
/// parameter, which is linear. The gate is a pure RATIO of two per-parameter increments measured
/// at opposite ends of the range, so it needs no absolute time budget and holds on any machine:
/// linear construction keeps the ratio near 1, and the quadratic law it replaced puts it above 3.</para>
///
/// <para>Both budgets are deliberately loose — comfortably above the measured behaviour and
/// comfortably below the broken law — so ordinary run-to-run jitter never trips them.</para>
/// </summary>
[Trait("Domain", "Training")]
[Trait("Purpose", "Benchmark")]
[Collection(SerialMeasurement.Name)]
public class RigConstructionScalingTests
{
    /// <summary>Measured ~0.4 KiB/element; the law this gate exists to catch is ~4.4 KiB.</summary>
    private const double MemoryBudgetBytesPerElement = 1536.0;

    /// <summary>Measured ~0.85 under linear construction; the quadratic law it replaced gives ~3.1.</summary>
    private const double MaxPerParameterCostGrowth = 2.0;

    private const long SmallTableElements = RigScalingTableSmall.Rows * 384L;
    private const long LargeTableElements = RigScalingTableLarge.Rows * 384L;

    [Fact]
    public void RigConstructionScalesWithTheModelRatherThanExplodingOnIt()
    {
        // Warm-up on the small table: it pays the process's one-time JIT / first-touch cost AND
        // sets the peak the large table is then measured against, so both measurements below see
        // an already-warm process.
        BuildRig(RigScalingTableSmall.ComputationGraph);
        long peakAfterSmall = PeakWorkingSetBytes();

        BuildRig(RigScalingTableLarge.ComputationGraph);
        double bytesPerElement =
            (PeakWorkingSetBytes() - peakAfterSmall) / (double)(LargeTableElements - SmallTableElements);

        double t2 = BuildSeconds(RigScalingStack2.ComputationGraph);
        double t4 = BuildSeconds(RigScalingStack4.ComputationGraph);
        double t8 = BuildSeconds(RigScalingStack8.ComputationGraph);
        double t12 = BuildSeconds(RigScalingStack12.ComputationGraph);

        // Per-added-parameter cost at the top of the range against the same at the bottom. Both
        // are differences, so the fixed per-build cost cancels out of each.
        double perParameterCostGrowth = ((t12 - t8) / 4.0) / ((t4 - t2) / 2.0);

        Assert.True(bytesPerElement <= MemoryBudgetBytesPerElement);
        Assert.True(perParameterCostGrowth <= MaxPerParameterCostGrowth);
    }

    private static void BuildRig(ComputationGraph model) =>
        TrainingRig.FromScratch(
            model, L1Loss.ComputationGraph, AdamOptimizer.ComputationGraph,
            [new TensorDataModelParam("x", ModelParamType.InputParam, TensorData([1L], (float[])[1f]))],
            new AdamOptimizerHyperparameters { LearningRate = 0.1f });

    private static double BuildSeconds(ComputationGraph model)
    {
        var sw = Stopwatch.StartNew();
        BuildRig(model);
        return sw.Elapsed.TotalSeconds;
    }

    private static long PeakWorkingSetBytes()
    {
        using var proc = Process.GetCurrentProcess();
        proc.Refresh();
        return proc.PeakWorkingSet64;
    }
}

using Shorokoo.Core.Inference;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// Interpreter↔lowered-graph parity for <see cref="ScheduleLowering"/>: each schedule is lowered
/// to graph math (int64 step counter → float32 value) and evaluated against the host
/// <see cref="Schedule.At"/> on both the <see cref="QuickExecutionEngine"/> (bit-exact) and ONNX
/// Runtime (transcendental ulp tolerance), densely around every piecewise boundary plus large
/// counters around the float32 2²⁴ integer-exactness limit.
/// </summary>
[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class ScheduleLoweringCoverageTests
{
    // ONNX Runtime's CPU float32 Cos/Pow may drift ~1 ulp from .NET's MathF; elementary
    // arithmetic, Cast, Clip, comparisons and Where are IEEE-deterministic and get no allowance.
    private const int TranscendentalUlps = 4;
    private const float PeakRelativeTolerance = 1f / (1 << 22);

    private static void AssertParity(Schedule schedule, IEnumerable<long> probeSteps, int ortUlps = 0)
    {
        long[] probes = [.. probeSteps.Where(p => p >= 0 && p <= int.MaxValue).Distinct().Order()];
        Assert.NotEmpty(probes);
        float[] expected = [.. probes.Select(p => schedule.At(p))];

        var steps = InputVector<int64>("steps");
        var value = schedule.LowerToGraph(steps);
        var graph = new InternalComputationGraph([steps], [value]);
        var input = TensorData([probes.Length], probes);

        var qee = ((TensorData)new QuickExecutionEngine { MaxDataElements = 1 << 22 }
            .Execute(graph, input)[0]).As<float32>().AccessMemory<float>().ToArray();
        AssertValuesWithin(expected, qee, 0, 0f);

        float absTol = ortUlps == 0 ? 0f : expected.Max(MathF.Abs) * PeakRelativeTolerance;
        var ort = new ComputeContext().Execute(graph, input)[0]
            .ToTensorData().As<float32>().AccessMemory<float>().ToArray();
        AssertValuesWithin(expected, ort, ortUlps, absTol);
    }

    private static void AssertValuesWithin(float[] expected, float[] actual, int tolUlps, float absTol)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(UlpDistance(expected[i], actual[i]) <= tolUlps
                || MathF.Abs(expected[i] - actual[i]) <= absTol);
    }

    private static long UlpDistance(float a, float b)
    {
        if (a == b) return 0;
        if (float.IsNaN(a) || float.IsNaN(b)) return long.MaxValue;
        return Math.Abs(OrderedBits(a) - OrderedBits(b));
    }

    private static long OrderedBits(float f)
    {
        long bits = BitConverter.SingleToInt32Bits(f);
        return bits >= 0 ? bits : int.MinValue - bits;
    }

    private static IEnumerable<long> DenseRange(long from, long toInclusive)
    {
        for (long s = from; s <= toInclusive; s++) yield return s;
    }

    private static IEnumerable<long> LargeSteps =>
        [1 << 20, (1 << 24) - 2, (1 << 24) - 1, 1 << 24, (1 << 24) + 1, (1 << 24) + 2,
         (1L << 24) + 7, 1L << 28, 2_000_000_000, int.MaxValue - 8];

    [Fact]
    public void TestFactoryShapeParity()
    {
        AssertParity(Schedules.Constant(3e-4f), [.. DenseRange(0, 8), .. LargeSteps]);
        AssertParity(Schedules.Linear(3e-4f, 1e-5f, 1000), [.. DenseRange(0, 1050), .. LargeSteps]);
        AssertParity(Schedules.Linear(-1f, 1f, 7), [.. DenseRange(0, 30), .. LargeSteps]);
        AssertParity(Schedules.Cosine(3e-4f, 1000), [.. DenseRange(0, 1050), .. LargeSteps], TranscendentalUlps);
        AssertParity(Schedules.CosineWithWarmup(3e-4f, 100, 1000), [.. DenseRange(0, 1150), .. LargeSteps], TranscendentalUlps);
        // warmupSteps <= 0 arm: WithWarmup(0) returns the cosine unchanged.
        AssertParity(Schedules.CosineWithWarmup(1f, 0, 100), [.. DenseRange(0, 120)], TranscendentalUlps);
        // warmupSteps >= totalSteps arm: the decay window degenerates to Max(1, …) = 1 step.
        AssertParity(Schedules.CosineWithWarmup(1f, 20, 10), [.. DenseRange(0, 40)], TranscendentalUlps);
        AssertParity(Schedules.StepDecay(1e-2f, 30, 0.5f), [.. DenseRange(0, 400), .. LargeSteps], TranscendentalUlps);
        AssertParity(Schedules.Exponential(1e-2f, 0.999f),
            [.. DenseRange(0, 200), 1000, 5000, 20000, 100000, .. LargeSteps], TranscendentalUlps);
        AssertParity(Schedules.OneCycle(0.1f, 1000), [.. DenseRange(0, 1100), .. LargeSteps], TranscendentalUlps);
        AssertParity(Schedules.OneCycle(0.3f, 500, pctStart: 0.1f, divFactor: 10f, finalDivFactor: 100f),
            [.. DenseRange(0, 560)], TranscendentalUlps);
        // pctStart = 1 edge: the whole run is the rising phase; down degenerates to Max(1, 0) = 1.
        AssertParity(Schedules.OneCycle(0.2f, 100, pctStart: 1f), [.. DenseRange(0, 120)], TranscendentalUlps);
    }

    [Fact]
    public void TestCombinatorParity()
    {
        AssertParity(Schedules.Cosine(1f, 100).Scale(0.5f), [.. DenseRange(0, 120), .. LargeSteps], TranscendentalUlps);
        AssertParity(Schedules.Linear(1f, 0f, 100).Clamp(0.2f, 0.8f), [.. DenseRange(0, 120), .. LargeSteps]);
        AssertParity(Schedules.Cosine(1f, 100).Shift(25), [.. DenseRange(0, 120)], TranscendentalUlps);
        // Negative shift: early steps read the inner schedule at negative positions.
        AssertParity(Schedules.Cosine(1f, 100).Shift(-10), [.. DenseRange(0, 130)], TranscendentalUlps);
        AssertParity(Schedules.Cosine(1f, 50).PerEpoch(10), [.. DenseRange(0, 600), .. LargeSteps], TranscendentalUlps);
        AssertParity(Schedules.Cosine(1f, 200).WithWarmup(50), [.. DenseRange(0, 300), .. LargeSteps], TranscendentalUlps);
        AssertParity(Schedules.Cosine(1f, 200).WithWarmup(50, startFactor: 0.25f),
            [.. DenseRange(0, 300), .. LargeSteps], TranscendentalUlps);
        AssertParity(Schedules.Constant(0.5f).Then(100, Schedules.Linear(0.5f, 0f, 200)),
            [.. DenseRange(0, 350), .. LargeSteps]);
        AssertParity(
            Schedules.Constant(1f).Then(50, Schedules.Cosine(1f, 100)).Then(150, Schedules.Constant(0.1f)),
            [.. DenseRange(0, 250), .. LargeSteps], TranscendentalUlps);
    }

    [Fact]
    public void TestCompositeParity()
    {
        AssertParity(Schedules.Cosine(3e-4f, 900).WithWarmup(100).Then(1000, Schedules.Constant(1e-5f)),
            [.. DenseRange(0, 1100), .. LargeSteps], TranscendentalUlps);
        AssertParity(Schedules.OneCycle(0.4f, 100).PerEpoch(8).Scale(2f).Clamp(0.05f, 0.5f),
            [.. DenseRange(0, 900), .. LargeSteps], TranscendentalUlps);
        // PerEpoch inside a re-based Then branch: the epoch index derives from the re-based step.
        AssertParity(
            Schedules.Cosine(1f, 40).PerEpoch(5).Then(300, Schedules.StepDecay(0.5f, 20, 0.7f).PerEpoch(3)),
            [.. DenseRange(0, 700)], TranscendentalUlps);
        AssertParity(Schedules.Linear(1f, 0f, 100).Shift(30).WithWarmup(20), [.. DenseRange(0, 200)]);
    }

    [Fact]
    public void TestLargeCounterParity()
    {
        // Steps past 2²⁴ are not exactly representable in float32; the graph's int64→float32 Cast
        // must round identically to the host's.
        AssertParity(Schedules.Linear(0f, 1f, 1 << 26),
            [.. DenseRange((1 << 24) - 3, (1 << 24) + 3), 1 << 25, (1 << 26) - 1, 1 << 26,
             (1 << 26) + 1, .. LargeSteps]);

        // int64 counters past int.MaxValue, which the harness's probe cap cannot reach.
        var schedule = Schedules.Linear(1f, 0f, 1 << 26).WithWarmup(1000);
        long[] probes = [(long)int.MaxValue + 1, 1L << 40, 5_000_000_000L, long.MaxValue / 4];
        var steps = InputVector<int64>("steps");
        var graph = new InternalComputationGraph([steps], [schedule.LowerToGraph(steps)]);
        var qee = ((TensorData)new QuickExecutionEngine { MaxDataElements = 1 << 22 }
            .Execute(graph, TensorData([probes.Length], probes))[0]).As<float32>().AccessMemory<float>().ToArray();
        for (int i = 0; i < probes.Length; i++)
            Assert.Equal(schedule.At(probes[i]), qee[i]);
    }

    [Fact]
    public void TestLoweringContractRejectsOpaqueSchedulesAndSwappedClampBounds()
    {
        Schedule opaque = new((ScheduleExpr?)null);
        Assert.False(opaque.CanLower());
        Assert.False(opaque.Scale(2f).CanLower());
        Assert.False(opaque.Clamp(0f, 1f).CanLower());
        Assert.False(opaque.Shift(5).CanLower());
        Assert.False(opaque.PerEpoch(10).CanLower());
        Assert.False(opaque.WithWarmup(5).CanLower());
        Assert.False(opaque.Then(10, Schedules.Constant(1f)).CanLower());
        Assert.False(Schedules.Constant(1f).Then(10, opaque).CanLower());
        Assert.True(Schedules.Constant(1f).CanLower());
        Assert.True(Schedules.Cosine(1f, 10).WithWarmup(0).CanLower());
        Assert.True(Schedules.Cosine(1f, 10).PerEpoch(2).Then(5, Schedules.Constant(0f)).CanLower());
        Assert.Throws<InvalidOperationException>(() => opaque.LowerToGraph(InputScalar<int64>("step")));

        Assert.Throws<ArgumentException>(() => Schedules.Constant(0.5f).Clamp(1f, 0.2f));
        Assert.NotNull(Schedules.Constant(0.5f).Clamp(0.2f, 1f));
    }

    [Fact]
    public void TestScalarLoweringMatchesAndRoundtrips()
    {
        var schedule = Schedules.Cosine(3e-4f, 200).WithWarmup(50).Then(400, Schedules.Constant(1e-5f));
        long[] probes = [0, 1, 49, 50, 51, 199, 249, 250, 251, 399, 400, 401, 1000, 1 << 24];
        foreach (long p in probes)
        {
            var step = InputScalar<int64>("step");
            var graph = new InternalComputationGraph([step], [schedule.LowerToGraph(step)]);
            float got = new ComputeContext().Execute(graph, TensorData([], p))[0]
                .ToTensorData().As<float32>().AccessMemory<float>()[0];
            float expected = schedule.At(p);
            Assert.True(UlpDistance(expected, got) <= TranscendentalUlps
                || MathF.Abs(expected - got) <= 3e-4f * PeakRelativeTolerance);
        }

        var s2 = InputScalar<int64>("step");
        var g2 = new InternalComputationGraph([s2], [schedule.LowerToGraph(s2)]);
        Assert.True(AutoTest.TestGraph(g2, sampleInputs: [TensorData([], 123L)],
            testQuickEngineExecution: true));
    }
}

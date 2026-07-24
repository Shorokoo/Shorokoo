using Shorokoo.Core.Inference;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage-purpose parity tests for <see cref="ScheduleLowering"/> (issue #39 spike): every
/// schedule shape constructible from the public <see cref="Schedules"/> factories and
/// <see cref="Schedule"/> combinators is lowered to graph math
/// (<c>int64 step counter → float32 value</c>) and evaluated against the host
/// <see cref="Schedule.At"/> across dense step ranges around every piecewise boundary
/// (warmup end, <c>Then</c> transitions, cycle edges, step 0) plus large counters around the
/// float32 2²⁴ integer-exactness limit.
///
/// Each schedule is evaluated on two engines through the same lowered graph:
/// <list type="bullet">
///   <item><see cref="QuickExecutionEngine"/> (pure managed, <c>MathF</c>-based elementwise
///         ops) — asserted bit-exact against the host at every probe step, pinning that the
///         lowering mirrors the host's float32 arithmetic operation for operation.</item>
///   <item><see cref="ComputeContext"/> (ONNX Runtime CPU) — exact for arithmetic/piecewise
///         schedules; schedules whose math includes <c>Cos</c> or <c>Pow</c> get the measured
///         transcendental tolerance (<see cref="TranscendentalUlps"/>), the tolerance contract
///         measured during the issue #39 spike.</item>
/// </list>
/// </summary>
[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class ScheduleLoweringCoverageTests
{
    // ───────────────────────────── tolerance contract ─────────────────────────────

    /// <summary>
    /// Tolerance contract for ONNX Runtime's CPU float32 transcendentals (Cos, Pow), which may
    /// drift from .NET's MathF — and hence from the host schedule evaluator — by ~1 ulp of the
    /// transcendental's own result. A value agrees when it is within this many ulps of the host
    /// value, or within this many ulps <em>at the schedule's peak magnitude</em>
    /// (<c>maxAbs(host values) · 2⁻²²</c>): where cancellation shrinks the result (the
    /// <c>1 + cos(π·t)</c> cosine tail approaching 0), a 1-ulp Cos difference becomes an
    /// absolute error at the pre-cancellation scale, not a relative one. Elementary arithmetic
    /// (add, sub, mul, div), Cast, Clip, comparisons, and Where are IEEE-deterministic and get
    /// no allowance.
    /// </summary>
    private const int TranscendentalUlps = 4;

    /// <summary><see cref="TranscendentalUlps"/> ulps at peak scale: 4 · 2⁻²⁴ = 2⁻²².</summary>
    private const float PeakRelativeTolerance = 1f / (1 << 22);

    // ───────────────────────────── parity harness ─────────────────────────────

    /// <summary>
    /// Lowers <paramref name="schedule"/> once (rank-1 batch form), evaluates it at every probe
    /// step on both engines, and compares each value to the host <see cref="Schedule.At"/>.
    /// QEE must match bit-exactly; ONNX Runtime within <paramref name="ortUlps"/>.
    /// </summary>
    private static void AssertParity(Schedule schedule, IEnumerable<long> probeSteps, int ortUlps = 0)
    {
        long[] probes = [.. probeSteps.Where(p => p >= 0 && p <= int.MaxValue).Distinct().Order()];
        Assert.NotEmpty(probes);
        float[] expected = [.. probes.Select(p => schedule.At((int)p))];

        var steps = InputVector<int64>("steps");
        var value = schedule.LowerToGraph(steps);
        var graph = (new InternalComputationGraph([steps], [value]));
        var input = TensorData([probes.Length], probes);

        var qee = ((TensorData)new QuickExecutionEngine { MaxDataElements = 1 << 22 }
            .Execute(graph, input)[0]).As<float32>().AccessMemory<float>().ToArray();
        AssertValuesWithin(expected, qee, 0, 0f, probes, "QuickExecutionEngine");

        float absTol = ortUlps == 0 ? 0f : expected.Max(MathF.Abs) * PeakRelativeTolerance;
        var ort = new ComputeContext().Execute(graph, input)[0]
            .ToTensorData().As<float32>().AccessMemory<float>().ToArray();
        AssertValuesWithin(expected, ort, ortUlps, absTol, probes, "OnnxRuntime");
    }

    private static void AssertValuesWithin(
        float[] expected, float[] actual, int tolUlps, float absTol, long[] probes, string engine)
    {
        Assert.Equal(expected.Length, actual.Length);
        List<string> mismatches = [];
        long maxUlps = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            long ulps = UlpDistance(expected[i], actual[i]);
            maxUlps = Math.Max(maxUlps, ulps);
            if (ulps > tolUlps && !(MathF.Abs(expected[i] - actual[i]) <= absTol))
                mismatches.Add($"step {probes[i]}: host {expected[i]:R} vs {engine} {actual[i]:R} ({ulps} ulps)");
        }
        Assert.True(mismatches.Count == 0,
            $"{engine}: {mismatches.Count} steps beyond {tolUlps} ulps / abs {absTol:R} (max {maxUlps} ulps):\n"
            + string.Join("\n", mismatches.Take(20)));
    }

    /// <summary>Ulp distance between two float32 values; 0 covers ±0, NaN pairs never match.</summary>
    private static long UlpDistance(float a, float b)
    {
        if (a == b) return 0;
        if (float.IsNaN(a) || float.IsNaN(b)) return long.MaxValue;
        return Math.Abs(OrderedBits(a) - OrderedBits(b));
    }

    /// <summary>Maps float bits to a lexicographically ordered integer line.</summary>
    private static long OrderedBits(float f)
    {
        long bits = BitConverter.SingleToInt32Bits(f);
        return bits >= 0 ? bits : int.MinValue - bits;
    }

    // ───────────────────────────── probe builders ─────────────────────────────

    /// <summary>Every step in <c>[from, toInclusive]</c> — dense coverage across boundaries.</summary>
    private static IEnumerable<long> DenseRange(long from, long toInclusive)
    {
        for (long s = from; s <= toInclusive; s++) yield return s;
    }

    /// <summary>
    /// Large counters: past the float32 integer-exactness limit (2²⁴, probed densely around the
    /// edge) up to near int.MaxValue, pinning that the graph's int64→float32 conversion rounds
    /// identically to the host's int→float32 conversion.
    /// </summary>
    private static IEnumerable<long> LargeSteps =>
        [1 << 20, (1 << 24) - 2, (1 << 24) - 1, 1 << 24, (1 << 24) + 1, (1 << 24) + 2,
         (1L << 24) + 7, 1L << 28, 2_000_000_000, int.MaxValue - 8];

    // ───────────────────────────── factory shapes ─────────────────────────────

    [Fact]
    public void TestConstantParity()
        => AssertParity(Schedules.Constant(3e-4f), [.. DenseRange(0, 8), .. LargeSteps]);

    [Fact]
    public void TestLinearParity()
        => AssertParity(Schedules.Linear(3e-4f, 1e-5f, 1000), [.. DenseRange(0, 1050), .. LargeSteps]);

    [Fact]
    public void TestLinearRisingSmallTotalParity()
        => AssertParity(Schedules.Linear(-1f, 1f, 7), [.. DenseRange(0, 30), .. LargeSteps]);

    [Fact]
    public void TestCosineParity()
        => AssertParity(Schedules.Cosine(3e-4f, 1000), [.. DenseRange(0, 1050), .. LargeSteps],
            ortUlps: TranscendentalUlps);

    [Fact]
    public void TestCosineWithWarmupParity()
        => AssertParity(Schedules.CosineWithWarmup(3e-4f, 100, 1000), [.. DenseRange(0, 1150), .. LargeSteps],
            ortUlps: TranscendentalUlps);

    // warmupSteps <= 0 arm: WithWarmup(0) returns the cosine unchanged (and stays lowerable).
    [Fact]
    public void TestCosineWithZeroWarmupParity()
        => AssertParity(Schedules.CosineWithWarmup(1f, 0, 100), [.. DenseRange(0, 120)],
            ortUlps: TranscendentalUlps);

    // warmupSteps >= totalSteps arm: the decay window degenerates to Max(1, …) = 1 step.
    [Fact]
    public void TestCosineWithOversizedWarmupParity()
        => AssertParity(Schedules.CosineWithWarmup(1f, 20, 10), [.. DenseRange(0, 40)],
            ortUlps: TranscendentalUlps);

    [Fact]
    public void TestStepDecayParity()
        => AssertParity(Schedules.StepDecay(1e-2f, 30, 0.5f), [.. DenseRange(0, 400), .. LargeSteps],
            ortUlps: TranscendentalUlps);

    [Fact]
    public void TestExponentialParity()
        => AssertParity(Schedules.Exponential(1e-2f, 0.999f),
            [.. DenseRange(0, 200), 1000, 5000, 20000, 100000, .. LargeSteps],
            ortUlps: TranscendentalUlps);

    [Fact]
    public void TestOneCycleDefaultParity()
        => AssertParity(Schedules.OneCycle(0.1f, 1000), [.. DenseRange(0, 1100), .. LargeSteps],
            ortUlps: TranscendentalUlps);

    [Fact]
    public void TestOneCycleCustomParity()
        => AssertParity(Schedules.OneCycle(0.3f, 500, pctStart: 0.1f, divFactor: 10f, finalDivFactor: 100f),
            [.. DenseRange(0, 560)], ortUlps: TranscendentalUlps);

    // pctStart = 1 edge: the whole run is the rising phase; down degenerates to Max(1, 0) = 1.
    [Fact]
    public void TestOneCycleFullRiseParity()
        => AssertParity(Schedules.OneCycle(0.2f, 100, pctStart: 1f), [.. DenseRange(0, 120)],
            ortUlps: TranscendentalUlps);

    // ───────────────────────────── combinators ─────────────────────────────

    [Fact]
    public void TestScaleParity()
        => AssertParity(Schedules.Cosine(1f, 100).Scale(0.5f), [.. DenseRange(0, 120), .. LargeSteps],
            ortUlps: TranscendentalUlps);

    [Fact]
    public void TestClampParity()
        => AssertParity(Schedules.Linear(1f, 0f, 100).Clamp(0.2f, 0.8f), [.. DenseRange(0, 120), .. LargeSteps]);

    [Fact]
    public void TestShiftParity()
        => AssertParity(Schedules.Cosine(1f, 100).Shift(25), [.. DenseRange(0, 120)],
            ortUlps: TranscendentalUlps);

    // Negative shift: early steps read the inner schedule at negative positions, which the
    // clamped-progress shapes tolerate — the lowering must match that host behavior too.
    [Fact]
    public void TestNegativeShiftParity()
        => AssertParity(Schedules.Cosine(1f, 100).Shift(-10), [.. DenseRange(0, 130)],
            ortUlps: TranscendentalUlps);

    [Fact]
    public void TestPerEpochParity()
        => AssertParity(Schedules.Cosine(1f, 50).PerEpoch(10), [.. DenseRange(0, 600), .. LargeSteps],
            ortUlps: TranscendentalUlps);

    [Fact]
    public void TestWithWarmupParity()
        => AssertParity(Schedules.Cosine(1f, 200).WithWarmup(50), [.. DenseRange(0, 300), .. LargeSteps],
            ortUlps: TranscendentalUlps);

    [Fact]
    public void TestWithWarmupStartFactorParity()
        => AssertParity(Schedules.Cosine(1f, 200).WithWarmup(50, startFactor: 0.25f),
            [.. DenseRange(0, 300), .. LargeSteps], ortUlps: TranscendentalUlps);

    [Fact]
    public void TestThenParity()
        => AssertParity(Schedules.Constant(0.5f).Then(100, Schedules.Linear(0.5f, 0f, 200)),
            [.. DenseRange(0, 350), .. LargeSteps]);

    [Fact]
    public void TestNestedThenParity()
        => AssertParity(
            Schedules.Constant(1f).Then(50, Schedules.Cosine(1f, 100)).Then(150, Schedules.Constant(0.1f)),
            [.. DenseRange(0, 250), .. LargeSteps], ortUlps: TranscendentalUlps);

    // ───────────────────────────── compositions ─────────────────────────────

    [Fact]
    public void TestWarmupCosineThenFloorParity()
        => AssertParity(
            Schedules.Cosine(3e-4f, 900).WithWarmup(100).Then(1000, Schedules.Constant(1e-5f)),
            [.. DenseRange(0, 1100), .. LargeSteps], ortUlps: TranscendentalUlps);

    [Fact]
    public void TestOneCyclePerEpochScaleClampParity()
        => AssertParity(
            Schedules.OneCycle(0.4f, 100).PerEpoch(8).Scale(2f).Clamp(0.05f, 0.5f),
            [.. DenseRange(0, 900), .. LargeSteps], ortUlps: TranscendentalUlps);

    // PerEpoch inside a re-based Then branch: the epoch index must be derived from the
    // re-based step (host contract), which the in-graph step/stepsPerEpoch division preserves —
    // a single global epoch input could not express this composition.
    [Fact]
    public void TestPerEpochInsideThenBranchesParity()
        => AssertParity(
            Schedules.Cosine(1f, 40).PerEpoch(5).Then(300, Schedules.StepDecay(0.5f, 20, 0.7f).PerEpoch(3)),
            [.. DenseRange(0, 700)], ortUlps: TranscendentalUlps);

    [Fact]
    public void TestShiftedLinearWithWarmupParity()
        => AssertParity(Schedules.Linear(1f, 0f, 100).Shift(30).WithWarmup(20), [.. DenseRange(0, 200)]);

    // ───────────────────────────── counter precision ─────────────────────────────

    // Steps past 2²⁴ are not exactly representable in float32; the host casts the int counter
    // to float32 (round-to-nearest) before the schedule math, and the graph's int64→float32
    // Cast must round identically, keeping parity exact even where the ideal value drifts.
    [Fact]
    public void TestLargeCounterConversionParity()
        => AssertParity(Schedules.Linear(0f, 1f, 1 << 26),
            [.. DenseRange((1 << 24) - 3, (1 << 24) + 3), 1 << 25, (1 << 26) - 1, 1 << 26,
             (1 << 26) + 1, .. LargeSteps]);

    // ───────────────────────────── lowering contract ─────────────────────────────

    [Fact]
    public void TestOpaqueScheduleCannotLower()
    {
        // The public host-lambda constructor was removed (#99); an opaque, non-lowerable schedule
        // can still be built internally (expr: null) to pin the defensive non-lowerable path.
        Schedule opaque = new(s => s * 0.1f, expr: null);
        Assert.False(opaque.CanLower());
        // Opaqueness is contagious through every combinator…
        Assert.False(opaque.Scale(2f).CanLower());
        Assert.False(opaque.Clamp(0f, 1f).CanLower());
        Assert.False(opaque.Shift(5).CanLower());
        Assert.False(opaque.PerEpoch(10).CanLower());
        Assert.False(opaque.WithWarmup(5).CanLower());
        Assert.False(opaque.Then(10, Schedules.Constant(1f)).CanLower());
        Assert.False(Schedules.Constant(1f).Then(10, opaque).CanLower());
        // …while factory-built schedules stay lowerable through the same combinators.
        Assert.True(Schedules.Constant(1f).CanLower());
        Assert.True(Schedules.Cosine(1f, 10).WithWarmup(0).CanLower());
        Assert.True(Schedules.Cosine(1f, 10).PerEpoch(2).Then(5, Schedules.Constant(0f)).CanLower());

        var step = InputScalar<int64>("step");
        Assert.Throws<InvalidOperationException>(() => opaque.LowerToGraph(step));
    }

    // Bounds-swapped Clamp is rejected at construction: deferred to Math.Clamp it would throw
    // on every host At call, while the lowered Clip (numpy-style, max wins) would silently
    // produce values the host contract rejects.
    [Fact]
    public void TestClampRejectsSwappedBounds()
    {
        Assert.Throws<ArgumentException>(() => Schedules.Constant(0.5f).Clamp(1f, 0.2f));
        Assert.NotNull(Schedules.Constant(0.5f).Clamp(0.2f, 1f));
    }

    // ───────────────────────────── scalar contract + durability ─────────────────────────────

    /// <summary>
    /// The rank-0 counter contract (the shape a checkpoint would persist): per-step scalar
    /// graphs match the host across the boundaries of a warmup + cosine + Then composite, and
    /// the lowered graph survives the ONNX save/load and C# codegen roundtrips byte-identically
    /// (<see cref="AutoTest.TestGraph"/>) — the durability property motivating the spike.
    /// </summary>
    [Fact]
    public void TestScalarLoweringMatchesAndRoundtrips()
    {
        var schedule = Schedules.Cosine(3e-4f, 200).WithWarmup(50).Then(400, Schedules.Constant(1e-5f));
        long[] probes = [0, 1, 49, 50, 51, 199, 249, 250, 251, 399, 400, 401, 1000, 1 << 24];
        foreach (long p in probes)
        {
            var step = InputScalar<int64>("step");
            var graph = (new InternalComputationGraph([step], [schedule.LowerToGraph(step)]));
            float got = new ComputeContext().Execute(graph, TensorData([], p))[0]
                .ToTensorData().As<float32>().AccessMemory<float>()[0];
            float expected = schedule.At((int)p);
            long ulps = UlpDistance(expected, got);
            Assert.True(ulps <= TranscendentalUlps
                    || MathF.Abs(expected - got) <= 3e-4f * PeakRelativeTolerance,
                $"step {p}: host {expected:R} vs graph {got:R} ({ulps} ulps)");
        }

        var s2 = InputScalar<int64>("step");
        var g2 = (new InternalComputationGraph([s2], [schedule.LowerToGraph(s2)]));
        Assert.True(AutoTest.TestGraph(g2, sampleInputs: [TensorData([], 123L)],
            testQuickEngineExecution: true));
    }
}

using System;
using System.Linq;
using Shorokoo.Core.Rng;
using Shorokoo.Runtime;
using static Shorokoo.Tests.RngDrawRunners;
using static Shorokoo.Tests.Modules.QeeAuditVerdicts;

namespace Shorokoo.Tests;

/// <summary>Emits the in-graph runtime uniform draw at the input's shape (fixed key/substreamIndex).</summary>
[Module]
public partial class RtUniformDraw
{
    public static Tensor<float32> Inline(Tensor<float32> x)
        => RuntimeRng.StandardUniform(x.ShapeTensor(), Scalar(123UL | (456UL << 32)), Scalar(0UL));
}

/// <summary>Emits the in-graph runtime normal draw at the input's shape (fixed key/substreamIndex).</summary>
[Module]
public partial class RtNormalDraw
{
    public static Tensor<float32> Inline(Tensor<float32> x)
        => RuntimeRng.StandardNormal(x.ShapeTensor(), Scalar(7UL | (9UL << 32)), Scalar(0UL));
}

/// <summary>Emits a plain <c>Globals.RandomUniform</c> draw — routed through the SHRK_RANDOM
/// lowering (<c>FastLowerRandomOps</c>), i.e. the in-graph counter-based path, not ONNX's
/// RandomUniformLike.</summary>
[Module]
public partial class RtLoweredUniform
{
    public static Tensor<float32> Inline(Tensor<float32> x)
        => RandomUniform(x.ShapeTensor(), 0f, 1f);
}

/// <summary>A trainable weight plus a runtime RNG feed: the feed forces the framework to inject
/// the <c>RngExecutionCounter</c> as model state, while the draw is zeroed so the model's output
/// is exactly the linear transform.</summary>
[Module]
public partial class RtFcWithRngFeed
{
    public static Tensor<float32> Inline(Tensor<float32> input, [Hyper] Scalar<int64> numOutFeatures)
    {
        var numInFeatures = input.ShapeTensor()[-1L];
        var weights = Shorokoo.Tests.Modules.InitSimple.Init([numOutFeatures, numInFeatures]);
        var y = input.MatMul(weights.Transpose(1, 0));
        return y + RandomUniform(y.ShapeTensor(), 0f, 1f) * Scalar(0f);
    }
}

/// <summary>Emits the in-graph raw-bits draws (U8/U16/U32/U64) at the input's shape.</summary>
[Module] public partial class RtBitsU8Draw  { public static Tensor<uint8>  Inline(Tensor<float32> x) => RuntimeRng.BitsU8 (x.ShapeTensor(), Scalar(111UL | (222UL << 32)), Scalar(0UL)); }
[Module] public partial class RtBitsU16Draw { public static Tensor<uint16> Inline(Tensor<float32> x) => RuntimeRng.BitsU16(x.ShapeTensor(), Scalar(111UL | (222UL << 32)), Scalar(0UL)); }
[Module] public partial class RtBitsU32Draw { public static Tensor<uint32> Inline(Tensor<float32> x) => RuntimeRng.BitsU32(x.ShapeTensor(), Scalar(111UL | (222UL << 32)), Scalar(0UL)); }
[Module] public partial class RtBitsU64Draw { public static Tensor<uint64> Inline(Tensor<float32> x) => RuntimeRng.BitsU64(x.ShapeTensor(), Scalar(111UL | (222UL << 32)), Scalar(0UL)); }

/// <summary>A plain <c>Globals.RandomBits</c> feed — routed through the SHRK_RANDOM_BITS
/// lowering (id-bearing keyed draw), i.e. the public runtime raw-bits path.</summary>
[Module] public partial class RtLoweredBits   { public static Tensor<uint32> Inline(Tensor<float32> x) => RandomBits<uint32>(x.ShapeTensor()); }
[Module] public partial class RtLoweredBits64 { public static Tensor<uint64> Inline(Tensor<float32> x) => RandomBits<uint64>(x.ShapeTensor()); }

/// <summary>
/// The op/dtype pairs the region-table uniform draw depends on, checked in-graph so both ONNX
/// Runtime and the Quick Execution Engine must agree: a computed (non-constant) uint64 table
/// gathered with computed indices, the restoring long division that builds the threshold table,
/// the running max that holds each threshold above the last, and a binary search over the table
/// checked against a linear scan.
///
/// <para>The table is <c>uint64</c> because <c>Max</c> on <c>int64</c> mis-orders operands in
/// [2^31, 2^32) and the running max walks straight through that band. The division uses
/// arithmetic selection rather than <c>Where</c>, multiplication by two rather than
/// <c>BitShift</c>, and <c>Greater</c> negated rather than <c>GreaterOrEqual</c> or
/// <c>LessOrEqual</c> — a uint64 <c>Where</c> is unimplemented in ORT, and these are the forms
/// it does implement.</para>
/// </summary>
[Module]
public partial class RngRegionSelectionOpsCheck
{
    private static Tensor<uint64> AtLeast(Tensor<uint64> a, Tensor<uint64> b)
        => Scalar(1UL) - ((Tensor<bit>)OnnxOp.Greater(b, a)).Cast<uint64>();

    private static Tensor<uint64> LongDivide(Tensor<uint64> cumulative, Tensor<uint64> total, int bits)
    {
        var remainder = cumulative;
        var quotient = cumulative * Scalar(0UL);
        for (int i = 0; i < bits; i++)
        {
            remainder = remainder * Scalar(2UL);
            var ge = AtLeast(remainder, total);
            remainder = remainder - (ge * total);
            quotient = (quotient * Scalar(2UL)) + ge;
        }
        return quotient;
    }

    public static Scalar<bit> Inline(Tensor<int64> c, Tensor<int64> t, Tensor<int64> s, Tensor<float32> x)
    {
        var mismatch = IntMismatch(LongDivide(c.Cast<uint64>(), t.Cast<uint64>(), 41).Cast<int64>(),
            Vector(0L, 733007751850L, 1466015503701L, 314146179364L, 2199023255551L, 999556025250L, 1099511627775L));

        // A [128] table that cannot be constant-folded: it carries the runtime element count.
        var n = x.ShapeTensor().Reduce(ReduceKind.Prod);
        var slots = OnnxOp.Range(Scalar(0L), Scalar(128L), Scalar(1L)).int64();
        var stride = n + Scalar(9L);
        var table = (slots * stride).Cast<uint64>();          // T[0] = 0, stride carries the runtime count

        // Gather with computed, descending indices — a real gather, not a slice in disguise.
        var pick = Scalar(126L) - OnnxOp.Range(Scalar(0L), Scalar(128L), Scalar(18L)).int64();
        var gathered = ((Tensor<uint64>)OnnxOp.Gather(table, pick, axis: 0)).Cast<int64>() - (pick * stride);
        mismatch = mismatch + IntMismatch(gathered, Vector(0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L));

        // The seven-round running max, run twice over the same non-monotone sequence: in uint64
        // scaled across 2^31, and in int64 left well below it. Scaling commutes with a max, so the
        // two must agree — and they do not if int64 Max is used above 2^31, which is why the
        // product's table is unsigned.
        var wave = Scalar(63L) - (slots * slots) % Scalar(64L);
        var scale = Scalar(70_000_000UL);
        var big = (wave * stride).Cast<uint64>() * scale;
        foreach (long back in (long[])[1L, 2L, 4L, 8L, 16L, 32L, 64L])
        {
            var shifted = (slots - Scalar(back)).Max(Scalar(0L));
            big = big.Max(OnnxOp.Gather(big, shifted, axis: 0).uint64());
            wave = wave.Max(OnnxOp.Gather(wave, shifted, axis: 0).int64());
        }
        mismatch = mismatch + IntMismatch(
            (big - (wave * stride).Cast<uint64>() * scale).Cast<int64>(), Vector(new long[128]));

        // Binary search for the last slot whose threshold is <= s, against a linear scan.
        var sel = s.Cast<uint64>();
        var lo = s * Scalar(0L);
        foreach (long step in (long[])[64L, 32L, 16L, 8L, 4L, 2L, 1L])
        {
            var probe = OnnxOp.Gather(table, lo + Scalar(step), axis: 0).uint64();
            lo = lo + (AtLeast(sel, probe).Cast<int64>() * Scalar(step));
        }
        var scan = AtLeast(sel.Reshape(Vector(-1L, 1L)), table).Cast<int64>()
            .Reduce(ReduceKind.Sum, Vector(1L), keepDims: false) - Scalar(1L);
        mismatch = mismatch + IntMismatch(lo - scan, Vector(0L, 0L, 0L, 0L));

        return mismatch == Scalar(0L);
    }
}

/// <summary>
/// The dense arbitrary-range uniform draw, checked in-graph against host-computed expectations so
/// ONNX Runtime and the Quick Execution Engine must both reproduce <c>RngDenseUniformOracle</c>
/// bit for bit. The bounds arrive as a runtime tensor, so nothing specializes on their values; a
/// batch of ranges shares one graph because the per-graph overheads dominate a table build.
/// </summary>
[Module]
public partial class RngDenseUniformOracleCheck
{
    public const int Ranges = 6, Draws = 32;

    public static Scalar<bit> Inline(Tensor<float32> bounds, Tensor<float32> expected)
    {
        var b = bounds.Vec();
        var want = expected.Vec();
        Scalar<int64> mismatch = Scalar(0L);
        for (long r = 0; r < Ranges; r++)
        {
            var drawn = RuntimeRng.Uniform(Vector((long)Draws),
                Scalar(0xA5A5_1234UL | (0x9E37UL << 32)), Scalar(0UL), b[2 * r], b[2 * r + 1]);
            var target = want.Slice(Scalar(r * Draws), Scalar(r * Draws + Draws));
            var differs = ((Tensor<bit>)OnnxOp.Not(OnnxOp.Equal(drawn, target))).Cast<int64>();
            var bothNaN = ((Tensor<bit>)OnnxOp.And(OnnxOp.IsNaN(drawn), OnnxOp.IsNaN(target))).Cast<int64>();
            mismatch = mismatch + (differs * (Scalar(1L) - bothNaN))
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
        return mismatch == Scalar(0L);
    }
}

/// <summary>
/// The selector thresholds the graph builds for a batch of ranges, [Ranges * 128], so a test can
/// hold them against the oracle's. Sampling draws cannot: a held-up threshold may own one selector
/// code in 2^41, and every draw-based check agrees with a table that has dropped it.
/// </summary>
[Module]
public partial class RngDenseThresholdTable
{
    public const int Ranges = 6, Slots = 128;

    public static Tensor<uint64> Inline(Tensor<float32> bounds)
    {
        var b = bounds.Vec();
        var built = RuntimeRng.BuildDenseTable(b[0], b[1]).Threshold;
        for (long r = 1; r < Ranges; r++)
            built = built.Concat(0, RuntimeRng.BuildDenseTable(b[2 * r], b[2 * r + 1]).Threshold);
        return built;
    }
}

/// <summary>
/// The in-graph counter-based runtime RNG (<see cref="RuntimeRng"/>): the ONNX-op Threefry
/// subgraph must reproduce the host generator (<see cref="Threefry2x32"/>) bit-for-bit —
/// execution-provider-independent — and produce well-distributed draws.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngRuntimeTests
{
    private const ulong BitsKey = 111UL | (222UL << 32);
    private const ulong DenseKey = 0xA5A5_1234UL | (0x9E37UL << 32);
    private const ulong UniformKey = 123UL | (456UL << 32);
    private const ulong NormalKey = 7UL | (9UL << 32);

    // Host reference for the runtime scheme: substreamIndex folds into the key, element i
    // indexes the whole counter; uniform = low 24 bits of x0 * 2^-24.
    private static float HostUniform(long i, ulong key, ulong substreamIndex)
        => RngTestOracle.DrawUniform(key, substreamIndex, i);

    // Host reference for the raw-bits scheme: E = 64/W elements pack into each generator value,
    // low lane first, so element i is lane i%E of the value at position i/E.
    private static ulong HostBits(long i, int width, ulong key, ulong substreamIndex)
        => RngTestOracle.DrawBits(key, substreamIndex, i, width);

    private static readonly (float Low, float High)[] DenseRanges =
    [
        (0f, 1f), (-1f, 1f), (4f, 12f), (4f, 8f), (1f, 2f), (-1f, 0f), (0.1f, 0.3f),
        (-0.0625f, 0.0625f), (1e-30f, 1e-20f), (0f, float.MaxValue), (-float.MaxValue, float.MaxValue),
        (-1e-40f, 1e-40f), (1e-45f, 1e-44f), (0f, float.Epsilon), (0f, 1.1754944e-38f),
        (5e-39f, 3e-38f), (-7.888609e-31f, 1.8446744e19f), (1f, 1.0000001f), (1e30f, 1.0000001e30f),
        (-float.MaxValue, -1.7e38f), (-0.1f, 0.3f), (3f, 3.5f), (100f, 1000f), (0f, float.PositiveInfinity),
        // Straddling the truncation floor: one endpoint inside the collapsed band, the other far
        // above it — the layouts where the band is partial on exactly one side, with and without
        // the off-lattice stubs (-5·2^39 is exactly on the lattice these ranges induce).
        (1e-30f, 1e30f), (-1e-30f, 1e30f), (-2748779069440f, 1e30f), (-1e30f, -2748779069440f),
        (-1e30f, -1e-30f), (-1e30f, 1e-30f), (-1e30f, 1e15f), (-1e15f, 1e30f),
        (1e15f, 1e30f), (-1e30f, 0f), (0f, 1e30f), (-1.5e-45f, 3f),
    ];

    private static long DenseSignedOrdinal(float x)
    {
        uint b = BitConverter.SingleToUInt32Bits(x);
        return (b & 0x8000_0000u) != 0 ? -(long)(b & 0x7FFF_FFFFu) : (long)b;
    }

    [Fact]
    public void TestDenseUniformOracleNeverLeavesTheRequestedRange()
    {
        foreach (var (low, high) in DenseRanges)
        {
            var table = RngDenseUniformOracle.Build(low, high);
            for (long i = 0; i < 400; i++)
            {
                float v = RngDenseUniformOracle.Draw(DenseKey, 0, i, low, high);
                Assert.True(v >= low && v < high);
                Assert.False(float.IsNaN(v) || float.IsInfinity(v));
            }
            Assert.True(table.Slots.Length <= RngDenseUniformOracle.MaxSlots);
        }
    }

    [Fact]
    public void TestDenseUniformOracleTableIsAWellFormedPartition()
    {
        foreach (var (low, high) in DenseRanges)
        {
            var table = RngDenseUniformOracle.Build(low, high);
            Assert.Equal(0, table.Slots[0].Threshold);
            Assert.True(table.Total > 0 && table.Total <= 1L << 62);
            long weights = 0;
            for (int i = 0; i < table.Slots.Length; i++)
            {
                weights += table.Slots[i].Weight;
                Assert.True(table.Slots[i].Weight > 0);
                Assert.InRange(table.Slots[i].IndexBits, 0, RngDenseUniformOracle.P);
                if (i > 0) Assert.True(table.Slots[i].Threshold > table.Slots[i - 1].Threshold);
                Assert.InRange(table.Slots[i].Threshold, 0, (1L << RngDenseUniformOracle.SelectorBits) - 1);
            }
            Assert.Equal(table.Total, weights);
        }
    }

    [Fact]
    public void TestDenseUniformOracleReachesEveryFloatWhereNothingIsTruncated()
    {
        foreach (var (low, high) in (( float Low, float High)[])[(4f, 12f), (4f, 8f), (1f, 2f), (0.1f, 0.3f), (3f, 3.5f), (100f, 1000f)])
        {
            var table = RngDenseUniformOracle.Build(low, high);
            long elements = 0;
            foreach (var slot in table.Slots) elements += 1L << slot.IndexBits;
            Assert.Equal(DenseSignedOrdinal(high) - DenseSignedOrdinal(low), elements);
        }
    }

    // On [0,1) above the truncation floor the dense draw IS Walker/Reynolds, bit for bit off the
    // same draw value — that identity is what fixes the 41/23 split. It does NOT hold below the
    // floor, where Walker keeps three more geometric binades down to 2^-41 and this draw switches
    // to an even lattice reaching 2^-61 and exact zero. The crossover is the first ordinal slot's
    // threshold, 8, so the disagreeing selectors carry probability 2^-38 — far past what sampling
    // random draws can reach, which is why both sides are asserted directly.
    [Fact]
    public void TestDenseUniformOracleIsWalkerReynoldsAboveTheTruncationFloor()
    {
        var table = RngDenseUniformOracle.Build(0f, 1f);
        long crossover = table.Slots[1].Threshold;
        Assert.Equal(8, crossover);

        uint Dense(long selector, ulong mantissa)
            => RngDenseUniformOracle.SampleBits(table, ((ulong)selector << RngDenseUniformOracle.P) | mantissa);
        uint Walker(long selector, ulong mantissa)
            => BitConverter.SingleToUInt32Bits(
                RngTestOracle.WalkerUniform(((ulong)selector << RngDenseUniformOracle.P) | mantissa));

        foreach (ulong m in (ulong[])[0UL, 1UL, 4919UL, 8388607UL])
        {
            for (int b = 3; b <= 40; b++)
                foreach (long s in (long[])[1L << b, (1L << b) + 1, (1L << (b + 1)) - 1])
                    Assert.Equal(Walker(s, m), Dense(s, m));
            for (long s = 0; s < crossover; s++)
                Assert.NotEqual(Walker(s, m), Dense(s, m));
        }
    }

    [Fact]
    public void TestDenseUniformOracleGivesEveryWeightedSlotSelectorCodes()
    {
        foreach (var (low, high) in DenseRanges)
        {
            var table = RngDenseUniformOracle.Build(low, high);
            for (int i = 0; i < table.Slots.Length; i++)
            {
                long end = i + 1 < table.Slots.Length
                    ? table.Slots[i + 1].Threshold : 1L << RngDenseUniformOracle.SelectorBits;
                Assert.True(end > table.Slots[i].Threshold);
            }
        }
    }

    [Fact]
    public void TestDenseUniformOracleSplitsSymmetricRangesExactlyInHalf()
    {
        for (int e = -60; e <= 60; e++)
        {
            float bound = MathF.ScaleB(1f, e);
            var table = RngDenseUniformOracle.Build(-bound, bound);
            long negative = 0;
            for (int i = 0; i < table.Slots.Length; i++)
            {
                long end = i + 1 < table.Slots.Length ? table.Slots[i + 1].Threshold : 1L << RngDenseUniformOracle.SelectorBits;
                if (table.Slots[i].Base < 0) negative += end - table.Slots[i].Threshold;
            }
            Assert.Equal(1L << (RngDenseUniformOracle.SelectorBits - 1), negative);
        }
    }

    [Fact]
    public void TestDenseUniformOracleHandlesDegenerateAndNonFiniteBounds()
    {
        Assert.Equal(3f, RngDenseUniformOracle.Draw(DenseKey, 0, 0, 3f, 3f));
        Assert.Equal(7f, RngDenseUniformOracle.Draw(DenseKey, 0, 0, 7f, 3f));
        Assert.True(float.IsNaN(RngDenseUniformOracle.Draw(DenseKey, 0, 0, float.NaN, 1f)));
        Assert.True(float.IsNaN(RngDenseUniformOracle.Draw(DenseKey, 0, 0, 1f, float.NaN)));
        for (long i = 0; i < 200; i++)
        {
            float wide = RngDenseUniformOracle.Draw(DenseKey, 0, i, float.NegativeInfinity, float.PositiveInfinity);
            Assert.False(float.IsNaN(wide) || float.IsInfinity(wide));
            float single = RngDenseUniformOracle.Draw(DenseKey, 0, i, 1f, 1.0000001f);
            Assert.Equal(1f, single);
        }
    }

    private static bool DenseMatchesOracle((float Low, float High)[] ranges, bool roundtrip = false)
    {
        const int n = RngDenseUniformOracleCheck.Draws;
        float[] bounds = new float[ranges.Length * 2];
        float[] expected = new float[ranges.Length * n];
        for (int r = 0; r < ranges.Length; r++)
        {
            bounds[2 * r] = ranges[r].Low;
            bounds[2 * r + 1] = ranges[r].High;
            for (long i = 0; i < n; i++)
                expected[r * n + i] = RngDenseUniformOracle.Draw(DenseKey, 0, i, ranges[r].Low, ranges[r].High);
        }
        return AutoTest.AdvancedTestGraph<RngDenseUniformOracleCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData([bounds.Length], bounds), TensorData([expected.Length], expected)],
            testOnnxRoundtrip: roundtrip, testCsRoundtrip: false);
    }

    private static ulong[] GraphThresholds((float Low, float High)[] ranges)
    {
        float[] bounds = new float[RngDenseThresholdTable.Ranges * 2];
        for (int r = 0; r < ranges.Length; r++)
            (bounds[2 * r], bounds[2 * r + 1]) = (ranges[r].Low, ranges[r].High);
        var g = ((ComputationGraph)typeof(RngDenseThresholdTable)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var input = TensorData([(long)bounds.Length], bounds);
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([input])).ToConcreteModel();
        return [.. ComputeContext.Default.Execute(concrete, input)[0].ToTensorData()
            .As<uint64>().AccessMemory().ToArray()];
    }

    // The graph's [128] table interleaves empty slots among the oracle's live ones, each carrying
    // the following live slot's threshold; the trailing empties carry 2^41. So the graph's
    // distinct thresholds, minus that top, are exactly the oracle's.
    private static void AssertThresholdsMatchOracle((float Low, float High)[] ranges)
    {
        ulong top = 1UL << RngDenseUniformOracle.SelectorBits;
        ulong[] graph = GraphThresholds(ranges);
        for (int r = 0; r < ranges.Length; r++)
        {
            var table = RngDenseUniformOracle.Build(ranges[r].Low, ranges[r].High);
            if (table.Fixed is not null) continue;
            ulong[] got = [.. graph.Skip(r * RngDenseThresholdTable.Slots)
                .Take(RngDenseThresholdTable.Slots).Distinct().Where(t => t != top).Order()];
            Assert.Equal([.. table.Slots.Select(s => (ulong)s.Threshold)], got);
        }
    }

    [Fact]
    public void TestInGraphDenseTableThresholdsMatchTheOracleSlotForSlot()
    {
        for (int b = 0; b * RngDenseThresholdTable.Ranges < DenseRanges.Length; b++)
            AssertThresholdsMatchOracle(DenseBatch(b));
    }

    private static (float Low, float High)[] DenseBatch(int index)
        => [.. DenseRanges.Skip(index * RngDenseUniformOracleCheck.Ranges).Take(RngDenseUniformOracleCheck.Ranges)];

    [Fact]
    public void TestInGraphDenseUniformMatchesTheOracleOnAdversarialRanges()
    {
        Assert.True(DenseMatchesOracle(DenseBatch(0), roundtrip: true));
        Assert.True(DenseMatchesOracle(DenseBatch(1)));
    }

    [Fact]
    public void TestInGraphDenseUniformMatchesTheOracleOnSubnormalAndFullDomainRanges()
    {
        Assert.True(DenseMatchesOracle(DenseBatch(2)));
        Assert.True(DenseMatchesOracle(DenseBatch(3)));
    }

    [Fact]
    public void TestInGraphDenseUniformMatchesTheOracleAcrossTheTruncationFloor()
    {
        Assert.True(DenseMatchesOracle(DenseBatch(4)));
        Assert.True(DenseMatchesOracle(DenseBatch(5)));
    }

    private static (float Low, float High) StraddlingNegativePowerOfTwo(int exponent)
    {
        float p = MathF.ScaleB(-1f, exponent);
        return (MathF.BitDecrement(p), MathF.BitIncrement(p));
    }

    [Fact]
    public void TestInGraphDenseUniformMatchesTheOracleWhenTheTopFloatIsANegativePowerOfTwo()
        => Assert.True(DenseMatchesOracle([.. (( int[])[0, 1, -1, 10, -20, 60]).Select(StraddlingNegativePowerOfTwo)]));

    [Fact]
    public void TestInGraphDenseUniformMatchesTheOracleOnDegenerateAndNonFiniteBounds()
    {
        Assert.True(DenseMatchesOracle([
            (3f, 3f), (7f, 3f), (float.NaN, 1f), (1f, float.NaN), (float.NaN, float.NaN), (0f, 0f)]));
        Assert.True(DenseMatchesOracle([
            (float.NegativeInfinity, float.PositiveInfinity), (float.NegativeInfinity, 0f),
            (-1f, float.PositiveInfinity), (float.PositiveInfinity, float.NegativeInfinity),
            (-0f, 0f), (1f, 1.0000001f)]));
    }

    [Fact]
    public void TestRegionSelectionOpsAgreeOnBothEngines()
    {
        Assert.True(AutoTest.AdvancedTestGraph<RngRegionSelectionOpsCheck>(
            hyperparamInputs: [],
            runtimeInputs: [
                TensorData(DType.Int64, [7L], 0L, 1L, 2L, 1L, (1L << 62) - 1, 5L, 1L << 40),
                TensorData(DType.Int64, [7L], 3L, 3L, 3L, 7L, 1L << 62, 11L, (1L << 41) + 1),
                TensorData(DType.Int64, [4L], 0L, 17L, 1000L, 1L << 40),
                TensorData([8L], new float[8])]));
    }

    [Fact]
    public void TestInGraphBitsAndUniformDrawsMatchTheHostGeneratorBitExactly()
    {
        var u8 = RunDrawRaw<RtBitsU8Draw>(4, 4);
        Assert.Equal(DType.UInt8, u8.DType);
        var u8v = u8.As<uint8>().AccessMemory().ToArray();
        for (long i = 0; i < 16; i++) Assert.Equal((byte)HostBits(i, 8, BitsKey, 0), u8v[i]);

        var u16 = RunDrawRaw<RtBitsU16Draw>(4, 4);
        Assert.Equal(DType.UInt16, u16.DType);
        var u16v = u16.As<uint16>().AccessMemory().ToArray();
        for (long i = 0; i < 16; i++) Assert.Equal((ushort)HostBits(i, 16, BitsKey, 0), u16v[i]);

        var u32 = RunDrawRaw<RtBitsU32Draw>(4, 4);
        Assert.Equal(DType.UInt32, u32.DType);
        var u32v = u32.As<uint32>().AccessMemory().ToArray();
        for (long i = 0; i < 16; i++) Assert.Equal((uint)HostBits(i, 32, BitsKey, 0), u32v[i]);

        var u64 = RunDrawRaw<RtBitsU64Draw>(4, 4);
        Assert.Equal(DType.UInt64, u64.DType);
        var u64v = u64.As<uint64>().AccessMemory().ToArray();
        for (long i = 0; i < 16; i++) Assert.Equal(HostBits(i, 64, BitsKey, 0), u64v[i]);

        var vals = RunDraw<RtUniformDraw>(4, 4);
        Assert.Equal(16, vals.Length);
        for (long i = 0; i < 16; i++) Assert.Equal(HostUniform(i, UniformKey, 0), vals[i]);

        // The normal carries a tolerance: its Ln/Sqrt/Cos/Sin kernels are EP-approximate, unlike
        // the integer bits path and the exactly-constructed uniform above.
        var normals = RunDraw<RtNormalDraw>(4, 4);
        Assert.Equal(16, normals.Length);
        for (long i = 0; i < 16; i++)
            Assert.Equal(RngTestOracle.DrawNormal(NormalKey, 0, i), normals[i], 1e-5f);
    }

    [Fact]
    public void TestBitsDrawsPackLanesLowFirstIntoTheSixtyFourBitDrawValueAndSliceTheTail()
    {
        var values = RunDrawRaw<RtBitsU64Draw>(4, 4).As<uint64>().AccessMemory().ToArray();
        var u8 = RunDrawRaw<RtBitsU8Draw>(4, 4).As<uint8>().AccessMemory().ToArray();
        var u16 = RunDrawRaw<RtBitsU16Draw>(4, 4).As<uint16>().AccessMemory().ToArray();
        var u32 = RunDrawRaw<RtBitsU32Draw>(4, 4).As<uint32>().AccessMemory().ToArray();

        void AssertPacksInto(int width, Func<int, ulong> element)
        {
            int lanes = 64 / width;
            for (int j = 0; j < 16 / lanes; j++)
            {
                ulong packed = 0;
                for (int l = 0; l < lanes; l++) packed |= element(j * lanes + l) << (l * width);
                Assert.Equal(values[j], packed);
            }
        }

        AssertPacksInto(8, i => u8[i]);
        AssertPacksInto(16, i => u16[i]);
        AssertPacksInto(32, i => u32[i]);

        Assert.Equal(u8.Take(5).ToArray(), RunDrawRaw<RtBitsU8Draw>(1, 5).As<uint8>().AccessMemory().ToArray());
        Assert.Equal(u16.Take(5).ToArray(), RunDrawRaw<RtBitsU16Draw>(1, 5).As<uint16>().AccessMemory().ToArray());
        Assert.Equal(u32.Take(5).ToArray(), RunDrawRaw<RtBitsU32Draw>(1, 5).As<uint32>().AccessMemory().ToArray());
    }

    [Fact]
    public void TestInGraphUniformIsInRangeAndSpreadAndNormalHasStandardMoments()
    {
        var uniform = RunDraw<RtUniformDraw>(8, 8);
        Assert.All(uniform, v => Assert.InRange(v, 0.0f, 0.99999997f));
        Assert.InRange(uniform.Average(), 0.4f, 0.6f);

        var normal = RunDraw<RtNormalDraw>(40, 40);
        double mean = normal.Average();
        double variance = normal.Select(v => (v - mean) * (v - mean)).Average();
        Assert.InRange(mean, -0.1, 0.1);
        Assert.InRange(variance, 0.85, 1.15);
    }

    [Fact]
    public void TestLoweredFeedsAreDeterministicAndKeyedUnderTheDefaultIdentity()
    {
        // A plain Globals.RandomUniform draw lowers to the in-graph counter-based RNG, so it is
        // bit-reproducible across executions (ONNX RandomUniformLike advanced state per Run).
        var a = RunDraw<RtLoweredUniform>(8, 8);
        Assert.Equal(64, a.Length);
        Assert.Equal(a, RunDraw<RtLoweredUniform>(8, 8));
        Assert.All(a, v => Assert.InRange(v, 0.0f, 0.99999997f));
        Assert.InRange(a.Average(), 0.3f, 0.7f);

        // "No config" means the DEFAULT deterministic identity (master seed 0), never the ONNX
        // random fallback: the draws are bit-exactly the host fold of the default runtime
        // master along the feed's ModelId (slot 1) — reconstructible offline.
        var g = ((ComputationGraph)typeof(RtLoweredUniform)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var input = TensorData([4L, 4L], Enumerable.Repeat(0f, 16).ToArray());
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([input])).ToConcreteModel();
        Assert.NotNull(concrete.TryGetRngSeed());
        var defaultKey = RngTestOracle.RunKey(RngConfig.Default, [1]);
        var vals = ComputeContext.Default.Execute(concrete, input)[0]
            .ToTensorData().As<float32>().AccessMemory().ToArray();
        for (long i = 0; i < 16; i++) Assert.Equal(HostUniform(i, defaultKey, 0), vals[i]);

        var bits = RunDrawRaw<RtLoweredBits>(4, 4);
        Assert.Equal(DType.UInt32, bits.DType);
        var bv = bits.As<uint32>().AccessMemory().ToArray();
        Assert.Equal(bv, RunDrawRaw<RtLoweredBits>(4, 4).As<uint32>().AccessMemory().ToArray());
        for (long i = 0; i < 16; i++) Assert.Equal((uint)HostBits(i, 32, defaultKey, 0), bv[i]);

        // The U64 path (unsigned BitShift + BitwiseOr above the int64 range) must survive the
        // full public feed -> keyed draw -> width-specialized function call.
        var bits64 = RunDrawRaw<RtLoweredBits64>(4, 4);
        Assert.Equal(DType.UInt64, bits64.DType);
        var b64 = bits64.As<uint64>().AccessMemory().ToArray();
        for (long i = 0; i < 16; i++) Assert.Equal(HostBits(i, 64, defaultKey, 0), b64[i]);
    }

    [Fact]
    public void TestBitsFeedIsLabelledAsSuchAndRebindingReplacesOnlyTheIdentityValue()
    {
        // A bits feed must be classified and described as a "bits feed", not silently as the
        // "normal feed" default the RngStreamKind switches fall through to.
        var bg = ((ComputationGraph)typeof(RtLoweredBits)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var bitsInput = TensorData([4L, 4L], Enumerable.Repeat(0f, 16).ToArray());
        var report = bg.ToConcreteArchitecture(bg.FromOrderedInputs([bitsInput])).GetRngStreamReport();
        var bitsStreams = report.Streams.Where(s => s.Kind == RngStreamKind.BitsFeed).ToList();
        Assert.NotEmpty(bitsStreams);
        Assert.Contains("bits feed", report.ToString());
        Assert.Contains("bits feed", report.EmitPinSkeleton());
        Assert.DoesNotContain("normal feed", report.ToString());

        // Re-binding is the RngSeed parameter's re-initialization: every draw's key — a split
        // chain rooted at that parameter — re-derives from it, with no node added or removed.
        var g = ((ComputationGraph)typeof(RtLoweredUniform)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var input = TensorData([4L, 4L], Enumerable.Repeat(0f, 16).ToArray());
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([input]))
            .ToConcreteModel(new RngConfig { MasterSeed = 1 });

        float[] Run() => ComputeContext.Default.Execute(concrete, input)[0]
            .ToTensorData().As<float32>().AccessMemory().ToArray();

        int nodeCount = concrete.Nodes.Count;
        Assert.Contains(concrete.Nodes, n => n.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM);
        var underSeed1 = Run();

        concrete.ApplyRngConfig(new RngConfig { MasterSeed = 2 });
        Assert.Equal(nodeCount, concrete.Nodes.Count);
        Assert.Contains(concrete.Nodes, n => n.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM);
        Assert.NotEqual(underSeed1, Run());

        concrete.ApplyRngConfig(new RngConfig { MasterSeed = 1 });
        Assert.Equal(underSeed1, Run());   // re-binding is exact, not approximate
    }

    [Fact]
    public void TestSplitIndexAndDrawPositionUseTheWholeSixtyFourBitRange()
    {
        static ulong Split(ulong k, ulong index)
        {
            var node = InternalOp.RngSplit(Scalar(k), Scalar(index), RngAlgorithms.Default);
            return ComputeContext.Default.Execute(new InternalComputationGraph([], [node]))[0]
                .ToTensorData().As<uint64>().AccessMemory().ToArray()[0];
        }

        // SHRK_RNG_SPLIT folds its parent key input with the index, bit-exact with the host
        // bijection (the split function is the versioned in-graph derivation primitive).
        var (px0, px1) = Threefry2x32.Bijection(7u, 0u, 1u, 2u);
        Assert.Equal(px0 | ((ulong)px1 << 32), Split(1UL | (2UL << 32), 7UL));

        // Distinct indices give distinct children over the ENTIRE range. The first two pairs
        // ALIAS under a 32-bit index (same low word, different high word); the third is in the
        // top half, where a signed reading would go negative. key's high bit is set too.
        const ulong key = 0x8000_0000_0000_0001UL;
        (ulong a, ulong b)[] pairs =
        [
            (7UL, 7UL + (1UL << 32)),
            (0UL, 1UL << 32),
            (0xFFFF_FFFF_FFFF_FFFEUL, 0xFFFF_FFFEUL),
        ];
        foreach (var (ia, ib) in pairs)
        {
            var a = Split(key, ia);
            var b = Split(key, ib);
            Assert.Equal(RngTestOracle.FoldKey(key, ia), a);
            Assert.Equal(RngTestOracle.FoldKey(key, ib), b);
            Assert.NotEqual(a, b);
        }

        // substreamIndex is the execution counter; under a 32-bit counter word, execution 2^32
        // repeated execution 0's draw exactly.
        const ulong drawKey = 0xDEAD_BEEF_FEED_FACEUL;
        static float[] Draw(ulong substreamIndex)
        {
            var g = new InternalComputationGraph([],
                [RuntimeRng.StandardUniform(Vector(4L), Scalar(drawKey), Scalar(substreamIndex))]);
            return ComputeContext.Default.Execute(g)[0]
                .ToTensorData().As<float32>().AccessMemory().ToArray();
        }

        var atZero = Draw(0);
        var atTwoPow32 = Draw(1UL << 32);
        var atTop = Draw(0xFFFF_FFFF_FFFF_FFFFUL);
        Assert.NotEqual(atZero, atTwoPow32);
        Assert.NotEqual(atZero, atTop);
        for (long i = 0; i < 4; i++)
        {
            Assert.Equal(RngTestOracle.DrawUniform(drawKey, 0, i), atZero[i]);
            Assert.Equal(RngTestOracle.DrawUniform(drawKey, 1UL << 32, i), atTwoPow32[i]);
            Assert.Equal(RngTestOracle.DrawUniform(drawKey, 0xFFFF_FFFF_FFFF_FFFFUL, i), atTop[i]);
        }
    }
}

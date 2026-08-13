using System;
using System.Collections.Generic;
using System.Linq;
using Shorokoo.Core.Inference;
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
/// A characterization pin on uint64 ops in ONNX Runtime: a computed (non-constant) uint64 table
/// gathered with computed indices, a restoring long division, a running max, and a binary search
/// over the table checked against a linear scan.
///
/// <para><b>The product no longer emits any of this.</b> The dense uniform draw builds no table,
/// divides not at all, and finds its block by counting seven thresholds. What survives is the
/// characterization: these behaviours were expensive to establish and nothing else records them.
/// The running max is why the int64-Max finding can say uint64 is the safe
/// width — it runs the same non-monotone scan twice, unsigned across 2^31 and signed below it, and
/// they agree only because the unsigned one is correct. The division uses arithmetic selection
/// rather than <c>Where</c>, multiplication by two rather than <c>BitShift</c>, and <c>Greater</c>
/// negated rather than <c>GreaterOrEqual</c>, because a uint64 <c>Where</c> is unimplemented in ORT
/// and these are the forms it does implement.</para>
///
/// <para>ONNX Runtime is the only engine that checks the result. <c>AutoTest</c> evaluates the
/// self-check bool on the default (ORT-backed) context; its Quick Execution Engine pass only
/// asserts that every output resolves to a valid dtype, and never reads the bool. A QEE that
/// disagreed on every op here would still pass — see Shorokoo#159.</para>
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
/// ONNX Runtime must reproduce <c>RngDenseUniformOracle</c> bit for bit. The bounds arrive as a
/// runtime tensor, so nothing specializes on their values; a batch of ranges shares one graph
/// because the per-graph overheads dominate a table build.
///
/// <para>The Quick Execution Engine runs the same graph but its result is not compared:
/// <c>AutoTest</c> only asserts that every output resolves to a valid dtype there, so this pins
/// the QEE's op coverage, not its values. Shorokoo#159 tracks closing that in general; the dense
/// draw's QEE values are checked through <see cref="RngDenseUniformOutput"/> instead.</para>
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
                Scalar(0xA5A5_1234UL | (0x9E37UL << 32)), Scalar((ulong)r), b[2 * r], b[2 * r + 1]);
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
/// The block thresholds the graph builds for a batch of ranges, [Ranges * 7], so a test can hold
/// them against the oracle's. Sampling draws cannot: a held-up threshold may own one code in 2^64,
/// and every draw-based check agrees with a table that has dropped it.
/// </summary>
[Module]
public partial class RngDenseThresholdTable
{
    public const int Ranges = 6, Slots = 7;

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
/// The dense draw's raw float32 output for a batch of ranges, so a test can read its bits
/// host-side. Nothing in-graph can: opset 21 has no bit-reinterpretation op, and <c>Equal</c>
/// forgives a NaN's payload and the sign of zero alike. Reading the output rather than a
/// self-check bool is also what lets the Quick Execution Engine be held to the same bits.
/// </summary>
[Module]
public partial class RngDenseUniformOutput
{
    public const int Ranges = 6, Draws = 4;

    public static Tensor<float32> Inline(Tensor<float32> bounds)
    {
        var b = bounds.Vec();
        var key = Scalar(0xA5A5_1234UL | (0x9E37UL << 32));
        var drawn = RuntimeRng.Uniform(Vector((long)Draws), key, Scalar(0UL), b[0], b[1]);
        for (long r = 1; r < Ranges; r++)
            drawn = drawn.Concat(0, RuntimeRng.Uniform(
                Vector((long)Draws), key, Scalar((ulong)r), b[2 * r], b[2 * r + 1]));
        return drawn;
    }
}

/// <summary>
/// The dense draw over [0,1) against <see cref="RuntimeRng.StandardUniform"/>, in-graph and off the
/// same key, so they consume the same generator values. Above the truncation floor they must agree
/// exactly: [0,1) does not straddle zero, so it keeps 41 weight classes and totals exactly 2^64,
/// which makes the scaling the identity and the block decode Walker/Reynolds' own. They part ways
/// only on draws below 2^23 — probability 2^-41 per element, unreachable by sampling — where the
/// lattice reaches exact zero and Walker instead folds everything into his bottom binade.
/// </summary>
[Module]
public partial class RngDenseIsWalkerOnTheUnitInterval
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var shape = x.ShapeTensor();
        var key = Scalar(0xA5A5_1234UL | (0x9E37UL << 32));
        var dense = RuntimeRng.Uniform(shape, key, Scalar(0UL), Scalar(0f), Scalar(1f));
        var walker = RuntimeRng.StandardUniform(shape, key, Scalar(0UL));
        return (dense - walker).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar() == Scalar(0f);
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
    // indexes the whole counter; uniform is Walker's geometric transform of the whole 64-bit value.
    private static float HostUniform(long i, ulong key, ulong substreamIndex)
        => RngTestOracle.DrawUniform(key, substreamIndex, i);

    // Host reference for a lowered Globals.RandomUniform, which is the dense arbitrary-range draw
    // over [0,1) rather than StandardUniform, so the dense oracle is its reference.
    private static float HostDenseUniform(long i, ulong key, ulong substreamIndex)
        => RngDenseUniformOracle.Draw(key, substreamIndex, i, 0f, 1f);

    // Host reference for the raw-bits scheme: E = 64/W elements pack into each generator value,
    // low lane first, so element i is lane i%E of the value at position i/E.
    private static ulong HostBits(long i, int width, ulong key, ulong substreamIndex)
        => RngTestOracle.DrawBits(key, substreamIndex, i, width);

    // One ordinal below 2^-37, the floor boundary a range topping out in weight class 130 induces.
    private static readonly float FloorBoundaryNeighbour = MathF.BitDecrement(MathF.ScaleB(1f, -37));

    // The adversarial ranges, in named groups of RngDenseUniformOracleCheck.Ranges. One group is
    // one graph and one Fact — building a graph costs far more than the ranges in it, and one wide
    // graph over all of them measured slower than eight narrow ones. Naming the groups rather than
    // slicing a flat list is what keeps the Fact names true when a range is added.
    private static readonly (float Low, float High)[] PlainRanges =
        [(0f, 1f), (-1f, 1f), (4f, 12f), (4f, 8f), (1f, 2f), (-1f, 0f)];

    private static readonly (float Low, float High)[] WideAndFullDomainRanges =
        [(0.1f, 0.3f), (-0.0625f, 0.0625f), (1e-30f, 1e-20f),
         (0f, float.MaxValue), (-float.MaxValue, float.MaxValue), (-1e-40f, 1e-40f)];

    private static readonly (float Low, float High)[] SubnormalRanges =
        [(1e-45f, 1e-44f), (0f, float.Epsilon), (0f, 1.1754944e-38f),
         (5e-39f, 3e-38f), (-7.888609e-31f, 1.8446744e19f), (1f, 1.0000001f)];

    private static readonly (float Low, float High)[] NarrowAndInfiniteBoundRanges =
        [(1e30f, 1.0000001e30f), (-float.MaxValue, -1.7e38f), (-0.1f, 0.3f),
         (3f, 3.5f), (100f, 1000f), (0f, float.PositiveInfinity)];

    // One endpoint inside the collapsed band, the other far above it — the band partial on exactly
    // one side, with and without an off-lattice cut (-5·2^39 lies on the lattice these induce).
    private static readonly (float Low, float High)[] AcrossTheTruncationFloorRanges =
        [(1e-30f, 1e30f), (-1e-30f, 1e30f), (-2748779069440f, 1e30f),
         (-1e30f, -2748779069440f), (-1e30f, -1e-30f), (-1e30f, 1e-30f)];

    private static readonly (float Low, float High)[] ReachingTheFloorFromOneSideRanges =
        [(-1e30f, 1e15f), (-1e15f, 1e30f), (1e15f, 1e30f),
         (-1e30f, 0f), (0f, 1e30f), (-1.5e-45f, 3f)];

    // The first two cut the collapsed span to under one delta, so it holds no lattice point.
    private static readonly (float Low, float High)[] SubDeltaCollapsedSpanRanges =
        [(FloorBoundaryNeighbour, 16f), (-16f, -FloorBoundaryNeighbour), (-1e30f, 1f),
         (-1f, 1e30f), (0.99999994f, 1f), (-float.Epsilon, 0f)];

    // Both rays hold a low partial; a layout keeping only one drops 6.1M floats in the first.
    private static readonly (float Low, float High)[] BothRaysLowPartialRanges =
        [(-1e30f, 2e18f), (-3.4028235e38f, 6.18969982749202e26f),
         (-1.2924697071141057e-26f, 2.3509885615147286e-38f), (-1e30f, 1.2379401e27f),
         (-1e15f, 4.7223665e21f), (-2.8e-45f, 1.1754944e-38f)];

    private static readonly (float Low, float High)[][] DenseRangeGroups =
        [PlainRanges, WideAndFullDomainRanges, SubnormalRanges, NarrowAndInfiniteBoundRanges,
         AcrossTheTruncationFloorRanges, ReachingTheFloorFromOneSideRanges,
         SubDeltaCollapsedSpanRanges, BothRaysLowPartialRanges];

    private static readonly (float Low, float High)[] DenseRanges = [.. DenseRangeGroups.SelectMany(g => g)];

    private static long DenseSignedOrdinal(float x)
    {
        uint b = BitConverter.SingleToUInt32Bits(x);
        return (b & 0x8000_0000u) != 0 ? -(long)(b & 0x7FFF_FFFFu) : (long)b;
    }

    private static int DenseClassOf(float x)
    {
        long z = DenseSignedOrdinal(x);
        return (int)Math.Max(1, (z >= 0 ? z : -z - 1) >> RngDenseUniformOracle.P);
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
            Assert.True(table.Blocks.Length <= DenseMaxBlocks);
        }
    }

    // At most five ever coexist, measured over 300k ranges, and the bound is deliberately not
    // tightened to that: knowing which combinations can coexist is the reasoning that once dropped
    // a partial class and made 6.1M floats unreachable.
    private const int DenseMaxBlocks = 7;

    private static long DenseElements(RngDenseUniformOracle.Block b)
        => b.Kind switch
        {
            RngDenseUniformOracle.Kind.Lattice => (long)b.Weight,
            RngDenseUniformOracle.Kind.TwoSided => (long)(b.C1 - b.C0 + 1) << (RngDenseUniformOracle.P + 1),
            RngDenseUniformOracle.Kind.OneSided => (long)(b.C1 - b.C0 + 1) << RngDenseUniformOracle.P,
            _ => (long)(b.Weight >> b.Shift),
        };

    [Fact]
    public void TestDenseUniformOracleTableIsAWellFormedPartition()
    {
        foreach (var (low, high) in DenseRanges)
        {
            var table = RngDenseUniformOracle.Build(low, high);
            Assert.Equal(0UL, table.Blocks[0].Threshold);
            Assert.InRange(table.Blocks.Length, 1, DenseMaxBlocks);
            UInt128 weights = 0;
            for (int i = 0; i < table.Blocks.Length; i++)
            {
                weights += table.Blocks[i].Weight;
                Assert.True(table.Blocks[i].Weight > 0);
                if (i > 0) Assert.True(table.Blocks[i].Threshold > table.Blocks[i - 1].Threshold);
            }
            Assert.True(weights <= (UInt128)1 << 64);
            Assert.Equal(weights, table.Total == 0 ? (UInt128)1 << 64 : table.Total);
        }
    }

    [Fact]
    public void TestDenseUniformOracleReachesEveryFloatWhereNothingIsTruncated()
    {
        foreach (var (low, high) in (( float Low, float High)[])[(4f, 12f), (4f, 8f), (1f, 2f), (0.1f, 0.3f), (3f, 3.5f), (100f, 1000f)])
        {
            var table = RngDenseUniformOracle.Build(low, high);
            long elements = 0;
            foreach (var block in table.Blocks) elements += DenseElements(block);
            Assert.Equal(DenseSignedOrdinal(high) - DenseSignedOrdinal(low), elements);
        }
    }

    // Below the floor the two part ways by construction, and neither side is reachable by
    // sampling, so both are asserted directly.
    [Fact]
    public void TestDenseUniformOracleIsWalkerReynoldsAboveTheTruncationFloor()
    {
        var table = RngDenseUniformOracle.Build(0f, 1f);
        Assert.Equal(0UL, table.Total);
        Assert.Equal(86, table.FloorClass);

        uint Dense(ulong draw) => RngDenseUniformOracle.SampleBits(table, draw);
        foreach (ulong m in (ulong[])[0UL, 1UL, 4919UL, 8388607UL])
            for (int bit = 23; bit <= 63; bit++)
            {
                ulong draw = (1UL << bit) | m | (0x5A5A5A5AUL << 24 & ((1UL << bit) - 1));
                Assert.Equal(BitConverter.SingleToUInt32Bits(RngTestOracle.WalkerUniform(draw)), Dense(draw));
            }

        for (ulong draw = 0; draw < 1UL << 23; draw += 7919)
            Assert.Equal(BitConverter.SingleToUInt32Bits(draw * MathF.ScaleB(1f, -64)), Dense(draw));
    }

    [Fact]
    public void TestDenseUniformOracleGivesEveryWeightedBlockItsExactWeightInCodes()
    {
        foreach (var (low, high) in DenseRanges)
        {
            var table = RngDenseUniformOracle.Build(low, high);
            for (int i = 0; i < table.Blocks.Length; i++)
            {
                UInt128 end = i + 1 < table.Blocks.Length ? table.Blocks[i + 1].Threshold
                    : table.Total == 0 ? (UInt128)1 << 64 : table.Total;
                Assert.Equal(table.Blocks[i].Weight, end - table.Blocks[i].Threshold);
            }
        }
    }

    private static ulong DenseNegativeWeight(RngDenseUniformOracle.Block b)
        => b.Kind switch
        {
            RngDenseUniformOracle.Kind.TwoSided => b.Weight >> 1,
            RngDenseUniformOracle.Kind.OneSided => b.Negative ? b.Weight : 0,
            RngDenseUniformOracle.Kind.Lattice => (ulong)Math.Clamp(-b.Base, 0, (long)b.Weight),
            _ => b.Base < 0 ? b.Weight : 0,
        };

    [Fact]
    public void TestDenseUniformOracleSplitsSymmetricRangesExactlyInHalf()
    {
        for (int e = -60; e <= 60; e++)
        {
            float bound = MathF.ScaleB(1f, e);
            var table = RngDenseUniformOracle.Build(-bound, bound);
            ulong negative = 0;
            foreach (var block in table.Blocks) negative += DenseNegativeWeight(block);
            Assert.Equal(1UL << 63, negative);
        }
    }

    // Nothing in the build clamps the total, so passing 2^64 would wrap silently.
    [Fact]
    public void TestDenseUniformOracleTotalNeverExceedsTheWeightAxis()
    {
        var random = new Random(20260811);
        for (int i = 0; i < 20000; i++)
        {
            long span = 1L << (1 + random.Next(31));
            long lowOrdinal = random.NextInt64(-(255L << RngDenseUniformOracle.P), 255L << RngDenseUniformOracle.P);
            long highOrdinal = lowOrdinal + random.NextInt64(1, span);
            (float low, float high) = (DenseFloatOfOrdinal(lowOrdinal), DenseFloatOfOrdinal(highOrdinal));
            if (float.IsNaN(low) || float.IsNaN(high) || !(low < high)) continue;

            var table = RngDenseUniformOracle.Build(low, high);
            UInt128 weights = 0;
            foreach (var block in table.Blocks) weights += block.Weight;
            Assert.True(weights <= (UInt128)1 << 64);
            Assert.Equal(weights, table.Total == 0 ? (UInt128)1 << 64 : table.Total);
            Assert.InRange(table.Blocks.Length, 1, DenseMaxBlocks);
            float drawn = RngDenseUniformOracle.Draw(DenseKey, 0, i, low, high);
            Assert.True(drawn >= low && drawn < high);
        }
    }

    // Total is itself the sum of the block weights, so a block built and then dropped leaves the
    // partition self-consistent and its floats unreachable. Only an independent measure sees that.
    [Fact]
    public void TestDenseUniformOracleBlocksCoverEveryFloatAboveTheFloor()
    {
        var random = new Random(20260812);
        for (int i = 0; i < 20000; i++)
        {
            float low = BitConverter.UInt32BitsToSingle((uint)random.NextInt64(1L << 32));
            float high = BitConverter.UInt32BitsToSingle((uint)random.NextInt64(1L << 32));
            if (float.IsNaN(low) || float.IsNaN(high) || !(low < high)) continue;
            var table = RngDenseUniformOracle.Build(low, high);
            if (table.Fixed is not null) continue;

            long floor = (long)table.FloorClass << RngDenseUniformOracle.P;
            long bandLow = Math.Clamp(-floor, table.ZLow, table.ZHigh);
            long bandHigh = Math.Clamp(floor, table.ZLow, table.ZHigh);
            UInt128 want = DenseOrdinalUnits(table.ZLow, bandLow, table.FloorClass)
                         + DenseOrdinalUnits(bandHigh, table.ZHigh, table.FloorClass);
            UInt128 got = 0;
            foreach (var block in table.Blocks)
                if (block.Kind != RngDenseUniformOracle.Kind.Lattice) got += block.Weight;
            Assert.Equal(want, got);
        }
    }

    // The weight of the ordinal run [from, to), summed one weight class at a time.
    private static UInt128 DenseOrdinalUnits(long from, long to, int floorClass)
    {
        UInt128 units = 0;
        for (long z = from; z < to;)
        {
            long magnitude = z >= 0 ? z : -z - 1;
            int cls = (int)Math.Max(1, magnitude >> RngDenseUniformOracle.P);
            long end = Math.Min(to, z >= 0
                ? ((magnitude >> RngDenseUniformOracle.P) + 1) << RngDenseUniformOracle.P
                : -((long)cls << RngDenseUniformOracle.P));
            units += (UInt128)(end - z) << (cls - floorClass);
            z = end;
        }
        return units;
    }

    // How the 2^64 draws divide over the weight axis: Remainder of the units take Quotient + 1 of
    // them and the rest take Quotient.
    private static (UInt128 Quotient, UInt128 Remainder) DenseSplit(float low, float high)
    {
        ulong total = RngDenseUniformOracle.Build(low, high).Total;
        UInt128 units = total == 0 ? (UInt128)1 << 64 : total;
        return (((UInt128)1 << 64) / units, ((UInt128)1 << 64) % units);
    }

    // The one rounding the blocks do not remove: a weight unit gets q or q+1 of the 2^64 draws, so
    // a single-unit float can take (q+1)/q of its due. Raising the depth shrank q from 4 to 1.
    [Fact]
    public void TestDenseUniformOracleScalingIsExactOnlyWhenTheTotalIsAPowerOfTwo()
    {
        Assert.Equal(((UInt128)1, (UInt128)0), DenseSplit(0f, 1f));
        Assert.Equal(((UInt128)1, (UInt128)0), DenseSplit(0f, float.PositiveInfinity));
        // Not the finite domain: clamping -infinity to -MaxValue costs it the top class's last float.
        Assert.Equal(((UInt128)1, (UInt128)1 << 40), DenseSplit(-float.MaxValue, float.MaxValue));

        // Every range's own quotient, row per group of DenseRangeGroups.
        UInt128[][] want =
        [
            [1UL, 1UL, 2UL, 2UL, 2UL, 1UL],
            [2UL, 1UL, 1UL, 1UL, 1UL, 129247667341929UL],
            [3074457345618258602UL, (UInt128)1 << 64, 2199023255552UL, 1033975716939UL, 2UL, 16777216UL],
            [16777216UL, 1UL, 2UL, 8UL, 1UL, 1UL],
            [1UL, 2UL, 2UL, 1UL, 1UL, 2UL],
            [2UL, 2UL, 1UL, 1UL, 1UL, 2UL],
            [1UL, 1UL, 2UL, 2UL, 16777216UL, (UInt128)1 << 64],
            [2UL, 2UL, 1UL, 2UL, 1UL, 2199022731264UL],
        ];
        for (int g = 0; g < DenseRangeGroups.Length; g++)
            for (int i = 0; i < DenseRangeGroups[g].Length; i++)
                Assert.Equal(want[g][i],
                    DenseSplit(DenseRangeGroups[g][i].Low, DenseRangeGroups[g][i].High).Quotient);
    }

    // The skew a single weight unit takes from that rounding, and the two ranges the header names
    // as exactly dyadic — hence exactly fair — among them.
    [Fact]
    public void TestDenseUniformWeightUnitSkewIsAtMostTwiceAndVanishesOnADyadicTotal()
    {
        double Skew(float low, float high)
        {
            var (quotient, remainder) = DenseSplit(low, high);
            return remainder == 0 ? 1.0 : (double)(quotient + 1) / (double)quotient;
        }
        foreach (var (low, high) in DenseRanges) Assert.InRange(Skew(low, high), 1.0, 2.0);
        Assert.Equal(1.0, Skew(0f, 1f));
        Assert.Equal(1.0, Skew(-1f, 1f));
        Assert.Equal(1.0, Skew(0f, float.PositiveInfinity));
        Assert.Equal(1.0, Skew(4f, 12f));
        Assert.Equal(2.0, Skew(0f, float.MaxValue));
        Assert.Equal(2.0, Skew(-float.MaxValue, float.MaxValue));
    }

    // The figure RuntimeRng's header quotes and nothing measured. Every float's run of weight units
    // maps to within one of the 2^64 draws it is due, so the distance is under the reachable float
    // count over 2^65 — and is exactly zero where the scaling is the identity.
    [Fact]
    public void TestDenseUniformStaysWithinTheDocumentedTotalVariationDistance()
    {
        double Distance(float low, float high)
        {
            var table = RngDenseUniformOracle.Build(low, high);
            long floats = 0;
            foreach (var block in table.Blocks) floats += DenseElements(block);
            return table.Total == 0 ? 0.0 : floats * Math.ScaleB(1.0, -65);
        }
        foreach (var (low, high) in DenseRanges) Assert.True(Distance(low, high) < Math.ScaleB(1.0, -35));
        Assert.Equal(0.0, Distance(0f, 1f));
        Assert.Equal(0.0, Distance(-1f, 1f));
        Assert.Equal(0.0, Distance(0f, float.PositiveInfinity));
    }

    // The smallest value the blocks can produce. Value order is ordinal order, and every block's
    // image is a contiguous ordinal run except the lattice, whose first point is its base.
    private static float DenseMinDrawable(RngDenseUniformOracle.Table table)
    {
        float min = float.PositiveInfinity;
        foreach (var b in table.Blocks)
            min = MathF.Min(min, b.Kind switch
            {
                RngDenseUniformOracle.Kind.Lattice =>
                    MathF.ScaleB((float)b.Base, table.FloorClass - 150),
                RngDenseUniformOracle.Kind.TwoSided =>
                    DenseFloatOfOrdinal(-(((long)b.C1 + 1) << RngDenseUniformOracle.P)),
                RngDenseUniformOracle.Kind.OneSided => b.Negative
                    ? DenseFloatOfOrdinal(-(((long)b.C1 + 1) << RngDenseUniformOracle.P))
                    : DenseFloatOfOrdinal((long)b.C0 << RngDenseUniformOracle.P),
                _ => DenseFloatOfOrdinal(b.Base),
            });
        return min;
    }

    // In the first two `low` lies inside the collapsed span and off the lattice, so truncation
    // makes it undrawable; everywhere else it is the base of its ray's partial class and owns
    // exactly the weight units one float of its class is worth.
    [Fact]
    public void TestDenseUniformDrawsLowOnlyWhereTruncationLeavesItOnTheGrid()
    {
        float Min(float low, float high) => DenseMinDrawable(RngDenseUniformOracle.Build(low, high));
        Assert.True(Min(FloorBoundaryNeighbour, 16f) > FloorBoundaryNeighbour);
        Assert.True(Min(-1f, 1e30f) > -1f);
        Assert.Equal(-16f, Min(-16f, -FloorBoundaryNeighbour));
        Assert.Equal(-1e30f, Min(-1e30f, 1f));
        Assert.Equal(0f, Min(0f, 1f));
        Assert.Equal(-1f, Min(-1f, 1f));
        Assert.Equal(4f, Min(4f, 12f));

        void OneUlpShare(float low, float high)
        {
            var table = RngDenseUniformOracle.Build(low, high);
            var block = table.Blocks.Single(b => b.Kind == RngDenseUniformOracle.Kind.Ordinals
                && b.Base == DenseSignedOrdinal(low));
            ulong units = 1UL << (DenseClassOf(low) - table.FloorClass);
            uint bits = BitConverter.SingleToUInt32Bits(low);
            Assert.Equal(units, 1UL << block.Shift);
            Assert.Equal(low, Min(low, high));
            Assert.Equal(bits, RngDenseUniformOracle.SampleBits(table, DenseDrawFor(table, block.Threshold)));
            Assert.Equal(bits, RngDenseUniformOracle.SampleBits(table, DenseDrawFor(table, block.Threshold + units - 1)));
            Assert.NotEqual(bits, RngDenseUniformOracle.SampleBits(table, DenseDrawFor(table, block.Threshold + units)));
        }
        OneUlpShare(0.1f, 0.3f);
        OneUlpShare(3f, 3.5f);
        OneUlpShare(100f, 1000f);
        OneUlpShare(-0.1f, 0.3f);
    }

    // A negative ray thinner than one lattice cell is dropped outright, not merely made rare.
    [Fact]
    public void TestDenseUniformGivesASubDeltaRayProbabilityExactlyZero()
    {
        UInt128 Negative(float low, float high)
        {
            UInt128 weight = 0;
            foreach (var block in RngDenseUniformOracle.Build(low, high).Blocks)
                weight += DenseNegativeWeight(block);
            return weight;
        }
        Assert.Equal((UInt128)0, Negative(-1f, 1e30f));
        Assert.Equal((UInt128)0, Negative(-1e-30f, 1e30f));
        Assert.Equal((UInt128)0, Negative(-1.5e-45f, 3f));
        Assert.Equal((UInt128)2, Negative(-2.8e-45f, 1.1754944e-38f));
        Assert.Equal((UInt128)1 << 63, Negative(-1f, 1f));
    }

    // What the draw actually emits: the real generator's values, off a fixed key so every statistic
    // below is deterministic, decoded by the table exactly as RngDenseUniformOracle.Draw would.
    private static float[] DenseStream(ulong key, ulong substreamIndex, float low, float high, int n)
    {
        var table = RngDenseUniformOracle.Build(low, high);
        float[] xs = new float[n];
        for (long i = 0; i < n; i++)
            xs[i] = BitConverter.UInt32BitsToSingle(RngDenseUniformOracle.SampleBits(
                table, RngTestOracle.DrawValue(key, substreamIndex, i)));
        return xs;
    }

    // The weight each class above the truncation floor is owed, read off the blocks rather than
    // sampled. Everything below the floor is the lattice, which is class 0 here because that is all
    // an observer can tell those floats apart as.
    private static Dictionary<int, UInt128> DenseClassUnits(RngDenseUniformOracle.Table table)
    {
        Dictionary<int, UInt128> units = [];
        void Add(int cls, UInt128 u) => units[cls] = units.GetValueOrDefault(cls) + u;
        foreach (var b in table.Blocks)
            switch (b.Kind)
            {
                case RngDenseUniformOracle.Kind.Lattice:
                    Add(0, b.Weight);
                    break;
                case RngDenseUniformOracle.Kind.TwoSided:
                case RngDenseUniformOracle.Kind.OneSided:
                {
                    int width = RngDenseUniformOracle.P
                        + (b.Kind == RngDenseUniformOracle.Kind.TwoSided ? 1 : 0);
                    for (int c = b.C0; c <= b.C1; c++)
                        Add(c, (UInt128)1 << (width + c - table.FloorClass));
                    break;
                }
                default:
                    Add(b.C0, b.Weight);
                    break;
            }
        return units;
    }

    private static double DenseClassChiSquare(float low, float high, int draws)
    {
        var table = RngDenseUniformOracle.Build(low, high);
        double total = table.Total == 0 ? Math.ScaleB(1.0, 64) : table.Total;
        Dictionary<int, long> seen = [];
        foreach (float x in DenseStream(DenseKey, 0, low, high, draws))
        {
            int cls = DenseClassOf(x);
            int bucket = cls < table.FloorClass ? 0 : cls;
            seen[bucket] = seen.GetValueOrDefault(bucket) + 1;
        }
        double chi = 0, pooledExpected = 0;
        long pooledSeen = 0;
        foreach (var (cls, weight) in DenseClassUnits(table))
        {
            double expected = draws * (double)weight / total;
            long observed = seen.GetValueOrDefault(cls);
            if (expected >= 5) chi += (observed - expected) * (observed - expected) / expected;
            else { pooledExpected += expected; pooledSeen += observed; }
        }
        return chi + (pooledExpected > 0
            ? (pooledSeen - pooledExpected) * (pooledSeen - pooledExpected) / pooledExpected
            : 0);
    }

    // Nothing else reads the drawn stream: the one other empirical check is a mean over 64 samples,
    // which a sampler returning only the two endpoints would pass.
    [Fact]
    public void TestDenseUniformDrawsOccupyEachWeightClassInProportionToItsBlockWeight()
        => Assert.All((( float Low, float High)[])
            [(0f, 1f), (-1f, 1f), (4f, 12f), (0.1f, 0.3f), (100f, 1000f),
             (1e-30f, 1e30f), (-1e30f, 1e15f), (-float.MaxValue, float.MaxValue)],
            r => Assert.True(DenseClassChiSquare(r.Low, r.High, 40000) < 40));

    [Fact]
    public void TestDenseUniformDrawsTrackTheContinuousUniformUnderKolmogorovSmirnov()
    {
        double Statistic(float low, float high)
        {
            float[] xs = DenseStream(DenseKey, 1, low, high, 20000);
            Array.Sort(xs);
            double worst = 0;
            for (int i = 0; i < xs.Length; i++)
            {
                double f = ((double)xs[i] - low) / ((double)high - low);
                worst = Math.Max(worst, Math.Max(Math.Abs(f - (double)i / xs.Length),
                                                 Math.Abs(f - (i + 1.0) / xs.Length)));
            }
            return worst * Math.Sqrt(xs.Length);
        }
        Assert.All((( float Low, float High)[])
            [(0f, 1f), (-1f, 1f), (4f, 12f), (0.1f, 0.3f), (100f, 1000f), (3f, 3.5f), (-0.1f, 0.3f)],
            r => Assert.True(Statistic(r.Low, r.High) < 2.0));
    }

    [Fact]
    public void TestConsecutiveDenseUniformDrawPositionsFillTheUnitSquareIndependently()
    {
        const int Draws = 50000, Grid = 8;
        double Statistic(float low, float high)
        {
            float[] xs = DenseStream(DenseKey, 2, low, high, Draws);
            int Cell(float x) => (int)Math.Min(Grid - 1, ((double)x - low) / ((double)high - low) * Grid);
            long[] cells = new long[Grid * Grid];
            for (int i = 0; i + 1 < Draws; i++) cells[Cell(xs[i]) * Grid + Cell(xs[i + 1])]++;
            double expected = (Draws - 1.0) / cells.Length, chi = 0;
            foreach (long c in cells) chi += (c - expected) * (c - expected) / expected;
            return chi;
        }
        Assert.All((( float Low, float High)[])
            [(0f, 1f), (-1f, 1f), (-1f, 0f), (4f, 12f), (0.1f, 0.3f), (3f, 3.5f), (100f, 1000f), (-0.1f, 0.3f)],
            r => Assert.True(Statistic(r.Low, r.High) < 130));
    }

    // Distinctness is all the other seed tests ask for, and two streams can be distinct and still
    // move together.
    [Fact]
    public void TestConsecutiveSeedsAndSiblingModelIdPathsGiveUncorrelatedStreams()
    {
        double Correlation(float[] a, float[] b)
        {
            double ma = 0, mb = 0;
            for (int i = 0; i < a.Length; i++) { ma += a[i]; mb += b[i]; }
            ma /= a.Length; mb /= b.Length;
            double sa = 0, sb = 0, sab = 0;
            for (int i = 0; i < a.Length; i++)
            {
                double da = a[i] - ma, db = b[i] - mb;
                sa += da * da; sb += db * db; sab += da * db;
            }
            return sab / Math.Sqrt(sa * sb);
        }
        double Worst(Func<int, ulong> key)
        {
            float[][] streams = [.. Enumerable.Range(0, 64).Select(i => DenseStream(key(i), 0, 0f, 1f, 1024))];
            double worst = 0;
            for (int i = 0; i < streams.Length; i++)
                for (int j = i + 1; j < streams.Length; j++)
                    worst = Math.Max(worst, Math.Abs(Correlation(streams[i], streams[j])));
            return worst;
        }
        Assert.True(Worst(s => RngTestOracle.RunKey(new RngConfig { MasterSeed = (ulong)s }, [1])) < 0.25);
        Assert.True(Worst(j => RngTestOracle.RunKey(RngConfig.Default, [1, j])) < 0.25);
        Assert.True(Worst(j => RngTestOracle.InitKey(RngConfig.Default, [3, j])) < 0.25);
    }

    // The size of the one approximation, and both numbers are quoted in RuntimeRng's header.
    [Fact]
    public void TestDenseUniformOracleReachesTheDocumentedFractionOfEachDomain()
    {
        double Reachable(float low, float high)
        {
            long floats = 0;
            foreach (var block in RngDenseUniformOracle.Build(low, high).Blocks)
                floats += DenseElements(block);
            return (double)floats / (DenseSignedOrdinal(high) - DenseSignedOrdinal(low));
        }
        Assert.Equal(0.331, Reachable(0f, 1f), 3);
        Assert.Equal(0.161, Reachable(-float.MaxValue, float.MaxValue), 3);
    }

    private static float DenseFloatOfOrdinal(long z)
        => BitConverter.UInt32BitsToSingle(z >= 0 ? (uint)z : 0x8000_0000u | (uint)-z);

    // The contract by exhaustion rather than sampling. Only ranges near zero have a small enough
    // total to enumerate — anything reaching the truncation floor totals about 2^64 — so the
    // whole-class blocks are left to the Walker and two-sided closed forms above.
    [Fact]
    public void TestDenseUniformOracleAssignsEveryWeightUnitExactlyOnce()
    {
        foreach (var (lowOrd, highOrd) in ((long Low, long High)[])
            [(0, 1000), (3, 1000), (-1000, 1000), (-1000, -3), (0, 1),
             ((1 << 23) - 500, (1 << 23) + 500), ((1 << 24) - 500, (1 << 24) + 500),
             (-(1 << 24) - 500, -(1 << 24) + 500), (-((1L << 25) + 500), -((1L << 25) - 500))])
        {
            (float low, float high) = (DenseFloatOfOrdinal(lowOrd), DenseFloatOfOrdinal(highOrd));
            var table = RngDenseUniformOracle.Build(low, high);
            Assert.NotEqual(0UL, table.Total);
            Dictionary<uint, ulong> seen = [];
            for (ulong s = 0; s < table.Total; s++)
            {
                ulong draw = (ulong)(((((UInt128)s << 64) + table.Total - 1) / table.Total));
                uint bits = RngDenseUniformOracle.SampleBits(table, draw);
                seen[bits] = seen.GetValueOrDefault(bits) + 1;
            }
            Dictionary<uint, ulong> want = [];
            for (long z = lowOrd; z < highOrd; z++)
                want[BitConverter.SingleToUInt32Bits(DenseFloatOfOrdinal(z))] =
                    1UL << (int)(Math.Max(1, (z >= 0 ? z : -z - 1) >> 23) - table.FloorClass);
            Assert.Equal([.. want.Keys.Order()], [.. seen.Keys.Order()]);
            foreach (var (bits, count) in seen) Assert.Equal(want[bits], count);
        }
    }

    // A NaN bound comes back bits intact, `low` winning when both are; an infinite bound clamps to
    // the finite extreme of its sign, except +inf as `high`, which is the ordinal one past the
    // largest finite; an empty or inverted range comes back as the clamped `low`, whose negative
    // zero was normalised to +0 on the way in.
    private static uint? DenseToyFixed(RngDenseUniformOracle.Format fmt, uint lowBits, uint highBits)
    {
        long lowMagnitude = lowBits & (fmt.SignBit - 1), highMagnitude = highBits & (fmt.SignBit - 1);
        if (lowMagnitude == 0) lowBits = 0;
        if (lowMagnitude > fmt.InfinityOrdinal) return lowBits;
        if (highMagnitude > fmt.InfinityOrdinal) return highBits;
        uint clamped = lowMagnitude == fmt.InfinityOrdinal
            ? (uint)((lowBits & (uint)fmt.SignBit) | (uint)fmt.MaxFiniteOrdinal) : lowBits;
        long zLow = DenseToySignedOrdinal(fmt, clamped);
        long zHigh = highMagnitude != fmt.InfinityOrdinal ? DenseToySignedOrdinal(fmt, highBits)
            : (highBits & (uint)fmt.SignBit) != 0 ? -fmt.MaxFiniteOrdinal : fmt.InfinityOrdinal;
        return zHigh <= zLow ? clamped : null;
    }

    private static long DenseToySignedOrdinal(RngDenseUniformOracle.Format fmt, uint bits)
        => (bits & (uint)fmt.SignBit) != 0 ? -(long)(bits & (uint)(fmt.SignBit - 1)) : bits;

    // Value of a signed ordinal in units of the format's smallest subnormal — exact integers, and
    // the only handle on which float a lattice point is that does not go through the construction.
    private static long DenseToyUnits(RngDenseUniformOracle.Format fmt, long z)
    {
        long magnitude = Math.Abs(z), field = magnitude >> fmt.P, significand = magnitude & fmt.SigMask;
        long units = field == 0 ? significand : (fmt.BinadeSize | significand) << (int)(field - 1);
        return z < 0 ? -units : units;
    }

    // F-2: the construction on a format small enough to enumerate. Every pair of representable
    // bounds, every code on the weight axis, checked against cell widths and lattice occupancy read
    // off an enumeration of the format rather than off the blocks.
    private static void DenseToyEnumerate(RngDenseUniformOracle.Format fmt)
    {
        int patterns = 1 << (1 + fmt.E + fmt.P);
        UInt128 axis = (UInt128)1 << fmt.W;
        uint Bits(long z) => z >= 0 ? (uint)z : (uint)(fmt.SignBit | -z);
        int Class(long z) => (int)Math.Max(1, (z >= 0 ? z : -z - 1) >> fmt.P);

        Dictionary<long, uint> byUnits = [];
        for (long z = -fmt.MaxFiniteOrdinal; z <= fmt.MaxFiniteOrdinal; z++)
            byUnits[DenseToyUnits(fmt, z)] = Bits(z);

        long[] want = new long[patterns], seen = new long[patterns];
        int kinds = 0;
        bool truncated = false, wrapped = false;
        for (uint lowBits = 0; lowBits < patterns; lowBits++)
            for (uint highBits = 0; highBits < patterns; highBits++)
            {
                var table = RngDenseUniformOracle.Build(fmt, lowBits, highBits);
                Assert.Equal(DenseToyFixed(fmt, lowBits, highBits), table.Fixed);
                if (table.Fixed is not null) continue;

                int classes = table.ZLow < 0 && table.ZHigh > 0 ? fmt.StraddleClasses : fmt.MaxClasses;
                int floorClass = Math.Max(1,
                    Math.Max(Class(table.ZLow), Class(table.ZHigh - 1)) - classes + 1);
                Assert.Equal(floorClass, table.FloorClass);

                long zFloor = (long)floorClass << fmt.P, delta = 1L << (floorClass - 1);
                long bandLow = Math.Clamp(-zFloor, table.ZLow, table.ZHigh);
                long bandHigh = Math.Clamp(zFloor, table.ZLow, table.ZHigh);
                long spanLow = DenseToyUnits(fmt, bandLow), spanHigh = DenseToyUnits(fmt, bandHigh);
                Array.Clear(want);
                for (long z = table.ZLow; z < bandLow; z++) want[Bits(z)] = 1L << (Class(z) - floorClass);
                for (long z = bandHigh; z < table.ZHigh; z++) want[Bits(z)] = 1L << (Class(z) - floorClass);
                for (long n = spanLow >> (floorClass - 1); (n + 1) * delta <= spanHigh; n++)
                    if (n * delta >= spanLow) want[byUnits[n * delta]]++;

                UInt128 total = table.Total == 0 ? axis : table.Total;
                Assert.Equal(total, (UInt128)want.Sum());
                Assert.True(total <= axis);

                UInt128 cumulative = 0;
                foreach (var block in table.Blocks)
                {
                    Assert.Equal((UInt128)block.Threshold, cumulative);
                    Assert.True(block.Weight > 0);
                    cumulative += block.Weight;
                    kinds |= 1 << (int)block.Kind;
                }
                Assert.Equal(total, cumulative);
                (truncated, wrapped) = (truncated || floorClass > 1, wrapped || table.Total == 0);

                Array.Clear(seen);
                for (ulong code = 0; code < total; code++)
                    seen[RngDenseUniformOracle.SampleBits(table, table.Total == 0 ? code
                        : ((code << fmt.W) + table.Total - 1) / table.Total)]++;
                Assert.True(seen.AsSpan().SequenceEqual(want));

                bool escaped = false;
                for (ulong draw = 0; draw < (ulong)axis; draw++)
                    escaped |= want[RngDenseUniformOracle.SampleBits(table, draw)] == 0;
                Assert.False(escaped);
            }
        Assert.Equal(0b1111, kinds);
        Assert.True(truncated && wrapped);
    }

    // Wide-exponent and wide-significand shapes, each with a draw width leaving the depth W - P
    // shallower than the classes the format holds, so truncation bites on the deeper ranges.
    [Fact] public void TestDenseUniformOracleAssignsEveryWeightUnitExactlyOnceOnAThreeByTwoFormat()
        => DenseToyEnumerate(new(E: 3, P: 2, W: 6));
    [Fact] public void TestDenseUniformOracleAssignsEveryWeightUnitExactlyOnceOnAFourByThreeFormat()
        => DenseToyEnumerate(new(E: 4, P: 3, W: 8));
    [Fact] public void TestDenseUniformOracleAssignsEveryWeightUnitExactlyOnceOnATwoByFourFormat()
        => DenseToyEnumerate(new(E: 2, P: 4, W: 6));
    [Fact] public void TestDenseUniformOracleAssignsEveryWeightUnitExactlyOnceOnAFiveByOneFormat()
        => DenseToyEnumerate(new(E: 5, P: 1, W: 8));

    // Against a closed form written out independently. The negative half runs downward from the
    // binade above — a negative ordinal takes its class from the pattern below it — hence 2 - m.
    [Fact]
    public void TestDenseUniformOracleDecodesBothSignsOfEveryWholeClass()
    {
        var table = RngDenseUniformOracle.Build(-1f, 1f);
        Assert.Equal(0UL, table.Total);
        Assert.Equal(87, table.FloorClass);

        foreach (ulong mantissa in (ulong[])[0UL, 1UL, 4919UL, 8388607UL])
            foreach (ulong sign in (ulong[])[0UL, 1UL])
                for (int lead = 24; lead <= 63; lead++)
                {
                    ulong draw = (1UL << lead) | (0x5A5A5AUL << 24 & ((1UL << lead) - 1))
                        | (sign << 23) | mantissa;
                    float f = 1f + mantissa * (1f / 8388608f);
                    float want = MathF.ScaleB(sign == 0 ? f : f - 3f, 87 + lead - 24 - 127);
                    Assert.Equal(BitConverter.SingleToUInt32Bits(want),
                        RngDenseUniformOracle.SampleBits(table, draw));
                }
    }

    [Fact]
    public void TestDenseUniformOracleHandlesDegenerateAndNonFiniteBounds()
    {
        Assert.Equal(3f, RngDenseUniformOracle.Draw(DenseKey, 0, 0, 3f, 3f));
        Assert.Equal(7f, RngDenseUniformOracle.Draw(DenseKey, 0, 0, 7f, 3f));
        Assert.True(float.IsNaN(RngDenseUniformOracle.Draw(DenseKey, 0, 0, float.NaN, 1f)));
        Assert.True(float.IsNaN(RngDenseUniformOracle.Draw(DenseKey, 0, 0, 1f, float.NaN)));
        // Which NaN, not merely that it is one. The in-graph check cannot see this: NaN never
        // equals itself, and opset 21 has no bit reinterpretation.
        float NaNOf(uint bits) => BitConverter.UInt32BitsToSingle(bits);
        Assert.Equal(BitConverter.SingleToUInt32Bits(float.NaN),
            RngDenseUniformOracle.Build(float.NaN, 1f).Fixed);
        Assert.Equal(0x7FC0_1234u, RngDenseUniformOracle.Build(NaNOf(0x7FC0_1234u), 1f).Fixed);
        Assert.Equal(0xFFD0_5678u, RngDenseUniformOracle.Build(1f, NaNOf(0xFFD0_5678u)).Fixed);
        Assert.Equal(0x7FC0_1234u,
            RngDenseUniformOracle.Build(NaNOf(0x7FC0_1234u), NaNOf(0xFFD0_5678u)).Fixed);
        for (long i = 0; i < 200; i++)
        {
            float wide = RngDenseUniformOracle.Draw(DenseKey, 0, i, float.NegativeInfinity, float.PositiveInfinity);
            Assert.False(float.IsNaN(wide) || float.IsInfinity(wide));
            Assert.Equal(1f, RngDenseUniformOracle.Draw(DenseKey, 0, i, 1f, 1.0000001f));
        }
    }

    // The smallest draw scaling to `scaled`, so a block's own first code can be hand-picked.
    private static ulong DenseDrawFor(RngDenseUniformOracle.Table table, ulong scaled)
        => table.Total == 0 ? scaled
         : (ulong)((((UInt128)scaled << 64) + table.Total - 1) / table.Total);

    // BitsOfClassMember spells -infinity at negative class 254 mantissa 0, and only the clamp of a
    // -infinity `low` to -MaxValue keeps that ordinal out of range. Sampling would never see the
    // clamp regress — the slot is 2^-24 of the draws at best — so it is hand-picked instead.
    [Fact]
    public void TestDenseUniformOracleClampsNegativeInfinityAwayFromTheInfinitySlot()
    {
        foreach (float high in (float[])[float.PositiveInfinity, float.MaxValue, 0f, -1e30f, -3.4e38f])
        {
            var table = RngDenseUniformOracle.Build(float.NegativeInfinity, high);
            Assert.Equal(-float.MaxValue, DenseFloatOfOrdinal(table.ZLow));
            var deepest = table.Blocks.Single(
                b => b.Kind == RngDenseUniformOracle.Kind.Ordinals && b.Base == table.ZLow);
            Assert.Equal(-float.MaxValue, BitConverter.UInt32BitsToSingle(
                RngDenseUniformOracle.SampleBits(table, DenseDrawFor(table, deepest.Threshold))));
        }
    }

    // Every int64 Max/Min operand the table build feeds ONNX Runtime stays inside (-2^31, 2^31),
    // where its kernel is sound: ordinals reach 255*2^23, lattice indices 2^23, classes 255,
    // shifts 40. Widening any of them past the sign boundary would silently mis-order.
    [Fact]
    public void TestDenseTableInt64MaxOperandsStayBelowTheSignBoundary()
    {
        const long ordinalBound = 255L << RngDenseUniformOracle.P, latticeBound = 1L << 24;
        void Check(float low, float high)
        {
            var table = RngDenseUniformOracle.Build(low, high);
            Assert.InRange(table.ZLow, -ordinalBound, ordinalBound);
            Assert.InRange(table.ZHigh, -ordinalBound, ordinalBound);
            Assert.InRange((long)table.FloorClass << RngDenseUniformOracle.P, 0, ordinalBound);
            foreach (var b in table.Blocks)
            {
                Assert.InRange(b.C0, 0, 255);
                Assert.InRange(b.C1, 0, 255);
                Assert.InRange(b.Shift, 0, 40);
                if (b.Kind == RngDenseUniformOracle.Kind.Lattice)
                {
                    Assert.InRange(b.Base, -latticeBound, latticeBound);
                    Assert.InRange(b.Base + (long)b.Weight, -latticeBound, latticeBound);
                }
                else if (b.Kind == RngDenseUniformOracle.Kind.Ordinals)
                {
                    Assert.InRange(b.Base, -ordinalBound, ordinalBound);
                    Assert.InRange(b.Base + DenseElements(b), -ordinalBound, ordinalBound);
                }
            }
        }

        foreach (var (low, high) in DenseRanges) Check(low, high);
        Check(float.NegativeInfinity, float.PositiveInfinity);
        Check(float.PositiveInfinity, float.NegativeInfinity);
        Check(float.NaN, float.NaN);
        var random = new Random(20260813);
        for (int i = 0; i < 20000; i++)
        {
            float low = BitConverter.UInt32BitsToSingle((uint)random.NextInt64(1L << 32));
            float high = BitConverter.UInt32BitsToSingle((uint)random.NextInt64(1L << 32));
            if (float.IsNaN(low) || float.IsNaN(high) || !(low < high)) continue;
            Check(low, high);
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
                expected[r * n + i] = RngDenseUniformOracle.Draw(DenseKey, (ulong)r, i, ranges[r].Low, ranges[r].High);
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

    private static uint[] GraphDrawBits((float Low, float High)[] ranges)
    {
        float[] bounds = new float[RngDenseUniformOutput.Ranges * 2];
        for (int r = 0; r < ranges.Length; r++)
            (bounds[2 * r], bounds[2 * r + 1]) = (ranges[r].Low, ranges[r].High);
        var g = ((ComputationGraph)typeof(RngDenseUniformOutput)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var input = TensorData([(long)bounds.Length], bounds);
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([input])).ToConcreteModel();
        return [.. ComputeContext.Default.Execute(concrete, input)[0].ToTensorData()
            .As<float32>().AccessMemory().ToArray().Select(BitConverter.SingleToUInt32Bits)];
    }

    // The same raw output off the Quick Execution Engine rather than ONNX Runtime. One concrete
    // model serves every group: the bounds are a runtime input, and the build dominates the run.
    private static uint[] QeeDrawBits(InternalComputationGraph model, (float Low, float High)[] ranges)
    {
        float[] bounds = new float[RngDenseUniformOutput.Ranges * 2];
        for (int r = 0; r < ranges.Length; r++)
            (bounds[2 * r], bounds[2 * r + 1]) = (ranges[r].Low, ranges[r].High);
        var input = TensorData([(long)bounds.Length], bounds);
        var rt = (RuntimeTensor)new QuickExecutionEngine().Run(model, [input])[model.Outputs[0]];
        return [.. rt.FloatData!.Value.Select(BitConverter.SingleToUInt32Bits)];
    }

    // The second engine's VALUES, which nothing else reads: AutoTest's Quick Execution Engine pass
    // asserts only that each output resolves to a valid dtype and never looks at a self-check bool,
    // so a QEE computing every draw wrong stays green (Shorokoo#159). Cross-engine bit-exactness is
    // the whole reason the draw lives in the graph, so it is asserted on the bits.
    [Fact]
    public void TestQuickEngineDenseUniformMatchesTheOracleBitForBitOnEveryAdversarialRange()
    {
        var g = ((ComputationGraph)typeof(RngDenseUniformOutput)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var seed = TensorData([(long)RngDenseUniformOutput.Ranges * 2], new float[RngDenseUniformOutput.Ranges * 2]);
        var model = g.ToConcreteArchitecture(g.FromOrderedInputs([seed])).ToConcreteModel();
        foreach (var ranges in DenseRangeGroups)
        {
            uint[] got = QeeDrawBits(model, ranges);
            for (int r = 0; r < ranges.Length; r++)
                for (long i = 0; i < RngDenseUniformOutput.Draws; i++)
                    Assert.Equal(
                        BitConverter.SingleToUInt32Bits(RngDenseUniformOracle.Draw(
                            DenseKey, (ulong)r, i, ranges[r].Low, ranges[r].High)),
                        got[r * RngDenseUniformOutput.Draws + i]);
        }
    }

    // Which NaN the graph returns, not merely that it returns one — the in-graph check compares
    // with Equal, which no NaN satisfies.
    [Fact]
    public void TestInGraphDenseUniformReproducesTheOraclesNaNPayloadBitForBit()
    {
        float NaNOf(uint bits) => BitConverter.UInt32BitsToSingle(bits);
        (float Low, float High)[] ranges =
            [(NaNOf(0x7FC0_1234u), 1f), (1f, NaNOf(0x7FC0_1234u)),
             (NaNOf(0x7FC0_1234u), NaNOf(0xFFD0_5678u)), (NaNOf(0xFFD0_5678u), 1f),
             (0f, 0f), (0f, 1f)];
        uint[] got = GraphDrawBits(ranges);
        for (int r = 0; r < ranges.Length; r++)
            for (long i = 0; i < RngDenseUniformOutput.Draws; i++)
                Assert.Equal(
                    BitConverter.SingleToUInt32Bits(RngDenseUniformOracle.Draw(
                        DenseKey, (ulong)r, i, ranges[r].Low, ranges[r].High)),
                    got[r * RngDenseUniformOutput.Draws + i]);
    }

    // -0f reaches the draw only through `low`, where it is normalised to +0f, so no draw and no
    // degenerate case returns the bit pattern 0x80000000 on any engine.
    [Fact]
    public void TestDenseUniformNeverReturnsNegativeZeroOnAnyEngine()
    {
        const uint negativeZero = 0x8000_0000u;
        (float Low, float High)[] ranges =
            [(-0f, 0f), (-0f, -0f), (0f, -0f), (-0f, -1f), (-0f, 1f), (-0f, float.Epsilon)];
        var g = ((ComputationGraph)typeof(RngDenseUniformOutput)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var seed = TensorData([(long)RngDenseUniformOutput.Ranges * 2],
            new float[RngDenseUniformOutput.Ranges * 2]);
        var model = g.ToConcreteArchitecture(g.FromOrderedInputs([seed])).ToConcreteModel();
        uint[] ort = GraphDrawBits(ranges), qee = QeeDrawBits(model, ranges);
        for (int r = 0; r < ranges.Length; r++)
            for (int i = 0; i < RngDenseUniformOutput.Draws; i++)
            {
                int k = r * RngDenseUniformOutput.Draws + i;
                uint oracle = BitConverter.SingleToUInt32Bits(RngDenseUniformOracle.Draw(
                    DenseKey, (ulong)r, i, ranges[r].Low, ranges[r].High));
                Assert.NotEqual(negativeZero, oracle);
                Assert.NotEqual(negativeZero, ort[k]);
                Assert.NotEqual(negativeZero, qee[k]);
                Assert.Equal(oracle, ort[k]);
                Assert.Equal(oracle, qee[k]);
            }
    }

    // An empty block carries the following one's threshold and the trailing empties carry the
    // total, so the graph's distinct thresholds minus that total are exactly the oracle's. A total
    // of 0 means 2^64, which no threshold can hold, so then nothing is dropped.
    private static void AssertThresholdsMatchOracle((float Low, float High)[] ranges)
    {
        ulong[] graph = GraphThresholds(ranges);
        for (int r = 0; r < ranges.Length; r++)
        {
            var table = RngDenseUniformOracle.Build(ranges[r].Low, ranges[r].High);
            if (table.Fixed is not null) continue;
            ulong[] got = [.. graph.Skip(r * RngDenseThresholdTable.Slots)
                .Take(RngDenseThresholdTable.Slots).Distinct()
                .Where(t => table.Total == 0 || t != table.Total).Order()];
            Assert.Equal([.. table.Blocks.Select(b => b.Threshold)], got);
        }
    }

    [Fact] public void TestInGraphDenseThresholdsMatchTheOracleOnPlainRanges()
        => AssertThresholdsMatchOracle(PlainRanges);
    [Fact] public void TestInGraphDenseThresholdsMatchTheOracleOnWideAndFullDomainRanges()
        => AssertThresholdsMatchOracle(WideAndFullDomainRanges);
    [Fact] public void TestInGraphDenseThresholdsMatchTheOracleOnSubnormalRanges()
        => AssertThresholdsMatchOracle(SubnormalRanges);
    [Fact] public void TestInGraphDenseThresholdsMatchTheOracleOnNarrowAndInfiniteBoundRanges()
        => AssertThresholdsMatchOracle(NarrowAndInfiniteBoundRanges);
    [Fact] public void TestInGraphDenseThresholdsMatchTheOracleAcrossTheTruncationFloor()
        => AssertThresholdsMatchOracle(AcrossTheTruncationFloorRanges);
    [Fact] public void TestInGraphDenseThresholdsMatchTheOracleReachingTheFloorFromOneSide()
        => AssertThresholdsMatchOracle(ReachingTheFloorFromOneSideRanges);
    [Fact] public void TestInGraphDenseThresholdsMatchTheOracleOnSubDeltaCollapsedSpans()
        => AssertThresholdsMatchOracle(SubDeltaCollapsedSpanRanges);
    [Fact] public void TestInGraphDenseThresholdsMatchTheOracleWhenBothRaysHoldALowPartial()
        => AssertThresholdsMatchOracle(BothRaysLowPartialRanges);

    [Fact] public void TestInGraphDenseUniformMatchesTheOracleOnPlainRanges()
        => Assert.True(DenseMatchesOracle(PlainRanges, roundtrip: true));
    [Fact] public void TestInGraphDenseUniformMatchesTheOracleOnWideAndFullDomainRanges()
        => Assert.True(DenseMatchesOracle(WideAndFullDomainRanges));
    [Fact] public void TestInGraphDenseUniformMatchesTheOracleOnSubnormalRanges()
        => Assert.True(DenseMatchesOracle(SubnormalRanges));
    [Fact] public void TestInGraphDenseUniformMatchesTheOracleOnNarrowAndInfiniteBoundRanges()
        => Assert.True(DenseMatchesOracle(NarrowAndInfiniteBoundRanges));
    [Fact] public void TestInGraphDenseUniformMatchesTheOracleAcrossTheTruncationFloor()
        => Assert.True(DenseMatchesOracle(AcrossTheTruncationFloorRanges));
    [Fact] public void TestInGraphDenseUniformMatchesTheOracleReachingTheFloorFromOneSide()
        => Assert.True(DenseMatchesOracle(ReachingTheFloorFromOneSideRanges));
    [Fact] public void TestInGraphDenseUniformMatchesTheOracleOnSubDeltaCollapsedSpans()
        => Assert.True(DenseMatchesOracle(SubDeltaCollapsedSpanRanges));
    [Fact] public void TestInGraphDenseUniformMatchesTheOracleWhenBothRaysHoldALowPartial()
        => Assert.True(DenseMatchesOracle(BothRaysLowPartialRanges));

    [Fact]
    public void TestInGraphDenseUniformOnTheUnitIntervalIsWalkerReynolds()
        => Assert.True(AutoTest.AdvancedTestGraph<RngDenseIsWalkerOnTheUnitInterval>(
            hyperparamInputs: [], runtimeInputs: [TensorData([2048L], new float[2048])]));

    private static (float Low, float High) StraddlingNegativePowerOfTwo(int exponent)
    {
        float p = MathF.ScaleB(-1f, exponent);
        return (MathF.BitDecrement(p), MathF.BitIncrement(p));
    }

    [Fact]
    public void TestInGraphDenseUniformMatchesTheOracleWhenTheTopFloatIsANegativePowerOfTwo()
        => Assert.True(DenseMatchesOracle([.. (( int[])[0, 1, -1, 10, -20, 60]).Select(StraddlingNegativePowerOfTwo)]));

    [Fact]
    public void TestInGraphDenseUniformMatchesTheOracleOnDegenerateBounds()
        => Assert.True(DenseMatchesOracle([
            (3f, 3f), (7f, 3f), (float.NaN, 1f), (1f, float.NaN), (float.NaN, float.NaN), (0f, 0f)]));

    [Fact]
    public void TestInGraphDenseUniformMatchesTheOracleOnNonFiniteBounds()
        => Assert.True(DenseMatchesOracle([
            (float.NegativeInfinity, float.PositiveInfinity), (float.NegativeInfinity, 0f),
            (-1f, float.PositiveInfinity), (float.PositiveInfinity, float.NegativeInfinity),
            (-0f, 0f), (1f, 1.0000001f)]));

    [Fact]
    public void TestRegionSelectionOpsBehaveAsCharacterizedInOnnxRuntime()
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
        for (long i = 0; i < 16; i++) Assert.Equal(HostDenseUniform(i, defaultKey, 0), vals[i]);

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

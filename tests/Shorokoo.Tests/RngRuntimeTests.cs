using System;
using System.Collections.Generic;
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
/// The uint64 op/dtype pairs a threshold-driven draw rests on, checked in-graph so both ONNX
/// Runtime and the Quick Execution Engine must agree: a computed (non-constant) uint64 table
/// gathered with computed indices, a restoring long division, a running max, and a binary search
/// over the table checked against a linear scan.
///
/// <para>The dense uniform draw no longer builds a 128-slot table, divides, or searches — its
/// thresholds are exact cumulative weights over seven blocks. These are kept as a pin on the ops
/// themselves, which stayed load-bearing elsewhere and were expensive to characterize: uint64 is
/// required because <c>Max</c> on <c>int64</c> mis-orders operands in [2^31, 2^32) and a running
/// max walks straight through that band, and the division uses arithmetic selection rather than
/// <c>Where</c>, multiplication by two rather than <c>BitShift</c>, and <c>Greater</c> negated
/// rather than <c>GreaterOrEqual</c> — a uint64 <c>Where</c> is unimplemented in ORT, and these
/// are the forms it does implement.</para>
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
    // indexes the whole counter; uniform = low 24 bits of x0 * 2^-24.
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
        // A collapsed span narrower than one delta: it holds no lattice point and degenerates to
        // its stub alone — the low end's when the span sits at the low end of the range, the high
        // end's when it sits at the high end.
        (FloorBoundaryNeighbour, 16f), (-16f, -FloorBoundaryNeighbour),
        // A two-sided block with a one-sided block above it, on each sign in turn.
        (-1e30f, 1f), (-1f, 1e30f),
        // One float: the top of a weight class, and the smallest subnormal of the negative ray.
        (0.99999994f, 1f), (-float.Epsilon, 0f),
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
            Assert.True(table.Blocks.Length <= DenseMaxBlocks);
        }
    }

    // The six kinds of block, plus the second partial class: lattice, both-signs classes,
    // one-sign classes, a partial class at each end of the range, and a stub at each end of the
    // lattice. No range can produce more, and a stub excludes the partial on its side.
    private const int DenseMaxBlocks = 7;

    private static long DenseElements(RngDenseUniformOracle.Block b)
        => b.Kind switch
        {
            RngDenseUniformOracle.Kind.Lattice => (long)b.Weight,
            RngDenseUniformOracle.Kind.TwoSided => (long)(b.C1 - b.C0 + 1) << (RngDenseUniformOracle.P + 1),
            RngDenseUniformOracle.Kind.OneSided => (long)(b.C1 - b.C0 + 1) << RngDenseUniformOracle.P,
            RngDenseUniformOracle.Kind.Ordinals => (long)(b.Weight >> b.Shift),
            _ => 1,
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

    // On [0,1) the dense draw IS Walker/Reynolds above the truncation floor, bit for bit. The range
    // does not straddle zero, so it keeps 41 classes and totals exactly 2^64: the scaling is the
    // identity, the lattice takes [0, 2^23) so a class block's offset is the draw itself, the
    // leading-bit search is Walker's leading-zero count over the 41-bit exponent field, and the
    // index's low bits are Walker's mantissa. It does NOT hold below the floor, where Walker folds
    // everything under 2^23 into one binade and this draw switches to an even lattice reaching
    // exact zero — past what sampling reaches, so both sides are asserted directly.
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
            Assert.Equal(table.Total, 2 * negative);
        }
    }

    // The truncation depth is chosen so the total tops out at exactly 2^64 — the largest value the
    // wrap sentinel can carry. Nothing in the build clamps it, so overflow would be silent: a
    // random sweep over the whole ordinal domain is what pins it.
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

    // Truncation is the one approximation, and this is its size: the fraction of a range's floats
    // that stays individually reachable. Both numbers are quoted in RuntimeRng's header.
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

    // Every weight unit decodes to a float the range contains, and each float takes exactly the
    // number of units its ulp is worth — the contract itself, by exhaustion rather than sampling.
    // Only ranges near zero have a small enough total to enumerate: any range deep enough to reach
    // the truncation floor totals about 2^64 by construction. So this covers the lattice, the
    // partial classes and the class-boundary crossings; the whole-class blocks are covered exactly
    // by the Walker and two-sided closed forms above.
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

    // The two-sided decode, against a closed form written out independently: on (-1,1) the range
    // straddles zero so it keeps 40 classes and totals exactly 2^64, and the draw's leading bit
    // picks the class while its low 24 bits are the sign and mantissa. The negative half runs
    // downward from the binade above, because a negative ordinal takes its class from the
    // magnitude pattern below it — hence 2 - mantissa rather than 1 + mantissa.
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

    // The graph's seven thresholds are the oracle's blocks in the same order, except that an empty
    // block carries the following one's threshold and the trailing empties carry the total. So the
    // graph's distinct thresholds, minus that total, are exactly the oracle's — and a total of 0
    // means 2^64, which no threshold can hold, so then nothing is dropped.
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

    [Fact]
    public void TestInGraphDenseUniformMatchesTheOracleOnSubDeltaCollapsedSpans()
        => Assert.True(DenseMatchesOracle(DenseBatch(6)));

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

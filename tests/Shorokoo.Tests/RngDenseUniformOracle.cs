using System;
using System.Collections.Generic;
using Shorokoo.Core.Rng;

namespace Shorokoo.Tests;

/// <summary>
/// Host oracle for the dense arbitrary-range uniform draw — <b>test-only</b>, and an independent
/// rebuild rather than a call into the graph.
///
/// <para>Floats are addressed by <b>signed ordinal</b> z: the bit pattern for x >= 0, minus the
/// magnitude's pattern for x &lt; 0. z is strictly monotone in the real value, so any range —
/// straddling or not — is a single interval [zLow, zHigh), the sign needs no separate draw, and
/// <c>high - low</c> is never computed as a float. Float z owns the half-open real interval
/// [V(z), V(z+1)), whose width is the ulp of weight class max(1, MagnitudeOrdinal(z) >> 23) — the
/// max(1,...) is what puts the subnormals and the smallest normal binade in one class. Note the
/// asymmetry that convention forces: a negative ordinal's class comes from the magnitude pattern
/// <b>below</b> it, so weight class c on the negative side is the magnitudes (c&lt;&lt;23,
/// (c+1)&lt;&lt;23], one ordinal off from the binade. Weight classes still tile without gaps, which
/// is what the decode needs.</para>
///
/// <para>The interval is partitioned into <b>blocks</b>, not a table: the weight axis need not run
/// in value order, so each kind of material becomes one contiguous block with a closed-form decode
/// and nothing is looked up. Above the truncation floor 2^(floorClass-127) the material is whole
/// weight classes — those present on both signs, then those present on one — plus at most one
/// partial class at each end of the range. Below the floor a coarse even lattice of spacing delta =
/// 2^(floorClass-150) carries the remaining mass, as a single region spanning both signs, with a
/// weight-1 stub at either end that the lattice does not reach.</para>
///
/// <para>Selection uses the WHOLE 64-bit draw. A block's threshold is its exact cumulative weight,
/// and the draw is scaled onto the weight axis by <c>floor(draw*total / 2^64)</c> — the high half of
/// the 128-bit product, Lemire's multiply-shift. The offset of the scaled draw above the chosen
/// block's threshold is in weight units. No block is quantized.</para>
///
/// <para>Within a whole-class block the index is the offset's LOW bits, not <c>offset >> shift</c>.
/// Both are exactly weight-preserving — a run of n indices each of weight 2^s spans n*2^s units, so
/// every residue mod n is hit exactly 2^s times — but the low-bits form is what makes the mantissa
/// fall out of the draw's low bits, which is what Walker/Reynolds does.</para>
///
/// <para>Depth depends on the range. A straddling range totals 2^(24+K), a one-sided one 2^(23+K),
/// so <see cref="StraddleClasses"/> = 40 and <see cref="MaxClasses"/> = 41 both top out at exactly
/// 2^64. That total is representable as the <see cref="Table.Total"/> sentinel 0, and no block ever
/// carries it as a threshold, so nothing needs a value above 2^64-1.</para>
///
/// <para>On [0,1) this draw is bit-for-bit Walker/Reynolds above the truncation floor: the range is
/// one-sided so K = 41, the total is exactly 2^64 and the scaling is the identity, the lattice
/// occupies [0, 2^23) so the class block's offset IS the draw, the leading-bit search IS a
/// leading-zero count, and the low 23 bits are the mantissa. Below the floor the two part ways by
/// construction: the lattice reaches exact zero where Walker stops at 2^-41 and doubles its bottom
/// binade's mass.</para>
/// </summary>
internal static class RngDenseUniformOracle
{
    public const int P = 23;
    public const int Bias = 127;
    public const int MinExponent = 1 - Bias;

    /// <summary>Truncation depth for a range that does not straddle zero.</summary>
    public const int MaxClasses = 41;

    /// <summary>Truncation depth for a range that does: one class shallower, because both signs of
    /// every class are present and the total would otherwise reach 2^65.</summary>
    public const int StraddleClasses = 40;

    private const long SignMask = 1L << 31;
    private const long SigMask = (1L << P) - 1;
    private const long BinadeSize = 1L << P;
    private const long MaxFiniteOrdinal = (254L << P) | SigMask;
    private const long InfinityOrdinal = 255L << P;

    internal enum Kind
    {
        /// <summary>Lattice points Base + idx, each of weight 1.</summary>
        Lattice,
        /// <summary>Whole weight classes C0..C1 present on both signs.</summary>
        TwoSided,
        /// <summary>Whole weight classes C0..C1 present on the sign given by <see cref="Block.Negative"/>.</summary>
        OneSided,
        /// <summary>A run of consecutive ordinals from Base, each of weight 2^Shift.</summary>
        Ordinals,
        /// <summary>The single float at ordinal Base, weight 1.</summary>
        OrdinalStub,
        /// <summary>The single lattice point Base, weight 1.</summary>
        LatticeStub,
    }

    /// <summary>One block of the weight axis. Threshold is its exact cumulative weight, Weight its
    /// own; the remaining fields are read according to <see cref="Kind"/>.</summary>
    internal readonly record struct Block(
        Kind Kind, ulong Threshold, ulong Weight, long Base, int C0, int C1, int Shift, bool Negative);

    /// <summary>Total is the summed weight, where <b>0 means exactly 2^64</b>. LatticeShift turns a
    /// lattice index into a bit pattern; FloorClass is the shallowest weight class kept whole.</summary>
    internal sealed record Table(
        Block[] Blocks, ulong Total, long ZLow, long ZHigh, int FloorClass, int LatticeShift, uint? Fixed);

    private static long SignedOrdinal(uint bits)
        => (bits & 0x8000_0000u) != 0 ? -(long)(bits & 0x7FFF_FFFFu) : (long)bits;

    private static uint BitsOfOrdinal(long z) => z >= 0 ? (uint)z : (uint)(SignMask | -z);

    private static long MagnitudeOrdinal(long z) => z >= 0 ? z : -z - 1;

    private static int ClassIndex(long z) => (int)Math.Max(1, MagnitudeOrdinal(z) >> P);

    /// <summary>|V(z)| divided by the lattice spacing 2^(L-P), truncated, plus whether it divided
    /// exactly. Pure int64: the significand carries at most 24 bits and the spacing only shifts.</summary>
    private static (long Quotient, bool Exact) MagnitudeOverSpacing(long magnitudeOrdinal, int floorExponent)
    {
        if (magnitudeOrdinal == 0) return (0, true);
        long significand;
        int exponent;
        if (magnitudeOrdinal < BinadeSize) { significand = magnitudeOrdinal; exponent = MinExponent; }
        else { exponent = (int)(magnitudeOrdinal >> P) - Bias; significand = BinadeSize | (magnitudeOrdinal & SigMask); }
        int shift = floorExponent - exponent;
        if (shift <= 0) return (shift <= -63 ? 0 : significand << -shift, true);
        if (shift >= 63) return (0, false);
        return (significand >> shift, (significand & ((1L << shift) - 1)) == 0);
    }

    private static long LatticeFloor(long z, int floorExponent)
    {
        var (q, exact) = MagnitudeOverSpacing(Math.Abs(z), floorExponent);
        return z >= 0 ? q : -(exact ? q : q + 1);
    }

    private static long LatticeCeil(long z, int floorExponent)
    {
        var (q, exact) = MagnitudeOverSpacing(Math.Abs(z), floorExponent);
        return z >= 0 ? (exact ? q : q + 1) : -q;
    }

    private static bool IsOnLattice(long z, int floorExponent)
        => MagnitudeOverSpacing(Math.Abs(z), floorExponent).Exact;

    /// <summary>The weight class c run on one sign: 2^23 consecutive ordinals. Negative classes are
    /// the magnitudes (c&lt;&lt;23, (c+1)&lt;&lt;23], so the run ends one ordinal below -(c&lt;&lt;23).</summary>
    private static long ClassRunStart(int c, bool negative)
        => negative ? -(((long)c + 1) << P) : (long)c << P;

    private static long ClassRunEnd(int c, bool negative)
        => negative ? -((long)c << P) : ((long)c + 1) << P;

    public static Table Build(float low, float high)
    {
        uint lowBits = BitConverter.SingleToUInt32Bits(low);
        uint highBits = BitConverter.SingleToUInt32Bits(high);
        if (float.IsNaN(low) || float.IsNaN(high))
            return new Table([], 0, 0, 0, 1, 0, 0x7FC0_0000u);

        uint clampedLow = float.IsInfinity(low)
            ? (uint)((low < 0 ? SignMask : 0) | MaxFiniteOrdinal) : lowBits;
        long zLow = SignedOrdinal(clampedLow);
        long zHigh = float.IsPositiveInfinity(high)
            ? InfinityOrdinal
            : SignedOrdinal(float.IsNegativeInfinity(high) ? (uint)(SignMask | MaxFiniteOrdinal) : highBits);

        if (zHigh <= zLow) return new Table([], 0, zLow, zHigh, 1, 0, clampedLow);

        int classes = zLow < 0 && zHigh > 0 ? StraddleClasses : MaxClasses;
        int topClass = Math.Max(ClassIndex(zLow), ClassIndex(zHigh - 1));
        int floorClass = Math.Max(1, topClass - classes + 1);
        int floorExponent = floorClass - Bias;
        long zFloor = (long)floorClass << P;
        long bandLow = Math.Clamp(-zFloor, zLow, zHigh);
        long bandHigh = Math.Clamp(zFloor, zLow, zHigh);

        var (negLow, negHigh, negC0, negC1) = SplitRun(zLow, bandLow, negative: true);
        var (posLow, posHigh, posC0, posC1) = SplitRun(bandHigh, zHigh, negative: false);

        // At most one partial class at each end of the whole range: an inner end of either run is a
        // class boundary by construction, so only the run holding zLow can have a low partial and
        // only the run holding zHigh a high partial.
        var lowPart = zLow < bandLow ? negLow : posLow;
        var highPart = bandHigh < zHigh ? posHigh : negHigh;

        bool negWhole = negC1 >= negC0, posWhole = posC1 >= posC0;
        int twoLow = 0, twoHigh = -1, oneLow = 0, oneHigh = -1;
        bool oneNegative = false;
        if (negWhole && posWhole)
        {
            int shared = Math.Min(negC1, posC1);
            (twoLow, twoHigh) = (Math.Max(negC0, posC0), shared);
            if (negC1 > shared) (oneLow, oneHigh, oneNegative) = (shared + 1, negC1, true);
            else if (posC1 > shared) (oneLow, oneHigh, oneNegative) = (shared + 1, posC1, false);
        }
        else if (negWhole) (oneLow, oneHigh, oneNegative) = (negC0, negC1, true);
        else if (posWhole) (oneLow, oneHigh, oneNegative) = (posC0, posC1, false);

        long latticeStart = 0, latticeCount = 0;
        long lowStub = 0, highStub = 0;
        bool hasLowStub = false, hasHighStub = false;
        if (bandLow < bandHigh)
        {
            long start = LatticeCeil(bandLow, floorExponent);
            long end = LatticeFloor(bandHigh, floorExponent);
            if (start > end) { (hasLowStub, lowStub) = (true, bandLow); }
            else
            {
                if (!IsOnLattice(bandLow, floorExponent)) (hasLowStub, lowStub) = (true, bandLow);
                (latticeStart, latticeCount) = (start, end - start);
                if (!IsOnLattice(bandHigh, floorExponent)) (hasHighStub, highStub) = (true, end);
            }
        }

        List<Block> blocks = [];
        if (latticeCount > 0)
            blocks.Add(new Block(Kind.Lattice, 0, (ulong)latticeCount, latticeStart, 0, 0, 0, false));
        if (twoHigh >= twoLow)
            blocks.Add(new Block(Kind.TwoSided, 0, ClassWeight(twoLow, twoHigh, floorClass, P + 1),
                0, twoLow, twoHigh, 0, false));
        if (oneHigh >= oneLow)
            blocks.Add(new Block(Kind.OneSided, 0, ClassWeight(oneLow, oneHigh, floorClass, P),
                0, oneLow, oneHigh, 0, oneNegative));
        Part[] partials = [lowPart, highPart];
        foreach (var part in partials)
            if (part.Count > 0)
                blocks.Add(new Block(Kind.Ordinals, 0, (ulong)part.Count << (part.Cls - floorClass),
                    part.Start, part.Cls, part.Cls, part.Cls - floorClass, part.Negative));
        if (hasLowStub) blocks.Add(new Block(Kind.OrdinalStub, 0, 1, lowStub, 0, 0, 0, false));
        if (hasHighStub) blocks.Add(new Block(Kind.LatticeStub, 0, 1, highStub, 0, 0, 0, false));

        Block[] table = [.. blocks];
        UInt128 cumulative = 0;
        for (int i = 0; i < table.Length; i++)
        {
            table[i] = table[i] with { Threshold = (ulong)cumulative };
            cumulative += table[i].Weight;
        }
        return new Table(table, (ulong)cumulative, zLow, zHigh, floorClass, floorClass - 1, null);
    }

    /// <summary>Summed weight of whole classes c0..c1 on one or both signs: each class contributes
    /// 2^perClass floats of weight 2^(c-floorClass), so the run is geometric.</summary>
    private static ulong ClassWeight(int c0, int c1, int floorClass, int perClass)
        => (1UL << (perClass + c0 - floorClass)) * ((1UL << (c1 - c0 + 1)) - 1);

    /// <summary>Decompose one sign's ordinal material [from, to) into a partial class at each end
    /// and the whole classes between. A partial that happens to cover its whole class is folded
    /// into the whole range instead, so the geometric blocks stay maximal. Classes ascend with the
    /// ordinal on the positive side and descend on the negative one.</summary>
    private static (Part Low, Part High, int C0, int C1) SplitRun(long from, long to, bool negative)
    {
        Part none = new(0, 0, 0, negative);
        if (from >= to) return (none, none, 0, -1);

        int cFrom = ClassIndex(from), cTo = ClassIndex(to - 1);
        long lowEnd = Math.Min(ClassRunEnd(cFrom, negative), to);
        long highStart = Math.Max(ClassRunStart(cTo, negative), from);
        bool lowWhole = from == ClassRunStart(cFrom, negative) && lowEnd == ClassRunEnd(cFrom, negative);
        bool highWhole = to == ClassRunEnd(cTo, negative) && highStart == ClassRunStart(cTo, negative);

        Part low = lowWhole ? none : new Part(from, lowEnd - from, cFrom, negative);
        Part high = highWhole || cTo == cFrom ? none : new Part(highStart, to - highStart, cTo, negative);
        return (low, high,
            negative ? cTo + (highWhole ? 0 : 1) : cFrom + (lowWhole ? 0 : 1),
            negative ? cFrom - (lowWhole ? 0 : 1) : cTo - (highWhole ? 0 : 1));
    }

    /// <summary>A partial weight class: <paramref name="Count"/> consecutive ordinals from
    /// <paramref name="Start"/>, all of class <paramref name="Cls"/>.</summary>
    internal readonly record struct Part(long Start, long Count, int Cls, bool Negative);

    /// <summary>The draw scaled onto the weight axis: floor(draw*total / 2^64), the high half of
    /// the 128-bit product. A total of 0 means exactly 2^64, where the scaling is the identity;
    /// mulhi by 0 is 0, so adding it back covers that case without a branch.</summary>
    public static ulong Scale(ulong draw, ulong total)
        => (ulong)(((UInt128)draw * total) >> 64) + (total == 0 ? draw : 0);

    /// <summary>Bit pattern of n*2^shift*delta. Exact: n carries at most 24 bits and scaling by a
    /// power of two only moves the exponent, so nothing rounds even into the subnormals.</summary>
    private static uint BitsOfLatticePoint(long n, int shift)
    {
        if (n == 0) return 0;
        uint sign = n < 0 ? (uint)SignMask : 0;
        ulong a = (ulong)Math.Abs(n);
        int hb = 63 - System.Numerics.BitOperations.LeadingZeroCount(a);
        int e = shift + hb;
        if (e < P) return sign | (uint)(a << shift);
        return sign | ((uint)(e - P + 1) << P) | (uint)((a - (1UL << hb)) << (P - hb));
    }

    /// <summary>Bit pattern of the mant'th float of weight class cls on the given sign. Positive
    /// classes are the binade; negative ones run down from the binade above, because a negative
    /// ordinal takes its class from the magnitude pattern below it.</summary>
    private static uint BitsOfClassMember(int cls, long mant, bool negative)
        => negative
            ? (uint)(SignMask | ((((long)cls + 1) << P) - mant))
            : (uint)(((long)cls << P) | mant);

    public static uint SampleBits(Table table, ulong draw)
    {
        if (table.Fixed is uint fixedResult) return fixedResult;
        ulong scaled = Scale(draw, table.Total);
        Block b = table.Blocks[0];
        for (int i = 1; i < table.Blocks.Length && table.Blocks[i].Threshold <= scaled; i++)
            b = table.Blocks[i];
        ulong d = scaled - b.Threshold;

        switch (b.Kind)
        {
            case Kind.Lattice:
                return BitsOfLatticePoint(b.Base + (long)d, table.LatticeShift);
            case Kind.TwoSided:
            case Kind.OneSided:
            {
                int width = b.Kind == Kind.TwoSided ? P + 1 : P;
                int m = width + b.C0 - table.FloorClass;
                ulong shifted = d + (1UL << m);
                int lead = 63 - System.Numerics.BitOperations.LeadingZeroCount(shifted);
                int cls = b.C0 + lead - m;
                long index = (long)(shifted & ((1UL << width) - 1));
                bool negative = b.Kind == Kind.TwoSided ? index >= BinadeSize : b.Negative;
                return BitsOfClassMember(cls, index & SigMask, negative);
            }
            case Kind.Ordinals:
                return BitsOfOrdinal(b.Base + (long)(d >> b.Shift));
            case Kind.LatticeStub:
                return BitsOfLatticePoint(b.Base, table.LatticeShift);
            default:
                return BitsOfOrdinal(b.Base);
        }
    }

    /// <summary>Element <paramref name="i"/> of a dense uniform draw over [low, high).</summary>
    public static float Draw(ulong key, ulong substreamIndex, long i, float low, float high,
        int rounds = Threefry2x32.Rounds)
        => BitConverter.UInt32BitsToSingle(
            SampleBits(Build(low, high), RngTestOracle.DrawValue(key, substreamIndex, i, rounds)));
}

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
/// [V(z), V(z+1)), whose width is the ulp of weight class max(1, MagnitudeOrdinal(z) >> P) — the
/// max(1,...) is what puts the subnormals and the smallest normal binade in one class. Note the
/// asymmetry that convention forces: a negative ordinal's class comes from the magnitude pattern
/// <b>below</b> it, so weight class c on the negative side is the magnitudes (c&lt;&lt;P,
/// (c+1)&lt;&lt;P], one ordinal off from the binade. Weight classes still tile without gaps, which
/// is what the decode needs.</para>
///
/// <para>The interval is partitioned into <b>blocks</b>, not a table: the weight axis need not run
/// in value order, so each kind of material becomes one contiguous block with a closed-form decode
/// and nothing is looked up. Above the truncation floor 2^(floorClass-Bias) the material is whole
/// weight classes — those present on both signs, then those present on one — plus a partial class
/// at each end of each ray. Below the floor a coarse even lattice of spacing delta =
/// 2^(floorClass-1-Bias-P) carries the remaining mass, as a single region spanning both signs.
/// Where the range cuts the span mid-cell the leftover sliver is dropped, not rounded up: its true
/// share is under one weight unit, so the axis cannot express it, and paying it a whole unit
/// over-weighted that region by up to 2^191. One consequence is deliberate — a bound lying strictly
/// inside the span, off the lattice, is not itself drawable, which is what truncation says of every
/// float down there.</para>
///
/// <para>Selection uses the WHOLE draw. A block's threshold is its exact cumulative weight, and the
/// draw is scaled onto the weight axis by <c>floor(draw*total / 2^W)</c> — the high half of the
/// double-width product, Lemire's multiply-shift. The offset of the scaled draw above the chosen
/// block's threshold is in weight units. No block is quantized.</para>
///
/// <para>Within a whole-class block the index is the offset's LOW bits, not <c>offset >> shift</c>.
/// Both are exactly weight-preserving — a run of n indices each of weight 2^s spans n*2^s units, so
/// every residue mod n is hit exactly 2^s times — but the low-bits form is what makes the mantissa
/// fall out of the draw's low bits, which is what Walker/Reynolds does.</para>
///
/// <para>Depth depends on the range. A straddling range totals 2^(P+1+K), a one-sided one 2^(P+K),
/// so <see cref="Format.StraddleClasses"/> and <see cref="Format.MaxClasses"/> both top out at
/// exactly 2^W. That total is representable as the <see cref="Table.Total"/> sentinel 0, and no
/// block ever carries it as a threshold, so nothing needs a value above 2^W-1.</para>
///
/// <para>On [0,1) this draw is bit-for-bit Walker/Reynolds above the truncation floor: the range is
/// one-sided so K = 41, the total is exactly 2^64 and the scaling is the identity, the lattice
/// occupies [0, 2^23) so the class block's offset plus 2^23 IS the draw, the leading-bit search IS a
/// leading-zero count, and the low 23 bits are the mantissa. Below the floor the two part ways by
/// construction: the lattice reaches exact zero where Walker stops at 2^-41 and doubles its bottom
/// binade's mass.</para>
/// </summary>
internal static class RngDenseUniformOracle
{
    public const int P = 23;

    /// <summary>A binary float format plus the width of the draw its weight axis is scaled onto.
    /// Everything else follows, so shrinking W alongside P gives a format small enough to enumerate
    /// exhaustively while the construction stays the same one float32 uses.</summary>
    internal readonly record struct Format(int E, int P, int W)
    {
        public static readonly Format Float32 = new(8, RngDenseUniformOracle.P, 64);

        public int Bias => (1 << (E - 1)) - 1;
        public int MinExponent => 1 - Bias;

        /// <summary>Truncation depth for a range that does not straddle zero.</summary>
        public int MaxClasses => W - P;

        /// <summary>Truncation depth for a range that does: one class shallower, because both signs
        /// of every class are present and the total would otherwise reach 2^(W+1).</summary>
        public int StraddleClasses => W - P - 1;

        public long SignBit => 1L << (E + P);
        public long SigMask => (1L << P) - 1;
        public long BinadeSize => 1L << P;
        public long MaxFiniteOrdinal => (((1L << E) - 2) << P) | SigMask;
        public long InfinityOrdinal => ((1L << E) - 1) << P;
    }

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
    }

    /// <summary>One block of the weight axis. Threshold is its exact cumulative weight, Weight its
    /// own; the remaining fields are read according to <see cref="Kind"/>.</summary>
    internal readonly record struct Block(
        Kind Kind, ulong Threshold, ulong Weight, long Base, int C0, int C1, int Shift, bool Negative);

    /// <summary>Total is the summed weight, and 0 is <b>overloaded</b>: on a drawing table it means
    /// exactly 2^W, but the three degenerate builds — NaN low, NaN high, and an empty or inverted
    /// range — also carry 0, there meaning no weight at all, over no blocks. <see cref="Fixed"/> is
    /// what separates the two and must be read first, as <see cref="SampleBits"/> does; the field
    /// alone cannot tell <c>Build(3f, 3f)</c> from <c>Build(0f, 1f)</c>. LatticeShift turns a
    /// lattice index into a bit pattern; FloorClass is the shallowest weight class kept whole.</summary>
    internal sealed record Table(
        Format Format, Block[] Blocks, ulong Total, long ZLow, long ZHigh,
        int FloorClass, int LatticeShift, uint? Fixed);

    private static long SignedOrdinal(Format fmt, uint bits)
        => (bits & (uint)fmt.SignBit) != 0 ? -(long)(bits & (uint)(fmt.SignBit - 1)) : bits;

    private static uint BitsOfOrdinal(Format fmt, long z) => z >= 0 ? (uint)z : (uint)(fmt.SignBit | -z);

    private static long MagnitudeOrdinal(long z) => z >= 0 ? z : -z - 1;

    private static int ClassIndex(Format fmt, long z) => (int)Math.Max(1, MagnitudeOrdinal(z) >> fmt.P);

    /// <summary>|V(z)| divided by the lattice spacing 2^(L-P), truncated, plus whether it divided
    /// exactly. Pure int64: the significand carries at most P+1 bits and the spacing only shifts.</summary>
    private static (long Quotient, bool Exact) MagnitudeOverSpacing(Format fmt, long magnitudeOrdinal, int floorExponent)
    {
        if (magnitudeOrdinal == 0) return (0, true);
        long significand;
        int exponent;
        if (magnitudeOrdinal < fmt.BinadeSize) { significand = magnitudeOrdinal; exponent = fmt.MinExponent; }
        else { exponent = (int)(magnitudeOrdinal >> fmt.P) - fmt.Bias; significand = fmt.BinadeSize | (magnitudeOrdinal & fmt.SigMask); }
        // Callers only ask about a magnitude inside the collapsed span, so the value is at most
        // 2^floorExponent and the shift is never negative. A shift of 0 falls through correctly:
        // the significand survives and the empty mask makes it exact.
        int shift = floorExponent - exponent;
        if (shift >= 63) return (0, false);
        return (significand >> shift, (significand & ((1L << shift) - 1)) == 0);
    }

    private static long LatticeFloor(Format fmt, long z, int floorExponent)
    {
        var (q, exact) = MagnitudeOverSpacing(fmt, Math.Abs(z), floorExponent);
        return z >= 0 ? q : -(exact ? q : q + 1);
    }

    private static long LatticeCeil(Format fmt, long z, int floorExponent)
    {
        var (q, exact) = MagnitudeOverSpacing(fmt, Math.Abs(z), floorExponent);
        return z >= 0 ? (exact ? q : q + 1) : -q;
    }

    /// <summary>The weight class c run on one sign: 2^P consecutive ordinals. Negative classes are
    /// the magnitudes (c&lt;&lt;P, (c+1)&lt;&lt;P], so the run ends one ordinal below -(c&lt;&lt;P).</summary>
    private static long ClassRunStart(Format fmt, int c, bool negative)
        => negative ? -(((long)c + 1) << fmt.P) : (long)c << fmt.P;

    private static long ClassRunEnd(Format fmt, int c, bool negative)
        => negative ? -((long)c << fmt.P) : ((long)c + 1) << fmt.P;

    public static Table Build(float low, float high)
        => Build(Format.Float32,
            BitConverter.SingleToUInt32Bits(low), BitConverter.SingleToUInt32Bits(high));

    public static Table Build(Format fmt, uint lowBits, uint highBits)
    {
        long lowMagnitude = lowBits & (fmt.SignBit - 1), highMagnitude = highBits & (fmt.SignBit - 1);
        // The NaN that came in, bits intact — IEEE 754 asks an operation to propagate an input
        // NaN's payload rather than mint a canonical one. `low` wins when both bounds are NaN.
        if (lowMagnitude > fmt.InfinityOrdinal) return new Table(fmt, [], 0, 0, 0, 1, 0, lowBits);
        if (highMagnitude > fmt.InfinityOrdinal) return new Table(fmt, [], 0, 0, 0, 1, 0, highBits);

        // An infinite bound clamps to the finite extreme of its own sign. As `low` that means
        // -MaxValue, or MaxValue for +infinity — so +infinity as `low` leaves every finite `high`
        // inverted and the draw is MaxValue, not +infinity, and Build(+inf, +inf) is not empty but
        // the one-float range [MaxValue, +inf). As `high`, +infinity instead becomes the ordinal
        // one PAST MaxValue, so the whole finite domain stays reachable.
        uint clampedLow = lowMagnitude == fmt.InfinityOrdinal
            ? (uint)((lowBits & (uint)fmt.SignBit) | (uint)fmt.MaxFiniteOrdinal) : lowBits;
        long zLow = SignedOrdinal(fmt, clampedLow);
        long zHigh = highMagnitude != fmt.InfinityOrdinal ? SignedOrdinal(fmt, highBits)
            : (highBits & (uint)fmt.SignBit) != 0 ? -fmt.MaxFiniteOrdinal : fmt.InfinityOrdinal;

        if (zHigh <= zLow) return new Table(fmt, [], 0, zLow, zHigh, 1, 0, clampedLow);

        int classes = zLow < 0 && zHigh > 0 ? fmt.StraddleClasses : fmt.MaxClasses;
        int topClass = Math.Max(ClassIndex(fmt, zLow), ClassIndex(fmt, zHigh - 1));
        int floorClass = Math.Max(1, topClass - classes + 1);
        int floorExponent = floorClass - fmt.Bias;
        long zFloor = (long)floorClass << fmt.P;
        long bandLow = Math.Clamp(-zFloor, zLow, zHigh);
        long bandHigh = Math.Clamp(zFloor, zLow, zHigh);

        var (negLow, negHigh, negC0, negC1) = SplitRun(fmt, zLow, bandLow, negative: true);
        var (posLow, posHigh, posC0, posC1) = SplitRun(fmt, bandHigh, zHigh, negative: false);

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
        // The lattice runs over the points the span wholly contains. At most one end is ever cut:
        // either the other end is a floor boundary and so on the lattice, or the range lies wholly
        // inside the span, which forces topClass == floorClass and hence both to 1, where delta is
        // the smallest subnormal and the lattice IS the float grid, so neither end is cut at all.
        // The partial cell left over at a cut end is dropped rather than rounded up to a weight
        // unit. Its true share is under one unit, which is below what the weight axis can express at
        // all, and paying it a whole unit was the scheme's single worst distortion: it over-weighted
        // the sliver's region by up to 2^191, and handed `low` some 2^64 times its due. Dropping it
        // also stops `low` being an exception to truncation — it lies below the floor, where nothing
        // is individually reachable.
        if (bandLow < bandHigh)
        {
            long start = LatticeCeil(fmt, bandLow, floorExponent);
            long end = LatticeFloor(fmt, bandHigh, floorExponent);
            if (start <= end) (latticeStart, latticeCount) = (start, end - start);
        }

        List<Block> blocks = [];
        if (latticeCount > 0)
            blocks.Add(new Block(Kind.Lattice, 0, (ulong)latticeCount, latticeStart, 0, 0, 0, false));
        if (twoHigh >= twoLow)
            blocks.Add(new Block(Kind.TwoSided, 0, ClassWeight(twoLow, twoHigh, floorClass, fmt.P + 1),
                0, twoLow, twoHigh, 0, false));
        if (oneHigh >= oneLow)
            blocks.Add(new Block(Kind.OneSided, 0, ClassWeight(oneLow, oneHigh, floorClass, fmt.P),
                0, oneLow, oneHigh, 0, oneNegative));
        // Every partial each ray reports, not a chosen two. Picking two needs an argument about
        // which can coexist, and the obvious one is wrong: when the range straddles zero and stops
        // INSIDE the floor class on the positive side, the negative ray's low partial and the
        // positive ray's low partial are both real, and dropping either loses floats outright.
        // Negative stays false: an Ordinals block takes its sign from the ordinal itself, so the
        // field is unread here, and the graph writes a literal 0 into those four slots.
        Part[] partials = [negLow, negHigh, posLow, posHigh];
        foreach (var part in partials)
            if (part.Count > 0)
                blocks.Add(new Block(Kind.Ordinals, 0, (ulong)part.Count << (part.Cls - floorClass),
                    part.Start, part.Cls, part.Cls, part.Cls - floorClass, false));

        Block[] table = [.. blocks];
        UInt128 cumulative = 0;
        for (int i = 0; i < table.Length; i++)
        {
            table[i] = table[i] with { Threshold = (ulong)cumulative };
            cumulative += table[i].Weight;
        }
        return new Table(fmt, table, (ulong)(cumulative & (((UInt128)1 << fmt.W) - 1)),
            zLow, zHigh, floorClass, floorClass - 1, null);
    }

    /// <summary>Summed weight of whole classes c0..c1 on one or both signs: each class contributes
    /// 2^perClass floats of weight 2^(c-floorClass), so the run is geometric.</summary>
    private static ulong ClassWeight(int c0, int c1, int floorClass, int perClass)
        => (1UL << (perClass + c0 - floorClass)) * ((1UL << (c1 - c0 + 1)) - 1);

    /// <summary>Decompose one sign's ordinal material [from, to) into a partial class at each end
    /// and the whole classes between. A partial that happens to cover its whole class is folded
    /// into the whole range instead, so the geometric blocks stay maximal. Classes ascend with the
    /// ordinal on the positive side and descend on the negative one.
    ///
    /// <para>Class 1 is where the run convention and <see cref="ClassIndex"/> disagree: the run is
    /// 2^P ordinals, but max(1, ...) also folds the subnormals into class 1, so a negative run
    /// ending above -2^P would mis-tile. <see cref="Build"/> never produces one — the band's edge
    /// is either -(floorClass&lt;&lt;P), which is at most -2^P, or below it — but the invariant
    /// lives at the caller, not here.</para></summary>
    private static (Part Low, Part High, int C0, int C1) SplitRun(Format fmt, long from, long to, bool negative)
    {
        Part none = new(0, 0, 0);
        if (from >= to) return (none, none, 0, -1);

        int cFrom = ClassIndex(fmt, from), cTo = ClassIndex(fmt, to - 1);
        long lowEnd = Math.Min(ClassRunEnd(fmt, cFrom, negative), to);
        long highStart = Math.Max(ClassRunStart(fmt, cTo, negative), from);
        bool lowWhole = from == ClassRunStart(fmt, cFrom, negative) && lowEnd == ClassRunEnd(fmt, cFrom, negative);
        bool highWhole = to == ClassRunEnd(fmt, cTo, negative) && highStart == ClassRunStart(fmt, cTo, negative);

        Part low = lowWhole ? none : new Part(from, lowEnd - from, cFrom);
        Part high = highWhole || cTo == cFrom ? none : new Part(highStart, to - highStart, cTo);
        return (low, high,
            negative ? cTo + (highWhole ? 0 : 1) : cFrom + (lowWhole ? 0 : 1),
            negative ? cFrom - (lowWhole ? 0 : 1) : cTo - (highWhole ? 0 : 1));
    }

    /// <summary>A partial weight class: <paramref name="Count"/> consecutive ordinals from
    /// <paramref name="Start"/>, all of class <paramref name="Cls"/>.</summary>
    internal readonly record struct Part(long Start, long Count, int Cls);

    /// <summary>The draw scaled onto the weight axis: floor(draw*total / 2^W), the high half of
    /// the double-width product. A total of 0 means exactly 2^W, where the scaling is the identity;
    /// mulhi by 0 is 0, so adding it back covers that case without a branch.</summary>
    public static ulong Scale(Format fmt, ulong draw, ulong total)
        => (ulong)(((UInt128)draw * total) >> fmt.W) + (total == 0 ? draw : 0);

    /// <summary>Bit pattern of n*2^shift*delta. Exact: n carries at most P+1 bits and scaling by a
    /// power of two only moves the exponent, so nothing rounds even into the subnormals.</summary>
    private static uint BitsOfLatticePoint(Format fmt, long n, int shift)
    {
        if (n == 0) return 0;
        uint sign = n < 0 ? (uint)fmt.SignBit : 0;
        ulong a = (ulong)Math.Abs(n);
        int hb = 63 - System.Numerics.BitOperations.LeadingZeroCount(a);
        int e = shift + hb;
        if (e < fmt.P) return sign | (uint)(a << shift);
        return sign | ((uint)(e - fmt.P + 1) << fmt.P) | (uint)((a - (1UL << hb)) << (fmt.P - hb));
    }

    /// <summary>Bit pattern of the mant'th float of weight class cls on the given sign. Positive
    /// classes are the binade; negative ones run down from the binade above, because a negative
    /// ordinal takes its class from the magnitude pattern below it.
    ///
    /// <para>The negative form spells -infinity at the top class, mantissa 0. Nothing here prevents
    /// it: it is unreachable only because a low bound of -infinity clamps to -MaxValue, so the
    /// ordinal -((2^E-1)&lt;&lt;P) never falls in range and that negative class is never whole.
    /// Change that clamp and the draw returns -infinity.</para></summary>
    private static uint BitsOfClassMember(Format fmt, int cls, long mant, bool negative)
        => negative
            ? (uint)(fmt.SignBit | ((((long)cls + 1) << fmt.P) - mant))
            : (uint)(((long)cls << fmt.P) | mant);

    public static uint SampleBits(Table table, ulong draw)
    {
        if (table.Fixed is uint fixedResult) return fixedResult;
        Format fmt = table.Format;
        ulong scaled = Scale(fmt, draw, table.Total);
        Block b = table.Blocks[0];
        for (int i = 1; i < table.Blocks.Length && table.Blocks[i].Threshold <= scaled; i++)
            b = table.Blocks[i];
        ulong d = scaled - b.Threshold;

        switch (b.Kind)
        {
            case Kind.Lattice:
                return BitsOfLatticePoint(fmt, b.Base + (long)d, table.LatticeShift);
            case Kind.TwoSided:
            case Kind.OneSided:
            {
                int width = b.Kind == Kind.TwoSided ? fmt.P + 1 : fmt.P;
                int m = width + b.C0 - table.FloorClass;
                ulong shifted = d + (1UL << m);
                int lead = 63 - System.Numerics.BitOperations.LeadingZeroCount(shifted);
                int cls = b.C0 + lead - m;
                long index = (long)(shifted & ((1UL << width) - 1));
                bool negative = b.Kind == Kind.TwoSided ? index >= fmt.BinadeSize : b.Negative;
                return BitsOfClassMember(fmt, cls, index & fmt.SigMask, negative);
            }
            default:
                return BitsOfOrdinal(fmt, b.Base + (long)(d >> b.Shift));
        }
    }

    /// <summary>Element <paramref name="i"/> of a dense uniform draw over [low, high).</summary>
    public static float Draw(ulong key, ulong substreamIndex, long i, float low, float high,
        int rounds = Threefry2x32.Rounds)
        => BitConverter.UInt32BitsToSingle(
            SampleBits(Build(low, high), RngTestOracle.DrawValue(key, substreamIndex, i, rounds)));
}

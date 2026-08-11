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
/// [V(z), V(z+1)), whose width is the ulp of weight class max(1, exponent field) — the max(1,...)
/// is what puts the subnormals and the smallest normal binade in one class.</para>
///
/// <para>The interval is partitioned into regions, each an arithmetic progression of
/// 2^IndexBits representable floats carrying one weight each. Above the truncation floor
/// 2^(L) the regions are the float grid itself, cut into class groups and then into descending
/// power-of-two blocks; below it a coarse even lattice of spacing 2^(L-P) carries the remaining
/// mass.</para>
///
/// <para>Selection uses the WHOLE 64-bit draw. A region's threshold is its exact cumulative
/// weight, and the draw is scaled onto the weight axis by <c>floor(draw*total / 2^64)</c> — the
/// high half of the 128-bit product, Lemire's multiply-shift. The offset of the scaled draw above
/// the chosen region's threshold is in weight units, so the index within the region is that offset
/// shifted right by <see cref="Slot.Shift"/>. No region is quantized: the old 41-bit selector
/// rounded a sub-2^-41 region UP to a 2^-41 share, over-weighting a weight-1 stub by 2^20.</para>
///
/// <para>Every weight is expressed in <b>spacing units</b> (multiples of the shallowest kept
/// class's ulp). A range straddling zero totals 2^(24+K), so K = 39 is the ceiling: the trailing
/// empty slots carry the total as their threshold, and at K = 40 that total is exactly 2^64 —
/// unrepresentable, so those slots would wrap to 0 and destroy the table's monotonicity, which
/// the search depends on. Scaling itself survives 2^64 (mulhi by a wrapped 0 total plus the draw
/// is the identity); the THRESHOLD table is what caps K, and no sentinel above 2^64-1 exists.</para>
///
/// <para>On [0,1) above the truncation floor this draw follows Walker/Reynolds' geometric law,
/// structurally: the total is an exact power of two, so the scaling is an exact shift, the binade
/// thresholds are 2^60, 2^59, … so the search IS a leading-zero count, and Shift extracts the
/// mantissa. It is no longer bit-for-bit against a 41/23 split, which read the mantissa from the
/// draw's LOW bits; the mantissa now comes from the bits under the leading one. Below the floor
/// the law does not hold at all: the lattice reaches exact zero where Walker stops at 2^-41.</para>
/// </summary>
internal static class RngDenseUniformOracle
{
    public const int P = 23;
    public const int Bias = 127;
    public const int MinExponent = 1 - Bias;
    public const int MaxClasses = 38;
    public const int SearchRounds = 7;
    public const int MaxSlots = 1 << SearchRounds;

    private const long SignMask = 1L << 31;
    private const long SigMask = (1L << P) - 1;
    private const long BinadeSize = 1L << P;
    private const long MaxFiniteOrdinal = (254L << P) | SigMask;
    private const long InfinityOrdinal = 255L << P;

    /// <summary>One region: 2^IndexBits floats, Base + idx·1 in ordinal space, or the lattice
    /// points (Base + idx)·2^LatticeShift·delta when <see cref="Lattice"/>. Threshold is the
    /// region's exact cumulative weight; Shift is how many weight units one of its points spans,
    /// as a power of two, so Weight == 2^(IndexBits + Shift) always.</summary>
    internal readonly record struct Slot(
        ulong Threshold, long Base, int IndexBits, int Shift, bool Lattice, int LatticeShift, ulong Weight);

    /// <summary>Total is the summed weight, where <b>0 means exactly 2^64</b>.</summary>
    internal sealed record Table(Slot[] Slots, ulong Total, long ZLow, long ZHigh, int FloorClass, uint? Fixed);

    private static long SignedOrdinal(uint bits)
        => (bits & 0x8000_0000u) != 0 ? -(long)(bits & 0x7FFF_FFFFu) : (long)bits;

    private static uint BitsOfOrdinal(long z) => z >= 0 ? (uint)z : (uint)(SignMask | -z);

    private static long MagnitudeOrdinal(long z) => z >= 0 ? z : -z - 1;

    private static int ClassIndex(long z) => (int)Math.Max(1, MagnitudeOrdinal(z) >> P);

    private static long ClassGroupEnd(long z)
        => z >= 0 ? ((z >> P) + 1) << P : -(((-z - 1) >> P) << P);

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

    public static Table Build(float low, float high)
    {
        uint lowBits = BitConverter.SingleToUInt32Bits(low);
        uint highBits = BitConverter.SingleToUInt32Bits(high);
        if (float.IsNaN(low) || float.IsNaN(high))
            return new Table([], 0, 0, 0, 1, 0x7FC0_0000u);

        uint clampedLow = float.IsInfinity(low)
            ? (uint)((low < 0 ? SignMask : 0) | MaxFiniteOrdinal) : lowBits;
        long zLow = SignedOrdinal(clampedLow);
        long zHigh = float.IsPositiveInfinity(high)
            ? InfinityOrdinal
            : SignedOrdinal(float.IsNegativeInfinity(high) ? (uint)(SignMask | MaxFiniteOrdinal) : highBits);

        if (zHigh <= zLow) return new Table([], 0, zLow, zHigh, 1, clampedLow);

        int topClass = Math.Max(ClassIndex(zLow), ClassIndex(zHigh - 1));
        int floorClass = Math.Max(1, topClass - MaxClasses + 1);
        int floorExponent = floorClass - Bias;
        long zFloor = (long)floorClass << P;
        long bandLow = Math.Clamp(-zFloor, zLow, zHigh);
        long bandHigh = Math.Clamp(zFloor, zLow, zHigh);

        List<Slot> slots = [];
        AddOrdinalRun(slots, zLow, bandLow, floorClass);
        AddBand(slots, bandLow, bandHigh, floorClass, floorExponent);
        AddOrdinalRun(slots, bandHigh, zHigh, floorClass);

        Slot[] table = [.. slots];
        ulong cumulative = 0;
        for (int i = 0; i < table.Length; i++)
        {
            table[i] = table[i] with { Threshold = cumulative };
            cumulative += table[i].Weight;
        }
        return new Table(table, cumulative, zLow, zHigh, floorClass, null);
    }

    /// <summary>The draw scaled onto the weight axis: floor(draw·total / 2^64), the high half of
    /// the 128-bit product. A total of 0 means exactly 2^64, where the scaling is the identity;
    /// mulhi by 0 is 0, so adding it back covers that case without a branch.</summary>
    public static ulong Scale(ulong draw, ulong total)
        => (ulong)(((UInt128)draw * total) >> 64) + (total == 0 ? draw : 0);

    private static void AddOrdinalRun(List<Slot> slots, long from, long to, int floorClass)
    {
        long cur = from;
        while (cur < to)
        {
            long groupEnd = Math.Min(ClassGroupEnd(cur), to);
            int cls = ClassIndex(cur);
            while (cur < groupEnd)
            {
                int bits = 63 - System.Numerics.BitOperations.LeadingZeroCount((ulong)(groupEnd - cur));
                slots.Add(new Slot(0, cur, bits, cls - floorClass, false, 0, 1UL << (bits + cls - floorClass)));
                cur += 1L << bits;
            }
        }
    }

    private static void AddBand(List<Slot> slots, long from, long to, int floorClass, int floorExponent)
    {
        if (from >= to) return;
        long start = LatticeCeil(from, floorExponent);
        long end = LatticeFloor(to, floorExponent);
        int shift = floorClass - 1;

        if (start > end) { slots.Add(new Slot(0, from, 0, 0, false, 0, 1)); return; }
        if (LatticeFloor(from, floorExponent) != start || !IsOnLattice(from, floorExponent))
            slots.Add(new Slot(0, from, 0, 0, false, 0, 1));

        AddLatticeRun(slots, start, Math.Min(end, 0), shift);
        AddLatticeRun(slots, Math.Max(start, 0), end, shift);

        if (!IsOnLattice(to, floorExponent)) slots.Add(new Slot(0, end, 0, 0, true, shift, 1));
    }

    private static bool IsOnLattice(long z, int floorExponent)
        => MagnitudeOverSpacing(Math.Abs(z), floorExponent).Exact;

    private static void AddLatticeRun(List<Slot> slots, long from, long to, int shift)
    {
        long cur = from;
        while (cur < to)
        {
            int bits = 63 - System.Numerics.BitOperations.LeadingZeroCount((ulong)(to - cur));
            slots.Add(new Slot(0, cur, bits, 0, true, shift, 1UL << bits));
            cur += 1L << bits;
        }
    }

    /// <summary>Bit pattern of n·2^shift·delta. Exact: n carries at most 24 bits and scaling by a
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

    public static uint SampleBits(Table table, ulong draw)
    {
        if (table.Fixed is uint fixedResult) return fixedResult;
        ulong scaled = Scale(draw, table.Total);
        int lo = 0, hi = table.Slots.Length - 1;
        while (lo < hi)
        {
            int mid = (lo + hi + 1) >> 1;
            if (table.Slots[mid].Threshold <= scaled) lo = mid; else hi = mid - 1;
        }
        Slot s = table.Slots[lo];
        long idx = (long)((scaled - s.Threshold) >> s.Shift);
        return s.Lattice ? BitsOfLatticePoint(s.Base + idx, s.LatticeShift) : BitsOfOrdinal(s.Base + idx);
    }

    /// <summary>Element <paramref name="i"/> of a dense uniform draw over [low, high).</summary>
    public static float Draw(ulong key, ulong substreamIndex, long i, float low, float high,
        int rounds = Threefry2x32.Rounds)
        => BitConverter.UInt32BitsToSingle(
            SampleBits(Build(low, high), RngTestOracle.DrawValue(key, substreamIndex, i, rounds)));
}

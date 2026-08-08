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
/// mass. One 64-bit draw decodes as a 41-bit region selector against a cumulative threshold table
/// plus the top bits of a 23-bit index.</para>
///
/// <para>Every weight is expressed in <b>spacing units</b> (multiples of the shallowest kept
/// class's ulp), which keeps the cumulative table inside int64 by construction and caps the
/// truncation depth at K = 38. Thresholds come from restoring binary long division, because
/// 2^41·C overflows int64 and binary64 is not usable — the Quick Execution Engine evaluates every
/// float dtype in binary32 (Shorokoo#157).</para>
/// </summary>
internal static class RngDenseUniformOracle
{
    public const int P = 23;
    public const int Bias = 127;
    public const int MinExponent = 1 - Bias;
    public const int MaxClasses = 38;
    public const int SelectorBits = 64 - P;
    public const int MaxSlots = 128;

    private const long SignMask = 1L << 31;
    private const long SigMask = (1L << P) - 1;
    private const long BinadeSize = 1L << P;
    private const long MaxFiniteOrdinal = (254L << P) | SigMask;
    private const long InfinityOrdinal = 255L << P;

    /// <summary>One region: 2^IndexBits floats, Base + idx·1 in ordinal space, or the lattice
    /// points (Base + idx)·2^LatticeShift·delta when <see cref="Lattice"/>.</summary>
    internal readonly record struct Slot(
        long Threshold, long Base, int IndexBits, bool Lattice, int LatticeShift, long Weight);

    internal sealed record Table(Slot[] Slots, long Total, long ZLow, long ZHigh, int FloorClass, uint? Fixed);

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

        long total = 0;
        foreach (Slot s in slots) total += s.Weight;

        Slot[] table = [.. slots];
        long cumulative = 0;
        for (int i = 0; i < table.Length; i++)
        {
            table[i] = table[i] with { Threshold = LongDivide(cumulative, total) };
            cumulative += table[i].Weight;
        }
        return new Table(table, total, zLow, zHigh, floorClass, null);
    }

    /// <summary>floor(cumulative·2^41 / total), exactly, in int64. Every intermediate stays under
    /// 2^63 because total never exceeds 2^62 — which is what pins the depth at K = 38.</summary>
    public static long LongDivide(long cumulative, long total)
    {
        long remainder = cumulative, quotient = 0;
        for (int i = 0; i < SelectorBits; i++)
        {
            remainder <<= 1;
            quotient <<= 1;
            if (remainder >= total) { remainder -= total; quotient++; }
        }
        return quotient;
    }

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
                slots.Add(new Slot(0, cur, bits, false, 0, 1L << (bits + cls - floorClass)));
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

        if (start > end) { slots.Add(new Slot(0, from, 0, false, 0, 1)); return; }
        if (LatticeFloor(from, floorExponent) != start || !IsOnLattice(from, floorExponent))
            slots.Add(new Slot(0, from, 0, false, 0, 1));

        AddLatticeRun(slots, start, Math.Min(end, 0), shift);
        AddLatticeRun(slots, Math.Max(start, 0), end, shift);

        if (!IsOnLattice(to, floorExponent)) slots.Add(new Slot(0, end, 0, true, shift, 1));
    }

    private static bool IsOnLattice(long z, int floorExponent)
        => MagnitudeOverSpacing(Math.Abs(z), floorExponent).Exact;

    private static void AddLatticeRun(List<Slot> slots, long from, long to, int shift)
    {
        long cur = from;
        while (cur < to)
        {
            int bits = 63 - System.Numerics.BitOperations.LeadingZeroCount((ulong)(to - cur));
            slots.Add(new Slot(0, cur, bits, true, shift, 1L << bits));
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
        long selector = (long)(draw >> P);
        long index = (long)(draw & (ulong)SigMask);
        int lo = 0, hi = table.Slots.Length - 1;
        while (lo < hi)
        {
            int mid = (lo + hi + 1) >> 1;
            if (table.Slots[mid].Threshold <= selector) lo = mid; else hi = mid - 1;
        }
        Slot s = table.Slots[lo];
        long idx = index >> (P - s.IndexBits);
        return s.Lattice ? BitsOfLatticePoint(s.Base + idx, s.LatticeShift) : BitsOfOrdinal(s.Base + idx);
    }

    /// <summary>Element <paramref name="i"/> of a dense uniform draw over [low, high).</summary>
    public static float Draw(ulong key, ulong substreamIndex, long i, float low, float high,
        int rounds = Threefry2x32.Rounds)
        => BitConverter.UInt32BitsToSingle(
            SampleBits(Build(low, high), RngTestOracle.DrawValue(key, substreamIndex, i, rounds)));
}

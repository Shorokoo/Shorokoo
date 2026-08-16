using System;
using Shorokoo.Core.Rng;

namespace Shorokoo.Tests;

/// <summary>
/// Host oracle for the dense standard-normal draw — <b>test-only</b>, and the contract the in-graph
/// decode must reproduce bit for bit.
///
/// <para>One 64-bit generator value per element: the top bit is the sign, the low 63 a position on
/// the <b>magnitude axis</b>. Sampling the magnitude and applying the sign afterwards is what makes
/// the draw exactly symmetric — <c>+a</c> and <c>-a</c> own mirror-image cells — which a signed
/// ordinal axis cannot give, since there <c>-2^k</c> would own half the cell <c>+2^k</c> does.</para>
///
/// <para>The axis is cut into three kinds of region, lowest first. <b>Lattice</b>: below the
/// truncation floor the density is constant to 2^-78, so an even lattice of the floor class's
/// spacing is correct and the uniform draw's own decode carries it unchanged. <b>Pieces</b>: a run
/// of <c>2^IndexBits</c> consecutive magnitudes inside one weight class, decoded by a degree-12
/// series that inverts the Gaussian CDF over the piece. <b>Cap</b>: past the last resolved magnitude
/// everything collapses onto one float.</para>
///
/// <para>Cells run <b>away from zero</b> and boundaries are <b>midpoints</b>: magnitude <c>a</c>
/// owns <c>[midpoint(a-1,a), midpoint(a,a+1))</c>, so a draw lands on the float nearest the value
/// its code names rather than the one below it. That costs a quarter of a relative ulp on average
/// instead of a half, and removes a systematic downward bias. The consequence the decode has to
/// carry: the first magnitude of a weight class owns 0.75 of its own ulp, the lower neighbour's ulp
/// being half as wide, which is what <see cref="ClassStartOffset"/> corrects.</para>
///
/// <para><b>What this oracle does and does not check.</b> It pins the <i>decode</i>. It does not
/// re-derive <see cref="DenseNormalTable"/>, which needs a 320-bit <c>erf</c> and is generated
/// offline; the table's own correctness — that every entry code is the exact ideal breakpoint, that
/// no float that earns a code is starved — is established by the census in the ShorokooDev harness,
/// not here. So oracle and graph share the table, and a fault in the table would be invisible to
/// both.</para>
/// </summary>
internal static class RngDenseNormalOracle
{
    public const int P = 23;
    private const int Bias = 127;
    private const ulong SignMask = 1UL << 31;
    private const ulong SignificandMask = (1UL << P) - 1;

    /// <summary>High half of the 128-bit product — what the graph builds from 32-bit products.</summary>
    private static ulong MulHigh(ulong a, ulong b) => (ulong)(((UInt128)a * b) >> 64);

    /// <summary>Largest piece whose entry code is at or below <paramref name="code"/>.</summary>
    public static int SelectPiece(ulong code)
    {
        int lo = 0, hi = DenseNormalTable.PieceCount - 1;
        while (lo < hi)
        {
            int mid = (lo + hi + 1) >> 1;
            if (DenseNormalTable.Entry[mid] <= code) lo = mid; else hi = mid - 1;
        }
        return lo;
    }

    /// <summary>
    /// The piece's position as a Q0.64 fraction of its float run. Horner, not a chain of explicit
    /// powers: that is one multiply-high per degree rather than two, and one truncation rather than
    /// two. The accumulator cannot overflow — every coefficient is non-negative and they sum to
    /// under 2^64, and <c>MulHigh(a, x) &lt;= a</c>.
    /// </summary>
    public static ulong Poly(int p, ulong code)
    {
        ulong x = (ulong)((UInt128)(code - DenseNormalTable.Entry[p]) * DenseNormalTable.Recip[p]
            >> (int)DenseNormalTable.RecipShift[p]);
        ulong acc = DenseNormalTable.C[DenseNormalTable.Degree - 1][p];
        for (int k = DenseNormalTable.Degree - 2; k >= 0; k--) acc = MulHigh(acc, x) + DenseNormalTable.C[k][p];
        return MulHigh(acc, x);
    }

    /// <summary>A quarter ulp in the piece's Q0.64 offset, and zero for a piece that does not begin a
    /// weight class. Rounding to nearest is otherwise carried by the series <b>origin</b>, which sits
    /// at the piece's first cell boundary; this only corrects the 0.75-ulp first cell.</summary>
    private static ulong ClassStartOffset(int p) => 1UL << (int)(62 - DenseNormalTable.IndexBits[p]);

    private static bool StartsClass(int p) => (DenseNormalTable.FirstOrdinal[p] & ((1L << P) - 1)) == 0;

    /// <summary>Index of the magnitude within its piece.</summary>
    public static ulong DecodeIndex(int p, ulong code)
    {
        int bits = (int)DenseNormalTable.IndexBits[p];
        ulong offset = StartsClass(p) ? ClassStartOffset(p) : 0UL;
        ulong index = (Poly(p, code) + offset) >> (64 - bits);
        return Math.Min(index, (1UL << bits) - 1);
    }

    /// <summary>Bit pattern of <c>n·2^shift·delta</c> — the coarse-lattice decode. Exact by
    /// construction: n carries at most P+1 significant bits and scaling by a power of two only moves
    /// the exponent, so nothing rounds even into the subnormals.</summary>
    private static ulong BitsOfLatticePoint(ulong n, int shift)
    {
        if (n == 0) return 0UL;
        int hb = 63 - System.Numerics.BitOperations.LeadingZeroCount(n);
        int e = shift + hb;
        if (e < P) return n << shift;
        return ((ulong)(e - P + 1) << P) | ((n - (1UL << hb)) << (P - hb));
    }

    /// <summary>Magnitude bit pattern for a position on the magnitude axis.</summary>
    public static ulong MagnitudeBits(ulong code)
    {
        if (code < DenseNormalTable.LatticeCodes)
        {
            ulong point = (ulong)((UInt128)code * ((UInt128)1 << P) / DenseNormalTable.LatticeCodes);
            return BitsOfLatticePoint(point, DenseNormalTable.FloorClass - 1);
        }
        if (code >= DenseNormalTable.CapCode) return (ulong)DenseNormalTable.CapOrdinal;
        int p = SelectPiece(code);
        return (ulong)DenseNormalTable.FirstOrdinal[p] + DecodeIndex(p, code);
    }

    /// <summary>The whole draw: one 64-bit generator value to one float32 bit pattern.</summary>
    public static ulong SampleBits(ulong draw)
        => ((draw >> DenseNormalTable.MagnitudeBits) << 31)
         | MagnitudeBits(draw & ((1UL << DenseNormalTable.MagnitudeBits) - 1));

    public static float Draw(ulong key, ulong substreamIndex, long i, int rounds = Threefry2x32.Rounds)
        => BitConverter.UInt32BitsToSingle(
            (uint)SampleBits(RngTestOracle.DrawValue(key, substreamIndex, i, rounds)));
}

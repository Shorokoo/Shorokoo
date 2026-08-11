using System.Collections.Generic;
using Shorokoo.Core.Rng;

namespace Shorokoo.Tests;

/// <summary>
/// Host-side RNG key oracle — <b>test-only</b>. Production code performs no host-side Threefry
/// (#136): the key tree is computed in-graph by the algorithm's <c>SHRK_RNG_SPLIT</c> chain and
/// host consumers resolve keys by <em>executing</em> that derivation (<c>RngKeyResolver</c>).
/// These helpers rebuild the fold order and counter scheme by hand so assertions do not compare
/// the graph against itself. (The <see cref="Threefry2x32"/> bijection is shared — it is the
/// reference generator, pinned to the published Random123 vectors.) Never resolve a key or a
/// draw by calling into the graph. Keys, split indices and draw positions are whole
/// <c>ulong</c>s; the 32-bit word split is confined to the helpers that hand values to the
/// reference generator or repack its output.
/// </summary>
internal static class RngTestOracle
{
    /// <summary>One Threefry key-tree fold step: <c>child = Bijection(counter: index, key)</c>.
    /// The index occupies BOTH counter words, so the whole 64-bit range is distinct.</summary>
    public static ulong FoldKey(ulong key, ulong index)
    {
        var (x0, x1) = Threefry2x32.Bijection(
            (uint)index, (uint)(index >> 32), (uint)key, (uint)(key >> 32));
        return x0 | ((ulong)x1 << 32);
    }

    private static ulong Fold((ulong root, IReadOnlyList<int> foldPath) spec)
    {
        var key = spec.root;
        foreach (var v in spec.foldPath) key = FoldKey(key, unchecked((ulong)(long)v));
        return key;
    }

    /// <summary>A trainable parameter's init stream key (oracle for the in-graph derivation).</summary>
    public static ulong InitKey(RngConfig config, IReadOnlyList<int> modelIdVals)
        => Fold(config.InitKeySpec(modelIdVals));

    /// <summary>A runtime feed's stream key (oracle for the in-graph derivation).</summary>
    public static ulong RunKey(RngConfig config, IReadOnlyList<int> modelIdVals)
        => Fold(config.RunKeySpec(modelIdVals));

    /// <summary>A runtime feed's stream key under an encoded identity (oracle).</summary>
    public static ulong RunKey(RngRuntimeIdentity identity, IReadOnlyList<int> path)
        => Fold(identity.RunKeySpec(path));

    /// <summary>
    /// The whole 64 bits a draw produces at stream position <paramref name="p"/> — the host oracle
    /// for <c>RuntimeRng.Draw</c>. The substream index folds into the key, leaving both counter
    /// words to the whole 64-bit position.
    /// </summary>
    public static ulong DrawValue(
        ulong key, ulong substreamIndex, long p, int rounds = Threefry2x32.Rounds)
    {
        var (dk0, dk1) = Threefry2x32.Bijection(
            (uint)substreamIndex, (uint)(substreamIndex >> 32), (uint)key, (uint)(key >> 32), rounds);
        var (x0, x1) = Threefry2x32.Bijection((uint)p, (uint)((ulong)p >> 32), dk0, dk1, rounds);
        return x0 | ((ulong)x1 << 32);
    }

    /// <summary>
    /// Lane <paramref name="i"/> of a draw cut into <paramref name="width"/>-bit lanes: E = 64/W
    /// lanes pack into each generator value, low lane first, so lane <c>i</c> is bits
    /// <c>[i*W, (i+1)*W)</c> of the stream — the value at position <c>i / E</c>, shifted down by
    /// <c>(i % E) * W</c>. Degenerates to one whole value per element at W = 64.
    /// </summary>
    public static ulong DrawLane(
        ulong key, ulong substreamIndex, long i, int width, int rounds = Threefry2x32.Rounds)
    {
        if (width is not (8 or 16 or 32 or 64)) throw new System.ArgumentOutOfRangeException(nameof(width));
        long lanes = 64 / width;
        return (DrawValue(key, substreamIndex, i / lanes, rounds) >> (int)(i % lanes) * width)
               & (ulong.MaxValue >> (64 - width));
    }

    /// <summary>Element <paramref name="i"/> of a raw-bits draw of the given uint width.</summary>
    public static ulong DrawBits(
        ulong key, ulong substreamIndex, long i, int width, int rounds = Threefry2x32.Rounds)
        => DrawLane(key, substreamIndex, i, width, rounds);

    /// <summary>Element <paramref name="i"/> of a standard-uniform draw — Walker's
    /// geometric transform of the whole 64-bit value at position <paramref name="i"/> (one per
    /// element): a 23-bit mantissa fraction in [1,2) times a geometric octave scale (2^-1-lz, lz =
    /// leading zeros of the 41-bit exponent field). Exact — mirrors <c>RuntimeRng.GeometricUniform</c>.</summary>
    public static float DrawUniform(
        ulong key, ulong substreamIndex, long i, int rounds = Threefry2x32.Rounds)
        => WalkerUniform(DrawValue(key, substreamIndex, i, rounds));

    /// <summary>Walker's transform of a whole generator value, split out so a test can drive it
    /// with a chosen value rather than a drawn one — the dense draw agrees with it only above the
    /// truncation floor, and the disagreeing values carry probability 2^-38.</summary>
    public static float WalkerUniform(ulong v)
    {
        float frac = 1.0f + (uint)(v & 0x7FFFFF) * (1.0f / 8388608.0f);          // [1,2), 23-bit mantissa
        ulong ef = (v >> 23) & ((1UL << 41) - 1);                                // 41-bit exponent field
        int p = ef == 0 ? 0 : 63 - System.Numerics.BitOperations.LeadingZeroCount(ef);
        return frac * (float)System.Math.Pow(2.0, -1 - (40 - p));               // frac · 2^(-1-leadingZeros)
    }

    /// <summary>Element <paramref name="i"/> of a standard-normal draw. Pair <c>j = i/2</c> takes its
    /// radius from the geometric uniform at even position <c>2j</c> (√(−2·ln w), deep tail) and its
    /// angle from a 24-bit uniform at odd position <c>2j+1</c>; even <paramref name="i"/> is the cosine
    /// arm, odd the sine. Mirrors <c>RuntimeRng.StandardNormal</c>.</summary>
    public static float DrawNormal(
        ulong key, ulong substreamIndex, long i, int rounds = Threefry2x32.Rounds)
    {
        long j = i / 2;
        float w = DrawUniform(key, substreamIndex, 2 * j, rounds);                          // geometric radius draw
        float u2 = (DrawValue(key, substreamIndex, 2 * j + 1, rounds) & 0x00FFFFFFuL) * (1.0f / 16777216.0f);
        float radius = System.MathF.Sqrt(-2.0f * System.MathF.Log(w));
        float theta = u2 * (2.0f * System.MathF.PI);
        return radius * (i % 2 == 0 ? System.MathF.Cos(theta) : System.MathF.Sin(theta));
    }
}

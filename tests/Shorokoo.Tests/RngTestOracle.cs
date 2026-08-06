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
    /// The generator word pair a draw produces at stream position <paramref name="p"/> — the host
    /// oracle for <c>RuntimeRng.Draw</c>. The substream index folds into the key, leaving both
    /// counter words to the whole 64-bit position.
    /// </summary>
    public static (uint x0, uint x1) DrawWords(
        ulong key, ulong substreamIndex, long p, int rounds = Threefry2x32.Rounds)
    {
        var (dk0, dk1) = Threefry2x32.Bijection(
            (uint)substreamIndex, (uint)(substreamIndex >> 32), (uint)key, (uint)(key >> 32), rounds);
        return Threefry2x32.Bijection((uint)p, (uint)((ulong)p >> 32), dk0, dk1, rounds);
    }

    /// <summary>Element <paramref name="i"/> of a standard-uniform draw (low 24 bits × 2⁻²⁴).</summary>
    public static float DrawUniform(
        ulong key, ulong substreamIndex, long i, int rounds = Threefry2x32.Rounds)
        => (DrawWords(key, substreamIndex, i, rounds).x0 & 0x00FFFFFFu) * (1.0f / 16777216.0f);

    /// <summary>
    /// Element <paramref name="i"/> of a raw-bits draw of the given uint width. U64 spends a whole
    /// position per element; the narrower widths pack E = 32/W elements into each generator word,
    /// low lane first, so element <c>i</c> is lane <c>i % E</c> of the word at position
    /// <c>i / E</c>.
    /// </summary>
    public static ulong DrawBits(
        ulong key, ulong substreamIndex, long i, int width, int rounds = Threefry2x32.Rounds)
    {
        if (width == 64)
        {
            var (w0, w1) = DrawWords(key, substreamIndex, i, rounds);
            return w0 | ((ulong)w1 << 32);
        }
        if (width is not (8 or 16 or 32)) throw new System.ArgumentOutOfRangeException(nameof(width));

        long lanes = 32 / width;
        return (DrawWords(key, substreamIndex, i / lanes, rounds).x0 >> (int)(i % lanes) * width)
               & (ulong.MaxValue >> (64 - width));
    }
}

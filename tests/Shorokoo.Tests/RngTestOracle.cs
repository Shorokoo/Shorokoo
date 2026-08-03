using System.Collections.Generic;
using Shorokoo.Core.Rng;

namespace Shorokoo.Tests;

/// <summary>
/// Host-side RNG key oracle — <b>test-only</b>.
///
/// <para>Production code performs no host-side Threefry (#136): the key tree is computed
/// exclusively in-graph by the algorithm's <c>SHRK_RNG_SPLIT</c> chain, and any host consumer
/// that needs a concrete key resolves it by <em>executing</em> that derivation
/// (<c>RngKeyResolver</c>). These helpers reimplement the fold independently, on the host, so
/// tests can assert the in-graph derivation against an oracle that does not share its
/// implementation — which is exactly what makes the assertions meaningful. Never replace any
/// of it with a call into the product.</para>
///
/// <para>Keys and split indices are whole <c>ulong</c> values, matching the interface. The
/// 32-bit word split is Threefry's own business and appears only inside <see cref="FoldKey"/>,
/// where the reference generator requires it.</para>
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
    /// The generator word pair a draw produces for element <paramref name="i"/> — the host
    /// oracle for <c>RuntimeRng.Draw</c>. The draw position folds into the key (the same
    /// bijection a split uses, at the draw's own round count), leaving both counter words to
    /// the whole 64-bit element index.
    /// </summary>
    public static (uint x0, uint x1) DrawWords(
        ulong key, ulong drawBase, long i, int rounds = Threefry2x32.Rounds)
    {
        var (dk0, dk1) = Threefry2x32.Bijection(
            (uint)drawBase, (uint)(drawBase >> 32), (uint)key, (uint)(key >> 32), rounds);
        return Threefry2x32.Bijection((uint)i, (uint)((ulong)i >> 32), dk0, dk1, rounds);
    }

    /// <summary>Element <paramref name="i"/> of a standard-uniform draw (low 24 bits × 2⁻²⁴).</summary>
    public static float DrawUniform(
        ulong key, ulong drawBase, long i, int rounds = Threefry2x32.Rounds)
        => (DrawWords(key, drawBase, i, rounds).x0 & 0x00FFFFFFu) * (1.0f / 16777216.0f);

    /// <summary>Element <paramref name="i"/> of a raw-bits draw of the given uint width.</summary>
    public static ulong DrawBits(
        ulong key, ulong drawBase, long i, int width, int rounds = Threefry2x32.Rounds)
    {
        var (x0, x1) = DrawWords(key, drawBase, i, rounds);
        return width switch
        {
            8 => (byte)x0,
            16 => (ushort)x0,
            32 => x0,
            64 => x0 | ((ulong)x1 << 32),
            _ => throw new System.ArgumentOutOfRangeException(nameof(width)),
        };
    }
}

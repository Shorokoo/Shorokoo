using System;
using System.Collections.Generic;
using System.Linq;

namespace Shorokoo.Core.Rng;

/// <summary>
/// A model's decoded <b>runtime</b> RNG identity, and the encoder/decoder for its stored form.
///
/// <para>Two names, deliberately distinct: <b>RngSeedData</b> is the stored <c>uint64[N]</c> — the
/// value of the ordinary non-trainable <c>RngSeed</c> parameter at reserved ModelId <c>[0]</c> —
/// and <see cref="RngRuntimeIdentity"/> is what it decodes to. Prefer "RngSeedData" over
/// "identity vector" for the stored form: this codebase already has an ONNX <c>Identity</c> op and
/// a <c>Vector&lt;T&gt;</c> type, so that phrase collides with two unrelated meanings.</para>
///
/// <para>Layout of the RngSeedData:</para>
///
/// <code>
///   [0]              algorithm id (see <see cref="AlgorithmIdOf"/>)
///   [1]              runtime master key (one whole uint64)
///   [2]              runtime override record count C
///   per record:      [L, path element × L, key]
/// </code>
///
/// Records carry <see cref="RngCollection.Runtime"/> overrides only and are written in a
/// canonical sorted order, so the same config always encodes to the same data and every
/// record's key sits at a fixed, wiring-time-computable offset — an overridden feed's
/// in-graph derivation chain roots at a <c>Gather</c> of that offset instead of the master
/// elements (see <c>FastWireRngKeyDerivation</c>). The init-collection identity is
/// deliberately NOT encoded: initialization randomness is drawn host-side and baked into
/// weights, so nothing in a saved model consumes the init tier (re-running initialization
/// takes an explicit <see cref="RngConfig"/>).
/// </summary>
internal sealed class RngRuntimeIdentity
{
    /// <summary>Elements before the first override record: [algId, key, count].</summary>
    public const int HeaderLength = 3;

    /// <summary>Index of the algorithm id element.</summary>
    public const int AlgorithmIdIndex = 0;

    /// <summary>Index of the runtime master key (a single whole uint64 element).</summary>
    public const int RunKeyIndex = 1;

    /// <summary>One runtime override record: the overridden stream's realized ModelId path, the
    /// replacement key (the override replaces the fully folded key), and the key's offset
    /// in the vector (for structural chain routing).</summary>
    public sealed record RuntimeOverrideRecord(int[] Path, ulong Key, int KeyOffset);

    public long AlgorithmId { get; }
    public ulong RunKey { get; }
    public IReadOnlyList<RuntimeOverrideRecord> Overrides { get; }

    private RngRuntimeIdentity(long algorithmId, ulong runKey, IReadOnlyList<RuntimeOverrideRecord> overrides)
    {
        AlgorithmId = algorithmId;
        RunKey = runKey;
        Overrides = overrides;
    }

    /// <summary>The configured algorithm, or null when the id is unknown (a file written under an
    /// id this build does not define). Consumers must fail loudly on null, never substitute.</summary>
    public RngAlgorithm? Algorithm => TryAlgorithmFromId(AlgorithmId);

    /// <summary>The stable identity-vector id of a configured algorithm — the encoding of the
    /// <see cref="RngAlgorithm"/> member, and nothing more. An id must not be pointed at a
    /// <i>different</i> member, since a stored vector would then decode to the wrong algorithm
    /// entirely; but it carries no promise about what that algorithm draws. Id 0 has already
    /// outlived one normal transform.</summary>
    public static long AlgorithmIdOf(RngAlgorithm algorithm) => algorithm switch
    {
        RngAlgorithm.Threefry2x32 => 0,
        RngAlgorithm.Threefry2x32Rounds13 => 1,
        _ => throw new NotSupportedException($"Unknown RNG algorithm '{algorithm}'."),
    };

    /// <summary>The configured algorithm for an identity-vector id, or null when unknown.</summary>
    public static RngAlgorithm? TryAlgorithmFromId(long id) => id switch
    {
        0 => RngAlgorithm.Threefry2x32,
        1 => RngAlgorithm.Threefry2x32Rounds13,
        _ => null,
    };

    /// <summary>
    /// Encodes <paramref name="config"/>'s runtime identity as the <c>RngSeed</c> parameter
    /// value. <see cref="Decode"/> is the exact inverse; the decoded identity derives every
    /// runtime stream key bit-identically to the in-graph SHRK_RNG_SPLIT chain.
    /// </summary>
    public static ulong[] Build(RngConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        var overrides = config.RuntimeOverridesSorted();
        var vec = new List<ulong>
        {
            (ulong)AlgorithmIdOf(config.Algorithm),
            config.RunMasterKey,
            (ulong)overrides.Count,
        };
        foreach (var (path, seed) in overrides)
        {
            vec.Add((ulong)path.Length);
            foreach (var p in path) vec.Add(unchecked((ulong)(long)p));
            vec.Add(seed);
        }
        return [.. vec];
    }

    /// <summary>
    /// Decodes the <c>RngSeedData</c> produced by <see cref="Build"/>. Malformed data throws —
    /// a corrupt identity must never silently fall back to a different derivation.
    /// </summary>
    public static RngRuntimeIdentity Decode(ulong[] rngSeedData)
    {
        if (rngSeedData is not { Length: >= HeaderLength })
            throw new ArgumentException(
                $"Malformed RngSeedData: length {rngSeedData?.Length ?? 0} " +
                $"(expected at least the {HeaderLength}-element header).", nameof(rngSeedData));

        var runKey = rngSeedData[RunKeyIndex];
        ulong count = rngSeedData[HeaderLength - 1];
        var records = new List<RuntimeOverrideRecord>();
        int i = HeaderLength;
        for (ulong r = 0; r < count; r++)
        {
            if (i >= rngSeedData.Length)
                throw new ArgumentException("Malformed RngSeedData: truncated override record.", nameof(rngSeedData));
            // Bound the length against the data BEFORE narrowing to int: a hostile or corrupt
            // record can claim any uint64 path length, and `i + pathLen + 1` in int arithmetic
            // would wrap negative, pass the check, and then allocate the claimed length.
            ulong claimedPathLen = rngSeedData[i++];
            // The record needs pathLen elements plus one key, so pathLen < remaining. Written as
            // a comparison rather than `pathLen + 1 > remaining` because the claim is untrusted:
            // adding to it overflows for a length near ulong.MaxValue and the check passes.
            ulong remaining = (ulong)(rngSeedData.Length - i);
            if (claimedPathLen == 0 || claimedPathLen >= remaining)
                throw new ArgumentException("Malformed RngSeedData: truncated override record.", nameof(rngSeedData));
            int pathLen = (int)claimedPathLen;
            int[] path = new int[pathLen];
            for (int j = 0; j < pathLen; j++) path[j] = checked((int)unchecked((long)rngSeedData[i++]));
            int keyOffset = i;
            var key = rngSeedData[i];
            i += 1;
            records.Add(new RuntimeOverrideRecord(path, key, keyOffset));
        }
        if (i != rngSeedData.Length)
            throw new ArgumentException("Malformed RngSeedData: trailing data after override records.", nameof(rngSeedData));
        return new RngRuntimeIdentity(checked((long)rngSeedData[AlgorithmIdIndex]), runKey, records);
    }

    /// <summary>
    /// A runtime stream's key derivation under this identity, as a <b>spec</b>: the matching
    /// override record's key (nothing left to fold) when one exists, else the runtime
    /// master plus the path still to be folded. The fold itself happens in-graph
    /// (<c>SHRK_RNG_SPLIT</c>) — this type performs no RNG computation (#136).
    /// </summary>
    public (ulong root, IReadOnlyList<int> foldPath) RunKeySpec(IReadOnlyList<int> path)
    {
        foreach (var rec in Overrides)
            if (rec.Path.Length == path.Count && rec.Path.SequenceEqual(path))
                return (rec.Key, []);
        return (RunKey, path);
    }

    /// <summary>Whether this identity's override PATH set equals <paramref name="paths"/> —
    /// the test for "re-bind is a pure parameter write" vs "the wiring pass must re-run".</summary>
    public bool HasSameOverridePaths(IReadOnlyList<int[]> paths)
    {
        if (paths.Count != Overrides.Count) return false;
        var mine = Overrides.Select(o => string.Join(",", o.Path)).ToHashSet();
        return paths.All(p => mine.Contains(string.Join(",", p)));
    }
}

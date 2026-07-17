using System;
using System.Collections.Generic;
using System.Linq;

namespace Shorokoo.Core.Rng;

/// <summary>
/// The encoded form of a model's <b>runtime</b> RNG identity — the value of the ordinary
/// non-trainable <c>RngSeed</c> parameter at reserved ModelId <c>[0]</c> (int64, shape [N]):
///
/// <code>
///   [0]              algorithm id (see <see cref="AlgorithmIdOf"/>)
///   [1], [2]         runtime master key words (k0, k1), each in [0, 2^32)
///   [3]              runtime override record count C
///   per record:      [L, path element × L, key word k0, key word k1]
/// </code>
///
/// Records carry <see cref="RngCollection.Runtime"/> overrides only and are written in a
/// canonical sorted order, so the same config always encodes to the same vector and every
/// record's key words sit at a fixed, wiring-time-computable offset — an overridden feed's
/// in-graph derivation chain roots at a <c>Gather</c> of that offset instead of the master
/// elements (see <c>FastWireRngKeyDerivation</c>). The init-collection identity is
/// deliberately NOT encoded: initialization randomness is drawn host-side and baked into
/// weights, so nothing in a saved model consumes the init tier (re-running initialization
/// takes an explicit <see cref="RngConfig"/>).
/// </summary>
internal sealed class RngRuntimeIdentity
{
    /// <summary>Elements before the first override record: [algId, k0, k1, count].</summary>
    public const int HeaderLength = 4;

    /// <summary>Index of the algorithm id element.</summary>
    public const int AlgorithmIdIndex = 0;

    /// <summary>Index of the runtime master key's first word; the second word follows it.</summary>
    public const int RunKeyIndex = 1;

    /// <summary>One runtime override record: the overridden stream's realized ModelId path, the
    /// replacement key words (the override replaces the fully folded key), and the vector offset
    /// of the record's first key word (for structural chain routing).</summary>
    public sealed record RuntimeOverrideRecord(int[] Path, (uint k0, uint k1) Key, int KeyOffset);

    public long AlgorithmId { get; }
    public (uint k0, uint k1) RunKey { get; }
    public IReadOnlyList<RuntimeOverrideRecord> Overrides { get; }

    private RngRuntimeIdentity(long algorithmId, (uint k0, uint k1) runKey, IReadOnlyList<RuntimeOverrideRecord> overrides)
    {
        AlgorithmId = algorithmId;
        RunKey = runKey;
        Overrides = overrides;
    }

    /// <summary>The configured algorithm, or null when the id is unknown (a file written by a
    /// newer framework version). Consumers must fail loudly on null, never substitute.</summary>
    public RngAlgorithm? Algorithm => TryAlgorithmFromId(AlgorithmId);

    /// <summary>The stable identity-vector id of a configured algorithm.</summary>
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
    /// runtime stream key bit-identically to <see cref="RngConfig.FoldRunKey"/>.
    /// </summary>
    public static long[] Build(RngConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        var overrides = config.RuntimeOverridesSorted();
        var vec = new List<long>
        {
            AlgorithmIdOf(config.Algorithm),
            config.RunMasterKey.k0,
            config.RunMasterKey.k1,
            overrides.Count,
        };
        foreach (var (path, seed) in overrides)
        {
            vec.Add(path.Length);
            foreach (var p in path) vec.Add(p);
            var (k0, k1) = RngConfig.SplitWords(seed);
            vec.Add(k0);
            vec.Add(k1);
        }
        return [.. vec];
    }

    /// <summary>
    /// Decodes an identity vector produced by <see cref="Build"/>. Malformed vectors throw —
    /// a corrupt identity must never silently fall back to a different derivation.
    /// </summary>
    public static RngRuntimeIdentity Decode(long[] identity)
    {
        if (identity is not { Length: >= HeaderLength })
            throw new ArgumentException(
                $"Malformed RngSeed identity: length {identity?.Length ?? 0} " +
                $"(expected at least the {HeaderLength}-element header).", nameof(identity));

        var runKey = ((uint)identity[RunKeyIndex], (uint)identity[RunKeyIndex + 1]);
        long count = identity[HeaderLength - 1];
        var records = new List<RuntimeOverrideRecord>();
        int i = HeaderLength;
        for (long r = 0; r < count; r++)
        {
            if (i >= identity.Length)
                throw new ArgumentException("Malformed RngSeed identity: truncated override record.", nameof(identity));
            int pathLen = checked((int)identity[i++]);
            if (pathLen <= 0 || i + pathLen + 2 > identity.Length)
                throw new ArgumentException("Malformed RngSeed identity: truncated override record.", nameof(identity));
            int[] path = new int[pathLen];
            for (int j = 0; j < pathLen; j++) path[j] = checked((int)identity[i++]);
            int keyOffset = i;
            var key = ((uint)identity[i], (uint)identity[i + 1]);
            i += 2;
            records.Add(new RuntimeOverrideRecord(path, key, keyOffset));
        }
        if (i != identity.Length)
            throw new ArgumentException("Malformed RngSeed identity: trailing data after override records.", nameof(identity));
        return new RngRuntimeIdentity(identity[AlgorithmIdIndex], runKey, records);
    }

    /// <summary>
    /// A runtime stream's key under this identity: the matching override record's key words when
    /// one exists, else the runtime master folded along the path — bit-identical to
    /// <see cref="RngConfig.FoldRunKey"/> of the encoding config, and to the in-graph
    /// SHRK_RNG_SPLIT chain the wiring emits.
    /// </summary>
    public (uint k0, uint k1) FoldRunKey(IReadOnlyList<int> path)
    {
        foreach (var rec in Overrides)
            if (rec.Path.Length == path.Count && rec.Path.SequenceEqual(path))
                return rec.Key;
        var key = RunKey;
        foreach (var v in path) key = RngConfig.FoldKey(key, v);
        return key;
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

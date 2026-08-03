using System;
using System.Collections.Generic;
using System.Linq;

namespace Shorokoo.Core.Rng;

/// <summary>
/// The encoded form of a model's <b>runtime</b> RNG identity — the value of the ordinary
/// non-trainable <c>RngSeed</c> parameter at reserved ModelId <c>[0]</c> (uint64, shape [N]):
///
/// <code>
///   [0]              scheme version (see <see cref="SchemeVersion"/>)
///   [1]              algorithm id (see <see cref="AlgorithmIdOf"/>)
///   [2]              runtime master key (one whole uint64)
///   [3]              runtime override record count C
///   per record:      [L, path element × L, key]
/// </code>
///
/// Records carry <see cref="RngCollection.Runtime"/> overrides only and are written in a
/// canonical sorted order, so the same config always encodes to the same vector and every
/// record's key sits at a fixed, wiring-time-computable offset — an overridden feed's
/// in-graph derivation chain roots at a <c>Gather</c> of that offset instead of the master
/// elements (see <c>FastWireRngKeyDerivation</c>). The init-collection identity is
/// deliberately NOT encoded: initialization randomness is drawn host-side and baked into
/// weights, so nothing in a saved model consumes the init tier (re-running initialization
/// takes an explicit <see cref="RngConfig"/>).
/// </summary>
internal sealed class RngRuntimeIdentity
{
    /// <summary>Elements before the first override record: [schemeVersion, algId, key, count].</summary>
    public const int HeaderLength = 4;

    /// <summary>Index of the scheme version element.</summary>
    public const int SchemeVersionIndex = 0;

    /// <summary>Index of the algorithm id element.</summary>
    public const int AlgorithmIdIndex = 1;

    /// <summary>Index of the runtime master key (a single whole uint64 element).</summary>
    public const int RunKeyIndex = 2;

    /// <summary>
    /// The version of the RNG scheme this identity was written under. <b>Bump this whenever the
    /// values a draw produces change</b>, not merely when the vector's layout changes.
    ///
    /// <para>The algorithm id alone cannot carry that: it maps <see cref="RngAlgorithm"/> to a
    /// small integer that stays stable across versions, so an identity written under an older
    /// scheme decodes as "algorithm 0" and silently means whatever "algorithm 0" means today. The
    /// registry's contract — a change in produced values is a new algorithm <em>name</em>, never a
    /// silent change — is only enforceable if something persisted moves with it. This is it.</para>
    ///
    /// <para>2: <c>Threefry2x32-BoxMuller.v2</c> — whole uint64 keys/indices/draw positions, and
    /// the draw position folded into the key. 1 was never written with a version element at all,
    /// so it is recognised by its Int64 element type instead (see
    /// <see cref="ReadIdentityVector"/>).</para>
    /// </summary>
    public const ulong SchemeVersion = 2;

    /// <summary>
    /// Reads an identity vector out of an <c>RngSeed</c> parameter's tensor data, rejecting a
    /// layout this version cannot read. Every site that decodes an identity goes through here, so
    /// a carrier written before the identity became uint64 gets one explanatory error rather than
    /// a bare <c>InvalidCastException</c> from whichever door it happened to arrive at.
    /// </summary>
    /// <exception cref="ArgumentNullException"><paramref name="data"/> is null.</exception>
    /// <exception cref="InvalidOperationException">The tensor's element type is not uint64. The
    /// vector's <em>contents</em> — length, scheme version, record structure — are validated by
    /// <see cref="Decode"/>, not here.</exception>
    public static ulong[] ReadIdentityVector(TensorData data)
    {
        if (data is null) throw new ArgumentNullException(nameof(data));
        if (data.DType != DType.UInt64)
            throw new InvalidOperationException(
                $"This model's RngSeed identity is {data.DType}, not UInt64. A UInt64 identity is " +
                "what every version from the 'Threefry2x32-BoxMuller.v2' algorithm on writes; an " +
                "Int64 one was written before RNG keys became whole uint64 values, and its draw " +
                "values are superseded (see RngAlgorithms: '...v1' -> '...v2'). Either way this " +
                "carrier cannot be read or re-keyed here — rebuild the concrete model from its " +
                "architecture under an explicit RngConfig.");
        return data.As<uint64>().AccessMemory().ToArray();
    }

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
    /// runtime stream key bit-identically to the in-graph SHRK_RNG_SPLIT chain.
    /// </summary>
    public static ulong[] Build(RngConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        var overrides = config.RuntimeOverridesSorted();
        var vec = new List<ulong>
        {
            SchemeVersion,
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
    /// Decodes an identity vector produced by <see cref="Build"/>. Malformed vectors throw —
    /// a corrupt identity must never silently fall back to a different derivation.
    /// </summary>
    public static RngRuntimeIdentity Decode(ulong[] identity)
    {
        if (identity is not { Length: >= HeaderLength })
            throw new ArgumentException(
                $"Malformed RngSeed identity: length {identity?.Length ?? 0} " +
                $"(expected at least the {HeaderLength}-element header).", nameof(identity));

        var schemeVersion = identity[SchemeVersionIndex];
        if (schemeVersion != SchemeVersion)
            throw new InvalidOperationException(
                $"This model's RngSeed identity was written under RNG scheme version " +
                $"{schemeVersion}; this build writes and reads version {SchemeVersion}. The scheme " +
                "version moves whenever the values a draw produces change, so running under it " +
                "would silently draw different numbers. Rebuild the concrete model from its " +
                "architecture under an explicit RngConfig.");

        var runKey = identity[RunKeyIndex];
        ulong count = identity[HeaderLength - 1];
        var records = new List<RuntimeOverrideRecord>();
        int i = HeaderLength;
        for (ulong r = 0; r < count; r++)
        {
            if (i >= identity.Length)
                throw new ArgumentException("Malformed RngSeed identity: truncated override record.", nameof(identity));
            // Bound the length against the vector BEFORE narrowing to int: a hostile or corrupt
            // record can claim any uint64 path length, and `i + pathLen + 1` in int arithmetic
            // would wrap negative, pass the check, and then allocate the claimed length.
            ulong claimedPathLen = identity[i++];
            // The record needs pathLen elements plus one key, so pathLen < remaining. Written as
            // a comparison rather than `pathLen + 1 > remaining` because the claim is untrusted:
            // adding to it overflows for a length near ulong.MaxValue and the check passes.
            ulong remaining = (ulong)(identity.Length - i);
            if (claimedPathLen == 0 || claimedPathLen >= remaining)
                throw new ArgumentException("Malformed RngSeed identity: truncated override record.", nameof(identity));
            int pathLen = (int)claimedPathLen;
            int[] path = new int[pathLen];
            for (int j = 0; j < pathLen; j++) path[j] = checked((int)unchecked((long)identity[i++]));
            int keyOffset = i;
            var key = identity[i];
            i += 1;
            records.Add(new RuntimeOverrideRecord(path, key, keyOffset));
        }
        if (i != identity.Length)
            throw new ArgumentException("Malformed RngSeed identity: trailing data after override records.", nameof(identity));
        return new RngRuntimeIdentity(checked((long)identity[AlgorithmIdIndex]), runKey, records);
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

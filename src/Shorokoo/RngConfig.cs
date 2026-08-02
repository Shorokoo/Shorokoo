using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Security.Cryptography;
using System.Text;

namespace Shorokoo;

/// <summary>The bit generator a <see cref="RngConfig"/> uses for keyed draws. Configuration, never
/// part of a model definition. Every algorithm shares one key tree (so switching preserves stream
/// identity — the same stream just draws different numbers); only the draw's bit generator differs.</summary>
public enum RngAlgorithm
{
    /// <summary>Threefry-2x32 (Random123), 20 rounds. The default: expressible as portable ONNX integer ops.</summary>
    Threefry2x32,

    /// <summary>Threefry-2x32 with the reduced 13-round bit generator (Random123 <c>threefry2x32x13</c>):
    /// still BigCrush-resistant, ~35% cheaper than the 20-round default — the faster, lower-margin choice.</summary>
    Threefry2x32Rounds13,
}

/// <summary>
/// A named RNG stream collection. Random sites belong to exactly one collection so that
/// initialization randomness (drawn once when parameters are materialized) and runtime
/// randomness (Dropout, sampling, noise — drawn every step) are separate, independently
/// seedable streams — the Flax "RNG collections" model.
/// </summary>
public enum RngCollection
{
    /// <summary>Trainable-parameter initialization randomness.</summary>
    Params,
    /// <summary>Per-step runtime randomness (Dropout, sampling, in-model noise).</summary>
    Runtime,
}

/// <summary>
/// The single configuration object for randomness. It carries the bit-generator
/// <see cref="Algorithm"/>, the <see cref="MasterSeed"/>, and any per-stream seed
/// overrides — and is bound at materialization / compile time (like hyperparameters),
/// never baked into the model definition.
///
/// <para>Key derivation: each stream's key is the collection's sub-master (init or
/// runtime, both derived from <see cref="MasterSeed"/>) folded along the consumer's
/// ModelId path — one Threefry bijection per path index. So changing
/// <see cref="MasterSeed"/> re-randomizes everything coherently; an
/// <see cref="Override(RngCollection, int[], ulong)"/> replaces exactly one stream's
/// key; and because keys derive from graph <em>position</em> rather than draw order,
/// inserting or reordering unrelated sites does not disturb other streams (and
/// <c>Rng.Pin</c> can freeze positions against refactoring).</para>
///
/// <para>Fully deterministic by default (<c>MasterSeed = 0</c>). Use
/// <see cref="NonDeterministic"/> for a fresh random stream each run.</para>
///
/// <para><b>Immutable.</b> A config is a value: every property is init-only and
/// <see cref="Override(RngCollection, int[], ulong)"/> returns a modified <em>copy</em>,
/// never touching the receiver — so configs can be shared, cached (including
/// <see cref="Default"/>), and handed across models without any risk of one caller's
/// tweak re-keying another's streams.</para>
/// </summary>
public sealed class RngConfig
{
    /// <summary>The master seed folded into every non-overridden stream key. Default 0.</summary>
    public ulong MasterSeed { get; init; }

    /// <summary>
    /// Explicit init-collection sub-master. When set, every trainable-parameter stream folds
    /// from this key instead of <c>Fold(MasterSeed, "init")</c> — re-rolling all weights while
    /// runtime streams stay put. Null (default) derives from <see cref="MasterSeed"/>.
    /// </summary>
    public ulong? InitMasterSeed { get; init; }

    /// <summary>
    /// Explicit runtime-collection sub-master. When set, every runtime feed stream folds from
    /// this key instead of <c>Fold(MasterSeed, "runtime")</c> — re-seeding all feeds while
    /// parameter init stays put. Null (default) derives from <see cref="MasterSeed"/>.
    /// </summary>
    public ulong? RunMasterSeed { get; init; }

    /// <summary>The bit generator. Default <see cref="RngAlgorithm.Threefry2x32"/>.</summary>
    public RngAlgorithm Algorithm { get; init; } = RngAlgorithm.Threefry2x32;

    /// <summary>
    /// When <c>true</c>, every <b>parameter-initialization</b> stream shares one key derived
    /// from <see cref="MasterSeed"/> alone (name-independent), so two parameters of the same
    /// shape and distribution
    /// receive identical values — the "tied" init that reproduces a layer's weights from a
    /// hand-built reference. Off by default (per-parameter, name-derived keys). Useful for
    /// closed-form reference tests and for debugging; not for real training, where distinct
    /// parameters should differ. Note this applies to the <see cref="RngCollection.Params"/>
    /// tier only — runtime feeds always fold along their ModelId path regardless.
    /// </summary>
    public bool SharedKey { get; init; }

    // (collection, ModelId path) -> seed. Immutable, like the config itself: Override
    // returns a copy carrying an extended dictionary.
    private readonly ImmutableDictionary<(RngCollection collection, string pathKey), ulong> _overrides
        = ImmutableDictionary<(RngCollection collection, string pathKey), ulong>.Empty;

    /// <summary>A fresh config with default properties (see the init-only property defaults).</summary>
    public RngConfig() { }

    /// <summary>Copy with a different override set — the <see cref="Override"/> mechanic.</summary>
    private RngConfig(RngConfig source,
        ImmutableDictionary<(RngCollection collection, string pathKey), ulong> overrides)
    {
        MasterSeed = source.MasterSeed;
        InitMasterSeed = source.InitMasterSeed;
        RunMasterSeed = source.RunMasterSeed;
        Algorithm = source.Algorithm;
        SharedKey = source.SharedKey;
        _overrides = overrides;
    }

    private static string PathKey(IReadOnlyList<int> modelIdPath) => string.Join(",", modelIdPath);

    /// <summary>The default deterministic configuration (master seed 0, Threefry-2x32).
    /// Safe to share: configs are immutable, so this instance can never be changed.</summary>
    public static RngConfig Default { get; } = new();

    /// <summary>
    /// A configuration seeded from system entropy, so each run draws a different stream.
    /// (The chosen seed is fixed for the lifetime of the returned object, so a single
    /// run remains internally consistent and its <see cref="MasterSeed"/> can be recorded.)
    /// </summary>
    public static RngConfig NonDeterministic()
    {
        Span<byte> b = stackalloc byte[8];
        RandomNumberGenerator.Fill(b);
        return new RngConfig { MasterSeed = BinaryPrimitives.ReadUInt64LittleEndian(b) };
    }

    /// <summary>
    /// Returns a copy of this config with a single stream pinned to <paramref name="seed"/>,
    /// overriding the master-seed derivation for that stream only — the receiver is never
    /// modified (configs are immutable values). The stream is addressed by its consumer's
    /// absolute ModelId path (as shown by the stream report / parameter infos), e.g.
    /// <c>Override(RngCollection.Params, [1, 1], 1234)</c> re-seeds the first
    /// sub-module's first parameter and nothing else. The override replaces the fully
    /// folded key, so it survives a <see cref="MasterSeed"/> change. Matching is exact
    /// (leaf streams); chain calls to stack overrides. An override that matches no
    /// stream fails loudly where its collection is consumed: <see cref="RngCollection.Runtime"/>
    /// at bind (<c>ApplyRngConfig</c>), <see cref="RngCollection.Params"/> at parameter
    /// initialization.
    /// </summary>
    public RngConfig Override(RngCollection collection, int[] modelIdPath, ulong seed)
    {
        ArgumentNullException.ThrowIfNull(modelIdPath);
        if (modelIdPath.Length == 0)
            throw new ArgumentException("ModelId path must be non-empty.", nameof(modelIdPath));
        return new RngConfig(this,
            _overrides.SetItem((collection, PathKey(modelIdPath)), seed));
    }

    /// <summary>All registered override addresses (collection + comma-joined path), for
    /// fail-loud validation: Runtime overrides are checked against the realized stream set at
    /// bind, Params overrides against the parameter inventory at initialization — an override
    /// that matches no stream throws instead of silently doing nothing.</summary>
    internal IEnumerable<(RngCollection collection, string pathKey)> OverrideKeys
        => _overrides.Keys;

    /// <summary>Every registered override as (collection, ModelId path, seed) — for serializing a
    /// config verbatim (issue #115), so a reconstructed config re-applies the same overrides.</summary>
    internal IEnumerable<(RngCollection collection, int[] path, ulong seed)> AllOverrides()
        => _overrides.Select(e =>
            (e.Key.collection, e.Key.pathKey.Split(',').Select(int.Parse).ToArray(), e.Value));

    /// <summary>Whether a stream has an explicit override.</summary>
    public bool HasOverride(RngCollection collection, int[] modelIdPath)
        => _overrides.ContainsKey((collection, PathKey(modelIdPath ?? throw new ArgumentNullException(nameof(modelIdPath)))));

    private bool TryGetOverride(RngCollection collection, IReadOnlyList<int> modelIdPath, out (uint k0, uint k1) key)
    {
        if (_overrides.Count > 0 &&
            _overrides.TryGetValue((collection, PathKey(modelIdPath)), out var seed))
        {
            key = SplitWords(seed);
            return true;
        }
        key = default;
        return false;
    }

    /// <summary>
    /// The init-collection master key: two 32-bit words of <c>Fold(MasterSeed, "init")</c>.
    /// Every trainable-parameter stream folds from this along the parameter's ModelId path,
    /// so overriding the init sub-master re-rolls all weights while runtime streams stay put.
    /// </summary>
    internal (uint k0, uint k1) InitMasterKey => SplitWords(InitMasterSeed ?? Fold(MasterSeed, "init"));

    /// <summary>The runtime-collection master key (Dropout masks, sampling, noise): words of <c>Fold(MasterSeed, "runtime")</c>.</summary>
    internal (uint k0, uint k1) RunMasterKey => SplitWords(RunMasterSeed ?? Fold(MasterSeed, "runtime"));

    /// <summary>
    /// The <see cref="RngCollection.Runtime"/> overrides in the canonical (sorted-by-path-text)
    /// record order used by the encoded <c>RngSeed</c> identity (see
    /// <see cref="Core.Rng.RngRuntimeIdentity"/>) and by the structural chain wiring — one
    /// deterministic order shared by every consumer, so record offsets computed at wiring time
    /// match the encoded vector exactly.
    /// </summary>
    internal IReadOnlyList<(int[] path, ulong seed)> RuntimeOverridesSorted()
        => _overrides
            .Where(e => e.Key.collection == RngCollection.Runtime)
            .OrderBy(e => e.Key.pathKey, StringComparer.Ordinal)
            .Select(e => (e.Key.pathKey.Split(',').Select(int.Parse).ToArray(), e.Value))
            .ToArray();

    internal static (uint k0, uint k1) SplitWords(ulong key)
        => ((uint)(key & 0xFFFFFFFF), (uint)(key >> 32));

    // NOTE (#136): there is deliberately no host-side key fold here. The key tree is computed
    // exclusively in-graph by the algorithm's SHRK_RNG_SPLIT chain; a host consumer that needs
    // a concrete key resolves it by EXECUTING that derivation (see RngKeyResolver), never by
    // running Threefry on the host. That removes the obstacle to a custom algorithm owning
    // its own split (#122) — the key tree is still algorithm-independent today.

    /// <summary>
    /// A trainable parameter's init stream key expressed as a <b>derivation spec</b> rather
    /// than a computed key: the root key words plus the ModelId path still to be folded into
    /// them. The fold itself is performed <b>in-graph</b> by a <c>SHRK_RNG_SPLIT</c> chain
    /// (see <c>FastInitKeyedDraws</c>) — no host-side Threefry — exactly as a runtime feed's
    /// chain derives its key from the <c>RngSeed</c> parameter.
    ///
    /// <para>The two short-circuits the old host fold applied are carried here as an empty
    /// fold path: an explicit per-stream override <b>replaces</b> the fully folded key, and
    /// <see cref="SharedKey"/> mode skips the fold so same-shape params tie (test/debug only).
    /// The root words are pure marshalling (<see cref="Fold"/> is a SHA-256 XOR, not RNG).</para>
    /// </summary>
    internal ((uint k0, uint k1) root, IReadOnlyList<int> foldPath) InitKeySpec(IEnumerable<int> modelIdVals)
    {
        var vals = modelIdVals as IReadOnlyList<int> ?? new List<int>(modelIdVals);
        if (TryGetOverride(RngCollection.Params, vals, out var overridden))
            return (overridden, []);
        return (InitMasterKey, SharedKey ? [] : vals);
    }

    /// <summary>
    /// A runtime feed's stream key as a <b>derivation spec</b> — see
    /// <see cref="InitKeySpec"/>. The fold is performed in-graph (the feed's
    /// <c>SHRK_RNG_SPLIT</c> chain); this spec lets a host-side consumer (the RNG stream
    /// report) resolve the same key by <em>executing</em> that derivation, never by
    /// recomputing it host-side.
    /// </summary>
    internal ((uint k0, uint k1) root, IReadOnlyList<int> foldPath) RunKeySpec(IEnumerable<int> modelIdVals)
    {
        var vals = modelIdVals as IReadOnlyList<int> ?? new List<int>(modelIdVals);
        if (TryGetOverride(RngCollection.Runtime, vals, out var overridden))
            return (overridden, []);
        return (RunMasterKey, vals);
    }

    /// <summary>
    /// Folds a stream name into the master seed: <c>masterSeed XOR (first 8 bytes of
    /// SHA-256(name), read little-endian)</c>. Deterministic and platform-independent (SHA-256
    /// gives identical bytes everywhere; the explicit little-endian read makes the fold
    /// endian-independent too).
    /// </summary>
    internal static ulong Fold(ulong masterSeed, string fullStreamName)
    {
        Span<byte> hash = stackalloc byte[32];
        SHA256.HashData(Encoding.UTF8.GetBytes(fullStreamName), hash);
        return masterSeed ^ BinaryPrimitives.ReadUInt64LittleEndian(hash);
    }
}

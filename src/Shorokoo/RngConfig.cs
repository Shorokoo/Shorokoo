using System;
using System.Buffers.Binary;
using System.Collections.Generic;
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
    /// When <c>true</c>, every stream shares one key derived from <see cref="MasterSeed"/>
    /// alone (name-independent), so two parameters of the same shape and distribution
    /// receive identical values — the "tied" init that reproduces a layer's weights from a
    /// hand-built reference. Off by default (per-parameter, name-derived keys). Useful for
    /// closed-form reference tests and for debugging; not for real training, where distinct
    /// parameters should differ.
    /// </summary>
    public bool SharedKey { get; init; }

    // (collection, ModelId path) -> seed. Insertion into a live config; frozen in
    // practice once bound (re-stamp after changing it to make it take effect).
    private readonly Dictionary<(RngCollection collection, string pathKey), ulong> _overrides = new();

    private static string PathKey(IReadOnlyList<int> modelIdPath) => string.Join(",", modelIdPath);

    /// <summary>The default deterministic configuration (master seed 0, Threefry-2x32).</summary>
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
    /// Pins a single stream to <paramref name="seed"/>, overriding the master-seed
    /// derivation for that stream only. The stream is addressed by its consumer's
    /// absolute ModelId path (as shown by the stream report / parameter infos), e.g.
    /// <c>Override(RngCollection.Params, [1, 1], 1234)</c> re-seeds the first
    /// sub-module's first parameter and nothing else. The override replaces the fully
    /// folded key, so it survives a <see cref="MasterSeed"/> change. Matching is exact
    /// (leaf streams); returns <c>this</c> for chaining.
    /// </summary>
    public RngConfig Override(RngCollection collection, int[] modelIdPath, ulong seed)
    {
        ArgumentNullException.ThrowIfNull(modelIdPath);
        if (modelIdPath.Length == 0)
            throw new ArgumentException("ModelId path must be non-empty.", nameof(modelIdPath));
        _overrides[(collection, PathKey(modelIdPath))] = seed;
        return this;
    }

    /// <summary>All registered override addresses (collection + comma-joined path), for
    /// bind-time validation: an override that matches no stream fails the bind loudly.</summary>
    internal IEnumerable<(RngCollection collection, string pathKey)> OverrideKeys
        => _overrides.Keys;

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
    /// Serializes this config's randomness state as the compact RNG key vector a model carries
    /// (as a single parameter-like tensor), in the smallest of three tiers:
    /// <list type="bullet">
    ///   <item><b>[1]</b> — the master seed alone, when every key derives from it;</item>
    ///   <item><b>[3]</b> — master + init master + run master, when a sub-master was set
    ///     explicitly but no per-stream override exists;</item>
    ///   <item><b>[3 + N]</b> — the three masters followed by every realized stream's resolved
    ///     key (init streams then runtime streams, in the caller's order — the stream report's
    ///     enumeration), when any per-stream override exists.</item>
    /// </list>
    /// ONNX-prep reconstruction is the inverse: tier 1/2 re-fold the masters along each
    /// realized path; tier 3 reads the expanded keys directly. Keys are (k0, k1) 32-bit word
    /// pairs packed as one long each.
    /// </summary>
    internal long[] BuildKeyVector(
        IEnumerable<IReadOnlyList<int>> initStreamPaths,
        IEnumerable<IReadOnlyList<int>> runStreamPaths)
    {
        static long Pack((uint k0, uint k1) key) => unchecked((long)(((ulong)key.k1 << 32) | key.k0));

        if (_overrides.Count == 0)
        {
            if (InitMasterSeed is null && RunMasterSeed is null)
                return [unchecked((long)MasterSeed)];
            return [unchecked((long)MasterSeed), Pack(InitMasterKey), Pack(RunMasterKey)];
        }

        var vec = new List<long>
        {
            unchecked((long)MasterSeed), Pack(InitMasterKey), Pack(RunMasterKey)
        };
        foreach (var p in initStreamPaths) vec.Add(Pack(FoldInitKey(p)));
        foreach (var p in runStreamPaths) vec.Add(Pack(FoldRunKey(p)));
        return [.. vec];
    }

    /// <summary>
    /// Reconstructs every realized stream's key from a compact key vector produced by
    /// <see cref="BuildKeyVector"/>, given the same stream enumeration (init paths then run
    /// paths, same order). Tier 1/2 re-fold from the stored masters; tier 3 reads the stored
    /// expansion. Returns (initKeys, runKeys) as (k0, k1) word pairs.
    /// </summary>
    internal static ((uint k0, uint k1)[] initKeys, (uint k0, uint k1)[] runKeys) ReconstructKeys(
        long[] keyVector,
        IReadOnlyList<IReadOnlyList<int>> initStreamPaths,
        IReadOnlyList<IReadOnlyList<int>> runStreamPaths)
    {
        static (uint k0, uint k1) Unpack(long packed)
            => (unchecked((uint)((ulong)packed & 0xFFFFFFFF)), unchecked((uint)((ulong)packed >> 32)));

        if (keyVector.Length >= 3 + initStreamPaths.Count + runStreamPaths.Count
            && keyVector.Length > 3)
        {
            // Tier 3: stored expansion.
            var init = new (uint, uint)[initStreamPaths.Count];
            var run = new (uint, uint)[runStreamPaths.Count];
            for (int i = 0; i < init.Length; i++) init[i] = Unpack(keyVector[3 + i]);
            for (int i = 0; i < run.Length; i++) run[i] = Unpack(keyVector[3 + init.Length + i]);
            return (init, run);
        }

        (uint, uint) initMaster, runMaster;
        if (keyVector.Length >= 3)
        {
            initMaster = Unpack(keyVector[1]);
            runMaster = Unpack(keyVector[2]);
        }
        else
        {
            var master = unchecked((ulong)keyVector[0]);
            initMaster = SplitWords(Fold(master, "init"));
            runMaster = SplitWords(Fold(master, "runtime"));
        }

        static (uint, uint) FoldPath((uint k0, uint k1) key, IReadOnlyList<int> path)
        {
            foreach (var v in path) key = FoldKey(key, v);
            return key;
        }
        var initKeys = new (uint, uint)[initStreamPaths.Count];
        var runKeys = new (uint, uint)[runStreamPaths.Count];
        for (int i = 0; i < initKeys.Length; i++) initKeys[i] = FoldPath(initMaster, initStreamPaths[i]);
        for (int i = 0; i < runKeys.Length; i++) runKeys[i] = FoldPath(runMaster, runStreamPaths[i]);
        return (initKeys, runKeys);
    }

    private static (uint k0, uint k1) SplitWords(ulong key)
        => ((uint)(key & 0xFFFFFFFF), (uint)(key >> 32));

    /// <summary>
    /// Host-side index fold — one Threefry bijection, bit-identical to the in-graph
    /// SHRK_RNG_SPLIT: child = Bijection(counter: (index, 0), key).
    /// </summary>
    internal static (uint k0, uint k1) FoldKey((uint k0, uint k1) key, long index)
        => Core.Rng.Threefry2x32.Bijection(unchecked((uint)index), 0u, key.k0, key.k1);

    /// <summary>
    /// A trainable parameter's stream key: an explicit per-stream override when one is set,
    /// else the init master folded along the parameter's ModelId path (specific ids — loop
    /// slots carry real iteration values). SharedKey mode skips the fold so same-shape
    /// params tie (test/debug only).
    /// </summary>
    internal (uint k0, uint k1) FoldInitKey(IEnumerable<int> modelIdVals)
    {
        var vals = modelIdVals as IReadOnlyList<int> ?? new List<int>(modelIdVals);
        if (TryGetOverride(RngCollection.Params, vals, out var overridden)) return overridden;
        var key = InitMasterKey;
        if (SharedKey) return key;
        foreach (var v in vals) key = FoldKey(key, v);
        return key;
    }

    /// <summary>
    /// A runtime feed's stream key: an explicit per-stream override when one is set, else
    /// the runtime master folded along the feed's ModelId path.
    /// </summary>
    internal (uint k0, uint k1) FoldRunKey(IEnumerable<int> modelIdVals)
    {
        var vals = modelIdVals as IReadOnlyList<int> ?? new List<int>(modelIdVals);
        if (TryGetOverride(RngCollection.Runtime, vals, out var overridden)) return overridden;
        var key = RunMasterKey;
        foreach (var v in vals) key = FoldKey(key, v);
        return key;
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

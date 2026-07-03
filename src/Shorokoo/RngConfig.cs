using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;

namespace Shorokoo;

/// <summary>The bit generator a <see cref="RngConfig"/> uses. Configuration, never part of a model definition.</summary>
public enum RngAlgorithm
{
    /// <summary>Threefry-2x32 (Random123), 20 rounds. The default: expressible as portable ONNX integer ops.</summary>
    Threefry2x32,
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
/// never baked into the model definition. A model's random sites store only stable
/// <em>stream names</em>; this object turns a stream name into the concrete key that
/// seeds the draw.
///
/// <para>Key derivation: <c>key(stream) = override(stream)</c> if one is set, else
/// <c>Fold(MasterSeed, streamName)</c>, where <c>Fold</c> hashes the UTF-8 stream name
/// (SHA-256) and folds it into the master seed. So changing <see cref="MasterSeed"/>
/// re-randomizes everything; an <see cref="Override(string, ulong)"/> touches exactly
/// one stream; and because keys derive from <em>names</em> rather than draw order,
/// inserting or reordering unrelated sites does not disturb other streams.</para>
///
/// <para>Fully deterministic by default (<c>MasterSeed = 0</c>). Use
/// <see cref="NonDeterministic"/> for a fresh random stream each run.</para>
/// </summary>
public sealed class RngConfig
{
    /// <summary>The master seed folded into every non-overridden stream key. Default 0.</summary>
    public ulong MasterSeed { get; init; }

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

    // Full-stream-name ("params/..", "runtime/..") -> seed. Insertion into a live config;
    // frozen in practice once bound.
    private readonly Dictionary<string, ulong> _overrides = new(StringComparer.Ordinal);

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
        return new RngConfig { MasterSeed = BitConverter.ToUInt64(b) };
    }

    /// <summary>
    /// Pins a single stream to <paramref name="seed"/>, overriding the master-seed
    /// derivation for that stream only. <paramref name="streamName"/> is the full name
    /// including the collection prefix, e.g. <c>"params/backbone.block3.conv1.weight"</c>
    /// or <c>"runtime/head.dropout#0"</c>. Returns <c>this</c> for chaining.
    /// </summary>
    public RngConfig Override(string streamName, ulong seed)
    {
        ArgumentNullException.ThrowIfNull(streamName);
        _overrides[streamName] = seed;
        return this;
    }

    /// <summary>The full stream name (with collection prefix) for a site.</summary>
    public static string StreamName(RngCollection collection, string name)
        => CollectionPrefix(collection) + "/" + name;

    internal static string CollectionPrefix(RngCollection collection) => collection switch
    {
        RngCollection.Params => "params",
        RngCollection.Runtime => "runtime",
        _ => throw new ArgumentOutOfRangeException(nameof(collection)),
    };

    /// <summary>Whether a stream has an explicit override.</summary>
    public bool HasOverride(RngCollection collection, string name)
        => _overrides.ContainsKey(StreamName(collection, name));

    /// <summary>The 64-bit key material for a stream, before splitting into generator words.</summary>
    internal ulong ResolveKey64(RngCollection collection, string name)
    {
        if (SharedKey)
            return Fold(MasterSeed, CollectionPrefix(collection) + "/__shared__");
        string full = StreamName(collection, name);
        return _overrides.TryGetValue(full, out var seed) ? seed : Fold(MasterSeed, full);
    }

    /// <summary>The stream's key as the two 32-bit words the generator consumes.</summary>
    internal (uint k0, uint k1) ResolveKey(RngCollection collection, string name)
    {
        ulong key = ResolveKey64(collection, name);
        return ((uint)(key & 0xFFFFFFFF), (uint)(key >> 32));
    }

    /// <summary>
    /// The init-collection master key: two 32-bit words of <c>Fold(MasterSeed, "init")</c>.
    /// Every trainable-parameter stream folds from this along the parameter's ModelId path,
    /// so overriding the init sub-master re-rolls all weights while runtime streams stay put.
    /// </summary>
    internal (uint k0, uint k1) InitMasterKey => SplitWords(Fold(MasterSeed, "init"));

    /// <summary>The runtime-collection master key (Dropout masks, sampling, noise): words of <c>Fold(MasterSeed, "runtime")</c>.</summary>
    internal (uint k0, uint k1) RunMasterKey => SplitWords(Fold(MasterSeed, "runtime"));

    private static (uint k0, uint k1) SplitWords(ulong key)
        => ((uint)(key & 0xFFFFFFFF), (uint)(key >> 32));

    /// <summary>
    /// Host-side index fold — one Threefry bijection, bit-identical to the in-graph
    /// SHRK_RNG_SPLIT: child = Bijection(counter: (index, 0), key).
    /// </summary>
    internal static (uint k0, uint k1) FoldKey((uint k0, uint k1) key, long index)
        => Core.Rng.Threefry2x32.Bijection(unchecked((uint)index), 0u, key.k0, key.k1);

    /// <summary>
    /// A trainable parameter's stream key: the init master folded along the parameter's
    /// ModelId path (specific ids — loop slots carry real iteration values). SharedKey mode
    /// skips the fold so same-shape params tie (test/debug only).
    /// </summary>
    internal (uint k0, uint k1) FoldInitKey(IEnumerable<int> modelIdVals)
    {
        var key = InitMasterKey;
        if (SharedKey) return key;
        foreach (var v in modelIdVals) key = FoldKey(key, v);
        return key;
    }

    /// <summary>A runtime feed's stream key: the runtime master folded along the feed's ModelId path.</summary>
    internal (uint k0, uint k1) FoldRunKey(IEnumerable<int> modelIdVals)
    {
        var key = RunMasterKey;
        foreach (var v in modelIdVals) key = FoldKey(key, v);
        return key;
    }

    /// <summary>
    /// Folds a stream name into the master seed: <c>masterSeed XOR (first 8 bytes of
    /// SHA-256(name))</c>. Deterministic and platform-independent (SHA-256 + XOR).
    /// </summary>
    internal static ulong Fold(ulong masterSeed, string fullStreamName)
    {
        Span<byte> hash = stackalloc byte[32];
        SHA256.HashData(Encoding.UTF8.GetBytes(fullStreamName), hash);
        return masterSeed ^ BitConverter.ToUInt64(hash);
    }
}

using System;
using System.Linq;
using Shorokoo.Core.Rng;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage tests for the host RNG core (<see cref="Threefry2x32"/>, <see cref="HostRng"/>)
/// and the <see cref="RngConfig"/> key-derivation surface. The Threefry tests pin the
/// implementation against the Random123 known-answer vectors; the rest assert the
/// properties the RNG design relies on — determinism, name-derived independence,
/// override isolation, and standard-distribution shape.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngCoreTests
{
    // Random123 known-answer test vectors for threefry2x32, 20 rounds
    // (tests/kat_vectors in DEShawResearch/random123): counter, key -> output.
    [Theory]
    [InlineData(0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x6b200159u, 0x99ba4efeu)]
    [InlineData(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0x1cb996fcu, 0xbb002be7u)]
    [InlineData(0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u, 0xc4923a9cu, 0x483df7a0u)]
    public void TestThreefry2x32KnownAnswerVectors(
        uint c0, uint c1, uint k0, uint k1, uint expected0, uint expected1)
    {
        var (x0, x1) = Threefry2x32.Bijection(c0, c1, k0, k1);
        Assert.Equal(expected0, x0);
        Assert.Equal(expected1, x1);
    }

    // Random123 known-answer vectors for threefry2x32, 13 rounds (the Crush-resistant fast
    // variant, RngAlgorithm.Threefry2x32Rounds13). The all-zero vector (9d1c5ec6, 8bd50731)
    // is the published threefry2x32x13 KAT; the others pin the reduced-round output against
    // regression. This anchors the 13-round injection schedule (after rounds 4/8/12, none
    // trailing) to a reference, not just to self-agreement with the in-graph lowering.
    [Theory]
    [InlineData(0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x9d1c5ec6u, 0x8bd50731u)]
    [InlineData(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xfd36d048u, 0x2d17272cu)]
    [InlineData(0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u, 0xba3e4725u, 0xf27d669eu)]
    public void TestThreefry2x32Rounds13KnownAnswerVectors(
        uint c0, uint c1, uint k0, uint k1, uint expected0, uint expected1)
    {
        var (x0, x1) = Threefry2x32.Bijection(c0, c1, k0, k1, Threefry2x32.Rounds13);
        Assert.Equal(expected0, x0);
        Assert.Equal(expected1, x1);
        // The round count genuinely changes the output (guards against an ignored/miswired
        // rounds parameter that would make the 13-round algorithm alias the 20-round default).
        Assert.NotEqual((x0, x1), Threefry2x32.Bijection(c0, c1, k0, k1, Threefry2x32.Rounds));
    }

    [Fact]
    public void TestThreefryIsPureFunction()
    {
        var a = Threefry2x32.Bijection(7, 42, 123, 456);
        var b = Threefry2x32.Bijection(7, 42, 123, 456);
        Assert.Equal(a, b);
        // Distinct counters (same key) and distinct keys (same counter) both diverge.
        Assert.NotEqual(a, Threefry2x32.Bijection(8, 42, 123, 456));
        Assert.NotEqual(a, Threefry2x32.Bijection(7, 42, 124, 456));
    }

    [Fact]
    public void TestHostUniformIsInRangeAndDeterministic()
    {
        var rng = new HostRng(1, 2);
        var a = rng.StandardUniform(10_000);
        var b = new HostRng(1, 2).StandardUniform(10_000);
        Assert.Equal(a, b);                                   // deterministic for a key
        Assert.All(a, v => Assert.InRange(v, 0.0f, 0.99999997f)); // [0,1), never 1
        Assert.InRange(a.Average(), 0.48f, 0.52f);            // ~0.5 mean
    }

    [Fact]
    public void TestHostNormalHasStandardMoments()
    {
        var z = new HostRng(99, 7).StandardNormal(100_000);
        double mean = z.Average();
        double var = z.Select(v => (v - mean) * (v - mean)).Average();
        Assert.InRange(mean, -0.02, 0.02);                    // ~0 mean
        Assert.InRange(var, 0.95, 1.05);                      // ~unit variance
    }

    [Fact]
    public void TestDistinctKeysGiveIndependentStreams()
    {
        var a = new HostRng(1, 0).StandardUniform(4096);
        var b = new HostRng(2, 0).StandardUniform(4096);
        // Different keys must not produce the same stream.
        Assert.NotEqual(a, b);
        // Correlation between the two streams should be near zero (independent).
        double ma = a.Average(), mb = b.Average();
        double cov = a.Zip(b, (x, y) => (x - ma) * (y - mb)).Average();
        Assert.InRange(cov, -0.01, 0.01);
    }

    [Fact]
    public void TestCounterBaseOffsetsTheStream()
    {
        // The stream at counterBase=k continues the base stream: block b at base k
        // equals block b+k at base 0. Two draws per block, so index 2k aligns.
        var full = new HostRng(5, 6).StandardUniform(64);
        var shifted = new HostRng(5, 6, counterBase: 3).StandardUniform(8);
        Assert.Equal(full.Skip(6).Take(8).ToArray(), shifted);
    }

    [Fact]
    public void TestConfigDefaultIsDeterministicMasterSeedZero()
    {
        Assert.Equal(0ul, RngConfig.Default.MasterSeed);
        Assert.Equal(RngAlgorithm.Threefry2x32, RngConfig.Default.Algorithm);
    }

    [Fact]
    public void TestKeyDerivationIsPathDerivedAndStable()
    {
        var cfg = new RngConfig { MasterSeed = 20260702 };
        var k1 = cfg.FoldInitKey([3, 1, 1]);
        var k2 = cfg.FoldInitKey([3, 1, 1]);
        var k3 = cfg.FoldInitKey([3, 1, 2]);
        Assert.Equal(k1, k2);                                 // stable for a path
        Assert.NotEqual(k1, k3);                              // sibling paths differ
        // Same path in the runtime collection is a different stream (distinct sub-master).
        Assert.NotEqual(k1, cfg.FoldRunKey([3, 1, 1]));
    }

    [Fact]
    public void TestMasterSeedChangeRerandomizesEveryStream()
    {
        var a = new RngConfig { MasterSeed = 1 };
        var b = new RngConfig { MasterSeed = 2 };
        Assert.NotEqual(a.FoldInitKey([1]), b.FoldInitKey([1]));
        Assert.NotEqual(a.FoldRunKey([1]), b.FoldRunKey([1]));
    }

    [Fact]
    public void TestOverrideIsolatesASingleStream()
    {
        var baseCfg = new RngConfig { MasterSeed = 7 };
        var cfg = new RngConfig { MasterSeed = 7 }
            .Override(RngCollection.Params, [1, 1], seed: 1234);

        // The overridden stream changes; siblings, sub-paths, and the runtime collection
        // keep their derived keys (matching is exact and per-collection).
        Assert.NotEqual(baseCfg.FoldInitKey([1, 1]), cfg.FoldInitKey([1, 1]));
        Assert.Equal(baseCfg.FoldInitKey([1, 2]), cfg.FoldInitKey([1, 2]));
        Assert.Equal(baseCfg.FoldInitKey([1, 1, 1]), cfg.FoldInitKey([1, 1, 1]));
        Assert.Equal(baseCfg.FoldRunKey([1, 1]), cfg.FoldRunKey([1, 1]));
        Assert.True(cfg.HasOverride(RngCollection.Params, [1, 1]));
        Assert.False(cfg.HasOverride(RngCollection.Params, [1, 2]));

        // The override replaces the fully folded key, so it survives a master-seed change.
        var otherMaster = new RngConfig { MasterSeed = 8 }
            .Override(RngCollection.Params, [1, 1], seed: 1234);
        Assert.Equal(cfg.FoldInitKey([1, 1]), otherMaster.FoldInitKey([1, 1]));
    }

    [Fact]
    public void TestOverrideReturnsACopyAndNeverMutatesTheReceiver()
    {
        // Configs are immutable values: Override returns a modified copy carrying every
        // property, and the receiver — crucially including the process-wide Default — is
        // untouched. (Guards against the shared-mutable-singleton hazard: one caller's
        // fluent tweak must never re-key another model's streams.)
        var baseCfg = new RngConfig { MasterSeed = 7, Algorithm = RngAlgorithm.Threefry2x32Rounds13 };
        var derived = baseCfg.Override(RngCollection.Params, [1, 1], seed: 1234);

        Assert.False(baseCfg.HasOverride(RngCollection.Params, [1, 1]));
        Assert.True(derived.HasOverride(RngCollection.Params, [1, 1]));
        Assert.Equal(baseCfg.MasterSeed, derived.MasterSeed);
        Assert.Equal(baseCfg.Algorithm, derived.Algorithm);

        // Stacking builds on the copy; earlier copies stay at their own override sets.
        var stacked = derived.Override(RngCollection.Runtime, [2], seed: 9);
        Assert.False(derived.HasOverride(RngCollection.Runtime, [2]));
        Assert.True(stacked.HasOverride(RngCollection.Params, [1, 1]));
        Assert.True(stacked.HasOverride(RngCollection.Runtime, [2]));

        _ = RngConfig.Default.Override(RngCollection.Params, [1, 1], seed: 7);
        Assert.False(RngConfig.Default.HasOverride(RngCollection.Params, [1, 1]));
    }
}

/// <summary>
/// The compact RNG key vector — the single tensor a model carries as the SOURCE OF TRUTH for
/// its randomness state. Three tiers: [master]; [master, initMaster, runMaster]; the masters
/// plus path-keyed, self-describing override records. <see cref="RngConfig.FromKeyVector"/>
/// must decode a config that derives every stream key bit-exactly like the original — with no
/// stream inventory and no enumeration-order contract.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngKeyVectorTests
{
    private static readonly int[][] Paths = [[1, 1], [2, 1], [3], [4, 0, 1], [4, 1, 1]];

    private static void AssertRoundTrips(RngConfig cfg)
    {
        var decoded = RngConfig.FromKeyVector(cfg.BuildKeyVector());
        foreach (var p in Paths)
        {
            Assert.Equal(cfg.FoldInitKey(p), decoded.FoldInitKey(p));
            Assert.Equal(cfg.FoldRunKey(p), decoded.FoldRunKey(p));
        }
    }

    [Fact]
    public void TestTier1MasterOnly()
    {
        var cfg = new RngConfig { MasterSeed = 42 };
        var vec = cfg.BuildKeyVector();
        Assert.Equal([42L], vec);
        AssertRoundTrips(cfg);
    }

    [Fact]
    public void TestTier2SubMasters()
    {
        // Explicit run sub-master: feeds re-seed, init stays derived from the master.
        var cfg = new RngConfig { MasterSeed = 42, RunMasterSeed = 777 };
        Assert.Equal(3, cfg.BuildKeyVector().Length);
        AssertRoundTrips(cfg);

        // The sub-master genuinely changed the runtime keys and left init untouched.
        var baseline = new RngConfig { MasterSeed = 42 };
        Assert.NotEqual(baseline.FoldRunKey(Paths[0]), cfg.FoldRunKey(Paths[0]));
        Assert.Equal(baseline.FoldInitKey(Paths[0]), cfg.FoldInitKey(Paths[0]));

        AssertRoundTrips(new RngConfig { MasterSeed = 42, InitMasterSeed = 888 });
    }

    [Fact]
    public void TestTier3PathKeyedOverrideRecords()
    {
        // Overrides in BOTH collections, multi-element paths included. Each record encodes
        // (collection, path length, path, seed), so decoding needs no stream enumeration.
        var cfg = new RngConfig { MasterSeed = 42, RunMasterSeed = 777 };
        cfg = cfg.Override(RngCollection.Runtime, [4, 1, 1], seed: 424242UL)
                 .Override(RngCollection.Params, [2, 1], seed: 7UL);

        var vec = cfg.BuildKeyVector();
        // 3 masters + count + (1 + 1 + 3 + 1) + (1 + 1 + 2 + 1) elements.
        Assert.Equal(3 + 1 + 6 + 5, vec.Length);
        AssertRoundTrips(cfg);

        var decoded = RngConfig.FromKeyVector(vec);
        Assert.True(decoded.HasOverride(RngCollection.Runtime, [4, 1, 1]));
        Assert.True(decoded.HasOverride(RngCollection.Params, [2, 1]));
        Assert.False(decoded.HasOverride(RngCollection.Runtime, [4, 0, 1]));

        // Sibling streams keep their derived keys; only the overridden ones deviate.
        var noOverride = new RngConfig { MasterSeed = 42, RunMasterSeed = 777 };
        Assert.Equal(noOverride.FoldRunKey([4, 0, 1]), cfg.FoldRunKey([4, 0, 1]));
        Assert.NotEqual(noOverride.FoldRunKey([4, 1, 1]), cfg.FoldRunKey([4, 1, 1]));
        Assert.NotEqual(noOverride.FoldInitKey([2, 1]), cfg.FoldInitKey([2, 1]));
    }

    [Fact]
    public void TestMalformedVectorFailsLoudly()
    {
        // Corrupt carriers must throw, never silently fall back to a different derivation.
        Assert.ThrowsAny<ArgumentException>(() => RngConfig.FromKeyVector([]));
        Assert.ThrowsAny<ArgumentException>(() => RngConfig.FromKeyVector([1L, 2L]));
        // Truncated override record (claims one record, supplies nothing).
        Assert.ThrowsAny<ArgumentException>(() => RngConfig.FromKeyVector([1L, 2L, 3L, 1L]));
        // Trailing garbage after the declared records.
        Assert.ThrowsAny<ArgumentException>(
            () => RngConfig.FromKeyVector([1L, 2L, 3L, 0L, 99L]));
    }

    [Fact]
    public void TestMalformedVectorRejectsAnInvalidCollectionTag()
    {
        // Regression: the collection tag must be exactly 0 (Params) or 1 (Runtime) — a
        // corrupted tag used to decode as Runtime whenever it merely wasn't 0, silently
        // accepting garbage as a valid override instead of failing loudly like every
        // other malformed field in the record.
        long[] vec = [1L, 2L, 3L, 1L, /* tag */ 7L, /* pathLen */ 1L, /* path */ 5L, /* seed */ 99L];
        var ex = Assert.ThrowsAny<ArgumentException>(() => RngConfig.FromKeyVector(vec));
        Assert.Contains("collection tag", ex.Message);
    }

    [Fact]
    public void TestMalformedVectorRejectsAnOverflowingPathLength()
    {
        // Regression: the truncated-record bounds check used plain `int` arithmetic, so a
        // corrupted pathLen near int.MaxValue could overflow the check into a false
        // negative and fall through to `new int[pathLen]` — crashing with an unhelpful
        // OutOfMemory/IndexOutOfRange instead of this method's own clean diagnostic. The
        // fixed check widens to `long` before comparing, so it must throw promptly here
        // (no multi-gigabyte allocation should ever be attempted).
        long[] vec = [1L, 2L, 3L, 1L, /* tag */ 0L, /* pathLen */ int.MaxValue, /* truncated */ 99L];
        var ex = Assert.ThrowsAny<ArgumentException>(() => RngConfig.FromKeyVector(vec));
        Assert.Contains("truncated", ex.Message);
    }
}

/// <summary>
/// The key-vector transport: ApplyRngConfig writes the compact vector as the graph's single
/// carrier node (mirrored into the saved file as a reserved-name initializer, lowered to a
/// plain CONSTANT at ONNX prep); it survives save/load bit-exactly and without duplication,
/// and the loaded model's key derivation reads it with no config object needed.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngKeyVectorTransportTests
{
    [Fact]
    public void TestKeyVectorCarrierRoundTripsWithoutDuplication()
    {
        var g = (FastComputationGraph)typeof(RngRuntimeLoopFeed)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var x = TensorData([8L], new float[8]);
        var steps = TensorData(System.Array.Empty<long>(), 2L);
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([x, steps]));

        var cfg = new RngConfig { MasterSeed = 11 };
        cfg = cfg.Override(RngCollection.Runtime, [1, 1, 1], seed: 424242UL);   // tier 3
        arch.ApplyRngConfig(cfg);

        // Binding writes exactly one node (the carrier) and nothing per-feed.
        Assert.Single(arch.Nodes, n => n.OpCode == InternalOpCodes.SHRK_RNG_KEY_VECTOR);

        var data = CompressedFormatUtils.SaveFastGraphToBinary(arch, compressed: true);
        var loaded = CompressedFormatUtils.LoadFastGraphFromBinary(data, isCompressed: true);
        var carried = loaded.TryGetRngKeyVector();
        Assert.NotNull(carried);
        Assert.Contains("Threefry", carried!.Value.algorithm);
        Assert.Equal(arch.TryGetRngKeyVector()!.Value.keyVector, carried.Value.keyVector);

        // The decoded carrier reproduces the config's derivation, override included — the
        // loaded model needs no config object.
        var decoded = RngConfig.FromKeyVector(carried.Value.keyVector);
        Assert.True(decoded.HasOverride(RngCollection.Runtime, [1, 1, 1]));
        Assert.Equal(cfg.FoldRunKey([1, 1, 1]), decoded.FoldRunKey([1, 1, 1]));
        Assert.Equal(cfg.FoldRunKey([1, 0, 1]), decoded.FoldRunKey([1, 0, 1]));

        // Exactly one carrier after each save/load cycle — the file's single representation
        // (the reserved-name initializer) never accumulates duplicates.
        Assert.Single(loaded.Nodes, n => n.OpCode == InternalOpCodes.SHRK_RNG_KEY_VECTOR);
        var loaded2 = CompressedFormatUtils.LoadFastGraphFromBinary(
            CompressedFormatUtils.SaveFastGraphToBinary(loaded, compressed: true), isCompressed: true);
        Assert.Single(loaded2.Nodes, n => n.OpCode == InternalOpCodes.SHRK_RNG_KEY_VECTOR);
        Assert.Equal(carried.Value.keyVector, loaded2.TryGetRngKeyVector()!.Value.keyVector);
    }

    [Fact]
    public void TestRebindingReplacesTheCarrier()
    {
        var g = (FastComputationGraph)typeof(RngRuntimeLoopFeed)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var x = TensorData([8L], new float[8]);
        var steps = TensorData(System.Array.Empty<long>(), 2L);
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([x, steps]));

        arch.ApplyRngConfig(new RngConfig { MasterSeed = 11 });
        Assert.Equal([11L], arch.TryGetRngKeyVector()!.Value.keyVector);   // tier 1: master only

        arch.ApplyRngConfig(new RngConfig { MasterSeed = 12 });
        Assert.Equal([12L], arch.TryGetRngKeyVector()!.Value.keyVector);
        Assert.Single(arch.Nodes, n => n.OpCode == InternalOpCodes.SHRK_RNG_KEY_VECTOR);
    }

    [Fact]
    public void TestRebindingLoweredStreamsFailsLoudly()
    {
        // A prepped-for-ONNX export lowers every key entity to a plain CONSTANT (the
        // interchange artifact must be ordinary ONNX). Re-importing such a file gives a
        // model whose streams can no longer be re-keyed: binding a config to it must
        // throw — silently recording an identity the baked draw keys don't use is
        // exactly the divergence the carrier-as-source-of-truth contract forbids.
        var g = (FastComputationGraph)typeof(RtLoweredUniform)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var input = TensorData([4L, 4L], new float[16]);
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([input]))
            .ToConcreteModel(new RngConfig { MasterSeed = 1 });

        var proto = Shorokoo.Core.Factory.FastOnnxModelBuilder.BuildOnnxModel(
            concrete, prepForOnnx: true);
        using var ms = new System.IO.MemoryStream();
        ProtoBuf.Serializer.Serialize(ms, proto);
        var imported = OnnxModelImporter.FromOnnxModelToFastGraph(ms.ToArray());

        var ex = Assert.Throws<System.InvalidOperationException>(
            () => imported.ApplyRngConfig(new RngConfig { MasterSeed = 2 }));
        Assert.Contains("lowered", ex.Message);
    }

    [Fact]
    public void TestBindingRequiresRealizedStreams()
    {
        // The concreteness contract at bind: an id-bearing feed without realized stream ids
        // (a graph that never went through ToConcreteArchitecture) fails loudly.
        var draw = RandomUniform(Vector(4L), 0f, 1f);
        var graph = new FastComputationGraph([], [draw]);
        var feed = graph.Nodes.Single(n => n.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM);
        feed.Attributes = feed.Attributes.SetAttributes(
            (OnnxOpAttributeNames.ShrkAttrLocalModelId, (long[])[1]));

        var ex = Assert.Throws<System.InvalidOperationException>(
            () => graph.ApplyRngConfig(new RngConfig { MasterSeed = 1 }));
        Assert.Contains("no realized stream ids", ex.Message);
    }
}

using System;
using System.Linq;
using Shorokoo.Core.Rng;
using Shorokoo.Runtime;

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
/// The encoded runtime RNG identity — the value of the ordinary <c>RngSeed</c> parameter at
/// reserved ModelId [0] (see <see cref="RngRuntimeIdentity"/>): an algorithm-id header, the
/// runtime master key words, and canonically sorted per-stream override records at fixed
/// offsets. <see cref="RngRuntimeIdentity.Decode"/> must derive every runtime stream key
/// bit-exactly like the encoding config. The init-collection identity is deliberately NOT
/// encoded — nothing in a saved model consumes it.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngRuntimeIdentityTests
{
    private static readonly int[][] Paths = [[1, 1], [2, 1], [3], [4, 0, 1], [4, 1, 1]];

    private static void AssertRoundTrips(RngConfig cfg)
    {
        var decoded = RngRuntimeIdentity.Decode(RngRuntimeIdentity.Build(cfg));
        Assert.Equal(RngRuntimeIdentity.AlgorithmIdOf(cfg.Algorithm), decoded.AlgorithmId);
        foreach (var p in Paths)
            Assert.Equal(cfg.FoldRunKey(p), decoded.FoldRunKey(p));
    }

    [Fact]
    public void TestHeaderOnlyIdentity()
    {
        var cfg = new RngConfig { MasterSeed = 42 };
        var vec = RngRuntimeIdentity.Build(cfg);
        // Header only: [algId, runK0, runK1, 0 overrides].
        Assert.Equal(RngRuntimeIdentity.HeaderLength, vec.Length);
        Assert.Equal(0L, vec[RngRuntimeIdentity.AlgorithmIdIndex]);
        Assert.Equal(cfg.RunMasterKey.k0, (uint)vec[RngRuntimeIdentity.RunKeyIndex]);
        Assert.Equal(cfg.RunMasterKey.k1, (uint)vec[RngRuntimeIdentity.RunKeyIndex + 1]);
        AssertRoundTrips(cfg);

        // The algorithm id header switches with the configured algorithm.
        var cfg13 = new RngConfig { MasterSeed = 42, Algorithm = RngAlgorithm.Threefry2x32Rounds13 };
        Assert.Equal(1L, RngRuntimeIdentity.Build(cfg13)[RngRuntimeIdentity.AlgorithmIdIndex]);
        AssertRoundTrips(cfg13);

        // An explicit run sub-master re-seeds the runtime tier and rides the same header.
        var subMaster = new RngConfig { MasterSeed = 42, RunMasterSeed = 777 };
        Assert.NotEqual(cfg.FoldRunKey(Paths[0]), subMaster.FoldRunKey(Paths[0]));
        AssertRoundTrips(subMaster);
    }

    [Fact]
    public void TestOverrideRecordsEncodeAtFixedOffsets()
    {
        // Runtime overrides only — a Params override is init-side material and must NOT be
        // persisted in the runtime identity.
        var cfg = new RngConfig { MasterSeed = 42, RunMasterSeed = 777 };
        cfg = cfg.Override(RngCollection.Runtime, [4, 1, 1], seed: 424242UL)
                 .Override(RngCollection.Params, [2, 1], seed: 7UL);

        var vec = RngRuntimeIdentity.Build(cfg);
        // Header + one record: length 3, its 3 path elements, 2 key words.
        Assert.Equal(RngRuntimeIdentity.HeaderLength + 1 + 3 + 2, vec.Length);
        Assert.Equal(1L, vec[RngRuntimeIdentity.HeaderLength - 1]);   // record count

        var decoded = RngRuntimeIdentity.Decode(vec);
        var rec = Assert.Single(decoded.Overrides);
        Assert.Equal((int[])[4, 1, 1], rec.Path);
        // The record replaces the fully folded key: its words are the override seed's words,
        // and they sit at the record's fixed key offset in the vector.
        Assert.Equal(RngConfig.SplitWords(424242UL), rec.Key);
        Assert.Equal(rec.Key.k0, (uint)vec[rec.KeyOffset]);
        Assert.Equal(rec.Key.k1, (uint)vec[rec.KeyOffset + 1]);

        // Derivation round-trips: the overridden stream deviates, siblings stay derived.
        AssertRoundTrips(cfg);
        var noOverride = new RngConfig { MasterSeed = 42, RunMasterSeed = 777 };
        Assert.Equal(noOverride.FoldRunKey([4, 0, 1]), decoded.FoldRunKey([4, 0, 1]));
        Assert.NotEqual(noOverride.FoldRunKey([4, 1, 1]), decoded.FoldRunKey([4, 1, 1]));
    }

    [Fact]
    public void TestMalformedIdentityFailsLoudly()
    {
        // Corrupt identities must throw, never silently fall back to a different derivation.
        Assert.ThrowsAny<ArgumentException>(() => RngRuntimeIdentity.Decode([]));
        Assert.ThrowsAny<ArgumentException>(() => RngRuntimeIdentity.Decode([0L, 1L, 2L]));
        // Truncated override record (claims one record, supplies nothing).
        Assert.ThrowsAny<ArgumentException>(() => RngRuntimeIdentity.Decode([0L, 1L, 2L, 1L]));
        // Trailing garbage after the declared records.
        Assert.ThrowsAny<ArgumentException>(() => RngRuntimeIdentity.Decode([0L, 1L, 2L, 0L, 99L]));
    }
}

/// <summary>
/// The identity transport: ApplyRngConfig writes the runtime identity into the ordinary
/// <c>RngSeed</c> parameter at reserved ModelId [0] — serialized as a plain initializer with
/// no reserved-name handling; it survives save/load bit-exactly and without duplication, the
/// loaded model's randomness is reproducible with no config object, and re-binding a LOADED
/// model is a parameter write that re-keys every draw (the re-bind-after-load pin).
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngSeedTransportTests
{
    private static int RngSeedNodeCount(FastComputationGraph graph)
        => graph.Nodes.Count(n =>
            n.IdentifierTemplate == Shorokoo.Core.Nodes.Processors.Fast
                .FastWireRngKeyDerivation.RngSeedIdentifierTemplate);

    [Fact]
    public void TestRngSeedIdentityRoundTripsWithoutDuplication()
    {
        var g = (FastComputationGraph)typeof(RngRuntimeLoopFeed)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var x = TensorData([8L], new float[8]);
        var steps = TensorData(System.Array.Empty<long>(), 2L);
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([x, steps]));

        var cfg = new RngConfig { MasterSeed = 11 };
        cfg = cfg.Override(RngCollection.Runtime, [1, 1, 1], seed: 424242UL);
        arch.ApplyRngConfig(cfg);

        // Exactly one RngSeed parameter, holding the encoded identity.
        Assert.Equal(1, RngSeedNodeCount(arch));
        Assert.Equal(RngRuntimeIdentity.Build(cfg), arch.TryGetRngSeed());

        var data = CompressedFormatUtils.SaveFastGraphToBinary(arch, compressed: true);
        var loaded = CompressedFormatUtils.LoadFastGraphFromBinary(data, isCompressed: true);
        var carried = loaded.TryGetRngSeed();
        Assert.NotNull(carried);
        Assert.Equal(arch.TryGetRngSeed(), carried);

        // The decoded identity reproduces the config's runtime derivation, override included
        // — the loaded model needs no config object.
        var decoded = RngRuntimeIdentity.Decode(carried!);
        Assert.Equal(RngRuntimeIdentity.AlgorithmIdOf(RngAlgorithm.Threefry2x32), decoded.AlgorithmId);
        Assert.Equal(cfg.FoldRunKey([1, 1, 1]), decoded.FoldRunKey([1, 1, 1]));
        Assert.Equal(cfg.FoldRunKey([1, 0, 1]), decoded.FoldRunKey([1, 0, 1]));

        // Exactly one RngSeed parameter after each save/load cycle — an ordinary initializer
        // never accumulates duplicates.
        Assert.Equal(1, RngSeedNodeCount(loaded));
        var loaded2 = CompressedFormatUtils.LoadFastGraphFromBinary(
            CompressedFormatUtils.SaveFastGraphToBinary(loaded, compressed: true), isCompressed: true);
        Assert.Equal(1, RngSeedNodeCount(loaded2));
        Assert.Equal(carried, loaded2.TryGetRngSeed());
    }

    [Fact]
    public void TestNonDefaultAlgorithmSurvivesSaveLoadByIdAndByBehavior()
    {
        // The algorithm identity rides the file in TWO forms — the RngSeed identity's
        // algorithm id (trusted by no-config parameter initialization and the lowering) and
        // the baked tagged draw functions (what the feeds actually execute) — and they can in
        // principle disagree. Bind the NON-default algorithm, round-trip the concrete model,
        // and pin both: the id decodes exactly, and the loaded model still draws 13-round
        // values (equal to its own pre-save draws, different from a default-algorithm model
        // under the same seed).
        var g = (FastComputationGraph)typeof(RngRuntimeLoopFeed)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var x = TensorData([8L], new float[8]);
        var steps = TensorData(System.Array.Empty<long>(), 2L);

        FastComputationGraph Concrete(RngConfig cfg) =>
            g.ToConcreteArchitecture(g.FromOrderedInputs([x, steps])).ToConcreteModel(cfg);
        float[] Run(FastComputationGraph m) => ComputeContext.Default.Execute(m, x, steps)[0]
            .ToTensorData().As<float32>().AccessMemory().ToArray();

        var m13 = Concrete(new RngConfig { MasterSeed = 11, Algorithm = RngAlgorithm.Threefry2x32Rounds13 });
        var before = Run(m13);
        var draws20 = Run(Concrete(new RngConfig { MasterSeed = 11 }));   // same seed, default rounds
        Assert.NotEqual(before, draws20);   // guard: the two algorithms genuinely differ here

        var loaded = CompressedFormatUtils.LoadFastGraphFromBinary(
            CompressedFormatUtils.SaveFastGraphToBinary(m13, compressed: true), isCompressed: true);

        var carried = loaded.TryGetRngSeed();
        Assert.NotNull(carried);
        Assert.Equal(RngRuntimeIdentity.AlgorithmIdOf(RngAlgorithm.Threefry2x32Rounds13),
            RngRuntimeIdentity.Decode(carried!).AlgorithmId);

        var after = Run(loaded);
        Assert.Equal(before, after);        // still draws its pre-save 13-round values
        Assert.NotEqual(draws20, after);    // and not the default algorithm's
    }

    [Fact]
    public void TestRebindingReplacesTheIdentityValue()
    {
        var g = (FastComputationGraph)typeof(RngRuntimeLoopFeed)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var x = TensorData([8L], new float[8]);
        var steps = TensorData(System.Array.Empty<long>(), 2L);
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([x, steps]));

        arch.ApplyRngConfig(new RngConfig { MasterSeed = 11 });
        Assert.Equal(RngRuntimeIdentity.Build(new RngConfig { MasterSeed = 11 }), arch.TryGetRngSeed());

        arch.ApplyRngConfig(new RngConfig { MasterSeed = 12 });
        Assert.Equal(RngRuntimeIdentity.Build(new RngConfig { MasterSeed = 12 }), arch.TryGetRngSeed());
        Assert.Equal(1, RngSeedNodeCount(arch));
    }

    [Fact]
    public void TestRebindAfterSaveLoadRekeysEveryDraw()
    {
        // THE re-bind-after-load pin: bind seed A -> save -> load -> ApplyRngConfig(B) ->
        // every draw changes AND matches a model bound to B directly. With the identity as an
        // ordinary parameter and keys derived in-graph from it, re-binding a loaded model is
        // a parameter write that re-keys every draw by construction — the divergence class
        // where a loaded model's recorded identity updated while the draws kept the old seed
        // is structurally impossible.
        var g = (FastComputationGraph)typeof(RngRuntimeLoopFeed)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var x = TensorData([8L], new float[8]);
        var steps = TensorData(System.Array.Empty<long>(), 2L);

        FastComputationGraph Concrete(RngConfig cfg) =>
            g.ToConcreteArchitecture(g.FromOrderedInputs([x, steps])).ToConcreteModel(cfg);
        float[] Run(FastComputationGraph m) => ComputeContext.Default.Execute(m, x, steps)[0]
            .ToTensorData().As<float32>().AccessMemory().ToArray();

        var seedA = new RngConfig { MasterSeed = 11 };
        var seedB = new RngConfig { MasterSeed = 12 };

        var modelA = Concrete(seedA);
        var drawsA = Run(modelA);

        var loaded = CompressedFormatUtils.LoadFastGraphFromBinary(
            CompressedFormatUtils.SaveFastGraphToBinary(modelA, compressed: true), isCompressed: true);
        Assert.Equal(drawsA, Run(loaded));            // load-and-run reproduces seed A

        loaded.ApplyRngConfig(seedB);                 // a parameter write on the loaded graph
        var rekeyed = Run(loaded);
        Assert.NotEqual(drawsA, rekeyed);             // every draw changed
        Assert.Equal(Run(Concrete(seedB)), rekeyed);  // and matches a direct seed-B model

        // Round-trip again after the re-bind: the new identity is what persists.
        var reloaded = CompressedFormatUtils.LoadFastGraphFromBinary(
            CompressedFormatUtils.SaveFastGraphToBinary(loaded, compressed: true), isCompressed: true);
        Assert.Equal(rekeyed, Run(reloaded));
    }

    [Fact]
    public void TestRebindOnLoadedModelCannotChangeOverrideSetOrAlgorithm()
    {
        // A loaded model's draws are baked function calls: seed VALUES re-key freely (the
        // parameter write), but the override SET's routing and the draw algorithm are
        // structural — changing either on a loaded graph must fail loudly, never silently
        // half-apply.
        var g = (FastComputationGraph)typeof(RngRuntimeLoopFeed)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var x = TensorData([8L], new float[8]);
        var steps = TensorData(System.Array.Empty<long>(), 2L);
        var model = g.ToConcreteArchitecture(g.FromOrderedInputs([x, steps]))
            .ToConcreteModel(new RngConfig { MasterSeed = 11 });
        var loaded = CompressedFormatUtils.LoadFastGraphFromBinary(
            CompressedFormatUtils.SaveFastGraphToBinary(model, compressed: true), isCompressed: true);

        var withOverride = new RngConfig { MasterSeed = 11 }
            .Override(RngCollection.Runtime, [1, 1, 1], 42UL);
        var ex1 = Assert.Throws<System.InvalidOperationException>(
            () => loaded.ApplyRngConfig(withOverride));
        Assert.Contains("override SET", ex1.Message);

        var ex2 = Assert.Throws<System.InvalidOperationException>(
            () => loaded.ApplyRngConfig(new RngConfig
            { MasterSeed = 11, Algorithm = RngAlgorithm.Threefry2x32Rounds13 }));
        Assert.Contains("algorithm", ex2.Message);
    }

    [Fact]
    public void TestLegacyBakedFileFailsRebindLoudly()
    {
        // A file saved before the RngSeed representation carries baked key-table constants
        // plus the reserved-name identity initializer — nothing left to re-key. Loading such
        // a file yields a graph with the legacy marker and no RngSeed parameter; binding it
        // must throw naming the situation (the old behavior silently updated only the
        // recorded identity). Simulate the loaded shape: no feeds, no RngSeed, the legacy
        // reserved-name tensor present as an ordinary data node.
        var g = (FastComputationGraph)typeof(RtLoweredUniform)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var input = TensorData([4L, 4L], new float[16]);
        var model = g.ToConcreteArchitecture(g.FromOrderedInputs([input]))
            .ToConcreteModel(new RngConfig { MasterSeed = 1 });

        // Strip the new representation down to the legacy shape.
        var legacy = CompressedFormatUtils.LoadFastGraphFromBinary(
            CompressedFormatUtils.SaveFastGraphToBinary(model, compressed: true), isCompressed: true);
        var seedNode = legacy.Nodes.Single(n =>
            n.IdentifierTemplate == Shorokoo.Core.Nodes.Processors.Fast
                .FastWireRngKeyDerivation.RngSeedIdentifierTemplate);
        seedNode.IdentifierTemplate = null;
        seedNode.FriendlyName = OnnxOpAttributeNames.ShrkRngKeysTensorName;

        var ex = Assert.Throws<System.InvalidOperationException>(
            () => legacy.ApplyRngConfig(new RngConfig { MasterSeed = 2 }));
        Assert.Contains(OnnxOpAttributeNames.ShrkRngKeysTensorName, ex.Message);
        Assert.Contains("cannot be re-keyed", ex.Message);
    }

    [Fact]
    public void TestModelWithoutRandomFeedsCarriesNothingRngRelated()
    {
        // A model with no runtime random feeds contains no RngSeed param, no chains, and
        // nothing RNG-related in its saved form; binding a config to it is a harmless no-op —
        // but a Runtime override, which can match nothing, still fails loudly.
        var g = (FastComputationGraph)typeof(RngInitTwoLinears)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var sample = TensorData([4L, 4L], Enumerable.Repeat(1f, 16).ToArray());
        var model = g.ToConcreteArchitecture(g.FromOrderedInputs([sample]))
            .ToConcreteModel(new RngConfig { MasterSeed = 7 });

        Assert.Equal(0, RngSeedNodeCount(model));
        Assert.Null(model.TryGetRngSeed());
        var loaded = CompressedFormatUtils.LoadFastGraphFromBinary(
            CompressedFormatUtils.SaveFastGraphToBinary(model, compressed: true), isCompressed: true);
        Assert.Equal(0, RngSeedNodeCount(loaded));
        Assert.DoesNotContain(loaded.Nodes, n =>
            n.FriendlyName == OnnxOpAttributeNames.ShrkRngKeysTensorName);

        model.ApplyRngConfig(new RngConfig { MasterSeed = 8 });   // no-op, no throw

        var ex = Assert.Throws<System.InvalidOperationException>(
            () => model.ApplyRngConfig(new RngConfig { MasterSeed = 8 }
                .Override(RngCollection.Runtime, [1], 1UL)));
        Assert.Contains("matches no runtime stream", ex.Message);
    }

    [Fact]
    public void TestBindingRequiresRealizedStreams()
    {
        // The concreteness contract at bind: an id-bearing feed without its key derivation
        // chain (a graph that never went through ToConcreteArchitecture) fails loudly.
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

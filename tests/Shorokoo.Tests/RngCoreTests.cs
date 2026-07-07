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
}

/// <summary>
/// The compact RNG key vector — the single parameter-like tensor a model carries to make its
/// randomness state self-contained. Three tiers: [master] when everything derives from the
/// master seed; [master, initMaster, runMaster] when a sub-master was set explicitly; the
/// 3 masters + the full per-stream expansion when any per-stream override exists. ONNX-prep
/// reconstruction must reproduce every stream key bit-exactly in all three tiers.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngKeyVectorTests
{
    private static readonly int[][] InitPaths = [[1, 1], [2, 1], [2, 2]];
    private static readonly int[][] RunPaths = [[3], [4, 0, 1], [4, 1, 1]];

    private static void AssertReconstructs(RngConfig cfg, long[] vec)
    {
        var (init, run) = RngConfig.ReconstructKeys(vec, InitPaths, RunPaths);
        for (int i = 0; i < InitPaths.Length; i++)
            Assert.Equal(cfg.FoldInitKey(InitPaths[i]), init[i]);
        for (int i = 0; i < RunPaths.Length; i++)
            Assert.Equal(cfg.FoldRunKey(RunPaths[i]), run[i]);
    }

    [Fact]
    public void TestTier1MasterOnly()
    {
        var cfg = new RngConfig { MasterSeed = 42 };
        var vec = cfg.BuildKeyVector(InitPaths, RunPaths);
        Assert.Single(vec);
        Assert.Equal(42L, vec[0]);
        AssertReconstructs(cfg, vec);
    }

    [Fact]
    public void TestTier2SubMasters()
    {
        // Explicit run sub-master: feeds re-seed, init stays derived from the master.
        var cfg = new RngConfig { MasterSeed = 42, RunMasterSeed = 777 };
        var vec = cfg.BuildKeyVector(InitPaths, RunPaths);
        Assert.Equal(3, vec.Length);
        AssertReconstructs(cfg, vec);

        // The sub-master genuinely changed the runtime keys and left init untouched.
        var baseline = new RngConfig { MasterSeed = 42 };
        Assert.NotEqual(baseline.FoldRunKey(RunPaths[0]), cfg.FoldRunKey(RunPaths[0]));
        Assert.Equal(baseline.FoldInitKey(InitPaths[0]), cfg.FoldInitKey(InitPaths[0]));

        var cfgInit = new RngConfig { MasterSeed = 42, InitMasterSeed = 888 };
        AssertReconstructs(cfgInit, cfgInit.BuildKeyVector(InitPaths, RunPaths));
    }

    [Fact]
    public void TestTier3FullExpansion()
    {
        // A single per-stream override (one loop iteration's feed) forces the full expansion,
        // and reconstruction reads the stored keys — override included — bit-exactly.
        var cfg = new RngConfig { MasterSeed = 42, RunMasterSeed = 777 };
        cfg.Override(RngCollection.Runtime, [4, 1, 1], seed: 424242UL);
        var vec = cfg.BuildKeyVector(InitPaths, RunPaths);
        Assert.Equal(3 + InitPaths.Length + RunPaths.Length, vec.Length);
        AssertReconstructs(cfg, vec);

        // Sibling streams keep their derived keys; only the overridden one deviates.
        var noOverride = new RngConfig { MasterSeed = 42, RunMasterSeed = 777 };
        Assert.Equal(noOverride.FoldRunKey([4, 0, 1]), cfg.FoldRunKey([4, 0, 1]));
        Assert.NotEqual(noOverride.FoldRunKey([4, 1, 1]), cfg.FoldRunKey([4, 1, 1]));
    }
}

/// <summary>
/// The key-vector transport: ApplyRngConfig injects the compact vector as a graph-carried
/// tensor (lowered to a plain CONSTANT at ONNX prep); it survives save/load, and
/// ApplyRngKeyVector reconstructs the exact feed stamps from it — no config object needed.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngKeyVectorTransportTests
{
    [Fact]
    public void TestKeyVectorCarrierRoundTripsAndRebindsFeeds()
    {
        var g = (FastComputationGraph)typeof(RngRuntimeLoopFeed)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var x = TensorData([8L], new float[8]);
        var steps = TensorData(System.Array.Empty<long>(), 2L);
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([x, steps]));

        var cfg = new RngConfig { MasterSeed = 11 };
        cfg.Override(RngCollection.Runtime, [1, 1, 1], seed: 424242UL);   // tier 3
        arch.ApplyRngConfig(cfg);

        var feed = arch.Nodes.Single(n => n.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM);
        var stampedTable = feed.Attributes.GetLongsVal(OnnxOpAttributeNames.ShrkAttrRngKeyTable);
        var stampedKey = feed.Attributes.GetLongsVal(OnnxOpAttributeNames.ShrkAttrRngExplicitKey);
        Assert.NotNull(stampedTable);
        Assert.True(stampedTable!.Length >= 4);

        // The carrier survives a save/load round trip: it rides the ONNX file as the
        // reserved-name initializer (with per-initializer metadata) and the loader rebuilds
        // the carrier node from it.
        var data = CompressedFormatUtils.SaveFastGraphToBinary(arch, compressed: true);
        var loaded = CompressedFormatUtils.LoadFastGraphFromBinary(data, isCompressed: true);
        var carried = loaded.TryGetRngKeyVector();
        Assert.NotNull(carried);
        Assert.True(carried!.Value.keyVector.Length > 3);                 // tier-3 expansion
        Assert.Contains("Threefry", carried.Value.algorithm);
        Assert.Equal(arch.TryGetRngKeyVector()!.Value.keyVector, carried.Value.keyVector);
        Assert.Equal(arch.TryGetRngKeyVector()!.Value.initStreamCount, carried.Value.initStreamCount);

        // Corrupt the feed's stamps, then reconstruct purely from the carried vector.
        feed.Attributes = feed.Attributes.SetAttributes(
            (OnnxOpAttributeNames.ShrkAttrRngKeyTable, (long[])[]),
            (OnnxOpAttributeNames.ShrkAttrRngExplicitKey, (long[])[0L, 0L]));
        arch.ApplyRngKeyVector();
        Assert.Equal(stampedTable, feed.Attributes.GetLongsVal(OnnxOpAttributeNames.ShrkAttrRngKeyTable));
        Assert.Equal(stampedKey, feed.Attributes.GetLongsVal(OnnxOpAttributeNames.ShrkAttrRngExplicitKey));
    }

    [Fact]
    public void TestTier1CarrierRebindsPrefixStamps()
    {
        var g = (FastComputationGraph)typeof(RngRuntimeLoopFeed)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var x = TensorData([8L], new float[8]);
        var steps = TensorData(System.Array.Empty<long>(), 2L);
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([x, steps]));

        arch.ApplyRngConfig(new RngConfig { MasterSeed = 11 });
        var carried = arch.TryGetRngKeyVector();
        Assert.Equal([11L], carried!.Value.keyVector);                    // tier 1: master only

        var feed = arch.Nodes.Single(n => n.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM);
        var stamped = feed.Attributes.GetLongsVal(OnnxOpAttributeNames.ShrkAttrRngExplicitKey);
        feed.Attributes = feed.Attributes.SetAttributes(
            (OnnxOpAttributeNames.ShrkAttrRngExplicitKey, (long[])[0L, 0L]));
        arch.ApplyRngKeyVector();
        Assert.Equal(stamped, feed.Attributes.GetLongsVal(OnnxOpAttributeNames.ShrkAttrRngExplicitKey));
    }
}

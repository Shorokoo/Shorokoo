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

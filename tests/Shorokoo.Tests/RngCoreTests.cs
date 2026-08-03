using System;
using System.Collections.Immutable;
using System.Linq;
using Shorokoo.Core.Rng;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.Processors.Fast;
using Shorokoo.Graph;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage tests for the host RNG core (<see cref="Threefry2x32"/>, the bit generator
/// behind the key folds) and the <see cref="RngConfig"/> key-derivation surface. The
/// Threefry tests pin the implementation against the Random123 known-answer vectors; the
/// rest assert the properties the RNG design relies on — determinism, name-derived
/// independence, and override isolation.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngCoreTests
{
    /// <summary>
    /// The framework-injected RNG execution counter is identified <b>structurally</b> — its leaf
    /// parameter part is named <c>RngExecutionCounter</c> under the <c>TrainableParam</c> category
    /// — not by a substring scan of the identifier string. So the false positives the old
    /// <c>name.Contains("RngExecutionCounter")</c> match produced (the counter name appearing as a
    /// module path segment or a mere name substring) are correctly rejected.
    /// </summary>
    [Fact]
    public void TestExecutionCounterIsIdentifiedStructurallyNotBySubstring()
    {
        // The real counter (at any dynamically-assigned slot) is recognized.
        var counter = ModelParamIdentifierTemplate.LocalTrainableParam(
            new ModelId(3), FastInjectRngDrawCounter.CounterName, 0, ImmutableArray<int>.Empty);
        Assert.True(FastInjectRngDrawCounter.IsExecutionCounter(counter));

        // A user parameter whose leaf name only CONTAINS the counter name is not the counter.
        var lookalike = ModelParamIdentifierTemplate.LocalTrainableParam(
            new ModelId(4), FastInjectRngDrawCounter.CounterName + "Stat", 0, ImmutableArray<int>.Empty);
        Assert.False(FastInjectRngDrawCounter.IsExecutionCounter(lookalike));

        // The strongest old false positive: an ordinary "weight" parameter nested in a user
        // MODULE that happens to be named RngExecutionCounter. Its path string contains the
        // counter name (so the old Contains match fired), but its leaf part is "weight".
        var moduleNamedLikeCounter = ModelParamIdentifierTemplate.LocalModule(
            new ModelId(4), FastInjectRngDrawCounter.CounterName, 0, ImmutableArray<int>.Empty);
        var nestedWeight = ModelParamIdentifierTemplate.LocalTrainableParam(
            new ModelId(0), "weight", 0, ImmutableArray<int>.Empty);
        var nested = new ModelParamIdentifierTemplate(moduleNamedLikeCounter, nestedWeight);
        Assert.Contains(FastInjectRngDrawCounter.CounterName, nested.ToString());   // old Contains would fire
        Assert.False(FastInjectRngDrawCounter.IsExecutionCounter(nested));          // structural match does not

        // An ordinary weight is not the counter.
        var weight = ModelParamIdentifierTemplate.LocalTrainableParam(
            new ModelId(1), "weight", 0, ImmutableArray<int>.Empty);
        Assert.False(FastInjectRngDrawCounter.IsExecutionCounter(weight));

        // A non-TrainableParam parameter whose leaf is exactly the counter name is not the
        // counter either — the category clause is load-bearing.
        var stateNamedLikeCounter = ModelParamIdentifierTemplate.LocalStateParam(
            new ModelId(5), FastInjectRngDrawCounter.CounterName, 0, ImmutableArray<int>.Empty);
        Assert.False(FastInjectRngDrawCounter.IsExecutionCounter(stateNamedLikeCounter));

        // A null identifier is not the counter.
        Assert.False(FastInjectRngDrawCounter.IsExecutionCounter((ModelParamIdentifierTemplate?)null));
    }

    /// <summary>
    /// The execution counter's materialization fallback value is an <c>int64[1]</c> zero — it must
    /// match what <c>CounterInit</c> produces, or a safetensors round-trip (which omits the counter
    /// and refills it from this value) would bind a wrong-shaped or wrong-valued counter.
    /// </summary>
    [Fact]
    public void TestExecutionCounterInitialValueIsInt64ScalarZero()
    {
        var v = FastInjectRngDrawCounter.ExecutionCounterInitialValue();
        Assert.Equal((long[])[1L], v.Shape.Dims.ToArray());
        // As<int64> also asserts the dtype (it throws on a non-int64 tensor).
        Assert.Equal((long[])[0L], v.As<int64>().AccessMemory().ToArray());
    }

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
    public void TestConfigDefaultIsDeterministicMasterSeedZero()
    {
        Assert.Equal(0ul, RngConfig.Default.MasterSeed);
        Assert.Equal(RngAlgorithm.Threefry2x32, RngConfig.Default.Algorithm);
    }

    [Fact]
    public void TestKeyDerivationIsPathDerivedAndStable()
    {
        var cfg = new RngConfig { MasterSeed = 20260702 };
        var k1 = RngTestOracle.InitKey(cfg, [3, 1, 1]);
        var k2 = RngTestOracle.InitKey(cfg, [3, 1, 1]);
        var k3 = RngTestOracle.InitKey(cfg, [3, 1, 2]);
        Assert.Equal(k1, k2);                                 // stable for a path
        Assert.NotEqual(k1, k3);                              // sibling paths differ
        // Same path in the runtime collection is a different stream (distinct sub-master).
        Assert.NotEqual(k1, RngTestOracle.RunKey(cfg, [3, 1, 1]));
    }

    [Fact]
    public void TestMasterSeedChangeRerandomizesEveryStream()
    {
        var a = new RngConfig { MasterSeed = 1 };
        var b = new RngConfig { MasterSeed = 2 };
        Assert.NotEqual(RngTestOracle.InitKey(a, [1]), RngTestOracle.InitKey(b, [1]));
        Assert.NotEqual(RngTestOracle.RunKey(a, [1]), RngTestOracle.RunKey(b, [1]));
    }

    [Fact]
    public void TestOverrideIsolatesASingleStream()
    {
        var baseCfg = new RngConfig { MasterSeed = 7 };
        var cfg = new RngConfig { MasterSeed = 7 }
            .Override(RngCollection.Params, [1, 1], seed: 1234);

        // The overridden stream changes; siblings, sub-paths, and the runtime collection
        // keep their derived keys (matching is exact and per-collection).
        Assert.NotEqual(RngTestOracle.InitKey(baseCfg, [1, 1]), RngTestOracle.InitKey(cfg, [1, 1]));
        Assert.Equal(RngTestOracle.InitKey(baseCfg, [1, 2]), RngTestOracle.InitKey(cfg, [1, 2]));
        Assert.Equal(RngTestOracle.InitKey(baseCfg, [1, 1, 1]), RngTestOracle.InitKey(cfg, [1, 1, 1]));
        Assert.Equal(RngTestOracle.RunKey(baseCfg, [1, 1]), RngTestOracle.RunKey(cfg, [1, 1]));
        Assert.True(cfg.HasOverride(RngCollection.Params, [1, 1]));
        Assert.False(cfg.HasOverride(RngCollection.Params, [1, 2]));

        // The override replaces the fully folded key, so it survives a master-seed change.
        var otherMaster = new RngConfig { MasterSeed = 8 }
            .Override(RngCollection.Params, [1, 1], seed: 1234);
        Assert.Equal(RngTestOracle.InitKey(cfg, [1, 1]), RngTestOracle.InitKey(otherMaster, [1, 1]));
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
/// reserved ModelId [0] (see <see cref="RngRuntimeIdentity"/>): a scheme-version + algorithm-id
/// header, the runtime master key, and canonically sorted per-stream override records at fixed
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
            Assert.Equal(RngTestOracle.RunKey(cfg, p), RngTestOracle.RunKey(decoded, p));
    }

    [Fact]
    public void TestPreV2IdentityIsRejectedWithAnExplanation()
    {
        // A carrier written before the identity became uint64 must fail with an explanation, not
        // a bare InvalidCastException. That every decode site routes through here is enforced
        // structurally (one guarded read, no other As<uint64>() on the identity) rather than by
        // this test — see TestNoUnguardedIdentityReadSurvives.
        var legacy = TensorData(DType.Int64, [3L], 0L, 42L, 0L);
        var ex = Assert.Throws<InvalidOperationException>(() => RngRuntimeIdentity.ReadRngSeedData(legacy));
        Assert.Contains("Int64", ex.Message);
        Assert.Contains("v2", ex.Message);

        // A correctly-typed vector from an older SCHEME is the case the element type cannot catch,
        // which is exactly why the version rides in the vector.
        var olderScheme = RngRuntimeIdentity.Build(new RngConfig { MasterSeed = 7 });
        olderScheme[RngRuntimeIdentity.SchemeVersionIndex] = RngRuntimeIdentity.SchemeVersion - 1;
        var ex2 = Assert.Throws<InvalidOperationException>(() => RngRuntimeIdentity.Decode(olderScheme));
        Assert.Contains("scheme version", ex2.Message);
    }

    [Fact]
    public void TestAlgorithmIdCannotStandInForTheSchemeVersion()
    {
        // Why the version element exists: the algorithm id tracks the RngAlgorithm enum, so it
        // varies with the CONFIGURED algorithm and not with the scheme. Two configs differing only
        // in algorithm get different ids but the same scheme version — so an id can never signal
        // "this build's draws differ from the one that wrote this". Only the version can, and
        // keeping it in step with draw-value changes is a convention no test can enforce.
        var a = RngRuntimeIdentity.Build(new RngConfig { MasterSeed = 1 });
        var b = RngRuntimeIdentity.Build(new RngConfig { MasterSeed = 1, Algorithm = RngAlgorithm.Threefry2x32Rounds13 });
        Assert.Equal(RngRuntimeIdentity.SchemeVersion, a[RngRuntimeIdentity.SchemeVersionIndex]);
        Assert.Equal(RngRuntimeIdentity.SchemeVersion, b[RngRuntimeIdentity.SchemeVersionIndex]);
        Assert.NotEqual(a[RngRuntimeIdentity.AlgorithmIdIndex], b[RngRuntimeIdentity.AlgorithmIdIndex]);
    }

    [Fact]
    public void TestHeaderOnlyIdentity()
    {
        var cfg = new RngConfig { MasterSeed = 42 };
        var vec = RngRuntimeIdentity.Build(cfg);
        // Header only: [schemeVersion, algId, runKey, 0 overrides].
        Assert.Equal(RngRuntimeIdentity.HeaderLength, vec.Length);
        Assert.Equal(0UL, vec[RngRuntimeIdentity.AlgorithmIdIndex]);
        Assert.Equal(cfg.RunMasterKey, vec[RngRuntimeIdentity.RunKeyIndex]);
        AssertRoundTrips(cfg);

        // The algorithm id header switches with the configured algorithm.
        var cfg13 = new RngConfig { MasterSeed = 42, Algorithm = RngAlgorithm.Threefry2x32Rounds13 };
        Assert.Equal(1UL, RngRuntimeIdentity.Build(cfg13)[RngRuntimeIdentity.AlgorithmIdIndex]);
        AssertRoundTrips(cfg13);

        // An explicit run sub-master re-seeds the runtime tier and rides the same header.
        var subMaster = new RngConfig { MasterSeed = 42, RunMasterSeed = 777 };
        Assert.NotEqual(RngTestOracle.RunKey(cfg, Paths[0]), RngTestOracle.RunKey(subMaster, Paths[0]));
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
        // Header + one record: length 3, its 3 path elements, 1 key.
        Assert.Equal(RngRuntimeIdentity.HeaderLength + 1 + 3 + 1, vec.Length);
        Assert.Equal(1UL, vec[RngRuntimeIdentity.HeaderLength - 1]);   // record count

        var decoded = RngRuntimeIdentity.Decode(vec);
        var rec = Assert.Single(decoded.Overrides);
        Assert.Equal((int[])[4, 1, 1], rec.Path);
        // The record replaces the fully folded key: it IS the override seed, and it sits at
        // the record's fixed key offset in the vector.
        Assert.Equal(424242UL, rec.Key);
        Assert.Equal(rec.Key, vec[rec.KeyOffset]);

        // Derivation round-trips: the overridden stream deviates, siblings stay derived.
        AssertRoundTrips(cfg);
        var noOverride = new RngConfig { MasterSeed = 42, RunMasterSeed = 777 };
        Assert.Equal(RngTestOracle.RunKey(noOverride, [4, 0, 1]), RngTestOracle.RunKey(decoded, [4, 0, 1]));
        Assert.NotEqual(RngTestOracle.RunKey(noOverride, [4, 1, 1]), RngTestOracle.RunKey(decoded, [4, 1, 1]));
    }

    [Fact]
    public void TestMalformedIdentityFailsLoudly()
    {
        // Corrupt identities must throw, never silently fall back to a different derivation.
        // These carry the current scheme version so they exercise the STRUCTURAL checks; a wrong
        // scheme version is a different condition, covered by TestPreV2IdentityIsRejectedAtEveryDecodeSite.
        const ulong v = RngRuntimeIdentity.SchemeVersion;
        Assert.ThrowsAny<ArgumentException>(() => RngRuntimeIdentity.Decode([]));
        Assert.ThrowsAny<ArgumentException>(() => RngRuntimeIdentity.Decode([v, 0UL, 42UL]));   // shorter than the header
        // Truncated override record (claims one record, supplies nothing).
        Assert.ThrowsAny<ArgumentException>(() => RngRuntimeIdentity.Decode([v, 0UL, 42UL, 1UL]));
        // Trailing garbage after the declared records.
        Assert.ThrowsAny<ArgumentException>(() => RngRuntimeIdentity.Decode([v, 0UL, 42UL, 0UL, 99UL]));
        // A record claiming a huge path length. The bound must be computed before narrowing to
        // int, or `i + pathLen + 1` wraps negative, passes the check, and allocates the claim —
        // turning a corrupt model file into an OutOfMemoryException instead of this error.
        Assert.ThrowsAny<ArgumentException>(() => RngRuntimeIdentity.Decode([v, 0UL, 42UL, 1UL, int.MaxValue]));
        Assert.ThrowsAny<ArgumentException>(() => RngRuntimeIdentity.Decode([v, 0UL, 42UL, 1UL, ulong.MaxValue]));
    }
}

/// <summary>
/// The identity transport: WithRngConfig writes the runtime identity into the ordinary
/// <c>RngSeed</c> parameter at reserved ModelId [0] — serialized as a plain initializer with
/// no reserved-name handling; it survives save/load bit-exactly and without duplication, the
/// loaded model's randomness is reproducible with no config object, and re-binding a LOADED
/// model is a parameter write that re-keys every draw (the re-bind-after-load pin).
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngSeedTransportTests
{
    private static int RngSeedNodeCount(ComputationGraph graph)
        => graph.ToInternal().Nodes.Count(n =>
            n.IdentifierTemplate == Shorokoo.Core.Nodes.Processors.Fast
                .FastWireRngKeyDerivation.RngSeedIdentifierTemplate);

    [Fact]
    public void TestRngSeedIdentityRoundTripsWithoutDuplication()
    {
        var g = (ComputationGraph)typeof(RngRuntimeLoopFeed)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var x = TensorData([8L], new float[8]);
        var steps = TensorData(System.Array.Empty<long>(), 2L);
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([x, steps]));

        var cfg = new RngConfig { MasterSeed = 11 };
        cfg = cfg.Override(RngCollection.Runtime, [1, 1, 1], seed: 424242UL);
        arch = arch.WithRngConfig(cfg);

        // Exactly one RngSeed parameter, holding the encoded identity.
        Assert.Equal(1, RngSeedNodeCount(arch));
        Assert.Equal(RngRuntimeIdentity.Build(cfg), arch.TryGetRngSeed());

        var data = CompressedFormatUtils.SaveFastGraphToBinary(arch, compressed: true);
        var loaded = CompressedFormatUtils.LoadFastGraphFromBinary(data);
        var carried = loaded.TryGetRngSeed();
        Assert.NotNull(carried);
        Assert.Equal(arch.TryGetRngSeed(), carried);

        // The decoded identity reproduces the config's runtime derivation, override included
        // — the loaded model needs no config object.
        var decoded = RngRuntimeIdentity.Decode(carried!);
        Assert.Equal(RngRuntimeIdentity.AlgorithmIdOf(RngAlgorithm.Threefry2x32), decoded.AlgorithmId);
        Assert.Equal(RngTestOracle.RunKey(cfg, [1, 1, 1]), RngTestOracle.RunKey(decoded, [1, 1, 1]));
        Assert.Equal(RngTestOracle.RunKey(cfg, [1, 0, 1]), RngTestOracle.RunKey(decoded, [1, 0, 1]));

        // Exactly one RngSeed parameter after each save/load cycle — an ordinary initializer
        // never accumulates duplicates.
        Assert.Equal(1, RngSeedNodeCount(loaded));
        var loaded2 = CompressedFormatUtils.LoadFastGraphFromBinary(
            CompressedFormatUtils.SaveFastGraphToBinary(loaded, compressed: true));
        Assert.Equal(1, RngSeedNodeCount(loaded2));
        Assert.Equal(carried, loaded2.TryGetRngSeed());
    }

    [Fact]
    public void TestNonDefaultAlgorithmSurvivesSaveLoadByIdAndByBehavior()
    {
        // The algorithm choice rides the file in TWO forms — the RngSeedData's
        // algorithm id (trusted by no-config parameter initialization and the lowering) and
        // the baked tagged draw functions (what the feeds actually execute) — and they can in
        // principle disagree. Bind the NON-default algorithm, round-trip the concrete model,
        // and pin both: the id decodes exactly, and the loaded model still draws 13-round
        // values (equal to its own pre-save draws, different from a default-algorithm model
        // under the same seed).
        var g = (ComputationGraph)typeof(RngRuntimeLoopFeed)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var x = TensorData([8L], new float[8]);
        var steps = TensorData(System.Array.Empty<long>(), 2L);

        ComputationGraph Concrete(RngConfig cfg) =>
            g.ToConcreteArchitecture(g.FromOrderedInputs([x, steps])).ToConcreteModel(cfg);
        float[] Run(ComputationGraph m) => ComputeContext.Default.Execute(m, x, steps)[0]
            .ToTensorData().As<float32>().AccessMemory().ToArray();

        var m13 = Concrete(new RngConfig { MasterSeed = 11, Algorithm = RngAlgorithm.Threefry2x32Rounds13 });
        var before = Run(m13);
        var draws20 = Run(Concrete(new RngConfig { MasterSeed = 11 }));   // same seed, default rounds
        Assert.NotEqual(before, draws20);   // guard: the two algorithms genuinely differ here

        var loaded = CompressedFormatUtils.LoadFastGraphFromBinary(
            CompressedFormatUtils.SaveFastGraphToBinary(m13, compressed: true));

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
        var g = (ComputationGraph)typeof(RngRuntimeLoopFeed)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var x = TensorData([8L], new float[8]);
        var steps = TensorData(System.Array.Empty<long>(), 2L);
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([x, steps]));

        arch = arch.WithRngConfig(new RngConfig { MasterSeed = 11 });
        Assert.Equal(RngRuntimeIdentity.Build(new RngConfig { MasterSeed = 11 }), arch.TryGetRngSeed());

        arch = arch.WithRngConfig(new RngConfig { MasterSeed = 12 });
        Assert.Equal(RngRuntimeIdentity.Build(new RngConfig { MasterSeed = 12 }), arch.TryGetRngSeed());
        Assert.Equal(1, RngSeedNodeCount(arch));
    }

    [Fact]
    public void TestRebindAfterSaveLoadRekeysEveryDraw()
    {
        // THE re-bind-after-load pin: bind seed A -> save -> load -> WithRngConfig(B) ->
        // every draw changes AND matches a model bound to B directly. With the identity as an
        // ordinary parameter and keys derived in-graph from it, re-binding a loaded model is
        // a parameter write that re-keys every draw by construction — the divergence class
        // where a loaded model's recorded identity updated while the draws kept the old seed
        // is structurally impossible.
        var g = (ComputationGraph)typeof(RngRuntimeLoopFeed)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var x = TensorData([8L], new float[8]);
        var steps = TensorData(System.Array.Empty<long>(), 2L);

        ComputationGraph Concrete(RngConfig cfg) =>
            g.ToConcreteArchitecture(g.FromOrderedInputs([x, steps])).ToConcreteModel(cfg);
        float[] Run(ComputationGraph m) => ComputeContext.Default.Execute(m, x, steps)[0]
            .ToTensorData().As<float32>().AccessMemory().ToArray();

        var seedA = new RngConfig { MasterSeed = 11 };
        var seedB = new RngConfig { MasterSeed = 12 };

        var modelA = Concrete(seedA);
        var drawsA = Run(modelA);

        var loaded = CompressedFormatUtils.LoadFastGraphFromBinary(
            CompressedFormatUtils.SaveFastGraphToBinary(modelA, compressed: true));
        Assert.Equal(drawsA, Run(loaded));            // load-and-run reproduces seed A

        loaded = loaded.WithRngConfig(seedB);         // a parameter write on the loaded graph
        var rekeyed = Run(loaded);
        Assert.NotEqual(drawsA, rekeyed);             // every draw changed
        Assert.Equal(Run(Concrete(seedB)), rekeyed);  // and matches a direct seed-B model

        // Round-trip again after the re-bind: the new identity is what persists.
        var reloaded = CompressedFormatUtils.LoadFastGraphFromBinary(
            CompressedFormatUtils.SaveFastGraphToBinary(loaded, compressed: true));
        Assert.Equal(rekeyed, Run(reloaded));
    }

    [Fact]
    public void TestRebindOnLoadedModelSupportsOverrideSetAndAlgorithmChanges()
    {
        // Since .srk persistence stopped lowering the SHRK_RANDOM_* feeds (issue #59 —
        // saved graphs keep the feed ops and their key-derivation chains verbatim), a
        // loaded model is no longer "baked": re-binding may change not only seed VALUES
        // but also the override SET (the routing is re-wired) and the draw algorithm —
        // and the result must be bit-identical to a model built under that config
        // directly. (Files written by pre-#59 versions still carry lowered draw-function
        // calls and keep the fail-loud structural-rebind guard in FastBindRngConfig.)
        var g = (ComputationGraph)typeof(RngRuntimeLoopFeed)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var x = TensorData([8L], new float[8]);
        var steps = TensorData(System.Array.Empty<long>(), 2L);

        ComputationGraph Concrete(RngConfig cfg) =>
            g.ToConcreteArchitecture(g.FromOrderedInputs([x, steps])).ToConcreteModel(cfg);
        float[] Run(ComputationGraph m) => ComputeContext.Default.Execute(m, x, steps)[0]
            .ToTensorData().As<float32>().AccessMemory().ToArray();

        var loaded = CompressedFormatUtils.LoadFastGraphFromBinary(
            CompressedFormatUtils.SaveFastGraphToBinary(
                Concrete(new RngConfig { MasterSeed = 11 }), compressed: true));

        var withOverride = new RngConfig { MasterSeed = 11 }
            .Override(RngCollection.Runtime, [1, 1, 1], 42UL);
        Assert.Equal(Run(Concrete(withOverride)), Run(loaded.WithRngConfig(withOverride)));

        var otherAlg = new RngConfig { MasterSeed = 11, Algorithm = RngAlgorithm.Threefry2x32Rounds13 };
        Assert.Equal(Run(Concrete(otherAlg)), Run(loaded.WithRngConfig(otherAlg)));
        Assert.NotEqual(Run(Concrete(new RngConfig { MasterSeed = 11 })),
            Run(loaded.WithRngConfig(otherAlg)));
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
        var g = (ComputationGraph)typeof(RtLoweredUniform)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var input = TensorData([4L, 4L], new float[16]);
        var model = g.ToConcreteArchitecture(g.FromOrderedInputs([input]))
            .ToConcreteModel(new RngConfig { MasterSeed = 1 });

        // Strip the new representation down to the legacy shape (mutating node surgery, so
        // work on a mutable copy of the loaded graph).
        var legacy = CompressedFormatUtils.LoadFastGraphFromBinary(
            CompressedFormatUtils.SaveFastGraphToBinary(model, compressed: true)).ToInternal();
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
        var g = (ComputationGraph)typeof(RngInitTwoLinears)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var sample = TensorData([4L, 4L], Enumerable.Repeat(1f, 16).ToArray());
        var model = g.ToConcreteArchitecture(g.FromOrderedInputs([sample]))
            .ToConcreteModel(new RngConfig { MasterSeed = 7 });

        Assert.Equal(0, RngSeedNodeCount(model));
        Assert.Null(model.TryGetRngSeed());
        var loaded = CompressedFormatUtils.LoadFastGraphFromBinary(
            CompressedFormatUtils.SaveFastGraphToBinary(model, compressed: true));
        Assert.Equal(0, RngSeedNodeCount(loaded));
        Assert.DoesNotContain(loaded.ToInternal().Nodes, n =>
            n.FriendlyName == OnnxOpAttributeNames.ShrkRngKeysTensorName);

        model = model.WithRngConfig(new RngConfig { MasterSeed = 8 });   // no-op, no throw

        var ex = Assert.Throws<System.InvalidOperationException>(
            () => model.WithRngConfig(new RngConfig { MasterSeed = 8 }
                .Override(RngCollection.Runtime, [1], 1UL)));
        Assert.Contains("matches no runtime stream", ex.Message);
    }

    [Fact]
    public void TestBindingRequiresRealizedStreams()
    {
        // The concreteness contract at bind: an id-bearing feed without its key derivation
        // chain (a graph that never went through ToConcreteArchitecture) fails loudly.
        var draw = RandomUniform(Vector(4L), 0f, 1f);
        var graph = new InternalComputationGraph([], [draw]);
        var feed = graph.Nodes.Single(n => n.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM);
        feed.Attributes = feed.Attributes.SetAttributes(
            (OnnxOpAttributeNames.ShrkAttrLocalModelId, (long[])[1]));

        var ex = Assert.Throws<System.InvalidOperationException>(
            () => graph.ApplyRngConfig(new RngConfig { MasterSeed = 1 }));
        Assert.Contains("no realized stream ids", ex.Message);
    }
}

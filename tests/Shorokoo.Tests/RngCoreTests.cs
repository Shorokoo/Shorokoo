using System;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Shorokoo.Core.Rng;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.Processors.Fast;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// The host RNG core: the <see cref="Threefry2x32"/> bit generator (pinned to the Random123
/// known-answer vectors) and the <see cref="RngConfig"/> key-derivation surface.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngCoreTests
{
    [Fact]
    public void TestExecutionCounterIsIdentifiedStructurallyNotBySubstringAndInitializesToInt64Zero()
    {
        var counter = ModelParamIdentifierTemplate.LocalTrainableParam(
            new ModelId(3), FastInjectRngDrawCounter.CounterName, 0, ImmutableArray<int>.Empty);
        var lookalike = ModelParamIdentifierTemplate.LocalTrainableParam(
            new ModelId(4), FastInjectRngDrawCounter.CounterName + "Stat", 0, ImmutableArray<int>.Empty);
        var nested = new ModelParamIdentifierTemplate(
            ModelParamIdentifierTemplate.LocalModule(
                new ModelId(4), FastInjectRngDrawCounter.CounterName, 0, ImmutableArray<int>.Empty),
            ModelParamIdentifierTemplate.LocalTrainableParam(
                new ModelId(0), "weight", 0, ImmutableArray<int>.Empty));
        var weight = ModelParamIdentifierTemplate.LocalTrainableParam(
            new ModelId(1), "weight", 0, ImmutableArray<int>.Empty);
        var stateNamedLikeCounter = ModelParamIdentifierTemplate.LocalStateParam(
            new ModelId(5), FastInjectRngDrawCounter.CounterName, 0, ImmutableArray<int>.Empty);

        Assert.True(FastInjectRngDrawCounter.IsExecutionCounter(counter));
        Assert.False(FastInjectRngDrawCounter.IsExecutionCounter(lookalike));
        Assert.Contains(FastInjectRngDrawCounter.CounterName, nested.ToString());
        Assert.False(FastInjectRngDrawCounter.IsExecutionCounter(nested));
        Assert.False(FastInjectRngDrawCounter.IsExecutionCounter(weight));
        Assert.False(FastInjectRngDrawCounter.IsExecutionCounter(stateNamedLikeCounter));
        Assert.False(FastInjectRngDrawCounter.IsExecutionCounter((ModelParamIdentifierTemplate?)null));

        var v = FastInjectRngDrawCounter.ExecutionCounterInitialValue();
        Assert.Equal((long[])[1L], v.Shape.Dims.ToArray());
        Assert.Equal((long[])[0L], v.As<int64>().AccessMemory().ToArray());
    }

    // Random123 known-answer vectors (tests/kat_vectors in DEShawResearch/random123) for
    // threefry2x32 at 20 and 13 rounds: counter, key, rounds -> output.
    [Fact]
    public void TestThreefry2x32MatchesKnownAnswerVectorsAtBothRoundCountsAndIsAPureFunction()
    {
        (uint c0, uint c1, uint k0, uint k1, int rounds, uint e0, uint e1)[] kat =
        [
            (0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, Threefry2x32.Rounds, 0x6b200159u, 0x99ba4efeu),
            (0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, Threefry2x32.Rounds, 0x1cb996fcu, 0xbb002be7u),
            (0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u, Threefry2x32.Rounds, 0xc4923a9cu, 0x483df7a0u),
            (0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, Threefry2x32.Rounds13, 0x9d1c5ec6u, 0x8bd50731u),
            (0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, Threefry2x32.Rounds13, 0xfd36d048u, 0x2d17272cu),
            (0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u, Threefry2x32.Rounds13, 0xba3e4725u, 0xf27d669eu),
        ];
        foreach (var (c0, c1, k0, k1, rounds, e0, e1) in kat)
        {
            Assert.Equal((e0, e1), Threefry2x32.Bijection(c0, c1, k0, k1, rounds));
            if (rounds == Threefry2x32.Rounds13)
                Assert.NotEqual((e0, e1), Threefry2x32.Bijection(c0, c1, k0, k1, Threefry2x32.Rounds));
        }

        var a = Threefry2x32.Bijection(7, 42, 123, 456);
        Assert.Equal(a, Threefry2x32.Bijection(7, 42, 123, 456));
        Assert.NotEqual(a, Threefry2x32.Bijection(8, 42, 123, 456));
        Assert.NotEqual(a, Threefry2x32.Bijection(7, 42, 124, 456));
    }

    [Fact]
    public void TestDefaultConfigAndPathDerivedKeysAreStableSiblingDistinctAndMasterSeeded()
    {
        Assert.Equal(0ul, RngConfig.Default.MasterSeed);
        Assert.Equal(RngAlgorithm.Threefry2x32, RngConfig.Default.Algorithm);

        var cfg = new RngConfig { MasterSeed = 20260702 };
        Assert.Equal(RngTestOracle.InitKey(cfg, [3, 1, 1]), RngTestOracle.InitKey(cfg, [3, 1, 1]));
        Assert.NotEqual(RngTestOracle.InitKey(cfg, [3, 1, 1]), RngTestOracle.InitKey(cfg, [3, 1, 2]));
        Assert.NotEqual(RngTestOracle.InitKey(cfg, [3, 1, 1]), RngTestOracle.RunKey(cfg, [3, 1, 1]));

        var a = new RngConfig { MasterSeed = 1 };
        var b = new RngConfig { MasterSeed = 2 };
        Assert.NotEqual(RngTestOracle.InitKey(a, [1]), RngTestOracle.InitKey(b, [1]));
        Assert.NotEqual(RngTestOracle.RunKey(a, [1]), RngTestOracle.RunKey(b, [1]));
    }

    [Fact]
    public void TestOverrideIsolatesOneStreamAndReturnsACopyLeavingTheReceiverUntouched()
    {
        var baseCfg = new RngConfig { MasterSeed = 7 };
        var cfg = new RngConfig { MasterSeed = 7 }.Override(RngCollection.Params, [1, 1], seed: 1234);

        Assert.NotEqual(RngTestOracle.InitKey(baseCfg, [1, 1]), RngTestOracle.InitKey(cfg, [1, 1]));
        Assert.Equal(RngTestOracle.InitKey(baseCfg, [1, 2]), RngTestOracle.InitKey(cfg, [1, 2]));
        Assert.Equal(RngTestOracle.InitKey(baseCfg, [1, 1, 1]), RngTestOracle.InitKey(cfg, [1, 1, 1]));
        Assert.Equal(RngTestOracle.RunKey(baseCfg, [1, 1]), RngTestOracle.RunKey(cfg, [1, 1]));
        Assert.True(cfg.HasOverride(RngCollection.Params, [1, 1]));
        Assert.False(cfg.HasOverride(RngCollection.Params, [1, 2]));

        // The override replaces the fully folded key, so it survives a master-seed change.
        var otherMaster = new RngConfig { MasterSeed = 8 }.Override(RngCollection.Params, [1, 1], seed: 1234);
        Assert.Equal(RngTestOracle.InitKey(cfg, [1, 1]), RngTestOracle.InitKey(otherMaster, [1, 1]));

        var typed = new RngConfig { MasterSeed = 7, Algorithm = RngAlgorithm.Threefry2x32Rounds13 };
        var derived = typed.Override(RngCollection.Params, [1, 1], seed: 1234);
        Assert.False(typed.HasOverride(RngCollection.Params, [1, 1]));
        Assert.True(derived.HasOverride(RngCollection.Params, [1, 1]));
        Assert.Equal(typed.MasterSeed, derived.MasterSeed);
        Assert.Equal(typed.Algorithm, derived.Algorithm);

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
/// runtime master key, and canonically sorted per-stream override records at fixed offsets.
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
    public void TestHeaderAndOverrideRecordsEncodeAtFixedOffsetsAndRoundTrip()
    {
        var cfg = new RngConfig { MasterSeed = 42 };
        var vec = RngRuntimeIdentity.Build(cfg);
        Assert.Equal(RngRuntimeIdentity.HeaderLength, vec.Length);
        Assert.Equal(0UL, vec[RngRuntimeIdentity.AlgorithmIdIndex]);
        Assert.Equal(cfg.RunMasterKey, vec[RngRuntimeIdentity.RunKeyIndex]);
        AssertRoundTrips(cfg);

        var cfg13 = new RngConfig { MasterSeed = 42, Algorithm = RngAlgorithm.Threefry2x32Rounds13 };
        Assert.Equal(1UL, RngRuntimeIdentity.Build(cfg13)[RngRuntimeIdentity.AlgorithmIdIndex]);
        AssertRoundTrips(cfg13);

        var subMaster = new RngConfig { MasterSeed = 42, RunMasterSeed = 777 };
        Assert.NotEqual(RngTestOracle.RunKey(cfg, Paths[0]), RngTestOracle.RunKey(subMaster, Paths[0]));
        AssertRoundTrips(subMaster);

        // Runtime overrides are encoded; a Params override is init-side material and is not.
        var withOverrides = subMaster
            .Override(RngCollection.Runtime, [4, 1, 1], seed: 424242UL)
            .Override(RngCollection.Params, [2, 1], seed: 7UL);
        var ovVec = RngRuntimeIdentity.Build(withOverrides);
        Assert.Equal(RngRuntimeIdentity.HeaderLength + 1 + 3 + 1, ovVec.Length);
        Assert.Equal(1UL, ovVec[RngRuntimeIdentity.HeaderLength - 1]);

        var decoded = RngRuntimeIdentity.Decode(ovVec);
        var rec = Assert.Single(decoded.Overrides);
        Assert.Equal((int[])[4, 1, 1], rec.Path);
        Assert.Equal(424242UL, rec.Key);
        Assert.Equal(rec.Key, ovVec[rec.KeyOffset]);
        AssertRoundTrips(withOverrides);
        Assert.Equal(RngTestOracle.RunKey(subMaster, [4, 0, 1]), RngTestOracle.RunKey(decoded, [4, 0, 1]));
        Assert.NotEqual(RngTestOracle.RunKey(subMaster, [4, 1, 1]), RngTestOracle.RunKey(decoded, [4, 1, 1]));
    }

    [Fact]
    public void TestMalformedIdentityFailsLoudly()
    {
        // Empty; shorter than the header; a truncated record; trailing garbage; and records
        // claiming a huge path length (the bound must be computed before narrowing to int, or
        // `i + pathLen + 1` wraps negative and the claim is allocated).
        ulong[][] malformed =
        [
            [],
            [0UL, 42UL],
            [0UL, 42UL, 1UL],
            [0UL, 42UL, 0UL, 99UL],
            [0UL, 42UL, 1UL, int.MaxValue],
            [0UL, 42UL, 1UL, ulong.MaxValue],
        ];
        foreach (var v in malformed)
            Assert.ThrowsAny<ArgumentException>(() => RngRuntimeIdentity.Decode(v));
    }
}

/// <summary>
/// The identity transport: WithRngConfig writes the runtime identity into the ordinary
/// <c>RngSeed</c> parameter at reserved ModelId [0] — a plain initializer that survives
/// save/load bit-exactly and without duplication, makes a loaded model reproducible with no
/// config object, and re-keys every draw when a LOADED model is re-bound.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngSeedTransportTests
{
    private static int RngSeedNodeCount(ComputationGraph graph)
        => graph.ToInternal().Nodes.Count(n =>
            n.IdentifierTemplate == Shorokoo.Core.Nodes.Processors.Fast
                .FastWireRngKeyDerivation.RngSeedIdentifierTemplate);

    private static ComputationGraph LoopFeedArch()
    {
        var g = (ComputationGraph)typeof(RngRuntimeLoopFeed)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        return g.ToConcreteArchitecture(g.FromOrderedInputs(
            [TensorData([8L], new float[8]), TensorData(Array.Empty<long>(), 2L)]));
    }

    private static ComputationGraph LoopFeedModel(RngConfig cfg) => LoopFeedArch().ToConcreteModel(cfg);

    private static float[] Run(ComputationGraph m) => ComputeContext.Default
        .Execute(m, TensorData([8L], new float[8]), TensorData(Array.Empty<long>(), 2L))[0]
        .ToTensorData().As<float32>().AccessMemory().ToArray();

    [Fact]
    public void TestRngSeedIdentityRoundTripsWithoutDuplicationAndRebindingReplacesItsValue()
    {
        var cfg = new RngConfig { MasterSeed = 11 }.Override(RngCollection.Runtime, [1, 1, 1], seed: 424242UL);
        var arch = LoopFeedArch().WithRngConfig(cfg);

        Assert.Equal(1, RngSeedNodeCount(arch));
        Assert.Equal(RngRuntimeIdentity.Build(cfg), arch.TryGetRngSeed());

        var loaded = CompressedFormatUtils.LoadFastGraphFromBinary(
            CompressedFormatUtils.SaveFastGraphToBinary(arch, compressed: true));
        var carried = loaded.TryGetRngSeed();
        Assert.NotNull(carried);
        Assert.Equal(arch.TryGetRngSeed(), carried);

        // The decoded identity reproduces the config's runtime derivation, override included.
        var decoded = RngRuntimeIdentity.Decode(carried!);
        Assert.Equal(RngRuntimeIdentity.AlgorithmIdOf(RngAlgorithm.Threefry2x32), decoded.AlgorithmId);
        Assert.Equal(RngTestOracle.RunKey(cfg, [1, 1, 1]), RngTestOracle.RunKey(decoded, [1, 1, 1]));
        Assert.Equal(RngTestOracle.RunKey(cfg, [1, 0, 1]), RngTestOracle.RunKey(decoded, [1, 0, 1]));

        Assert.Equal(1, RngSeedNodeCount(loaded));
        var loaded2 = CompressedFormatUtils.LoadFastGraphFromBinary(
            CompressedFormatUtils.SaveFastGraphToBinary(loaded, compressed: true));
        Assert.Equal(1, RngSeedNodeCount(loaded2));
        Assert.Equal(carried, loaded2.TryGetRngSeed());

        var rebound = LoopFeedArch().WithRngConfig(new RngConfig { MasterSeed = 11 });
        Assert.Equal(RngRuntimeIdentity.Build(new RngConfig { MasterSeed = 11 }), rebound.TryGetRngSeed());
        rebound = rebound.WithRngConfig(new RngConfig { MasterSeed = 12 });
        Assert.Equal(RngRuntimeIdentity.Build(new RngConfig { MasterSeed = 12 }), rebound.TryGetRngSeed());
        Assert.Equal(1, RngSeedNodeCount(rebound));
    }

    [Fact]
    public void TestSaveLoadCarriesTheAlgorithmAndRebindingALoadedModelRekeysEveryDraw()
    {
        var seedA = new RngConfig { MasterSeed = 11 };
        var seedB = new RngConfig { MasterSeed = 12 };
        var alg13 = new RngConfig { MasterSeed = 11, Algorithm = RngAlgorithm.Threefry2x32Rounds13 };

        var draws20 = Run(LoopFeedModel(seedA));
        var m13 = LoopFeedModel(alg13);
        var before = Run(m13);
        Assert.NotEqual(before, draws20);

        var loaded13 = CompressedFormatUtils.LoadFastGraphFromBinary(
            CompressedFormatUtils.SaveFastGraphToBinary(m13, compressed: true));
        var carried13 = loaded13.TryGetRngSeed();
        Assert.NotNull(carried13);
        Assert.Equal(RngRuntimeIdentity.AlgorithmIdOf(RngAlgorithm.Threefry2x32Rounds13),
            RngRuntimeIdentity.Decode(carried13!).AlgorithmId);
        Assert.Equal(before, Run(loaded13));
        Assert.NotEqual(draws20, Run(loaded13));

        var loadedA = CompressedFormatUtils.LoadFastGraphFromBinary(
            CompressedFormatUtils.SaveFastGraphToBinary(LoopFeedModel(seedA), compressed: true));
        Assert.Equal(draws20, Run(loadedA));

        var rebound = loadedA.WithRngConfig(seedB);
        var rekeyed = Run(rebound);
        Assert.NotEqual(draws20, rekeyed);
        Assert.Equal(Run(LoopFeedModel(seedB)), rekeyed);
        Assert.Equal(rekeyed, Run(CompressedFormatUtils.LoadFastGraphFromBinary(
            CompressedFormatUtils.SaveFastGraphToBinary(rebound, compressed: true))));

        // A loaded model keeps its feed ops (#59), so a re-bind may also change the override
        // SET and the draw algorithm, matching a model built under that config directly.
        var withOverride = seedA.Override(RngCollection.Runtime, [1, 1, 1], 42UL);
        Assert.Equal(Run(LoopFeedModel(withOverride)), Run(loadedA.WithRngConfig(withOverride)));
        var reboundTo13 = Run(loadedA.WithRngConfig(alg13));
        Assert.Equal(Run(LoopFeedModel(alg13)), reboundTo13);
        Assert.NotEqual(draws20, reboundTo13);
    }

    [Fact]
    public void TestRngIdentityProjectsTheRuntimeTierLosslesslyAndRefusesAnUnrecognizedAlgorithmId()
    {
        var cfg = new RngConfig { MasterSeed = 11, Algorithm = RngAlgorithm.Threefry2x32Rounds13 }
            .Override(RngCollection.Runtime, [1, 1, 1], seed: 424242UL);
        var model = LoopFeedModel(cfg);
        var identity = model.TryGetRngIdentity()!;

        Assert.Equal(RngAlgorithm.Threefry2x32Rounds13, identity.Algorithm);
        Assert.Equal(cfg.RunMasterKey, identity.RunMasterKey);
        Assert.Equal((int[])[1, 1, 1], Assert.Single(identity.Overrides).ModelIdPath);
        Assert.Equal(424242UL, identity.TryGetOverride([1, 1, 1]));
        Assert.Null(identity.TryGetOverride([1, 0, 1]));
        Assert.Null(LoopFeedArch().TryGetRngIdentity());

        Assert.Equal(model.TryGetRngSeed(), model.WithRngConfig(identity.ToRuntimeConfig()).TryGetRngSeed());
        Assert.Equal(Run(model), Run(model.WithRngConfig(identity.ToRuntimeConfig())));
        Assert.Equal(model.TryGetRngSeed(),
            LoopFeedArch().WithRngConfig(cfg.Override(RngCollection.Params, [1, 0], 7UL)).TryGetRngSeed());

        const ulong unknownId = 9999;
        var arch = LoopFeedArch().WithRngConfig(RngConfig.Default).ToInternal();
        var seedData = arch.TryGetRngSeed()!;
        seedData[RngRuntimeIdentity.AlgorithmIdIndex] = unknownId;
        var seedNode = arch.Nodes.Single(n => n.IdentifierTemplate == Shorokoo.Core.Nodes.Processors.Fast
            .FastWireRngKeyDerivation.RngSeedIdentifierTemplate);
        seedNode.Attributes = seedNode.Attributes.SetAttributes(
            (OnnxOpAttributeNames.ShrkAttrTensorData, (object?)TensorData.Create(
                new Shape(seedData.Length), DType.UInt64,
                OnnxUtils.CreateTensorValue(new Shape(seedData.Length), seedData))));

        var tampered = ComputationGraph.FromInternal(arch, GraphKind.ConcreteArchitecture);
        Assert.Contains(unknownId.ToString(),
            Assert.Throws<NotSupportedException>(() => tampered.TryGetRngIdentity()).Message);
    }

    [Fact]
    public void TestWithRngOverrideRekeysOneStreamAndLeavesTheRestOfTheIdentityIntact()
    {
        var cfg = new RngConfig { MasterSeed = 11 }.Override(RngCollection.Runtime, [1, 1, 1], seed: 424242UL);
        var model = LoopFeedModel(cfg);

        var added = model.WithRngOverride(RngCollection.Runtime, [1, 0, 1], 99UL);
        var addedIdentity = added.TryGetRngIdentity()!;
        Assert.Equal(cfg.RunMasterKey, addedIdentity.RunMasterKey);
        Assert.Equal(424242UL, addedIdentity.TryGetOverride([1, 1, 1]));
        Assert.Equal(99UL, addedIdentity.TryGetOverride([1, 0, 1]));
        Assert.Equal(Run(LoopFeedModel(cfg.Override(RngCollection.Runtime, [1, 0, 1], 99UL))), Run(added));

        var replaced = added.WithRngOverride(RngCollection.Runtime, [1, 1, 1], 7UL);
        Assert.Equal(7UL, replaced.TryGetRngIdentity()!.TryGetOverride([1, 1, 1]));
        Assert.Equal(99UL, replaced.TryGetRngIdentity()!.TryGetOverride([1, 0, 1]));
        Assert.Equal(1, RngSeedNodeCount(replaced));
        Assert.Equal(Run(model), Run(replaced.WithRngConfig(cfg)));

        Assert.Contains("matches no runtime stream", Assert.Throws<InvalidOperationException>(
            () => model.WithRngOverride(RngCollection.Runtime, [9], 1UL)).Message);
        Assert.Contains("records no Params-collection identity", Assert.Throws<ArgumentException>(
            () => model.WithRngOverride(RngCollection.Params, [1, 1, 1], 1UL)).Message);
    }

    [Fact]
    public void TestModelWithoutRandomFeedsCarriesNothingRngRelatedAndBindingRequiresRealizedStreams()
    {
        var g = (ComputationGraph)typeof(RngInitTwoLinears)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var sample = TensorData([4L, 4L], Enumerable.Repeat(1f, 16).ToArray());
        var model = g.ToConcreteArchitecture(g.FromOrderedInputs([sample]))
            .ToConcreteModel(new RngConfig { MasterSeed = 7 });

        Assert.Equal(0, RngSeedNodeCount(model));
        Assert.Null(model.TryGetRngSeed());
        Assert.Null(model.TryGetRngIdentity());
        Assert.Contains("no bound RNG identity", Assert.Throws<InvalidOperationException>(
            () => model.WithRngOverride(RngCollection.Runtime, [1], 1UL)).Message);
        Assert.Equal(0, RngSeedNodeCount(CompressedFormatUtils.LoadFastGraphFromBinary(
            CompressedFormatUtils.SaveFastGraphToBinary(model, compressed: true))));

        model = model.WithRngConfig(new RngConfig { MasterSeed = 8 });   // no-op, no throw
        var unmatched = Assert.Throws<InvalidOperationException>(
            () => model.WithRngConfig(new RngConfig { MasterSeed = 8 }
                .Override(RngCollection.Runtime, [1], 1UL)));
        Assert.Contains("matches no runtime stream", unmatched.Message);

        // An id-bearing feed with no key-derivation chain (never concretized) fails loudly.
        var draw = RandomUniform(Vector(4L), 0f, 1f);
        var graph = new InternalComputationGraph([], [draw]);
        var feed = graph.Nodes.Single(n => n.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM);
        feed.Attributes = feed.Attributes.SetAttributes(
            (OnnxOpAttributeNames.ShrkAttrLocalModelId, (long[])[1]));
        var unrealized = Assert.Throws<InvalidOperationException>(
            () => graph.ApplyRngConfig(new RngConfig { MasterSeed = 1 }));
        Assert.Contains("no realized stream ids", unrealized.Message);
    }
}

/// <summary>
/// The graph-only-RNG guard (#136): no production code may run the RNG algorithm host-side.
/// Key splits/folds and draws alike are computed by the in-graph tagged functions; a host
/// consumer that needs a concrete key resolves it by <em>executing</em> that derivation
/// (<c>RngKeyResolver</c>). The C# <see cref="Threefry2x32"/> generator survives only as a
/// test oracle (<see cref="RngTestOracle"/>). Source-level rather than behavioural: a
/// reintroduced host fold computes the same numbers for the built-ins and would only break
/// once a custom algorithm exists.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngNoHostRngTests
{
    private static string ProductionSourceRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !Directory.Exists(Path.Combine(dir.FullName, "src", "Shorokoo")))
            dir = dir.Parent;
        Assert.NotNull(dir);
        return Path.Combine(dir!.FullName, "src", "Shorokoo");
    }

    // `Rounds`/`Rounds13` are round-count constants parameterising the in-graph function
    // builders, so they are exempt — but only as exact member names.
    private static readonly Regex[] HostRngUse =
    [
        new(@"Threefry2x32\s*\.\s*(?!Rounds13\b|Rounds\b)\w+", RegexOptions.Compiled),
        new(@"using\s+static\s+[\w\.]*\bThreefry2x32\s*;", RegexOptions.Compiled),
        new(@"using\s+\w+\s*=\s*[\w\.]*\bThreefry2x32\s*;", RegexOptions.Compiled),
    ];

    // Strips comments and string literals so prose does not trip the guard, and so a call
    // split across lines is still seen as one text.
    private static string StripCommentsAndStrings(string source)
    {
        source = Regex.Replace(source, @"/\*.*?\*/", " ", RegexOptions.Singleline);
        source = Regex.Replace(source, @"//[^\n]*", " ");
        source = Regex.Replace(source, @"@""(?:[^""]|"""")*""", " ");
        source = Regex.Replace(source, @"""(?:\\.|[^""\\])*""", " ");
        return source;
    }

    private static string[] ProductionSourceFiles() => Directory
        .EnumerateFiles(ProductionSourceRoot(), "*.cs", SearchOption.AllDirectories)
        .Where(f => !f.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}") &&
                    !f.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}"))
        .ToArray();

    [Fact]
    public void TestNoProductionCodeRunsTheRngAlgorithmHostSideAndTheGuardStillDetectsEveryEvasion()
    {
        var files = ProductionSourceFiles();
        // A guard that silently scans nothing is worse than no guard.
        Assert.True(files.Length > 100);

        var offenders = files
            // Threefry2x32.cs DEFINES the generator; defining it is not calling it. Everything
            // else in that file is still swept.
            .SelectMany(f => HostRngUse
                .SelectMany(rx => rx.Matches(StripCommentsAndStrings(File.ReadAllText(f)))
                    .Where(_ => Path.GetFileName(f) != "Threefry2x32.cs" || !rx.ToString().StartsWith("Threefry2x32"))
                    .Select(m => $"{Path.GetFileName(f)}: {m.Value.Trim()}")))
            .Distinct()
            .ToArray();
        Assert.Empty(offenders);

        string[] mustFlag =
        [
            "var k = Threefry2x32.Bijection(a, b, c, d);",
            "var k = Shorokoo.Core.Rng.Threefry2x32.Bijection(a, b, c, d);",
            "var k = Threefry2x32\n    .Bijection(a, b, c, d);",
            "using static Shorokoo.Core.Rng.Threefry2x32;",
            "using TF = Shorokoo.Core.Rng.Threefry2x32;",
            "var k = Threefry2x32.RoundsFold(key, i);",
        ];
        foreach (var sample in mustFlag)
            Assert.True(HostRngUse.Any(rx => rx.IsMatch(StripCommentsAndStrings(sample))));

        string[] mustNotFlag =
        [
            "int rounds = Threefry2x32.Rounds;",
            "int rounds = Threefry2x32.Rounds13;",
            "// see Threefry2x32.Bijection for the host oracle",
            "/* Threefry2x32.Bijection */",
            "var name = \"Threefry2x32.Bijection\";",
        ];
        foreach (var sample in mustNotFlag)
            Assert.False(HostRngUse.Any(rx => rx.IsMatch(StripCommentsAndStrings(sample))));

        // The deleted host key fold must not come back under its old names.
        var flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Static |
                    System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic;
        foreach (var name in (string[])["FoldKey", "FoldInitKey", "FoldRunKey"])
        {
            Assert.Null(typeof(RngConfig).GetMethod(name, flags));
            Assert.Null(typeof(RngRuntimeIdentity).GetMethod(name, flags));
        }
    }
}

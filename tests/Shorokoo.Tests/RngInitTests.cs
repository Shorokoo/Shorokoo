using System;
using System.Linq;
using Shorokoo.Core.Rng;
using Shorokoo.Modules.Layers;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// Two same-shape Linear weights at distinct module paths (so distinct parameters,
/// each KaimingUniform-initialized on a [4,4] weight). Distinct paths — not two bare
/// identical Init calls, which would be a single common-subexpression parameter.
/// </summary>
[Module]
public partial class RngInitTwoLinears
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var y1 = Linear.Call(Scalar(4L), Scalar(false), x);   // weight [4,4]
        var y2 = Linear.Call(Scalar(4L), Scalar(false), y1);  // weight [4,4], distinct path
        return y2;
    }
}

/// <summary>A uint32 state parameter initialized with raw random bits: exercises RandomBits
/// inside a (state) parameter initializer. Bits produce unsigned integers, so this must be a
/// state parameter (trainable parameters are float — they carry gradients), keyed on the
/// parameter's own init stream exactly like a uniform/normal init draw.</summary>
[StateInitializer(Ownership = StateOwnership.ModuleOwned)]
public static partial class RngBitsStateInit
{
    public static Tensor<uint32> Inline(Vector<int64> shape) => RandomBits<uint32>(shape);
}

[Module]
public partial class RngBitsInitLayer
{
    public static Tensor<uint32> Inline(Tensor<float32> x) => RngBitsStateInit.Init([Scalar(4L), Scalar(4L)]);
}

/// <summary>A TRAINABLE float32 parameter whose initializer builds its value from BOTH a uniform
/// draw and a raw-bits draw (bits → cast → float). Proves RandomBits can be an intermediate step
/// toward a float trainable weight (the output is float32, so it is a valid trainable param), and
/// that two different RNG ops coexist in one initializer (each keyed to its own sub-stream).</summary>
[TrainableParamInitializer]
public static partial class BitsIntermediateTrainableInit
{
    public static Tensor<float32> Inline(Vector<int64> shape)
    {
        var u = RandomUniform(shape, 0f, 1f);
        var fromBits = RandomBits<uint32>(shape).Cast<float32>() * Scalar(1.0f / 4294967296.0f);  // uint32 → [0,1)
        return u * fromBits;   // float32 output ⇒ valid trainable parameter
    }
}

[Module]
public partial class BitsIntermediateTrainableLayer
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var w = BitsIntermediateTrainableInit.Init(x.ShapeTensor());   // trainable [4,4] weight
        return x * w;
    }
}

/// <summary>
/// End-to-end coverage for per-parameter initialization RNG (phase 2). Concretizes
/// <see cref="RngInitTwoLinears"/> and initializes it under various
/// <see cref="RngConfig"/>s, asserting the properties the design promises:
/// same-shape parameters now differ, initialization is reproducible for a config,
/// and the master seed re-randomizes everything.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngInitTests
{
    private static ComputationGraph ConcreteArch()
    {
        var g = RngInitTwoLinears.ComputationGraph;
        var sample = TensorData([4L, 4L], Enumerable.Repeat(1f, 16).ToArray());
        return g.ToConcreteArchitecture(g.FromOrderedInputs([sample]));
    }

    private static float[][] InitWeights(RngConfig? cfg = null)
    {
        var arch = ConcreteArch();
        var pl = arch.InitializeTrainableParams(rngConfig: cfg);
        // Both Linear weights are [4,4] = 16 elements.
        return pl.ModelParams
            .Select(p => p.ToTensorData().As<float32>().AccessMemory().ToArray())
            .Where(v => v.Length == 16)
            .ToArray();
    }

    private static uint[] MaterializeBitsState(RngConfig cfg)
    {
        var g = RngBitsInitLayer.ComputationGraph;
        var sample = TensorData([4L, 4L], Enumerable.Repeat(1f, 16).ToArray());
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([sample]));
        var pl = arch.InitializeTrainableParams(rngConfig: cfg);
        return pl.ModelParams
            .Select(p => p.ToTensorData())
            .Where(td => td.DType == DType.UInt32)
            .SelectMany(td => td.As<uint32>().AccessMemory().ToArray())
            .ToArray();
    }

    [Fact]
    public void TestTrainableInitUsesBitsIntermediateAndTwoRngOps()
    {
        // A float32 trainable weight whose initializer uses a uniform draw AND a bits draw:
        // nothing in the trainable-param path forbids a non-float intermediate or more than one
        // RNG op per initializer (each draw is keyed to a distinct sub-stream by ordinal).
        float[] Materialize(RngConfig cfg)
        {
            var g = BitsIntermediateTrainableLayer.ComputationGraph;
            var sample = TensorData([4L, 4L], Enumerable.Repeat(1f, 16).ToArray());
            var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([sample]));
            var pl = arch.InitializeTrainableParams(rngConfig: cfg);
            return pl.ModelParams
                .Select(p => p.ToTensorData().As<float32>().AccessMemory().ToArray())
                .Single(v => v.Length == 16);   // the [4,4] trainable weight
        }

        var a = Materialize(new RngConfig { MasterSeed = 5 });
        var b = Materialize(new RngConfig { MasterSeed = 5 });
        var c = Materialize(new RngConfig { MasterSeed = 6 });

        Assert.Equal(a, b);                          // reproducible for a config
        Assert.NotEqual(a, c);                       // master seed re-randomizes
        Assert.All(a, v => Assert.InRange(v, 0.0f, 1.0f));   // u * bits/2^32 ∈ [0,1)
        Assert.Contains(a, v => v != 0.0f);          // real draws, not a zeroed fallback
    }

    [Fact]
    public void TestBitsInitializerMaterializesKeyedAndReproducible()
    {
        // RandomBits<uint32> in a state-parameter initializer keys on the parameter's own init
        // stream, materializes to real uint bits, is reproducible for a config, and re-rolls
        // with the master seed — the raw-bits analogue of the uniform/normal init properties.
        var a = MaterializeBitsState(new RngConfig { MasterSeed = 5 });
        var b = MaterializeBitsState(new RngConfig { MasterSeed = 5 });
        var c = MaterializeBitsState(new RngConfig { MasterSeed = 6 });

        Assert.Equal(16, a.Length);            // the [4,4] uint32 state param materialized
        Assert.Equal(a, b);                    // reproducible for a config
        Assert.NotEqual(a, c);                 // master seed re-randomizes
        Assert.Contains(a, v => v != 0u);      // real bits, not a zeroed fallback
    }

    [Fact]
    public void TestSameShapeParamsAreNotIdentical()
    {
        var w = InitWeights();
        Assert.Equal(2, w.Length);
        // The core bug the design fixes: two same-shape parameters previously received
        // identical values; keyed by their (distinct) canonical names they now differ.
        Assert.False(w[0].SequenceEqual(w[1]));
    }

    [Fact]
    public void TestInitializationIsReproducibleForAConfig()
    {
        var a = InitWeights(new RngConfig { MasterSeed = 123 });
        var b = InitWeights(new RngConfig { MasterSeed = 123 });
        Assert.Equal(a.Length, b.Length);
        for (int i = 0; i < a.Length; i++)
            Assert.Equal(a[i], b[i]);
    }

    [Fact]
    public void TestMasterSeedChangesAllValues()
    {
        var a = InitWeights(new RngConfig { MasterSeed = 1 });
        var b = InitWeights(new RngConfig { MasterSeed = 2 });
        for (int i = 0; i < a.Length; i++)
            Assert.False(a[i].SequenceEqual(b[i]), $"param {i} unchanged across seeds");
    }

    [Fact]
    public void TestKaimingValuesAreFiniteAndInBound()
    {
        // KaimingUniform bound for fanIn=4 is sqrt(6/4) ≈ 1.22474; values stay within it.
        foreach (var v in InitWeights())
            foreach (var x in v)
            {
                Assert.True(float.IsFinite(x));
                Assert.InRange(x, -1.2248f, 1.2248f);
            }
    }

    [Fact]
    public void TestUnmatchedParamsOverrideFailsInitialization()
    {
        // Mirror of the Runtime-side bind check: a Params override that matches no trainable
        // parameter must fail initialization loudly — a silently inactive override is exactly
        // the re-keying hazard explicit seeding exists to prevent.
        var cfg = new RngConfig { MasterSeed = 1 };
        cfg = cfg.Override(RngCollection.Params, [9, 9, 9], 1UL);
        var ex = Assert.Throws<InvalidOperationException>(() => InitWeights(cfg));
        Assert.Contains("matches no trainable parameter", ex.Message);
    }

    [Fact]
    public void TestMatchedParamsOverrideReSeedsExactlyOneParam()
    {
        // Overriding one weight's stream by its ModelId path re-rolls that weight only.
        var baseline = InitWeights(new RngConfig { MasterSeed = 5 });

        var arch = ConcreteArch();
        var firstWeightPath = arch.GetConcreteModelParamInfos().ParamInfos
            .Single(p => p.Shape.Dims.SequenceEqual((long[])[4, 4]) && p.ModelId.Vals[0] == 1)
            .ModelId.Vals.ToArray();
        var cfg = new RngConfig { MasterSeed = 5 };
        cfg = cfg.Override(RngCollection.Params, firstWeightPath, 4242UL);

        var overridden = InitWeights(cfg);
        Assert.False(baseline[0].SequenceEqual(overridden[0]));   // re-seeded
        Assert.Equal(baseline[1], overridden[1]);                 // untouched
    }
}

/// <summary>
/// The init-value derivation pinned to FROZEN constants — the cross-version seed contract.
/// Every other init test is relational (the system compared against itself), so a silent
/// change anywhere in the chain — master → "init" sub-master fold → per-path FoldInitKey →
/// in-graph keyed draw (counter (elementIndex, drawOrdinal), draw rounds) → uniform
/// transform → Kaiming scaling — would keep them all green while breaking every seed
/// anyone has ever shared. These values were generated from the implementation that
/// defines the derivation (regenerated once when init moved from host-precomputed noise
/// to the in-graph keyed draw — the deliberate breaking change that unified init with the
/// feed convention); a red here means "MasterSeed 123 no longer produces the weights it
/// used to" and must never be fixed by regenerating the constants without a deliberate,
/// breaking-change decision.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngInitFrozenDerivationTests
{
    [Fact]
    public void TestInitKeyDerivationIsFrozen()
    {
        // Layer 1: the key derivation alone (fold order, the "init" label, sub-master wiring).
        // Resolved through PRODUCTION (RngKeyResolver executes the in-graph split chain, which is
        // how a real init key is derived since #136) — asserting via the test oracle instead
        // would pin only the oracle and stay green if the product's derivation changed.
        var cfg = new RngConfig { MasterSeed = 123 };
        var keys = Core.Rng.RngKeyResolver.Resolve(
            [cfg.InitKeySpec((int[])[1, 1]), cfg.InitKeySpec((int[])[2, 1])]);
        Assert.Equal(0x33e150fc_0177f47cUL, keys[0]);
        Assert.Equal(0x2a93ecfc_3c6c3147UL, keys[1]);

        // ...and the independent host oracle agrees with what the graph computed.
        foreach (var (path, i) in new[] { ((int[])[1, 1], 0), ((int[])[2, 1], 1) })
            Assert.Equal(RngTestOracle.InitKey(cfg, path), keys[i]);
    }

    [Fact]
    public void TestInitValuesAreFrozen()
    {
        // Layer 2: the full materialized values (draw composition: counter scheme, rounds,
        // uniform transform, substreamIndex ordinal, initializer scaling). REFERENCE: golden.
        // Exact equality is safe cross-backend: the uniform path is Threefry integer ops
        // plus IEEE-exact float multiply/add — no transcendental kernels involved.
        float[] expected0 = [-1.1163274f, 1.1247115f, -0.20118715f, -0.8630716f, 0.12048453f, 0.73705673f, -0.38930926f, -0.9366948f, 0.7735388f, -0.49744576f, -0.60573745f, -0.41470495f, -1.003003f, 0.19222532f, 0.8099788f, 0.49284714f];
        float[] expected1 = [-0.88179505f, 0.22158815f, 0.46890008f, 1.0455909f, -1.1027482f, 0.91218925f, -0.5450415f, 0.36076564f, -0.54581296f, 0.6172559f, -0.40583524f, 0.3620881f, -0.5337995f, -0.24915563f, 1.085321f, 0.67871165f];

        var g = RngInitTwoLinears.ComputationGraph;
        var sample = TensorData([4L, 4L], System.Linq.Enumerable.Repeat(1f, 16).ToArray());
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([sample]));
        var pl = arch.InitializeTrainableParams(rngConfig: new RngConfig { MasterSeed = 123 });
        var ws = pl.ModelParams
            .Select(p => p.ToTensorData().As<float32>().AccessMemory().ToArray())
            .Where(v => v.Length == 16).ToArray();
        Assert.Equal(2, ws.Length);
        Assert.Equal(expected0, ws[0]);   // weight at ModelId [1, 1]
        Assert.Equal(expected1, ws[1]);   // weight at ModelId [2, 1]
    }

    [Fact]
    public void TestBatchedKeyResolutionMatchesTheHostOracle()
    {
        // The resolver folds a whole tree LEVEL per batched split (#138): M parent keys and M
        // counters in, M child keys out as one [M] uint64 vector. Grouping is the failure surface
        // — specs are bucketed by depth and each group's results are scattered back by index, so
        // a mis-grouped depth or an off-by-one scatter silently hands back another stream's key.
        //
        // So resolve a set that deliberately mixes depths (which is what splits the work into
        // groups) and group sizes, and check every key against the independent host oracle.
        var cfg = new RngConfig { MasterSeed = 77 };
        int[][] paths =
        [
            [1], [2], [3],                     // depth 1, group of 3
            [1, 1], [1, 2], [2, 1], [7, 9],    // depth 2, group of 4
            [1, 2, 3],                         // depth 3, group of 1 (M == 1 edge case)
            [4, 5, 6, 7, 8, 9],                // depth 6
        ];

        var keys = Core.Rng.RngKeyResolver.Resolve([.. paths.Select(p => cfg.InitKeySpec(p))]);

        Assert.Equal(paths.Length, keys.Count);
        for (int i = 0; i < paths.Length; i++)
            Assert.Equal(RngTestOracle.InitKey(cfg, paths[i]), keys[i]);
        // Distinct paths must stay distinct — a packing bug that returned one row for every
        // stream would still satisfy a per-element check done carelessly.
        Assert.Equal(paths.Length, keys.Distinct().Count());
    }

    [Fact]
    public void TestMultiDrawInitValuesAreFrozen()
    {
        // Layer 3: an initializer that draws TWICE (a uniform and a raw-bits draw, combined).
        // Both draws share the parameter's ONE stream key and are separated only by their
        // substreamIndex sub-stream ordinal, so this golden is what pins the ordinal assignment:
        // renumber the draws (or key them identically) and every value here moves, while the
        // relational assertions in TestTrainableInitUsesBitsIntermediateAndTwoRngOps — same
        // config reproduces, a new seed re-randomizes, values stay in range — all still hold.
        //
        // Exact equality is safe cross-backend: Threefry integer ops, an IEEE
        // round-to-nearest uint32 -> float32 conversion, and multiplies (one by a power of
        // two, hence exact). No transcendental kernels involved.
        float[] expected =
        [0.12127531f, 0.045944285f, 0.71740365f, 0.025424859f, 0.1510541f, 0.14533761f, 0.0006900568f, 0.3375456f, 0.34560895f, 0.17979373f, 0.14335075f, 0.010312658f, 0.112886935f, 0.4531033f, 0.27203277f, 0.15625617f];

        var g = BitsIntermediateTrainableLayer.ComputationGraph;
        var sample = TensorData([4L, 4L], System.Linq.Enumerable.Repeat(1f, 16).ToArray());
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([sample]));
        var pl = arch.InitializeTrainableParams(rngConfig: new RngConfig { MasterSeed = 123 });
        var w = pl.ModelParams
            .Select(p => p.ToTensorData().As<float32>().AccessMemory().ToArray())
            .Single(v => v.Length == 16);
        Assert.Equal(expected, w);
    }
}

/// <summary>Helper module holding the random draw that <see cref="RngInitNestedDrawInit"/> factors out.</summary>
[Module]
public partial class RngInitNestedDrawHelper
{
    public static Tensor<float32> Inline(Vector<int64> shape)
        => RandomUniform(shape, low: -1.0f, high: 1.0f);
}

/// <summary>
/// A custom initializer whose random draw is nested inside a called function instead of
/// inline in its own body — keyed per-parameter initialization reaches it by flattening
/// the initializer body before the noise substitution.
/// </summary>
[TrainableParamInitializer]
public static partial class RngInitNestedDrawInit
{
    public static Tensor<float32> Inline(Vector<int64> shape)
        => RngInitNestedDrawHelper.Call(shape);
}

[Module]
public partial class RngInitNestedDrawLayer
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var w = RngInitNestedDrawInit.Init(x.ShapeTensor());
        return x * w;
    }
}

/// <summary>
/// Initialization-side draws must never silently escape the keyed scheme into unkeyed
/// backend randomness. A draw factored into a called function is brought into the scheme
/// by flattening the initializer body before the noise substitution (first test); the
/// other escape — <c>FastInitializeModelParams</c> invoked with a config but a
/// missing/incomplete parameter inventory, which used to silently disable the noise
/// injection for all/some parameters while the config's own override validation still ran
/// and passed — fails loudly instead.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngInitFailLoudTests
{
    // Returns a mutable graph: these tests drive the Fast processor directly.
    private static InternalComputationGraph ConcreteArch()
    {
        var g = RngInitTwoLinears.ComputationGraph;
        var sample = TensorData([4L, 4L], Enumerable.Repeat(1f, 16).ToArray());
        return g.ToConcreteArchitecture(g.FromOrderedInputs([sample])).ToInternal();
    }

    [Fact]
    public void TestDrawNestedInCalledFunctionIsInlinedAndKeyed()
    {
        // The draw sits in RngInitNestedDrawHelper, called by the initializer body. Before
        // flattening was added, the top-level substitution found nothing to intercept and
        // the nested draw resolved through the generic ONNX fallback to real
        // backend-random, non-reproducible values — with no error and no entry in the RNG
        // stream report. Flattening makes the draw top-level, so it draws keyed noise
        // by the parameter's own stream like an inline draw.
        float[] Init(ulong seed)
        {
            var g = RngInitNestedDrawLayer.ComputationGraph;
            var sample = TensorData([2L, 2L], 1f, 1f, 1f, 1f);
            var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([sample]));
            return arch.InitializeTrainableParams(rngConfig: new RngConfig { MasterSeed = seed })
                .ModelParams.Single().ToTensorData().As<float32>().AccessMemory().ToArray();
        }

        var a = Init(123);
        Assert.Equal(4, a.Length);
        Assert.All(a, x => Assert.InRange(x, -1.0f, 1.0f));   // the helper's declared U(-1, 1)
        Assert.True(a.Distinct().Count() > 1);                // not a degenerate fill
        Assert.Equal(a, Init(123));                           // reproducible for a config
        Assert.False(a.SequenceEqual(Init(124)));             // derived from the master seed
    }

    [Fact]
    public void TestConfigWithoutInventoryFailsAtEntry()
    {
        // A non-null config with paramInfos: null used to silently skip the noise injection
        // for every parameter — un-keyed initializers, backend randomness — while the
        // Params-override validation (gated only on the config) still ran, making the
        // config look engaged. Now the pairing is enforced at entry.
        var ex = Assert.Throws<ArgumentNullException>(() =>
            Shorokoo.Core.Nodes.Processors.Fast.FastInitializeModelParams.Process(
                ConcreteArch(), null, new RngConfig { MasterSeed = 1 }, paramInfos: null));
        Assert.Contains("without the parameter inventory", ex.Message);
    }

    [Fact]
    public void TestParamMissingFromInventoryFailsNamingIt()
    {
        // An inventory miss on one parameter used to skip that parameter's noise injection
        // while its siblings stayed keyed — a silent keyed/un-keyed mix. Now it throws,
        // naming the parameter (the mirror of the unmatched-override check).
        var arch = ConcreteArch();
        var full = arch.GetConcreteModelParamInfos();
        var missing = full.ParamInfos[0];
        var partial = new Shorokoo.Core.ConcreteModelParamInfos(full.ParamInfos.RemoveAt(0));

        var ex = Assert.Throws<InvalidOperationException>(() =>
            Shorokoo.Core.Nodes.Processors.Fast.FastInitializeModelParams.Process(
                arch, null, new RngConfig { MasterSeed = 1 }, partial));
        Assert.Contains("missing from the supplied parameter inventory", ex.Message);
        Assert.Contains($"[{string.Join(", ", missing.ModelId.Vals)}]", ex.Message);
    }
}

/// <summary>
/// The two normal-family consumers in one module: a KaimingNormal-initialized [4,4] weight
/// (in-graph Box–Muller path, run at parameter initialization) and a Globals.RandomNormal feed
/// (in-graph Box–Muller path, lowered to the keyed counter RNG). The weight is kept live via
/// a ×0 term, so with a zero input the module's output equals the feed draw exactly.
/// </summary>
[Module]
public partial class RngNormalBothCollections
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var w = Shorokoo.Modules.Initializers.KaimingNormal.Init([Scalar(4L), Scalar(4L)]);
        var feed = RandomNormal(x.ShapeTensor(), mean: 0.0f, scale: 1.0f);
        return feed + w.Reduce(ReduceKind.Sum, keepDims: false).Scalar() * Scalar(0.0f);
    }
}

/// <summary>
/// The NORMAL-family value derivation pinned to FROZEN constants, for both consumer kinds —
/// the cross-version seed contract for normals (the sibling of
/// <see cref="RngInitFrozenDerivationTests"/>, which pins the uniform family). Every
/// Box–Muller composition variant (cos↔sin, u₁↔u₂, 1−u₁↔u₁, uniform-to-element pairing)
/// yields a perfect N(0,1) distribution, so the moments tests can never detect a composition
/// change: only value pins hold the convention fixed. Both consumers now draw via the same
/// in-graph keyed lowering (fold → key constant/table → per-element SHRK lowering → ONNX
/// Ln/Sqrt/Cos kernels): parameter initialization keys off the init sub-master with
/// substreamIndex = the draw's ordinal, the runtime feed keys off the runtime sub-master with
/// substreamIndex = the execution counter — distinct streams, pinned independently, never
/// compared. One Fact covers both, at both round counts. All values are asserted at 1e-6
/// (ORT transcendental kernels may drift in the last ULP across backends; a composition
/// change shifts values by O(1)). A red here means "this seed no longer draws the normals
/// it used to" and must never be fixed by regenerating the constants without a deliberate,
/// breaking-change decision — the init constants were regenerated exactly once, when init
/// moved from host-precomputed noise to the in-graph keyed draw.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngNormalFrozenDerivationTests
{
    private static (float[] init, float[] feed) Run(RngConfig cfg)
    {
        var g = RngNormalBothCollections.ComputationGraph;
        var input = TensorData([4L, 4L], Enumerable.Repeat(0f, 16).ToArray());
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([input]));
        // Filter to the float32 weight before casting: the param list also carries the
        // framework-injected RngExecutionCounter, which is int64 state.
        var init = arch.InitializeTrainableParams(rngConfig: cfg).ModelParams
            .Select(p => p.ToTensorData())
            .Where(t => t.DType == DType.Float32)
            .Select(t => t.As<float32>().AccessMemory().ToArray())
            .Single(v => v.Length == 16);
        var concrete = arch.ToConcreteModel(cfg);
        var feed = ComputeContext.Default.Execute(concrete, input)[0]
            .ToTensorData().As<float32>().AccessMemory().ToArray();
        return (init, feed);
    }

    [Fact]
    public void TestNormalInitAndDrawValuesAreFrozen()
    {
        // REFERENCE: golden — generated once from the implementation that defines the convention.
        float[] init20 = [0.12544397f, 0.2957119f, 1.614189f, -0.22173794f, -0.23703626f, -0.64295983f, -0.1786294f, -1.4764216f, 0.15099204f, -0.019193964f, -0.21473941f, 1.033891f, 1.3871936f, 0.59315336f, -0.41766375f, 0.006978817f];
        float[] feed20 = [-0.2854576f, -1.0614587f, 0.69347787f, 1.1629281f, -0.63950145f, 1.7594889f, 1.6418929f, -2.4083176f, 0.79176825f, -0.48223278f, 0.48083737f, 0.38064465f, -0.3447332f, 0.0259849f, 0.062860526f, -0.43736157f];
        float[] init13 = [0.10458848f, -1.9170773f, 0.12625404f, 0.056145065f, -1.4316688f, -0.37182125f, 0.019850086f, 0.9272645f, -1.0287207f, 1.1623243f, -0.9364095f, 0.21012756f, 0.55460495f, -0.6630122f, 0.30105424f, -0.8519283f];
        float[] feed13 = [-0.2670085f, -0.9534051f, 0.28634885f, 0.93654203f, 0.9747834f, -0.14879523f, -1.5747236f, 0.99790245f, -1.1938162f, 0.9022896f, -0.8663206f, 0.3107173f, 1.0289081f, 1.3187166f, 0.5506851f, -0.7555348f];

        var (i20, f20) = Run(new RngConfig { MasterSeed = 123 });
        for (int i = 0; i < 16; i++)
        {
            Assert.Equal(init20[i], i20[i], 1e-6f);
            Assert.Equal(feed20[i], f20[i], 1e-6f);
        }

        var (i13, f13) = Run(new RngConfig { MasterSeed = 123, Algorithm = RngAlgorithm.Threefry2x32Rounds13 });
        for (int i = 0; i < 16; i++)
        {
            Assert.Equal(init13[i], i13[i], 1e-6f);
            Assert.Equal(feed13[i], f13[i], 1e-6f);
        }
    }
}

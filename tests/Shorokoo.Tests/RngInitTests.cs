using System;
using System.Linq;
using Shorokoo.Core.Rng;
using Shorokoo.Modules.Initializers;
using Shorokoo.Modules.Layers;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>Two same-shape Linear weights at distinct module paths — distinct parameters, each
/// KaimingUniform-initialized on a [4,4] weight.</summary>
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

/// <summary>A uint32 state parameter initialized with raw random bits: RandomBits inside a
/// (state) parameter initializer, keyed on the parameter's own init stream.</summary>
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

/// <summary>A TRAINABLE float32 parameter built from BOTH a uniform draw and a raw-bits draw
/// (bits → cast → float): a non-float intermediate and two RNG ops in one initializer, each
/// keyed to its own sub-stream.</summary>
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

/// <summary>A UniformRange-initialized parameter whose bounds arrive as hyperparameters, so the
/// range is a runtime value the initializer cannot specialize on.</summary>
[Module]
public partial class RngUniformRangeRuntimeBounds
{
    public const int N = 256;

    public static Tensor<float32> Inline(
        Tensor<float32> x, [Hyper] Scalar<float32> low, [Hyper] Scalar<float32> high)
        => UniformRange.Init([Scalar((long)N)], low, high);
}

/// <summary>The same runtime range through the public runtime feed — the Scalar-bound overload of
/// <c>Globals.RandomUniform</c>.</summary>
[Module]
public partial class RngRuntimeFeedRuntimeBounds
{
    public static Tensor<float32> Inline(
        Tensor<float32> x, [Hyper] Scalar<float32> low, [Hyper] Scalar<float32> high)
        => RandomUniform([Scalar((long)RngUniformRangeRuntimeBounds.N)], low, high);
}

/// <summary>
/// End-to-end coverage for per-parameter initialization RNG: same-shape parameters differ,
/// initialization is reproducible for a config, the master seed re-randomizes everything, and
/// Params overrides must match a real parameter.
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

    // Both Linear weights are [4,4] = 16 elements.
    private static float[][] InitWeights(RngConfig? cfg = null) => ConcreteArch()
        .InitializeTrainableParams(rngConfig: cfg).ModelParams
        .Select(p => p.ToTensorData().As<float32>().AccessMemory().ToArray())
        .Where(v => v.Length == 16)
        .ToArray();

    private static float[] Materialize(ComputationGraph g, RngConfig cfg)
    {
        var sample = TensorData([4L, 4L], Enumerable.Repeat(1f, 16).ToArray());
        return g.ToConcreteArchitecture(g.FromOrderedInputs([sample]))
            .InitializeTrainableParams(rngConfig: cfg).ModelParams
            .Select(p => p.ToTensorData().As<float32>().AccessMemory().ToArray())
            .Single(v => v.Length == 16);
    }

    private static uint[] MaterializeBitsState(RngConfig cfg)
    {
        var g = RngBitsInitLayer.ComputationGraph;
        var sample = TensorData([4L, 4L], Enumerable.Repeat(1f, 16).ToArray());
        return g.ToConcreteArchitecture(g.FromOrderedInputs([sample]))
            .InitializeTrainableParams(rngConfig: cfg).ModelParams
            .Select(p => p.ToTensorData())
            .Where(td => td.DType == DType.UInt32)
            .SelectMany(td => td.As<uint32>().AccessMemory().ToArray())
            .ToArray();
    }

    [Fact]
    public void TestSameShapeParamsDifferAndInitIsReproducibleMasterSeededAndInKaimingBound()
    {
        var w = InitWeights();
        Assert.Equal(2, w.Length);
        Assert.False(w[0].SequenceEqual(w[1]));   // the core bug the design fixes

        // KaimingUniform bound for fanIn=4 is sqrt(6/4) ≈ 1.22474.
        foreach (var v in w)
            foreach (var x in v)
            {
                Assert.True(float.IsFinite(x));
                Assert.InRange(x, -1.2248f, 1.2248f);
            }

        var a = InitWeights(new RngConfig { MasterSeed = 123 });
        var b = InitWeights(new RngConfig { MasterSeed = 123 });
        Assert.Equal(a.Length, b.Length);
        for (int i = 0; i < a.Length; i++) Assert.Equal(a[i], b[i]);

        var s1 = InitWeights(new RngConfig { MasterSeed = 1 });
        var s2 = InitWeights(new RngConfig { MasterSeed = 2 });
        for (int i = 0; i < s1.Length; i++) Assert.False(s1[i].SequenceEqual(s2[i]));
    }

    [Fact]
    public void TestBitsAndMultiDrawInitializersMaterializeKeyedReproducibleValues()
    {
        // A float32 trainable weight whose initializer uses a uniform draw AND a bits draw.
        var bg = BitsIntermediateTrainableLayer.ComputationGraph;
        var a = Materialize(bg, new RngConfig { MasterSeed = 5 });
        Assert.Equal(a, Materialize(bg, new RngConfig { MasterSeed = 5 }));
        Assert.NotEqual(a, Materialize(bg, new RngConfig { MasterSeed = 6 }));
        Assert.All(a, v => Assert.InRange(v, 0.0f, 1.0f));   // u * bits/2^32 ∈ [0,1)
        Assert.Contains(a, v => v != 0.0f);                  // real draws, not a zeroed fallback

        // RandomBits<uint32> in a state-parameter initializer: the raw-bits analogue.
        var bits = MaterializeBitsState(new RngConfig { MasterSeed = 5 });
        Assert.Equal(16, bits.Length);
        Assert.Equal(bits, MaterializeBitsState(new RngConfig { MasterSeed = 5 }));
        Assert.NotEqual(bits, MaterializeBitsState(new RngConfig { MasterSeed = 6 }));
        Assert.Contains(bits, v => v != 0u);
    }

    [Fact]
    public void TestParamsOverrideMustMatchAParameterAndReSeedsExactlyOneStream()
    {
        var unmatched = new RngConfig { MasterSeed = 1 }.Override(RngCollection.Params, [9, 9, 9], 1UL);
        var ex = Assert.Throws<InvalidOperationException>(() => InitWeights(unmatched));
        Assert.Contains("matches no trainable parameter", ex.Message);

        var baseline = InitWeights(new RngConfig { MasterSeed = 5 });
        var firstWeightPath = ConcreteArch().GetConcreteModelParamInfos().ParamInfos
            .Single(p => p.Shape.Dims.SequenceEqual((long[])[4, 4]) && p.ModelId.Vals[0] == 1)
            .ModelId.Vals.ToArray();
        var overridden = InitWeights(
            new RngConfig { MasterSeed = 5 }.Override(RngCollection.Params, firstWeightPath, 4242UL));

        Assert.False(baseline[0].SequenceEqual(overridden[0]));   // re-seeded
        Assert.Equal(baseline[1], overridden[1]);                 // untouched
    }
}

/// <summary>
/// The init-value derivation pinned end to end — the cross-version seed contract. Every other init
/// test is relational, so a silent change anywhere in the chain (master → "init" sub-master fold →
/// per-path key fold → in-graph keyed draw → uniform transform → Kaiming scaling) would keep them
/// green while breaking every seed anyone has ever shared.
///
/// <para>Layer 1, the keys, is frozen to literal constants: a red there means "MasterSeed 123 no
/// longer produces the keys it used to" and must never be fixed by regenerating them without a
/// deliberate, breaking-change decision. Layers 2 and 3 are rebuilt from the host oracles instead,
/// which is stronger — the oracles are independent reimplementations, so they catch a graph that
/// drifts from the contract, where a frozen constant only catches a graph that drifts from its own
/// past. A red there is either a real derivation change or a deliberate one, and the oracle says
/// which.</para>
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngInitFrozenDerivationTests
{
    [Fact]
    public void TestInitKeysValuesAndBatchedResolutionAreFrozenAndMatchTheHostOracle()
    {
        // Layer 1: the key derivation alone, resolved through PRODUCTION (RngKeyResolver
        // executes the in-graph split chain) and cross-checked against the host oracle.
        var cfg = new RngConfig { MasterSeed = 123 };
        var keys = Core.Rng.RngKeyResolver.Resolve(
            [cfg.InitKeySpec((int[])[1, 1]), cfg.InitKeySpec((int[])[2, 1])]);
        Assert.Equal(0x33e150fc_0177f47cUL, keys[0]);
        Assert.Equal(0x2a93ecfc_3c6c3147UL, keys[1]);
        Assert.Equal(RngTestOracle.InitKey(cfg, (int[])[1, 1]), keys[0]);
        Assert.Equal(RngTestOracle.InitKey(cfg, (int[])[2, 1]), keys[1]);

        // The resolver folds a whole tree level per batched split (#138): specs are bucketed by
        // depth and scattered back by index, so mix depths and group sizes (including M == 1).
        var batchCfg = new RngConfig { MasterSeed = 77 };
        int[][] paths =
        [
            [1], [2], [3],
            [1, 1], [1, 2], [2, 1], [7, 9],
            [1, 2, 3],
            [4, 5, 6, 7, 8, 9],
        ];
        var batched = Core.Rng.RngKeyResolver.Resolve([.. paths.Select(p => batchCfg.InitKeySpec(p))]);
        Assert.Equal(paths.Length, batched.Count);
        for (int i = 0; i < paths.Length; i++)
            Assert.Equal(RngTestOracle.InitKey(batchCfg, paths[i]), batched[i]);
        Assert.Equal(paths.Length, batched.Distinct().Count());

        // Layer 2: the full materialized values (counter scheme, rounds, uniform transform,
        // substreamIndex ordinal, initializer bounds). REFERENCE: the dense uniform oracle, an
        // independent host rebuild — KaimingUniform draws U(-bound, bound) directly with bound =
        // sqrt(6/fanIn), and fanIn is 4 for both [4,4] weights. Exact equality is safe
        // cross-backend: the draw is Threefry integer ops plus exact bit assembly.
        float kaiming = MathF.Sqrt(6f / 4f);
        float[] expected0 = [.. Enumerable.Range(0, 16)
            .Select(i => RngDenseUniformOracle.Draw(keys[0], 0, i, -kaiming, kaiming))];
        float[] expected1 = [.. Enumerable.Range(0, 16)
            .Select(i => RngDenseUniformOracle.Draw(keys[1], 0, i, -kaiming, kaiming))];

        var g = RngInitTwoLinears.ComputationGraph;
        var sample = TensorData([4L, 4L], Enumerable.Repeat(1f, 16).ToArray());
        var ws = g.ToConcreteArchitecture(g.FromOrderedInputs([sample]))
            .InitializeTrainableParams(rngConfig: cfg).ModelParams
            .Select(p => p.ToTensorData().As<float32>().AccessMemory().ToArray())
            .Where(v => v.Length == 16).ToArray();
        Assert.Equal(2, ws.Length);
        Assert.Equal(expected0, ws[0]);   // weight at ModelId [1, 1]
        Assert.Equal(expected1, ws[1]);   // weight at ModelId [2, 1]

        // Layer 3: an initializer that draws TWICE. Both draws share the parameter's ONE stream
        // key and are separated only by their substreamIndex ordinal — rebuilding both draws from
        // the oracles pins that ordinal assignment, which the relational assertions above cannot
        // see.
        ulong multiKey = RngTestOracle.InitKey(cfg, (int[])[1]);
        float[] multiDraw = [.. Enumerable.Range(0, 16).Select(i =>
            RngDenseUniformOracle.Draw(multiKey, 0, i, 0f, 1f)
            * ((uint)RngTestOracle.DrawBits(multiKey, 1, i, 32) * (1.0f / 4294967296.0f)))];

        var mg = BitsIntermediateTrainableLayer.ComputationGraph;
        var w = mg.ToConcreteArchitecture(mg.FromOrderedInputs([sample]))
            .InitializeTrainableParams(rngConfig: cfg).ModelParams
            .Select(p => p.ToTensorData().As<float32>().AccessMemory().ToArray())
            .Single(v => v.Length == 16);
        Assert.Equal(multiDraw, w);
    }

    private static readonly RngConfig RangeCfg = new() { MasterSeed = 4242 };

    private static TensorData[] RangeInputs(float low, float high) =>
        [TensorData(DType.Float32, [], low), TensorData(DType.Float32, [], high),
         TensorData(DType.Float32, [1L], 0f)];

    private static (float[] vals, ulong key) UniformRangeParam(float low, float high)
    {
        var g = RngUniformRangeRuntimeBounds.ComputationGraph;
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([.. RangeInputs(low, high)]));
        var vals = arch.InitializeTrainableParams(rngConfig: RangeCfg).ModelParams
            .Select(p => p.ToTensorData())
            .Single(t => t.DType == DType.Float32 && t.Shape.Count == RngUniformRangeRuntimeBounds.N)
            .As<float32>().AccessMemory().ToArray();
        var path = arch.GetRngStreamReport().Streams
            .Single(s => s.Shape is { Count: 1 } sh && sh[0] == RngUniformRangeRuntimeBounds.N)
            .ModelIdPath;
        return (vals, RngTestOracle.InitKey(RangeCfg, [.. path]));
    }

    private static bool DrawsTheRange(float low, float high)
    {
        var (v, key) = UniformRangeParam(low, high);
        return v.Length == RngUniformRangeRuntimeBounds.N
            && v.All(x => float.IsFinite(x) && x >= low && x < high)
            && Enumerable.Range(0, v.Length).All(i =>
                BitConverter.SingleToUInt32Bits(v[i]) ==
                BitConverter.SingleToUInt32Bits(RngDenseUniformOracle.Draw(key, 0, i, low, high)));
    }

    private static bool FeedStaysInRange(float low, float high)
    {
        var g = RngRuntimeFeedRuntimeBounds.ComputationGraph;
        var inputs = RangeInputs(low, high);
        var model = g.ToConcreteArchitecture(g.FromOrderedInputs([.. inputs])).ToConcreteModel(RangeCfg);
        var v = ComputeContext.Default.Execute(model, [.. inputs.Cast<IData>()])[0]
            .ToTensorData().As<float32>().AccessMemory().ToArray();
        return v.Length == RngUniformRangeRuntimeBounds.N
            && v.All(x => float.IsFinite(x) && x >= low && x < high);
    }

    /// <summary>
    /// UniformRange over bounds the initializer cannot see at trace time: every draw is finite,
    /// inside [low, high), and bit-identical to the dense oracle over that same range. The last
    /// three ranges are the ones the retired affine transform u·(high−low)+low got wrong — the
    /// widest range overflowed to +Infinity (and 0·∞ to NaN), and a range narrower than one ulp
    /// of its own endpoint rounded up onto the excluded <c>high</c>.
    /// </summary>
    [Fact]
    public void TestUniformRangeDrawsItsRuntimeBoundsDenselyAndNeverReturnsHigh()
    {
        Assert.True(DrawsTheRange(2f, 5f));
        Assert.True(DrawsTheRange(-1f, 1f));
        Assert.True(DrawsTheRange(-1.8e38f, 1.8e38f));
        Assert.True(DrawsTheRange(0f, float.Epsilon));
        Assert.True(DrawsTheRange(1f, 1.0000001f));
    }

    /// <summary>The public runtime feed overload carries its graph-scalar bounds to the draw too
    /// (dropping them would draw [0, 1) and leave every range below).</summary>
    [Fact]
    public void TestRuntimeUniformFeedHonoursItsRuntimeBounds()
    {
        Assert.True(FeedStaysInRange(2f, 5f));
        Assert.True(FeedStaysInRange(-1.8e38f, 1.8e38f));
    }
}

/// <summary>Helper module holding the random draw that <see cref="RngInitNestedDrawInit"/> factors out.</summary>
[Module]
public partial class RngInitNestedDrawHelper
{
    public static Tensor<float32> Inline(Vector<int64> shape)
        => RandomUniform(shape, low: -1.0f, high: 1.0f);
}

/// <summary>A custom initializer whose random draw is nested inside a called function instead of
/// inline in its own body.</summary>
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
/// Initialization-side draws must never silently escape the keyed scheme into unkeyed backend
/// randomness. A draw factored into a called function is brought into the scheme by flattening
/// the initializer body before the noise substitution; the other escape —
/// <c>FastInitializeModelParams</c> invoked with a config but a missing/incomplete parameter
/// inventory — fails loudly instead of silently disabling the injection.
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
    public void TestConfigWithoutInventoryFailsAtEntryAndAMissingParamIsNamed()
    {
        var nullInventory = Assert.Throws<ArgumentNullException>(() =>
            Shorokoo.Core.Nodes.Processors.Fast.FastInitializeModelParams.Process(
                ConcreteArch(), null, new RngConfig { MasterSeed = 1 }, paramInfos: null));
        Assert.Contains("without the parameter inventory", nullInventory.Message);

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

    /// <summary>The ONNX fallback carries its bounds as attributes, so an unkeyed feed whose range
    /// is in-graph has nowhere to put them — a hard error, never a silently dropped range.</summary>
    [Fact]
    public void TestUnkeyedFeedWithRuntimeBoundsIsAHardError()
    {
        var g = GraphBuilder.BuildInternalComputationGraphFromDelegate(
            (Func<Tensor<float32>>)(() => RandomUniform([Scalar(4L)], Scalar(2f), Scalar(5f))));
        var feed = g.Nodes.Single(n => n.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM);
        var attrs = feed.Attributes.GetAttributeVals().ToDictionary();
        attrs[OnnxOpAttributeNames.ShrkAttrLocalModelId] = (long[])[];
        feed.Attributes = OnnxCSharpAttributes.FromCSharpVals(attrs, feed.Attributes.AttributeDefs);

        var ex = Assert.Throws<InvalidOperationException>(
            () => Shorokoo.Core.Nodes.Processors.Fast.FastLowerRandomOps.Process(g));
        Assert.Contains("cannot express a range computed in-graph", ex.Message);
    }
}

/// <summary>
/// The two normal-family consumers in one module: a KaimingNormal-initialized [4,4] weight and
/// a Globals.RandomNormal feed. The weight is kept live via a ×0 term, so with a zero input the
/// module's output equals the feed draw exactly.
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
/// the sibling of <see cref="RngInitFrozenDerivationTests"/>. Every Box–Muller composition
/// variant yields a perfect N(0,1), so the moments tests can never detect a composition change:
/// only value pins hold the convention fixed. Values are asserted at 1e-6 (ORT transcendental
/// kernels may drift in the last ULP; a composition change shifts values by O(1)).
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
        var feed = ComputeContext.Default.Execute(arch.ToConcreteModel(cfg), input)[0]
            .ToTensorData().As<float32>().AccessMemory().ToArray();
        return (init, feed);
    }

    [Fact]
    public void TestNormalInitAndDrawValuesAreFrozen()
    {
        // REFERENCE: golden — generated once from the implementation that defines the convention.
        float[] init20 = [0.32684076f, 0.31919587f, 0.71540254f, 0.47326648f, -0.53483117f, 0.82311344f, 0.76074445f, -0.22252876f, 0.1262496f, 0.32773456f, -0.33518276f, -0.5254864f, -0.36883605f, 0.08743811f, -0.22421674f, 0.13269918f];
        float[] feed20 = [-0.7269528f, 0.33580682f, -0.19701481f, -0.23019204f, 0.48736975f, -1.9013742f, 0.62898695f, -0.20801696f, -0.3274576f, 0.6395818f, -0.28467518f, 1.5134908f, 1.9615656f, 0.07030752f, -0.015374133f, -0.89534664f];
        float[] init13 = [0.80691016f, -0.120621406f, 0.7277567f, 0.6014911f, 1.019367f, 1.08257f, -0.8566765f, -0.7944496f, 0.49256578f, -0.9301721f, 0.6375022f, 0.32377562f, -0.7371908f, 0.4374376f, -0.39587018f, -0.21394667f];
        float[] feed13 = [-1.6076908f, 0.710971f, -2.258631f, 1.7556456f, 0.36330792f, 0.80508655f, -1.818318f, -0.3107102f, -1.5659105f, 0.5310641f, -1.1968004f, -0.0999485f, -0.109400675f, -0.8416264f, -0.293053f, -0.12692356f];

        var (i20, f20) = Run(new RngConfig { MasterSeed = 123 });
        var (i13, f13) = Run(new RngConfig { MasterSeed = 123, Algorithm = RngAlgorithm.Threefry2x32Rounds13 });
        for (int i = 0; i < 16; i++)
        {
            Assert.Equal(init20[i], i20[i], 1e-6f);
            Assert.Equal(feed20[i], f20[i], 1e-6f);
            Assert.Equal(init13[i], i13[i], 1e-6f);
            Assert.Equal(feed13[i], f13[i], 1e-6f);
        }
    }
}

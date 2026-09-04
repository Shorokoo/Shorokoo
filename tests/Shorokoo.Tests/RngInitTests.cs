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

/// <summary>The normal counterpart — the Scalar-parameter overload of <c>Globals.RandomNormal</c>,
/// whose mean and scale arrive as hyperparameters.</summary>
[Module]
public partial class RngRuntimeFeedRuntimeNormal
{
    public static Tensor<float32> Inline(
        Tensor<float32> x, [Hyper] Scalar<float32> mean, [Hyper] Scalar<float32> scale)
        => RandomNormal([Scalar((long)RngUniformRangeRuntimeBounds.N)], mean, scale);
}

/// <summary>An initializer drawing N(mean, scale) straight through the Scalar-parameter feed, so
/// the distribution reaches the keyed init draw as inputs rather than attributes.</summary>
[TrainableParamInitializer]
public static partial class RngNormalParamsInit
{
    public static Tensor<float32> Inline(Vector<int64> shape, Scalar<float32> mean, Scalar<float32> scale)
        => RandomNormal(shape, mean, scale);
}

/// <summary>The initializer-side counterpart of <see cref="RngRuntimeFeedRuntimeNormal"/>.</summary>
[Module]
public partial class RngNormalParamsInitLayer
{
    public static Tensor<float32> Inline(
        Tensor<float32> x, [Hyper] Scalar<float32> mean, [Hyper] Scalar<float32> scale)
        => RngNormalParamsInit.Init([Scalar((long)RngUniformRangeRuntimeBounds.N)], mean, scale);
}

/// <summary>A XavierUniformGain-initialized parameter whose gain arrives as a hyperparameter, so
/// the same site and stream key serve every gain the test materializes.</summary>
[Module]
public partial class RngXavierGainRuntimeGain
{
    public static Tensor<float32> Inline(Tensor<float32> x, [Hyper] Scalar<float32> gain)
        => XavierUniformGain.Init([Scalar(4L), Scalar(4L)], gain);
}

/// <summary>The KaimingUniformGain counterpart of <see cref="RngXavierGainRuntimeGain"/>.</summary>
[Module]
public partial class RngKaimingGainRuntimeGain
{
    public static Tensor<float32> Inline(Tensor<float32> x, [Hyper] Scalar<float32> gain)
        => KaimingUniformGain.Init([Scalar(4L), Scalar(4L)], gain);
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
/// The init-value derivation pinned to FROZEN constants — the cross-version seed contract. Every
/// other init test is relational, so a silent change anywhere in the chain (master → "init"
/// sub-master fold → per-path key fold → in-graph keyed draw → uniform transform → Kaiming
/// bounds) would keep them green while breaking every seed anyone has ever shared. A red here
/// means "MasterSeed 123 no longer produces the weights it used to" and must never be fixed by
/// regenerating the constants without a deliberate, breaking-change decision.
///
/// <para>Recomputing any layer's expectation from the oracles would forfeit exactly that: a
/// coordinated change to graph and oracle would pass silently. The oracles cross-check the same
/// constants from a separate Fact instead, so both properties hold at once.</para>
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngInitFrozenDerivationTests
{
    // MasterSeed 123: the two [4,4] KaimingUniform weights at ModelId [1, 1] and [2, 1], and the
    // two-draw initializer's [4,4] weight.
    private static readonly float[] FrozenWeight11 = [0.3378866f, 1.2052094f, 0.37335718f, 0.7997707f, 1.0675026f, -0.9041115f, 0.10083773f, -0.7819222f, -1.023829f, -0.70204467f, -0.3367729f, 0.31696287f, -1.0910206f, -0.6310536f, 0.6859728f, 1.1874193f];
    private static readonly float[] FrozenWeight21 = [1.0079714f, 0.1395562f, -0.016731672f, -1.0395613f, 0.21706164f, -0.5711288f, 0.91078484f, 1.1091135f, 0.9176603f, 0.4229469f, -0.14127223f, 0.5303491f, 0.1532544f, 0.57974875f, 0.9801079f, -1.1877112f];
    private static readonly float[] FrozenMultiDraw = [0.31585765f, 0.21880347f, 0.17880033f, 0.23017395f, 0.2856346f, 0.55870205f, 0.14583084f, 0.17104696f, 0.5075757f, 0.074125335f, 0.2884086f, 0.12671219f, 0.017829021f, 0.14411132f, 0.33035496f, 0.088769004f];

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
        // substreamIndex ordinal, initializer bounds). REFERENCE: golden. Exact equality is safe
        // cross-backend — the draw is Threefry integer ops plus exact bit assembly.
        var g = RngInitTwoLinears.ComputationGraph;
        var sample = TensorData([4L, 4L], Enumerable.Repeat(1f, 16).ToArray());
        var ws = g.ToConcreteArchitecture(g.FromOrderedInputs([sample]))
            .InitializeTrainableParams(rngConfig: cfg).ModelParams
            .Select(p => p.ToTensorData().As<float32>().AccessMemory().ToArray())
            .Where(v => v.Length == 16).ToArray();
        Assert.Equal(2, ws.Length);
        Assert.Equal(FrozenWeight11, ws[0]);
        Assert.Equal(FrozenWeight21, ws[1]);

        // Layer 3: an initializer that draws TWICE. Both draws share the parameter's ONE stream
        // key and are separated only by their substreamIndex ordinal — this golden pins that
        // ordinal assignment, which the relational assertions above cannot see.
        var mg = BitsIntermediateTrainableLayer.ComputationGraph;
        var w = mg.ToConcreteArchitecture(mg.FromOrderedInputs([sample]))
            .InitializeTrainableParams(rngConfig: cfg).ModelParams
            .Select(p => p.ToTensorData().As<float32>().AccessMemory().ToArray())
            .Single(v => v.Length == 16);
        Assert.Equal(FrozenMultiDraw, w);
    }

    // The host oracles are independent reimplementations, so holding the same frozen constants up
    // to them catches a graph that drifts from the CONTRACT, where the freeze above only catches a
    // graph that drifts from its own past. KaimingUniform draws U(-bound, bound) directly with
    // bound = sqrt(6/fanIn), and fanIn is 4 for both [4,4] weights.
    [Fact]
    public void TestTheFrozenInitValuesAreWhatTheHostOraclesIndependentlyDerive()
    {
        var cfg = new RngConfig { MasterSeed = 123 };
        float kaiming = MathF.Sqrt(6f / 4f);
        float[] Kaiming(int[] path)
        {
            ulong key = RngTestOracle.InitKey(cfg, path);
            return [.. Enumerable.Range(0, 16)
                .Select(i => RngDenseUniformOracle.Draw(key, 0, i, -kaiming, kaiming))];
        }
        Assert.Equal(FrozenWeight11, Kaiming([1, 1]));
        Assert.Equal(FrozenWeight21, Kaiming([2, 1]));

        ulong multiKey = RngTestOracle.InitKey(cfg, (int[])[1]);
        float[] multiDraw = [.. Enumerable.Range(0, 16).Select(i =>
            RngDenseUniformOracle.Draw(multiKey, 0, i, 0f, 1f)
            * ((uint)RngTestOracle.DrawBits(multiKey, 1, i, 32) * (1.0f / 4294967296.0f)))];
        Assert.Equal(FrozenMultiDraw, multiDraw);
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

    private static bool FeedDrawsTheRange(float low, float high)
    {
        var g = RngRuntimeFeedRuntimeBounds.ComputationGraph;
        var inputs = RangeInputs(low, high);
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([.. inputs]));
        var v = ComputeContext.Default.Execute(arch.ToConcreteModel(RangeCfg), [.. inputs.Cast<IData>()])[0]
            .ToTensorData().As<float32>().AccessMemory().ToArray();
        var path = arch.GetRngStreamReport().Streams
            .Single(s => s.Kind == RngStreamKind.UniformFeed).ModelIdPath;
        ulong key = RngTestOracle.RunKey(RangeCfg, [.. path]);
        return v.Length == RngUniformRangeRuntimeBounds.N
            && v.All(x => float.IsFinite(x) && x >= low && x < high)
            && Enumerable.Range(0, v.Length).All(i =>
                BitConverter.SingleToUInt32Bits(v[i]) ==
                BitConverter.SingleToUInt32Bits(RngDenseUniformOracle.Draw(key, 0, i, low, high)));
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

    /// <summary>The public runtime feed overload carries its graph-scalar bounds to the draw too,
    /// and draws bit-for-bit what the dense oracle does over them — a range check alone stays green
    /// under drift that changes every drawn bit, and under dropped bounds wherever the range happens
    /// to contain [0, 1).</summary>
    [Fact]
    public void TestRuntimeUniformFeedDrawsItsRuntimeBoundsBitForBit()
    {
        Assert.True(FeedDrawsTheRange(2f, 5f));
        Assert.True(FeedDrawsTheRange(-1f, 1f));
        Assert.True(FeedDrawsTheRange(-1.8e38f, 1.8e38f));
        Assert.True(FeedDrawsTheRange(1f, 1.0000001f));
    }

    private static float[] NormalFeed(float mean, float scale)
    {
        var g = RngRuntimeFeedRuntimeNormal.ComputationGraph;
        var inputs = RangeInputs(mean, scale);
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([.. inputs]));
        return ComputeContext.Default.Execute(arch.ToConcreteModel(RangeCfg), [.. inputs.Cast<IData>()])[0]
            .ToTensorData().As<float32>().AccessMemory().ToArray();
    }

    private static float[] NormalInit(float mean, float scale)
    {
        var g = RngNormalParamsInitLayer.ComputationGraph;
        return g.ToConcreteArchitecture(g.FromOrderedInputs([.. RangeInputs(mean, scale)]))
            .InitializeTrainableParams(rngConfig: RangeCfg).ModelParams
            .Select(p => p.ToTensorData())
            .Single(t => t.DType == DType.Float32 && t.Shape.Count == RngUniformRangeRuntimeBounds.N)
            .As<float32>().AccessMemory().ToArray();
    }

    private static bool AffineMapsTheStandardDraw(Func<float, float, float[]> draw, float mean, float scale)
    {
        var z = draw(0f, 1f);
        var v = draw(mean, scale);
        return z.Length == RngUniformRangeRuntimeBounds.N && z.Distinct().Count() > 1
            && Enumerable.Range(0, z.Length).All(i =>
                MathF.Abs(v[i] - (z[i] * scale + mean)) <= 1e-5f * (1f + MathF.Abs(v[i])));
    }

    /// <summary>The public runtime feed's graph-scalar overload carries mean and scale to the draw
    /// itself: the same stream, mapped by exactly the values the model was handed at run time. A
    /// dropped scale, a dropped mean, and dropping both each leave a distinguishable output.</summary>
    [Fact]
    public void TestRuntimeNormalFeedDrawsItsRuntimeMeanAndScale()
    {
        Assert.True(AffineMapsTheStandardDraw(NormalFeed, 3f, 2f));
        Assert.True(AffineMapsTheStandardDraw(NormalFeed, -1.5f, 0.25f));
        Assert.True(AffineMapsTheStandardDraw(NormalFeed, 0f, 7f));
        Assert.True(AffineMapsTheStandardDraw(NormalFeed, 9f, 1f));
        Assert.True(NormalFeed(100f, 0f).All(x => x == 100f));
    }

    /// <summary>The same graph-scalar parameters through a parameter initializer, keyed on the
    /// parameter's own init stream.</summary>
    [Fact]
    public void TestNormalInitializerDrawsItsRuntimeMeanAndScale()
    {
        Assert.True(AffineMapsTheStandardDraw(NormalInit, 3f, 2f));
        Assert.True(AffineMapsTheStandardDraw(NormalInit, -1.5f, 0.25f));
        Assert.True(NormalInit(100f, 0f).All(x => x == 100f));
    }

    /// <summary>One built model, re-parameterized per execution without a rebuild.</summary>
    [Fact]
    public void TestRuntimeNormalFeedReparameterizesWithoutRebuild()
    {
        var g = RngRuntimeFeedRuntimeNormal.ComputationGraph;
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([.. RangeInputs(0f, 1f)]));
        var model = arch.ToConcreteModel(RangeCfg);
        float[] Run(float mean, float scale) => ComputeContext.Default
            .Execute(model, [.. RangeInputs(mean, scale).Cast<IData>()])[0]
            .ToTensorData().As<float32>().AccessMemory().ToArray();

        var z = Run(0f, 1f);
        Assert.True(z.Distinct().Count() > 1);
        foreach (var (mean, scale) in ((float, float)[])[(7f, 4f), (-2f, 0.5f), (0f, 1f)])
            Assert.All(Enumerable.Range(0, z.Length).Zip(Run(mean, scale)),
                p => Assert.Equal(z[p.First] * scale + mean, p.Second, 1e-4f));
    }

    /// <summary>The operator's two forms are the same two draw inputs: the graph-scalar overload
    /// wires <c>mean</c>/<c>scale</c> as node inputs and sets no attributes, the literal overload
    /// sets the attributes and wires no inputs.</summary>
    [Fact]
    public void TestNormalFeedCarriesItsParametersAsInputsOrAttributesNeverBoth()
    {
        static Shorokoo.Core.Graph.FastNode Feed(Func<Tensor<float32>> f) => GraphBuilder
            .BuildInternalComputationGraphFromDelegate(f).Nodes
            .Single(n => n.OpCode == InternalOpCodes.SHRK_RANDOM_NORMAL);

        var tensorForm = Feed(() => RandomNormal([Scalar(4L)], Scalar(3f), Scalar(2f)));
        Assert.NotNull(tensorForm.Inputs[4]);
        Assert.NotNull(tensorForm.Inputs[5]);
        Assert.Null(tensorForm.Attributes.GetFloatVal(OnnxOpAttributeNames.AttrMean));
        Assert.Null(tensorForm.Attributes.GetFloatVal(OnnxOpAttributeNames.AttrScale));

        var literalForm = Feed(() => RandomNormal([Scalar(4L)], 3f, 2f));
        Assert.True(literalForm.Inputs.Count < 5 || literalForm.Inputs[4] is null);
        Assert.True(literalForm.Inputs.Count < 6 || literalForm.Inputs[5] is null);
        Assert.Equal(3f, literalForm.Attributes.GetFloatVal(OnnxOpAttributeNames.AttrMean));
        Assert.Equal(2f, literalForm.Attributes.GetFloatVal(OnnxOpAttributeNames.AttrScale));
    }

    private static float[] GainDraw(ComputationGraph g, float gain)
    {
        TensorData[] inputs =
            [TensorData(DType.Float32, [], gain), TensorData(DType.Float32, [1L], 0f)];
        return g.ToConcreteArchitecture(g.FromOrderedInputs([.. inputs]))
            .InitializeTrainableParams(rngConfig: RangeCfg).ModelParams
            .Select(p => p.ToTensorData())
            .Single(t => t.DType == DType.Float32 && t.Shape.Count == 16)
            .As<float32>().AccessMemory().ToArray();
    }

    private static bool GainIsSignAgnostic(ComputationGraph g, float gain)
    {
        var pos = GainDraw(g, gain);
        return pos.Length == 16 && pos.Distinct().Count() > 1 && pos.SequenceEqual(GainDraw(g, -gain));
    }

    /// <summary>U(-b, b) is symmetric, so a negative <c>gain</c> must draw the distribution its
    /// magnitude does. Passing the signed bound straight through instead inverts the range, and an
    /// inverted range yields <c>low</c> for every element — a silent constant fill.</summary>
    [Fact]
    public void TestGainInitializersTakeTheBoundsMagnitudeAndNeverFillConstant()
    {
        Assert.True(GainIsSignAgnostic(RngXavierGainRuntimeGain.ComputationGraph, 2.5f));
        Assert.True(GainIsSignAgnostic(RngKaimingGainRuntimeGain.ComputationGraph, 1.4142135f));
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

    private static string UnkeyedTensorParamFeedError(Func<Tensor<float32>> draw, string opCode)
    {
        var g = GraphBuilder.BuildInternalComputationGraphFromDelegate(draw);
        var feed = g.Nodes.Single(n => n.OpCode == opCode);
        var attrs = feed.Attributes.GetAttributeVals().ToDictionary();
        attrs[OnnxOpAttributeNames.ShrkAttrLocalModelId] = (long[])[];
        feed.Attributes = OnnxCSharpAttributes.FromCSharpVals(attrs, feed.Attributes.AttributeDefs);
        return Assert.Throws<InvalidOperationException>(
            () => Shorokoo.Core.Nodes.Processors.Fast.FastLowerRandomOps.Process(g)).Message;
    }

    /// <summary>The ONNX fallback carries its distribution as attributes, so an unkeyed feed whose
    /// parameters are in-graph has nowhere to put them — a hard error, never silently dropped.</summary>
    [Fact]
    public void TestUnkeyedFeedWithRuntimeDistributionIsAHardError()
    {
        var uniform = UnkeyedTensorParamFeedError(
            () => RandomUniform([Scalar(4L)], Scalar(2f), Scalar(5f)), InternalOpCodes.SHRK_RANDOM_UNIFORM);
        Assert.Contains("cannot express one computed in-graph", uniform);
        Assert.Contains(InternalOpCodes.SHRK_RANDOM_UNIFORM, uniform);

        var normal = UnkeyedTensorParamFeedError(
            () => RandomNormal([Scalar(4L)], Scalar(2f), Scalar(5f)), InternalOpCodes.SHRK_RANDOM_NORMAL);
        Assert.Contains("cannot express one computed in-graph", normal);
        Assert.Contains(InternalOpCodes.SHRK_RANDOM_NORMAL, normal);
    }

    /// <summary>An id-bearing feed with no key chain is reachable from public API — a module
    /// output handed to <c>OnnxEngine.Eval</c>, and a ConcreteArchitecture handed to
    /// <c>ComputeContext.Execute</c>, which <c>RequireConcretized</c> admits. Both must fail with
    /// the product's own catchable exception; today a Debug build hits FastLowerRandomOps'
    /// <c>Debug.Assert</c> instead, which outside a test host kills the process.
    /// Tracked as Shorokoo/Shorokoo#220.</summary>
    [Fact(Skip = "Shorokoo/Shorokoo#220: a Debug.Assert on a user-reachable path fires instead of the product's own exception")]
    public void TestIdBearingFeedWithoutKeyChainFailsWithACatchableExceptionNotAnAssertion()
    {
        var sample = TensorData([1L, 4L], 1f, 2f, 3f, 4f);
        var moduleGraph = RngInitTwoLinears.ComputationGraph;
        var arch = moduleGraph.ToConcreteArchitecture(moduleGraph.FromOrderedInputs([sample]));

        Assert.IsType<InvalidOperationException>(Record.Exception(
            () => OnnxEngine.Eval(RngInitTwoLinears.Call(Tensor([1L, 4L], 1f, 2f, 3f, 4f)))));
        Assert.IsType<InvalidOperationException>(Record.Exception(
            () => ComputeContext.Default.Execute(arch, sample)));
        Assert.Null(Record.Exception(
            () => ComputeContext.Default.Execute(arch.ToConcreteModel(), sample)));
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
/// the sibling of <see cref="RngInitFrozenDerivationTests"/>. Any composition of the draw yields
/// a perfect N(0,1), so the moments tests can never detect a composition change: only value pins
/// hold the convention fixed. Values are asserted at 1e-6 — the draw itself is bit-exact, but
/// KaimingNormal scales it by an in-graph <c>√(2/fanIn)</c> whose kernel may drift in the last
/// ULP; a composition change shifts values by O(1).
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

    // REFERENCE: goldens — generated once from the implementation that defines each convention.
    private static readonly (RngAlgorithm Algorithm, float[] Init, float[] Feed)[] Frozen =
    [
        (RngAlgorithm.Threefry2x32,
            [-0.46155134f, 1.1823097f, 1.7006428f, -1.0548751f, 0.5553944f, -0.037981175f, -0.32655385f, 0.43091658f, -0.5052699f, 0.1411822f, -0.09921032f, -0.61696446f, -0.23968527f, 0.22614606f, -0.3482982f, -2.1913755f],
            [-1.4262838f, 0.78313416f, -0.22856675f, 0.07745038f, 0.3881657f, -0.40209898f, -0.44436496f, -0.5162606f, -0.7766776f, 1.1438937f, 1.1135756f, -1.682844f, 0.5379951f, 0.4911052f, -0.78660655f, 1.0551703f]),
        (RngAlgorithm.Threefry2x32Rounds13,
            [-0.648924f, -0.62196916f, 0.61931646f, 0.067791395f, 0.11458076f, 0.49879357f, 0.59926426f, -0.3936729f, 1.1705743f, 0.8123495f, -0.38091052f, 0.3156036f, 0.8826961f, -0.22946525f, -0.3551115f, -0.5530873f],
            [0.61773705f, -0.8109559f, 0.041261587f, -0.8039563f, -1.0408375f, -1.5433217f, 0.52736086f, -0.2665317f, 0.89446217f, -1.389097f, 0.74119544f, -0.67193127f, -0.91855776f, 0.46575305f, -0.7654304f, -2.0355716f]),
    ];

    [Fact]
    public void TestNormalInitAndDrawValuesAreFrozenPerAlgorithm()
    {
        foreach ((RngAlgorithm algorithm, float[] init, float[] feed) in Frozen)
        {
            var (i, f) = Run(new RngConfig { MasterSeed = 123, Algorithm = algorithm });
            for (int k = 0; k < 16; k++)
            {
                Assert.Equal(init[k], i[k], 1e-6f);
                Assert.Equal(feed[k], f[k], 1e-6f);
            }
        }
    }
}

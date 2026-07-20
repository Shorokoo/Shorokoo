using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage tests for the modern normalization / parametric-activation layers
/// (RMSNorm, PReLU, GatedLinear.GLU), in the <see cref="NNLibraryCoverageTests"/>
/// one-liner style: each [Fact] drives a self-checking module from
/// NormActTestModules.cs through <see cref="AutoTest.AdvancedTestGraph{TModule}"/>
/// (ONNX roundtrip, CS codegen, QEE). Value correctness lives inside the
/// modules (they return Scalar&lt;bit&gt;); see the module docs for the closed
/// forms checked.
/// </summary>
[Trait("Domain", "Modules")]
[Trait("Purpose", "Coverage")]
public class NormActModuleTests
{
    /// <summary>[i * scale + offset for i in 0..N) as a float32 TensorData — varied, non-degenerate values.</summary>
    private static TensorData RangeTensor(long[] dims, float scale = 1f, float offset = 0f)
    {
        long total = 1;
        foreach (var d in dims) total *= d;
        return TensorData(DType.Float32, dims,
            Enumerable.Range(0, (int)total).Select(i => (object)(i * scale + offset)).ToArray());
    }

    [Fact]
    public void TestRMSNormCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<RMSNormNormalizes>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 4L], 1.5f, -3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<RMSNormMatchesManual>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 4L], 0.7f, 1f)]));
    }

    [Fact]
    public void TestPReLUCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<PReLUClosedForm>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [7L], -3f, -1f, -0.5f, 0f, 0.5f, 1f, 3f)]));
    }

    /// <summary>
    /// PReLUChannelwise closed-form value coverage (design §7-1): at init (every
    /// per-channel slope = 0.25) the layer equals the relu-built reference
    /// <c>relu(x) − a·relu(−x)</c> with a hand-built <c>[1, C, 1, …]</c> constant
    /// 0.25 slope, at BOTH rank 4 (<c>[N,C,H,W]</c>) and rank 2 (<c>[N,C]</c>, the
    /// spatial-collapsed view inside <see cref="NNPReLUChannelwiseClosedForm"/>).
    /// Channel count C = 3 in both. This pins the relu form + the rank-generic
    /// <c>[C] → [1, C, 1, …]</c> broadcast; it does NOT discriminate per-channel vs
    /// shared (their inits coincide) — that is the rig param-shape / divergence
    /// checks in <see cref="NormActTrainingCoverageTests"/>.
    /// </summary>
    [Fact]
    public void TestPReLUChannelwiseClosedFormCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<NNPReLUChannelwiseClosedForm>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 3L, 2L, 2L], 0.5f, -3f)]));
    }

    [Fact]
    public void TestGLUCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<GLUMatchesManual>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 6L], 0.5f, -2f)]));
    }

    /// <summary>
    /// Design §7-1/§7-3 — the new param-free <c>[Module] GLU</c> (baked <c>dim = -1</c>):
    /// <see cref="GLUModuleMatchesManual"/> asserts <c>GLU.Call(x)</c> equals an
    /// INDEPENDENT hand-split <c>a · sigmoid(b)</c> reference (the module = the math),
    /// and <see cref="GLUModuleEqualsHelper"/> asserts the forwarder is bit-for-bit
    /// identical to the underlying <c>GatedLinear.GLU(x, -1)</c> helper. Both are driven
    /// at two ranks — <c>[2, 6]</c> (last axis 6 → 3) and <c>[2, 3, 4]</c> (last axis
    /// 4 → 2) — confirming the rank-generic last-axis split the module delegates to.
    /// Distinct method/module names from <see cref="GLUMatchesManual"/> (the helper check).
    /// </summary>
    [Fact]
    public void TestGluModuleClosedFormCoverage()
    {
        // Module ≡ independent hand-split reference, rank 2 then rank 3.
        Assert.True(AutoTest.AdvancedTestGraph<GLUModuleMatchesManual>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 6L], 0.5f, -2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<GLUModuleMatchesManual>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 4L], 0.3f, -1.5f)]));

        // Module forwarder ≡ static helper, bit-for-bit, same two ranks.
        Assert.True(AutoTest.AdvancedTestGraph<GLUModuleEqualsHelper>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 6L], 0.5f, -2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<GLUModuleEqualsHelper>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 4L], 0.3f, -1.5f)]));
    }

    /// <summary>
    /// Design §7-2 — output shape: the <c>[Module] GLU</c> halves the LAST axis,
    /// <c>[N, …, 2H] → [N, …, H]</c>, preserving the leading dims (asserted via
    /// <c>ShapeTensor</c> inside <see cref="GLUModuleHalvesLastAxis"/>). Driven at
    /// rank 2 (<c>[2, 6] → [2, 3]</c>) and rank 3 (<c>[2, 3, 4] → [2, 3, 2]</c>) to
    /// confirm the rank-genericity of the split.
    /// </summary>
    [Fact]
    public void TestGluModuleHalvesLastAxisCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<GLUModuleHalvesLastAxis>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 6L], 0.5f, -2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<GLUModuleHalvesLastAxis>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 4L], 0.3f, -1.5f)]));
    }

    /// <summary>
    /// LocalResponseNorm design §7-1 (the load-bearing parity anchor +
    /// closed-form value): <see cref="NNLocalResponseNormMatchesOp"/> asserts the
    /// hand-rolled primitive <c>[Module] LocalResponseNorm</c> equals the native
    /// ONNX <c>LRN</c> op (<see cref="LRNHelper.Lrn{T}"/>) at the ONNX/PyTorch
    /// defaults — the single most important check (primitive forward ≡ ORT kernel);
    /// <see cref="NNLocalResponseNormClosedForm"/> asserts the module equals an
    /// independent hand-built reference (Pad axis-1 + channel Slice-sum of x²,
    /// <c>pool = k + (α/5)·Σ</c>, <c>x·pool^(−β)</c>) with NON-default
    /// <c>α/β/k = 2e-4/0.5/2</c>, exercising the clamped EDGE channel (0) and the
    /// full-window INTERIOR channel (2). Driven on a <c>[1,5,2,2]</c> input with
    /// distinct ascending values.
    /// </summary>
    [Fact]
    public void TestLocalResponseNormParityAndClosedFormCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<NNLocalResponseNormMatchesOp>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 5L, 2L, 2L], 0.3f, -1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNLocalResponseNormClosedForm>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 5L, 2L, 2L], 0.3f, -1f)]));
    }

    /// <summary>
    /// LocalResponseNorm design §7-3 (alpha/beta/k are LIVE hypers + the
    /// arbitrary-size helper path):
    /// <see cref="NNLocalResponseNormHypersLive"/> drives the module with
    /// non-default <c>α/β/k = 1e-3/0.5/2</c> and asserts it tracks the native op
    /// with the SAME params (proving the float hypers are not baked to their
    /// defaults); <see cref="NNLrnHelperArbitrarySizeClosedForm"/> asserts
    /// <see cref="LRNHelper.Lrn{T}"/> with a NON-baked <c>size=3</c> matches a hand
    /// reference, covering the arbitrary-size path the module (baked <c>size=5</c>)
    /// lacks. (An even <c>size</c> is not tested: the ONNX Runtime CPU LRN kernel
    /// asserts <c>size % 2 == 1</c> and refuses even windows — an external-backend
    /// limitation, not a Shorokoo bug; see the module doc.) Both on a
    /// <c>[1,5,2,2]</c> input.
    /// </summary>
    [Fact]
    public void TestLocalResponseNormHypersAndSizeCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<NNLocalResponseNormHypersLive>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 5L, 2L, 2L], 0.3f, -1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNLrnHelperArbitrarySizeClosedForm>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 5L, 2L, 2L], 0.3f, -1f)]));
    }
}

/// <summary>
/// Training-rig coverage for the modern norm / activation layers, in the
/// <see cref="NNLibraryTrainingCoverageTests"/> style: each model is a tiny
/// no-hyper wrapper (layer hypers fixed via Model(...)) trained one step with
/// L2Loss + SGD; the loss must be finite and at least one trainable param must
/// move (gradient flowed through the layer).
/// </summary>
[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class NormActTrainingCoverageTests
{
    private static TensorDataStruct MakeBatch(string fieldName, string structName, TensorData data)
    {
        var def = new TensorStructDef(
            new[] { new TensorStructFieldDef(fieldName, DataStructure.Tensor, data.Shape.Dims.Length, data.DType) },
            structName);
        return new TensorDataStruct(def, new Dictionary<string, IData> { { fieldName, data } });
    }

    private static float[] Floats(IData data) => ((TensorData<float32>)data).AccessMemory().ToArray();

    /// <summary>One SGD step of <paramref name="modelGraph"/> + L2Loss against a zero target:
    /// asserts the loss is finite and at least one trainable param moved.</summary>
    private static void AssertTrainsAndMovesAParam(InternalComputationGraph modelGraph, long[] inShape, float[] input)
    {
        var inputData = TensorData(inShape, input);
        long rows = inShape[0];
        var targetData = TensorData([rows], new float[rows]);

        var rig = TrainingRig.FromScratch(
            modelGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            new NamedModelParam[] { new TensorDataModelParam("input", ModelParamType.InputParam, inputData) },
            0.01f);

        var ctx = new ComputeContext();
        var compiled = ctx.Compile(rig.TrainingStepPureGraph);
        var initial = rig.CreateDefaultCheckpoint();
        var step = rig.TrainStep(initial,
            MakeBatch("input", "ModelInput", inputData),
            MakeBatch("targets", "Target", targetData),
            compiled);

        Assert.True(float.IsFinite(step.Loss), $"training-step loss must be finite; got {step.Loss}");
        Assert.NotEmpty(rig.TrainableParamStructDef.Fields);
        bool anyMoved = rig.TrainableParamStructDef.Fields.Any(field =>
        {
            var before = Floats(initial.TrainableParams.Fields[field.Name]);
            var after = Floats(step.Checkpoint.TrainableParams.Fields[field.Name]);
            return before.Zip(after).Any(p => MathF.Abs(p.First - p.Second) > 1e-9f);
        });
        Assert.True(anyMoved, "no trainable param moved — no gradient flowed through the layer");
    }

    [Fact]
    public void TestRMSNormTrainsAndMovesGain()
        => AssertTrainsAndMovesAParam(NormActRMSNormModel.ComputationGraph, [3L, 4L],
            Enumerable.Range(0, 12).Select(i => i * 0.5f - 2f).ToArray());

    [Fact]
    public void TestPReLUTrainsAndMovesAlpha()
        => AssertTrainsAndMovesAParam(NormActPReLUModel.ComputationGraph, [3L, 4L],
            Enumerable.Range(0, 12).Select(i => i * 0.5f - 3f).ToArray());

    /// <summary>
    /// Design §7-4 (param-free rig smoke): GLU has NO trainable param of its own, so the
    /// rig is driven through <see cref="NormActGLUModel"/>, which fronts GLU with a tiny
    /// trainable scalar pre-weight <c>w</c> (the <see cref="NNInstanceNormAffineFalseParamModel"/>
    /// trick): <c>[N, 2H] → (scale by w) → GLU → [N, H] → per-row mean → [N]</c>. After one
    /// SGD step the loss must be finite and the upstream scalar <c>w</c> must move — which it
    /// can ONLY do if a finite gradient flowed back THROUGH the differentiable
    /// Split/Sigmoid/Mul gate. Confirms GLU composes inside a trainable model and backprops.
    /// Input <c>[3, 4]</c> (H = 2; last axis 4 → 2).
    /// </summary>
    [Fact]
    public void TestGluModelTrainsThroughGate()
        => AssertTrainsAndMovesAParam(NormActGLUModel.ComputationGraph, [3L, 4L],
            Enumerable.Range(0, 12).Select(i => i * 0.5f - 3f).ToArray());

    /// <summary>
    /// LocalResponseNorm design §7-4 (param-free rig smoke): LRN has NO trainable
    /// param of its own, so the rig is driven through <see cref="NormActLRNModel"/>,
    /// which fronts the LRN primitive graph with a tiny trainable scalar pre-weight
    /// <c>w</c> (the <see cref="NNInstanceNormAffineFalseParamModel"/> / GLU trick):
    /// <c>[N, C, H, W] → (scale by w) → LocalResponseNorm → per-sample mean → [N]</c>.
    /// After one SGD step the loss must be finite and the upstream scalar <c>w</c>
    /// must move — which it can ONLY do if a finite gradient flowed back THROUGH the
    /// differentiable LRN graph (the analytic LRN VJP / the primitive Pad/Slice/Pow
    /// chain). Confirms LocalResponseNorm composes inside a trainable model and
    /// backprops. Input <c>[2, 5, 2, 2]</c> (N=2, C=5, 2×2 spatial).
    /// </summary>
    [Fact]
    public void TestLocalResponseNormModelTrainsThroughLayer()
        => AssertTrainsAndMovesAParam(NormActLRNModel.ComputationGraph, [2L, 5L, 2L, 2L],
            Enumerable.Range(0, 40).Select(i => i * 0.3f - 1f).ToArray());

    /// <summary>
    /// Per-channel-vs-shared DISCRIMINATOR (design §7 check 2, the load-bearing one):
    /// because both modules init every slope to 0.25 they are numerically identical at
    /// init, so a value check cannot tell them apart — the slope param's SHAPE does.
    /// Build a rig over <see cref="NormActPReLUChannelwiseModel"/> on a <c>[N, C]</c>
    /// input with C = 4 and assert its single trainable slope field materializes to C
    /// (4) elements; build the sibling <see cref="NormActPReLUSharedSlopeModel"/> (shared
    /// <see cref="PReLU"/>, identical wrapper) on the SAME input and assert its slope
    /// field is <c>[1]</c> (1 element). The C-vs-1 element count is read off the
    /// materialized checkpoint param (as <see cref="NNLibraryTrainingCoverageTests"/>'s
    /// non-scalar-param helper does), the only check that distinguishes the two modules.
    /// </summary>
    [Fact]
    public void TestPReLUChannelwiseSlopeIsPerChannel()
    {
        const long c = 4L;
        var input = Enumerable.Range(0, (int)(3L * c)).Select(i => i * 0.5f - 3f).ToArray();
        var inputData = TensorData([3L, c], input);

        NamedModelParam[] Inputs() => new NamedModelParam[]
            { new TensorDataModelParam("input", ModelParamType.InputParam, inputData) };

        // Per-channel: the single trainable slope materializes to [C] (4 elements).
        var cwRig = TrainingRig.FromScratch(
            NormActPReLUChannelwiseModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            Inputs(), 0.01f);
        Assert.Single(cwRig.TrainableParamStructDef.Fields);
        var cwSlope = cwRig.TrainableParamStructDef.Fields[0];
        var cwInit = cwRig.CreateDefaultCheckpoint();
        Assert.Equal((int)c, Floats(cwInit.TrainableParams.Fields[cwSlope.Name]).Length);

        // Shared sibling: its slope is [1] (1 element) regardless of C.
        var sharedRig = TrainingRig.FromScratch(
            NormActPReLUSharedSlopeModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            Inputs(), 0.01f);
        Assert.Single(sharedRig.TrainableParamStructDef.Fields);
        var sharedSlope = sharedRig.TrainableParamStructDef.Fields[0];
        var sharedInit = sharedRig.CreateDefaultCheckpoint();
        Assert.Equal(1, Floats(sharedInit.TrainableParams.Fields[sharedSlope.Name]).Length);
    }

    /// <summary>
    /// Per-channel divergence after a TrainStep (design §7 check 3, the strongest
    /// discriminator): a shared <c>[1]</c> slope can never produce different
    /// per-channel slopes, but a true per-channel <c>[C]</c> slope can. Train
    /// <see cref="NormActPReLUChannelwiseModel"/> one step on a <c>[N, C]</c> input
    /// engineered so different channels see different-signed activations: channel 0 is
    /// all-negative, channel 1 all-positive (channels 2/3 mixed). The PReLU slope
    /// gradient for channel <c>c</c> is <c>Σ_n min(0, x_{n,c})</c>, so the all-negative
    /// channel gets a non-zero slope gradient while the all-positive channel gets ~0 —
    /// after one SGD step the post-step <c>[C]</c> slope entries are NOT all equal
    /// (they all start at 0.25). At least two channels must differ, proving the slopes
    /// are independent per channel.
    /// </summary>
    [Fact]
    public void TestPReLUChannelwiseSlopesDivergeAfterStep()
    {
        const long n = 3L, c = 4L;
        // Per-channel-asymmetric input ([N, C], row-major): channel 0 all-negative,
        // channel 1 all-positive, channels 2/3 mixed — so the per-channel slope
        // gradients Σ_n min(0, x_{n,c}) differ across channels.
        float[] input =
        {
            -1f,  2f, -0.5f, 1.5f,   // row 0: ch0<0, ch1>0, ch2<0, ch3>0
            -2f,  3f,  0.5f, -1f,    // row 1
            -3f,  1f, -1.5f,  2f,    // row 2
        };
        var inputData = TensorData([n, c], input);
        var targetData = TensorData([n], new float[n]);

        var rig = TrainingRig.FromScratch(
            NormActPReLUChannelwiseModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            new NamedModelParam[] { new TensorDataModelParam("input", ModelParamType.InputParam, inputData) },
            0.1f);

        var ctx = new ComputeContext();
        var compiled = ctx.Compile(rig.TrainingStepPureGraph);
        var initial = rig.CreateDefaultCheckpoint();

        // Sanity: the single slope param is [C] and starts uniform at 0.25.
        Assert.Single(rig.TrainableParamStructDef.Fields);
        string slopeName = rig.TrainableParamStructDef.Fields[0].Name;
        float[] slope0 = Floats(initial.TrainableParams.Fields[slopeName]);
        Assert.Equal((int)c, slope0.Length);
        Assert.All(slope0, v => Assert.True(MathF.Abs(v - 0.25f) < 1e-5f, $"slope init must be 0.25; got {v}"));

        var step = rig.TrainStep(initial,
            MakeBatch("input", "ModelInput", inputData),
            MakeBatch("targets", "Target", targetData),
            compiled);
        Assert.True(float.IsFinite(step.Loss), $"training-step loss must be finite; got {step.Loss}");

        float[] slope1 = Floats(step.Checkpoint.TrainableParams.Fields[slopeName]);
        Assert.Equal((int)c, slope1.Length);

        // The post-step [C] slopes must NOT all be equal — at least two channels differ
        // (a shared [1] slope could only ever produce one value). Compare against the
        // first channel's post-step value with a tolerance above float noise.
        bool diverged = slope1.Skip(1).Any(v => MathF.Abs(v - slope1[0]) > 1e-6f);
        Assert.True(diverged,
            $"post-step per-channel slopes must differ; got [{string.Join(", ", slope1)}]");
    }
}

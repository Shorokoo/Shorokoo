using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// Forward coverage for the modern normalization / parametric-activation layers (RMSNorm, PReLU,
/// GatedLinear.GLU, LocalResponseNorm): each row drives a self-checking module from
/// NormActTestModules.cs through <see cref="AutoTest.AdvancedTestGraph{TModule}"/> (ONNX roundtrip,
/// CS codegen, QEE). Value correctness lives inside the modules, which return
/// <c>Scalar&lt;bit&gt;</c>; see the module docs for the closed forms checked.
/// </summary>
[Trait("Domain", "Modules")]
[Trait("Purpose", "Coverage")]
public class NormActModuleTests
{
    /// <summary>[i * scale + offset for i in 0..N) as a float32 TensorData.</summary>
    private static TensorData RangeTensor(long[] dims, float scale = 1f, float offset = 0f)
    {
        long total = 1;
        foreach (var d in dims) total *= d;
        return TensorData(DType.Float32, dims,
            Enumerable.Range(0, (int)total).Select(i => (object)(i * scale + offset)).ToArray());
    }

    private static void Run<TModule>(long[] dims, float scale, float offset)
        => Assert.True(AutoTest.AdvancedTestGraph<TModule>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor(dims, scale, offset)]));

    [Fact]
    public void TestNormActForwardCoverage()
    {
        Run<RMSNormNormalizes>([2L, 4L], 1.5f, -3f);
        Run<RMSNormMatchesManual>([2L, 4L], 0.7f, 1f);

        Assert.True(AutoTest.AdvancedTestGraph<PReLUClosedForm>(hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [7L], -3f, -1f, -0.5f, 0f, 0.5f, 1f, 3f)]));
        // PReLUChannelwise at init (every per-channel slope 0.25) == relu(x) − a·relu(−x) with a
        // hand-built [1, C, 1, …] 0.25 slope; pins the rank-generic [C] → [1, C, 1, …] broadcast.
        Run<NNPReLUChannelwiseClosedForm>([1L, 3L, 2L, 2L], 0.5f, -3f);

        Run<GLUMatchesManual>([2L, 6L], 0.5f, -2f);
        // The param-free [Module] GLU (baked dim = -1) against an independent hand-split
        // a · sigmoid(b) reference, and its forwarder against the GatedLinear.GLU(x, -1) helper —
        // both at rank 2 (last axis 6 → 3) and rank 3 (last axis 4 → 2).
        Run<GLUModuleMatchesManual>([2L, 6L], 0.5f, -2f);
        Run<GLUModuleMatchesManual>([2L, 3L, 4L], 0.3f, -1.5f);
        Run<GLUModuleEqualsHelper>([2L, 6L], 0.5f, -2f);
        Run<GLUModuleEqualsHelper>([2L, 3L, 4L], 0.3f, -1.5f);
        // Output shape: [N, …, 2H] → [N, …, H], leading dims preserved, at both ranks.
        Run<GLUModuleHalvesLastAxis>([2L, 6L], 0.5f, -2f);
        Run<GLUModuleHalvesLastAxis>([2L, 3L, 4L], 0.3f, -1.5f);

        // LocalResponseNorm: the primitive [Module] against the native ONNX LRN op at the
        // ONNX/PyTorch defaults, against an independent Pad/Slice-sum reference with non-default
        // α/β/k, with α/β/k driven as LIVE hypers, and the arbitrary-size (size=3) helper path.
        Run<NNLocalResponseNormMatchesOp>([1L, 5L, 2L, 2L], 0.3f, -1f);
        Run<NNLocalResponseNormClosedForm>([1L, 5L, 2L, 2L], 0.3f, -1f);
        Run<NNLocalResponseNormHypersLive>([1L, 5L, 2L, 2L], 0.3f, -1f);
        Run<NNLrnHelperArbitrarySizeClosedForm>([1L, 5L, 2L, 2L], 0.3f, -1f);
    }
}

/// <summary>
/// Training-rig coverage for the modern norm / activation layers: each model is a tiny no-hyper
/// wrapper (layer hypers fixed via Model(...)) trained one step with L2Loss + SGD; the loss must
/// be finite and at least one trainable param must move. The param-free layers (GLU, LRN) are
/// fronted with a trainable scalar pre-weight, which can only move if a finite gradient flowed
/// back through the layer.
/// </summary>
[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class NormActTrainingCoverageTests
{
    private static TensorDataStruct MakeBatch(string fieldName, string structName, TensorData data)
    {
        var def = new TensorStructDef(
            [new TensorStructFieldDef(fieldName, DataStructure.Tensor, data.Shape.Dims.Length, data.DType)],
            structName);
        return new TensorDataStruct(def, new Dictionary<string, IData> { { fieldName, data } });
    }

    private static float[] Floats(IData data) => ((TensorData<float32>)data).AccessMemory().ToArray();

    private static float[] Ramp(int count, float scale, float offset)
        => [.. Enumerable.Range(0, count).Select(i => i * scale + offset)];

    private static void AssertTrainsAndMovesAParam(ComputationGraph modelGraph, long[] inShape, float[] input)
    {
        var inputData = TensorData(inShape, input);
        long rows = inShape[0];
        var targetData = TensorData([rows], new float[rows]);

        var rig = TrainingRig.FromScratch(
            modelGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, inputData)],
            0.01f);

        var initial = rig.CreateInitialCheckpoint();
        var step = rig.TrainStep(initial,
            MakeBatch("input", "ModelInput", inputData),
            MakeBatch("targets", "Target", targetData));

        Assert.True(float.IsFinite(step.Loss!.Value));
        Assert.NotEmpty(rig.TrainableParamStructDef.Fields);
        Assert.True(rig.TrainableParamStructDef.Fields.Any(field =>
        {
            var before = Floats(initial.TrainableParams.Fields[field.Name]);
            var after = Floats(step.TrainableParams.Fields[field.Name]);
            return before.Zip(after).Any(p => MathF.Abs(p.First - p.Second) > 1e-9f);
        }));
    }

    [Fact]
    public void TestNormActTrainsAndMovesAParam()
    {
        AssertTrainsAndMovesAParam(NormActRMSNormModel.ComputationGraph, [3L, 4L], Ramp(12, 0.5f, -2f));
        AssertTrainsAndMovesAParam(NormActPReLUModel.ComputationGraph, [3L, 4L], Ramp(12, 0.5f, -3f));
        // GLU has no trainable param of its own: [N, 2H] → scale by w → GLU → [N, H] → row mean.
        AssertTrainsAndMovesAParam(NormActGLUModel.ComputationGraph, [3L, 4L], Ramp(12, 0.5f, -3f));
        // Likewise LRN: [N, C, H, W] → scale by w → LocalResponseNorm → per-sample mean → [N].
        AssertTrainsAndMovesAParam(NormActLRNModel.ComputationGraph, [2L, 5L, 2L, 2L], Ramp(40, 0.3f, -1f));
    }

    /// <summary>
    /// Per-channel-vs-shared discriminator: both modules init every slope to 0.25, so only the
    /// slope param's SHAPE tells them apart — channelwise materializes to [C], shared to [1].
    /// </summary>
    [Fact]
    public void TestPReLUChannelwiseSlopeIsPerChannel()
    {
        const long c = 4L;
        var inputData = TensorData([3L, c], Ramp((int)(3L * c), 0.5f, -3f));
        NamedModelParam[] Inputs() => [new TensorDataModelParam("input", ModelParamType.InputParam, inputData)];

        var cwRig = TrainingRig.FromScratch(
            NormActPReLUChannelwiseModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            Inputs(), 0.01f);
        Assert.Single(cwRig.TrainableParamStructDef.Fields);
        var cwSlope = cwRig.TrainableParamStructDef.Fields[0];
        Assert.Equal((int)c, Floats(cwRig.CreateInitialCheckpoint().TrainableParams.Fields[cwSlope.Name]).Length);

        var sharedRig = TrainingRig.FromScratch(
            NormActPReLUSharedSlopeModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            Inputs(), 0.01f);
        Assert.Single(sharedRig.TrainableParamStructDef.Fields);
        var sharedSlope = sharedRig.TrainableParamStructDef.Fields[0];
        Assert.Equal(1, Floats(sharedRig.CreateInitialCheckpoint().TrainableParams.Fields[sharedSlope.Name]).Length);
    }

    /// <summary>
    /// Per-channel divergence after a TrainStep: the PReLU slope gradient for channel c is
    /// Σ_n min(0, x_{n,c}), so on an input whose channel 0 is all-negative and channel 1
    /// all-positive the post-step [C] slopes must not all be equal — which a shared [1] slope
    /// could never produce.
    /// </summary>
    [Fact]
    public void TestPReLUChannelwiseSlopesDivergeAfterStep()
    {
        const long n = 3L, c = 4L;
        float[] input =
        [
            -1f,  2f, -0.5f, 1.5f,
            -2f,  3f,  0.5f, -1f,
            -3f,  1f, -1.5f,  2f,
        ];
        var inputData = TensorData([n, c], input);
        var targetData = TensorData([n], new float[n]);

        var rig = TrainingRig.FromScratch(
            NormActPReLUChannelwiseModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, inputData)],
            0.1f);

        var initial = rig.CreateInitialCheckpoint();
        Assert.Single(rig.TrainableParamStructDef.Fields);
        string slopeName = rig.TrainableParamStructDef.Fields[0].Name;
        float[] slope0 = Floats(initial.TrainableParams.Fields[slopeName]);
        Assert.Equal((int)c, slope0.Length);
        Assert.All(slope0, v => Assert.True(MathF.Abs(v - 0.25f) < 1e-5f));

        var step = rig.TrainStep(initial,
            MakeBatch("input", "ModelInput", inputData),
            MakeBatch("targets", "Target", targetData));
        Assert.True(float.IsFinite(step.Loss!.Value));

        float[] slope1 = Floats(step.TrainableParams.Fields[slopeName]);
        Assert.Equal((int)c, slope1.Length);
        Assert.True(slope1.Skip(1).Any(v => MathF.Abs(v - slope1[0]) > 1e-6f));
    }
}

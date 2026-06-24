using Shorokoo.Runtime;
using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage for the classic layers added on top of the baseline NN library:
/// Conv3d (NCDHW). Conv3d is value-checked the same way as the other conv
/// variants — its self-checking module (NNConv3dMatchesStaticConv) returns a
/// Scalar&lt;bit&gt; comparing the dynamic SHRK_CONV geometry to a static-attribute
/// NN.Conv with identical geometry and weights — driven through
/// <see cref="AutoTest.AdvancedTestGraph{TModule}"/> (ONNX roundtrip, CS codegen, QEE).
/// BatchNorm1d is covered by the rig-based <see cref="ClassicLayerTrainingCoverageTests"/>
/// instead (its StateUpdate links are not executable in the plain inference pipeline).
/// </summary>
[Trait("Domain", "Modules")]
[Trait("Purpose", "Coverage")]
public class ClassicLayerModuleTests
{
    /// <summary>[i * scale + offset for i in 0..N) as a float32 TensorData — varied, non-degenerate values for the conv check.</summary>
    private static TensorData RangeTensor(long[] dims, float scale = 1f, float offset = 0f)
    {
        long total = 1;
        foreach (var d in dims) total *= d;
        return TensorData(DType.Float32, dims,
            Enumerable.Range(0, (int)total).Select(i => (object)(i * scale + offset)).ToArray());
    }

    [Fact]
    public void TestConv3dLayerCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<NNConv3dMatchesStaticConv>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 5L, 5L, 5L], 0.05f, -2f)]));
    }
}

/// <summary>
/// Training-rig coverage for BatchNorm1d, in the
/// <see cref="NNLibraryTrainingCoverageTests"/> style: a tiny [N, C]
/// training-mode model is driven through TrainingRig.FromScratch + a single
/// TrainStep, asserting the model state is populated, the loss is finite, and
/// the running statistics are EMA-updated by the training pass. BatchNorm1d
/// carries StateUpdate links, so it cannot be exercised via AutoTest (the plain
/// inference pipeline has no STATE_UPDATE_LINK op).
/// </summary>
[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class ClassicLayerTrainingCoverageTests
{
    private static TensorDataStruct MakeBatch(string fieldName, string structName, TensorData data)
    {
        var def = new TensorStructDef(
            new[] { new TensorStructFieldDef(fieldName, DataStructure.Tensor, data.Shape.Dims.Length, data.DType) },
            structName);
        return new TensorDataStruct(def, new Dictionary<string, IData> { { fieldName, data } });
    }

    private static float[] Floats(IData data) => ((TensorData<float32>)data).AccessMemory().ToArray();

    /// <summary>
    /// BatchNorm1d training path over a [4, 3] input: the model state
    /// (running mean/var) is populated at init; one TrainStep produces a finite
    /// loss; and the training pass EMA-updates the running statistics (every
    /// running-stat model-state field moves).
    /// </summary>
    [Fact]
    public void TestBatchNorm1dTrainModeUpdatesRunningStats()
    {
        var vals = Enumerable.Range(0, 12).Select(i => (float)i).ToArray();
        var inputData = TensorData([4L, 3L], vals);
        var targetData = TensorData([3L], new float[] { 0f, 0f, 0f });

        var rig = TrainingRig.FromScratch(
            NNBatchNorm1dTrainGradModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            new NamedModelParam[] { new TensorDataModelParam("input", ModelParamType.InputParam, inputData) },
            0.01f);

        var initial = rig.CreateDefaultCheckpoint();
        Assert.NotEmpty(initial.ModelState.Fields);

        var ctx = new ComputeContext();
        var compiled = ctx.Compile(rig.TrainingStepPureGraph);
        var step = rig.TrainStep(initial,
            MakeBatch("input", "ModelInput", inputData),
            MakeBatch("targets", "Target", targetData),
            compiled);

        Assert.True(float.IsFinite(step.Loss), $"BatchNorm1d training-mode loss must be finite; got {step.Loss}");

        // The training pass must EMA-update the running statistics.
        Assert.NotEmpty(rig.ModelStateDef.Fields);
        foreach (var field in rig.ModelStateDef.Fields)
        {
            var before = Floats(initial.ModelState.Fields[field.Name]);
            var after = Floats(step.Checkpoint.ModelState.Fields[field.Name]);
            Assert.True(before.Zip(after).Any(p => MathF.Abs(p.First - p.Second) > 1e-7f),
                $"running stat '{field.Name}' was not updated by a training-mode pass");
        }
    }
}

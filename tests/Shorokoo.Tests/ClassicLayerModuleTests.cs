using Shorokoo.Runtime;
using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage for the classic layers added on top of the baseline NN library. Conv3d is
/// value-checked by the self-checking <c>NNConv3dForwardGolden</c> module; BatchNorm1d
/// carries StateUpdate links (no STATE_UPDATE_LINK op in the plain inference pipeline)
/// so it is covered by <see cref="ClassicLayerTrainingCoverageTests"/> instead.
/// </summary>
[Trait("Domain", "Modules")]
[Trait("Purpose", "Coverage")]
public class ClassicLayerModuleTests
{
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
        Assert.True(AutoTest.AdvancedTestGraph<NNConv3dForwardGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 5L, 5L, 5L], 0.05f, -2f)]));
    }
}

/// <summary>
/// Training-rig coverage for BatchNorm1d: a tiny [N, C] training-mode model driven
/// through TrainingRig.FromScratch + a single TrainStep.
/// </summary>
[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class ClassicLayerTrainingCoverageTests
{
    private static TensorDataStruct MakeBatch(string fieldName, string structName, TensorData data)
    {
        TensorStructFieldDef[] fields =
            [new TensorStructFieldDef(fieldName, DataStructure.Tensor, data.Shape.Dims.Length, data.DType)];
        return new TensorDataStruct(new TensorStructDef(fields, structName),
            new Dictionary<string, IData> { { fieldName, data } });
    }

    private static float[] Floats(IData data) => ((TensorData<float32>)data).AccessMemory().ToArray();

    [Fact]
    public void TestBatchNorm1dTrainModeStatePopulatedFiniteLossAndRunningStatsUpdated()
    {
        var vals = Enumerable.Range(0, 12).Select(i => (float)i).ToArray();
        var inputData = TensorData([4L, 3L], vals);
        float[] targets = [0f, 0f, 0f];
        var targetData = TensorData([3L], targets);

        NamedModelParam[] sample =
            [new TensorDataModelParam("input", ModelParamType.InputParam, inputData)];

        var rig = TrainingRig.FromScratch(
            NNBatchNorm1dTrainGradModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, 0.01f);

        var initial = rig.CreateInitialCheckpoint();
        Assert.NotEmpty(initial.ModelState.Fields);

        var step = rig.TrainStep(initial,
            MakeBatch("input", "ModelInput", inputData),
            MakeBatch("targets", "Target", targetData));

        Assert.True(float.IsFinite(step.Loss!.Value));

        Assert.NotEmpty(rig.ModelStateDef.Fields);
        foreach (var field in rig.ModelStateDef.Fields)
        {
            var before = Floats(initial.ModelState.Fields[field.Name]);
            var after = Floats(step.ModelState.Fields[field.Name]);
            Assert.True(before.Zip(after).Any(p => MathF.Abs(p.First - p.Second) > 1e-7f));
        }
    }
}

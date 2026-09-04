using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage for the KLDivLoss closed forms and the extra initializers
/// (TruncatedNormal / LeCunNormal); the exact closed forms live inside the
/// self-checking modules in LossInitTestModules.cs.
/// </summary>
[Trait("Domain", "Modules")]
[Trait("Purpose", "Coverage")]
public class LossInitModuleTests
{
    [Fact]
    public void TestKLDivLossClosedFormAndInitializerPropsCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<KLDivClosedForm>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [1L], 0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<InitializerProps>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [1L], 0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<ScalarInitializerValues>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [1L], 0f)]));
    }
}

/// <summary>
/// Training-rig smoke coverage for KLDivLoss: it satisfies the rig's
/// (predictions, targets) → scalar loss contract, so it composes through
/// <see cref="TrainingRig.FromScratch"/> + one <c>TrainStep</c>.
/// </summary>
[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class LossInitTrainingTests
{
    [Fact]
    public void TestKLDivLossThroughTrainingRigProducesFiniteLoss()
    {
        float[] logProbs = [-0.6931472f, -0.6931472f];
        float[] probs = [0.5f, 0.5f];
        var inputData = TensorData([2L], logProbs);
        var targetData = TensorData([2L], probs);

        NamedModelParam[] sample =
            [new TensorDataModelParam("input", ModelParamType.InputParam, inputData)];

        var rig = TrainingRig.FromScratch(
            Shorokoo.Tests.Modules.ScalarMultiplyModel.ComputationGraph,
            KLDivLoss.ComputationGraph,
            SGDOptimizer.ComputationGraph,
            sample,
            0.01f);

        var initial = rig.CreateInitialCheckpoint();

        TensorStructFieldDef[] inputFields =
            [new TensorStructFieldDef("input", DataStructure.Tensor, 1, DType.Float32)];
        TensorStructFieldDef[] targetFields =
            [new TensorStructFieldDef("targets", DataStructure.Tensor, 1, DType.Float32)];

        var step = rig.TrainStep(
            initial,
            new TensorDataStruct(new TensorStructDef(inputFields, "ModelInput"),
                new Dictionary<string, IData> { { "input", inputData } }),
            new TensorDataStruct(new TensorStructDef(targetFields, "Target"),
                new Dictionary<string, IData> { { "targets", targetData } }));

        Assert.NotNull(step);
        Assert.NotNull(step.TrainableParams);
        Assert.True(float.IsFinite(step.Loss!.Value));
    }
}

/// <summary>
/// Coverage for the rank-0 trainable-parameter initializers: a parameter created by
/// <c>ScalarZeros</c>/<c>ScalarOnes</c>/<c>ScalarConstant</c> is a true rank-0 scalar in
/// the checkpoint (not a <c>[1]</c>-shaped rank-1 tensor), and it trains.
/// </summary>
[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class ScalarInitializerTrainingTests
{
    [Fact]
    public void TestScalarInitializerYieldsRank0TrainableParamThatTrains()
    {
        float[] input = [1f, 2f, 3f, 4f];
        float[] target = [0f, 0f, 0f, 0f];
        var rig = TrainingRig.FromScratch(
            Shorokoo.Tests.Modules.ScalarGainModel.ComputationGraph,
            L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData([4L], input))],
            0.01f);

        var initial = rig.CreateInitialCheckpoint();
        var field = Assert.Single(rig.TrainableParamStructDef.Fields);
        Assert.Equal(0, field.Rank);

        var w0 = (TensorData<float32>)initial.TrainableParams.Fields[field.Name];
        Assert.Empty(w0.Shape.Dims);
        Assert.Equal(1f, w0.AccessMemory()[0]);

        TensorStructFieldDef[] inputFields =
            [new TensorStructFieldDef("input", DataStructure.Tensor, 1, DType.Float32)];
        TensorStructFieldDef[] targetFields =
            [new TensorStructFieldDef("targets", DataStructure.Tensor, 1, DType.Float32)];
        var step = rig.TrainStep(
            initial,
            new TensorDataStruct(new TensorStructDef(inputFields, "ModelInput"),
                new Dictionary<string, IData> { { "input", TensorData([4L], input) } }),
            new TensorDataStruct(new TensorStructDef(targetFields, "Target"),
                new Dictionary<string, IData> { { "targets", TensorData([4L], target) } }));

        var w1 = (TensorData<float32>)step.TrainableParams.Fields[field.Name];
        Assert.Empty(w1.Shape.Dims);
        Assert.True(float.IsFinite(step.Loss!.Value));
        Assert.True(MathF.Abs(w1.AccessMemory()[0] - 1f) > 1e-7f);
    }

    [Fact]
    public void TestRank0TrainableParamAndRank0ModuleStateCoexist()
    {
        float[] input = [1f, 2f, 3f, 4f];
        var rig = TrainingRig.FromScratch(
            Shorokoo.Tests.Modules.ScalarGainWithScalarStateModel.ComputationGraph,
            L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData([4L], input))],
            0.01f);

        var initial = rig.CreateInitialCheckpoint();
        var param = Assert.Single(rig.TrainableParamStructDef.Fields);
        var state = Assert.Single(rig.ModelStateDef.Fields);
        Assert.Equal(0, param.Rank);
        Assert.Equal(0, state.Rank);
        Assert.Empty(((TensorData<float32>)initial.TrainableParams.Fields[param.Name]).Shape.Dims);
        Assert.Empty(((TensorData<float32>)initial.ModelState.Fields[state.Name]).Shape.Dims);
    }
}

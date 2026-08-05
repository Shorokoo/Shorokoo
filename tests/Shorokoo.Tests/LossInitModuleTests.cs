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

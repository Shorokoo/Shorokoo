using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage for the KLDivLoss closed forms and the extra initializers
/// (TruncatedNormal / LeCunNormal), in the one-liner self-checking-module style
/// (each [Fact] drives a Scalar&lt;bit&gt; module from LossInitTestModules.cs
/// through <see cref="AutoTest.AdvancedTestGraph{TModule}"/>). The exact closed
/// forms / properties checked live inside the modules.
/// </summary>
[Trait("Domain", "Modules")]
[Trait("Purpose", "Coverage")]
public class LossInitModuleTests
{
    [Fact]
    public void TestKLDivLossClosedFormCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<KLDivClosedForm>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [1L], 0f)]));
    }

    [Fact]
    public void TestInitializerPropsCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<InitializerProps>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [1L], 0f)]));
    }
}

/// <summary>
/// Training-rig smoke coverage for KLDivLoss: it satisfies the rig's
/// (predictions, targets) → scalar loss contract, so it composes through
/// <see cref="TrainingRig.FromScratch"/> + one <c>TrainStep</c>. The model
/// (<see cref="Shorokoo.Tests.Modules.ScalarMultiplyModel"/>, weight init 1.0)
/// passes the input through unchanged, so feeding log-probabilities as input
/// keeps the KL term well-defined; the targets are a valid probability
/// distribution (non-negative, summing to 1). The step's loss must be finite.
/// </summary>
[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class LossInitTrainingTests
{
    [Fact]
    public void TestKLDivLossThroughTrainingRig()
    {
        // input = log-probabilities of a uniform 2-way distribution (ln 0.5).
        var inputData = TensorData([2L], new float[] { -0.6931472f, -0.6931472f });
        // targets = valid probabilities (non-negative, sum to 1).
        var targetData = TensorData([2L], new float[] { 0.5f, 0.5f });

        var rig = TrainingRig.FromScratch(
            Shorokoo.Tests.Modules.ScalarMultiplyModel.ComputationGraph,
            KLDivLoss.ComputationGraph,
            SGDOptimizer.ComputationGraph,
            new NamedModelParam[]
            {
                new TensorDataModelParam("input", ModelParamType.InputParam, inputData),
            },
            0.01f);

        var initial = rig.CreateDefaultCheckpoint();

        var modelInputDef = new TensorStructDef(
            new[] { new TensorStructFieldDef("input", DataStructure.Tensor, 1, DType.Float32) },
            "ModelInput");
        var targetDef = new TensorStructDef(
            new[] { new TensorStructFieldDef("targets", DataStructure.Tensor, 1, DType.Float32) },
            "Target");

        var inputBatch = new TensorDataStruct(modelInputDef,
            new Dictionary<string, IData> { { "input", inputData } });
        var targetBatch = new TensorDataStruct(targetDef,
            new Dictionary<string, IData> { { "targets", targetData } });

        var ctx = new ComputeContext();
        var compiled = ctx.Compile(rig.TrainingStepPureGraph);
        var step = rig.TrainStep(initial, inputBatch, targetBatch, compiled);

        Assert.NotNull(step.Checkpoint);
        Assert.NotNull(step.Checkpoint.TrainableParams);
        Assert.True(float.IsFinite(step.Loss),
            $"KLDivLoss must produce a finite loss through the rig; got {step.Loss}");
    }
}

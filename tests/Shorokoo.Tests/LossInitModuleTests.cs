using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;
using Shorokoo.Runtime;
using static Shorokoo.Tests.NNLibraryTrainingFixtures;

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
/// Rank-0 parameters through the training rig: a parameter from <c>ScalarZeros</c>/<c>ScalarOnes</c>,
/// and module-owned state from a rank-0 <c>[StateInitializer]</c>, are true rank-0 scalars in the
/// checkpoint rather than <c>[1]</c>-shaped rank-1 tensors, survive a loop body one per iteration
/// slot, and train. None of them takes a shape input — the case the shape derivation and the
/// definition-vs-reference resolution in trainable-param lowering have to get right without one.
/// (The seeded values of all three, <c>ScalarConstant</c> included, are pinned through the ONNX
/// round trip by <c>ScalarInitializerValues</c> above.)
/// </summary>
[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class Rank0ParamTrainingTests
{
    [Fact]
    public void TestRank0ParamsAndStateAreShapelessAndTrain()
    {
        float[] input = [1f, 2f, 3f, 4f];
        var rig = TrainingRig.FromScratch(
            Rank0ScalarModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData([4L], input))],
            0.01f);
        var initial = rig.CreateInitialCheckpoint();

        var gain = rig.TrainableParamStructDef.Fields.Single(f => f.Name.Contains("ScalarOnes"));
        var bias = rig.TrainableParamStructDef.Fields.Single(f => f.Name.Contains("ScalarZeros"));
        var calls = Assert.Single(rig.ModelStateDef.Fields);
        int?[] ranks = [gain.Rank, bias.Rank, calls.Rank];
        Assert.Equal<int?>([0, 0, 0], ranks);
        Assert.Equal<float>([1f], Floats(initial.TrainableParams.Fields[gain.Name]));
        Assert.Equal<float>([0f], Floats(initial.TrainableParams.Fields[bias.Name]));
        Assert.Equal<float>([0f], Floats(initial.ModelState.Fields[calls.Name]));

        var step = rig.TrainStep(initial,
            MakeBatch("input", "ModelInput", TensorData([4L], input)),
            MakeBatch("targets", "Target", TensorData([4L], new float[4])));

        Assert.True(float.IsFinite(step.Loss!.Value));
        Assert.True(AnyParamMoved(rig, initial, step));
        Assert.Equal<float>([1f], Floats(step.ModelState.Fields[calls.Name]));
    }

    [Fact]
    public void TestALoopBodyRealizesOneRank0ParamPerIterationSlot()
    {
        var rig = TrainingRig.FromScratch(
            Rank0ParamsInLoopModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData([4L], new float[4]))],
            0.01f);
        var initial = rig.CreateInitialCheckpoint();

        var fields = rig.TrainableParamStructDef.Fields;   // 2 params x 3 trips, all distinct
        float[] seeds = [.. fields.Select(f => Floats(initial.TrainableParams.Fields[f.Name]).Single())];
        Assert.Equal<float>([1f, 1f, 1f, 0f, 0f, 0f], seeds);
        Assert.All(fields, f => Assert.Equal(0, f.Rank));
    }

    [Fact]
    public void TestAnInitializerThatStatesItsShapeNowhereIsRejectedByName()
    {
        var ex = Assert.Throws<InvalidOperationException>(() => TrainingRig.FromScratch(
            ShapelessInitModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData([4L], new float[4]))],
            0.01f));
        Assert.Contains(nameof(InitShapelessZeros), ex.Message);
    }

    // Pins Shorokoo/Shorokoo#237: an initializer whose Inline hands an input straight back exports an
    // ONNX function with a nameless output, and the model fails to load with an ORT schema error that
    // names neither the initializer nor the fix. ScalarConstant works around it by writing its body as
    // `Scalar(1.0f) * value`; unskipping this must not need any change to the test.
    [Fact(Skip = "Shorokoo/Shorokoo#237: an identity initializer body exports a nameless function output")]
    public void TestAnInitializerReturningItsInputUnchangedLoads()
        => Assert.True(AutoTest.AdvancedTestGraph<IdentityInitModel>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [2L], 1f, 2f)]));
}

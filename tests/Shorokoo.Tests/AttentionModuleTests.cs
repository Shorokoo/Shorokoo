using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;
using Shorokoo.Runtime;

using static Shorokoo.Tests.TransformerTrainingFixtures;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage for the Transformer / Attention stack. The self-checking modules embed
/// their value validation inside the module's <c>Inline</c> (returning a
/// <c>Scalar&lt;bit&gt;</c>), so each AutoTest call is a one-liner asserting the check bit.
/// Inputs are per-element-distinct so the softmax is non-uniform.
/// </summary>
[Trait("Domain", "Modules")]
[Trait("Purpose", "Coverage")]
public class AttentionModuleTests
{
    private static TensorData Sdpa3x2() => TensorData(DType.Float32, [1L, 1L, 3L, 2L],
        0.1f, 0.9f, 0.5f, -0.3f, -0.7f, 0.4f);

    private static TensorData Mha3x4() => TensorData(DType.Float32, [1L, 3L, 4L],
        0.1f, 0.2f, -0.3f, 0.4f,
        0.5f, -0.6f, 0.7f, 0.8f,
        -0.9f, 0.15f, 0.25f, -0.35f);

    private static TensorData RoPE3x4() => TensorData(DType.Float32, [1L, 1L, 3L, 4L],
        0.1f, 0.9f, 0.5f, -0.3f,
        -0.7f, 0.4f, 0.2f, 0.8f,
        0.6f, -0.5f, 0.35f, -0.15f);

    private static TensorData RoPE2x4() => TensorData(DType.Float32, [1L, 1L, 2L, 4L],
        0.1f, 0.9f, 0.5f, -0.3f,
        -0.7f, 0.4f, 0.2f, 0.8f);

    private static TensorData Memory5x4() => TensorData(DType.Float32, [1L, 5L, 4L],
        0.3f, -0.1f, 0.45f, -0.2f,
        0.6f, 0.05f, -0.55f, 0.15f,
        -0.25f, 0.7f, 0.1f, -0.4f,
        0.8f, -0.3f, 0.2f, 0.55f,
        -0.65f, 0.35f, -0.05f, 0.5f);

    [Fact]
    public void TestSdpaMhaAndRoPECoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AttnSdpaForwardGolden>(
            hyperparamInputs: [], runtimeInputs: [Sdpa3x2()]));
        Assert.True(AutoTest.AdvancedTestGraph<AttnSdpaCausalMasksFuture>(
            hyperparamInputs: [], runtimeInputs: [Sdpa3x2()]));
        Assert.True(AutoTest.AdvancedTestGraph<MhaForwardGolden>(
            hyperparamInputs: [], runtimeInputs: [Mha3x4()]));
        Assert.True(AutoTest.AdvancedTestGraph<RoPEPositionZeroIsIdentity>(
            hyperparamInputs: [], runtimeInputs: [RoPE3x4()]));
        Assert.True(AutoTest.AdvancedTestGraph<RoPEPreservesNorm>(
            hyperparamInputs: [], runtimeInputs: [RoPE3x4()]));
        Assert.True(AutoTest.AdvancedTestGraph<RoPEClosedFormPositionOne>(
            hyperparamInputs: [], runtimeInputs: [RoPE2x4()]));
    }

    [Fact]
    public void TestTransformerDecoderLayerCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<DecoderLayerShapeCheck>(
            hyperparamInputs: [], runtimeInputs: [Mha3x4(), Memory5x4()]));
        Assert.True(AutoTest.AdvancedTestGraph<DecoderLayerNoBiasGolden>(
            hyperparamInputs: [], runtimeInputs: [Mha3x4(), Memory5x4()]));
        Assert.True(AutoTest.AdvancedTestGraph<DecoderLayerWithBiasGolden>(
            hyperparamInputs: [], runtimeInputs: [Mha3x4(), Memory5x4()]));
    }
}

internal static class TransformerTrainingFixtures
{
    internal static readonly TensorStructFieldDef[] TargetFields =
        [new TensorStructFieldDef("targets", DataStructure.Tensor, 2, DType.Float32)];

    internal static float[] Floats(int count, float seed)
    {
        var vals = new float[count];
        for (var i = 0; i < count; i++)
            vals[i] = seed * (((i * 7) % 11) - 5);
        return vals;
    }

    internal static bool AnyFieldChanged(TensorDataStruct before, TensorDataStruct after)
    {
        foreach (var f in before.Definition.Fields)
        {
            if (after.Fields[f.Name] is not TensorData a || before.Fields[f.Name] is not TensorData b)
                continue;
            var av = a.As<float32>().AccessMemory<float>().ToArray();
            var bv = b.As<float32>().AccessMemory<float>().ToArray();
            for (var i = 0; i < av.Length && i < bv.Length; i++)
                if (MathF.Abs(av[i] - bv[i]) > 1e-7f)
                    return true;
        }
        return false;
    }
}

/// <summary>
/// Training-rig smoke coverage for the Transformer encoder layer: a tiny mean-pooling model is
/// driven through <see cref="TrainingRig.FromScratch"/> + <c>CreateInitialCheckpoint</c> + one
/// <see cref="TrainingRig.TrainStep"/>.
/// </summary>
[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class TransformerEncoderTrainingCoverageTests
{
    [Fact]
    public void TestTransformerEncoderLayerTrainStepCoverage()
    {
        long[] inputShape = [2L, 3L, 4L];
        long[] outShape = [2L, 4L];

        NamedModelParam[] encoderSample =
        [
            new TensorDataModelParam("input", ModelParamType.InputParam,
                TensorData(inputShape, Floats(24, seed: 0.07f))),
        ];

        var encoderRig = TrainingRig.FromScratch(
            TransformerEncoderMeanPoolModel.ComputationGraph,
            L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph,
            encoderSample, 0.01f);

        var encoderInitial = encoderRig.CreateInitialCheckpoint();
        Assert.NotEmpty(encoderRig.TrainableParamStructDef.Fields);

        TensorStructFieldDef[] encoderInputFields =
            [new TensorStructFieldDef("input", DataStructure.Tensor, 3, DType.Float32)];
        var targetDef = new TensorStructDef(TargetFields, "Target");

        var encoderStep = encoderRig.TrainStep(
            encoderInitial,
            new TensorDataStruct(new TensorStructDef(encoderInputFields, "ModelInput"),
                new Dictionary<string, IData> { { "input", TensorData(inputShape, Floats(24, seed: 0.07f)) } }),
            new TensorDataStruct(targetDef,
                new Dictionary<string, IData> { { "targets", TensorData(outShape, new float[8]) } }));

        Assert.True(float.IsFinite(encoderStep.Loss!.Value));
        Assert.NotEmpty(encoderStep.TrainableParams.Fields);
        Assert.True(AnyFieldChanged(encoderInitial.TrainableParams, encoderStep.TrainableParams));
    }
}

/// <summary>
/// Training-rig smoke coverage for the Transformer decoder layer: a tiny mean-pooling model is
/// driven through <see cref="TrainingRig.FromScratch"/> + <c>CreateInitialCheckpoint</c> + one
/// <see cref="TrainingRig.TrainStep"/>.
/// </summary>
[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class TransformerDecoderTrainingCoverageTests
{
    [Fact]
    public void TestTransformerDecoderLayerTrainStepCoverage()
    {
        long[] inputShape = [2L, 3L, 4L];
        long[] memShape = [2L, 5L, 4L];
        long[] outShape = [2L, 4L];

        NamedModelParam[] decoderSample =
        [
            new TensorDataModelParam("tgt", ModelParamType.InputParam,
                TensorData(inputShape, Floats(24, seed: 0.07f))),
            new TensorDataModelParam("memory", ModelParamType.InputParam,
                TensorData(memShape, Floats(40, seed: 0.05f))),
        ];

        var decoderRig = TrainingRig.FromScratch(
            TransformerDecoderMeanPoolModel.ComputationGraph,
            L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph,
            decoderSample, 0.01f);

        var decoderInitial = decoderRig.CreateInitialCheckpoint();
        Assert.NotEmpty(decoderRig.TrainableParamStructDef.Fields);

        TensorStructFieldDef[] decoderInputFields =
        [
            new TensorStructFieldDef("tgt", DataStructure.Tensor, 3, DType.Float32),
            new TensorStructFieldDef("memory", DataStructure.Tensor, 3, DType.Float32),
        ];

        var decoderStep = decoderRig.TrainStep(
            decoderInitial,
            new TensorDataStruct(new TensorStructDef(decoderInputFields, "ModelInput"),
                new Dictionary<string, IData>
                {
                    { "tgt", TensorData(inputShape, Floats(24, seed: 0.07f)) },
                    { "memory", TensorData(memShape, Floats(40, seed: 0.05f)) },
                }),
            new TensorDataStruct(new TensorStructDef(TargetFields, "Target"),
                new Dictionary<string, IData> { { "targets", TensorData(outShape, new float[8]) } }));

        Assert.True(float.IsFinite(decoderStep.Loss!.Value));
        Assert.NotEmpty(decoderStep.TrainableParams.Fields);
        Assert.True(AnyFieldChanged(decoderInitial.TrainableParams, decoderStep.TrainableParams));
    }
}

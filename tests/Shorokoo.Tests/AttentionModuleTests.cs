using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage for the Transformer / Attention stack
/// (<see cref="Shorokoo.Modules.Layers.Attention"/>,
/// <see cref="Shorokoo.Modules.Layers.MultiHeadAttention"/>,
/// <see cref="Shorokoo.Modules.Layers.TransformerEncoderLayer"/>). The self-checking
/// modules embed their value validation inside the module's <c>Inline</c> (returning a
/// <c>Scalar&lt;bit&gt;</c>), so each AutoTest call is a one-liner asserting the check bit.
///
/// <para>SDPA/MHA inputs use per-element-distinct values (not the all-0.1 of
/// <c>TensorDataWithSmallVals</c>) so the softmax is non-uniform and the attention math is
/// genuinely exercised — equal logits would make every reference trivially match.</para>
/// </summary>
[Trait("Domain", "Modules")]
[Trait("Purpose", "Coverage")]
public class AttentionModuleTests
{
    [Fact]
    public void TestScaledDotProductAttentionCoverage()
    {
        // [1, 1, L=3, d=2] with distinct entries → a non-uniform attention pattern.
        Assert.True(AutoTest.AdvancedTestGraph<AttnSdpaMatchesManual>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [1L, 1L, 3L, 2L],
                0.1f, 0.9f, 0.5f, -0.3f, -0.7f, 0.4f)]));

        Assert.True(AutoTest.AdvancedTestGraph<AttnSdpaCausalMasksFuture>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [1L, 1L, 3L, 2L],
                0.1f, 0.9f, 0.5f, -0.3f, -0.7f, 0.4f)]));
    }

    [Fact]
    public void TestMultiHeadAttentionCoverage()
    {
        // [N=1, L=3, embedDim=4], distinct entries.
        Assert.True(AutoTest.AdvancedTestGraph<MhaMatchesManualReference>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [1L, 3L, 4L],
                0.1f, 0.2f, -0.3f, 0.4f,
                0.5f, -0.6f, 0.7f, 0.8f,
                -0.9f, 0.15f, 0.25f, -0.35f)]));
    }

    [Fact]
    public void TestRoPECoverage()
    {
        // [N=1, H=1, L=3, d=4] with per-element-distinct values → a non-trivial rotation.
        // Position 0 is the identity (mθ = 0 ⇒ cos = 1, sin = 0).
        Assert.True(AutoTest.AdvancedTestGraph<RoPEPositionZeroIsIdentity>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [1L, 1L, 3L, 4L],
                0.1f, 0.9f, 0.5f, -0.3f,
                -0.7f, 0.4f, 0.2f, 0.8f,
                0.6f, -0.5f, 0.35f, -0.15f)]));

        // RoPE is an orthogonal rotation ⇒ it preserves each position's vector norm.
        Assert.True(AutoTest.AdvancedTestGraph<RoPEPreservesNorm>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [1L, 1L, 3L, 4L],
                0.1f, 0.9f, 0.5f, -0.3f,
                -0.7f, 0.4f, 0.2f, 0.8f,
                0.6f, -0.5f, 0.35f, -0.15f)]));

        // Closed-form rotation at sequence position 1 (d = 4, base = 10000 ⇒ θ0 = 1, θ1 = 0.01):
        // pins the half-split pairing (0,2)/(1,3), the frequency formula and the sign convention.
        // Input is [1, 1, 2, 4] (L = 2 so position 1 exists).
        Assert.True(AutoTest.AdvancedTestGraph<RoPEClosedFormPositionOne>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [1L, 1L, 2L, 4L],
                0.1f, 0.9f, 0.5f, -0.3f,
                -0.7f, 0.4f, 0.2f, 0.8f)]));
    }

    [Fact]
    public void TestTransformerDecoderLayerCoverage()
    {
        // Shape: tgt [N=1, Lt=3, E=4] + memory [N=1, Lm=5, E=4] with Lt != Lm ⇒ out [1, 3, 4].
        Assert.True(AutoTest.AdvancedTestGraph<DecoderLayerShapeCheck>(
            hyperparamInputs: [],
            runtimeInputs:
            [
                TensorData(DType.Float32, [1L, 3L, 4L],
                    0.1f, 0.2f, -0.3f, 0.4f,
                    0.5f, -0.6f, 0.7f, 0.8f,
                    -0.9f, 0.15f, 0.25f, -0.35f),
                TensorData(DType.Float32, [1L, 5L, 4L],
                    0.3f, -0.1f, 0.45f, -0.2f,
                    0.6f, 0.05f, -0.55f, 0.15f,
                    -0.25f, 0.7f, 0.1f, -0.4f,
                    0.8f, -0.3f, 0.2f, 0.55f,
                    -0.65f, 0.35f, -0.05f, 0.5f),
            ]));

        // Structural closed-form re-derivation (Lt != Lm), no bias.
        Assert.True(AutoTest.AdvancedTestGraph<DecoderLayerMatchesManualNoBias>(
            hyperparamInputs: [],
            runtimeInputs:
            [
                TensorData(DType.Float32, [1L, 3L, 4L],
                    0.1f, 0.2f, -0.3f, 0.4f,
                    0.5f, -0.6f, 0.7f, 0.8f,
                    -0.9f, 0.15f, 0.25f, -0.35f),
                TensorData(DType.Float32, [1L, 5L, 4L],
                    0.3f, -0.1f, 0.45f, -0.2f,
                    0.6f, 0.05f, -0.55f, 0.15f,
                    -0.25f, 0.7f, 0.1f, -0.4f,
                    0.8f, -0.3f, 0.2f, 0.55f,
                    -0.65f, 0.35f, -0.05f, 0.5f),
            ]));

        // Same structural closed-form with useBias = true (zero biases added everywhere).
        Assert.True(AutoTest.AdvancedTestGraph<DecoderLayerMatchesManualWithBias>(
            hyperparamInputs: [],
            runtimeInputs:
            [
                TensorData(DType.Float32, [1L, 3L, 4L],
                    0.1f, 0.2f, -0.3f, 0.4f,
                    0.5f, -0.6f, 0.7f, 0.8f,
                    -0.9f, 0.15f, 0.25f, -0.35f),
                TensorData(DType.Float32, [1L, 5L, 4L],
                    0.3f, -0.1f, 0.45f, -0.2f,
                    0.6f, 0.05f, -0.55f, 0.15f,
                    -0.25f, 0.7f, 0.1f, -0.4f,
                    0.8f, -0.3f, 0.2f, 0.55f,
                    -0.65f, 0.35f, -0.05f, 0.5f),
            ]));
    }
}

/// <summary>
/// Training-rig smoke coverage for <see cref="Shorokoo.Modules.Layers.TransformerEncoderLayer"/>:
/// a tiny model (<see cref="TransformerEncoderMeanPoolModel"/>) wrapping one encoder layer is
/// driven through <see cref="TrainingRig.FromScratch"/> + <c>CreateDefaultCheckpoint</c> + one
/// <see cref="TrainingRig.TrainStep"/>, asserting a finite loss and that a trainable parameter
/// actually moved. Mirrors the helper-driven style of
/// <see cref="Shorokoo.Tests.TrainingRigCoverageTests"/>.
/// </summary>
[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class AttentionTrainingCoverageTests
{
    [Fact]
    public void TestTransformerEncoderLayerTrainStepCoverage()
    {
        long[] inputShape = [2L, 3L, 4L];   // [N, L, embedDim]
        long[] outShape = [2L, 4L];         // mean-pooled [N, embedDim]

        var sample = new NamedModelParam[]
        {
            new TensorDataModelParam("input", ModelParamType.InputParam,
                TensorData(inputShape, Floats(24, seed: 0.07f))),
        };

        var rig = TrainingRig.FromScratch(
            TransformerEncoderMeanPoolModel.ComputationGraph,
            L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph,
            sample, 0.01f);

        var initial = rig.CreateDefaultCheckpoint();
        Assert.NotEmpty(rig.TrainableParamStructDef.Fields);

        var modelInputDef = new TensorStructDef(
            new[] { new TensorStructFieldDef("input", DataStructure.Tensor, 3, DType.Float32) }, "ModelInput");
        var targetDef = new TensorStructDef(
            new[] { new TensorStructFieldDef("targets", DataStructure.Tensor, 2, DType.Float32) }, "Target");
        var inputBatch = new TensorDataStruct(modelInputDef,
            new Dictionary<string, IData> { { "input", TensorData(inputShape, Floats(24, seed: 0.07f)) } });
        var targetBatch = new TensorDataStruct(targetDef,
            new Dictionary<string, IData> { { "targets", TensorData(outShape, new float[8]) } });

        var ctx = new ComputeContext();
        var compiled = ctx.Compile(rig.TrainingStepPureGraph);
        var step = rig.TrainStep(initial, inputBatch, targetBatch, compiled);

        Assert.True(float.IsFinite(step.Loss));
        Assert.NotEmpty(step.Checkpoint.TrainableParams.Fields);
        Assert.True(AnyFieldChanged(initial.TrainableParams, step.Checkpoint.TrainableParams),
            "no trainable parameter moved after a TrainStep (gradient did not flow)");
    }

    [Fact]
    public void TestTransformerDecoderLayerTrainStepCoverage()
    {
        long[] tgtShape = [2L, 3L, 4L];     // [N, Lt, embedDim]
        long[] memShape = [2L, 5L, 4L];     // [N, Lm, embedDim]
        long[] outShape = [2L, 4L];         // mean-pooled over Lt → [N, embedDim]

        // Two graph inputs (tgt, memory) — names must match the model's Inline params.
        var sample = new NamedModelParam[]
        {
            new TensorDataModelParam("tgt", ModelParamType.InputParam,
                TensorData(tgtShape, Floats(24, seed: 0.07f))),
            new TensorDataModelParam("memory", ModelParamType.InputParam,
                TensorData(memShape, Floats(40, seed: 0.05f))),
        };

        var rig = TrainingRig.FromScratch(
            TransformerDecoderMeanPoolModel.ComputationGraph,
            L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph,
            sample, 0.01f);

        var initial = rig.CreateDefaultCheckpoint();
        Assert.NotEmpty(rig.TrainableParamStructDef.Fields);

        var modelInputDef = new TensorStructDef(
            new[]
            {
                new TensorStructFieldDef("tgt", DataStructure.Tensor, 3, DType.Float32),
                new TensorStructFieldDef("memory", DataStructure.Tensor, 3, DType.Float32),
            }, "ModelInput");
        var targetDef = new TensorStructDef(
            new[] { new TensorStructFieldDef("targets", DataStructure.Tensor, 2, DType.Float32) }, "Target");
        var inputBatch = new TensorDataStruct(modelInputDef,
            new Dictionary<string, IData>
            {
                { "tgt", TensorData(tgtShape, Floats(24, seed: 0.07f)) },
                { "memory", TensorData(memShape, Floats(40, seed: 0.05f)) },
            });
        var targetBatch = new TensorDataStruct(targetDef,
            new Dictionary<string, IData> { { "targets", TensorData(outShape, new float[8]) } });

        var ctx = new ComputeContext();
        var compiled = ctx.Compile(rig.TrainingStepPureGraph);
        var step = rig.TrainStep(initial, inputBatch, targetBatch, compiled);

        Assert.True(float.IsFinite(step.Loss));
        Assert.NotEmpty(step.Checkpoint.TrainableParams.Fields);
        Assert.True(AnyFieldChanged(initial.TrainableParams, step.Checkpoint.TrainableParams),
            "no trainable parameter moved after a TrainStep (gradient did not flow)");
    }

    /// <summary>Deterministic small distinct floats so the attention/FFN path is non-degenerate.</summary>
    private static float[] Floats(int count, float seed)
    {
        var vals = new float[count];
        for (var i = 0; i < count; i++)
            vals[i] = seed * (((i * 7) % 11) - 5);   // spread over [-5·seed, 5·seed], distinct pattern
        return vals;
    }

    private static bool AnyFieldChanged(TensorDataStruct before, TensorDataStruct after)
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

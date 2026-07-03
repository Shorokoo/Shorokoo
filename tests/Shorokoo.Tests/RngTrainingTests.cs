using System.Collections.Generic;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.Processors.Training;
using Shorokoo.Modules.Layers;
using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;
using Shorokoo.Runtime;
using Shorokoo.Tests.Modules;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;

namespace Shorokoo.Tests;

/// <summary>Scalar weight + always-on Dropout: the smallest model whose training step draws
/// runtime randomness every step.</summary>
[Module]
public partial class RngRigDropoutModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var weight = InitScalarWeight.Init(Vector(1L));
        var h = input * weight;
        return Dropout.Call(Scalar(0.5f), Scalar(true), h);
    }
}

/// <summary>
/// End-to-end determinism of a keyed training rig: binding an <see cref="RngConfig"/> at
/// <see cref="TrainingRig.FromScratch"/> stamps the model's runtime feeds (Dropout masks)
/// before loss composition and autodiff, so the whole trajectory — losses and updated
/// weights across steps — reproduces bit-for-bit from the master seed, and re-keys under a
/// different one.
/// </summary>
[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class RngTrainingTests
{
    private static readonly TensorStructDef ModelInputDef = new(
        new[] { new TensorStructFieldDef("input", DataStructure.Tensor, 1, DType.Float32) },
        "ModelInput");

    private static readonly TensorStructDef TargetDef = new(
        new[] { new TensorStructFieldDef("targets", DataStructure.Tensor, 1, DType.Float32) },
        "Target");

    private static (float[] losses, TrainingRig rig, TrainingCheckpoint finalCheckpoint) TrainLosses(
        RngConfig? rngConfig, int steps)
    {
        var sample = new NamedModelParam[]
        {
            new TensorDataModelParam("input", ModelParamType.InputParam,
                TensorData([8L], new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f })),
        };

        var rig = TrainingRig.FromScratch(
            RngRigDropoutModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph, sample, rngConfig, 0.05f);

        var inputBatch = new TensorDataStruct(ModelInputDef,
            new Dictionary<string, IData>
                { { "input", TensorData([8L], new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }) } });
        var targetBatch = new TensorDataStruct(TargetDef,
            new Dictionary<string, IData>
                { { "targets", TensorData([8L], new float[8]) } });

        var ctx = new ComputeContext();
        var compiled = ctx.Compile(rig.TrainingStepPureGraph);
        var checkpoint = rig.CreateDefaultCheckpoint();

        var losses = new float[steps];
        for (int i = 0; i < steps; i++)
        {
            var step = rig.TrainStep(checkpoint, inputBatch, targetBatch, compiled);
            losses[i] = step.Loss;
            checkpoint = step.Checkpoint;
        }
        return (losses, rig, checkpoint);
    }

    [Fact]
    public void TestKeyedRigTrainsDeterministicallyAndRekeysUnderNewMaster()
    {
        var (lossesA1, rigA, finalA) = TrainLosses(new RngConfig { MasterSeed = 5 }, steps: 3);
        var (lossesA2, _, _) = TrainLosses(new RngConfig { MasterSeed = 5 }, steps: 3);
        var (lossesB, _, _) = TrainLosses(new RngConfig { MasterSeed = 6 }, steps: 3);

        // The generator-managed drawBase: the injected RngExecutionCounter is ordinary model
        // state riding the checkpoint, advanced +1 per step — after 3 steps it reads 3, so a
        // resumed run at step 4 draws exactly what the uninterrupted run would.
        var counterField = finalA.ModelState.Fields.Single(f => f.Key.Contains("RngExecutionCounter"));
        var counterValue = ((TensorData<float32>)counterField.Value).AccessMemory()[0];
        Assert.Equal(3f, counterValue);

        // The stamped feed rides through loss composition and autodiff into the
        // training-step graph — the stamp is visible on the step graph itself.
        Assert.Contains(rigA.TrainingStepPureGraph.Nodes, n =>
            n.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM &&
            n.Attributes.GetLongsVal(ShrkAttrRngExplicitKey) is { Length: 2 });

        // Same master seed -> bit-identical trajectory across independent rig builds.
        Assert.Equal(lossesA1, lossesA2);
        // Different master seed -> different Dropout streams -> different trajectory.
        Assert.NotEqual(lossesA1, lossesB);
        Assert.All(lossesA1, l => Assert.True(float.IsFinite(l)));
    }
}

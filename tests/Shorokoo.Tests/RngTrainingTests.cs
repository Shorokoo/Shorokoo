using System.Collections.Generic;
using Shorokoo.Core.Nodes.Processors.Training;
using Shorokoo.Modules.Layers;
using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;
using Shorokoo.Runtime;
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
/// <see cref="TrainingRig.FromScratch"/> keys the model's runtime feeds (Dropout masks) and its
/// parameter initialization before loss composition and autodiff, so the whole trajectory —
/// losses and updated weights across steps — reproduces bit-for-bit from the master seed,
/// re-keys under a different one, and resumes exactly from a mid-run checkpoint.
/// </summary>
[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class RngTrainingTests
{
    private static readonly TensorStructDef ModelInputDef = new(
        [new TensorStructFieldDef("input", DataStructure.Tensor, 1, DType.Float32)],
        "ModelInput");

    private static readonly TensorStructDef TargetDef = new(
        [new TensorStructFieldDef("targets", DataStructure.Tensor, 1, DType.Float32)],
        "Target");

    private static TrainingRig BuildDropoutRig(RngConfig? rngConfig)
    {
        var sample = new NamedModelParam[]
        {
            new TensorDataModelParam("input", ModelParamType.InputParam,
                TensorData([8L], 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f)),
        };
        return TrainingRig.FromScratch(
            RngRigDropoutModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph, sample, [0.05f], rngConfig);
    }

    private static (TensorDataStruct inputBatch, TensorDataStruct targetBatch) MakeBatches()
    {
        var inputBatch = new TensorDataStruct(ModelInputDef,
            new Dictionary<string, IData>
                { { "input", TensorData([8L], 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f) } });
        var targetBatch = new TensorDataStruct(TargetDef,
            new Dictionary<string, IData>
                { { "targets", TensorData([8L], new float[8]) } });
        return (inputBatch, targetBatch);
    }

    private static (float[] losses, TrainingRig rig, TrainingCheckpoint finalCheckpoint) TrainLosses(
        RngConfig? rngConfig, int steps)
    {
        var rig = BuildDropoutRig(rngConfig);
        var (inputBatch, targetBatch) = MakeBatches();

        var checkpoint = rig.CreateInitialCheckpoint();
        var losses = new float[steps];
        for (int i = 0; i < steps; i++)
        {
            var step = rig.TrainStep(checkpoint, inputBatch, targetBatch);
            losses[i] = step.Loss!.Value;
            checkpoint = step;
        }
        return (losses, rig, checkpoint);
    }

    [Fact]
    public void TestKeyedRigTrainsDeterministicallyRekeysUnderANewMasterAndKeysInitialization()
    {
        var (lossesA1, rigA, finalA) = TrainLosses(new RngConfig { MasterSeed = 5 }, steps: 3);
        var (lossesA2, _, _) = TrainLosses(new RngConfig { MasterSeed = 5 }, steps: 3);
        var (lossesB, _, _) = TrainLosses(new RngConfig { MasterSeed = 6 }, steps: 3);

        // The generator-managed substreamIndex: the injected RngExecutionCounter is ordinary
        // int64 model state riding the checkpoint, advanced +1 per step.
        var counterField = finalA.ModelState.Fields.Single(f => f.Key.Contains("RngExecutionCounter"));
        Assert.Equal(3L, ((TensorData<int64>)counterField.Value).AccessMemory()[0]);

        // The RngSeed parameter rides through loss composition and autodiff into the training
        // step graph, which is what its Dropout feeds' key chains derive from.
        Assert.NotNull(rigA.TrainingStepPureGraph.TryGetRngSeed());
        Assert.Contains(rigA.TrainingStepPureGraph.ToInternal().Nodes, n =>
            n.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM);

        Assert.Equal(lossesA1, lossesA2);      // same master -> bit-identical trajectory
        Assert.NotEqual(lossesA1, lossesB);    // different master -> different Dropout streams
        Assert.All(lossesA1, l => Assert.True(float.IsFinite(l)));

        // The rig's RngConfig must key parameter INITIALIZATION too, not only runtime feeds.
        float[] InitialWeight(RngConfig cfg)
        {
            var sample = new NamedModelParam[]
            {
                new TensorDataModelParam("input", ModelParamType.InputParam,
                    TensorData([1L, 3L], 0.1f, 0.2f, 0.3f)),
            };
            var rig = TrainingRig.FromScratch(
                SwitchInitLinear.ComputationGraph, L2Loss.ComputationGraph,
                SGDOptimizer.ComputationGraph, sample, [0.05f], cfg);
            var ckpt = rig.CreateInitialCheckpoint();
            var name = rig.TrainableParamStructDef.Fields[0].Name;
            return ((TensorData<float32>)ckpt.TrainableParams.Fields[name]).AccessMemory().ToArray();
        }

        var w5 = InitialWeight(new RngConfig { MasterSeed = 5 });
        Assert.Equal(w5, InitialWeight(new RngConfig { MasterSeed = 5 }));
        Assert.NotEqual(w5, InitialWeight(new RngConfig { MasterSeed = 6 }));
    }

    [Fact]
    public void TestMidRunCheckpointResumeReplaysUninterruptedTrajectoryExactly()
    {
        const int totalSteps = 6, resumeAt = 3;
        var cfg = new RngConfig { MasterSeed = 5 };

        var (fullLosses, _, _) = TrainLosses(cfg, totalSteps);
        // A second, independent rig trains to step k and checkpoints there.
        var (_, _, ckpt) = TrainLosses(cfg, resumeAt);
        var (inputBatch, targetBatch) = MakeBatches();

        var path = Path.Combine(Path.GetTempPath(), $"rng_resume_{Guid.NewGuid():N}.safetensors");
        try
        {
            ckpt.Save(path);

            // "Fresh process": a brand-new rig + compiled graph loads the checkpoint. The int64
            // substreamIndex counter rides in ModelState, so the resumed steps draw the masks of
            // executions k, k+1, … — not 0, 1, … over again.
            var rigC = BuildDropoutRig(cfg);
            var resumed = rigC.LoadCheckpoint(path);
            Assert.Equal(resumeAt, resumed.Step);

            var resumedLosses = new float[totalSteps - resumeAt];
            for (int i = 0; i < resumedLosses.Length; i++)
            {
                var step = rigC.TrainStep(resumed, inputBatch, targetBatch);
                resumedLosses[i] = step.Loss!.Value;
                resumed = step;
            }

            // The NotEqual keeps the Equal non-vacuous: the trajectory isn't periodic, so
            // matching the tail is a genuine resume signal.
            Assert.Equal(fullLosses[resumeAt..], resumedLosses);
            Assert.NotEqual(fullLosses[..(totalSteps - resumeAt)], resumedLosses);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }
}

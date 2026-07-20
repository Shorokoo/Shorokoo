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
/// <see cref="TrainingRig.FromScratch"/> keys the model's runtime feeds (Dropout masks)
/// before loss composition and autodiff, so the whole trajectory — losses and updated
/// weights across steps — reproduces bit-for-bit from the master seed, and re-keys under a
/// different one.
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
                TensorData([8L], new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f })),
        };
        return TrainingRig.FromScratch(
            RngRigDropoutModel.ComputationGraph, L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph, sample, rngConfig, 0.05f);
    }

    private static (TensorDataStruct inputBatch, TensorDataStruct targetBatch) MakeBatches()
    {
        var inputBatch = new TensorDataStruct(ModelInputDef,
            new Dictionary<string, IData>
                { { "input", TensorData([8L], new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }) } });
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
        // resumed run at step 4 draws exactly what the uninterrupted run would. The cast
        // pins the framework-counter convention — int64 state end-to-end (see
        // FastInjectRngDrawCounter.CounterInit for the saturation rationale).
        var counterField = finalA.ModelState.Fields.Single(f => f.Key.Contains("RngExecutionCounter"));
        var counterValue = ((TensorData<int64>)counterField.Value).AccessMemory()[0];
        Assert.Equal(3L, counterValue);

        // The RngSeed parameter rides through loss composition and autodiff into the
        // training-step graph — the step graph itself carries the model's RNG identity,
        // which is what its Dropout feeds' key chains derive from.
        Assert.NotNull(rigA.TrainingStepPureGraph.TryGetRngSeed());
        Assert.Contains(rigA.TrainingStepPureGraphInternal.Nodes, n =>
            n.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM);

        // Same master seed -> bit-identical trajectory across independent rig builds.
        Assert.Equal(lossesA1, lossesA2);
        // Different master seed -> different Dropout streams -> different trajectory.
        Assert.NotEqual(lossesA1, lossesB);
        Assert.All(lossesA1, l => Assert.True(float.IsFinite(l)));
    }

    [Fact]
    public void TestKeyedRigInitializesWeightsFromTheMasterSeed()
    {
        // The rig's RngConfig must key parameter INITIALIZATION, not only runtime feeds:
        // a random initializer's weights come from the config's master seed. (SwitchInitLinear
        // is a single Linear whose weight is drawn by KaimingUniform — a random initializer.)
        float[] InitialWeight(RngConfig cfg)
        {
            var sample = new NamedModelParam[]
            {
                new TensorDataModelParam("input", ModelParamType.InputParam,
                    TensorData([1L, 3L], 0.1f, 0.2f, 0.3f)),
            };
            var rig = TrainingRig.FromScratch(
                SwitchInitLinear.ComputationGraph, L2Loss.ComputationGraph,
                SGDOptimizer.ComputationGraph, sample, cfg, 0.05f);
            var ckpt = rig.CreateDefaultCheckpoint();
            var name = rig.TrainableParamStructDef.Fields[0].Name;
            return ((TensorData<float32>)ckpt.TrainableParams.Fields[name]).AccessMemory().ToArray();
        }

        var w5 = InitialWeight(new RngConfig { MasterSeed = 5 });
        var w5again = InitialWeight(new RngConfig { MasterSeed = 5 });
        var w6 = InitialWeight(new RngConfig { MasterSeed = 6 });

        Assert.Equal(w5, w5again);   // deterministic under a fixed master seed
        Assert.NotEqual(w5, w6);     // init weights are keyed by the master seed (not constant)
    }

    /// <summary>
    /// The drawBase counter's resume guarantee, pinned end-to-end on a rig that draws runtime
    /// randomness every step: save a Dropout rig's checkpoint mid-run, resume it in a
    /// brand-new rig + compiled graph, and the resumed run replays the uninterrupted run's
    /// remaining steps bit-exactly. Previously this held only by proxy (the counter was
    /// checked to increment and ride the checkpoint; save/load resume was covered only on a
    /// randomness-free model).
    /// </summary>
    [Fact]
    public void TestMidRunCheckpointResumeReplaysUninterruptedTrajectoryExactly()
    {
        const int totalSteps = 6, resumeAt = 3;
        var cfg = new RngConfig { MasterSeed = 5 };

        var (fullLosses, _, _) = TrainLosses(cfg, totalSteps);

        // A second, independent rig (built inside TrainLosses) trains to step k and
        // checkpoints there.
        var (_, _, ckpt) = TrainLosses(cfg, resumeAt);
        var (inputBatch, targetBatch) = MakeBatches();

        var path = Path.Combine(Path.GetTempPath(), $"rng_resume_{Guid.NewGuid():N}.safetensors");
        try
        {
            ckpt.Save(path);

            // "Fresh process": a brand-new rig + compiled graph loads the checkpoint. The
            // int64 drawBase counter rides in ModelState, so the resumed steps draw the
            // masks of executions k, k+1, … — not 0, 1, … over again.
            var rigC = BuildDropoutRig(cfg);
            var compiledC = new ComputeContext().Compile(rigC.TrainingStepPureGraph);
            var resumed = rigC.LoadCheckpoint(path);
            Assert.Equal(resumeAt, resumed.Step);

            var resumedLosses = new float[totalSteps - resumeAt];
            for (int i = 0; i < resumedLosses.Length; i++)
            {
                var step = rigC.TrainStep(resumed, inputBatch, targetBatch, compiledC);
                resumedLosses[i] = step.Loss;
                resumed = step.Checkpoint;
            }

            // Bit-exact continuation: the resumed losses ARE the uninterrupted run's steps
            // k..N. The NotEqual keeps that Equal non-vacuous: it pins that the trajectory
            // isn't periodic (the opening steps don't repeat at k..N), so per-step mask
            // variation is real and matching the tail is a genuine resume signal.
            Assert.Equal(fullLosses[resumeAt..], resumedLosses);
            Assert.NotEqual(fullLosses[..(totalSteps - resumeAt)], resumedLosses);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }
}

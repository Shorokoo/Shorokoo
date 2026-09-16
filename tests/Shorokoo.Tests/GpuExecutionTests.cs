using Shorokoo.Core.Inference.Abstractions;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// Smoke tests that actually execute a small Shorokoo computation graph on the
/// local machine's GPU (NOT skipped).
///
/// These tests assume the host has an ONNX Runtime–compatible GPU available.
/// On a machine without one they will fail with a clear EP-loading error.
/// They are deliberately excluded from the coverage suite (no
/// Purpose=Coverage); run them on a CUDA machine with --filter "Purpose=Hardware".
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Hardware")]
[Collection(DeviceMemorySettings.Name)]
public class GpuExecutionTests
{
    /// <summary>
    /// Explicitly exercises the CUDA execution provider.
    /// </summary>
    [CudaFact]
    public void CudaProvider_AddTwoFloat32Scalars_ReturnsCorrectSum()
    {
        // [CudaFact] already gates on a GPU backend being loaded; confirm the
        // bound factory really is a CUDA one rather than a silent CPU fallback.
        var backend = InferenceBackend.Factory.GetType().Assembly.GetName().Name ?? "";
        Assert.EndsWith("GPU", backend);

        var result = AddTwoScalars(2.0f, 3.0f);
        Assert.Equal(5.0f, result);
    }

    /// <summary>
    /// The device-memory surface end to end, on both settings of every knob: the shipped
    /// defaults build and run a session — they are set explicitly, so the leg establishes them
    /// rather than inheriting whatever the process holds — and so does a budgeted power-of-two
    /// arena with per-run shrinkage on, with the card reporting a reading either way.
    /// </summary>
    [CudaFact]
    public void CudaProvider_RunsUnderEveryDeviceMemoryConfigurationAndReportsTheCardsUsage()
    {
        var limit = DeviceMemory.LimitBytes;
        var arenaExtend = DeviceMemory.ArenaExtend;
        var shrink = DeviceMemory.ShrinkArenaAfterRun;
        try
        {
            DeviceMemory.LimitBytes = null;
            DeviceMemory.ArenaExtend = ArenaExtendStrategy.SameAsRequested;
            DeviceMemory.ShrinkArenaAfterRun = false;
            DeviceMemory.ResetPeak();
            Assert.Equal(5.0f, AddTwoScalars(2.0f, 3.0f));

            DeviceMemory.LimitBytes = 2L * 1024 * 1024 * 1024;
            DeviceMemory.ArenaExtend = ArenaExtendStrategy.NextPowerOfTwo;
            DeviceMemory.ShrinkArenaAfterRun = true;
            Assert.Equal(5.0f, AddTwoScalars(2.0f, 3.0f));

            var reading = DeviceMemory.Sample();
            Assert.NotNull(reading);
            Assert.True(reading!.Value.TotalBytes > 0);
            Assert.True(reading.Value.UsedBytes > 0);
            Assert.Equal(reading.Value.UsedBytes, DeviceMemory.PeakUsedBytes);
        }
        finally
        {
            DeviceMemory.LimitBytes = limit;
            DeviceMemory.ArenaExtend = arenaExtend;
            DeviceMemory.ShrinkArenaAfterRun = shrink;
            DeviceMemory.ResetPeak();
        }
    }

    /// <summary>
    /// The interaction the whole integration turns on, and the one no CPU test can reach: state
    /// left in the provider's own memory across steps, while the arena it lives in is budgeted,
    /// extends by request, and is handed back after every run. The trained result has to be the
    /// same as an ordinary step loop's.
    /// </summary>
    [CudaFact]
    public void CudaProvider_AResidentRunTrainsTheSameUnderABudgetedAndShrinkingArena()
    {
        var limit = DeviceMemory.LimitBytes;
        var arenaExtend = DeviceMemory.ArenaExtend;
        var shrink = DeviceMemory.ShrinkArenaAfterRun;
        try
        {
            DeviceMemory.LimitBytes = 2L * 1024 * 1024 * 1024;
            DeviceMemory.ArenaExtend = ArenaExtendStrategy.SameAsRequested;
            DeviceMemory.ShrinkArenaAfterRun = true;

            var (input, target) = (TrainingRigHelpers.InBatch(1f, 2f, 3f, 4f),
                                   TrainingRigHelpers.TargetBatch(2f, 4f, 6f, 8f));
            var expected = StepLoopWeights(input, target);

            var rig = ScalarRig();
            using var run = rig.BeginResidentRun();
            run.Step(input, target);
            run.Step(input, target);
            var published = run.StepToCheckpoint(input, target);

            Assert.Equal(3, published.Step);
            var actual = Weights(published);
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(expected[i], actual[i], precision: 4);
        }
        finally
        {
            DeviceMemory.LimitBytes = limit;
            DeviceMemory.ArenaExtend = arenaExtend;
            DeviceMemory.ShrinkArenaAfterRun = shrink;
        }
    }

    private static TrainingRig ScalarRig() => TrainingRig.FromScratch(
        ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, AdamWOptimizer.ComputationGraph,
        [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData([4L], [1f, 2f, 3f, 4f]))],
        new AdamWOptimizerHyperparameters { LearningRate = 0.1f });

    private static float[] StepLoopWeights(TensorDataStruct input, TensorDataStruct target)
    {
        var rig = ScalarRig();
        var ckpt = rig.CreateInitialCheckpoint();
        for (int i = 0; i < 3; i++) ckpt = rig.TrainStep(ckpt, input, target);
        return Weights(ckpt);
    }

    private static float[] Weights(TrainingCheckpoint checkpoint) =>
        TrainingRigHelpers.FlattenStruct(checkpoint.TrainableParams);

    private static float AddTwoScalars(float left, float right)
    {
        var a = InputScalar<float32>();
        var b = InputScalar<float32>();
        var c = a + b;

        var graph = new InternalComputationGraph([a, b], [c]);
        var ctx = new ComputeContext();
        var results = ctx.Execute(
            (graph),
            TensorData([], left),
            TensorData([], right));

        return results[0].ToTensorData().As<float32>().AccessMemory<float>()[0];
    }
}

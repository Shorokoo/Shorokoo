using System;
using Shorokoo;
using Shorokoo.Graph;
using Shorokoo.Modules.Optimizers;
using Shorokoo.Runtime;
using static Shorokoo.Globals;

// ── Train ────────────────────────────────────────────────────────────────────

var baseGraph    = StackedLinear.ComputationGraph;
var exampleInput = TensorData([4L, 8L], new float[32]);
var model        = baseGraph.Specialize(baseGraph.FromOrderedInputs([TensorData([], 3L)]));

var rig = TrainingRig.FromScratch(
    model, Losses.L2Loss, Optimizers.Adam,
    model.FromOrderedInputs([exampleInput]),
    new AdamOptimizerHyperparameters { LearningRate = 1e-3f });

// Toy data: gradually increasing values; targets all zeros.
float[] batch1X = new float[32];
float[] batch2X = new float[32];
float[] batch1Y = new float[32];
float[] batch2Y = new float[32];
for (int i = 0; i < 32; i++)
{
    batch1X[i] = (float)(i + 1)  / 100f;
    batch2X[i] = (float)(i + 33) / 100f;
}

TensorDataStruct[] trainInputs = [
    rig.InputDef.FromOrderedData(TensorData([4L, 8L], batch1X)),
    rig.InputDef.FromOrderedData(TensorData([4L, 8L], batch2X)),
];
TensorDataStruct[] trainTargets = [
    rig.TargetDef.FromOrderedData(TensorData([4L, 8L], batch1Y)),
    rig.TargetDef.FromOrderedData(TensorData([4L, 8L], batch2Y)),
];

var result = rig.Fit(trainInputs, trainTargets, numEpochs: 20);
Console.WriteLine($"Final loss: {result.EpochLosses[^1]:F6}");

var firstLoss = result.EpochLosses[0];
var lastLoss  = result.EpochLosses[^1];
Console.WriteLine($"Loss went from {firstLoss:F6} to {lastLoss:F6}");
if (!float.IsFinite(lastLoss)) throw new Exception("Loss is not finite!");

// ── Run ──────────────────────────────────────────────────────────────────────

var savePath = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "readme-validation.skpt");
Persistence.SaveTrainingCheckpointToSkpt(result.FinalCheckpoint, savePath);
Console.WriteLine($"Checkpoint saved to: {savePath}");

var inferenceInput = TensorData([4L, 8L], new float[32]);   // same [4 × 8] shape the rig trained on
var concrete       = result.FinalCheckpoint.ToInferenceModel();

ReadOnlySpan<float> prediction = ComputeContext.Default
    .Execute(concrete, inferenceInput)[0]
    .ToTensorData<float32>().AccessMemory();

Console.WriteLine($"Inference output ({prediction.Length} values): [{string.Join(", ", prediction.ToArray())}]");
if (prediction.Length != 32) throw new Exception($"Expected 32 output values, got {prediction.Length}");

// ── Reload, as a later process would ─────────────────────────────────────────

var reloaded = Persistence.Load(savePath);
var evaluation = Persistence.LoadEvaluationModel(savePath);
var (reloadedRig, reloadedCheckpoint) = TrainingRig.Load(savePath);

ReadOnlySpan<float> reloadedPrediction = ComputeContext.Default
    .Execute(reloaded, inferenceInput)[0]
    .ToTensorData<float32>().AccessMemory();
if (!reloadedPrediction.SequenceEqual(prediction))
    throw new Exception("Reloaded model disagrees with the checkpoint's inference model.");

var validationLoss = ComputeContext.Default
    .Execute(evaluation, TensorData([4L, 8L], batch1X), TensorData([4L, 8L], batch1Y))[0]
    .ToTensorData<float32>().ValueAt<float>(0);
Console.WriteLine($"Validation loss from the file alone: {validationLoss:F6}");
if (!float.IsFinite(validationLoss)) throw new Exception("Validation loss is not finite!");

if (reloadedCheckpoint.Step != result.FinalCheckpoint.Step)
    throw new Exception("Reloaded checkpoint resumed at the wrong step.");
if (reloadedRig.TrainStep(reloadedCheckpoint, trainInputs[0], trainTargets[0]).Loss is not { } resumedLoss
    || !float.IsFinite(resumedLoss))
    throw new Exception("Resumed training step did not produce a finite loss.");

Console.WriteLine("\nREADME validation passed.");

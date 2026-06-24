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

var savePath = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "readme-validation.safetensors");
result.FinalCheckpoint.Save(savePath);
Console.WriteLine($"Checkpoint saved to: {savePath}");

var inferenceInput = TensorData([1L, 8L], new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f });
var concrete       = result.FinalCheckpoint.ToInferenceModel(model, inferenceInput);

ReadOnlySpan<float> prediction = ComputeContext.Default
    .Execute(concrete, inferenceInput)[0]
    .ToTensorData<float32>().AccessMemory();

Console.WriteLine($"Inference output ({prediction.Length} values): [{string.Join(", ", prediction.ToArray())}]");
if (prediction.Length != 8) throw new Exception($"Expected 8 output values, got {prediction.Length}");

Console.WriteLine("\nREADME validation passed.");

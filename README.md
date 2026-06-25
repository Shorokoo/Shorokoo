# Shorokoo

## What Shorokoo is

Shorokoo is a .NET 10 / C# neural-network framework. With it, you can:

- **Define models** in C# — no Python.
- **Train** them with built-in or custom losses and optimizers.
- **Run** on CPU or GPU.
- **Interoperate** with the wider ML ecosystem: export models as `.onnx`, load pretrained weights from `.safetensors`.

The primary public namespace is `Shorokoo`.

## From model to prediction

The example below walks through the full lifecycle: defining a model, training it, and running inference. No Python, no framework boilerplate — just C# classes and LINQ-style method chains.

### 1. Define

A `[Module]` class is the building block. Its `Inline` method describes the forward pass as ordinary C# expressions; the source generator produces `Call`, `Model`, and `ComputationGraph` from it automatically. Control flow that depends on tensor values — `LoopAPI.Iterate` for loops, `.IfElse` for branches — is embedded directly in the method body.

```csharp
[Module]
public partial class StackedLinear
{
    // Weight-tied feedforward: the same (w, b) applied `depth` times.
    // Intermediate layers get ReLU; the output layer does not.
    public static Tensor<float32> Inline(
        Tensor<float32> x,
        [Hyper] Scalar<int64> depth)
    {
        var n = x.DimTensor(1);
        var w = XavierUniform.Init([n, n]);    // trainable weight matrix
        var b = Zeros.Init([n]).Vec();          // trainable bias

        var h = x;
        foreach (var ctx in LoopAPI.Iterate(depth))
        {
            var z      = h.MatMul(w.Transpose([1L, 0L])) + b;
            var isLast = ctx.IterationIndex + Scalar(1L) == depth;
            h = isLast.IfElse(z, z.Relu());
        }
        return h;
    }
}
```

### 2. Train

`Specialize` bakes the depth hyperparameter into the graph so the training rig only sees tensor inputs. Then `TrainingRig.FromScratch` wires the model, loss, and optimizer together — gradients are computed automatically, no backward pass to write.

```csharp
var baseGraph    = StackedLinear.ComputationGraph;
var exampleInput = TensorData([4L, 8L], new float[32]);
var model        = baseGraph.Specialize(baseGraph.FromOrderedInputs([TensorData([], 3L)]));

var rig = TrainingRig.FromScratch(
    model, Losses.L2Loss, Optimizers.Adam,
    model.FromOrderedInputs([exampleInput]),
    new AdamOptimizerHyperparameters { LearningRate = 1e-3f });

// Fit iterates all batches on every epoch — supply as many as you like.
var rng     = new Random(42);
float[] batch1X = Enumerable.Range(0, 32).Select(_ => (float)rng.NextDouble()).ToArray();
float[] batch1Y = Enumerable.Range(0, 32).Select(_ => (float)rng.NextDouble()).ToArray();
float[] batch2X = Enumerable.Range(0, 32).Select(_ => (float)rng.NextDouble()).ToArray();
float[] batch2Y = Enumerable.Range(0, 32).Select(_ => (float)rng.NextDouble()).ToArray();

TensorDataStruct[] trainInputs = [
    rig.InputDef.FromOrderedData(TensorData([4L, 8L], batch1X)),
    rig.InputDef.FromOrderedData(TensorData([4L, 8L], batch2X)),
];
TensorDataStruct[] trainTargets = [
    rig.TargetDef.FromOrderedData(TensorData([4L, 8L], batch1Y)),
    rig.TargetDef.FromOrderedData(TensorData([4L, 8L], batch2Y)),
];

var result = rig.Fit(trainInputs, trainTargets, numEpochs: 20);
Console.WriteLine($"Final loss: {result.EpochLosses[^1]:F4}");
```

### 3. Run

Save the checkpoint, then bind the trained weights into a concrete model with one call and execute. `ToInferenceModel` inlines all sub-modules and substitutes the trained parameter values.

```csharp
result.FinalCheckpoint.Save("my-model.safetensors");   // persist trained weights

var inferenceInput = TensorData([1L, 8L], new float[8]);   // your [1 × 8] input
var concrete       = result.FinalCheckpoint.ToInferenceModel(model, inferenceInput);

ReadOnlySpan<float> prediction = ComputeContext.Default
    .Execute(concrete, inferenceInput)[0]
    .ToTensorData<float32>().AccessMemory();
```

## Documentation

The full documentation index — which page covers what — lives in
[Documentation/README.md](Documentation/README.md).

## Installation

Shorokoo ships as NuGet packages. Install the meta-package plus **one** backend for
your platform:

```bash
dotnet add package Shorokoo               # runtime + NN library + source generator
dotnet add package Shorokoo.LinuxCPU      # or Shorokoo.LinuxGPU / Shorokoo.WinCPU / Shorokoo.WinGPU
```

| Package | What it is |
|---|---|
| `Shorokoo` | Meta-package: pulls in everything below except a backend |
| `Shorokoo.Core` | The runtime: tensors, autodiff, training, ONNX import/export |
| `Shorokoo.Modules` | Baseline NN library: ready-made layers, losses, optimizers, initializers |
| `Shorokoo.CodeGen` | Source generator for the `[Module]` syntax (flows in with the meta-package) |
| `Shorokoo.LinuxCPU` | ONNX Runtime backend, Linux x64 CPU |
| `Shorokoo.LinuxGPU` | ONNX Runtime backend, Linux x64 GPU (CUDA) |
| `Shorokoo.WinCPU` | ONNX Runtime backend, Windows x64 CPU |
| `Shorokoo.WinGPU` | ONNX Runtime backend, Windows x64 GPU (CUDA) |

(The backends share an infrastructure package, `Shorokoo.OnnxRuntime`, that
they pull in themselves — you never install it directly.)

## Samples

- [`samples/RetinaNet`](samples/RetinaNet) — ResNet backbones and a RetinaNet detector
  built entirely from Shorokoo modules.

## License

[MIT](LICENSE)

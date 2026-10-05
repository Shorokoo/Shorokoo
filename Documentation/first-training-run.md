# Your first training run

One program, end to end: define a model, train it, checkpoint and resume it, evaluate it,
and read its memory use. Each step links to the reference page that covers it in full.
The last section lists what a first run does not surface but a long run depends on; read it
before you size a real job.

## Setup

Reference `Shorokoo`, `Shorokoo.Modules` and one backend package for your platform
(`Shorokoo.LinuxCPU`, `Shorokoo.LinuxGPU`, `Shorokoo.WinCPU` or `Shorokoo.WinGPU`). With one
backend referenced, `ComputeContext.Default` uses it with no setup. To choose one explicitly,
see [inference.md](inference.md).

```csharp
using Shorokoo;                     // TrainingRig, TrainingCheckpoint, Persistence, RngConfig
using Shorokoo.Core.Backends;       // DeviceMemory
using Shorokoo.Graph;               // ComputationGraph
using Shorokoo.Modules;             // [Module], [Hyper]
using Shorokoo.Modules.Layers;      // Linear
using Shorokoo.Modules.Losses;      // CrossEntropyLoss
using Shorokoo.Modules.Optimizers;  // AdamWOptimizer, AdamWOptimizerHyperparameters
using Shorokoo.Runtime;             // ComputeContext
using static Shorokoo.Globals;      // TensorData(...), Scalar(...)
```

## 1. Define the model

A model is a `[Module]` class with a static `Inline` method. `[Hyper]` parameters make it a
family of models; pick one member with `Specialize`. See
[defining-models.md](defining-models.md). In a top-level-statements `Program.cs`, the class
goes after the statements (or in a file of its own).

```csharp
[Module]
public partial class Classifier
{
    public static Tensor<float32> Inline(Tensor<float32> x,
        [Hyper] Scalar<int64> hidden, [Hyper] Scalar<int64> classes)
    {
        var h = Linear.Call(hidden, Scalar(true), x).Relu();
        return Linear.Call(classes, Scalar(true), h);
    }
}
```

```csharp
const long Batch = 32, Features = 16, Classes = 4;

var family = Classifier.ComputationGraph;
var model = family.Specialize(family.FromOrderedInputs(
    [TensorData([], 64L), TensorData([], Classes)]));   // hidden = 64, classes = 4
```

## 2. Build the rig

`TrainingRig.FromScratch` composes model, loss and optimizer into one training step. It
takes one sample per model input, which fixes the input's shape. The build is the slow part
of a first run. [What construction costs](training.md#what-construction-costs) says how it
scales, and [Watching a long build](training.md#watching-a-long-build) shows how to follow it.

```csharp
var rig = TrainingRig.FromScratch(
    model,
    CrossEntropyLoss.ComputationGraph,
    AdamWOptimizer.ComputationGraph,
    [TensorData([Batch, Features], new float[Batch * Features])],
    new AdamWOptimizerHyperparameters
    {
        LearningRate = 1e-3f, Beta1 = 0.9f, Beta2 = 0.999f, Epsilon = 1e-8f, WeightDecay = 0.01f,
    },
    rngConfig: new RngConfig { MasterSeed = 42 });
```

`CrossEntropyLoss` takes logits `[N, C]` and `int64` class indices `[N]`, or with extra axes,
`[N, C, d1, …]` against `[N, d1, …]`. The class axis is always axis 1. See the
[losses table](nn-library.md#losses-shorokoomoduleslosses).

The full signature, the other optimizers and schedules are in
[training.md](training.md#trainingrig-api).

## 3. Train

A resident run keeps the training state on the device between steps. `Step` returns the loss
and downloads nothing else.

```csharp
var run = rig.BeginResidentRun();
for (long step = 0; step < 99; step++)
{
    var (x, y) = MakeBatch(step);
    float loss = run.Step(rig.InputDef.FromOrderedData(x), rig.TargetDef.FromOrderedData(y));
    if (step % 20 == 0) Console.WriteLine($"step {step}: loss {loss:F4}");
}
```

A step **consumes** the tensors it is fed: after the call, `x` and `y` are gone, and reading
them throws `ObjectDisposedException`. To keep a tensor, feed it as `x.Shared()`. The rules
are in [What a training step consumes](training.md#what-a-training-step-consumes).

## 4. Checkpoint

`StepToCheckpoint` takes one more step and hands out the state as a `TrainingCheckpoint`,
left where the step put it (on a GPU, the card's memory; the save writes it from there). Save it
as a `.skpt`, which carries everything needed to resume.

```csharp
var (lx, ly) = MakeBatch(99);
TrainingCheckpoint checkpoint = run.StepToCheckpoint(rig.InputDef.FromOrderedData(lx),
                                                     rig.TargetDef.FromOrderedData(ly));
run.Dispose();
Persistence.SaveTrainingCheckpointToSkpt(checkpoint, "run.skpt");
```

The save is atomic ([skpt-checkpoints.md](skpt-checkpoints.md#facts)). What the file holds is in
[Training checkpoints](skpt-checkpoints.md#training-checkpoints).

## 5. Resume

`TrainingRig.Load` rebuilds the rig from the file and returns it with the resumed checkpoint,
in this process or a new one. Its full signature is
`Load(string filePath, ComputeContext? mergeContext = null, ComputeContext? runtimeContext = null,
IProgress<BuildProgress>? progress = null, TrainingBackend? trainingBackend = null)`.

```csharp
var (resumedRig, resumed) = TrainingRig.Load("run.skpt");
using (var resumedRun = resumedRig.BeginResidentRun(resumed))
{
    for (long step = 100; step < 110; step++)
    {
        var (x, y) = MakeBatch(step);
        resumedRun.Step(resumedRig.InputDef.FromOrderedData(x), resumedRig.TargetDef.FromOrderedData(y));
    }
    Console.WriteLine($"resumed to step {resumedRun.CurrentStep}");
}
```

## 6. Evaluate

A training `.skpt` also loads without a rig. `Persistence.LoadEvaluationModel` returns the
model composed with its loss, `[model inputs…, targets] → loss`. `Persistence.Load` returns
the trained model. Both run through `ComputeContext`. See
[Bind trained weights into an inference model](training.md#bind-trained-weights-into-an-inference-model).

```csharp
var eval = Persistence.LoadEvaluationModel("run.skpt");
var (vx, vy) = MakeBatch(1000);
float evalLoss = ComputeContext.Default.Execute(eval, vx.Shared(), vy)[0].ToTensorData().CopyMemory<float>()[0];

var inference = Persistence.Load("run.skpt");
var logits = ComputeContext.Default.Execute(inference, vx)[0].ToTensorData();   // [32, 4]
```

`vx` is fed `.Shared()` to the first run so that the second can still use it.

## 7. Measure memory

```csharp
if (DeviceMemory.Read() is { } card)   // null with no CUDA runtime
    Console.WriteLine($"device: {card.UsedBytes >> 20} of {card.TotalBytes >> 20} MiB used, all processes; "
                      + $"this process {card.ProcessBytes >> 20} MiB");
using (var p = System.Diagnostics.Process.GetCurrentProcess())
    Console.WriteLine($"host working set {p.WorkingSet64 >> 20} MiB");
```

`DeviceMemory.Read()` reports the whole card, every process included, and `ProcessBytes` this
process's own share of it. The other readings, and
what each covers, are in [Device memory](inference.md#device-memory-gpu-backends).

The batch helper used above:

```csharp
static (TensorData x, TensorData y) MakeBatch(long seed)
{
    var rnd = new Random((int)seed);
    var x = new float[Batch * Features];
    var y = new long[Batch];
    for (int i = 0; i < Batch; i++)
    {
        y[i] = rnd.Next((int)Classes);
        for (int f = 0; f < Features; f++) x[i * Features + f] = (float)rnd.NextDouble() + (f % Classes == y[i] ? 1f : 0f);
    }
    return (TensorData([Batch, Features], x), TensorData([Batch], y));
}
```

## Before you scale up

None of these shows in a small first run. Each can end a long one.

- **Checkpoint size.** A `.skpt` data entry of 2 GiB or more as stored, or an archive of 4 GiB or more,
  is refused when saved; a Zstd entry may decompress to more. A flat safetensors file loads back whatever its size and the size of each tensor in
  it. See [skpt-checkpoints.md](skpt-checkpoints.md).
- **Save cost.** What a save allocates and how long it takes grow with the checkpoint:
  [What a save costs](training.md#what-a-save-costs).
- **Out of memory.** A failed allocation arrives as `CR009` and says which pool ran out. On
  Windows (WDDM), device memory is charged to the process's commit, so a host memory limit
  also caps device memory:
  [When a training step runs out of memory](training.md#when-a-training-step-runs-out-of-memory).
- **Device-memory readings.** They are one record for the whole process and read device 0 only,
  and this process's share of the card can be unavailable in a container:
  [limitations.md](limitations.md#device-memory-readings-are-process-wide-and-device-0s).
- **Speed on a card.** `float32` is computed in full precision on a GPU too, which on a
  convolutional network costs the step 2.3 to 2.6 times what TensorFloat-32 does. A context can
  allow TensorFloat-32 for its runs: [Precision](inference.md#precision-gpu-backends).
- **Reproducibility.** A fixed seed reproduces a run bit for bit on the CPU backend, including
  across a save and resume. On a GPU it does so only on the ONNX Runtime backends, and only on a
  context that asks for deterministic compute: [Seeding the run](training.md#seeding-the-run).
- **Build cost.** Building a rig grows with the parameter count, and each new process pays it
  again: [What construction costs](training.md#what-construction-costs).
- **Namespaces.** The `using` lines each type needs are in [orientation.md](orientation.md).

To initialize one parameter from another's value, see "Writing your own" under
[Initializers](nn-library.md#initializers-shorokoomodulesinitializers).

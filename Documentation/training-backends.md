# Training backends: who computes the gradient

Related: [training.md](training.md) · [inference.md](inference.md) · [limitations.md](limitations.md)

## Facts

- A `TrainingRig`'s **training backend** decides who computes the gradient of its training step.
  `TrainingBackend.Shorokoo`, the default, is Shorokoo's own automatic differentiation.
  `TrainingBackend.Native` leaves the gradient to the **execution backend** of the rig's
  `RuntimeContext` — the backend that runs the step.
- Only the gradient changes hands. The model, the loss, the optimizer, schedules, hyperparameters,
  random draws (Dropout masks and the like), the step's inputs and outputs and the checkpoint layout
  are Shorokoo's on either backend.
- The training backend is **runtime configuration, never written to a checkpoint**, exactly like the
  two compute contexts. A checkpoint saved by a rig on one backend resumes on a rig on the other.
- A step is handed to the execution backend in a **format**: `"onnx"` for `TrainingBackend.Shorokoo`,
  `"onnx-autograd/1"` for `TrainingBackend.Native`. Every backend runs `"onnx"`. Only a backend that
  computes gradients itself runs `"onnx-autograd/1"`; the ONNX Runtime backends do not.
- Asking for `TrainingBackend.Native` on a context whose backend does not accept its format is refused
  when the rig is built, with a `NotSupportedException` — never at the first step.

`TrainingBackend` and `TrainingFormats` are in namespace `Shorokoo` (covered by `using Shorokoo;`).

## Choosing one

Pass it to `FromScratch`, after the compute contexts, together with a runtime context whose backend
computes gradients:

```csharp
var rig = TrainingRig.FromScratch(
    model, loss, optimizer, sampleInputs,
    new AdamWOptimizerHyperparameters { LearningRate = 0.001f },
    runtimeContext: new ComputeContext(backend),
    trainingBackend: TrainingBackend.Native);
```

`backend` here is an `IShorokooBackend` that accepts `TrainingFormats.OnnxAutoGrad` — today, the
PyTorch backends (see [Training on PyTorch](#training-on-pytorch)). Everything else is the ordinary
rig: `TrainStep`, `Fit`, `Train`, `BeginResidentRun`, checkpoints and `.skpt` files work unchanged.

To move an existing rig, derive a new one; the model, loss, optimizer, hyperparameters, seed and
contexts carry over, and only the training step is rebuilt:

```csharp
var native   = rig.WithTrainingBackend(TrainingBackend.Native);
var shorokoo = native.WithTrainingBackend(TrainingBackend.Shorokoo);
```

Every other `With…` derivation (`WithLoss`, `WithOptimizer`, `WithScheduler`, `WithSeed`) keeps the
rig's training backend, as it keeps its contexts. A rig rebuilt from a `.skpt` alone takes one the
same way it takes its contexts: `TrainingRig.Load(path, runtimeContext: ctx, trainingBackend:
TrainingBackend.Native)`; left out, it is `TrainingBackend.Shorokoo`. Read the current one off
`rig.TrainingBackend`; its `ToString()` is the name and the format, e.g. `Native (onnx-autograd/1)`.

Whether a backend runs a format is its own answer:
`backend.AcceptsTrainingFormat(TrainingFormats.OnnxAutoGrad)`.

## Training on PyTorch

The [PyTorch backends](pytorch-backend.md) accept both formats, so a rig whose runtime context runs
on one can leave its gradient to **torch autograd**:

```csharp
using Shorokoo.PyTorch.Cpu;                  // or Shorokoo.PyTorch.Cuda: new TorchCudaBackend()
using Shorokoo.Runtime;

using var torch = new ComputeContext(new TorchCpuBackend());
var rig = TrainingRig.FromScratch(
    model, loss, optimizer, sampleInputs,
    new AdamWOptimizerHyperparameters { LearningRate = 0.001f },
    runtimeContext: torch,
    trainingBackend: TrainingBackend.Native);

var checkpoint = rig.CreateInitialCheckpoint();
checkpoint = rig.TrainStep(checkpoint, inputs, targets);
```

A PyTorch runtime context with the default `TrainingBackend.Shorokoo` works too: torch then runs
Shorokoo's own backward pass like any other operators. `TrainingBackend.Native` is what hands the
gradient to torch.

What the step does on torch:

- The forward pass up to the loss runs with gradient recording on, each trainable parameter as a
  leaf; `torch.autograd.grad` takes the gradient there; the optimizer update runs with recording
  off, and everything the step returns is detached. The step never keeps an autograd graph past its
  own run.
- The optimizer, schedules, hyperparameters and random draws are still Shorokoo's operators. A
  Dropout mask is drawn by Shorokoo's counter-based generator on either backend, so the masks are
  identical bit for bit, and a rig trained on torch follows the rig trained by Shorokoo step for
  step — loss, parameters, model state and optimizer state — up to floating-point rounding, away
  from the non-smooth points below.
- A checkpoint is the same checkpoint: saved from a torch rig, it resumes on an ONNX Runtime rig,
  and the other way round.

Limits of training on torch, besides those of [the native path](#what-differs-on-the-native-path):

- The model must be one the PyTorch backend runs: an operator it does not translate refuses the
  step's session when it is created, at the first step, with a `TorchUnsupportedModelException`
  naming it (see the backend's [limitations](pytorch-backend.md#limitations)).
- An operator the backend runs but does not differentiate through refuses the step the same way,
  naming it, when it lies between a parameter and the loss. Operators whose outputs carry no
  gradient — comparisons, shapes, indices — end the path rather than refusing it.
- Only floating-point parameters (`Float32`, `Float16`, `BFloat16`, `Float64`) are differentiated.
- Memory: torch keeps what the backward pass needs of the forward pass until the gradient is taken,
  with no rematerialization.

## The formats

| Training backend | Format (`TrainingFormats`) | Who computes the gradient | Runs on |
|---|---|---|---|
| `TrainingBackend.Shorokoo` (default) | `Onnx` = `"onnx"` | Shorokoo, at build: the backward pass is written out as ordinary operators | every backend |
| `TrainingBackend.Native` | `OnnxAutoGrad` = `"onnx-autograd/1"` | the execution backend, when it runs the step | a backend that accepts the format |

Both are ONNX, compiled and run through the same path — one compiled session per input shape, output
aliasing of the updated state, resident runs — so the rest of [training.md](training.md) holds for
both.

## What an `onnx-autograd/1` step contains

This section is for the author of a backend that accepts the format; a user choosing
`TrainingBackend.Native` needs none of it.

The step is the default step with its backward pass left out. The forward pass, the loss, the
optimizer update, the scheduled hyperparameters and the random draws are all present as ordinary ONNX
operators, and the model records `shrk_training_format` = `onnx-autograd/1` in its metadata
(`TrainingFormats.MetadataKey`). Where the default step has the backward pass, it has **exactly one**
extra node:

| | |
|---|---|
| domain | `ai.shorokoo.training` (`TrainingFormats.AutoGradDomain`), imported at version `1` |
| op type | `AutoGrad` (`TrainingFormats.AutoGradOpType`) |
| inputs | `loss`, then `wrt_0 … wrt_n-1` |
| outputs | `grad_0 … grad_n-1` |
| attributes | none; no subgraph |

- `grad_i` is the gradient of the **sum** of `loss` with respect to `wrt_i` — the seed is a tensor of
  ones shaped like `loss` — and has the shape and element type of `wrt_i`. `loss` and every `wrt_i`
  are floating-point.
- The forward pass it differentiates is the part of the graph that computes `loss` from the `wrt_i`:
  each `wrt_i` is treated as a leaf, whatever produced it.
- Where `loss` does not depend on `wrt_i`, `grad_i` is zeros.
- The node sits at the top level of the graph, never inside an `If` or `Loop` body, and a step has one.
  Each `wrt_i` is one of the model's trainable parameters, and the optimizer update reads `grad_i`.

A backend that runs the format overrides one member of `IShorokooBackend`:

```csharp
public bool AcceptsTrainingFormat(string format)
    => format is TrainingFormats.Onnx or TrainingFormats.OnnxAutoGrad;
```

A backend that wraps another must forward it, as it forwards every member with a default body.

## What differs on the native path

- **No memory-aware pass.** The default path rewrites the backward pass it builds —
  rematerialization, activation checkpointing, scheduling — to lower peak memory. On the native path
  there is no backward pass in the graph to rewrite: what the gradient keeps alive is the execution
  backend's business. `[Module(Checkpoint = true)]` has no effect there.
- **Ops without a Shorokoo gradient.** An operator Shorokoo cannot differentiate refuses a default
  rig at build; on the native path it trains if the execution backend differentiates it.
- **Non-smooth points.** Where a function has no derivative — `Relu` at 0, ties in `Max`/`Min`, the
  bounds of `Clip`, `Abs` at 0, a cast to an integer and back — the two backends may pick different
  one-sided values. Elsewhere they compute the same gradient, up to floating-point rounding.
- **Loops still unroll.** A loop on the path from the parameters to the loss must have a constant
  trip count, exactly as on the default path, and is refused at build otherwise.
- **The step graph carries the node.** `rig.TrainingStepPureGraph` holds the gradient node, so it runs
  only through its rig. Compiling it yourself (`ComputeContext.Compile`) is refused with an
  `InvalidOperationException` that says so. Saving it as `.srk` keeps the node as Shorokoo's own.

## Errors

- `NotSupportedException` at `FromScratch`, `WithTrainingBackend` or `Load`: *"The training backend
  Native (onnx-autograd/1) leaves the gradient to the execution backend, and the runtime context's
  backend … does not run steps in format 'onnx-autograd/1'."* Train on a context whose backend accepts
  the format, or use `TrainingBackend.Shorokoo`, whose steps every backend runs.
- `InvalidOperationException` from `ComputeContext.Compile` on a native rig's `TrainingStepPureGraph`:
  such a step runs only through its `TrainingRig`.

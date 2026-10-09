# The JAX backend

Related: [pytorch-backend.md](pytorch-backend.md) · [inference.md](inference.md) · [training-backends.md](training-backends.md)

Shorokoo can run a model on **JAX** instead of ONNX Runtime. The backend translates the model
Shorokoo hands every backend (serialized ONNX) into Python that calls `jax.numpy` and `jax.lax`,
and **XLA compiles it**, once per signature of input shapes and element types, into one program
for the device. Two sub-backends exist: **CPU** and **CUDA** (Linux).

It is also a **training backend**: a rig whose runtime context runs on JAX can leave its gradient to
JAX (`TrainingBackend.Native`), and XLA then compiles the whole training step (forward pass,
backward pass and optimizer update) as one program. See [Training](#training).

It shares the translation, the embedded CPython and the Python environment with the
[PyTorch backend](pytorch-backend.md); read that page for the environment. This page covers
what is JAX's own.

## Facts

- Packages: `Shorokoo.Jax.Cpu` (**Linux x64 and Windows x64**) and `Shorokoo.Jax.Cuda` (**Linux x64**
  only). Each brings `Shorokoo.Jax`, `Shorokoo.PythonTranslation` and `Shorokoo.PythonHost`; JAX
  itself lives in the Python environment.
- A JAX backend is **always named**: `new ComputeContext(new JaxCpuBackend())`. It is never a
  candidate for [auto-discovery](backends-and-devices.md#auto-discovery).
- The **Python environment** is resolved as for PyTorch: the backend's options, then
  `SHOROKOO_PYTHON_ENV`, then one provisioned with uv into your user cache
  ([details](pytorch-backend.md#the-python-environment)). The provisioned environments hold PyTorch
  **and** JAX, one per CUDA major version: the CPU one serves both CPU backends, the CUDA 13 one both
  CUDA backends. A process runs **one** environment, shared by every PyTorch and JAX backend in it.
- **A model is compiled per input shape.** The first run with a new signature of input shapes and
  element types compiles it (XLA); later runs with that signature reuse the program. Where the
  model's inputs have fixed shapes, it is compiled when its session is created.
- **Unsupported models fail when the session is created**, with a `JaxUnsupportedModelException`
  naming the operator where one is at fault: what JAX cannot hold (strings, sequences, operators
  whose output shape their input's values decide), and, where the inputs' shapes are fixed, a shape,
  count or axis the graph computes from an input's values. See [Shapes](#shapes) and [Limitations](#limitations).

## Installing

```bash
dotnet add package Shorokoo
dotnet add package Shorokoo.Jax.Cpu        # or Shorokoo.Jax.Cuda
```

Provisioning is shared with the PyTorch backend: install [uv](https://docs.astral.sh/uv/) and the
first run downloads the environment (about 1.5 GB for CPU, 7 GB for CUDA).

An environment you provide must have `jax` and `numpy` installed (`jax[cuda13]` for the CUDA
backend), besides what the [PyTorch page](pytorch-backend.md#how-it-is-resolved) asks of one:

```bash
uv venv --managed-python -p 3.12 /opt/shorokoo-env
uv pip install --python /opt/shorokoo-env jax numpy
```

## Choosing the backend

```csharp
using Shorokoo.Jax.Cpu;
using Shorokoo.Runtime;

using var jax = new ComputeContext(new JaxCpuBackend());
var compiled = jax.Compile(model);
var output = compiled.Execute(input);
```

Everything that works on a `ComputeContext` works here. Contexts on ONNX Runtime, PyTorch and JAX
live side by side in one process; a tensor crosses between runtimes by copy.

As for PyTorch, constructing a backend does nothing; `Start()` resolves the environment, starts
JAX up front, and returns the environment it runs in.

### CPU and CUDA

| | `JaxCpuBackend` | `JaxCudaBackend` |
|---|---|---|
| package | `Shorokoo.Jax.Cpu` | `Shorokoo.Jax.Cuda` |
| platforms | Linux x64, Windows x64 | Linux x64 |
| device | the host | `cuda:N` — device 0 by default, `new JaxCudaBackend(1)` for another |
| environment | the CPU environment (jax `0.11.2`) | the CUDA 13 environment (`jax[cuda13]` `0.11.2`) |
| machine needs | nothing | an NVIDIA driver recent enough for CUDA 13 |
| `Description.Device` / `MemorySpace` | `Cpu` / host | `Cuda` / `MemorySpace.Cuda(N)` |

As with PyTorch, start a CUDA backend before a CPU one where a program uses both, so the process
gets the CUDA environment; a CUDA backend on a machine with no fit driver, or on Windows, refuses to
start *before* anything is provisioned (`PythonEnvironmentFailure.DeviceUnavailable` or
`UnsupportedPlatform`).

On CUDA, JAX takes the card's memory as needed, **not** 75 % up front: the backend sets
`XLA_PYTHON_CLIENT_PREALLOCATE=false` unless your environment sets it first.

## Shapes

XLA compiles a program for fixed shapes. While JAX traces the model for a signature of input shapes,
every value computed from shapes and constants alone is known (`Shape`, `Size`, and arithmetic,
`Gather`, `Concat`, `Slice`, `Cast` … of those), so a shape the graph computes from its inputs'
*shapes* reaches `Reshape`, `Expand`, `Tile`, `ConstantOfShape`, `Range`, `Slice`, `Pad`, `TopK`
and the rest as a number. A shape, count or axis computed from an input's **values** cannot
compile: such a model is refused with a `JaxUnsupportedModelException`
(`Reason = UnsupportedUsage`) naming the operator that needed the number, when the session is
created if the inputs' shapes are fixed, else by the run that first compiles it.

Control flow follows the same rule:

- An `If` whose condition is known while tracing takes its branch then. One whose condition is an
  input's value compiles both branches into XLA's conditional, which requires both to produce
  outputs of the same shapes and element types; one that does not is refused, naming `If`.
- A `Loop` whose trip count and conditions are known is unrolled (up to 64 iterations; beyond, it is
  compiled once as an XLA loop, its body seeing its iteration number as a value). One whose count is
  known but whose condition is an input's value runs its whole count as an XLA loop that stops
  changing its values once the condition turns false, and stays differentiable. One whose trip
  count is itself an input's value, or that has none, compiles as an XLA while loop, which JAX cannot
  differentiate: a training step with one between its parameters and its loss is refused, naming
  `Loop`. Either kind runs without scan outputs, whose length would depend on the values; one
  with scan outputs is refused, naming `Loop`.
- A random draw inside a loop compiled as an XLA loop draws afresh every iteration.

A training rig unrolls the loops on its loss path anyway (see
[training-backends.md](training-backends.md#what-differs-on-the-native-path)).

## Training

A `TrainingRig` trains on a JAX context like on any other. With
`trainingBackend: TrainingBackend.Native` the gradient is computed by **JAX**:

```csharp
using var jax = new ComputeContext(new JaxCpuBackend());
var rig = TrainingRig.FromScratch(model, loss, optimizer, sampleInputs, hyperparameters,
    runtimeContext: jax, trainingBackend: TrainingBackend.Native);
```

`jax.value_and_grad` differentiates the forward pass, and XLA compiles it, the backward pass and
the optimizer update as **one program**. Details and limits are in
[Training on JAX](training-backends.md#training-on-jax).

## Values

Every fixed-stride element type is supported, including `Complex64`, `Complex128` and the four 8-bit
float kinds, besides the usual integer, float, `Float16`, `BFloat16` and `Bool` types; 64-bit types
are 64-bit (the backend enables JAX's `jax_enable_x64` for the process). `UInt4` and `Int4` are
refused, as everywhere. JAX has **no string tensors and no sequences**: `CreateStringTensor` and
`CreateSequence` throw `NotSupportedException`, and a model with a string input or output, a string
constant or a sequence or optional operator is refused.

Nothing writes into a JAX array in place but XLA itself, over an input a run **donates** to it (see
[Runs](#runs)); every other output is memory of its own.

## Runs

| | CPU | CUDA |
|---|---|---|
| **Output aliasing** | XLA buffer donation (below); every output is still handed back as a host array of its own | XLA buffer donation (below): an output XLA writes over a donated input is in that input's memory on the card |
| **Where inputs and outputs are** | host memory | the card: every input is placed there before the run, and every output stays there |
| **Cancellation** (`RunSettings.CancellationToken`) | a run cancelled before it starts is refused; a run is one XLA program and is not stopped part way | same |
| **`DeviceMemory.LimitBytes`** | ignored | ignored: JAX's allocator is the process's and takes no per-run limit |
| **`RunSettings.ShrinkArenaAfterRun`** | ignored | ignored |
| **`RunSettings.Log`** | each Python warning the run raises, as a `Warning` from `JAX`; those raised compiling a session go to its context's settings (see [Log messages](backends-and-devices.md#log-messages)) | same |
| **Arena statistics** | none | JAX's allocator on the device (`Device.memory_stats`) |
| **`TraceNodePlacement`** | every node on `cpu` | every node on `cuda:N` |
| **`DeterministicCompute`** | not applied | not applied: XLA's kernels run as they otherwise would |
| **`Precision.AllowTensorFloat32`** | no effect: `float32` in full precision | off by default: every product and convolution is compiled at `Precision.HIGHEST`; on, at `Precision.HIGH`: XLA computes products in TensorFloat-32, and each convolution in TensorFloat-32 or in full precision, whichever kernel its autotuner finds faster when it compiles the program |
| **Log severity** | Python warnings a run raises are shown at `Warning` and below, not above | same |
| **When a run returns** | once the program has run | once the program is dispatched: the card may still be computing, and JAX reads an output only once it is computed, so the host's next work overlaps the card's |

**Donation.** A session donates to XLA the inputs of the output aliases it is built with — for a
training rig's step, every state field (weight, optimizer moment, step counter) with the output that
replaces it — as `jax.jit(..., donate_argnums=...)` does, and XLA writes an output over a donated
input wherever its own analysis lets it. A run donates such an input as it is only where it
consumed it and fed it as no other input; an input it only reads — a `.Shared()` tensor, a checkpoint
a resident run handed out or began from `.Shared()` — is copied in the run's memory first and the
copy donated, so it is never written over or deleted. XLA orders the program so that nothing reads a
donated input after it is written over, and computes an output it cannot write over its input into
memory of its own, so results are the same with or without donation. On a card a resident run thus
holds its state once rather than twice; on the CPU every input and output crosses into and out of
JAX by copy anyway, so donation only spares JAX's own buffers.

Floating-point products and convolutions run in the operands' full precision unless the context
allows TensorFloat-32 ([Precision](gpu-backends.md#precision-gpu-backends)): the precision is compiled
into each program, from the session's context, so sessions of either kind run side by side. A
determinant (`Det`) and its gradient are factorized in full precision either way.

## Limitations

- **Operators.** Every operator the PyTorch backend translates, except those JAX cannot hold: the
  string operators (`TfIdfVectorizer` included), sequences and optionals (`SequenceMap`, `Optional`,
  …), and the operators whose output shape their input's values decide (`NonZero`, `Unique`,
  `Compress`, `NonMaxSuppression`, `ImageDecoder`). Each is refused when the session is created,
  naming it (`Reason = UnknownOperator`).
- **Shapes from values** are refused, as [Shapes](#shapes) describes.
- **ONNX random draws differ from ONNX Runtime's.** `RandomNormal`, `RandomUniform`, their `Like`
  forms, `Bernoulli`, `Multinomial` and a training-mode `Dropout` draw from JAX's generator, with a
  fresh key every run (a node with a `seed` draws the same values every run). Shorokoo's own keyed
  draws are integer arithmetic and agree with ONNX Runtime exactly.
- **A float constant equal to an iota loses the sign of a −0.** XLA's simplifier replaces a
  constant that compares equal to an iota, such as `[-0.0, 1.0]`, with an iota, so the −0 becomes
  +0: `1 / c` gives `[inf, 1]` where the constant says `[-inf, 1]`. A constant that is not an iota,
  such as `[-0.0, 5.0]`, keeps its sign. Accepted as XLA's behaviour
  ([#441](https://github.com/Shorokoo/Shorokoo/issues/441)).
- **Compilation takes time.** The first run of each input-shape signature pays XLA's compilation; a
  session keeps the programs of its 16 most recent signatures.
- **Device memory is JAX's, per process**, and a context's per-session memory settings do not reach
  it.
- **Crash isolation**: the interpreter runs in your process, so a fault inside JAX or XLA takes the
  process down with it.

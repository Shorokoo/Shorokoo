# The JAX backend

Related: [pytorch-backend.md](pytorch-backend.md) · [inference.md](inference.md) · [training-backends.md](training-backends.md)

Shorokoo can run a model on **JAX** instead of ONNX Runtime. The backend translates the model
Shorokoo hands every backend (serialized ONNX) into Python that calls `jax.numpy` and `jax.lax`,
and **XLA compiles it** — once per signature of input shapes and element types — into one program
for the device. Two sub-backends exist: **CPU** and **CUDA** (Linux).

It is also a **training backend**: a rig whose runtime context runs on JAX can leave its gradient to
JAX (`TrainingBackend.Native`), and XLA then compiles the whole training step — forward pass,
backward pass and optimizer update — as one program. See [Training](#training).

It shares everything but the operators with the [PyTorch backend](pytorch-backend.md): the same
translation, the same embedded CPython, the same Python environment. Read that page for the
environment; this one says what is JAX's own.

## Facts

- Packages: `Shorokoo.Jax.Cpu` (**Linux x64 and Windows x64**) and `Shorokoo.Jax.Cuda` (**Linux x64**
  only: JAX's CUDA plugin is built for no other system). Each brings `Shorokoo.Jax` (the backend),
  `Shorokoo.PythonTranslation` (the translator it shares with PyTorch) and `Shorokoo.PythonHost`
  (the embedded interpreter). Neither ships JAX itself: that lives in the Python environment.
- A JAX backend is **always named**: `new ComputeContext(new JaxCpuBackend())`. It is never a
  candidate for [auto-discovery](inference.md#auto-discovery).
- The **Python environment** is resolved exactly as for PyTorch — the backend's options, then
  `SHOROKOO_PYTHON_ENV`, then one provisioned with uv into your user cache
  ([details](pytorch-backend.md#the-python-environment)). The provisioned environments hold PyTorch
  **and** JAX, one per CUDA major version: the CPU one serves both CPU backends, the CUDA 13 one both
  CUDA backends. A process runs **one** environment, shared by every PyTorch and JAX backend in it.
- **A model is compiled per input shape.** The first run with a new signature of input shapes and
  element types compiles it (XLA); later runs with that signature reuse the program. Where the
  model's inputs have fixed shapes, it is compiled when its session is created.
- **Unsupported models fail when the session is created**, with a `JaxUnsupportedModelException`
  naming the operator: what JAX cannot hold (strings, sequences, operators whose output shape their
  input's values decide), and — where the inputs' shapes are fixed — a shape, count or axis the graph
  computes from an input's values. See [Shapes](#shapes) and [Limitations](#limitations).

## Installing

```bash
dotnet add package Shorokoo
dotnet add package Shorokoo.Jax.Cpu        # or Shorokoo.Jax.Cuda
```

Provisioning is the PyTorch backend's: install [uv](https://docs.astral.sh/uv/) and the first run
downloads CPython 3.12 and the locked packages into the cache. The CPU environment is about 1.5 GB,
the CUDA one about 7 GB; the NVIDIA libraries in it serve PyTorch and JAX alike.

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

Everything that works on a `ComputeContext` works here: one-shot `Execute`, `Compile` and repeated
runs, consumed and `.Shared()` feeds, `TensorData.To(context)`. Contexts on ONNX Runtime, PyTorch
and JAX live side by side in one process; a tensor crosses between runtimes by copy.

As for PyTorch, constructing a backend does nothing yet; `Start()` resolves the environment and
starts JAX up front, and returns the environment it runs in.

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

On CUDA, JAX takes the card's memory as it needs it, **not** 75 % of it up front as JAX does by
default, because the card is shared with the other backends in the process. The backend sets
`XLA_PYTHON_CLIENT_PREALLOCATE=false` unless your program's environment sets it first.

## Shapes

XLA compiles a program for fixed shapes. While JAX traces the model for a signature of input shapes,
every value computed from shapes and constants alone is known — `Shape`, `Size`, and arithmetic,
`Gather`, `Concat`, `Slice`, `Cast` … of those — so a shape the graph computes from its inputs'
*shapes* reaches `Reshape`, `Expand`, `Tile`, `ConstantOfShape`, `Range`, `Slice`, `Pad`, `TopK`
and the rest as a number, as it must. What cannot compile is a shape, count or axis computed from an
input's **values**: such a model is refused with a `JaxUnsupportedModelException`
(`Reason = UnsupportedUsage`) naming the operator that needed the number — when the session is
created where the inputs' shapes are fixed, else by the run that first compiles it.

Control flow follows the same rule:

- An `If` whose condition is known while tracing takes its branch then. One whose condition is an
  input's value compiles both branches into XLA's conditional, which requires both to produce
  outputs of the same shapes and element types; one that does not is refused, naming `If`.
- A `Loop` whose trip count and conditions are known is unrolled (up to 64 iterations; beyond, it is
  compiled once as an XLA loop, its body seeing its iteration number as a value). One whose trip
  count or condition is an input's value compiles as an XLA while loop — without scan outputs, whose
  length would then depend on the values; one with scan outputs is refused, naming `Loop`.

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

The step's forward pass is written as a function of the trainable parameters, `jax.value_and_grad`
differentiates it, and the optimizer update reads the gradients — all in one traced function, so XLA
compiles the forward pass, the backward pass it derives and the update as **one program**, fusing
across them. What that means for the rig is in
[Training on JAX](training-backends.md#training-on-jax).

## Values

Every fixed-stride element type is supported, including `Complex64`, `Complex128` and the four 8-bit
float kinds, besides the usual integer, float, `Float16`, `BFloat16` and `Bool` types; 64-bit types
are 64-bit (the backend enables JAX's `jax_enable_x64` for the process). `UInt4` and `Int4` are
refused, as everywhere. JAX has **no string tensors and no sequences**: `CreateStringTensor` and
`CreateSequence` throw `NotSupportedException`, and a model with a string input or output, a string
constant or a sequence or optional operator is refused.

Every output a run hands back is memory of its own. A JAX array is never written in place, so no run
writes an output into an input it consumed: a session binds no output aliases.

## Runs

| | CPU | CUDA |
|---|---|---|
| **Output aliasing** | none: nothing is written in place | none |
| **Resident runs** (`RunRetainingOutputs`) | nothing to retain: outputs are on the host | a kept output stays on the card; inputs already there are read in place |
| **Cancellation** (`RunSettings.CancellationToken`) | a run cancelled before it starts is refused; a run is one XLA program and is not stopped part way | same |
| **`DeviceMemory.LimitBytes`** | ignored | ignored: JAX's allocator is the process's and takes no per-run limit |
| **`RunSettings.ShrinkArenaAfterRun`** | ignored | ignored |
| **Arena statistics** | none | JAX's allocator on the device (`Device.memory_stats`) |
| **`TraceNodePlacement`** | every node on `cpu` | every node on `cuda:N` |
| **Log severity** | Python warnings a run raises are shown at `Warning` and below, not above | same |

Floating-point products and convolutions are computed in full precision: XLA's default on a card
may round `float32` operands to TensorFloat-32, which the backend does not ask for.

## Limitations

- **Operators.** Every operator the PyTorch backend translates, except those JAX cannot hold: the
  string operators (`TfIdfVectorizer` included), sequences and optionals (`SequenceMap`, `Optional`,
  …), and the operators whose output shape their input's values decide — `NonZero`, `Unique`,
  `Compress`, `NonMaxSuppression`, `ImageDecoder`. Each is refused when the session is created,
  naming it (`Reason = UnknownOperator`).
- **Shapes from values** are refused, as [Shapes](#shapes) describes.
- **ONNX random draws differ from ONNX Runtime's.** `RandomNormal`, `RandomUniform`, their `Like`
  forms, `Bernoulli`, `Multinomial` and a training-mode `Dropout` draw from JAX's generator, with a
  fresh key every run (a node with a `seed` draws the same values every run). Shorokoo's own keyed
  draws are integer arithmetic and agree with ONNX Runtime exactly.
- **Compilation takes time.** The first run of each input-shape signature pays XLA's compilation; a
  session keeps the programs of its 16 most recent signatures.
- **Device memory is JAX's, per process**, and a context's per-session memory settings do not reach
  it.
- **Crash isolation**: the interpreter runs in your process, so a fault inside JAX or XLA takes the
  process down with it.

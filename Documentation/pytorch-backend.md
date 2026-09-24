# The PyTorch backend

Related: [inference.md](inference.md) · [operator-support.md](operator-support.md) · [training-backends.md](training-backends.md)

Shorokoo can run a model on **PyTorch** instead of ONNX Runtime. The backend translates the
model Shorokoo hands every backend (serialized ONNX) into Python that calls PyTorch, and runs
it in a CPython interpreter embedded in your .NET process. Two sub-backends exist: **CPU** and
**CUDA**.

It is an execution backend like the ONNX Runtime ones — the same `ComputeContext`, the same
`TensorData`, the same compile-once-run-many workflow — with two differences you have to know
about: it is **never picked for you** (you always name it), and it needs a **Python
environment**, which it can provision itself the first time you use it.

## Facts

- Packages: `Shorokoo.PyTorch.Cpu` and `Shorokoo.PyTorch.Cuda`, for **Linux x64**. Each
  brings `Shorokoo.PyTorch` (the translator and the backend) and `Shorokoo.PythonHost` (the
  embedded interpreter). Neither ships PyTorch itself: that lives in the Python environment.
- A torch backend is **always named**: `new ComputeContext(new TorchCpuBackend())`. It is
  never a candidate for [auto-discovery](inference.md#auto-discovery), so referencing it
  beside an ONNX Runtime package does not make discovery ambiguous, and `ComputeContext.Default`
  stays on ONNX Runtime.
- The **Python environment** is, in order: the one you name in the backend's options; the one
  the `SHOROKOO_PYTHON_ENV` environment variable names; or one provisioned with
  [uv](https://docs.astral.sh/uv/) from the package's lock file into your user cache, on first
  use. See [The Python environment](#the-python-environment).
- **Unsupported models fail when the session is created**, never on the first run, with a
  `TorchUnsupportedModelException` naming the operator (or attribute) the backend cannot run.
  Operator coverage is partial today; see [Limitations](#limitations).
- One process runs **one** Python interpreter over **one** environment, shared by every torch
  backend in it.

## Installing

```bash
dotnet add package Shorokoo
dotnet add package Shorokoo.PyTorch.Cpu      # or Shorokoo.PyTorch.Cuda
```

If you want the backend to provision its own environment (the default), install **uv** and
make sure it is on `PATH` — or point `SHOROKOO_UV` at it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

The first run that needs PyTorch then downloads CPython 3.12 and the locked packages into the
cache. That needs the network and takes a while: the CPU environment is under 1 GB, the CUDA
one several GB (the NVIDIA libraries come with it). Every later run, in any process, reuses it.

On a machine without network access, provision an environment elsewhere (or build one by
hand, see below) and name it with `SHOROKOO_PYTHON_ENV`.

## Choosing the backend

```csharp
using Shorokoo.PyTorch.Cpu;
using Shorokoo.Runtime;

using var torch = new ComputeContext(new TorchCpuBackend());
var compiled = torch.Compile(model);
var output = compiled.Execute(input);
```

Everything that works on a `ComputeContext` works here: one-shot `Execute`, `Compile` and
repeated runs, consumed and `.Shared()` feeds, `TensorData.To(context)`. Contexts on ONNX
Runtime and on PyTorch live side by side in one process; a tensor crosses between them by
copy, since they are different runtimes.

Constructing a backend does nothing yet: the environment is resolved and PyTorch started the
first time a session or a tensor is needed. To fail at startup instead, call `Start()`, which
returns the environment it runs in:

```csharp
var backend = new TorchCpuBackend();
var environment = backend.Start();          // throws PythonEnvironmentException if it cannot
Console.WriteLine(environment);             // "/home/me/.cache/shorokoo/python-envs/cpu-… (CPython 3.12.11, Provisioned)"
```

### CPU and CUDA

| | `TorchCpuBackend` | `TorchCudaBackend` |
|---|---|---|
| package | `Shorokoo.PyTorch.Cpu` | `Shorokoo.PyTorch.Cuda` |
| device | the host | `cuda:N` — device 0 by default, `new TorchCudaBackend(1)` for another |
| environment | PyTorch's CPU build | PyTorch built for CUDA 13 |
| machine needs | nothing | an NVIDIA driver recent enough for CUDA 13; the CUDA libraries come with the environment |
| `Description.Device` / `MemorySpace` | `Cpu` / host | `Cuda` / `MemorySpace.Cuda(N)` |

On CUDA, a tensor moved to the context (`TensorData.To(context)`) is on the card, and a run
reads it there. A run hands its outputs back in host memory, except those a resident run asks
to keep on the card, which stay there (`IsHostAccessible` is false) until copied home.

**One environment per process.** The CUDA build of PyTorch also runs on the CPU, the CPU build
cannot run on the card, and the process gets whichever environment the first torch backend
started. If a program uses both, start the CUDA one first:

```csharp
var cuda = new TorchCudaBackend();
cuda.Start();                                // the CUDA environment, for the whole process
var cpu = new TorchCpuBackend();             // shares it
```

A CUDA backend started in a process already running the CPU build fails with
`PythonEnvironmentFailure.DeviceUnavailable`, saying so.

## The Python environment

### How it is resolved

1. **The backend's options.** `new TorchCpuBackend(new PythonEnvironmentOptions { EnvironmentPath = "/opt/torch-env" })`
   uses that virtual environment as it is. Nothing is installed into it.
2. **`SHOROKOO_PYTHON_ENV`**, a virtual environment directory, used the same way.
3. **The cache.** Otherwise the environment described by the package's lock file, under
   `$XDG_CACHE_HOME/shorokoo/python-envs/` (or `~/.cache/…`; `%LOCALAPPDATA%\shorokoo\…` on
   Windows), in a folder named after the lock and a hash of it — `cpu-6279c3b3d69b31c7`. If it
   is not there yet it is created with uv: `uv python install 3.12`, `uv venv`, then
   `uv pip install` of the lock. Two processes starting at once build it once — the second
   waits on a lock file beside it — and a build that was interrupted is started over rather
   than used. `PythonEnvironmentOptions.CacheDirectory` moves the cache;
   `PythonEnvironmentOptions.UvPath` or `SHOROKOO_UV` names the uv to use.

An environment you provide must be a **CPython 3.12** virtual environment (a folder with a
`pyvenv.cfg`) whose base interpreter has a shared `libpython3.12.so`, with `torch` and `numpy`
installed. uv's own Python builds have the shared library; so does the environment made by

```bash
uv venv --managed-python -p 3.12 /opt/torch-env
VIRTUAL_ENV=/opt/torch-env uv pip install torch numpy --index-url https://download.pytorch.org/whl/cpu
```

A distribution's system Python often lacks it, which is refused with
`PythonEnvironmentFailure.LibPythonNotFound`.

### When it goes wrong

Every environment problem is a `PythonEnvironmentException` whose `Failure` says which, and
whose message names what is missing:

| `Failure` | meaning |
|---|---|
| `EnvironmentNotFound` | the named environment does not exist |
| `NotAVirtualEnvironment` | the named folder has no `pyvenv.cfg` |
| `WrongPythonVersion` | the environment is not Python 3.12 |
| `LibPythonNotFound` | its base interpreter has no shared library to embed |
| `UvNotFound` | provisioning needs uv, and there is none |
| `NetworkUnavailable` | provisioning could not reach the package index |
| `ProvisioningFailed` | uv failed for another reason; its output is quoted |
| `ProvisioningTimedOut` | another process held the provisioning lock too long |
| `MissingPackage` | the environment cannot import `torch` or `numpy` |
| `EnvironmentConflict` | the process already runs Python over a different environment |
| `InterpreterFailed` | CPython itself would not start |
| `DeviceUnavailable` | a CUDA backend, and PyTorch sees no such device |

## Training

A `TrainingRig` trains on a torch context like on any other: pass it as the rig's `runtimeContext`.
With `trainingBackend: TrainingBackend.Native` the gradient is computed by **torch autograd** rather
than by Shorokoo:

```csharp
using var torch = new ComputeContext(new TorchCpuBackend());
var rig = TrainingRig.FromScratch(model, loss, optimizer, sampleInputs, hyperparameters,
    runtimeContext: torch, trainingBackend: TrainingBackend.Native);
```

Both torch backends accept both training formats (`AcceptsTrainingFormat` is true for
`TrainingFormats.Onnx` and `TrainingFormats.OnnxAutoGrad`). Everything else about the rig — the
optimizer, schedules, random draws, checkpoints, resident runs — is unchanged, and a torch-trained
rig agrees with a Shorokoo-trained one step for step up to floating-point rounding. What the step
does on torch, and its limits, are in [Training on PyTorch](training-backends.md#training-on-pytorch).

## Values

Every element type PyTorch has is supported, which includes a few ONNX Runtime cannot build
from bytes: `Complex64`, `Complex128` and the four 8-bit float kinds, besides the usual integer,
float, `Float16`, `BFloat16` and `Bool` types. `UInt4` and `Int4` are refused, as everywhere.
String tensors work, and always stay on the host; sequences of tensors work.

Every output a run hands back is memory of its own — never an input's or a weight's, even where
the model returns one of those unchanged — so writing to an output never changes anything
else.

## Limitations

- **Operator coverage is partial.** The elementwise math and activation operators, the
  comparisons and logic operators, the reductions, `MatMul`/`Gemm`, the shape operators
  (`Reshape`, `Transpose`, `Concat`, `Split`, `Squeeze`/`Unsqueeze`, `Shape`, `Expand`, `Tile`,
  `Pad`, `Constant`, `ConstantOfShape`, `Range`, …), `Gather`/`GatherElements`/`Slice`/`Compress`,
  and `If`/`Loop` are translated, and so are Shorokoo's own random draws (Dropout masks and the
  like), which reach the backend as integer operators. Convolution and pooling, normalization,
  recurrent networks, the ONNX random operators, sequences, strings, signal, image and
  quantization operators are not yet: a model using one is refused when its session is created,
  naming it.
- **Linux x64 only.** The lock files are resolved for Linux x64, and so are the packages.
- **Device memory settings are not applied.** PyTorch's caching allocator is process-wide, so
  a context's `DeviceMemory` budget still bounds the tensors the context holds, but a session's
  arena settings do not reach PyTorch, and sessions report no arena statistics.
- **No output aliasing.** A run that consumes its inputs releases them; it never writes an
  output into one.
- **Cancellation** is honoured before a run starts, not during it.
- **Crash isolation**: the interpreter runs in your process, so a fault inside PyTorch takes
  the process down with it.

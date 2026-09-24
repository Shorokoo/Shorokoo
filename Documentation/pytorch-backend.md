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

- Packages: `Shorokoo.PyTorch.Cpu` and `Shorokoo.PyTorch.Cuda`, for **Linux x64 and Windows x64**. Each
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
| environment lock | torch `2.14.0+cpu` | Linux: PyPI's torch `2.14.0` (CUDA 13); Windows: `2.14.0+cu130` from PyTorch's `cu130` index |
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

**No driver, no provisioning.** The CUDA package's manifest declares the driver it needs
(`RequiresCudaDriver = "13.0"`), so `BackendPackage.Probe` answers
`BackendRejection.MissingCudaDriver` on a machine with no NVIDIA driver, one too old for CUDA 13,
or one that sees no device — saying which. Starting the backend on such a machine fails the same
way, with `PythonEnvironmentFailure.DeviceUnavailable`, *before* anything is provisioned: it never
downloads several gigabytes of CUDA libraries for a card that is not there.

## The Python environment

### How it is resolved

1. **The backend's options.** `new TorchCpuBackend(new PythonEnvironmentOptions { EnvironmentPath = "/opt/torch-env" })`
   uses that virtual environment as it is. Nothing is installed into it.
2. **`SHOROKOO_PYTHON_ENV`**, a virtual environment directory, used the same way.
3. **The cache.** Otherwise the environment described by the package's lock file for this
   platform (Linux x64 or Windows x64 — each has its own), under
   `$XDG_CACHE_HOME/shorokoo/python-envs/` (or `~/.cache/…`; `%LOCALAPPDATA%\shorokoo\…` on
   Windows), in a folder named after the lock and a hash of it — `cpu-b568d348aeca20e7`. If it
   is not there yet it is created with uv: `uv python install 3.12`, `uv venv`, then
   `uv pip install --require-hashes` of the lock, which records the hash of every file it may
   install. Two processes starting at once build it once — the second
   waits on a lock file beside it — and a build that was interrupted is started over rather
   than used. uv runs without the `UV_` variables that could make it install something else or
   somewhere else (it keeps `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, `UV_NATIVE_TLS` and
   `UV_HTTP_TIMEOUT`, and the proxy and certificate variables), and each step is stopped if
   provisioning outlasts `PythonEnvironmentOptions.ProvisioningTimeout`.
   `PythonEnvironmentOptions.CacheDirectory` moves the cache;
   `PythonEnvironmentOptions.UvPath` or `SHOROKOO_UV` names the uv to use. An environment built
   from an older lock stays in the cache until you delete its folder.

An environment you provide must be a **CPython 3.12** virtual environment (a folder with a
`pyvenv.cfg`) whose base interpreter has a shared library — `libpython3.12.so` on Linux,
`python312.dll` beside `python.exe` on Windows — with `torch` and `numpy` installed, and `pillow`
for a model that uses `ImageDecoder`. uv's own Python builds have the shared library; so does the
environment made by

```bash
uv venv --managed-python -p 3.12 /opt/torch-env
uv pip install --python /opt/torch-env torch numpy pillow --index-url https://download.pytorch.org/whl/cpu
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
| `ProvisioningTimedOut` | provisioning ran past `ProvisioningTimeout`: another process held its lock, or a uv step had not finished and was stopped |
| `MissingPackage` | the environment cannot import `torch` or `numpy` |
| `EnvironmentConflict` | the process already runs Python over a different environment |
| `InterpreterFailed` | CPython itself would not start |
| `DeviceUnavailable` | a CUDA backend, and no NVIDIA driver fit for CUDA 13, or PyTorch sees no such device |

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
else. The one exception is an output written into an input the run *consumed* (see
[output aliasing](#runs)), which nothing else holds any more.

## Runs

What a run does with the settings every backend is handed, on each device:

| | CPU | CUDA |
|---|---|---|
| **Output aliasing** (a run writing an output into a consumed input) | yes, for an output produced by `Add`/`Sub`/`Mul`/`Div` | the same on the card for an output kept there; any output fetched home is copied into a consumed host input |
| **Resident runs** (`RunRetainingOutputs`) | nothing to retain: outputs are on the host | a kept tensor output stays on the card; inputs already there are read in place |
| **Cancellation** (`RunSettings.CancellationToken`) | stops before the next node | stops before the next node |
| **`DeviceMemory.LimitBytes`** | ignored, as on every CPU backend | caps each run's allocations (see below) |
| **`RunSettings.ShrinkArenaAfterRun`** | ignored | `torch.cuda.empty_cache()` after the run |
| **Arena statistics** / `RunStats` | none | torch's caching allocator on the device |
| **`TraceNodePlacement`** | every node on `cpu` | every node on `cuda:N` |
| **Log severity** | Python warnings a run raises are shown at `Warning` and below, not above | same |

**Output aliasing.** A compiled graph that pairs an output with an input it replaces — the
training rig pairs each updated parameter with the parameter — has a run that *consumed* that
input write the output into its memory, where the graph proves nothing reads the input
afterwards. torch writes an `Add`, `Sub`, `Mul` or `Div` into memory it is handed (`out=`), so such
an output costs no memory at all: the optimizer's `p - lr * g` is written over `p`. Other operators
cannot be told where to write, so on the CPU their pairs are not bound. On CUDA, an output the run
fetches back is copied home into the consumed host tensor rather than into memory of its own. A
write is also declined wherever something the rest of the run still reads could be the input's
memory under another name: torch hands back views where ONNX Runtime copies (`Transpose`,
`Expand`, `Slice`, a `Cast` to the same type), and those are checked when the run gets there.

**Intermediate values.** A run lets go of each value the graph computes right after the last node
that reads it — in the main graph, in a function's body and in a branch or loop body alike — as
ONNX Runtime frees a buffer after its last use, so a run's peak is what is live at once, not the
sum of everything the graph computes. In a training step whose gradient torch takes, what the
backward pass needs is kept by torch's autograd graph until the gradient is taken.

**Device memory on CUDA.** torch has one caching allocator per device for the whole process,
where ONNX Runtime gives each session an arena of its own, so the per-session settings map as
far as they can and no further:

- `LimitBytes` (which the budget of a context sets per session) caps a run at what is allocated on
  the device when it starts plus the limit, through torch's per-process memory fraction; the
  allocator's cached blocks are handed back first, so they cannot serve the run past the cap. A run
  that needs more fails with an `InvalidOperationException` naming the limit. Because the cap is the
  process's, capped runs on one device take turns, and a run of *another* context on that device
  while one is in progress is capped too.
- `ArenaExtend` has no counterpart: torch's allocator grows its own way, configured process-wide by
  `PYTORCH_CUDA_ALLOC_CONF` before the backend starts.
- `ReadArenaStatistics` reads `torch.cuda.memory_stats`: `InUseBytes`, `MaxInUseBytes`,
  `TotalAllocatedBytes`, `AllocationCount`, `ArenaExtensionCount` (segments held) and
  `ArenaShrinkageCount` (segments released) are the device allocator's — every session's on that
  device, not one session's — and `LimitBytes` is the session's limit or -1. torch records no
  largest single allocation and no reserves, so `MaxAllocSizeBytes` and `ReserveCount` are 0.
- There is no pinned host arena: host–device copies go through ordinary host memory, so
  `ReadPinnedArenaStatistics` is null.

**Cancellation** is checked before every node, so a run stops at the next node boundary —
inside a `Loop` too, once per iteration — and throws `OperationCanceledException` carrying the
token. A single node that is one long kernel is not interrupted.

## Limitations

- **Operator coverage is that of the operators Shorokoo builds.** The elementwise math and activation operators, the
  comparisons and logic operators, the reductions, `MatMul`/`Gemm`, the shape operators
  (`Reshape`, `Transpose`, `Concat`, `Split`, `Squeeze`/`Unsqueeze`, `Shape`, `Expand`, `Tile`,
  `Pad`, `Constant`, `ConstantOfShape`, `Range`, `OneHot`, `EyeLike`, `ReverseSequence`,
  `TensorScatter`, …), the indexing operators (`Gather`, `GatherElements`, `GatherND`, `Slice`,
  `Compress`, `ScatterElements`, `ScatterND`, `TopK`, `Unique`, `NonZero`), `If`/`Loop`,
  convolution and pooling, normalization and the losses, `Einsum`/`Det`/`MatMulInteger`, the
  image and geometry operators (`Resize`, `GridSample`, `AffineGrid`, `RoiAlign`,
  `NonMaxSuppression`, `CenterCropPad`, `Col2Im`, `DepthToSpace`/`SpaceToDepth`, `ImageDecoder`,
  …), the
  recurrent networks (`RNN`, `GRU`, `LSTM` — `layout=1` included, which ONNX Runtime's CPU
  kernels refuse), the signal operators (`DFT`, `STFT`, the windows, `MelWeightMatrix`), the
  quantization operators (`QuantizeLinear`, `DequantizeLinear`, `DynamicQuantizeLinear`,
  `QLinearMatMul`, `QLinearConv`), sequences and optionals (`SequenceMap` included), the string
  operators (`TfIdfVectorizer` included), and the ONNX random operators and `Dropout` are
  translated, as are functions that take attributes and Shorokoo's own random draws, which reach
  the backend as integer operators. `ImageDecoder`, which ONNX Runtime has no kernel for, decodes
  with Pillow every format the ONNX specification names (BMP, JPEG, JPEG 2000, TIFF, PNG, WebP and
  the portable any-maps). A model using an operator the backend does not translate — one from a
  domain other than ONNX's own, say — is refused when its session is created, naming it.
- **ONNX random draws differ from ONNX Runtime's.** `RandomNormal`, `RandomUniform`, their `Like`
  forms, `Bernoulli`, `Multinomial` and a training-mode `Dropout` draw from PyTorch's generator:
  the distribution, shape and element type are the operator's, and a node with a `seed` draws the
  same values on every run, but not the values ONNX Runtime draws. Shorokoo's own keyed draws
  are integer arithmetic, and agree with ONNX Runtime exactly.
- **Linux x64 and Windows x64 only**, the platforms there are lock files for.
- **Device memory is torch's, per process** — see [Runs](#runs) for what a context's settings
  can and cannot reach.
- **Node placement records no bytes.** A traced session names the device every node ran on, which
  is the session's; the per-node byte counts are 0.
- **Crash isolation**: the interpreter runs in your process, so a fault inside PyTorch takes
  the process down with it.

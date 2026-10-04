# The PyTorch backend

Related: [inference.md](inference.md) · [operator-support.md](operator-support.md) · [training-backends.md](training-backends.md)

Shorokoo can run a model on **PyTorch** instead of ONNX Runtime. The backend translates the
model Shorokoo hands every backend (serialized ONNX) into Python that calls PyTorch, and runs
it in a CPython interpreter embedded in your .NET process. Two sub-backends exist: **CPU** and
**CUDA**.

It uses the same `ComputeContext`, `TensorData` and compile-once-run-many workflow as the
ONNX Runtime backends, with two differences: it is **never picked for you** (you always name
it), and it needs a **Python environment**, which it can provision on first use.

## Facts

- Packages: `Shorokoo.PyTorch.Cpu` and `Shorokoo.PyTorch.Cuda`, for **Linux x64 and Windows x64**. Each
  brings `Shorokoo.PyTorch`, `Shorokoo.PythonTranslation` (shared with the
  [JAX backend](jax-backend.md)) and `Shorokoo.PythonHost`. PyTorch itself lives in the Python
  environment.
- A torch backend is **always named**: `new ComputeContext(new TorchCpuBackend())`. It is
  never a candidate for [auto-discovery](inference.md#auto-discovery), and
  `ComputeContext.Default` stays on ONNX Runtime.
- The **Python environment** is, in order: the one you name in the backend's options; the one
  the `SHOROKOO_PYTHON_ENV` environment variable names; or one provisioned with
  [uv](https://docs.astral.sh/uv/) from the package's lock file into your user cache, on first
  use. See [The Python environment](#the-python-environment).
- **Unsupported models fail when the session is created** with a
  `TorchUnsupportedModelException` naming the operator (or attribute) the backend cannot run.
  Operator coverage is partial; see [Limitations](#limitations).
- One process runs **one** Python interpreter over **one** environment, shared by every torch
  and [JAX](jax-backend.md) backend in it. The environments provisioned from the lock files hold
  PyTorch and JAX together, one per CUDA major version.

## Installing

```bash
dotnet add package Shorokoo
dotnet add package Shorokoo.PyTorch.Cpu      # or Shorokoo.PyTorch.Cuda
```

For the backend to provision its own environment (the default), install **uv** on `PATH`,
or point `SHOROKOO_UV` at it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

The first run that needs PyTorch downloads CPython 3.12 and the locked packages into the
cache. This needs the network: the CPU environment is about 1.5 GB, the CUDA one about 7 GB
(including the NVIDIA libraries). Both also hold JAX, for the
[JAX backend](jax-backend.md). Every later run, in any process, reuses it.

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
repeated runs, consumed and `.Shared()` feeds, `TensorData.To(context)`. ONNX Runtime and
PyTorch contexts live side by side in one process; a tensor crosses between them by copy.

Constructing a backend does nothing: the environment is resolved and PyTorch started the
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
| environment lock | torch `2.14.0+cpu`, jax `0.11.2` | Linux: PyPI's torch `2.14.0` (CUDA 13) and `jax[cuda13]` `0.11.2`; Windows: `2.14.0+cu130` from PyTorch's `cu130` index, and jax `0.11.2` for the CPU |
| `Description.Device` / `MemorySpace` | `Cpu` / host | `Cuda` / `MemorySpace.Cuda(N)` |

On CUDA, a tensor moved to the context (`TensorData.To(context)`) is on the card and read
there. Every input is placed on the card before a run, and every tensor output stays there
(`TensorData.IsHostResident` is false): reading its values copies them to the host and leaves it
there, and `ToHost()` makes a copy of it in host memory. String
tensors and sequences are read and left in host memory.

**One environment per process.** The process gets whichever environment the first torch
backend started. The CUDA build of PyTorch also runs on the CPU; the CPU build cannot run on
the card. If a program uses both, start the CUDA one first:

```csharp
var cuda = new TorchCudaBackend();
cuda.Start();                                // the CUDA environment, for the whole process
var cpu = new TorchCpuBackend();             // shares it
```

A CUDA backend started in a process already running the CPU build fails with
`PythonEnvironmentFailure.DeviceUnavailable`.

**No driver, no provisioning.** The CUDA package's manifest declares the driver it needs
(`RequiresCudaDriver = "13.0"`), so `BackendPackage.Probe` answers
`BackendRejection.MissingCudaDriver` on a machine with no NVIDIA driver, one too old for CUDA 13,
or one that sees no device, saying which. Starting the backend on such a machine fails with
`PythonEnvironmentFailure.DeviceUnavailable` *before* anything is provisioned, so it never
downloads the CUDA libraries for a card that is not there.

**Beside an ONNX Runtime CUDA backend.** A process can run `TorchCudaBackend` together with
`WinGpuBackend`, `LinuxGpuBackend` or a CUDA backend loaded through `IsolatedBackend`, whichever
starts first: every CUDA backend runs on one pinned copy of cuDNN and cuBLAS, which is the release
the CUDA environment's PyTorch carries — see
[The NVIDIA libraries the CUDA backends run on](inference.md#the-nvidia-libraries-the-cuda-backends-run-on).
When the CUDA environment is provisioned, or first used, its copies of those libraries become
hard links to the shared cache's, filling the cache from them if it is empty, so nothing more is
downloaded and PyTorch loads the very same files as the other backends. An environment you name
is never modified: one whose PyTorch carries the pinned release runs beside another CUDA backend
on its own copy of that release. On Windows, one whose PyTorch bundles another release cannot load
it into a process that already holds the pinned one, and starting it there fails with
`PythonEnvironmentFailure.CudaLibraryConflict`, naming the copies held; on Linux its PyTorch binds
to the pinned copy already loaded, and runs where that release has everything PyTorch takes from
it, and where it lacks something, starting it fails with `CudaLibraryConflict` too.

## The Python environment

### How it is resolved

1. **The backend's options.** `new TorchCpuBackend(new PythonEnvironmentOptions { EnvironmentPath = "/opt/torch-env" })`
   uses that virtual environment as it is. Nothing is installed into it.
2. **`SHOROKOO_PYTHON_ENV`**, a virtual environment directory, used the same way.
3. **The cache.** Otherwise the environment described by the package's lock file for this
   platform (Linux x64 or Windows x64 — each has its own), under
   `$XDG_CACHE_HOME/shorokoo/python-envs/` (or `~/.cache/…`; `%LOCALAPPDATA%\shorokoo\…` on
   Windows), in a folder named after the lock and its hash (`cpu-b568d348aeca20e7`). If absent
   it is created with uv (`uv python install 3.12`, `uv venv`, `uv pip install --require-hashes`
   of the lock). Concurrent processes build it once; an interrupted build is started over.
   Other `UV_` variables are ignored; uv keeps only `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`,
   `UV_NATIVE_TLS`, `UV_HTTP_TIMEOUT` and the proxy and certificate variables. Provisioning is
   stopped past `PythonEnvironmentOptions.ProvisioningTimeout`.
   `PythonEnvironmentOptions.CacheDirectory` moves the cache;
   `PythonEnvironmentOptions.UvPath` or `SHOROKOO_UV` names the uv to use. An environment built
   from a different lock stays in the cache until you delete its folder.

An environment you provide must be a **CPython 3.12** virtual environment (a folder with a
`pyvenv.cfg`) whose base interpreter has a shared library — `libpython3.12.so` on Linux,
`python312.dll` beside `python.exe` on Windows — with `torch` and `numpy` installed, `pillow`
for a model that uses `ImageDecoder`, and `google-re2` for one that uses `RegexFullMatch`. uv's
own Python builds have the shared library; so does the environment made by

```bash
uv venv --managed-python -p 3.12 /opt/torch-env
uv pip install --python /opt/torch-env torch numpy pillow google-re2 \
    --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
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
| `MissingPackage` | the environment cannot import `torch` or `numpy` (`jax` for a JAX backend) |
| `EnvironmentConflict` | the process already runs Python over a different environment |
| `InterpreterFailed` | CPython itself would not start |
| `DeviceUnavailable` | a CUDA backend, and no NVIDIA driver fit for CUDA 13, or PyTorch sees no such device |
| `UnsupportedPlatform` | the machine is not Linux or Windows on x64, the platforms there are lock files for |
| `CudaLibraryConflict` | PyTorch cannot load the CUDA libraries its environment ships, because the process already holds another release of them; the message names the copies held |

## Training

A `TrainingRig` trains on a torch context passed as its `runtimeContext`. With
`trainingBackend: TrainingBackend.Native` the gradient is computed by **torch autograd** rather
than by Shorokoo:

```csharp
using var torch = new ComputeContext(new TorchCpuBackend());
var rig = TrainingRig.FromScratch(model, loss, optimizer, sampleInputs, hyperparameters,
    runtimeContext: torch, trainingBackend: TrainingBackend.Native);
```

Both torch backends accept both training formats (`AcceptsTrainingFormat` is true for
`TrainingFormats.Onnx` and `TrainingFormats.OnnxAutoGrad`). The optimizer, schedules, random
draws, checkpoints and resident runs are unchanged, and a torch-trained rig agrees with a
Shorokoo-trained one step for step up to floating-point rounding. What the step does on torch,
and its limits, are in [Training on PyTorch](training-backends.md#training-on-pytorch).

## Values

Every element type PyTorch has is supported: the usual integer, float, `Float16`, `BFloat16`
and `Bool` types, plus `Complex64`, `Complex128` and the four 8-bit float kinds. `UInt4` and `Int4` are refused, as everywhere.
String tensors work, and always stay on the host; sequences of tensors work.

Every output is memory of its own, even where the model returns an input or weight
unchanged, except an output written into a *consumed* input
([output aliasing and placement](#runs)), which no other value's range overlaps.

## Runs

What a run does with the settings every backend is handed, on each device:

| | CPU | CUDA |
|---|---|---|
| **Output aliasing** (a run writing an output into a consumed input) | yes, for an output produced by `Add`/`Sub`/`Mul`/`Div` | the same, on the card |
| **Placement** (a run writing its values into ranges of consumed inputs) | yes: floating-point element-wise operators, matrix products, `Gemm`, `Softmax`, `LogSoftmax`, `Gelu`, `Clip` and the normalizations, fills, concatenations and copies of views; a convolution by a copy; not in a training step torch differentiates | the same, a convolution written into its range by cuDNN |
| **Where inputs and outputs are** | host memory | the card for every tensor, the host for strings and sequences: every input is moved there before the run, and every output stays there |
| **Cancellation** (`RunSettings.CancellationToken`) | stops before the next node | stops before the next node |
| **`DeviceMemory.LimitBytes`** | ignored, as on every CPU backend | caps each run's allocations (see below) |
| **`RunSettings.ShrinkArenaAfterRun`** | ignored | `torch.cuda.empty_cache()` after the run |
| **Arena statistics** / `RunStats` | none | torch's caching allocator on the device |
| **`TraceNodePlacement`** | every node on `cpu` | every node on `cuda:N` |
| **`DeterministicCompute`** | not applied | not applied: torch's kernels run as they otherwise would |
| **`Precision.AllowTensorFloat32`** | no effect: `float32` in full precision | off by default: each run sets `torch.backends.cuda.matmul.allow_tf32` and `torch.backends.cudnn.allow_tf32` off as it starts; on, it sets both on, and products, cuDNN convolutions and recurrent layers run in TensorFloat-32 |
| **Log severity** | Python warnings a run raises are shown at `Warning` and below, not above | same |

**Output aliasing.** An output paired with a consumed input (as the training rig pairs each
updated parameter with the parameter) is written into the input's memory when it comes from
`Add`, `Sub`, `Mul` or `Div`, so the optimizer's `p - lr * g` costs no memory. Other operators'
pairs are not bound. A write is declined where the input's memory may still be read through a
view (`Transpose`, `Expand`, `Slice`, a same-type `Cast`).

**Placement.** A run that consumes inputs writes its values of a mebibyte or more into ranges of
their memory where the graph proves it safe ([A run that writes into what it
consumed](inference.md#a-run-that-writes-into-what-it-consumed)): each with torch's own operator
writing into the range — an element-wise operator's or a matrix product's `out=` form (on
floating-point values), `Softmax`, `LogSoftmax` and `Gelu` through theirs, `Gemm` as a product
written into the range and scaled and summed there, `Clip` and the normalizations computed step by
step in the range, a fill, a concatenation part by part, or a copy of what a slice, reshape or
transpose reads, which an output that views an input is copied out by anyway. A convolution is
written into its range by cuDNN on a card; on the CPU, where torch's convolutions take no tensor to
write into, it is computed and copied into the range, which allocates what an unplaced run
allocates there and frees it at once. An operator with no such form is not placed. Each computes
exactly what an unplaced run computes. Where the values go is planned per run signature, for up to
8 of them, and not for a model over 16 MiB, a graph of over 20 000 nodes, or a run of a graph not
compiled. Every output a run wrote into one input stands on that input's memory, which torch frees
with the last of them.

**Intermediate values.** A run computes the graph's nodes in the order ONNX Runtime would — the
order a training step's memory-aware pass plans for — and reads each value's shape (`Shape`,
`Size`) as soon as the value is made, so no value is held only for its shape. It releases each
value after its last reader (in function, branch and loop bodies too), so its peak is what is
live at once. Where nothing reads an operand after a node, the node's result is written over that
operand rather than into memory of its own: an element-wise operator, `Clip`, `Gelu`, `Where`,
`Softmax`, `LogSoftmax`, `LayerNormalization` or `BatchNormalization` over an operand of the
result's type and shape that the run made itself, or that it consumed — a chain of such nodes
holds one value at a time. In a training step whose gradient torch takes, autograd keeps what the
backward pass needs until the gradient is taken, and nothing is written over.

**Device memory on CUDA.** torch has one caching allocator per device for the whole process,
so the per-session settings map only partly:

- `LimitBytes` caps a run at what the allocator holds when it starts plus the limit (cached
  blocks are released first). A run that needs more fails with an `InvalidOperationException`
  naming the limit. The cap is process-wide, so other torch runs on that device wait for a
  capped run to finish.
- Configure torch's allocator itself with `PYTORCH_CUDA_ALLOC_CONF` before the backend starts.
- `ReadArenaStatistics` reads `torch.cuda.memory_stats` for the whole device, not one session:
  `InUseBytes`, `MaxInUseBytes`, `TotalAllocatedBytes`, `AllocationCount`,
  `ArenaExtensionCount` (segments held), `ArenaShrinkageCount` (segments released),
  `RequestedInUseBytes` (the bytes requested, without the allocator's rounding).
  `LimitBytes` is the session's limit or -1; `MaxAllocSizeBytes` and `ReserveCount` are 0.
- `ReadPinnedArenaStatistics` is null: there is no pinned host arena.

**Cancellation** is checked before every node (once per iteration inside a `Loop`) and throws
`OperationCanceledException` carrying the token. One long kernel is not interrupted.

**Precision on CUDA.** `float32` is computed in full `float32` precision unless the context allows
TensorFloat-32 ([Precision](inference.md#precision-gpu-backends)). torch's two switches for it are
the whole process's, so a run sets them from its session as it starts — about 0.3 µs — whatever
they were before, and runs of sessions that set them differently do not overlap: runs in one
precision run on the cards beside one another, and a run in the other waits until they are done —
runs arriving in the first precision meanwhile wait behind it, so neither keeps the other out. A
run on the CPU sets neither.

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
  …), the recurrent networks (`RNN`, `GRU`, `LSTM` — `layout=1` included, which ONNX Runtime's CPU
  kernels refuse), the signal operators (`DFT`, `STFT`, the windows, `MelWeightMatrix`), the
  quantization operators (`QuantizeLinear`, `DequantizeLinear`, `DynamicQuantizeLinear`,
  `QLinearMatMul`, `QLinearConv`), sequences and optionals (`SequenceMap` included), the string
  operators (`TfIdfVectorizer` included), and the ONNX random operators and `Dropout` are
  translated, as are functions that take attributes and Shorokoo's own random draws, which reach
  the backend as integer operators. `ImageDecoder`, which ONNX Runtime has no kernel for, decodes
  with Pillow every format the ONNX specification names (BMP, JPEG, JPEG 2000, TIFF, PNG, WebP and
  the portable any-maps). A model using an operator the backend does not translate, such as one
  from a domain other than ONNX's own, is refused when its session is created, naming it.
- **ONNX random draws differ from ONNX Runtime's.** `RandomNormal`, `RandomUniform`, their `Like`
  forms, `Bernoulli`, `Multinomial` and a training-mode `Dropout` draw from PyTorch's generator
  (a node with a `seed` repeats its values every run). Shorokoo's own keyed draws agree with
  ONNX Runtime exactly.
- **Two text operators follow the ONNX reference implementation where ONNX Runtime departs from
  it.** `StringSplit` without a delimiter splits at runs of any whitespace — tabs, newlines and
  Unicode spaces as well as the space, which is all ONNX Runtime splits at — and splits an empty
  string at a delimiter into one empty piece, where ONNX Runtime counts none. `StringNormalizer`
  changes case by Unicode's full mappings in every locale (`ß` upper-cases to `SS`, `Σ`
  lower-cases to `σ`), where ONNX Runtime leaves what its locale does not map, which in the C
  locale is everything beyond ASCII.
- **`MaxPool` can index the padding of a window whose maximum is −inf.** The backend pads the
  input with −inf itself and lets torch's pooling take each window's first maximum, so where a
  padded position comes before the input element holding a −inf maximum, the `Indices` output
  names the padding: −1, or in general a position outside the window. The values are right.
  Accepted as the backend's behaviour ([#425](https://github.com/Shorokoo/Shorokoo/issues/425)).
- **Linux x64 and Windows x64 only**, the platforms there are lock files for. A backend can be
  constructed anywhere; elsewhere it refuses to start with `PythonEnvironmentFailure.UnsupportedPlatform`.
- **Device memory is torch's, per process** — see [Runs](#runs) for what a context's settings
  can and cannot reach.
- **Node placement records no bytes**: per-node byte counts in a trace are 0.
- **Crash isolation**: the interpreter runs in your process, so a fault inside PyTorch takes
  the process down with it.

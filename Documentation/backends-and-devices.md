# Backends and devices

Choose the backend a model runs on, load one at runtime, and run one model on two devices in
one process.

Related: [inference.md](inference.md) · [tensors-in-a-run.md](tensors-in-a-run.md) · [gpu-backends.md](gpu-backends.md) · [pytorch-backend.md](pytorch-backend.md) · [jax-backend.md](jax-backend.md)

## Backend selection

- Add a backend package: `Shorokoo.LinuxCPU`, `Shorokoo.LinuxGPU`, `Shorokoo.WinCPU`, or
  `Shorokoo.WinGPU`. Each brings the native ONNX Runtime (CPU or CUDA) for its platform. A
  program that names its backends may reference several, or load them at runtime — see
  [Loading a backend at runtime](#loading-a-backend-at-runtime).
- With one backend package referenced, auto-discovery finds it on the first run. Set it
  explicitly when a deployment holds more than one, when you want a startup failure instead
  of a first-run one, or when the backend DLL is not next to `Shorokoo.dll`:

  ```csharp
  using Shorokoo.Core.Backends;
  using Shorokoo.LinuxCPU;                                // the package you referenced

  DefaultBackend.Instance = new LinuxCpuBackend();
  ```

- `DefaultBackend.Instance` is the backend for a `ComputeContext` that names none, and for a
  tensor fed without a context naming another. The first resolved is cached. Reassigning it
  does not unload a native ONNX Runtime already bound, nor reach a `ComputeContext.Default`
  that has already resolved — assign it at startup.
- A `ComputeContext` constructed with a backend runs there — see
  [One model, two devices](#one-model-two-devices).
- PyTorch (`Shorokoo.PyTorch.Cpu`, `Shorokoo.PyTorch.Cuda`) and JAX (`Shorokoo.Jax.Cpu`,
  `Shorokoo.Jax.Cuda`) backends run a model in an embedded Python. They are never
  discovered — name one per context, `new ComputeContext(new TorchCpuBackend())` — so they
  never make discovery ambiguous. See [pytorch-backend.md](pytorch-backend.md) and
  [jax-backend.md](jax-backend.md).

### What a backend provides

`IShorokooBackend` is the interface every backend implements; its comments are the full
contract. In short, a backend provides:

- **Its run memory** — `RunMemoryOf(elementType)` for a tensor and `SequenceRunMemory` for a
  sequence: where its sessions read their inputs and leave their outputs. The defaults are the
  backend's own memory (`MemorySpace`) in its own runtime (`RuntimeIdentity`), and host memory
  for strings and sequences.
- **Sessions** (`CreateSession`) that read every input in the run memory, refuse any value
  outside it with an exception, and leave every output there (`IShorokooSession`). A session
  never moves a value.
- **The moves**, as operations of their own, which Shorokoo calls to place an input before a run
  and to bring a tensor home: into the backend's memory, `CreateTensorInBackendMemory`,
  `CreateUninitializedTensorInBackendMemory` and `TryCopyHostToTensorRange`; out of it,
  `CopyTensorToHost` and `TryCopyTensorRangeToHost`. Their defaults serve a backend whose memory
  the host reads; a backend that computes in memory of its own overrides each of them.
- **Values in its runtime's host memory** — `CreateTensor`, `CreateTensorFromRawBytes`,
  `CreateUninitializedHostTensor` (one a later move fills, of any size), `CreateStringTensor`,
  `CreateSequence` — and `Release`, the one path its memory goes back by.

### The backend types

One backend per package, in a namespace equal to the package id. **The type name spells
`Cpu`/`Gpu`; the package spells `CPU`/`GPU`** — `WinGpuBackend`, *not* `WinGPUBackend`:

| package (= namespace) | backend type | fully qualified |
|---|---|---|
| `Shorokoo.LinuxCPU` | `LinuxCpuBackend` | `Shorokoo.LinuxCPU.LinuxCpuBackend` |
| `Shorokoo.LinuxGPU` | `LinuxGpuBackend` | `Shorokoo.LinuxGPU.LinuxGpuBackend` |
| `Shorokoo.WinCPU` | `WinCpuBackend` | `Shorokoo.WinCPU.WinCpuBackend` |
| `Shorokoo.WinGPU` | `WinGpuBackend` | `Shorokoo.WinGPU.WinGpuBackend` |

All four implement `IShorokooBackend` with a parameterless constructor; the GPU ones use the
CUDA provider on device 0, the CPU ones ORT's default provider.

| package | backend type | fully qualified |
|---|---|---|
| `Shorokoo.PyTorch.Cpu` | `TorchCpuBackend` | `Shorokoo.PyTorch.Cpu.TorchCpuBackend` |
| `Shorokoo.PyTorch.Cuda` | `TorchCudaBackend` | `Shorokoo.PyTorch.Cuda.TorchCudaBackend` |
| `Shorokoo.Jax.Cpu` | `JaxCpuBackend` | `Shorokoo.Jax.Cpu.JaxCpuBackend` |
| `Shorokoo.Jax.Cuda` | `JaxCudaBackend` | `Shorokoo.Jax.Cuda.JaxCudaBackend` |

The sessions of the four ONNX Runtime backends run their operators on ONNX Runtime's thread pools
of the process, made with its environment when the first of these backends is built, rather than
each on an intra-op pool of its own (`SessionsShareThreadPools`, true unless set otherwise:
`new WinCpuBackend { SessionsShareThreadPools = false }`). A pool's threads spin for a while after a
run, waiting for more work, so sessions with pools of their own run one after another contend: two
compiled graphs run in turn, or a compiled graph whose runs consuming their inputs place values
through a session of their own
([A run that writes into what it consumed](tensors-in-a-run.md#a-run-that-writes-into-what-it-consumed)). A session
built with an intra-op thread count of its own (`CreateSession`'s `intraOpThreads`) keeps a pool of
its own of that size; where something else made ONNX Runtime's environment first, every session
keeps its own.

### ONNX Runtime's log messages

Where Shorokoo makes ONNX Runtime's environment, which it does when the first of the four ONNX
Runtime backends is built, the environment logs through `OrtLog` (namespace
`Shorokoo.Core.Backends`). Messages the environment logs and messages a session logs both go
there, and nowhere else:

- `OrtLog.Severity` is the least severe message passed on, `ShorokooLogSeverity.Warning` unless
  set otherwise. ONNX Runtime's environment is set to the same level.
- `OrtLog.Sink` receives each message passed on, as an `OrtLogMessage`: `Severity`, `Category`,
  `LogId`, `CodeLocation` and `Message`, with no terminal colour codes. Its default,
  `OrtLog.WriteToStandardError`, writes one line per message to the standard error stream
  (`[ONNX Runtime Warning] <code location>: <message>`). Set it to `null` to drop every message.

```csharp
using Shorokoo.Core.Backends;

OrtLog.Severity = ShorokooLogSeverity.Error;                          // warnings dropped
OrtLog.Sink = m => logger.Log(m.Severity.ToString(), m.Message);      // your own logger
OrtLog.Sink = null;                                                   // nothing at all
```

Both settings can be changed at any time and apply to the next message. The sink is called on
whichever thread ONNX Runtime logs from, possibly several at once, so it must be thread-safe. An
exception it throws is dropped, because none can be raised through ONNX Runtime.

The sessions Shorokoo builds log only at `Fatal`. A session built through
`IShorokooBackend.CreateSession` at a severity of its own passes on only what is at least as
severe as both that severity and `OrtLog.Severity`.

A backend loaded with `IsolatedBackend.Load` runs its own copy of ONNX Runtime, and that copy
logs through the same `OrtLog`, at the same severity.

A program that makes ONNX Runtime's environment itself (`OrtEnv.CreateInstanceWithOptions`)
before the first backend is built keeps the logging it made the environment with, and its
sessions keep per-session thread pools ([The backend types](#the-backend-types)).

### Auto-discovery

If `DefaultBackend.Instance` is never assigned, its first read resolves and caches a backend:

1. If one of the four backend assemblies targeting the running OS is **already loaded** and
   exposes a backend, it is used, and the folder is never probed — so a deployment carrying
   several backends can run without being refused. (Naming a backend type in any method your
   program runs can load it.) Backends loaded by `IsolatedBackend.Load` are never candidates.
2. Otherwise the folder next to `Shorokoo.dll` is probed for the known
   `Shorokoo.{Platform}.dll` files targeting the running OS. Nothing else is searched.
   Referencing a package suffices: it copies its DLL to your output folder.

A single candidate is taken as-is — even a GPU backend with no CUDA runtime present. **Two or
more are refused** with an `InvalidOperationException` (step 1 says `already loaded in this
process` instead of `deployed in '<folder>'`):

> `Several Shorokoo backends are deployed in '<folder>': Shorokoo.WinCPU (CPU),
> Shorokoo.WinGPU (CUDA). Discovery picks the backend for a program that named none, and
> this deployment gives it no way to choose. Say which you mean: assign
> DefaultBackend.Instance before the first run to make one of them the default.
> To run several at once, give each ComputeContext its own backend -- new ComputeContext(new
> LinuxGpuBackend()) -- and where they need separate native ONNX Runtimes, load
> them with IsolatedBackend.Load.`

Discovery never prefers the GPU: both packages ship their native ONNX Runtime at the same
path, so only one native is actually deployed. Backends for other OSes are ignored, so a
deployment with all four is refused on either OS. A shared library referencing a backend is
the usual accidental cause — keep backends in executables (see
[Or keep it to two processes](#or-keep-it-to-two-processes)); to deploy two on purpose, see
[Deploying two backends](#deploying-two-backends).

If no backend is found, the first run throws `InvalidOperationException`:

> `No Shorokoo backend is set and none was found in '<folder>'. Set one at
> startup -- e.g. DefaultBackend.Instance = new LinuxCpuBackend(); (or the
> backend from whichever Shorokoo.{WinCPU,WinGPU,LinuxCPU,LinuxGPU} package you
> reference) -- or add such a package as a dependency.`

### Which device am I on?

```csharp
using Shorokoo.Core.Backends;

Console.WriteLine(DefaultBackend.Describe());         // Shorokoo.WinGPU (CUDA device 0)
Console.WriteLine(ComputeContext.Default.Backend);    // the same, at the point work is submitted
```

`BackendDescription` carries `Name` (the supplying assembly), `Device` (`ComputeDevice.Cpu`,
`Cuda`, or `Other` for a custom execution provider) and `CudaDeviceId` (null off CUDA). Log it
with training runs.

- `DefaultBackend.Current` is the live backend **or null**; unlike `Instance` and
  `Describe()`, reading it does not trigger resolution.
- `DefaultBackend.RequireDevice(ComputeDevice.Cpu)` throws, naming the live backend, unless it
  is on that device — use it at startup in a program that must not share the GPU:

  ```csharp
  DefaultBackend.RequireDevice(ComputeDevice.Cpu);   // before anything runs
  ```

### Loading a backend at runtime

`BackendPackage.TryLoad` loads a backend from a path, returning a reason instead of throwing
when it does not fit the machine — so one executable can carry several and pick at startup:

```csharp
using Shorokoo.Core.Backends;

if (BackendPackage.TryLoad("plugins/Shorokoo.WinGPU.dll", out var gpu, out var why))
{
    using var cuda = new ComputeContext(gpu!);
    // ...
}
else
{
    Console.WriteLine($"No GPU backend here: {why.Detail}");   // fall back, warn, or stop
}
```

`BackendPackage.Probe` answers without loading the backend, from the file's metadata.
`BackendProbe.Reason` gives the cause (`WrongOperatingSystem`, `MissingNative`,
`MissingCudaRuntime`, `MissingCudaDriver`, …) and `Detail` the offending file or library.
Wrong OS, architecture or missing natives are decided from metadata and paths alone. A backend
requiring the CUDA runtime is checked by binding it, which creates this process's CUDA context
on the card; one needing only the driver (the [PyTorch](pytorch-backend.md) and
[JAX](jax-backend.md) CUDA backends) is checked through the driver API, creating no context,
and refused as `MissingCudaDriver` with no driver, one too old, or no device.

Natives are looked for flat beside the backend assembly and under `runtimes/<rid>/native/`
next to it — a NuGet install uses the `runtimes/` layout on every platform. Each backend
loaded this way gets its own load context, so several run side by side without sharing a
native runtime.

### One model, two devices

A `ComputeContext` constructed with a backend compiles and runs there, and contexts may name
different backends:

```csharp
using Shorokoo.Core.Backends;
using Shorokoo.LinuxCPU;
using Shorokoo.LinuxGPU;

var cpu  = new ComputeContext(new LinuxCpuBackend());
var cuda = new ComputeContext(new LinuxGpuBackend());

var onHost = cpu.Execute(graph, input.Shared());   // the host, reading input and leaving it
var onCard = cuda.Execute(graph, input);           // the same graph, the same input, the card;
                                                   // its outputs are on the card
```

Both run the same model, one C# build; each context compiles its own session. `ComputeContext.Backend` and `CompiledGraph.Backend` name
the backend. Tensors you build (allocating backend `HostBackend.Instance`) are tied to no
runtime, so building and exporting needs none and either context accepts them. A tensor not
readable in place is copied per run when consumed, or once per (tensor, runtime) when fed
`.Shared()`, for as long as it lives ([Feeding a run](tensors-in-a-run.md#feeding-a-run-consumed-shared-or-tried)).
A tensor on the card (`TensorData.IsHostResident` false) — every output of a run there, and the
state a [training step](training.md#keeping-training-state-on-the-device) hands back — crosses
via the host when it is fed to the CPU context or moved with `ToHost()`; reading its values
copies them to the host and leaves it on the card.

#### Deploying two backends

Both packages deliver `runtimes/<rid>/native/libonnxruntime.so` at the same path, so
referencing both normally deploys only one native — which is why
[auto-discovery](#auto-discovery) refuses it.

**One runtime, two providers (usual).** The CUDA build of ONNX Runtime includes the CPU
provider. Deploy the GPU package, and the CPU package without its native:

```xml
<PackageReference Include="Shorokoo.LinuxGPU" Version="..." />
<PackageReference Include="Shorokoo.LinuxCPU" Version="..." ExcludeAssets="native" />
```

**Then name the default explicitly, before anything runs** — discovery sees two candidates and
refuses on the first read of `DefaultBackend.Instance`, possibly inside a framework call.
Constructing a backend is not enough; assign it:

```csharp
DefaultBackend.Instance = new LinuxCpuBackend();   // startup, before anything runs

var cpu  = new ComputeContext(DefaultBackend.Instance);
var cuda = new ComputeContext(new LinuxGpuBackend());
```

**Two runtimes** (two ONNX Runtime builds or versions in one process): give each native its own
folder and load the second with `IsolatedBackend`:

```xml
<!-- The glue, referenced directly: it carries the target that reads the items below, and
     it is the assembly every isolated backend loads. Coming in through a platform package
     is the usual route, and this recipe cuts that route on purpose. -->
<PackageReference Include="Shorokoo.OnnxRuntime" Version="..." />

<PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.30.0"
                  ExcludeAssets="all" GeneratePathProperty="true" />
<PackageReference Include="Microsoft.ML.OnnxRuntime.Gpu.Linux" Version="1.30.0"
                  ExcludeAssets="all" GeneratePathProperty="true" />

<ShorokooBackendNatives BackendId="cpu"
  Include="$(PkgMicrosoft_ML_OnnxRuntime)/runtimes/linux-x64/native/*" />
<ShorokooBackendNatives BackendId="cuda"
  Include="$(PkgMicrosoft_ML_OnnxRuntime_Gpu_Linux)/runtimes/linux-x64/native/*" />
```

Each `ShorokooBackendNatives` item lands in `ort/<BackendId>/`; deploy a package's whole
native folder, since providers must sit beside their core. The direct `Shorokoo.OnnxRuntime`
reference is required: without it the items are silently ignored and the first
`IsolatedBackend.Load` throws `FileNotFoundException` for the missing native. It is not a
backend and never a discovery candidate.

**Deploy the backend assembly beside its native, not beside `Shorokoo.dll`** (two there is
the ambiguity discovery refuses), and point `ProbeDirectory` at it:

```xml
<PackageReference Include="Shorokoo.LinuxGPU" Version="..." ExcludeAssets="all"
                  GeneratePathProperty="true" />
<None Include="$(PkgShorokoo_LinuxGPU)/lib/net10.0/Shorokoo.LinuxGPU.dll"
      Link="ort/cuda/Shorokoo.LinuxGPU.dll" CopyToOutputDirectory="PreserveNewest" />
```

```csharp
var cudaDirectory = Path.Combine(AppContext.BaseDirectory, "ort", "cuda");
var cuda = new ComputeContext(IsolatedBackend.Load(new IsolatedBackendSpec
{
    Name = "cuda:0",
    BackendAssembly = "Shorokoo.LinuxGPU",
    NativeRuntimePath = Path.Combine(cudaDirectory, "libonnxruntime.so"),
    ProbeDirectory = cudaDirectory,
}));
```

The ONNX Runtime wrapper and glue are found beside the backend first, then beside
`Shorokoo.dll`. `Name` is the label `Backend.Name` reports; loading one native twice under two
names is refused, and loading the same spec twice returns the same backend. A loaded backend
lives until the process exits. Only the wrapper, glue and backend assembly are private to it;
the core assembly and your model are shared, so contexts can exchange data.

Sequence outputs work on any provider; ONNX Runtime materializes them in host memory — see
[A sequence's elements live in host memory](limitations.md#a-sequences-elements-live-in-host-memory).

#### Two CUDA backends in one process

CUDA backends of different frameworks run side by side in one process, whichever starts first,
because they run on one copy of cuDNN and cuBLAS — see
[The NVIDIA libraries the CUDA backends run on](gpu-backends.md#the-nvidia-libraries-the-cuda-backends-run-on).

#### Or keep it to two processes

Two executables over a shared, backend-free model library suit separate jobs — a long training
run and an occasional check:

```xml
<!-- Model.csproj — the model, its modules, its losses. No backend. -->
<ItemGroup>
  <PackageReference Include="Shorokoo" Version="..." />
</ItemGroup>
```

```xml
<!-- Train.csproj — the long run. -->
<ItemGroup>
  <ProjectReference Include="../Model/Model.csproj" />
  <PackageReference Include="Shorokoo.WinGPU" Version="..." />
</ItemGroup>
```

```xml
<!-- Check.csproj — the causality check, the shape probe, the debugging execution. -->
<ItemGroup>
  <ProjectReference Include="../Model/Model.csproj" />
  <PackageReference Include="Shorokoo.WinCPU" Version="..." />
</ItemGroup>
```

A backend reference in the shared library — a `PackageReference`, or a `ProjectReference` to
an executable carrying one — flows into both outputs and makes
[auto-discovery](#auto-discovery) refuse. Keep backends in executables, and reference no
executable.

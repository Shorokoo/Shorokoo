# GPU backends: libraries, precision and device memory

What the CUDA backends run on and how they compute: the NVIDIA libraries, the precision of
`float32` work, the device memory a context may hold, and how to read what a run did on the card.

Related: [inference.md](inference.md) · [backends-and-devices.md](backends-and-devices.md) · [training-memory.md](training-memory.md)

## The NVIDIA libraries the CUDA backends run on

Every CUDA backend runs on the same release of cuDNN and cuBLAS (with cuBLASLt), CUDA 13 builds,
pinned per platform beside the lock of the CUDA Python environment: the releases that
environment's PyTorch carries.

| | cuDNN | cuBLAS |
|---|---|---|
| Windows x64 | 9.24.0.43 — 393 MiB download, 542 MiB on disk | 13.0.0.19 — 382 MiB download, 504 MiB on disk |
| Linux x64 | 9.24.0.43 — 528 MiB download, 921 MiB on disk | 13.1.1.3 — 404 MiB download, 568 MiB on disk |

They live in a per-user cache, one folder per release: `%LOCALAPPDATA%\shorokoo\cuda\` on
Windows, `$XDG_CACHE_HOME/shorokoo/cuda/` (or `~/.cache/shorokoo/cuda/`) on Linux, beside the
Python environments — `cudnn-9.24.0.43-cu13\`, `cublas-13.0.0.19-cu13\`. A folder is filled once,
the first time a CUDA backend needs it, under a lock file beside it, and marked complete only once
every file is in place and checked; one left half-filled is filled again. It is filled from:

1. **A copy that matches exactly**: every file of the release present, each with the SHA-256 its
   PyPI wheel records. The provisioned CUDA Python environments beside the cache are looked in
   first — `torch\lib` on Windows, the NVIDIA wheels' own folders on Linux — since their PyTorch
   carries the pinned release byte for byte, whichever backend runs first. Then, on Windows, cuDNN
   is looked for on `PATH`, in `%CUDNN_PATH%` and under `%ProgramFiles%\NVIDIA\CUDNN`, and cuBLAS
   on `PATH`, in `%CUDA_PATH%` and in the CUDA 13 toolkits under
   `%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA`; on Linux, on `LD_LIBRARY_PATH`, in
   `$CUDNN_PATH`, `$CUDA_PATH` and `$CUDA_HOME`, in the CUDA 13 toolkits under `/usr/local` and in
   the system's library folders. The copy's files are hard-linked into the cache where the volume
   allows, so nothing is downloaded or stored twice, and copied otherwise. Any other copy — another
   release, a build for another CUDA major, a file that differs — is ignored, never mixed in.
2. **Otherwise the release's wheel from PyPI**, checked against the SHA-256 the pin records. A
   download through which no data arrives for a minute is given up, as is one not done within the
   hour a fill may take; and a download ended with its process leaves no partial wheel behind.

A filled folder is looked at again each time it is used, without reading its files: one whose
size, or whose last write, has changed since the folder was filled — an installed copy it was
linked to, written over in place, say — has the folder checked whole, file by file against the
pin, and filled again where any file is no longer the pinned one. A folder copied from another
machine, its files' times moved, passes that check and is kept.

With neither available — offline, no copy that matches — the first CUDA session fails with an
`InvalidOperationException` naming the library, where it looked and the size of the download. A
cache folder that cannot be written — a full disk — fails it with an `IOException` instead. To
run offline, fill the cache once while online, copy the folders from a machine that has them, or
install exactly the pinned release. `CudaLibraries.Prepare()` (in
`Shorokoo.Core.Backends`) fills the cache and loads the libraries at a moment of your choosing,
at startup rather than on the first CUDA run; it is also what a backend of your own that loads
CUDA libraries calls first.

Within a process every CUDA backend binds to that one copy. The ONNX Runtime CUDA backend loads
the cache's files by full path before its provider loads anything by name, so it never runs on
the cuDNN or cuBLAS that `PATH` happens to offer. A provisioned
[PyTorch CUDA environment](pytorch-backend.md#cpu-and-cuda)'s own copies — `torch\lib` on Windows,
the `nvidia` wheels' folders on Linux — are made hard links to the cache's files when the
environment is provisioned, or first used, so PyTorch loads the very same files. This is what
lets an ONNX Runtime CUDA backend and a PyTorch one share a process in either order: a process
holding two releases of cuDNN cannot run both, since cuDNN's libraries import one another's
internal entry points by name and bind to whichever copy of a name was loaded first.

So whatever else in the process loads cuDNN or cuBLAS has to do so once the pinned copies are
loaded: a CUDA session of your own built on ONNX Runtime directly, say, calls
`CudaLibraries.Prepare()` before it touches the CUDA provider at all — setting an
`OrtCUDAProviderOptions`' options with `UpdateOptions` already loads the provider, and cuBLAS with
it, as appending it to a session does. A process that already holds another
release when the ONNX Runtime CUDA backend prepares the pinned one — any copy, under the name a
pinned file is loaded by, that is not that file — is refused before anything is loaded: the
backend's first session fails with an `InvalidOperationException` naming the copies held. Loading
the pinned files beside them would not help, as what imports those names binds to the copies
loaded first. A copy held that is the cache's own file — a provisioned PyTorch environment's,
linked to it — is recognised at a glance; any other is read whole and checked against the pin,
which for a PyTorch environment you named, its copies not linked to the cache, reads up to a
gigabyte once per process.

The other NVIDIA libraries both stacks load — the CUDA runtime, cuFFT, nvrtc, nvJitLink — are
each backend's own: the ONNX Runtime backend and Shorokoo's allocator on the card load them by
name, from the machine's CUDA 13 runtime or the copy the process already holds under that name,
and PyTorch loads its environment's. Two copies of these run side by side: they call one another
only through public, versioned entry points, and both work in the card's one context per process,
so memory either one allocates is good to the other.

## Precision (GPU backends)

`float32` is computed in full `float32` precision on every backend and every device: a product, a
convolution or a recurrent layer of `float32` operands is computed in `float32` throughout, so a
model computes the same on a card as on the host, but for the order its sums are added in. Measured
on an RTX 4090, a 1024-square `MatMul`, a 64-channel 3×3 `Conv` and a 256-unit `LSTM` land within
5e-6 of the host's results (the largest difference over the largest value), on every CUDA
backend.

A card from NVIDIA's Ampere generation on can compute those operators faster in **TensorFloat-32**
(TF32): on its tensor cores, with each operand's significand rounded from 24 bits to 11 and the sums
kept in `float32`. A context asks for it with its `Precision`:

```csharp
using Shorokoo.Core.Backends;
using Shorokoo.Runtime;

var fast = new ComputeContext(new LinuxGpuBackend())
{
    Precision = new PrecisionSettings { AllowTensorFloat32 = true },
};
```

The same three operators then land up to 2.5e-3 from the host's, some five hundred times further
than in full precision. What full precision costs against it, measured on an RTX 4090: a 4096-square
`MatMul` takes 1.57 times as long, and a convolutional network's training step 2.6 times as long
on ONNX Runtime and 2.3 times on PyTorch; the training steps of the other model families measured
were within their run-to-run noise.

| Backend | By default | With `AllowTensorFloat32` |
|---|---|---|
| ONNX Runtime CUDA (`WinGpuBackend`, `LinuxGpuBackend`) | the CUDA provider's `use_tf32` is `0` | `use_tf32` is `1`: cuBLAS products and cuDNN convolutions and recurrent layers in TF32 |
| [PyTorch](pytorch-backend.md#runs) CUDA | each run sets `torch.backends.cuda.matmul.allow_tf32` and `torch.backends.cudnn.allow_tf32` off as it starts | each run sets both on: products, cuDNN convolutions and recurrent layers in TF32 |
| [JAX](jax-backend.md#runs) CUDA | every product and convolution is compiled at `Precision.HIGHEST` | compiled at `Precision.HIGH`: products and recurrent layers in TF32; each convolution in TF32 or in full precision, whichever kernel XLA's autotuner finds faster when it compiles the program |
| every CPU backend | full precision | no effect: full precision |

- **It is read when a session is built**, like [`DeviceMemory`](#device-memory-gpu-backends): a
  graph compiled on the context keeps the precision the context carries, and a training rig's steps
  compute in its `runtimeContext`'s. Set it on the context from the start.
- **It allows TF32, and never requires it.** A backend uses it where its kernels choose to: XLA
  puts a small product through a full-precision kernel all the same, and a convolution through
  whichever kernel it timed fastest, which may be either.
- **Only `float32` changes.** Products of `Float16`, `BFloat16` and `Double` operands compute as
  they do either way.
- **PyTorch's switches are the whole process's.** A run on a card sets them from its own session as
  it starts, so a session's precision does not depend on what ran before it, nor on code of your own
  that sets them. Runs of sessions that set them differently do not overlap: while a run that allows
  TF32 is running, runs on the cards that do not wait, and the other way round.
- **`NVIDIA_TF32_OVERRIDE=0`** in a process's environment turns TF32 off in cuBLAS and cuDNN for the
  whole process, a context that allows it included.

## Device memory (GPU backends)

On ONNX Runtime every session allocates through an allocator of Shorokoo's — one per device, the
host or a card, for the whole process, whichever runtime or backend built the session — rather
than through an arena of its own, and the tensors a context places on a card, and every tensor
made in host memory outside a run, come from the same one. A session's blocks are carved from address space reserved for it with no memory behind it:
memory is committed under a block as it is carved, and handed back to the system as the session
lets it go. Each request is rounded up and served by its size:

- **On the host**, a block under 64 KiB is an allocation of its own from the C runtime's heap,
  which may keep its pages for the process's next allocations; a larger one is a whole number of
  4 KiB pages, its memory committed 64 KiB at a time (`VirtualAlloc` on Windows, `mmap` and
  `mprotect` on Linux) and handed back the same way (`VirtualFree`, `madvise`).
- **On a card**, a block of up to a mebibyte is carved from memory every session on the card
  shares, packed into the card's 2 MiB pages as the CUDA runtime packs small allocations; a larger
  one is a whole number of 2 MiB pages of its own, mapped into one contiguous range with CUDA's
  virtual memory management (`cuMemCreate`, `cuMemMap`). Where the driver does not offer it, every
  block is a `cudaMalloc` of its own, a multiple of 512 bytes up to a mebibyte and of an eighth of
  the power of two below it above that.

So a block holds its own memory: on a card a block over a mebibyte has its pages to itself, and
a smaller one shares a page with other small blocks; on the host a block shares at most the
64 KiB at each of its ends with its neighbours.

- **What a session lets go of is kept for its next runs**, so a loop's runs find their blocks
  waiting. A request no kept block serves is carved from the memory the session has committed,
  whichever block it last belonged to, so a session fed one shape after another reuses its warm
  memory as an arena does rather than taking each new size from the system. A session never holds
  more than the most one of its runs has used — what it had in use as the run began, and every
  block the run was handed, each counted once: where a request would take it past that mark, the
  smallest larger block it keeps serves it, or the blocks it kept longest give way to it, and what
  is still over the mark goes back to the system — on the host at once, on a card as the run ends.
  So a loop whose runs repeat finds every block it needs, and a session fed varied shapes holds
  what its busiest run used, not a block of every size it has seen. What the tensors placed on a
  card let go of is kept the same way, for the next tensor placed there, counting what is placed
  between two runs on the card as one run; so is what a tensor made in host memory outside a run
  lets go of under a mebibyte, while from a mebibyte its memory goes back to the system as it goes,
  as the C runtime's heap hands back a block that large: nothing says another of its size follows.
  The small blocks every session on the card shares keep no more memory committed than they had in
  use at their busiest.
- **What is kept goes back to the system**:
  - at the end of a run that asks for it (`RunSettings.ShrinkArenaAfterRun`, always on under a
    budget): what its session keeps, and what the placed tensors left on that device;
  - as a session is disposed, or collected undisposed: everything it keeps, and each block it
    still has out — an output a caller keeps — as that goes;
  - before a session's budget, or the device itself, would refuse a request for want of room;
  - when the program calls `DeviceMemory.ReleaseCached()`: everything kept on every device, for
    every session and every placed tensor, with no run to make. It returns the bytes it handed back.

  Nothing else returns it. A long-lived program doing varied work holds, besides what is in use,
  up to the busiest run of each session it keeps alive and of the tensors it placed on each device,
  and nothing of a session it disposed.
- **Sessions run without ONNX Runtime's memory pattern.** With it, a session's runs from the second
  on take their planned tensors as one block laid out by the first run; that block came out larger
  than what the runs have in use at once with a block per tensor on every graph measured — 33–50%
  on a scenario of slices, fills and concatenations, 3–24% on training steps, 7–37% on inference —
  for fewer allocations, which cost little: training steps measured without the pattern ran within
  2% of their time with it on the host and on a card, the smallest, a 4×2 linear, within the 8% its
  own runs vary by. So `AllocationCount` counts a run's tensors one by one.
- **A request that cannot be served fails the call that made it**, as ONNX Runtime's own
  allocators fail one: a block the budget leaves no room for, or one the device does not have,
  fails the run — or the placing of the tensor — with an allocation failure, and leaves the
  device as usable as it was for the next one. The text names the memory:

  ```
  [ErrorCode:RuntimeException] Non-zero status code returned while running Expand node. Name:'N3'
  Status Message: Failed to allocate 4503599627370496 bytes on CUDA device 0: the card has no such
  block free (CUDA refused it), with everything Shorokoo's cuda_allocator kept for reuse
  handed back.
  ```

Settings apply to the sessions the context compiles; `CompiledGraph.DeviceMemory` reports what a
graph got:

```csharp
using Shorokoo.Core.Backends;
using Shorokoo.Runtime;

var ctx = new ComputeContext
{
    DeviceMemory = new DeviceMemorySettings
    {
        LimitBytes = 16L * 1024 * 1024 * 1024,   // a 16 GiB budget on what ctx holds on the card
    },
};

var compiled = ctx.Compile(graph);
compiled.Execute([batch1]);

Console.WriteLine(compiled.DeviceMemory.LimitBytes);   // what the budget left the session's last run
```

| setting | on | what it does | default | read |
|---|---|---|---|---|
| `LimitBytes` | `DeviceMemorySettings` | the most the context holds on the card; a run's session may allocate the budget less what the context holds there | `null` — no budget | on every transfer onto the context and every run of it — see [A context's device-memory budget](#a-contexts-device-memory-budget) |
| `ShrinkArenaAfterRun` | `RunSettings` | as the run ends, its session's allocator hands back to the device what it keeps for the session, and what it keeps of tensors that are gone | `false` — and forced on under a budget | on every run |

- `ShrinkArenaAfterRun` makes the next run commit its memory again — on a card mapping each 2 MiB
  page anew, which costs tens of microseconds a block on Windows — so outside a budget use it only
  when the card is shared. On the CPU backend it hands the session's host memory back the same
  way, and the next run's pages are each zeroed by the system as they are first touched.
- `LimitBytes` is hard: a run that needs more fails with an allocation failure naming the limit,
  so a figure set too low fails work that would fit.
- Context settings are `init`-only. `CompiledGraph.Execute` / `Run` can override
  `ShrinkArenaAfterRun` per call on an unbudgeted context; the context's one-shot entry points
  and a rig's `TrainStep` use the context's.
- **Scope is the context, never the process.** Two contexts may have different budgets and never
  affect each other's sessions; to use different settings, compile on another context — e.g. one
  for a training loop, one for variable-shape inference. Every session on a card, and every tensor
  placed there, allocates through that card's allocator, held for the process's life; each
  session's figures and limit are its own, and the budget counts placed tensors by attachment.

The static `DeviceMemory` class reports what the card is doing:

```csharp
using var run = rig.BeginResidentRun(checkpoint);
for (int step = 0; step < steps; step++)
{
    run.Step(input.Shared(), target.Shared());   // read, so the next step can use them
    DeviceMemory.Sample();
}
Console.WriteLine($"this process at its peak {DeviceMemory.PeakProcessBytes / (1024 * 1024)} MiB");
Console.WriteLine($"the card at its peak {DeviceMemory.PeakUsedBytes / (1024 * 1024)} MiB");
```

`Read()` returns a `DeviceMemoryReading` (`UsedBytes`, `FreeBytes`, `TotalBytes`,
`ProcessBytes`); `Sample()` also folds it into `PeakUsedBytes` and `PeakProcessBytes`;
`ResetPeak()` restarts both.

- `UsedBytes`, `FreeBytes` and `TotalBytes` are the **device's**, including other processes.
- `ProcessBytes` is **this process's** share: everything it holds on the card — weights and
  optimizer state, every session's memory, the CUDA context and its libraries' workspaces — and
  nothing another process holds. `PeakProcessBytes` is therefore the one figure for "the most this
  run held on the card". It is the figure Task Manager shows per process on Windows, and
  `nvidia-smi` lists per process where it can.
- `ProcessBytes` is read from DXGI for a Windows card driven by WDDM (a GeForce, or any card
  driving a display), where `nvidia-smi` prints `[N/A]` for it, and from NVML everywhere else. It
  is `null` where neither answers for the card: where DXGI does not, and NVML is not installed,
  gives no per-process figure, or does not see this process's id, as in a container with its own
  process-id namespace. The two figures of one reading are taken one after the other, so while
  this process allocates on another thread its share can momentarily read above the card's.
- The peaks are the largest of your own `Sample()` calls; nothing samples on its own. A
  sample cost about a microsecond on an RTX 4090 under WDDM, and NVML's process list about 120
  microseconds, so one per step is cheap — and, unlike an external poller
  such as `nvidia-smi`, cannot miss the step.
- The peaks are 0 until a sample folds into them, and `PeakProcessBytes` stays 0 while no sampled
  reading carried a process figure — in a container with its own process-id namespace, say, where
  neither DXGI nor NVML gives one. A 0 therefore means "no figure" as well as "nothing sampled".
- With no CUDA runtime, `Read()` and `Sample()` return `null` and the peaks stay 0, so the calls
  can stay in CPU code.
- The first reading initializes this process's CUDA context (a few hundred MiB) if none exists;
  take it after the backend is up. Readings use the thread's current CUDA device (device 0
  for the shipped GPU backends).

## A context's device-memory budget

`DeviceMemorySettings.LimitBytes`, on a context whose memory is a card's, budgets
**everything that context holds there**: its attached tensors in card memory and, during a run,
what that run's session allocates. It is per context, not per session or process:

```csharp
using var ctx = new ComputeContext(gpu)
{
    DeviceMemory = new DeviceMemorySettings { LimitBytes = 2L * 1024 * 1024 * 1024 },
};

var onCard = big.CopyTo(ctx);                     // on the card, and on ctx's books
var use = ctx.ReadDeviceMemoryUse();
Console.WriteLine($"{use.AttachedBytes} of {use.LimitBytes} bytes attached, {use.AvailableBytes} left");
```

**What it counts.** Tensors placed by `To` or `CopyTo`, read or
copied there by the context's runs, or left there as its runs' outputs — every output of a run
on the card, until it is deleted, collected or detached.
`ReadDeviceMemoryUse()` reports `AttachedBytes`, `AttachedTensors` and `LimitBytes` (`null`
with no budget, or for a host context). A tensor on two contexts counts on both; collected or
`Detach`ed tensors drop out, and so do dead ones once their memory is released. A tensor deleted
with `DeleteAsync` while a run, or the session of a graph loaded with `LoadCompiled`, still reads
it keeps counting until that reader stands down, since its memory is still on the card until then.

**A transfer it cannot take is refused before allocating** — `To`, `CopyTo`, a run's card copy
of a tensor it cannot read in place, and a `To` that
merely attaches a tensor already on the card. Structs and sequences are checked whole (a
run-produced sequence per element as copied), and a failure undoes what was placed. The refusal
is an `InvalidOperationException`:

```
CopyTo(context) of Tensor (8388608,):Float32 asks this compute context for 33554432 bytes of CUDA
device 0 memory, which its device-memory budget cannot give: the budget
(DeviceMemorySettings.LimitBytes) is 67108864 bytes, and 50331648 bytes of it are attached to the
context there, in 1 tensor(s), leaving 16777216. Delete what the context no longer needs, or give
the context a larger budget.
```

**A run's session gets what the context leaves it.** What a run's session may allocate on the
card is the budget less the *discount*: what the context holds on the card apart from the session
for the length of the run — attached tensors, and those the run reads or copies there (for a
tensor another runtime holds on the same card, both it and its copy). A tensor already on the card
is read in place and never counts against the session. A host tensor fed to a card run is copied
onto the card before the run, outside the session, and counted in the discount:

- `.Shared()`: the copy is kept for later reads;
- consumed: the copy is the run's, and goes when the run no longer reads it. A tensor fed to
  several inputs is copied once, and counted once.

What the session allocates is its weights, everything the run computes, and the run's outputs
until it returns. From then on every output on the card is counted with the attached tensors, in
the discount of every later run until it goes, so delete each output once you are done with it.
Outputs [written into consumed memory](tensors-in-a-run.md#a-run-that-writes-into-what-it-consumed) count that
memory once, for what of it is still held, while any of them is attached.

A run whose discount leaves its session nothing is refused before taking anything; one whose
session needs more than it was left fails with an allocation failure naming the limit:

```
[ErrorCode:RuntimeException] Non-zero status code returned while running Expand node. Name:'N3'
Status Message: Failed to allocate 167772160 bytes on CUDA device 0: Shorokoo's cuda_allocator may
hold 163577848 bytes there for this session, what the device-memory budget
(DeviceMemorySettings.LimitBytes) leaves its run, and it holds 512.
```

**The limit follows the discount, run by run.** On ONNX Runtime a session's allocator holds it to
its limit as each block is asked for, so the session takes the limit each run is left without
being built again, and what the context frees is the next run's room.
`CompiledGraph.DeviceMemory.LimitBytes` is the limit its last run was given. A backend whose
sessions take a limit only when they are built (PyTorch) builds one with the budget less the
discount, rounded up to the next sixty-fourth of the budget, and builds it again, before taking
anything, only when the discount grows past that: at most sixty-four times over a compiled
graph's life, plus once each time the remainder halves within its last sixty-fourth. Its limit
only comes down, until the graph is compiled again, and its statistics restart with each build.

**One at a time.** Under a budget the context's runs, transfers onto it and compiles on it are
serialized, and every run hands back what its session's allocator keeps for it as it ends,
whatever `ShrinkArenaAfterRun` says. None of this applies without `LimitBytes`.

What the budget does *not* count is in
[Known limitations](limitations.md#a-device-memory-budget-counts-tensors-not-what-the-allocator-keeps).

## What one session's allocator did

To read **one session's own allocator** (CPU or GPU), ask the compiled graph:

```csharp
using Shorokoo.Core.Backends;
using Shorokoo.Runtime;

var compiled = ctx.Compile(graph);
compiled.Execute(inputs);

if (compiled.ReadArenaStatistics() is { } allocated)
    Console.WriteLine($"{allocated.InUseBytes} in use, {allocated.MaxInUseBytes} at its highest, "
                    + $"{allocated.TotalAllocatedBytes} taken from the device");
```

`ArenaStatistics` has ten figures: `InUseBytes`, `RequestedInUseBytes`, `MaxInUseBytes`,
`MaxAllocSizeBytes`, `TotalAllocatedBytes`, `LimitBytes` (the most the session may allocate; `-1`
with no budget), `AllocationCount`, `ArenaExtensionCount`, `ArenaShrinkageCount` and
`ReserveCount`. It is `null` on a backend that reports none.

- On ONNX Runtime they are the session's own figures in the allocator it allocates through: on the
  card for a CUDA session, in host memory otherwise. `ArenaExtensionCount` is the blocks the
  session holds from the device now, in use or kept, and falls as they go back;
  `ArenaShrinkageCount` is the blocks it has handed back; `ReserveCount` is zero.
- `RequestedInUseBytes` is the part of `InUseBytes` the callers asked for; the rest is what the
  allocator added to round each request up to its size class. A backend whose allocator does not
  report what was requested (JAX) reports `InUseBytes` here, rounding included.
- `TotalAllocatedBytes` is what the session holds from the device, in use or kept for its next
  runs. It is not a bound on card usage; use `DeviceMemory.Read()` for that.
- `MaxInUseBytes` is a lifetime high-water mark; it cannot be reset and handing memory back does
  not lower it. For per-run figures use [`RunStats`](#per-run-statistics-on-the-context).
- A session's weights are among what it allocates, so it is non-zero before the first run, and the
  first run's peak includes them.
- A run's outputs are blocks of their session's: `InUseBytes` after a run is the weights and the
  outputs a caller still holds, and an output's block leaves it when the output goes.

## What crossed the bus

A device session that hands part of a graph to the host stages the crossings through a **pinned
host arena**:

```csharp
if (compiled.ReadPinnedArenaStatistics() is { } pinned)
    Console.WriteLine($"{pinned.MaxInUseBytes} of pinned host memory at its highest");
```

Same ten figures, not included in `ReadArenaStatistics()`; `null` on CPU backends. Zero for a
graph that stays on the card.

## Per-run statistics on the context

Collection is **off by default** and free until enabled:

```csharp
using var ctx = new ComputeContext
{
    Diagnostics = new DiagnosticSettings { CollectRunStatistics = true },
};

var rig = TrainingRig.FromScratch(model, loss, optimizer, sample, hypers, runtimeContext: ctx);
var checkpoint = rig.CreateInitialCheckpoint();
for (int step = 0; step < steps; step++)
    checkpoint = rig.TrainStep(checkpoint, inputs.Shared(), targets.Shared());

var stats = ctx.RunStats;
Console.WriteLine($"{stats.RunCount} runs, peak {stats.PeakBytes / (1024 * 1024)} MiB, "
                + $"{stats.ArenaExtensionCount} blocks taken from the device");

foreach (var run in stats.RecentRuns.TakeLast(5))
    Console.WriteLine($"run {run.RunNumber}: {run.PeakBytes} ({run.PeakKind}), "
                    + $"its session stood at {run.PriorPeakBytes} before it");
```

`RunStats` covers every run of the context across **all** its sessions.

- **Aggregates are exact; per-run detail is bounded.** `PeakBytes`, `RunCount`,
  `AllocationCount`, `ArenaExtensionCount`, `ArenaShrinkageCount`, `LargestAllocationBytes` and
  `ArenaBytes` cover the whole history. `RecentRuns` keeps the last
  `DiagnosticSettings.RecentRunCapacity` records (1000 by default; zero keeps none).
- **`PeakKind`**: a run that raised its session's high-water mark is
  `MemoryFigureKind.Measured`; one that stayed below an earlier mark is
  `MemoryFigureKind.UpperBound` (it used at most that).
- **`PriorPeakBytes`** is the mark the run found. On the first run it is the weights, so
  `PeakBytes - PriorPeakBytes` is the run's own use; later it is how far the run exceeded the
  record (zero for `UpperBound` runs).
- `PeakBytes` is the largest mark of any one session, not a sum across sessions.
- `ArenaBytes` (taken from the device) is usually above `PeakBytes`, but shrinking runs —
  every run under a budget — can leave it below; compare ordering, not figures.
  `ArenaExtensionCount` counts blocks held, so it undercounts after shrinkage.

## Did part of my GPU graph run on the host?

A CUDA session leaves operators the provider cannot run to the host: among them the bitwise
operators of every random draw (`Dropout`, a random initializer), which the CUDA provider has no
kernels for, so a draw is computed on the host and copied to the card. ONNX Runtime's own notice
that it added such copies arrives as a `Verbose` message ([Log messages](backends-and-devices.md#log-messages)), once
per session built; ask for placement instead. Two signals:

```csharp
switch (compiled.OutputPlacement)
{
    case SessionOutputPlacement.Device: break;               // all of it stayed on the card
    case SessionOutputPlacement.Mixed:                       // some of it did not
    case SessionOutputPlacement.Host: break;                 // none of it did
    case SessionOutputPlacement.Unknown: break;              // this backend does not say
}
```

`OutputPlacement` is free, and says where the outputs are **computed**; every tensor output but
a string one comes back on the card whatever it says. A CPU session reports `Host`. On a card, a graph run wholly by CUDA
reports `Device`; one with an unsupported operator (e.g. `Det`) reports `Mixed` or `Host`
depending on where its outputs are computed. An output the host consumed after the card computed
it counts as host memory, in the pinned arena.

For **which** nodes fell back, trace them. This turns on ONNX Runtime's profiler for every run
of the session, so keep it to diagnosis:

```csharp
using var traced = new ComputeContext
{
    Diagnostics = new DiagnosticSettings { TraceNodePlacement = true },
};
var compiled = traced.Compile(graph);
compiled.Execute(inputs);

if (compiled.ReadNodePlacement() is { } placement)
{
    foreach (var share in placement.Providers)
        Console.WriteLine($"{share.Provider}: {share.NodeCount} nodes, {share.OutputBytes} bytes out");

    foreach (var node in placement.NodesOn("CPUExecutionProvider"))
        Console.WriteLine($"  {node.Name} ({node.OpType}) fell back, {node.ActivationBytes} bytes in");
}
```

`Providers` lists each provider that ran anything, busiest first; more than one on a GPU session
*is* fallback. **Reading the trace stops the recording**: later runs are not included, and a
second read returns the same trace. `Nodes` is in execution order; inserted `MemcpyToHost` /
`MemcpyFromHost` nodes get indices above all original nodes, so treat
`NodeExecution.NodeIndex` as a name, not an order.

| what | where | cost | null / none when |
|---|---|---|---|
| `DeviceMemory.Read()` | static, the whole card, and this process's share of it | about a microsecond through DXGI; about 120 microseconds through NVML | no CUDA runtime; `ProcessBytes` alone is null where the driver attributes no memory to this process |
| `CompiledGraph.ReadArenaStatistics()` | one session's allocator | a call into the backend | the backend reports none |
| `CompiledGraph.ReadPinnedArenaStatistics()` | one session's pinned host arena | a call into the backend | the backend stages nothing (every CPU one) |
| `ComputeContext.ReadDeviceMemoryUse()` | what the context holds in its memory, against its budget | a walk over its list | never null; `LimitBytes` is null with no budget in force |
| `ComputeContext.RunStats` | every run of the context | two allocator reads per run, once switched on | `CollectRunStatistics` is off |
| `CompiledGraph.OutputPlacement` | one session | nothing | the backend does not report it (`Unknown`) |
| `CompiledGraph.ReadNodePlacement()` | one session, per node | a profiler on every run of that session | `TraceNodePlacement` is off |

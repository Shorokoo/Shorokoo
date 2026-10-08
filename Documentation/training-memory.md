# Training memory

What a training step reports when an allocation fails, what a training process's memory holds,
and how large a process memory limit must be.

Related: [training.md](training.md) · [gpu-backends.md](gpu-backends.md#device-memory-gpu-backends) · [first-training-run.md](first-training-run.md)

## When a training step runs out of memory

An allocation failure in a step is rethrown as a `ComputeContextException` with code `CR009`
reporting:

- **Which pool**: `HOST memory` (a failed host allocation; a bare `bad allocation` is host even on
  a GPU) or `DEVICE memory` (the accelerator's own). On ONNX Runtime the allocator every session
  allocates through names the memory it failed to allocate, the card's or the host's. Where the
  backend names no allocator on a session with device memory, the report says so.
- **What the step held**: trainable parameters, model state, optimizer state and the batch, each
  with tensor count and size, plus the five largest tensors.
- **The card's figures** (where a CUDA runtime is installed; `DeviceMemory.Read()`): used, free and
  total across processes, how much of it is this process's, and the most this session may allocate
  where a budget limits it.
- **This process's memory**: working set, commit charge and managed heap against the limit in force
  (cgroup/container, Job Object, or machine RAM).

A process memory limit can fail a device allocation too, with the same message as a full card —
see [Sizing a process memory limit](#sizing-a-process-memory-limit). The report distinguishes three
cases: the device is full; the device has room but the process is at its limit (raise the limit);
or both have room and the session was held to less than it asked for — by a budget, or by an
allocator keeping blocks it is not using.

```
[CR009] Compute context operation failed in TrainingRig.TrainStep: allocating memory for the training
step at step 1 failed. The failing allocation was for DEVICE memory — the accelerator's own (backend
'Shorokoo.WinGPU'). Training state held for this operation: 296 tensor(s), 1.83 GiB in total (...).
Device: 12.59 GiB of 23.99 GiB in use across all processes, 11.9 GiB of it this process's, 11.4 GiB
free. Host process: working set
9.61 GiB, commit 27.4 GiB, managed heap 3.02 GiB; against a configured memory limit of 28 GiB (98%
used). The device has room, yet this process is close to its own memory limit — and on Windows/WDDM a
device allocation is backed by system commit, so a limit meant to bound HOST memory bounds DEVICE
memory too ... This is the limit, not the model: re-run with it raised or removed. Underlying failure:
[ErrorCode:RuntimeException] ... Failed to allocate 2359296 bytes on CUDA device 0: the card has no
such block free (CUDA refused it) ...
```

The backend's text is kept verbatim at the end and the original exception as `InnerException`.
Other failures keep their type and message. If the host runs out while the **garbage collector**
needs memory, the runtime fails fast (`Fatal error. 0xE0004743`) with no exception to wrap; use the
last CR009 report as the lead.

## Reading a training process's memory

A process's private bytes (its commit charge) are more than what it holds:

- **The .NET heap keeps what it collected.** Building a rig, loading a checkpoint and saving one
  each allocate the training state or a large share of it for a moment, and after a collection the
  runtime keeps that memory committed for reuse rather than returning it. A forced
  `GC.Collect()` leaves it committed; `GC.Collect(2, GCCollectionMode.Aggressive, blocking: true,
  compacting: true)` returns it. So compare `GC.GetTotalMemory(false)` (what is live) with
  `GC.GetGCMemoryInfo().TotalCommittedBytes` (what the heap holds) before reading the rest of the
  commit as native. Under a container's or a Job Object's memory limit the runtime caps its heap
  below the limit (at 75% of it by default) and collects harder as it nears the cap.
- **On Windows, a card's memory is commit too** — see
  [Sizing a process memory limit](#sizing-a-process-memory-limit).

What a rig itself holds is the model's state once over at most. A rig built from scratch keeps its
initial values — parameters, model state and optimizer state — for `CreateInitialCheckpoint`, where
the runs that computed them left them: on the card when its merge context runs on one, attached to
that context. A rig from `TrainingRig.Load` keeps none, where every parameter declares a concrete
shape (the case for a model built by Shorokoo): it computes them the first time something asks,
since the checkpoint it loads replaces them. A resident run holds the state on its device; each
checkpoint it hands out is that state where it is, which the run then only reads, so its next step
writes beside it: a second copy of the state on the device for as long as you hold the checkpoint.
Once you drop it, it is freed by the collection the rig triggers after a later step once the state
dropped since the last one exceeds its budget for superseded state: 32 MiB, doubling up to 8 GiB
while the rig finds you keeping your checkpoints. Until then it stays beside the run's state, so a
run that saves and drops a checkpoint every N steps holds, besides that state, at most the budget of
dropped state: two copies of a state larger than the budget, briefly, and up to the budget's worth
of dropped copies of a smaller one.

## Sizing a process memory limit

On Windows, a card driven by WDDM (a GeForce, or any card driving a display) backs every allocation
on it with system commit. A GPU process's commit charge therefore includes everything it holds on
the card, the blocks its allocator keeps for reuse among it, and a process memory limit — a Job
Object's, or a container's — bounds the card's memory along with the host's: a device allocation
that would take the process past the limit fails as though the card were full.

**Such a limit must cover the process's peak commit charge**, which on a WDDM-driven card already
includes the device memory the process holds (`ProcessBytes`). The device share is usually the larger
part of it: training runs of 49M, 88M and 164M parameters on one card peaked at 13.2, 15.4 and
20.3 GiB of commit, of which 10.4, 12.6 and 16.9 GiB was held on the card. A limit sized from a
host-only figure — a CPU run's commit, say — must add the device peak, or it fails such a run early. Read this process's
device share as `DeviceMemory.Read()?.ProcessBytes` ([Device memory](gpu-backends.md#device-memory-gpu-backends));
a `CR009` report gives both figures at the moment an allocation fails.

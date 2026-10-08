# Tensors in a run: feeding, placement and lifetime

What a run does with the tensors it is fed and the ones it returns: which it consumes and which
it only reads, where its outputs are, when it writes into the memory of its inputs, when a
tensor's memory is freed, and how a tensor moves from one context to another.

Related: [inference.md](inference.md) · [backends-and-devices.md](backends-and-devices.md) · [training.md](training.md)

## Feeding a run: consumed, shared or tried

A tensor fed **as it is** is taken by the run when it starts — dead from then — and its memory
is released before the call returns, or reused for the run's values
([below](#a-run-that-writes-into-what-it-consumed)):

```csharp
var result = compiled.Execute(batch)[0].ToTensorData();
// batch is consumed: reading it throws, naming the run that took it
```

Pass `.Shared()` to keep it: the run only reads it, under a reader lock:

```csharp
var weights = TensorData([4L, 4L], w);
var first  = compiled.Execute(x1, weights.Shared());
var second = compiled.Execute(x2, weights.Shared());   // weights is still there
```

| Fed as | The run | Afterwards |
|---|---|---|
| `t` | takes it when it starts; refuses it while another run is reading it | dead |
| `t.Shared()` | reads it, holding a reader lock | alive and unchanged |
| `t.TryConsume()` | takes it if nothing else is reading it, reads it otherwise | dead, or alive if it was read |

A tensor fed as it is while another run reads it makes the call throw
`InvalidOperationException` naming that run.

- **Consumption is irrevocable.** A run that fails or is stopped after starting has consumed
  its as-is feeds. A run refused before starting (cancelled token, dead feed, feed being
  read) takes nothing — except in a race, where a feed that dies or starts being read
  mid-setup refuses the run part-way and what it took stays consumed.
- **One tensor fed twice in one call** is taken once: read if any occurrence is `.Shared()`,
  else consumed if any is bare, else tried.
- **Composites apply the mode to everything they hold.** `TensorDataStruct`,
  `TensorDataSequence` and `OptionalTensorData` have `.Shared()` and `.TryConsume()`, as does
  a training checkpoint ([What a training step consumes](training.md#what-a-training-step-consumes)).
  A sequence that owns its elements (made by a sequence's `To`, `CopyTo` or `ToHost`) is
  handled per element: refused if another run reads any element; fed `.Shared()`, none can be
  deleted during the run; an element fed separately counts as another occurrence, so
  `Execute(e.Shared(), s)` reads `e` and consumes the rest of `s`. A struct field can carry
  its own mode — `def.FromOrderedData(tokens, mask.Shared())` — and a `.Shared()` field is
  always read; a struct fed `.Shared()` has every field read.
- **The error names the call to change.** Reading a consumed tensor throws
  `ObjectDisposedException` naming the run (graph and context) and input that took it.
- `.Shared()` / `.TryConsume()` on a tensor, struct, sequence or optional return a
  `SharedInput` (an `IData` with a `Mode`). On a training checkpoint they return a new
  checkpoint over the same tensors with its `FeedMode` set, kept by its derivations
  (`WithStep`, …) and `rig.AdoptCheckpoint`. `Run` reads each `NamedModelParam`'s `FeedMode`
  (`null` = as it is); on a parameter they return a copy with `FeedMode` set, so
  `graph.Run(p.Shared())` leaves `p` unchanged.

**Memory the run cannot read in place** — every tensor built from a C# array, and one from
another device or runtime — is fed through a copy in the run's memory, which Shorokoo makes
before the run with the backend's own move there
([Where a run's inputs and outputs are](#where-a-runs-inputs-and-outputs-are)):

- **Consumed**: the tensor is dead and its memory released at the feed; the run consumes the
  copy. On a card, the copy is made on the card, outside what the session allocates, and a
  budget counts it ([A context's device-memory budget](gpu-backends.md#a-contexts-device-memory-budget)).
- **Read**: the copy is made on first read and kept — a `TensorData` held by the source,
  attached to the reading context (visible in `context.Tensors`), and reused by later shared
  reads for as long as the source lives. Deleting or consuming the source retires it, and so
  does a training step letting go of the copies of the batch it read.

## Where a run's inputs and outputs are

Every backend has a **run memory**: the memory its sessions read their inputs in and leave
their outputs in. On a CUDA backend it is that card's memory for every tensor but a string one;
a string tensor and a sequence are in the host memory of the backend's runtime. On a CPU backend
it is host memory. That memory belongs to the backend: two backends on one card share it only
where they share a runtime.

- **Inputs.** Shorokoo puts every input in the run memory before the run: a tensor there
  already is handed over as it is, and any other — a C# array, a tensor of another device or
  runtime, a card's output fed to a run on the host — through a copy made with the backend's own
  moves. A session is handed nothing else; one handed a value outside its run memory refuses it
  with an `InvalidOperationException` rather than copying it.
- **Outputs.** Every output comes back in the run memory, as a new `TensorData` attached to the
  context that ran it — on a GPU backend, on the card, `IsHostResident` false. Nothing moves it
  afterwards. Reading its values copies them to the host and leaves it on the card; `ToHost()`
  makes a copy of it in host memory, `To(context)` puts it on another context, and a run on
  another context that is fed it moves it there as one of its inputs.
- **An output holds nothing of the session's.** On ONNX Runtime a session allocates through an
  allocator of Shorokoo's, and an output is the block its run wrote it into — or a range of the
  memory of an input the run consumed, where the run wrote it there
  ([below](#a-run-that-writes-into-what-it-consumed)). Keeping it keeps that memory and nothing else
  of the session's alive — the session may run on, sit idle or be disposed — and nothing copies it
  after the run. [Device memory](gpu-backends.md#device-memory-gpu-backends) says how blocks are laid out and what
  the allocator keeps between runs.

```csharp
var compiled = cuda.Compile(graph);
var y = compiled.Execute(x)[0].ToTensorData();     // on the card
float[] values = y.CopyMemory<float>();            // one copy across the bus; y stays on the card
var onCpu = cpu.Execute(graph, y.Shared());        // copied to the host as cpu's input
var next = compiled.Execute(y);                    // read where it is, and consumed: nothing crosses
```

On a CUDA backend an operator the provider runs on the host reads its inputs from the card and
writes its outputs back there ([Shorokoo/Shorokoo#493](https://github.com/Shorokoo/Shorokoo/issues/493)).
On a CPU backend nothing of this is visible: its run memory is the host's.

## A run that writes into what it consumed

ONNX Runtime holds every input until the run ends, so a consumed input's memory cannot be
freed mid-run. Instead a run can write **into** it, so what it writes there needs no memory of its
own. It does so two ways, both only into an input the run consumed — `.Shared()` memory is left as
it was — and fed as no other input. Both inputs and outputs are in the backend's run memory, so
where they are never stands in the way: a host tensor consumed by a card run is copied onto the
card first, and written into as that copy.

**Output aliasing.** An output a lowering marks as safe is written over the whole of an input of
its element type and the shape the session settled at build time. The only such lowering is the
training rig's step, which pairs each updated state field with the one it replaces
([A step writes its state over the state it consumed](training.md#a-step-writes-its-state-over-the-state-it-consumed));
a graph you compile yourself marks no output.

**Placement.** For a compiled graph, a run's values of a mebibyte or more — outputs and
intermediates alike — are written into ranges of the consumed inputs' memory where the graph the
backend runs proves a range free for the value:

- everything that reads what the range held runs before the value is written: by the graph's own
  edges, or on ONNX Runtime in the order the session runs its nodes — one at a time, in the order
  of the graph it writes out, which on a card holds for the nodes the CUDA provider runs;
- nothing reads it after the run: an output, or a view of one, is never written over;
- an operator that reads what it overwrites reads each element where it writes it — an
  element-wise operator in place, a slice at its own offset, a part of a concatenation already in
  its slot, a fused normalization's sum over its own input;
- on a card, the value is computed on the card: a node the CUDA provider has no kernel for runs on
  the host, and what it computes is not placed.

Where the values go is planned the first time a run *signature* — which inputs are consumed, the
shape of every input, the outputs asked for — runs, and kept for it.

| | ONNX Runtime | PyTorch |
|---|---|---|
| **How** | a second session, its placed values bound to their ranges: over the model, where that keeps the first session's order and can bind every placed value, and otherwise over the graph ONNX Runtime runs, its fusions made | a translation writing each placed value with torch's own operator: an `out=` form, the steps of a composed operator (`Gemm`, `Clip`, a normalization) in the range, cuDNN's convolution on a card, a fill, a concatenation part by part, or a copy of what a view reads or of a CPU convolution's result |
| **When it applies** | from the signature's first run, where placing saves — by a model of what the run holds at each node, in the order the second session runs them, less what that session holds of its own — more than the larger of a mebibyte and a sixty-fourth of the plain run. The first placed run is measured, and the runs after it run unplaced where it did not save that much, counting what the second session holds | from the first run: nothing placed allocates |
| **Not used** | where the second session would run other operators than the first; on an execution provider other than ONNX Runtime's CPU and CUDA ones; past 8 signatures; in a run of a graph not compiled (`Execute` or `Run` on the context); for a graph of over 20 000 nodes | in a training step whose gradient torch takes; for a model over 16 MiB; past 8 signatures; in a run of a graph not compiled; for a graph of over 20 000 nodes |

On ONNX Runtime a session keeps what it builds the second session from. A model of 16 MiB or less
it keeps in memory; a larger one on the host, in a temporary file of its own, deleted with the
session. On a card, a session over a model over 16 MiB reads the weights the model carries from
copies in the card's memory, which the second session reads too, and keeps no model: it writes out
the graph it runs, which the second session is built from, into a temporary folder kept for its
life. A session whose outputs are written into its inputs, such as a training step's, also keeps
the graph it runs, which it writes out to prove those outputs, in a temporary folder for its life,
on the host and on a card. The second session reads the weights fed to the session from the context's
memory as they are. Of the weights the model carries it holds a copy of its own for a model of
16 MiB or less, on the host as on a card; for a larger one, on the host, only the packed copies
ONNX Runtime makes of a product's weights, the rest mapped from the files the graph was written
into, and on a card none.

**Outputs on consumed memory.** An output written into an input stands on that input's memory —
its **block** — as a `TensorData` of its own over its range, never overlapping another's. Several
outputs of one run may stand on one block. How the block is freed depends on whose memory it is:

- **On ONNX Runtime**, where the block is one Shorokoo's allocator carved from its reserved memory
  — on the host a tensor of 64 KiB or more, made from your data or by a run; on a card any tensor,
  where the driver offers CUDA's virtual memory management
  ([Device memory](gpu-backends.md#device-memory-gpu-backends)): each output frees its own range as it ends, and
  what of the block no output stands on is freed as the run ends. The whole pages inside a range go
  back — 4 KiB on the host, 2 MiB on a card (512 bytes for a card block of a mebibyte or less) — to
  the allocator, as a tensor's own memory does. All but the block's first page, which the allocator
  knows the block by: it goes with the block's last output, and until then a block of a mebibyte or
  less on a card, and one on the host, keeps it; a larger block on a card hands its memory back all
  the same, keeping only its address.
- **On PyTorch**, and on a card whose driver does not offer virtual memory management, the block is
  freed when the last output on it ends, not before: torch frees a tensor's storage whole, and so
  does `cudaFree` a block. A run places outputs in such a block only where they leave at most a
  mebibyte of it unused.

A device-memory budget counts a block once, for what of it is still held, for as long as any
tensor on it is attached ([A tensor's lifetime](#a-tensors-lifetime-locks-and-deletion)).

Otherwise nothing differs: outputs are new `TensorData` attached to the running context, and
values are the same.

## A tensor's lifetime: locks and deletion

A `TensorData` **is** its memory: one object per allocation, or per range of a block several
tensors stand on — the memory of an input a run consumed and wrote them into
([A run that writes into what it consumed](#a-run-that-writes-into-what-it-consumed)). Ranges
never overlap, and each tensor's range is freed with it, or the block with the last tensor
standing on it ([Outputs on consumed memory](#a-run-that-writes-into-what-it-consumed)). A tensor records the
backend that allocated it (`AllocatingBackend`), which releases it, and where it lives: `Space`
(the device) and `Location` (device plus runtime). It does not know which contexts it is attached
to — see [Moving data between contexts](#moving-data-between-contexts).

**How a tensor ends** (disposing a context is not one of them):

| | |
|---|---|
| **Deleted** | `Delete()`, `Dispose()`, `TryDelete()` or `DeleteAsync(...)` — below. |
| **Consumed** | fed to a run as it is — or through `.TryConsume()` with nothing else reading it — which takes it when the run starts; see [Feeding a run](#feeding-a-run-consumed-shared-or-tried). |
| **Moved into an attribute** | `MoveToAttribute()`, which takes its contents — see [core-types.md](core-types.md#the-two-conversions-and-which-one-spends-its-source). |

A run's read-copy of a tensor ([above](#feeding-a-run-consumed-shared-or-tried)) ends when its
source ends, or releases its copies (as a training step does for its batch).
Elements a sequence owns end with the sequence, except one a run is reading on its own
account; a sequence whose owned element a run is reading cannot be disposed.

Every access to a dead tensor — reading, feeding, `To`, `CopyTo`, `ToHost`, `Shared()`,
`TryConsume()`, `MoveToAttribute` — throws `ObjectDisposedException` saying why (for a
consumed one, which run took it). `Shape`, `DType`, `ToString()`, `IsDisposed`,
`AllocatingBackend`, `Space`, `Device` and `Location` keep working. Ending a dead tensor is a
no-op, so double disposal is harmless.

An unreferenced tensor is reclaimed by the GC through its backend; deleting only chooses
*when*. Runtime-allocated buffers (run outputs, card copies) are native and freed at once — a
tensor standing on a block with others frees the block when it is the last; a tensor built from a
C# array frees its native read-copies at once and leaves the array to the GC.

**What a run holds.** A run holds a reader lock on every tensor it reads until it returns; any
number of runs may read one tensor. While locked, `Delete()` and `Dispose()` throw
`InvalidOperationException` and `TryDelete()` declines. `ToHost()`, `CopyTo(...)`,
`CopyMemory()`, `ValueAt()`, `CopyRawMemory()`, `MoveToAttribute()`'s copy, and the copy
`AccessMemory()` makes of a tensor on a card take the same lock while copying. A span
`AccessMemory()` hands out over host memory is not covered: keep the tensor alive and unfed while
you hold one. One over a card tensor's host copy is valid on its own.

Locks are taken inside the run, one feed at a time. A feed deleted from another thread while a
run is starting makes `Execute` throw `ObjectDisposedException`, with any feeds already taken
staying consumed — see
[A feed deleted while a run is starting loses that run](limitations.md#a-feed-deleted-while-a-run-is-starting-loses-that-run).

Disposing a `ComputeContext` throws while a run of it is in flight or it holds a lock;
disposing a compiled graph throws while one of its runs is in flight.

**Deleting:**

| | What it does |
|---|---|
| `Delete()` / `Dispose()` | The same operation: ends the tensor and releases its memory now. Throws if a run is reading it. |
| `bool TryDelete()` | The same if no run is reading the tensor. If one is, it changes **nothing** and returns `false`. `true` for a tensor already dead. |
| `Task<bool> DeleteAsync(timeout, cancellationToken)` | Ends the tensor at once, asks whatever is reading it to stop, and waits up to `timeout` for the memory to come back. |

With `DeleteAsync` the tensor is deleted either way; `false` means only that the memory had not
come back in time (it will when the run ends — do not retry). A timeout never rolls back the
stop request, and the token cancels the wait, not the deletion. The wait is bounded like
[stopping a run](inference.md#stopping-a-run): by the longest operator in flight.

**Host memory.** A tensor built from a C# array belongs to no context; its allocating backend
is `HostBackend.Instance`, readable by every host backend. `ComputeContext.Host` names host
memory as a target for `To` and `CopyTo`; `Compile`, `Execute`, `Run` and `Eval` on it refuse,
and it cannot be disposed.

The framework's own host memory is one managed array per tensor, which holds at most
`Array.MaxLength` bytes (just under 2 GiB). A larger tensor loaded or copied into host memory goes
into host memory of a backend instead, through one bounded buffer (8 MiB) a piece at a time, never
whole in a managed array: of the target context's backend, and of the one `ComputeContext.Default`
runs on for `ComputeContext.Host`, `ToHost()` and a load into host memory.

A graph's literals are
[`TensorAttribute`s](core-types.md#two-kinds-of-concrete-tensor-tensordata-and-tensorattribute),
immutable and without lifetime. One past `Array.MaxLength` bytes keeps its elements in host memory
of a backend — of the one that made the tensor it was moved from, where that tensor was a host
tensor owning its runtime value whole, and of the one `ComputeContext.Default` runs on, where it was
copied in (see [the two conversions](core-types.md#the-two-conversions-and-which-one-spends-its-source))
— released when the attribute is garbage-collected. A run or a compile hands its session a
`Constant`'s value that large as it hands it such a weight, in a subgraph too; any other
tensor-valued node attribute that large is refused with `NotSupportedException`, naming the node.

## Moving data between contexts

A tensor's memory never moves. None of these changes the tensor it is called on:

| | Result |
|---|---|
| `t.To(context)` | `t` itself, if `context`'s backend can read its memory as it stands; otherwise a new copy in `context`'s memory. Either way attached to `context`. |
| `t.CopyTo(context)` | Always a new, independent copy in `context`'s memory, attached to `context`. |
| `t.ToHost()` | `t` itself, if the host can read its memory; otherwise a new copy in host memory, attached to nothing. |

A backend reads memory in place only on **the same device and the same runtime**. Host memory
from a C# array counts as every host backend's. Two backends over one ONNX Runtime share card
allocations; a CPU backend reads them through a copy; two isolated runtimes on one card copy
through the host. A run's outputs are in its backend's memory
([Where a run's inputs and outputs are](#where-a-runs-inputs-and-outputs-are)).

```csharp
var onCard = cuda.Compile(model).Execute(input)[0].ToTensorData();  // device memory
var onHost = onCard.To(cpu);        // one copy across the bus; onCard is untouched
var same   = onHost.To(otherCpu);   // no copy: the very same object
var mine   = onHost.CopyTo(cpu);    // always a copy
```

**Where `To` or `ToHost` needs no copy, the result is `t`**: feeding it as it is consumes `t`,
and `using var h = t.ToHost();` deletes `t`. Use `CopyTo` for an independent tensor, or
`.Shared()` to keep `t` past a run.

**Attachment is bookkeeping, not ownership.** `context.Tensors` is a weak list of the context's
run outputs, what its runs read, and what `To` and `CopyTo` placed. It never keeps a tensor alive
or ends one. `context.Detach(t)` removes a tensor (refused while a
run of that context reads it) without deleting it. Disposing a context releases its sessions
and leaves every tensor. A budgeted context counts this list — see
[A context's device-memory budget](gpu-backends.md#a-contexts-device-memory-budget).

`TensorDataStruct` and `TensorDataSequence` take the same three operations. A struct comes back
as itself where nothing was copied; a sequence is copied whole if any element must be.

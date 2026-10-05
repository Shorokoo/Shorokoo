# Known limitations

**Permanent** limitations cannot work, and each says why. **Current** limitations
could be lifted by future work. For per-operator support see
[operator-support.md](operator-support.md).

## Permanent limitations

### Efficient backprop through fully dynamic convolutions

A convolution whose *kernel spatial shape* is known only at run time (the weight
shape is computed by the graph) has no efficient backward pass: the weight
gradient is itself a convolution whose pads, strides, dilations and kernel extent
must be known when the backward graph is built.

Give convolution weights a concrete shape and backprop works normally. Shapes
derived from `[Hyper]` values qualify; they are resolved by
`ToConcreteArchitecture`.

### Per-iteration convolution geometry in a dynamic loop

`NN.Conv` accepts geometry (`pads`, `strides`, `dilations`, `kernel_shape`,
`group`) computed in the graph, and resolves it to a static ONNX attribute at
concretization. (Conv is the only operator with that overload.) Geometry that
differs between iterations of a loop that stays rolled cannot be one static
attribute. A loop stays rolled when the unroll declines it, usually because its
trip count is not a compile-time constant. Such a graph is refused at
concretization:

```
the 'pads' geometry of 'shrk_Conv' varies per iteration of a loop that was not
unrolled, so it cannot be lowered to the static 'pads' attribute of 'Conv' ...
```

Either give the loop a compile-time-constant trip count
(`LoopAPI.Iterate(Scalar(3L))`), which normally unrolls it into one Conv node per
iteration, or compute the geometry from something that does not track the
iteration: a literal, a `[Hyper]` value, or an input's shape. Geometry computed
*from* a dynamic loop's result is also fine. The check is on where the geometry
comes from, so iteration-dependent geometry is refused even if its value happens
to repeat.

### Variables first assigned inside a loop body

A variable assigned inside a loop *before ever being read in that loop* cannot be
used after the loop, because there is no initial value for the zero-iteration
case. The graph is rejected as **FW046** when the value is returned from the
graph, and as the node's own **NOD001** when it feeds another node. Initialize
the variable in the loop body with `LoopAPI.Init(x)` (or read it once, e.g.
`OnnxOp.Identity(x)`) before the first assignment.

`ctx.IterationIndex` is refused the same way. To carry it out, `LoopAPI.Init` a
local and assign the index to it.

### Carrying a value computed outside the loop body

A bare assignment of a value computed *outside* the loop creates no node in the
body, so the loop has nothing to hand back, and after the loop the variable would
be the outside value even when the loop ran. Shorokoo rejects that shape. Wrap
the value with `LoopAPI.Carry` so the body produces it:

```csharp
var carry = n + Scalar(5L);
foreach (var ctx in LoopAPI.Iterate(trips))
{
    LoopAPI.Init(carry);
    carry = LoopAPI.Carry(n);   // not `carry = n;`
}
return carry;                   // the loop's result, or n + 5 after zero iterations
```

A value the body computes (`carry = carry + Scalar(1L)`, or anything built from
the iteration index) needs no wrapping. `LoopAPI.Carry` has overloads for
`Scalar<T>`, `Vector<T>` and `Tensor<T>`; for any other type, move the assignment
out of the loop. The build reports the bare assignment as the **MSG005** warning,
and concretizing raises `FW023` if it is left unfixed; `FW023` also catches
shapes MSG005 does not see, such as an alias whose first link is computed outside
the loop.

A **lagged** carry (a local holding what another carry held one iteration ago)
works unwrapped only when read inside the body or scanned:

```csharp
foreach (var ctx in LoopAPI.Iterate(trips))
{
    sum  = sum + prev;   // acc's value from the previous iteration
    prev = acc;
    acc  = acc + Scalar(1.0f);
}
```

For anything more, write `prev = LoopAPI.Carry(acc)`. Unwrapped, these are
refused:

| shape | code |
|---|---|
| reading the lagged value after the loop | **FW046** |
| lagging a local the **enclosing** loop also carries (it has nothing to carry it out by) | **FW048** |
| two ordinary carries that end on one body value, starting from different pre-loop values | **FW051** |
| a lag chain deeper than one step (`prev2 = prev; prev = acc;`), a local trailing a body value the loop does not carry, or one whose first link comes from outside the loop | **FW049** (**FW023** for the last, when declared with `LoopAPI.Init`) |

Wrapping each assignment fixes all of them; wrap every link of a chain, not only
some:

```csharp
sum   = sum + prev2;
prev2 = LoopAPI.Carry(prev);
prev  = LoopAPI.Carry(acc);
acc   = acc + Scalar(1.0f);
```

A lagged local created inside the enclosing loop's body (a nested recurrence)
needs no wrapping.

### Scanning inside a nested loop, read after the enclosing loop

`ctx.Scan` in an inner loop produces a stacked tensor once per iteration of the
*enclosing* loop. The enclosing loop's body can use it, but it cannot be read
after the enclosing loop: there is no value for the zero-iteration case, and a
stack has no pre-loop value to declare with `LoopAPI.Init`. The graph is refused,
naming the scan:

```csharp
var acc = x;
Variable? scanned = null;
foreach (var ctx0 in LoopAPI.Iterate(outerTrips))
    foreach (var ctx1 in LoopAPI.Iterate(innerTrips))
    {
        acc = acc + Scalar(1.0f);
        scanned = (Variable)ctx1.Scan(acc);
    }
return (Tensor<float32>)scanned!;   // refused: FW046
```

Consume the inner scan inside the enclosing body, or scan on the enclosing loop's
context: `ctx0.Scan(v)` works from an inner body and records one entry per
*outer* iteration. `v` must survive the nested loop; a body-local the nested loop
assigns before reading is refused as **FW050** unless you `LoopAPI.Init` it in
the nested body.

### Two exit conditions in one loop body

A loop has **one** exit condition, evaluated at the end of the iteration. A second
`ctx.Break` or `ctx.ContinueWhile` in the same body is refused as **FW052**.
Combine them into one call, and express a condition tested mid-body as an
`IfElse` over the rest of it.

## Current limitations (could be lifted)

### Little-endian platforms only

Shorokoo runs only on little-endian machines, which covers every platform .NET
officially supports (x64, x86, Arm64, Arm32). Tensor data and every file format
Shorokoo reads or writes (SafeTensors, ONNX raw data, `.srk`, `.skpt`, training
checkpoints) are little-endian and moved unconverted. On a big-endian host (such
as IBM Z) the first use of Shorokoo throws a `TypeInitializationException` whose
inner exception is a `PlatformNotSupportedException`.

### Moving a tensor between memory spaces copies it

A `TensorData` never moves: `To` hands it to a context whose backend can read it
where it is and copies otherwise, and `CopyTo` always copies; see
[Moving data between contexts](inference.md#moving-data-between-contexts). Host
backends read the framework's host memory directly. Two CUDA contexts on one
device share an allocation only when they share a native ONNX Runtime; two
*isolated* backends over one card copy through the host. There is no
device-to-device path: a tensor going to a different card goes through the host.

A tensor in memory Shorokoo cannot identify (a device value from an unknown
execution provider, or one wrapped around a runtime value without naming its
backend) reports its space as unknown; only the backend that made it reads it in
place, and `To` / `CopyTo` onto any other context copy it through that backend. A
value wrapped without its backend that the host *can* read reports its space as
the host: its accessors and `ToHost()` use it directly, but a run reads it through
a copy and `To` a running context copies it. One that the host cannot read and
whose producer is unrecorded cannot be copied, and a run fed one is refused
before it takes anything; wrap it with
`TensorData.Create(shape, dtype, value, backend)`, naming the backend.

### A feed deleted while a run is starting loses that run

A run holds every tensor it is fed until it returns: one fed `.Shared()` (or via
`.TryConsume()` while another run reads it) under a reader lock, so `Delete()` /
`Dispose()` throw and `TryDelete()` declines; one fed as it is by consuming it;
see [A tensor's lifetime](inference.md#a-tensors-lifetime-locks-and-deletion).
Holds are taken inside the run, one feed at a time. A deletion from another thread
during the call's setup (the `Execute` / `Run` call, input expansion and naming,
and the holding of earlier feeds) makes the run throw `ObjectDisposedException`,
saying the tensor was deleted.

Nothing reads freed memory and no run returns a wrong answer, but the run is lost.
A run refused before it starts takes nothing; one refused part-way (a feed died,
or another run started reading a feed this one would consume) leaves what it had
already taken consumed. The arrangement in
[One model, two devices](inference.md#one-model-two-devices), staging the next
batch while the other device reads the last one, is where this is easy to hit.
Give the concurrent run its own tensor (`CopyTo`) or wait for it to return.

### A device-memory budget counts tensors, not what the allocator keeps

A context's `DeviceMemorySettings.LimitBytes` counts the bytes of the tensors
attached to it on the card, plus what the session of its executing run
allocates; see
[A context's device-memory budget](inference.md#a-contexts-device-memory-budget).
That count is exact, but the card also holds:

- **What the allocator keeps for reuse.** Every session on a card, whichever
  runtime built it, and every tensor placed there, allocates through one
  allocator, held for the life of the process. A block a live session lets go
  of is kept for its next runs, and one a placed tensor lets go of for the next
  tensor placed — each up to the most one of its runs has used — until a run
  hands memory back (`ShrinkArenaAfterRun`, always on under a budget), the
  program calls `DeviceMemory.ReleaseCached()`, or the card has no room for a
  request. Between a budgeted context's runs that is what its tensors let go of
  since the last one; an unbudgeted context's live sessions keep up to what
  their busiest runs used.
- **Rounding.** A block on a card is its request rounded up to a multiple of
  512 bytes up to a mebibyte, and to whole 2 MiB pages above that — so a tensor
  over a mebibyte holds up to just under 2 MiB more than its bytes. On a driver
  without CUDA's virtual memory management a block over a mebibyte is rounded
  to an eighth of the power of two below it instead, up to an eighth more. A
  session's limit counts its blocks whole; the budget counts a tensor by its
  bytes.
- **A session's weights between its runs.** Each compiled graph's weights stay in
  its session's memory and count only against that session's runs, so several
  compiled graphs hold all their weights at once while the budget sees one at a
  time.
- **Memory a dead tensor still holds.** A tensor leaves the books when it dies,
  possibly before its memory returns: one deleted with `DeleteAsync` while a run
  reads it, or consumed by another context's run, holds its memory until that run
  finishes. A budgeted context's own runs cannot cause this; another context's run
  reading or consuming a tensor on this context's books can.
- **What the CUDA runtime holds for itself**: each process's CUDA context, and
  what the libraries the execution provider calls allocate on their own.

Leave headroom, and read `DeviceMemory.Read()` for what the card is carrying.

A budgeted context also does one thing at a time (a transfer onto it waits for
its run in flight), so staging the next batch onto it from another thread does
not overlap the current step. Staging through a second context over the same
backend keeps the overlap.

### A fed input's buffer is recycled only where the run consumed it

ONNX Runtime never reuses a graph input's buffer for an intermediate, whatever the
session or run options, so every fed input is resident for the whole run. A run
that consumes an input — [fed as it is](inference.md#feeding-a-run-consumed-shared-or-tried)
rather than `.Shared()` — writes its values into that input's memory where the
graph proves it safe and it saves memory
([A run that writes into what it consumed](inference.md#a-run-that-writes-into-what-it-consumed)).
What that still leaves:

- **A shared input** is resident for the whole run and written into by nothing.
- **Values under a mebibyte** are left to the runtime.
- **Order the graph does not state, on PyTorch.** A value goes into a range only
  after everything reading what the range held, by the graph's own edges; two
  independent branches never share a range. ONNX Runtime's own order is read from
  the session that runs the graph, and a value goes in once the session has run
  every reader.
- **A node the card has no kernel for.** On a card, ONNX Runtime runs it on the
  host, and its output is host memory: it is not written into an input on the card.
- **The weights of a second session.** On ONNX Runtime the values are written by a
  second session, which holds its own copy of the weights the model carries where
  it cannot share them: a model's of 16 MiB or less, on the host as on a card, and
  of a larger model on the host the packed copies ONNX Runtime makes of a
  product's weights (it keeps a weight it is handed beside its packed copy, so
  handing it one saves nothing). Placing pays only where it saves more than that
  copy.
- **A training step.** The state it consumes it already writes over
  ([A step writes its state over the state it consumed](training.md#a-step-writes-its-state-over-the-state-it-consumed)),
  and a batch it consumes is read by the backward pass as well as the forward one —
  the first layer's weight gradient reads the input — so values go into its memory
  only once that gradient is made, late in the step, and save at most the batch's
  own size.

### Some outputs written into one consumed input are freed together

Outputs a run wrote into the memory of one input it consumed stand on that memory
together. On ONNX Runtime, where the input was carved from Shorokoo's reserved
memory, each frees its own pages as it ends; otherwise the memory is freed only
when the last of them ends
([Outputs on consumed memory](inference.md#a-run-that-writes-into-what-it-consumed)):

- **On PyTorch** torch frees a tensor's storage whole, with the last tensor
  reading it, and has no call that frees part of one.
- **On a card without CUDA's virtual memory management** every block is a
  `cudaMalloc` of its own, which `cudaFree` frees whole.
- **On the host, an input under 64 KiB** is an allocation of the C runtime's
  heap, freed whole; no value that small is written into anything.
- **Pages are whole.** A page two outputs' ranges share — a 2 MiB page on a card,
  a 4 KiB one on the host — is held until both have ended. CUDA maps a card's
  memory 2 MiB at a time at the finest (the driver's minimum granularity for the
  card).

Copy an output out (`CopyTo`) to keep it apart from the others.

### A sequence's elements live in host memory

A sequence holds host-memory tensors on every backend. ONNX Runtime can pack a
device tensor into a sequence but reads elements back with a host copy, which
crashes the process with an uncatchable access violation. The ONNX Runtime
backend's `CreateSequence` therefore refuses an element in the provider's device
memory, naming the tensor:

```
A tensor (2:Float) in Shorokoo.WinGPU's own device memory cannot be an element of a sequence:
ONNX Runtime can pack it into one but reads an element back with a host copy, so nothing could
ever read it again. ...
```

The framework passes it host copies only, so this reaches only direct callers of
`CreateSequence`. A refusal, like any failure of `CreateSequence`, releases every
value it was given; the source tensor is untouched, so bring it home with
`TensorData.ToHost()` and build the sequence again.

Models whose *outputs* are sequences (`SequenceAt`, `SplitToSequence`, anything
producing an ONNX sequence type) are unaffected: a run leaves a sequence output in
host memory on every execution provider, which is where a backend reads and leaves
sequences. They run on CUDA backends, including ones loaded through
`IsolatedBackend.Load` or `BackendPackage.TryLoad`, and a sequence moves to another
context element by element.

### Device-memory readings are process-wide, and device 0's

Device-memory configuration is per context, session and run
(`ComputeContext.DeviceMemory`, `RunSettings`; see
[Device memory](inference.md#device-memory-gpu-backends)), so two models on one
host can have separate budgets.

Reporting is process-wide. `DeviceMemory.Read()` and `Sample()` query the CUDA
device current for the calling thread (device 0, which the shipped GPU backends
use) and return the whole device's usage, including other processes, and this
process's share of it (`ProcessBytes`). `PeakUsedBytes` and `PeakProcessBytes`
are one record each for the process: two contexts training side by side in one
process share them, and neither can be split between contexts.
`CompiledGraph.ReadArenaStatistics()` reads one session's allocator and
`ComputeContext.ReadDeviceMemoryUse()` what a context holds against its budget
(see [What one session's allocator did](inference.md#what-one-sessions-allocator-did)).

### Backprop through dynamic loops

Gradients through a `Loop` whose trip count is known only at run time are
rejected with `AutoDiffNotSupportedException`: autodiff would need a tape or
forward re-execution, and Shorokoo supports neither. A loop with a statically
known trip count (`LoopAPI.Iterate(n)` with `n` a compile-time constant) is
unrolled and differentiates normally.

"Statically known" means known when the training graph is built: a count read off
an input's shape does not qualify, because the rig compiles one trainstep for all
input shapes. A count computed from constants qualifies
(`LoopAPI.Iterate(Scalar(3L) * Scalar(2L))` is folded before the unroll), and so
does a `[Hyper]`.

The rejection also covers module-owned state updated inside such a loop.

### A training rig feeds its model tensors and optional tensors

A model may take a `TensorSequence<T>` for inference, but a `TrainingRig` accepts
only tensor and optional-tensor inputs. `TrainingRig.FromScratch` refuses such a
model before building anything, naming the input, with a
`NotSupportedException`. Train a model that takes the sequence's tensors as
separate inputs, or builds the sequence from them itself.

### Conditional execution in a training graph

In a training graph, an `IfElse` arm runs, forward and backward, only on the
steps that take it, as it does in inference: the forward values its backward
reads are computed in the arm, and its backward runs behind the arm's own
condition. The arm that did not run contributes a *selected* zero to the
gradient, so a non-finite derivative there (`sqrt` of a negative, division by
zero) cannot poison the weights.

A value read by arms of `IfElse`s on different conditions, or both by an arm
and outside it, belongs to no single arm. It runs on every step, as in inference, and so
does the part of the backward pass that differentiates it: an operation that
would *fail* there (not merely return a non-finite number) off its branch must
stay off the differentiated path.

A random draw in an arm, such as a `Dropout` mask, trains like the rest of the
arm. The backward pass reads the mask the forward pass drew; it does not draw
again. The execution counter that gives each step fresh masks (see
[rng-configuration.md](rng-configuration.md)) advances only on the steps that
take the arm. On the other steps it carries through unchanged. A draw in an arm
of an `IfElse` nested in another's arm is the same: it, and its counter, run only
on the steps whose conditions select every arm around it. A draw that is also read
outside the arm runs on every step, and so does its counter.

### Gradient (activation) checkpointing

`[Module(Checkpoint = true)]` marks every call of the module as a segment whose
forward activations are recomputed in the backward pass instead of kept, like
PyTorch's `torch.utils.checkpoint`; see
[Activation checkpointing](nn-library.md#activation-checkpointing). It is always
honoured, even on steps the automatic pass below skips. The finest grain is a
module; a single tensor cannot be checkpointed.

For attention, `queryChunks: c` on `Attention.ScaledDotProductAttention` divides
the score-sized **transients** by `c` but not what the step retains for the
backward pass, at a compute cost; see
[Sizing an attention run](nn-library.md#attention-memory).

Building a training rig also runs an automatic memory-aware pass that reorders
nodes and recomputes tensors where that improves a combined compute-and-memory
objective. It charges a step's memory as the backend of the rig's runtime context
lays a run out: ONNX Runtime's allocation plan, with the temporaries its CPU
kernels hold beside their outputs on the host; on PyTorch, the translation's —
each value freed after its last read, a view as its input's memory, a result
written over a contiguous operand dying there. It judges its search by the
backend's own model of a run: each strategy takes a step only where that model
scores it better, and the strategies are chosen among by it, so that by that model
the step it hands over holds no more there than the one it was handed. PyTorch's
model is quick to ask, so the rematerializer also weighs by it the three candidates
its own figures score best where they leave it nothing better to take; ONNX
Runtime's builds a session each time it is asked, and is asked only about each
strategy's steps. ONNX Runtime's model reads the
graph ONNX Runtime optimizes and runs, in the order it runs it, with the buffers
its allocation plan keeps for later values of the same shape and the scratch its
kernels take: convolutions' and recurrent layers' working buffers on the host, and
on a card a reduction over inner axes. PyTorch's adds to the translation's layout
the temporaries of a layer normalization and, on the CPU, of a convolution, and the
copies torch makes of a value whose strides a reshape or a matrix product cannot
use as they lie — an element-wise result lies as its operands do, so a product of
transposed attention heads is transposed too — and a recurrent layer's stacked
outputs. Both run a loop's body once per iteration where its trip count follows
from the shapes fed, and hold what a sequence holds for as long as the sequence
lives. Where a model cannot tell a value's shape, the pass's own figures decide.
It is conservative and has no
opt-out; use the attribute to force a trade it would not take. It picks the
tensors it recomputes, but not whole-module segments: checkpointing each layer of
a two-layer transformer encoder holds a further 5–6% less than the pass alone on
ONNX Runtime, for a tenth more step time or more, and up to 4% less on PyTorch. The
pass cuts a step's peak memory by anywhere from nothing to about a third on ONNX
Runtime, for at most a few percent more computation; on PyTorch it can trade more
— a two-layer encoder's step holds over 40% less, for about a fifth more
computation. A step containing a scope — a recurrent op's backward pass is a loop
— is searched only where the backend's model answers for it, which it does for a
loop whose trip count follows from the shapes fed, and for no branch (`If`); a
one-layer LSTM's step then holds about a tenth less on PyTorch and a sixth to a
fifth less on ONNX Runtime, by reordering alone. Any other step with a scope gets only
its checkpoint attributes applied. Separately, the
training-step session is compiled for the shapes it is fed, which removes most
shape arithmetic from the executed graph.

The `Shorokoo.Core.AutoDiffCheckpointing` namespace is internal despite being
public in the assembly; no API returns its types, so do not build on them. Ten other
`Shorokoo.Core.*` namespaces carry documented public API, listed in
[orientation.md](orientation.md#public-core-namespaces); treat the rest as unsupported.

### Quick Execution Engine value computation is bounded

The Quick Execution Engine (QEE) propagates **dtype and shape** for every
supported operator, but computes **values** only for small tensors (up to
`MaxDataElements`, default 256 elements). It also leaves the value uncomputed for
`ReduceMax`, `ReduceMin`, `ReduceMean`, `ReduceLogSum` and `ReduceLogSumExp` over
an axis of extent 0, because their empty-group result depends on the element type.
The ONNX spec fixes it for four of them: the type's lowest value for `ReduceMax`
(−inf, the integer minimum, false), its highest for `ReduceMin` (+inf, the integer
maximum, true), and −inf for `ReduceLogSum` and `ReduceLogSumExp`; it leaves
`ReduceMean` of an empty group undefined. Every backend gives those four the
spec's value for every dtype it has a kernel for — the ONNX Runtime backend
through the rewrites listed in
[operator-support.md](operator-support.md#reductions). `ReduceMean` of an empty
group is what each backend's kernel gives; for a floating-point type the PyTorch
and JAX backends give NaN. Use the ONNX Runtime backend (`OnnxEngine.Eval` /
`ComputeContext`) for real numeric execution.

### Uniform draws resolve a bounded span of magnitudes

A uniform draw uses one 64-bit generator value per element and resolves `float32`
values only over the top 41 weight classes of the range (40 when it straddles
zero). A class is `max(1, exponent field)`: a binade, except that the subnormals
and the smallest normal binade share one. Below that floor the draw keeps its due
probability mass but on an even lattice, so those floats are not individually
drawable; about 33% of the floats in `[0, 1)` and 16% of all finite `float32`
values can be drawn. At extreme ranges, a single float can take up to twice its
due share when the range's total weight is not a power of two (it is exact for
`[0, 1)` and `[-1, 1)`), and a side worth less than one weight unit (the negative
side of `[-1, 1e30)`) has probability zero. See [uniform-draws.md](uniform-draws.md).

### Normal draws stop at 8 sigma and resolve a bounded span of magnitudes

A normal draw uses one 64-bit generator value per element (1 sign bit, 63 for the
magnitude) and resolves magnitudes from 2⁻³⁹ (1.818989e-12) to 8.

- **Above**: everything at or past 8 sigma decodes to exactly `8.0f`, which
  carries the 1.244e-15 tail mass (about one draw in 800 trillion); a scaled draw
  never leaves `mean ± 8·scale`.
- **Below**: magnitudes under 2⁻³⁹ lie on an even lattice, so they are not
  individually drawable; the region keeps its exact mass (1.4513e-12, about one
  draw in 690 billion).
- **Near the top**: 577,209 magnitudes between 7.6011825 and 8 are never drawn.

In all, 720,265,872 of the 4,278,190,080 finite `float32` values (16.8%) can be
drawn. See [normal-draws.md](normal-draws.md).

### ONNX `Scan` import

A model containing a `Scan` node is rejected at import; Shorokoo executes `Loop`
and does not rewrite one into the other. Express the iteration as an explicit
`Loop` (slice each per-iteration input with `Gather` on the iteration index and
let the `Loop` stack its scan outputs), or re-export the model in that form. In
Shorokoo, use `LoopAPI` and `ctx.Scan`.

### ONNX `SequenceMap` import

A model containing `SequenceMap` is rejected at import: lowering it needs
whole-graph type inference (its extra inputs are mapped when sequence-typed and
broadcast when tensor-typed), and ONNX Runtime has no `SequenceMap` kernel.
Express the mapping as a `Loop` over `SequenceLength` using
`SequenceAt`/`SequenceInsert` (in Shorokoo, with `LoopAPI`).

### C# emission of a runtime-built TensorStruct

`SaveToCSharp()`, and the `DebugRequests` snapshots built on it, name a
TensorStruct by its `IStruct` interface. A struct assembled at runtime from a
`TensorStructDef` has none, so emission is refused with **FW053**. Structs built
with `Globals.TensorStruct<T>` / `TensorStructCreate<T>`, generic interfaces
included, are unaffected. Declare the struct as an `IStruct` interface instead.

### ONNX opset 21 only

Shorokoo reads and writes a single ONNX opset: **21**.

**Import** — `Persistence.ImportOnnx`, `OnnxModelImporter.FromOnnxModel` and the payload of a
`.srk` alike — requires the model's default-domain (`""` / `ai.onnx`) opset import to be 21,
wherever it sits in the model's `opset_import` list, and so for each function that declares
one. A model that declares no default-domain opset, or any other one, is refused with
**FW060**, naming the opset found and the one required. Convert such a model to opset 21
before importing it, for example with `onnx.version_converter.convert_version(model, 21)` in
Python.

An opset-21 model is also refused with **FW060**, naming the node, when a node is an operator
ONNX introduced after opset 21 (`Attention`, `RMSNormalization`, `RotaryEmbedding`,
`TensorScatter`, `Swish`, `BitCast`, `CumProd`, …) or carries an attribute ONNX added after
it (`DequantizeLinear.output_dtype`, `QuantizeLinear.precision`,
`Cast`/`CastLike.round_mode`): opset 21 does not define them, so the model is malformed.

**Export and every save** stamp the model, and each of its functions, at exactly 21 — an
exported `.onnx`, the model a backend's session is built from and a `.srk` payload alike.
Nothing raises the stamp. The post-21 operators an authored graph can name never reach the
file:

- `Swish` and `RMSNormalization` lower inline to opset-21 primitives.
- `TensorScatter` is written as a `Concat` plus `GatherElements` (see
  [operator-support.md](operator-support.md)), in an exported file and a saved `.srk` or
  `.skpt` alike; a reloaded graph carries that decomposition in its place and computes the
  same values.
- `Attention`, `AttentionWithKVCache`, `RotaryEmbedding`, `BitCast` and `CumProd`
  throw `NotImplementedException` at their `OnnxOp` entry points.

No node carries a post-21 attribute: the low-level `NodeBuilder` refuses to build one with
**FW060**, naming the operator and the attribute. A node built with `NodeBuilder` can be a
post-21 operator; writing it is refused with **FW060**, naming the node, rather than emitted.

The opset 22–26 respecifications of the operators opset 21 defines only widen dtype lists
(bfloat16, float4e2m1, float8e8m0, int2/uint2, all unsupported in Shorokoo; see below) besides
adding those optional attributes, so an opset-21 model expresses everything Shorokoo runs.

### Sub-byte and complex dtypes

`Float16` and `BFloat16` are fully supported: `.safetensors` `F16`/`BF16`
payloads load and save (`SafeTensorLoader`), `TensorDataConversion` converts
float32→f16/bf16 with round-to-nearest-even, and ONNX f16/bf16 initializers
(`raw_data` or int32-packed `int32_data`) import and export. The Quick Execution
Engine stores f16/bf16 values as float32, so QEE values do not model the precision
loss.

`Int4`/`UInt4` are unsupported (no sub-byte storage); materializing them raises
`UnsupportedDTypeException` (`DT001`/`DT002`/`DT010`/`DT011`). Also unsupported as
element types: the float8 family (`Float8E4M3FN`, `Float8E4M3FNUZ`, `Float8E5M2`,
`Float8E5M2FNUZ`, `Float8E8M0`), `Float4E2M1`, `Int2`/`UInt2`, and
`Complex64`/`Complex128`. The [PyTorch](pytorch-backend.md#values) and
[JAX](jax-backend.md#values) backends hold complex and float8 tensors at the
backend level (`IShorokooTensorValue`), but a Shorokoo model cannot declare them.

### Gradient coverage

Most differentiable operators (opset 21 plus the post-21 additions) have gradient
support, through a gradient rule or a registered decomposition. The rest raise
`AutoDiffNotSupportedException` with an error code naming the op. Per-operator
status is in [operator-support.md](operator-support.md).

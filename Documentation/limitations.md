# Known limitations

**Permanent** limitations cannot work, and each says why. **Current** limitations
could be lifted by future work. For per-operator support see
[operator-support.md](operator-support.md).

## Permanent limitations

### Efficient backprop through fully dynamic convolutions

A convolution whose *kernel spatial shape* is known only at run time (the weight
tensor's shape is computed by the graph) cannot get an efficient backward pass.
The weight gradient of a convolution is itself a convolution whose attributes
(pads, strides, dilations and the kernel extent) must be known when the backward
graph is built; with a fully dynamic kernel there is no fixed backward graph to
build. This is a property of graph-mode autodiff.

Give convolution weights a concrete shape and backprop works normally. Shapes
derived from `[Hyper]` values qualify: they are resolved when the architecture is
concretized via `ToConcreteArchitecture`.

### Per-iteration convolution geometry in a dynamic loop

`NN.Conv` lets you compute a convolution's geometry (`pads`, `strides`,
`dilations`, `kernel_shape`, `group`) in the graph instead of writing it as a
literal, and resolves it to a static ONNX attribute when the architecture is
concretized. (Conv is currently the only operator with that overload.) A static
attribute holds one value for every execution of its node. That fails in one
place: geometry that differs between iterations of a loop that stays rolled. The
whole loop is then a single Conv node, and no single attribute value is right for
all its iterations.

A loop stays rolled when the unroll declines it, most often because its trip
count is not a compile-time constant; a few body shapes decline too. Such a graph
is refused at concretization, naming the geometry that varies:

```
the 'pads' geometry of 'shrk_Conv' varies per iteration of a loop that was not
unrolled, so it cannot be lowered to the static 'pads' attribute of 'Conv' ...
```

Either give the loop a compile-time-constant trip count, which normally unrolls
it (`LoopAPI.Iterate(Scalar(3L))`) so each iteration becomes its own Conv node
with its own geometry, or compute the geometry from something that does not
track the iteration: a literal, a `[Hyper]` value, or an input's shape.

Constant and input-derived geometry inside a dynamic loop is fine, and so is
geometry computed *from* a dynamic loop's result: that value is computed once,
and the Conv reading it runs once. What is refused is geometry that follows a
loop's iteration where it is used, even if its value happens to repeat. The
check is on where the geometry comes from, since a carry's value is not knowable
at build time.

### Variables first assigned inside a loop body

A variable assigned inside a loop *before ever being read in that loop* cannot be
used after the loop. Shorokoo cannot recover its initial value (needed for the
zero-iteration case) and rejects the graph: as **FW046** when the value is
returned from the graph, and as the node's own **NOD001** when it feeds another
node. Both carry the same guidance. Initialize the variable inside the loop body
with `LoopAPI.Init(x)` (or read it once, e.g. `OnnxOp.Identity(x)`) before the
first assignment.

`ctx.IterationIndex` is refused the same way: it is the loop's own per-iteration
counter and has no value once the loop has exited. To carry it out,
`LoopAPI.Init` a local and assign the index to it.

### Carrying a value computed outside the loop body

A loop hands back its result by re-tracing the node that produced the body's
value and pointing the variable at the loop's output. A bare assignment of a
value computed *outside* the loop creates no node in the body, so there is
nothing to re-trace, and after the loop the variable is that outside value,
indistinguishable from its other uses. Shorokoo rejects that shape rather than
return the body's value even when the loop ran zero times.

Wrap the value with `LoopAPI.Carry` so the body produces it:

```csharp
var carry = n + Scalar(5L);
foreach (var ctx in LoopAPI.Iterate(trips))
{
    LoopAPI.Init(carry);
    carry = LoopAPI.Carry(n);   // not `carry = n;`
}
return carry;                   // the loop's result, or n + 5 after zero iterations
```

A value the body already computes (`carry = carry + Scalar(1L)`, or anything
built from the iteration index) needs no wrapping. `LoopAPI.Carry` has overloads
for `Scalar<T>`, `Vector<T>` and `Tensor<T>`; for any other carry type, move the
assignment out of the loop.

The build reports the bare-assignment form as the **MSG005** warning on the
offending line, and concretizing raises `FW023` if it is left unfixed. `FW023`
also catches shapes MSG005 does not see, such as an alias whose first link is
computed outside the loop.

A **lagged** carry (a local holding what another carry held one iteration ago)
needs the same wrapper in every case but one: reading the lagged value inside the
body, or scanning it, works unwrapped:

```csharp
foreach (var ctx in LoopAPI.Iterate(trips))
{
    sum  = sum + prev;   // acc's value from the previous iteration
    prev = acc;
    acc  = acc + Scalar(1.0f);
}
```

For anything more, write `prev = LoopAPI.Carry(acc)`. A bare assignment gives the
lagged local no body node of its own (it shares the node of the carry it trails),
and these shapes are refused rather than answered wrongly:

| shape | code |
|---|---|
| reading the lagged value after the loop | **FW046** |
| lagging a local the **enclosing** loop also carries (it has nothing to carry it out by) | **FW048** |

Two ordinary carries that end on one value, having started from different
pre-loop values, need no lag but share a body value too. The loop has one node to
hand back and two variables wanting it, so this is refused as **FW051**; wrap
each assignment and each gets its own node.

A lagged local created inside the enclosing loop's body (a nested recurrence,
say) crosses no boundary and needs no wrapping.

The loop identifies a local trailing **one of its own carries** by one iteration.
A deeper chain (`prev2 = prev; prev = acc;`), a local trailing a body value the
loop does not carry, and one whose first link comes from outside the loop are
refused as **FW049**; the last shape, when *declared* with `LoopAPI.Init`, is
caught earlier as **FW023**. One extra identification pass sees one lag step, so
the depth limit follows from the number of passes. Wrapping every link works;
wrapping only some leaves the rest sharing a body value:

```csharp
sum   = sum + prev2;
prev2 = LoopAPI.Carry(prev);
prev  = LoopAPI.Carry(acc);
acc   = acc + Scalar(1.0f);
```

### Scanning inside a nested loop, read after the enclosing loop

`ctx.Scan` in an inner loop produces a stacked tensor once per iteration of the
*enclosing* loop. The enclosing loop can pass that tensor to its own body, but
cannot hand it back after the loop: as with any value the body assigns before
reading it, there is no value to return for the zero-iteration case, and a stack
has no pre-loop value to declare with `LoopAPI.Init`. Shorokoo refuses the graph,
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

Consume the inner loop's scan output inside the enclosing loop's body, or scan on
the enclosing loop's own context: `ctx0.Scan(v)` works from an inner body and
records one entry per *outer* iteration.

That route needs `v` to survive the nested loop, since the enclosing loop records
the value that loop ends with. A body-local the nested loop assigns before
reading does not, and is refused as **FW050**; `LoopAPI.Init` it in the nested
body and the scan records it.

### Two exit conditions in one loop body

A loop carries **one** exit condition, evaluated at the end of the iteration, so
a second `ctx.Break` or `ctx.ContinueWhile` in the same body would replace the
first. Two of them are refused as **FW052**. Combine them into a single call, and
express a condition tested part-way through the body as an `IfElse` over the rest
of it.

## Current limitations (could be lifted)

### Little-endian platforms only

Shorokoo runs only on little-endian machines, which covers every platform .NET
officially supports (x64, x86, Arm64, Arm32). Tensor data, ONNX raw data and
every file format Shorokoo reads or writes (SafeTensors, `.srk`, `.skpt` and
training checkpoints) are little-endian, and tensor bytes move between memory
and disk unconverted. On a big-endian host (such as IBM Z) all of them would be
silently misread, so the first use of Shorokoo there fails with a
`TypeInitializationException` whose inner exception is a
`PlatformNotSupportedException`. Supporting big-endian hosts would mean
byte-swapping every tensor and persisted field on the way in and out.

### Moving a tensor between memory spaces copies it

A `TensorData` is its memory and never moves: `To` hands it to a context whose
backend can read it where it is and copies it otherwise, and `CopyTo` always
copies; see [Moving data between contexts](inference.md#moving-data-between-contexts).
Every host backend reads the framework's own host memory as it stands. Two CUDA
contexts on one device share an allocation only when they share a native ONNX
Runtime; a device allocation means nothing to a runtime that did not make it, so
two *isolated* backends over one card copy through the host. Crossing between
host and device is a real copy, once per crossing. A tensor cannot be in two
spaces at once, and there is no direct device-to-device path: a tensor going to a
different card goes through the host.

A tensor in memory Shorokoo has no name for (a device value produced by a backend
on an execution provider it does not know, or one wrapped around a runtime value
without naming the backend that made it) reports its space as unknown. Two such
allocations compare equal as spaces without being in the same place, so only the
backend that made one reads it where it is; `To` and `CopyTo` onto any other
context copy it, through that backend. A value wrapped without its backend that
the host *can* read reports its space as the host: its accessors read it, and
`ToHost()` returns it as it is. Since which runtime made it is unknown, no runtime
is handed it directly: a run reads it through a copy, and `To` a context that
runs copies it. One whose producer was not recorded and that the host cannot read
cannot be copied at all, and a run fed one is refused before it takes anything;
wrap such a value with `TensorData.Create(shape, dtype, value, backend)`, naming
the backend.

### A feed deleted while a run is starting loses that run

A run holds every tensor it is fed until it returns. One it reads (fed
`.Shared()`, or through `.TryConsume()` while another run was reading it) it
holds under a reader lock, so deleting it from another thread *while the run
holds it* is refused: `Delete()` and `Dispose()` throw, `TryDelete()` declines,
and the run reads on. One fed as it is, the run holds by taking it: the tensor is
dead from then on, so a later `Delete()` has nothing to do; see
[A tensor's lifetime](inference.md#a-tensors-lifetime-locks-and-deletion). Either
hold is taken inside the run, one feed at a time, and everything before it is
unprotected: the `Execute` / `Run` call itself, the expansion and naming of its
inputs, and the holding of earlier feeds. A deletion in that window ends the
tensor, so the lock or take the run then requests is refused and the call throws
`ObjectDisposedException`, saying the tensor was deleted.

The failure is clean (nothing reads freed memory, and no run returns a wrong
answer), but the run is lost, and not always alone. A run checks every feed
before it takes any, so one refused before it starts takes nothing; but a feed
that dies in this window, or that another run starts reading where this one would
consume it, refuses the run part-way, and what it had taken stays consumed. The
window is the whole of the call's setup. It is also the arrangement
[One model, two devices](inference.md#one-model-two-devices) invites (staging
the next batch while the other device is still reading the last one), where it is
easy to write by accident. Give the concurrent run its own tensor (`CopyTo`) or
wait for it to return.

Nothing detects an impending deletion; the hold gives a refusal when the run
reaches for a tensor that is gone. Closing the window would mean taking the hold
where the caller still holds the tensor, at the entry point before the inputs are
expanded, which would also have to cover `Run`, `Eval`, and the one-shot paths
that build their own session.

### A device-memory budget counts tensors, not arenas

A context's `DeviceMemorySettings.LimitBytes` is kept by counting what is on its
books: the bytes of the tensors attached to it in the card's memory, and the
arena limit of whichever of its runs is executing, which is cut to what those
tensors leave; see
[A context's device-memory budget](inference.md#a-contexts-device-memory-budget).
That count is exact. It omits memory the card holds all the same:

- **The allocator tensors are placed from.** A tensor put on a card (by `To`,
  `CopyTo`, `AllocateUninitialized`, or a run copying a host tensor there to read
  it) comes from one allocator per card and runtime, shared by every context over
  that runtime (a backend loaded in isolation has its own runtime and allocator)
  and held for the life of the process: an ONNX Runtime tensor frees itself
  through the allocator that made it, so the allocator must outlive every tensor
  it served. Nothing asks it to shrink, so a deleted tensor's bytes return to it
  rather than to the card, and it keeps holding the most that was ever on the card
  through it at once, whatever any budget counts now.
- **What an arena keeps spare.** An arena takes blocks, not bytes, and a
  partly-used block cannot be handed back. Every run under a budget hands back
  what it can as it ends, but a session's arena can hold more than is in use.
- **A session's weights between its runs.** A compiled graph's weights live in its
  session's arena for as long as the session does, and count only against that
  session's own runs. A context that has compiled several graphs with large
  weights holds all of them at once, which the budget sees one at a time. A
  session rebuilt for a lower limit leaves its old arena alive as long as outputs
  its runs left there are.
- **Memory a dead tensor still holds.** The books count live tensors, and a tensor
  leaves them when it dies, which can be before its memory comes back. One deleted
  with `DeleteAsync` while a run is reading it holds its memory until that run
  returns; one a run of another context consumes holds it until that run is
  finished with it. A budgeted context's own runs close that window, since nothing
  is placed on the context while one is in flight; a run of another context,
  reading or consuming a tensor this context also has on its books, leaves it open
  until that run returns, and for that long the budget undercounts the card.

A budget is a ceiling on what the context counts, and the card can hold more:
leave headroom, and read `DeviceMemory.Read()` for what the card is carrying.

Two costs follow from how the budget is kept. A session's arena limit only comes
down: a context that lets go of what it held keeps the smaller arenas its graphs
were rebuilt with until the graph is compiled again. And under a budget a context
does one thing at a time (a transfer onto it waits for its run in flight), so
staging the next batch onto a budgeted context from another thread does not
overlap the step it is staged for. Staging through a second context over the same
backend keeps the overlap: the budgeted run reads the batch where it is, and
counts it then.

### A fed input's buffer is not recycled inside the run

ONNX Runtime's memory planner reuses a buffer only where a kernel declares the
reuse and the input's use count says that kernel is its last reader. It seeds
every graph input with one extra use count so a caller can still read a feed
after `Run` returns, so the test never passes for a feed, and no session or run
option changes it. Shorokoo's memory-aware pass models this the same way: every
fed input is resident for the whole step.

That costs nothing on a training step, whose peak is intermediates and whose
output is a scalar loss. It matters for a pipeline over an input so large that it
dominates the peak, with an input-shaped output: the one buffer that can never be
recycled is the largest in the run.

[Feeding the input as it is](inference.md#feeding-a-large-input-without-a-second-copy),
rather than `.Shared()`, is the available lever: the run consumes it, which
releases it when the run returns instead of when the caller lets go, and
allocating on the context removes the managed copy beside it. Neither makes the
bytes available to the run's own intermediates. Only writing an output into the
input through ONNX Runtime's I/O binding does, which ORT permits and checks
nothing about: it saves one input-sized buffer, only where an output matches that
input's dtype and shape and nothing reads the input after the output is written,
and nothing where the output is a loss. A training step does this for the state
it replaces, whose lowering proves which outputs qualify; see
[A run that writes an output into what it consumed](inference.md#a-run-that-writes-an-output-into-what-it-consumed).
A graph you compile yourself marks no outputs, so your own pipeline still holds
its input beside its input-shaped output.

### A sequence's elements live in host memory

A sequence holds host-memory tensors, on every backend. ONNX Runtime will pack a
tensor an execution provider left on a card into a sequence but cannot read it
back: its `GetValue` copies an element with a plain host `memcpy` whatever
allocator it is given, so the first read dereferences a device address from the
host and kills the process with an uncatchable access violation. Such a sequence
is write-only.

So the ONNX Runtime backend's `CreateSequence` refuses an element in the
provider's own memory. The framework passes it host copies only, so only a direct
caller of `CreateSequence` meets this; a refusal, like any failure of
`CreateSequence`, releases every value it was given. The tensor those values were
copied from is untouched: bring it home with `TensorData.ToHost()` and build the
sequence again. The refusal names the tensor:

```
A tensor (2:Float) in Shorokoo.WinGPU's own device memory cannot be an element of a sequence:
ONNX Runtime can pack it into one but reads an element back with a host copy, so nothing could
ever read it again. ...
```

Models whose *outputs* are sequences (`SequenceAt`, `SplitToSequence`, anything
producing an ONNX sequence type) are unaffected. ONNX Runtime materializes a
run's sequence output in host memory whichever execution provider produced it,
and a sequence output flagged in `Execute(inputs, retainOnDevice)` comes back
there too, since only a tensor is left on the card. Such a model runs on a CUDA
backend, and on one loaded through `IsolatedBackend.Load` or
`BackendPackage.TryLoad`, like any other; its elements read, and the sequence
moves to another context element by element.

A device-aware copy inside ONNX Runtime's `GetValue` would make a device-resident
sequence readable and lift this refusal.

### Device-memory readings are the device's, and device 0's

Device-memory configuration is per context, per session and per run:
`ComputeContext.DeviceMemory` for the budget a context keeps on its card and the
sessions it compiles, `RunSettings` for what a run does; see
[Device memory](inference.md#device-memory-gpu-backends). Two contexts may differ,
so a host running two models can give them separate budgets and arena strategies.

The *reporting* is process-wide. `DeviceMemory.Read()` and `Sample()` query the
CUDA device current for the calling thread (device 0, which the shipped GPU
backends use) and return the whole device's usage, not this process's share, so
another process on the card is in your figures. `PeakUsedBytes` is likewise one
record for the process.

Per-allocator figures exist for what they cover:
`CompiledGraph.ReadArenaStatistics()` reads one session's own arena, and
`ComputeContext.ReadDeviceMemoryUse()` what a context holds against its budget;
see [What one session's arena did](inference.md#what-one-sessions-arena-did).
None gives this process's share of the card: nothing reads the allocator tensors
are placed from, and a process with several sessions has several arenas to add
up. For that total, the device's own figure is all there is.

### Backprop through dynamic loops

Reverse-mode autodiff through a `Loop` whose trip count is known only at run time
requires either recording per-iteration intermediates (a tape) or re-executing
the forward body during the backward pass. Shorokoo's graph-mode autodiff
supports neither, so gradients through dynamic loops are rejected with
`AutoDiffNotSupportedException`. Loops with a statically known trip count can be
unrolled (iterate with `LoopAPI.Iterate(n)` where `n` is a compile-time constant)
and differentiate normally.

"Statically known" means known when the training graph is built, not when it
runs, so a count read off an input's shape does not qualify: the rig compiles one
trainstep for all input shapes and specializes only the ONNX session per shape. A
count computed from constants qualifies however it is written
(`LoopAPI.Iterate(Scalar(3L) * Scalar(2L))` is folded before the unroll), and so
does a `[Hyper]`, which is a constant once the model is concretized.

The rejection also covers module-owned state inside such a loop: an update
registered there has no value the training graph can carry out of the body.

### A training rig feeds its model tensors and optional tensors

A model may take a `TensorSequence<T>` for inference, but a `TrainingRig` cannot
be built over one: its stages past concretization (the representative inputs that
seed its shape inference, the batch definition its steps are checked against)
know a model input only as a tensor or an optional tensor.
`TrainingRig.FromScratch` refuses such a model before building anything, naming
the input, with a `NotSupportedException`. Train a model that takes the
sequence's tensors as separate inputs, or that builds the sequence from them
itself.

### Conditional execution in a training graph

An `IfElse` branch runs only when its condition selects it. A training graph
keeps that for the branch as a whole, but not for the forward values the backward
pass reads: reverse-mode autodiff needs the forward's intermediates, and a value
read unconditionally cannot be computed conditionally, so those are hoisted out
of the branch and computed on every step.

Gradients are unaffected: the arm that did not run contributes exactly zero, and
that zero is *selected*, not produced by multiplication, so an arm whose
derivative is not a number (`sqrt` of a value negative on that path, a division
by a value zero there) cannot poison the weights. The cost is the extra work, and
an operation that would *fail* off its branch, rather than return a non-finite
number, must stay off the differentiated path.

### Gradient (activation) checkpointing

Activation checkpointing is a per-module attribute: `[Module(Checkpoint = true)]`
marks every call of the module as a segment whose forward activations are
recomputed in the backward pass instead of kept, like PyTorch's
`torch.utils.checkpoint`; see
[Activation checkpointing](nn-library.md#activation-checkpointing). It is the one
manual memory lever, and it is always honoured: the rig applies it before its own
compute-versus-memory objective, even to a step so small that the automatic pass
below would skip it. The finest grain is a module; a single tensor cannot be
checkpointed.

Attention has one more lever, which is not checkpointing: passing
`queryChunks: c` to `Attention.ScaledDotProductAttention` splits the query axis
into `c` blocks, which divides the score-sized **transients** by `c` but not what
the step retains across the backward pass. The saving is shape-dependent and
costs compute, so measure both. See
[Sizing an attention run](nn-library.md#attention-memory) for the arithmetic and
the real cost of the quadratic term.

Building a training rig also runs an internal memory-aware pass over the lowered
training-step graph, which reorders nodes and recomputes tensors instead of
keeping them alive where that improves a combined compute-and-memory objective.
It recomputes as gradient checkpointing does (a chain of producers back to values
that are live anyway, cloned once and shared by every gradient that reads it,
with placement and chain depth chosen by evaluating the candidate graph), but
conservatively, because it is automatic and has no opt-out: it takes trades its
objective accepts, and the attribute above asks for one it would not. Its memory
model is ONNX Runtime's own allocation plan for the step (the order ORT runs, and
ORT's reuse of a dead buffer for the next tensor of the same shape), so it
optimizes what gets allocated. The framework's memory benchmark records the
resident peak of one training step next to the modelled one; on the graph the
pass returns, the two agree within about 10% except for a conv stack, whose
im2col workspace the model does not see, and the LSTM step, whose Loop body the
model walks once. On the graph before the pass the model runs up to 13% high on
attention. Measured that way, unoptimized to optimized: an MLP 5.5 to 4.2 MiB, a
conv stack 28.6 to 24.7, a one-layer transformer encoder 19.5 to 18.9, a
two-layer one 37.3 to 35.8, dense attention unchanged, chunked attention 5.8 to
5.0, for at most a few percent more kernel time. Any graph carrying a scope,
whether its backward pass runs through a recurrent op or it has a forward `If`,
comes back with only its checkpoint hints applied, because the evaluator walks a scope body once where ORT
runs it per iteration, and on the LSTM step acting on that model made the real
peak worse. The attribute above is still honoured there: it is applied before
this bail-out.

Two things the rig does around that pass matter more for a step's memory. The
training-step session is compiled for the shapes it is fed, so ORT resolves every
intermediate shape at session build and folds the shape arithmetic (most of a
step's kernels, and outputs that pinned activations alive) out of the executed
graph; the encoder steps above dropped from 27.4 and 50.6 MiB to 19.5 and 37.2
before the pass touched them. And the session runs ORT's full optimization level
minus the common-subexpression pass, which would otherwise merge every
recomputation the pass emits back into the tensor it exists to free.

A reflection dump over the `Shorokoo` assembly turns up three types from that
pass: `GraphEvaluationResult`, `NodeEvaluationInfo` and
`GraphOptimizationResult`, in the namespace `Shorokoo.Core.AutoDiffCheckpointing`.
They are the pass's internal report, public only as an artefact of the assembly
layout; everything that produces or consumes them is internal, so no API you can
call returns one. Treat them as unsupported.

That warning covers this one namespace, not the `Shorokoo.Core.*` prefix. Ten
namespaces under it carry documented public API, among them
`Shorokoo.Core.Interpreter` (`QuickExecutionEngine`) and
`Shorokoo.Core.Training` (the learning-rate `Schedule` / `Schedules` API).
[orientation.md](orientation.md#public-core-namespaces) lists them.

### Quick Execution Engine value computation is bounded

The Quick Execution Engine (QEE) propagates output **dtype and shape** for every
supported operator, but materializes concrete **values** only for small tensors
(up to `MaxDataElements`, default 256 elements); larger tensors flow through as
shape/dtype only. Reducing **over** an axis of extent 0 is also bounded: each
group is empty, and five reductions (`ReduceMax`, `ReduceMin`, `ReduceMean`,
`ReduceLogSum`, `ReduceLogSumExp`) leave the value uncomputed, because their
empty-group result is backend-specific rather than fixed by the ONNX spec (see
[operator-support.md](operator-support.md#reductions)). Use the ONNX Runtime
backend (`OnnxEngine.Eval` / `ComputeContext`) for real numeric execution.

### Uniform draws resolve a bounded span of magnitudes

A uniform draw addresses `float32` values directly, but only over the top 41
weight classes of the requested range (40 when the range straddles zero): one
64-bit generator value per element cannot separate more. A class is
`max(1, exponent field)`, so it is a binade except at the bottom, where the
subnormals and the smallest normal binade share one. Below that floor the draw
still carries the probability mass its width earns, but on an even lattice rather
than the float grid, so those floats are not individually drawable; about 33% of
the floats in `[0, 1)` and 16% of the whole finite `float32` domain can come out
of a draw. Two further effects appear only at extreme ranges: a single float can
take up to twice its due share when the range's total weight is not a power of
two (it is exact when it is, `[0, 1)` and `[-1, 1)` included), and a side of the
range worth less than one weight unit (the negative side of `[-1, 1e30)`, say)
gets probability exactly zero. Resolution is bounded by the generator bits spent
per element, so a deeper draw is possible but costs more bits. See
[uniform-draws.md](uniform-draws.md) for the full contract.

### Normal draws stop at 8 sigma and resolve a bounded span of magnitudes

A normal draw spends one 64-bit generator value per element (one bit of sign, 63
for the magnitude), which buys 42 weight classes of resolved magnitudes, from
2⁻³⁹ (1.818989e-12) up to 8. Neither end is reachable by accident, but both are
hard. **Above**: every position at or past 8 sigma decodes to exactly `8.0f`, so
the tail is clipped there and that one float carries all 1.244e-15 of the mass
beyond it (about one draw in 800 trillion); a scaled draw never leaves
`mean ± 8·scale`. **Below**: magnitudes under 2⁻³⁹ ride an even lattice instead
of the float grid, so they are not individually drawable. The region still
carries exactly its due mass (1.4513e-12, about one draw in 690 billion), and the
normal density is constant to within 2⁻⁷⁸ across it, so what is lost is
resolution among numerically interchangeable values, not fairness. **Near the
top**: above 7.6008 a float's cell is worth under one position, and 577,209
magnitudes below 8 (the lowest 7.6011825) get none and never come out. In all,
720,265,872 of the 4,278,190,080 finite `float32` values (16.8%) can come out of a
draw. Both limits are set by the generator bits spent per element, so a wider
window is possible but costs more bits. See [normal-draws.md](normal-draws.md)
for the full contract, which states both magnitudes.

### ONNX `Scan` import

`Scan` cannot be imported. Shorokoo executes `Loop`, not `Scan`, and does not
rewrite one into the other, so a model containing a `Scan` node is rejected at
import. Workaround: express the iteration as an explicit `Loop` (slice each
per-iteration input inside the body with `Gather` on the iteration index, and let
the `Loop` stack its scan outputs), or re-export the model from the source
framework in that form. In Shorokoo, build the equivalent with `LoopAPI` and
`ctx.Scan`.

### ONNX `SequenceMap` import

`SequenceMap` cannot be imported. Lowering it to a `Loop` requires whole-graph
type inference: its variadic additional inputs are mapped per element when
sequence-typed but broadcast when tensor-typed, which cannot be told apart without
inferring element types, and the per-output accumulator sequences need a typed
`SequenceEmpty` seed. The ONNX Runtime backend has no `SequenceMap` kernel to fall
back on. The importer rejects the model with an error. Workaround: express the
mapping as an explicit `Loop` over `SequenceLength` using
`SequenceAt`/`SequenceInsert` (in Shorokoo, build it with `LoopAPI`); that form is
fully supported.

### C# emission of a runtime-built TensorStruct

`SaveToCSharp()`, and so the `DebugRequests` snapshots built on it, names a
TensorStruct by the `IStruct` interface it was reflected from, since that is what
the emitted source must write. A struct assembled at runtime from a
`TensorStructDef` has no such interface, so emission is refused with **FW053**.
Every other graph emits; a struct built with `Globals.TensorStruct<T>` /
`TensorStructCreate<T>`, generic interfaces included, carries its type and is
unaffected. Workaround: declare the struct as an `IStruct` interface instead of
building its definition by hand.

### ONNX opset range and export stamping

Import reads every standard-domain (`ai.onnx`) node against its opset-21 definition,
extended with the attributes opsets 22–26 add (the range of the bundled ONNX Runtime 1.26,
which pins ONNX 1.21). It ignores the opset the model declares and converts nothing, so
**models older than opset 21 are not supported**. Where an operator's signature changed
after the model's opset, the node keeps its old form and ONNX Runtime refuses the compiled
model. For example, `axes` is an attribute of `Squeeze`, `Unsqueeze` and `ReduceSum`, and
`split` of `Split`, before opset 13, and `axes` of the other reductions before opset 18;
such a model fails with `[ErrorCode:InvalidGraph] … has input size 1 not in range [min=2,
max=2]`. Convert an older model to opset 21 before importing it, for example with
`onnx.version_converter.convert_version(model, 21)` in Python.

Export stamps models at the **opset-21 baseline**, and the exporter raises each
model's stamp only as far as the graph requires.

The exporter holds a floor for each post-21 operator (`RMSNormalization` and
`RotaryEmbedding` at 23; `Attention`, `Swish` and `TensorScatter` at 24, where
`Attention` is defined at 23 but ORT 1.26's CPU provider registers its kernel
only at 24+; `BitCast` and `CumProd` at 26). Nothing you export as ONNX reaches
one of those floors today, by three routes. `Attention`, `AttentionWithKVCache`,
`RotaryEmbedding`, `BitCast` and `CumProd` throw `NotImplementedException` at
their `OnnxOp` entry points, so the node cannot be authored. `Swish` and
`RMSNormalization` lower inline to opset-21 primitives (`Mul`/`Sigmoid` and
`ReduceMean`/`Sqrt`/`Div`/`Mul`), so the node is never built. `TensorScatter` is
built and kept as itself, but the exporter decomposes it into a `Concat` of the
cache and the update and one `GatherElements` over the pair (see
[operator-support.md](operator-support.md)), so the file still stamps at 21. No
post-21 operator node is emitted from an authored graph. The floors are kept for
the ops that cannot be authored, for when a runtime registers them at a usable
opset, and are live for a saved architecture, which keeps `TensorScatter` as
authored and so stamps at 24. The raise you will see is the attribute-driven one
below, on an imported model.

The baseline stays at 21 rather than 26 because the opset stamp selects kernel
versions in ONNX Runtime, and ORT's CPU provider has gaps at the bumped versions.
For example, the opset-22 respecifications of `GlobalLpPool` and
`RandomNormalLike` only added bfloat16 to their type constraints, yet ORT 1.26's
CPU provider registers no opset-22 kernels for them: a model blanket-stamped at
opset ≥ 22 fails to load while the identical opset-21 model runs.

The lower stamp does not reduce coverage. The opset 22–26 respecifications of
existing operators only widen dtype lists (bfloat16 at 22; float4e2m1 at 23;
float8e8m0 at 24; int2/uint2 at 25; all unsupported in Shorokoo, see the dtype
section below), plus three new optional attributes that Shorokoo imports and
honors: `DequantizeLinear.output_dtype` (opset 23), `QuantizeLinear.precision`
(23), and `Cast`/`CastLike.round_mode` (24, float8e8m0-only semantics). When such
an attribute carries a non-default value the exporter raises that model's stamp
accordingly. That raise comes only from an imported model: no `Ops`/`OnnxOp`
entry point accepts any of the three attributes, so a graph built from
`Ops`/`OnnxOp` alone exports at the opset-21 baseline. (The low-level
`NodeBuilder` surface is the exception: it can stamp any attribute a node
definition declares, `precision` and `round_mode` included, and a node built that
way raises the stamp as an imported one does.) The opset-21 operator versions are
semantically complete for everything else Shorokoo can represent.

### Sub-byte and complex dtypes

`Float16` and `BFloat16` are fully supported: `.safetensors` files with
`F16`/`BF16` payloads load and save (`SafeTensorLoader`), constant
folding/conversion roundtrips through `TensorDataConversion` (float32→f16/bf16
rounds to nearest-even), and ONNX models with f16/bf16 initializers import (both
the `raw_data` and the int32-packed `int32_data` encodings) and export. The Quick
Execution Engine stores f16/bf16 *values* in float32 storage, so QEE-propagated
values do not model the precision loss; real rounding happens in the ONNX Runtime
backend and in the constant-conversion paths.

`Int4`/`UInt4` are unsupported: there is no sub-byte tensor storage, and any
attempt to materialize them raises `UnsupportedDTypeException` (error codes
`DT001`/`DT002`/`DT010`/`DT011`). The same applies to the narrow dtypes of recent
opsets: the float8 family (`Float8E4M3FN`, `Float8E4M3FNUZ`, `Float8E5M2`,
`Float8E5M2FNUZ`, plus `Float8E8M0` from opset 24), `Float4E2M1` (opset 23), and
`Int2`/`UInt2` (opset 25) are not supported as tensor element types.
`Complex64`/`Complex128` are not supported either. This concerns Shorokoo's own
tensor storage (`TensorData` and the graph DSL): the
[PyTorch](pytorch-backend.md#values) and [JAX](jax-backend.md#values) backends
hold complex and float8 tensors at the backend level (`IShorokooTensorValue`), but
a Shorokoo model cannot declare them.

### Gradient coverage

Most differentiable operators in the supported set (opset 21 plus the post-21
additions) have gradient support: either a gradient rule written for the
operator, or a registered decomposition into simpler operators that the engine
differentiates instead. The rest raise `AutoDiffNotSupportedException` with an
error code naming the op. Per-operator status is in
[operator-support.md](operator-support.md).

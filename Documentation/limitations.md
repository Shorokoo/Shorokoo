# Known limitations

This page lists the framework's known limitations, split into **permanent**
limitations (things that cannot work, with the reasoning) and **current**
limitations (things that could be lifted by future work). For per-operator
support details see [operator-support.md](operator-support.md).

## Permanent limitations

### Efficient backprop through fully dynamic convolutions

A convolution whose *kernel spatial shape* is only known at run time (i.e. the
weight tensor's shape is itself computed by the graph) cannot get an efficient
backward pass. The weight gradient of a convolution is itself a convolution
whose attributes (pads, strides, dilations — and critically the kernel extent)
must be known when the backward graph is built; with a fully dynamic kernel
there is no fixed backward graph to build. This is a structural property of
graph-mode autodiff, not a missing feature.

In practice: give convolution weights a concrete shape (the usual case — e.g.
shapes derived from `[Hyper]` values are resolved when the architecture is
concretized via `ToConcreteArchitecture`), and backprop works normally.

### Per-iteration convolution geometry in a dynamic loop

`NN.Conv` lets you compute a convolution's geometry — `pads`, `strides`,
`dilations`, `kernel_shape`, `group` — in the graph rather than writing it as a
literal, and resolves it to a static ONNX attribute when the architecture is
concretized. (Conv is currently the only operator with that overload.) A static
attribute holds one value for every execution of its node, which is fine
everywhere except one place: geometry that differs from one iteration of a loop
to the next, in a loop that stays rolled. There the whole loop is a single Conv
node, and no single attribute value is right for all of its iterations.

A loop stays rolled whenever the unroll declines it — most often because its
trip count is not a compile-time constant, though a handful of body shapes
decline too. Such a graph is refused when the architecture is concretized,
naming the geometry that varies:

```
the 'pads' geometry of 'shrk_Conv' varies per iteration of a loop that was not
unrolled, so it cannot be lowered to the static 'pads' attribute of 'Conv' ...
```

Either give the loop a compile-time-constant trip count, which normally unrolls
it — `LoopAPI.Iterate(Scalar(3L))` — so each iteration becomes its own Conv node
with its own geometry; or compute the geometry from something that does not
track the iteration: a literal, a `[Hyper]` value, or an input's shape.

Constant and input-derived geometry inside a dynamic loop is fine, and so is
geometry computed *from* a dynamic loop's result — that value is computed once,
and the Conv reading it runs once. What is refused is geometry that still
follows a loop's iteration where it is used, whether or not the value it ends up
holding happens to repeat: the check is on where the geometry comes from, since
a carry's value is not knowable at build time.

### Variables first assigned inside a loop body

A variable that is assigned inside a loop *before ever being read in that same
loop* cannot be used after the loop. Shorokoo cannot recover the variable's
initial value (needed for the zero-iteration case) and conservatively rejects
the graph. Initialize the variable explicitly inside the loop body with
`LoopAPI.Init(x)` (or read it once, e.g. `OnnxOp.Identity(x)`) before the first
assignment.

### Carrying a value computed outside the loop body

A loop hands its result back by re-tracing the node that produced the body's
value and pointing the variable at the loop's output instead. A bare assignment
of a value computed *outside* the loop creates no node in the body, so there is
nothing to re-trace — and after the loop the variable is simply that outside
value, indistinguishable from every other use of it. Shorokoo rejects that shape
rather than silently returning the body's value even when the loop ran zero
times.

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

A value the body already computes — `carry = carry + Scalar(1L)`, or anything
built from the iteration index — needs no wrapping. `LoopAPI.Carry` has
overloads for `Scalar<T>`, `Vector<T>` and `Tensor<T>`; for any other carry
type, move the assignment out of the loop.

The build reports this as the **MSG005** warning on the offending line, and
concretizing the graph raises `FW023` if it is left unfixed.

## Current limitations (could be lifted)

### A generic module called from a module that is itself called

A generic `[Module]` lowers and runs through the ordinary route, and a non-generic module may
call one — `GenericLayer.Call<float32>(x)` — and lower too. What does not yet work is putting a
third module on top: the generic call site is then left unspecialized
([#286](https://github.com/Shorokoo/Shorokoo/issues/286)). `ToConcreteArchitecture` does not
refuse it — it returns an architecture whose spliced inputs are wired to nothing, and the model
fails later at session creation with an opaque `Node input '…' is not a graph input, initializer,
or output of a previous node`. Call the generic module from the module you concretize, rather
than from one it calls.

### Carrying a value from the previous iteration

A local that holds what another local held **one iteration ago** is not recognised as a loop
carry, and every read of it is silently pinned to its value from before the loop
([#274](https://github.com/Shorokoo/Shorokoo/issues/274)):

```csharp
sum = sum + prev;   // prev is acc's value from the previous iteration — reads x every time
prev = acc;
acc  = acc + Scalar(1.0f);
```

The result is wrong rather than rejected, and every engine agrees on it. `LoopAPI.Init` does not
help — this is a different shape from the one it addresses. Carry the lagged value explicitly
(compute it inside the body from the carry itself) until this is fixed.

### Calling an outer loop's ctx.Scan from an inner loop

`ctx.Scan` on an **enclosing** loop's context, called from inside a nested loop's body, throws a
`KeyNotFoundException` naming nothing
([#275](https://github.com/Shorokoo/Shorokoo/issues/275)). Scan on the context of the loop whose
body you are in.

### Scanning inside a nested loop

A value produced by `ctx.Scan` in an inner loop can be used inside the enclosing
loop's body, but cannot be read after the enclosing loop: the enclosing loop does
not carry it out, so the model fails to build or is rejected at execution instead
of returning the stacked value
([#255](https://github.com/Shorokoo/Shorokoo/issues/255)). Consume the inner
loop's scan output inside the enclosing body, or move the scan out to the
enclosing loop.

### Backprop through dynamic loops

Reverse-mode autodiff through a `Loop` whose trip count is only known at run
time requires either recording per-iteration intermediates (a tape) or
re-executing the forward body during the backward pass. Shorokoo's graph-mode
autodiff currently supports neither, so gradients through dynamic loops are
rejected with `AutoDiffNotSupportedException`. Loops with a statically known
trip count can be unrolled (iterate with `LoopAPI.Iterate(n)` where `n` is a
compile-time constant) and then differentiate normally.

### Gradient (activation) checkpointing

Activation checkpointing is a per-module attribute: `[Module(Checkpoint = true)]`
marks every call of the module as a segment whose forward activations are
recomputed in the backward pass instead of kept, PyTorch's
`torch.utils.checkpoint` — see
[Activation checkpointing](nn-library.md#activation-checkpointing). It is the
one memory lever you reach for by hand, and it is honoured unconditionally: the
rig applies it before its own compute-versus-memory objective, and even to a
step so small that the automatic pass below would skip it. There is no finer
grain than a module, and no way to checkpoint a single tensor.

Attention has one more lever, and it is not checkpointing: passing
`queryChunks: c` to `Attention.ScaledDotProductAttention` splits the query axis
into `c` blocks, which divides the score-sized **transients** by `c` but not
what the step retains across the backward pass. What it saves is shape-dependent
and costs compute, so measure both. See
[Sizing an attention run](nn-library.md#attention-memory) for the arithmetic and
for what the quadratic term actually costs.

Building a training rig also runs an internal memory-aware pass over the lowered
training-step graph, which reorders nodes and recomputes tensors rather than
keeping them alive where that improves a combined compute-and-memory objective.
It recomputes the way gradient checkpointing does — a chain of producers back to
values that are live anyway, cloned once and shared by every gradient that reads
it, with the placement and the depth of the chain chosen by evaluating the
candidate graph — but conservatively, because it is automatic and has no opt-out:
it takes trades its objective accepts, and the attribute above is how you ask for
one it would not. Its model of memory is ONNX Runtime's own allocation plan for the
step (the order ORT actually runs, and ORT's habit of handing a dead buffer to the
next tensor of the same shape rather than returning it), so what it optimizes is
what gets allocated; the framework's own memory benchmark records the resident peak
of one training step next to the modelled one, and on the graph the pass returns the
two agree to within about 10% everywhere except a conv stack, whose im2col workspace
the model does not see, and the LSTM step, whose Loop body the model walks once. On
the graph before the pass the model runs up to 13% high on attention. Measured that
way, unoptimized to optimized: an MLP 5.5 to 4.2 MiB, a conv stack 28.6 to 24.7, a
one-layer transformer encoder 19.5 to 18.9, a two-layer one 37.3 to 35.8, dense
attention unchanged, chunked attention 5.8 to 5.0 — for at most a few percent more
kernel time. The pass leaves a graph whose backward pass runs through
a recurrent op untouched — and not a recurrent one only: any graph carrying a scope at all, a
forward `If` included, comes back with nothing but its checkpoint hints applied,
because the evaluator walks a scope body once where ORT runs it per iteration, and
on the LSTM step acting on that model made the real peak worse. The attribute above
is still honoured there: it is applied before this bail-out, being what you asked
for rather than what the objective chose.

Two things the rig does around that pass matter more than the pass itself for a
step's memory. The training-step session is compiled for the shapes it is fed, so
ORT resolves every intermediate shape at session build and folds the shape
arithmetic — most of a step's kernels, and outputs that pinned activations alive —
out of the executed graph (the encoder steps above dropped from 27.4 and 50.6 MiB to
19.5 and 37.2 before the pass touched them). And the session runs ORT's full
optimization level minus the common-subexpression pass, which would otherwise merge
every recomputation the pass emits back into the tensor it exists to free.

That pass is also where three types a reflection dump over the `Shorokoo`
assembly turns up come from — `GraphEvaluationResult`, `NodeEvaluationInfo` and
`GraphOptimizationResult`, in the namespace `Shorokoo.Core.AutoDiffCheckpointing`.
Despite the namespace name they are that pass's internal report, and they are
public only as an artefact of the assembly layout: everything that produces or
consumes them is internal, so no API you can call ever hands you one. Treat them
as unsupported and do not build on them.

### Quick Execution Engine value computation is bounded

The Quick Execution Engine (QEE) always propagates output **dtype and shape**
for every supported operator, but only materializes concrete **values** for
small tensors (up to `MaxDataElements`, default 256 elements). Larger tensors
flow through QEE as shape/dtype-only. Size is not the only bound: reducing **over** an axis
of extent 0 leaves each group empty, and five reductions — `ReduceMax`, `ReduceMin`,
`ReduceMean`, `ReduceLogSum`, `ReduceLogSumExp` — leave the value uncomputed in that case,
because their empty-group result is backend-specific rather than fixed by the ONNX spec (see
[operator-support.md](operator-support.md#reductions)). Use the ONNX Runtime backend
(`OnnxEngine.Eval` / `ComputeContext`) for real numeric execution.

### Uniform draws resolve a bounded span of magnitudes

A uniform draw addresses `float32` values directly, but only over the top 41 weight classes
of the requested range (40 when the range straddles zero) — one 64-bit generator value per
element cannot separate more than that. A class is `max(1, exponent field)`, so it is a
binade except at the bottom, where the subnormals and the smallest normal binade share one. Below that floor the draw still carries the
probability mass its width earns, but on an even lattice rather than on the float grid, so
those floats are not individually drawable; about 33% of the floats in `[0, 1)` and 16% of
the whole finite `float32` domain can come out of a draw. Two further consequences show up
only at extreme ranges: a single float can take up to twice its due share when the range's
total weight is not a power of two (it is exact when it is, `[0, 1)` and `[-1, 1)`
included), and a side of the range worth less than one weight unit — the negative side of
`[-1, 1e30)`, say — gets probability exactly zero. Resolution here is bounded by the
generator bits spent per element, so a deeper draw is possible but costs more of them. See
[uniform-draws.md](uniform-draws.md) for the full contract.

### Normal draws stop at 8 sigma and resolve a bounded span of magnitudes

A normal draw spends one 64-bit generator value per element — one bit of sign, 63 for the
magnitude — which buys 42 weight classes of resolved magnitudes, from 2⁻³⁹ (1.818989e-12) up
to 8. Neither end is reachable by accident, but both are hard. **Above**: every position at or
past 8 sigma decodes to exactly `8.0f`, so the tail is clipped there and that one float
carries all 1.244e-15 of the mass beyond it (about one draw in 800 trillion); a scaled draw
never leaves `mean ± 8·scale`. **Below**: magnitudes under 2⁻³⁹ ride an even lattice instead
of the float grid, so they are not individually drawable — the region still carries exactly
the mass it is due (1.4513e-12, about one draw in 690 billion), and the normal density is
constant to within 2⁻⁷⁸ across it, so what is lost is resolution among numerically
interchangeable values, not fairness. **Near the top**: above 7.6008 a float's cell is worth under
one position, and 577,209 magnitudes below 8 — the lowest of them 7.6011825 — get none and never
come out. In all, 720,265,872 of the 4,278,190,080 finite `float32` values — 16.8% — can come out
of a draw. Both limits are set by the generator bits spent per element, so a wider window is
possible but costs more of them. See [normal-draws.md](normal-draws.md) for the full contract,
which states both magnitudes.

### ONNX `Scan` import

`Scan` cannot be imported. Shorokoo executes `Loop`, not `Scan`, and does not
rewrite one into the other, so a model containing a `Scan` node is rejected at
import. Workaround: express the iteration as an explicit `Loop` — slice each
per-iteration input inside the body with `Gather` on the iteration index, and
let the `Loop` stack its scan outputs — or re-export the model from the source
framework with the `Scan` already expressed that way. In Shorokoo, build the
equivalent with `LoopAPI` and `ctx.Scan`.

### ONNX `SequenceMap` import

`SequenceMap` cannot be imported. Lowering it to a `Loop` requires whole-graph
type inference: its variadic additional inputs are mapped per-element when
sequence-typed but broadcast when tensor-typed — indistinguishable without
inferring the element types — and the per-output accumulator sequences need a typed
`SequenceEmpty` seed. The ONNX Runtime execution backend has no `SequenceMap`
kernel to fall back on either. The importer rejects the model with an
error. Workaround: express the mapping as an explicit `Loop` over
`SequenceLength` using `SequenceAt`/`SequenceInsert` (in Shorokoo, build it
with `LoopAPI`) — that form is fully supported.

### ONNX opset range and export stamping

Import accepts standard-domain (`ai.onnx`) models from opset 7 through
opset 26 — the range implemented by the bundled ONNX Runtime 1.26 (which pins
ONNX 1.21). Export, however, stamps models at the **opset-21 baseline**,
and the exporter auto-raises each model's opset stamp only as far as the
graph actually requires.

The exporter holds a floor for each post-21 operator (`RMSNormalization` and
`RotaryEmbedding` at 23; `Attention`, `Swish` and `TensorScatter` at 24 —
`Attention` is defined at 23, but ORT 1.26's CPU provider only registers its
kernel at 24+; `BitCast` and `CumProd` at 26). None of those floors is
reachable from the `Ops`/`OnnxOp` authoring surface today, though:
`Attention`, `AttentionWithKVCache`, `RotaryEmbedding`, `TensorScatter`,
`BitCast` and `CumProd` throw `NotImplementedException` at their `OnnxOp`
entry points, and `Swish` and `RMSNormalization` lower inline to opset-21
primitives (`Mul`/`Sigmoid` and `ReduceMean`/`Sqrt`/`Div`/`Mul`), so no
post-21 operator node is ever emitted from an authored graph. The floors are
kept as the restore point for when a runtime registers those operators at a
usable opset. In practice the raise you will see is the attribute-driven one
described below, on an imported model.

The baseline stays at 21 rather than 26: the opset stamp selects
kernel versions in ONNX Runtime, and ORT's CPU provider has gaps at the
bumped versions. For example, the opset-22 respecifications of `GlobalLpPool`
and `RandomNormalLike` only added bfloat16 to their type constraints, yet ORT
1.26's CPU provider registers no opset-22 kernels for them — a model
blanket-stamped at opset ≥ 22 fails to load even though the identical
opset-21 model runs fine.

The lower stamp does not reduce coverage: the opset 22–26
respecifications of pre-existing operators only widen dtype lists (bfloat16
at 22; float4e2m1 at 23; float8e8m0 at 24; int2/uint2 at 25 — all
unsupported in Shorokoo, see the dtype section below), plus three new
optional attributes that Shorokoo imports and honors —
`DequantizeLinear.output_dtype` (opset 23), `QuantizeLinear.precision` (23),
and `Cast`/`CastLike.round_mode` (24, float8e8m0-only semantics). When such
an attribute carries a non-default value the exporter raises that model's
stamp accordingly. That raise only ever comes from an imported model, though:
none of the three attributes is reachable from the `Ops`/`OnnxOp` authoring
surface — no entry point there accepts one — so an authored graph never
carries them, and a graph built from `Ops`/`OnnxOp` alone exports at the
opset-21 baseline. (The low-level `NodeBuilder` surface is the exception: it
can stamp any attribute a node definition declares, `precision` and
`round_mode` included, and a node built that way raises the stamp exactly as
an imported one does.) The opset-21 operator versions remain semantically
complete for everything else Shorokoo can represent.

### Sub-byte and complex dtypes

`Float16` and `BFloat16` are fully supported: `.safetensors` files with
`F16`/`BF16` payloads load and save (`SafeTensorLoader`), constant
folding/conversion roundtrips through `TensorDataConversion` (float32→f16/bf16
rounds to nearest-even), and ONNX models with f16/bf16 initializers import
(both the `raw_data` and the int32-packed `int32_data` encodings) and export.
Note that the Quick Execution Engine stores f16/bf16 *values* in float32
storage, so QEE-propagated values don't model the precision loss — real
rounding happens in the ONNX Runtime backend and in the constant-conversion
paths.

`Int4`/`UInt4` remain unsupported: there is no sub-byte tensor storage, and
any attempt to materialize them raises `UnsupportedDTypeException` (error
codes `DT001`/`DT002`/`DT010`/`DT011`). The same applies to the narrow dtypes
introduced in recent opsets: the float8 family (`Float8E4M3FN`,
`Float8E4M3FNUZ`, `Float8E5M2`, `Float8E5M2FNUZ`, plus `Float8E8M0` added at
opset 24), `Float4E2M1` (opset 23), and `Int2`/`UInt2` (opset 25) are not
supported as tensor element types. `Complex64`/`Complex128` are likewise not
supported.

### Gradient coverage

Most differentiable operators in the supported set (opset 21 plus the
post-21 additions) have registered gradients; the rest raise
`AutoDiffNotSupportedException` with an error code naming the op.
The current per-operator status is tracked in
[operator-support.md](operator-support.md).

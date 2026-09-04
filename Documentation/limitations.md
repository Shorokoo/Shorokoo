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

### Variables first assigned inside a loop body

A variable that is assigned inside a loop *before ever being read in that same
loop* cannot be used after the loop. Shorokoo cannot recover the variable's
initial value (needed for the zero-iteration case) and conservatively rejects
the graph. Initialize the variable explicitly inside the loop body with
`LoopAPI.Init(x)` (or read it once, e.g. `OnnxOp.Identity(x)`) before the first
assignment.

## Current limitations (could be lifted)

### Backprop through dynamic loops

Reverse-mode autodiff through a `Loop` whose trip count is only known at run
time requires either recording per-iteration intermediates (a tape) or
re-executing the forward body during the backward pass. Shorokoo's graph-mode
autodiff currently supports neither, so gradients through dynamic loops are
rejected with `AutoDiffNotSupportedException`. Loops with a statically known
trip count can be unrolled (iterate with `LoopAPI.Iterate(n)` where `n` is a
compile-time constant) and then differentiate normally.

### Gradient (activation) checkpointing

There is no way to ask Shorokoo to trade compute for activation memory: no
attribute, option, or API marks a module, block, or tensor for recomputation
during the backward pass. If a training step does not fit, the levers are the
usual ones — a smaller batch, a shorter sequence, or a smaller model.

Building a training rig does run an internal memory-aware pass over the lowered
training-step graph, which may reorder nodes and recompute a tensor rather than
keep it alive, but only where that improves a fixed combined compute-and-memory
metric. The pass is automatic, has no settings, and reports nothing; do not
count on it to make a step fit that otherwise would not.

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
flow through QEE as shape/dtype-only. Use the ONNX Runtime backend
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
equivalent with `LoopAPI` and `ctx.Scan`, which is fully supported.

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

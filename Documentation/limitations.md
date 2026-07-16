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

### Quick Execution Engine value computation is bounded

The Quick Execution Engine (QEE) always propagates output **dtype and shape**
for every supported operator, but only materializes concrete **values** for
small tensors (up to `MaxDataElements`, default 256 elements). Larger tensors
flow through QEE as shape/dtype-only. Use the ONNX Runtime backend
(`OnnxEngine.Eval` / `ComputeContext`) for real numeric execution.

### ONNX `Scan` import: stacked-output attributes

Imported `Scan` nodes are lowered to an equivalent `Loop` (trip count from the
scan input's shape, per-iteration slicing via `Gather`, scan outputs stacked by
the Loop). Any `scan_input_axes` and `scan_input_directions` are supported.
Because a `Loop` always stacks its scan outputs along axis 0 in iteration
order, non-zero `scan_output_axes` and reverse `scan_output_directions` are
rejected at import time with a `NotSupportedException` naming the attribute.
Workaround: export the model with default output axes/directions and apply the
`Transpose`/reversal downstream of the Scan instead. The opset-8 form of Scan
(implicit batch dimension + `sequence_lens` input) is likewise rejected;
re-export at opset 9 or later.

### ONNX `SequenceMap` import

`SequenceMap` cannot be imported. Lowering it to a `Loop` requires whole-graph
type inference: its variadic additional inputs are mapped per-element when
sequence-typed but broadcast when tensor-typed — indistinguishable at the
proto level — and the per-output accumulator sequences need a typed
`SequenceEmpty` seed. The ONNX Runtime execution backend has no `SequenceMap`
kernel to fall back on either. The importer rejects the model with an
error. Workaround: express the mapping as an explicit `Loop` over
`SequenceLength` using `SequenceAt`/`SequenceInsert` (in Shorokoo, build it
with `LoopAPI`) — that form is fully supported.

### ONNX opset range and export stamping

Import accepts standard-domain (`ai.onnx`) models from opset 7 through
opset 26 — the range implemented by the bundled ONNX Runtime 1.26 (which pins
ONNX 1.21). Export, however, stamps models at the **opset-21 baseline**,
and the exporter auto-raises each model's opset stamp only as far
as the post-21 operators actually present in the graph require (e.g. a graph
containing `Attention` is stamped opset 23; one containing `BitCast`,
opset 26).

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
stamp accordingly. The opset-21 operator versions remain semantically
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

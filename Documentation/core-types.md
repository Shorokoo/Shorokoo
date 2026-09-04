# Core types: tensors, scalars, vectors, dtypes

Related: [defining-models.md](defining-models.md) · [inference.md](inference.md)

## Facts

- Three graph-value shapes, all generic over a dtype marker `T : IVarType`:
  - `Scalar<T>` — rank 0.
  - `Vector<T>` — rank 1 (also used for dynamic shapes, e.g. `Vector<int64>`).
  - `Tensor<T>` — rank N. `Scalar<T>`, `Vector<T>`, and `Tensor<T>` are distinct
    value-struct handles, all implementing the common `IValue` interface.
- `IValue` — the base interface for any graph value *handle* (`Tensor<T>`,
  `Scalar<T>`, `Vector<T>`, and the sequence / optional / struct handles).
  User-facing code holds `IValue` handles; the framework wires them into the
  computation graph as needed.
- `Variable` — the graph-side node a handle points at, and the argument type of the
  execution entry points. It is deliberately *not* an `IValue`; see
  [`Variable` and `IValue`](#variable-and-ivalue).
- Dtype marker types (used as the generic argument): `bit` (boolean), `int8`,
  `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `float16`,
  `bfloat16`, `float32`, `float64`. Example: `Tensor<float32>`, `Scalar<int64>`,
  `Scalar<bit>`.
- `DType` is the runtime dtype descriptor (`DType.Float32`, `DType.Int64`,
  `DType.Bool`, …). Use marker types in signatures; use `DType` when working with
  runtime/untyped APIs.
- A graph value is symbolic. To get concrete numbers you must evaluate it — see
  [inference.md](inference.md).
- `TensorData` / `TensorData<T>` hold concrete (materialized) values, not graph nodes.
- `OptionalTensorData` is the concrete value of an `OptionalTensor` input:
  `OptionalTensorData.Some(tensor)` for a present value, `OptionalTensorData.None(dtype)`
  for an absent one. Both are `IData`, so they feed execution like any other input (see
  [defining-models.md](defining-models.md#omittable-parameters-defaulted-hypers--optional-inputs)).

## `Variable` and `IValue`

`Variable` (namespace `Shorokoo.Core`) is the graph value itself — the non-generic node
every handle points at, carrying the runtime dtype and rank rather than a C# type
parameter. It is what the execution entry points take: `OnnxEngine.Eval(Variable)` and
its multi-output overloads, and the same `Eval` forms on `ComputeContext`.

You rarely have to name the type, because `Tensor<T>`, `Scalar<T>` and `Vector<T>` (and
the sequence / optional / struct handles) each declare an **implicit** conversion to
`Variable`, so an op result goes straight in. The catch is that `Variable` deliberately
does **not** implement `IValue`, and those conversions live on the concrete handle
types, not on the interface — so a handle you are holding as `IValue` is not accepted
and needs an explicit `ToVariable()`:

```csharp
var y = x.Relu();                                  // Tensor<float32>
TensorData r1 = OnnxEngine.Eval(y);                // implicit Tensor<float32> → Variable

IValue handle = y;
TensorData r2 = OnnxEngine.Eval(handle.ToVariable());   // Eval(handle) would not compile
```

`Variable.ToValue()` goes the other way, returning the natural handle for the value —
`Scalar<T>` at rank 0, `Vector<T>` at rank 1, `Tensor<T>` otherwise (and
`OptionalTensor<T>` / `TensorSequence<T>` / `TensorStruct<T>` for the other structural
kinds).

## Factory helpers (`using static Shorokoo.Globals;`)

| Call | Returns | Notes |
|---|---|---|
| `Scalar(1L)` / `Scalar(0.1f)` / `Scalar(true)` | `Scalar<int64/float32/bit>` | Type inferred from literal. |
| `Scalar<float32>(x)` | `Scalar<float32>` | Explicit dtype. |
| `Vector(1L, 3L, 224L, 224L)` | `Vector<int64>` | Shape literal / 1-D vector. |
| `VectorFill(length, 0f)` | `Vector<float32>` | Fill of given length. |
| `VectorRange(start, limit, delta)` | `Vector<T>` | Numeric range. |
| `Tensor([2L,3L], v0, v1, ...)` | `Tensor<T>` | From dims + flat values. |
| `TensorData([1L,3L,2L,2L], myFloats)` | `TensorData<float32>` | Materialized data from dims + a flat `float[]`. |
| `TensorFill(shape, TensorData([1], 0f))` | `Tensor<T>` | Constant-filled tensor. |
| `Tensor<float32>.Fill(shape, TensorData(...))` | `Tensor<float32>` | Static fill on the type. |
| `RandomUniform(shape, low = 0f, high = 1f)` | `Tensor<float32>` | Random feed over the half-open `[low, high)`; all but `shape` are optional. Keyed by the model's [RNG identity](rng-configuration.md) — no per-site seed. What the draw returns: [uniform-draws.md](uniform-draws.md). |
| `RandomUniform(shape, Scalar<float32> low, Scalar<float32> high)` | `Tensor<float32>` | Same feed over a range computed **in-graph** (both bounds required). The bounds reach the draw itself, so the range is exact at any width; a graph-scalar range needs a keyed (concrete, id-bearing) model. |
| `RandomNormal(shape, mean = 0f, scale = 1f)` | `Tensor<float32>` | Random feed over N(`mean`, `scale`); all but `shape` are optional. Keyed by the model's [RNG identity](rng-configuration.md) — no per-site seed. What the draw returns: [normal-draws.md](normal-draws.md). |
| `RandomNormal(shape, Scalar<float32> mean, Scalar<float32> scale)` | `Tensor<float32>` | Same feed over a distribution computed **in-graph** (both required). The parameters reach the draw itself, so one built model can be re-parameterized per run; a graph-scalar distribution needs a keyed (concrete, id-bearing) model. |

**Implicit primitive → `Scalar<T>` conversion.** Wherever a `Scalar<T>` is expected, a bare
primitive value converts to one automatically, so the `Scalar(...)` wrapper is usually
optional — e.g. `Scalar<int64> n = 32;` or `myScalar.Clip(0f, 6f)`. The element type comes
from the **target context**, not the literal: `Scalar<float32> x = 5;` builds a `float32`
scalar. Reach for the explicit `Scalar(...)` / `Scalar<T>(...)` helpers when there is no
`Scalar<T>` target to infer from — e.g. `var x = Scalar(1L);`, since a bare `var x = 1L;`
is a plain `long`, not a scalar.

**`PrimitiveParam`** (namespace `Shorokoo.Core`) is what carries that convention into
method signatures. It is a one-value box with an implicit conversion *from* every supported
C# primitive (`bool`, the integer types, `float`, `double`, `Float16`, `BFloat16`) and *on
to* `Scalar<T>` / `Tensor<T>`, so a parameter typed as it accepts a literal of any of them
and converts the value to the receiver's element type. You never name it or construct one —
it shows up only when you read a signature, e.g. `Tensor<T>.Clip(PrimitiveParam min,
PrimitiveParam max)` and the mixed operand operators (`Tensor<T> + PrimitiveParam`).
`Tensor<T>` carries that `Clip` overload *and* `Clip(Scalar<T>, Scalar<T>)`, while
`Scalar<T>` and `Vector<T>` carry only the latter; the difference is in the overload set,
not in what you can write, since the direct primitive → `Scalar<T>` conversions above
already cover the literal case — `x.Clip(0f, 6f)` compiles on all three. Reach for the
`Scalar<T>` form when a bound is computed in-graph rather than being a constant.

First-argument convention for `Tensor(...)` / `TensorData(...)`: the first argument is
the **shape (dims)**. Pass a collection literal (`[1]`, `[1L,3L,224L,224L]`) for the
`long[]` overload, or a bare `long` (e.g. `1`) for the 1-D convenience overload. The
remaining arguments are the flat element values (`params T[]`), so you can pass an
existing array directly: `TensorData([1L,3L,224L,224L], myPixelArray)`. A **rank-0**
(scalar) value takes the empty dims literal — `TensorData([], 0.01f)`, one element and no
dimensions. That is the shape a scalar graph input wants, e.g. the value handed to
[`Specialize`](inference.md#hardcoding-hypers-with-specialize) for a scalar `[Hyper]`.
`TensorData([1], 0.01f)` is not the same thing: it is rank 1 with a single element.

## Operators and fluent methods on `Tensor<T>`

- Arithmetic: `+ - * / % ^ & | << >>`, unary `-`, logical `!`.
- Comparisons return `Tensor<bit>`: `> >= < <= == !=`.
- Shape ops: `.Reshape(shape, keepAxes)` (see below), `.Transpose(dims...)`, `.Squeeze(axes)`,
  `.Unsqueeze(axis)`, `.Expand(shape)`, `.Flatten(axis)`, `.Concat(axis, others...)`,
  `.Slice(start, end, axes, steps)`, `.Pad(mode, pads, val)`, `.Tile(repeats)`.
- Indexing: `.Gather(indices, axis)` and `.GatherND(indices, batchDims)` — both default to
  ONNX's 0, so `table.Gather(tokens)` gathers rows.
- Math/activations: `.Relu()`, `.Sigmoid()`, `.Tanh()`, `.Softmax(axis)`, `.Gelu()`,
  `.Sqrt()`, `.Exp()`, `.Ln()`, `.Abs()`, trig (`.Sin()`, `.Cos()`, …).
- Linear algebra: `.MatMul(other)`.
- Reductions: `.Reduce(ReduceKind.Sum | Prod | Mean | Max | Min, axes, keepDims)`,
  `.ArgMax(axis)`, `.ArgMin(axis)`, `.TopK(k, axis)`.
- Casts: `.Cast<V>()`.
- Shape introspection (returns graph values): `.TShape`, `.ShapeTensor(start, end)`,
  `.DimTensor(axis)`, `.SizeTensor(...)`, `.TRank`.

### Mixing shapes in one operator

The arithmetic and comparison operators are declared for every pairing of the three value
shapes plus a bare primitive literal, so `Tensor<T> + Scalar<T>`, `Scalar<T> * Vector<T>`
and `Vector<T> - 1f` all exist. The result takes the wider of the two operand shapes:

| left ⊕ right | `Tensor<T>` | `Vector<T>` | `Scalar<T>` | literal |
|---|---|---|---|---|
| **`Tensor<T>`** | `Tensor<T>` | `Tensor<T>` | `Tensor<T>` | `Tensor<T>` |
| **`Vector<T>`** | `Tensor<T>` | `Vector<T>` | `Vector<T>` | `Vector<T>` |
| **`Scalar<T>`** | `Tensor<T>` | `Vector<T>` | `Scalar<T>` | `Scalar<T>` |
| **literal** | `Tensor<T>` | `Vector<T>` | `Scalar<T>` | — |

Comparisons follow the same table with the element type replaced by `bit` — `Tensor<T> >
Scalar<T>` gives `Tensor<bit>`, `Scalar<T> <= Vector<T>` gives `Vector<bit>`. Both operands
must share one `T`; there is no mixed-dtype form (see [Anti-patterns](#anti-patterns)).

The two shift operators are the exception: C# draws shift candidates from the **left**
operand's type alone, so the left operand can never be a bare literal and the right operand
must be no wider than the left. `Tensor<T> << Vector<T>` and `Vector<T> << 1L` exist;
`Vector<T> << Tensor<T>` and `1L << Tensor<T>` do not.

### Reductions and `keepDims`

`x.Reduce(kind, axes, keepDims)` **drops** the reduced dimensions by default, as in PyTorch
and NumPy — so `x.Reduce(ReduceKind.Mean).Scalar()` reduces over every axis to a rank-0
scalar, and `x.Reduce(ReduceKind.Sum, Vector(1L))` turns `[N, C]` into `[N]`. Pass
`keepDims: true` to keep them as length-1 axes instead (`[N, 1]`), which is what you want
when the result has to broadcast back against the input.

Note that ONNX itself defaults the other way: its `keepdims` attribute is `1`, so a
`Reduce*` node with the attribute omitted keeps the reduced dimensions. The fluent `.Reduce`
follows the eager frameworks instead, the same choice it makes for `Reshape` below, and always
emits the attribute explicitly. (The lower-level `NN.Reduce` takes a `bool?` with no default,
where `null` omits the attribute and so keeps ONNX's reading.)

**This default changed.** It was previously `true`. Code that omits `keepDims` now gets the
reduced dimensions dropped, with no compile error to flag it — so a reduction whose result is
broadcast back against its own input needs an explicit `keepDims: true`. Usually the shapes
stop matching and you get an error, but where the remaining dimensions happen to agree
(`[N, C]` with `N == C`, common in attention and square hidden dims) it broadcasts along the
wrong axis and silently computes the wrong numbers.

### `Reshape` and copying dimensions from the input

`x.Reshape(newShape)` follows the conventions you know from PyTorch, TensorFlow, and
NumPy: at most one `-1` entry means "infer this dimension from the element count," and a
`0` entry is a **literal zero-sized dimension**. This is worth calling out because raw
ONNX `Reshape` (with its default `allowzero=0`) disagrees: there a `0` means "copy the
dimension at this position from the input tensor" — a convention ONNX inherited from
Caffe that trips up users arriving from the eager frameworks.

Shorokoo exposes the copy-dim behavior through the explicit `keepAxes` parameter
instead: list the **output positions** whose dimensions should be copied from the input,
and omit those entries from `newShape`. The classic batch-preserving flatten of
`x : [N, C, H, W]` — ONNX `Reshape(x, [0, -1])` — is spelled:

```csharp
x.Reshape([Scalar(-1L)], keepAxes: [0])   // → [N, C·H·W]; N need not be known at build time
```

`keepAxes: [0, 1]` with `newShape = [-1]` similarly yields `[N, C, H·W]`, and so on. The
kept dimensions are resolved at run time by ONNX Runtime, so they work even when the
input's dimensions are unknown while the graph is being built (no `.DimTensor(...)`
plumbing needed).

Lowering: `Reshape` always emits an ONNX `Reshape` node. Without `keepAxes` the node
carries `allowzero=1`, matching the PyTorch reading of `0`; with `keepAxes` it carries
`allowzero=0` and a shape input with `0` at each kept position. Note that ONNX rejects
combining `-1` with a literal `0` under `allowzero=1`, so a zero-sized dimension and an
inferred dimension cannot appear in the same plain `Reshape` call — but `-1` combines
freely with `keepAxes`.

## Higher-level ops (`using static Shorokoo.NN;`)

`NN` holds ops that don't read as instance methods. Signatures for common ones (the
`NN` class has the full list):

```csharp
Tensor<T> Conv<T>(Tensor<T> x, Tensor<T> w, Vector<T> b, AutoPad autoPad,
                  long[] dilations, long group, long[] kernelShape,
                  long[] pads, long[] strides);
Tensor<T> MaxPool<T>(Tensor<T> x, bool ceilMode, long[] dilations, long[] kernelShape,
                     long[] pads, long storageOrder, long[] strides,
                     AutoPad autoPad = AutoPad.NotSet);
Tensor<T> GlobalAveragePool<T>(Tensor<T> input);
Tensor<T> GroupNormalization<T>(Tensor<T> x, Tensor<T> scale, Tensor<T> bias,
                                long numGroups, long stashType = 1L,
                                float epsilon = 1e-05f);
```

Note `numGroups`/`epsilon` here are plain C# `long`/`float` op attributes (not
`Scalar<...>`). Enums used by these ops: `AutoPad`, `PadMode`, `ReduceKind`,
`RoundMode`.

## Example

```csharp
using Shorokoo;
using static Shorokoo.Globals;
using static Shorokoo.NN;

var x = TensorFill(Vector(1L, 3L, 224L, 224L), TensorData([1], 0.1f)); // [1,3,224,224]
var w = RandomNormal(Vector(64L, 3L, 7L, 7L));
var b = VectorFill(64L, 0f);

var y = Conv(x, w, b, AutoPad.NotSet,
             dilations: [1L, 1L], group: 1L,
             kernelShape: [7L, 7L], pads: [3L, 3L, 3L, 3L], strides: [2L, 2L]);
var activated = y.Relu();
```

## Reading concrete values out of a result

Execution returns `TensorData` (see [inference.md](inference.md)). To read the numbers,
cast to the typed `TensorData<T>` and call `AccessMemory()`, which returns a
`ReadOnlySpan<primitive>`:

```csharp
TensorData result = OnnxEngine.Eval(y);
ReadOnlySpan<float> values = ((TensorData<float32>)result).AccessMemory();
float first = values[0];
```

`AccessMemory()` maps each dtype marker to its CLR primitive: `float32`→`float`,
`float64`→`double`, `int64`→`long`, `int32`→`int`, `bit`→`bool`, `float16`→`Float16`,
`bfloat16`→`BFloat16`, etc. A boxed `TensorData.Data` (`object[]`) also exists; prefer
`AccessMemory()`.

## Anti-patterns

- Do not mix dtypes in one op (e.g. add `Tensor<float32>` to `Tensor<int64>`); cast
  first with `.Cast<float32>()`.
- Do not assume `.TShape` gives compile-time dimensions — it is a graph value
  (`Vector<int64>`) resolved at evaluation, not a C# array.
- Do not call `new Tensor<T>(...)` directly; use the `Globals` factories or op results.

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
- `TensorData` / `TensorData<T>` hold concrete (materialized) values, not graph nodes —
  what you feed a run, and what a run gives back.
- `TensorAttribute` holds concrete values too, but the ones written into a graph's own
  *description*: a `Constant`'s value, a `ConstantOfShape`'s fill, a trainable parameter's
  weights. It is immutable, belongs to no compute context and is not disposable.
  `TensorData.MoveToAttribute()` and `TensorAttribute.CopyToTensorData()` convert between the
  two, and the first **spends** its source — see
  [Two kinds of concrete tensor](#two-kinds-of-concrete-tensor-tensordata-and-tensorattribute).
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
| `TensorFill(shape, 0f)` | `Tensor<T>` | Constant-filled tensor. |
| `Tensor<float32>.Fill(shape, TensorData(...).MoveToAttribute())` | `Tensor<float32>` | Static fill on the type. The fill value is written into the graph, so it is a [`TensorAttribute`](#two-kinds-of-concrete-tensor-tensordata-and-tensorattribute) — and `MoveToAttribute()` spends the `TensorData` it is taken from. |
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

var x = TensorFill(Vector(1L, 3L, 224L, 224L), 0.1f); // [1,3,224,224]
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
GC.KeepAlive(result);   // see "What a TensorData holds" below — the span is a window, not a copy
```

`AccessMemory()` maps each dtype marker to its CLR primitive: `float32`→`float`,
`float64`→`double`, `int64`→`long`, `int32`→`int`, `bit`→`bool`, `float16`→`Float16`,
`bfloat16`→`BFloat16`, etc. A boxed `TensorData.Data` (`object[]`) also exists; prefer
`AccessMemory()`.

## Two kinds of concrete tensor: `TensorData` and `TensorAttribute`

Concrete numbers reach Shorokoo in two roles, and each role has its own type.

| | `TensorData` | `TensorAttribute` |
|---|---|---|
| What it is | a **run's** data: an input you feed, an output you read, training state | a **graph's** data: a literal written into the description itself |
| Where the bytes are | in one compute context's memory — the host, or a particular card | nowhere in particular: a shape, a dtype and the elements |
| Lifetime | a handle on an allocation; `IDisposable` | none — immutable, shared, not disposable |
| Where you meet it | `Eval`, `Execute`, `Run`, `TrainStep`, a checkpoint's tensors | `Constant`, `ConstantOfShape`, a trainable parameter's initial value |

The split is not bookkeeping. A description has to mean the same thing everywhere: a graph you
build here, export to `.onnx`, and read back on a machine with a different card must be the
same graph. A `TensorData` cannot promise that — it names memory that belongs to one compute
context, so a graph holding one could only be built where that context is — and it has a
lifetime, which a description does not. Nothing frees a literal, and a graph whose constant
could be disposed out from under it is not a description of anything.

So the slots that take a tensor-valued *operator attribute* take a `TensorAttribute`:

- `OnnxOp.Constant(value)` — a graph constant;
- `OnnxOp.ConstantOfShape(shape, value)`, and `Tensor<T>.Fill(shape, value)` on top of it —
  the fill value;
- `Globals.TrainableTensor(value, name)` — a trainable parameter's value, which is also where
  a checkpoint's weights land when they are bound into a model.

Everything else that takes concrete numbers — feeding a run, reading a result, a training
batch, a `TrainingCheckpoint`'s state — still takes and returns `TensorData`.

### Building one

Most of the time you never name the type. `Scalar(1L)`, `Vector(1L, 2L)`,
`Tensor([2L, 2L], …)`, `TensorFill(shape, 0f)` and `VectorFill(64L, 0f)` build their literal
for you and hand back a graph value. You name it when you are holding the numbers already:

```csharp
using System.Runtime.InteropServices;   // MemoryMarshal, for the second form

// From a TensorData you are finished with — the usual case.
TensorAttribute weights = TensorData([64L, 3L, 7L, 7L], myFloats).MoveToAttribute();

// Or straight from the bytes, with no TensorData in between.
TensorAttribute fill = TensorAttribute.Create(
    new Shape(1L), DType.Float32, MemoryMarshal.AsBytes<float>([0.1f]));

Variable w = Globals.TrainableTensor(weights, "conv1.weight");
```

`Shape` is a class rather than a collection type, so the shape argument is `new Shape(…)` or a
`long[]` — a bare `[1L]` collection literal does not convert to it.

An attribute answers `Shape`, `DType`, `HasValues`, `Bytes` (or `Values` for `DType.String`),
`Elements<V>()` and `CopyToTensorData()`, and that is the whole of it: there is no `Dispose`, no
`Context`, no `Space`. `HasValues` is false for one case only — a model definition saved
*without* its weights, whose parameter slots keep dtype and shape and no elements until a
checkpoint is bound back onto them. Reading the elements of one of those throws, and says so.

### The two conversions, and which one spends its source

| | Costs | Afterwards |
|---|---|---|
| `TensorData.MoveToAttribute()` | nothing, where the tensor holds its own array and is the only handle on it; a copy otherwise | **the tensor is spent**: it is disposed, and reading it throws `ObjectDisposedException` |
| `TensorAttribute.CopyToTensorData()` | a copy, always | both usable; the attribute is unchanged, and the copy is on `ComputeContext.Host` |

The asymmetry is about size. Binding a checkpoint's weights into a graph is the direction that
runs hot — a 165 M-parameter model is some 660 MB — so it moves, and moving means the source is
gone. The other direction copies because an attribute is immutable and shared by every graph
that captured it: a writable tensor over the same bytes would be a way to edit a description
through the back door.

That is also why the move falls back to a copy wherever handing the array over would leave
somebody else able to write it, or wherever there is no array to hand over in the first place:

- **A second handle names the same bytes.** `GiveAccessTo` hands out another handle, and
  surrendering yours says nothing about that one — it could still write through
  `AccessModifiableMemory`, and the description the graph captured would change under it. An
  attribute taken while you are the only handle cannot be written by anyone.
- **The elements are a runtime value's, or strings.** There is no managed array to give: the
  bytes are a runtime's own buffer, or a string tensor's variable-length elements, and only a
  copy gets them out.

The case the no-copy path exists for — a checkpoint's tensor, parsed for the bind and named by
nothing else — is the sole-handle case, so the size argument above is untouched.

`MoveToAttribute()` refuses a tensor attached to a compute context, and says which call fixes
it. A result that came back from `Execute` on a context of your own belongs to that context,
so send it home first:

```csharp
TensorData result = compiled.Execute(input)[0].ToTensorData();
Variable literal = Globals.Tensor(result.Detach().MoveToAttribute());
```

`OnnxEngine.Eval`, `ComputeContext.Eval` and `ComputeContext.Default` already hand their
results back on `ComputeContext.Host`, so those need no `Detach()` — see
[Moving data between contexts](inference.md#moving-data-between-contexts).

### Changing code that passed a `TensorData` as an attribute

The slots listed above used to take a `TensorData`. Code that passed one does not compile any
more, and the fix is a `.MoveToAttribute()` at the call site:

```csharp
// before
Tensor<float32>.Fill(shape, Globals.TensorData(1, 1.0f));
Globals.TrainableTensor(myWeights, "w");

// after
Tensor<float32>.Fill(shape, Globals.TensorData(1, 1.0f).MoveToAttribute());
Globals.TrainableTensor(myWeights.MoveToAttribute(), "w");     // myWeights is spent
```

One trap, from the move:

- **A literal you meant to reuse has to be built twice, or converted once.** `MoveToAttribute()`
  spends its tensor, so one `TensorData` cannot serve two call sites. Convert once and pass the
  `TensorAttribute` to both — an attribute *is* shareable, being immutable — or build a fresh
  `TensorData` per site.

A factory that takes a `TensorData` for you does **not** spend it. `Globals.TensorFill<T>(shape,
TensorData<T>)` copies, so the tensor you passed is still yours and may be passed again:

```csharp
var fill = (TensorData<float32>)TensorData([1L], 0.5f);
var a = TensorFill((Vector<int64>)[Scalar(2L)], fill);
var b = TensorFill((Vector<int64>)[Scalar(3L)], fill);   // fine: fill is untouched
```

A fill value is one element, so the copy costs nothing. Spending a tensor is something you ask
for by name, never something a factory does to an argument you handed it.

## What a `TensorData` holds, and when its values go away

A `TensorData` is a **handle** on an allocation, not the allocation itself. More than one
handle can name the same bytes, and the allocation is released when the last one lets go — so
disposing a tensor means letting go of your own name for the memory, never pulling it away from
something else that is reading it. The whole model — handles, what a run holds while it runs,
and the two calls that do mean "free these bytes now" — is in
[A tensor's lifetime](inference.md#a-tensors-lifetime-handles-locks-and-deletion). What matters
while you are reading a result is this:

- **Disposing is optional.** A tensor you simply drop is reclaimed like any other object, and
  nothing in the framework hands you a tensor you are obliged to dispose. Dispose when you want
  the memory back at a known moment — a long loop that produces large tensors is the case that
  motivates it.
- **Disposing is about this handle.** It is idempotent, it leaves every other handle on the
  same bytes reading, and it frees nothing while a run is still reading them.
- **The disposed handle stops reading.** `AccessMemory()`, `AccessRawMemory()`, `.Data` and
  `.DebugData` throw `ObjectDisposedException` rather than reading freed memory. `.Shape`,
  `.DType`, `.ToString()` and `.IsDisposed` keep working, so a disposed tensor can still say
  what it was.

Operations that build one tensor from another copy, so the source keeps its values — unless
they say otherwise in so many words. Three say otherwise, and each **spends** the tensor it is
called on: `MoveToAttribute()`
([above](#the-two-conversions-and-which-one-spends-its-source)), `Donate()`, and `TransferTo`
across memory spaces — the last two in
[inference.md](inference.md#feeding-a-large-input-without-a-second-copy).
`TensorDataSequence.Create(...)` copies the tensors you pass it, and disposing the sequence
releases only the sequence's own copies.

**A span is a window, not a copy.** `AccessMemory()` and `AccessRawMemory()` point straight
into the allocation, and nothing ties the span's lifetime to the tensor's. A span outlives the
bytes it points at if you let go of the last handle on them — by disposing the tensor, and also
by letting it simply become unreachable while you are still reading.

That second one catches people out, because being *in scope* is not the same as being
*reachable*: the runtime retires a local at its last read, and taking the span **is** the
tensor's last read. So copying out of the span does not by itself make you safe — the copy
happens after the tensor is already collectable, and allocating the array is exactly the sort
of thing that triggers a collection:

```csharp
TensorData result = OnnxEngine.Eval(y);
float[] values = ((TensorData<float32>)result).AccessMemory().ToArray();  // NOT safe
```

Keep the tensor alive across the read instead — with `GC.KeepAlive` after it, or by reading
through something that outlives the span (a field, a collection, a later use of the tensor):

```csharp
TensorData result = OnnxEngine.Eval(y);
float[] values = ((TensorData<float32>)result).AccessMemory().ToArray();
GC.KeepAlive(result);                                                     // safe
```

Or reach for the copying accessors, which do both for you — `CopyMemory<V>()` and
`CopyRawMemory()` return an array the caller owns, `ValueAt<V>(int)` reads one element, and each
keeps the tensor alive across the read. Where the whole buffer is being copied anyway, they are
the shorter and safer form.

### A tensor whose values are not on the host

Disposal is not the only reason a tensor's elements cannot be read. A tensor produced by a
[resident training run](training.md#keeping-training-state-on-the-device) is left in the execution
provider's own memory, where a host read would dereference a device address. `IsHostResident` says
which it is, and the accessors throw `InvalidOperationException` rather than reading it — naming
`StepToCheckpoint`, which is what brings that state home. A tensor from any other route is
host-resident, so this only arises for a run that asked for residency.

## Anti-patterns

- Do not mix dtypes in one op (e.g. add `Tensor<float32>` to `Tensor<int64>`); cast
  first with `.Cast<float32>()`.
- Do not assume `.TShape` gives compile-time dimensions — it is a graph value
  (`Vector<int64>`) resolved at evaluation, not a C# array.
- Do not call `new Tensor<T>(...)` directly; use the `Globals` factories or op results.

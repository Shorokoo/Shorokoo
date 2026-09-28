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
  User code holds `IValue` handles; the framework wires them into the
  computation graph.
- `Variable` — the graph-side node a handle points at, and the argument type of the
  execution entry points. It is *not* an `IValue`; see
  [`Variable` and `IValue`](#variable-and-ivalue).
- Dtype marker types (used as the generic argument): `bit` (boolean), `int8`,
  `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `float16`,
  `bfloat16`, `float32`, `float64`. Example: `Tensor<float32>`, `Scalar<int64>`,
  `Scalar<bit>`.
- `DType` is the runtime dtype descriptor (`DType.Float32`, `DType.Int64`,
  `DType.Bool`, …). Use marker types in signatures and `DType` with
  runtime/untyped APIs.
- A graph value is symbolic. To get numbers, evaluate it; see
  [inference.md](inference.md).
- `TensorData` / `TensorData<T>` hold concrete (materialized) values, not graph nodes:
  what you feed a run, and what a run returns.
- `TensorAttribute` holds concrete values written into a graph's own *description*: a
  `Constant`'s value, a `ConstantOfShape`'s fill, a trainable parameter's weights. It is
  immutable, belongs to no compute context and is not disposable.
  `TensorData.MoveToAttribute()` and `TensorAttribute.CopyToTensorData()` convert between the
  two, and the first **spends** its source; see
  [Two kinds of concrete tensor](#two-kinds-of-concrete-tensor-tensordata-and-tensorattribute).
- `OptionalTensorData` is the concrete value of an `OptionalTensor` input:
  `OptionalTensorData.Some(tensor)` for a present value, `OptionalTensorData.None(dtype)`
  for an absent one. Both are `IData`, so they feed execution like any other input (see
  [defining-models.md](defining-models.md#omittable-parameters-defaulted-hypers--optional-inputs)).

## `Variable` and `IValue`

`Variable` (namespace `Shorokoo.Core`) is the graph value itself: the non-generic node
every handle points at, carrying the runtime dtype and rank instead of a C# type
parameter. The execution entry points take it: `OnnxEngine.Eval(Variable)` and its
multi-output overloads, and the same `Eval` forms on `ComputeContext`.

You rarely name the type, because `Tensor<T>`, `Scalar<T>` and `Vector<T>` (and the
sequence / optional / struct handles) each declare an **implicit** conversion to
`Variable`, so an op result goes straight in. But `Variable` does **not** implement
`IValue`, and the conversions live on the concrete handle types, not on the interface, so
a handle held as `IValue` needs an explicit `ToVariable()`:

```csharp
var y = x.Relu();                                  // Tensor<float32>
TensorData r1 = OnnxEngine.Eval(y);                // implicit Tensor<float32> → Variable

IValue handle = y;
TensorData r2 = OnnxEngine.Eval(handle.ToVariable());   // Eval(handle) would not compile
```

`Variable.ToValue()` goes the other way, returning the natural handle for the value:
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
| `Tensor<float32>.Fill(shape, TensorData(...).MoveToAttribute())` | `Tensor<float32>` | Static fill on the type. The fill value is written into the graph, so it is a [`TensorAttribute`](#two-kinds-of-concrete-tensor-tensordata-and-tensorattribute); `MoveToAttribute()` spends the `TensorData` it is taken from. |
| `RandomUniform(shape, low = 0f, high = 1f)` | `Tensor<float32>` | Random feed over the half-open `[low, high)`; all but `shape` are optional. Keyed by the model's [RNG identity](rng-configuration.md); no per-site seed. What the draw returns: [uniform-draws.md](uniform-draws.md). |
| `RandomUniform(shape, Scalar<float32> low, Scalar<float32> high)` | `Tensor<float32>` | Same feed over a range computed **in-graph** (both bounds required). The bounds reach the draw itself, so the range is exact at any width; a graph-scalar range needs a keyed (concrete, id-bearing) model. |
| `RandomNormal(shape, mean = 0f, scale = 1f)` | `Tensor<float32>` | Random feed over N(`mean`, `scale`); all but `shape` are optional. Keyed by the model's [RNG identity](rng-configuration.md); no per-site seed. What the draw returns: [normal-draws.md](normal-draws.md). |
| `RandomNormal(shape, Scalar<float32> mean, Scalar<float32> scale)` | `Tensor<float32>` | Same feed over a distribution computed **in-graph** (both required). The parameters reach the draw itself, so one built model can be re-parameterized per run; a graph-scalar distribution needs a keyed (concrete, id-bearing) model. |

**Implicit primitive → `Scalar<T>` conversion.** Wherever a `Scalar<T>` is expected, a bare
primitive converts to one, so the `Scalar(...)` wrapper is usually optional, e.g.
`Scalar<int64> n = 32;` or `myScalar.Clip(0f, 6f)`. The element type comes from the
**target**, not the literal: `Scalar<float32> x = 5;` builds a `float32` scalar. Use the
explicit `Scalar(...)` / `Scalar<T>(...)` helpers when there is no `Scalar<T>` target to
infer from, e.g. `var x = Scalar(1L);`, since `var x = 1L;` is a plain `long`.

**`PrimitiveParam`** (namespace `Shorokoo.Core`) carries that convention into method
signatures. It is a one-value box with an implicit conversion *from* every supported C#
primitive (`bool`, the integer types, `float`, `double`, `Float16`, `BFloat16`) and *on to*
`Scalar<T>` / `Tensor<T>`, so a parameter of that type accepts a literal of any of them and
converts it to the receiver's element type. You never construct one; it appears in
signatures such as `Tensor<T>.Clip(PrimitiveParam min, PrimitiveParam max)` and the mixed
operand operators (`Tensor<T> + PrimitiveParam`). `Tensor<T>` has that `Clip` overload *and*
`Clip(Scalar<T>, Scalar<T>)`, while `Scalar<T>` and `Vector<T>` have only the latter. What
you can write is the same, since the primitive → `Scalar<T>` conversion covers literals:
`x.Clip(0f, 6f)` compiles on all three. Use the `Scalar<T>` form when a bound is computed
in-graph.

First-argument convention for `Tensor(...)` / `TensorData(...)`: the first argument is
the **shape (dims)**. Pass a collection literal (`[1]`, `[1L,3L,224L,224L]`) for the
`long[]` overload, or a bare `long` (e.g. `1`) for the 1-D convenience overload. The
remaining arguments are the flat element values (`params T[]`), so you can pass an
existing array: `TensorData([1L,3L,224L,224L], myPixelArray)`. A **rank-0** (scalar) value
takes the empty dims literal: `TensorData([], 0.01f)`, one element and no dimensions. That
is the shape a scalar graph input wants, e.g. the value passed to
[`Specialize`](inference.md#hardcoding-hypers-with-specialize) for a scalar `[Hyper]`.
`TensorData([1], 0.01f)` is different: rank 1 with a single element.

## Operators and fluent methods on `Tensor<T>`

- Arithmetic: `+ - * / % ^ & | << >>`, unary `-`, logical `!`.
- Comparisons return `Tensor<bit>`: `> >= < <= == !=`.
- Shape ops: `.Reshape(shape, keepAxes)` (see below), `.Transpose(dims...)`, `.Squeeze(axes)`,
  `.Unsqueeze(axis)`, `.Expand(shape)`, `.Flatten(axis)`, `.Concat(axis, others...)`,
  `.Slice(start, end, axes, steps)`, `.Pad(mode, pads, val)`, `.Tile(repeats)`.
- Indexing: `.Gather(indices, axis)` and `.GatherND(indices, batchDims)`, both defaulting to
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

Comparisons follow the same table with the element type replaced by `bit`: `Tensor<T> >
Scalar<T>` gives `Tensor<bit>`, `Scalar<T> <= Vector<T>` gives `Vector<bit>`. Both operands
must share one `T`; there is no mixed-dtype form (see [Anti-patterns](#anti-patterns)).

The shift operators are the exception: C# draws shift candidates from the **left**
operand's type alone, so the left operand cannot be a bare literal and the right operand
must be no wider than the left. `Tensor<T> << Vector<T>` and `Vector<T> << 1L` exist;
`Vector<T> << Tensor<T>` and `1L << Tensor<T>` do not.

### Reductions and `keepDims`

`x.Reduce(kind, axes, keepDims)` **drops** the reduced dimensions by default, as in PyTorch
and NumPy: `x.Reduce(ReduceKind.Mean).Scalar()` reduces over every axis to a rank-0
scalar, and `x.Reduce(ReduceKind.Sum, Vector(1L))` turns `[N, C]` into `[N]`. Pass
`keepDims: true` to keep them as length-1 axes (`[N, 1]`), which you need when the result
must broadcast back against the input.

ONNX defaults the other way: its `keepdims` attribute is `1`, so a `Reduce*` node with the
attribute omitted keeps the reduced dimensions. The fluent `.Reduce` follows the eager
frameworks, as `Reshape` does below, and always emits the attribute. (The lower-level
`NN.Reduce` takes a `bool?` with no default, where `null` omits the attribute and so keeps
ONNX's reading.)

A reduction whose result is broadcast back against its own input needs an explicit
`keepDims: true`. Without it the shapes usually stop matching and you get an error, but
where the remaining dimensions happen to agree (`[N, C]` with `N == C`, common in attention
and square hidden dims) it broadcasts along the wrong axis and silently computes the wrong
numbers.

### `Reshape` and copying dimensions from the input

`x.Reshape(newShape)` follows PyTorch, TensorFlow and NumPy: at most one `-1` entry means
"infer this dimension from the element count," and a `0` entry is a **literal zero-sized
dimension**. Raw ONNX `Reshape` (with its default `allowzero=0`) differs: there a `0` means
"copy the dimension at this position from the input tensor," a convention inherited from
Caffe.

Shorokoo exposes copy-dim through the explicit `keepAxes` parameter: list the **output
positions** whose dimensions are copied from the input, and omit those entries from
`newShape`. The batch-preserving flatten of `x : [N, C, H, W]` (ONNX
`Reshape(x, [0, -1])`) is:

```csharp
x.Reshape([Scalar(-1L)], keepAxes: [0])   // → [N, C·H·W]; N need not be known at build time
```

`keepAxes: [0, 1]` with `newShape = [-1]` yields `[N, C, H·W]`, and so on. ONNX Runtime
resolves the kept dimensions at run time, so they work when the input's dimensions are
unknown at build time (no `.DimTensor(...)` plumbing needed).

Lowering: `Reshape` always emits an ONNX `Reshape` node. Without `keepAxes` the node
carries `allowzero=1`, matching the PyTorch reading of `0`; with `keepAxes` it carries
`allowzero=0` and a shape input with `0` at each kept position. ONNX rejects combining
`-1` with a literal `0` under `allowzero=1`, so a zero-sized dimension and an inferred
dimension cannot appear in the same plain `Reshape` call, but `-1` combines freely with
`keepAxes`.

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

`numGroups`/`epsilon` here are plain C# `long`/`float` op attributes, not
`Scalar<...>`. Enums used by these ops: `AutoPad`, `PadMode`, `ReduceKind`,
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

Execution returns `TensorData` (see [inference.md](inference.md)), and so do a run's
outputs and a checkpoint's parameters. Read the numbers off it, naming the CLR type the
elements are stored as:

```csharp
TensorData result = OnnxEngine.Eval(y);
float[] values = result.CopyMemory<float>();   // the whole buffer, in an array you own
float first    = result.ValueAt<float>(0);     // one element
```

Each dtype marker has one CLR storage type: `float32`→`float`, `float64`→`double`,
`int64`→`long`, `int32`→`int`, `bit`→`bool`, `float16`→`Float16`, `bfloat16`→`BFloat16`, etc.
Any other type throws an `InvalidCastException` naming the tensor's dtype and its storage
type; the buffer is never reinterpreted. `CopyRawMemory()` is the same copy as bytes, for
any dtype.

On a typed `TensorData<T>` (what `As<T>()` returns) the storage type follows from `T`:
`result.As<float32>().CopyMemory()` is a `float[]`, and `ValueAt(i)` a `float`.
`AccessMemory<V>()` / `AccessMemory()` return a `ReadOnlySpan` over the storage instead of
a copy; see "What a TensorData holds" below for why a span needs the tensor kept alive. A
boxed `TensorData.Data` (`object[]`) exists for diagnostics: the storage bytes, each boxed
(the strings themselves for a `utf8` tensor), not the element values.

## Two kinds of concrete tensor: `TensorData` and `TensorAttribute`

Concrete numbers reach Shorokoo in two roles, each with its own type.

| | `TensorData` | `TensorAttribute` |
|---|---|---|
| What it is | a **run's** data: an input you feed, an output you read, training state | a **graph's** data: a literal written into the description itself |
| Where the bytes are | in the memory one backend allocated: the host, or a particular card | nowhere in particular: a shape, a dtype and the elements |
| Lifetime | its own memory, released through that backend when deleted or collected; `IDisposable` | none: immutable, shared, not disposable |
| Where you meet it | `Eval`, `Execute`, `Run`, `TrainStep`, a checkpoint's tensors | `Constant`, `ConstantOfShape`, a trainable parameter's initial value |

A graph description must mean the same thing everywhere, including after export to
`.onnx` and reload on a machine with a different card. A `TensorData` is memory one backend
allocated and has a lifetime, so it cannot be part of a description.

The slots that take a tensor-valued *operator attribute* therefore take a
`TensorAttribute`:

- `OnnxOp.Constant(value)`: a graph constant;
- `OnnxOp.ConstantOfShape(shape, value)`, and `Tensor<T>.Fill(shape, value)` on top of it:
  the fill value;
- `Globals.TrainableTensor(value, name)`: a trainable parameter's value, which is also where
  a checkpoint's weights land when they are bound into a model.

Everything else that takes concrete numbers (feeding a run, reading a result, a training
batch, a `TrainingCheckpoint`'s state) takes and returns `TensorData`.

### Building one

Usually you never name the type. `Scalar(1L)`, `Vector(1L, 2L)`, `Tensor([2L, 2L], …)`,
`TensorFill(shape, 0f)` and `VectorFill(64L, 0f)` build their literal and return a graph
value. You name it when you already hold the numbers:

```csharp
using System.Runtime.InteropServices;   // MemoryMarshal, for the second form

// From a TensorData you are finished with — the usual case.
TensorAttribute weights = TensorData([64L, 3L, 7L, 7L], myFloats).MoveToAttribute();

// Or straight from the bytes, with no TensorData in between.
TensorAttribute fill = TensorAttribute.Create(
    new Shape(1L), DType.Float32, MemoryMarshal.AsBytes<float>([0.1f]));

Variable w = Globals.TrainableTensor(weights, "conv1.weight");
```

`Shape` is a class, not a collection type, so the shape argument is `new Shape(…)` or a
`long[]`; a bare `[1L]` collection literal does not convert to it.

An attribute exposes `Shape`, `DType`, `HasValues`, `Bytes` (or `Values` for `DType.Utf8`),
`Elements<V>()` and `CopyToTensorData()`, and nothing else: no `Dispose`, no
`AllocatingBackend`, no `Space`. `HasValues` is false in one case only: a model definition
saved *without* its weights, whose parameter slots keep dtype and shape but no elements
until a checkpoint is bound onto them. Reading the elements of one throws, and says so.

### The two conversions, and which one spends its source

| | Costs | Afterwards |
|---|---|---|
| `TensorData.MoveToAttribute()` | nothing, where the tensor holds its own array; a copy otherwise | **the tensor ends**: it is dead, and reading it throws `ObjectDisposedException` saying it was moved into an attribute |
| `TensorAttribute.CopyToTensorData()` | a copy, always | both usable; the attribute is unchanged, and the copy is in the framework's own host memory |

Binding a checkpoint's weights into a graph is the hot direction (a 165 M-parameter model
is some 660 MB), so it moves, and the source is gone. The other direction copies because an
attribute is immutable and shared by every graph that captured it; a writable tensor over
the same bytes would let you edit a description.

The move takes the tensor's own array where it has one (a tensor built from a C# array,
which nothing else can write once the tensor is dead). It copies where there is no array
to hand over: the elements are a runtime's own buffer, read back through the backend that
made it when that buffer is on a card, or a string tensor's variable-length elements. The
source memory is released either way. A checkpoint's tensor, parsed for the bind, has its
own array, so binding weights takes the no-copy path.

`MoveToAttribute()` takes any live tensor, a run's output included, and refuses only one a
run is still reading. To build a literal from a result and keep the result, move a copy:

```csharp
TensorData result = compiled.Execute(input)[0].ToTensorData();
Variable literal = Globals.Tensor(result.CopyTo(ComputeContext.Host).MoveToAttribute());
```

### Passing a `TensorData` as an attribute

The slots listed above take a `TensorAttribute`; give them a `TensorData` by moving it with
`.MoveToAttribute()` at the call site:

```csharp
Tensor<float32>.Fill(shape, Globals.TensorData(1, 1.0f).MoveToAttribute());
Globals.TrainableTensor(myWeights.MoveToAttribute(), "w");     // myWeights is spent
```

One trap follows from the move:

- **A literal you want to reuse must be built twice, or converted once.** `MoveToAttribute()`
  spends its tensor, so one `TensorData` cannot serve two call sites. Convert once and pass
  the `TensorAttribute` to both (an attribute is immutable, so shareable), or build a fresh
  `TensorData` per site.

A factory that takes a `TensorData` does **not** spend it. `Globals.TensorFill<T>(shape,
TensorData<T>)` copies, so the tensor you passed stays yours and may be passed again:

```csharp
var fill = TensorData([1L], 0.5f).As<float32>();
var a = TensorFill((Vector<int64>)[Scalar(2L)], fill);
var b = TensorFill((Vector<int64>)[Scalar(3L)], fill);   // fine: fill is untouched
```

A fill value is one element, so the copy costs nothing. Only a call named for it spends a
tensor; a factory never spends an argument.

## What a `TensorData` holds, and when its values go away

A `TensorData` **is** its memory: one object per allocation, released through the backend
that made it. No second tensor names the same bytes, so deleting a tensor (`Delete()` and
`Dispose()` are the same call) releases them. The full model (how a tensor ends, what a run
holds while it runs, and the calls that delete) is in
[A tensor's lifetime](inference.md#a-tensors-lifetime-locks-and-deletion). While reading a
result:

- **Deleting is optional.** A tensor you drop is reclaimed like any other object, and the
  framework never hands you a tensor you must delete. Delete when you want the memory back
  at a known moment, e.g. in a long loop producing large tensors.
- **`Delete()` waits for nobody and interrupts nobody.** It throws while a run is reading the
  tensor instead of freeing memory under it; `TryDelete()` declines instead, and
  `DeleteAsync(...)` asks the run to stop. Deleting a dead tensor does nothing.
- **A dead tensor stops reading.** `AccessMemory()`, `AccessRawMemory()`, `.Data` and
  `.DebugData` throw `ObjectDisposedException` instead of reading freed memory, and the
  message says how the tensor died. `.Shape`, `.DType`, `.ToString()` and `.IsDisposed` keep
  working.

Operations that build one tensor from another copy, or return the tensor itself, and leave
the source as it was, unless they say otherwise. Two do, and each **ends** the tensor:
`MoveToAttribute()` ([above](#the-two-conversions-and-which-one-spends-its-source)), and
feeding it to a run as it is, which the run consumes when it starts; pass it `.Shared()` to
have the run only read it ([inference.md](inference.md#feeding-a-run-consumed-shared-or-tried)).
`TensorDataSequence.Create(...)` copies the tensors you pass it, and disposing the sequence
releases only its own copies.

`To(context)`, where the context can read the tensor as it stands, and `ToHost()`, where
the host already can, return the very tensor you called them on. Feeding the result to a
run as it is consumes the original, and deleting it (a `using` over it included) deletes
the original. `CopyTo` always gives an independent tensor; see
[Moving data between contexts](inference.md#moving-data-between-contexts).

**A span is a window, not a copy.** `AccessMemory()` and `AccessRawMemory()` point into the
tensor's memory, and nothing ties the span's lifetime to the tensor's. A span outlives the
bytes it points at if the tensor is deleted, or becomes unreachable while you are still
reading.

Being *in scope* is not being *reachable*: the runtime retires a local at its last read,
and taking the span **is** the tensor's last read. So copying out of the span is not safe
by itself: the copy happens after the tensor is collectable, and allocating the array can
trigger a collection:

```csharp
TensorData result = OnnxEngine.Eval(y);
float[] values = result.AccessMemory<float>().ToArray();  // NOT safe
```

Keep the tensor alive across the read, with `GC.KeepAlive` after it or by reading through
something that outlives the span (a field, a collection, a later use of the tensor):

```csharp
TensorData result = OnnxEngine.Eval(y);
float[] values = result.AccessMemory<float>().ToArray();
GC.KeepAlive(result);                                     // safe
```

Or use the copying accessors, which keep the tensor alive across the read:
`CopyMemory<V>()` and `CopyRawMemory()` return an array the caller owns, and
`ValueAt<V>(int)` reads one element. When you are copying the whole buffer anyway, they are
shorter and safer.

### A tensor whose values are not on the host

A tensor on a card is in the execution provider's own memory, where a host read would
dereference a device address. That is one a run left there (an output kept with
`CompiledGraph.Execute(inputs, retainOnDevice)`, or the state of a
[resident training run](training.md#keeping-training-state-on-the-device)), or one put there
by `To`, `CopyTo` or `AllocateUninitialized` on a device context. `IsHostResident` says
which it is, and the accessors throw `InvalidOperationException` instead of reading it,
naming `ToHost()`, which copies the tensor into host memory, and, for a resident run's
state, `StepToCheckpoint`, which brings that state home. A tensor built from a C# array is
host-resident, and so is a run output nobody asked to keep on the card, which ONNX Runtime
fetches to the host.

## Anti-patterns

- Do not mix dtypes in one op (e.g. add `Tensor<float32>` to `Tensor<int64>`); cast
  first with `.Cast<float32>()`.
- Do not assume `.TShape` gives compile-time dimensions — it is a graph value
  (`Vector<int64>`) resolved at evaluation, not a C# array.
- Do not call `new Tensor<T>(...)` directly; use the `Globals` factories or op results.

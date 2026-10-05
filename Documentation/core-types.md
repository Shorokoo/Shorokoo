# Core types: tensors, scalars, vectors, dtypes

Related: [defining-models.md](defining-models.md) · [inference.md](inference.md)

## Facts

- Three graph-value shapes, all generic over a dtype marker `T : IVarType`:
  - `Scalar<T>` — rank 0.
  - `Vector<T>` — rank 1 (also used for dynamic shapes, e.g. `Vector<int64>`).
  - `Tensor<T>` — rank N. All three are distinct value-struct handles implementing
    `IValue`.
- `IValue` — the base interface for any graph value *handle* (`Tensor<T>`,
  `Scalar<T>`, `Vector<T>`, and the sequence / optional / struct handles).
- `Variable` — the graph node a handle points at, and the argument type of the
  execution entry points. It is *not* an `IValue`; see
  [`Variable` and `IValue`](#variable-and-ivalue).
- Dtype markers (the generic argument): `bit` (boolean), `int8`, `int16`, `int32`,
  `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `float16`, `bfloat16`, `float32`,
  `float64`. Example: `Tensor<float32>`, `Scalar<int64>`, `Scalar<bit>`.
- `DType` is the runtime dtype descriptor (`DType.Float32`, `DType.Int64`,
  `DType.Bool`, …), used by runtime/untyped APIs.
- A graph value is symbolic; evaluate it to get numbers ([inference.md](inference.md)).
- `TensorData` / `TensorData<T>` hold concrete values: what you feed a run, and what
  it returns.
- `TensorAttribute` holds concrete values written into a graph's *description*
  (constants, fills, parameter weights). `TensorData.MoveToAttribute()` (which
  **spends** its source) and `TensorAttribute.CopyToTensorData()` convert; see
  [Two kinds of concrete tensor](#two-kinds-of-concrete-tensor-tensordata-and-tensorattribute).
- `OptionalTensorData` is the value of an `OptionalTensor` input:
  `OptionalTensorData.Some(tensor)` or `OptionalTensorData.None(dtype)`. Both are
  `IData` and feed execution like any input (see
  [defining-models.md](defining-models.md#omittable-parameters-defaulted-hypers--optional-inputs)).

## `Variable` and `IValue`

`Variable` (namespace `Shorokoo.Core`) is the non-generic graph node every handle
points at, carrying the dtype and rank at runtime. `OnnxEngine.Eval(Variable)`, its
multi-output overloads and the `Eval` forms on `ComputeContext` take it.

`Tensor<T>`, `Scalar<T>`, `Vector<T>` and the sequence / optional / struct handles
convert **implicitly** to `Variable`, so an op result goes straight in. The
conversions are on the concrete types, not on `IValue`, so a handle held as `IValue`
needs `ToVariable()`:

```csharp
var y = x.Relu();                                  // Tensor<float32>
TensorData r1 = OnnxEngine.Eval(y);                // implicit Tensor<float32> → Variable

IValue handle = y;
TensorData r2 = OnnxEngine.Eval(handle.ToVariable());   // Eval(handle) would not compile
```

`Variable.ToValue()` returns the natural handle: `Scalar<T>` at rank 0, `Vector<T>`
at rank 1, `Tensor<T>` otherwise (or `OptionalTensor<T>` / `TensorSequence<T>` /
`TensorStruct<T>`).

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
| `Tensor<float32>.Fill(shape, TensorData(...).MoveToAttribute())` | `Tensor<float32>` | The fill value is a [`TensorAttribute`](#two-kinds-of-concrete-tensor-tensordata-and-tensorattribute); `MoveToAttribute()` spends the `TensorData`. |
| `RandomUniform(shape, low = 0f, high = 1f)` | `Tensor<float32>` | Random feed over `[low, high)`. Keyed by the model's [RNG identity](rng-configuration.md); no per-site seed. See [uniform-draws.md](uniform-draws.md). |
| `RandomUniform(shape, Scalar<float32> low, Scalar<float32> high)` | `Tensor<float32>` | Range computed **in-graph** (both bounds required), exact at any width; needs a keyed (concrete, id-bearing) model. |
| `RandomNormal(shape, mean = 0f, scale = 1f)` | `Tensor<float32>` | Random feed over N(`mean`, `scale`). Keyed by the model's [RNG identity](rng-configuration.md); no per-site seed. See [normal-draws.md](normal-draws.md). |
| `RandomNormal(shape, Scalar<float32> mean, Scalar<float32> scale)` | `Tensor<float32>` | Parameters computed **in-graph** (both required), so they can change per run; needs a keyed (concrete, id-bearing) model. |

**Implicit primitive → `Scalar<T>` conversion.** Where a `Scalar<T>` is expected, a
bare primitive converts, e.g. `Scalar<int64> n = 32;` or `myScalar.Clip(0f, 6f)`. The
element type comes from the **target**: `Scalar<float32> x = 5;` is `float32`. With
no target (`var x = 1L;` is a `long`), use `Scalar(1L)` or `Scalar<T>(...)`.

**`PrimitiveParam`** (namespace `Shorokoo.Core`) appears in signatures such as
`Tensor<T>.Clip(PrimitiveParam min, PrimitiveParam max)` and
`Tensor<T> + PrimitiveParam`. It converts implicitly from `bool`, the integer types,
`float`, `double`, `Float16` and `BFloat16`, to the receiver's element type; you
never construct one. `x.Clip(0f, 6f)` compiles on all three shapes; use
`Clip(Scalar<T>, Scalar<T>)` when a bound is computed in-graph.

`Tensor(...)` / `TensorData(...)` take the **shape** first: a collection literal
(`[1]`, `[1L,3L,224L,224L]`) for the `long[]` overload, or a bare `long` for the 1-D
overload. The rest are the flat values (`params T[]`), so an existing array works:
`TensorData([1L,3L,224L,224L], myPixelArray)`. A **rank-0** value takes empty dims,
`TensorData([], 0.01f)`, as a scalar graph input needs (e.g. for
[`Specialize`](inference.md#hardcoding-hypers-with-specialize) of a scalar
`[Hyper]`). `TensorData([1], 0.01f)` is rank 1.

## Operators and fluent methods on `Tensor<T>`

- Arithmetic: `+ - * / % ^ & | << >>`, unary `-`, logical `!`.
- Comparisons return `Tensor<bit>`: `> >= < <= == !=`.
- Shape ops: `.Reshape(shape, keepAxes)` (see below), `.Transpose(dims...)`, `.Squeeze(axes)`,
  `.Unsqueeze(axis)`, `.Expand(shape)`, `.Flatten(axis)`, `.Concat(axis, others...)`,
  `.Slice(start, end, axes, steps)`, `.Pad(mode, pads, val)`, `.Tile(repeats)`.
- Indexing: `.Gather(indices, axis)` and `.GatherND(indices, batchDims)`, both
  defaulting to 0, so `table.Gather(tokens)` gathers rows.
- Math/activations: `.Relu()`, `.Sigmoid()`, `.Tanh()`, `.Softmax(axis)`, `.Gelu()`,
  `.Sqrt()`, `.Exp()`, `.Ln()`, `.Abs()`, trig (`.Sin()`, `.Cos()`, …).
- Linear algebra: `.MatMul(other)`.
- Reductions: `.Reduce(ReduceKind.Sum | Prod | Mean | Max | Min, axes, keepDims)`,
  `.ArgMax(axis)`, `.ArgMin(axis)`, `.TopK(k, axis)`.
- Casts: `.Cast<V>()`.
- Shape introspection (returns graph values): `.TShape`, `.ShapeTensor(start, end)`,
  `.DimTensor(axis)`, `.SizeTensor(...)`, `.TRank`.

### Mixing shapes in one operator

Arithmetic and comparison operators exist for every pairing of the three shapes and
a bare literal. The result takes the wider shape:

| left ⊕ right | `Tensor<T>` | `Vector<T>` | `Scalar<T>` | literal |
|---|---|---|---|---|
| **`Tensor<T>`** | `Tensor<T>` | `Tensor<T>` | `Tensor<T>` | `Tensor<T>` |
| **`Vector<T>`** | `Tensor<T>` | `Vector<T>` | `Vector<T>` | `Vector<T>` |
| **`Scalar<T>`** | `Tensor<T>` | `Vector<T>` | `Scalar<T>` | `Scalar<T>` |
| **literal** | `Tensor<T>` | `Vector<T>` | `Scalar<T>` | — |

Comparisons follow the same table with element type `bit` (`Scalar<T> <= Vector<T>`
gives `Vector<bit>`). Both operands must share one `T` (see
[Anti-patterns](#anti-patterns)).

Shift operators are the exception: the left operand cannot be a literal, and the
right must be no wider than the left. `Tensor<T> << Vector<T>` and
`Vector<T> << 1L` exist; `Vector<T> << Tensor<T>` and `1L << Tensor<T>` do not.

### Reductions and `keepDims`

`x.Reduce(kind, axes, keepDims)` **drops** reduced dimensions by default, as in
PyTorch and NumPy: `x.Reduce(ReduceKind.Mean).Scalar()` gives a rank-0 scalar, and
`x.Reduce(ReduceKind.Sum, Vector(1L))` turns `[N, C]` into `[N]`. Pass
`keepDims: true` to keep them as length-1 axes (`[N, 1]`).

ONNX defaults the other way (`keepdims=1`); the fluent `.Reduce` always emits the
attribute. `NN.Reduce` takes a `bool?` with no default, where `null` omits it and
so keeps ONNX's default.

A result broadcast back against its input needs `keepDims: true`. Without it the
shapes usually mismatch and throw, but when the remaining dimensions happen to agree
(`[N, C]` with `N == C`) it silently broadcasts along the wrong axis.

### `Reshape` and copying dimensions from the input

`x.Reshape(newShape)` follows PyTorch and NumPy: one `-1` entry is inferred from the
element count, and `0` is a **literal zero-sized dimension**. (Raw ONNX `Reshape`
with its default `allowzero=0` reads `0` as "copy the input's dimension".)

To copy dimensions from the input, list their **output positions** in `keepAxes`
and omit them from `newShape`. The batch-preserving flatten of
`x : [N, C, H, W]` (ONNX `Reshape(x, [0, -1])`) is:

```csharp
x.Reshape([Scalar(-1L)], keepAxes: [0])   // → [N, C·H·W]; N need not be known at build time
```

`keepAxes: [0, 1]` with `newShape = [-1]` gives `[N, C, H·W]`. Kept dimensions are
resolved at run time, so they work when the input's dimensions are unknown at build
time.

`Reshape` emits an ONNX `Reshape` with `allowzero=1`, or with `keepAxes`,
`allowzero=0` and `0` at each kept position. ONNX rejects `-1` together with a
literal `0` under `allowzero=1`, so a plain `Reshape` cannot mix a zero-sized and an
inferred dimension; `-1` combines freely with `keepAxes`.

## Higher-level ops (`using static Shorokoo.NN;`)

`NN` holds ops that don't read as instance methods. Common signatures (the `NN`
class has the full list):

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

`numGroups`/`epsilon` are plain C# op attributes, not `Scalar<...>`. Enums: `AutoPad`,
`PadMode`, `ReduceKind`, `RoundMode`.

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

Execution, run outputs and checkpoint parameters are `TensorData` (see
[inference.md](inference.md)). Read values by naming the CLR storage type, wherever the tensor
is:

```csharp
TensorData result = OnnxEngine.Eval(y);
float[] values = result.CopyMemory<float>();   // the whole buffer, in an array you own
float first    = result.ValueAt<float>(0);     // one element
```

A run's outputs are in the memory of the backend that ran it — on a GPU backend, the card's.
Reading a tensor on a card copies its values to the host and returns them; the tensor stays where
it is, alive and unchanged.

Storage types: `float32`→`float`, `float64`→`double`, `int64`→`long`, `int32`→`int`,
`bit`→`bool`, `float16`→`Float16`, `bfloat16`→`BFloat16`, etc. Any other type throws
`InvalidCastException` naming the dtype and its storage type. `CopyRawMemory()`
copies the bytes, for any dtype.

On a typed `TensorData<T>` (from `As<T>()`) the type is implied:
`result.As<float32>().CopyMemory()` is a `float[]`, `ValueAt(i)` a `float`.
`AccessMemory<V>()` / `AccessMemory()` return a `ReadOnlySpan` instead of a copy (see
[below](#accessmemory-in-place-on-the-host-through-a-copy-on-a-card)). `TensorData.Data`
(`object[]`) is for diagnostics: each storage byte boxed (the strings for a `utf8` tensor), not
the element values.

**Every accessor reads; none writes.** A tensor's contents are fixed when it is built — from a C#
array or bytes, by a load, or as a run's output. A tensor with other values is the output of a
run: build a graph that computes them and run it.

## Two kinds of concrete tensor: `TensorData` and `TensorAttribute`

| | `TensorData` | `TensorAttribute` |
|---|---|---|
| What it is | a **run's** data: an input you feed, an output you read, training state | a **graph's** data: a literal written into the description itself |
| Where the bytes are | memory one backend allocated: the host or a particular card | nowhere in particular: a shape, a dtype and the elements |
| Lifetime | released through its backend when deleted or collected; `IDisposable` | none: immutable, shared, not disposable |
| Where you meet it | `Eval`, `Execute`, `Run`, `TrainStep`, a checkpoint's tensors | `Constant`, `ConstantOfShape`, a trainable parameter's initial value |

A graph description must be portable and permanent, so tensor-valued operator
attributes take a `TensorAttribute`:

- `OnnxOp.Constant(value)`;
- `OnnxOp.ConstantOfShape(shape, value)`, and `Tensor<T>.Fill(shape, value)` built on
  it;
- `Globals.TrainableTensor(value, name)`, also where checkpoint weights land when
  bound into a model.

Everything else uses `TensorData`.

### Building one

The literal factories build attributes for you. Build one yourself when you already
hold the numbers:

```csharp
using System.Runtime.InteropServices;   // MemoryMarshal, for the second form

// From a TensorData you are finished with — the usual case.
TensorAttribute weights = TensorData([64L, 3L, 7L, 7L], myFloats).MoveToAttribute();

// Or straight from the bytes, with no TensorData in between.
TensorAttribute fill = TensorAttribute.Create(
    new Shape(1L), DType.Float32, MemoryMarshal.AsBytes<float>([0.1f]));

Variable w = Globals.TrainableTensor(weights, "conv1.weight");
```

`Shape` is a class: pass `new Shape(…)` or a `long[]`, not a bare `[1L]` literal.

An attribute exposes only `Shape`, `DType`, `HasValues`, `Bytes` (or `Values` for
`DType.Utf8`), `Elements<V>()` and `CopyToTensorData()`. `HasValues` is false only for
parameter slots of a model saved *without* its weights, until a checkpoint is bound;
reading their elements throws.

### The two conversions, and which one spends its source

| | Costs | Afterwards |
|---|---|---|
| `TensorData.MoveToAttribute()` | nothing, where the tensor holds its own array; a copy otherwise | **the tensor ends**: it is dead, and reading it throws `ObjectDisposedException` saying it was moved into an attribute |
| `TensorAttribute.CopyToTensorData()` | a copy, always | both usable; the attribute is unchanged, and the copy is in the framework's own host memory |

A tensor built from a C# array, including a checkpoint tensor parsed for binding,
moves without a copy. A runtime-owned buffer (possibly on a card) or a string tensor
is copied; the source memory is released either way. An attribute keeps its elements in
one managed array, so a tensor of more bytes than one holds (`Array.MaxLength`, just
under 2 GiB) is refused with `NotSupportedException` and left as it was.

`MoveToAttribute()` accepts any live tensor, a run output included, but refuses one a
run is still reading. To keep a result and also make a literal of it, move a copy:

```csharp
TensorData result = compiled.Execute(input)[0].ToTensorData();
Variable literal = Globals.Tensor(result.CopyTo(ComputeContext.Host).MoveToAttribute());
```

### Passing a `TensorData` as an attribute

Move it with `.MoveToAttribute()` at the call site:

```csharp
Tensor<float32>.Fill(shape, Globals.TensorData(1, 1.0f).MoveToAttribute());
Globals.TrainableTensor(myWeights.MoveToAttribute(), "w");     // myWeights is spent
```

The move spends the tensor, so to reuse a literal, convert once and pass the
(shareable) `TensorAttribute` to each site.

Factories that take a `TensorData` do **not** spend it. `Globals.TensorFill<T>(shape,
TensorData<T>)` copies, so you can pass the same tensor again:

```csharp
var fill = TensorData([1L], 0.5f).As<float32>();
var a = TensorFill((Vector<int64>)[Scalar(2L)], fill);
var b = TensorFill((Vector<int64>)[Scalar(3L)], fill);   // fine: fill is untouched
```

## What a `TensorData` holds, and when its values go away

A `TensorData` **is** its memory: one object per allocation, or per range of the
memory of an input a run consumed and wrote values into, never shared with
another tensor. `Delete()` (the same as `Dispose()`) releases it. The full lifetime
model is in [A tensor's lifetime](inference.md#a-tensors-lifetime-locks-and-deletion).
In short:

- **Deleting is optional.** A dropped tensor is reclaimed like any object. Delete
  to free memory at a known moment, e.g. in a long loop producing large tensors — a
  tensor standing on a block with others frees its own range, or on a backend that
  frees a block whole, its share of the block with the last of them.
- **`Delete()` never waits or interrupts.** It throws while a run is reading the
  tensor; `TryDelete()` declines instead, and `DeleteAsync(...)` asks the run to
  stop. Deleting a dead tensor does nothing.
- **A dead tensor stops reading.** `AccessMemory()`, `AccessRawMemory()`, `.Data`
  and `.DebugData` throw `ObjectDisposedException`, saying how it died. `.Shape`,
  `.DType`, `.ToString()` and `.IsDisposed` keep working.

Operations leave their source untouched unless stated. Two **end** it:
`MoveToAttribute()` ([above](#the-two-conversions-and-which-one-spends-its-source))
and feeding it to a run as it is, which consumes it; feed `.Shared()` to have the run
only read it ([inference.md](inference.md#feeding-a-run-consumed-shared-or-tried)).
`TensorDataSequence.Create(...)` copies its tensors, and disposing the sequence
releases only its copies.

`To(context)`, when the context can read the tensor in place, and `ToHost()`, when it
is already on the host, return the same tensor, so consuming or deleting the result
(including via `using`) affects the original. `CopyTo` always returns an independent
tensor; see [Moving data between contexts](inference.md#moving-data-between-contexts).

### `AccessMemory()`: in place on the host, through a copy on a card

`AccessMemory()` and `AccessRawMemory()` return a `ReadOnlySpan`, so nothing can be written
through them, and do the fastest thing for where the tensor is.

**On a card**, the tensor's values are first copied into a managed array in host memory, and the
span is over that array, which it keeps alive while you use it. The tensor holds the copy weakly
and every later call reuses it while it lives: a loop reading `t.AccessMemory()[i]` copies the
tensor once, not once per element. Once nothing uses the copy it goes at the next garbage
collection, and a call after that copies again.

**On the host**, the span is a window onto the tensor's own memory, not a copy, and it does not
keep the tensor alive. If the tensor is deleted, or becomes unreachable while you read, the span
points at freed memory. Taking the span can be the tensor's last use, after which it is
collectable even while still in scope, so this is **not** safe:

```csharp
TensorData result = OnnxEngine.Eval(y);
float[] values = result.AccessMemory<float>().ToArray();  // NOT safe
```

Keep the tensor alive across the read, with `GC.KeepAlive` or a later use:

```csharp
TensorData result = OnnxEngine.Eval(y);
float[] values = result.AccessMemory<float>().ToArray();
GC.KeepAlive(result);                                     // safe
```

Or use `CopyMemory<V>()`, `CopyRawMemory()` or `ValueAt<V>(int)`, which keep the
tensor alive for you.

### A tensor whose values are on a card

Every output of a run on a GPU backend (`Eval`, `Execute`, `Run`, a compiled graph's
runs), the state a [training step](training.md#keeping-training-state-on-the-device)
on a GPU hands back, and anything placed by `To` or `CopyTo` on a device context is
in the card's memory; `IsHostResident` is false. It is read like any other tensor:
`CopyMemory`, `ValueAt`, `AccessMemory` and the rest copy its values to the host and
return them, and the tensor stays on the card. Tensors built from C# arrays, and the
outputs of a run on a CPU backend, are host-resident. `ToHost()` makes a tensor of
its own in host memory — `checkpoint.ToHost()` does so for a whole training
checkpoint. Nothing moves a run's output to the host but `ToHost()`, `To` or
`CopyTo`, or a run on a host backend it is fed to — see
[Where a run's inputs and outputs are](inference.md#where-a-runs-inputs-and-outputs-are).

## Anti-patterns

- Do not mix dtypes in one op (e.g. add `Tensor<float32>` to `Tensor<int64>`); cast
  first with `.Cast<float32>()`.
- Do not assume `.TShape` gives compile-time dimensions — it is a graph value
  (`Vector<int64>`) resolved at evaluation, not a C# array.
- Do not call `new Tensor<T>(...)` directly; use the `Globals` factories or op results.

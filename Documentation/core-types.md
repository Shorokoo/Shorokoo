# Core types: tensors, scalars, vectors, dtypes

Related: [defining-models.md](defining-models.md) · [inference.md](inference.md)

## Facts

- Three graph-value shapes, all generic over a dtype marker `T : IVarType`:
  - `Scalar<T>` — rank 0.
  - `Vector<T>` — rank 1 (also used for dynamic shapes, e.g. `Vector<int64>`).
  - `Tensor<T>` — rank N. `Scalar<T>`, `Vector<T>`, and `Tensor<T>` are distinct
    value-struct handles, all implementing the common `IValue` interface.
- `IValue` — the base interface for any graph value (`Tensor<T>`, `Scalar<T>`,
  `Vector<T>`, and the sequence / optional / struct handles). User-facing code holds
  `IValue` handles; the framework converts them to the internal graph node as needed.
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
| `RandomUniform(shape, low = 0f, high = 1f, seed = null)` | `Tensor<float32>` | Random init; all but `shape` are optional. |
| `RandomNormal(shape, mean = 0f, scale = 1f, seed = null)` | `Tensor<float32>` | Random init; `RandomNormal(shape, seed: 0)` is valid. |

**Implicit primitive → `Scalar<T>` conversion.** Wherever a `Scalar<T>` is expected, a bare
primitive value converts to one automatically, so the `Scalar(...)` wrapper is usually
optional — e.g. `Scalar<int64> n = 32;` or `myScalar.Clip(0f, 6f)`. The element type comes
from the **target context**, not the literal: `Scalar<float32> x = 5;` builds a `float32`
scalar. Reach for the explicit `Scalar(...)` / `Scalar<T>(...)` helpers when there is no
`Scalar<T>` target to infer from — e.g. `var x = Scalar(1L);`, since a bare `var x = 1L;`
is a plain `long`, not a scalar.

First-argument convention for `Tensor(...)` / `TensorData(...)`: the first argument is
the **shape (dims)**. Pass a collection literal (`[1]`, `[1L,3L,224L,224L]`) for the
`long[]` overload, or a bare `long` (e.g. `1`) for the 1-D convenience overload. The
remaining arguments are the flat element values (`params T[]`), so you can pass an
existing array directly: `TensorData([1L,3L,224L,224L], myPixelArray)`.

## Operators and fluent methods on `Tensor<T>`

- Arithmetic: `+ - * / % ^ & | << >>`, unary `-`, logical `!`.
- Comparisons return `Tensor<bit>`: `> >= < <= == !=`.
- Shape ops: `.Reshape(shape)`, `.Transpose(dims...)`, `.Squeeze(axes)`,
  `.Unsqueeze(axis)`, `.Expand(shape)`, `.Flatten(axis)`, `.Concat(axis, others...)`,
  `.Slice(start, end, axes, steps)`, `.Pad(mode, pads, val)`, `.Tile(repeats)`.
- Math/activations: `.Relu()`, `.Sigmoid()`, `.Tanh()`, `.Softmax(axis)`, `.Gelu()`,
  `.Sqrt()`, `.Exp()`, `.Ln()`, `.Abs()`, trig (`.Sin()`, `.Cos()`, …).
- Linear algebra: `.MatMul(other)`.
- Reductions: `.Reduce(ReduceKind.Sum | Prod | Mean | Max | Min, axes, keepDims)`,
  `.ArgMax(axis)`, `.ArgMin(axis)`, `.TopK(k, axis)`.
- Casts: `.Cast<V>()`.
- Shape introspection (returns graph values): `.TShape`, `.ShapeTensor(start, end)`,
  `.DimTensor(axis)`, `.SizeTensor(...)`, `.TRank`.

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
var w = RandomNormal(Vector(64L, 3L, 7L, 7L), seed: 0);
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

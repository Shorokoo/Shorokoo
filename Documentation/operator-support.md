# Operator support matrix

Shorokoo supports the standard `ai.onnx` domain from **opset 21** up to **opset 26**, the
maximum the bundled ONNX Runtime 1.30 loads; import does not convert older models.
Exported models are stamped at the
**opset-21 baseline**; only a few post-21 attributes on imported (or
`NodeBuilder`-built) nodes raise the stamp, and no post-21 operator reaches an
exported ONNX file. See [limitations.md](limitations.md) for the details and the reason.

Every operator Shorokoo defines is listed below: the opset-21 set plus the
post-21 additions (`Attention`, `RMSNormalization`, `RotaryEmbedding` at opset 23;
`Swish`, `TensorScatter` at 24; `BitCast`, `CumProd` at 26), grouped by family.
Each introducing opset is also the operator's export floor (see
[limitations.md](limitations.md)), except `Attention`, floored at 24 because Shorokoo
defines it with the opset-24 inputs (`nonpad_kv_seqlen`).

- **Build & run**: the operator can be constructed and executes on the ONNX
  Runtime backend. Footnotes flag spec-legal corners restricted in-framework or by
  ONNX Runtime's CPU kernels. ❌ means it cannot be built: no definition, or the
  entry point throws (the note says which).
- **QEE**: the Quick Execution Engine propagates **dtype and shape** for every
  supported operator and **values** for small tensors (see
  [limitations.md](limitations.md)). 🟡 means values are not, or only partly,
  computed.
- **Gradient**: reverse-mode autodiff. Unsupported attribute combinations throw
  `AutoDiffNotSupportedException`.

Symbols: ✅ full support · 🟡 partial (see the family's notes) · ❌ not
supported (see notes) · N/A not applicable (non-differentiable output, leaf
operator, or operator not available).

Some ✅ entries in the last two columns are computed or differentiated through a
registered **decomposition into simpler operators**; the result is the same.
Computing one leaves your graph unchanged. **Differentiating one does not**: the
training graph autodiff returns contains the decomposition in place of the
operator (a `Softsign` becomes `Abs`, `Add` and `Div`), while your inference model
is untouched. The exporter decomposes only `TensorScatter`; a `Softsign` exports
as a `Softsign`.

A ✅ in the first column also covers a few corners where ONNX Runtime's kernels
compute otherwise than the spec. The ONNX Runtime backend rewrites each such call,
when it builds a session, into an equivalent one its kernels compute as the spec
does — for a graph you built and an imported model alike. Only the session's model
is rewritten: your graph, an ONNX export, generated C# and `.srk` keep the operator
as written. The family notes name the corners;
[debugging.md](debugging.md#backend-kernel-workarounds) lists them all, in the
order they apply.

## Elementwise math & activations

| Op | Build & run | QEE | Gradient |
|---|---|---|---|
| Abs | ✅ | ✅ | ✅ |
| Acos | ✅ | ✅ | ✅ |
| Acosh | ✅ | ✅ | ✅ |
| Add | ✅ | ✅ | ✅ |
| Asin | ✅ | ✅ | ✅ |
| Asinh | ✅ | ✅ | ✅ |
| Atan | ✅ | ✅ | ✅ |
| Atanh | ✅ | ✅ | ✅ |
| BitCast | ❌ [1] | ✅ | N/A (bit reinterpretation) |
| Cast | ✅ | 🟡 [2] | ✅ [3] |
| CastLike | ✅ | 🟡 [2] | ✅ [3] |
| Ceil | ✅ | ✅ | N/A [4] |
| Celu | ✅ | ✅ | ✅ |
| Clip | ✅ | ✅ | 🟡 [5] |
| Cos | ✅ | ✅ | ✅ |
| Cosh | ✅ | ✅ | ✅ |
| CumProd | ❌ [1] | ✅ | ✅ |
| CumSum | ✅ | ✅ | ✅ |
| Div | ✅ | ✅ | ✅ |
| Elu | ✅ | ✅ | ✅ |
| Erf | ✅ | 🟡 [6] | ✅ |
| Exp | ✅ | ✅ | ✅ |
| Floor | ✅ | ✅ | N/A [4] |
| Gelu | ✅ | ✅ | ✅ [7] |
| HardSigmoid | ✅ | ✅ | ✅ |
| HardSwish | ✅ | ✅ | ✅ |
| Hardmax | ✅ | ✅ | N/A [8] |
| LeakyRelu | ✅ | ✅ | ✅ |
| Log | ✅ | ✅ | ✅ |
| LogSoftmax | ✅ | ✅ | ✅ |
| Max | ✅ | ✅ | ✅ [9] |
| Mean | ✅ | ✅ | ✅ |
| Min | ✅ | ✅ | ✅ [9] |
| Mish | ✅ | ✅ | ✅ |
| Mod | ✅ | ✅ | ✅ [10] |
| Mul | ✅ | ✅ | ✅ |
| Neg | ✅ | ✅ | ✅ |
| PRelu | 🟡 [11] | ✅ | ✅ |
| Pow | 🟡 [12] | ✅ | ✅ [13] |
| Reciprocal | ✅ | ✅ | ✅ |
| Relu | 🟡 [14] | ✅ | ✅ |
| Round | ✅ | ✅ | N/A [4] |
| Selu | ✅ | ✅ | ✅ |
| Shrink | ✅ | ✅ | ✅ |
| Sigmoid | ✅ | ✅ | ✅ |
| Sign | ✅ | ✅ | N/A [4] |
| Sin | ✅ | ✅ | ✅ |
| Sinh | ✅ | ✅ | ✅ |
| Softmax | ✅ | ✅ | ✅ |
| Softplus | ✅ | ✅ | ✅ |
| Softsign | ✅ | ✅ | ✅ |
| Sqrt | ✅ | ✅ | ✅ |
| Sub | ✅ | ✅ | ✅ |
| Sum | ✅ | ✅ | ✅ |
| Swish | ✅ [15] | ✅ | ✅ |
| Tan | ✅ | ✅ | ✅ |
| Tanh | ✅ | ✅ | ✅ |
| ThresholdedRelu | ✅ | ✅ | ✅ |

1. `OnnxOp.BitCast` and `OnnxOp.CumProd` throw `NotImplementedException` and have
   no `NN.*` wrapper: neither has an opset-21 equivalent to lower to.
2. QEE values use float32/int64 storage, so narrowing-integer wrap and
   float16/bfloat16 rounding are not modeled; string/complex/int4 casts propagate
   dtype only.
3. Gradient is cast back to the source dtype; integer-target rounding is
   ignored (straight-through estimator).
4. Piecewise-constant output: zero gradient wherever defined.
5. Gradient flows to the data input only; `min`/`max` are treated as constants.
6. QEE values for float inputs only.
7. Both `approximate="none"` and `approximate="tanh"` are differentiated exactly.
8. One-hot output; zero gradient by convention.
9. Ties share the gradient equally.
10. Float inputs: d/da = 1, d/db = −q with the fmod-consistent quotient; integer
    Mod is piecewise constant.
11. Float tensors only; the spec also allows (u)int32/64 inputs.
12. Float base only; the spec also allows int32/int64 bases.
13. The exponent's gradient uses ln(base) and is NaN for base ≤ 0 (harmless when
    the exponent is a constant).
14. Float tensors only; the spec also allows signed integers since opset 14.
15. Lowers inline to `Mul`/`Sigmoid` (`y = x * sigmoid(alpha * x)`), so it runs on
    any execution provider; ONNX Runtime runs a `Swish` node only in a model stamped at
    opset 24.

## Comparisons & logic

| Op | Build & run | QEE | Gradient |
|---|---|---|---|
| And | ✅ | ✅ | N/A |
| BitShift | ✅ | ✅ | N/A |
| BitwiseAnd | 🟡 [1] | ✅ | N/A |
| BitwiseNot | 🟡 [1] | ✅ | N/A |
| BitwiseOr | 🟡 [1] | ✅ | N/A |
| BitwiseXor | 🟡 [1] | ✅ | N/A |
| Equal | ✅ | ✅ | N/A |
| Greater | ✅ | ✅ | N/A |
| GreaterOrEqual | ✅ | ✅ | N/A |
| IsInf | ✅ | ✅ | N/A |
| IsNaN | ✅ | ✅ | N/A |
| Less | ✅ | ✅ | N/A |
| LessOrEqual | ✅ | ✅ | N/A |
| Not | ✅ | ✅ | N/A |
| Or | ✅ | ✅ | N/A |
| Where | ✅ [2] | ✅ | ✅ [3] |
| Xor | ✅ | ✅ | N/A |

Boolean/integer outputs are non-differentiable, hence N/A.

1. Unsigned integer tensors only; the spec also allows signed integers.
2. ONNX Runtime has no `Where` kernel over int8, int16, uint16, uint32, uint64,
   bfloat16 or bool values. Its backend rewrites such a call when it builds a
   session: a bool one as `Or(And(c, x), And(Not(c), y))`, every other one as a
   `Where` over int32 (for int8, int16, uint16), int64 (for uint32, uint64) or
   float32 (for bfloat16) with the result cast back. The graph and its export
   keep the `Where` ([#423](https://github.com/Shorokoo/Shorokoo/issues/423)).
   Every selected value comes out exact, with one exception: ONNX Runtime's
   float32, float64 and float16 `Where` gives +0 where it selects −0 from `x`
   (a −0 from `y` keeps its sign), and the float32 path of bfloat16 inherits it.
   Accepted as ONNX Runtime's behaviour
   ([#439](https://github.com/Shorokoo/Shorokoo/issues/439)).
3. The condition is non-differentiable; both branches get broadcast-aware
   gradients.

## Reductions

| Op | Build & run | QEE | Gradient |
|---|---|---|---|
| ArgMax | ✅ | ✅ | N/A [1] |
| ArgMin | ✅ | ✅ | N/A [1] |
| ReduceL1 | ✅ [5, 6] | ✅ | ✅ |
| ReduceL2 | ✅ [5] | ✅ | ✅ |
| ReduceLogSum | ✅ [5] | 🟡 [4] | ✅ |
| ReduceLogSumExp | ✅ [5] | 🟡 [4] | ✅ |
| ReduceMax | ✅ [5] | 🟡 [4] | ✅ [2] |
| ReduceMean | ✅ [5, 6] | 🟡 [4] | ✅ |
| ReduceMin | ✅ [5] | 🟡 [4] | ✅ [2] |
| ReduceProd | ✅ [5, 6] | ✅ | ✅ [3] |
| ReduceSum | ✅ [5, 6] | ✅ | ✅ |
| ReduceSumSquare | ✅ [5, 6] | ✅ | ✅ |

`noop_with_empty_axes` set with no axes, or an empty axes tensor, means no axis is
reduced: each element is a group of its own, and the output, of the input's shape,
holds the reduction of each one-element group. That is the element itself for
`ReduceSum`, `ReduceMean`, `ReduceMax`, `ReduceMin`, `ReduceProd` and
`ReduceLogSumExp`, ±inf included; its absolute value for `ReduceL1` and `ReduceL2`;
its square for `ReduceSumSquare`; and its logarithm for `ReduceLogSum`. This is the
reading of the ONNX reference implementation and of the operators' function bodies,
and every backend and QEE compute it; an empty input gives an empty output of its
shape and type.

1. Integer index output: non-differentiable.
2. Ties share the gradient equally.
3. The gradient uses prod/x and is NaN when an element of a group of two or more
   is exactly 0; a one-element group's gradient is 1.
4. Only when a **reduced axis has extent 0** (so each group is empty): QEE leaves
   the value uncomputed, because the empty-group result depends on dtype (-inf,
   +inf, 0, -inf, -inf for `float32`; the type's minimum/maximum for an integer
   `ReduceMax`/`ReduceMin`; false/true for `bool`). Note 5 says how the ONNX
   Runtime backend computes these. `ReduceSum`, `ReduceSumSquare`, `ReduceL1`,
   `ReduceL2` and `ReduceProd` fold to their identity (0, or 1 for `Prod`); an
   empty **kept** axis folds to the empty result for all ten.
5. ONNX Runtime's reduction kernels depart from the spec over an **empty input**,
   and its backend rewrites the affected calls when it builds a session, for built
   and imported graphs alike; the graph, its export, generated C# and `.srk` keep
   the reduction as written. A call whose input is a scalar or a nonempty
   `Constant` is left alone. The rewrites, in the order they apply:
   - Negative axes: ONNX Runtime ignores a negative axis of an empty input,
     reducing only the non-negative ones. The axes are made non-negative — as a
     constant when they are constant and the input's rank is known, in the graph
     otherwise ([#422](https://github.com/Shorokoo/Shorokoo/issues/422)).
   - `noop_with_empty_axes` set with no axes, or an empty axes tensor: ONNX Runtime
     reduces every axis of an empty input where the output is that empty input.
     The call keeps one reduction of the same data and axes with `keepdims` 0, the
     reduced axes put back by an `Unsqueeze`, and an empty input with empty axes is
     viewed with a trailing axis of one that alone is reduced — no branch and no
     copy ([#409](https://github.com/Shorokoo/Shorokoo/issues/409)).
   - A `bool` `ReduceMax`/`ReduceMin`: ONNX Runtime throws when a reduced axis of a
     `bool` input has extent 0. The call becomes a uint8 `ReduceMax` of the input cast
     back — over the negated input, negated back, for `ReduceMin` — whose empty group
     gives 0, that is false: exact for every input, with no branch. A reduction with no
     axes and `noop_with_empty_axes` set is left alone
     ([#382](https://github.com/Shorokoo/Shorokoo/issues/382)).
6. Over an integer input, Shorokoo's folding and QEE compute the sum, mean, product,
   L1 norm and sum of squares in the declared width, wrapping as two's complement:
   int32 `ReduceSum([2147483647, 2147483647])` is −2, and the mean of the same pair is
   −1. ONNX Runtime's CPU kernels accumulate an int32 or int64 group in double and clamp
   the result to the type's range, so that sum and mean both give 2147483647; an int64
   group is also rounded once its accumulation passes 2^53
   (`ReduceSum([9007199254740993, 0])` gives 9007199254740992). A result within the
   type and below 2^53 is exact on both. Accepted as ONNX Runtime's behaviour.

## Shape & data movement

| Op | Build & run | QEE | Gradient |
|---|---|---|---|
| Compress | ✅ | ✅ | ✅ |
| Concat | ✅ | ✅ | ✅ |
| Constant | 🟡 [1] | ✅ | N/A (leaf) |
| ConstantOfShape | ✅ | ✅ | N/A [2] |
| DepthToSpace | 🟡 [3] | ✅ | ✅ |
| Expand | ✅ | ✅ | ✅ |
| EyeLike | ✅ | ✅ | N/A (structural) |
| Flatten | ✅ | ✅ | ✅ |
| Gather | ✅ | ✅ | ✅ [4] |
| GatherElements | ✅ | ✅ | ✅ |
| GatherND | ✅ | ✅ | 🟡 [5] |
| Identity | ✅ | ✅ | ✅ |
| NonZero | 🟡 [6] | ✅ [7] | N/A [2] |
| OneHot | ✅ | ✅ | N/A [8] |
| Pad | ✅ | ✅ | 🟡 [9] |
| Range | 🟡 [19] | ✅ | N/A [2] |
| Reshape | ✅ | ✅ | ✅ |
| ReverseSequence | 🟡 [10] | ✅ | ✅ |
| Scatter | ❌ [11] | N/A | N/A |
| ScatterElements | ✅ | ✅ | 🟡 [12] |
| ScatterND | ✅ | ✅ | 🟡 [12] |
| Shape | ✅ | ✅ | N/A [2] |
| Size | ✅ | ✅ | N/A [2] |
| Slice | ✅ | ✅ | 🟡 [13] |
| SpaceToDepth | ✅ | ✅ | ✅ |
| Split | ✅ | ✅ | ✅ |
| Squeeze | ✅ | ✅ | ✅ |
| TensorScatter | ✅ [14] | ✅ [15] | ✅ [16] |
| Tile | ✅ | ✅ | ✅ |
| TopK | ✅ | ✅ [17] | ✅ |
| Transpose | ✅ | ✅ | ✅ |
| Trilu | ✅ | ✅ | ✅ |
| Unique | ✅ | 🟡 [18] | ✅ |
| Unsqueeze | ✅ | ✅ | ✅ |

1. The `sparse_value` attribute is unsupported; all dense value variants work.
   On the [JAX backend](jax-backend.md#limitations), XLA replaces a float
   constant that compares equal to an iota, such as `[-0.0, 1.0]`, with an iota,
   so its −0 becomes +0. Accepted as XLA's behaviour
   ([#441](https://github.com/Shorokoo/Shorokoo/issues/441)).
2. Index/shape/count output: non-differentiable.
3. Float tensors only; the spec allows all tensor types.
4. A negative `axis` requires a statically known input rank.
5. `batch_dims = 0` only; `batch_dims > 0` throws.
6. Signed numeric tensors only; the spec also allows bool/unsigned inputs.
7. Exact shape and values when the input data is known to QEE, rank-only
   otherwise.
8. Indices and depth are non-differentiable; the values pair is treated as
   constant.
9. Constant mode only; reflect/edge/wrap throw.
10. Numeric tensors only; the spec allows all tensor types.
11. Deprecated in ONNX since opset 11 and not implemented; use ScatterElements.
12. Reductions `none`/`add` only; `mul`/`min`/`max` throw.
13. Exact when a `steps` input is wired (any stride, including negative); without
    `steps`, negative starts/ends are clamped approximately.
14. Exported as `update` concatenated onto `past_cache` along the sequence axis
    plus one `GatherElements`, so the model stamps at opset 21; a saved
    architecture keeps the operator. The decomposition is checked element for
    element against ONNX Runtime's opset-24 kernel in both modes, with and
    without `write_indices`, at ranks 2 to 4, including empty inputs and every
    element type (bool and bfloat16 included). The spec's preconditions are **not checked**, and violating them
    gives silently wrong results rather than an error:
    - `sequence_length <= max_sequence_length` (an over-long window returns
      garbage);
    - in `linear` mode, `write_indices + sequence_length <= max_sequence_length`;
    - each write index non-negative (a negative one shifts the write; one at or
      past `max_sequence_length` drops it), one per batch (a single-entry
      `write_indices` is broadcast over every batch);
    - `axis` must not normalize to 0, the batch axis. `OnnxOp.TensorScatter`
      refuses a literal 0, but a rank-2 cache must name axis 1 or −1, since the
      default −2 normalizes to 0.

    Check these yourself when the window or indices are computed.
15. Values computed through the same decomposition.
16. Differentiated through the same decomposition: `present_cache`'s gradient
    reaches `past_cache` outside the written window and `update` inside it, per
    batch and in both modes. `write_indices` is non-differentiable.
17. Values for small tensors honor `largest` (ties go to the lower index); with
    `k` wired but unknown, the shape is rank-only with bounds.
18. The flatten form computes all four outputs for small tensors (both `sorted`
    modes); the `axis` form is shape-only, with a data-dependent unique count.
19. A model that returns a `Range` as an output beside a `Gather` the `Range`
    drives fails on ONNX Runtime: its graph optimization fuses the `Gather` and
    removes the `Range` although it is an output, and reading the model's outputs
    throws `UnsupportedDTypeException` (`OU002`). Accepted as ONNX Runtime's
    behaviour ([#432](https://github.com/Shorokoo/Shorokoo/issues/432)).
    An int64 or int32 `Range` counts its elements exactly on every backend, a span
    `limit − start` beyond 2^53 or beyond its type included, except as below.
    ONNX Runtime's kernel takes `limit − start` in the input type, which wraps
    beyond it, and computes the count in double precision, so on ONNX Runtime an
    int64 `Range` is rewritten to count in uint64 and scale a `Range(0, count, 1)`
    by `delta`, refusing a count no tensor holds, and an int32 `Range` is rewritten
    the same way on its inputs cast to int64, its elements cast back. Two forms are
    left as written. One whose three inputs are `Constant`s spanning less than
    2^53 for int64, or within int32 for int32, the kernel counts exactly. One whose
    `delta` is a `Constant` 1 or −1, whatever its `start` and `limit`, such as
    `Range(0, n, 1)` or `Range(s, s + n, 1)`, so that a position or index `Range`
    pays nothing: the kernel counts it exactly for a span below 2^53, and for a
    larger span that does not wrap gives no elements or refuses the call, as the
    spec's count is 0 or one no tensor holds. Where its span wraps its type, ONNX
    Runtime gives the elements of the wrapped difference — `Range(long.MaxValue −
    2, long.MinValue + 2, 1)` gives five where the spec gives none — or refuses
    the call (`Tensor storage size overflowed`): accepted as ONNX Runtime's
    behaviour ([#447](https://github.com/Shorokoo/Shorokoo/issues/447),
    [#450](https://github.com/Shorokoo/Shorokoo/issues/450)).

## Convolution & pooling

| Op | Build & run | QEE | Gradient |
|---|---|---|---|
| AveragePool | ✅ [9] | 🟡 [1] | 🟡 [2] |
| Conv | ✅ | 🟡 [1] | 🟡 [3] |
| ConvTranspose | ✅ [10] | 🟡 [1] | 🟡 [4] |
| DeformConv | ✅ | 🟡 [1] | ❌ [5] |
| GlobalAveragePool | ✅ | 🟡 [1] | ✅ |
| GlobalLpPool | ✅ | 🟡 [1] | ✅ |
| GlobalMaxPool | ✅ | 🟡 [1] | ✅ |
| LpPool | ✅ [9] | 🟡 [1] | ✅ |
| MaxPool | ✅ [9] | 🟡 [1] | 🟡 [6] |
| MaxRoiPool | ✅ | 🟡 [1] | 🟡 [7] |
| MaxUnpool | ✅ | 🟡 [8] | ✅ |

1. Shape/dtype only; use the ONNX Runtime backend for values.
2. `ceil_mode=1` throws; everything else (count_include_pad, SAME auto_pad,
   dilations, overlapping windows) is supported.
3. Backward requires explicit pads (SAME_UPPER/SAME_LOWER not resolved); weight
   gradient is 2-D only. Grouped convolutions are supported.
4. As Conv (2-D weight gradient only, no SAME auto_pad, grouped supported,
   `output_padding` handled); `output_shape` is ignored in the backward.
5. Differentiation throws.
6. Exact except `storage_order=1` (throws); ties route the gradient to the first
   maximum.
7. Recompute-and-mask approximation; `rois` gets no gradient.
8. Shape from the `output_shape` input when known; values not computed.
9. Where ONNX Runtime's pooling kernels depart from the spec, its backend rewrites
   the call when it builds a session, for built and imported graphs alike; the
   graph, its export, generated C# and `.srk` keep the pool as written.
   - **Padding** (`AveragePool`, `LpPool`, `MaxPool`): `SAME_UPPER`/`SAME_LOWER`
     with dilation above 1 or stride above the kernel, and explicit pads as large
     as the kernel, are rebuilt as `Pad`, a pool ONNX Runtime computes as the spec
     does, and `Slice`, with no branch; where the stated input dimensions leave
     nothing to crop, ONNX Runtime removes the `Slice` when it builds the session ([#379](https://github.com/Shorokoo/Shorokoo/issues/379), [#408](https://github.com/Shorokoo/Shorokoo/issues/408)).
   - **`MaxPool` indices** over int8 or uint8: ONNX Runtime gives a window holding
     only the type's lowest value a wrong index. When the `Indices` output is
     read, the pool is computed over the input cast to float32 and its values cast
     back ([#420](https://github.com/Shorokoo/Shorokoo/issues/420)).
   - A `MaxPool` whose `Indices` output is read, with pads as large as the kernel:
     the padding written into the input is the element type's lowest value, so in a
     window whose input elements are all that value every position ties and the pool
     takes the window's start, which may be written padding. Such an index is carried
     to the window's first input element, the spec's.
   - On the [PyTorch backend](pytorch-backend.md#limitations), which pads the
     input with −inf itself, a window whose maximum is −inf takes a padded
     position before the input element holding it as its first maximum: its
     index names the padding, such as −1. The values are right. Accepted as the
     backend's behaviour ([#425](https://github.com/Shorokoo/Shorokoo/issues/425)).
10. With `SAME_UPPER`/`SAME_LOWER` and a stride above the kernel's extent plus
    `output_padding`, the output is `in · stride` long and reaches past the full
    transposed convolution, where it holds the bias alone. ONNX Runtime gives the
    full transposed convolution instead; its backend rewrites such a call when it
    builds a session as the call without padding or bias, a `Pad` by the negated
    padding, and the bias ([#444](https://github.com/Shorokoo/Shorokoo/issues/444)).

## Normalization & losses

| Op | Build & run | QEE | Gradient |
|---|---|---|---|
| BatchNormalization | ✅ [1] | 🟡 [2] | ✅ [3] |
| Dropout | 🟡 [4] | ✅ | ✅ [4] |
| GroupNormalization | ✅ | 🟡 [5] | ✅ |
| InstanceNormalization | ✅ | 🟡 [5] | ✅ |
| LRN | ✅ | 🟡 [5] | ✅ |
| LayerNormalization | ✅ | 🟡 [5] | ✅ [6] |
| LpNormalization | ✅ | 🟡 [5] | ✅ |
| MeanVarianceNormalization | ✅ | 🟡 [5] | ✅ |
| NegativeLogLikelihoodLoss | ✅ | 🟡 [5] | ✅ |
| RMSNormalization | ✅ [7] | ✅ | ✅ |
| SoftmaxCrossEntropyLoss | ✅ | 🟡 [5] | ✅ |

1. With `training_mode=1` the node is decomposed into primitives on export, with
   the same results.
2. Values in inference mode; training mode is shape/dtype only.
3. Both modes (batch-stats backward); gradients into the running-mean/var
   outputs throw.
4. Training-mode Dropout is not supported: ONNX Runtime may constant-fold one
   whose data input is constant into a no-drop identity, unguarded. Shorokoo's
   dropout layers build masks from the keyed RNG feed (see
   rng-configuration.md), so this affects only hand-authored graphs. The
   mask-based gradient throws if the forward mask is unavailable.
5. Shape/dtype only.
6. Gradients into the optional Mean/InvStdDev outputs are treated as zero.
7. Lowers inline to opset-21 primitives
   (`y = x / sqrt(mean(x², suffix axes) + epsilon) * scale`, via
   `ReduceMean`/`Sqrt`/`Div`/`Mul`), so it runs on any execution provider.

## MatMul & linear algebra

| Op | Build & run | QEE | Gradient |
|---|---|---|---|
| Attention | ❌ [1] | 🟡 [2] | ❌ [3] |
| Det | ✅ | 🟡 [4] | ✅ |
| Einsum | ✅ | 🟡 [5] | 🟡 [6] |
| Gemm | ✅ | ✅ | ✅ |
| MatMul | ✅ [7] | ✅ | ✅ |
| RotaryEmbedding | ❌ [1] | 🟡 [2] | ❌ [3] |

1. `OnnxOp.Attention`, `OnnxOp.AttentionWithKVCache`, `OnnxOp.RotaryEmbedding` and
   their `NN.*` wrappers throw `NotImplementedException`: none has an opset-21
   equivalent. Build attention from primitives, or use the NN library's
   `Attention.ScaledDotProductAttention` / `MultiHeadAttention` and its rotary
   helper `Attention.ApplyRoPE`; see [nn-library.md](nn-library.md).
2. Shape/dtype only.
3. Not implemented; raises `AutoDiffNotSupportedException` (`AD003`).
4. Shape/dtype only.
5. Shape inferred from the equation (matmul/transpose/reduce/diagonal forms;
   exotic equations give unknown shape); values not computed.
6. Repeated subscripts within one operand (e.g. `"ii->i"`) are unsupported;
   ellipsis is supported.
7. With a contraction dimension of 0 the product is zeros. ONNX Runtime's
   kernel, and the `FusedMatMul` it fuses a `Transpose` into, compute a matrix
   or vector times a matrix and a vector times a vector as the spec does; for
   other operands with a dimension of 0 they can leave the output unwritten or
   give it the left operand's batch dimension where that is 1 and the right
   one's is not. Its backend corrects this when the session is built with every
   input's dimensions stated, as a training step is for the shapes it is fed,
   outside a loop body: the product goes through an `If` on either operand or
   the product being empty, giving zeros of the product's shape where one is,
   which ONNX Runtime folds away wherever the shapes follow from those
   dimensions. Where an operand's shape is computed from the data (a
   `NonZero`, a `TopK` with a computed `k`, a `Reshape` or `Expand` to a
   computed shape) that `If` runs on every run, at the cost of a few shape
   operations. A `Constant` operand with a dimension of 0 gives those zeros
   whatever the dimensions. A session built without stated dimensions, and a
   `MatMul` in a loop body, run the `MatMul` as written, and the kernel runs
   on the empty operands in every session: it refuses an empty batch against
   an operand with no batch dimension or one of 1, a `FusedMatMul` that moves
   the batch axis of rank-3 operands can stop the process (SIGFPE), and a
   wrongly shaped product can fail a run where ONNX Runtime reuses its memory.
   Accepted as ONNX Runtime's behaviour
   ([#451](https://github.com/Shorokoo/Shorokoo/issues/451)).

## Quantization

| Op | Build & run | QEE | Gradient |
|---|---|---|---|
| ConvInteger | 🟡 [1] | 🟡 [2] | N/A [3] |
| DequantizeLinear | ✅ | ✅ | N/A [3] |
| DynamicQuantizeLinear | ✅ | ✅ | N/A [3] |
| MatMulInteger | 🟡 [1] | ✅ | N/A [3] |
| QLinearConv | ✅ | 🟡 [2] | N/A [3] |
| QLinearMatMul | ✅ | 🟡 [2] | N/A [3] |
| QuantizeLinear | ✅ [4] | ✅ | N/A [3] |

1. Scalar weight zero point only; a per-channel `w_zero_point` is not
   representable.
2. Shape/dtype only.
3. Non-differentiable; no straight-through estimator, so no quantization-aware
   training.
4. `NN.QuantizeLinear` and `OnnxOp.QuantizeLinear` do not expose the opset-23
   `precision` attribute (it selects float8/float4 targets, which Shorokoo does
   not support). An imported model carrying it round-trips and re-exports at
   opset 23.

## Recurrent (RNN family)

| Op | Build & run | QEE | Gradient |
|---|---|---|---|
| GRU | 🟡 [1] | 🟡 [2] | 🟡 [3] |
| LSTM | 🟡 [1] | 🟡 [2] | 🟡 [3] [4] |
| RNN | 🟡 [1] | 🟡 [2] | 🟡 [3] |

1. ONNX Runtime's CPU kernels require `hidden_size` (optional per spec) and reject
   `layout=1`; those forms build and infer shapes but do not execute on CPU.
2. Shape/dtype only (including bidirectional shapes).
3. Forward or reverse direction only, default activations, `layout=0`;
   bidirectional, custom activations, `clip`, `layout=1` and a wired
   `sequence_lens` throw.
4. LSTM peephole weights (`P`) and `input_forget=1` also throw.

## Image & geometry

| Op | Build & run | QEE | Gradient |
|---|---|---|---|
| AffineGrid | ✅ | 🟡 [1] | ✅ [2] |
| CenterCropPad | 🟡 [3] | ✅ | ✅ |
| Col2Im | ✅ [14] | 🟡 [1] | ✅ |
| GridSample | ✅ | 🟡 [1] | 🟡 [4] |
| ImageDecoder | 🟡 [13] | 🟡 [5] | N/A |
| NonMaxSuppression | ✅ | 🟡 [6] | N/A (index output) |
| Resize | ✅ [7] [15] | 🟡 [8] | 🟡 [9] |
| RoiAlign | ✅ | 🟡 [1] | 🟡 [10] |
| Upsample | ✅ [11] | 🟡 [1] | 🟡 [12] |

1. Shape/dtype only.
2. 2-D and 3-D grids, `align_corners` both ways.
3. Numeric tensors only; the spec allows all tensor types.
4. Bilinear with zeros padding only; nearest/bicubic and border/reflection padding
   throw. Gradients flow into both input and grid.
5. Shape/dtype only; output is rank-3 uint8.
6. Exact `[0, 3]` when `max_output_boxes_per_class` is absent or 0, otherwise rank
   plus an upper bound.
7. Negative `axes` entries (spec-legal since opset 18) run on every backend and in
   QEE. ONNX Runtime's kernel refuses them, so on that backend the `axes` rewrite
   in note 15 counts them from the front.
8. Full shape inference (scales/sizes, `axes`, `keep_aspect_ratio_policy`, all
   modes); values not computed.
9. Nearest mode with the asymmetric coordinate transform only; others throw.
10. Average mode with half-pixel transform only; max mode throws;
    `rois`/`batch_indices` get no gradient.
11. Deprecated; exported as an equivalent Resize.
12. Nearest mode only.
13. ONNX Runtime has no `ImageDecoder` kernel, so session creation refuses the
    model. It runs on the [PyTorch backend](pytorch-backend.md); the
    [JAX backend](jax-backend.md) refuses it (its output shape depends on the
    decoded bytes).
14. ONNX Runtime computes `Col2Im` over one spatial axis wrongly. Its backend
    rewrites such a call when it builds a session, for built and imported graphs
    alike, as a two-axis `Col2Im` (second extent 1) plus `Squeeze`; the graph, its
    export, generated C# and `.srk` keep the one-axis `Col2Im`
    ([#381](https://github.com/Shorokoo/Shorokoo/issues/381)).
15. ONNX Runtime's Resize kernel departs from the spec in the cases below, and its
    backend rewrites the call when it builds a session, for built and imported
    graphs alike; the graph, its export, generated C# and `.srk` keep the `Resize`
    as written.
    - `tf_crop_and_resize` along an axis whose length is unchanged: ONNX Runtime
      ignores that axis's `roi`. The axis is resized to `2L−1` under the same
      `roi` and every second element kept, which gives the spec's coordinates
      exactly ([#380](https://github.com/Shorokoo/Shorokoo/issues/380)).
    - A cubic rank-4 `tf_crop_and_resize` whose scales for axes 0 and 3 are 1 and
      for axis 1 is not: ONNX Runtime's channels-last route places out-of-roi
      `extrapolation_value`s at the wrong elements. The input is regrouped so the
      resize runs over its last two axes, and the result regrouped back. A
      `keep_aspect_ratio_policy` of `not_larger` or `not_smaller` is kept, over
      the regrouped positions of the axes it names; one that names axis 0 or 3
      never takes the channels-last route and is left as it stands
      ([#421](https://github.com/Shorokoo/Shorokoo/issues/421)).
    - A `Resize` with an `axes` attribute: ONNX Runtime's transpose optimizer
      moves a `Transpose` through it as though its `roi`, `scales` and `sizes`
      held one entry per input axis, so the session fails, or reorders them
      wrongly when the named axes are all of them out of order; its kernel also
      refuses negative axes. The call is written out over every axis, each axis
      it does not name taking scale 1, its own extent as size and the `roi`
      `[0, 1]`. A `not_larger` or `not_smaller` policy over a subset of the axes
      takes its common scale over those axes alone, which no call over every axis
      reproduces, so such a call keeps its axes, counted from the front, and reads
      its input through `OptionalGetElement(Optional(x))` — the same tensor,
      which the transpose optimizer cannot move a `Transpose` through
      ([#429](https://github.com/Shorokoo/Shorokoo/issues/429)).

## Random

| Op | Build & run | QEE | Gradient |
|---|---|---|---|
| Bernoulli | ✅ | 🟡 [1] | N/A [2] |
| Multinomial | ✅ | 🟡 [1] | N/A [2] |
| RandomNormal | ✅ | 🟡 [1] | N/A (leaf) |
| RandomNormalLike | ✅ | 🟡 [1] | N/A [2] |
| RandomUniform | ✅ | 🟡 [1] | N/A (leaf) |
| RandomUniformLike | ✅ | 🟡 [1] | N/A [2] |

1. Shape/dtype only. At execution, two nodes with the same seed produce identical
   streams.
2. No reparameterization path; gradients stop here.

## Sequences & optional

| Op | Build & run | QEE | Gradient |
|---|---|---|---|
| ConcatFromSequence | ✅ | ✅ | ✅ |
| Optional | ✅ | ✅ | ✅ |
| OptionalGetElement | 🟡 [1] | ✅ | ✅ |
| OptionalHasElement | 🟡 [1] | ✅ | N/A (bool output) |
| SequenceAt | ✅ | ✅ | ✅ |
| SequenceConstruct | ✅ | ✅ | ✅ |
| SequenceEmpty | ✅ | ✅ | N/A (leaf) |
| SequenceErase | ✅ | ✅ | ✅ |
| SequenceInsert | ✅ | ✅ | ✅ |
| SequenceLength | ✅ | ✅ | N/A (int64 output) |
| SequenceMap | ❌ [2] | N/A | N/A |
| SplitToSequence | 🟡 [3] | ✅ | ✅ [4] |

1. In-framework construction takes optional-typed inputs only; the opset-18 plain
   tensor/sequence input form is handled on import.
2. Cannot be imported; see [limitations.md](limitations.md). Use an explicit Loop
   over `SequenceLength`.
3. ONNX Runtime applies `keepdims` even when a `split` input is given (the spec
   ignores it then) and fails on chunk extents ≠ 1; QEE follows the spec.
4. The `split` input is non-differentiable.

## Strings & text

| Op | Build & run | QEE | Gradient |
|---|---|---|---|
| RegexFullMatch | ✅ | 🟡 [1] | N/A |
| StringConcat | ✅ | 🟡 [1] | N/A |
| StringNormalizer | ✅ | 🟡 [1] | N/A |
| StringSplit | ✅ | 🟡 [1] | N/A |
| TfIdfVectorizer | ✅ | 🟡 [2] | N/A |

String operators are non-differentiable.

1. Shape/dtype only, including data-dependent split extents bounded by
   `maxsplit`.
2. Shape/dtype only.

## Signal

| Op | Build & run | QEE | Gradient |
|---|---|---|---|
| BlackmanWindow | ✅ | ✅ | N/A [1] |
| DFT | ✅ | ✅ | 🟡 [2] |
| HammingWindow | ✅ | ✅ | N/A [1] |
| HannWindow | ✅ | ✅ | N/A [1] |
| MelWeightMatrix | ✅ | 🟡 [3] | N/A [1] |
| STFT | 🟡 [4] | 🟡 [3] | ✅ [5] |

1. Integer size/parameter inputs: non-differentiable.
2. Forward, inverse and `onesided` supported; a `dft_length` that pads or
   truncates the axis is not handled in the backward.
3. Shape/dtype only.
4. The signal must be rank-3 `[batch, length, 1|2]`; the rank-2 real-signal form
   is not representable.
5. Gradients for signal and window, including the windowless
   `frame_length`-driven form.

## Control flow

| Op | Build & run | QEE | Gradient |
|---|---|---|---|
| If | ✅ | ✅ | ✅ [1] |
| Loop | ✅ | 🟡 [2] | 🟡 [3] |
| Scan | ❌ [4] | ❌ [4] | ❌ [4] |

1. Gradients flow through both branches; the condition is non-differentiable.
2. Values for statically known trip counts; otherwise per-iteration shapes are
   merged conservatively.
3. Static trip counts are unrolled and differentiate normally; dynamic ones throw
   `AutoDiffNotSupportedException`; see [limitations.md](limitations.md).
4. A model containing `Scan` is rejected at import; see
   [limitations.md](limitations.md). Use an explicit `Loop` (in Shorokoo,
   `LoopAPI` with `ctx.Scan`).

## Shorokoo-specific operators

Shorokoo graphs also contain internal, non-ONNX operators (dynamic-geometry
convolution, hyperparameter-driven initializer samplers, `If`/`Loop` body markers,
training and sequence plumbing). All are lowered or removed before ONNX export,
and they may appear in graph dumps under names such as `ShrkConv` or `LoopOpen`.

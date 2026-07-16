# Operator support matrix

Shorokoo supports the standard `ai.onnx` domain up to **opset 26** — the
maximum implemented by the bundled ONNX Runtime 1.26. Exported models are
stamped at the **opset-21 baseline**; the exporter then auto-raises
each model's opset stamp only as far as the post-21 operators actually
present in the graph require (e.g. a graph containing `RMSNormalization` is
stamped opset 23; `Attention` is stamped opset 24 because ONNX Runtime's CPU
kernel registers at 24+). See [limitations.md](limitations.md) for why the baseline stays
at 21. Every supported operator is listed below — the full opset-21 set plus
the post-21 additions (`Attention`, `RMSNormalization`, `RotaryEmbedding`
at opset 23; `Swish`, `TensorScatter` at opset 24; `BitCast`, `CumProd` at
opset 26) — grouped by family and alphabetical within each family. The three
columns mean:

- **Build & run** — the operator can be constructed in a Shorokoo graph
  (definition covers the spec's inputs/outputs/attributes) and executes on the
  ONNX Runtime backend. Footnotes flag spec-legal corners that are restricted
  in-framework or by ONNX Runtime's CPU kernels.
- **QEE** — the Quick Execution Engine propagates output **dtype and shape**
  for every supported operator, and concrete **values** for small tensors (see
  [limitations.md](limitations.md) for the size bound). 🟡 here means values
  are not (or only partially) computed; shape/dtype inference still works.
- **Gradient** — reverse-mode autodiff. Unsupported attribute combinations on
  partially supported operators throw `AutoDiffNotSupportedException`.

Symbols: ✅ full support · 🟡 partial (see the family's notes) · ❌ not
supported (see notes) · N/A not applicable (non-differentiable output, leaf
operator, or operator not available).

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
| BitCast | ✅ | ✅ | N/A (bit reinterpretation) |
| Cast | ✅ | 🟡 [1] | ✅ [2] |
| CastLike | ✅ | 🟡 [1] | ✅ [2] |
| Ceil | ✅ | ✅ | N/A [3] |
| Celu | ✅ | ✅ | ✅ |
| Clip | ✅ | ✅ | 🟡 [4] |
| Cos | ✅ | ✅ | ✅ |
| Cosh | ✅ | ✅ | ✅ |
| CumProd | ✅ | ✅ | ✅ |
| CumSum | ✅ | ✅ | ✅ |
| Div | ✅ | ✅ | ✅ |
| Elu | ✅ | ✅ | ✅ |
| Erf | ✅ | 🟡 [5] | ✅ |
| Exp | ✅ | ✅ | ✅ |
| Floor | ✅ | ✅ | N/A [3] |
| Gelu | ✅ | ✅ | ✅ [6] |
| HardSigmoid | ✅ | ✅ | ✅ |
| HardSwish | ✅ | ✅ | ✅ |
| Hardmax | ✅ | ✅ | N/A [7] |
| LeakyRelu | ✅ | ✅ | ✅ |
| Log | ✅ | ✅ | ✅ |
| LogSoftmax | ✅ | ✅ | ✅ |
| Max | ✅ | ✅ | ✅ [8] |
| Mean | ✅ | ✅ | ✅ |
| Min | ✅ | ✅ | ✅ [8] |
| Mish | ✅ | ✅ | ✅ |
| Mod | ✅ | ✅ | ✅ [9] |
| Mul | ✅ | ✅ | ✅ |
| Neg | ✅ | ✅ | ✅ |
| PRelu | 🟡 [10] | ✅ | ✅ |
| Pow | 🟡 [11] | ✅ | ✅ [12] |
| Reciprocal | ✅ | ✅ | ✅ |
| Relu | 🟡 [13] | ✅ | ✅ |
| Round | ✅ | ✅ | N/A [3] |
| Selu | ✅ | ✅ | ✅ |
| Shrink | ✅ | ✅ | ✅ |
| Sigmoid | ✅ | ✅ | ✅ |
| Sign | ✅ | ✅ | N/A [3] |
| Sin | ✅ | ✅ | ✅ |
| Sinh | ✅ | ✅ | ✅ |
| Softmax | ✅ | ✅ | ✅ |
| Softplus | ✅ | ✅ | ✅ |
| Softsign | ✅ | ✅ | ✅ |
| Sqrt | ✅ | ✅ | ✅ |
| Sub | ✅ | ✅ | ✅ |
| Sum | ✅ | ✅ | ✅ |
| Swish | 🟡 [14] | ✅ | ✅ |
| Tan | ✅ | ✅ | ✅ |
| Tanh | ✅ | ✅ | ✅ |
| ThresholdedRelu | ✅ | ✅ | ✅ |

1. QEE stores values in float32/int64 storage, so narrowing-integer wrap and
   float16/bfloat16 rounding are not modeled in QEE values (real rounding
   happens at execution); string/complex/int4 casts propagate dtype only.
2. Gradient is cast back to the source dtype; integer-target rounding is
   ignored (straight-through-estimator convention).
3. Piecewise-constant output — zero gradient everywhere it is defined.
4. Gradient flows to the data input only; the optional `min`/`max` inputs are
   treated as constants.
5. QEE values for float inputs only; integer inputs propagate shape/dtype.
6. Both `approximate="none"` (exact erf) and `approximate="tanh"` forms are
   matched by the corresponding derivative.
7. One-hot output; zero gradient by convention.
8. Ties share the gradient equally.
9. Float inputs get the almost-everywhere derivative (d/da = 1, d/db = −q with
   the fmod-consistent quotient); integer Mod is piecewise constant.
10. Float tensors only; the spec also allows (u)int32/64 inputs.
11. Float base only; the spec also allows int32/int64 bases.
12. The exponent's gradient term uses ln(base) and is NaN for base ≤ 0
    (standard caveat; harmless when the exponent is a constant).
13. Float tensors only; the spec also allows signed integers since opset 14.
14. ONNX Runtime 1.26 ships no `Swish` kernel on any execution provider, so
    Swish graphs execute through the Quick Execution Engine only. Equivalent
    ORT-executable form: `x * Sigmoid(alpha * x)` built from primitives.

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
| Where | 🟡 [2] | ✅ | ✅ [3] |
| Xor | ✅ | ✅ | N/A |

All boolean/integer outputs are non-differentiable, hence N/A gradients.

1. Unsigned integer tensors only; the spec also allows signed integers.
2. ONNX Runtime's CPU provider has no bool-element `Where` kernel; selecting
   between bool-valued branches works in QEE only.
3. The condition input is non-differentiable; both value branches get
   broadcast-aware gradients.

## Reductions

| Op | Build & run | QEE | Gradient |
|---|---|---|---|
| ArgMax | ✅ | ✅ | N/A [1] |
| ArgMin | ✅ | ✅ | N/A [1] |
| ReduceL1 | ✅ | ✅ | ✅ |
| ReduceL2 | ✅ | ✅ | ✅ |
| ReduceLogSum | ✅ | ✅ | ✅ |
| ReduceLogSumExp | ✅ | ✅ | ✅ |
| ReduceMax | ✅ | ✅ | ✅ [2] |
| ReduceMean | ✅ | ✅ | ✅ |
| ReduceMin | ✅ | ✅ | ✅ [2] |
| ReduceProd | ✅ | ✅ | ✅ [3] |
| ReduceSum | ✅ | ✅ | ✅ |
| ReduceSumSquare | ✅ | ✅ | ✅ |

1. Integer index output — non-differentiable.
2. Ties share the gradient equally.
3. The gradient uses prod/x and is NaN when an element is exactly 0.

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
| Range | ✅ | ✅ | N/A [2] |
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
| TensorScatter | ✅ | 🟡 [16] | ❌ [17] |
| Tile | ✅ | ✅ | ✅ |
| TopK | ✅ | ✅ [14] | ✅ |
| Transpose | ✅ | ✅ | ✅ |
| Trilu | ✅ | ✅ | ✅ |
| Unique | ✅ | 🟡 [15] | ✅ |
| Unsqueeze | ✅ | ✅ | ✅ |

1. The `sparse_value` attribute is unsupported (no in-framework sparse-tensor
   representation); all dense value variants work.
2. Index/shape/count output — non-differentiable.
3. Float tensors only; the spec allows all tensor types.
4. A negative `axis` requires a statically known input rank.
5. `batch_dims = 0` only; `batch_dims > 0` throws.
6. Signed numeric tensors only; the spec also allows bool/unsigned inputs.
7. Output extent is data-dependent: exact shape and values when the input data
   is known to QEE, rank-only otherwise.
8. Indices and depth are non-differentiable; the values pair is treated as
   constant.
9. Constant mode only; reflect/edge/wrap modes throw.
10. Numeric tensors only; the spec allows all tensor types.
11. Deprecated in ONNX (since opset 11) and not implemented — use
    ScatterElements instead.
12. Reductions `none`/`add` only; `mul`/`min`/`max` throw.
13. Exact whenever a `steps` input is wired (any stride, including negative);
    the faster path used when `steps` is absent retains an approximate
    clamping of negative starts/ends.
14. Values computed for small tensors honoring `largest` (ties resolved to the
    lower index); when `k` is wired but unknown the shape degrades to a bounded
    rank-only claim.
15. Flatten form computes all four outputs for small tensors (both `sorted`
    modes); the `axis` form is shape-only and its unique-count extent stays
    data-dependent (rank-only with bounds).
16. Shape/dtype inference only; values are not computed.
17. Gradient is not implemented; differentiation raises
    `AutoDiffNotSupportedException` (error code `AD003`).

## Convolution & pooling

| Op | Build & run | QEE | Gradient |
|---|---|---|---|
| AveragePool | ✅ | 🟡 [1] | 🟡 [2] |
| Conv | ✅ | 🟡 [1] | 🟡 [3] |
| ConvTranspose | ✅ | 🟡 [1] | 🟡 [4] |
| DeformConv | ✅ | 🟡 [1] | ❌ [5] |
| GlobalAveragePool | ✅ | 🟡 [1] | ✅ |
| GlobalLpPool | ✅ | 🟡 [1] | ✅ |
| GlobalMaxPool | ✅ | 🟡 [1] | ✅ |
| LpPool | ✅ | 🟡 [1] | 🟡 [6] |
| MaxPool | ✅ | 🟡 [1] | 🟡 [7] |
| MaxRoiPool | ✅ | 🟡 [1] | 🟡 [8] |
| MaxUnpool | ✅ | 🟡 [9] | ✅ |

1. Shape/dtype inference only — values are never computed for these heavy
   operators; use the ONNX Runtime backend for numbers.
2. `ceil_mode=1` throws; everything else (count_include_pad, SAME auto_pad,
   dilations, overlapping windows) is supported.
3. Backward requires explicit pads (`auto_pad` SAME_UPPER/SAME_LOWER is not
   resolved); the weight gradient is 2-D only. Grouped convolutions are fully
   supported.
4. As Conv (2-D-only weight gradient, no SAME auto_pad, grouped supported,
   `output_padding` handled); the `output_shape` attribute is ignored in the
   backward.
5. The bilinear-sampling adjoint is not implemented; differentiation throws.
6. Backward ignores `ceil_mode`, `dilations`, and `auto_pad`.
7. Exact for every attribute combination except `storage_order=1` (throws);
   ties route the gradient to the first maximum.
8. Recompute-and-mask approximation (deprecated operator); the `rois` input
   gets no gradient.
9. Shape comes from the `output_shape` input's values when known; element
   values are not computed.

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
| RMSNormalization | ✅ | ✅ | ✅ |
| SoftmaxCrossEntropyLoss | ✅ | 🟡 [5] | ✅ |

1. With `training_mode=1` the node is decomposed into primitive ops at export
   time, producing the same results as the fused kernel.
2. Values in inference mode; training mode is shape/dtype only.
3. Both inference and training modes (batch-stats backward); gradients flowing
   into the running-mean/running-var outputs throw.
4. Inference mode is fully supported. TRAINING-mode Dropout is not supported:
   ONNX Runtime's constant folding may evaluate a training-mode Dropout whose
   data input is provably constant as the inference-mode identity (a silent
   no-drop mask) at session load, and Shorokoo does not guard against it.
   Shorokoo's own dropout layers do not emit the op — masks are built in-graph
   from the keyed RNG feed (see rng-configuration.md) — so this only affects
   hand-authored graphs. The gradient path (mask-based) throws if the forward
   mask is unavailable.
5. Shape/dtype inference only; values are not computed.
6. Gradients into the optional Mean/InvStdDev outputs are treated as zero.

## MatMul & linear algebra

| Op | Build & run | QEE | Gradient |
|---|---|---|---|
| Attention | ✅ | 🟡 [1] | ❌ [2] |
| Det | ✅ | 🟡 [3] | ✅ |
| Einsum | ✅ | 🟡 [4] | 🟡 [5] |
| Gemm | ✅ | ✅ | ✅ |
| MatMul | ✅ | ✅ | ✅ |
| RotaryEmbedding | ✅ | 🟡 [1] | ❌ [2] |

1. Shape/dtype inference only; values are not computed.
2. Gradient is not implemented; differentiation raises
   `AutoDiffNotSupportedException` (error code `AD003`).
3. Shape/dtype only; determinant values are not computed.
4. Shape is inferred from the equation (matmul/transpose/reduce/diagonal
   forms; exotic equations degrade to unknown shape); values are not computed.
5. Repeated subscripts within a single operand (e.g. `"ii->i"`) are
   unsupported; ellipsis is supported.

## Quantization

| Op | Build & run | QEE | Gradient |
|---|---|---|---|
| ConvInteger | 🟡 [1] | 🟡 [2] | N/A [3] |
| DequantizeLinear | ✅ | ✅ | N/A [3] |
| DynamicQuantizeLinear | ✅ | ✅ | N/A [3] |
| MatMulInteger | 🟡 [1] | ✅ | N/A [3] |
| QLinearConv | ✅ | 🟡 [2] | N/A [3] |
| QLinearMatMul | ✅ | 🟡 [2] | N/A [3] |
| QuantizeLinear | ✅ | ✅ | N/A [3] |

1. The weight zero point is scalar only; a per-channel 1-D `w_zero_point` is
   not representable in-framework.
2. Shape/dtype only; values are not computed.
3. Quantized operators are non-differentiable — no straight-through estimator
   is provided, so there is no quantization-aware training path.

## Recurrent (RNN family)

| Op | Build & run | QEE | Gradient |
|---|---|---|---|
| GRU | 🟡 [1] | 🟡 [2] | 🟡 [3] |
| LSTM | 🟡 [1] | 🟡 [2] | 🟡 [3] [4] |
| RNN | 🟡 [1] | 🟡 [2] | 🟡 [3] |

1. ONNX Runtime's CPU recurrent kernels require the `hidden_size` attribute
   (optional per spec) and reject `layout=1`; those spec-legal forms still
   build and get full QEE shape inference, but cannot execute on the CPU
   provider.
2. Shape/dtype inference only (including bidirectional shapes and the
   hidden-size fallback chain); step-by-step values are not computed.
3. Forward **and** reverse direction, single-direction only, default
   activations, `layout=0`; bidirectional, custom activations, `clip`,
   `layout=1`, and a wired `sequence_lens` input throw.
4. LSTM additionally: peephole weights (`P`) and `input_forget=1` throw.

## Image & geometry

| Op | Build & run | QEE | Gradient |
|---|---|---|---|
| AffineGrid | ✅ | 🟡 [1] | ✅ [2] |
| CenterCropPad | 🟡 [3] | ✅ | ✅ |
| Col2Im | ✅ | 🟡 [1] | ✅ |
| GridSample | ✅ | 🟡 [1] | 🟡 [4] |
| ImageDecoder | ✅ | 🟡 [5] | N/A |
| NonMaxSuppression | ✅ | 🟡 [6] | N/A (index output) |
| Resize | 🟡 [7] | 🟡 [8] | 🟡 [9] |
| RoiAlign | ✅ | 🟡 [1] | 🟡 [10] |
| Upsample | ✅ [11] | 🟡 [1] | 🟡 [12] |

1. Shape/dtype inference only; values are not computed.
2. Both 2-D and 3-D grids, `align_corners` both ways.
3. Numeric tensors only; the spec allows all tensor types.
4. Bilinear interpolation with zeros padding only; nearest/bicubic modes and
   border/reflection padding throw. Gradients flow into both the input and
   the grid.
5. Shape/dtype only (no image codec in QEE); output is rank-3 uint8.
6. The selected count is data-dependent: exact `[0, 3]` when
   `max_output_boxes_per_class` is absent or 0, otherwise rank plus an upper
   bound on the output extent.
7. ONNX Runtime's CPU Resize kernel rejects negative `axes` entries (allowed
   by spec since opset 18); QEE handles them.
8. Full shape inference (scales/sizes precedence, `axes`,
   `keep_aspect_ratio_policy`, all modes); interpolated values are not
   computed.
9. Nearest mode with the asymmetric coordinate transform only; linear/cubic
   modes and other coordinate transforms throw.
10. Average mode with the half-pixel coordinate transform only; max mode
    throws; `rois`/`batch_indices` get no gradient.
11. Deprecated operator; supported by lowering to an equivalent Resize at
    export.
12. Nearest mode only; other modes throw.

## Random

| Op | Build & run | QEE | Gradient |
|---|---|---|---|
| Bernoulli | ✅ | 🟡 [1] | N/A [2] |
| Multinomial | ✅ | 🟡 [1] | N/A [2] |
| RandomNormal | ✅ | 🟡 [1] | N/A (leaf) |
| RandomNormalLike | ✅ | 🟡 [1] | N/A [2] |
| RandomUniform | ✅ | 🟡 [1] | N/A (leaf) |
| RandomUniformLike | ✅ | 🟡 [1] | N/A [2] |

1. Shape/dtype only — random values are inherently not computable ahead of
   execution. Seeded determinism works at execution time: two nodes with the
   same seed produce identical streams.
2. Sampling has no reparameterization path; gradients stop here.

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

1. In-framework construction accepts optional-typed inputs only; the opset-18
   plain tensor/sequence input form is handled when importing ONNX models.
2. Cannot be imported — see [limitations.md](limitations.md). Express the
   mapping as an explicit Loop over `SequenceLength` instead.
3. ONNX Runtime's kernel deviates from spec: it applies `keepdims` even when a
   `split` input is given (the spec says it is ignored then) and fails on
   chunk extents ≠ 1. QEE follows the spec.
4. The `split` input is non-differentiable.

## Strings & text

| Op | Build & run | QEE | Gradient |
|---|---|---|---|
| RegexFullMatch | ✅ | 🟡 [1] | N/A |
| StringConcat | ✅ | 🟡 [1] | N/A |
| StringNormalizer | ✅ | 🟡 [1] | N/A |
| StringSplit | ✅ | 🟡 [1] | N/A |
| TfIdfVectorizer | ✅ | 🟡 [2] | N/A |

String operators are non-differentiable, hence N/A gradients.

1. Shape/dtype only: string element values never enter QEE (variable-length
   text has no fixed-width value storage); shapes — including the
   data-dependent split extents, bounded by `maxsplit` — are inferred.
2. Shape/dtype only; the term-frequency values are not computed.

## Signal

| Op | Build & run | QEE | Gradient |
|---|---|---|---|
| BlackmanWindow | ✅ | ✅ | N/A [1] |
| DFT | ✅ | ✅ | 🟡 [2] |
| HammingWindow | ✅ | ✅ | N/A [1] |
| HannWindow | ✅ | ✅ | N/A [1] |
| MelWeightMatrix | ✅ | 🟡 [3] | N/A [1] |
| STFT | 🟡 [4] | 🟡 [3] | ✅ [5] |

1. All inputs are integer sizes/parameters — non-differentiable.
2. Forward, inverse, and `onesided` transforms are supported; a `dft_length`
   that pads or truncates the transform axis is not handled in the backward.
3. Shape/dtype only; values are not computed.
4. The signal must be rank-3 `[batch, length, 1|2]`; the spec's rank-2
   real-signal form is not representable in-framework.
5. Full overlap-add adjoint for both the signal and the window, including the
   windowless `frame_length`-driven form.

## Control flow

| Op | Build & run | QEE | Gradient |
|---|---|---|---|
| If | ✅ | ✅ | ✅ [1] |
| Loop | ✅ | 🟡 [2] | 🟡 [3] |
| Scan | 🟡 [4] | 🟡 [2] | 🟡 [3] |

1. Gradients are routed through both branches; the condition is
   non-differentiable.
2. Values are computed for statically known trip counts; with unknown bounds,
   per-iteration output shapes are merged conservatively.
3. Loops with a statically known trip count are unrolled and differentiate
   normally; dynamic trip counts throw `AutoDiffNotSupportedException` — see
   [limitations.md](limitations.md).
4. Imported Scan nodes are lowered to an equivalent Loop. Any
   `scan_input_axes`/`scan_input_directions` are supported; non-zero
   `scan_output_axes`, reverse `scan_output_directions`, and the opset-8 form
   are rejected at import — see [limitations.md](limitations.md). QEE and
   gradients then behave exactly as for Loop.

## Shorokoo-specific operators

These internal operators appear in Shorokoo graphs but are not part of the
ONNX standard; they are lowered or rewritten before ONNX export.

- **ShrkConv** (internal) — convolution with dynamic geometry: pads, strides,
  and dilations arrive as runtime inputs instead of attributes; lowered to a
  standard `Conv` before autodiff and export.
- **ShrkRandomNormal / ShrkRandomUniform** (internal) — hyperparameter-driven
  initializer samplers; gradient leaves by construction.
- **IfOpen/IfClose, LoopOpen/LoopClose, LoopFakeInput, LoopIndexVariable,
  LoopScanVariable** (internal) — the open/close node pairs that represent
  `If` and `Loop` bodies inside Shorokoo graphs.
- **StateUpdateLink, WithStateDeps, TrainableParamIdRef** (internal) —
  structural sentinels used by the training pipeline; non-differentiable
  pass-throughs.
- **SequenceConcat, SequenceSlice** (internal) — sequence plumbing used by
  loop lowering; non-differentiable structural ops.

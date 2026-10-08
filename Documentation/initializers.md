# Initializers

The ready-made initializers of `Shorokoo.Modules`, trainable scalars, and writing an initializer of
your own.

Related: [nn-library.md](nn-library.md) · [defining-models.md](defining-models.md) · [rng-configuration.md](rng-configuration.md)

## Initializers (`Shorokoo.Modules.Initializers`)

All are `[TrainableParamInitializer]`s taking the parameter's shape first:
`Zeros.Init([outFeatures])`, `KaimingUniform.Init([outC, inC, k, k])`. The `Scalar*`
entries take **no shape** and create a rank-0 parameter ([Trainable scalars](#trainable-scalars)).
Extra arguments are `Init` args after the shape, e.g. `UniformRange.Init([shape], Scalar(lo), Scalar(hi))`.

| Initializer | Fills with | Notes |
|---|---|---|
| `Zeros` | 0.0 | biases, BatchNorm beta |
| `Ones` | 1.0 | BatchNorm/LayerNorm gamma |
| `Constant` | `value` | deterministic; any rank; `Constant.Init([shape], Scalar(v))`; PyTorch `constant_` |
| `ScalarZeros` | 0.0, rank 0 | `ScalarZeros.Init()`; trainable counterpart of `OptimizerScalarZeros` |
| `ScalarOnes` | 1.0, rank 0 | `ScalarOnes.Init()`; trainable counterpart of `OptimizerScalarOnes` |
| `ScalarConstant` | `value`, rank 0 | `ScalarConstant.Init(Scalar(v))` |
| `Uniform` | U(0, 1) | seeded |
| `Normal` | N(0, 1) | seeded; PyTorch's `nn.Embedding` default |
| `UniformRange` | U(low, high) | seeded; any rank; args `low`, `high` (expects `low ≤ high`); exact at any width, `high` never returned ([uniform-draws.md](uniform-draws.md)); PyTorch `uniform_(a, b)` |
| `NormalDist` | N(mean, std) | seeded; any rank; args `mean`, `std` (expects `std ≥ 0`); PyTorch `normal_(mean, std)` |
| `XavierUniform` | U(−a, a), a = √(6 / (fanIn + fanOut)) | gain 1; seeded; rank ≥ 2 |
| `XavierNormal` | N(0, √(2 / (fanIn + fanOut))) | gain 1; seeded; rank ≥ 2 |
| `KaimingUniform` | U(−b, b), b = √(6 / fanIn) | ReLU gain √2, fan-in; seeded; rank ≥ 2; default weight init of `Linear` and the conv layers |
| `KaimingNormal` | N(0, √(2 / fanIn)) | ReLU gain √2, fan-in; seeded; rank ≥ 2 |
| `XavierUniformGain` | U(−a, a), a = gain·√(6 / (fanIn + fanOut)) | seeded; rank ≥ 2; arg `gain` (the `calculate_gain` multiplier, computed by you); = `XavierUniform` at gain 1 |
| `XavierNormalGain` | N(0, gain·√(2 / (fanIn + fanOut))) | seeded; rank ≥ 2; arg `gain`; = `XavierNormal` at gain 1 |
| `KaimingUniformGain` | U(−b, b), b = gain·√(3 / fanIn) | seeded; rank ≥ 2; arg `gain`; = `KaimingUniform` at gain √2. **Base factor is √(3 / fanIn), not √(6 / fanIn)** |
| `KaimingNormalGain` | N(0, gain·√(1 / fanIn)) | seeded; rank ≥ 2; arg `gain`; = `KaimingNormal` at gain √2. **Base factor is √(1 / fanIn), not √(2 / fanIn)** |
| `TruncatedNormal` | N(0, 1) clamped to [−2, 2] | seeded; clamp approximation, not rejection sampling |
| `LeCunNormal` | N(0, √(1 / fanIn)) | seeded; rank ≥ 2; JAX/Flax `lecun_normal` |
| `Orthogonal` | (semi-)orthogonal matrix | seeded; rank ≥ 2; gain 1; **approximate**: 15 Newton–Schulz iterations from a Gaussian, not exact QR; PyTorch `orthogonal_` |
| `RecurrentUniform` | U(−1/√H, 1/√H) | seeded; PyTorch RNN/LSTM/GRU default; arg `H` (`Scalar(hiddenSize)`), not read from the shape, so gate-stacked `4H`/`3H` axes keep the `1/√H` bound; used by the `Recurrent` layers |

- **Seeded determinism**: each random initializer draws from its own stream, derived
  from the model's [RNG configuration](rng-configuration.md) (master seed 0 by default)
  and the parameter's place in the model. Materialization is reproducible for a config,
  and two same-shaped parameters of the same class get **distinct** values. Bind a
  different master seed to re-roll everything; no seed appears in the model definition.
- **Uniform initializers** (`Uniform`, `UniformRange`, `XavierUniform`,
  `KaimingUniform`, `RecurrentUniform`, `XavierUniformGain`, `KaimingUniformGain`) fill
  from the half-open `[low, high)` with the guarantees in [uniform-draws.md](uniform-draws.md).
- **Normal initializers** (`Normal`, `NormalDist`, the Xavier/Kaiming normals and
  `LeCunNormal`) scale a standard N(0, 1) draw by the std in `float32`. No value exceeds
  `8·std` (`mean ± 8·std` for `NormalDist`). [normal-draws.md](normal-draws.md)'s
  bit-exactness covers the draw only; fan-scaled initializers can differ in the last ulp
  between providers. `TruncatedNormal` and `Orthogonal` do not have the `8·std` bound.
- **Fan-in/fan-out**: `fanIn = prod(shape) / shape[0]`, `fanOut = prod(shape) / shape[1]`
  (PyTorch convention for `[out, in]` and `[outC, inC/g, k...]`). Hence rank ≥ 2: use
  `Zeros`/`Ones`/`Uniform`/`Normal` for biases.

<a id="trainable-scalars"></a>
### Trainable scalars (rank 0)

A learned scalar — a temperature, a residual scale, a `gamma` — is a rank-0 parameter:

```csharp
var gamma = ScalarOnes.Init();                        // seeded at 1: starts as a no-op
var beta  = ScalarZeros.Init();                       // seeded at 0
var temp  = ScalarConstant.Init(Scalar(0.125f));      // seeded at 1/√d
return x * gamma + beta;                              // broadcasts against any shape
```

Do **not** use `Ones.Init([Scalar(1L)])` instead: it persists in the checkpoint as a
`[1]`-shaped parameter. The `Scalar*` initializers persist as rank 0. The optimizer-state
equivalents are `OptimizerScalarZeros` / `OptimizerScalarOnes`
([Optimizers](losses-and-optimizers.md#optimizers-shorokoomodulesoptimizers)).

**Writing your own.** An initializer states its parameter's shape either as its **first**
`Inline` parameter (a shape vector) or, for rank 0, as its `Scalar<T>` return type. Both
misuses are refused by name at the `Init` call, even on a branch a specialization removes: a
shape baked into the body of a no-argument `Inline`, and a shaped initializer whose first
argument is not an `int64` shape vector (`Init(wte)` handing it another parameter's value,
say). Any other input comes after the shape. An optimizer-owned `[StateInitializer]` is
exempt: the rig calls it once per trainable parameter, and the state takes whatever shape
it returns ([Custom optimizers](training-hyperparameters.md#custom-optimizers)). The body is an ordinary
graph body (tensor ops, loops, `IfElse`). The rules below apply to `[StateInitializer]` and
`[TrainableParamInitializer]` alike.

**It can call other initializers.** An `Init(...)` call inside an initializer body is
evaluated as a value; only the top-level initializer defines a parameter. Either kind may
call either kind:

```csharp
[TrainableParamInitializer]
public static partial class NormalDist02
{
    public static Tensor<float32> Inline(Vector<int64> shape)
        => NormalDist.Init(shape, Scalar(0f), Scalar(0.02f));
}
```

Draws are keyed on the parameter being created. Each draw site gets its own sub-stream,
and a site inside a `LoopAPI.Iterate` body draws a fresh sample on every iteration.

**It can start from another parameter's value.** A `Tensor<T>` input may be another
trainable parameter. Initializers run in dependency order, so the input is the value the
model starts from (re-drawing it inside the body would give a different matrix).

```csharp
[TrainableParamInitializer]
public static partial class ProductOf
{
    public static Tensor<float32> Inline(Vector<int64> shape, Tensor<float32> a, Tensor<float32> b)
        => a.MatMul(b);
}

Scalar<int64> vocab = Scalar(50257L), d = Scalar(384L);

var emb  = NormalDist02.Init([vocab, d]);
var wv   = NormalDist02.Init([d, d]);
var bank = ProductOf.Init([vocab, d], emb, wv);   // starts as emb · wv, for the emb the model has

return x.MatMul(emb).MatMul(wv) + x.MatMul(bank);   // every one of the three is read by the model
```

- The **shape** input must fold to a constant at the call site. (A rank-0 initializer
  has no shape input, so its first input may be a parameter.) Chains of dependencies work.
- A value **passed in** may be **computed from parameters** (`emb * Scalar(2f)`, or a
  module's output on one), as plain tensor arithmetic over parameters created outside any
  loop. A passed-in value computed through a loop or branch, over a per-iteration
  parameter, or that draws randomness is refused at concretization, naming the initializer.
- The initializer's **own body** may still draw. Each `Init(...)` call in it draws from its
  own sub-stream, keyed on the parameter being created, so two parameters built by the same
  initializer get different draws.
- A source the model reads **nowhere else**, or one created **inside a loop**, is refused
  by name. Create the source outside the loop and use it in the model, or fold its
  computation — a draw included — into the reading initializer's body.

A table that starts as `E · W`, with `E` the token embedding the model starts from and `W` a
fresh `N(0, 0.02)` matrix that is no parameter, draws `W` in the body:

```csharp
[TrainableParamInitializer]
public static partial class NormalValueBankInit
{
    public static Tensor<float32> Inline(Vector<int64> shape, Tensor<float32> tokenEmbedding)
    {
        var d = tokenEmbedding.DimTensor(1);
        return tokenEmbedding.MatMul(NormalDist.Init([d, d], Scalar(0f), Scalar(0.02f)));
    }
}

var wte   = NormalDist02.Init([vocab, d]);
var bank0 = NormalValueBankInit.Init([vocab, d], wte);   // wte · W0
var bank1 = NormalValueBankInit.Init([vocab, d], wte);   // wte · W1, a different W
```

Drawing `W` at the call site instead —
`wte.MatMul(RandomNormal([d, d], Scalar(0f), Scalar(0.02f)))` passed to an
initializer that returns its argument — is refused: that draw is part of the value passed in.

**It may not create or reference a model.** No `Foo.Model(...)`, no `Foo.Call(...)` of
any `[Module]` (even a parameter-free one), no `ModelSequence`, no
`IModel.GetTrainableParam`, no model hyperparameter read, no model-typed input. Such a
body is refused with **FW055**, naming the initializer, when the using module's graph (or
the calling initializer's body) is built. Compute layer outputs in the `[Module]` and pass
them in as `Tensor<T>` inputs.

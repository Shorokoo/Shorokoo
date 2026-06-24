# Shorokoo.Modules

Baseline neural-network library for [Shorokoo](https://github.com/Shorokoo/Shorokoo):
ready-made layers, loss functions, optimizers, and initializers built from Shorokoo modules.

- **Initializers** (`Shorokoo.Modules.Initializers`) — `Zeros`, `Ones`, `Uniform`,
  `Normal`, `XavierUniform`, `XavierNormal`, `KaimingUniform`, `KaimingNormal`,
  `TruncatedNormal`, `LeCunNormal`. All shape-only `[TrainableParamInitializer]`s;
  the random ones are seeded (deterministic), and Xavier/Kaiming/LeCun compute
  fan-in/fan-out in-graph from the shape vector.
- **Layers** (`Shorokoo.Modules.Layers`) — `Linear`, `Conv1d`, `Conv2d`, `Conv3d`
  (hyperparameter-driven geometry via the dynamic Conv lowering),
  `ConvTranspose2d` (default geometry, kernel inferred from the weight),
  `BatchNorm2d`/`BatchNorm1d` (training/eval flag, running stats via `StateUpdate`),
  `LayerNorm`, `RMSNorm`, `GroupNorm`, `InstanceNorm2d`, `Dropout` (training flag),
  `Embedding`, `MultiHeadAttention` / `TransformerEncoderLayer` (+ the
  `Attention.ScaledDotProductAttention` helper), `LeakyReLU`/`ELU` (hyper alpha),
  `PReLU` (learnable slope), and the `Pooling` / `GatedLinear.GLU` helpers
  (`MaxPool2d`, `AvgPool2d`, `GlobalAvgPool2d`, `GlobalMaxPool2d`, `Flatten`).
  Plain activations are tensor one-liners — `x.Relu()`, `x.Gelu()`,
  `x.Sigmoid()`, `x.Tanh()`, `x.Softmax(axis)` — and need no modules.
- **Losses** (`Shorokoo.Modules.Losses`) — `L2Loss` (MSE), `L1Loss`,
  `HuberLoss(delta)` / `SmoothL1Loss`, `CrossEntropyLoss` (logits + int64
  class indices), `NLLLoss`, `BCELoss`, `BCEWithLogitsLoss`, `KLDivLoss`
  (log-probs + probs). All map (predictions, targets) → scalar loss.
- **Optimizers** (`Shorokoo.Modules.Optimizers`) — `SGDOptimizer`,
  `SGDMomentumOptimizer`, `AdamOptimizer` (with bias correction),
  `AdamWOptimizer`, `RMSpropOptimizer`, `AdagradOptimizer`, with strongly
  typed hyperparameter sets and learning-rate schedules (`Schedules.*`).
  Optimizer state (moments, velocity, accumulators) is created inside each
  module via optimizer-owned `[StateInitializer]`s — `OptimizerStateZeros`
  (param-shaped) and `OptimizerScalarZeros` (a rank-0 scalar, e.g. Adam's
  timestep) — and threaded with `StateUpdate`; never declared in the `Inline`
  signature.

```bash
dotnet add package Shorokoo.Modules
```

```csharp
using Shorokoo.Modules.Optimizers;
using Shorokoo.Modules.Losses;

var rig = TrainingRig.FromScratch(
    MyModel.ComputationGraph,
    CrossEntropyLoss.ComputationGraph,
    AdamOptimizer.ComputationGraph,
    sampleInputs,
    new AdamOptimizerHyperparameters { LearningRate = 1e-3f });
```

Documentation: https://github.com/Shorokoo/Shorokoo

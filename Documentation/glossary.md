# Glossary

| Term | Meaning | Where |
|---|---|---|
| `IValue` | Base interface for any graph value (`Tensor<T>`, `Scalar<T>`, `Vector<T>`, sequences, optionals). | `core-types.md` |
| `Tensor<T>` / `Scalar<T>` / `Vector<T>` | Symbolic graph values of rank N / 0 / 1, generic over dtype marker `T`. | `core-types.md` |
| dtype marker (`float32`, `int64`, `bit`, …) | The generic type argument naming a tensor's element type. | `core-types.md` |
| `DType` | Runtime dtype descriptor (`DType.Float32`, …). | `core-types.md` |
| `TensorData` / `TensorData<T>` | Concrete materialized values (results, constants, weights). | `inference.md` |
| `AccessMemory()` | Reads a `TensorData<T>`'s values as a `ReadOnlySpan<primitive>`. | `core-types.md` |
| `IData` | Interface implemented by `TensorData`; the input type `ComputeContext.Execute` accepts. | `inference.md` |
| `[Module]` | Attribute marking a `partial class` whose `Inline` becomes a computation graph. | `defining-models.md` |
| `[Hyper]` | Attribute marking a scalar hyperparameter (bound on `Model(...)`, before the tensor inputs). An optional default — `[Hyper(0.9f)]` — makes the parameter omittable (the default is used when omitted) and seeds the generated optimizer hyperparameter set. | `defining-models.md` |
| `[TrainableParamInitializer]` / `[StateInitializer]` | Attributes for classes that produce trainable weights / non-trainable state. | `defining-models.md` |
| `Inline` | The `static` method the generator reads to build the graph. | `defining-models.md` |
| `Model` / `Call` / `ComputationGraph` | Generated members: bind hypers, run on inputs, get the full graph. | `defining-models.md` |
| `FastComputationGraph` | The internal graph representation that gets executed, exported, or trained. | `inference.md`, `onnx-and-weights.md` |
| `OnnxEngine.Eval` | One-shot: run a graph value and return `TensorData`. | `inference.md` |
| `ComputeContext` / `CompiledGraph` | Compile a graph once and run it many times. | `inference.md` |
| `QuickExecutionEngine` | CPU-only interpreter for debugging / shape inference (small tensors only). | `inference.md` |
| `LoopAPI.Iterate` | Build a graph loop over a `Scalar<int64>` count. | `defining-models.md` |
| `.IfElse(a, b)` | Data-dependent branch on a `Scalar<bit>`. | `defining-models.md` |
| `NN` | Static class of higher-level ops (`Conv`, `MaxPool`, `GlobalAveragePool`, …). | `core-types.md` |
| `Globals` | Static factory helpers (`Scalar`, `Vector`, `Tensor`, `TensorData`, `TensorFill`). | `core-types.md` |
| `Globals.StateUpdate` | Register a state mutation (optimizer/BatchNorm state) inside a module. | `training.md` |
| `TrainingRig` | Entry point that composes model+loss+optimizer and runs autodiff. | `training.md` |
| `TrainingCheckpoint` | Holds trainable params, model state, optimizer state, and the global `Step` (advances each `TrainStep`; schedules resume from it). | `training.md` |
| `HyperValue` | An optimizer hyperparameter's value: a baked `float`, a `Schedule`, or `HyperValue.Runtime(seed)`; its kind decides constant-vs-runtime wiring. | `training.md` |
| `Schedule` / `Schedules` | A `step → value` schedule (`Schedules.Cosine`, `OneCycle`, …) with fluent combinators (`WithWarmup`, `Then`, `Scale`, `Clamp`, `Shift`, `PerEpoch`). | `training.md` |
| `IOptimizerHyperparameters` / `<Optimizer>Hyperparameters` | The named, defaulted hyperparameter set; source-generated per optimizer (e.g. `AdamWOptimizerHyperparameters`). | `training.md` |
| autodiff / `[AutoDiff]` | Automatic gradient generation; per-op derivative rules. | `training.md` |
| `SafeTensor` / `.safetensors` | Weight file format (PyTorch/HF-compatible). | `onnx-and-weights.md` |
| `ModelParamList` | A named set of parameter values (e.g. loaded weights). | `onnx-and-weights.md`, `training.md` |
| `NamedModelParam` / `TensorDataModelParam` | A named param value; the concrete wrapper around one `TensorData`. | `training.md` |
| `ModelParamType` | Enum tagging a param: `Undefined/HyperParam/TrainableParam/InputParam/OutputParam`. | `training.md` |
| `TensorDataStruct` | Struct-shaped bundle of named `TensorData` fields; the input/target form `Train` expects. | `training.md` |
| `Specialize` | Bakes a partial set of named inputs (typically `[Hyper]`s) into constants, folds them through, and drops them from the input list. Optional first step of the lowering pipeline; returns a copy. | `inference.md` |
| `ToConcreteArchitecture` | Lowers a module graph into a concrete architecture (inlines sub-modules so trainable params are top-level). Required before `ToConcreteModel`/`InitializeTrainableParams`. | `onnx-and-weights.md` |
| `ToConcreteModel` | Binds a `ModelParamList` (weights) into a concrete-architecture graph by name for inference. | `onnx-and-weights.md` |
| naming scheme (`ModelIdNamingScheme` / `SimplePatternNamingScheme`) | Maps third-party (e.g. PyTorch) parameter names onto Shorokoo's, so loaded weights bind; built with the format or pattern DSL. | `param-naming-format-dsl.md`, `param-naming-pattern-dsl.md` |
| `DebugRequests` | Saves graph snapshots at chosen points of `ToConcreteArchitecture` lowering, as compilable C#. | `debugging.md` |
| `.srk` / `.zsrk` | Shorokoo's own (un)compressed graph file format. | `onnx-and-weights.md` |
| backend / execution provider | The loaded platform assembly (`LinuxCPU`/`LinuxGPU`/`WinCPU`/`WinGPU`) that runs ORT. | `inference.md` |

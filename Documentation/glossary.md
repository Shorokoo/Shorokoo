# Glossary

| Term | Meaning | Where |
|---|---|---|
| `IValue` | Base interface for any graph value handle (`Tensor<T>`, `Scalar<T>`, `Vector<T>`, sequences, optionals). | `core-types.md` |
| `Variable` / `ToVariable()` | The graph-side node a handle points at, and the argument type of `OnnxEngine.Eval` / `ComputeContext.Eval`. `Tensor<T>`, `Scalar<T>`, `Vector<T>` convert to it implicitly; `IValue` is *not* a `Variable`, so an `IValue`-typed handle needs an explicit `handle.ToVariable()`. | `core-types.md` |
| `Tensor<T>` / `Scalar<T>` / `Vector<T>` | Symbolic graph values of rank N / 0 / 1, generic over dtype marker `T`. | `core-types.md` |
| dtype marker (`float32`, `int64`, `bit`, …) | The generic type argument naming a tensor's element type. | `core-types.md` |
| `DType` | Runtime dtype descriptor (`DType.Float32`, …). | `core-types.md` |
| `TensorData` / `TensorData<T>` | The concrete values a *run* takes and gives back: inputs, outputs, training state. One object per allocation, released through the backend that made it; it ends by being deleted (`Delete()` / `Dispose()`), consumed by a run it was fed to as it is or through `.TryConsume()`, or moved into an attribute. A copy a run made of another tensor, and an element a sequence holds as its own, end with what they belong to. | `core-types.md`, `inference.md` |
| `TensorAttribute` | The concrete values written into a *graph's description*: a `Constant`'s value, a `ConstantOfShape`'s fill, a trainable parameter's weights. Immutable, attached to no context, not disposable. `TensorData.MoveToAttribute()` converts one way and spends the tensor; `TensorAttribute.CopyToTensorData()` converts back and copies. | `core-types.md` |
| `ComputeContext.Host` | A name for host memory as a context: a target for `TensorData.To` and `CopyTo`. It holds nothing, compiles and runs nothing, and cannot be disposed. `TensorData.ToHost()` brings a tensor within the host's reach. | `inference.md` |
| `To` / `CopyTo` / `ToHost` | `t.To(context)` is `t` itself where the context's backend can read its memory as it stands, and a copy there otherwise; `CopyTo` always copies; `ToHost()` is `t` itself where the host can read it. None of them touches `t` — but where `To` or `ToHost` hands back `t` itself, feeding the result as it is consumes `t` and deleting it deletes `t`; `CopyTo` is the independent copy. | `inference.md` |
| `AccessMemory()` | Reads a `TensorData<T>`'s values as a `ReadOnlySpan<primitive>`, valid only while the tensor is; `CopyMemory<T>()` reads them into an array of your own safely. | `core-types.md` |
| `IData` | Interface implemented by `TensorData` and the other run values; the input type `ComputeContext.Execute` accepts. | `inference.md` |
| consumed / `.Shared()` / `.TryConsume()` | How a run treats an input. Fed as it is, a tensor is *consumed*: the run takes it when it starts, and it is dead afterwards. `.Shared()` has the run only read it; `.TryConsume()` consumes it only if nothing else is reading it. On tensors, structs, sequences and optionals both return a `SharedInput`; on a training checkpoint they return a new checkpoint over the same tensors with its `FeedMode` set, which its derivations (`WithStep`, …) and `AdoptCheckpoint` keep, since they share its tensors. | `inference.md`, `training.md` |
| output aliasing | A run writing an output into the memory of an input it consumed rather than into memory of its own — done only for outputs a graph's lowering proved nothing reads the input after (a training step's updated state), and only where the two agree in memory, shape and element type. The output is still a new `TensorData`. | `inference.md`, `training.md` |
| `[Module]` | Attribute marking a `partial class` whose `Inline` becomes a computation graph. | `defining-models.md` |
| `[Hyper]` | Attribute marking a hyperparameter (bound on `Model(...)`, before the tensor inputs); `Scalar<T>`, `Vector<T>` or `Tensor<T>`, at any supported dtype. An optional **scalar-only** default — `[Hyper(0.9f)]` — makes the parameter omittable (the default is used when omitted) and seeds the generated optimizer hyperparameter set. | `defining-models.md` |
| `[TrainableParamInitializer]` / `[StateInitializer]` | Attributes for classes that produce trainable weights / non-trainable state. | `defining-models.md` |
| `Inline` | The `static` method the generator reads to build the graph. | `defining-models.md` |
| `Model` / `Call` / `ComputationGraph` | Generated members: bind hypers, run on inputs, get the full graph. | `defining-models.md` |
| `ComputationGraph` | The readonly graph users hold; carries a reliable `Kind` (`GraphKind.Module` / `ConcreteArchitecture` / `ConcreteModel`). | `inference.md`, `onnx-and-weights.md` |
| `GraphKind` | What a graph *is*: `Module`, `ConcreteArchitecture`, or `ConcreteModel`. Stamped by the producing path, checked by kind-gated operations, recorded in the `.srk` header. | `inference.md`, `onnx-and-weights.md` |
| `OnnxEngine.Eval` | One-shot: run a graph value and return `TensorData`. | `inference.md` |
| `ComputeContext` / `CompiledGraph` | Compile a graph once and run it many times. | `inference.md` |
| `QuickExecutionEngine` | CPU-only interpreter for debugging / shape inference (small tensors only). | `inference.md` |
| `LoopAPI.Iterate` | Build a graph loop over a `Scalar<int64>` count. | `defining-models.md` |
| `.IfElse(a, b)` | Data-dependent branch on a `Scalar<bit>`. | `defining-models.md` |
| `NN` | Static class of higher-level ops (`Conv`, `MaxPool`, `GlobalAveragePool`, …). | `core-types.md` |
| `Globals` | Static factory helpers (`Scalar`, `Vector`, `Tensor`, `TensorData`, `TensorFill`). | `core-types.md` |
| `Globals.StateUpdate` | Register a state mutation (optimizer/BatchNorm state) inside a module. | `training.md` |
| `TrainingRig` | Entry point that composes model+loss+optimizer and runs autodiff. | `training.md` |
| `TrainingCheckpoint` | Holds trainable params, model state, optimizer state, the global `Step` (advances each `TrainStep`; schedules resume from it), the host-owned run counters `Epoch` / `BatchIndex`, and the producing step's `Loss`. The last three are `null` when unknown — presence-gated on disk, so an absent counter reads back `null`, never a sentinel `0`. | `training.md` |
| `Hyperparameter` | An optimizer hyperparameter's source: `Hyperparameter.Baked(v)` (a bare value or a `TensorData`), a `Schedule`, or `Hyperparameter.Runtime(shape)`; its `Kind` decides baked-vs-scheduled-vs-runtime wiring, and its dtype and rank are whatever the optimizer's `Scalar<T>` / `Vector<T>` / `Tensor<T>` declares. | `training.md` |
| `Schedule` / `Schedules` | A `step → value` schedule (`Schedules.Cosine`, `OneCycle`, …) with fluent combinators (`WithWarmup`, `Then`, `Scale`, `Clamp`, `Shift`, `PerEpoch`). | `training.md` |
| `IOptimizerHyperparameters` / `<Optimizer>Hyperparameters` | The named, defaulted hyperparameter set; source-generated per optimizer (e.g. `AdamWOptimizerHyperparameters`). | `training.md` |
| autodiff | Automatic gradient generation; per-op derivative rules. | `training.md` |
| `SafeTensor` / `.safetensors` | Weight file format (PyTorch/HF-compatible). | `onnx-and-weights.md` |
| `ModelParamList` | A named set of parameter values (e.g. loaded weights). | `onnx-and-weights.md`, `training.md` |
| `NamedModelParam` / `TensorDataModelParam` | A named param value; the concrete wrapper around one `TensorData`. | `training.md` |
| `ModelParamType` | Enum tagging a param: `Undefined/HyperParam/TrainableParam/InputParam/OutputParam`. | `training.md` |
| `TensorDataStruct` | Struct-shaped bundle of named `TensorData` fields; the input/target form `Train` expects. | `training.md` |
| `Specialize` | Bakes a partial set of named inputs (typically `[Hyper]`s) into constants, folds them through, and drops them from the input list. Optional first step of the lowering pipeline; returns a copy. | `inference.md` |
| `ToConcreteArchitecture` | Lowers a module graph into a concrete architecture (inlines sub-modules so trainable params are top-level) and fixes its parameter space — shapes, count, and which params exist — from the values supplied. On the `ComputationGraph` route those inputs stay live; re-supply the same values at `Execute`. Required before `ToConcreteModel`/`InitializeTrainableParams`. | `onnx-and-weights.md`, `inference.md` |
| `ToConcreteModel` | Binds a `ModelParamList` (weights) into a concrete-architecture graph by name for inference. | `onnx-and-weights.md` |
| naming scheme (`ModelIdNamingScheme` / `SimplePatternNamingScheme`) | Maps third-party (e.g. PyTorch) parameter names onto Shorokoo's, so loaded weights bind; built with the format or pattern DSL. | `param-naming-format-dsl.md`, `param-naming-pattern-dsl.md` |
| `DebugRequests` | Saves graph snapshots at chosen points of `ToConcreteArchitecture` lowering, as compilable C#. | `debugging.md` |
| `BuildProgress` / `SynchronousBuildProgress` | A `progress:` sink handed to a build (`FromScratch`, `ToConcreteArchitecture`, `Load`, a `With…` derivation) reports each stage as it is entered, so a build that runs for minutes is visibly alive. | `training.md`, `debugging.md` |
| `.srk` / `.zsrk` | Shorokoo's own (un)compressed graph file format. | `onnx-and-weights.md` |
| `.skpt` / `Persistence` | Shorokoo's native checkpoint container (a STORED zip with a `config.json` manifest, or the same content as a directory of real files). `Persistence.From(...).Save(...)` / `Persistence.Load` save and reload a concrete model with its weights; `Persistence.SaveTrainingCheckpointToSkpt` writes a whole training checkpoint — rig constituents included — from which `Persistence.Load` returns the runnable model, `Persistence.LoadEvaluationModel` the model composed with its loss, and `TrainingRig.Load` the rig *and* its resumed checkpoint, all from that file alone. | `skpt-checkpoints.md` |
| backend / execution provider | The loaded platform assembly (`LinuxCPU`/`LinuxGPU`/`WinCPU`/`WinGPU`) that runs ORT. Several can be live at once, one per `ComputeContext`. | `inference.md` |
| isolated backend | A backend loaded into a load context of its own so it binds a native ONNX Runtime no other backend shares (`IsolatedBackend.Load`). | `inference.md` |

# Namespaces and usings

Related: [core-types.md](core-types.md) · [defining-models.md](defining-models.md)

## Facts

- Language/runtime: C#, .NET 10 (`net10.0`).
- Main library namespace: `Shorokoo` (plus sub-namespaces such as `Shorokoo.Modules`,
  `Shorokoo.Modules.Losses`/`.Optimizers`, and `Shorokoo.Graph`).
- Models are pure C# graphs — there is no Python. Pretrained weights are loaded from
  `.safetensors`.

## Standard usings for model code

```csharp
using Shorokoo;                 // Tensor<T>, Scalar<T>, Vector<T>, LoopAPI, attributes
using Shorokoo.Modules;         // [Module], [Hyper], [TrainableParamInitializer]
using static Shorokoo.Globals;  // Scalar(...), Vector(...), TensorData(...), TensorFill(...)
using static Shorokoo.NN;       // Conv, MaxPool, GlobalAveragePool, LayerNormalization, ...
```

Other usings are introduced by the page that needs them (for example
`Shorokoo.Onnx` for weight loading, `Shorokoo.Graph` for binding weights).

<a id="public-core-namespaces"></a>
## `Shorokoo.Core.*` is not all internal

Some of the API you are meant to call sits under `Shorokoo.Core.*`, so the prefix is not a
"do not touch" marker. Nine of those namespaces carry documented public API:

| Namespace | What it holds | Introduced by |
|---|---|---|
| `Shorokoo.Core` | `Module<…>` / `CallbackModule<…>`, `GraphBuilder`, `Variable`, `PrimitiveParam`, the parameter-naming schemes | [core-types.md](core-types.md), [defining-models.md](defining-models.md), [onnx-and-weights.md](onnx-and-weights.md) |
| `Shorokoo.Core.Training` | `Schedule` and `Schedules` — the learning-rate schedule factories and combinators | [training.md](training.md#schedule-factories-and-combinators) |
| `Shorokoo.Core.Inference` | `QuickExecutionEngine`, the CPU interpreter used for debugging and shape inference | [inference.md](inference.md#debugging-engine-no-onnxruntime) |
| `Shorokoo.Core.Inference.Abstractions` | `InferenceBackend`, `ComputeDevice`, `DeviceMemory`, `Float16`/`BFloat16` and the session interfaces | [inference.md](inference.md) |
| `Shorokoo.Core.Factory` | `FastOnnxModelBuilder` | [onnx-and-weights.md](onnx-and-weights.md) |
| `Shorokoo.Core.Factory.IR` | `ModelProto` and the rest of the ONNX protobuf types | [onnx-and-weights.md](onnx-and-weights.md) |
| `Shorokoo.Core.Graph` | `ModelId` and `ModelParamIdentifierTemplate` — the parameter identities the naming DSLs match on, and what `ToModelId` hands back | [param-naming-format-dsl.md](param-naming-format-dsl.md), [param-naming-pattern-dsl.md](param-naming-pattern-dsl.md) |
| `Shorokoo.Core.Nodes.NodeDefinitions` | `DataStructure`, and the low-level `NodeBuilder` / `NodeDefinition` op-authoring surface | [nn-library.md](nn-library.md), [operator-support.md](operator-support.md) |
| `Shorokoo.Core.Utils` | `CompressedFormatUtils` and `SrkFileFormat` / `SrkHeader` — Shorokoo's own `.srk` / `.zsrk` graph files | [onnx-and-weights.md](onnx-and-weights.md) |

The rest of `Shorokoo.Core.*` has no documented entry point: nothing here tells you to call it,
and it is not covered by these pages. Treat it as unsupported unless a page introduces it.
`Shorokoo.Core.AutoDiffCheckpointing` is the case the documentation warns off explicitly, and
[limitations.md](limitations.md#gradient-activation-checkpointing) says why.

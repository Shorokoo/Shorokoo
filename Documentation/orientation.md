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
"do not touch" marker. Six of those namespaces are supported public surface — they are the
ones the pages here write `using` lines for:

| Namespace | What it holds | Introduced by |
|---|---|---|
| `Shorokoo.Core` | `Module<…>` / `CallbackModule<…>`, `GraphBuilder`, `Variable`, `PrimitiveParam`, the parameter-naming schemes | [core-types.md](core-types.md), [defining-models.md](defining-models.md) |
| `Shorokoo.Core.Training` | `Schedule` and `Schedules` — the learning-rate schedule factories and combinators | [training.md](training.md#schedule-factories-and-combinators) |
| `Shorokoo.Core.Inference.Abstractions` | `InferenceBackend`, `ComputeDevice`, `DeviceMemory`, `Float16`/`BFloat16` and the session interfaces | [inference.md](inference.md) |
| `Shorokoo.Core.Factory` | `FastOnnxModelBuilder` | [onnx-and-weights.md](onnx-and-weights.md) |
| `Shorokoo.Core.Factory.IR` | `ModelProto` and the rest of the ONNX protobuf types | [onnx-and-weights.md](onnx-and-weights.md) |
| `Shorokoo.Core.Nodes.NodeDefinitions` | `NodeDefinition` / `NodeBuilder` and the ONNX op metadata | [nn-library.md](nn-library.md) |

Anything else under `Shorokoo.Core.*` is public only as an artefact of the assembly layout and
is not documented here — treat it as internal. `Shorokoo.Core.AutoDiffCheckpointing` is the
worked example of that, and
[limitations.md](limitations.md#gradient-activation-checkpointing) says why.

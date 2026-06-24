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
using static Shorokoo.NN;       // Conv, MaxPool, GlobalAveragePool, Erf, ...
```

Other usings are introduced by the page that needs them (for example
`Shorokoo.Onnx` for weight loading, `Shorokoo.Graph` for binding weights).

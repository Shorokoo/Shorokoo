# Shorokoo

Define, train, and run neural networks in pure C# — no Python required.

- **Define models as C# classes** — strongly typed `Tensor<float32>`, `Scalar<int64>`, shapes checked as you build.
- **Train them** — reverse-mode autodiff, optimizers, learning-rate schedules, checkpointing.
- **Run them fast** — execution backed by ONNX Runtime on CPU or GPU.
- **Interoperate** — export models as `.onnx`, load pretrained weights from `.safetensors`.

## Getting started

Install this package plus **one** backend for your platform, and (recommended)
the source generator for the `[Module]` syntax:

```bash
dotnet add package Shorokoo
dotnet add package Shorokoo.LinuxCPU      # or Shorokoo.LinuxGPU / Shorokoo.WinCPU / Shorokoo.WinGPU
dotnet add package Shorokoo.CodeGen
```

For ready-made layers, losses, and optimizers also add:

```bash
dotnet add package Shorokoo.Modules
```

```csharp
using Shorokoo;
using Shorokoo.Modules;
using static Shorokoo.Globals;
using static Shorokoo.NN;

[Module]
public partial class Dense
{
    public static Tensor<float32> Inline(
        Tensor<float32> x,
        [Hyper] Scalar<int64> outFeatures)
    {
        // ... build the layer from tensor ops ...
    }
}
```

## Documentation

Guides, API reference, and samples: https://github.com/Shorokoo/Shorokoo

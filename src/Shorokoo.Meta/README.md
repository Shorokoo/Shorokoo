# Shorokoo

Define, train, and run neural networks in pure C#. Shorokoo is a .NET deep-learning
framework with strongly typed tensors, reverse-mode autodiff, and ONNX interop.

This is the **meta-package**: it brings the runtime (`Shorokoo.Core`), the ready-made
layers (`Shorokoo.Modules`), and the `[Module]` **source generator**
(`Shorokoo.CodeGen`). Install this plus **exactly one backend**:

```
dotnet add package Shorokoo
dotnet add package Shorokoo.LinuxCPU   # or Shorokoo.LinuxGPU / Shorokoo.WinCPU / Shorokoo.WinGPU
```

## Documentation

Guides, API reference, and samples: https://github.com/Shorokoo/Shorokoo

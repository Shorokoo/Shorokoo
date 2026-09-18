# Shorokoo

Define, train, and run neural networks in pure C#. Shorokoo is a .NET deep-learning
framework with strongly typed tensors, reverse-mode autodiff, and ONNX interop.

This is the **meta-package**: it brings the runtime (`Shorokoo.Core`), the ready-made
layers (`Shorokoo.Modules`), and the `[Module]` **source generator**
(`Shorokoo.CodeGen`). Install this plus **a backend**:

```
dotnet add package Shorokoo
dotnet add package Shorokoo.LinuxCPU   # or Shorokoo.LinuxGPU / Shorokoo.WinCPU / Shorokoo.WinGPU
```

One backend is the usual case and needs no configuration — Shorokoo discovers it at first
use. A program that runs on more than one device references a backend per device and names
the one it wants, rather than leaving it to discovery.

## Documentation

Guides, API reference, and samples: https://github.com/Shorokoo/Shorokoo

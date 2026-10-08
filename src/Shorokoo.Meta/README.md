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

The guides for this version ship inside the package, in its `docs/` folder: start at
`docs/README.md`, or at `docs/first-training-run.md` for a first program end to end. Restored
to the default global packages folder, they are in `~/.nuget/packages/shorokoo/<version>/docs/`
(`%UserProfile%\.nuget\packages\shorokoo\<version>\docs\` on Windows).

The same guides online (in the package, this link names the commit the package was built from):
https://github.com/Shorokoo/Shorokoo/tree/main/Documentation

Samples: https://github.com/Shorokoo/Shorokoo/tree/main/samples

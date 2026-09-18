# Shorokoo.LinuxCPU

[Shorokoo](https://github.com/Shorokoo/Shorokoo) execution backend for
**Linux x64 (CPU)**, powered by ONNX Runtime.

```bash
dotnet add package Shorokoo
dotnet add package Shorokoo.LinuxCPU
```

Referenced on its own, this backend is discovered at first use and needs no
configuration. Running on more than one device in one process takes more than adding a
second backend package: they deliver their native ONNX Runtime at the same path, so two
referenced the ordinary way are one of them deployed twice. See
[Deploying two backends](https://github.com/Shorokoo/Shorokoo/blob/main/Documentation/inference.md#deploying-two-backends).

Documentation: https://github.com/Shorokoo/Shorokoo

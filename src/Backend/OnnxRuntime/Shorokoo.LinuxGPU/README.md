# Shorokoo.LinuxGPU

[Shorokoo](https://github.com/Shorokoo/Shorokoo) execution backend for
**Linux x64 GPU (CUDA)**, powered by ONNX Runtime.

```bash
dotnet add package Shorokoo
dotnet add package Shorokoo.LinuxGPU
```

Requires a CUDA-capable GPU and the CUDA/cuDNN versions matching the bundled
ONNX Runtime release. Referenced on its own, this backend is discovered at first use.
An application that runs on the card and the host references a backend for each and
names the one it wants.

Documentation: https://github.com/Shorokoo/Shorokoo

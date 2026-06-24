# Shorokoo.WinGPU

[Shorokoo](https://github.com/Shorokoo/Shorokoo) execution backend for
**Windows x64 GPU (CUDA)**, powered by ONNX Runtime.

```bash
dotnet add package Shorokoo
dotnet add package Shorokoo.WinGPU
```

Requires a CUDA-capable GPU and the CUDA/cuDNN versions matching the bundled
ONNX Runtime release. Reference exactly one backend package per application;
Shorokoo discovers the backend at first use.

Documentation: https://github.com/Shorokoo/Shorokoo

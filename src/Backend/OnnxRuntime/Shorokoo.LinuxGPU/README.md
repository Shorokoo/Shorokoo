# Shorokoo.LinuxGPU

[Shorokoo](https://github.com/Shorokoo/Shorokoo) execution backend for
**Linux x64 GPU (CUDA)**, powered by ONNX Runtime.

```bash
dotnet add package Shorokoo
dotnet add package Shorokoo.LinuxGPU
```

Requires a CUDA-capable GPU and the CUDA/cuDNN versions matching the bundled
ONNX Runtime release. Referenced on its own, this backend is discovered at first use. Running on the card and
the host in one process takes more than adding the CPU package too: both deliver their
native ONNX Runtime at the same path, so two referenced the ordinary way are one of them
deployed twice. See
[Deploying two backends](https://github.com/Shorokoo/Shorokoo/blob/main/Documentation/inference.md#deploying-two-backends).

Documentation: https://github.com/Shorokoo/Shorokoo

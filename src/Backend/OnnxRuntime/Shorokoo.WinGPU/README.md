# Shorokoo.WinGPU

[Shorokoo](https://github.com/Shorokoo/Shorokoo) execution backend for
**Windows x64 GPU (CUDA)**, powered by ONNX Runtime.

```bash
dotnet add package Shorokoo
dotnet add package Shorokoo.WinGPU
```

Requires a CUDA-capable GPU and a CUDA 13.x runtime. The cuDNN and cuBLAS it runs on are the
releases Shorokoo pins, shared with every other CUDA backend of the process: taken from an
installed copy that matches them exactly, or else downloaded once into a per-user cache on first
use. See
[The NVIDIA libraries the CUDA backends run on](https://github.com/Shorokoo/Shorokoo/blob/main/Documentation/inference.md#the-nvidia-libraries-the-cuda-backends-run-on). Referenced on its own, this backend is discovered at first use. Running on the card and
the host in one process takes more than adding the CPU package too: both deliver their
native ONNX Runtime at the same path, so two referenced the ordinary way are one of them
deployed twice. See
[Deploying two backends](https://github.com/Shorokoo/Shorokoo/blob/main/Documentation/inference.md#deploying-two-backends).

Documentation: https://github.com/Shorokoo/Shorokoo

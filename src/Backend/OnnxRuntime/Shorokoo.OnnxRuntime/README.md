# Shorokoo.OnnxRuntime

ONNX Runtime backend glue for [Shorokoo](https://github.com/Shorokoo/Shorokoo)
(managed, platform-neutral).

You normally do **not** install this package directly. Install one of the
platform backends instead — they depend on this package and add the native
ONNX Runtime binaries for your platform:

- `Shorokoo.LinuxCPU` — Linux x64, CPU
- `Shorokoo.LinuxGPU` — Linux x64, GPU (CUDA)
- `Shorokoo.WinCPU` — Windows x64, CPU
- `Shorokoo.WinGPU` — Windows x64, GPU (CUDA)

Documentation: https://github.com/Shorokoo/Shorokoo

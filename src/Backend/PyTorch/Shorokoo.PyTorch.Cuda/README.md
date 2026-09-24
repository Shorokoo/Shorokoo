# Shorokoo.PyTorch.Cuda

[Shorokoo](https://github.com/Shorokoo/Shorokoo) execution backend running on
**an NVIDIA GPU (CUDA 13)** through PyTorch, in an embedded CPython (Linux x64).

```bash
dotnet add package Shorokoo
dotnet add package Shorokoo.PyTorch.Cuda
```

It is never discovered: name it for the contexts that should run on it,
`new ComputeContext(new TorchCudaBackend())`. The Python environment is provisioned with
[uv](https://docs.astral.sh/uv/) on first use, or named with `SHOROKOO_PYTHON_ENV`.

Documentation: https://github.com/Shorokoo/Shorokoo/blob/main/Documentation/pytorch-backend.md

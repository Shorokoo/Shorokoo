# Shorokoo.PyTorch.Cpu

[Shorokoo](https://github.com/Shorokoo/Shorokoo) execution backend running on
**the CPU** through PyTorch, in an embedded CPython (Linux x64 and Windows x64).

```bash
dotnet add package Shorokoo
dotnet add package Shorokoo.PyTorch.Cpu
```

It is never discovered: name it for the contexts that should run on it,
`new ComputeContext(new TorchCpuBackend())`. The Python environment is provisioned with
[uv](https://docs.astral.sh/uv/) on first use, or named with `SHOROKOO_PYTHON_ENV`.

Documentation: https://github.com/Shorokoo/Shorokoo/blob/main/Documentation/pytorch-backend.md

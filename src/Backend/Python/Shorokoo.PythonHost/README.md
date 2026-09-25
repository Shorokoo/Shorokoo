# Shorokoo.PythonHost

The embedded CPython host behind [Shorokoo](https://github.com/Shorokoo/Shorokoo)'s
Python-based backends. It resolves the Python environment a backend needs — an explicit
path, the `SHOROKOO_PYTHON_ENV` variable, or an environment provisioned with
[uv](https://docs.astral.sh/uv/) from a lock file into the user cache — and starts one
interpreter per process through pythonnet.

You normally do not install this directly: install `Shorokoo.PyTorch.Cpu` or
`Shorokoo.PyTorch.Cuda`, which bring it.

Documentation: https://github.com/Shorokoo/Shorokoo/blob/main/Documentation/pytorch-backend.md

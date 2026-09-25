# Shorokoo.PythonTranslation

The ONNX-to-Python translation behind [Shorokoo](https://github.com/Shorokoo/Shorokoo)'s
Python-based backends: it writes a Shorokoo graph (serialized ONNX) as a Python module whose every
node calls a helper of the backend's support package, and refuses — while a session is being
created — whatever the backend cannot run.

You normally do not install this directly: install `Shorokoo.PyTorch.Cpu`, `Shorokoo.PyTorch.Cuda`,
`Shorokoo.Jax.Cpu` or `Shorokoo.Jax.Cuda`, which bring it.

Documentation: https://github.com/Shorokoo/Shorokoo/blob/main/Documentation/pytorch-backend.md and
https://github.com/Shorokoo/Shorokoo/blob/main/Documentation/jax-backend.md

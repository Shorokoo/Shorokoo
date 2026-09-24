# Shorokoo.PyTorch

The device-neutral logic of [Shorokoo](https://github.com/Shorokoo/Shorokoo)'s PyTorch
backend: it translates a Shorokoo graph (serialized ONNX) into Python that calls PyTorch, and
runs it in an embedded CPython.

You normally do not install this directly: install `Shorokoo.PyTorch.Cpu` or
`Shorokoo.PyTorch.Cuda`, which bring it.

Documentation: https://github.com/Shorokoo/Shorokoo/blob/main/Documentation/pytorch-backend.md

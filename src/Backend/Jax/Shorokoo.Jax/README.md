# Shorokoo.Jax

The device-neutral logic of [Shorokoo](https://github.com/Shorokoo/Shorokoo)'s JAX backend: it
translates a Shorokoo graph (serialized ONNX) into Python that calls `jax.numpy` and `jax.lax`, and
runs it in an embedded CPython, compiled by XLA once per signature of input shapes. It also computes
a training step's gradient with JAX, so that XLA compiles the whole step.

You normally do not install this directly: install `Shorokoo.Jax.Cpu` or `Shorokoo.Jax.Cuda`,
which bring it.

Documentation: https://github.com/Shorokoo/Shorokoo/blob/main/Documentation/jax-backend.md

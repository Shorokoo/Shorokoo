# Shorokoo.Jax.Cpu

A [Shorokoo](https://github.com/Shorokoo/Shorokoo) execution backend that runs on the CPU through
JAX: each model is translated to Python calling `jax.numpy` and compiled by XLA once per signature
of input shapes. A training rig built with `TrainingBackend.Native` on it has its gradient computed
by JAX, and XLA compiles the whole training step.

```csharp
using var context = new ComputeContext(new JaxCpuBackend());
```

JAX comes from a Python environment: the one `SHOROKOO_PYTHON_ENV` names, or one provisioned with
[uv](https://docs.astral.sh/uv/) into the user cache on first use.

Documentation: https://github.com/Shorokoo/Shorokoo/blob/main/Documentation/jax-backend.md

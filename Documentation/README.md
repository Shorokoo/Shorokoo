# Shorokoo Documentation

Which document covers what. For an overview of Shorokoo and an end-to-end example
(define → train → run), see the [project README](../README.md).

## 1. Define models as C# classes

- [defining-models.md](defining-models.md) — declare a model or layer with `[Module]`, expose `[Hyper]` parameters, create trainable weights, compose sub-modules, add control flow (`IfElse`, `LoopAPI.Iterate`), and specialize one module into many architecture variants.
- [core-types.md](core-types.md) — the tensors, scalars, vectors, dtypes, shapes, and `NN` ops you build a model out of.

## 2. Train them

- [first-training-run.md](first-training-run.md) — start here: one program that trains, checkpoints, resumes, evaluates and measures a model, and the limits a long run depends on.
- [training.md](training.md) — compose model + loss + optimizer with `TrainingRig`, run training steps on the device, seed a run, and the types the training API takes.
- [training-hyperparameters.md](training-hyperparameters.md) — each optimizer's hyperparameter set; baked, scheduled and runtime hyperparameters; schedule factories and combinators; custom optimizers.
- [training-data.md](training-data.md) — feed a rig through a data loader, resume at the next batch, and read the training history.
- [training-checkpoints.md](training-checkpoints.md) — save and resume a run across process restarts, what a save costs, and bind trained weights into an inference model.
- [training-memory.md](training-memory.md) — what a training step reports when it runs out of memory, what a process's memory holds, and how large a process memory limit must be.
- [training-backends.md](training-backends.md) — who computes a training step's gradient: Shorokoo's own autodiff (`TrainingBackend.Shorokoo`, the default) or the execution backend (`TrainingBackend.Native`), the step formats a backend accepts, and what differs on the native path.
- [nn-library.md](nn-library.md) — the `Shorokoo.Modules` package's ready-made layers (`Linear`, `Conv2d`, `BatchNorm2d`, attention, recurrent layers, …) and a small network trained with them.
- [initializers.md](initializers.md) — the ready-made initializers, trainable scalars, and writing your own.
- [losses-and-optimizers.md](losses-and-optimizers.md) — the sixteen losses and their knobs, and the thirteen optimizers and the state each keeps.
- [rng-configuration.md](rng-configuration.md) — seed and reproduce a model's randomness with `RngConfig`: parameter initialization and runtime draws (Dropout masks, sampling), master-seed re-rolls, per-stream overrides, and how the identity rides save/load.
- [rng-pinning.md](rng-pinning.md) — keep a module's random streams stable under refactoring with `Rng.Pin` and the stream report's per-scope pin skeleton.
- [uniform-draws.md](uniform-draws.md) — what a uniform draw returns: the half-open interval, degenerate and non-finite bounds, how finely a range is resolved, and the known imperfections.
- [normal-draws.md](normal-draws.md) — what a normal draw returns: exact symmetry, round-to-nearest, how finely the magnitude axis is resolved, the 8-sigma cap, and why the values are identical on every execution provider.

## 3. Run on CPU or GPU

- [inference.md](inference.md) — execute a model (`OnnxEngine.Eval`, `ComputeContext`), compile once and run many times, fix or hardcode `[Hyper]` parameters, stop a run, and use the CPU interpreter for debugging.
- [backends-and-devices.md](backends-and-devices.md) — pick the backend, let auto-discovery find it or load one at runtime, and run one model on two devices.
- [tensors-in-a-run.md](tensors-in-a-run.md) — feed a run (consumed, `.Shared()` or `.TryConsume()`), where its outputs are, when it writes into its inputs' memory, a tensor's lifetime, and moving data between contexts.
- [gpu-backends.md](gpu-backends.md) — the NVIDIA libraries the CUDA backends run on, `float32` precision and TensorFloat-32, device-memory budgets, and what a run did on the card.
- [pytorch-backend.md](pytorch-backend.md) — run a model on PyTorch (CPU or CUDA) instead of ONNX Runtime: the `Shorokoo.PyTorch.*` packages, the Python environment they provision or use, and what they do not run yet.
- [jax-backend.md](jax-backend.md) — run a model on JAX (CPU or CUDA), compiled by XLA once per input shape: the `Shorokoo.Jax.*` packages, what a program compiled for fixed shapes can and cannot run, and training with XLA compiling the whole step.

## 4. Interoperate with the ML ecosystem

- [onnx-and-weights.md](onnx-and-weights.md) — export/import `.onnx`, save/load Shorokoo's own `.srk`/`.zsrk` graphs, and load `.safetensors` weights and bind them into a model.
- [skpt-checkpoints.md](skpt-checkpoints.md) — Shorokoo's native `.skpt` checkpoint (a single file, or the same content as a directory): save a concrete model (definition + weights) with `Persistence.From(...).Save(...)` / `.SaveAsDirectory(...)` and load it back with `Persistence.Load`; save a full training checkpoint — weights, optimizer state, run counters and the rig's own constituents — and get from that one file to a runnable model (`Persistence.Load`), a validation-loss model (`Persistence.LoadEvaluationModel`), or the whole rig plus its resumed checkpoint (`TrainingRig.Load`); and the container/manifest format itself.

## Reference

- [orientation.md](orientation.md) — namespaces and `using` directives, including which `Shorokoo.Core.*` namespaces are documented public API.
- [glossary.md](glossary.md) — term lookup.
- [operator-support.md](operator-support.md) — per-operator support matrix (build & run, QEE, gradients) for the full supported operator set (ONNX opset 21, the one opset Shorokoo reads and writes, plus the post-21 additions it lowers or refuses).
- [param-naming-format-dsl.md](param-naming-format-dsl.md) / [param-naming-pattern-dsl.md](param-naming-pattern-dsl.md) — the two DSLs for mapping parameter names when binding third-party weights (`ToConcreteModel(weights, namingScheme)`).
- [debugging.md](debugging.md) — watch a long build stage by stage with a `progress:` sink, and snapshot the graph at chosen points of `ToConcreteArchitecture` lowering with `DebugRequests`.
- [limitations.md](limitations.md) — known limitations, permanent and otherwise.

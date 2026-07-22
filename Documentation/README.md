# Shorokoo Documentation

Which document covers what. For an overview of Shorokoo and an end-to-end example
(define → train → run), see the [project README](../README.md).

## 1. Define models as C# classes

- [defining-models.md](defining-models.md) — declare a model or layer with `[Module]`, expose `[Hyper]` parameters, create trainable weights, compose sub-modules, and add control flow (`IfElse`, `LoopAPI.Iterate`).
- [core-types.md](core-types.md) — the tensors, scalars, vectors, dtypes, shapes, and `NN` ops you build a model out of.

## 2. Train them

- [training.md](training.md) — compose model + loss + optimizer with `TrainingRig`, run the training loop, and save / resume checkpoints across process restarts.
- [nn-library.md](nn-library.md) — the `Shorokoo.Modules` package: ready-made initializers, layers (`Linear`, `Conv2d`, `BatchNorm2d`, …), losses, and optimizers to build and train with.
- [rng-configuration.md](rng-configuration.md) — seed and reproduce a model's randomness with `RngConfig`: parameter initialization and runtime draws (Dropout masks, sampling), master-seed re-rolls, per-stream overrides, and how the identity rides save/load.
- [rng-pinning.md](rng-pinning.md) — keep a module's random streams stable under refactoring with `Rng.Pin` and the stream report's per-scope pin skeleton.

## 3. Run on CPU or GPU

- [inference.md](inference.md) — execute a model (`OnnxEngine.Eval`, `ComputeContext`), pick the backend, read output values, and use the CPU interpreter for debugging.

## 4. Interoperate with the ML ecosystem

- [onnx-and-weights.md](onnx-and-weights.md) — export/import `.onnx`, save/load Shorokoo's own `.srk`/`.zsrk` graphs, and load `.safetensors` weights and bind them into a model.
- [skpt-checkpoints.md](skpt-checkpoints.md) — Shorokoo's native `.skpt` single-file checkpoint: save a concrete model (definition + weights) with `Checkpoint.From(...).Save(...)`, load it back with `Checkpoint.Load`, and the container/manifest format itself.

## Reference

- [orientation.md](orientation.md) — namespaces and `using` directives.
- [glossary.md](glossary.md) — term lookup.
- [operator-support.md](operator-support.md) — per-operator support matrix (build & run, QEE, gradients) for the full supported operator set (opset 21 plus the post-21 additions through opset 26).
- [param-naming-format-dsl.md](param-naming-format-dsl.md) / [param-naming-pattern-dsl.md](param-naming-pattern-dsl.md) — the two DSLs for mapping parameter names when binding third-party weights (`ToConcreteModel(weights, namingScheme)`).
- [debugging.md](debugging.md) — snapshot the graph at chosen points of `ToConcreteArchitecture` lowering with `DebugRequests`.
- [limitations.md](limitations.md) — known limitations, permanent and otherwise.

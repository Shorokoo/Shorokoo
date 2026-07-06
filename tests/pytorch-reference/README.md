# PyTorch reference generators

Scripts here produce **external-truth reference constants** for the forward-value
checks in the NN-library coverage modules (e.g. `NNLinearMatchesManualMatMul` in
`tests/Shorokoo.Tests/Modules/NNLibraryTestModules.cs`).

Those modules replace the old "recompute the layer's own math by hand and assert
agreement" pattern (a tautology). Each now runs the layer forward and compares —
*in-graph*, returning `Scalar<bit>` via `AutoTest.AdvancedTestGraph` — against a
reference frozen as a one-line `Vector<float32>` in the module, tagged with a
provenance comment:

- `// REFERENCE: PyTorch` — produced by the equivalent PyTorch op on the same
  seeded weights + input (the scripts here); an external source of truth.
- `// REFERENCE: golden` — self-generated: the layer's own forward output, frozen.
  Catches regressions; upgraded to a PyTorch reference over time.

## Regenerating a PyTorch reference

The reference is deterministic from the test's `MasterSeed` (default 0) and its
fixed input, so it is reproduced at test time — the `.safetensors` below is only
the bridge to PyTorch during generation.

1. Export the seeded weights + the exact test input to a `.safetensors` file
   (materialize the layer's concrete architecture at the test seed,
   `InitializeTrainableParams`, and save each param plus an `"input"` tensor via
   `SafeTensorLoader.SaveSafeTensors`).
2. Run the matching script to print the C# `Vector(...)` initializer text:

   ```bash
   pip install torch safetensors numpy       # CPU wheel is fine
   python3 tests/pytorch-reference/linear.py /tmp/linear.safetensors
   ```
3. Paste the numbers into the module's `reference` vector.

A `golden` reference needs no PyTorch: run the layer forward once at the test
seed + input and freeze the flattened output.

## Collapsing large outputs

Outputs of <=30 values are compared elementwise (the current examples: Linear
[2,4]=8, Conv2d [1,3,3,3]=27). For a layer whose output exceeds ~30 values,
collapse it in-graph to ~19 numbers first — a single `MatMul` against a baked
`[19, n]` projection matrix `W` with `W[c,i] = w(i)` when `i mod 19 == c` else 0
(a fixed, fp-stable linear sketch; the weight `w(i)` is an integer hash of the
position, not the value). Replicate the same formula on the PyTorch side when a
collapsed output needs a PyTorch reference.

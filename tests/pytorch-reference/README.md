# PyTorch reference generators

Scripts here produce the **frozen reference constants** for
`ModuleForwardValueTests` from the equivalent PyTorch op, giving those tests an
*external* source of truth (rather than re-running the layer's own math by hand,
which is a tautology).

## How a forward-value test is anchored

1. A fixture `[Module]` (e.g. `ParityLinear`) wraps one layer as a single
   tensor-in / tensor-out graph.
2. The `_GoldenReferenceGen` harness (in `Shorokoo.Tests`, `Purpose=Manual`)
   materializes the fixture from a **fixed seed** (`MasterSeed = 12345`), runs it
   forward on a **fixed input**, and:
   - prints the collapsed output for **golden** (self-generated) references, and
   - exports the seeded weights + input to `.safetensors` for **PyTorch** references.
3. For a PyTorch reference, the matching script here loads that `.safetensors`,
   runs the PyTorch analog on the **identical** weights + input, and prints the
   output as C# `float[]` initializer text.
4. The values are pasted into `ModuleForwardValueTests` and marked with a
   provenance comment: `// REFERENCE: PyTorch` or `// REFERENCE: golden`.

Because the fixture init is deterministic for a seed, the test reproduces the
exact same weights at runtime — it does not load the `.safetensors`; that file is
only the bridge to PyTorch during generation.

## Regenerating

```bash
pip install torch safetensors numpy      # CPU wheel is fine

# export weights + input (and print golden values) from the C# side:
PILOT_DIR=/tmp/pilot dotnet test tests/Shorokoo.Tests/Shorokoo.Tests.csproj \
    --filter "FullyQualifiedName~_GoldenReferenceGen"

# produce a PyTorch reference:
python3 tests/pytorch-reference/linear.py /tmp/pilot/linear.safetensors
```

Collapse convention (large outputs): outputs of >30 values are reduced to 19
prime-strided, position-weighted sums (`ModuleForwardValueTests.Collapse`) —
order- and value-sensitive but fp-stable and transcendental-free, so it is
portable across machines. Replicate it on the PyTorch side when a collapsed
output needs a PyTorch reference.

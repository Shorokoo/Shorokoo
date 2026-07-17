# Test Data Directory

This directory holds permanent test fixtures for the Shorokoo test suite,
organized by domain and purpose.

Small fixtures (parameter-id listings, architecture text/JSON) are committed.
**Large model checkpoints and tensor archives are not** — they are kept out of
the repo entirely.
Developers download them manually when they want to run the opt-in
`Purpose=Manual` tests (see [Downloading model checkpoints](#downloading-model-checkpoints)).
Everything matching `*.safetensors`, `*.zsafetensor`, `*.pth`, `*.pt`, `*.onnx`
here is git-ignored, so a manual download can never be committed by accident.

## Structure

```
test-data/
├── models/                    # downloaded model weights (git-ignored, not committed)
│   └── resnet18/
│       └── resnet18.safetensors      # ← download manually; consumed by RealCheckpointTests
├── integration/
│   ├── images/                # test images (git-ignored)
│   └── intermediate-layers/   # golden intermediate outputs (no longer committed)
└── modules/fclayer/           # small committed fixtures (architecture/param-id text)
```

## Downloading model checkpoints

The `Purpose=Manual` tests (e.g. `RealCheckpointTests`) skip themselves with a
clear message unless their checkpoint is present, so a clean checkout stays
green without any download. To run them, fetch the data first.

### ResNet18 (consumed by `RealCheckpointTests`)

A small (~47 MB) public, torchvision-compatible ResNet18 checkpoint:

```bash
mkdir -p tests/test-data/models/resnet18
curl -L https://huggingface.co/timm/resnet18.a1_in1k/resolve/main/model.safetensors \
  -o tests/test-data/models/resnet18/resnet18.safetensors
```

Then run the opt-in tests (they are excluded from the default `Purpose=Coverage`
suite):

```bash
dotnet test tests/Shorokoo.Tests/Shorokoo.Tests.csproj --filter "Purpose=Manual"
```

Any torchvision-named ResNet18 `.safetensors` works — the test asserts the
canonical landmark tensors (`conv1.weight` `[64,3,7,7]`, `fc.weight` `[1000,512]`,
…), not an exact byte hash.

### ResNet18 prediction (consumed by `RealCheckpointPredictionTests`)

`RealCheckpointPredictionTests` completes the E-3 **prediction** half: it binds the
downloaded checkpoint onto the Shorokoo `ResNet18` graph through
`TorchvisionResNet18NamingScheme` (the PyTorch→Shorokoo name map) and runs a forward
pass. It checks both the top-1 class **and** full-distribution parity against PyTorch
(every logit/probability, not just the argmax). The parity target is **baked into the
test** (`RealCheckpointPredictionTests.ReferenceLogits`), so the test needs no reference
file — only the checkpoint above plus the preprocessed input tensor.

Generate the input (downloads the canonical PyTorch Samoyed image, applies the timm eval
transform, writes `sample-input-dog.safetensors` — no torch needed):

```bash
pip install numpy pillow safetensors
python tests/test-data/models/resnet18/make-sample-input.py
```

Then run the prediction, parity, and name-map bijection tests:

```bash
dotnet test tests/Shorokoo.Tests/Shorokoo.Tests.csproj --filter "Purpose=Manual"
```

The image and generated tensor are git-ignored; the tests skip cleanly until both the
checkpoint and the input tensor are present. Observed agreement with PyTorch is at the
float32 noise floor (max |Δlogit| ~1e-5, max |Δprob| ~1e-6).

To **regenerate** the baked `ReferenceLogits` (only if the image, preprocessing, or
checkpoint change), run the reference model and paste its printed array — it writes no file:

```bash
pip install torch safetensors          # CPU torch is enough; torchvision/timm not needed
python tests/test-data/models/resnet18/make-reference-logits.py
```

## Real-pretrained-model parity (release check E-6)

Full bit-exact parity against recorded PyTorch outputs (ResNet18/50, ViT-Tiny,
RetinaNet — release-test-plan check **E-6**) is a **manual** release-time
exercise. Its golden weights and intermediate-layer outputs are no longer
committed; regenerate or re-export them with PyTorch exporters and the Shorokoo
architecture generators, then place them under the matching
`models/` / `integration/intermediate-layers/` paths.
Treat any regenerated file as golden data: re-validate it against the recorded
provenance before relying on it.

## Guidelines

### When to add committed data here
- Small reference fixtures needed by multiple tests
- Small reference outputs for validation tests

### When NOT to commit data here
- Model checkpoints / tensor archives (download them manually instead — they are
  git-ignored)
- Experimental, investigation-specific, or short-lived data

### Data provenance
When adding or regenerating data, document: (1) the source (e.g. PyTorch/HF model
hub, custom-trained), (2) the model version/checkpoint, (3) any preprocessing.

## How this data is used

The automated coverage suite (`Purpose=Coverage`) exercises this directory only
via the small committed fixtures. Tests that need a downloaded checkpoint are
tagged `Purpose=Manual` and skip unless the file is present (see
`RealCheckpointTests`). Real-pretrained-model parity (E-6) is validated manually
at release time.

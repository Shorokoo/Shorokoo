# .skpt checkpoints

Related: [onnx-and-weights.md](onnx-and-weights.md) · [training.md](training.md) · [inference.md](inference.md)

## Facts

- A `.skpt` file is Shorokoo's native checkpoint container: **one file** holding a
  model definition and its weights, reloadable as a runnable model with
  `Checkpoint.Load`. There is no loose-directory form.
- The file is a **standard zip archive** — any unzip tool can list and extract its
  entries — whose entries are all **STORED** (uncompressed), so tensor data remains
  range-readable through the zip central directory. Data payloads are 64-byte aligned
  inside the file.
- Data entries can **opt into Zstd compression** (`.WithZstdCompressedData()`): the zip
  framing stays STORED, and a single Zstd layer lives inside the entry's bytes, declared
  per entry in the manifest (`compression: "zstd"`). The default remains uncompressed —
  and byte-for-byte identical to output without the option. See
  [the compression trade-off](#compressed-data-entries-the-trade-off).
- A single `config.json` manifest is the **only source of wiring**: entries never
  reference each other; every mapping (model → serialization format, parameter →
  stored tensor, data entry → storage format) lives in the manifest.
- Saves are **atomic** (staged to a temp file, committed by rename): a crash mid-save
  never corrupts an existing checkpoint. The target directory must already exist.
- This version writes exactly one checkpoint shape: an **inference checkpoint of a
  concrete model** (definition + weights). Training-rig state, precompiled artifacts,
  and additional weight sets (e.g. EMA) are future extensions of the same container.
- `.skpt` replaces nothing: ONNX export and `.safetensors`/`.srk` files remain
  separate, on-demand projections (see [onnx-and-weights.md](onnx-and-weights.md)).

## Save and load

```csharp
using Shorokoo;   // Checkpoint

// graph: a ComputationGraph of kind ConcreteModel — fully lowered, weights materialized.
Checkpoint.From(concreteModel)
    .WithModel()      // include the model definition
    .WithWeights()    // include the weights
    .Save("model.skpt");

var loaded = Checkpoint.Load("model.skpt");   // ConcreteModel, weights bound
var outputs = ComputeContext.Default.Execute(loaded, inputs);
```

Round-trip is exact: the loaded model's weight bytes are identical to the saved
model's, and execution on the same inputs is bit-identical.

`Checkpoint.From` requires a `GraphKind.ConcreteModel`; lower a module graph with
`ToConcreteArchitecture(inputHints, ...).ToConcreteModel(...)` first. This version
requires both `.WithModel()` and `.WithWeights()` — the builder shape exists so later
versions can add contents without changing the call pattern.

To shrink the file, opt into per-entry Zstd compression of the data tree:

```csharp
Checkpoint.From(concreteModel)
    .WithModel()
    .WithWeights()
    .WithZstdCompressedData()       // optional level 1–22, default 3
    .Save("model.skpt");

var loaded = Checkpoint.Load("model.skpt");   // decompression is transparent
```

Loading honors each entry's manifest-declared compression; nothing changes on the
read side of the API.

## Compressed data entries: the trade-off

Compression is a per-entry, opt-in trade of **size against range-readability**:

- An uncompressed (default) data entry is STORED verbatim and 64-byte aligned, so a
  future reader can memory-map or range-read the tensor bytes straight out of the file
  through the zip central directory.
- A Zstd-compressed entry is smaller on disk but must be decompressed in full before
  any tensor in it can be read — it **forfeits mmap/range reads**, and therefore also
  skips the 64-byte alignment (alignment would buy nothing).
- Compression is recorded **only in the manifest** (`compression: "zstd"` in the
  entry's data-registry record), never inferred from an entry's file extension — the
  same rule `.srk` v2 follows with its header. The entry's manifest `sha256` covers
  the **stored (compressed) bytes**, so integrity checking never requires
  decompression.
- `config.json` and `models/*.srk` are never compressed by the option (the `.srk`
  payload is already Zstd-compressed internally), and the zip framing itself stays
  STORED — any unzip tool still lists and extracts every entry; a compressed data
  entry extracts to a `.zst`-decodable byte stream.
- A manifest/stored mismatch — an entry marked `"zstd"` whose bytes are not a Zstd
  frame, or one marked `"none"` whose bytes are — fails loudly on load, naming the
  entry.

## Inspecting a .skpt

`Checkpoint.Inspect("model.skpt")` identifies the container and summarizes its
manifest — whole-archive metadata, the model and data registries, the
mapping-set names — reading only the zip central directory and `config.json`,
never the tensor data. The recorded per-entry sha256s are reported as written
but not verified (a full `Checkpoint.Load` verifies them), and cheap sanity
observations flag manifest/archive mismatches, compressed entries where STORED
is expected, and unknown manifest keys. See the inspection section in
[onnx-and-weights.md](onnx-and-weights.md#identify-and-summarize-a-file-checkpointinspect).

## Container layout

```
model.skpt
├── config.json                the manifest: all metadata and all wiring
├── models/
│   └── model.srk              the model definition (srk2 encoding, weights stripped)
└── data/
    └── weights.safetensors    tensor data (safetensors layout)
```

- `models/model.srk` is the model **definition**: a valid `.srk` v2 concrete-model
  file in which each weight tensor is replaced by a placeholder of the same
  dtype/shape whose values are elided (an empty, marker-tagged initializer payload) —
  placeholders cost almost nothing on disk and no weight-sized allocation in memory.
  The model's RNG identity parameter is part of the definition — not
  a weight — and stays embedded, so a reloaded model reproduces the original's
  randomness (see [rng-configuration.md](rng-configuration.md)).
- `data/weights.safetensors` holds the real weight bytes once, as a plain
  [safetensors](https://huggingface.co/docs/safetensors) file. Tensor names are the
  model's internal parameter identifiers, as wired by the manifest — extract the entry
  with any unzip tool and read it with any safetensors reader.
- The trees are optional and the layout is extensible: future versions add more
  `models/` entries, more `data/` kinds, `precompiledmodels/`, and `sample_inputs/`
  without a container change.

## The `config.json` manifest

```jsonc
{
  "format": "skpt",                       // format identifier
  "skptVersion": 1,                       // format major version
  "createdUtc": "2026-07-21T13:32:39Z",
  "producer": { "shorokoo": "0.1.0" },    // framework version that wrote the file

  // Model registry: per model, where its definition lives and how it is encoded.
  "models": {
    "model": {
      "entry": "models/model.srk",
      "format": "srk2",                   // the .srk v2 container encoding
      "stage": "concrete-model",          // lifecycle stage of the serialized graph
      "sha256": "6824d4…"                 // hash of the entry's bytes (the graph hash)
    }
  },

  // Tensor mappings: per model, named mapping sets resolving each parameter to a
  // tensor inside a data entry. Only the "default" set is written today; the shape
  // allows parallel sets (e.g. EMA weights) later.
  "tensorMappings": {
    "model": {
      "default": {
        "tensors": {
          "[1]:TrainableParam#0…": { "data": "weights", "tensor": "[1]:TrainableParam#0…" }
        }
      }
    }
  },

  // Data registry: per data entry, its storage format, compression, and hash.
  "data": {
    "weights": {
      "entry": "data/weights.safetensors",
      "format": "safetensors",
      "compression": "none",              // "none" or "zstd"; never inferred from the name
      "sha256": "734485…"                 // hash of the entry's bytes as stored (compressed)
    }
  }
}
```

Rules:

- **Keys are add-only.** A reader ignores unknown keys; removing or re-typing a key is
  a major-version event (a bump of `skptVersion`). A file with a higher `skptVersion`
  is refused with a clear message rather than half-read.
- **Integrity is checked on load.** Every entry the manifest references must exist and
  match its recorded `sha256`; a missing entry, a hash mismatch, or a tensor mapping
  that does not cover the model's parameters exactly fails loudly, naming the
  offending entry or parameter.

## Current limits

- One model per file, the `default` mapping set only.
- Data entries are bounded by the in-memory safetensors path — checkpoints with ≥ 2 GB
  of tensor data in a single entry are not yet supported (compressed or not; the bound
  applies to both the stored and the decompressed bytes).
- Training state (optimizer/scheduler, constituent models of a rig) is not stored;
  for training resume across process restarts, use `TrainingCheckpoint.Save` /
  `TrainingRig.LoadCheckpoint` (see [training.md](training.md)).

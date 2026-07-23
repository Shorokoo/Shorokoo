# .skpt checkpoints

Related: [onnx-and-weights.md](onnx-and-weights.md) · [training.md](training.md) · [inference.md](inference.md)

## Facts

- A `.skpt` file is Shorokoo's native checkpoint container: **one file** holding a
  model definition and its weights, reloadable as a runnable model with
  `Persistence.Load`. There is no loose-directory form.
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
- This version writes an **inference checkpoint of a concrete model** (definition +
  weights). It can also carry **additional named weight sets** over the same parameters
  (e.g. an `ema` set alongside `default`), selected at load time — see
  [Named weight sets](#named-weight-sets-default--ema).
- A `.skpt` can also persist a **training checkpoint** — the trainable weights, model
  state, optimizer state and global step of a training run — with the state split into
  separate per-kind `data/` entries alongside the concrete inference model, so a run
  resumes across process restarts and the same file also loads as an inference model. See
  [Training checkpoints](#training-checkpoints). Precompiled artifacts and the full
  training-rig (constituent models, schedules) are future extensions of the same container.
- A checkpoint can carry a **host user-data bag** — an arbitrary JSON object you attach
  at save and read back verbatim on load (e.g. your data-pipeline state), stored as
  `data/user-data.json` and never interpreted by Shorokoo. See
  [Host user-data bag](#host-user-data-bag).
- `.skpt` replaces nothing: ONNX export and `.safetensors`/`.srk` files remain
  separate, on-demand projections (see [onnx-and-weights.md](onnx-and-weights.md)).

## Save and load

```csharp
using Shorokoo;   // Persistence

// graph: a ComputationGraph of kind ConcreteModel — fully lowered, weights materialized.
Persistence.From(concreteModel)
    .WithModel()      // include the model definition
    .WithWeights()    // include the weights
    .Save("model.skpt");

var loaded = Persistence.Load("model.skpt");   // ConcreteModel, weights bound
var outputs = ComputeContext.Default.Execute(loaded, inputs);
```

Round-trip is exact: the loaded model's weight bytes are identical to the saved
model's, and execution on the same inputs is bit-identical.

`Persistence.From` requires a `GraphKind.ConcreteModel`; lower a module graph with
`ToConcreteArchitecture(inputHints, ...).ToConcreteModel(...)` first. This version
requires both `.WithModel()` and `.WithWeights()` — the builder shape exists so later
versions can add contents without changing the call pattern.

To shrink the file, opt into per-entry Zstd compression of the data tree:

```csharp
Persistence.From(concreteModel)
    .WithModel()
    .WithWeights()
    .WithZstdCompressedData()       // optional level 1–22, default 3
    .Save("model.skpt");

var loaded = Persistence.Load("model.skpt");   // decompression is transparent
```

Loading honors each entry's manifest-declared compression; nothing changes on the
read side of the API.

## Training checkpoints

A training run's state — the trainable weights, model state, optimizer state and the
global step — persists into a `.skpt` too, so training resumes across process restarts
in the native container (inspectable manifest, per-entry Zstd, atomic write, provenance
metadata), sharing one on-disk format with inference checkpoints.

```csharp
using Shorokoo;   // Persistence, TrainingRig, TrainingCheckpoint

// checkpoint: a TrainingCheckpoint from rig.CreateDefaultCheckpoint() / TrainStep().
// exampleInput: any sample model input — only its shape matters (drives concretization).
Persistence.SaveTrainingCheckpointToSkpt(
    checkpoint, modelGraph, exampleInput, "run.skpt");

// Resume in a fresh process: rebuild the rig from the same graphs, then load.
var rig     = TrainingRig.FromScratch(modelGraph, lossGraph, optimizerGraph, sample, hypers);
var resumed = rig.LoadCheckpoint("run.skpt");   // reads .skpt or legacy flat, auto-detected
var next    = rig.TrainStep(resumed, inputBatch, targetBatch, compiled);
```

To compose the container's features, use the builder form:

```csharp
Persistence.ForTrainingCheckpoint(checkpoint, modelGraph, exampleInput)
    .WithZstdCompressedData()                          // per-entry Zstd (optional level 1–22)
    .WithMetadata(runName: "nightly-42", gitCommit: "9f3c1ba")
    .Save("run.skpt");
```

What the file carries:

- **The concrete inference model** in `models/model.srk` (definition, weights stripped),
  built from the checkpoint's trained weights. The trainable weights double as the model's
  `default` weight set, so the same file loads as a runnable inference model with
  `Persistence.Load("run.skpt")` — no separate export step.
- **The training state, split by kind into separate `data/` entries**: the trainable
  weights (`data/trainable.safetensors`), the model state (`data/model_state.safetensors`,
  omitted for a stateless model) and the optimizer state
  (`data/optimizer_state.safetensors`, omitted for a stateless optimizer like plain SGD).
  Each entry stores its kind's tensors keyed by struct field name.
- **The global step** and which data entry holds each kind, recorded in the manifest's
  `training` block (so `Persistence.Inspect` reports them without reading tensor data).

Round-trip is exact: reloaded trainable params, model state and optimizer state are
bit-identical, the step is preserved, and a resumed `TrainStep` reproduces the pre-save
trajectory. Loading validates against the rig's struct definitions with the same fail-loud
contract as the legacy flat format — a checkpoint from a different model or optimizer, a
missing kind, a rank mismatch, or a tampered entry (sha256) fails loudly.

Reconstruct without a rig by supplying the struct defs directly:

```csharp
TrainingCheckpoint ckpt = Persistence.LoadTrainingCheckpoint(
    "run.skpt", trainableParamDef, modelStateDef, optimizerStateDef);
```

`Persistence.SaveTrainingCheckpoint(checkpoint, path)` still writes the **legacy flat**
[safetensors format](training.md); the `.skpt` path is opt-in via
`SaveTrainingCheckpointToSkpt` / `ForTrainingCheckpoint`. Both `LoadTrainingCheckpoint`
and `TrainingRig.LoadCheckpoint` read either shape — the on-disk form is detected from the
file — so old and new checkpoints load through one entry point.

## Provenance metadata

A checkpoint records its **producer** (framework version) and **creation time**
automatically. You can attach your own **provenance metadata** on top — a
free-form `string → string` bag written into the manifest, so the checkpoint is
self-documenting for reproducibility. It is cheap to write at save time and
impossible to reconstruct later.

```csharp
Persistence.From(concreteModel)
    .WithModel()
    .WithWeights()
    .WithMetadata(
        gitCommit: "9f3c1ba",
        datasetId: "imagenet-1k@v2",
        runName:   "nightly-run-42",
        license:   "Apache-2.0")
    .Save("model.skpt");
```

Four well-known keys — git commit, dataset id, run name, license — are surfaced
as named parameters; any other pairs go in the map argument, and calls
accumulate:

```csharp
.WithMetadata(new Dictionary<string, string> { ["experiment"] = "ablation-7" },
              gitCommit: "9f3c1ba")
```

`Persistence.Inspect` echoes the metadata back (in its own section, distinct from
the auto producer/created fields):

```csharp
var info = Persistence.Inspect("model.skpt");
foreach (var (key, value) in info.Skpt!.UserMetadata ?? new Dictionary<string, string>())
    Console.WriteLine($"{key} = {value}");
```

What provenance metadata **is and is not**:

- **Purely informational.** It never affects manifest identity checks or weight
  binding — `Persistence.Load` ignores it entirely, so a checkpoint loads and
  binds identically with or without it. It is trusted only as far as its writer:
  Shorokoo does not sign, interpret, or validate the values (a git commit is not
  checked to exist), and nothing is auto-populated from the environment — you
  supply every value.
- **Add-only, like the rest of the manifest.** A reader tolerates keys it does
  not know. The values are stored verbatim; the human-readable inspection output
  sanitizes control characters for display only, so a value can never forge a
  line in the summary — but the structured `UserMetadata` property keeps it raw.
- **Absent by default.** Supply none and the manifest's `userMetadata` key is
  simply not written — the output is byte-for-byte identical to a checkpoint
  saved without provenance.

## Host user-data bag

Provenance metadata is a flat `string → string` map for **humans** to read in
`Inspect`. When your resuming **program** needs to read back structured state,
attach a **user-data bag** instead: an arbitrary JSON object you serialize at
save and read back verbatim on load, stored as `data/user-data.json`.

Its motivating use is the **data-pipeline state** — which corpus, the
shuffle/augmentation strategy, the stream position — the one part of a run
Shorokoo cannot reconstruct for you, because it does not own your dataloader. The
bag carries your own bytes and hands them back, making a `.skpt` a self-contained
resume unit, without interpreting them.

```csharp
Persistence.From(concreteModel)
    .WithModel()
    .WithWeights()
    .WithUserData(new PipelineState        // any type System.Text.Json can serialize
    {
        Corpus      = "imagenet-1k",
        ShuffleSeed = 12345,
        Epoch       = 3,
        Shards      = ["a.tar", "b.tar", "c.tar"],
    })
    .Save("model.skpt");
```

Read it back through `Inspect` — as the raw DOM, or deserialized into your type:

```csharp
var info = Persistence.Inspect("model.skpt");

System.Text.Json.Nodes.JsonObject? bag = info.Skpt!.UserData;   // null when absent
PipelineState? state = info.Skpt!.GetUserData<PipelineState>();  // default when absent
```

`WithUserData(JsonObject value)` takes a `System.Text.Json.Nodes.JsonObject`
directly if you would rather build the DOM by hand.

What the user-data bag **is and is not**:

- **A JSON object at the root.** The one structural rule: the value must
  serialize to a JSON *object* (a property bag), so a bare list or scalar is
  rejected at save with a clear error — wrap it in an object first (e.g.
  `{ "items": [ … ] }`). The values *under* the root may be any valid JSON.
- **Never interpreted.** Shorokoo validates well-formedness only — it never
  schema-checks the shape or meaning of the values, and never fails a load on a
  data mismatch (that check, if you want one, is your code). The bag wires
  nothing: `Persistence.Load` ignores it entirely, binding a checkpoint
  identically with or without it.
- **`$`-prefixed top-level keys are reserved** for Shorokoo and rejected at save;
  use any other key. (Only the root's keys are reserved — nested objects may use
  any keys.)
- **Summarized, not dumped.** `Inspect`'s text summary shows a one-line key count
  (`user-data: 4 keys`), never the nested contents; the full object stays
  available through the `UserData` property.
- **Absent by default.** Supply none and no `data/user-data.json` entry is
  written — the output is byte-for-byte identical to a checkpoint saved without
  it. The bag is always stored uncompressed, independent of
  `.WithZstdCompressedData()`.

Distinct from a Shorokoo-defined data-pipeline format: there is none. If Shorokoo
ever grows a first-class dataloader, a replayable pipeline state could supersede
this bag — until then it is your bytes, round-tripped.

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

## Named weight sets (default + ema)

A checkpoint can carry **more than one named set of weights over the same model
parameters** — the motivating case being EMA / averaged weights kept alongside the raw
weights. The model definition is stored once; each set is a mapping from the model's
parameters to stored tensors. The parameterless `.WithWeights()` writes the model's own
weights as the `default` set; add another set with `.WithWeights(setName, values)`,
where `values` maps each weight-parameter identifier to that set's tensor:

```csharp
// emaWeights: IReadOnlyDictionary<string, TensorData> keyed by the model's weight-
// parameter identifiers, covering exactly the same parameters as the default weights.
Persistence.From(concreteModel)
    .WithModel()
    .WithWeights()                    // the "default" set (the model's own weights)
    .WithWeights("ema", emaWeights)   // an additional set over the same parameters
    .Save("model.skpt");

var raw = Persistence.Load("model.skpt");            // binds "default"
var smoothed = Persistence.Load("model.skpt", "ema"); // binds "ema"
```

- **Selection at load.** `Persistence.Load(path)` binds `default`; `Persistence.Load(path,
  set)` binds the named set. An unknown set name fails loudly, listing the sets the file
  declares.
- **Shared data is stored once.** A set's tensor whose bytes (dtype, shape and content)
  are identical to one already stored — in the `default` set or an earlier additional
  set — is **referenced, not copied**. Only a set's genuinely distinct tensors are
  written, into its own `data/<setName>.safetensors` entry. So an EMA set that differs
  from the raw weights in only a few tensors adds only those few tensors to the file.
- **Coverage is exact.** An additional set must map every weight parameter the model
  declares (the same parameters the `default` weights span), each with a matching dtype
  and shape; a missing or stray parameter, or a shape/dtype mismatch, fails loudly at
  save.
- **`config.json` records every set.** Each set is a named entry under a model's
  `tensorMappings`; the data registry gains one entry per set that has distinct tensors.
- **`default`-only is unchanged.** A save with no additional set is byte-for-byte the
  single-set output — the feature adds nothing to a file that does not use it. The set
  name must be a non-empty identifier over `[A-Za-z0-9._-]`, distinct from the reserved
  `default` set and `weights` data key.

Computing EMA / averaged weights is a **training** concern and out of the container's
scope; `.WithWeights(setName, values)` only carries and selects the parallel versions.

## Inspecting a .skpt

`Persistence.Inspect("model.skpt")` identifies the container and summarizes its
manifest — whole-archive metadata (producer, creation time, any
[user provenance metadata](#provenance-metadata), and a one-line count of the
[host user-data bag](#host-user-data-bag) with the full object on `Skpt.UserData`),
the model and data registries, the mapping-set names — reading only the zip
central directory, `config.json`, and (when present) the small `data/user-data.json`
entry, never the tensor data. The recorded per-entry sha256s are reported as written
but not verified (a full `Persistence.Load` verifies them), and cheap sanity
observations flag manifest/archive mismatches, compressed entries where STORED
is expected, and unknown manifest keys. See the inspection section in
[onnx-and-weights.md](onnx-and-weights.md#identify-and-summarize-a-file-checkpointinspect).


A foreign `.safetensors` file (e.g. PyTorch/timm weights) lands as a native
checkpoint in one call — the strict safetensors import (see
[onnx-and-weights.md](onnx-and-weights.md#weight-exchange-with-naming-schemes-exportsafetensors--importsafetensors))
followed by this same writer:

```csharp
ComputationGraph model = Persistence.ImportSafeTensorsToCheckpoint(
    arch, "foreign.safetensors", "model.skpt", scheme);
```

## Container layout

```
model.skpt
├── config.json                the manifest: all metadata and all wiring
├── models/
│   └── model.srk              the model definition (srk2 encoding, weights stripped)
└── data/
    ├── weights.safetensors    tensor data (safetensors layout)
    └── user-data.json         optional host user-data bag (JSON object)
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
- `data/user-data.json` holds the optional [host user-data bag](#host-user-data-bag) —
  a JSON object you attach and read back verbatim; present only when you supply one, and
  ignored by load.
- A [training checkpoint](#training-checkpoints) adds more `data/` entries — one per
  training-state kind (`data/trainable.safetensors`, and, when non-empty,
  `data/model_state.safetensors` and `data/optimizer_state.safetensors`) — and a
  `training` block in the manifest.
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

  // Optional user-supplied provenance metadata (omitted entirely when none is given).
  // Purely descriptive: it wires nothing and never affects load.
  "userMetadata": {
    "gitCommit": "9f3c1ba",
    "datasetId": "imagenet-1k@v2",
    "runName": "nightly-run-42",
    "license": "Apache-2.0"
  },

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
  // tensor inside a data entry. "default" is always present; additional sets (e.g.
  // "ema") map the same parameters, sharing data entries where the bytes are identical.
  "tensorMappings": {
    "model": {
      "default": {
        "tensors": {
          "[1]:TrainableParam#0…": { "data": "weights", "tensor": "[1]:TrainableParam#0…" },
          "[1]:TrainableParam#1…": { "data": "weights", "tensor": "[1]:TrainableParam#1…" }
        }
      },
      "ema": {
        "tensors": {
          // the first tensor differs from default → stored in the "ema" data entry
          "[1]:TrainableParam#0…": { "data": "ema", "tensor": "[1]:TrainableParam#0…" },
          // the second is byte-identical to default → shared, referenced back into "weights"
          "[1]:TrainableParam#1…": { "data": "weights", "tensor": "[1]:TrainableParam#1…" }
        }
      }
    }
  },

  // Data registry: per data entry, its storage format, compression, and hash. An
  // additional set contributes one entry holding only its distinct tensors.
  "data": {
    "weights": {
      "entry": "data/weights.safetensors",
      "format": "safetensors",
      "compression": "none",              // "none" or "zstd"; never inferred from the name
      "sha256": "734485…"                 // hash of the entry's bytes as stored (compressed)
    },
    "ema": {
      "entry": "data/ema.safetensors",
      "format": "safetensors",
      "compression": "none",
      "sha256": "9af0c1…"
    },

    // Optional host user-data bag (issue #101): format "json", never referenced by a
    // tensor mapping, so load ignores it. Present only when you attach one.
    "userData": {
      "entry": "data/user-data.json",
      "format": "json",
      "compression": "none",
      "sha256": "1c0ffe…"
    }
  },

  // Training block: present only in a training checkpoint (omitted for an inference
  // checkpoint). Records the global step and which data entry holds each state kind —
  // an empty kind (e.g. model state for a stateless model) is absent and has no entry.
  "training": {
    "checkpointVersion": 1,
    "step": 42,
    "kinds": {
      "trainableParams": "trainable",         // → data/trainable.safetensors (also the default set)
      "modelState": "model_state",            // → data/model_state.safetensors
      "optimizerState": "optimizer_state"     // → data/optimizer_state.safetensors
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

- One model per file. Any number of named weight sets over that model's parameters
  (see [Named weight sets](#named-weight-sets-default--ema)).
- Data entries are bounded by the in-memory safetensors path — checkpoints with ≥ 2 GB
  of tensor data in a single entry are not yet supported (compressed or not; the bound
  applies to both the stored and the decompressed bytes).
- A [training checkpoint](#training-checkpoints) stores the trainable weights, model
  state, optimizer state and global step of a run. The full training **rig** — the
  constituent model/loss/optimizer/scheduler graphs and their schedules — is not yet
  carried; resume rebuilds the rig from the same graphs, then `LoadCheckpoint`s the file.

# .skpt checkpoints

Related: [onnx-and-weights.md](onnx-and-weights.md) · [training.md](training.md) · [inference.md](inference.md)

## Facts

- A `.skpt` is Shorokoo's native checkpoint container: a model definition and its
  weights, reloadable as a runnable model with `Persistence.Load`.
- It has two forms with identical content: a **single file** (a standard zip archive,
  for distribution) and a **directory** of real files (for training runs). `Save` /
  `SaveAsDirectory` pick the form; loading and inspection accept either. See
  [The directory form](#the-directory-form).
- Zip entries are all **STORED** (uncompressed) and data payloads are 64-byte aligned,
  so tensor data is range-readable. Data entries can opt into Zstd
  (`.WithZstdCompressedData()`); see
  [the compression trade-off](#compressed-data-entries-the-trade-off).
- The `config.json` manifest is the **only source of wiring**: entries never reference
  each other.
- Saves are **atomic** (staged beside the target, committed by rename), so a crash never
  corrupts an existing checkpoint. The target's parent directory must already exist.
  Every other save API is atomic too; see [onnx-and-weights.md](onnx-and-weights.md#facts)
  and [training.md](training.md#save-and-resume-a-checkpoint-across-process-restarts).
- A single-file training-checkpoint save returns a `SaveReport` (committed size, time
  spent writing, flushing and committing); a directory save returns `void`. See
  [What a save costs](training.md#what-a-save-costs).
- A `.skpt` can carry [named weight sets](#named-weight-sets-default--ema) (e.g. `ema`
  alongside `default`), a [training checkpoint](#training-checkpoints) that
  `TrainingRig.Load` resumes from the file alone, and a
  [host user-data bag](#host-user-data-bag) of your own JSON.
- ONNX export and `.safetensors`/`.srk` files are separate, on-demand projections (see
  [onnx-and-weights.md](onnx-and-weights.md)).

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

Round-trip is exact: weight bytes and execution results are bit-identical.

`Persistence.From` requires a `GraphKind.ConcreteModel`; lower a module graph with
`ToConcreteArchitecture(inputHints, ...).ToConcreteModel(...)` first. Both
`.WithModel()` and `.WithWeights()` are required.

To shrink the file, compress the data entries:

```csharp
Persistence.From(concreteModel)
    .WithModel()
    .WithWeights()
    .WithZstdCompressedData()       // optional level 1–22, default 3
    .Save("model.skpt");

var loaded = Persistence.Load("model.skpt");   // decompression is transparent
```

## The directory form

The directory form holds `config.json` at the root and the models/ and data/ entries as
real files, with byte-identical content and the same manifest.

```csharp
Persistence.From(concreteModel)
    .WithModel()
    .WithWeights()
    .SaveAsDirectory("run17.skpt");        // a directory named run17.skpt

var loaded = Persistence.Load("run17.skpt");   // loads either form transparently
```

```
run17.skpt/
├── config.json
├── models/model.srk
└── data/weights.safetensors
```

Training checkpoints save the same way (`Persistence.ForTrainingCheckpoint(ckpt)
.SaveAsDirectory(path)`). Every `.skpt` load entry point (`Persistence.Load`,
`TrainingRig.Load`, `rig.LoadCheckpointFromSkpt`) accepts either form; a directory path
is the directory form. The save method alone picks the form, never the path's name.

Use the **directory** for working runs: unchanged entries stay untouched files
(diff/rsync-friendly), and one tensor file can be fetched, read or replaced on its own.
Use the **single file** to hand someone one artifact.

Convert between the forms; content is byte-identical and every entry's sha256 is
verified in transit:

```csharp
Persistence.ExtractSkpt("model.skpt", "model-dir.skpt");   // file → directory
Persistence.PackSkpt("model-dir.skpt", "model2.skpt");     // directory → file
```

Guarantees specific to the directory form:

- **Atomic commit by directory rename.** A save stages the tree in a `.tmp-` sibling and
  renames it onto the target; an interrupted save leaves the previous checkpoint (or
  nothing) plus debris no load reads. Replacing takes two renames: a failure between
  them rolls back, but a hard crash there can leave the target **absent**, with the
  previous tree under a `.tmp-` name. A polling reader should treat not-found, or a
  failed SHA-256 check mid-swap, as retryable. The single-file replace is one rename.
- **Path safety on read.** An entry path in `config.json` that escapes the checkpoint
  root (`../…`, absolute) **fails loudly**; extraction never writes outside the target.
  The check is lexical: symlinks inside an untrusted checkpoint directory are followed.
- **The manifest rules.** Stray files are ignored by load (and flagged by `Inspect`),
  and conversion carries only manifest-referenced entries.

## Training checkpoints

A training run's state persists into a `.skpt`, sharing one format with inference
checkpoints:

```csharp
using Shorokoo;   // Persistence, TrainingRig, TrainingCheckpoint

// checkpoint: from rig.CreateInitialCheckpoint() / TrainStep(); it carries its rig.
// (For a bare checkpoint, attach a rig first via rig.AdoptCheckpoint(checkpoint).)
Persistence.SaveTrainingCheckpointToSkpt(checkpoint, "run.skpt");

// Resume in a fresh process: rebuild the rig from the same graphs, then load.
var rig     = TrainingRig.FromScratch(modelGraph, lossGraph, optimizerGraph, sample, hypers);
var resumed = rig.LoadCheckpointFromSkpt("run.skpt");
var next    = rig.TrainStep(resumed, inputBatch, targetBatch);   // trainstep compiled internally, cached per fed shape
```

Or resume from the file **alone**: the static `TrainingRig.Load` rebuilds the rig from
the constituents the file carries and returns it with the loaded checkpoint:

```csharp
var (rig, resumed) = TrainingRig.Load("run.skpt");   // resumed.Rig is that rig
var next = rig.TrainStep(resumed, inputBatch, targetBatch);
```

A resumed step continues the saved trajectory. `TrainingRig.Load` costs most of a
build (see [below](#without-a-rig-in-hand)); pass a `progress:` sink to
[watch it stage by stage](training.md#watching-a-long-build), including the file and
payload reads. Signature:
`TrainingRig.Load(path, mergeContext, runtimeContext, progress)`, contexts defaulting to `ComputeContext.Default` (contexts are never
persisted). A flat safetensors checkpoint has no constituents; handed one, it fails
loudly, pointing at `rig.LoadCheckpoint`.

The builder form composes the container's features:

```csharp
Persistence.ForTrainingCheckpoint(checkpoint)
    .WithZstdCompressedData()                          // per-entry Zstd (optional level 1–22)
    .WithMetadata(runName: "nightly-42", gitCommit: "9f3c1ba")
    .Save("run.skpt");
```

What the file carries:

- **The concrete inference model** (`models/model.srk`), the trained weights bound into
  the rig's concrete architecture, as `checkpoint.ToInferenceModel()` does. The
  trainable weights double as its `default` weight set, so `Persistence.Load("run.skpt")`
  loads a runnable model.
- **The training state**, each tensor addressed individually through `tensorMappings`.
  Trainable weights and model state ride in the model's `default` mapping, stored once.
  Optimizer state has its own `default` mapping under the optimizer's model key, keyed
  `{parameterIdentifier}#opt{slot}`. Bytes live in `data/trainable.safetensors`,
  `data/model_state.safetensors` (omitted for a stateless model) and
  `data/optimizer_state.safetensors` (omitted for a stateless optimizer like plain SGD),
  keyed by struct field name.
- **The run counters and the step's loss** (global step, epoch, batch index, loss) in
  the manifest's `training` block, so `Persistence.Inspect` reports them without reading
  tensors. The training loop owns them (`TrainStep` advances the step and carries
  epoch/batch through). Unknown epoch/batch, and the loss of an initial or bare
  checkpoint, are omitted and reload as `null`, never `0`.
- **The [training history](training.md#the-training-history)**, when non-empty, as
  `data/history.safetensors` (registry key `history`), one tensor per column: `step` (`int64[n]`), `loss` (`float32[n]`), `epoch`
  and `batch_index` (`int64[n]`) with `epoch_present` / `batch_index_present`
  (`bool[n]`; `false` reads back `null`), and per hyperparameter
  `hyperparameter/<name>` (its dtype, shape `[n, …valueShape]`) with
  `hyperparameter_present/<name>` (`bool[n]`). A file without it, or a load whose
  components omit `CheckpointComponents.History`, gives an empty history. Save
  `checkpoint.WithoutHistory()` to write none.
- **The rig's constituents**: `models/model-arch.srk`, `models/loss.srk`,
  `models/optimizer.srk`, and `models/scheduler.srk` when any hyperparameter is
  scheduled, plus the manifest's `training.rig` block (their registry keys, the
  hyperparameter bindings in the optimizer's order, the RNG config). Model-input shapes
  ride on the architecture. Every training `.skpt` carries these.

Round-trip is exact: state is bit-identical, counters are preserved, and a resumed
`TrainStep` reproduces the pre-save trajectory. Loading fails loudly, naming the
tensor, on a mapped state tensor the rig does not declare, a declared one the file does
not map, a tampered entry (sha256), or an element type or dimension mismatch with the
rig's parameters (see [training.md](training.md)).

### Without a rig in hand

Only the last option builds a training rig.

**To run the model**, load it; this costs a graph read and a weight bind:

```csharp
var model = Persistence.Load("run.skpt");                  // ConcreteModel, weights bound
```

**To score a validation set**, load the model composed with its training loss. Inputs
are the model's plus the target; the output is the scalar loss:

```csharp
var eval = Persistence.LoadEvaluationModel("run.skpt");    // [model inputs…, targets] → loss
var loss = ComputeContext.Default.Execute(eval, batch, targets)[0].ToTensorData().ValueAt<float>(0);
```

This splices two graphs and binds the trained weights; no trainstep, autodiff,
optimizer lowering or initializer runs. There is no weight set to select. When the loss
ignores its target the evaluation model takes the model inputs alone; check with
`Persistence.EvaluationModelTakesTarget(path)`.

**To continue training**, rebuild the rig from the file:

```csharp
var (rig, ckpt) = TrainingRig.Load("run.skpt");
```

This redoes composition, autodiff and the optimizer's per-parameter lowering. The
model's initializers are deferred, since the checkpoint overwrites their values;
`rig.CreateInitialCheckpoint()` runs them on demand, giving the values an eager build
would.

Format pairs: `Persistence.SaveTrainingCheckpoint` / `Persistence.LoadTrainingCheckpoint`
(and `rig.LoadCheckpoint`) handle the **flat** [safetensors format](training.md);
`SaveTrainingCheckpointToSkpt` / `ForTrainingCheckpoint`, `rig.LoadCheckpointFromSkpt`
and `TrainingRig.Load` handle `.skpt`, which `Persistence.Load` and
`Persistence.LoadEvaluationModel` also read. No entry point sniffs bytes: the wrong
format fails immediately, naming both formats and the right entry point. Identify an
unknown file with `Persistence.Inspect`.

## Provenance metadata

A checkpoint records its **producer** (framework version) and **creation time**. You
can add your own `string → string` **provenance metadata** to the manifest:

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

Four well-known keys are named parameters; other pairs go in the map argument, and
calls accumulate:

```csharp
.WithMetadata(new Dictionary<string, string> { ["experiment"] = "ablation-7" },
              gitCommit: "9f3c1ba")
```

`Persistence.Inspect` reports it in its own section:

```csharp
var info = Persistence.Inspect("model.skpt");
foreach (var (key, value) in info.Skpt!.UserMetadata ?? new Dictionary<string, string>())
    Console.WriteLine($"{key} = {value}");
```

- **Informational only.** Load and binding ignore it; Shorokoo never validates it or
  fills it from the environment.
- Values are stored verbatim; the text summary sanitizes control characters, while
  `UserMetadata` stays raw. Readers tolerate unknown keys.
- **Absent by default**: no `userMetadata` key, byte-identical output.

## Host user-data bag

For structured state your **program** reads back on resume (typically data-pipeline
state: corpus, shuffle strategy, stream position), attach a **user-data bag**: a JSON
object stored as `data/user-data.json` and returned verbatim.

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

Read it back through `Inspect`:

```csharp
var info = Persistence.Inspect("model.skpt");

System.Text.Json.Nodes.JsonObject? bag = info.Skpt!.UserData;   // null when absent
PipelineState? state = info.Skpt!.GetUserData<PipelineState>();  // default when absent
```

`WithUserData(JsonObject value)` takes a `System.Text.Json.Nodes.JsonObject` directly.

- The root must be a JSON *object*; a list or scalar is rejected at save (wrap it, e.g.
  `{ "items": [ … ] }`).
- Only well-formedness is validated; load ignores the bag.
- `$`-prefixed **top-level** keys are reserved and rejected at save.
- `Inspect`'s text summary shows only a key count (`user-data: 4 keys`).
- **Absent by default** (byte-identical output); always stored uncompressed.

## Compressed data entries: the trade-off

- An uncompressed (default) data entry is STORED and 64-byte aligned, so it can be
  memory-mapped or range-read.
- A Zstd entry is smaller but must be decompressed in full to read any tensor; it is not
  aligned.
- Compression is recorded **only in the manifest** (`compression: "zstd"`), never
  inferred from the extension. The `sha256` covers the **stored (compressed) bytes**.
- `config.json` and `models/*.srk` (already Zstd-compressed internally) are never
  compressed by the option. The zip framing stays STORED; a compressed entry extracts to
  a `.zst`-decodable stream.
- An entry whose bytes contradict its declared compression fails loudly on load.

## Named weight sets (default + ema)

A checkpoint can carry **several weight sets over the same parameters** (e.g. EMA
weights). `.WithWeights()` writes the model's weights as `default`; add a set with
`.WithWeights(setName, values)`:

```csharp
// emaWeights: IReadOnlyDictionary<string, TensorData> keyed by weight-parameter identifier.
Persistence.From(concreteModel)
    .WithModel()
    .WithWeights()                    // the "default" set (the model's own weights)
    .WithWeights("ema", emaWeights)   // an additional set over the same parameters
    .Save("model.skpt");

var raw = Persistence.Load("model.skpt");            // binds "default"
var smoothed = Persistence.Load("model.skpt", "ema"); // binds "ema"
```

- An unknown set name fails loudly, listing the file's sets.
- A tensor identical (dtype, shape, bytes) to one already stored is referenced, not
  copied; a set's distinct tensors go in `data/<setName>.safetensors`.
- A set must cover exactly the weight parameters, with matching dtype and shape, or the
  save fails loudly.
- A `default`-only save is byte-identical to the single-set output. Set names are
  non-empty identifiers over `[A-Za-z0-9._-]`, excluding `default` and `weights`.

Computing EMA weights is a training concern; the container only carries and selects them.

## Inspecting a .skpt

`Persistence.Inspect("model.skpt")` summarizes either form, reading only the zip
central directory (or file listing), `config.json` and `data/user-data.json`:
producer, creation time, [provenance metadata](#provenance-metadata), a
[user-data](#host-user-data-bag) key count, the registries and mapping-set names. It
never reads tensor data or verifies sha256s, and flags manifest/archive mismatches,
unexpected compression and unknown keys. See
[onnx-and-weights.md](onnx-and-weights.md#identify-and-summarize-a-file-persistenceinspect).


To land a foreign `.safetensors` file as a checkpoint in one call (strict import, see
[onnx-and-weights.md](onnx-and-weights.md#weight-exchange-with-naming-schemes-exportsafetensors--importsafetensors)):

```csharp
ComputationGraph model = Persistence.ImportSafeTensorsToCheckpoint(
    arch, "foreign.safetensors", "model.skpt", scheme);
```

## Container layout

Zip entry paths in the single file, real paths in the directory form:

```
model.skpt
├── config.json                the manifest: all metadata and all wiring
├── models/
│   └── model.srk              the model definition (srk1 encoding, weights stripped)
└── data/
    ├── weights.safetensors    tensor data (safetensors layout)
    └── user-data.json         optional host user-data bag (JSON object)
```

- `models/model.srk` is a valid `.srk` concrete-model file whose weight tensors are
  zero-cost placeholders of the same dtype/shape. The RNG identity parameter stays
  embedded, so a reloaded model reproduces the original's randomness (see
  [rng-configuration.md](rng-configuration.md)).
- `data/weights.safetensors` is a plain [safetensors](https://huggingface.co/docs/safetensors)
  file keyed by internal parameter identifiers; any safetensors reader can read it.
- A [training checkpoint](#training-checkpoints) replaces `data/weights.safetensors` with
  its state and history entries and adds the rig's `models/` entries.

## The `config.json` manifest

```jsonc
{
  "format": "skpt",
  "skptVersion": 1,
  "createdUtc": "2026-07-21T13:32:39Z",
  "producer": { "shorokoo": "0.1.0" },    // framework version that wrote the file

  // Optional provenance metadata; never affects load.
  "userMetadata": {
    "gitCommit": "9f3c1ba",
    "datasetId": "imagenet-1k@v2",
    "runName": "nightly-run-42",
    "license": "Apache-2.0"
  },

  // Model registry. A training checkpoint also registers "modelArch", "loss",
  // "optimizer" and (if scheduled) "scheduler"; only "optimizer" has a tensor mapping.
  "models": {
    "model": {
      "entry": "models/model.srk",
      "format": "srk1",
      "stage": "concrete-model",
      "sha256": "6824d4…"                 // hash of the entry's bytes
    }
  },

  // Named mapping sets, parameter → tensor in a data entry. "default" is always present.
  // Training: optimizer state maps under "optimizer", keyed "{parameterIdentifier}#opt{slot}", e.g.
  //   "[1]:TrainableParam#0…#opt0": { "data": "optimizer_state", "tensor": "TrainableParam#0…_opt_0" }.
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
          // differs from default → stored in "ema"
          "[1]:TrainableParam#0…": { "data": "ema", "tensor": "[1]:TrainableParam#0…" },
          // identical to default → shared
          "[1]:TrainableParam#1…": { "data": "weights", "tensor": "[1]:TrainableParam#1…" }
        }
      }
    }
  },

  // Data registry.
  "data": {
    "weights": {
      "entry": "data/weights.safetensors",
      "format": "safetensors",
      "compression": "none",              // "none" or "zstd"
      "sha256": "734485…"                 // hash of the bytes as stored
    },
    "ema": {
      "entry": "data/ema.safetensors",
      "format": "safetensors",
      "compression": "none",
      "sha256": "9af0c1…"
    },

    // Training, non-empty history only; unmapped:
    //   "history": { "entry": "data/history.safetensors", "format": "safetensors", … },

    // Optional user-data bag; unmapped, ignored by load.
    "userData": {
      "entry": "data/user-data.json",
      "format": "json",
      "compression": "none",
      "sha256": "1c0ffe…"
    }
  },

  // Training checkpoints only. epoch, batchIndex and loss are omitted when unknown.
  "training": {
    "checkpointVersion": 1,
    "step": 42,                           // 0-based
    "epoch": 3,                           // 0-based
    "batchIndex": 17,                     // 0-based, within the epoch
    "loss": 0.3125,                       // loss of the step that produced the checkpoint

    // The rig recipe; model-input shapes live in the architecture.
    "rig": {
      "rigVersion": 1,
      "archModel": "modelArch",           // model-registry keys
      "lossModel": "loss",
      "optimizerModel": "optimizer",
      "schedulerModel": "scheduler",      // omitted when nothing is scheduled

      // In the optimizer's order. "baked": constant inline ("value" = base64 little-endian
      // bytes); "runtime": shape only; "scheduled": the scheduler's output of that name.
      "hyperparameters": [
        { "name": "learningRate", "kind": "scheduled" },
        { "name": "weightDecay", "kind": "baked", "dtype": "Float32", "shape": [], "value": "zcz…" },
        { "name": "gradScale", "kind": "runtime", "shape": [] }
      ],

      // See rng-configuration.md.
      "rng": {
        "masterSeed": 12345,
        "initMasterSeed": 999,            // omitted to derive from masterSeed
        "runMasterSeed": 7,               // omitted likewise
        "algorithm": "Threefry2x32",
        "overrides": [                    // omitted when none
          { "collection": "Params", "path": [1, 3], "seed": 42 }
        ]
      }
    }
  }
}
```

Rules:

- **Keys a reader does not interpret are ignored.** A `skptVersion` other than `1` is
  refused with a clear message.
- **Integrity is checked on load.** A missing entry, a sha256 mismatch, or a mapping that
  does not cover the model's parameters exactly fails loudly, naming the entry or
  parameter.

## Current limits

- One **weight-bearing** model per file (the `model` entry), with any number of
  [named weight sets](#named-weight-sets-default--ema). A training checkpoint's extra
  `models/` entries bind no weights.
- A single data entry must hold under 2 GB of tensor data, both stored and decompressed.
- The flat safetensors training format carries no rig constituents: rebuild the rig from
  the same graphs, then `rig.LoadCheckpoint`.
- Precompiled artifacts are not supported.

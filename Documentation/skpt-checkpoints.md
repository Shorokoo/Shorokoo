# .skpt checkpoints

Related: [onnx-and-weights.md](onnx-and-weights.md) · [training.md](training.md) · [inference.md](inference.md)

## Facts

- A `.skpt` is Shorokoo's native checkpoint container: a model definition and its
  weights, reloadable as a runnable model with `Persistence.Load`. It has **two on-disk
  forms with identical content**: a **single file** (the distribution form) and a
  **directory** of real files (the working form for training runs — diffable,
  rsync-friendly, entries readable and replaceable as plain files). Saving picks the
  form explicitly (`Save` vs `SaveAsDirectory`); loading and inspection accept either,
  and `ExtractSkpt` / `PackSkpt` convert both ways. See
  [The directory form](#the-directory-form).
- The single-file form is a **standard zip archive** — any unzip tool can list and
  extract its entries — whose entries are all **STORED** (uncompressed), so tensor data
  remains range-readable through the zip central directory. Data payloads are 64-byte
  aligned inside the file. (In the directory form alignment is moot — a real file is
  already page-aligned and range-readable.)
- Data entries can **opt into Zstd compression** (`.WithZstdCompressedData()`): the zip
  framing stays STORED, and a single Zstd layer lives inside the entry's bytes, declared
  per entry in the manifest (`compression: "zstd"`). The default remains uncompressed —
  and byte-for-byte identical to output without the option. See
  [the compression trade-off](#compressed-data-entries-the-trade-off).
- A single `config.json` manifest is the **only source of wiring**: entries never
  reference each other; every mapping (model → serialization format, parameter →
  stored tensor, data entry → storage format) lives in the manifest.
- Saves are **atomic** in both `.skpt` forms (staged beside the target — a temp file or a
  temp directory — and committed by rename): a crash mid-save never corrupts an existing
  checkpoint, and an interrupted directory save is never visible at the target path.
  (The directory form's *replace* takes two renames, so a hard crash in that one window can
  leave the target briefly **absent** — never half-written; see
  [The directory form](#the-directory-form).)
  The target's parent directory must already exist. The flat safetensors training
  checkpoint (`checkpoint.Save`) is written through the same atomic file path, as is
  every save/export API on the `Persistence.*` facade — `ExtractSkpt` / `PackSkpt`,
  `Persistence.ExportSafeTensors`, `Persistence.ExportOnnx`. The raw layers below that
  facade write **in place** — see
  [onnx-and-weights.md](onnx-and-weights.md#facts). See also
  [training.md](training.md#save-and-resume-a-checkpoint-across-process-restarts).
- This version writes an **inference checkpoint of a concrete model** (definition +
  weights). It can also carry **additional named weight sets** over the same parameters
  (e.g. an `ema` set alongside `default`), selected at load time — see
  [Named weight sets](#named-weight-sets-default--ema).
- A `.skpt` can also persist a **training checkpoint** — the trainable weights, model
  state, optimizer state, the run counters (global step, epoch, batch index) and the step's
  loss of a training run — with every state tensor addressed individually through the manifest's
  tensor mappings, alongside the concrete inference model, so a run
  resumes across process restarts and the same file also loads as an inference model. Every
  training `.skpt` also carries the **rig's constituents** — the concrete architecture, the loss
  and optimizer graphs, the composed scheduler when any hyperparameter is scheduled, the
  hyperparameter bindings and the RNG config — so `TrainingRig.Load(path)` rebuilds the whole
  rig, and its resumed checkpoint, from the file alone. See
  [Training checkpoints](#training-checkpoints). Precompiled artifacts are a future extension
  of the same container.
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

## The directory form

The same checkpoint can be saved as a **directory** instead of a single file:
`config.json` at the root, the models/ and data/ entries as real files and folders,
with byte-identical content and the same manifest describing both forms.

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
.SaveAsDirectory(path)`), and every `.skpt` load entry point — `Persistence.Load`,
`TrainingRig.Load`, `rig.LoadCheckpointFromSkpt`,
`Persistence.LoadTrainingCheckpointFromSkpt` — accepts either form; a directory path
is unambiguously the directory form (no content sniffing).

When to use which:

- **Directory** — the working form: a run writing a checkpoint every N steps leaves
  unchanged entries as untouched files (diff/rsync-friendly), a partial download can
  fetch one tensor file, and reading or replacing one weight set is a plain file
  operation. This is where the rest of the field landed (safetensors repos, Orbax,
  `.mlpackage`).
- **Single file** — the distribution form: one artifact to hand someone.

Which form a save writes is always the explicit choice of method — never inferred from
the path, since a directory checkpoint may itself be named `run17.skpt`.

Convert between the forms at will; entry content is byte-identical in both directions,
and every entry's recorded sha256 is verified in transit:

```csharp
Persistence.ExtractSkpt("model.skpt", "model-dir.skpt");   // file → directory
Persistence.PackSkpt("model-dir.skpt", "model2.skpt");     // directory → file
```

Guarantees specific to the directory form:

- **Atomic commit by directory rename.** A save stages the whole tree in a `.tmp-`
  sibling directory and commits by renaming it onto the target, so the target path
  never names a half-written checkpoint: an interrupted save leaves the previous
  checkpoint in place (or, for a first save, no target at all) plus staged debris that
  no load path reads. Replacing an existing checkpoint takes two renames (a directory
  rename cannot overwrite): a failure between them rolls the previous checkpoint back
  into place, and only a hard crash inside that tiny window can leave the target
  absent — the previous tree then still exists, complete, under a `.tmp-` sibling
  name. A concurrent reader can therefore observe the checkpoint briefly *missing*
  during a replace (never silently half-written); a polling reader should treat
  not-found — or a failed SHA-256 check from catching the swap mid-read — as
  retryable. The single-file form's replace is one atomic rename with no such window.
- **Path safety on read.** Entry paths come from `config.json`, so a hostile manifest
  could name `../…` or an absolute path; every read resolves the entry against the
  checkpoint root and **fails loudly** on any path that escapes it (the same rule the
  ONNX external-data reader applies to its `location` field). Extraction of a hostile
  zip is bounded the same way — nothing is ever written outside the target directory.
  The check is lexical, like the ONNX rule: it stops `..` and absolute paths, while a
  symlink planted inside an untrusted checkpoint directory is followed by the
  filesystem — treat a checkpoint directory from an untrusted source accordingly.
- **The manifest still rules.** Only the manifest and the entries it references are a
  checkpoint; stray files in the directory are ignored by load (and flagged by
  `Inspect`), and conversion carries only manifest-referenced entries.

## Training checkpoints

A training run's state — the trainable weights, model state, optimizer state, the run
counters (global step, epoch, batch index) and the step's loss — persists into a `.skpt`
too, so training resumes across process restarts
in the native container (inspectable manifest, per-entry Zstd, atomic write, provenance
metadata), sharing one on-disk format with inference checkpoints.

```csharp
using Shorokoo;   // Persistence, TrainingRig, TrainingCheckpoint

// checkpoint: a TrainingCheckpoint from rig.CreateInitialCheckpoint() / TrainStep(). It carries
// its rig, which is the source of the self-describing inference model — no model graph or example
// input is needed. (For a bare checkpoint, attach a rig first via rig.AdoptCheckpoint(checkpoint).)
Persistence.SaveTrainingCheckpointToSkpt(checkpoint, "run.skpt");

// Resume in a fresh process: rebuild the rig from the same graphs, then load.
var rig     = TrainingRig.FromScratch(modelGraph, lossGraph, optimizerGraph, sample, hypers);
var resumed = rig.LoadCheckpointFromSkpt("run.skpt");
var next    = rig.TrainStep(resumed, inputBatch, targetBatch);   // trainstep compiled once internally
```

Or resume from the file **alone**, with no model/loss/optimizer graphs in hand — the static
`TrainingRig.Load` rebuilds the rig from the constituents the checkpoint carries and returns it
together with the checkpoint loaded against it:

```csharp
var (rig, resumed) = TrainingRig.Load("run.skpt");   // resumed.Rig is that rig
var next = rig.TrainStep(resumed, inputBatch, targetBatch);
```

The rebuilt rig re-derives its trainstep exactly as a fresh build does, so a resumed step
continues the saved trajectory — and takes as long, so on a large model set `mergeContext`'s
`Progress` sink to [watch it stage by stage](training.md#watching-a-long-build) rather than wait
blind. Its two optional arguments are the compute contexts that seed the rebuilt rig
(`TrainingRig.Load(path, mergeContext, runtimeContext)`, each defaulting to
`ComputeContext.Default`) — contexts are never persisted, so a reloaded run gets fresh ones.
Handed a flat safetensors checkpoint — which stores training state only, with no constituents to
rebuild from — it fails loudly, pointing at `rig.LoadCheckpoint`.

To compose the container's features, use the builder form:

```csharp
Persistence.ForTrainingCheckpoint(checkpoint)
    .WithZstdCompressedData()                          // per-entry Zstd (optional level 1–22)
    .WithMetadata(runName: "nightly-42", gitCommit: "9f3c1ba")
    .Save("run.skpt");
```

What the file carries:

- **The concrete inference model** in `models/model.srk` (definition, weights stripped),
  built from the checkpoint's trained weights bound into the rig's retained concrete
  architecture — the same weight-bind `checkpoint.ToInferenceModel()` uses, so the container's
  self-describing model can never disagree with extraction. The trainable weights double as the model's
  `default` weight set, so the same file loads as a runnable inference model with
  `Persistence.Load("run.skpt")` — no separate export step.
- **The training state, every tensor addressed individually** through the manifest's
  `tensorMappings`. The trainable weights and model state are parameters the architecture
  owns, so they ride in the model's `default` mapping — the very mapping inference loading
  binds, keyed by parameter identifier, so the bytes live once. The optimizer state gets
  its own `default` mapping under the optimizer constituent's model key, keyed
  `{parameterIdentifier}#opt{slot}` — one entry per (trainable parameter × optimizer state
  slot) instance, composing the arch-owned parameter identity with the optimizer-owned
  slot index. The bytes live in separate `data/` entries (`data/trainable.safetensors`;
  `data/model_state.safetensors`, omitted for a stateless model; and
  `data/optimizer_state.safetensors`, omitted for a stateless optimizer like plain SGD),
  each storing its tensors keyed by struct field name.
- **The run counters and the step's loss** — the global step, plus the epoch and batch index
  and the loss of the step that produced the checkpoint — recorded in the manifest's
  `training` block (so `Persistence.Inspect` reports the counters without reading tensor
  data). Step, epoch and batch index are host-owned: the
  training loop advances them (`TrainStep` advances the step and carries epoch/batch through
  unchanged), and they are persisted so a resumed run restores its position. A checkpoint whose
  epoch/batch position is genuinely unknown omits them and reloads them as `null`, and so does
  one no training step produced a loss for (an initial or bare checkpoint) — never a
  sentinel `0`.
- **The rig's constituents** — the concrete architecture, the loss graph, the optimizer graph
  and, when any hyperparameter is scheduled, one composed scheduler model — as ordinary `models/`
  entries (`models/model-arch.srk`, `models/loss.srk`, `models/optimizer.srk`,
  `models/scheduler.srk`), with the non-graph part of the recipe recorded in the manifest's
  `training.rig` block: the model-registry keys naming which `models/` entries those
  constituents are (`archModel`, `lossModel`, `optimizerModel`, and `schedulerModel` when a
  scheduler was composed), the hyperparameter bindings, in the optimizer's declared order, and
  the RNG config the rig was built with. The model-input shapes ride on the architecture itself,
  not the manifest. This is what `TrainingRig.Load` rebuilds a rig from, and every training `.skpt`
  this version writes carries it.

Round-trip is exact: reloaded trainable params, model state and optimizer state are
bit-identical, the counters are preserved, and a resumed `TrainStep` reproduces the pre-save
trajectory. Loading validates against the rig's struct definitions with the same fail-loud
contract as the flat format — a checkpoint from a different model or optimizer (a state
tensor mapped that the rig does not declare, or a declared one the checkpoint does not
map — the mismatch is named), a rank mismatch, or a tampered entry (sha256) fails loudly.

Reconstruct without a rig by supplying the struct defs directly:

```csharp
TrainingCheckpoint ckpt = Persistence.LoadTrainingCheckpointFromSkpt(
    "run.skpt", trainableParamDef, modelStateDef, optimizerStateDef);
```

Each on-disk format has its own save/load pair. `Persistence.SaveTrainingCheckpoint` /
`Persistence.LoadTrainingCheckpoint` (and `rig.LoadCheckpoint`) handle the **flat**
[safetensors format](training.md); `SaveTrainingCheckpointToSkpt` /
`ForTrainingCheckpoint` and `LoadTrainingCheckpointFromSkpt` (and
`rig.LoadCheckpointFromSkpt`, and the static `TrainingRig.Load`, which rebuilds the rig
from the file alone) handle the `.skpt` container. No load entry point sniffs
the file's bytes to pick a format: handing one the other format fails immediately with an
error naming both formats and the entry point that reads the file's actual format. To
identify a genuinely unknown file first, use `Persistence.Inspect`.

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
  same rule `.srk` v1 follows with its header. The entry's manifest `sha256` covers
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

`Persistence.Inspect("model.skpt")` identifies the container — either form: a
directory path is inspected as [the directory form](#the-directory-form), its file
listing playing the central directory's role — and summarizes its
manifest — whole-archive metadata (producer, creation time, any
[user provenance metadata](#provenance-metadata), and a one-line count of the
[host user-data bag](#host-user-data-bag) with the full object on `Skpt.UserData`),
the model and data registries, the mapping-set names — reading only the zip
central directory, `config.json`, and (when present) the small `data/user-data.json`
entry, never the tensor data. The recorded per-entry sha256s are reported as written
but not verified (a full `Persistence.Load` verifies them), and cheap sanity
observations flag manifest/archive mismatches, compressed entries where STORED
is expected, and unknown manifest keys. See the inspection section in
[onnx-and-weights.md](onnx-and-weights.md#identify-and-summarize-a-file-persistenceinspect).


A foreign `.safetensors` file (e.g. PyTorch/timm weights) lands as a native
checkpoint in one call — the strict safetensors import (see
[onnx-and-weights.md](onnx-and-weights.md#weight-exchange-with-naming-schemes-exportsafetensors--importsafetensors))
followed by this same writer:

```csharp
ComputationGraph model = Persistence.ImportSafeTensorsToCheckpoint(
    arch, "foreign.safetensors", "model.skpt", scheme);
```

## Container layout

The layout is the same in both forms — zip entry paths in the single file, real file
paths in [the directory form](#the-directory-form):

```
model.skpt
├── config.json                the manifest: all metadata and all wiring
├── models/
│   └── model.srk              the model definition (srk1 encoding, weights stripped)
└── data/
    ├── weights.safetensors    tensor data (safetensors layout)
    └── user-data.json         optional host user-data bag (JSON object)
```

- `models/model.srk` is the model **definition**: a valid `.srk` v1 concrete-model
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
- A [training checkpoint](#training-checkpoints) writes **no** `data/weights.safetensors`:
  in its place stand the per-kind state entries (`data/trainable.safetensors`, and, when
  non-empty, `data/model_state.safetensors` and `data/optimizer_state.safetensors`), which
  the inference model's own `default` mapping points into — so the trainable bytes live
  once and serve both roles. It adds a `training` block to the manifest (the run counters —
  step, epoch, batch index — and the step's loss); every state tensor is wired individually
  through `tensorMappings`, never routed by entry. It also adds the rig's constituents as
  further `models/` entries — `models/model-arch.srk`,
  `models/loss.srk`, `models/optimizer.srk`, and `models/scheduler.srk` when any
  hyperparameter is scheduled — described by the `training` block's `rig` record.
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

  // Model registry: per model, where its definition lives and how it is encoded. An
  // inference checkpoint registers the one "model" entry; a training checkpoint also
  // registers its rig's constituents here — "modelArch", "loss", "optimizer", and
  // "scheduler" when any hyperparameter is scheduled — which the training block's "rig"
  // record names by these keys. Of those, only "optimizer" carries a tensor mapping (the
  // optimizer state, below); "modelArch", "loss" and "scheduler" carry none.
  "models": {
    "model": {
      "entry": "models/model.srk",
      "format": "srk1",                   // the .srk container encoding
      "stage": "concrete-model",          // lifecycle stage of the serialized graph
      "sha256": "6824d4…"                 // hash of the entry's bytes (the graph hash)
    }
  },

  // Tensor mappings: per model, named mapping sets resolving each parameter to a
  // tensor inside a data entry. "default" is always present; additional sets (e.g.
  // "ema") map the same parameters, sharing data entries where the bytes are identical.
  // In a training checkpoint, this is also how every training-state tensor is addressed:
  // the trainable weights and model state through the model's "default" mapping (whose
  // entries then point at the "trainable" / "model_state" data entries), and the
  // optimizer state through a "default" mapping under the optimizer constituent's model
  // key ("optimizer"), keyed "{parameterIdentifier}#opt{slot}" — one entry per
  // (parameter × state slot) instance, e.g.
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
  // checkpoint). Records the host-owned run counters (step, epoch, batch index) and the
  // loss of the step that produced the checkpoint; the training *state* is addressed per
  // tensor through "tensorMappings" above, so the block routes no state — what it does
  // route is its "rig" record, whose fields are model-registry keys naming the "models"
  // entries the rig is rebuilt from. epoch, batchIndex and loss are nullable and
  // presence-gated: a checkpoint whose position is genuinely unknown, or that no training
  // step produced a loss for, omits them and reads them back as null (never a sentinel 0).
  "training": {
    "checkpointVersion": 1,               // training-block version
    "step": 42,                           // 0-based global training step
    "epoch": 3,                           // 0-based epoch counter; omitted when unknown
    "batchIndex": 17,                     // 0-based batch index within the epoch; omitted when unknown
    "loss": 0.3125,                       // loss of the step that produced the checkpoint;
                                          // omitted on an initial/bare checkpoint

    // The rig recipe: the model-registry keys of the constituents the rig is rebuilt from
    // (their graphs are the "models" entries above; of them only "optimizer" carries a
    // tensor mapping — the optimizer state's, above — while the arch, loss and scheduler
    // entries carry none), plus the non-graph part — the hyperparameter bindings and the
    // RNG config.
    // Written by every training .skpt this version produces; absent (⇒ null) on a file
    // written before rig constituents existed, which resumes only by the host rebuilding
    // the rig from the same source graphs — as does a file whose stored architecture predates
    // dims-only model-input shapes (see the pre-release caveat in the rules below).
    // Model-input shapes are not recorded here: the serialized architecture carries them itself.
    "rig": {
      "rigVersion": 1,                    // rig-block version
      "archModel": "modelArch",           // registry key of the concrete-architecture entry
      "lossModel": "loss",                // registry key of the loss constituent
      "optimizerModel": "optimizer",      // registry key of the optimizer constituent
      "schedulerModel": "scheduler",      // registry key of the composed scheduler; omitted
                                          // when no hyperparameter is scheduled

      // The optimizer's hyperparameter bindings, in its declared order. "kind" decides
      // reconstruction: "baked" carries its constant inline ("value" is base64 of the raw
      // little-endian bytes at "dtype"/"shape"), "runtime" records only the host-declared
      // shape, "scheduled" takes the scheduler model's output of the same name. A
      // hyperparameter's dtype otherwise comes from the optimizer constituent itself.
      "hyperparameters": [
        { "name": "learningRate", "kind": "scheduled" },
        { "name": "weightDecay", "kind": "baked", "dtype": "Float32", "shape": [], "value": "zcz…" },
        { "name": "gradScale", "kind": "runtime", "shape": [] }
      ],

      // The RNG config the rig was built with, so a rebuilt rig reproduces the same keyed
      // initialization and runtime randomness (see rng-configuration.md).
      "rng": {
        "masterSeed": 12345,
        "initMasterSeed": 999,            // explicit init sub-master; omitted to derive from masterSeed
        "runMasterSeed": 7,               // explicit runtime sub-master; omitted likewise
        "algorithm": "Threefry2x32",      // the bit-generator algorithm name
        "overrides": [                    // per-stream overrides; omitted when there are none
          { "collection": "Params", "path": [1, 3], "seed": 42 }
        ]
      }
    }
  }
}
```

Rules:

- **Keys are add-only.** A reader ignores unknown keys; removing or re-typing a key is
  a major-version event (a bump of `skptVersion`). `skptVersion` is `1` — the only version
  that has existed — and a file declaring any other value is refused with a clear message
  rather than half-read. There is no read path for another version and no compatibility
  shim: every format below is version 1, and stays there until a breaking change earns a
  bump.
- **Pre-release caveat: a payload break can land inside version 1.** Add-only governs the
  manifest's *keys*; while Shorokoo is pre-release, what an entry's **payload** records can
  still change in a read-breaking way without a `skptVersion` bump — and once has, in the
  stored architecture. The full caveat, its blast radius and the recovery live with the
  container the break happened in: see
  [the pre-release caveat](onnx-and-weights.md#the-srk-container).
- **Integrity is checked on load.** Every entry the manifest references must exist and
  match its recorded `sha256`; a missing entry, a hash mismatch, or a tensor mapping
  that does not cover the model's parameters exactly fails loudly, naming the
  offending entry or parameter.

## Current limits

- One **weight-bearing** model per file — the `model` registry entry, the only one a
  tensor mapping binds weights into. Any number of named weight sets over that model's
  parameters (see [Named weight sets](#named-weight-sets-default--ema)). A
  [training checkpoint](#training-checkpoints) registers further `models/` entries
  alongside it — `modelArch`, `loss`, `optimizer`, and `scheduler` when any hyperparameter
  is scheduled — as the graphs its rig is rebuilt from; no tensor mapping binds weights
  into any of them (the `modelArch`, `loss` and `scheduler` entries carry no mapping at all,
  and the `optimizer` entry's mapping addresses the optimizer *state*, not weights).
- Data entries are bounded by the in-memory safetensors path — checkpoints with ≥ 2 GB
  of tensor data in a single entry are not yet supported (compressed or not; the bound
  applies to both the stored and the decompressed bytes).
- A [training checkpoint](#training-checkpoints) carries the rig's constituent
  model/loss/optimizer/scheduler graphs alongside the run's state, so `TrainingRig.Load`
  resumes from the file alone. The flat safetensors format cannot carry constituents:
  resuming from one means rebuilding the rig from the same graphs, then loading the file
  with `rig.LoadCheckpoint`. Precompiled artifacts are still a future extension of the
  container.
- Resuming from the file alone does not reach back across pre-release payload breaks: a
  training `.skpt` whose architecture was written before model-input shapes became dims-only
  is rejected by `TrainingRig.Load` and has to be rebuilt from its source graphs and re-saved
  (see [the pre-release caveat](onnx-and-weights.md#the-srk-container)). Its state and its
  inference model still load.

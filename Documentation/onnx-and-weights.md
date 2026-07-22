# ONNX export/import and weights

Related: [inference.md](inference.md) · [core-types.md](core-types.md) · [skpt-checkpoints.md](skpt-checkpoints.md)

## Facts

- A model is a `ComputationGraph` (from `MyModule.ComputationGraph` or one of the
  lowering steps). Every graph carries a reliable `Kind` (`GraphKind.Module` /
  `ConcreteArchitecture` / `ConcreteModel`) stamped where it was produced. It can be
  converted to an ONNX `ModelProto`, or saved in Shorokoo's own `.srk`/`.zsrk` format.
- There is no one-call `graph.ToOnnxFile(path)`. Build the `ModelProto`, then save it
  with `OnnxModelExporter` (or serialize it yourself with protobuf).
- Models larger than protobuf's 2 GB message ceiling are handled with the standard
  ONNX **external data** mechanism: `OnnxModelExporter.SaveWithExternalData` on
  export, and transparent side-file loading on import.
- Pretrained weights are loaded from `.safetensors` (and compressed `.zsafetensor`).

## Export to ONNX

```csharp
using System.IO;
using Shorokoo.Core.Factory;      // FastOnnxModelBuilder
using Shorokoo.Core.Factory.IR;   // ModelProto

ModelProto model = FastOnnxModelBuilder.BuildOnnxModel(graph);  // graph: ComputationGraph, kind ConcreteModel
using var stream = File.Create("model.onnx");
ProtoBuf.Serializer.Serialize(stream, model);   // ProtoBuf = protobuf-net (a transitive dependency)
```

`ModelProto` is the protobuf type in `Shorokoo.Core.Factory.IR`; it is serialized with
protobuf-net's `ProtoBuf.Serializer` (the `ProtoBuf` namespace ships via protobuf-net,
which the library already depends on). `OnnxModelExporter.Save(model, path)`
(namespace `Shorokoo.Onnx`) does the same serialization, plus a clear error — instead
of a protobuf failure — when the model's tensor data exceeds the 2 GB protobuf message
ceiling, pointing at the external-data option below.

### Large models: external data

Protobuf caps any single message at 2 GB, so a self-contained `.onnx` cannot hold
weights approaching that size. The standard ONNX answer is **external data**:
initializer bytes live in a side file next to the model, and each externalized
`TensorProto` carries `data_location = EXTERNAL` plus `location`/`offset`/`length`
entries. Shorokoo supports it on both paths:

```csharp
using Shorokoo.Onnx;   // OnnxModelExporter, OnnxExternalDataOptions

// Opt-in export mode: initializers of at least SizeThreshold bytes go to
// "model.onnx.data"; smaller ones stay inline. The .onnx + .onnx.data pair is
// deterministic (byte-identical across runs for the same ModelProto).
OnnxModelExporter.SaveWithExternalData(model, "model.onnx");
OnnxModelExporter.SaveWithExternalData(model, "model.onnx",
    new OnnxExternalDataOptions { SizeThreshold = 1024, Alignment = 4096 });
```

`SaveWithExternalData` applies to **concrete models only**: it externalizes the
top-level graph initializers, which is complete exactly when every weight lives
there. A model that still contains module-stage machinery or unmaterialized
parameters is refused up front (`XD008`) with the actual vs required kind named.

- Tensor data is written in initializer order, each tensor aligned to `Alignment`
  bytes (default 4096) for mmap-friendly access.
- Self-contained (all-inline) export remains the default and is unchanged; with no
  initializer at or above the threshold, `SaveWithExternalData` writes no side file
  (removing a stale one from a previous save of the same path) and its output is
  identical to `Save`.
- The exported pair is standard ONNX — stock onnxruntime loads it directly.
- The passed `ModelProto` is left unmodified.

`BuildOnnxModel(ComputationGraph graph, OpSetVersion opset = OPS_21,
bool prepForOnnx = false)` requires a
`GraphKind.ConcreteModel` graph — anything else fails fast with `FW045` naming the
actual and required kinds (only a concrete model can satisfy the vanilla-ONNX
guarantee). It clones the graph (no mutation), lowers it for ONNX, and emits
nodes, subgraphs, and functions.
The default `OPS_21` is the export **baseline**: if the graph (including
function bodies) contains operators introduced after opset 21 (e.g.
`Attention`, opset 23), the model's `opset_import` is raised automatically
just far enough to cover them. Models up to opset 26 execute on the bundled
ONNX Runtime 1.26 — see [limitations.md](limitations.md) for the stamping
policy.

### The vanilla dialect is a guarantee

`BuildOnnxModel` only ever writes **vanilla ONNX**: every node is a standard
ONNX op or a call to a `FunctionProto` emitted into the same file, so the model
loads in any stock ONNX runtime with no Shorokoo involvement. It therefore
requires a **concrete model** (from `ToConcreteArchitecture` →
`ToConcreteModel`). Exporting a module-stage graph — one still carrying
Shorokoo's internal orchestration ops (`ShrkCreateModule`, `ShrkModelInvoke`,
…) — throws at export time with the offending ops named, instead of writing a
file that only fails later when a third-party runtime rejects the custom ops.
Module-stage graphs are persisted with the `.srk`/`.zsrk` format below, which
uses Shorokoo's internal dialect and is re-imported by Shorokoo only.

### Graph input/output names and shapes

Exported graph inputs and outputs are named from the model's signature — the
names by which the graph's inputs are addressed in Shorokoo (e.g. `[Hyper]` /
input parameter names), deduplicated deterministically (`x`, `x_2`, …).
Unnamed slots fall back to `input_{i}` / `output_{i}`. Every input and output
`ValueInfoProto` carries its dtype; dimension info is stamped wherever it is
known — a statically known rank produces that many symbolic (dynamic) dims
named `{name}_dim{i}`, a known rank-0 value is stamped as a true scalar, and a
rank-agnostic `Tensor<T>` boundary stays fully dynamic. Tools like Netron or
`InferenceSession.InputMetadata` therefore see the model's logical signature
directly, and `OnnxModelImporter` round-trips the names.

### Parameters in the exported graph

A concrete model's parameters — trainable weights and state params (e.g.
BatchNorm running stats) alike — are emitted as `graph.initializer`
`TensorProto`s, following ONNX convention; they are never baked into
`Constant` op-nodes. Each initializer carries two Shorokoo metadata props:
`IsTrainable` (`"true"`/`"false"`) and `IdentifierTemplate` (the parameter's
name). Initializer *tensor names* are internal ids (`N{k}_T{s}`), so match
parameters by the `IdentifierTemplate` metadata, not by tensor name.
Re-importing an exported model rebuilds the parameters with their
trainability and names intact, so a loaded model remains trainable and its
weights can be re-bound by name with `ToConcreteModel`.

## Import from ONNX

```csharp
using Shorokoo.Onnx;   // OnnxModelImporter

ComputationGraph g1 = OnnxModelImporter.FromOnnxModel("model.onnx");
ComputationGraph g2 = OnnxModelImporter.FromOnnxModel(byteArray);
ComputationGraph g3 = OnnxModelImporter.FromOnnxModel(stream);
```

Models written by Shorokoo carry a graph-kind metadata tag (`shrk_graph_kind` in
the model's metadata props), so an imported graph's `Kind` is the kind it was
saved with. Foreign models have no tag and are classified by op-scanning. A tag
that is structurally impossible for the model's content (a hand-edited or
corrupt file) fails the import loudly.

Models using ONNX external data (the standard layout for large third-party models)
load transparently from a **file path** — `location` keys resolve against the model
file's directory, honoring `offset`/`length` slicing. When importing from bytes or a
stream, pass the directory the side files live in:

```csharp
ComputationGraph g = OnnxModelImporter.FromOnnxModel(
    byteArray, externalDataDirectory: "/path/to/model/dir");
```

External-data loading fails loudly (naming the tensor and the file) rather than
zero-filling: a missing side file, a `location` escaping the model's directory, an
out-of-range `offset`/`length`, a `length` contradicting the tensor's shape/dtype,
or an external-data model imported from a stream/bytes without
`externalDataDirectory` all throw a `ModelException`.

## Save/load Shorokoo graph format (`.srk` / `.zsrk`)

With `CompressedFormatUtils`:

```csharp
string path = CompressedFormatUtils.SaveFastGraphToFile("model.zsrk", graph);     // compressed
ComputationGraph g = CompressedFormatUtils.LoadFastGraphFromFile("model.zsrk");
byte[] bytes = CompressedFormatUtils.SaveFastGraphToBinary(graph, compressed: true);
```

`.zsrk` = Zstandard-compressed; `.srk` = uncompressed. The extension is auto-selected
from the `compressed` flag, but it is only a hint for humans: **how a file parses is
decided by its content, never by its extension** — a renamed file loads identically.

### The `.srk` v2 container

Every file written today is a self-describing, versioned container:

```
magic "SRK\x02" | u16 headerLen (little-endian) | JSON header | payload
```

The payload is the graph serialized as an ONNX `ModelProto` (Shorokoo's internal
dialect allowed), wrapped in **exactly one** compression layer when the header says
so. Header fields (add-only across minor revisions; unknown fields are ignored):

```jsonc
{
  "srkVersion": 2,
  "stage": "module" | "concrete-architecture" | "concrete-model",
  "compression": "none" | "zstd",
  "payloadSha256": "…",   // lowercase hex SHA-256 of the payload bytes as stored
  "producer": { "shorokoo": "…", "irVersion": 10, "opsets": { "": 21, "…": 1 } }
}
```

- `stage` records the graph's `GraphKind` (see
  [inference.md](inference.md#the-lowering-pipeline)). The writer records the
  graph's **stamped** kind (`graph.Kind`), and the loader stamps the loaded
  graph's `Kind` from the header (falling back to op-scan classification for
  legacy/foreign data), so loaders can refuse a mismatched file up front instead
  of failing at run time with `No Op registered for ShrkCreateModule`. Pass the
  optional `requiredStage` argument to enforce it:

  ```csharp
  // Throws a clear stage-mismatch error if model.zsrk holds a module-stage graph:
  var g = CompressedFormatUtils.LoadFastGraphFromFile(
      "model.zsrk", requiredStage: GraphKind.ConcreteModel);
  ```

- `payloadSha256` makes corruption and truncation fail loudly, with an error naming
  the file and the failure.
- `SrkFileFormat.TryReadHeaderFromFile(path)` reads the header (`SrkHeader`) without
  loading the graph — useful to identify a file cheaply; it returns `null` for
  pre-container legacy files. For a format-agnostic version of the same idea, see
  [`Persistence.Inspect`](#identify-and-summarize-a-file-checkpointinspect).
- `producer` is informational; the payload dialect remains versioned by the embedded
  ONNX `ir_version`/opsets themselves.

Legacy files from before the container (bare protobuf, single-Zstd, and the retired
double-Zstd layout of `SaveCompressedArchitecture`) still load — the loader sniffs
their layout from content. Writers emit v2 only; the `SaveCompressedArchitecture` /
`LoadCompressedArchitecture` helpers are `[Obsolete]` shims over the v2 writer and
the universal loader.

Unlike `BuildOnnxModel`, this format is Shorokoo's **internal dialect**: it
accepts any graph — module-stage graphs with their internal ops included —
keeps internal `N{k}_T{s}` tensor names, and is only loadable by Shorokoo
(`LoadFastGraphFromFile` / `OnnxModelImporter`). Use it for Shorokoo-to-Shorokoo
persistence; use `BuildOnnxModel` for anything meant to leave Shorokoo.

## Load pretrained weights (SafeTensors)

With `SafeTensorLoader`:

```csharp
ModelParamList weights = SafeTensorLoader.LoadModelParamSet("weights.safetensors");
Dictionary<string, TensorData> byName = SafeTensorLoader.LoadTensorDictionary("weights.safetensors");
TensorData single = SafeTensorLoader.LoadSingleTensor("bias.safetensors");
List<SafeTensor> all = SafeTensorLoader.LoadSafeTensors("weights.safetensors");
```

A truncated file (interrupted download or copy, disk full) is refused up front with an
error naming truncation, the file, and the declared vs. actual byte counts: the declared
header length and every tensor's `data_offsets` range are validated against the actual
file length before any tensor is materialized. Training checkpoints
(`TrainingRig.LoadCheckpoint`) share this loader, so a truncated checkpoint fails the
same way, naming the checkpoint path.

Save:

```csharp
SafeTensorLoader.SaveSafeTensors("out.safetensors", listOfSafeTensors);
```

Compressed (`.zsafetensor`) variants live in `CompressedFormatUtils`:
`SaveCompressedSafeTensors`, `LoadCompressedSafeTensors`,
`SaveCompressedModelParamSet`, `LoadCompressedModelParamSet`.

## Weight exchange with naming schemes (`ExportSafeTensors` / `ImportSafeTensors`)

`SafeTensorLoader` above is the raw tensor-file layer: it moves tensors, knows
nothing about models, and leaves name matching to you. The **model-level
boundary** lives on `Persistence` (namespace `Shorokoo`, the same class as the
[.skpt save/load](skpt-checkpoints.md) entry points):

```csharp
using Shorokoo;        // Persistence
using Shorokoo.Core;   // naming schemes

// Export: concrete model → standard .safetensors (canonical names by default).
Persistence.ExportSafeTensors(model, "weights.safetensors");
Persistence.ExportSafeTensors(model, "weights.safetensors", scheme);   // PyTorch/timm names

// Import: concrete architecture + .safetensors → concrete model, strictly checked.
ComputationGraph m1 = Persistence.ImportSafeTensors(arch, "weights.safetensors");
ComputationGraph m2 = Persistence.ImportSafeTensors(arch, "foreign.safetensors", scheme);

// One-call native landing: foreign safetensors → .skpt checkpoint (+ the bound model).
ComputationGraph m3 = Persistence.ImportSafeTensorsToCheckpoint(
    arch, "foreign.safetensors", "model.skpt", scheme);
```

- `ExportSafeTensors` requires a **concrete model** (`GraphKind.ConcreteModel`)
  and writes every weight parameter to a single plain safetensors file (the RNG
  identity parameter is model definition, not a weight, and is not exported).
  The write is atomic; the output loads in any safetensors implementation.
- `ImportSafeTensors` requires a **concrete architecture** (the unbound stage
  weights bind into) and returns the bound concrete model. The binding is the
  standard `ToConcreteModel(weights, scheme)` path — what import adds is
  **strictness**. Where `ToConcreteModel` silently drops names that do not
  resolve, import fails loudly, naming the offending tensor, on:
  - a source tensor that maps to no parameter (a training-checkpoint file is
    recognized by its marker and redirected to `TrainingRig.LoadCheckpoint`);
  - a required parameter with no source tensor (or one the scheme fails to name);
  - two source tensors mapping to one parameter (an ambiguous scheme);
  - a dtype or shape mismatch after mapping.

  All validation runs before any binding, so a failed import never yields a
  partially bound model. Truncated/corrupt files are refused by the loader's
  declared-vs-actual size checks, naming the file. A safetensors `__metadata__`
  block is ignored (it is metadata, not a tensor).
- `ImportSafeTensorsToCheckpoint` performs the same import and lands the result
  as a native `.skpt` via the standard
  `Persistence.From(model).WithModel().WithWeights().Save(...)` writer (atomic;
  nothing is written when the import fails).

### Naming

With **no scheme**, tensors carry the parameters' **canonical Shorokoo ids**
(e.g. `TrainableParam#0.FCLayer#0.InitSimple#0`) — export and import are exact
mirrors, so `ExportSafeTensors(model, path)` → `ImportSafeTensors(arch, path)`
reproduces the model bit-identically.

With a **scheme**, the mapping is applied at the boundary in both directions:

```csharp
SimplePatternScheme[] patterns =
[
    new SimplePatternScheme("TrainableParam#0.FCLayer#0.InitSimple#0", "fc.weight"),
    new SimplePatternScheme("TrainableParam#0.FCLayer#0.InitSimple#1", "fc.bias"),
];
var scheme = new SimplePatternNamingScheme(
    patterns, arch.GetShorokooIdNamingScheme(), ModuleParamSetNamingScheme.PyTorchFrameworkId);

Persistence.ExportSafeTensors(model, "torch.safetensors", scheme);  // writes fc.weight / fc.bias
var bound = Persistence.ImportSafeTensors(arch, "torch.safetensors", scheme);
```

Both DSLs work for **import**: the
[pattern DSL](param-naming-pattern-dsl.md) (`SimplePatternNamingScheme`) and the
[ModelId format DSL](param-naming-format-dsl.md) (`ModelIdNamingScheme`).
**Export with a scheme** needs the canonical-id → name direction, which only the
pattern DSL provides (its patterns are written against canonical id strings —
`ModuleParamSetNamingScheme.ToName(string)`); a plain `ModelIdNamingScheme` maps
ModelIds, which a bound model no longer carries, and is refused with
`NotSupportedException`. Export also refuses — naming the parameters — a scheme
that leaves any weight unnamed or maps two weights to one tensor name, so
weights are never silently dropped or overwritten.

## ONNX model exchange (`ExportOnnx` / `ImportOnnx`)

The ONNX mirror of the safetensors boundary: `ExportOnnx` writes a concrete model
to a standard, externally-loadable `.onnx`, and `ImportOnnx` turns a foreign
vanilla `.onnx` back into a native, runnable `ComputationGraph` (and, in one call,
a native `.skpt`).

```csharp
// Export: concrete model → standard vanilla .onnx (loads in any ONNX runtime).
Persistence.ExportOnnx(model, "model.onnx");

// Import: foreign vanilla .onnx → native runnable ComputationGraph.
ComputationGraph g = Persistence.ImportOnnx("foreign.onnx");
ComputationGraph gRenamed = Persistence.ImportOnnx("foreign.onnx", scheme);

// One-call native landing: foreign .onnx → .skpt checkpoint (+ the imported model).
ComputationGraph landed = Persistence.ImportOnnxToCheckpoint("foreign.onnx", "model.skpt");
```

- `ExportOnnx` requires a **concrete model** (`GraphKind.ConcreteModel`) and writes
  **vanilla ONNX** — every node a standard op or emitted function call — so the file
  loads in any conforming runtime. A graph carrying Shorokoo-internal ops is refused,
  naming them. The write is atomic. (It is the `Persistence`-facade wrapper over
  [`FastOnnxModelBuilder.BuildOnnxModel`](#export-to-onnx); use that directly when you
  need the `ModelProto`.)
- `ImportOnnx` builds the graph through the existing ONNX reader, so it composes with
  ONNX **external data** (a `.data` side file resolves against the model file's
  directory, exactly as [`OnnxModelImporter`](#import-from-onnx)). At the boundary each
  foreign initializer — which vanilla ONNX carries as a plain constant — is **promoted
  to a canonical Shorokoo parameter** so the model can be named, checkpointed and
  reloaded natively: its identifier becomes `[k]:TrainableParam#0.name#0`, where `name`
  is the ONNX initializer name by default, or the scheme's translation of it when a
  `namingScheme` is given. A Shorokoo-produced `.onnx` already carries canonical
  identifiers, which are kept as-is.
- `ImportOnnxToCheckpoint` performs the same import and lands the result straight in a
  native `.skpt` via the container writer (see [.skpt](skpt-checkpoints.md)); the write
  is atomic, so a failed import leaves any existing checkpoint untouched.

Importing vanilla ONNX is **lossy by design**: the vanilla dialect drops a concrete
model's module structure and hyper defaults, so `ExportOnnx` → `ImportOnnx` reproduces
the model's **inference values**, not its structure (the outputs match on any input).
For a structural round-trip, use the native `.skpt` container (`Persistence.From` /
`Persistence.Load`).

`ImportOnnx` **fails loudly**, naming the op and the file, on a construct the reader
cannot ingest (an op outside the vanilla ONNX dialect Shorokoo reads, or a node in an
unknown domain); a truncated or garbage file fails loudly naming the file. Two
initializers that resolve to one canonical name are refused, naming both, so no weight
can silently overwrite another. (A Shorokoo **internal-dialect** `.onnx` — the payload
inside `.srk` — is not a vanilla model; load it with `Persistence.Load` /
`CompressedFormatUtils`, not `ImportOnnx`.)

The `namingScheme` is the same `ModuleParamSetNamingScheme` surface `ImportSafeTensors`
takes; for `ImportOnnx` it translates each foreign initializer **name string** to the
canonical Shorokoo name used as the parameter identifier, so the
[pattern DSL](param-naming-pattern-dsl.md) (`SimplePatternNamingScheme`, whose patterns
match name strings) is the tool here — a plain
[`ModelIdNamingScheme`](param-naming-format-dsl.md) maps ModelIds, which a freshly
imported ONNX graph does not carry, so it leaves the ONNX names unchanged.

## Identify and summarize a file (`Persistence.Inspect`)

`Persistence.Inspect(path)` (namespace `Shorokoo`) answers "what is this file?"
**without loading it**: it identifies any Shorokoo-produced artifact and
summarizes its contents from headers/prefixes only, so inspecting a multi-GB
file is fast and cheap.

```csharp
ArtifactInspection result = Persistence.Inspect("run.safetensors");
Console.WriteLine(result);          // human-readable multi-line summary
switch (result.Kind)
{
    case ArtifactKind.SrkGraph:           /* result.Srk */                          break;
    case ArtifactKind.SafeTensors:        /* result.SafeTensors */                  break;
    case ArtifactKind.TrainingCheckpoint: /* result.TrainingCheckpoint (+ .SafeTensors) */ break;
    case ArtifactKind.CompressedSafeTensors: /* result.SafeTensors (decompressed header) */ break;
    case ArtifactKind.SkptCheckpoint:     /* result.Skpt */                         break;
    case ArtifactKind.NotRecognized:      /* result.Observations say what was seen */ break;
}
```

Recognized formats and what is reported:

| `Kind` | Recognized by | Reported |
|---|---|---|
| `SrkGraph` | `.srk` v2 magic (or a sniffed legacy v1 layout) | the v2 header — format version, lifecycle stage, compression, payload SHA-256, producer (`result.Srk.Header`, an `SrkHeader`); legacy files report the sniffed layout instead (`result.Srk.LegacyLayout`) |
| `SafeTensors` | 8-byte header-length prefix + valid JSON header | tensor listing (name, dtype, shape, byte size), total payload size, `__metadata__` (`result.SafeTensors`) |
| `TrainingCheckpoint` | the `__shorokoo_checkpoint__` marker tensor in a SafeTensors header | checkpoint format version, global step, and the per-section (`trainable` / `model_state` / `opt_state`) tensor listing (`result.TrainingCheckpoint`); `result.SafeTensors` is populated too |
| `CompressedSafeTensors` | Zstd frame magic whose decompressed content starts with a valid SafeTensors length prefix + JSON header (`.zsafetensor`, written by `CompressedFormatUtils.SaveCompressedSafeTensors`) | the same details as `SafeTensors` (`result.SafeTensors`), read by stream-decompressing only the prefix and header — the tensor payload is never decompressed; sizes describe the decompressed content |
| `SkptCheckpoint` | a zip archive with a root `config.json` manifest declaring format `"skpt"` (see [skpt-checkpoints.md](skpt-checkpoints.md)) | whole-archive metadata (`.skpt` version, created time, producer), the model registry (per model: entry path, format, stage, graph hash), the data registry (per entry: storage format, compression, declared size, recorded sha256 — reported **unverified**), and the mapping-set names (`result.Skpt`) |
| `NotRecognized` | anything else — including a zip without a readable `skpt` manifest | a structured result — **content problems never throw**; a missing file and I/O errors (permissions, disk) do |

- Reads are bounded to headers and prefixes; tensor payload bytes are never
  materialized. The one exception is a checkpoint's 16-byte marker (the format
  version and step live there). For a `.skpt`, only the zip central directory
  and the `config.json` entry are read — the recorded per-entry sha256s are
  reported as written, never checked (a full `Persistence.Load` verifies them).
  Because the payload is untouched, `Inspect` also succeeds on a file whose
  payload is corrupt — it reports the header while a full load would fail the
  SHA-256 check.
- `result.Observations` lists cheap sanity findings visible from the header
  alone, e.g. declared tensor extents pointing past the end of the file
  (truncation), trailing bytes beyond the declared data, an unreadable /
  future-version container header, or — for a `.skpt` — a manifest entry with
  no matching archive entry (and vice versa), a compressed entry where STORED
  is expected, unknown manifest keys, and empty registries. Through the
  compression layer of a `.zsafetensor` only header-internal checks apply — a
  compressed file's size has no fixed relation to the decompressed extents, so
  truncation of the tensor payload is not detectable from the header.
- A `.zsafetensor` that contains the `__shorokoo_checkpoint__` marker (a
  training checkpoint saved compressed) reports `CompressedSafeTensors`, not
  `TrainingCheckpoint`: the marker's version/step payload sits inside the
  compressed tensor data, beyond `Inspect`'s bounded header-only reads. An
  observation notes the marker and suggests decompressing to inspect fully.
- `.onnx` files are out of scope (standard ONNX tooling covers them). A bare
  serialized ONNX model is byte-identical to the legacy v1 `.srk` layout, so
  such a file reports as a legacy `SrkGraph` with an observation noting the
  ambiguity.
- There is no console I/O in the library — `ToString()` on the result (and on
  each listed tensor) formats the summary; printing is up to you.

## Bind loaded weights into a model (for inference)

Loading a file gives you a `ModelParamList`; it does not yet change any model. A
model's `ComputationGraph` starts with weights from its `[TrainableParamInitializer]`s.
To run with loaded weights, bind them into a concrete graph with `ToConcreteModel`
(extension methods in namespace `Shorokoo.Graph`), then execute that graph:

```csharp
using Shorokoo;
using Shorokoo.Graph;          // ToConcreteArchitecture, ToConcreteModel, InitializeTrainableParams
using static Shorokoo.Globals;

ModelParamList weights = SafeTensorLoader.LoadModelParamSet("weights.safetensors");

// Lower the module graph to a concrete architecture first. This inlines sub-modules so the
// trainable parameters are visible at the top level; pass sample inputs as shape hints.
var input = TensorData([1L, 3L, 224L, 224L], myPixelFloatArray);
ComputationGraph arch = MyModel.ComputationGraph.ToConcreteArchitecture(
    MyModel.ComputationGraph.FromOrderedInputs([input]));  // arch.Kind == GraphKind.ConcreteArchitecture

// Bind by parameter name into a concrete (weight-filled) graph:
ComputationGraph concrete = arch.ToConcreteModel(weights);  // concrete.Kind == GraphKind.ConcreteModel

// Run it. Execute takes IData[] inputs (TensorData implements IData) and returns
// NamedModelParam[]; read each output via ToTensorData().AccessMemory().
var outputs = new ComputeContext().Execute(concrete, input);
ReadOnlySpan<float> values = outputs[0].ToTensorData<float32>().AccessMemory();
```

Notes:
- The full lowering pipeline is **`Specialize` → `ToConcreteArchitecture` →
  `ToConcreteModel`**. The optional first step, `Specialize`, bakes a partial set
  of named inputs (typically `[Hyper]`s) into constants and drops them from the
  input list; skip it when the model has no inputs to hardcode. See
  [inference.md](inference.md#the-lowering-pipeline).
- `ToConcreteModel`, `InitializeTrainableParams`, and `GetConcreteModelParamInfos` require a
  graph whose `Kind` is `GraphKind.ConcreteArchitecture` (from `ToConcreteArchitecture`) and check
  it first. Called on a raw `MyModel.ComputationGraph` (kind `Module`), they fail fast naming the
  actual and required kinds — the trainable parameters would still be nested inside sub-functions.
- `ToConcreteModel()` with no argument fills defaults (equivalent to
  `arch.ToConcreteModel(arch.InitializeTrainableParams())`).
- Binding is **by name**. The default uses Shorokoo's naming scheme; weights exported
  from PyTorch/timm usually need name remapping (use the `ToConcreteModel(weights,
  namingScheme)` overload) before they bind. Unmatched names are silently dropped.
  Two DSLs build the remapping scheme: the
  [ModelId format DSL](param-naming-format-dsl.md) (`ModelIdNamingScheme`) and the
  [pattern DSL](param-naming-pattern-dsl.md) (`SimplePatternNamingScheme`).
- Prefer `Persistence.ImportSafeTensors` (above) when the file is supposed to cover the
  model exactly: it runs the same binding but fails loudly on unmatched or mismatched
  tensors instead of silently dropping them.

A `SafeTensor` exposes `.Name`, `.Data` (`TensorData`), `.DataType` (e.g. `"F32"`,
`"I64"`), `.Shape`, and `.Metadata`. Use `SafeTensorLoader.DTypeToSafeTensorDType` to
map a `DType` to a SafeTensor dtype string.

## Notes / known limitations

- Parameter names from external frameworks (PyTorch/timm) may need remapping to match
  this framework's trainable-parameter names before they bind.

## Anti-patterns

- Do not expect a one-call graph-to-file helper; build the `ModelProto` first, then
  save it (`OnnxModelExporter.Save` / `SaveWithExternalData`, or serialize it
  yourself).

# ONNX export/import and weights

Related: [inference.md](inference.md) · [core-types.md](core-types.md) · [skpt-checkpoints.md](skpt-checkpoints.md)

## Facts

- A model is a `ComputationGraph` whose `Kind` is `GraphKind.Module`,
  `ConcreteArchitecture` or `ConcreteModel`. It can be exported to ONNX or saved as
  `.srk`/`.zsrk`.
- `Persistence.ExportOnnx(graph, path)` writes a concrete model to `.onnx` in one call.
  Use `FastOnnxModelBuilder.BuildOnnxModel` + `OnnxModelExporter` when you need the
  `ModelProto` in between.
- Models over protobuf's 2 GB limit use ONNX **external data** (`externalData:` on
  export; side files load transparently on import). A concrete model holds a weight of any
  size; ONNX export with external data and `ExportSafeTensors` write one past 2 GiB from where
  the model holds it, 8 MiB at a time.
- `.srk`/`.zsrk` and `.zsafetensor` files have no size limit of their own: they are
  saved and loaded as streams, never held whole, and a `.srk` holds a weight of any size.
- Pretrained weights load from `.safetensors` (and compressed `.zsafetensor`).
- Every save API is **atomic** (staged beside the target, committed by rename), so an
  interrupted write never damages the existing file: the `Persistence.*` facade,
  `checkpoint.Save` (`TrainingCheckpoint.Save`, used by
  `Persistence.SaveTrainingCheckpoint`), `OnnxModelExporter`,
  `SafeTensorLoader.SaveSafeTensors` and `CompressedFormatUtils`. The target's directory
  must already exist. A save tolerates a file another process holds for a moment, as an
  antivirus scanner or the search indexer on Windows often does just after a write: a
  commit rename that meets a sharing or lock violation is retried for up to about 0.3 s
  before the save fails with that error. Any other failure fails it at once.

## Export to ONNX

```csharp
using System.IO;
using Shorokoo.Core.Factory;      // FastOnnxModelBuilder
using Shorokoo.Core.Factory.IR;   // ModelProto

ModelProto model = FastOnnxModelBuilder.BuildOnnxModel(graph);  // graph: ComputationGraph, kind ConcreteModel
using var stream = File.Create("model.onnx");
ProtoBuf.Serializer.Serialize(stream, model);   // ProtoBuf = protobuf-net (a transitive dependency)
```

Serializing the proto yourself, as above, is not atomic. `OnnxModelExporter.Save(model,
path)` (namespace `Shorokoo.Onnx`) writes it atomically and gives a clear error when
tensor data exceeds protobuf's 2 GB limit. A crash can leave a `.tmp-` sibling, which
the next successful save of that target removes.
[`Persistence.ExportOnnx`](#onnx-model-exchange-exportonnx--importonnx) wraps both steps.

### Large models: external data

External data puts initializer bytes in a side file; each externalized `TensorProto`
carries `data_location = EXTERNAL` and `location`/`offset`/`length`:

```csharp
using Shorokoo.Onnx;   // OnnxExternalDataOptions, OnnxModelExporter

// Initializers of at least SizeThreshold bytes go to "model.onnx.data"; omit
// externalData (the default) for the self-contained form.
Persistence.ExportOnnx(graph, "model.onnx", externalData: new OnnxExternalDataOptions());
Persistence.ExportOnnx(graph, "model.onnx",
    externalData: new OnnxExternalDataOptions { SizeThreshold = 1024, Alignment = 4096 });

// From a ModelProto; deterministic for the same proto and file name.
OnnxModelExporter.SaveWithExternalData(model, "model.onnx");
```

A weight of more bytes than one managed array holds (`Array.MaxLength`, just under 2 GiB) has
no `raw_data` in the `ModelProto`: `BuildOnnxModel` carries it beside the proto, and
`SaveWithExternalData` always writes it to the side file, whatever `SizeThreshold`, straight
from where the model holds it, 8 MiB at a time. A self-contained save of such a model is
refused with `XD007`, and serializing the proto yourself (`ProtoBuf.Serializer.Serialize`)
throws `NotSupportedException`, naming the tensor, since the proto alone would lose it. A
[`.srk`](#saveload-shorokoo-graph-format-srk--zsrk) holds it inline.

- **Concrete models only**; anything else is refused with `XD008`, naming the actual
  and required kind.
- Tensors are written in initializer order, aligned to `Alignment` bytes (default
  4096).
- With nothing at or above the threshold, no side file is written (a stale one is
  removed) and the output equals `Save`'s.
- The side file's name is recorded in each `location`, so output is deterministic per
  target name.
- The pair is standard ONNX; stock onnxruntime loads it. The passed `ModelProto` is not
  modified.
- The side file is committed first, then the `.onnx`; any in-process failure rolls both
  back, and only a hard crash between the two renames can leave them mismatched.

`BuildOnnxModel(ComputationGraph graph, bool prepForOnnx = false)` requires a
`GraphKind.ConcreteModel` graph (else `FW045`, naming both kinds). It does not modify the
graph. The model and each of its functions are stamped at ONNX opset 21, the one opset
Shorokoo reads and writes; nothing raises the stamp. No post-21 operator is emitted from an
authored graph: each either throws `NotImplementedException` or is lowered to opset-21
primitives. A `NodeBuilder`-built post-21 operator is refused with `FW060`, naming it, and
`NodeBuilder` refuses to build a node carrying a post-21 attribute. See [limitations.md](limitations.md#onnx-opset-21-only) and
[operator-support.md](operator-support.md) per operator.

Each exported input stores its **representative shape** (the dims it was concretized
at; see [inference.md](inference.md#the-lowering-pipeline)) in `ValueInfoProto` metadata
key `shrk_repr_input`, so the model re-imports as the same concrete graph. An absent
`OptionalTensor` records `-1`. The `ValueInfoProto` dims themselves stay symbolic.

### The vanilla dialect is a guarantee

`BuildOnnxModel` writes only **vanilla ONNX** (standard ops and `FunctionProto`s in the
same file), loadable by any ONNX runtime. A module-stage graph, still carrying internal
ops (`ShrkCreateModule`, `ShrkModelInvoke`, …), throws at export naming them; persist
it as `.srk`/`.zsrk` instead. Nested initializer calls are inlined and unreachable
functions are dropped. `.srk` keeps each body as authored, including its calls.

### Graph input/output names and shapes

- Inputs and outputs are named from the model's signature, deduplicated as `x`, `x_2`, …;
  unnamed slots become `input_{i}` / `output_{i}`. A renamed output's own name is kept in
  metadata and restored by `ImportOnnx`.
- Each carries its dtype and rank (so `onnx.checker` passes): the declared rank
  (`Scalar<T>`, `Vector<T>`, …), otherwise the rank observed at the concretization
  samples, even for an output whose rank varies with its inputs.
- Dims are **symbolic** (`{name}_dim{i}`), so any size of that rank is accepted; rank-0
  values are true scalars.

### Parameters in the exported graph

Parameters (trainable weights and state such as BatchNorm running stats) are
`graph.initializer` `TensorProto`s, never `Constant` nodes, with metadata props
`IsTrainable` (`"true"`/`"false"`) and `IdentifierTemplate` (the parameter's name).
Tensor names are internal (`N{k}_T{s}`), so match by `IdentifierTemplate`. Re-import
restores names and trainability, so the model stays trainable and re-bindable with
`ToConcreteModel`.

## Import from ONNX

```csharp
using Shorokoo.Onnx;   // OnnxModelImporter

ComputationGraph g1 = OnnxModelImporter.FromOnnxModel("model.onnx");
ComputationGraph g2 = OnnxModelImporter.FromOnnxModel(byteArray);
ComputationGraph g3 = OnnxModelImporter.FromOnnxModel(stream);
```

Models must be ONNX opset 21 models. One whose default-domain (`ai.onnx`) opset import is
missing or names another opset is refused with `FW060`, naming the opset found and the one
required, and so is an opset-21 model with a node that is an operator, or carries an
attribute, ONNX introduced after opset 21. Import converts nothing: convert such a model to
opset 21 first, for example with `onnx.version_converter.convert_version(model, 21)` — see
[limitations.md](limitations.md#onnx-opset-21-only).

An operator Shorokoo does not support fails the import with `NotSupportedException`, naming
the operator and its node. That includes every node outside the standard domain (`""` /
`ai.onnx`) that does not call one of the model's own functions — a `com.microsoft::Attention`,
say, or a `com.microsoft::Relu`: an operator in another domain is not the standard operator
of the same name.

Shorokoo-written models carry a `shrk_graph_kind` metadata prop that sets the imported
`Kind`; foreign models are classified by op-scanning. A tag impossible for the content
fails the import.

Each input records a **representative shape**: the stored one for a Shorokoo export;
otherwise the declared shape with every symbolic, unset or negative dimension as `1`
(`[N, 3, 224, 224]` → `[1, 3, 224, 224]`). A sequence input records its elements'
shape. An input with no declared shape fails with **`FW058`**; give it one (or override
any) by ONNX input name (for a sequence, its element shape):

```csharp
var shapes = new Dictionary<string, long[]> { ["input"] = [1, 3, 224, 224] };
ComputationGraph g = OnnxModelImporter.FromOnnxModel("model.onnx", shapes);
// also: FromOnnxModel(byteArray, shapes, externalDataDirectory), FromOnnxModel(stream, shapes, ...)
```

A given shape must match the declared rank and fixed dims; a mismatch or an unknown key
throws `ArgumentException`.

Outputs record shapes by the same rules. An output with no declared shape is evaluated
at the input shapes (running the model once at zeros if an op, e.g. some string ops,
requires it); one still unshaped fails with **`FW058`**.

A call to a model function with the wrong number of inputs is refused on import.

External data loads transparently from a **file path** (`location` resolves against the
model's directory, honoring `offset`/`length`). From bytes or a stream, pass the
directory:

```csharp
ComputationGraph g = OnnxModelImporter.FromOnnxModel(
    byteArray, externalDataDirectory: "/path/to/model/dir");
```

A missing side file, a `location` escaping the model's directory, an out-of-range
`offset`/`length`, a `length` contradicting shape/dtype, or a missing
`externalDataDirectory` throws `ModelException`, naming the tensor and file.

Imported from a file path, a model's weights are held once in host memory, each read straight
into its own tensor: the file is scanned rather than parsed whole, and a tensor of at least
1 KiB whose bytes lie flat in the file (`raw_data`, or `float_data` / `double_data` of a
float / double tensor) is read from where it lies, as external data in the model's own file is.
A weight stored as varints (`int32_data`, `int64_data`, `uint64_data`) or strings is decoded
as it is parsed. From bytes or a stream, the protobuf is parsed whole. A sparse initializer
is refused, naming the file: the importer does not read one, so store it dense. A model is read
to a depth of 100 messages below it, protobuf's own default recursion limit, and one nested
deeper is refused however it is read.

## Save/load Shorokoo graph format (`.srk` / `.zsrk`)

```csharp
using Shorokoo.Core.Utils;   // CompressedFormatUtils, SrkFileFormat

string path = CompressedFormatUtils.SaveFastGraphToFile("model.zsrk", graph);     // compressed
ComputationGraph g = CompressedFormatUtils.LoadFastGraphFromFile("model.zsrk");
byte[] bytes = CompressedFormatUtils.SaveFastGraphToBinary(graph, compressed: true);
```

`.zsrk` is Zstandard-compressed, `.srk` uncompressed. The extension follows the
`compressed` flag, but **content, never the extension, decides how a file parses**.

`SaveFastGraphToFile` is atomic and, alone among savers, creates the target directory.
For a concrete model, a [`.skpt`](skpt-checkpoints.md) is the richer container; a
module-stage graph can only be saved as `.srk`.

A `.srk` file has no size limit of its own. `SaveFastGraphToFile` streams the container
to the file, each weight written from where it lies, and `LoadFastGraphFromFile` reads
the file as a stream — once to check `payloadSha256`, once to parse the payload — so
neither ever holds the file whole. Each weight is read into storage of its own as the
payload is parsed; one of more bytes than one managed array holds (`Array.MaxLength`, just
under 2 GiB) is read 8 MiB at a time into host memory of the backend
`ComputeContext.Default` runs on, where the loaded graph holds it. `SaveFastGraphToBinary`
and `LoadFastGraphFromBinary` hold the container in one array, so `SaveFastGraphToBinary`
refuses a container of more than `Array.MaxLength` bytes with `NotSupportedException`
naming `SaveFastGraphToFile`.

### The `.srk` container

```
magic "SRK\x01" | u16 headerLen (little-endian) | JSON header | payload
```

The payload is the graph as an ONNX `ModelProto` (internal dialect allowed), with **at
most one** compression layer. Header (unknown fields are ignored):

```jsonc
{
  "srkVersion": 1,
  "stage": "module" | "concrete-architecture" | "concrete-model",
  "compression": "none" | "zstd",
  "payloadSha256": "…",   // lowercase hex SHA-256 of the payload bytes as stored
  "producer": { "shorokoo": "…", "irVersion": 10, "opsets": { "": 21, "…": 1 } }
}
```

- `stage` is the graph's `GraphKind` (see
  [inference.md](inference.md#the-lowering-pipeline)); the loader restores `Kind` from it
  (op-scanning foreign data). Enforce it with `requiredStage`:

  ```csharp
  // Throws a clear stage-mismatch error if model.zsrk holds a module-stage graph:
  var g = CompressedFormatUtils.LoadFastGraphFromFile(
      "model.zsrk", requiredStage: GraphKind.ConcreteModel);
  ```

- `payloadSha256` makes corruption and truncation fail loudly, naming the file.
- `SrkFileFormat.TryReadHeaderFromFile(path)` returns the `SrkHeader` without loading the
  graph, or `null` for non-`.srk` data. See also
  [`Persistence.Inspect`](#identify-and-summarize-a-file-persistenceinspect).
- `producer` is informational.

A file without the magic fails to load with a clear error. `.srk` accepts any graph,
keeps internal tensor names, and is readable only by Shorokoo
(`LoadFastGraphFromFile` / `OnnxModelImporter`); use `BuildOnnxModel` for anything
leaving Shorokoo.

## Load pretrained weights (SafeTensors)

```csharp
ModelParamList weights = SafeTensorLoader.LoadModelParamSet("weights.safetensors");
Dictionary<string, TensorData> byName = SafeTensorLoader.LoadTensorDictionary("weights.safetensors");
TensorData single = SafeTensorLoader.LoadSingleTensor("bias.safetensors");
List<SafeTensor> all = SafeTensorLoader.LoadSafeTensors("weights.safetensors");
```

A truncated file is refused before any tensor is read, with an error naming the file and
declared vs actual byte counts. `TrainingRig.LoadCheckpoint` uses the same loader.

Save:

```csharp
SafeTensorLoader.SaveSafeTensors("out.safetensors", listOfSafeTensors);
```

For a concrete model's weights prefer
[`Persistence.ExportSafeTensors`](#weight-exchange-with-naming-schemes-exportsafetensors--importsafetensors),
which also matches names.

Compressed `.zsafetensor` variants are in `CompressedFormatUtils`:
`SaveCompressedSafeTensors`, `LoadCompressedSafeTensors`,
`SaveCompressedModelParamSet`, `LoadCompressedModelParamSet`. The save writes a standard
SafeTensors file as one Zstandard frame declaring its decompressed size, compressed
straight into the target with each tensor written from its own storage, and the load
decodes the file as it reads it, so a file of any size is written and read without
either form of it being held whole.

## Weight exchange with naming schemes (`ExportSafeTensors` / `ImportSafeTensors`)

`SafeTensorLoader` only moves tensors. The model-level API is on `Persistence`
(namespace `Shorokoo`; see also [.skpt save/load](skpt-checkpoints.md)):

```csharp
using Shorokoo;        // Persistence
using Shorokoo.Core;   // naming schemes

// Export: concrete model → standard .safetensors (canonical names by default).
Persistence.ExportSafeTensors(model, "weights.safetensors");
Persistence.ExportSafeTensors(model, "weights.safetensors", scheme);   // PyTorch/timm names

// Import: concrete architecture + .safetensors → concrete model, strictly checked.
ComputationGraph m1 = Persistence.ImportSafeTensors(arch, "weights.safetensors");
ComputationGraph m2 = Persistence.ImportSafeTensors(arch, "foreign.safetensors", scheme);

// Foreign safetensors → .skpt checkpoint (+ the bound model).
ComputationGraph m3 = Persistence.ImportSafeTensorsToCheckpoint(
    arch, "foreign.safetensors", "model.skpt", scheme);
```

- `ExportSafeTensors` takes a **concrete model** and writes every weight (not the RNG
  identity parameter) to one plain safetensors file. Each is written from where the model
  holds it, one past 2 GiB 8 MiB at a time.
- `ImportSafeTensors` takes a **concrete architecture** and binds like
  `ToConcreteModel(weights, scheme)`, but fails loudly, naming the tensor, on:
  - a tensor mapping to no parameter (a training-checkpoint file is detected and
    redirected to `TrainingRig.LoadCheckpoint`);
  - a parameter with no tensor, or one the scheme cannot name;
  - two tensors mapping to one parameter;
  - a dtype or shape mismatch.

  Validation precedes binding, so no partial model results. `__metadata__` is ignored.
- `ImportSafeTensorsToCheckpoint` does the same, then saves a `.skpt`; nothing is
  written if the import fails.

### Naming

With **no scheme**, tensors use **canonical Shorokoo ids** (e.g.
`TrainableParam#0.fc#0.weight#0`; see
[Parameter names](defining-models.md#parameter-names)), and export → import is
bit-identical.

A **scheme** maps names in both directions:

```csharp
SimplePatternScheme[] patterns =
[
    new SimplePatternScheme("TrainableParam#0.fc#0.weight#0", "fc.weight"),
    new SimplePatternScheme("TrainableParam#0.fc#0.bias#0", "fc.bias"),
];
var scheme = new SimplePatternNamingScheme(
    patterns, arch.GetShorokooIdNamingScheme(), ModuleParamSetNamingScheme.PyTorchFrameworkId);

Persistence.ExportSafeTensors(model, "torch.safetensors", scheme);  // writes fc.weight / fc.bias
var bound = Persistence.ImportSafeTensors(arch, "torch.safetensors", scheme);
```

Import accepts both the [pattern DSL](param-naming-pattern-dsl.md)
(`SimplePatternNamingScheme`) and the [ModelId format DSL](param-naming-format-dsl.md)
(`ModelIdNamingScheme`). Export needs the pattern DSL
(`ModuleParamSetNamingScheme.ToName(string)`); a `ModelIdNamingScheme` throws
`NotSupportedException`. Export also refuses, naming the parameters, a scheme that
leaves a weight unnamed or maps two weights to one name.

## ONNX model exchange (`ExportOnnx` / `ImportOnnx`)

```csharp
// Export: concrete model → standard vanilla .onnx (loads in any ONNX runtime).
Persistence.ExportOnnx(model, "model.onnx");

// Past protobuf's 2 GB ceiling: large initializers to a "model.onnx.data" side file.
Persistence.ExportOnnx(model, "model.onnx", externalData: new OnnxExternalDataOptions());

// Import: foreign vanilla .onnx → native runnable ComputationGraph.
ComputationGraph g = Persistence.ImportOnnx("foreign.onnx");
ComputationGraph gRenamed = Persistence.ImportOnnx("foreign.onnx", scheme);

// An input the file declares no shape for is given one (keyed by ONNX input name).
var shapes = new Dictionary<string, long[]> { ["input"] = [1, 3, 224, 224] };
ComputationGraph gShaped = Persistence.ImportOnnx("foreign.onnx", shapes);
ComputationGraph gBoth = Persistence.ImportOnnx("foreign.onnx", scheme, shapes);

// Foreign .onnx → .skpt checkpoint (+ the imported model).
ComputationGraph landed = Persistence.ImportOnnxToCheckpoint("foreign.onnx", "model.skpt");
ComputationGraph landedShaped = Persistence.ImportOnnxToCheckpoint("foreign.onnx", "model.skpt", shapes);
```

- `ExportOnnx` takes a **concrete model** and writes [vanilla ONNX](#the-vanilla-dialect-is-a-guarantee);
  internal ops are refused, naming them. A self-contained model over 2 GB is refused with
  `XD007`; pass `externalData` for the [external-data pair](#large-models-external-data).
  It wraps [`FastOnnxModelBuilder.BuildOnnxModel`](#export-to-onnx) plus
  `OnnxModelExporter`.
- `ImportOnnx` reads like [`OnnxModelImporter`](#import-from-onnx) (external data and
  shape rules included). Each foreign initializer becomes a canonical parameter
  `[k]:TrainableParam#0.name#0`, where `name` is the initializer name or the scheme's
  translation; a Shorokoo-produced `.onnx` keeps its identifiers. Every entry point has
  an `inputShapes` overload: `ImportOnnx(path, inputShapes)`,
  `ImportOnnx(path, namingScheme, inputShapes)`, and the same two for
  `ImportOnnxToCheckpoint`. To compile it on a device with its weights read straight into
  device memory, use `ComputeContext.ImportCompiledOnnx` (see
  [Loading a saved model onto the device](inference.md#loading-a-saved-model-onto-the-device)).
- `ImportOnnxToCheckpoint` saves the result as a `.skpt` (see
  [.skpt](skpt-checkpoints.md)); a failed import leaves any existing checkpoint untouched.

Vanilla ONNX drops module structure and hyper defaults, so export → import reproduces
**outputs**, not structure. For a structural round-trip use `.skpt`
(`Persistence.From` / `Persistence.Load`). One operator reloads in a different form: a
`TensorScatter` node, which opset 21 does not define, is saved — and so reloaded — as its
opset-21 decomposition, which computes the same values (see
[ONNX opset 21 only](limitations.md#onnx-opset-21-only)).

`ImportOnnx` fails loudly, naming op and file, on an op outside the vanilla dialect or
an unknown domain, and naming the file on a truncated or garbage file. Two initializers
resolving to one name are refused, naming both. An internal-dialect `.onnx` (a `.srk`
payload) loads with `Persistence.Load` / `CompressedFormatUtils`, not `ImportOnnx`.

For `ImportOnnx` the `namingScheme` maps initializer **name strings**, so use the
[pattern DSL](param-naming-pattern-dsl.md) (`SimplePatternNamingScheme`); a
[`ModelIdNamingScheme`](param-naming-format-dsl.md) leaves the names unchanged.

## Identify and summarize a file (`Persistence.Inspect`)

`Persistence.Inspect(path)` (namespace `Shorokoo`) identifies a Shorokoo artifact from
headers alone, so multi-GB files inspect fast. A directory is inspected as a `.skpt`
[directory form](skpt-checkpoints.md#the-directory-form); `result.FileSizeBytes` is then
the total of its files.

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

| `Kind` | Recognized by | Reported |
|---|---|---|
| `SrkGraph` | `.srk` magic (a file with the magic but no readable header that parses as SafeTensors is `SafeTensors`) | the header: version, stage, compression, payload SHA-256, producer (`result.Srk.Header`, an `SrkHeader`) |
| `SafeTensors` | 8-byte length prefix + valid JSON header | tensors (name, dtype, shape, bytes), payload size, `__metadata__` (`result.SafeTensors`) |
| `TrainingCheckpoint` | the `__shorokoo_checkpoint__` marker tensor | format version, run counters (step, epoch, batch index), tensors per section (`trainable` / `model_state` / `opt_state`, plus `history` for a [training history](training.md#the-training-history)) (`result.TrainingCheckpoint`); `result.SafeTensors` too |
| `CompressedSafeTensors` | Zstd frame decompressing to a SafeTensors prefix + header (`.zsafetensor`, from `CompressedFormatUtils.SaveCompressedSafeTensors`) | as `SafeTensors`, decompressing only the header; sizes are decompressed |
| `SkptCheckpoint` | a zip or **directory** with a root `config.json` declaring `"skpt"` (see [skpt-checkpoints.md](skpt-checkpoints.md)) | version, created time, producer, model registry (entry, format, stage, hash), data registry (format, compression, size, sha256 **unverified**), mapping-set names (`result.Skpt`) |
| `NotRecognized` | anything else, including a zip without a readable `skpt` manifest, or a directory whose `config.json` is missing, too large, or of another format | a structured result: **content problems never throw**; a missing file and I/O errors do |

- No tensor payload is read, except a checkpoint's 16-byte marker (`int64[2]`: version,
  step) and the optional 8-byte epoch and batch-index scalars; the loss is not read.
  For a `.skpt`, only the zip central directory (or directory listing), `config.json`
  and `data/user-data.json` are read. Sha256s are never verified, so a corrupt payload
  still inspects.
- `result.Observations` lists header-level findings: extents past end of file,
  trailing bytes, an unreadable or future-version header, and for `.skpt` manifest /
  archive mismatches, unexpected compression, unknown keys, empty registries, and (in
  the directory form) entry paths escaping the root. For a `.zsafetensor`, payload
  truncation is not detectable.
- A compressed training checkpoint reports `CompressedSafeTensors`; an observation notes
  the marker and suggests decompressing.
- `.onnx` files are `NotRecognized`; use ONNX tooling.
- `ToString()` on the result and each tensor formats the summary; the library prints
  nothing.

## Bind loaded weights into a model (for inference)

Loaded weights (`ModelParamList`) change no model until you bind them into a concrete
graph with `ToConcreteModel` (namespace `Shorokoo.Graph`):

```csharp
using Shorokoo;
using Shorokoo.Graph;          // ToConcreteArchitecture, ToConcreteModel, InitializeTrainableParams
using static Shorokoo.Globals;

ModelParamList weights = SafeTensorLoader.LoadModelParamSet("weights.safetensors");

// Lower to a concrete architecture; sample inputs are shape hints, one per input in order
// (or a ModelParamList of NamedModelParams to bind by name).
var input = TensorData([1L, 3L, 224L, 224L], myPixelFloatArray);
ComputationGraph arch = MyModel.ComputationGraph.ToConcreteArchitecture([input]);
// arch.Kind == GraphKind.ConcreteArchitecture

// Bind by parameter name into a concrete (weight-filled) graph:
ComputationGraph concrete = arch.ToConcreteModel(weights);  // concrete.Kind == GraphKind.ConcreteModel

// Execute takes IData[] and returns NamedModelParam[].
var outputs = new ComputeContext().Execute(concrete, input);
float[] values = outputs[0].ToTensorData().CopyMemory<float>();
```

Notes:
- The pipeline is **`Specialize` → `ToConcreteArchitecture` → `ToConcreteModel`**;
  `Specialize` (optional) bakes named inputs into constants. See
  [inference.md](inference.md#the-lowering-pipeline).
- `ToConcreteModel`, `InitializeTrainableParams` and `GetConcreteModelParamInfos`
  require `GraphKind.ConcreteArchitecture`; on a `Module` graph they fail fast naming
  both kinds.
- `ToConcreteModel()` with no argument equals
  `arch.ToConcreteModel(arch.InitializeTrainableParams())`.
- Binding is **by name**, and unmatched names are silently dropped. PyTorch/timm
  weights usually need a scheme (`ToConcreteModel(weights, namingScheme)`), built with
  the [ModelId format DSL](param-naming-format-dsl.md) (`ModelIdNamingScheme`) or the
  [pattern DSL](param-naming-pattern-dsl.md) (`SimplePatternNamingScheme`).
- `Persistence.ImportSafeTensors` binds the same way but fails on unmatched or
  mismatched tensors.

A `SafeTensor` exposes `.Name`, `.Data` (`TensorData`), `.DataType` (e.g. `"F32"`,
`"I64"`), `.Shape` and `.Metadata`; `SafeTensorLoader.DTypeToSafeTensorDType` maps a
`DType` to its dtype string.

## Notes / known limitations

- PyTorch/timm parameter names may need remapping before they bind (see above).

## Anti-patterns

- Handing `OnnxModelExporter.Save` a graph: it takes a `ModelProto`. Use
  `Persistence.ExportOnnx(graph, path)`.
- Wrapping a save in your own stage-and-rename: every save API already does it.
  Serializing a `ModelProto` yourself with `ProtoBuf.Serializer` is the exception; it
  truncates the target first.
- Saving into a directory that does not exist yet (the staged temp file lives there).
  Create it first.

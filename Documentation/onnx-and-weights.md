# ONNX export/import and weights

Related: [inference.md](inference.md) · [core-types.md](core-types.md)

## Facts

- A model is a `FastComputationGraph` (from `MyModule.ComputationGraph` or built
  directly). It can be converted to an ONNX `ModelProto`, or saved in Shorokoo's own
  `.srk`/`.zsrk` format.
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

ModelProto model = FastOnnxModelBuilder.BuildOnnxModel(graph);  // graph: FastComputationGraph
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

- Tensor data is written in initializer order, each tensor aligned to `Alignment`
  bytes (default 4096) for mmap-friendly access.
- Self-contained (all-inline) export remains the default and is unchanged; with no
  initializer at or above the threshold, `SaveWithExternalData` writes no side file
  (removing a stale one from a previous save of the same path) and its output is
  identical to `Save`.
- The exported pair is standard ONNX — stock onnxruntime loads it directly.
- The passed `ModelProto` is left unmodified.

`BuildOnnxModel(FastComputationGraph fastGraph, OpSetVersion opset = OPS_21,
IR_VERSION irVersion = IR_10, bool prepForOnnx = false)` clones the graph (no
mutation), lowers it for ONNX, and emits nodes, subgraphs, and functions.
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

FastComputationGraph g1 = OnnxModelImporter.FromOnnxModelToFastGraph("model.onnx");
FastComputationGraph g2 = OnnxModelImporter.FromOnnxModelToFastGraph(byteArray);
FastComputationGraph g3 = OnnxModelImporter.FromOnnxModelToFastGraph(stream);
```

Models using ONNX external data (the standard layout for large third-party models)
load transparently from a **file path** — `location` keys resolve against the model
file's directory, honoring `offset`/`length` slicing. When importing from bytes or a
stream, pass the directory the side files live in:

```csharp
FastComputationGraph g = OnnxModelImporter.FromOnnxModelToFastGraph(
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
FastComputationGraph g = CompressedFormatUtils.LoadFastGraphFromFile("model.zsrk");
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

- `stage` records where the graph sits in the lowering pipeline (see
  [inference.md](inference.md#the-lowering-pipeline)), so loaders can refuse a
  mismatched file up front instead of failing at run time with
  `No Op registered for ShrkCreateModule`. Pass the optional `requiredStage`
  argument to enforce it:

  ```csharp
  // Throws a clear stage-mismatch error if model.zsrk holds a module-stage graph:
  var g = CompressedFormatUtils.LoadFastGraphFromFile(
      "model.zsrk", requiredStage: SrkGraphStage.ConcreteModel);
  ```

- `payloadSha256` makes corruption and truncation fail loudly, with an error naming
  the file and the failure.
- `SrkFileFormat.TryReadHeaderFromFile(path)` reads the header (`SrkHeader`) without
  loading the graph — useful to identify a file cheaply; it returns `null` for
  pre-container legacy files.
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

Save:

```csharp
SafeTensorLoader.SaveSafeTensors("out.safetensors", listOfSafeTensors);
```

Compressed (`.zsafetensor`) variants live in `CompressedFormatUtils`:
`SaveCompressedSafeTensors`, `LoadCompressedSafeTensors`,
`SaveCompressedModelParamSet`, `LoadCompressedModelParamSet`.

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
FastComputationGraph arch = MyModel.ComputationGraph.ToConcreteArchitecture(
    MyModel.ComputationGraph.FromOrderedInputs([input]));

// Bind by parameter name into a concrete (weight-filled) graph:
FastComputationGraph concrete = arch.ToConcreteModel(weights);

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
  concrete architecture graph from `ToConcreteArchitecture`. Called on a raw `MyModel.ComputationGraph`
  whose sub-modules are not inlined, they throw, because the trainable parameters are still nested
  inside sub-functions.
- `ToConcreteModel()` with no argument fills defaults (equivalent to
  `arch.ToConcreteModel(arch.InitializeTrainableParams())`).
- Binding is **by name**. The default uses Shorokoo's naming scheme; weights exported
  from PyTorch/timm usually need name remapping (use the `ToConcreteModel(weights,
  namingScheme)` overload) before they bind. Unmatched names are silently dropped.
  Two DSLs build the remapping scheme: the
  [ModelId format DSL](param-naming-format-dsl.md) (`ModelIdNamingScheme`) and the
  [pattern DSL](param-naming-pattern-dsl.md) (`SimplePatternNamingScheme`).

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

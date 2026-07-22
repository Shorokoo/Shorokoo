using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Utils;
using Shorokoo.Graph;
using Shorokoo.Onnx;

namespace Shorokoo
{
    /// <summary>
    /// Save/load entry points for .skpt checkpoints — Shorokoo's native single-file
    /// checkpoint container. A .skpt is a standard zip archive whose entries are all
    /// STORED (uncompressed), wired together by a single config.json manifest; this
    /// version saves and loads an inference checkpoint of a concrete model (model
    /// definition + weights) with bit-identical execution on round-trip.
    ///
    /// <code>
    /// Checkpoint.From(concreteModel)
    ///     .WithModel()
    ///     .WithWeights()
    ///     .Save("model.skpt");
    ///
    /// var loaded = Checkpoint.Load("model.skpt");   // concrete model, weights bound
    /// </code>
    ///
    /// The write is atomic (staged to a temp file and committed by rename), so a crash
    /// mid-save never corrupts an existing checkpoint. See <see cref="SkptFileFormat"/>
    /// for the container layout and manifest schema.
    /// </summary>
    public static class Checkpoint
    {
        /// <summary>
        /// Starts a checkpoint of <paramref name="concreteModel"/>. The graph must be a
        /// <see cref="GraphKind.ConcreteModel"/> — fully lowered, every parameter
        /// materialized. Select the contents with <see cref="CheckpointBuilder.WithModel"/> and
        /// <see cref="CheckpointBuilder.WithWeights"/>, then commit with
        /// <see cref="CheckpointBuilder.Save"/>.
        /// </summary>
        public static CheckpointBuilder From(ComputationGraph concreteModel)
        {
            if (concreteModel is null) throw new ArgumentNullException(nameof(concreteModel));
            if (concreteModel.Kind != GraphKind.ConcreteModel)
                throw new InvalidOperationException(SrkFileFormat.KindMismatchMessage(
                    "Checkpoint.From", "a 'concrete-model' graph", concreteModel.Kind,
                    "Lower the graph with ToConcreteArchitecture(inputHints, ...).ToConcreteModel(...) first."));
            return new CheckpointBuilder(concreteModel);
        }

        /// <summary>
        /// Loads a .skpt checkpoint saved by <see cref="CheckpointBuilder.Save"/>: reads the
        /// manifest, verifies every referenced entry's SHA-256, loads the model definition and
        /// binds the checkpoint's weights back onto its parameters. Returns the runnable
        /// <see cref="GraphKind.ConcreteModel"/>. A manifest referencing a missing entry,
        /// an entry failing its SHA-256 check, or a weight that does not match its parameter
        /// fails loudly naming the entry; unknown manifest keys are ignored (the format's
        /// keys are add-only across minor revisions).
        /// </summary>
        public static ComputationGraph Load(string filePath)
        {
            if (string.IsNullOrWhiteSpace(filePath))
                throw new ArgumentException("Checkpoint path cannot be null or empty.", nameof(filePath));

            var fileBytes = File.ReadAllBytes(filePath);
            using var fileStream = new MemoryStream(fileBytes, writable: false);
            using var archive = OpenArchive(fileStream, filePath);

            var configEntry = archive.GetEntry(SkptFileFormat.ConfigEntryName)
                ?? throw new InvalidDataException(
                    $"'{filePath}' is not a .skpt checkpoint — the archive contains no " +
                    $"'{SkptFileFormat.ConfigEntryName}' manifest.");
            var manifest = SkptFileFormat.ParseManifest(ReadEntryBytes(configEntry, filePath), filePath);
            ValidateManifestIdentity(manifest, filePath);

            var (modelKey, modelEntry) = SingleModel(manifest, filePath);
            var graph = LoadModelDefinition(archive, modelKey, modelEntry, filePath);
            BindWeights(archive, manifest, modelKey, graph, filePath);

            return new ComputationGraph(graph, GraphKind.ConcreteModel);
        }

        private static ZipArchive OpenArchive(Stream stream, string filePath)
        {
            try
            {
                return new ZipArchive(stream, ZipArchiveMode.Read, leaveOpen: true);
            }
            catch (InvalidDataException e)
            {
                throw new InvalidDataException(
                    $"'{filePath}' is not a .skpt checkpoint — it does not open as a zip archive. ({e.Message})", e);
            }
        }

        private static void ValidateManifestIdentity(SkptManifest manifest, string filePath)
        {
            if (manifest.Format != SkptFileFormat.FormatName)
                throw new InvalidDataException(
                    $"'{filePath}': '{SkptFileFormat.ConfigEntryName}' does not declare format " +
                    $"'{SkptFileFormat.FormatName}' (found '{manifest.Format}') — not a .skpt checkpoint.");

            // skptVersion is required and >= 1; 0 means the field was absent from the JSON,
            // which is a malformed manifest, not version skew. (A present-but-wrong-typed
            // value would already have failed in ParseManifest.)
            if (manifest.SkptVersion == 0)
                throw new InvalidDataException(
                    $"'{filePath}': invalid .skpt manifest — required field 'skptVersion' is missing or zero.");
            if (manifest.SkptVersion != SkptFileFormat.CurrentVersion)
                throw new InvalidDataException(
                    $"'{filePath}': .skpt version {manifest.SkptVersion} is not supported by this Shorokoo " +
                    $"build (supported: {SkptFileFormat.CurrentVersion}). The file was written by " +
                    (manifest.SkptVersion > SkptFileFormat.CurrentVersion
                        ? "a newer framework version."
                        : "an older, unsupported framework version."));
        }

        private static (string Key, SkptModelEntry Entry) SingleModel(SkptManifest manifest, string filePath)
        {
            if (manifest.Models is null || manifest.Models.Count == 0)
                throw new InvalidDataException(
                    $"'{filePath}': the .skpt manifest declares no models — nothing to load.");
            if (manifest.Models.Count > 1)
                throw new InvalidDataException(
                    $"'{filePath}': the .skpt manifest declares {manifest.Models.Count} models; this " +
                    "Shorokoo build loads single-model checkpoints only. The file was likely written " +
                    "by a newer framework version.");
            var kv = manifest.Models.First();
            return (kv.Key, kv.Value);
        }

        private static InternalComputationGraph LoadModelDefinition(
            ZipArchive archive, string modelKey, SkptModelEntry modelEntry, string filePath)
        {
            if (string.IsNullOrEmpty(modelEntry.Entry))
                throw new InvalidDataException(
                    $"'{filePath}': the manifest's model '{modelKey}' names no archive entry.");
            if (modelEntry.Format != SkptFileFormat.ModelFormatSrk2)
                throw new InvalidDataException(
                    $"'{filePath}': model '{modelKey}' uses unsupported serialization format " +
                    $"'{modelEntry.Format}' (supported: '{SkptFileFormat.ModelFormatSrk2}'). " +
                    "The file was likely written by a newer framework version.");
            if (SrkFileFormat.TryParseStageName(modelEntry.Stage) != GraphKind.ConcreteModel)
                throw new InvalidDataException(
                    $"'{filePath}': model '{modelKey}' records stage '{modelEntry.Stage}', but this " +
                    "Shorokoo build loads 'concrete-model' checkpoints only.");

            var modelBytes = ReadEntry(archive, modelEntry.Entry, $"model '{modelKey}'", filePath);
            VerifySha256(modelBytes, modelEntry.Sha256, modelEntry.Entry, filePath);

            var (graph, _) = CompressedFormatUtils.LoadFastGraphCore(
                modelBytes, origin: $"{filePath}!{modelEntry.Entry}", requiredStage: GraphKind.ConcreteModel);
            return graph;
        }

        /// <summary>
        /// Resolves the model's "default" tensor mapping set and injects each mapped tensor
        /// into its parameter node (the graph carries dtype/shape-true zero placeholders where
        /// weights were stripped at save time). Every parameter must be mapped and every
        /// mapping entry must land on a parameter — a mismatch means the checkpoint does not
        /// belong to this model definition, and fails loudly naming the parameter.
        /// </summary>
        private static void BindWeights(
            ZipArchive archive, SkptManifest manifest, string modelKey,
            InternalComputationGraph graph, string filePath)
        {
            if (manifest.TensorMappings is null
                || !manifest.TensorMappings.TryGetValue(modelKey, out var mappingSets)
                || mappingSets is null)
                throw new InvalidDataException(
                    $"'{filePath}': the .skpt manifest has no tensor mappings for model '{modelKey}'.");
            if (!mappingSets.TryGetValue(SkptFileFormat.DefaultMappingSetName, out var mappingSet)
                || mappingSet is null)
                throw new InvalidDataException(
                    $"'{filePath}': model '{modelKey}' has no '{SkptFileFormat.DefaultMappingSetName}' " +
                    "tensor mapping set — this Shorokoo build binds the default set.");
            var tensorRefs = mappingSet.Tensors ?? new Dictionary<string, SkptTensorRef>();

            var tensorsByDataKey = new Dictionary<string, Dictionary<string, TensorData>>(StringComparer.Ordinal);
            var unboundRefs = new HashSet<string>(tensorRefs.Keys, StringComparer.Ordinal);

            foreach (var node in graph.Nodes)
            {
                if (node.OpCode != InternalOpCodes.MODEL_PARAM_DATA) continue;
                if (node.IdentifierTemplate ==
                        Core.Nodes.Processors.Fast.FastWireRngKeyDerivation.RngSeedIdentifierTemplate)
                    continue;

                var paramId = node.IdentifierTemplate;
                if (string.IsNullOrEmpty(paramId))
                    throw new InvalidDataException(
                        $"'{filePath}': the model definition carries a parameter with no identifier; " +
                        "cannot resolve its weights in the checkpoint.");
                if (!tensorRefs.TryGetValue(paramId, out var tensorRef) || tensorRef is null)
                    throw new InvalidDataException(
                        $"'{filePath}': the '{SkptFileFormat.DefaultMappingSetName}' mapping of model " +
                        $"'{modelKey}' has no entry for parameter '{paramId}'. Does the checkpoint " +
                        "belong to this model?");
                unboundRefs.Remove(paramId);

                var tensors = ResolveDataEntry(archive, manifest, tensorRef, paramId, tensorsByDataKey, filePath);
                if (string.IsNullOrEmpty(tensorRef.Tensor) || !tensors.TryGetValue(tensorRef.Tensor, out var loaded))
                    throw new InvalidDataException(
                        $"'{filePath}': parameter '{paramId}' maps to tensor '{tensorRef.Tensor}' in data " +
                        $"entry '{tensorRef.Data}', but that entry contains no such tensor.");

                var placeholder = node.GetTensorData()
                    ?? throw new InvalidDataException(
                        $"'{filePath}': parameter '{paramId}' in the model definition carries no tensor placeholder.");
                if (placeholder.DType.ToIVarType() != loaded.DType.ToIVarType()
                    || !placeholder.Shape.Dims.SequenceEqual(loaded.Shape.Dims))
                    throw new InvalidDataException(
                        $"'{filePath}': tensor '{tensorRef.Tensor}' (dtype {loaded.DType}, shape " +
                        $"[{string.Join(",", loaded.Shape.Dims)}]) does not match parameter '{paramId}' " +
                        $"(dtype {placeholder.DType}, shape [{string.Join(",", placeholder.Shape.Dims)}]).");

                node.Attributes = node.Attributes.SetAttributes(
                    (OnnxOpAttributeNames.ShrkAttrTensorData, (object?)loaded));
            }

            if (unboundRefs.Count > 0)
                throw new InvalidDataException(
                    $"'{filePath}': the '{SkptFileFormat.DefaultMappingSetName}' mapping of model " +
                    $"'{modelKey}' maps parameter '{unboundRefs.First()}' " +
                    (unboundRefs.Count > 1 ? $"(and {unboundRefs.Count - 1} more) " : string.Empty) +
                    "which the model definition does not declare. Does the checkpoint belong to this model?");
        }

        /// <summary>
        /// Returns the parsed tensors of the data entry a tensor reference points at, reading
        /// and SHA-256-verifying each data entry at most once per load.
        /// </summary>
        private static Dictionary<string, TensorData> ResolveDataEntry(
            ZipArchive archive, SkptManifest manifest, SkptTensorRef tensorRef, string paramId,
            Dictionary<string, Dictionary<string, TensorData>> tensorsByDataKey, string filePath)
        {
            var dataKey = tensorRef.Data;
            if (string.IsNullOrEmpty(dataKey))
                throw new InvalidDataException(
                    $"'{filePath}': the mapping for parameter '{paramId}' names no data entry.");
            if (tensorsByDataKey.TryGetValue(dataKey, out var cached))
                return cached;

            if (manifest.Data is null || !manifest.Data.TryGetValue(dataKey, out var dataEntry) || dataEntry is null)
                throw new InvalidDataException(
                    $"'{filePath}': the mapping for parameter '{paramId}' references data entry " +
                    $"'{dataKey}', which the manifest's data registry does not declare.");
            if (string.IsNullOrEmpty(dataEntry.Entry))
                throw new InvalidDataException(
                    $"'{filePath}': the manifest's data entry '{dataKey}' names no archive entry.");
            if (dataEntry.Format != SkptFileFormat.DataFormatSafeTensors)
                throw new InvalidDataException(
                    $"'{filePath}': data entry '{dataKey}' uses unsupported storage format " +
                    $"'{dataEntry.Format}' (supported: '{SkptFileFormat.DataFormatSafeTensors}'). " +
                    "The file was likely written by a newer framework version.");
            if (dataEntry.Compression is not null && dataEntry.Compression != SkptFileFormat.CompressionNone)
                throw new InvalidDataException(
                    $"'{filePath}': data entry '{dataKey}' declares unsupported compression " +
                    $"'{dataEntry.Compression}' (supported: '{SkptFileFormat.CompressionNone}'). " +
                    "The file was likely written by a newer framework version.");

            var dataBytes = ReadEntry(archive, dataEntry.Entry, $"data entry '{dataKey}'", filePath);
            VerifySha256(dataBytes, dataEntry.Sha256, dataEntry.Entry, filePath);

            var tensors = SafeTensorLoader.ParseSafeTensorBytes(dataBytes)
                .ToDictionary(t => t.Name, t => t.Data, StringComparer.Ordinal);
            tensorsByDataKey[dataKey] = tensors;
            return tensors;
        }

        private static byte[] ReadEntry(ZipArchive archive, string entryPath, string role, string filePath)
        {
            var entry = archive.GetEntry(entryPath)
                ?? throw new InvalidDataException(
                    $"'{filePath}': the manifest references entry '{entryPath}' (for {role}), " +
                    "but the archive contains no such entry.");
            return ReadEntryBytes(entry, filePath);
        }

        private static byte[] ReadEntryBytes(ZipArchiveEntry entry, string filePath)
        {
            // entry.Length is the uncompressed size declared in the archive's directory; a
            // corrupt or hostile file can declare up to ~4 GiB. Reject oversize entries with
            // the loader's usual named error rather than letting the (int) cast below throw a
            // context-free OverflowException. This .skpt version reads in-memory entries only.
            if (entry.Length > int.MaxValue)
                throw new InvalidDataException(
                    $"'{filePath}': entry '{entry.FullName}' declares an uncompressed size of {entry.Length} " +
                    "bytes, which exceeds the maximum this .skpt version reads.");
            using var entryStream = entry.Open();
            using var buffer = new MemoryStream((int)entry.Length);
            entryStream.CopyTo(buffer);
            return buffer.ToArray();
        }

        private static void VerifySha256(byte[] bytes, string? expected, string entryPath, string filePath)
        {
            if (string.IsNullOrEmpty(expected))
                throw new InvalidDataException(
                    $"'{filePath}': the manifest records no sha256 for entry '{entryPath}' — " +
                    "required by .skpt version 1.");
            var actual = SkptFileFormat.Sha256Hex(bytes);
            if (!string.Equals(actual, expected, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException(
                    $"'{filePath}': entry '{entryPath}' fails its SHA-256 check — the checkpoint is " +
                    $"corrupt or was modified (manifest records {expected}, entry hashes to {actual}).");
        }
    }

    /// <summary>
    /// Builder for a .skpt checkpoint, started by <see cref="Checkpoint.From"/>. Select what
    /// the checkpoint contains — this Shorokoo version writes exactly one shape, the inference
    /// checkpoint <see cref="WithModel"/> + <see cref="WithWeights"/> — then commit it to disk
    /// with <see cref="Save"/>.
    /// </summary>
    public sealed class CheckpointBuilder
    {
        private readonly ComputationGraph _model;
        private bool _withModel;
        private bool _withWeights;

        internal CheckpointBuilder(ComputationGraph model) => _model = model;

        /// <summary>Includes the model definition (serialized as .srk v2, weights stripped) in the checkpoint.</summary>
        public CheckpointBuilder WithModel()
        {
            _withModel = true;
            return this;
        }

        /// <summary>Includes the model's weights (stored as a safetensors data entry) in the checkpoint.</summary>
        public CheckpointBuilder WithWeights()
        {
            _withWeights = true;
            return this;
        }

        /// <summary>
        /// Saves the checkpoint as a single .skpt file. The write is atomic (staged to a temp
        /// file beside <paramref name="filePath"/> and committed by rename), so a crash
        /// mid-save never corrupts an existing checkpoint; the target's directory must already
        /// exist. The model definition is stored with its weight tensors stripped to zero
        /// placeholders — the weights live once, in the checkpoint's data tree — and the RNG
        /// identity parameter (part of the model's definition, not a weight) stays embedded.
        /// </summary>
        public void Save(string filePath)
        {
            if (string.IsNullOrWhiteSpace(filePath))
                throw new ArgumentException("Checkpoint path cannot be null or empty.", nameof(filePath));
            if (!_withModel || !_withWeights)
                throw new InvalidOperationException(
                    "Checkpoint.Save: this Shorokoo version writes exactly one checkpoint shape — " +
                    "model definition + weights. Call .WithModel().WithWeights() before Save.");

            var source = _model.ToInternal();
            var weightNodes = CollectWeightNodes(source);

            var tensors = new List<SafeTensor>(weightNodes.Count);
            var tensorRefs = new Dictionary<string, SkptTensorRef>(StringComparer.Ordinal);
            foreach (var node in weightNodes)
            {
                var data = node.GetTensorData()!;
                tensors.Add(new SafeTensor(node.IdentifierTemplate!, data,
                    SafeTensorLoader.DTypeToSafeTensorDType(data.DType), data.Shape.Dims));
                tensorRefs[node.IdentifierTemplate!] = new SkptTensorRef
                {
                    Data = SkptFileFormat.DefaultDataKey,
                    Tensor = node.IdentifierTemplate,
                };
            }

            byte[] weightsBytes;
            using (var buffer = new MemoryStream())
            {
                SafeTensorLoader.SaveSafeTensorsToStream(buffer, tensors);
                weightsBytes = buffer.ToArray();
            }

            var modelBytes = CompressedFormatUtils.SaveFastGraphToBinary(
                StripWeights(source, weightNodes), GraphKind.ConcreteModel, compressed: true);

            var manifest = new SkptManifest
            {
                Format = SkptFileFormat.FormatName,
                SkptVersion = SkptFileFormat.CurrentVersion,
                CreatedUtc = DateTime.UtcNow.ToString("yyyy-MM-dd'T'HH:mm:ss'Z'"),
                Producer = new SkptProducerInfo { Shorokoo = ShorokooVersion.VersionString },
                Models = new Dictionary<string, SkptModelEntry>
                {
                    [SkptFileFormat.DefaultModelKey] = new SkptModelEntry
                    {
                        Entry = SkptFileFormat.ModelEntryPath,
                        Format = SkptFileFormat.ModelFormatSrk2,
                        Stage = SrkFileFormat.StageName(GraphKind.ConcreteModel),
                        Sha256 = SkptFileFormat.Sha256Hex(modelBytes),
                    },
                },
                TensorMappings = new Dictionary<string, Dictionary<string, SkptMappingSet>>
                {
                    [SkptFileFormat.DefaultModelKey] = new Dictionary<string, SkptMappingSet>
                    {
                        [SkptFileFormat.DefaultMappingSetName] = new SkptMappingSet { Tensors = tensorRefs },
                    },
                },
                Data = new Dictionary<string, SkptDataEntry>
                {
                    [SkptFileFormat.DefaultDataKey] = new SkptDataEntry
                    {
                        Entry = SkptFileFormat.WeightsEntryPath,
                        Format = SkptFileFormat.DataFormatSafeTensors,
                        Compression = SkptFileFormat.CompressionNone,
                        Sha256 = SkptFileFormat.Sha256Hex(weightsBytes),
                    },
                },
            };

            SkptFileFormat.ZipEntrySpec[] entries =
            [
                new(SkptFileFormat.ConfigEntryName, SkptFileFormat.SerializeManifest(manifest), Align: false),
                new(SkptFileFormat.ModelEntryPath, modelBytes, Align: false),
                new(SkptFileFormat.WeightsEntryPath, weightsBytes, Align: true),
            ];
            AtomicFileWriter.WriteFile(filePath,
                stream => SkptFileFormat.WriteStoredZip(stream, entries, DateTime.UtcNow));
        }

        /// <summary>
        /// The model's weight parameters: every MODEL_PARAM_DATA node except the RNG identity
        /// parameter (which is model definition, not a weight, and stays embedded). Each must
        /// carry a unique, non-empty identifier — it names the tensor in the checkpoint — and
        /// its tensor data.
        /// </summary>
        private static List<FastNode> CollectWeightNodes(InternalComputationGraph graph)
        {
            var nodes = new List<FastNode>();
            var seenIds = new HashSet<string>(StringComparer.Ordinal);
            foreach (var node in graph.Nodes)
            {
                if (node.OpCode != InternalOpCodes.MODEL_PARAM_DATA) continue;
                if (node.IdentifierTemplate ==
                        Core.Nodes.Processors.Fast.FastWireRngKeyDerivation.RngSeedIdentifierTemplate)
                    continue;

                if (string.IsNullOrEmpty(node.IdentifierTemplate))
                    throw new InvalidOperationException(
                        "Checkpoint.Save: a model parameter carries no identifier, so its tensor " +
                        "cannot be named in the checkpoint.");
                if (!seenIds.Add(node.IdentifierTemplate))
                    throw new InvalidOperationException(
                        $"Checkpoint.Save: two model parameters share the identifier " +
                        $"'{node.IdentifierTemplate}'; parameter identifiers must be unique to map tensors.");
                if (node.GetTensorData() is null)
                    throw new InvalidOperationException(
                        $"Checkpoint.Save: parameter '{node.IdentifierTemplate}' carries no tensor data; " +
                        "a concrete model must have every parameter materialized.");

                nodes.Add(node);
            }
            return nodes;
        }

        /// <summary>
        /// Returns a copy of the graph with each weight parameter's tensor replaced by a
        /// dtype/shape-true zero placeholder (the same clone-and-swap
        /// <see cref="InternalComputationGraph.WithUpdatedStates"/> uses). The placeholders
        /// keep the serialized definition a valid concrete model — and Zstd inside the .srk
        /// payload collapses them to almost nothing — while the real bytes live once, in the
        /// checkpoint's data tree.
        /// </summary>
        private static InternalComputationGraph StripWeights(
            InternalComputationGraph graph, List<FastNode> weightNodes)
        {
            var indexByKey = new Dictionary<FastNodeKey, int>();
            for (int i = 0; i < graph.Nodes.Count; i++)
                indexByKey[graph.Nodes[i].Key] = i;

            var stripped = graph.Clone();
            foreach (var node in weightNodes)
            {
                var data = node.GetTensorData()!;
                var clonedNode = stripped.Nodes[indexByKey[node.Key]];
                clonedNode.Attributes = clonedNode.Attributes.SetAttributes(
                    (OnnxOpAttributeNames.ShrkAttrTensorData,
                     (object?)Globals.TensorDataWithDefaultVals(data.DType, data.Shape.Dims)));
            }
            return stripped;
        }
    }
}

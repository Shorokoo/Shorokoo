using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Nodes;
using Shorokoo.Core;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Utils;
using Shorokoo.Graph;
using Shorokoo.Onnx;

namespace Shorokoo
{
    /// <summary>
    /// The single entry point for getting Shorokoo models and weights on and off disk:
    /// native <c>.skpt</c> checkpoints, weight interchange (safetensors), training-run
    /// checkpoints, and read-only artifact inspection.
    ///
    /// <para>Its primary job is the native <c>.skpt</c> container — Shorokoo's single-file
    /// checkpoint. A .skpt is a standard zip archive whose entries are all
    /// STORED (uncompressed), wired together by a single config.json manifest; this
    /// version saves and loads an inference checkpoint of a concrete model (model
    /// definition + weights) with bit-identical execution on round-trip.</para> Data-tree
    /// entries may opt into Zstd compression inside their STORED bytes via
    /// <see cref="CheckpointBuilder.WithZstdCompressedData"/>; the manifest records it
    /// per entry and <see cref="Load(string)"/> decompresses transparently.
    ///
    /// <code>
    /// Persistence.From(concreteModel)
    ///     .WithModel()
    ///     .WithWeights()
    ///     .Save("model.skpt");
    ///
    /// var loaded = Persistence.Load("model.skpt");   // concrete model, weights bound
    /// </code>
    ///
    /// The write is atomic (staged to a temp file and committed by rename), so a crash
    /// mid-save never corrupts an existing checkpoint. See <see cref="SkptFileFormat"/>
    /// for the container layout and manifest schema. (The read-only <see cref="Inspect"/>
    /// facility lives in the other half of this partial class, in ArtifactInspection.cs.)
    ///
    /// The safetensors boundary lives here too: <c>ExportSafeTensors</c> writes a model's
    /// weights to a standard .safetensors file and <c>ImportSafeTensors</c> binds a foreign
    /// .safetensors file onto an architecture (see <c>Persistence.SafeTensors.cs</c>).
    /// Training-run state routes through <see cref="SaveTrainingCheckpoint(TrainingCheckpoint, string)"/> /
    /// <see cref="LoadTrainingCheckpoint"/>.
    /// </summary>
    // Partial: Persistence.Inspect (read-only artifact identification) lives in
    // ArtifactInspection.cs and the safetensors weight-exchange boundary in
    // Persistence.SafeTensors.cs — one Persistence facade, several persistence concerns.
    public static partial class Persistence
    {
        /// <summary>
        /// Starts a checkpoint of <paramref name="concreteModel"/>. The graph must be a
        /// <see cref="GraphKind.ConcreteModel"/> — fully lowered, every parameter
        /// materialized. Select the contents with <see cref="CheckpointBuilder.WithModel"/> and
        /// <see cref="CheckpointBuilder.WithWeights()"/>, then commit with
        /// <see cref="CheckpointBuilder.Save"/>.
        /// </summary>
        public static CheckpointBuilder From(ComputationGraph concreteModel)
        {
            if (concreteModel is null) throw new ArgumentNullException(nameof(concreteModel));
            if (concreteModel.Kind != GraphKind.ConcreteModel)
                throw new InvalidOperationException(SrkFileFormat.KindMismatchMessage(
                    "Persistence.From", "a 'concrete-model' graph", concreteModel.Kind,
                    "Lower the graph with ToConcreteArchitecture(inputHints, ...).ToConcreteModel(...) first."));
            return new CheckpointBuilder(concreteModel);
        }

        /// <summary>
        /// Loads a .skpt checkpoint saved by <see cref="CheckpointBuilder.Save"/>, binding the
        /// <c>default</c> tensor mapping set. See <see cref="Load(string, string)"/> to select
        /// another set (e.g. <c>ema</c>).
        /// </summary>
        public static ComputationGraph Load(string filePath)
            => Load(filePath, SkptFileFormat.DefaultMappingSetName);

        /// <summary>
        /// Loads a .skpt checkpoint saved by <see cref="CheckpointBuilder.Save"/>, binding the
        /// named tensor mapping <paramref name="set"/>: reads the manifest, verifies every
        /// referenced entry's SHA-256 (always over the stored bytes, so a compressed entry is
        /// integrity-checked without decompressing), removes each data entry's manifest-declared
        /// compression layer, loads the model definition and binds the selected set's weights
        /// back onto its parameters. A checkpoint may carry several sets over shared data (e.g.
        /// <c>default</c> raw weights and an <c>ema</c> smoothed set); pass the set's name to
        /// pick one. Returns the runnable <see cref="GraphKind.ConcreteModel"/>. An unknown set
        /// name fails loudly, listing the sets the file declares. A manifest referencing a
        /// missing entry, an entry failing its SHA-256 check, a manifest/stored compression
        /// mismatch, or a weight that does not match its parameter likewise fails loudly naming
        /// the entry; unknown manifest keys are ignored (the format's keys are add-only across
        /// minor revisions).
        /// </summary>
        public static ComputationGraph Load(string filePath, string set)
        {
            if (string.IsNullOrWhiteSpace(filePath))
                throw new ArgumentException("Checkpoint path cannot be null or empty.", nameof(filePath));
            if (string.IsNullOrWhiteSpace(set))
                throw new ArgumentException("Mapping set name cannot be null or empty.", nameof(set));

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
            BindWeights(archive, manifest, modelKey, set, graph, filePath);

            return new ComputationGraph(graph, GraphKind.ConcreteModel);
        }

        // ---- Training checkpoints ----
        // Training-run state (trainable params, model + optimizer state, global step) persists
        // through this same facade in one of two on-disk shapes. The legacy shape is the
        // self-contained flat sectioned-safetensors file TrainingCheckpoint owns; the native
        // shape is a .skpt container (issue #95) that carries the concrete inference model plus
        // the training state split into per-kind data entries, so a training checkpoint gains the
        // container's benefits (inspectable manifest, per-entry Zstd, atomic write, provenance
        // metadata) and shares one format with inference checkpoints. New saves opt into the .skpt
        // shape via SaveTrainingCheckpointToSkpt / ForTrainingCheckpoint (in
        // Persistence.TrainingCheckpoint.cs); LoadTrainingCheckpoint reads either shape.

        /// <summary>
        /// Saves a <see cref="TrainingCheckpoint"/> — trainable parameters, model state, optimizer
        /// state and the global step — in the legacy flat sectioned-safetensors format, so a
        /// training run can resume across process restarts. Delegates to
        /// <see cref="TrainingCheckpoint.Save(string)"/>; the write is atomic (temp file + rename). A
        /// <c>.safetensors</c> extension is conventional. To write the native .skpt container
        /// instead (carrying the inference model and per-kind data entries), use
        /// <see cref="SaveTrainingCheckpointToSkpt"/> / <see cref="ForTrainingCheckpoint"/>.
        /// </summary>
        public static void SaveTrainingCheckpoint(TrainingCheckpoint checkpoint, string filePath)
        {
            if (checkpoint is null) throw new ArgumentNullException(nameof(checkpoint));
            checkpoint.Save(filePath);
        }

        /// <summary>
        /// Saves a <see cref="TrainingCheckpoint"/> into a rotating series — the file is written to
        /// <c>{directory}/{filePrefix}{step}{fileSuffix}</c> — and prunes older members so only the
        /// <paramref name="keepLast"/> most recent survive (the "keep last N training checkpoints"
        /// use case). Rotation orders strictly by the global step encoded in each name, so it is
        /// correct regardless of filesystem timestamp resolution or zero-padding, never touches
        /// files outside the series, and — running only after the atomic commit — never fails the
        /// save; failures surface only through <paramref name="onWarning"/>. Returns the path
        /// written. Delegates to <see cref="TrainingCheckpoint.Save(string, string, string, int, Action{string})"/>.
        /// </summary>
        public static string SaveTrainingCheckpoint(
            TrainingCheckpoint checkpoint,
            string directory,
            string filePrefix,
            string fileSuffix,
            int keepLast,
            Action<string>? onWarning = null)
        {
            if (checkpoint is null) throw new ArgumentNullException(nameof(checkpoint));
            return checkpoint.Save(directory, filePrefix, fileSuffix, keepLast, onWarning);
        }

        /// <summary>
        /// Loads a <see cref="TrainingCheckpoint"/> saved by either <see cref="SaveTrainingCheckpoint(TrainingCheckpoint, string)"/>
        /// (the legacy flat safetensors file) or <see cref="SaveTrainingCheckpointToSkpt"/> (the
        /// native .skpt container) — the shape is detected from the file's bytes. Either way the
        /// checkpoint is reconstructed against the given struct defs (which pin the expected shapes,
        /// so a checkpoint from a different model or optimizer fails loudly). To resume a whole rig,
        /// prefer <see cref="TrainingRig.LoadCheckpoint"/>, which supplies these defs from the rig.
        /// </summary>
        public static TrainingCheckpoint LoadTrainingCheckpoint(
            string filePath,
            TensorStructDef trainableParamDef,
            TensorStructDef modelStateDef,
            TensorStructDef optimizerStateDef)
        {
            if (string.IsNullOrWhiteSpace(filePath))
                throw new ArgumentException("Checkpoint path cannot be null or empty.", nameof(filePath));

            // Sniff the container shape from the leading bytes: a .skpt is a zip (PK\x03\x04),
            // the legacy flat checkpoint a safetensors file (8-byte header-length prefix). The
            // read is bounded — four bytes — and routes to the matching reconstructor.
            byte[] prefix = new byte[4];
            using (var probe = File.OpenRead(filePath))
                probe.ReadAtLeast(prefix, prefix.Length, throwOnEndOfStream: false);

            return LooksLikeZipArchive(prefix)
                ? LoadTrainingCheckpointFromSkpt(filePath, trainableParamDef, modelStateDef, optimizerStateDef)
                : TrainingCheckpoint.Load(filePath, trainableParamDef, modelStateDef, optimizerStateDef);
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
        /// Resolves the model's <paramref name="setName"/> tensor mapping set and injects each
        /// mapped tensor into its parameter node (the graph carries dtype/shape-true placeholders
        /// where weights were stripped at save time). Every parameter must be mapped and every
        /// mapping entry must land on a parameter — a mismatch means the checkpoint does not
        /// belong to this model definition, and fails loudly naming the parameter. An unknown
        /// <paramref name="setName"/> fails loudly, listing the sets the manifest declares.
        /// </summary>
        private static void BindWeights(
            ZipArchive archive, SkptManifest manifest, string modelKey, string setName,
            InternalComputationGraph graph, string filePath)
        {
            if (manifest.TensorMappings is null
                || !manifest.TensorMappings.TryGetValue(modelKey, out var mappingSets)
                || mappingSets is null)
                throw new InvalidDataException(
                    $"'{filePath}': the .skpt manifest has no tensor mappings for model '{modelKey}'.");
            if (!mappingSets.TryGetValue(setName, out var mappingSet) || mappingSet is null)
            {
                var available = mappingSets.Keys.Count == 0
                    ? "<none>"
                    : string.Join(", ", mappingSets.Keys.OrderBy(k => k, StringComparer.Ordinal));
                throw new InvalidDataException(
                    $"'{filePath}': model '{modelKey}' has no '{setName}' tensor mapping set. " +
                    $"Available sets: {available}.");
            }
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
                        $"'{filePath}': the '{setName}' mapping of model " +
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
                    $"'{filePath}': the '{setName}' mapping of model " +
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
            var storedBytes = ReadEntry(archive, dataEntry.Entry, $"data entry '{dataKey}'", filePath);
            // The manifest sha256 covers the entry's bytes as stored in the archive — for a
            // compressed entry, the compressed bytes — so integrity is checked here, before
            // and without decompression (mirroring .srk v2's payloadSha256 semantics).
            VerifySha256(storedBytes, dataEntry.Sha256, dataEntry.Entry, filePath);
            var dataBytes = DecodeDataEntryPayload(storedBytes, dataEntry, dataKey, filePath);

            var tensors = SafeTensorLoader.ParseSafeTensorBytes(dataBytes)
                .ToDictionary(t => t.Name, t => t.Data, StringComparer.Ordinal);
            tensorsByDataKey[dataKey] = tensors;
            return tensors;
        }

        /// <summary>
        /// Removes a data entry's manifest-declared compression layer (none today for
        /// "none", one Zstd layer for "zstd"). The declared compression is cross-checked
        /// against the stored bytes' framing, so a manifest/stored mismatch in either
        /// direction fails loudly naming the entry instead of feeding garbage to the
        /// safetensors parser. The Zstd-frame sniff cannot misfire on a genuine
        /// uncompressed payload: every supported data format is safetensors, whose first
        /// 8 bytes are a little-endian header length, and the Zstd magic in bytes 0–3
        /// would put that length beyond the 2 GiB entry cap enforced on read.
        /// </summary>
        private static byte[] DecodeDataEntryPayload(
            byte[] storedBytes, SkptDataEntry dataEntry, string dataKey, string filePath)
        {
            switch (dataEntry.Compression ?? SkptFileFormat.CompressionNone)
            {
                case SkptFileFormat.CompressionNone:
                    if (SkptFileFormat.LooksLikeZstdFrame(storedBytes))
                        throw new InvalidDataException(
                            $"'{filePath}': data entry '{dataKey}' ('{dataEntry.Entry}') declares compression " +
                            $"'{SkptFileFormat.CompressionNone}' but its stored bytes are a Zstd frame — " +
                            "the manifest and the stored entry disagree; the checkpoint is corrupt or was modified.");
                    return storedBytes;

                case SkptFileFormat.CompressionZstd:
                    if (!SkptFileFormat.LooksLikeZstdFrame(storedBytes))
                        throw new InvalidDataException(
                            $"'{filePath}': data entry '{dataKey}' ('{dataEntry.Entry}') declares compression " +
                            $"'{SkptFileFormat.CompressionZstd}' but its stored bytes are not a Zstd frame — " +
                            "the manifest and the stored entry disagree; the checkpoint is corrupt or was modified.");
                    try
                    {
                        // ZstdSharp allocates from the frame's declared content size, capped at
                        // 2 GiB — the same bound ReadEntryBytes enforces on stored entries — so a
                        // hostile frame cannot demand an unbounded allocation.
                        return CompressedFormatUtils.Decompress(storedBytes);
                    }
                    catch (Exception e)
                    {
                        throw new InvalidDataException(
                            $"'{filePath}': failed to Zstd-decompress data entry '{dataKey}' " +
                            $"('{dataEntry.Entry}') — the checkpoint is corrupt or was modified. ({e.Message})", e);
                    }

                default:
                    throw new InvalidDataException(
                        $"'{filePath}': data entry '{dataKey}' declares unsupported compression " +
                        $"'{dataEntry.Compression}' (supported: '{SkptFileFormat.CompressionNone}', " +
                        $"'{SkptFileFormat.CompressionZstd}'). " +
                        "The file was likely written by a newer framework version.");
            }
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
    /// Builder for a .skpt checkpoint, started by <see cref="Persistence.From"/>. Select what
    /// the checkpoint contains — this Shorokoo version writes exactly one shape, the inference
    /// checkpoint <see cref="WithModel"/> + <see cref="WithWeights()"/> — then commit it to disk
    /// with <see cref="Save"/>.
    /// </summary>
    public sealed class CheckpointBuilder
    {
        private readonly ComputationGraph _model;
        private bool _withModel;
        private bool _withWeights;
        private int? _zstdDataCompressionLevel;
        private Dictionary<string, string>? _userMetadata;
        private JsonObject? _userData;
        private readonly List<(string Name, IReadOnlyDictionary<string, TensorData> Values)> _extraSets = new();

        internal CheckpointBuilder(ComputationGraph model) => _model = model;

        /// <summary>Includes the model definition (serialized as .srk v2, weights stripped) in the checkpoint.</summary>
        public CheckpointBuilder WithModel()
        {
            _withModel = true;
            return this;
        }

        /// <summary>Includes the model's weights (stored as a safetensors data entry) in the
        /// checkpoint, as the <c>default</c> tensor mapping set (bound by the parameterless
        /// <see cref="Persistence.Load(string)"/>).</summary>
        public CheckpointBuilder WithWeights()
        {
            _withWeights = true;
            return this;
        }

        /// <summary>
        /// Attaches an additional named weight set <paramref name="setName"/> (e.g. <c>ema</c>)
        /// over the same model parameters, selectable at load time with
        /// <see cref="Persistence.Load(string, string)"/>. <paramref name="values"/> maps each of
        /// the model's weight-parameter identifiers to that set's tensor, and must cover exactly
        /// the model's weight parameters (the same set the <c>default</c> weights span), each with
        /// a matching dtype and shape. Sets share data: a tensor byte-identical to one already
        /// stored (by the default set or an earlier additional set) is referenced, not stored
        /// again; only a set's genuinely distinct tensors are written, into its own
        /// <c>data/&lt;setName&gt;.safetensors</c> entry. Computing the weights (e.g. an EMA) is
        /// the caller's concern; this only carries and selects the parallel versions.
        /// <paramref name="setName"/> must be a non-empty identifier over
        /// <c>[A-Za-z0-9._-]</c>, distinct from the reserved <c>default</c> set and the
        /// <c>weights</c> data key, and not repeated. Requires <see cref="WithWeights()"/>.
        /// </summary>
        public CheckpointBuilder WithWeights(string setName, IReadOnlyDictionary<string, TensorData> values)
        {
            if (string.IsNullOrWhiteSpace(setName))
                throw new ArgumentException("Weight set name cannot be null or empty.", nameof(setName));
            if (setName == SkptFileFormat.DefaultMappingSetName)
                throw new ArgumentException(
                    $"'{SkptFileFormat.DefaultMappingSetName}' is the reserved name of the model's own " +
                    "weight set (written by WithWeights()); name the additional set something else.",
                    nameof(setName));
            if (setName == SkptFileFormat.DefaultDataKey)
                throw new ArgumentException(
                    $"'{SkptFileFormat.DefaultDataKey}' is reserved for the default weights data entry; " +
                    "name the additional set something else.", nameof(setName));
            if (setName == SkptFileFormat.UserDataDataKey)
                throw new ArgumentException(
                    $"'{SkptFileFormat.UserDataDataKey}' is reserved for the host user-data entry; " +
                    "name the additional set something else.", nameof(setName));
            if (!IsValidSetName(setName))
                throw new ArgumentException(
                    $"Weight set name '{setName}' must be a non-empty identifier over [A-Za-z0-9._-] " +
                    "(it names an archive entry).", nameof(setName));
            if (_extraSets.Any(s => s.Name == setName))
                throw new ArgumentException($"Weight set '{setName}' is already attached.", nameof(setName));
            if (values is null) throw new ArgumentNullException(nameof(values));
            if (values.Count == 0)
                throw new ArgumentException($"Weight set '{setName}' carries no tensors.", nameof(values));

            _extraSets.Add((setName, values));
            return this;
        }

        private static bool IsValidSetName(string name)
        {
            foreach (var c in name)
                if (!(char.IsAsciiLetterOrDigit(c) || c is '.' or '_' or '-'))
                    return false;
            return true;
        }

        /// <summary>
        /// Opt-in: Zstd-compress the checkpoint's data-tree entries (the weights entry, in
        /// this version's single checkpoint shape), recording <c>compression: "zstd"</c> in
        /// each compressed entry's data-registry record. The zip framing stays STORED — the
        /// Zstd layer lives inside the entry's bytes, mirroring how .srk v2 declares
        /// compression in its header. The manifest sha256 covers the stored (compressed)
        /// bytes, so integrity checking never needs decompression. The trade: a compressed
        /// entry is smaller on disk but forfeits memory-mapped/range reads, so it also skips
        /// the 64-byte payload alignment uncompressed data entries carry. The manifest and
        /// model entries (config.json, models/*.srk) are never compressed by this option,
        /// and without it the output is byte-for-byte the uncompressed default.
        /// </summary>
        /// <param name="compressionLevel">Zstandard compression level (1–22, default
        /// <see cref="CompressedFormatUtils.DefaultCompressionLevel"/>).</param>
        public CheckpointBuilder WithZstdCompressedData(
            int compressionLevel = CompressedFormatUtils.DefaultCompressionLevel)
        {
            if (compressionLevel is < 1 or > 22)
                throw new ArgumentOutOfRangeException(nameof(compressionLevel), compressionLevel,
                    "Zstandard compression level must be between 1 and 22.");
            _zstdDataCompressionLevel = compressionLevel;
            return this;
        }

        /// <summary>
        /// Opt-in: attaches user-supplied provenance metadata to the checkpoint, recorded under
        /// the manifest's dedicated <c>userMetadata</c> key. This is descriptive, reproducibility
        /// metadata — "cheap to write at save time, impossible to reconstruct later" — echoed
        /// back by <see cref="Persistence.Inspect"/>; it is trusted only as far as its writer and
        /// never affects load: <see cref="Persistence.Load(string)"/> ignores it entirely, binding a
        /// checkpoint identically with or without it. Nothing here is interpreted or validated
        /// (a git commit is not checked to exist) and nothing is auto-populated from the
        /// environment — the caller supplies every value.
        ///
        /// <para>The four well-known keys (<paramref name="gitCommit"/>,
        /// <paramref name="datasetId"/>, <paramref name="runName"/>, <paramref name="license"/>)
        /// are conveniences that write the corresponding <see cref="SkptFileFormat"/> keys; a
        /// non-null one overrides any same-key entry in <paramref name="metadata"/>. Any other
        /// key/value pairs go in <paramref name="metadata"/>. Calls accumulate, so metadata can
        /// be built up across several invocations. Values are stored verbatim (control characters
        /// included); the human-readable inspection output sanitizes them for display only.</para>
        ///
        /// <para>Supplying no metadata at all — never calling this method, or calling it with
        /// nothing set — leaves the manifest's <c>userMetadata</c> key absent, keeping the
        /// output byte-for-byte identical to a checkpoint saved without provenance.</para>
        /// </summary>
        /// <param name="metadata">Arbitrary provenance key/value pairs. Keys must be non-empty
        /// and values non-null.</param>
        /// <param name="gitCommit">Convenience for the <see cref="SkptFileFormat.MetadataGitCommitKey"/> key.</param>
        /// <param name="datasetId">Convenience for the <see cref="SkptFileFormat.MetadataDatasetIdKey"/> key.</param>
        /// <param name="runName">Convenience for the <see cref="SkptFileFormat.MetadataRunNameKey"/> key.</param>
        /// <param name="license">Convenience for the <see cref="SkptFileFormat.MetadataLicenseKey"/> key.</param>
        public CheckpointBuilder WithMetadata(
            IReadOnlyDictionary<string, string>? metadata = null,
            string? gitCommit = null,
            string? datasetId = null,
            string? runName = null,
            string? license = null)
        {
            if (metadata is not null)
                foreach (var (key, value) in metadata)
                    AddMetadata(key, value);
            if (gitCommit is not null) AddMetadata(SkptFileFormat.MetadataGitCommitKey, gitCommit);
            if (datasetId is not null) AddMetadata(SkptFileFormat.MetadataDatasetIdKey, datasetId);
            if (runName is not null) AddMetadata(SkptFileFormat.MetadataRunNameKey, runName);
            if (license is not null) AddMetadata(SkptFileFormat.MetadataLicenseKey, license);
            return this;
        }

        /// <summary>
        /// Records one metadata entry, allocating the bag lazily so a call that sets nothing
        /// leaves it null (and thus the <c>userMetadata</c> key absent). Keys must be non-empty
        /// and values non-null — the only structural check; values are otherwise uninterpreted.
        /// </summary>
        private void AddMetadata(string key, string value)
        {
            if (string.IsNullOrEmpty(key))
                throw new ArgumentException(
                    "Persistence.WithMetadata: a metadata key must be non-empty.", nameof(key));
            if (value is null)
                throw new ArgumentNullException(nameof(value),
                    $"Persistence.WithMetadata: the value for metadata key '{key}' is null.");
            (_userMetadata ??= new Dictionary<string, string>(StringComparer.Ordinal))[key] = value;
        }

        /// <summary>
        /// Opt-in: attaches a host-owned <b>user-data bag</b> to the checkpoint (issue #101) —
        /// an arbitrary JSON object the host serializes at save and reads back verbatim on load,
        /// stored as the <see cref="SkptFileFormat.UserDataEntryPath"/> data entry and wired into
        /// the manifest's data registry (<c>format: "json"</c>, sha256, so the manifest stays the
        /// verification root). It carries the one part of a run Shorokoo cannot reconstruct — the
        /// data-pipeline state (which corpus, shuffle/augmentation strategy, stream position) —
        /// because Shorokoo does not own the dataloader.
        ///
        /// <para><paramref name="value"/> is serialized with <see cref="System.Text.Json"/>
        /// (using <paramref name="options"/> when supplied) and its serialized <b>root must be a
        /// JSON object</b>: a value that serializes to an array, string, number, boolean, or null
        /// is rejected with an <see cref="ArgumentException"/> (wrap it in an object first). The
        /// values under the root may be any valid JSON — Shorokoo validates well-formedness only,
        /// never the shape or meaning of the values, and never fails a load on a data mismatch
        /// (that check, if wanted, is host code). Read the bag back via
        /// <see cref="SkptArtifactInfo.UserData"/> / <see cref="SkptArtifactInfo.GetUserData{T}"/>.</para>
        ///
        /// <para>Top-level keys beginning with <c>'$'</c> are reserved for Shorokoo and rejected.
        /// The bag never affects a <see cref="Persistence.Load(string)"/>: it is not referenced by
        /// any tensor mapping, so loading binds a checkpoint identically with or without it, and
        /// it is stored uncompressed regardless of <see cref="WithZstdCompressedData"/>. Not
        /// calling this method leaves the entry absent, keeping the output byte-for-byte identical
        /// to a checkpoint saved without user data. The last call wins.</para>
        /// </summary>
        /// <param name="value">The host object to serialize; its serialized root must be a JSON object.</param>
        /// <param name="options">Optional serializer options for <paramref name="value"/>.</param>
        public CheckpointBuilder WithUserData<T>(T value, JsonSerializerOptions? options = null)
        {
            JsonNode? node;
            try
            {
                node = JsonSerializer.SerializeToNode(value, options);
            }
            catch (JsonException e)
            {
                throw new ArgumentException(
                    "Persistence.WithUserData: the value could not be serialized to JSON.",
                    nameof(value), e);
            }
            if (node is not JsonObject obj)
                throw new ArgumentException(
                    "Persistence.WithUserData: the user-data root must be a JSON object, but the " +
                    $"value serialized to {DescribeJsonRoot(node)}. A user-data bag needs a JSON " +
                    "object at its root; wrap a list or scalar in an object (e.g. { \"items\": [ … ] }).",
                    nameof(value));
            // SerializeToNode returns a fresh, unparented node — take it directly after validating.
            ValidateNoReservedUserDataKeys(obj);
            _userData = obj;
            return this;
        }

        /// <summary>
        /// Attaches a host user-data bag from a <see cref="JsonObject"/> DOM directly (issue #101);
        /// see <see cref="WithUserData{T}(T, JsonSerializerOptions?)"/> for the semantics. The
        /// object is deep-cloned, so later mutation of <paramref name="value"/> does not affect the
        /// checkpoint.
        /// </summary>
        public CheckpointBuilder WithUserData(JsonObject value)
        {
            if (value is null) throw new ArgumentNullException(nameof(value));
            // Deep-clone: detaches the node from any parent DOM and snapshots it against later mutation.
            var snapshot = value.DeepClone().AsObject();
            ValidateNoReservedUserDataKeys(snapshot);
            _userData = snapshot;
            return this;
        }

        /// <summary>Names the JSON kind a value serialized to, for the non-object-root error.</summary>
        private static string DescribeJsonRoot(JsonNode? node) => node switch
        {
            null => "JSON null",
            JsonArray => "a JSON array",
            JsonValue => "a JSON scalar (string, number, or boolean)",
            _ => "a non-object JSON value",
        };

        /// <summary>
        /// Rejects top-level user-data keys beginning with <c>'$'</c> — reserved for Shorokoo
        /// (issue #101). Only the root's own keys are checked; nested objects may use any keys,
        /// since the bag's values are never interpreted.
        /// </summary>
        private static void ValidateNoReservedUserDataKeys(JsonObject userData)
        {
            foreach (var (key, _) in userData)
                if (key.Length > 0 && key[0] == SkptFileFormat.ReservedUserDataKeyPrefix)
                    throw new ArgumentException(
                        $"Persistence.WithUserData: top-level user-data key '{key}' begins with " +
                        $"'{SkptFileFormat.ReservedUserDataKeyPrefix}', which is reserved for Shorokoo; rename it.",
                        nameof(userData));
        }

        /// <summary>
        /// Saves the checkpoint as a single .skpt file. The write is atomic (staged to a temp
        /// file beside <paramref name="filePath"/> and committed by rename), so a crash
        /// mid-save never corrupts an existing checkpoint; the target's directory must already
        /// exist. The model definition is stored with its weight tensors stripped to
        /// metadata-only placeholders — the weights live once, in the checkpoint's data tree —
        /// and the RNG identity parameter (part of the model's definition, not a weight)
        /// stays embedded.
        /// </summary>
        public void Save(string filePath)
        {
            if (string.IsNullOrWhiteSpace(filePath))
                throw new ArgumentException("Checkpoint path cannot be null or empty.", nameof(filePath));
            if (!_withModel || !_withWeights)
                throw new InvalidOperationException(
                    "Persistence.Save: this Shorokoo version writes exactly one checkpoint shape — " +
                    "model definition + weights. Call .WithModel().WithWeights() before Save.");

            var source = _model.ToInternal();
            var weightNodes = CollectWeightNodes(source, "Persistence.Save");

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

            // Encodes one safetensors data-entry payload for storage: opt-in Zstd wraps the
            // bytes in a single Zstd layer (the zip framing stays STORED, the manifest records
            // compression "zstd" and a sha256 of the stored/compressed bytes, and the entry
            // skips the 64-byte alignment a compressed entry cannot use); otherwise the bytes
            // are STORED verbatim and aligned. Applied uniformly to every data-tree entry.
            (byte[] Stored, string Compression, bool Align) EncodeDataEntry(byte[] rawBytes)
                => _zstdDataCompressionLevel is int level
                    ? (CompressedFormatUtils.Compress(rawBytes, level), SkptFileFormat.CompressionZstd, false)
                    : (rawBytes, SkptFileFormat.CompressionNone, true);

            var (weightsStoredBytes, weightsCompression, weightsAlign) = EncodeDataEntry(weightsBytes);

            var dataEntries = new Dictionary<string, SkptDataEntry>(StringComparer.Ordinal)
            {
                [SkptFileFormat.DefaultDataKey] = new SkptDataEntry
                {
                    Entry = SkptFileFormat.WeightsEntryPath,
                    Format = SkptFileFormat.DataFormatSafeTensors,
                    Compression = weightsCompression,
                    Sha256 = SkptFileFormat.Sha256Hex(weightsStoredBytes),
                },
            };
            var mappingSets = new Dictionary<string, SkptMappingSet>(StringComparer.Ordinal)
            {
                [SkptFileFormat.DefaultMappingSetName] = new SkptMappingSet { Tensors = tensorRefs },
            };
            // Body entries after the manifest: the model definition, the default weights entry,
            // and one entry per additional set that has genuinely distinct tensors. config.json
            // is prepended once the manifest below is complete, keeping the default-only file
            // byte-identical to the single-set output.
            var bodyEntries = new List<SkptFileFormat.ZipEntrySpec>
            {
                new(SkptFileFormat.ModelEntryPath, modelBytes, Align: false),
                new(SkptFileFormat.WeightsEntryPath, weightsStoredBytes, Align: weightsAlign),
            };

            // Content-addressed dedup index over the data already stored: an additional set's
            // tensor byte-identical (dtype, shape, raw bytes) to one here is referenced, never
            // stored again. Seeded with the default set's tensors (the "weights" entry); each
            // newly stored additional-set tensor is added, so later sets share with earlier ones.
            var storedByContent = new Dictionary<string, (string DataKey, string Tensor)>(StringComparer.Ordinal);
            if (_extraSets.Count > 0)
                foreach (var node in weightNodes)
                    storedByContent.TryAdd(ContentKey(node.GetTensorData()!),
                        (SkptFileFormat.DefaultDataKey, node.IdentifierTemplate!));

            foreach (var (setName, values) in _extraSets)
            {
                var setRefs = new Dictionary<string, SkptTensorRef>(StringComparer.Ordinal);
                var newTensors = new List<SafeTensor>();
                var extraKeys = new HashSet<string>(values.Keys, StringComparer.Ordinal);
                foreach (var node in weightNodes)
                {
                    var paramId = node.IdentifierTemplate!;
                    if (!values.TryGetValue(paramId, out var value) || value is null)
                        throw new InvalidOperationException(
                            $"Persistence.Save: weight set '{setName}' has no entry for parameter " +
                            $"'{paramId}'; an additional set must cover every model weight parameter " +
                            "(the same parameters the default weights span).");
                    extraKeys.Remove(paramId);

                    var modelData = node.GetTensorData()!;
                    if (value.DType.ToIVarType() != modelData.DType.ToIVarType()
                        || !value.Shape.Dims.SequenceEqual(modelData.Shape.Dims))
                        throw new InvalidOperationException(
                            $"Persistence.Save: weight set '{setName}' tensor for parameter '{paramId}' " +
                            $"(dtype {value.DType}, shape [{string.Join(",", value.Shape.Dims)}]) does not " +
                            $"match the model parameter (dtype {modelData.DType}, shape " +
                            $"[{string.Join(",", modelData.Shape.Dims)}]).");

                    var contentKey = ContentKey(value);
                    if (storedByContent.TryGetValue(contentKey, out var existing))
                    {
                        setRefs[paramId] = new SkptTensorRef { Data = existing.DataKey, Tensor = existing.Tensor };
                    }
                    else
                    {
                        newTensors.Add(new SafeTensor(paramId, value,
                            SafeTensorLoader.DTypeToSafeTensorDType(value.DType), value.Shape.Dims));
                        setRefs[paramId] = new SkptTensorRef { Data = setName, Tensor = paramId };
                        storedByContent[contentKey] = (setName, paramId);
                    }
                }
                if (extraKeys.Count > 0)
                    throw new InvalidOperationException(
                        $"Persistence.Save: weight set '{setName}' maps parameter '{extraKeys.First()}'" +
                        (extraKeys.Count > 1 ? $" (and {extraKeys.Count - 1} more)" : string.Empty) +
                        ", which the model does not declare as a weight parameter.");

                mappingSets[setName] = new SkptMappingSet { Tensors = setRefs };

                // A set fully shared with already-stored data adds no data entry — its mapping
                // references the existing entries. Only its distinct tensors are written.
                if (newTensors.Count > 0)
                {
                    byte[] setBytes;
                    using (var buffer = new MemoryStream())
                    {
                        SafeTensorLoader.SaveSafeTensorsToStream(buffer, newTensors);
                        setBytes = buffer.ToArray();
                    }
                    var (setStored, setCompression, setAlign) = EncodeDataEntry(setBytes);
                    var setEntryPath = $"data/{setName}.safetensors";
                    dataEntries[setName] = new SkptDataEntry
                    {
                        Entry = setEntryPath,
                        Format = SkptFileFormat.DataFormatSafeTensors,
                        Compression = setCompression,
                        Sha256 = SkptFileFormat.Sha256Hex(setStored),
                    };
                    bodyEntries.Add(new(setEntryPath, setStored, Align: setAlign));
                }
            }

            // Host user-data bag (issue #101): a JSON object stored as its own data entry and
            // wired into the data registry (format "json", sha256). It is never referenced by a
            // tensor mapping, so Load ignores it; and it is stored uncompressed regardless of the
            // Zstd option (it is small config-like JSON, not a range-read tensor payload). Absent
            // unless WithUserData was called, so the no-user-data output stays byte-identical.
            if (_userData is not null)
            {
                var userDataBytes = SkptFileFormat.SerializeUserData(_userData);
                dataEntries[SkptFileFormat.UserDataDataKey] = new SkptDataEntry
                {
                    Entry = SkptFileFormat.UserDataEntryPath,
                    Format = SkptFileFormat.DataFormatJson,
                    Compression = SkptFileFormat.CompressionNone,
                    Sha256 = SkptFileFormat.Sha256Hex(userDataBytes),
                };
                bodyEntries.Add(new(SkptFileFormat.UserDataEntryPath, userDataBytes, Align: false));
            }

            var manifest = new SkptManifest
            {
                Format = SkptFileFormat.FormatName,
                SkptVersion = SkptFileFormat.CurrentVersion,
                CreatedUtc = DateTime.UtcNow.ToString("yyyy-MM-dd'T'HH:mm:ss'Z'"),
                Producer = new SkptProducerInfo { Shorokoo = ShorokooVersion.VersionString },
                // Null (never allocated) when no metadata was supplied, so JsonIgnoreCondition
                // .WhenWritingNull omits the key entirely and the output stays byte-identical.
                UserMetadata = _userMetadata,
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
                    [SkptFileFormat.DefaultModelKey] = mappingSets,
                },
                Data = dataEntries,
            };

            var entries = new List<SkptFileFormat.ZipEntrySpec>(bodyEntries.Count + 1)
            {
                new(SkptFileFormat.ConfigEntryName, SkptFileFormat.SerializeManifest(manifest), Align: false),
            };
            entries.AddRange(bodyEntries);
            AtomicFileWriter.WriteFile(filePath,
                stream => SkptFileFormat.WriteStoredZip(stream, entries, DateTime.UtcNow));
        }

        /// <summary>
        /// Content-addressed key for tensor dedup across mapping sets: dtype + shape + a SHA-256
        /// of the raw bytes, so two tensors share a key exactly when a load would bind identical
        /// bytes into an identically-shaped parameter.
        /// </summary>
        private static string ContentKey(TensorData data)
            => $"{data.DType}|{string.Join(",", data.Shape.Dims)}|" +
               SkptFileFormat.Sha256Hex(data.AccessRawMemory());

        /// <summary>
        /// The model's weight parameters: every MODEL_PARAM_DATA node except the RNG identity
        /// parameter (which is model definition, not a weight, and stays embedded). Each must
        /// carry a unique, non-empty identifier — it names the tensor in the checkpoint — and
        /// its tensor data. <paramref name="operation"/> names the caller in errors
        /// (<c>Persistence.Save</c> and <c>Persistence.ExportSafeTensors</c> share this walk).
        /// </summary>
        internal static List<FastNode> CollectWeightNodes(InternalComputationGraph graph, string operation)
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
                        $"{operation}: a model parameter carries no identifier, so its tensor " +
                        "cannot be named in the checkpoint.");
                if (!seenIds.Add(node.IdentifierTemplate))
                    throw new InvalidOperationException(
                        $"{operation}: two model parameters share the identifier " +
                        $"'{node.IdentifierTemplate}'; parameter identifiers must be unique to map tensors.");
                if (node.GetTensorData() is null)
                    throw new InvalidOperationException(
                        $"{operation}: parameter '{node.IdentifierTemplate}' carries no tensor data; " +
                        "a concrete model must have every parameter materialized.");

                nodes.Add(node);
            }
            return nodes;
        }

        /// <summary>
        /// Returns a copy of the graph with each weight parameter's tensor replaced by a
        /// dtype/shape-true <see cref="WeightPlaceholderTensorData"/> (the same
        /// clone-and-swap <see cref="InternalComputationGraph.WithUpdatedStates"/> uses).
        /// The placeholders are metadata-only — no values array is ever allocated, so
        /// stripping adds no per-weight peak memory — and serialize as empty,
        /// values-elided initializers that keep the definition a loadable concrete model,
        /// while the real bytes live once, in the checkpoint's data tree.
        /// </summary>
        internal static InternalComputationGraph StripWeights(
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
                     (object?)new WeightPlaceholderTensorData(data.Shape, data.DType)));
            }
            return stripped;
        }
    }
}

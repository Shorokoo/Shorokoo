using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using Shorokoo.Core;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Utils;
using Shorokoo.Graph;
using Shorokoo.Onnx;

namespace Shorokoo
{
    // Native .skpt persistence for training checkpoints (issue #95). A training-checkpoint .skpt
    // is a strict superset of an inference .skpt: it carries the concrete inference model
    // (models/model.srk) and a "default" weight set — which doubles as the run's trainable weights,
    // so Persistence.Load reads it straight back as an inference model — plus per-kind data entries
    // for the remaining training state (model state, optimizer state) and a manifest training block
    // recording the global step and which data entry holds each kind. The trainable weights live
    // once: the model's default weight mapping references the trainable data entry rather than
    // duplicating the bytes. This half of the Persistence facade owns that save/load; the container
    // primitives (writer, manifest schema, sha256/decompression helpers) are shared with the
    // inference path in Persistence.cs.
    public static partial class Persistence
    {
        /// <summary>
        /// Saves a <see cref="TrainingCheckpoint"/> as a native <c>.skpt</c> container: the concrete
        /// inference model (built from the checkpoint's trained weights via
        /// <see cref="TrainingCheckpoint.ToInferenceModel"/>) plus the training state split into
        /// per-kind data entries (trainable weights, model state, optimizer state), with the global
        /// step recorded in the manifest. The trainable-weights entry doubles as the model's default
        /// weight set, so the file also loads as an inference checkpoint via
        /// <see cref="Load(string)"/>. Reload the training state with
        /// <see cref="LoadTrainingCheckpoint"/> / <see cref="TrainingRig.LoadCheckpoint"/>.
        ///
        /// <para>The write is atomic (staged to a temp file and committed by rename). For per-entry
        /// Zstd compression or provenance metadata, use the builder form
        /// <see cref="ForTrainingCheckpoint"/>.</para>
        /// </summary>
        /// <param name="checkpoint">The training state to persist. Must carry its
        /// <see cref="TrainingCheckpoint.Rig"/> — the self-describing inference model is bound into the
        /// rig's retained concrete architecture, so no model graph or example input is needed.</param>
        /// <param name="filePath">Target <c>.skpt</c> path; its directory must already exist.</param>
        public static void SaveTrainingCheckpointToSkpt(
            TrainingCheckpoint checkpoint, string filePath)
            => ForTrainingCheckpoint(checkpoint).Save(filePath);

        /// <summary>
        /// Starts a native <c>.skpt</c> training-checkpoint save (issue #95). Compose the container
        /// features — <see cref="TrainingCheckpointBuilder.WithZstdCompressedData"/> and
        /// <see cref="TrainingCheckpointBuilder.WithMetadata"/> — then commit with
        /// <see cref="TrainingCheckpointBuilder.Save"/>. See
        /// <see cref="SaveTrainingCheckpointToSkpt"/> for the parameters and on-disk shape. The
        /// <paramref name="checkpoint"/> must carry its <see cref="TrainingCheckpoint.Rig"/>: the
        /// inference model is bound into the rig's retained concrete architecture (the same path
        /// <see cref="TrainingCheckpoint.ToInferenceModel()"/> uses), so a rig-less checkpoint throws.
        /// </summary>
        public static TrainingCheckpointBuilder ForTrainingCheckpoint(TrainingCheckpoint checkpoint)
        {
            if (checkpoint is null) throw new ArgumentNullException(nameof(checkpoint));
            if (checkpoint.Rig is null)
                throw new InvalidOperationException(
                    "SaveTrainingCheckpointToSkpt requires a training rig, but this checkpoint has none " +
                    "attached — the rig is the source of the self-describing inference model. Adopt one " +
                    "via rig.AdoptCheckpoint(checkpoint), or load the checkpoint against a rig " +
                    "(rig.LoadCheckpoint(path)), then save.");
            return new TrainingCheckpointBuilder(checkpoint);
        }

        /// <summary>
        /// Serializes one training-state <see cref="TensorDataStruct"/> to safetensors bytes, keyed
        /// by struct field name (no section prefix — each kind is its own data entry). Fields must be
        /// plain tensors; a nested-struct field fails loudly, mirroring the legacy flat writer.
        /// </summary>
        internal static byte[] SerializeTrainingKind(TensorDataStruct data, string kindLabel)
        {
            var tensors = new List<SafeTensor>(data.Definition.Fields.Length);
            foreach (var fieldDef in data.Definition.Fields)
            {
                if (data.Fields[fieldDef.Name] is not TensorData td)
                    throw new NotSupportedException(
                        $"Training-checkpoint {kindLabel} field '{fieldDef.Name}' is not a plain tensor; " +
                        "nested-struct fields are not supported by checkpoint serialization.");
                tensors.Add(new SafeTensor(
                    fieldDef.Name, td, SafeTensorLoader.DTypeToSafeTensorDType(td.DType), td.Shape.Dims));
            }
            using var buffer = new MemoryStream();
            SafeTensorLoader.SaveSafeTensorsToStream(buffer, tensors);
            return buffer.ToArray();
        }

        // ---- Load ----

        /// <summary>
        /// Reconstructs a <see cref="TrainingCheckpoint"/> from a native <c>.skpt</c> container
        /// written by <see cref="SaveTrainingCheckpointToSkpt"/>. Validated against the expected
        /// struct defs with the same fail-loud contract as <see cref="TrainingCheckpoint.Load"/>:
        /// every referenced entry's SHA-256 is verified, each kind's tensors must match the given
        /// def field-for-field (missing field, rank mismatch, or a stray tensor fails loudly), and a
        /// kind the def expects but the file omits fails loudly. Routed to by
        /// <see cref="LoadTrainingCheckpoint"/> when the file is a .skpt.
        ///
        /// <para><paramref name="components"/> selects which parts to load (<c>null</c> ⇒ everything
        /// present), exactly as the flat path (<see cref="TrainingCheckpoint.LoadFlat"/>) does: a
        /// dropped or absent kind is filled from <paramref name="rigForDefaults"/>'s initial values
        /// when a rig is supplied (<see cref="CheckpointComponents.InferenceState"/> → rig initial
        /// params + model state, <see cref="CheckpointComponents.OptimizerState"/> → rig initial
        /// optimizer state, <see cref="CheckpointComponents.Counters"/> → 0,
        /// <see cref="CheckpointComponents.Loss"/> → a <c>null</c> loss); without a rig, an
        /// absent-but-expected kind fails loud.</para>
        /// </summary>
        internal static TrainingCheckpoint LoadTrainingCheckpointFromSkpt(
            string filePath,
            TensorStructDef trainableParamDef,
            TensorStructDef modelStateDef,
            TensorStructDef optimizerStateDef,
            CheckpointComponents? components,
            TrainingRig? rigForDefaults)
        {
            if (trainableParamDef is null) throw new ArgumentNullException(nameof(trainableParamDef));
            if (modelStateDef is null) throw new ArgumentNullException(nameof(modelStateDef));
            if (optimizerStateDef is null) throw new ArgumentNullException(nameof(optimizerStateDef));

            var fileBytes = File.ReadAllBytes(filePath);
            using var fileStream = new MemoryStream(fileBytes, writable: false);
            using var archive = OpenArchive(fileStream, filePath);

            var configEntry = archive.GetEntry(SkptFileFormat.ConfigEntryName)
                ?? throw new InvalidDataException(
                    $"'{filePath}' is not a .skpt checkpoint — the archive contains no " +
                    $"'{SkptFileFormat.ConfigEntryName}' manifest.");
            var manifest = SkptFileFormat.ParseManifest(ReadEntryBytes(configEntry, filePath), filePath);
            ValidateManifestIdentity(manifest, filePath);

            var training = manifest.Training
                ?? throw new InvalidDataException(
                    $"'{filePath}': the .skpt manifest has no 'training' block — this is an inference " +
                    "checkpoint, not a training checkpoint. Load it with Persistence.Load instead.");
            if (training.CheckpointVersion == 0)
                throw new InvalidDataException(
                    $"'{filePath}': invalid training-checkpoint block — required field 'checkpointVersion' " +
                    "is missing or zero.");
            if (training.CheckpointVersion != SkptFileFormat.TrainingCheckpointVersion)
                throw new InvalidDataException(
                    $"'{filePath}': training-checkpoint block version {training.CheckpointVersion} is not " +
                    $"supported by this Shorokoo build (supported: {SkptFileFormat.TrainingCheckpointVersion}). " +
                    "The file was likely written by " +
                    (training.CheckpointVersion > SkptFileFormat.TrainingCheckpointVersion
                        ? "a newer framework version." : "an older, unsupported framework version."));

            // Step/epoch/batch are host-owned int64 scalars read straight from the manifest (the
            // manifest training block has stored them as int64 since #102, so #105's in-memory int64
            // widening needs no format change here — no truncation on read). Epoch and batchIndex are
            // add-only, nullable fields (issues #100 / #111): a .skpt that omits them — one written
            // before they existed, or one whose position is genuinely unknown — deserializes them to
            // null, never a sentinel 0.
            var kinds = training.Kinds ?? new Dictionary<string, string>();

            bool Want(CheckpointComponents c) => components is null || (components.Value & c) != 0;
            bool KindPresent(string kindName) =>
                kinds.TryGetValue(kindName, out var key) && !string.IsNullOrEmpty(key);

            // Counters (step/epoch/batch) ride with the Counters component; the loss is its own
            // independent Loss component. Each filtered out ⇒ 0 (step) / null (epoch, batch, loss),
            // mirroring the flat path.
            long step = Want(CheckpointComponents.Counters) ? training.Step : 0L;
            long? epoch = Want(CheckpointComponents.Counters) ? training.Epoch : null;
            long? batchIndex = Want(CheckpointComponents.Counters) ? training.BatchIndex : null;
            float? loss = Want(CheckpointComponents.Loss) ? training.Loss : null;

            TensorDataStruct trainable, modelState, optState;

            if (Want(CheckpointComponents.InferenceState)
                && (KindPresent(SkptFileFormat.TrainingKindTrainableParams) || trainableParamDef.Fields.Length == 0))
            {
                trainable = ReconstructTrainingKind(
                    archive, manifest, kinds, SkptFileFormat.TrainingKindTrainableParams, trainableParamDef, filePath);
                modelState = ReconstructTrainingKind(
                    archive, manifest, kinds, SkptFileFormat.TrainingKindModelState, modelStateDef, filePath);
            }
            else if (rigForDefaults is not null)
            {
                trainable = rigForDefaults.InitialTrainableStruct;
                modelState = rigForDefaults.InitialModelStateStruct;
            }
            else
            {
                trainable = ReconstructTrainingKind(
                    archive, manifest, kinds, SkptFileFormat.TrainingKindTrainableParams, trainableParamDef, filePath);
                modelState = ReconstructTrainingKind(
                    archive, manifest, kinds, SkptFileFormat.TrainingKindModelState, modelStateDef, filePath);
            }

            if (Want(CheckpointComponents.OptimizerState)
                && (KindPresent(SkptFileFormat.TrainingKindOptimizerState) || optimizerStateDef.Fields.Length == 0))
            {
                optState = ReconstructTrainingKind(
                    archive, manifest, kinds, SkptFileFormat.TrainingKindOptimizerState, optimizerStateDef, filePath);
            }
            else if (rigForDefaults is not null)
            {
                optState = rigForDefaults.InitialOptimizerStateStruct;
            }
            else
            {
                optState = ReconstructTrainingKind(
                    archive, manifest, kinds, SkptFileFormat.TrainingKindOptimizerState, optimizerStateDef, filePath);
            }

            return new TrainingCheckpoint(trainable, modelState, optState, step, epoch, batchIndex, rig: null, loss: loss);
        }

        /// <summary>
        /// Rebuilds one training-state kind against <paramref name="def"/>. A kind the manifest omits
        /// is valid only when the def carries no fields (then an empty struct); a def with fields but
        /// no matching entry fails loudly (the checkpoint is missing state this model/optimizer needs).
        /// </summary>
        private static TensorDataStruct ReconstructTrainingKind(
            ZipArchive archive, SkptManifest manifest, IReadOnlyDictionary<string, string> kinds,
            string kindName, TensorStructDef def, string filePath)
        {
            if (!kinds.TryGetValue(kindName, out var dataKey) || string.IsNullOrEmpty(dataKey))
            {
                if (def.Fields.Length > 0)
                    throw new InvalidDataException(
                        $"'{filePath}': the training checkpoint carries no '{kindName}' kind, but this " +
                        $"model/optimizer expects it ({def.Fields.Length} field(s)). Does the checkpoint " +
                        "match this model/optimizer?");
                return new TensorDataStruct(def, Array.Empty<KeyValuePair<string, IData>>());
            }

            var tensors = ReadTrainingKindTensors(archive, manifest, dataKey, kindName, filePath);
            return ReadTrainingKindSection(tensors, def, kindName, filePath);
        }

        /// <summary>
        /// Reads, SHA-256-verifies and decodes the safetensors data entry a training kind points at,
        /// returning its tensors by name. Mirrors the inference path's data-entry handling (integrity
        /// checked over the stored bytes, then any Zstd layer removed) via the shared helpers.
        /// </summary>
        private static Dictionary<string, TensorData> ReadTrainingKindTensors(
            ZipArchive archive, SkptManifest manifest, string dataKey, string kindName, string filePath)
        {
            if (manifest.Data is null || !manifest.Data.TryGetValue(dataKey, out var dataEntry) || dataEntry is null)
                throw new InvalidDataException(
                    $"'{filePath}': training kind '{kindName}' references data entry '{dataKey}', which the " +
                    "manifest's data registry does not declare.");
            if (string.IsNullOrEmpty(dataEntry.Entry))
                throw new InvalidDataException(
                    $"'{filePath}': the manifest's data entry '{dataKey}' (training kind '{kindName}') names " +
                    "no archive entry.");
            if (dataEntry.Format != SkptFileFormat.DataFormatSafeTensors)
                throw new InvalidDataException(
                    $"'{filePath}': data entry '{dataKey}' uses unsupported storage format " +
                    $"'{dataEntry.Format}' (supported: '{SkptFileFormat.DataFormatSafeTensors}'). " +
                    "The file was likely written by a newer framework version.");

            var storedBytes = ReadEntry(archive, dataEntry.Entry, $"data entry '{dataKey}'", filePath);
            VerifySha256(storedBytes, dataEntry.Sha256, dataEntry.Entry, filePath);
            var dataBytes = DecodeDataEntryPayload(storedBytes, dataEntry, dataKey, filePath);
            return SafeTensorLoader.ParseSafeTensorBytes(dataBytes)
                .ToDictionary(t => t.Name, t => t.Data, StringComparer.Ordinal);
        }

        /// <summary>
        /// Reconstructs a <see cref="TensorDataStruct"/> for one kind against <paramref name="def"/>:
        /// every def field must be present with a matching rank, and no stray tensor may remain — the
        /// field-name-keyed analogue of <see cref="TrainingCheckpoint"/>'s section reader.
        /// </summary>
        private static TensorDataStruct ReadTrainingKindSection(
            IReadOnlyDictionary<string, TensorData> tensors, TensorStructDef def, string kindName, string filePath)
        {
            var fields = new List<KeyValuePair<string, IData>>(def.Fields.Length);
            foreach (var fieldDef in def.Fields)
            {
                if (!tensors.TryGetValue(fieldDef.Name, out var td))
                    throw new InvalidDataException(
                        $"'{filePath}': training kind '{kindName}' is missing field '{fieldDef.Name}'. " +
                        "Does the checkpoint match this model/optimizer?");
                if (fieldDef.Rank is int rank && td.Shape.Dims.Length != rank)
                    throw new InvalidDataException(
                        $"'{filePath}': training-kind '{kindName}' field '{fieldDef.Name}' has rank " +
                        $"{td.Shape.Dims.Length}, expected {rank}.");
                fields.Add(new KeyValuePair<string, IData>(fieldDef.Name, td));
            }

            foreach (var name in tensors.Keys)
                if (def.GetField(name) is null)
                    throw new InvalidDataException(
                        $"'{filePath}': training kind '{kindName}' has unexpected tensor '{name}' not in " +
                        "this model/optimizer's definition.");

            return new TensorDataStruct(def, fields);
        }
    }

    /// <summary>
    /// Builder for a native <c>.skpt</c> training checkpoint, started by
    /// <see cref="Persistence.ForTrainingCheckpoint"/>. Optionally compose the container's features —
    /// <see cref="WithZstdCompressedData"/> for per-entry Zstd, <see cref="WithMetadata"/> for
    /// provenance — then commit with <see cref="Save"/>.
    /// </summary>
    public sealed class TrainingCheckpointBuilder
    {
        private readonly TrainingCheckpoint _checkpoint;
        private int? _zstdDataCompressionLevel;
        private Dictionary<string, string>? _userMetadata;

        internal TrainingCheckpointBuilder(TrainingCheckpoint checkpoint)
        {
            _checkpoint = checkpoint;
        }

        /// <summary>
        /// Opt-in: Zstd-compress the checkpoint's data-tree entries (the trainable/model/optimizer
        /// state entries), recording <c>compression: "zstd"</c> per compressed entry. The zip
        /// framing stays STORED and the manifest/model entries are never compressed — mirroring the
        /// inference builder's <see cref="CheckpointBuilder.WithZstdCompressedData"/>.
        /// </summary>
        /// <param name="compressionLevel">Zstandard level (1–22, default
        /// <see cref="CompressedFormatUtils.DefaultCompressionLevel"/>).</param>
        public TrainingCheckpointBuilder WithZstdCompressedData(
            int compressionLevel = CompressedFormatUtils.DefaultCompressionLevel)
        {
            if (compressionLevel is < 1 or > 22)
                throw new ArgumentOutOfRangeException(nameof(compressionLevel), compressionLevel,
                    "Zstandard compression level must be between 1 and 22.");
            _zstdDataCompressionLevel = compressionLevel;
            return this;
        }

        /// <summary>
        /// Opt-in: attaches user-supplied provenance metadata under the manifest's <c>userMetadata</c>
        /// key — descriptive, reproducibility metadata echoed back by <see cref="Persistence.Inspect"/>
        /// and never interpreted or used at load, exactly as for
        /// <see cref="CheckpointBuilder.WithMetadata"/>. The four well-known keys override any same-key
        /// entry in <paramref name="metadata"/>; calls accumulate.
        /// </summary>
        public TrainingCheckpointBuilder WithMetadata(
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
        /// Commits the training checkpoint as a single <c>.skpt</c> file. The write is atomic (staged
        /// to a temp file beside <paramref name="filePath"/> and committed by rename), so a crash
        /// mid-save never corrupts an existing checkpoint; the target's directory must already exist.
        /// See <see cref="Persistence.SaveTrainingCheckpointToSkpt"/> for the on-disk shape.
        /// </summary>
        public void Save(string filePath)
        {
            if (string.IsNullOrWhiteSpace(filePath))
                throw new ArgumentException("Checkpoint path cannot be null or empty.", nameof(filePath));

            // Build the concrete inference model — the self-describing "models/" half of the
            // container. This is the SAME weight-bind the extraction path uses: bind the checkpoint's
            // trainable params (the default weight set) and its model state (running stats etc., so a
            // stateful model like BatchNorm still concretizes) into the rig's retained concrete arch,
            // by canonical identity. Sourcing it from the rig (not a re-supplied model graph +
            // example input) means the container's self-describing model and ToInferenceModel() can
            // never diverge. Each parameter is mapped below to its own per-kind data entry, so no
            // weight bytes are duplicated.
            const string operation = "Persistence.SaveTrainingCheckpointToSkpt";
            var source = _checkpoint.Rig!.BindInferenceWeights(_checkpoint);
            var weightNodes = CheckpointBuilder.CollectWeightNodes(source, operation);

            // Default weight mapping: each model parameter (keyed by its full identifier) points at
            // its tensor in the trainable or model-state data entry, named by the checkpoint's struct
            // field name (the identifier's canonical dotted portion). Every parameter must have a
            // matching trainable or model-state field, or the model and checkpoint disagree.
            var trainableFieldNames = new HashSet<string>(
                _checkpoint.TrainableParams.Definition.Fields.Select(f => f.Name), StringComparer.Ordinal);
            var modelStateFieldNames = new HashSet<string>(
                _checkpoint.ModelState.Definition.Fields.Select(f => f.Name), StringComparer.Ordinal);
            var tensorRefs = new Dictionary<string, SkptTensorRef>(StringComparer.Ordinal);
            foreach (var node in weightNodes)
            {
                var fieldName = Core.Nodes.Processors.Training.FastDiscoverParamsHelpers
                    .ExtractTemplateString(node.IdentifierTemplate!);
                string dataKey =
                    trainableFieldNames.Contains(fieldName) ? SkptFileFormat.TrainableDataKey
                    : modelStateFieldNames.Contains(fieldName) ? SkptFileFormat.ModelStateDataKey
                    : throw new InvalidOperationException(
                        $"{operation}: the model's parameter '{node.IdentifierTemplate}' has no matching " +
                        $"trainable or model-state field '{fieldName}' in the checkpoint. The model graph " +
                        "and the checkpoint do not correspond.");
                tensorRefs[node.IdentifierTemplate!] = new SkptTensorRef { Data = dataKey, Tensor = fieldName };
            }

            // Serialize each training-state kind to safetensors (keyed by field name). The trainable
            // entry carries every trainable field (the authoritative source for reconstruction); the
            // model/optimizer state entries are written only when their struct is non-empty.
            var trainableBytes = Persistence.SerializeTrainingKind(_checkpoint.TrainableParams, "trainable");
            var modelBytes = CompressedFormatUtils.SaveFastGraphToBinary(
                CheckpointBuilder.StripWeights(source, weightNodes), GraphKind.ConcreteModel, compressed: true);

            (byte[] Stored, string Compression, bool Align) EncodeDataEntry(byte[] rawBytes)
                => _zstdDataCompressionLevel is int level
                    ? (CompressedFormatUtils.Compress(rawBytes, level), SkptFileFormat.CompressionZstd, false)
                    : (rawBytes, SkptFileFormat.CompressionNone, true);

            var dataEntries = new Dictionary<string, SkptDataEntry>(StringComparer.Ordinal);
            var bodyEntries = new List<SkptFileFormat.ZipEntrySpec>
            {
                new(SkptFileFormat.ModelEntryPath, modelBytes, Align: false),
            };
            var kinds = new Dictionary<string, string>(StringComparer.Ordinal);

            void AddKind(string kindName, string dataKey, string entryPath, byte[] rawBytes)
            {
                var (stored, compression, align) = EncodeDataEntry(rawBytes);
                dataEntries[dataKey] = new SkptDataEntry
                {
                    Entry = entryPath,
                    Format = SkptFileFormat.DataFormatSafeTensors,
                    Compression = compression,
                    Sha256 = SkptFileFormat.Sha256Hex(stored),
                };
                bodyEntries.Add(new(entryPath, stored, Align: align));
                kinds[kindName] = dataKey;
            }

            AddKind(SkptFileFormat.TrainingKindTrainableParams, SkptFileFormat.TrainableDataKey,
                SkptFileFormat.TrainableEntryPath, trainableBytes);
            if (_checkpoint.ModelState.Definition.Fields.Length > 0)
                AddKind(SkptFileFormat.TrainingKindModelState, SkptFileFormat.ModelStateDataKey,
                    SkptFileFormat.ModelStateEntryPath,
                    Persistence.SerializeTrainingKind(_checkpoint.ModelState, "model state"));
            if (_checkpoint.OptimizerState.Definition.Fields.Length > 0)
                AddKind(SkptFileFormat.TrainingKindOptimizerState, SkptFileFormat.OptimizerStateDataKey,
                    SkptFileFormat.OptimizerStateEntryPath,
                    Persistence.SerializeTrainingKind(_checkpoint.OptimizerState, "optimizer state"));

            var manifest = new SkptManifest
            {
                Format = SkptFileFormat.FormatName,
                SkptVersion = SkptFileFormat.CurrentVersion,
                CreatedUtc = DateTime.UtcNow.ToString("yyyy-MM-dd'T'HH:mm:ss'Z'"),
                Producer = new SkptProducerInfo { Shorokoo = ShorokooVersion.VersionString },
                UserMetadata = _userMetadata,
                Models = new Dictionary<string, SkptModelEntry>
                {
                    [SkptFileFormat.DefaultModelKey] = new SkptModelEntry
                    {
                        Entry = SkptFileFormat.ModelEntryPath,
                        Format = SkptFileFormat.ModelFormatSrk1,
                        Stage = SrkFileFormat.StageName(GraphKind.ConcreteModel),
                        Sha256 = SkptFileFormat.Sha256Hex(modelBytes),
                    },
                },
                TensorMappings = new Dictionary<string, Dictionary<string, SkptMappingSet>>
                {
                    [SkptFileFormat.DefaultModelKey] = new Dictionary<string, SkptMappingSet>(StringComparer.Ordinal)
                    {
                        [SkptFileFormat.DefaultMappingSetName] = new SkptMappingSet { Tensors = tensorRefs },
                    },
                },
                Data = dataEntries,
                Training = new SkptTrainingInfo
                {
                    CheckpointVersion = SkptFileFormat.TrainingCheckpointVersion,
                    Step = _checkpoint.Step,
                    // Epoch / batch index are host-owned run counters that may be genuinely unknown
                    // (null — no loader / no explicit counter). Nullable and add-only: a null value is
                    // omitted by the manifest serializer (WhenWritingNull) and reads back null, never a
                    // sentinel 0.0 — the same presence-gated treatment as the loss below.
                    Epoch = _checkpoint.Epoch,
                    BatchIndex = _checkpoint.BatchIndex,
                    // Loss is a host-owned run-progress scalar, its own savable component (independent
                    // of the counters). The .skpt builder writes every available component, so the loss
                    // is written iff the checkpoint carries one. Nullable and add-only: null (an
                    // initial/bare checkpoint) is omitted by the manifest serializer (WhenWritingNull),
                    // so it reads back as null — never a sentinel 0.0. A dropped Loss component on LOAD
                    // (or a null value) reads back null.
                    Loss = _checkpoint.Loss,
                    Kinds = kinds,
                },
            };

            var entries = new List<SkptFileFormat.ZipEntrySpec>(bodyEntries.Count + 1)
            {
                new(SkptFileFormat.ConfigEntryName, SkptFileFormat.SerializeManifest(manifest), Align: false),
            };
            entries.AddRange(bodyEntries);
            AtomicFileWriter.WriteFile(filePath,
                stream => SkptFileFormat.WriteStoredZip(stream, entries, DateTime.UtcNow));
        }
    }
}

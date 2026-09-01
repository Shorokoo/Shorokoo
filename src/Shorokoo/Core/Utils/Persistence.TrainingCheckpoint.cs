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
using Shorokoo.Runtime;

namespace Shorokoo
{
    // Native .skpt persistence for training checkpoints (issue #95). A training-checkpoint .skpt
    // is a strict superset of an inference .skpt: it carries the concrete inference model
    // (models/model.srk) and a "default" weight set — which doubles as the run's trainable weights,
    // so Persistence.Load reads it straight back as an inference model — plus data entries for the
    // remaining training state (model state, optimizer state) and a manifest training block
    // recording the global step. Every state tensor is addressed individually through the
    // manifest's tensorMappings (issue #184): the trainable weights and model state ride in the
    // inference model's default mapping — which thereby doubles as the training-state mapping, so
    // the bytes live once — and the optimizer state gets its own mapping under the optimizer
    // constituent's model key, keyed per (parameter × slot) instance. This half of the Persistence
    // facade owns that save/load; the container primitives (writer, manifest schema,
    // sha256/decompression helpers) are shared with the inference path in Persistence.cs.
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
        /// plain tensors; a nested-struct field fails loudly, mirroring the flat writer.
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
        /// Loads a <see cref="TrainingCheckpoint"/> from a native <c>.skpt</c> container written by
        /// <see cref="SaveTrainingCheckpointToSkpt"/> / <see cref="ForTrainingCheckpoint"/>. This
        /// entry point reads that format only: handed a flat safetensors checkpoint it fails
        /// immediately, naming <see cref="LoadTrainingCheckpoint"/> as the entry point for that
        /// shape (a caller with a genuinely unknown file identifies it with <see cref="Inspect"/>
        /// first). The checkpoint is reconstructed against the given struct defs (which pin the
        /// expected shapes, so a checkpoint from a different model or optimizer fails loudly). The
        /// result carries no <see cref="TrainingCheckpoint.Rig"/>; to resume a whole rig, prefer
        /// <see cref="TrainingRig.LoadCheckpointFromSkpt"/> (which supplies these defs from the
        /// rig) or the from-file-alone
        /// <see cref="TrainingRig.Load(string, ComputeContext?, ComputeContext?)"/>.
        /// </summary>
        public static TrainingCheckpoint LoadTrainingCheckpointFromSkpt(
            string filePath,
            TensorStructDef trainableParamDef,
            TensorStructDef modelStateDef,
            TensorStructDef optimizerStateDef)
        {
            if (string.IsNullOrWhiteSpace(filePath))
                throw new ArgumentException("Checkpoint path cannot be null or empty.", nameof(filePath));

            VerifySkptContainer(filePath,
                "Load a flat safetensors training checkpoint with Persistence.LoadTrainingCheckpoint.");
            return LoadTrainingCheckpointFromSkpt(
                filePath, trainableParamDef, modelStateDef, optimizerStateDef,
                components: null, rigForDefaults: null);
        }

        /// <summary>
        /// Reconstructs a <see cref="TrainingCheckpoint"/> from a native <c>.skpt</c> container
        /// written by <see cref="SaveTrainingCheckpointToSkpt"/>, resolving every state tensor
        /// individually through the manifest's tensor mappings (issue #184): trainable params and
        /// model state through the inference model's <c>default</c> mapping, optimizer state through
        /// the optimizer constituent's per-instance mapping. Validated against the expected struct
        /// defs with the same fail-loud contract as <see cref="TrainingCheckpoint.Load"/>: every
        /// referenced entry's SHA-256 is verified, and the mapped tensors must cover each def
        /// field-for-field (a missing field, a rank mismatch, or a mapped tensor no def declares
        /// fails loudly, naming the mismatch). Backs the public
        /// <see cref="LoadTrainingCheckpointFromSkpt(string, TensorStructDef, TensorStructDef, TensorStructDef)"/>
        /// and the rig-supplied <see cref="TrainingCheckpoint.LoadFromSkpt"/>; callers verify the
        /// container shape first.
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
                    $"readable by this Shorokoo build, which reads version " +
                    $"{SkptFileFormat.TrainingCheckpointVersion} only.");

            // Step/epoch/batch are host-owned int64 scalars read straight from the manifest, stored
            // as int64 so the in-memory int64 counters survive the round trip with no truncation on
            // read. Epoch and batchIndex are nullable: a .skpt whose position is genuinely unknown
            // omits them, and they deserialize to null rather than a sentinel 0.
            //
            // Every state tensor is addressed individually through the manifest's tensorMappings
            // (issue #184): the trainable params and model state through the inference model's
            // 'default' mapping — the very mapping Persistence.Load binds, so the bytes live once —
            // and the optimizer state through the optimizer constituent's mapping, keyed per
            // (parameter × slot) instance. Which def each tensor belongs to is re-derived from the
            // identifiers, with the same fail-loud coverage contract as before: a def field with no
            // mapped tensor, a mapped tensor no def declares, or a rank mismatch names the culprit.
            var modelMapping = GetDefaultMappingTensors(manifest, SkptFileFormat.DefaultModelKey);
            var optimizerMapping = GetDefaultMappingTensors(
                manifest, training.Rig?.OptimizerModel ?? SkptFileFormat.OptimizerModelKey);
            var tensorsByDataKey = new Dictionary<string, Dictionary<string, TensorData>>(StringComparer.Ordinal);

            bool Want(CheckpointComponents c) => components is null || (components.Value & c) != 0;

            // Counters (step/epoch/batch) ride with the Counters component; the loss is its own
            // independent Loss component. Each filtered out ⇒ 0 (step) / null (epoch, batch, loss),
            // mirroring the flat path.
            long step = Want(CheckpointComponents.Counters) ? training.Step : 0L;
            long? epoch = Want(CheckpointComponents.Counters) ? training.Epoch : null;
            long? batchIndex = Want(CheckpointComponents.Counters) ? training.BatchIndex : null;
            float? loss = Want(CheckpointComponents.Loss) ? training.Loss : null;

            TensorDataStruct trainable, modelState, optState;

            if (Want(CheckpointComponents.InferenceState)
                && (modelMapping is not null
                    || (trainableParamDef.Fields.Length == 0 && modelStateDef.Fields.Length == 0)))
            {
                (trainable, modelState) = ReconstructArchOwnedState(
                    archive, manifest, modelMapping, trainableParamDef, modelStateDef, tensorsByDataKey, filePath);
            }
            else if (rigForDefaults is not null)
            {
                trainable = rigForDefaults.InitialTrainableStruct;
                modelState = rigForDefaults.InitialModelStateStruct;
            }
            else
            {
                (trainable, modelState) = ReconstructArchOwnedState(
                    archive, manifest, modelMapping, trainableParamDef, modelStateDef, tensorsByDataKey, filePath);
            }

            if (Want(CheckpointComponents.OptimizerState)
                && (optimizerMapping is not null || optimizerStateDef.Fields.Length == 0))
            {
                optState = ReconstructOptimizerState(
                    archive, manifest, optimizerMapping, optimizerStateDef, tensorsByDataKey, filePath);
            }
            else if (rigForDefaults is not null)
            {
                optState = rigForDefaults.InitialOptimizerStateStruct;
            }
            else
            {
                optState = ReconstructOptimizerState(
                    archive, manifest, optimizerMapping, optimizerStateDef, tensorsByDataKey, filePath);
            }

            return new TrainingCheckpoint(trainable, modelState, optState, step, epoch, batchIndex, rig: null, loss: loss);
        }

        /// <summary>The tensors of a model's <c>default</c> mapping set, or null when the manifest
        /// carries no such set (distinct from an existing-but-empty set, which returns its empty
        /// dictionary — a model that genuinely has no parameters).</summary>
        private static Dictionary<string, SkptTensorRef>? GetDefaultMappingTensors(
            SkptManifest manifest, string modelKey)
            => manifest.TensorMappings is not null
               && manifest.TensorMappings.TryGetValue(modelKey, out var sets)
               && sets is not null
               && sets.TryGetValue(SkptFileFormat.DefaultMappingSetName, out var set)
               && set?.Tensors is not null
                ? set.Tensors
                : null;

        /// <summary>
        /// Rebuilds the arch-owned training state — the trainable parameters and the model state —
        /// from the inference model's <c>default</c> tensor mapping (issue #184). The mapping's keys
        /// are full parameter identifiers; which def each tensor belongs to is re-derived by
        /// matching the identifier's canonical dotted portion against the def field names. Coverage
        /// is exact both ways: a def field with no mapped tensor, or a mapped tensor neither def
        /// declares, fails loudly naming it — that is what catches a checkpoint from a different
        /// model.
        /// </summary>
        private static (TensorDataStruct Trainable, TensorDataStruct ModelState) ReconstructArchOwnedState(
            ZipArchive archive, SkptManifest manifest, IReadOnlyDictionary<string, SkptTensorRef>? mapping,
            TensorStructDef trainableParamDef, TensorStructDef modelStateDef,
            Dictionary<string, Dictionary<string, TensorData>> tensorsByDataKey, string filePath)
        {
            if (mapping is null)
            {
                if (trainableParamDef.Fields.Length > 0 || modelStateDef.Fields.Length > 0)
                    throw new InvalidDataException(
                        $"'{filePath}': the .skpt manifest has no '{SkptFileFormat.DefaultMappingSetName}' " +
                        $"tensor mapping for model '{SkptFileFormat.DefaultModelKey}', but this model expects " +
                        $"{trainableParamDef.Fields.Length} trainable and {modelStateDef.Fields.Length} " +
                        "model-state field(s). Does the checkpoint match this model?");
                return (new TensorDataStruct(trainableParamDef, Array.Empty<KeyValuePair<string, IData>>()),
                        new TensorDataStruct(modelStateDef, Array.Empty<KeyValuePair<string, IData>>()));
            }

            var byField = new Dictionary<string, (string Id, SkptTensorRef Ref)>(StringComparer.Ordinal);
            foreach (var (id, tensorRef) in mapping)
            {
                var fieldName = Core.Nodes.Processors.Training.FastDiscoverParamsHelpers
                    .ExtractTemplateString(id);
                if (!byField.TryAdd(fieldName, (id, tensorRef)))
                    throw new InvalidDataException(
                        $"'{filePath}': parameters '{byField[fieldName].Id}' and '{id}' in the " +
                        $"'{SkptFileFormat.DefaultMappingSetName}' mapping of model " +
                        $"'{SkptFileFormat.DefaultModelKey}' share the canonical name '{fieldName}'; the " +
                        "training state cannot be resolved unambiguously.");
            }

            var trainable = BuildStateStruct(archive, manifest, byField, trainableParamDef,
                "trainable parameter", "Does the checkpoint match this model?", tensorsByDataKey, filePath);
            var modelState = BuildStateStruct(archive, manifest, byField, modelStateDef,
                "model-state field", "Does the checkpoint match this model?", tensorsByDataKey, filePath);

            if (byField.Count > 0)
            {
                var stray = byField.Values.First().Id;
                throw new InvalidDataException(
                    $"'{filePath}': the checkpoint maps parameter '{stray}'" +
                    (byField.Count > 1 ? $" (and {byField.Count - 1} more)" : string.Empty) +
                    ", which this model's trainable/model-state definitions do not declare. Does the " +
                    "checkpoint match this model?");
            }
            return (trainable, modelState);
        }

        /// <summary>
        /// Rebuilds the optimizer state from the optimizer constituent's <c>default</c> tensor
        /// mapping (issue #184), whose keys are composite
        /// <c>{parameterIdentifier}#opt{slot}</c> per-instance identifiers — the arch-owned
        /// parameter identity plus the optimizer-owned slot index. Each key is resolved to its
        /// struct field name by construction (never by parsing field names), and coverage against
        /// <paramref name="def"/> is exact both ways, so a checkpoint whose optimizer state does not
        /// match the rig's optimizer fails loudly naming the mismatched instance.
        /// </summary>
        private static TensorDataStruct ReconstructOptimizerState(
            ZipArchive archive, SkptManifest manifest, IReadOnlyDictionary<string, SkptTensorRef>? mapping,
            TensorStructDef def,
            Dictionary<string, Dictionary<string, TensorData>> tensorsByDataKey, string filePath)
        {
            if (mapping is null)
            {
                if (def.Fields.Length > 0)
                    throw new InvalidDataException(
                        $"'{filePath}': the training checkpoint carries no optimizer-state tensor mapping, " +
                        $"but this optimizer expects {def.Fields.Length} state field(s). Does the " +
                        "checkpoint match this optimizer?");
                return new TensorDataStruct(def, Array.Empty<KeyValuePair<string, IData>>());
            }

            var byField = new Dictionary<string, (string Id, SkptTensorRef Ref)>(StringComparer.Ordinal);
            foreach (var (id, tensorRef) in mapping)
            {
                if (!SkptFileFormat.TryParseOptimizerStateId(id, out var paramId, out var slot))
                    throw new InvalidDataException(
                        $"'{filePath}': optimizer-state mapping key '{id}' is not a " +
                        $"'<parameter>{SkptFileFormat.OptimizerStateIdSeparator}<slot>' composite " +
                        "identifier.");
                var fieldName = TrainingRig.OptimizerStateFieldName(
                    Core.Nodes.Processors.Training.FastDiscoverParamsHelpers.ExtractTemplateString(paramId),
                    slot);
                if (!byField.TryAdd(fieldName, (id, tensorRef)))
                    throw new InvalidDataException(
                        $"'{filePath}': optimizer-state mapping keys '{byField[fieldName].Id}' and '{id}' " +
                        $"resolve to the same state instance '{fieldName}'.");
            }

            var optState = BuildStateStruct(archive, manifest, byField, def,
                "optimizer-state instance", "Does the checkpoint match this optimizer?",
                tensorsByDataKey, filePath);

            if (byField.Count > 0)
            {
                var stray = byField.Values.First().Id;
                throw new InvalidDataException(
                    $"'{filePath}': the checkpoint maps optimizer-state tensor '{stray}'" +
                    (byField.Count > 1 ? $" (and {byField.Count - 1} more)" : string.Empty) +
                    ", which this optimizer's state definition does not declare. Does the checkpoint " +
                    "match this optimizer?");
            }
            return optState;
        }

        /// <summary>
        /// Materializes one state struct against <paramref name="def"/> from the per-field mapping
        /// index, consuming each claimed entry (so the caller can fail loudly on leftovers): every
        /// def field must have a mapped tensor of the expected rank, resolved through the shared
        /// data-entry reader (SHA-256-verified and decoded at most once per entry via
        /// <paramref name="tensorsByDataKey"/>).
        /// </summary>
        private static TensorDataStruct BuildStateStruct(
            ZipArchive archive, SkptManifest manifest,
            Dictionary<string, (string Id, SkptTensorRef Ref)> byField,
            TensorStructDef def, string role, string mismatchHint,
            Dictionary<string, Dictionary<string, TensorData>> tensorsByDataKey, string filePath)
        {
            var fields = new List<KeyValuePair<string, IData>>(def.Fields.Length);
            foreach (var fieldDef in def.Fields)
            {
                if (!byField.Remove(fieldDef.Name, out var mapped))
                    throw new InvalidDataException(
                        $"'{filePath}': the checkpoint maps no tensor for {role} '{fieldDef.Name}'. " +
                        mismatchHint);
                var tensors = ResolveDataEntry(
                    archive, manifest, mapped.Ref, mapped.Id, tensorsByDataKey, filePath);
                if (string.IsNullOrEmpty(mapped.Ref.Tensor)
                    || !tensors.TryGetValue(mapped.Ref.Tensor, out var td))
                    throw new InvalidDataException(
                        $"'{filePath}': '{mapped.Id}' maps to tensor '{mapped.Ref.Tensor}' in data entry " +
                        $"'{mapped.Ref.Data}', but that entry contains no such tensor.");
                if (fieldDef.Rank is int rank && td.Shape.Dims.Length != rank)
                    throw new InvalidDataException(
                        $"'{filePath}': {role} '{fieldDef.Name}' has rank {td.Shape.Dims.Length}, " +
                        $"expected {rank}.");
                fields.Add(new KeyValuePair<string, IData>(fieldDef.Name, td));
            }
            return new TensorDataStruct(def, fields);
        }

        // ---- Rig reconstruction from a .skpt file alone (issue #115, folding in #106) ----

        /// <summary>
        /// Rebuilds a <see cref="TrainingRig"/> from the constituents serialized in a native
        /// <c>.skpt</c> checkpoint (#115) — the concrete architecture, loss, optimizer, and composed
        /// scheduler <c>models/</c> entries plus the rig block's hyperparameter bindings and RNG config
        /// — with no host-supplied source graphs. The model-input shapes ride on the arch itself (its
        /// self-describing MODEL_TENSOR_INPUT nodes), not the manifest. Backs the static
        /// <see cref="TrainingRig.Load(string, ComputeContext?, ComputeContext?)"/>. A file with no rig
        /// block (a flat checkpoint, which carries training state only) fails loudly.
        /// </summary>
        internal static TrainingRig ReconstructRigFromSkpt(
            string filePath, ComputeContext mergeContext, ComputeContext runtimeContext)
        {
            VerifySkptContainer(filePath,
                "A flat checkpoint stores training state only — no rig constituents to rebuild " +
                "from; rebuild the rig from its source graphs and resume with rig.LoadCheckpoint(path).");
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
                    "checkpoint, not a training checkpoint, so there is no rig to reconstruct.");
            var rig = training.Rig
                ?? throw new InvalidDataException(
                    $"'{filePath}': this training checkpoint stores no rig constituents. Rebuild the " +
                    "rig from its source graphs and resume with rig.LoadCheckpoint(path) instead.");
            if (rig.RigVersion == 0)
                throw new InvalidDataException(
                    $"'{filePath}': invalid rig block — required field 'rigVersion' is missing or zero.");
            if (rig.RigVersion != SkptFileFormat.TrainingRigVersion)
                throw new InvalidDataException(
                    $"'{filePath}': rig block version {rig.RigVersion} is not readable by this Shorokoo " +
                    $"build, which reads version {SkptFileFormat.TrainingRigVersion} only.");

            var archGraph = LoadConstituentGraph(archive, manifest, rig.ArchModel ?? SkptFileFormat.ArchModelKey, filePath);
            var lossGraph = LoadConstituentGraph(archive, manifest, rig.LossModel ?? SkptFileFormat.LossModelKey, filePath);
            var optimizerGraph = LoadConstituentGraph(archive, manifest, rig.OptimizerModel ?? SkptFileFormat.OptimizerModelKey, filePath);
            var schedulerGraph = rig.SchedulerModel is string schedKey
                ? LoadConstituentGraph(archive, manifest, schedKey, filePath)
                : null;

            // Model-input shapes are not read from the manifest: the deserialized arch's
            // MODEL_TENSOR_INPUT nodes carry their own representative-input attribute (round-tripped
            // as NodeProtos in the native .srk dialect), so the reconstructed arch is self-describing.

            var bindings = rig.Hyperparameters
                ?? throw new InvalidDataException(
                    $"'{filePath}': the rig block records no hyperparameter bindings.");
            var hypers = new Hyperparameter[bindings.Count];
            var names = new string[bindings.Count];
            for (int h = 0; h < bindings.Count; h++)
            {
                var b = bindings[h];
                names[h] = b.Name ?? $"hyperparam_{h}";
                hypers[h] = b.Kind switch
                {
                    SkptFileFormat.HyperKindBaked => Hyperparameter.Baked(ReadBakedHyper(b, names[h], filePath)),
                    SkptFileFormat.HyperKindRuntime => Hyperparameter.Runtime(
                        b.Shape ?? throw new InvalidDataException(
                            $"'{filePath}': runtime hyperparameter '{names[h]}' records no shape in the rig block.")),
                    SkptFileFormat.HyperKindScheduled => Hyperparameter.Scheduled(
                        TrainingRig.SplitSchedulerOutput(
                            schedulerGraph ?? throw new InvalidDataException(
                                $"'{filePath}': hyperparameter '{names[h]}' is scheduled but the checkpoint " +
                                "carries no scheduler constituent."),
                            names[h])),
                    _ => throw new InvalidDataException(
                        $"'{filePath}': hyperparameter '{names[h]}' records the unknown kind " +
                        $"'{b.Kind ?? "<none>"}'."),
                };
            }

            var rngConfig = DeserializeRngConfig(rig.Rng, filePath);

            return TrainingRig.ReconstructFromConstituents(
                archGraph, lossGraph, optimizerGraph, hypers, names, rngConfig,
                mergeContext, runtimeContext);
        }

        /// <summary>
        /// Reads a baked hyperparameter's constant off its binding: the recorded dtype and shape name how
        /// to decode the recorded base64 bytes. All three are required — a hyperparameter is any supported
        /// dtype at any shape, so an untyped or unshaped value is unreadable, not an older float32 scalar.
        /// </summary>
        private static TensorData ReadBakedHyper(SkptRigHyperparameter binding, string name, string filePath)
        {
            if (string.IsNullOrEmpty(binding.DType))
                throw new InvalidDataException(
                    $"'{filePath}': baked hyperparameter '{name}' records no dtype in the rig block.");
            if (binding.Value is null)
                throw new InvalidDataException(
                    $"'{filePath}': baked hyperparameter '{name}' records no value in the rig block.");
            if (binding.Shape is null)
                throw new InvalidDataException(
                    $"'{filePath}': baked hyperparameter '{name}' records no shape in the rig block.");
            var dtype = DType.FromName(binding.DType)
                ?? throw new InvalidDataException(
                    $"'{filePath}': baked hyperparameter '{name}' records the unknown dtype '{binding.DType}'.");
            try
            {
                return Globals.TensorData(dtype, binding.Shape, binding.Value);
            }
            catch (Exception e) when (e is FormatException or ArgumentException)
            {
                throw new InvalidDataException(
                    $"'{filePath}': baked hyperparameter '{name}' has a value that is not a valid " +
                    $"'{dtype}' tensor of shape [{string.Join(", ", binding.Shape)}].", e);
            }
        }

        /// <summary>Loads one constituent model graph from its <c>models/</c> entry, verifying SHA-256
        /// and stamping it with its recorded stage (#115).</summary>
        private static ComputationGraph LoadConstituentGraph(
            ZipArchive archive, SkptManifest manifest, string modelKey, string filePath)
        {
            if (manifest.Models is null || !manifest.Models.TryGetValue(modelKey, out var entry) || entry is null)
                throw new InvalidDataException(
                    $"'{filePath}': the rig references model '{modelKey}', which the manifest's model " +
                    "registry does not declare.");
            if (string.IsNullOrEmpty(entry.Entry))
                throw new InvalidDataException(
                    $"'{filePath}': the manifest's rig model '{modelKey}' names no archive entry.");
            if (entry.Format != SkptFileFormat.ModelFormatSrk1)
                throw new InvalidDataException(
                    $"'{filePath}': rig model '{modelKey}' uses unsupported serialization format " +
                    $"'{entry.Format}' (supported: '{SkptFileFormat.ModelFormatSrk1}').");

            var bytes = ReadEntry(archive, entry.Entry, $"rig model '{modelKey}'", filePath);
            VerifySha256(bytes, entry.Sha256, entry.Entry, filePath);
            var (graph, kind) = CompressedFormatUtils.LoadFastGraphCore(
                bytes, origin: $"{filePath}!{entry.Entry}", requiredStage: null);
            return new ComputationGraph(graph, kind);
        }

        /// <summary>Reconstructs an <see cref="RngConfig"/> from its manifest form (#115).</summary>
        private static RngConfig DeserializeRngConfig(SkptRngConfigInfo? info, string filePath)
        {
            if (info is null)
                throw new InvalidDataException(
                    $"'{filePath}': the rig block records no RNG config; the rig cannot be reconstructed.");
            var algorithm = Enum.TryParse<RngAlgorithm>(info.Algorithm, out var a)
                ? a
                : throw new InvalidDataException(
                    $"'{filePath}': the rig block records the unknown RNG algorithm " +
                    $"'{info.Algorithm ?? "<none>"}'.");
            var config = new RngConfig
            {
                MasterSeed = info.MasterSeed,
                InitMasterSeed = info.InitMasterSeed,
                RunMasterSeed = info.RunMasterSeed,
                Algorithm = algorithm,
            };
            foreach (var o in info.Overrides ?? new List<SkptRngOverride>())
            {
                if (!Enum.TryParse<RngCollection>(o.Collection, out var collection))
                    throw new InvalidDataException(
                        $"'{filePath}': the rig block records an override for the unknown RNG collection " +
                        $"'{o.Collection ?? "<none>"}'.");
                config = config.Override(collection, o.Path ?? Array.Empty<int>(), o.Seed);
            }
            return config;
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
            // matching trainable or model-state field, or the model and checkpoint disagree. This
            // one mapping is also how the training state is addressed on reload (issue #184): each
            // arch-owned state tensor rides in it per tensor, so it must cover the checkpoint's
            // trainable and model-state fields exactly — both directions are validated here.
            var trainableFieldNames = new HashSet<string>(
                _checkpoint.TrainableParams.Definition.Fields.Select(f => f.Name), StringComparer.Ordinal);
            var modelStateFieldNames = new HashSet<string>(
                _checkpoint.ModelState.Definition.Fields.Select(f => f.Name), StringComparer.Ordinal);
            var tensorRefs = new Dictionary<string, SkptTensorRef>(StringComparer.Ordinal);
            var identifierByField = new Dictionary<string, string>(StringComparer.Ordinal);
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
                identifierByField[fieldName] = node.IdentifierTemplate!;
            }
            foreach (var fieldName in trainableFieldNames.Concat(modelStateFieldNames))
                if (!identifierByField.ContainsKey(fieldName))
                    throw new InvalidOperationException(
                        $"{operation}: the checkpoint's state field '{fieldName}' has no matching " +
                        "parameter in the model, so its tensor cannot be addressed in the manifest's " +
                        "tensor mapping. The model graph and the checkpoint do not correspond.");

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

            void AddDataEntry(string dataKey, string entryPath, byte[] rawBytes)
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
            }

            AddDataEntry(SkptFileFormat.TrainableDataKey, SkptFileFormat.TrainableEntryPath, trainableBytes);
            if (_checkpoint.ModelState.Definition.Fields.Length > 0)
                AddDataEntry(SkptFileFormat.ModelStateDataKey, SkptFileFormat.ModelStateEntryPath,
                    Persistence.SerializeTrainingKind(_checkpoint.ModelState, "model state"));
            if (_checkpoint.OptimizerState.Definition.Fields.Length > 0)
                AddDataEntry(SkptFileFormat.OptimizerStateDataKey, SkptFileFormat.OptimizerStateEntryPath,
                    Persistence.SerializeTrainingKind(_checkpoint.OptimizerState, "optimizer state"));

            // Optimizer-state tensor mapping (issue #184): one entry per (trainable parameter ×
            // state slot) instance, keyed by the composite identifier — the parameter's full
            // identifier (its identity in the arch) plus the optimizer-owned slot index — under the
            // optimizer constituent's model key. The rig generates the optimizer-state def
            // param-major with TrainingRig.OptimizerStateFieldName, which is re-checked per field
            // here so the mapping can never silently drift from the stored tensor names.
            Dictionary<string, SkptTensorRef>? optimizerStateRefs = null;
            var optFields = _checkpoint.OptimizerState.Definition.Fields;
            if (optFields.Length > 0)
            {
                var trainableFields = _checkpoint.TrainableParams.Definition.Fields;
                if (trainableFields.Length == 0 || optFields.Length % trainableFields.Length != 0)
                    throw new InvalidOperationException(
                        $"{operation}: the checkpoint's optimizer state has {optFields.Length} field(s) " +
                        $"over {trainableFields.Length} trainable parameter(s) — not a whole number of " +
                        "state slots per parameter, so the per-instance tensor mapping cannot be built.");
                int slots = optFields.Length / trainableFields.Length;
                optimizerStateRefs = new Dictionary<string, SkptTensorRef>(StringComparer.Ordinal);
                for (int i = 0; i < optFields.Length; i++)
                {
                    var paramName = trainableFields[i / slots].Name;
                    int slot = i % slots;
                    if (optFields[i].Name != TrainingRig.OptimizerStateFieldName(paramName, slot))
                        throw new InvalidOperationException(
                            $"{operation}: optimizer-state field '{optFields[i].Name}' is not the expected " +
                            $"'{TrainingRig.OptimizerStateFieldName(paramName, slot)}' — the checkpoint's " +
                            "optimizer state does not follow the rig's param-major slot layout.");
                    optimizerStateRefs[SkptFileFormat.MakeOptimizerStateId(identifierByField[paramName], slot)] =
                        new SkptTensorRef
                        {
                            Data = SkptFileFormat.OptimizerStateDataKey,
                            Tensor = optFields[i].Name,
                        };
                }
            }

            // The model registry: the inference model (the one Persistence.Load binds) plus the rig's
            // constituent graphs as ordinary models/ entries (#115). Constituents carry no tensor
            // mapping — a from-file reconstruction re-derives the trainstep rather than binding weights
            // into them.
            var models = new Dictionary<string, SkptModelEntry>(StringComparer.Ordinal)
            {
                [SkptFileFormat.DefaultModelKey] = new SkptModelEntry
                {
                    Entry = SkptFileFormat.ModelEntryPath,
                    Format = SkptFileFormat.ModelFormatSrk1,
                    Stage = SrkFileFormat.StageName(GraphKind.ConcreteModel),
                    Sha256 = SkptFileFormat.Sha256Hex(modelBytes),
                },
            };

            // Serialize the rig constituents (#115, folding in #106): the concrete architecture (drives
            // trainstep re-derivation), the loss and optimizer module graphs, and — when any
            // hyperparameter is scheduled — the composed scheduler model, plus the non-graph recipe
            // (input shapes, hyperparameter bindings, RNG config) in the rig block.
            var rigInfo = AppendRigConstituents(_checkpoint.Rig!, models, bodyEntries);

            // Tensor mappings: the inference model's default set (which doubles as the
            // trainable/model-state training mapping — one entry per tensor, bytes stored once) plus,
            // when the optimizer is stateful, the optimizer constituent's per-instance set.
            var tensorMappings = new Dictionary<string, Dictionary<string, SkptMappingSet>>
            {
                [SkptFileFormat.DefaultModelKey] = new Dictionary<string, SkptMappingSet>(StringComparer.Ordinal)
                {
                    [SkptFileFormat.DefaultMappingSetName] = new SkptMappingSet { Tensors = tensorRefs },
                },
            };
            if (optimizerStateRefs is not null)
                tensorMappings[SkptFileFormat.OptimizerModelKey] =
                    new Dictionary<string, SkptMappingSet>(StringComparer.Ordinal)
                    {
                        [SkptFileFormat.DefaultMappingSetName] = new SkptMappingSet { Tensors = optimizerStateRefs },
                    };

            var manifest = new SkptManifest
            {
                Format = SkptFileFormat.FormatName,
                SkptVersion = SkptFileFormat.CurrentVersion,
                CreatedUtc = DateTime.UtcNow.ToString("yyyy-MM-dd'T'HH:mm:ss'Z'"),
                Producer = new SkptProducerInfo { Shorokoo = ShorokooVersion.VersionString },
                UserMetadata = _userMetadata,
                Models = models,
                TensorMappings = tensorMappings,
                Data = dataEntries,
                Training = new SkptTrainingInfo
                {
                    CheckpointVersion = SkptFileFormat.TrainingCheckpointVersion,
                    Rig = rigInfo,
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

        /// <summary>
        /// Serializes the rig's constituents (#115, folding in #106) into <paramref name="models"/> /
        /// <paramref name="bodyEntries"/> as ordinary <c>models/</c> entries — the concrete
        /// architecture, the loss and optimizer module graphs, and (when any hyperparameter is
        /// scheduled) the composed scheduler model — and returns the non-graph recipe (hyperparameter
        /// bindings, RNG config) as the manifest's rig block. Model-input shapes are NOT recorded: the
        /// serialized arch is self-describing (its MODEL_TENSOR_INPUT nodes carry the shape).
        /// </summary>
        private static SkptRigInfo AppendRigConstituents(
            TrainingRig rig,
            Dictionary<string, SkptModelEntry> models,
            List<SkptFileFormat.ZipEntrySpec> bodyEntries)
        {
            void AddModel(string key, string entryPath, ComputationGraph graph)
            {
                var bytes = CompressedFormatUtils.SaveFastGraphToBinary(graph, compressed: true);
                models[key] = new SkptModelEntry
                {
                    Entry = entryPath,
                    Format = SkptFileFormat.ModelFormatSrk1,
                    Stage = SrkFileFormat.StageName(graph.Kind),
                    Sha256 = SkptFileFormat.Sha256Hex(bytes),
                };
                bodyEntries.Add(new(entryPath, bytes, Align: false));
            }

            AddModel(SkptFileFormat.ArchModelKey, SkptFileFormat.ArchEntryPath, rig.ConcreteArchConstituent);
            AddModel(SkptFileFormat.LossModelKey, SkptFileFormat.LossEntryPath, rig.LossConstituent);
            AddModel(SkptFileFormat.OptimizerModelKey, SkptFileFormat.OptimizerEntryPath, rig.OptimizerConstituent);

            var (schedulerGraph, _) = rig.BuildComposedSchedulerModel();
            string? schedulerKey = null;
            if (schedulerGraph is not null)
            {
                AddModel(SkptFileFormat.SchedulerModelKey, SkptFileFormat.SchedulerEntryPath, schedulerGraph);
                schedulerKey = SkptFileFormat.SchedulerModelKey;
            }

            // No model-input shapes are recorded here: the arch's MODEL_TENSOR_INPUT nodes serialize
            // as NodeProtos in the native .srk dialect and carry their own representative-input
            // attribute, so the reconstructed arch is self-describing (a from-file load reads the
            // shapes straight off it via ReadRepresentativeInputs).

            // Hyperparameter bindings, in optimizer order. A baked one records its constant inline, at
            // the dtype the optimizer declares it at; a scheduled one maps to the scheduler model's
            // output of the same name; a runtime one is host-supplied.
            var names = rig.HyperparameterNames;
            var hyperBindings = new List<SkptRigHyperparameter>(rig.Hyperparameters.Count);
            for (int h = 0; h < rig.Hyperparameters.Count; h++)
            {
                var hv = rig.Hyperparameters[h];
                var name = h < names.Count ? names[h] : $"hyperparam_{h}";
                var kind = hv.Kind switch
                {
                    HyperparameterKind.Baked => SkptFileFormat.HyperKindBaked,
                    HyperparameterKind.Scheduled => SkptFileFormat.HyperKindScheduled,
                    HyperparameterKind.Runtime => SkptFileFormat.HyperKindRuntime,
                    _ => throw new InvalidOperationException($"Unknown hyperparameter kind {hv.Kind}."),
                };
                var binding = new SkptRigHyperparameter { Name = name, Kind = kind };
                if (hv.Kind == HyperparameterKind.Baked)
                {
                    // Normalized to the declared dtype at rig build, so this is the very constant the
                    // training-step graph carries.
                    binding.DType = hv.BakedDType.ToString();
                    binding.Shape = hv.BakedValue.Shape.Dims;
                    binding.Value = Convert.ToBase64String(hv.BakedValue.AccessRawMemory());
                }
                else if (hv.Kind == HyperparameterKind.Runtime)
                {
                    // A runtime hyperparameter's shape is declared by the host, so nothing else in the
                    // file records it; its dtype still comes from the optimizer constituent.
                    binding.Shape = [.. hv.RuntimeShape];
                }
                hyperBindings.Add(binding);
            }

            return new SkptRigInfo
            {
                RigVersion = SkptFileFormat.TrainingRigVersion,
                ArchModel = SkptFileFormat.ArchModelKey,
                LossModel = SkptFileFormat.LossModelKey,
                OptimizerModel = SkptFileFormat.OptimizerModelKey,
                SchedulerModel = schedulerKey,
                Hyperparameters = hyperBindings,
                Rng = SerializeRngConfig(rig.RngConfig),
            };
        }

        /// <summary>Serializes an <see cref="RngConfig"/> to its manifest form (#115).</summary>
        private static SkptRngConfigInfo SerializeRngConfig(RngConfig rng)
        {
            var overrides = rng.AllOverrides()
                .Select(o => new SkptRngOverride
                {
                    Collection = o.collection.ToString(),
                    Path = o.path,
                    Seed = o.seed,
                })
                .ToList();
            return new SkptRngConfigInfo
            {
                MasterSeed = rng.MasterSeed,
                InitMasterSeed = rng.InitMasterSeed,
                RunMasterSeed = rng.RunMasterSeed,
                Algorithm = rng.Algorithm.ToString(),
                Overrides = overrides.Count > 0 ? overrides : null,
            };
        }
    }
}

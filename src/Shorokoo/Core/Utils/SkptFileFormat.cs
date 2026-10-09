using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;

namespace Shorokoo.Core.Utils
{
    /// <summary>Producer metadata recorded in a .skpt manifest (informational).</summary>
    public sealed class SkptProducerInfo
    {
        /// <summary>Version of the Shorokoo framework that wrote the checkpoint.</summary>
        [JsonPropertyName("shorokoo")]
        public string? Shorokoo { get; set; }

        /// <summary>Round-trips producer fields this reader does not interpret, unchanged.</summary>
        [JsonExtensionData]
        public Dictionary<string, JsonElement>? AdditionalFields { get; set; }
    }

    /// <summary>One model in a .skpt manifest's model registry.</summary>
    public sealed class SkptModelEntry
    {
        /// <summary>Archive path of the serialized model definition (e.g. "models/model.srk").</summary>
        [JsonPropertyName("entry")]
        public string? Entry { get; set; }

        /// <summary>Serialization format of the entry; "srk1" is the only format written today.</summary>
        [JsonPropertyName("format")]
        public string? Format { get; set; }

        /// <summary>Lifecycle stage of the serialized graph, in .srk stage-name form
        /// (see <see cref="SrkFileFormat.StageName"/>); "concrete-model" today.</summary>
        [JsonPropertyName("stage")]
        public string? Stage { get; set; }

        /// <summary>Lowercase hex SHA-256 of the entry's bytes as stored in the archive.
        /// Doubles as the model's graph hash in this format version.</summary>
        [JsonPropertyName("sha256")]
        public string? Sha256 { get; set; }

        /// <summary>Round-trips model fields this reader does not interpret, unchanged.</summary>
        [JsonExtensionData]
        public Dictionary<string, JsonElement>? AdditionalFields { get; set; }
    }

    /// <summary>Resolution of one model tensor reference to a tensor inside a data entry.</summary>
    public sealed class SkptTensorRef
    {
        /// <summary>Key of the data entry (in the manifest's data registry) that stores the tensor.</summary>
        [JsonPropertyName("data")]
        public string? Data { get; set; }

        /// <summary>Name of the tensor inside the data entry.</summary>
        [JsonPropertyName("tensor")]
        public string? Tensor { get; set; }

        /// <summary>Round-trips reference fields this reader does not interpret, unchanged.</summary>
        [JsonExtensionData]
        public Dictionary<string, JsonElement>? AdditionalFields { get; set; }
    }

    /// <summary>A named set of tensor mappings for one model (e.g. the "default" weights).</summary>
    public sealed class SkptMappingSet
    {
        /// <summary>Per model tensor reference (parameter identifier), where its bytes live.</summary>
        [JsonPropertyName("tensors")]
        public Dictionary<string, SkptTensorRef>? Tensors { get; set; }

        /// <summary>Round-trips mapping-set fields this reader does not interpret, unchanged.</summary>
        [JsonExtensionData]
        public Dictionary<string, JsonElement>? AdditionalFields { get; set; }
    }

    /// <summary>One data entry in a .skpt manifest's data registry.</summary>
    public sealed class SkptDataEntry
    {
        /// <summary>Archive path of the data payload (e.g. "data/weights.safetensors").</summary>
        [JsonPropertyName("entry")]
        public string? Entry { get; set; }

        /// <summary>Storage format of the entry; "safetensors" is the only format written today.</summary>
        [JsonPropertyName("format")]
        public string? Format { get; set; }

        /// <summary>Compression of the entry's bytes: "none" (default) or "zstd" (a single
        /// Zstd layer inside the STORED zip bytes, mirroring .srk's header-declared
        /// compression). Always taken from here, never inferred from the entry's extension.</summary>
        [JsonPropertyName("compression")]
        public string? Compression { get; set; }

        /// <summary>Lowercase hex SHA-256 of the entry's bytes as stored in the archive —
        /// for a compressed entry, the compressed bytes. Integrity is thus checkable
        /// without decompressing, mirroring .srk's payloadSha256.</summary>
        [JsonPropertyName("sha256")]
        public string? Sha256 { get; set; }

        /// <summary>Round-trips data fields this reader does not interpret, unchanged.</summary>
        [JsonExtensionData]
        public Dictionary<string, JsonElement>? AdditionalFields { get; set; }
    }

    /// <summary>
    /// The training-checkpoint block of a .skpt manifest (issue #95). Present only when the file
    /// persists a <see cref="Shorokoo.TrainingCheckpoint"/>; absent for an ordinary inference
    /// checkpoint. It records the checkpoint's host-owned run counters (step, epoch, batch index)
    /// and the rig constituents; the training state itself is addressed per tensor through the
    /// manifest's <c>tensorMappings</c> (issue #184) — the trainable weights and model state by the
    /// inference model's <c>default</c> mapping (which thereby doubles as the training-state
    /// mapping, so no bytes are duplicated), the optimizer state by the optimizer constituent's
    /// <c>default</c> mapping under composite per-instance identifiers (see
    /// <see cref="SkptFileFormat.MakeOptimizerStateId"/>). Like the rest of the manifest, keys a
    /// reader does not interpret are ignored.
    /// </summary>
    public sealed class SkptTrainingInfo
    {
        /// <summary>Training-checkpoint block version; <see cref="SkptFileFormat.TrainingCheckpointVersion"/>
        /// for files written today. 0 means the field was absent (a malformed block).</summary>
        [JsonPropertyName("checkpointVersion")]
        public int CheckpointVersion { get; set; }

        /// <summary>The 0-based global training step the checkpoint sits at.</summary>
        [JsonPropertyName("step")]
        public long Step { get; set; }

        /// <summary>The 0-based epoch counter the checkpoint sits at — a host-owned run counter
        /// (issue #100), or <c>null</c> when the position is genuinely unknown (a checkpoint trained
        /// without a data loader / explicit counter — issue #111). Add-only and nullable: the serializer
        /// omits it when null (never a sentinel 0.0), and a manifest that omits it reads back as
        /// <c>null</c>.</summary>
        [JsonPropertyName("epoch")]
        public long? Epoch { get; set; }

        /// <summary>The 0-based batch index within the current epoch — a host-owned run counter
        /// (issue #100), or <c>null</c> when the position is genuinely unknown (issue #111). Add-only and
        /// nullable on the same terms as <see cref="Epoch"/>: omitted when null, absent ⇒ null.</summary>
        [JsonPropertyName("batchIndex")]
        public long? BatchIndex { get; set; }

        /// <summary>The loss of the training step that produced this checkpoint — a host-owned
        /// run-progress scalar, its own savable component (independent of the counters). Add-only and
        /// nullable: absent (⇒ read back as <c>null</c>) on an initial/bare checkpoint no step
        /// produced, or when the <see cref="Shorokoo.CheckpointComponents.Loss"/> component is
        /// filtered out on load. The serializer omits it when null (never a sentinel 0.0).</summary>
        [JsonPropertyName("loss")]
        public float? Loss { get; set; }

        /// <summary>The serialized <see cref="Shorokoo.TrainingRig"/> constituents (issue #115, folding
        /// in #106): enough to rebuild the whole rig — <c>trainstep</c> and all — from the checkpoint
        /// file alone, with no host-supplied source graphs. Present on every training <c>.skpt</c>;
        /// absent (⇒ <c>null</c>) on a <c>.skpt</c> that holds no training rig. Backs the
        /// <see cref="Shorokoo.CheckpointComponents.TrainingRig"/> flag and the static
        /// <see cref="Shorokoo.TrainingRig.Load(string, Shorokoo.Runtime.ComputeContext?, Shorokoo.Runtime.ComputeContext?, System.IProgress{Shorokoo.Graph.BuildProgress}, Shorokoo.TrainingBackend?)"/>.</summary>
        [JsonPropertyName("rig")]
        public SkptRigInfo? Rig { get; set; }

        /// <summary>Round-trips training fields this reader does not interpret, unchanged.</summary>
        [JsonExtensionData]
        public Dictionary<string, JsonElement>? AdditionalFields { get; set; }
    }

    /// <summary>
    /// One optimizer hyperparameter's persisted binding (issue #115 / #106), in the optimizer's
    /// declared order. <see cref="Kind"/> decides reconstruction: <c>"baked"</c> reads its constant off
    /// this entry's own <see cref="DType"/> / <see cref="Shape"/> / <see cref="Value"/>; <c>"runtime"</c>
    /// is rebuilt as a host-supplied runtime hyperparameter at the recorded <see cref="Shape"/>;
    /// <c>"scheduled"</c> takes the scheduler model's output named <see cref="Name"/> as its
    /// <c>counters → value</c> graph. A hyperparameter's dtype always comes from the optimizer
    /// constituent's own declaration, which is the source of truth on load as at build; only a baked
    /// binding records a dtype, as the key to reading its bytes back.
    /// </summary>
    public sealed class SkptRigHyperparameter
    {
        /// <summary>The hyperparameter's name (the optimizer's declared name, or <c>hyperparam_{i}</c>).</summary>
        [JsonPropertyName("name")]
        public string? Name { get; set; }

        /// <summary>The source kind: <c>"baked"</c>, <c>"scheduled"</c>, or <c>"runtime"</c>.</summary>
        [JsonPropertyName("kind")]
        public string? Kind { get; set; }

        /// <summary>
        /// The baked constant's dtype (a <see cref="Shorokoo.DType"/> name, e.g. <c>"Float32"</c>,
        /// <c>"Int32"</c>, <c>"Bool"</c>). Required on a <c>"baked"</c> entry — a hyperparameter is any
        /// supported dtype, so the value cannot be read without it — and absent otherwise.
        /// </summary>
        [JsonPropertyName("dtype")]
        public string? DType { get; set; }

        /// <summary>
        /// The hyperparameter's dims (empty for a scalar). Required on a <c>"baked"</c> entry, to read its
        /// bytes back, and on a <c>"runtime"</c> one, whose shape is declared by the host rather than
        /// derivable from the file. Absent on a <c>"scheduled"</c> entry, whose shape its scheduler graph
        /// carries.
        /// </summary>
        [JsonPropertyName("shape")]
        public long[]? Shape { get; set; }

        /// <summary>
        /// The baked constant, as base64 of its raw little-endian bytes at <see cref="DType"/> and
        /// <see cref="Shape"/>. Base64 rather than a decimal literal so every dtype (including
        /// <c>float16</c> / <c>bfloat16</c>) round-trips bit-exactly. Required on a <c>"baked"</c> entry,
        /// absent otherwise.
        /// </summary>
        [JsonPropertyName("value")]
        public string? Value { get; set; }

        /// <summary>Round-trips fields this reader does not interpret, unchanged.</summary>
        [JsonExtensionData]
        public Dictionary<string, JsonElement>? AdditionalFields { get; set; }
    }

    /// <summary>One <see cref="Shorokoo.RngConfig"/> per-stream override (issue #115).</summary>
    public sealed class SkptRngOverride
    {
        /// <summary>The stream collection: <c>"Params"</c> or <c>"Runtime"</c>.</summary>
        [JsonPropertyName("collection")]
        public string? Collection { get; set; }

        /// <summary>The consumer's absolute ModelId path.</summary>
        [JsonPropertyName("path")]
        public int[]? Path { get; set; }

        /// <summary>The override seed.</summary>
        [JsonPropertyName("seed")]
        public ulong Seed { get; set; }

        /// <summary>Round-trips fields this reader does not interpret, unchanged.</summary>
        [JsonExtensionData]
        public Dictionary<string, JsonElement>? AdditionalFields { get; set; }
    }

    /// <summary>The <see cref="Shorokoo.RngConfig"/> a rig was built with (issue #115), recorded so a
    /// reconstructed rig reproduces the same keyed initialization and runtime randomness.</summary>
    public sealed class SkptRngConfigInfo
    {
        /// <summary>The master seed folded into every non-overridden stream key.</summary>
        [JsonPropertyName("masterSeed")]
        public ulong MasterSeed { get; set; }

        /// <summary>Explicit init-collection sub-master, or <c>null</c> to derive from the master seed.</summary>
        [JsonPropertyName("initMasterSeed")]
        public ulong? InitMasterSeed { get; set; }

        /// <summary>Explicit runtime-collection sub-master, or <c>null</c> to derive from the master seed.</summary>
        [JsonPropertyName("runMasterSeed")]
        public ulong? RunMasterSeed { get; set; }

        /// <summary>The bit-generator algorithm name (the <see cref="Shorokoo.RngAlgorithm"/> enum name).</summary>
        [JsonPropertyName("algorithm")]
        public string? Algorithm { get; set; }

        /// <summary>Per-stream overrides; omitted when none.</summary>
        [JsonPropertyName("overrides")]
        public List<SkptRngOverride>? Overrides { get; set; }

        /// <summary>Round-trips fields this reader does not interpret, unchanged.</summary>
        [JsonExtensionData]
        public Dictionary<string, JsonElement>? AdditionalFields { get; set; }
    }

    /// <summary>
    /// The serialized training-rig constituents (issue #115, folding in #106): the model registry
    /// keys of the concrete architecture, loss, optimizer, and (optional) composed scheduler model
    /// entries, plus the hyperparameter bindings and the RNG config — the non-graph part of the recipe.
    /// Model-input shapes are NOT recorded here: the arch's <c>MODEL_TENSOR_INPUT</c> nodes serialize as
    /// NodeProtos in the native <c>.srk</c> dialect and carry the shape themselves, so the arch is
    /// self-describing. Together with those <c>models/</c> entries this is enough to rebuild the rig from
    /// the file alone. Keys a reader does not interpret are ignored.
    /// </summary>
    public sealed class SkptRigInfo
    {
        /// <summary>Rig-block version; <see cref="SkptFileFormat.TrainingRigVersion"/> for files written today.
        /// 0 means the field was absent (a malformed block).</summary>
        [JsonPropertyName("rigVersion")]
        public int RigVersion { get; set; }

        /// <summary>Model-registry key of the rig's concrete-architecture entry.</summary>
        [JsonPropertyName("archModel")]
        public string? ArchModel { get; set; }

        /// <summary>Model-registry key of the loss constituent entry.</summary>
        [JsonPropertyName("lossModel")]
        public string? LossModel { get; set; }

        /// <summary>Model-registry key of the optimizer constituent entry.</summary>
        [JsonPropertyName("optimizerModel")]
        public string? OptimizerModel { get; set; }

        /// <summary>Model-registry key of the composed scheduler entry, or <c>null</c> when the rig has
        /// no scheduled hyperparameter (nothing to compose).</summary>
        [JsonPropertyName("schedulerModel")]
        public string? SchedulerModel { get; set; }

        /// <summary>The hyperparameter bindings, in the optimizer's declared order. A baked binding
        /// carries its own dtype and value; see <see cref="SkptRigHyperparameter"/>.</summary>
        [JsonPropertyName("hyperparameters")]
        public List<SkptRigHyperparameter>? Hyperparameters { get; set; }

        /// <summary>The RNG config the rig was built with.</summary>
        [JsonPropertyName("rng")]
        public SkptRngConfigInfo? Rng { get; set; }

        /// <summary>The answers the backend's model of a run gave the rig's memory-aware pass — each
        /// the peak it told, or null where it could not — keyed by the SHA-256 of the question, which
        /// names the model asked about and the build of the backend asked. A rig loaded on the same
        /// build of the same backend takes them in place of asking; one loaded elsewhere asks anew.
        /// <c>null</c> when the pass asked nothing.</summary>
        [JsonPropertyName("runModelAnswers")]
        public Dictionary<string, long?>? RunModelAnswers { get; set; }

        /// <summary>Round-trips fields this reader does not interpret, unchanged.</summary>
        [JsonExtensionData]
        public Dictionary<string, JsonElement>? AdditionalFields { get; set; }
    }

    /// <summary>
    /// The config.json manifest of a .skpt checkpoint — the single source of wiring: archive
    /// entries never reference each other directly; every mapping (model → serialization
    /// format, model tensor references → data items, data item → storage format) lives here.
    /// A reader ignores keys it does not interpret; they are preserved in the extension-data
    /// bags.
    /// </summary>
    public sealed class SkptManifest
    {
        /// <summary>Format identifier; always <see cref="SkptFileFormat.FormatName"/>.</summary>
        [JsonPropertyName("format")]
        public string? Format { get; set; }

        /// <summary>Format major version; <see cref="SkptFileFormat.CurrentVersion"/> for files written today.</summary>
        [JsonPropertyName("skptVersion")]
        public int SkptVersion { get; set; }

        /// <summary>Creation time of the checkpoint, ISO-8601 UTC.</summary>
        [JsonPropertyName("createdUtc")]
        public string? CreatedUtc { get; set; }

        /// <summary>Producer metadata (framework version).</summary>
        [JsonPropertyName("producer")]
        public SkptProducerInfo? Producer { get; set; }

        /// <summary>Optional, user-supplied provenance metadata (git commit, dataset id, run
        /// name, license, and arbitrary key/value pairs) recorded at save time. Purely
        /// informational — trusted only as far as its writer: it never affects manifest
        /// identity checks or weight binding. Absent (null, and omitted from the JSON) unless the saver supplied
        /// it.</summary>
        [JsonPropertyName("userMetadata")]
        public Dictionary<string, string>? UserMetadata { get; set; }

        /// <summary>Model registry: model key → serialized model definition.</summary>
        [JsonPropertyName("models")]
        public Dictionary<string, SkptModelEntry>? Models { get; set; }

        /// <summary>Tensor mappings: model key → mapping-set name → tensor mapping set.
        /// Only the "default" set is written today; the shape allows parallel sets
        /// (e.g. EMA weights) later.</summary>
        [JsonPropertyName("tensorMappings")]
        public Dictionary<string, Dictionary<string, SkptMappingSet>>? TensorMappings { get; set; }

        /// <summary>Data registry: data key → stored tensor payload.</summary>
        [JsonPropertyName("data")]
        public Dictionary<string, SkptDataEntry>? Data { get; set; }

        /// <summary>Training-checkpoint block (issue #95): present only when the file persists a
        /// <see cref="Shorokoo.TrainingCheckpoint"/>, recording its run counters and rig
        /// constituents; the training state is addressed per tensor through
        /// <see cref="TensorMappings"/> (issue #184). Absent (null, omitted from the JSON) for an
        /// inference checkpoint.</summary>
        [JsonPropertyName("training")]
        public SkptTrainingInfo? Training { get; set; }

        /// <summary>Round-trips manifest fields this reader does not interpret, unchanged.</summary>
        [JsonExtensionData]
        public Dictionary<string, JsonElement>? AdditionalFields { get; set; }
    }

    /// <summary>
    /// The .skpt single-file checkpoint container: a standard zip archive whose entries are
    /// all STORED (uncompressed, method 0) — so tensor payloads remain range-readable through
    /// the zip central directory — bound together by a single config.json manifest
    /// (<see cref="SkptManifest"/>). Data-tree entry payloads are additionally aligned to
    /// <see cref="DataAlignment"/> bytes within the file so future memory-mapped/range reads
    /// stay possible. A data entry may opt into a single Zstd layer inside its STORED bytes,
    /// declared by the manifest's per-entry compression field — such an entry is not
    /// range-readable and skips the alignment. This class owns the container constants,
    /// the manifest schema (de)serialization, and the STORED zip writer; the save/load
    /// entry points live on <see cref="Shorokoo.Persistence"/>.
    /// </summary>
    public static class SkptFileFormat
    {
        /// <summary>Manifest format identifier.</summary>
        public const string FormatName = "skpt";

        /// <summary>Current manifest major version.</summary>
        public const int CurrentVersion = 1;

        /// <summary>Archive path of the manifest.</summary>
        public const string ConfigEntryName = "config.json";

        /// <summary>Archive path of the inference model definition entry. A training checkpoint adds
        /// the rig's constituent graphs beside it under <c>models/</c>.</summary>
        public const string ModelEntryPath = "models/model.srk";

        /// <summary>Archive path of the weights data entry an inference save writes. A training
        /// checkpoint writes its own per-kind training-state entries under <c>data/</c> instead.</summary>
        public const string WeightsEntryPath = "data/weights.safetensors";

        /// <summary>Archive path of the host user-data bag (issue #101).</summary>
        public const string UserDataEntryPath = "data/user-data.json";

        /// <summary>Model serialization format name for .srk payloads.</summary>
        public const string ModelFormatSrk1 = "srk1";

        /// <summary>Data storage format name for safetensors payloads.</summary>
        public const string DataFormatSafeTensors = "safetensors";

        /// <summary>Data storage format name for the host user-data bag — a JSON object
        /// (issue #101). Carried verbatim; never schema-checked or interpreted.</summary>
        public const string DataFormatJson = "json";

        /// <summary>Data compression name for uncompressed payloads.</summary>
        public const string CompressionNone = "none";

        /// <summary>Data compression name for Zstd-compressed payloads (one Zstd layer
        /// inside the entry's STORED zip bytes; the zip framing itself stays method 0).</summary>
        public const string CompressionZstd = "zstd";

        /// <summary>Alignment (bytes) of data-tree entry payloads within the archive.</summary>
        public const int DataAlignment = 64;

        /// <summary>Manifest key of the single model this slice writes.</summary>
        internal const string DefaultModelKey = "model";

        /// <summary>Manifest key of the single data entry this slice writes.</summary>
        internal const string DefaultDataKey = "weights";

        /// <summary>Name of the (only, for now) tensor mapping set.</summary>
        internal const string DefaultMappingSetName = "default";

        /// <summary>Manifest data-registry key of the host user-data bag (issue #101). Reserved:
        /// no weight set may take this name, so a user-data entry never collides with one.</summary>
        internal const string UserDataDataKey = "userData";

        /// <summary>Top-level user-data keys beginning with this character are reserved for
        /// Shorokoo and rejected from a host-supplied bag (issue #101).</summary>
        internal const char ReservedUserDataKeyPrefix = '$';

        // ---- Training-checkpoint container (issue #95) ----
        // A training-checkpoint .skpt is a superset of an inference .skpt: it carries the concrete
        // inference model (models/) plus its default weight set — which doubles as the run's
        // trainable weights — and adds data entries for the remaining training state (model state,
        // optimizer state). Every state tensor is addressed individually through tensorMappings
        // (issue #184): the trainable weights and model state ride in the inference model's
        // "default" mapping, keyed by parameter identifier; the optimizer state gets its own
        // "default" mapping under the optimizer constituent's model key, keyed by the composite
        // per-instance identifier below. The global step is small metadata and rides in the
        // manifest's dedicated training block rather than a data entry. A file carrying that block
        // reconstructs a TrainingCheckpoint; one without it is an ordinary inference checkpoint.

        /// <summary>Current training-checkpoint manifest-block version (see <see cref="SkptTrainingInfo"/>).</summary>
        public const int TrainingCheckpointVersion = 1;

        /// <summary>Manifest data-registry key of the trainable-weights entry. Doubles as the
        /// model's default weight set (referenced by the "default" tensor mapping) and as the
        /// training checkpoint's trainable-parameter storage.</summary>
        internal const string TrainableDataKey = "trainable";

        /// <summary>Manifest data-registry key of the model-state entry.</summary>
        internal const string ModelStateDataKey = "model_state";

        /// <summary>Manifest data-registry key of the optimizer-state entry.</summary>
        internal const string OptimizerStateDataKey = "optimizer_state";

        /// <summary>Archive path of the trainable-weights data entry.</summary>
        internal const string TrainableEntryPath = "data/trainable.safetensors";

        /// <summary>Archive path of the model-state data entry.</summary>
        internal const string ModelStateEntryPath = "data/model_state.safetensors";

        /// <summary>Archive path of the optimizer-state data entry.</summary>
        internal const string OptimizerStateEntryPath = "data/optimizer_state.safetensors";

        /// <summary>Manifest data-registry key of the training-history entry: one safetensors tensor
        /// per history column (see <see cref="Shorokoo.TrainingHistoryColumns"/>), present only when
        /// the checkpoint's history is non-empty.</summary>
        internal const string HistoryDataKey = "history";

        /// <summary>Archive path of the training-history data entry.</summary>
        internal const string HistoryEntryPath = "data/history.safetensors";

        /// <summary>
        /// Separator of the composite optimizer-state tensor identifier (issue #184). A single
        /// optimizer-state tensor's identity is a model parameter (arch-owned, named by its full
        /// parameter identifier) times a state slot (optimizer-owned, a 0-based index), so the
        /// optimizer constituent's tensor mapping keys each tensor as
        /// <c>{parameterIdentifier}#opt{slot}</c>. The optimizer's own ModelIds do not enumerate
        /// per-parameter state instances, hence the composite rather than a plain identifier.
        /// </summary>
        internal const string OptimizerStateIdSeparator = "#opt";

        /// <summary>Composes the optimizer-state mapping key for one (parameter, slot) instance
        /// (see <see cref="OptimizerStateIdSeparator"/>).</summary>
        internal static string MakeOptimizerStateId(string parameterIdentifier, int slot)
            => parameterIdentifier + OptimizerStateIdSeparator
               + slot.ToString(System.Globalization.CultureInfo.InvariantCulture);

        /// <summary>
        /// Splits a composite optimizer-state mapping key back into its parameter identifier and
        /// slot index. The separator is matched at its last occurrence (so a pathological
        /// identifier containing the separator still round-trips); false for a key that does not
        /// end in <c>#opt&lt;digits&gt;</c>.
        /// </summary>
        internal static bool TryParseOptimizerStateId(string id, out string parameterIdentifier, out int slot)
        {
            parameterIdentifier = string.Empty;
            slot = 0;
            int at = id.LastIndexOf(OptimizerStateIdSeparator, StringComparison.Ordinal);
            if (at <= 0) return false;
            var slotPart = id.AsSpan(at + OptimizerStateIdSeparator.Length);
            if (slotPart.IsEmpty || !int.TryParse(
                    slotPart, System.Globalization.NumberStyles.None,
                    System.Globalization.CultureInfo.InvariantCulture, out slot))
                return false;
            parameterIdentifier = id.Substring(0, at);
            return true;
        }

        // ---- Training-rig constituents (issue #115, folding in #106) ----
        // A training .skpt also carries the rig's constituent graphs as ordinary models/ entries — the
        // concrete architecture, loss, optimizer, and (when any hyperparameter is scheduled) a composed
        // scheduler model — plus the non-graph recipe (input shapes, hyperparameter bindings, RNG
        // config) in the manifest training block's rig section, so a fresh process rebuilds the whole
        // rig from the file alone. These sit alongside the "model" inference-model entry, which stays
        // the one Persistence.Load binds. No constituent binds weights (the rig re-derives), so the
        // arch, loss and scheduler entries carry no tensor mapping at all; the optimizer entry carries
        // the optimizer-state mapping, and only when the optimizer is stateful.

        /// <summary>Current rig-block version (see <see cref="SkptRigInfo"/>). The block carries no
        /// per-input <c>inputShapes</c> field: the arch's <c>MODEL_TENSOR_INPUT</c> nodes serialize as
        /// NodeProtos carrying their representative-input shape, so the arch is self-describing.</summary>
        public const int TrainingRigVersion = 1;

        /// <summary>Model-registry key of the rig's concrete-architecture constituent.</summary>
        internal const string ArchModelKey = "modelArch";

        /// <summary>Model-registry key of the loss constituent.</summary>
        internal const string LossModelKey = "loss";

        /// <summary>Model-registry key of the optimizer constituent.</summary>
        internal const string OptimizerModelKey = "optimizer";

        /// <summary>Model-registry key of the composed scheduler constituent.</summary>
        internal const string SchedulerModelKey = "scheduler";

        /// <summary>Archive path of the rig's concrete-architecture constituent entry.</summary>
        internal const string ArchEntryPath = "models/model-arch.srk";

        /// <summary>Archive path of the loss constituent entry.</summary>
        internal const string LossEntryPath = "models/loss.srk";

        /// <summary>Archive path of the optimizer constituent entry.</summary>
        internal const string OptimizerEntryPath = "models/optimizer.srk";

        /// <summary>Archive path of the composed scheduler constituent entry.</summary>
        internal const string SchedulerEntryPath = "models/scheduler.srk";

        /// <summary>Hyperparameter-binding kind name: a fixed value baked into the graph.</summary>
        internal const string HyperKindBaked = "baked";

        /// <summary>Hyperparameter-binding kind name: driven in-graph by the scheduler constituent.</summary>
        internal const string HyperKindScheduled = "scheduled";

        /// <summary>Hyperparameter-binding kind name: supplied by the host each step.</summary>
        internal const string HyperKindRuntime = "runtime";

        /// <summary>Well-known user-metadata key: the source-control commit the checkpoint
        /// was produced from (see <see cref="SkptManifest.UserMetadata"/>).</summary>
        public const string MetadataGitCommitKey = "gitCommit";

        /// <summary>Well-known user-metadata key: an identifier of the training/evaluation
        /// dataset (see <see cref="SkptManifest.UserMetadata"/>).</summary>
        public const string MetadataDatasetIdKey = "datasetId";

        /// <summary>Well-known user-metadata key: the name of the run that produced the
        /// checkpoint (see <see cref="SkptManifest.UserMetadata"/>).</summary>
        public const string MetadataRunNameKey = "runName";

        /// <summary>Well-known user-metadata key: the checkpoint's license
        /// (see <see cref="SkptManifest.UserMetadata"/>).</summary>
        public const string MetadataLicenseKey = "license";

        private static readonly JsonSerializerOptions ManifestJsonOptions = new()
        {
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            WriteIndented = true,
        };

        /// <summary>Serializes a manifest to the UTF-8 bytes of the config.json entry.</summary>
        internal static byte[] SerializeManifest(SkptManifest manifest)
            => JsonSerializer.SerializeToUtf8Bytes(manifest, ManifestJsonOptions);

        /// <summary>
        /// Parses a config.json entry. Unknown keys are tolerated (and preserved in the
        /// extension-data bags); malformed JSON fails loudly naming <paramref name="origin"/>.
        /// </summary>
        internal static SkptManifest ParseManifest(byte[] configBytes, string origin)
        {
            SkptManifest? manifest;
            try
            {
                manifest = JsonSerializer.Deserialize<SkptManifest>(configBytes, ManifestJsonOptions);
            }
            catch (JsonException e)
            {
                throw new InvalidDataException(
                    $"'{origin}': corrupt .skpt checkpoint — '{ConfigEntryName}' does not parse as JSON: {e.Message}", e);
            }
            if (manifest is null)
                throw new InvalidDataException(
                    $"'{origin}': corrupt .skpt checkpoint — '{ConfigEntryName}' is JSON null.");
            return manifest;
        }

        /// <summary>Lowercase hex SHA-256, as recorded in manifest sha256 fields.</summary>
        internal static string Sha256Hex(ReadOnlySpan<byte> data)
            => Convert.ToHexString(SHA256.HashData(data)).ToLowerInvariant();

        private static readonly JsonSerializerOptions UserDataJsonOptions = new()
        {
            WriteIndented = true,
        };

        /// <summary>Serializes the host user-data bag to the UTF-8 bytes of the
        /// <see cref="UserDataEntryPath"/> entry (issue #101). The bag is written verbatim; its
        /// values are never interpreted.</summary>
        internal static byte[] SerializeUserData(JsonObject userData)
            => JsonSerializer.SerializeToUtf8Bytes(userData, UserDataJsonOptions);

        /// <summary>
        /// Parses the host user-data entry back into its JSON DOM (issue #101). Validates
        /// well-formedness only: the bytes must be valid JSON whose root is an object — never the
        /// shape or meaning of the values. Throws <see cref="InvalidDataException"/> for malformed
        /// JSON or a non-object root (a <see cref="JsonException"/> is wrapped).
        /// </summary>
        internal static JsonObject ParseUserData(byte[] bytes, string origin)
        {
            JsonNode? node;
            try
            {
                node = JsonNode.Parse(bytes);
            }
            catch (JsonException e)
            {
                throw new InvalidDataException(
                    $"'{origin}': the '{UserDataEntryPath}' entry does not parse as JSON: {e.Message}", e);
            }
            if (node is not JsonObject obj)
                throw new InvalidDataException(
                    $"'{origin}': the '{UserDataEntryPath}' entry is not a JSON object at its root.");
            return obj;
        }

        /// <summary>
        /// Whether the bytes open with the Zstd frame magic (28 B5 2F FD, little-endian
        /// 0xFD2FB528). Used to cross-check a data entry's stored bytes against its
        /// manifest-declared compression — never to decide the compression itself, which
        /// only ever comes from the manifest.
        /// </summary>
        internal static bool LooksLikeZstdFrame(ReadOnlySpan<byte> data)
            => data.Length >= 4 && data[0] == 0x28 && data[1] == 0xB5 && data[2] == 0x2F && data[3] == 0xFD;

        /// <summary>
        /// Writes checkpoint entries as real files under <paramref name="stagingRoot"/> — the
        /// directory form of a .skpt (issue #183). Each entry's archive-relative path becomes a
        /// file path (creating subdirectories as needed) with byte-identical content to the zip
        /// form; the zip's payload alignment is moot here (a real file is already page-aligned
        /// and range-readable), so <see cref="ZipEntrySpec.Align"/> is ignored. Every file is
        /// flushed to disk, since the atomic directory commit is a rename of
        /// <paramref name="stagingRoot"/> itself. Entry paths resolve through the same
        /// inside-the-root rule the directory reader enforces, so nothing is ever written
        /// outside the staging root; <paramref name="origin"/> names the checkpoint being
        /// written (or converted) in that failure, not the transient staging path.
        /// </summary>
        internal static void WriteDirectoryEntries(
            string stagingRoot, IReadOnlyList<ZipEntrySpec> entries, string origin)
        {
            if (entries is null) throw new ArgumentNullException(nameof(entries));
            var rootFull = Path.GetFullPath(stagingRoot);

            // Resolve every entry and create the needed subdirectories once, up front — never
            // per entry. This is load-bearing for the stale-staging sweep's guarantee: after
            // this point nothing here recreates a directory, so if a concurrent sweep judges
            // this staging tree abandoned and renames it away mid-write, the next CreateNew
            // fails loudly (DirectoryNotFoundException) instead of silently rebuilding a
            // partial tree that could then be committed.
            var resolvedPaths = new string[entries.Count];
            var directories = new HashSet<string>(StringComparer.Ordinal);
            for (int i = 0; i < entries.Count; i++)
            {
                resolvedPaths[i] = SkptDirectoryContainer.ResolveEntryPath(rootFull, entries[i].Name, origin);
                directories.Add(Path.GetDirectoryName(resolvedPaths[i])!);
            }
            foreach (var dir in directories)
                Directory.CreateDirectory(dir);

            for (int i = 0; i < entries.Count; i++)
            {
                using var fs = new FileStream(
                    resolvedPaths[i], FileMode.CreateNew, FileAccess.Write, FileShare.None);
                entries[i].Payload.WriteTo(PayloadDestination(fs), entries[i].Name);
                fs.Flush(flushToDisk: true);
            }
        }

        /// <summary>
        /// Test hook: wraps the stream an entry's payload is written to, in either form, so a test
        /// can write a payload of gibibytes of zeros as a sparse file that holds no disk space. The
        /// wrapper writes through to the stream it is given, at that stream's position, and leaves
        /// it open. Thread-scoped, so a hook installed by one parallel test is invisible to every
        /// other thread; still reset it in a <c>finally</c>.
        /// </summary>
        [ThreadStatic]
        internal static Func<Stream, Stream>? PayloadDestinationInjection;

        private static Stream PayloadDestination(Stream stream)
            => PayloadDestinationInjection?.Invoke(stream) ?? stream;

        #region STORED zip writer

        /// <summary>One entry to be written into a .skpt archive.</summary>
        /// <param name="Name">Archive path (forward slashes, ASCII).</param>
        /// <param name="Payload">Entry bytes; always STORED verbatim.</param>
        /// <param name="Align">Pad the local header (via a zipalign-style extra field) so the
        /// entry's payload starts at a <see cref="DataAlignment"/>-byte file offset.</param>
        internal readonly record struct ZipEntrySpec(string Name, EntryPayload Payload, bool Align)
        {
            /// <summary>An entry whose bytes are already in hand.</summary>
            public ZipEntrySpec(string Name, byte[] Data, bool Align)
                : this(Name, EntryPayload.Of(Data), Align)
            {
            }
        }

        /// <summary>
        /// The bytes of one entry: held, produced on demand by a writer, or copied out of another
        /// checkpoint's stored entry. Only a held entry is ever whole in memory, and an entry of any
        /// size can be produced or copied. A produced entry's writer runs into sinks that measure it
        /// — its length, then its CRC-32 and SHA-256, which the manifest and the zip headers need
        /// before its first byte is written — and once more into the destination. A copied entry is
        /// read in bounded pieces, once to measure it when its CRC-32 is asked for and once into the
        /// destination, and each read is checked against the SHA-256 its checkpoint records for it.
        /// Writing an entry therefore costs no managed copy of what it carries, however large that is.
        /// </summary>
        internal sealed class EntryPayload
        {
            private const int CopyBufferBytes = 1 << 20;

            private readonly byte[]? _bytes;
            private readonly Action<Stream>? _produce;
            private readonly Func<Stream>? _open;
            private readonly Action<string>? _verify;
            private readonly string? _origin;
            private uint? _crc32;
            private string? _sha256;

            private EntryPayload(
                byte[]? bytes, Action<Stream>? produce, Func<Stream>? open, Action<string>? verify,
                string? origin, long length, uint? crc32, string? sha256)
            {
                _bytes = bytes;
                _produce = produce;
                _open = open;
                _verify = verify;
                _origin = origin;
                Length = length;
                _crc32 = crc32;
                _sha256 = sha256;
            }

            /// <summary>An entry over bytes already in hand.</summary>
            public static EntryPayload Of(byte[] bytes)
                => new(bytes ?? throw new ArgumentNullException(nameof(bytes)),
                    null, null, null, null, bytes.Length, null, null);

            /// <summary>An entry produced by <paramref name="produce"/>, which must write the same
            /// bytes every time it runs. It runs here twice: once to count its bytes, and once to
            /// hash them.</summary>
            public static EntryPayload Produced(Action<Stream> produce)
            {
                if (produce is null) throw new ArgumentNullException(nameof(produce));
                using (var counter = new MeasuringStream(destination: null, hash: false, long.MaxValue, Overrun))
                    produce(counter);
                using var sink = new MeasuringStream(destination: null, hash: true, long.MaxValue, Overrun);
                produce(sink);
                return new(null, produce, null, null, null, sink.Length, sink.Crc32, sink.Sha256Hex());
            }

            /// <summary>
            /// An entry of <paramref name="length"/> bytes copied from the stream
            /// <paramref name="open"/> returns, which is opened afresh for each read of it and
            /// disposed after. Every read passes the SHA-256 of what it read to
            /// <paramref name="verify"/>, which refuses bytes that are not the entry's; a read that
            /// yields other than <paramref name="length"/> bytes is refused as a truncated or
            /// overlong entry; <paramref name="origin"/> names the entry, and its checkpoint, in that
            /// refusal.
            /// </summary>
            public static EntryPayload Copied(Func<Stream> open, long length, Action<string> verify, string origin)
            {
                if (open is null) throw new ArgumentNullException(nameof(open));
                if (verify is null) throw new ArgumentNullException(nameof(verify));
                return new(null, null, open, verify, origin, length, null, null);
            }

            private static Exception Overrun(long length)
                => new NotSupportedException(
                    $"A .skpt entry would be at least {length} bytes, more than a stream can address.");

            /// <summary>The entry's size in bytes.</summary>
            public long Length { get; }

            /// <summary>The zip CRC-32 of the entry's bytes.</summary>
            public uint Crc32
            {
                get
                {
                    if (_crc32 is null)
                    {
                        if (_bytes is not null) _crc32 = SkptFileFormat.Crc32(_bytes);
                        else MeasureCopy();
                    }
                    return _crc32!.Value;
                }
            }

            /// <summary>Lowercase hex SHA-256 of the entry's bytes, as a manifest records it.</summary>
            public string Sha256
            {
                get
                {
                    if (_sha256 is null)
                    {
                        if (_bytes is not null) _sha256 = Sha256Hex(_bytes);
                        else MeasureCopy();
                    }
                    return _sha256!;
                }
            }

            private void MeasureCopy()
            {
                using var sink = new MeasuringStream(destination: null, hash: true, Length, CopiedLengthDiffers);
                _sha256 = CopyInto(sink);
                _crc32 = sink.Crc32;
            }

            /// <summary>Reads a copied entry into <paramref name="sink"/> through one bounded
            /// buffer, holding it to its length and to the SHA-256 its checkpoint records, and
            /// returns that SHA-256.</summary>
            private string CopyInto(MeasuringStream sink)
            {
                var buffer = System.Buffers.ArrayPool<byte>.Shared.Rent(CopyBufferBytes);
                try
                {
                    using var source = _open!();
                    int read;
                    while ((read = source.Read(buffer, 0, CopyBufferBytes)) > 0)
                        sink.Write(buffer.AsSpan(0, read));
                }
                finally
                {
                    System.Buffers.ArrayPool<byte>.Shared.Return(buffer);
                }
                if (sink.Length != Length) throw CopiedLengthDiffers(sink.Length);
                var sha256 = sink.Sha256Hex();
                _verify!(sha256);
                return sha256;
            }

            private Exception CopiedLengthDiffers(long read)
                => new InvalidDataException(
                    $"{_origin} reads as {(read > Length ? "at least " : string.Empty)}{read} bytes, but the " +
                    $"checkpoint declares {Length} bytes for it; the checkpoint is truncated or corrupt.");

            /// <summary>Writes the entry's bytes to <paramref name="destination"/>. A produced entry
            /// is held to the length it measured at: one that writes more or fewer bytes has changed
            /// under the save, and is refused rather than written into an archive whose headers and
            /// manifest describe other bytes. Only the length is checked — a producer that writes
            /// other bytes of the same length is not detected here, and fails its SHA-256 check on
            /// load. A copied entry is held to its length and to its recorded SHA-256 alike.</summary>
            public void WriteTo(Stream destination, string entryName)
            {
                if (_bytes is not null)
                {
                    destination.Write(_bytes);
                    return;
                }
                if (_open is not null)
                {
                    using var copied = new MeasuringStream(
                        destination, hash: true, Length, CopiedLengthDiffers, crc32: false);
                    CopyInto(copied);
                    return;
                }
                Exception Changed(long written) => new InvalidOperationException(
                    $"The .skpt entry '{entryName}' wrote {written} bytes, but measured " +
                    $"{Length} bytes moments earlier: its content changed while it was being saved.");
                using var counted = new MeasuringStream(destination, hash: false, Length, Changed);
                _produce!(counted);
                if (counted.Length != Length) throw Changed(counted.Length);
            }
        }

        /// <summary>A write-only stream that counts, and optionally SHA-256s — and CRC-32s, unless
        /// told not to — what is written through it, forwarding it to <c>destination</c> when there
        /// is one. A write that would take it past <c>limit</c> bytes throws <c>overrun</c>'s
        /// exception before any of that write is hashed or forwarded.</summary>
        private sealed class MeasuringStream : Stream, ILengthOnlyStream
        {
            private readonly Stream? _destination;
            private readonly IncrementalHash? _sha256;
            private readonly long _limit;
            private readonly Func<long, Exception> _overrun;
            private readonly System.IO.Hashing.Crc32? _crc32;

            public MeasuringStream(
                Stream? destination, bool hash, long limit, Func<long, Exception> overrun, bool crc32 = true)
            {
                _destination = destination;
                _limit = limit;
                _overrun = overrun;
                if (hash) _sha256 = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
                if (hash && crc32) _crc32 = new System.IO.Hashing.Crc32();
            }

            public override long Length => _length;
            private long _length;

            public uint Crc32 => _crc32?.GetCurrentHashAsUInt32() ?? 0;

            public string Sha256Hex() => Convert.ToHexString(_sha256!.GetHashAndReset()).ToLowerInvariant();

            public override void Write(ReadOnlySpan<byte> buffer)
            {
                if (_length + buffer.Length > _limit) throw _overrun(_length + buffer.Length);
                _length += buffer.Length;
                if (_sha256 is not null)
                {
                    _sha256.AppendData(buffer);
                    _crc32?.Append(buffer);
                }
                _destination?.Write(buffer);
            }

            public override void Write(byte[] buffer, int offset, int count)
                => Write(buffer.AsSpan(offset, count));

            public bool IsLengthOnly => _destination is null && _sha256 is null;

            public void Advance(long count)
            {
                if (!IsLengthOnly)
                    throw new InvalidOperationException("Only a stream that just counts can be advanced without writing.");
                if (_length + count > _limit) throw _overrun(_length + count);
                _length += count;
            }

            public override void WriteByte(byte value) => Write([value]);

            public override bool CanRead => false;
            public override bool CanSeek => false;
            public override bool CanWrite => true;
            public override long Position
            {
                get => _length;
                set => throw new NotSupportedException();
            }
            public override void Flush() => _destination?.Flush();
            public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
            public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
            public override void SetLength(long value) => throw new NotSupportedException();

            protected override void Dispose(bool disposing)
            {
                if (disposing) _sha256?.Dispose();
                base.Dispose(disposing);
            }
        }

        // The zip writer is hand-rolled because System.IO.Compression.ZipArchive cannot pad
        // local headers, and payload alignment is a container rule. Writing STORED-only zip
        // is small and fully determined: local headers + payloads, then the central
        // directory, then end-of-central-directory. Reading always goes through the BCL
        // ZipArchive — an independent implementation, which doubles as a standardness check.
        //
        // A size or offset that does not fit its 32-bit field, and an entry count that does not fit
        // its 16-bit one, go in Zip64 records (APPNOTE 4.5.3, 4.3.14, 4.3.15): the field holds its
        // all-ones marker and the Zip64 record the value. Only what does not fit moves, so an
        // archive that needs none of it is written without any Zip64 record at all.

        private const uint LocalFileHeaderSignature = 0x04034b50;
        private const uint CentralDirectoryHeaderSignature = 0x02014b50;
        private const uint EndOfCentralDirectorySignature = 0x06054b50;
        private const uint Zip64EndOfCentralDirectorySignature = 0x06064b50;
        private const uint Zip64EndOfCentralDirectoryLocatorSignature = 0x07064b50;
        private const ushort ZipVersionStored = 10;    // 1.0 — enough for STORED entries
        private const ushort ZipVersionZip64 = 45;     // 4.5 — Zip64 records
        private const ushort Zip64ExtraFieldId = 0x0001;
        private const ushort AlignmentExtraFieldId = 0xd935;    // zipalign's padding field
        private const int LocalFileHeaderSize = 30;
        private const int CentralDirectoryHeaderSize = 46;
        private const int Zip64EndOfCentralDirectorySize = 56;

        /// <summary>
        /// Test hook: the largest value the writer puts in a 32-bit zip field and the largest entry
        /// count it puts in a 16-bit one, lowered so a small archive is written with the Zip64
        /// records a large one needs. <c>null</c> — the default — leaves the fields' own limits:
        /// one below the all-ones marker each field reserves. Thread-scoped, so a value installed by
        /// one parallel test is invisible to every other thread; still reset it in a <c>finally</c>.
        /// </summary>
        [ThreadStatic]
        internal static (long Field32, int Count16)? Zip64ThresholdInjection;

        /// <summary>The largest value the writer puts in a 32-bit zip field and the largest entry
        /// count it puts in a 16-bit one; past either, it writes the Zip64 record.</summary>
        internal static (long Field32, int Count16) Zip64Limits
            => Zip64ThresholdInjection ?? (uint.MaxValue - 1L, ushort.MaxValue - 1);

        /// <summary>
        /// Writes <paramref name="entries"/> as a STORED-only zip archive. Every entry is
        /// method-0 (no compression); entries flagged <see cref="ZipEntrySpec.Align"/> get a
        /// padding extra field so their payload starts at a <see cref="DataAlignment"/>-byte
        /// offset. A size, offset or count past what its classic zip field holds is written in
        /// Zip64 records, and only then. Each entry's payload is written straight from its
        /// <see cref="EntryPayload"/>, so a produced or copied entry streams into the archive
        /// without ever being held whole.
        /// </summary>
        internal static void WriteStoredZip(Stream stream, IReadOnlyList<ZipEntrySpec> entries, DateTime timestampUtc)
        {
            if (stream is null) throw new ArgumentNullException(nameof(stream));
            if (entries is null) throw new ArgumentNullException(nameof(entries));

            (ushort dosTime, ushort dosDate) = ToDosDateTime(timestampUtc);
            var (max32, max16) = Zip64Limits;

            // Lay the archive out before writing any of it: every entry's size is known up front,
            // so each header knows whether it needs Zip64 before a byte of the archive is written.
            long offset = 0;
            var records = new List<(ZipEntrySpec Entry, byte[] NameBytes, bool Zip64Sizes, int ExtraLength,
                long HeaderOffset)>(entries.Count);
            foreach (var entry in entries)
            {
                var nameBytes = Encoding.ASCII.GetBytes(entry.Name);
                if (nameBytes.Length != entry.Name.Length)
                    throw new ArgumentException($"Zip entry name '{entry.Name}' is not ASCII.", nameof(entries));

                // A local header whose sizes do not fit carries both of them in a Zip64 extra field.
                bool zip64Sizes = entry.Payload.Length > max32;
                int extraLength = zip64Sizes ? 20 : 0;
                if (entry.Align)
                {
                    long payloadStart = offset + LocalFileHeaderSize + nameBytes.Length + extraLength;
                    int padding = (int)((DataAlignment - payloadStart % DataAlignment) % DataAlignment);
                    // The padding rides in a well-formed extra field, which needs 4 bytes for
                    // its own id+size header — bump undersized paddings by one alignment unit.
                    if (padding is > 0 and < 4) padding += DataAlignment;
                    extraLength += padding;
                }

                records.Add((entry, nameBytes, zip64Sizes, extraLength, offset));
                offset += LocalFileHeaderSize + nameBytes.Length + extraLength + entry.Payload.Length;
            }

            using var writer = new BinaryWriter(stream, Encoding.ASCII, leaveOpen: true);
            foreach (var (entry, nameBytes, zip64Sizes, extraLength, _) in records)
            {
                writer.Write(LocalFileHeaderSignature);
                writer.Write(zip64Sizes ? ZipVersionZip64 : ZipVersionStored); // version needed to extract
                writer.Write((ushort)0);                  // general purpose flags
                writer.Write((ushort)0);                  // method 0 = STORED
                writer.Write(dosTime);
                writer.Write(dosDate);
                writer.Write(entry.Payload.Crc32);
                uint size32 = zip64Sizes ? uint.MaxValue : (uint)entry.Payload.Length;
                writer.Write(size32);                     // compressed size (== uncompressed)
                writer.Write(size32);                     // uncompressed size
                writer.Write((ushort)nameBytes.Length);
                writer.Write((ushort)extraLength);
                writer.Write(nameBytes);
                int padding = extraLength;
                if (zip64Sizes)
                {
                    writer.Write(Zip64ExtraFieldId);
                    writer.Write((ushort)16);
                    writer.Write(entry.Payload.Length);   // uncompressed size
                    writer.Write(entry.Payload.Length);   // compressed size
                    padding -= 20;
                }
                if (padding > 0)
                {
                    writer.Write(AlignmentExtraFieldId);
                    writer.Write((ushort)(padding - 4));
                    writer.Write(new byte[padding - 4]);
                }
                writer.Flush();
                entry.Payload.WriteTo(PayloadDestination(stream), entry.Name);
            }

            long centralDirectoryOffset = offset;
            foreach (var (entry, nameBytes, zip64Sizes, _, headerOffset) in records)
            {
                bool zip64Offset = headerOffset > max32;
                int zip64Length = (zip64Sizes ? 16 : 0) + (zip64Offset ? 8 : 0);
                int extraLength = zip64Length > 0 ? 4 + zip64Length : 0;
                ushort version = zip64Length > 0 ? ZipVersionZip64 : ZipVersionStored;
                uint size32 = zip64Sizes ? uint.MaxValue : (uint)entry.Payload.Length;

                writer.Write(CentralDirectoryHeaderSignature);
                writer.Write(version);                    // version made by
                writer.Write(version);                    // version needed to extract
                writer.Write((ushort)0);                  // general purpose flags
                writer.Write((ushort)0);                  // method 0 = STORED
                writer.Write(dosTime);
                writer.Write(dosDate);
                writer.Write(entry.Payload.Crc32);
                writer.Write(size32);
                writer.Write(size32);
                writer.Write((ushort)nameBytes.Length);
                writer.Write((ushort)extraLength);        // the central copy carries no padding
                writer.Write((ushort)0);                  // comment length
                writer.Write((ushort)0);                  // disk number start
                writer.Write((ushort)0);                  // internal attributes
                writer.Write((uint)0);                    // external attributes
                writer.Write(zip64Offset ? uint.MaxValue : (uint)headerOffset);
                writer.Write(nameBytes);
                if (zip64Length > 0)
                {
                    writer.Write(Zip64ExtraFieldId);
                    writer.Write((ushort)zip64Length);
                    if (zip64Sizes)
                    {
                        writer.Write(entry.Payload.Length);   // uncompressed size
                        writer.Write(entry.Payload.Length);   // compressed size
                    }
                    if (zip64Offset) writer.Write(headerOffset);
                }
                offset += CentralDirectoryHeaderSize + nameBytes.Length + extraLength;
            }

            long centralDirectorySize = offset - centralDirectoryOffset;
            // The end record's count, size and offset all take their markers once any of them needs
            // the Zip64 end record, so a reader that looks for that record by any one of them finds it.
            bool zip64End = records.Count > max16 || centralDirectorySize > max32 || centralDirectoryOffset > max32;
            if (zip64End)
            {
                writer.Write(Zip64EndOfCentralDirectorySignature);
                writer.Write((ulong)(Zip64EndOfCentralDirectorySize - 12)); // size of the rest of this record
                writer.Write(ZipVersionZip64);            // version made by
                writer.Write(ZipVersionZip64);            // version needed to extract
                writer.Write(0u);                         // this disk
                writer.Write(0u);                         // disk with central directory
                writer.Write((long)records.Count);        // entries on this disk
                writer.Write((long)records.Count);        // entries in total
                writer.Write(centralDirectorySize);
                writer.Write(centralDirectoryOffset);

                writer.Write(Zip64EndOfCentralDirectoryLocatorSignature);
                writer.Write(0u);                         // disk with the Zip64 end record
                writer.Write(offset);                     // offset of the Zip64 end record
                writer.Write(1u);                         // total disks
            }

            writer.Write(EndOfCentralDirectorySignature);
            writer.Write((ushort)0);                      // this disk
            writer.Write((ushort)0);                      // disk with central directory
            ushort count16 = zip64End ? ushort.MaxValue : (ushort)records.Count;
            writer.Write(count16);
            writer.Write(count16);
            writer.Write(zip64End ? uint.MaxValue : (uint)centralDirectorySize);
            writer.Write(zip64End ? uint.MaxValue : (uint)centralDirectoryOffset);
            writer.Write((ushort)0);                      // comment length
        }

        private static (ushort Time, ushort Date) ToDosDateTime(DateTime t)
        {
            if (t.Year < 1980) t = new DateTime(1980, 1, 1, 0, 0, 0, t.Kind);
            var time = (ushort)((t.Hour << 11) | (t.Minute << 5) | (t.Second / 2));
            var date = (ushort)(((t.Year - 1980) << 9) | (t.Month << 5) | t.Day);
            return (time, date);
        }

        // The zip CRC-32 (System.IO.Hashing's, vectorized), since every byte of a checkpoint's
        // state passes through it.
        private static uint Crc32(ReadOnlySpan<byte> data) => System.IO.Hashing.Crc32.HashToUInt32(data);

        #endregion
    }
}

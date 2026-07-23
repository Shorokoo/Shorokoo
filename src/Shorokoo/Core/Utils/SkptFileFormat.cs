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

        /// <summary>Round-trips producer fields added by newer minor revisions of the format.</summary>
        [JsonExtensionData]
        public Dictionary<string, JsonElement>? AdditionalFields { get; set; }
    }

    /// <summary>One model in a .skpt manifest's model registry.</summary>
    public sealed class SkptModelEntry
    {
        /// <summary>Archive path of the serialized model definition (e.g. "models/model.srk").</summary>
        [JsonPropertyName("entry")]
        public string? Entry { get; set; }

        /// <summary>Serialization format of the entry; "srk2" is the only format written today.</summary>
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

        /// <summary>Round-trips model fields added by newer minor revisions of the format.</summary>
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

        /// <summary>Round-trips reference fields added by newer minor revisions of the format.</summary>
        [JsonExtensionData]
        public Dictionary<string, JsonElement>? AdditionalFields { get; set; }
    }

    /// <summary>A named set of tensor mappings for one model (e.g. the "default" weights).</summary>
    public sealed class SkptMappingSet
    {
        /// <summary>Per model tensor reference (parameter identifier), where its bytes live.</summary>
        [JsonPropertyName("tensors")]
        public Dictionary<string, SkptTensorRef>? Tensors { get; set; }

        /// <summary>Round-trips mapping-set fields added by newer minor revisions of the format.</summary>
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
        /// Zstd layer inside the STORED zip bytes, mirroring .srk v2's header-declared
        /// compression). Always taken from here, never inferred from the entry's extension.</summary>
        [JsonPropertyName("compression")]
        public string? Compression { get; set; }

        /// <summary>Lowercase hex SHA-256 of the entry's bytes as stored in the archive —
        /// for a compressed entry, the compressed bytes. Integrity is thus checkable
        /// without decompressing, mirroring .srk v2's payloadSha256.</summary>
        [JsonPropertyName("sha256")]
        public string? Sha256 { get; set; }

        /// <summary>Round-trips data fields added by newer minor revisions of the format.</summary>
        [JsonExtensionData]
        public Dictionary<string, JsonElement>? AdditionalFields { get; set; }
    }

    /// <summary>
    /// The training-checkpoint block of a .skpt manifest (issue #95). Present only when the file
    /// persists a <see cref="Shorokoo.TrainingCheckpoint"/>; absent for an ordinary inference
    /// checkpoint (so those stay byte-identical to files from builds predating this field). It
    /// records the checkpoint's global step and which manifest data-registry entry holds each
    /// training-state kind — the trainable weights (which also serve as the model's default weight
    /// set), the model state, and the optimizer state. A kind whose struct is empty is omitted
    /// from <see cref="Kinds"/> and carries no data entry. Like the rest of the manifest, its keys
    /// are add-only across minor revisions.
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

        /// <summary>Training-state kind name → the manifest data-registry key that stores it
        /// (e.g. <c>"trainableParams" → "trainable"</c>). A kind with an empty struct is absent.</summary>
        [JsonPropertyName("kinds")]
        public Dictionary<string, string>? Kinds { get; set; }

        /// <summary>Round-trips training fields added by newer minor revisions of the format.</summary>
        [JsonExtensionData]
        public Dictionary<string, JsonElement>? AdditionalFields { get; set; }
    }

    /// <summary>
    /// The config.json manifest of a .skpt checkpoint — the single source of wiring: archive
    /// entries never reference each other directly; every mapping (model → serialization
    /// format, model tensor references → data items, data item → storage format) lives here.
    /// Keys are add-only across minor revisions: a reader ignores unknown keys (they are
    /// preserved in the extension-data bags); removing or re-typing a key is a major-version
    /// event (a bump of <see cref="SkptVersion"/>).
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
        /// identity checks or weight binding, and its keys are add-only like the rest of the
        /// manifest. Absent (null, and omitted from the JSON) unless the saver supplied it,
        /// so a checkpoint written without metadata is byte-identical to one from a build
        /// that predates this field.</summary>
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
        /// <see cref="Shorokoo.TrainingCheckpoint"/>, recording its global step and per-kind data
        /// entries. Absent (null, omitted from the JSON) for an inference checkpoint, so those stay
        /// byte-identical to files from builds predating this field.</summary>
        [JsonPropertyName("training")]
        public SkptTrainingInfo? Training { get; set; }

        /// <summary>Round-trips manifest fields added by newer minor revisions of the format.</summary>
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

        /// <summary>Archive path of the (single, for now) model definition entry.</summary>
        public const string ModelEntryPath = "models/model.srk";

        /// <summary>Archive path of the (single, for now) weights data entry.</summary>
        public const string WeightsEntryPath = "data/weights.safetensors";

        /// <summary>Archive path of the host user-data bag (issue #101).</summary>
        public const string UserDataEntryPath = "data/user-data.json";

        /// <summary>Model serialization format name for .srk v2 payloads.</summary>
        public const string ModelFormatSrk2 = "srk2";

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
        // trainable weights — and adds per-kind data entries for the remaining training state
        // (model state, optimizer state). The global step is small metadata and rides in the
        // manifest's dedicated training block rather than a data entry. A file carrying that block
        // reconstructs a TrainingCheckpoint; one without it is an ordinary inference checkpoint.

        /// <summary>Current training-checkpoint manifest-block version (see <see cref="SkptTrainingInfo"/>).</summary>
        public const int TrainingCheckpointVersion = 1;

        /// <summary>Training-state kind name: the trainable parameters (also the model's default
        /// weight set, so the checkpoint is self-describing for inference).</summary>
        public const string TrainingKindTrainableParams = "trainableParams";

        /// <summary>Training-state kind name: the model state (running stats, etc.; empty for
        /// stateless models).</summary>
        public const string TrainingKindModelState = "modelState";

        /// <summary>Training-state kind name: the optimizer state (moment buffers, step, etc.;
        /// empty for basic SGD).</summary>
        public const string TrainingKindOptimizerState = "optimizerState";

        /// <summary>Manifest data-registry key of the trainable-weights entry. Doubles as the
        /// model's default weight set (referenced by the "default" tensor mapping) and as the
        /// trainable-params training kind.</summary>
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

        #region STORED zip writer

        /// <summary>One entry to be written into a .skpt archive.</summary>
        /// <param name="Name">Archive path (forward slashes, ASCII).</param>
        /// <param name="Data">Entry bytes; always STORED verbatim.</param>
        /// <param name="Align">Pad the local header (via a zipalign-style extra field) so the
        /// entry's payload starts at a <see cref="DataAlignment"/>-byte file offset.</param>
        internal readonly record struct ZipEntrySpec(string Name, byte[] Data, bool Align);

        // The zip writer is hand-rolled because System.IO.Compression.ZipArchive cannot pad
        // local headers, and payload alignment is a container rule. Writing STORED-only zip
        // is small and fully determined: local headers + payloads, then the central
        // directory, then end-of-central-directory. Reading always goes through the BCL
        // ZipArchive — an independent implementation, which doubles as a standardness check.

        private const uint LocalFileHeaderSignature = 0x04034b50;
        private const uint CentralDirectoryHeaderSignature = 0x02014b50;
        private const uint EndOfCentralDirectorySignature = 0x06054b50;
        private const ushort ZipVersionStored = 10;    // 1.0 — enough for STORED entries
        private const ushort AlignmentExtraFieldId = 0xd935;    // zipalign's padding field
        private const int LocalFileHeaderSize = 30;

        /// <summary>
        /// Writes <paramref name="entries"/> as a STORED-only zip archive. Every entry is
        /// method-0 (no compression); entries flagged <see cref="ZipEntrySpec.Align"/> get a
        /// padding extra field so their payload starts at a <see cref="DataAlignment"/>-byte
        /// offset. Entry sizes are capped below the Zip64 threshold — large-tensor streaming
        /// is out of scope for this format version.
        /// </summary>
        internal static void WriteStoredZip(Stream stream, IReadOnlyList<ZipEntrySpec> entries, DateTime timestampUtc)
        {
            if (stream is null) throw new ArgumentNullException(nameof(stream));
            if (entries is null) throw new ArgumentNullException(nameof(entries));

            (ushort dosTime, ushort dosDate) = ToDosDateTime(timestampUtc);

            long offset = 0;
            var records = new List<(ZipEntrySpec Entry, byte[] NameBytes, uint Crc, long HeaderOffset)>(entries.Count);

            using var writer = new BinaryWriter(stream, Encoding.ASCII, leaveOpen: true);
            foreach (var entry in entries)
            {
                var nameBytes = Encoding.ASCII.GetBytes(entry.Name);
                if (nameBytes.Length != entry.Name.Length)
                    throw new ArgumentException($"Zip entry name '{entry.Name}' is not ASCII.", nameof(entries));

                int extraLength = 0;
                if (entry.Align)
                {
                    long payloadStart = offset + LocalFileHeaderSize + nameBytes.Length;
                    extraLength = (int)((DataAlignment - payloadStart % DataAlignment) % DataAlignment);
                    // The padding rides in a well-formed extra field, which needs 4 bytes for
                    // its own id+size header — bump undersized paddings by one alignment unit.
                    if (extraLength is > 0 and < 4) extraLength += DataAlignment;
                }

                uint crc = Crc32(entry.Data);
                records.Add((entry, nameBytes, crc, offset));

                writer.Write(LocalFileHeaderSignature);
                writer.Write(ZipVersionStored);           // version needed to extract
                writer.Write((ushort)0);                  // general purpose flags
                writer.Write((ushort)0);                  // method 0 = STORED
                writer.Write(dosTime);
                writer.Write(dosDate);
                writer.Write(crc);
                writer.Write((uint)entry.Data.Length);    // compressed size (== uncompressed)
                writer.Write((uint)entry.Data.Length);    // uncompressed size
                writer.Write((ushort)nameBytes.Length);
                writer.Write((ushort)extraLength);
                writer.Write(nameBytes);
                if (extraLength > 0)
                {
                    writer.Write(AlignmentExtraFieldId);
                    writer.Write((ushort)(extraLength - 4));
                    writer.Write(new byte[extraLength - 4]);
                }
                writer.Write(entry.Data);

                offset += LocalFileHeaderSize + nameBytes.Length + extraLength + entry.Data.Length;
            }

            long centralDirectoryOffset = offset;
            // Local-header offsets and the central-directory offset are stored as 32-bit
            // fields (a header offset is always < centralDirectoryOffset, so this one check
            // bounds them all). Beyond 4 GiB the format needs Zip64, which this version does
            // not write — fail loudly here rather than silently truncate an offset and emit a
            // corrupt archive. The count field is likewise 16-bit.
            if (centralDirectoryOffset > uint.MaxValue)
                throw new NotSupportedException(
                    $"The .skpt archive would be {centralDirectoryOffset} bytes; archives at or beyond " +
                    "4 GiB need Zip64, which this .skpt version does not write.");
            if (records.Count > ushort.MaxValue)
                throw new NotSupportedException(
                    $"The .skpt archive has {records.Count} entries; more than {ushort.MaxValue} need Zip64, " +
                    "which this .skpt version does not write.");

            foreach (var (entry, nameBytes, crc, headerOffset) in records)
            {
                writer.Write(CentralDirectoryHeaderSignature);
                writer.Write(ZipVersionStored);           // version made by
                writer.Write(ZipVersionStored);           // version needed to extract
                writer.Write((ushort)0);                  // general purpose flags
                writer.Write((ushort)0);                  // method 0 = STORED
                writer.Write(dosTime);
                writer.Write(dosDate);
                writer.Write(crc);
                writer.Write((uint)entry.Data.Length);
                writer.Write((uint)entry.Data.Length);
                writer.Write((ushort)nameBytes.Length);
                writer.Write((ushort)0);                  // extra length (central copy carries none)
                writer.Write((ushort)0);                  // comment length
                writer.Write((ushort)0);                  // disk number start
                writer.Write((ushort)0);                  // internal attributes
                writer.Write((uint)0);                    // external attributes
                writer.Write((uint)headerOffset);
                writer.Write(nameBytes);
                offset += 46 + nameBytes.Length;
            }

            writer.Write(EndOfCentralDirectorySignature);
            writer.Write((ushort)0);                      // this disk
            writer.Write((ushort)0);                      // disk with central directory
            writer.Write((ushort)records.Count);
            writer.Write((ushort)records.Count);
            writer.Write((uint)(offset - centralDirectoryOffset));
            writer.Write((uint)centralDirectoryOffset);
            writer.Write((ushort)0);                      // comment length
        }

        private static (ushort Time, ushort Date) ToDosDateTime(DateTime t)
        {
            if (t.Year < 1980) t = new DateTime(1980, 1, 1, 0, 0, 0, t.Kind);
            var time = (ushort)((t.Hour << 11) | (t.Minute << 5) | (t.Second / 2));
            var date = (ushort)(((t.Year - 1980) << 9) | (t.Month << 5) | t.Day);
            return (time, date);
        }

        private static readonly uint[] Crc32Table = BuildCrc32Table();

        private static uint[] BuildCrc32Table()
        {
            var table = new uint[256];
            for (uint i = 0; i < 256; i++)
            {
                uint c = i;
                for (int k = 0; k < 8; k++)
                    c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
                table[i] = c;
            }
            return table;
        }

        private static uint Crc32(ReadOnlySpan<byte> data)
        {
            uint c = 0xFFFFFFFFu;
            foreach (byte b in data)
                c = Crc32Table[(c ^ b) & 0xFF] ^ (c >> 8);
            return c ^ 0xFFFFFFFFu;
        }

        #endregion
    }
}

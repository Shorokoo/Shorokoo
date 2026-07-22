using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Text.Json;
using Shorokoo.Core.Utils;
using ZstdSharp;

namespace Shorokoo
{
    /// <summary>What kind of Shorokoo-produced artifact a file was identified as.</summary>
    public enum ArtifactKind
    {
        /// <summary>The file matches no format Shorokoo writes; see the observations for what was seen.</summary>
        NotRecognized,

        /// <summary>A serialized computation graph: a .srk v2 container, or one of the legacy
        /// pre-container v1 layouts (bare ONNX protobuf, single- or double-Zstd).</summary>
        SrkGraph,

        /// <summary>A SafeTensors weights file (8-byte header-length prefix + JSON header + tensor data).</summary>
        SafeTensors,

        /// <summary>A SafeTensors file written by <see cref="TrainingCheckpoint.Save"/>, recognized
        /// via its <c>__shorokoo_checkpoint__</c> marker tensor.</summary>
        TrainingCheckpoint,

        /// <summary>A Zstd-compressed SafeTensors archive (.zsafetensor), as written by
        /// <see cref="CompressedFormatUtils.SaveCompressedSafeTensors"/>: a Zstd frame whose
        /// decompressed content is a SafeTensors file. Recognized by stream-decompressing the
        /// 8-byte length prefix and the JSON header only — the tensor payload is never
        /// decompressed.</summary>
        CompressedSafeTensors,

        /// <summary>A .skpt checkpoint container (written by <see cref="CheckpointBuilder.Save"/>):
        /// a STORED zip archive wired by a root config.json manifest.</summary>
        SkptCheckpoint,
    }

    /// <summary>One tensor as declared in a SafeTensors header: name, dtype, shape and payload size —
    /// metadata only, the tensor data itself is never read.</summary>
    public sealed class InspectedTensorInfo
    {
        /// <summary>Tensor name as recorded in the header. For a training checkpoint's per-section
        /// listing this is the field name with the <c>section/</c> prefix already stripped.</summary>
        public string Name { get; }

        /// <summary>SafeTensors dtype string, e.g. "F32", "I64", "BF16".</summary>
        public string DType { get; }

        /// <summary>Declared shape; empty for a rank-0 scalar.</summary>
        public long[] Shape { get; }

        /// <summary>Declared payload size in bytes (from the header's data offsets).</summary>
        public long ByteSize { get; }

        /// <summary>Declared start offset within the tensor-data area (data_offsets[0]); kept
        /// internally so the checkpoint-marker read can locate its 16 bytes.</summary>
        internal long DataStartOffset { get; }

        internal InspectedTensorInfo(string name, string dtype, long[] shape, long byteSize, long dataStartOffset)
        {
            Name = name;
            DType = dtype;
            Shape = shape;
            ByteSize = byteSize;
            DataStartOffset = dataStartOffset;
        }

        /// <summary>"name: F32[2, 3], 24 bytes".</summary>
        public override string ToString()
            => $"{Name}: {DType}[{string.Join(", ", Shape)}], {ByteSize} bytes";
    }

    /// <summary>Details of a <see cref="ArtifactKind.SrkGraph"/> artifact.</summary>
    public sealed class SrkArtifactInfo
    {
        /// <summary>The parsed v2 container header (format version, stage, compression, payload
        /// SHA-256, producer). Null for legacy v1 layouts (which carry no header) and for
        /// containers whose header could not be read — see the inspection's observations.</summary>
        public SrkHeader? Header { get; }

        /// <summary>For a legacy pre-container file, names the sniffed v1 layout; null for v2.</summary>
        public string? LegacyLayout { get; }

        /// <summary>Size of the payload as stored in the file (after compression): everything
        /// following the v2 header, or the whole file for a legacy layout. Null when unknown
        /// (unreadable header).</summary>
        public long? PayloadSizeBytes { get; }

        internal SrkArtifactInfo(SrkHeader? header, string? legacyLayout, long? payloadSizeBytes)
        {
            Header = header;
            LegacyLayout = legacyLayout;
            PayloadSizeBytes = payloadSizeBytes;
        }
    }

    /// <summary>Details of a SafeTensors artifact (also populated for training checkpoints,
    /// which are SafeTensors files).</summary>
    public sealed class SafeTensorsArtifactInfo
    {
        /// <summary>Size in bytes of the JSON header.</summary>
        public long HeaderSizeBytes { get; }

        /// <summary>Every tensor declared in the header, in header order, with full names.</summary>
        public IReadOnlyList<InspectedTensorInfo> Tensors { get; }

        /// <summary>Total declared tensor payload size: the largest declared end offset.</summary>
        public long TotalTensorBytes { get; }

        /// <summary>The header's <c>__metadata__</c> entries as raw JSON value text, or null when absent.</summary>
        public IReadOnlyDictionary<string, string>? GlobalMetadata { get; }

        internal SafeTensorsArtifactInfo(
            long headerSizeBytes, IReadOnlyList<InspectedTensorInfo> tensors,
            long totalTensorBytes, IReadOnlyDictionary<string, string>? globalMetadata)
        {
            HeaderSizeBytes = headerSizeBytes;
            Tensors = tensors;
            TotalTensorBytes = totalTensorBytes;
            GlobalMetadata = globalMetadata;
        }
    }

    /// <summary>Details of a <see cref="ArtifactKind.TrainingCheckpoint"/> artifact, read from the
    /// checkpoint marker and the SafeTensors header.</summary>
    public sealed class TrainingCheckpointArtifactInfo
    {
        /// <summary>Checkpoint format version from the marker tensor
        /// (<see cref="TrainingCheckpoint"/> writes version 1).</summary>
        public long FormatVersion { get; }

        /// <summary>The 0-based global training step the checkpoint was saved at.</summary>
        public long Step { get; }

        /// <summary>Per-section tensor listing, keyed "trainable" / "model_state" / "opt_state"
        /// (always all three, possibly empty); tensor names have the section prefix stripped.</summary>
        public IReadOnlyDictionary<string, IReadOnlyList<InspectedTensorInfo>> Sections { get; }

        internal TrainingCheckpointArtifactInfo(
            long formatVersion, long step,
            IReadOnlyDictionary<string, IReadOnlyList<InspectedTensorInfo>> sections)
        {
            FormatVersion = formatVersion;
            Step = step;
            Sections = sections;
        }
    }

    /// <summary>One model in an inspected .skpt manifest's model registry — manifest metadata
    /// only, the model definition entry itself is never read.</summary>
    public sealed class SkptModelSummary
    {
        /// <summary>The model's key in the manifest's model registry (e.g. "model").</summary>
        public string Key { get; }

        /// <summary>Archive path of the serialized model definition (e.g. "models/model.srk"),
        /// or null when the manifest names none.</summary>
        public string? EntryPath { get; }

        /// <summary>Serialization format of the entry ("srk2" for files written today).</summary>
        public string? Format { get; }

        /// <summary>Lifecycle stage of the serialized graph, in .srk stage-name form
        /// ("concrete-model" for files written today).</summary>
        public string? Stage { get; }

        /// <summary>The recorded SHA-256 of the entry's bytes — the model's graph hash in this
        /// format version. Reported exactly as recorded, never verified by Inspect.</summary>
        public string? GraphHash { get; }

        internal SkptModelSummary(string key, string? entryPath, string? format, string? stage, string? graphHash)
        {
            Key = key;
            EntryPath = entryPath;
            Format = format;
            Stage = stage;
            GraphHash = graphHash;
        }

        /// <summary>"model: models/model.srk (srk2, stage concrete-model), graph hash 6824…".</summary>
        public override string ToString()
            => $"{Key}: {EntryPath ?? "<no entry>"} ({Format ?? "<unrecorded>"}, " +
               $"stage {Stage ?? "<unrecorded>"}), graph hash {GraphHash ?? "<unrecorded>"}";
    }

    /// <summary>One data entry in an inspected .skpt manifest's data registry — manifest and
    /// central-directory metadata only, the payload itself is never read.</summary>
    public sealed class SkptDataSummary
    {
        /// <summary>The entry's key in the manifest's data registry (e.g. "weights").</summary>
        public string Key { get; }

        /// <summary>Archive path of the data payload (e.g. "data/weights.safetensors"),
        /// or null when the manifest names none.</summary>
        public string? EntryPath { get; }

        /// <summary>Storage format ("safetensors" for files written today).</summary>
        public string? Format { get; }

        /// <summary>Compression of the entry's bytes ("none" for files written today).</summary>
        public string? Compression { get; }

        /// <summary>Uncompressed size the zip central directory declares for the entry, or null
        /// when the manifest names no entry or the archive lacks it. Declared only — no payload
        /// bytes are read to confirm it.</summary>
        public long? DeclaredSizeBytes { get; }

        /// <summary>The recorded SHA-256 of the entry's bytes — reported exactly as recorded,
        /// never verified by Inspect (a full <see cref="Persistence.Load"/> verifies it).</summary>
        public string? Sha256 { get; }

        internal SkptDataSummary(
            string key, string? entryPath, string? format, string? compression,
            long? declaredSizeBytes, string? sha256)
        {
            Key = key;
            EntryPath = entryPath;
            Format = format;
            Compression = compression;
            DeclaredSizeBytes = declaredSizeBytes;
            Sha256 = sha256;
        }

        /// <summary>"weights: data/weights.safetensors (safetensors, compression none),
        /// 1184 bytes declared, sha256 7344… (unverified)".</summary>
        public override string ToString()
            => $"{Key}: {EntryPath ?? "<no entry>"} ({Format ?? "<unrecorded>"}, " +
               $"compression {Compression ?? "<unrecorded>"}), " +
               (DeclaredSizeBytes is { } size ? $"{size} bytes declared, " : "size unknown, ") +
               $"sha256 {Sha256 ?? "<unrecorded>"} (unverified)";
    }

    /// <summary>Details of a <see cref="ArtifactKind.SkptCheckpoint"/> artifact, read from the
    /// zip central directory and the config.json manifest alone — no other entry is opened,
    /// and no tensor payload is touched.</summary>
    public sealed class SkptArtifactInfo
    {
        /// <summary>Format identifier from the manifest; always "skpt" (recognition requires it).</summary>
        public string? FormatName { get; }

        /// <summary>Manifest major version (skptVersion); 0 when the field is missing —
        /// see the observations.</summary>
        public int SkptVersion { get; }

        /// <summary>Version of the Shorokoo framework that wrote the file, or null when unrecorded.</summary>
        public string? Producer { get; }

        /// <summary>Creation time as recorded (ISO-8601 UTC), or null when unrecorded.</summary>
        public string? CreatedUtc { get; }

        /// <summary>The model registry, in manifest order.</summary>
        public IReadOnlyList<SkptModelSummary> Models { get; }

        /// <summary>The data registry, in manifest order.</summary>
        public IReadOnlyList<SkptDataSummary> DataEntries { get; }

        /// <summary>Distinct tensor-mapping-set names across all models (e.g. "default"),
        /// in manifest order.</summary>
        public IReadOnlyList<string> MappingSetNames { get; }

        internal SkptArtifactInfo(
            string? formatName, int skptVersion, string? producer, string? createdUtc,
            IReadOnlyList<SkptModelSummary> models, IReadOnlyList<SkptDataSummary> dataEntries,
            IReadOnlyList<string> mappingSetNames)
        {
            FormatName = formatName;
            SkptVersion = skptVersion;
            Producer = producer;
            CreatedUtc = createdUtc;
            Models = models;
            DataEntries = dataEntries;
            MappingSetNames = mappingSetNames;
        }
    }

    /// <summary>
    /// Result of <see cref="Persistence.Inspect"/>: what the file is (<see cref="Kind"/>), the
    /// per-kind details (exactly the properties matching the kind are non-null), and cheap
    /// sanity <see cref="Observations"/> visible from the header alone.
    /// </summary>
    public sealed class ArtifactInspection
    {
        /// <summary>The inspected file's path, as passed to <see cref="Persistence.Inspect"/>.</summary>
        public string FilePath { get; }

        /// <summary>Total size of the file in bytes.</summary>
        public long FileSizeBytes { get; }

        /// <summary>What the file was identified as.</summary>
        public ArtifactKind Kind { get; }

        /// <summary>Graph-container details; non-null iff <see cref="Kind"/> is <see cref="ArtifactKind.SrkGraph"/>.</summary>
        public SrkArtifactInfo? Srk { get; }

        /// <summary>SafeTensors details; non-null for <see cref="ArtifactKind.SafeTensors"/>,
        /// <see cref="ArtifactKind.TrainingCheckpoint"/> (a checkpoint is a SafeTensors file), and
        /// <see cref="ArtifactKind.CompressedSafeTensors"/> (read through the compression layer
        /// from the decompressed header; sizes describe the decompressed content).</summary>
        public SafeTensorsArtifactInfo? SafeTensors { get; }

        /// <summary>Training-checkpoint details; non-null iff <see cref="Kind"/> is <see cref="ArtifactKind.TrainingCheckpoint"/>.</summary>
        public TrainingCheckpointArtifactInfo? TrainingCheckpoint { get; }

        /// <summary>.skpt container details; non-null iff <see cref="Kind"/> is
        /// <see cref="ArtifactKind.SkptCheckpoint"/>.</summary>
        public SkptArtifactInfo? Skpt { get; }

        /// <summary>Cheap sanity observations visible from the header alone — e.g. declared tensor
        /// extents pointing past the end of the file, or an unreadable container header. Empty
        /// for a clean file.</summary>
        public IReadOnlyList<string> Observations { get; }

        internal ArtifactInspection(
            string filePath, long fileSizeBytes, ArtifactKind kind,
            SrkArtifactInfo? srk, SafeTensorsArtifactInfo? safeTensors,
            TrainingCheckpointArtifactInfo? trainingCheckpoint, IReadOnlyList<string> observations)
        {
            FilePath = filePath;
            FileSizeBytes = fileSizeBytes;
            Kind = kind;
            Srk = srk;
            SafeTensors = safeTensors;
            TrainingCheckpoint = trainingCheckpoint;
            Observations = observations;
        }

        /// <summary>Constructor for .skpt results: exactly the <see cref="Skpt"/> details are
        /// populated (a separate overload so .skpt support stays additive to the general one).</summary>
        internal ArtifactInspection(
            string filePath, long fileSizeBytes, ArtifactKind kind,
            SkptArtifactInfo skpt, IReadOnlyList<string> observations)
        {
            FilePath = filePath;
            FileSizeBytes = fileSizeBytes;
            Kind = kind;
            Skpt = skpt;
            Observations = observations;
        }

        /// <summary>The most tensors a section/tensor listing prints before eliding the rest.</summary>
        private const int MaxListedTensors = 50;

        /// <summary>Human-readable multi-line summary of the whole inspection.</summary>
        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.Append(FilePath).Append(": ").AppendLine(Kind switch
            {
                ArtifactKind.SrkGraph when Srk?.Header is not null =>
                    $"Shorokoo graph (.srk v{Srk.Header.SrkVersion} container)",
                ArtifactKind.SrkGraph when Srk?.LegacyLayout is not null =>
                    $"Shorokoo graph (legacy pre-container .srk, {Srk.LegacyLayout})",
                ArtifactKind.SrkGraph => "Shorokoo graph (.srk container, header not readable)",
                ArtifactKind.SafeTensors => "SafeTensors weights file",
                ArtifactKind.TrainingCheckpoint => "Shorokoo training checkpoint (SafeTensors)",
                ArtifactKind.CompressedSafeTensors => "Zstd-compressed SafeTensors archive (.zsafetensor)",
                ArtifactKind.SkptCheckpoint => "Shorokoo checkpoint container (.skpt)",
                _ => "not recognized as a Shorokoo artifact",
            });
            sb.Append("  file size: ").Append(FileSizeBytes).AppendLine(" bytes");

            if (Srk is not null)
            {
                if (Srk.Header is { } h)
                {
                    sb.Append("  stage: ").AppendLine(h.Stage ?? "<unrecorded>");
                    sb.Append("  compression: ").AppendLine(h.Compression ?? "<unrecorded>");
                    // A parsed header always comes with a known payload size.
                    sb.Append("  payload: ").Append(Srk.PayloadSizeBytes!.Value).Append(" bytes as stored, sha256 ")
                      .AppendLine(h.PayloadSha256 ?? "<unrecorded>");
                    if (h.Producer is { } p)
                    {
                        sb.Append("  producer: Shorokoo ").Append(p.Shorokoo ?? "<unknown>")
                          .Append(", ONNX IR ").Append(p.IrVersion);
                        if (p.Opsets is { Count: > 0 })
                            sb.Append(", opsets ").Append(string.Join(", ", p.Opsets.Select(
                                kv => $"{(kv.Key.Length == 0 ? "ai.onnx" : kv.Key)}:{kv.Value}")));
                        sb.AppendLine();
                    }
                }
                else if (Srk.LegacyLayout is not null)
                {
                    sb.AppendLine("  no header: legacy layouts record no stage/compression/producer metadata");
                }
            }

            if (Skpt is { } skpt)
            {
                sb.Append("  skpt version: ").Append(skpt.SkptVersion)
                  .Append(", created: ").Append(skpt.CreatedUtc ?? "<unrecorded>")
                  .Append(", producer: Shorokoo ").AppendLine(skpt.Producer ?? "<unknown>");

                sb.Append("  models (").Append(skpt.Models.Count).AppendLine("):");
                AppendItemList(sb, skpt.Models, indent: "    ");
                sb.Append("  data entries (").Append(skpt.DataEntries.Count)
                  .AppendLine("; sha256 recorded, not verified):");
                AppendItemList(sb, skpt.DataEntries, indent: "    ");
                sb.Append("  mapping sets: ").AppendLine(skpt.MappingSetNames.Count == 0
                    ? "<none>"
                    : string.Join(", ", skpt.MappingSetNames.Take(MaxListedTensors))
                      + (skpt.MappingSetNames.Count > MaxListedTensors
                          ? $", … and {skpt.MappingSetNames.Count - MaxListedTensors} more"
                          : string.Empty));
            }

            if (TrainingCheckpoint is { } ckpt)
            {
                sb.Append("  checkpoint format version: ").Append(ckpt.FormatVersion)
                  .Append(", global step: ").Append(ckpt.Step).AppendLine();
                foreach (var (section, tensors) in ckpt.Sections)
                {
                    sb.Append("  ").Append(section).Append(" (").Append(tensors.Count)
                      .AppendLine(tensors.Count == 1 ? " tensor):" : " tensors):");
                    AppendTensorList(sb, tensors, indent: "    ");
                }
            }
            else if (SafeTensors is { } st)
            {
                sb.Append("  tensors (").Append(st.Tensors.Count).Append("), total data ")
                  .Append(st.TotalTensorBytes)
                  .AppendLine(Kind == ArtifactKind.CompressedSafeTensors ? " bytes decompressed:" : " bytes:");
                AppendTensorList(sb, st.Tensors, indent: "    ");
            }

            foreach (var note in Observations)
                sb.Append("  note: ").AppendLine(note);

            return sb.ToString().TrimEnd('\r', '\n');
        }

        private static void AppendTensorList(
            StringBuilder sb, IReadOnlyList<InspectedTensorInfo> tensors, string indent)
        {
            foreach (var t in tensors.Take(MaxListedTensors))
                sb.Append(indent).AppendLine(t.ToString());
            if (tensors.Count > MaxListedTensors)
                sb.Append(indent).Append("… and ").Append(tensors.Count - MaxListedTensors)
                  .AppendLine(" more");
        }

        /// <summary>Prints each item's ToString on its own indented line, eliding past
        /// <see cref="MaxListedTensors"/> — the same shape as the tensor listings.</summary>
        private static void AppendItemList<T>(StringBuilder sb, IReadOnlyList<T> items, string indent)
        {
            foreach (var item in items.Take(MaxListedTensors))
                sb.Append(indent).AppendLine(item!.ToString());
            if (items.Count > MaxListedTensors)
                sb.Append(indent).Append("… and ").Append(items.Count - MaxListedTensors)
                  .AppendLine(" more");
        }
    }

    /// <summary>
    /// Read-only inspection of Shorokoo-produced files: <see cref="Inspect"/> identifies what a
    /// file is and summarizes its contents from headers/prefixes only — tensor payloads are never
    /// loaded, so inspecting a multi-GB file is fast and cheap. (Partial: the .skpt save/load
    /// entry points live in Persistence.cs — one <c>Persistence</c> facade, three concerns.)
    /// </summary>
    public static partial class Persistence
    {
        // SafeTensors caps its JSON header at 100 MB; anything declaring more is not a
        // SafeTensors file (and bounds our header read).
        private const long MaxSafeTensorsHeaderBytes = 100_000_000;

        /// <summary>Shared observation for every sniffed legacy v1 layout, so the
        /// Observations collection reads the same however the layout was detected.</summary>
        private const string LegacyNoHeaderObservation =
            "legacy pre-container .srk layouts record no stage/compression/producer " +
            "header; loading the file is the only way to learn more.";

        /// <summary>
        /// Identifies <paramref name="filePath"/> and summarizes its contents without loading
        /// tensor data. Recognized formats: .srk graph files (the v2 container by its header;
        /// the legacy v1 layouts by content sniffing), SafeTensors weights files (header only),
        /// Zstd-compressed SafeTensors archives (.zsafetensor; the length prefix and JSON header
        /// are stream-decompressed, the tensor payload never), training checkpoints written
        /// by <see cref="TrainingCheckpoint.Save"/> (via the
        /// checkpoint marker; the marker's 16 bytes are the only payload bytes ever read),
        /// and .skpt checkpoint containers written by <see cref="CheckpointBuilder.Save"/>
        /// (a zip archive with a root config.json manifest — only the zip central directory
        /// and the manifest entry are read; recorded sha256s are reported, never verified).
        /// Anything else — including corrupt or truncated headers — yields a structured
        /// <see cref="ArtifactKind.NotRecognized"/> (or partially-detailed) result with
        /// observations rather than an exception: content problems never throw. A missing
        /// file (<see cref="FileNotFoundException"/>) and I/O errors (permissions, disk)
        /// do.
        /// </summary>
        public static ArtifactInspection Inspect(string filePath)
        {
            if (string.IsNullOrWhiteSpace(filePath))
                throw new ArgumentException("Path cannot be null or empty.", nameof(filePath));
            if (!File.Exists(filePath))
                throw new FileNotFoundException($"File not found: {filePath}", filePath);

            using var stream = File.OpenRead(filePath);
            long fileLen = stream.Length;
            var observations = new List<string>();

            byte[] prefix = new byte[8];
            int prefixRead = stream.ReadAtLeast(prefix, prefix.Length, throwOnEndOfStream: false);

            if (fileLen == 0)
                return NotRecognized(filePath, fileLen, observations, "the file is empty.");

            if (prefixRead >= 3 && prefix[0] == (byte)'S' && prefix[1] == (byte)'R' && prefix[2] == (byte)'K')
                return InspectSrkContainer(filePath, stream, fileLen, prefix, prefixRead, observations);

            if (prefixRead >= 4 && LooksLikeZipArchive(prefix))
                return InspectZipArchive(filePath, stream, fileLen, observations);

            if (prefixRead >= 4 && LooksLikeZstd(prefix))
                return InspectZstdFrame(filePath, stream, fileLen, observations);

            if (prefixRead == 8 && TryInspectSafeTensors(filePath, stream, fileLen, prefix, observations) is { } st)
                return st;

            // The bare-protobuf legacy v1 .srk layout — the encoding is a plain ONNX
            // model's too; content cannot tell the two apart.
            if (LooksLikeOnnxModelProto(prefix.AsSpan(0, prefixRead)))
            {
                observations.Add(LegacyNoHeaderObservation);
                observations.Add("a bare serialized ONNX model is indistinguishable from a plain " +
                    ".onnx file by content; if this is a .onnx export, use standard ONNX tooling instead.");
                return new ArtifactInspection(filePath, fileLen, ArtifactKind.SrkGraph,
                    new SrkArtifactInfo(header: null, legacyLayout: "bare ONNX protobuf", fileLen),
                    safeTensors: null, trainingCheckpoint: null, observations);
            }

            return NotRecognized(filePath, fileLen, observations,
                $"the file starts with bytes {Convert.ToHexString(prefix.AsSpan(0, Math.Min(prefixRead, 4)))}, " +
                "matching no format Shorokoo writes (.srk container, legacy .srk layout, " +
                "SafeTensors, or .skpt zip container).");
        }

        private static ArtifactInspection NotRecognized(
            string filePath, long fileLen, List<string> observations, string reason)
        {
            observations.Add(reason);
            return new ArtifactInspection(filePath, fileLen, ArtifactKind.NotRecognized,
                srk: null, safeTensors: null, trainingCheckpoint: null, observations);
        }

        /// <summary>Zstd frame magic: 28 B5 2F FD.</summary>
        private static bool LooksLikeZstd(ReadOnlySpan<byte> data)
            => data.Length >= 4 && data[0] == 0x28 && data[1] == 0xB5 && data[2] == 0x2F && data[3] == 0xFD;

        /// <summary>
        /// A serialized ONNX ModelProto opens with field 1 (ir_version) as a small varint —
        /// 0x08 then a single non-zero byte — followed by the next field's protobuf tag.
        /// Requiring that follow-on byte to be a valid tag (field number ≥ 1, wire type
        /// 0/1/2/5) keeps near-misses from matching — e.g. a Zstd-compressed SafeTensors
        /// file whose 8-byte header-length prefix happens to start 08 xx 00.
        /// </summary>
        private static bool LooksLikeOnnxModelProto(ReadOnlySpan<byte> data)
        {
            if (data.Length < 3 || data[0] != 0x08) return false;
            if (data[1] == 0 || (data[1] & 0x80) != 0) return false;   // ir_version: single-byte varint ≥ 1
            int fieldNumber = data[2] >> 3;
            int wireType = data[2] & 0x7;
            return fieldNumber >= 1 && wireType is 0 or 1 or 2 or 5;
        }

        // ---- .srk container ("SRK" prefix present) ----

        private static ArtifactInspection InspectSrkContainer(
            string filePath, FileStream stream, long fileLen,
            byte[] prefix, int prefixRead, List<string> observations)
        {
            // Reuse SrkFileFormat's header parser on exactly magic + length + header bytes, the
            // same way TryReadHeaderFromFile does — but downgrade its exceptions (unsupported
            // major version, truncation, malformed JSON) to observations: Inspect never throws
            // on content.
            SrkHeader? header = null;
            long? payloadSize = null;
            try
            {
                if (prefixRead < 6)
                {
                    // Always throws for an "SRK"-prefixed fragment this short (truncated
                    // container or unreadable version byte) — the catch turns it into an
                    // observation.
                    SrkFileFormat.TryReadHeader(prefix.AsSpan(0, prefixRead).ToArray(), filePath);
                }
                else
                {
                    int headerLen = prefix[4] | (prefix[5] << 8);
                    var buf = new byte[6 + headerLen];
                    prefix.AsSpan(0, 6).CopyTo(buf);
                    stream.Position = 6;
                    int bodyRead = stream.ReadAtLeast(buf.AsSpan(6), headerLen, throwOnEndOfStream: false);
                    header = SrkFileFormat.TryReadHeader(
                        bodyRead < headerLen ? buf.AsSpan(0, 6 + bodyRead).ToArray() : buf, filePath);
                    payloadSize = fileLen - 6 - headerLen;

                    if (header is not null)
                    {
                        if (payloadSize <= 0)
                            observations.Add("the container ends at its header — the graph payload " +
                                "is empty (truncated file?).");
                        if (header.Compression is not ("none" or "zstd"))
                            observations.Add($"the header declares unknown compression '{header.Compression}' " +
                                "(likely written by a newer Shorokoo version).");
                        if (header.Stage is not null && header.TryGetStage() is null)
                            observations.Add($"the header records the unknown stage '{header.Stage}' " +
                                "(likely written by a newer Shorokoo version).");
                    }
                }
            }
            catch (InvalidDataException e)
            {
                header = null;
                payloadSize = null;
                observations.Add($"the container header is not readable: {e.Message}");
            }

            return new ArtifactInspection(filePath, fileLen, ArtifactKind.SrkGraph,
                new SrkArtifactInfo(header, legacyLayout: null, payloadSize),
                safeTensors: null, trainingCheckpoint: null, observations);
        }

        // ---- Zip archive (.skpt container) ----

        /// <summary>Caps the config.json read of a .skpt candidate (the same bound as the
        /// SafeTensors header): a manifest declaring more is not something Shorokoo wrote,
        /// and a hostile declared size cannot force a huge allocation.</summary>
        private const long MaxSkptManifestBytes = MaxSafeTensorsHeaderBytes;

        /// <summary>The most per-entry archive observations (STORED violations, unreferenced
        /// entries) reported per category before eliding the rest — a hostile central
        /// directory can declare hundreds of thousands of entries.</summary>
        private const int MaxZipEntryObservations = 20;

        /// <summary>Zip signatures a .skpt candidate can open with: a local file header
        /// (PK\3\4) or, for an entry-less archive, the end-of-central-directory record (PK\5\6).</summary>
        private static bool LooksLikeZipArchive(ReadOnlySpan<byte> data)
            => data.Length >= 4 && data[0] == (byte)'P' && data[1] == (byte)'K'
               && ((data[2] == 3 && data[3] == 4) || (data[2] == 5 && data[3] == 6));

        private static ArtifactInspection InspectZipArchive(
            string filePath, FileStream stream, long fileLen, List<string> observations)
        {
            // Bounded read by construction: opening a ZipArchive in Read mode parses only the
            // end-of-central-directory record and the central directory, and the only entry
            // ever opened is config.json (capped below). Tensor payloads are never touched —
            // which is also why every recorded sha256 is reported unverified.
            List<(string Name, long Length, long CompressedLength)> zipEntries;
            byte[] configBytes;
            int configCount = 0;
            try
            {
                stream.Position = 0;
                using var archive = new ZipArchive(stream, ZipArchiveMode.Read, leaveOpen: true);
                ZipArchiveEntry? configEntry = null;
                zipEntries = new List<(string, long, long)>(archive.Entries.Count);
                foreach (var entry in archive.Entries)
                {
                    zipEntries.Add((entry.FullName, entry.Length, entry.CompressedLength));
                    if (entry.FullName == SkptFileFormat.ConfigEntryName)
                    {
                        configCount++;
                        configEntry ??= entry;
                    }
                }

                if (configEntry is null)
                    return NotRecognized(filePath, fileLen, observations,
                        $"a zip archive with no root '{SkptFileFormat.ConfigEntryName}' manifest — " +
                        "not a .skpt checkpoint (and no other Shorokoo format is zip-based).");
                // A negative declared size (a hostile zip64 length read back as a negative
                // long) must be rejected here too, or the allocation below would throw.
                if (configEntry.Length is < 0 or > MaxSkptManifestBytes)
                    return NotRecognized(filePath, fileLen, observations,
                        $"a zip archive whose '{SkptFileFormat.ConfigEntryName}' declares " +
                        $"{configEntry.Length} bytes — not a readable .skpt manifest.");

                // A truncated entry yields fewer bytes than declared; the manifest parse
                // below then reports it as unreadable rather than anything throwing here.
                var buf = new byte[(int)configEntry.Length];
                using var entryStream = configEntry.Open();
                int read = entryStream.ReadAtLeast(buf, buf.Length, throwOnEndOfStream: false);
                configBytes = read == buf.Length ? buf : buf[..read];
            }
            catch (InvalidDataException e)
            {
                return NotRecognized(filePath, fileLen, observations,
                    "a zip-signature file whose archive structure is not readable — truncated " +
                    $"or corrupt ({e.Message}).");
            }

            SkptManifest manifest;
            try
            {
                manifest = SkptFileFormat.ParseManifest(configBytes, filePath);
            }
            catch (InvalidDataException e)
            {
                return NotRecognized(filePath, fileLen, observations,
                    $"a zip archive whose '{SkptFileFormat.ConfigEntryName}' is not a readable " +
                    $"manifest ({e.Message}).");
            }

            if (manifest.Format != SkptFileFormat.FormatName)
                return NotRecognized(filePath, fileLen, observations,
                    $"a zip archive with a '{SkptFileFormat.ConfigEntryName}' entry, but it declares " +
                    $"format '{manifest.Format ?? "<none>"}' rather than '{SkptFileFormat.FormatName}' — " +
                    "a zip, not a .skpt checkpoint.");

            return new ArtifactInspection(filePath, fileLen, ArtifactKind.SkptCheckpoint,
                SummarizeSkptManifest(manifest, zipEntries, configCount, observations), observations);
        }

        /// <summary>
        /// Builds the .skpt summary from the parsed manifest plus the zip central-directory
        /// listing, adding the cheap sanity observations along the way: manifest/archive
        /// mismatches in both directions, STORED-expectation violations, unknown manifest
        /// keys, empty trees, and version/field anomalies. sha256 values are carried exactly
        /// as recorded — verifying them needs full entry reads, which is
        /// <see cref="Load"/>'s job, not Inspect's.
        /// </summary>
        private static SkptArtifactInfo SummarizeSkptManifest(
            SkptManifest manifest,
            List<(string Name, long Length, long CompressedLength)> zipEntries,
            int configCount, List<string> observations)
        {
            if (manifest.SkptVersion == 0)
                observations.Add("required manifest field 'skptVersion' is missing or zero.");
            else if (manifest.SkptVersion != SkptFileFormat.CurrentVersion)
                observations.Add($".skpt version {manifest.SkptVersion} is not the version this " +
                    $"build reads ({SkptFileFormat.CurrentVersion}); Persistence.Load would refuse " +
                    "the file" + (manifest.SkptVersion > SkptFileFormat.CurrentVersion
                        ? " (likely written by a newer Shorokoo version)." : "."));

            if (configCount > 1)
                observations.Add($"the archive contains {configCount} entries named " +
                    $"'{SkptFileFormat.ConfigEntryName}'; only the first was read.");

            // Declared (uncompressed) sizes by entry name, from the central directory alone.
            var declaredSizes = new Dictionary<string, long>(StringComparer.Ordinal);
            foreach (var (name, length, _) in zipEntries)
                declaredSizes.TryAdd(name, length);
            var referenced = new HashSet<string>(StringComparer.Ordinal)
                { SkptFileFormat.ConfigEntryName };

            var unknownKeys = new List<string>();
            void CollectUnknown(string where, Dictionary<string, JsonElement>? bag)
            {
                if (bag is null) return;
                foreach (var key in bag.Keys)
                    unknownKeys.Add($"'{key}' ({where})");
            }
            CollectUnknown("top level", manifest.AdditionalFields);
            CollectUnknown("producer", manifest.Producer?.AdditionalFields);

            var models = new List<SkptModelSummary>();
            foreach (var (key, m) in manifest.Models ?? new())
            {
                models.Add(new SkptModelSummary(key, m?.Entry, m?.Format, m?.Stage, m?.Sha256));
                CollectUnknown($"model '{key}'", m?.AdditionalFields);
                if (string.IsNullOrEmpty(m?.Entry))
                    observations.Add($"the manifest's model '{key}' names no archive entry.");
                else if (!declaredSizes.ContainsKey(m.Entry))
                    observations.Add($"the manifest's model '{key}' references entry '{m.Entry}', " +
                        "but the archive has no such entry.");
                else
                    referenced.Add(m.Entry);
                if (m is not null && m.Format != SkptFileFormat.ModelFormatSrk2)
                    observations.Add($"model '{key}' uses the unknown serialization format " +
                        $"'{m.Format ?? "<none>"}' (likely written by a newer Shorokoo version).");
                if (m?.Stage is not null && SrkFileFormat.TryParseStageName(m.Stage) is null)
                    observations.Add($"model '{key}' records the unknown stage '{m.Stage}' " +
                        "(likely written by a newer Shorokoo version).");
                if (string.IsNullOrEmpty(m?.Sha256))
                    observations.Add($"the manifest records no sha256 for model '{key}' — " +
                        "required by .skpt version 1.");
            }

            var dataEntries = new List<SkptDataSummary>();
            foreach (var (key, d) in manifest.Data ?? new())
            {
                long? declaredSize = d?.Entry is { } entryPath
                    && declaredSizes.TryGetValue(entryPath, out var size) ? size : null;
                dataEntries.Add(new SkptDataSummary(
                    key, d?.Entry, d?.Format, d?.Compression, declaredSize, d?.Sha256));
                CollectUnknown($"data entry '{key}'", d?.AdditionalFields);
                if (string.IsNullOrEmpty(d?.Entry))
                    observations.Add($"the manifest's data entry '{key}' names no archive entry.");
                else if (!declaredSizes.ContainsKey(d.Entry))
                    observations.Add($"the manifest's data entry '{key}' references entry " +
                        $"'{d.Entry}', but the archive has no such entry.");
                else
                    referenced.Add(d.Entry);
                if (d is not null && d.Format != SkptFileFormat.DataFormatSafeTensors)
                    observations.Add($"data entry '{key}' uses the unknown storage format " +
                        $"'{d.Format ?? "<none>"}' (likely written by a newer Shorokoo version).");
                if (d?.Compression is not null && d.Compression != SkptFileFormat.CompressionNone)
                    observations.Add($"data entry '{key}' declares the unknown compression " +
                        $"'{d.Compression}' (likely written by a newer Shorokoo version).");
                if (string.IsNullOrEmpty(d?.Sha256))
                    observations.Add($"the manifest records no sha256 for data entry '{key}' — " +
                        "required by .skpt version 1.");
            }

            var mappingSetNames = new List<string>();
            foreach (var (modelKey, sets) in manifest.TensorMappings ?? new())
            {
                if (manifest.Models is null || !manifest.Models.ContainsKey(modelKey))
                    observations.Add($"tensor mappings cover model '{modelKey}', which the model " +
                        "registry does not declare.");
                foreach (var (setName, set) in sets ?? new())
                {
                    if (!mappingSetNames.Contains(setName))
                        mappingSetNames.Add(setName);
                    CollectUnknown($"mapping set '{modelKey}/{setName}'", set?.AdditionalFields);
                }
            }

            if (models.Count == 0)
                observations.Add("the manifest declares no models.");
            if (dataEntries.Count == 0)
                observations.Add("the manifest declares no data entries.");
            if (mappingSetNames.Count == 0)
                observations.Add("the manifest declares no tensor mapping sets.");

            if (unknownKeys.Count > 0)
                observations.Add("the manifest carries unknown key(s) — tolerated, keys are " +
                    "add-only across minor revisions: " + string.Join(", ", unknownKeys.Take(8)) +
                    (unknownKeys.Count > 8 ? $", … and {unknownKeys.Count - 8} more." : "."));

            int storedViolations = 0, unreferenced = 0;
            foreach (var (name, length, compressedLength) in zipEntries)
            {
                // Method 0 (STORED) always stores exactly the declared size, so a size
                // mismatch proves a compressed entry. (An entry that compressed to exactly
                // its own size would slip through — the check stays central-directory-cheap.)
                if (compressedLength != length && ++storedViolations <= MaxZipEntryObservations)
                    observations.Add($"archive entry '{name}' is compressed ({compressedLength} " +
                        $"bytes stored for {length} declared) — .skpt entries are expected " +
                        "STORED (uncompressed).");
                if (!referenced.Contains(name) && !name.EndsWith("/", StringComparison.Ordinal)
                    && ++unreferenced <= MaxZipEntryObservations)
                    observations.Add($"archive entry '{name}' is not referenced by the manifest.");
            }
            if (storedViolations > MaxZipEntryObservations)
                observations.Add($"… and {storedViolations - MaxZipEntryObservations} more " +
                    "compressed entries.");
            if (unreferenced > MaxZipEntryObservations)
                observations.Add($"… and {unreferenced - MaxZipEntryObservations} more entries " +
                    "not referenced by the manifest.");

            return new SkptArtifactInfo(
                manifest.Format, manifest.SkptVersion, manifest.Producer?.Shorokoo,
                manifest.CreatedUtc, models, dataEntries, mappingSetNames);
        }

        // ---- Zstd frame (legacy v1 compressed layouts) ----

        private static ArtifactInspection InspectZstdFrame(
            string filePath, FileStream stream, long fileLen, List<string> observations)
        {
            // Same sniff as the v1 shim, but bounded: stream-decompress only the first 8 inner
            // bytes to classify the frame's content instead of unwrapping the whole payload.
            byte[] inner = new byte[8];
            int innerRead;
            try
            {
                stream.Position = 0;
                using var ds = new DecompressionStream(stream);
                innerRead = ds.ReadAtLeast(inner, inner.Length, throwOnEndOfStream: false);

                // .zsafetensor probe — deliberately BEFORE the legacy-.srk classification:
                // it is the more specific format, and recognizing it first eliminates the
                // residual ambiguity class where a SafeTensors 8-byte header-length prefix
                // imitates the opening bytes of a serialized ONNX model. The decompression
                // stream sits right after the 8 prefix bytes, so the probe can continue
                // reading the (bounded) JSON header sequentially.
                if (innerRead == 8
                    && TryInspectCompressedSafeTensors(filePath, ds, fileLen, inner, observations) is { } zst)
                    return zst;
            }
            catch (Exception e) when (e is ZstdException or InvalidDataException or EndOfStreamException)
            {
                return NotRecognized(filePath, fileLen, observations,
                    $"a Zstd frame that fails to decompress — corrupt or truncated ({e.Message}).");
            }

            string? layout =
                LooksLikeOnnxModelProto(inner.AsSpan(0, innerRead)) ? "single-Zstd"
                : innerRead >= 4 && LooksLikeZstd(inner) ? "double-Zstd"
                : null;
            if (layout is null)
                return NotRecognized(filePath, fileLen, observations,
                    "a Zstd frame whose decompressed content is not a serialized ONNX model — " +
                    "possibly a .zsafetensor archive or foreign compressed data.");

            observations.Add(LegacyNoHeaderObservation);
            return new ArtifactInspection(filePath, fileLen, ArtifactKind.SrkGraph,
                new SrkArtifactInfo(header: null, layout, fileLen),
                safeTensors: null, trainingCheckpoint: null, observations);
        }

        // ---- Zstd-compressed SafeTensors archive (.zsafetensor) ----

        /// <summary>
        /// Probes whether the Zstd frame's decompressed content is a SafeTensors file (a
        /// .zsafetensor archive, as written by
        /// <see cref="CompressedFormatUtils.SaveCompressedSafeTensors"/>). <paramref name="ds"/>
        /// is positioned just past the 8 decompressed prefix bytes (<paramref name="innerPrefix"/>);
        /// when the prefix declares a plausible header length, the probe stream-decompresses
        /// exactly the JSON header — bounded by <see cref="MaxSafeTensorsHeaderBytes"/>, like the
        /// uncompressed path — and never the tensor payload. Returns null when the content is not
        /// a compressed SafeTensors archive (so the legacy-.srk sniff proceeds), or a final
        /// result: recognized <see cref="ArtifactKind.CompressedSafeTensors"/>, or NotRecognized
        /// for a frame that positively declared a SafeTensors header but could not deliver it.
        ///
        /// Note the prefix declares a *decompressed* extent, so the uncompressed path's
        /// declared-length-vs-file-size checks do not transfer: the compressed file size has no
        /// fixed relation to the decompressed content, and only header-internal consistency is
        /// verifiable here without decompressing the payload.
        /// </summary>
        private static ArtifactInspection? TryInspectCompressedSafeTensors(
            string filePath, DecompressionStream ds, long fileLen,
            byte[] innerPrefix, List<string> observations)
        {
            long headerLen = BitConverter.ToInt64(innerPrefix, 0);
            if (headerLen <= 0 || headerLen > MaxSafeTensorsHeaderBytes)
                return null;

            // Read the header through the stream with a growing buffer instead of allocating
            // the declared length up front: unlike the uncompressed path, headerLen has no
            // file-size bound here, so a tiny hostile file declaring a 100 MB header must not
            // cost a 100 MB allocation — memory tracks what the stream actually delivers.
            var headerBytes = new byte[(int)Math.Min(headerLen, 64 * 1024)];
            int headerRead = 0;
            try
            {
                while (headerRead < headerLen)
                {
                    if (headerRead == headerBytes.Length)
                    {
                        var grown = new byte[(int)Math.Min(headerLen, (long)headerBytes.Length * 4)];
                        headerBytes.CopyTo(grown, 0);
                        headerBytes = grown;
                    }
                    int n = ds.Read(headerBytes, headerRead, headerBytes.Length - headerRead);
                    if (n == 0) break;   // clean end of the decompressed stream
                    headerRead += n;
                }
            }
            catch (Exception e) when (e is ZstdException or InvalidDataException or EndOfStreamException)
            {
                return NotRecognized(filePath, fileLen, observations,
                    $"a Zstd frame whose decompressed content declares a plausible SafeTensors header " +
                    $"({headerLen} bytes) but fails to decompress that far — a corrupt or truncated " +
                    $".zsafetensor archive? ({e.Message})");
            }

            if (headerRead < headerLen)
                return NotRecognized(filePath, fileLen, observations,
                    $"a Zstd frame whose decompressed content declares a SafeTensors header of " +
                    $"{headerLen} bytes but ends after {headerRead} of them — a truncated " +
                    ".zsafetensor archive or foreign compressed data.");

            if (ParseSafeTensorsHeader(headerBytes) is not { } parsed)
                return null;

            observations.AddRange(parsed.Observations);

            // A checkpoint saved compressed: the marker is visible in the header, but its
            // 16-byte [version, step] payload sits at its declared offset inside the
            // compressed tensor data (TrainingCheckpoint.Save writes it last). Reaching it
            // through the non-seekable Zstd stream would mean decompressing everything before
            // it — not a bounded header read — so the archive keeps the CompressedSafeTensors
            // kind and the observation says what more is known and why it stops there.
            int markerIndex = parsed.Tensors.FindIndex(
                t => t.Name == TrainingCheckpoint.CheckpointMarkerName);
            if (markerIndex >= 0)
                observations.Add(
                    $"the archive's header carries a '{TrainingCheckpoint.CheckpointMarkerName}' " +
                    "marker — a Shorokoo training checkpoint stored compressed — but the marker's " +
                    $"version/step payload sits {parsed.Tensors[markerIndex].DataStartOffset} bytes " +
                    "into the compressed tensor data, beyond Inspect's bounded header-only reads; " +
                    "decompress the archive (CompressedFormatUtils.LoadCompressedSafeTensors) to " +
                    "read them.");

            return new ArtifactInspection(filePath, fileLen, ArtifactKind.CompressedSafeTensors,
                srk: null,
                new SafeTensorsArtifactInfo(headerLen, parsed.Tensors, parsed.MaxEnd, parsed.GlobalMetadata),
                trainingCheckpoint: null, observations);
        }

        // ---- SafeTensors / training checkpoint ----

        private static ArtifactInspection? TryInspectSafeTensors(
            string filePath, FileStream stream, long fileLen, byte[] prefix, List<string> observations)
        {
            long headerLen = BitConverter.ToInt64(prefix, 0);
            if (headerLen <= 0 || headerLen > fileLen - 8 || headerLen > MaxSafeTensorsHeaderBytes)
                return null;

            var headerBytes = new byte[headerLen];
            stream.Position = 8;
            if (stream.ReadAtLeast(headerBytes, headerBytes.Length, throwOnEndOfStream: false) < headerLen)
                return null;

            if (ParseSafeTensorsHeader(headerBytes) is not { } parsed)
                return null;

            // Compare in subtracted form: fileLen - dataStart >= 0 is guaranteed by the
            // headerLen bound above, so a hostile huge maxEnd cannot wrap the comparison
            // (dataStart + maxEnd could). These checks are specific to the uncompressed
            // layout, where the declared extents and the file size share a coordinate
            // system — they do not transfer to the .zsafetensor path.
            long dataStart = 8 + headerLen;
            long dataBytesInFile = fileLen - dataStart;
            observations.AddRange(parsed.Observations);
            if (parsed.MaxEnd > dataBytesInFile)
                observations.Add($"declared tensor data extends {parsed.MaxEnd - dataBytesInFile} bytes " +
                    "past the end of the file — the file is truncated or corrupt.");
            else if (dataBytesInFile > parsed.MaxEnd)
                observations.Add($"the file has {dataBytesInFile - parsed.MaxEnd} trailing bytes beyond " +
                    "the declared tensor data.");

            var safeTensorsInfo = new SafeTensorsArtifactInfo(
                headerLen, parsed.Tensors, parsed.MaxEnd, parsed.GlobalMetadata);

            var checkpoint = TryReadCheckpointMarker(
                stream, fileLen, dataStart, parsed.Tensors, observations);
            return new ArtifactInspection(filePath, fileLen,
                checkpoint is null ? ArtifactKind.SafeTensors : ArtifactKind.TrainingCheckpoint,
                srk: null, safeTensorsInfo, checkpoint, observations);
        }

        /// <summary>
        /// Parses SafeTensors JSON header bytes into the tensor listing — shared by the plain
        /// (<see cref="TryInspectSafeTensors"/>) and Zstd-compressed
        /// (<see cref="TryInspectCompressedSafeTensors"/>) paths. Returns null when the bytes
        /// are not a SafeTensors header. The returned observations cover header-internal
        /// consistency only (invalid data_offsets extents); how the declared extents relate to
        /// what the file actually holds is layout-specific, so those checks stay with the
        /// callers.
        /// </summary>
        private static (List<InspectedTensorInfo> Tensors, Dictionary<string, string>? GlobalMetadata,
            long MaxEnd, List<string> Observations)? ParseSafeTensorsHeader(byte[] headerBytes)
        {
            JsonDocument doc;
            try
            {
                doc = JsonDocument.Parse(headerBytes);
            }
            catch (JsonException)
            {
                return null;
            }

            using (doc)
            {
                if (doc.RootElement.ValueKind != JsonValueKind.Object)
                    return null;

                // Collected locally so a file that stops parsing as SafeTensors halfway
                // through leaves no stray observations on the fall-through result.
                var stObservations = new List<string>();
                var tensors = new List<InspectedTensorInfo>();
                Dictionary<string, string>? globalMetadata = null;
                long maxEnd = 0;
                foreach (var prop in doc.RootElement.EnumerateObject())
                {
                    if (prop.Name == "__metadata__")
                    {
                        if (prop.Value.ValueKind == JsonValueKind.Object)
                        {
                            globalMetadata = new Dictionary<string, string>();
                            foreach (var meta in prop.Value.EnumerateObject())
                                globalMetadata[meta.Name] = meta.Value.ValueKind == JsonValueKind.String
                                    ? meta.Value.GetString()! : meta.Value.GetRawText();
                        }
                        continue;
                    }

                    // Every non-metadata entry must be a tensor descriptor, or this is not a
                    // SafeTensors header at all (e.g. some unrelated JSON-bearing format).
                    if (prop.Value.ValueKind != JsonValueKind.Object
                        || !prop.Value.TryGetProperty("dtype", out var dtypeEl)
                        || dtypeEl.ValueKind != JsonValueKind.String
                        || !prop.Value.TryGetProperty("shape", out var shapeEl)
                        || shapeEl.ValueKind != JsonValueKind.Array
                        || !prop.Value.TryGetProperty("data_offsets", out var offsetsEl)
                        || offsetsEl.ValueKind != JsonValueKind.Array
                        || offsetsEl.GetArrayLength() != 2)
                        return null;

                    long[] shape;
                    long start, end;
                    try
                    {
                        shape = [.. shapeEl.EnumerateArray().Select(e => e.GetInt64())];
                        start = offsetsEl[0].GetInt64();
                        end = offsetsEl[1].GetInt64();
                    }
                    // FormatException: a non-integer number; InvalidOperationException: a
                    // non-number element. Either way this is not a SafeTensors header.
                    catch (Exception e) when (e is FormatException or InvalidOperationException)
                    {
                        return null;
                    }

                    // A crafted start/end pair can wrap end - start even when end >= start,
                    // so flag both orderings-gone-wrong and the wrapped difference, and clamp
                    // the reported size to keep the listing sane.
                    long size = unchecked(end - start);
                    if (start < 0 || end < start || size < 0)
                    {
                        stObservations.Add($"tensor '{prop.Name}' declares an invalid extent " +
                            $"(data_offsets [{start}, {end}]).");
                        size = 0;
                    }
                    tensors.Add(new InspectedTensorInfo(prop.Name, dtypeEl.GetString()!, shape, size, start));
                    maxEnd = Math.Max(maxEnd, end);
                }

                return (tensors, globalMetadata, maxEnd, stObservations);
            }
        }

        private static TrainingCheckpointArtifactInfo? TryReadCheckpointMarker(
            FileStream stream, long fileLen, long dataStart,
            List<InspectedTensorInfo> tensors, List<string> observations)
        {
            // Locate the marker in the header listing; its declared extent gives the file
            // position of its 16 payload bytes ([version, step] as two int64s) — the only
            // payload bytes Inspect ever reads.
            int markerIndex = tensors.FindIndex(t => t.Name == TrainingCheckpoint.CheckpointMarkerName);
            if (markerIndex < 0)
                return null;

            // Bounds in subtracted form (fileLen - 16 cannot underflow for any file large
            // enough to hold a marker entry): markerStart + 16 could wrap past the check
            // for a crafted offset near long.MaxValue.
            var marker = tensors[markerIndex];
            long markerStart = dataStart + marker.DataStartOffset;
            if (marker.DType != "I64" || marker.ByteSize < 16 || markerStart < dataStart || markerStart > fileLen - 16)
            {
                observations.Add($"the file carries a '{TrainingCheckpoint.CheckpointMarkerName}' marker, " +
                    "but it is malformed — not a readable Shorokoo training checkpoint.");
                return null;
            }

            var markerBytes = new byte[16];
            try
            {
                stream.Position = markerStart;
                stream.ReadExactly(markerBytes);
            }
            // Belt-and-braces: the guard above bounds the read, but a seek/read that still
            // fails must degrade to the malformed-marker result, not escape Inspect.
            catch (Exception e) when (e is IOException or EndOfStreamException)
            {
                observations.Add($"the file carries a '{TrainingCheckpoint.CheckpointMarkerName}' marker, " +
                    $"but it is malformed and could not be read ({e.Message}) — not a readable Shorokoo " +
                    "training checkpoint.");
                return null;
            }
            long version = BitConverter.ToInt64(markerBytes, 0);
            long step = BitConverter.ToInt64(markerBytes, 8);

            if (version != TrainingCheckpoint.CheckpointFormatVersion)
                observations.Add($"checkpoint format version {version} is not the version this build " +
                    $"writes ({TrainingCheckpoint.CheckpointFormatVersion}); " +
                    "the file may come from a newer Shorokoo version.");

            string[] sectionNames =
            [
                TrainingCheckpoint.TrainableSection,
                TrainingCheckpoint.ModelStateSection,
                TrainingCheckpoint.OptimizerStateSection,
            ];
            var sections = new Dictionary<string, IReadOnlyList<InspectedTensorInfo>>();
            foreach (var section in sectionNames)
            {
                var prefix = section + "/";
                sections[section] =
                [
                    .. tensors
                        .Where(t => t.Name.StartsWith(prefix, StringComparison.Ordinal))
                        .Select(t => new InspectedTensorInfo(
                            t.Name.Substring(prefix.Length), t.DType, t.Shape, t.ByteSize, t.DataStartOffset)),
                ];
            }

            foreach (var t in tensors)
            {
                if (t.Name == TrainingCheckpoint.CheckpointMarkerName) continue;
                if (!sectionNames.Any(s => t.Name.StartsWith(s + "/", StringComparison.Ordinal)))
                    observations.Add($"tensor '{t.Name}' sits outside the known checkpoint sections " +
                        $"({string.Join(", ", sectionNames)}).");
            }

            return new TrainingCheckpointArtifactInfo(version, step, sections);
        }
    }
}

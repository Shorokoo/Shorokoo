using System;
using System.Collections.Generic;
using System.IO;
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

    /// <summary>
    /// Result of <see cref="Checkpoint.Inspect"/>: what the file is (<see cref="Kind"/>), the
    /// per-kind details (exactly the properties matching the kind are non-null), and cheap
    /// sanity <see cref="Observations"/> visible from the header alone.
    /// </summary>
    public sealed class ArtifactInspection
    {
        /// <summary>The inspected file's path, as passed to <see cref="Checkpoint.Inspect"/>.</summary>
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

        /// <summary>Checkpoint details; non-null iff <see cref="Kind"/> is <see cref="ArtifactKind.TrainingCheckpoint"/>.</summary>
        public TrainingCheckpointArtifactInfo? TrainingCheckpoint { get; }

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
    }

    /// <summary>
    /// Read-only inspection of Shorokoo-produced files: <see cref="Inspect"/> identifies what a
    /// file is and summarizes its contents from headers/prefixes only — tensor payloads are never
    /// loaded, so inspecting a multi-GB file is fast and cheap. (Partial: the .skpt save/load
    /// entry points live in Checkpoint.cs — one <c>Checkpoint</c> facade, two concerns.)
    /// </summary>
    public static partial class Checkpoint
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
        /// are stream-decompressed, the tensor payload never), and training checkpoints written
        /// by <see cref="TrainingCheckpoint.Save"/> (via the
        /// checkpoint marker; the marker's 16 bytes are the only payload bytes ever read).
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
                "matching no format Shorokoo writes (.srk container, legacy .srk layout, or SafeTensors).");
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

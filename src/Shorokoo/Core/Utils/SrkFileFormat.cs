using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Graph;

namespace Shorokoo.Core.Utils
{
    /// <summary>
    /// The lifecycle stage of a serialized graph. Recorded in the .srk v2 header so a
    /// loader can route or refuse a file at load time instead of failing deep inside
    /// execution (e.g. <c>No Op registered for ShrkCreateModule</c>).
    /// </summary>
    public enum SrkGraphStage
    {
        /// <summary>A pre-lowering module graph: still contains module/function
        /// invocation machinery (<c>ShrkModelInvoke</c> etc.).</summary>
        Module,

        /// <summary>A lowered architecture from <c>ToConcreteArchitecture</c>: every
        /// trainable parameter is visible at top level but values are not yet
        /// materialized (parameters still carry their initializer functions).</summary>
        ConcreteArchitecture,

        /// <summary>A weight-filled, runnable graph from <c>ToConcreteModel</c>.</summary>
        ConcreteModel,
    }

    /// <summary>Producer metadata recorded in the .srk v2 header (informational; the
    /// payload dialect remains versioned by the embedded ONNX model itself).</summary>
    public sealed class SrkProducerInfo
    {
        /// <summary>Version of the Shorokoo framework that wrote the file.</summary>
        [JsonPropertyName("shorokoo")]
        public string? Shorokoo { get; set; }

        /// <summary>ONNX IR version of the embedded ModelProto.</summary>
        [JsonPropertyName("irVersion")]
        public long IrVersion { get; set; }

        /// <summary>Opset imports of the embedded ModelProto, keyed by domain
        /// (<c>""</c> is the default ONNX domain).</summary>
        [JsonPropertyName("opsets")]
        public Dictionary<string, long>? Opsets { get; set; }
    }

    /// <summary>
    /// The JSON header of a .srk v2 container. Fields are add-only across minor
    /// revisions of the format; removing or re-typing a field is a major-version
    /// event (a bump of the magic's version byte and <see cref="SrkVersion"/>).
    /// Unknown fields written by newer minor revisions are preserved in
    /// <see cref="AdditionalFields"/> and ignored.
    /// </summary>
    public sealed class SrkHeader
    {
        /// <summary>Container format version; <see cref="SrkFileFormat.CurrentVersion"/> for files written today.</summary>
        [JsonPropertyName("srkVersion")]
        public int SrkVersion { get; set; }

        /// <summary>Graph stage name: "module", "concrete-architecture" or "concrete-model".
        /// Parse with <see cref="TryGetStage"/>.</summary>
        [JsonPropertyName("stage")]
        public string? Stage { get; set; }

        /// <summary>Compression of the payload: "none" or "zstd". Exactly one
        /// compression layer, ever — detected from here, never from the file extension.</summary>
        [JsonPropertyName("compression")]
        public string? Compression { get; set; }

        /// <summary>Lowercase hex SHA-256 of the payload bytes as stored in the file
        /// (i.e. after compression). Allows integrity checking and truncation detection
        /// without decompressing.</summary>
        [JsonPropertyName("payloadSha256")]
        public string? PayloadSha256 { get; set; }

        /// <summary>Producer metadata (framework version, embedded ONNX ir/opsets).</summary>
        [JsonPropertyName("producer")]
        public SrkProducerInfo? Producer { get; set; }

        /// <summary>Round-trips header fields added by newer minor revisions of the format.</summary>
        [JsonExtensionData]
        public Dictionary<string, JsonElement>? AdditionalFields { get; set; }

        /// <summary>Parses <see cref="Stage"/>; null when the name is missing or unknown
        /// (e.g. a stage introduced by a newer framework version).</summary>
        public SrkGraphStage? TryGetStage() => SrkFileFormat.TryParseStageName(Stage);
    }

    /// <summary>
    /// The .srk v2 self-describing container for serialized <see cref="FastComputationGraph"/>s:
    ///
    /// <code>magic "SRK\x02" | u16 headerLen (little-endian) | JSON header | payload</code>
    ///
    /// The payload is an ONNX ModelProto (internal dialect allowed), optionally wrapped in
    /// exactly one Zstd layer as declared by the header — compression is detected from the
    /// header, never from the file extension (".zsrk" vs ".srk" is a human-readable hint
    /// with no parsing significance). Legacy v1 files (bare protobuf, single-Zstd and the
    /// retired double-Zstd layout) remain readable through the content-sniffing shim in
    /// <see cref="Read"/>; writers emit v2 only.
    ///
    /// Save/load entry points live on <see cref="CompressedFormatUtils"/>
    /// (<c>SaveFastGraphToFile</c> / <c>LoadFastGraphFromFile</c> and the binary variants);
    /// this class owns the container layout, header schema, stage detection and the v1 shim.
    /// </summary>
    public static class SrkFileFormat
    {
        /// <summary>Current container major version. The version is also baked into the
        /// last magic byte, so a major break is visible before the header is parsed.</summary>
        public const int CurrentVersion = 2;

        /// <summary>Magic bytes opening every v2 file: "SRK" followed by the format major version.</summary>
        public static ReadOnlySpan<byte> Magic => [(byte)'S', (byte)'R', (byte)'K', CurrentVersion];

        private const int MagicLength = 4;
        private const int HeaderLengthFieldSize = 2;
        private const string CompressionNone = "none";
        private const string CompressionZstd = "zstd";

        private static readonly JsonSerializerOptions HeaderJsonOptions = new()
        {
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        };

        #region Stage names and detection

        /// <summary>Canonical header name of a stage ("module", "concrete-architecture", "concrete-model").</summary>
        public static string StageName(SrkGraphStage stage) => stage switch
        {
            SrkGraphStage.Module => "module",
            SrkGraphStage.ConcreteArchitecture => "concrete-architecture",
            SrkGraphStage.ConcreteModel => "concrete-model",
            _ => throw new ArgumentOutOfRangeException(nameof(stage), stage, null),
        };

        /// <summary>Inverse of <see cref="StageName"/>; null for unknown or missing names.</summary>
        public static SrkGraphStage? TryParseStageName(string? name) => name switch
        {
            "module" => SrkGraphStage.Module,
            "concrete-architecture" => SrkGraphStage.ConcreteArchitecture,
            "concrete-model" => SrkGraphStage.ConcreteModel,
            _ => null,
        };

        /// <summary>
        /// Ops whose presence marks a graph as still module-stage (pre-lowering). This is the
        /// union of the internal ops that <c>ToConcreteArchitecture</c> guarantees are gone from
        /// its output plus the module-machinery ops that only ever exist before lowering.
        /// </summary>
        private static readonly HashSet<string> ModuleStageOps =
        [
            InternalOpCodes.MODULE_SET_HYPERPARAMS,
            InternalOpCodes.MODEL_INVOKE,
            InternalOpCodes.FUNCTION_INVOKE,
            InternalOpCodes.MODEL_HYPERPARAM,
            InternalOpCodes.GET_MODEL_ID,
            InternalOpCodes.NEW_MODEL_LIKE,
            InternalOpCodes.CREATE_MODULE,
            InternalOpCodes.MODEL_PARAM_REF,
            InternalOpCodes.MODEL_PARAM_MODEL_REF,
            InternalOpCodes.MODEL_PARAM_ID_REF,
            InternalOpCodes.TENSOR_STRUCT_CREATE,
            InternalOpCodes.TENSOR_STRUCT_GETFIELD,
            InternalOpCodes.MODEL_TENSORSTRUCT_INPUT,
            InternalOpCodes.AUTO_GRAD,
            InternalOpCodes.GENERIC_TYPE_INPUT,
        ];

        /// <summary>
        /// Classifies a graph's lifecycle stage from its content: module machinery present →
        /// <see cref="SrkGraphStage.Module"/>; unmaterialized trainable parameters
        /// (<c>MODEL_PARAM</c> nodes) present → <see cref="SrkGraphStage.ConcreteArchitecture"/>;
        /// otherwise <see cref="SrkGraphStage.ConcreteModel"/>. This is what the writer records
        /// in the header, and what the v1 shim falls back to when enforcing a required stage on
        /// a header-less legacy file.
        /// </summary>
        public static SrkGraphStage DetectStage(FastComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            foreach (var node in graph.Nodes)
            {
                if (ModuleStageOps.Contains(node.OpCode) ||
                    node.OpCode.StartsWith(InternalOpCodes.SUBMODEL, StringComparison.Ordinal))
                    return SrkGraphStage.Module;
            }

            return graph.Nodes.Any(n => n.OpCode == InternalOpCodes.MODEL_PARAM)
                ? SrkGraphStage.ConcreteArchitecture
                : SrkGraphStage.ConcreteModel;
        }

        /// <summary>
        /// Throws a clear stage-mismatch error naming both stages (and the file, via
        /// <paramref name="origin"/>) when <paramref name="actual"/> differs from
        /// <paramref name="required"/>.
        /// </summary>
        internal static void EnforceStage(SrkGraphStage actual, SrkGraphStage required, string origin)
        {
            if (actual == required) return;

            var hint = actual == SrkGraphStage.Module
                ? " A module-stage graph cannot execute; lower it first with " +
                  "ToConcreteArchitecture(...) (and ToConcreteModel(...) for a runnable model), then save that."
                : string.Empty;
            throw new InvalidOperationException(
                $"'{origin}' contains a '{StageName(actual)}'-stage graph, but a " +
                $"'{StageName(required)}'-stage graph is required here.{hint}");
        }

        #endregion

        #region Writing

        /// <summary>
        /// Wraps serialized ONNX bytes in a v2 container: applies the (single, optional) Zstd
        /// layer, records stage/compression/payload-hash/producer in the JSON header, and
        /// prepends magic + header length.
        /// </summary>
        internal static byte[] Write(
            byte[] onnxBytes,
            SrkGraphStage stage,
            bool compress,
            int compressionLevel,
            long irVersion,
            IReadOnlyCollection<KeyValuePair<string, long>> opsets)
        {
            var payload = compress
                ? CompressedFormatUtils.Compress(onnxBytes, compressionLevel)
                : onnxBytes;

            var header = new SrkHeader
            {
                SrkVersion = CurrentVersion,
                Stage = StageName(stage),
                Compression = compress ? CompressionZstd : CompressionNone,
                PayloadSha256 = Sha256Hex(payload),
                Producer = new SrkProducerInfo
                {
                    Shorokoo = ShorokooVersion.Value,
                    IrVersion = irVersion,
                    Opsets = opsets.ToDictionary(kv => kv.Key, kv => kv.Value),
                },
            };

            var headerBytes = JsonSerializer.SerializeToUtf8Bytes(header, HeaderJsonOptions);
            if (headerBytes.Length > ushort.MaxValue)
                throw new InvalidOperationException(
                    $".srk header is {headerBytes.Length} bytes; the u16 length field caps it at {ushort.MaxValue}.");

            var result = new byte[MagicLength + HeaderLengthFieldSize + headerBytes.Length + payload.Length];
            Magic.CopyTo(result);
            result[MagicLength] = (byte)(headerBytes.Length & 0xFF);
            result[MagicLength + 1] = (byte)(headerBytes.Length >> 8);
            headerBytes.CopyTo(result, MagicLength + HeaderLengthFieldSize);
            payload.CopyTo(result, MagicLength + HeaderLengthFieldSize + headerBytes.Length);
            return result;
        }

        private static readonly Lazy<string> ShorokooVersion = new(() =>
        {
            var assembly = typeof(SrkFileFormat).Assembly;
            return assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion
                ?? assembly.GetName().Version?.ToString()
                ?? string.Empty;
        });

        private static string Sha256Hex(byte[] payload)
            => Convert.ToHexString(SHA256.HashData(payload)).ToLowerInvariant();

        #endregion

        #region Reading

        /// <summary>True when the bytes start with the v2 magic ("SRK\x02").</summary>
        public static bool IsSrkV2(byte[] data)
            => data is not null && data.Length >= MagicLength && data.AsSpan(0, MagicLength).SequenceEqual(Magic);

        /// <summary>
        /// Reads and validates the v2 header without touching the payload. Returns null for
        /// data without the v2 magic (a legacy v1 file, or not a .srk at all); throws
        /// <see cref="InvalidDataException"/> for a v2 file whose header is truncated,
        /// malformed, or of an unsupported major version.
        /// </summary>
        public static SrkHeader? TryReadHeader(byte[] data, string? origin = null)
            => IsSrkV2(data) ? ReadHeaderCore(data, origin ?? "<in-memory .srk data>", out _) : null;

        /// <summary>File-path convenience over <see cref="TryReadHeader(byte[], string?)"/>;
        /// error messages name the file.</summary>
        public static SrkHeader? TryReadHeaderFromFile(string filePath)
            => TryReadHeader(File.ReadAllBytes(filePath), filePath);

        /// <summary>
        /// Extracts the serialized ONNX model bytes from any .srk layout, deciding by content
        /// only. For a v2 container this validates the header and the payload SHA-256, then
        /// removes the header-declared compression layer; corruption and truncation fail
        /// loudly with a message naming <paramref name="origin"/>. For legacy v1 data (no v2
        /// magic) the shim sniffs Zstd framing: bare protobuf, single-Zstd
        /// (<c>SaveFastGraphToFile</c>) and the retired double-Zstd
        /// (<c>SaveCompressedArchitecture</c>) layouts all load; the returned header is null.
        /// </summary>
        /// <param name="data">Raw file/stream bytes.</param>
        /// <param name="origin">Name used in error messages, typically the file path.</param>
        public static (SrkHeader? Header, byte[] OnnxBytes) Read(byte[] data, string? origin = null)
        {
            if (data is null) throw new ArgumentNullException(nameof(data));
            origin ??= "<in-memory .srk data>";

            if (IsSrkV2(data))
            {
                var header = ReadHeaderCore(data, origin, out var payloadOffset);
                var payload = data.AsSpan(payloadOffset).ToArray();

                if (string.IsNullOrEmpty(header.PayloadSha256))
                    throw new InvalidDataException(
                        $"'{origin}': invalid .srk v2 header — required field 'payloadSha256' is missing.");
                var actualSha = Sha256Hex(payload);
                if (!string.Equals(actualSha, header.PayloadSha256, StringComparison.OrdinalIgnoreCase))
                    throw new InvalidDataException(
                        $"'{origin}': payload SHA-256 mismatch — the file is corrupt or truncated " +
                        $"(header records {header.PayloadSha256}, payload hashes to {actualSha}).");

                return header.Compression switch
                {
                    CompressionNone => (header, payload),
                    CompressionZstd => (header, DecompressPayload(payload, origin)),
                    _ => throw new InvalidDataException(
                        $"'{origin}': .srk header declares unsupported compression " +
                        $"'{header.Compression}' (supported: '{CompressionNone}', '{CompressionZstd}'). " +
                        "The file was likely written by a newer Shorokoo version."),
                };
            }

            // v1 shim: no container. Sniff Zstd framing by content — bare protobuf needs no
            // unwrapping; SaveFastGraphToFile wrote one Zstd layer; the retired
            // SaveCompressedArchitecture wrote two.
            var bytes = data;
            for (int layer = 0; layer < 2 && LooksLikeZstd(bytes); layer++)
                bytes = DecompressPayload(bytes, origin);
            return (null, bytes);
        }

        private static SrkHeader ReadHeaderCore(byte[] data, string origin, out int payloadOffset)
        {
            if (data.Length < MagicLength + HeaderLengthFieldSize)
                throw new InvalidDataException(
                    $"'{origin}': truncated .srk v2 file — {data.Length} bytes is too short to hold the header length field.");

            int headerLen = data[MagicLength] | (data[MagicLength + 1] << 8);
            payloadOffset = MagicLength + HeaderLengthFieldSize + headerLen;
            if (data.Length < payloadOffset)
                throw new InvalidDataException(
                    $"'{origin}': truncated .srk v2 file — the header declares {headerLen} bytes " +
                    $"but only {data.Length - MagicLength - HeaderLengthFieldSize} bytes follow the length field.");

            SrkHeader? header;
            try
            {
                header = JsonSerializer.Deserialize<SrkHeader>(
                    data.AsSpan(MagicLength + HeaderLengthFieldSize, headerLen), HeaderJsonOptions);
            }
            catch (JsonException e)
            {
                throw new InvalidDataException(
                    $"'{origin}': corrupt .srk v2 file — the JSON header does not parse: {e.Message}", e);
            }
            if (header is null)
                throw new InvalidDataException($"'{origin}': corrupt .srk v2 file — the JSON header is null.");

            if (header.SrkVersion != CurrentVersion)
                throw new InvalidDataException(
                    $"'{origin}': .srk container version {header.SrkVersion} is not supported by this " +
                    $"Shorokoo build (supported: {CurrentVersion}). The file was likely written by a newer framework version.");

            return header;
        }

        /// <summary>Zstd frame magic: 28 B5 2F FD (little-endian 0xFD2FB528).</summary>
        private static bool LooksLikeZstd(byte[] data)
            => data.Length >= 4 && data[0] == 0x28 && data[1] == 0xB5 && data[2] == 0x2F && data[3] == 0xFD;

        private static byte[] DecompressPayload(byte[] payload, string origin)
        {
            try
            {
                return CompressedFormatUtils.Decompress(payload);
            }
            catch (Exception e)
            {
                throw new InvalidDataException(
                    $"'{origin}': failed to Zstd-decompress the payload — the file is corrupt or truncated. ({e.Message})", e);
            }
        }

        #endregion
    }
}

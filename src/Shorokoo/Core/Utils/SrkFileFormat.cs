using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Graph;

namespace Shorokoo.Core.Utils
{
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
        public GraphKind? TryGetStage() => SrkFileFormat.TryParseStageName(Stage);
    }

    /// <summary>
    /// The .srk v2 self-describing container for serialized <see cref="InternalComputationGraph"/>s:
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
        private const int MagicPrefixLength = 3;
        private const int HeaderLengthFieldSize = 2;
        private const string CompressionNone = "none";
        private const string CompressionZstd = "zstd";

        /// <summary>The most Zstd layers any historical (pre-container) .srk layout ever wrapped:
        /// the retired double-Zstd architecture writer. The v1 shim unwraps up to this many,
        /// then treats anything still Zstd-framed as corrupt.</summary>
        private const int MaxV1ZstdLayers = 2;

        private static readonly JsonSerializerOptions HeaderJsonOptions = new()
        {
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        };

        #region Stage names and detection

        /// <summary>Canonical header name of a stage ("module", "concrete-architecture", "concrete-model").</summary>
        public static string StageName(GraphKind stage) => stage switch
        {
            GraphKind.Module => "module",
            GraphKind.ConcreteArchitecture => "concrete-architecture",
            GraphKind.ConcreteModel => "concrete-model",
            _ => throw new ArgumentOutOfRangeException(nameof(stage), stage, null),
        };

        /// <summary>Inverse of <see cref="StageName"/>; null for unknown or missing names.</summary>
        public static GraphKind? TryParseStageName(string? name) => name switch
        {
            "module" => GraphKind.Module,
            "concrete-architecture" => GraphKind.ConcreteArchitecture,
            "concrete-model" => GraphKind.ConcreteModel,
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
        /// <see cref="GraphKind.Module"/>; unmaterialized trainable parameters
        /// (<c>MODEL_PARAM</c> nodes) present → <see cref="GraphKind.ConcreteArchitecture"/>;
        /// otherwise <see cref="GraphKind.ConcreteModel"/>. This is what the writer records
        /// in the header, and what the v1 shim falls back to when enforcing a required stage on
        /// a header-less legacy file.
        /// </summary>
        public static GraphKind DetectStage(InternalComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            foreach (var node in graph.Nodes)
            {
                if (ModuleStageOps.Contains(node.OpCode) ||
                    node.OpCode.StartsWith(InternalOpCodes.SUBMODEL, StringComparison.Ordinal))
                    return GraphKind.Module;
            }

            return graph.Nodes.Any(n => n.OpCode == InternalOpCodes.MODEL_PARAM)
                ? GraphKind.ConcreteArchitecture
                : GraphKind.ConcreteModel;
        }

        /// <summary>
        /// Reads the graph-kind metadata tag (<see cref="Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames.ShrkMetaGraphKind"/>)
        /// stamped into a serialized model by the ONNX builders, so a saved graph can be
        /// reloaded as the same kind. Null when the model carries no (recognizable) tag —
        /// foreign models, or files written before the tag existed.
        /// </summary>
        internal static GraphKind? TryReadKindTag(Shorokoo.Core.Factory.IR.ModelProto model)
        {
            if (model is null) throw new ArgumentNullException(nameof(model));
            foreach (var prop in model.MetadataProps)
                if (prop.Key == OnnxOpAttributeNames.ShrkMetaGraphKind)
                    return TryParseStageName(prop.Value);
            return null;
        }

        /// <summary>
        /// Checks whether <paramref name="kind"/> is a valid stamp for the graph's content.
        /// Returns null when valid, else a sentence naming the violated requirement:
        /// <list type="bullet">
        /// <item><see cref="GraphKind.Module"/> — must not have initialized model parameters.</item>
        /// <item><see cref="GraphKind.ConcreteArchitecture"/> — parameter number and shapes must be
        /// statically known (no module-stage ops), and no model parameter may be initialized.</item>
        /// <item><see cref="GraphKind.ConcreteModel"/> — parameters must be statically known
        /// (no module-stage ops) and every parameter initialized (no unmaterialized
        /// <c>MODEL_PARAM</c> nodes).</item>
        /// </list>
        /// </summary>
        internal static string? DescribeKindViolation(InternalComputationGraph graph, GraphKind kind)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            // The RngSeed identity parameter at reserved ModelId [0] is not a weight
            // (binding an RNG config IS its initialization, legitimate from architecture
            // stage on), so it never counts as an "initialized model parameter" here.
            //
            // A FUNCTION_INVOKE that calls a plain Function is executable content, not
            // module machinery — a reimported concrete model legitimately carries such
            // calls (e.g. the RNG draw functions emitted at export). Only calls whose
            // target is module-typed (or unresolved) mark the graph as module-stage.
            int moduleOps = 0, uninitializedParams = 0, initializedParams = 0;
            foreach (var node in graph.Nodes)
            {
                bool isModuleOp = node.OpCode == InternalOpCodes.FUNCTION_INVOKE
                    ? node.TargetFunction is not { FunctionType: Shorokoo.Core.Nodes.OnnxNodes.FunctionType.Function }
                    : ModuleStageOps.Contains(node.OpCode) ||
                      node.OpCode.StartsWith(InternalOpCodes.SUBMODEL, StringComparison.Ordinal);
                if (isModuleOp)
                    moduleOps++;
                else if (node.OpCode == InternalOpCodes.MODEL_PARAM)
                    uninitializedParams++;
                else if (node.OpCode == InternalOpCodes.MODEL_PARAM_DATA &&
                         node.IdentifierTemplate !=
                             Shorokoo.Core.Nodes.Processors.Fast.FastWireRngKeyDerivation.RngSeedIdentifierTemplate)
                    initializedParams++;
            }

            switch (kind)
            {
                case GraphKind.Module:
                    if (initializedParams > 0)
                        return $"a module graph cannot have initialized model parameters, " +
                               $"but this graph carries {initializedParams} initialized parameter(s).";
                    return null;

                case GraphKind.ConcreteArchitecture:
                    if (moduleOps > 0)
                        return "a concrete architecture's parameter number and shapes must be statically " +
                               $"known, but this graph still contains {moduleOps} module-stage op(s).";
                    if (initializedParams > 0)
                        return "a concrete architecture must not have initialized model parameters, " +
                               $"but this graph carries {initializedParams} initialized parameter(s).";
                    return null;

                case GraphKind.ConcreteModel:
                    if (moduleOps > 0)
                        return "a concrete model's parameters must be statically known, " +
                               $"but this graph still contains {moduleOps} module-stage op(s).";
                    if (uninitializedParams > 0)
                        return "a concrete model must have all model parameters initialized, " +
                               $"but this graph carries {uninitializedParams} unmaterialized parameter(s).";
                    return null;

                default:
                    throw new ArgumentOutOfRangeException(nameof(kind), kind, null);
            }
        }

        /// <summary>
        /// Op-scan classification of a serialized model: the <see cref="DetectStage(InternalComputationGraph)"/>
        /// rules applied to a <see cref="Shorokoo.Core.Factory.IR.ModelProto"/>'s main graph, nested
        /// subgraph attributes, and function bodies. Used where only the serialized artifact exists
        /// (e.g. the <c>SaveWithExternalData</c> concrete-model gate) — a ModelProto carries no
        /// stamped kind.
        /// </summary>
        internal static GraphKind DetectStage(Shorokoo.Core.Factory.IR.ModelProto model)
        {
            if (model is null) throw new ArgumentNullException(nameof(model));

            // Serialization rewrites MODEL_PARAM nodes and function invokes into calls
            // to "Functions"-domain FunctionProtos, so those in-memory opcodes are not
            // visible on the wire. Recover them from the function metadata: a call to an
            // initializer-typed function IS a serialized unmaterialized parameter, and a
            // call to a module-typed function is module machinery. Calls to plain
            // Function-typed functions stay unclassified — vanilla concrete-model
            // exports legitimately contain them.
            var fnTypeByName = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (var fn in model.Functions)
                foreach (var prop in fn.MetadataProps)
                    if (prop.Key == Shorokoo.Core.Function.IRFunctionTypeParamName)
                        fnTypeByName[fn.Name] = prop.Value;

            bool sawModelParam = false;
            bool sawModuleOp = false;

            void ScanNodes(IEnumerable<Shorokoo.Core.Factory.IR.NodeProto> nodes)
            {
                foreach (var node in nodes)
                {
                    if (ModuleStageOps.Contains(node.OpType) ||
                        node.OpType.StartsWith(InternalOpCodes.SUBMODEL, StringComparison.Ordinal))
                        sawModuleOp = true;
                    else if (node.OpType == InternalOpCodes.MODEL_PARAM)
                        sawModelParam = true;
                    else if (fnTypeByName.TryGetValue(node.OpType, out var fnType))
                    {
                        if (fnType == nameof(Shorokoo.Core.Nodes.OnnxNodes.FunctionType.TrainableParamInitializer) ||
                            fnType == nameof(Shorokoo.Core.Nodes.OnnxNodes.FunctionType.StateParamInitializer))
                            sawModelParam = true;
                        else if (fnType == nameof(Shorokoo.Core.Nodes.OnnxNodes.FunctionType.Module) ||
                                 fnType == nameof(Shorokoo.Core.Nodes.OnnxNodes.FunctionType.ModuleSignature))
                            sawModuleOp = true;
                    }

                    foreach (var attr in node.Attributes)
                    {
                        if (attr.G is not null) ScanNodes(attr.G.Nodes);
                        foreach (var g in attr.Graphs) ScanNodes(g.Nodes);
                    }
                }
            }

            if (model.Graph is not null) ScanNodes(model.Graph.Nodes);
            foreach (var fn in model.Functions) ScanNodes(fn.Nodes);

            return sawModuleOp ? GraphKind.Module
                : sawModelParam ? GraphKind.ConcreteArchitecture
                : GraphKind.ConcreteModel;
        }

        /// <summary>
        /// Shared remedy sentence for kind/stage-mismatch errors: unstamped data is
        /// classified by op-scanning, which cannot tell a machinery-free module body or
        /// a parameterless architecture from a concrete model — the validated re-stamp
        /// is the way out when the stamp itself is what's wrong.
        /// </summary>
        internal const string WithKindRemedyHint =
            "If the stamp itself is wrong (op-scanning of unstamped data can misjudge " +
            "machinery-free graphs), re-stamp the graph with ComputationGraph.WithKind.";

        /// <summary>
        /// Throws a clear stage-mismatch error naming both stages (and the file, via
        /// <paramref name="origin"/>) when <paramref name="actual"/> differs from
        /// <paramref name="required"/>.
        /// </summary>
        internal static void EnforceStage(GraphKind actual, GraphKind required, string origin)
        {
            if (actual == required) return;

            var hint = actual == GraphKind.Module
                ? " A module-stage graph cannot execute; lower it first with " +
                  "ToConcreteArchitecture(...) (and ToConcreteModel(...) for a runnable model), then save that."
                : string.Empty;
            throw new InvalidOperationException(
                $"'{origin}' contains a '{StageName(actual)}'-stage graph, but a " +
                $"'{StageName(required)}'-stage graph is required here.{hint}" +
                " If the file's recorded stage is wrong, load it without requiredStage and " +
                "re-stamp the graph with ComputationGraph.WithKind.");
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
            GraphKind stage,
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
                    // Same version the ONNX exporter stamps as producer_version — one source of
                    // truth (Version.cs strips any "+build-metadata"), so the header records e.g. "0.1.0".
                    Shorokoo = Shorokoo.ShorokooVersion.VersionString,
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

        private static string Sha256Hex(ReadOnlySpan<byte> payload)
            => Convert.ToHexString(SHA256.HashData(payload)).ToLowerInvariant();

        #endregion

        #region Reading

        /// <summary>True when the bytes start with the v2 magic ("SRK\x02").</summary>
        public static bool IsSrkV2(byte[] data)
            => data is not null && data.Length >= MagicLength && data.AsSpan(0, MagicLength).SequenceEqual(Magic);

        /// <summary>True when the bytes open with the "SRK" container prefix, whatever the
        /// trailing major-version byte. A serialized ONNX ModelProto opens with 0x08 and a Zstd
        /// frame with 0x28, so no legacy v1 layout can collide with this prefix.</summary>
        private static bool StartsWithMagicPrefix(byte[] data)
            => data is not null && data.Length >= MagicPrefixLength
               && data[0] == (byte)'S' && data[1] == (byte)'R' && data[2] == (byte)'K';

        /// <summary>
        /// Throws when <paramref name="data"/> is an .srk container whose major version this
        /// build does not understand (the "SRK" prefix is present but the version byte is not
        /// <see cref="CurrentVersion"/>). This makes a major-version break a clear, pre-header
        /// error instead of letting the file fall through to the legacy content shim and fail
        /// obscurely as unparseable protobuf.
        /// </summary>
        private static void ThrowIfUnsupportedContainerVersion(byte[] data, string origin)
        {
            if (!StartsWithMagicPrefix(data) || IsSrkV2(data))
                return;
            int version = data.Length > MagicPrefixLength ? data[MagicPrefixLength] : -1;
            throw new InvalidDataException(
                $"'{origin}': .srk container major version {version} is not supported by this Shorokoo " +
                $"build (supported: {CurrentVersion}). The file was likely written by a newer framework version.");
        }

        /// <summary>
        /// Reads and validates the v2 header without touching the payload. Returns null for
        /// data without the "SRK" container prefix (a legacy v1 file, or not a .srk at all);
        /// throws <see cref="InvalidDataException"/> for a v2 file whose header is truncated or
        /// malformed, or for an .srk container of an unsupported major version.
        /// </summary>
        public static SrkHeader? TryReadHeader(byte[] data, string? origin = null)
        {
            origin ??= "<in-memory .srk data>";
            if (IsSrkV2(data))
                return ReadHeaderCore(data, origin, out _);
            ThrowIfUnsupportedContainerVersion(data, origin);
            return null;
        }

        /// <summary>
        /// File-path convenience over <see cref="TryReadHeader(byte[], string?)"/> that reads only
        /// the container prefix and header from disk — never the (potentially multi-GB) payload —
        /// so identifying/routing a file is cheap. Error messages name the file.
        /// </summary>
        public static SrkHeader? TryReadHeaderFromFile(string filePath)
        {
            using var stream = File.OpenRead(filePath);

            Span<byte> prefix = stackalloc byte[MagicLength + HeaderLengthFieldSize];
            int prefixRead = stream.ReadAtLeast(prefix, prefix.Length, throwOnEndOfStream: false);
            if (prefixRead < prefix.Length)
                // Too short to be a v2 container; hand what we have to the shared parser, which
                // returns null for legacy/other data and throws for a truncated SRK container.
                return TryReadHeader(prefix[..prefixRead].ToArray(), filePath);

            // Legacy v1 / non-.srk data has no header to read; only a container carries one.
            if (!(prefix[0] == (byte)'S' && prefix[1] == (byte)'R' && prefix[2] == (byte)'K'))
                return null;

            int headerLen = prefix[MagicLength] | (prefix[MagicLength + 1] << 8);
            var buf = new byte[prefix.Length + headerLen];
            prefix.CopyTo(buf);
            int bodyRead = stream.ReadAtLeast(buf.AsSpan(prefix.Length), headerLen, throwOnEndOfStream: false);

            // Hand exactly magic+length+header (as far as it was read) to the shared parser: it
            // validates the magic/version, declared length and JSON without needing the payload.
            return TryReadHeader(bodyRead < headerLen ? buf[..(prefix.Length + bodyRead)] : buf, filePath);
        }

        /// <summary>
        /// Extracts the serialized ONNX model bytes from any .srk layout, deciding by content
        /// only. For a v2 container this validates the header and the payload SHA-256, then
        /// removes the header-declared compression layer; corruption and truncation fail
        /// loudly with a message naming <paramref name="origin"/>. For legacy v1 data (no v2
        /// magic) the shim sniffs Zstd framing: bare protobuf, single-Zstd
        /// (<c>SaveFastGraphToFile</c>) and the retired double-Zstd architecture-writer
        /// layouts all load; the returned header is null.
        /// </summary>
        /// <param name="data">Raw file/stream bytes.</param>
        /// <param name="origin">Name used in error messages, typically the file path.</param>
        public static (SrkHeader? Header, byte[] OnnxBytes) Read(byte[] data, string? origin = null)
        {
            if (data is null) throw new ArgumentNullException(nameof(data));
            origin ??= "<in-memory .srk data>";

            if (data.Length == 0)
                throw new InvalidDataException($"'{origin}': the file is empty — not a valid .srk file.");

            if (IsSrkV2(data))
            {
                var header = ReadHeaderCore(data, origin, out var payloadOffset);
                // Hash and (for the compressed case) decompress the payload straight from the
                // file buffer — no intermediate whole-payload copy for a large model.
                var payload = data.AsSpan(payloadOffset);

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
                    CompressionNone => (header, payload.ToArray()),
                    CompressionZstd => (header, DecompressPayload(payload, origin)),
                    _ => throw new InvalidDataException(
                        $"'{origin}': .srk header declares unsupported compression " +
                        $"'{header.Compression}' (supported: '{CompressionNone}', '{CompressionZstd}'). " +
                        "The file was likely written by a newer Shorokoo version."),
                };
            }

            // An .srk container whose major version this build cannot read must fail clearly
            // here, before the legacy shim mistakes its header bytes for content.
            ThrowIfUnsupportedContainerVersion(data, origin);

            // v1 shim: no container. Sniff Zstd framing by content — bare protobuf needs no
            // unwrapping; SaveFastGraphToFile wrote one Zstd layer; the retired
            // the retired double-Zstd writer wrote two (the maximum any legacy layout used).
            var bytes = data;
            for (int layer = 0; layer < MaxV1ZstdLayers && LooksLikeZstd(bytes); layer++)
                bytes = DecompressPayload(bytes, origin);
            if (LooksLikeZstd(bytes))
                throw new InvalidDataException(
                    $"'{origin}': still Zstd-compressed after unwrapping {MaxV1ZstdLayers} layers " +
                    "(the maximum for any legacy .srk layout) — the file is corrupt or was written by an unsupported tool.");
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

            // srkVersion is required and always >= 1; a 0 means the field was absent (or mis-cased,
            // so it landed in AdditionalFields), which is a malformed header, not a version skew.
            if (header.SrkVersion == 0)
                throw new InvalidDataException(
                    $"'{origin}': invalid .srk v2 header — required field 'srkVersion' is missing or zero.");

            if (header.SrkVersion != CurrentVersion)
                throw new InvalidDataException(
                    $"'{origin}': .srk container version {header.SrkVersion} is not supported by this Shorokoo " +
                    $"build (supported: {CurrentVersion}). The file was written by " +
                    (header.SrkVersion > CurrentVersion
                        ? "a newer framework version."
                        : "an older, unsupported framework version."));

            return header;
        }

        /// <summary>Zstd frame magic: 28 B5 2F FD (little-endian 0xFD2FB528).</summary>
        private static bool LooksLikeZstd(byte[] data)
            => data.Length >= 4 && data[0] == 0x28 && data[1] == 0xB5 && data[2] == 0x2F && data[3] == 0xFD;

        private static byte[] DecompressPayload(ReadOnlySpan<byte> payload, string origin)
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

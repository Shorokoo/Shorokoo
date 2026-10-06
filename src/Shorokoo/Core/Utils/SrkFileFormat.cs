using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Graph;

namespace Shorokoo.Core.Utils
{
    /// <summary>Producer metadata recorded in the .srk header (informational; the
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
    /// The JSON header of a .srk container. Fields the reader does not interpret
    /// are preserved in <see cref="AdditionalFields"/> and ignored.
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

        /// <summary>Round-trips header fields this reader does not interpret, unchanged.</summary>
        [JsonExtensionData]
        public Dictionary<string, JsonElement>? AdditionalFields { get; set; }

        /// <summary>Parses <see cref="Stage"/>; null when the name is missing or is not one
        /// this build defines.</summary>
        public GraphKind? TryGetStage() => SrkFileFormat.TryParseStageName(Stage);
    }

    /// <summary>
    /// The .srk self-describing container for serialized <see cref="InternalComputationGraph"/>s:
    ///
    /// <code>magic "SRK\x01" | u16 headerLen (little-endian) | JSON header | payload</code>
    ///
    /// The payload is an ONNX ModelProto (internal dialect allowed), optionally wrapped in
    /// exactly one Zstd layer as declared by the header — compression is detected from the
    /// header, never from the file extension (".zsrk" vs ".srk" is a human-readable hint
    /// with no parsing significance). A file that does not open with the container magic is
    /// not a .srk file.
    ///
    /// Save/load entry points live on <see cref="CompressedFormatUtils"/>
    /// (<c>SaveFastGraphToFile</c> / <c>LoadFastGraphFromFile</c> and the binary variants);
    /// this class owns the container layout, header schema and stage detection.
    /// </summary>
    public static class SrkFileFormat
    {
        /// <summary>Current container major version. The version is also baked into the
        /// last magic byte, so a major break is visible before the header is parsed.</summary>
        public const int CurrentVersion = 1;

        /// <summary>Magic bytes opening every .srk file: "SRK" followed by the format major version.</summary>
        public static ReadOnlySpan<byte> Magic => [(byte)'S', (byte)'R', (byte)'K', CurrentVersion];

        private const int MagicLength = 4;
        private const int MagicPrefixLength = 3;
        private const int HeaderLengthFieldSize = 2;
        private const string CompressionNone = "none";
        private const string CompressionZstd = "zstd";

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
        /// Classifies a graph's lifecycle stage from its content: module machinery
        /// (<see cref="InternalOpCodes.ModuleStageOps"/> — the same canonical inventory whose
        /// absence <c>ToConcreteArchitecture</c> asserts on its output) present →
        /// <see cref="GraphKind.Module"/>; unmaterialized trainable parameters
        /// (<c>MODEL_PARAM</c> nodes) present → <see cref="GraphKind.ConcreteArchitecture"/>;
        /// otherwise <see cref="GraphKind.ConcreteModel"/> — unless an input carries no
        /// representative shape, which every concrete graph records on each input
        /// (<see cref="Shorokoo.Core.Graph.RepresentativeInputShapes"/>), or an output no recorded
        /// shape, making the graph a <see cref="GraphKind.Module"/>; and unless an output records
        /// the unresolved marker (<see cref="Shorokoo.Core.Graph.RecordedOutputShapes.UnresolvedShape"/>),
        /// which only a <see cref="GraphKind.ConcreteArchitecture"/> carries. This is what the writer records
        /// in the header, and the fallback classification used when a file's recorded stage is
        /// missing or is not one this build defines.
        /// </summary>
        public static GraphKind DetectStage(InternalComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            foreach (var node in graph.Nodes)
            {
                if (InternalOpCodes.IsModuleStageOp(node.OpCode))
                    return GraphKind.Module;
            }

            // Every concrete graph records a representative shape on each of its inputs, and the
            // shape it produces on each of its outputs; a graph missing one was never concretized.
            if (Shorokoo.Core.Graph.RepresentativeInputShapes.FirstInputWithoutShape(graph) is not null
                || Shorokoo.Core.Graph.RecordedOutputShapes.FirstOutputWithoutShape(graph) is not null)
                return GraphKind.Module;

            // An output whose shape the weights decide is left unsettled on an architecture alone:
            // a graph carrying one is no concrete model, whatever parameters it has left.
            var byOps = DetectStageByOps(graph);
            return byOps == GraphKind.ConcreteModel
                   && Shorokoo.Core.Graph.RecordedOutputShapes.FirstUnresolvedOutput(graph) is not null
                ? GraphKind.ConcreteArchitecture
                : byOps;
        }

        /// <summary>
        /// <see cref="DetectStage(InternalComputationGraph)"/> from the ops alone, not the inputs'
        /// representative shapes: what an importer stamps a foreign graph it is about to give those
        /// shapes, and refuses where it cannot.
        /// </summary>
        internal static GraphKind DetectStageByOps(InternalComputationGraph graph)
        {
            if (graph.Nodes.Any(n => InternalOpCodes.IsModuleStageOp(n.OpCode)))
                return GraphKind.Module;
            return graph.Nodes.Any(n => n.OpCode == InternalOpCodes.MODEL_PARAM)
                ? GraphKind.ConcreteArchitecture
                : GraphKind.ConcreteModel;
        }

        /// <summary>
        /// Reads the graph-kind metadata tag (<see cref="Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames.ShrkMetaGraphKind"/>)
        /// stamped into a serialized model by the ONNX builders, so a saved graph can be
        /// reloaded as the same kind. Null when the model carries no (recognizable) tag, as a
        /// foreign model does.
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

            // The RngSeed parameter at reserved ModelId [0] is not a weight
            // (binding an RNG config IS its initialization, legitimate from architecture
            // stage on), so it never counts as an "initialized model parameter" here.
            //
            // Module machinery is classified by FastNodeClassification.IsModuleStageMachinery —
            // the classification question, deliberately not the executability one the execution
            // gate asks (IsUnrunnableModuleOp): an initializer-typed invoke marks a graph as
            // pre-lowering here while running perfectly well there.
            int moduleOps = 0, uninitializedParams = 0, initializedParams = 0;
            foreach (var node in graph.Nodes)
            {
                if (node.IsModuleStageMachinery())
                    moduleOps++;
                else if (node.OpCode == InternalOpCodes.MODEL_PARAM)
                    uninitializedParams++;
                else if (node.OpCode == InternalOpCodes.MODEL_PARAM_DATA &&
                         node.IdentifierTemplate !=
                             Shorokoo.Core.Nodes.Processors.Fast.FastWireRngKeyDerivation.RngSeedIdentifierTemplate)
                    initializedParams++;
            }

            if (kind is GraphKind.ConcreteArchitecture or GraphKind.ConcreteModel
                && moduleOps == 0
                && Shorokoo.Core.Graph.RepresentativeInputShapes.FirstInputWithoutShape(graph) is { } unshaped)
                return "every input of a concrete graph records the shape it was concretized at, " +
                       $"but this graph's input '{unshaped}' carries none.";
            if (kind is GraphKind.ConcreteArchitecture or GraphKind.ConcreteModel
                && moduleOps == 0
                && Shorokoo.Core.Graph.RecordedOutputShapes.FirstOutputWithoutShape(graph) is { } unrecorded)
                return "every output of a concrete graph records the shape it has at the samples the graph " +
                       $"was concretized at, but this graph's output '{unrecorded}' records none.";

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
                    if (Shorokoo.Core.Graph.RecordedOutputShapes.FirstUnresolvedOutput(graph) is { } unresolved)
                        return "every output of a concrete model records a settled shape, but this graph's " +
                               $"output {unresolved} records none, only the marker a concrete architecture " +
                               "carries for a shape its weights decide.";
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
                    if (InternalOpCodes.IsModuleStageOp(node.OpType))
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
        /// The stamp-reading gates' form: <see cref="Mismatch"/> plus the shared
        /// <see cref="WithKindRemedyHint"/>, since there the stamp is what may be wrong.
        /// </summary>
        internal static string KindMismatchMessage(
            string operation, string requiredDescription, GraphKind actual, string? hint = null)
            => Mismatch(operation, requiredDescription, actual, hint) + " " + WithKindRemedyHint;

        /// <summary>
        /// The op-scanning gates' form (<c>InternalComputationGraphExtensions.RequireRunnableOps</c>):
        /// <see cref="Mismatch"/> ending at the caller's own hint. <see cref="WithKindRemedyHint"/>
        /// would misdirect there, because the stamp is not what is wrong — machinery cannot run
        /// whatever the graph is stamped, and an eager-evaluation caller holds no
        /// <see cref="ComputationGraph"/> to re-stamp anyway.
        /// </summary>
        internal static string MachineryMismatchMessage(
            string operation, string requiredDescription, GraphKind actual, string hint)
            => Mismatch(operation, requiredDescription, actual, hint);

        /// <summary>The one leading sentence behind every graph-kind-mismatch error — the operation,
        /// the requirement, the actual kind — so that wording cannot drift between the gates. Each
        /// public form above adds its own ending; file loads use <see cref="EnforceStage"/>.</summary>
        private static string Mismatch(
            string operation, string requiredDescription, GraphKind actual, string? hint)
            => $"{operation} requires {requiredDescription}, but this graph is a '{StageName(actual)}'." +
               (string.IsNullOrEmpty(hint) ? string.Empty : " " + hint);

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
        /// Writes a .srk container to <paramref name="destination"/>: magic, header length, the JSON
        /// header recording stage/compression/payload-hash/producer, then the payload
        /// <paramref name="writePayload"/> writes — <paramref name="payloadLength"/> bytes of
        /// serialized ONNX — through the (single, optional) Zstd layer. Nothing holds the payload
        /// whole: it is hashed as it is written, and the hash, which the header carries ahead of
        /// the payload, is written into its place in the header once the payload is done. The
        /// destination must therefore be seekable.
        /// </summary>
        internal static void Write(
            Stream destination,
            GraphKind stage,
            bool compress,
            int compressionLevel,
            long irVersion,
            IReadOnlyCollection<KeyValuePair<string, long>> opsets,
            long payloadLength,
            Action<Stream> writePayload)
        {
            if (!destination.CanSeek)
                throw new ArgumentException("A .srk container is written to a seekable stream.", nameof(destination));

            var header = new SrkHeader
            {
                SrkVersion = CurrentVersion,
                Stage = StageName(stage),
                Compression = compress ? CompressionZstd : CompressionNone,
                // A stand-in of the hash's own length, overwritten once the payload is hashed.
                PayloadSha256 = new string('0', Sha256HexLength),
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
            int hashAt = headerBytes.AsSpan().IndexOf(PayloadSha256Key) + PayloadSha256Key.Length;

            long start = destination.Position;
            destination.Write(Magic);
            destination.Write([(byte)(headerBytes.Length & 0xFF), (byte)(headerBytes.Length >> 8)]);
            destination.Write(headerBytes);

            string hash;
            using (var hashed = new Sha256WriteStream(destination))
            {
                if (compress)
                    CompressedFormatUtils.WriteZstdFrame(hashed, compressionLevel, payloadLength, writePayload);
                else
                    writePayload(hashed);
                hash = hashed.Sha256Hex();
            }

            long end = destination.Position;
            destination.Position = start + MagicLength + HeaderLengthFieldSize + hashAt;
            destination.Write(Encoding.ASCII.GetBytes(hash));
            destination.Position = end;
        }

        private const int Sha256HexLength = 64;

        /// <summary>The header's hash key and the opening quote of its value, as the writer spells them.</summary>
        private static ReadOnlySpan<byte> PayloadSha256Key => "\"payloadSha256\":\""u8;

        /// <summary>A forward write-only stream passing every byte on and taking their SHA-256.</summary>
        private sealed class Sha256WriteStream(Stream inner) : Stream
        {
            private readonly IncrementalHash _hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);

            /// <summary>The lowercase hex SHA-256 of everything written so far.</summary>
            public string Sha256Hex() => Convert.ToHexStringLower(_hash.GetCurrentHash());

            public override void Write(ReadOnlySpan<byte> buffer)
            {
                inner.Write(buffer);
                _hash.AppendData(buffer);
            }

            public override void Write(byte[] buffer, int offset, int count) => Write(buffer.AsSpan(offset, count));
            public override void Flush() => inner.Flush();
            public override bool CanRead => false;
            public override bool CanSeek => false;
            public override bool CanWrite => true;
            public override long Length => throw new NotSupportedException();
            public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
            public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
            public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
            public override void SetLength(long value) => throw new NotSupportedException();

            protected override void Dispose(bool disposing)
            {
                if (disposing) _hash.Dispose();
                base.Dispose(disposing);
            }
        }

        #endregion

        #region Reading

        /// <summary>True when the bytes start with the current .srk magic ("SRK\x01").</summary>
        public static bool IsSrkContainer(byte[] data)
            => data is not null && data.Length >= MagicLength && data.AsSpan(0, MagicLength).SequenceEqual(Magic);

        /// <summary>True when the bytes open with the "SRK" container prefix, whatever the
        /// trailing major-version byte.</summary>
        private static bool StartsWithMagicPrefix(byte[] data)
            => data is not null && data.Length >= MagicPrefixLength
               && data[0] == (byte)'S' && data[1] == (byte)'R' && data[2] == (byte)'K';

        /// <summary>
        /// Throws when <paramref name="data"/> is an .srk container whose major version this
        /// build does not understand (the "SRK" prefix is present but the version byte is not
        /// <see cref="CurrentVersion"/>). This makes a major-version break a clear, pre-header
        /// error naming the offending version.
        /// </summary>
        private static void ThrowIfUnsupportedContainerVersion(byte[] data, string origin)
        {
            if (!StartsWithMagicPrefix(data) || IsSrkContainer(data))
                return;
            int version = data.Length > MagicPrefixLength ? data[MagicPrefixLength] : -1;
            throw new InvalidDataException(
                $"'{origin}': .srk container major version {version} is not readable by this Shorokoo " +
                $"build, which reads version {CurrentVersion} only.");
        }

        /// <summary>
        /// Reads and validates the container header without touching the payload. Returns null
        /// for data without the "SRK" container prefix (not a .srk file at all); throws
        /// <see cref="InvalidDataException"/> for a .srk file whose header is truncated or
        /// malformed, or for an .srk container of an unsupported major version.
        /// </summary>
        public static SrkHeader? TryReadHeader(byte[] data, string? origin = null)
        {
            origin ??= InMemoryOrigin;
            if (IsSrkContainer(data))
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
                // Too short to be a container; hand what we have to the shared parser, which
                // returns null for non-.srk data and throws for a truncated SRK container.
                return TryReadHeader(prefix[..prefixRead].ToArray(), filePath);

            // Non-.srk data has no header to read; only a container carries one.
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
        /// Extracts the serialized ONNX model bytes from a .srk container. Validates the header
        /// and the payload SHA-256, then removes the header-declared compression layer;
        /// corruption and truncation fail loudly with a message naming <paramref name="origin"/>.
        /// Data that does not open with the container magic is not a .srk file and throws.
        /// </summary>
        /// <param name="data">Raw file/stream bytes.</param>
        /// <param name="origin">Name used in error messages, typically the file path.</param>
        public static (SrkHeader Header, byte[] OnnxBytes) Read(byte[] data, string? origin = null)
        {
            if (data is null) throw new ArgumentNullException(nameof(data));
            using var container = new MemoryStream(data, writable: false);
            var (header, payload, _) = OpenPayload(container, origin ?? InMemoryOrigin);
            using (payload)
            {
                if (ReferenceEquals(payload, container))
                    return (header, data[(int)container.Position..]);
                using var onnx = new MemoryStream();
                payload.CopyTo(onnx);
                return (header, onnx.ToArray());
            }
        }

        /// <summary>The name a container held in memory goes by in error messages.</summary>
        internal const string InMemoryOrigin = "<in-memory .srk data>";

        /// <summary>
        /// Opens the serialized ONNX payload of the .srk container that <paramref name="container"/>
        /// holds from where it stands to its end, validated as <see cref="Read"/> validates it: the
        /// header read and checked, the payload's SHA-256 taken by reading it through once and held
        /// to the header's, and the header-declared compression layer removed as the returned
        /// stream is read. Nothing reads the payload whole: a container of any size opens. The
        /// returned stream is <paramref name="container"/> itself, positioned at the payload, when
        /// the payload is not compressed, else a decoder reading from it whose failure is the
        /// payload failing to decompress, refused naming <paramref name="origin"/>. With it comes
        /// the most bytes the payload can hold, which bounds every length its fields declare: the
        /// bytes past the header, or what its Zstd frames decompress to at most
        /// (<see cref="CompressedFormatUtils.ZstdContentSizeBound"/>).
        /// </summary>
        internal static (SrkHeader Header, Stream Payload, long MaxLength) OpenPayload(Stream container, string origin)
        {
            long start = container.Position;
            var prefix = new byte[MagicLength + HeaderLengthFieldSize];
            int prefixRead = container.ReadAtLeast(prefix, prefix.Length, throwOnEndOfStream: false);
            if (prefixRead == 0)
                throw new InvalidDataException($"'{origin}': the file is empty — not a valid .srk file.");

            var head = prefix;
            if (prefixRead == prefix.Length && IsSrkContainer(prefix))
            {
                int headerLen = prefix[MagicLength] | (prefix[MagicLength + 1] << 8);
                head = new byte[prefix.Length + headerLen];
                prefix.CopyTo(head, 0);
                int bodyRead = container.ReadAtLeast(head.AsSpan(prefix.Length), headerLen, throwOnEndOfStream: false);
                head = head[..(prefix.Length + bodyRead)];
            }
            else
                head = prefix[..prefixRead];

            if (!IsSrkContainer(head))
            {
                // A "SRK"-prefixed file of an unsupported major version fails with a clear
                // version error; anything else is simply not a .srk container.
                ThrowIfUnsupportedContainerVersion(head, origin);
                throw new InvalidDataException(
                    $"'{origin}': not a Shorokoo .srk container — the file does not open with the " +
                    $"'SRK\\x{CurrentVersion:X2}' container magic.");
            }

            var header = ReadHeaderCore(head, origin, out var payloadOffset);
            long payloadStart = start + payloadOffset;

            if (string.IsNullOrEmpty(header.PayloadSha256))
                throw new InvalidDataException(
                    $"'{origin}': invalid .srk header — required field 'payloadSha256' is missing.");
            container.Position = payloadStart;
            var actualSha = Convert.ToHexStringLower(SHA256.HashData(container));
            if (!string.Equals(actualSha, header.PayloadSha256, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException(
                    $"'{origin}': payload SHA-256 mismatch — the file is corrupt or truncated " +
                    $"(header records {header.PayloadSha256}, payload hashes to {actualSha}).");

            container.Position = payloadStart;
            return header.Compression switch
            {
                CompressionNone => (header, container, container.Length - payloadStart),
                CompressionZstd => (header, new DecodingReadStream(
                    new ZstdSharp.DecompressionStream(container, leaveOpen: true),
                    e => new InvalidDataException(
                        $"'{origin}': failed to Zstd-decompress the payload — the file is corrupt or truncated. ({e.Message})", e)),
                    ZstdPayloadBound(container, payloadStart, origin)),
                _ => throw new InvalidDataException(
                    $"'{origin}': .srk header declares unsupported compression " +
                    $"'{header.Compression}' (supported: '{CompressionNone}', '{CompressionZstd}')."),
            };
        }

        /// <summary>The most bytes the Zstd payload at <paramref name="payloadStart"/> decompresses
        /// to, the container left positioned there.</summary>
        private static long ZstdPayloadBound(Stream container, long payloadStart, string origin)
        {
            long bound = CompressedFormatUtils.ZstdContentSizeBound(container, reason => new InvalidDataException(
                $"'{origin}': {reason} — the file is corrupt or truncated."));
            container.Position = payloadStart;
            return bound;
        }

        private static SrkHeader ReadHeaderCore(byte[] data, string origin, out int payloadOffset)
        {
            if (data.Length < MagicLength + HeaderLengthFieldSize)
                throw new InvalidDataException(
                    $"'{origin}': truncated .srk file — {data.Length} bytes is too short to hold the header length field.");

            int headerLen = data[MagicLength] | (data[MagicLength + 1] << 8);
            payloadOffset = MagicLength + HeaderLengthFieldSize + headerLen;
            if (data.Length < payloadOffset)
                throw new InvalidDataException(
                    $"'{origin}': truncated .srk file — the header declares {headerLen} bytes " +
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
                    $"'{origin}': corrupt .srk file — the JSON header does not parse: {e.Message}", e);
            }
            if (header is null)
                throw new InvalidDataException($"'{origin}': corrupt .srk file — the JSON header is null.");

            // srkVersion is required and always >= 1; a 0 means the field was absent (or mis-cased,
            // so it landed in AdditionalFields), which is a malformed header, not a version skew.
            if (header.SrkVersion == 0)
                throw new InvalidDataException(
                    $"'{origin}': invalid .srk header — required field 'srkVersion' is missing or zero.");

            if (header.SrkVersion != CurrentVersion)
                throw new InvalidDataException(
                    $"'{origin}': .srk container version {header.SrkVersion} is not readable by this " +
                    $"Shorokoo build, which reads version {CurrentVersion} only.");

            return header;
        }

        #endregion
    }
}

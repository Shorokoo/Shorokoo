using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using ProtoBuf;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Modules;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Onnx;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Runtime;
using ZstdSharp;

namespace Shorokoo.Core.Utils
{
    /// <summary>
    /// Utility class for loading and saving compressed Shorokoo data formats.
    /// Supports:
    /// - .zsafetensor: Zstandard-compressed safetensors files
    /// - .srk / .zsrk: Shorokoo graph files — written and read as self-describing
    ///   containers (see <see cref="SrkFileFormat"/>). The extensions are hints for
    ///   humans only; the header, not the extension, declares the compression.
    /// </summary>
    public static class CompressedFormatUtils
    {
        /// <summary>
        /// Default Zstandard compression level (3 is a good balance between speed and compression ratio)
        /// </summary>
        public const int DefaultCompressionLevel = 3;

        /// <summary>
        /// File extension for compressed safetensors files
        /// </summary>
        public const string CompressedSafeTensorExtension = ".zsafetensor";

        /// <summary>
        /// File extension for compressed Shorokoo architecture files
        /// </summary>
        public const string CompressedArchitectureExtension = ".zsrk";
        public const string UncompressedArchitectureExtension = ".srk";
        public const string JsonArchitectureExtension = ".json";

        #region Compressed SafeTensor Loading

        /// <summary>
        /// Load a compressed SafeTensor file (.zsafetensor) into a ModelParamList
        /// </summary>
        /// <param name="filePath">Path to the .zsafetensor file</param>
        /// <param name="paramType">Type of model parameters</param>
        /// <returns>ModelParamList containing all tensors from the file</returns>
        public static ModelParamList LoadCompressedModelParamSet(string filePath, ModelParamType paramType = ModelParamType.TrainableParam)
        {
            var tensors = ReadCompressedSafeTensors(filePath);
            var paramDict = tensors.ToDictionary(t => t.Name, t => t.Data);
            return new ModelParamList(paramDict, paramType);
        }

        /// <summary>
        /// Load a compressed SafeTensor file (.zsafetensor) into a list of SafeTensor objects
        /// </summary>
        /// <param name="filePath">Path to the .zsafetensor file</param>
        /// <returns>List of SafeTensor objects containing tensor data and metadata</returns>
        public static List<SafeTensor> LoadCompressedSafeTensors(string filePath)
        {
            return ReadCompressedSafeTensors(filePath);
        }

        /// <summary>
        /// The tensors of a compressed SafeTensors file, decoded as they are read: neither the file
        /// nor its decompressed payload is ever whole in memory, only the tensors themselves
        /// (Shorokoo/Shorokoo#436). The size its frames must declare for the payload bounds what
        /// its header may claim, so a tensor the payload cannot hold is refused before it is
        /// allocated; a failure of the decoder itself is the file failing to decompress, and is
        /// refused as that rather than as whatever the reader was reading when it happened.
        /// </summary>
        private static List<SafeTensor> ReadCompressedSafeTensors(string filePath)
            => ReadCompressedSafeTensors(filePath, (_, _) => ComputeContext.Host);

        /// <summary><see cref="ReadCompressedSafeTensors(string)"/>, putting each tensor where
        /// <paramref name="placement"/> names for it, and passing over one it names none for
        /// (<see cref="SafeTensorLoader.ReadSafeTensors"/>).</summary>
        internal static List<SafeTensor> ReadCompressedSafeTensors(
            string filePath, Func<string, long, ComputeContext?> placement)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException($"Compressed file not found: {filePath}");
            using var file = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read,
                bufferSize: 1 << 16, FileOptions.SequentialScan);
            long declared = DeclaredZstdContentSize(file, reason => new InvalidDataException(
                $"'{filePath}': {reason} — the file is corrupt or not a compressed SafeTensors file."));
            file.Position = 0;
            using var decoded = new DecodingReadStream(
                new DecompressionStream(file, leaveOpen: true),
                e => new InvalidDataException(
                    $"'{filePath}': failed to Zstd-decompress the file — it is corrupt or truncated. ({e.Message})", e));
            return SafeTensorLoader.ReadSafeTensors(decoded, declared, placement, filePath);
        }

        /// <summary>
        /// Load a compressed SafeTensor file (.zsafetensor) into a single TensorData
        /// </summary>
        /// <param name="filePath">Path to the .zsafetensor file containing exactly one tensor</param>
        /// <returns>TensorData containing the single tensor</returns>
        /// <exception cref="InvalidOperationException">Thrown if file contains zero or multiple tensors</exception>
        public static TensorData LoadCompressedSingleTensor(string filePath)
        {
            var tensors = LoadCompressedSafeTensors(filePath);

            if (tensors.Count == 0)
                throw new InvalidOperationException($"Compressed SafeTensor file '{filePath}' contains no tensors");

            if (tensors.Count > 1)
                throw new InvalidOperationException($"Compressed SafeTensor file '{filePath}' contains {tensors.Count} tensors, expected exactly 1");

            return tensors.First().Data;
        }

        /// <summary>
        /// Load a compressed SafeTensor file (.zsafetensor) into a Dictionary of tensor names to TensorData
        /// </summary>
        /// <param name="filePath">Path to the .zsafetensor file</param>
        /// <returns>Dictionary mapping tensor names to TensorData</returns>
        public static Dictionary<string, TensorData> LoadCompressedTensorDictionary(string filePath)
        {
            var tensors = LoadCompressedSafeTensors(filePath);
            return tensors.ToDictionary(t => t.Name, t => t.Data);
        }

        #endregion

        #region Compressed SafeTensor Saving

        /// <summary>
        /// Save tensors to a compressed SafeTensor file (.zsafetensor): the SafeTensors file is
        /// written straight through a Zstd compressor into the file, each tensor from its own
        /// storage by the piece, so neither the file nor its uncompressed form is ever held whole
        /// and a file of any size is written. The payload is one Zstd frame declaring its
        /// decompressed size, which the loader holds the SafeTensors header to. The write is atomic
        /// (staged beside the target and committed by rename), so a failed or interrupted save
        /// leaves any previous file untouched; the target's directory must already exist.
        /// </summary>
        /// <param name="filePath">Path for the output .zsafetensor file</param>
        /// <param name="tensors">List of SafeTensor objects to save</param>
        /// <param name="globalMetadata">Optional global metadata to include</param>
        /// <param name="compressionLevel">Zstandard compression level (1-22, default: 3)</param>
        public static void SaveCompressedSafeTensors(string filePath, List<SafeTensor> tensors, Dictionary<string, object>? globalMetadata = null, int compressionLevel = DefaultCompressionLevel)
        {
            AtomicFileWriter.WriteFile(filePath, file => WriteZstdFrame(file, compressionLevel,
                frame => SafeTensorLoader.SaveSafeTensorsToStream(frame, tensors, globalMetadata)));
        }

        /// <summary>
        /// Save a ModelParamList to a compressed SafeTensor file (.zsafetensor)
        /// </summary>
        /// <param name="filePath">Path for the output .zsafetensor file</param>
        /// <param name="paramSet">ModelParamList to save</param>
        /// <param name="compressionLevel">Zstandard compression level (1-22, default: 3)</param>
        public static void SaveCompressedModelParamSet(string filePath, ModelParamList paramSet, int compressionLevel = DefaultCompressionLevel)
        {
            var tensors = paramSet.ModelParams.Select(p => new SafeTensor(
                p.ParamName,
                p.ToTensorData(),
                SafeTensorLoader.DTypeToSafeTensorDType(p.Type),
                p.ToTensorData().Shape.Dims
            )).ToList();

            SaveCompressedSafeTensors(filePath, tensors, compressionLevel: compressionLevel);
        }

        #endregion

        #region Architecture Save / Load

        /// <summary>
        /// Serialize <paramref name="graph"/> to a self-describing .srk container
        /// (see <see cref="SrkFileFormat"/>): the ONNX payload built via
        /// <see cref="FastOnnxModelBuilder"/>, wrapped in exactly one optional Zstd layer,
        /// behind a JSON header recording the container version, the graph's lifecycle
        /// stage, the compression, the payload SHA-256, and producer info.
        /// <para>The container is returned as one array, so it is refused, with
        /// <see cref="NotSupportedException"/>, when it is larger than an array holds
        /// (<see cref="Array.MaxLength"/> bytes); <see cref="SaveFastGraphToFile(string, ComputationGraph, bool, bool, int)"/>
        /// writes a container of any size.</para>
        /// </summary>
        public static byte[] SaveFastGraphToBinary(
            ComputationGraph graph, bool compressed = true, int compressionLevel = DefaultCompressionLevel)
            // The header records the graph's stamped kind — the authoritative stage, no op-scan.
            => SaveFastGraphToBinary(graph.ToInternal(), graph.Kind, compressed, compressionLevel);

        /// <summary>
        /// Internal-graph form of <see cref="SaveFastGraphToBinary(ComputationGraph, bool, int)"/>.
        /// With no stamped kind available, the header stage falls back to op-scanning
        /// (<see cref="SrkFileFormat.DetectStage(InternalComputationGraph)"/>).
        /// </summary>
        internal static byte[] SaveFastGraphToBinary(
            InternalComputationGraph graph, GraphKind? stage = null, bool compressed = true,
            int compressionLevel = DefaultCompressionLevel)
            => SaveFastGraphToBinary(graph, stage, compressed, compressionLevel, Array.MaxLength);

        /// <summary>
        /// <see cref="SaveFastGraphToBinary(InternalComputationGraph, GraphKind?, bool, int)"/>
        /// refusing a container of more than <paramref name="maxBytes"/> bytes: up front, before
        /// any of it is written, when the uncompressed payload alone exceeds it, else as soon as
        /// the container being written does.
        /// </summary>
        internal static byte[] SaveFastGraphToBinary(
            InternalComputationGraph graph, GraphKind? stage, bool compressed, int compressionLevel, long maxBytes)
        {
            NotSupportedException TooLarge(string what) => new(
                $"{what} more than the {maxBytes:N0} bytes one array holds, so SaveFastGraphToBinary cannot " +
                "return this graph's .srk container. Save it with CompressedFormatUtils.SaveFastGraphToFile, " +
                "which streams a container of any size to the file.");

            var srk = SrkPayload.Of(graph, stage);
            if (!compressed && srk.Writer.Length > maxBytes)
                throw TooLarge($"The container's payload alone is {srk.Writer.Length:N0} bytes,");
            using var buffer = new BoundedMemoryStream(maxBytes, () => TooLarge("The container is"));
            srk.WriteContainer(buffer, compressed, compressionLevel);
            return buffer.ToArray();
        }

        /// <summary>
        /// A graph prepared for a .srk container: the ONNX model it is persisted as, ready to be
        /// streamed (<see cref="OnnxStreamingWriter"/>), and the stage the header records.
        /// </summary>
        private sealed class SrkPayload
        {
            public required GraphKind Stage { get; init; }
            public required long IrVersion { get; init; }
            public required KeyValuePair<string, long>[] Opsets { get; init; }
            public required OnnxStreamingWriter Writer { get; init; }

            public static SrkPayload Of(InternalComputationGraph graph, GraphKind? stage)
            {
                var resolvedStage = stage ?? SrkFileFormat.DetectStage(graph);
                // The stage rides in the payload's metadata too (not just this container's
                // header), so the graph reloads as the same kind even when the payload
                // travels as a bare ONNX model. Persistence must be faithful, so the
                // execution lowerings are disabled: a saved graph keeps its
                // STATE_UPDATE_LINK / WITH_STATE_DEPS machinery, state initializers, and
                // SHRK_RANDOM_* / SHRK_RNG_* feed ops verbatim.
                // emitInputsAsNodes: the .srk on-disk dialect serializes every top-level model-input op as an
                // ordinary NodeProto (carrying its attributes — e.g. a MODEL_TENSOR_INPUT's representative-
                // input shape), not a graph-input ValueInfoProto, so a saved graph stays self-describing
                // across the round-trip: the representative attributes are serialized on those nodes exactly
                // as the in-memory build set them.
                var model = FastOnnxModelBuilder.BuildInternalOnnxModel(
                    graph, stage: resolvedStage, applyExecutionLowerings: false, emitInputsAsNodes: true);
                return new SrkPayload
                {
                    Stage = resolvedStage,
                    IrVersion = model.IrVersion,
                    Opsets = [.. model.OpsetImports.Select(o => new KeyValuePair<string, long>(o.Domain, o.Version))],
                    // The model's weights are streamed into the payload from where they lie, so the
                    // container is never held whole; only the model besides each raw_data of at least
                    // OnnxStreamingWriter.MinStreamedBytes is built in one array.
                    Writer = OnnxStreamingWriter.Prepare(model),
                };
            }

            public void WriteContainer(Stream destination, bool compressed, int compressionLevel)
                => SrkFileFormat.Write(destination, Stage, compressed, compressionLevel, IrVersion, Opsets,
                    Writer.Length, Writer.WriteTo);
        }

        /// <summary>A memory stream refusing to grow past <paramref name="maxBytes"/>, with the
        /// exception <paramref name="refusal"/> makes, before it allocates for the bytes past it.</summary>
        private sealed class BoundedMemoryStream(long maxBytes, Func<Exception> refusal) : MemoryStream
        {
            private void Admit(int count)
            {
                if (Position + count > maxBytes) throw refusal();
            }

            public override void Write(ReadOnlySpan<byte> buffer)
            {
                Admit(buffer.Length);
                base.Write(buffer);
            }

            public override void Write(byte[] buffer, int offset, int count)
            {
                Admit(count);
                base.Write(buffer, offset, count);
            }

            public override void WriteByte(byte value)
            {
                Admit(1);
                base.WriteByte(value);
            }
        }

        /// <summary>
        /// Inverse of <see cref="SaveFastGraphToBinary(ComputationGraph, bool, int)"/>:
        /// deserialize .srk container bytes into a <see cref="ComputationGraph"/> via
        /// <see cref="OnnxModelImporter"/>, stamped with the header stage (or, for a header
        /// whose stage is unknown to this build, the op-scanned kind). The container is
        /// validated against its header (compression, payload SHA-256); data that is not a
        /// .srk container is refused (see <see cref="SrkFileFormat.Read"/>).
        /// </summary>
        /// <param name="data">.srk container bytes.</param>
        /// <param name="requiredStage">When set, refuse (with a clear stage-mismatch error)
        /// data whose graph is not of this <see cref="GraphKind"/> — e.g. reject a
        /// module-stage graph where a runnable concrete model is required. The header stage
        /// is checked before the payload is parsed.</param>
        public static ComputationGraph LoadFastGraphFromBinary(
            byte[] data, GraphKind? requiredStage = null)
        {
            var (graph, kind) = LoadFastGraphCore(data, origin: SrkFileFormat.InMemoryOrigin, requiredStage);
            return new ComputationGraph(graph, kind);
        }

        /// <summary>
        /// Shared load path: container payload extraction, header-first stage enforcement,
        /// and graph import. The returned kind is the header stage when the file carries a
        /// known one, else the op-scanned fallback
        /// (<see cref="SrkFileFormat.DetectStage(InternalComputationGraph)"/>), used when a
        /// header records a stage name this build does not define.
        /// </summary>
        internal static (InternalComputationGraph Graph, GraphKind Kind) LoadFastGraphCore(
            byte[] data, string origin, GraphKind? requiredStage)
        {
            if (data is null) throw new ArgumentNullException(nameof(data));
            using var container = new MemoryStream(data, writable: false);
            return LoadFastGraphCore(container, origin, requiredStage);
        }

        /// <summary>
        /// <see cref="LoadFastGraphCore(byte[], string, GraphKind?)"/> over the container
        /// <paramref name="container"/> holds from where it stands to its end, which must be
        /// seekable: its payload is hashed in one read through it (<see cref="SrkFileFormat.OpenPayload"/>)
        /// and parsed in a second (<see cref="OnnxStreamingReader"/>), never held whole, each weight
        /// read into storage of its own — past what one array holds, into host memory.
        /// </summary>
        internal static (InternalComputationGraph Graph, GraphKind Kind) LoadFastGraphCore(
            Stream container, string origin, GraphKind? requiredStage)
        {
            var (header, payload, maxLength) = SrkFileFormat.OpenPayload(container, origin);
            using var _ = payload;

            if (requiredStage is not null)
            {
                var stage = header.TryGetStage() ?? throw new InvalidDataException(
                    $"'{origin}': the .srk header records the unknown stage '{header.Stage}', " +
                    $"so the required '{SrkFileFormat.StageName(requiredStage.Value)}' stage " +
                    "cannot be verified.");
                SrkFileFormat.EnforceStage(stage, requiredStage.Value, origin);
            }

            InternalComputationGraph graph;
            GraphKind? taggedKind;
            try
            {
                (graph, taggedKind) = OnnxModelImporter.FromModelProtoWithKindTag(
                    OnnxStreamingReader.ReadModel(payload, maxLength, origin));
            }
            catch (Exception e) when (e is ProtoBuf.ProtoException
                or EndOfStreamException
                or IndexOutOfRangeException
                or ArgumentOutOfRangeException
                or OverflowException
                or FormatException)
            {
                // These are the exceptions the protobuf/ONNX layer raises on malformed payload
                // bytes — garbage or truncation. Name the file and the cause instead of surfacing a bare
                // deep-in-the-importer exception. Deliberately NOT catching Exception broadly:
                // a NullReferenceException/InvalidOperationException/NotSupportedException from a
                // valid-but-unsupported graph (or a framework bug), and OutOfMemoryException, must
                // propagate as themselves rather than be mislabeled "corrupt file".
                throw new InvalidDataException(
                    $"'{origin}': not a readable Shorokoo graph file — failed to parse the ONNX payload " +
                    $"({e.GetType().Name}: {e.Message}). The file is corrupt or not a .srk file.", e);
            }

            return (graph, header.TryGetStage() ?? taggedKind ?? SrkFileFormat.DetectStage(graph));
        }

        /// <summary>
        /// Save <paramref name="graph"/> directly to <paramref name="filename"/> as a
        /// .srk container (see <see cref="SaveFastGraphToBinary(ComputationGraph, bool, int)"/>). Inverse of
        /// <see cref="LoadFastGraphFromFile"/>. With <paramref name="overrideExtension"/>
        /// the extension is normalized to .zsrk/.srk purely as a hint for humans — the
        /// extension has no parsing significance; the header records the compression.
        /// The container is streamed to the file, each weight from where it lies, so a graph's
        /// weights may be of any size, a weight past what one array holds included, and may total
        /// any size. The model besides each tensor's <c>raw_data</c> of at least 1,024 bytes — its
        /// structure, smaller <c>raw_data</c>, string tensors and typed data fields — is built in
        /// one array, and a graph whose model besides those exceeds <see cref="Array.MaxLength"/>
        /// bytes is refused with <see cref="NotSupportedException"/> before anything is written.
        /// The write is atomic: the container is staged in a <c>.tmp-</c> sibling and committed
        /// by rename, so a failed or interrupted save leaves any previous file untouched.
        /// </summary>
        public static string SaveFastGraphToFile(
            string filename, ComputationGraph graph, bool compressed = true,
            bool overrideExtension = true, int compressionLevel = DefaultCompressionLevel)
            => SaveFastGraphToFile(filename, graph.ToInternal(), graph.Kind, compressed,
                overrideExtension, compressionLevel);

        /// <summary>
        /// Internal-graph form of
        /// <see cref="SaveFastGraphToFile(string, ComputationGraph, bool, bool, int)"/>; the
        /// header stage is <paramref name="stage"/> when given, else op-scanned.
        /// </summary>
        internal static string SaveFastGraphToFile(
            string filename, InternalComputationGraph graph, GraphKind? stage = null,
            bool compressed = true, bool overrideExtension = true,
            int compressionLevel = DefaultCompressionLevel)
        {
            filename = Path.GetFullPath(filename);

            if (overrideExtension)
                filename = Path.ChangeExtension(filename,
                    compressed ? CompressedArchitectureExtension : UncompressedArchitectureExtension);

            string? directoryPath = Path.GetDirectoryName(filename);
            if (directoryPath is not null)
                Directory.CreateDirectory(directoryPath);

            var srk = SrkPayload.Of(graph, stage);
            AtomicFileWriter.WriteFile(filename, stream => srk.WriteContainer(stream, compressed, compressionLevel));
            return filename;
        }

        /// <summary>
        /// Load a <see cref="ComputationGraph"/> from a .srk container file —
        /// the content decides how the file parses, never the extension, so a renamed
        /// file loads identically. See <see cref="LoadFastGraphFromBinary"/> for the
        /// container handling and the <paramref name="requiredStage"/> contract; errors
        /// name <paramref name="filename"/>. The file is read as a stream — once to verify
        /// its payload hash, once to parse the payload — and never held whole, so a file of
        /// any size loads; a weight past what one array holds is read into host memory of the
        /// backend <see cref="ComputeContext.Default"/> runs on.
        /// </summary>
        public static ComputationGraph LoadFastGraphFromFile(
            string filename, GraphKind? requiredStage = null)
        {
            using var file = OpenSrkFile(filename);
            var (graph, kind) = LoadFastGraphCore(file, filename, requiredStage);
            return new ComputationGraph(graph, kind);
        }

        private static FileStream OpenSrkFile(string filePath)
            => new(filePath, FileMode.Open, FileAccess.Read, FileShare.Read, bufferSize: 1 << 16);

        /// <summary>
        /// Parses the ONNX model out of a .srk container file, by content, streaming the payload.
        /// Shared by the JSON/introspection helpers below, which parse the ModelProto without
        /// building a full <see cref="InternalComputationGraph"/>.
        /// </summary>
        private static ModelProto ReadArchitectureModelFromFile(string filePath)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException($"Architecture file not found: {filePath}");
            using var file = OpenSrkFile(filePath);
            var (_, payload, maxLength) = SrkFileFormat.OpenPayload(file, filePath);
            using (payload)
                return OnnxStreamingReader.ReadModel(payload, maxLength, filePath);
        }

        /// <summary>
        /// Generates a text listing of all node names and tensor names found in an
        /// architecture file (a .srk container), by deserializing into intermediate JSON
        /// objects rather than a full <see cref="InternalComputationGraph"/>.
        ///
        /// The returned string lists all node names first, followed by an empty line, then all
        /// tensor names. Names are listed in the order they are first encountered when scanning
        /// the JSON representation of the file.
        /// </summary>
        /// <param name="filePath">Path to the architecture file (a .srk container)</param>
        /// <returns>Formatted text listing of node names and tensor names</returns>
        public static string GetNodeAndTensorNameListing(string filePath)
        {
            // Deserialize to IR.ModelProto — do NOT go all the way to InternalComputationGraph
            var model = ReadArchitectureModelFromFile(filePath);

            // Clear raw data so the JSON serialization stays compact
            foreach (var initializer in model.Graph.Initializers)
                initializer.ResetRawData();

            // Serialize to JSON then parse with JsonDocument for generic traversal
            var json = JsonSerializer.Serialize(model);
            using var doc = JsonDocument.Parse(json);
            var graph = doc.RootElement.GetProperty("Graph");

            // Collect node names in the order they appear in Graph.Nodes[*].Name
            var nodeNames = new List<string>();
            foreach (var node in graph.GetProperty("Nodes").EnumerateArray())
            {
                if (node.TryGetProperty("Name", out var nameEl)
                    && nameEl.ValueKind == JsonValueKind.String)
                {
                    var name = nameEl.GetString();
                    if (!string.IsNullOrEmpty(name))
                        nodeNames.Add(name);
                }
            }

            // Collect tensor names in first-occurrence order as encountered in the JSON:
            //   1. Each node's Inputs and Outputs arrays (in node array order)
            //   2. Graph.Initializers[*].Name
            //   3. Graph.Inputs[*].Name
            //   4. Graph.Outputs[*].Name
            var tensorNames = new List<string>();
            var seenTensors = new HashSet<string>();

            void AddTensorName(string? name)
            {
                if (!string.IsNullOrEmpty(name) && seenTensors.Add(name))
                    tensorNames.Add(name);
            }

            foreach (var node in graph.GetProperty("Nodes").EnumerateArray())
            {
                if (node.TryGetProperty("Inputs", out var inputs))
                    foreach (var input in inputs.EnumerateArray())
                        if (input.ValueKind == JsonValueKind.String)
                            AddTensorName(input.GetString());

                if (node.TryGetProperty("Outputs", out var outputs))
                    foreach (var output in outputs.EnumerateArray())
                        if (output.ValueKind == JsonValueKind.String)
                            AddTensorName(output.GetString());
            }

            if (graph.TryGetProperty("Initializers", out var initializers))
                foreach (var init in initializers.EnumerateArray())
                    if (init.TryGetProperty("Name", out var nameEl)
                        && nameEl.ValueKind == JsonValueKind.String)
                        AddTensorName(nameEl.GetString());

            if (graph.TryGetProperty("Inputs", out var graphInputs))
                foreach (var inp in graphInputs.EnumerateArray())
                    if (inp.TryGetProperty("Name", out var nameEl)
                        && nameEl.ValueKind == JsonValueKind.String)
                        AddTensorName(nameEl.GetString());

            if (graph.TryGetProperty("Outputs", out var graphOutputs))
                foreach (var outp in graphOutputs.EnumerateArray())
                    if (outp.TryGetProperty("Name", out var nameEl)
                        && nameEl.ValueKind == JsonValueKind.String)
                        AddTensorName(nameEl.GetString());

            // Format: node names, empty line, tensor names (no trailing newline)
            var lines = new List<string>(nodeNames.Count + 1 + tensorNames.Count);
            lines.AddRange(nodeNames);
            lines.Add(string.Empty);
            lines.AddRange(tensorNames);

            return string.Join(Environment.NewLine, lines);
        }

        /// <summary>
        /// Converts a compressed Shorokoo architecture file (.zsrk) to a pretty-printed JSON
        /// string. Raw tensor data is stripped before serialization so the output stays
        /// human-readable and compact.
        ///
        /// This is primarily useful for diffing two runs of a generation step to identify
        /// which fields or nodes differ between runs.
        /// </summary>
        /// <param name="filePath">Path to the architecture file (a .srk container) to convert.</param>
        /// <returns>Pretty-printed JSON string representing the ModelProto.</returns>
        public static string ToJson(string filePath)
        {
            var model = ReadArchitectureModelFromFile(filePath);

            // Strip all raw tensor data so the JSON stays compact and human-readable.
            // This covers initializer tensors, inline attribute tensors (e.g. Constant nodes),
            // and tensors nested inside function graphs.
            StripRawData(model.Graph);
            foreach (var function in model.Functions)
                StripRawData(function.Nodes);

            var options = new JsonSerializerOptions { WriteIndented = true };
            return JsonSerializer.Serialize(model, options);
        }

        /// <summary>Clears RawData from all initializers and node attribute tensors in a graph.</summary>
        private static void StripRawData(GraphProto graph)
        {
            foreach (var initializer in graph.Initializers)
                initializer.ResetRawData();

            StripRawData(graph.Nodes);
        }

        /// <summary>Clears RawData from TensorProto attributes embedded in a collection of nodes.</summary>
        private static void StripRawData(IEnumerable<NodeProto> nodes)
        {
            foreach (var node in nodes)
                foreach (var attr in node.Attributes)
                    attr.T?.ResetRawData();
        }

        /// <summary>
        /// Converts a compressed Shorokoo architecture file (.zsrk) to JSON and writes it to
        /// <paramref name="targetPath"/>. If <paramref name="targetPath"/> is <c>null</c> the
        /// output path is derived from <paramref name="sourcePath"/> by replacing its extension
        /// with <c>.json</c>.
        /// </summary>
        /// <param name="sourcePath">Path to the .zsrk file.</param>
        /// <param name="targetPath">
        /// Optional explicit path for the output .json file. When omitted the file is saved
        /// next to the source file with a <c>.json</c> extension.
        /// </param>
        /// <returns>The path where the JSON file was written.</returns>
        /// <remarks>The write is atomic (staged beside the target and committed by rename), so a
        /// failed conversion leaves any previous file at the target path untouched; the target's
        /// directory must already exist.</remarks>
        public static string SaveAsJson(string sourcePath, string? targetPath = null)
        {
            targetPath ??= Path.ChangeExtension(sourcePath, JsonArchitectureExtension);
            var json = ToJson(sourcePath);
            var bytes = Encoding.UTF8.GetBytes(json);
            AtomicFileWriter.WriteFile(targetPath, stream => stream.Write(bytes));
            return targetPath;
        }

        /// <summary>
        /// Compares two Shorokoo architecture files (compressed .zsrk or uncompressed .bin) by
        /// their JSON representations after stripping all raw tensor data.  This provides a
        /// human-friendly, structure-only comparison that is stable across serialization runs
        /// and useful for debugging regressions in architecture generation.
        /// </summary>
        /// <param name="pathA">Path to the first architecture file.</param>
        /// <param name="pathB">Path to the second architecture file.</param>
        /// <returns>
        /// <c>true</c> when the two files are structurally identical (same JSON); <c>false</c>
        /// otherwise.
        /// </returns>
        public static bool CompareJson(string pathA, string pathB)
        {
            var jsonA = ToJson(pathA);
            var jsonB = ToJson(pathB);
            return string.Equals(jsonA, jsonB, StringComparison.Ordinal);
        }

        /// <summary>
        /// Compares two Shorokoo architecture files and returns the first differing line, or
        /// <c>null</c> when the files are identical.  Useful for quickly locating structural
        /// divergences between two serialization runs.
        /// </summary>
        /// <param name="pathA">Path to the first architecture file.</param>
        /// <param name="pathB">Path to the second architecture file.</param>
        /// <returns>
        /// A tuple <c>(lineNumber, lineA, lineB)</c> describing the first line that differs, or
        /// <c>null</c> when the JSON representations are identical.
        /// </returns>
        public static (int LineNumber, string LineA, string LineB)? FindFirstJsonDiff(string pathA, string pathB)
        {
            var jsonA = ToJson(pathA);
            var jsonB = ToJson(pathB);
            if (string.Equals(jsonA, jsonB, StringComparison.Ordinal))
                return null;

            var linesA = jsonA.Split('\n');
            var linesB = jsonB.Split('\n');
            int minCount = Math.Min(linesA.Length, linesB.Length);
            for (int i = 0; i < minCount; i++)
            {
                if (!string.Equals(linesA[i], linesB[i], StringComparison.Ordinal))
                    return (i + 1, linesA[i], linesB[i]);
            }

            // One file has more lines than the other
            if (linesA.Length != linesB.Length)
            {
                int lineNumber = minCount + 1;
                string lineA = linesA.Length > minCount ? linesA[minCount] : "<end>";
                string lineB = linesB.Length > minCount ? linesB[minCount] : "<end>";
                return (lineNumber, lineA, lineB);
            }

            return null;
        }

        #endregion

        #region Generic Compression Utilities

        /// <summary>
        /// Decompress a Zstandard-compressed file, read as a stream (see <see cref="DecompressStream(Stream)"/>).
        /// </summary>
        /// <param name="filePath">Path to the compressed file</param>
        /// <returns>Decompressed byte array</returns>
        public static byte[] DecompressFile(string filePath)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException($"Compressed file not found: {filePath}");

            using var file = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read,
                bufferSize: 1 << 16, FileOptions.SequentialScan);
            return DecompressStream(file);
        }

        /// <summary>
        /// Decompress a Zstandard-compressed stream, decoding it as it is read. The result is one
        /// array, so decompressed data larger than an array holds (<see cref="Array.MaxLength"/>
        /// bytes) is refused with <see cref="NotSupportedException"/> as soon as it passes that.
        /// </summary>
        /// <param name="stream">Stream containing compressed data</param>
        /// <returns>Decompressed byte array</returns>
        public static byte[] DecompressStream(Stream stream)
            => DecompressStream(stream, Array.MaxLength);

        internal static byte[] DecompressStream(Stream stream, long maxBytes)
        {
            using var decompressed = new BoundedMemoryStream(maxBytes, () => new NotSupportedException(
                $"The decompressed data is more than the {maxBytes:N0} bytes one array holds. " +
                "Read it as a stream with ZstdSharp.DecompressionStream instead."));
            using (var decoder = new DecompressionStream(stream, leaveOpen: true))
                decoder.CopyTo(decompressed);
            return decompressed.ToArray();
        }

        /// <summary>
        /// Decompress Zstandard-compressed bytes
        /// </summary>
        /// <param name="compressedBytes">Compressed byte array</param>
        /// <returns>Decompressed byte array</returns>
        public static byte[] Decompress(byte[] compressedBytes)
            => Decompress((ReadOnlySpan<byte>)compressedBytes);

        /// <summary>
        /// Decompress Zstandard-compressed bytes from a span, without first copying them into a
        /// dedicated array — lets callers unwrap a slice of a larger buffer (e.g. a container
        /// payload) with no intermediate allocation.
        /// </summary>
        public static byte[] Decompress(ReadOnlySpan<byte> compressedBytes)
        {
            using var decompressor = new Decompressor();
            return decompressor.Unwrap(compressedBytes).ToArray();
        }

        /// <summary>
        /// Compress bytes using Zstandard and write to a file. The write is atomic (staged
        /// beside the target and committed by rename), so a failed or interrupted write leaves
        /// any previous file untouched; the target's directory must already exist.
        /// </summary>
        /// <param name="filePath">Path for the output file</param>
        /// <param name="uncompressedBytes">Bytes to compress</param>
        /// <param name="compressionLevel">Zstandard compression level (1-22, default: 3)</param>
        public static void CompressToFile(string filePath, byte[] uncompressedBytes, int compressionLevel = DefaultCompressionLevel)
        {
            AtomicFileWriter.WriteFile(filePath, stream => CompressToStream(stream, uncompressedBytes, compressionLevel));
        }

        /// <summary>
        /// Compress bytes using Zstandard and write to a stream, as one frame declaring its
        /// decompressed size, compressed straight into the stream with no compressed copy held.
        /// </summary>
        /// <param name="stream">Stream to write the compressed data to</param>
        /// <param name="uncompressedBytes">Bytes to compress</param>
        /// <param name="compressionLevel">Zstandard compression level (1-22, default: 3)</param>
        public static void CompressToStream(Stream stream, byte[] uncompressedBytes, int compressionLevel = DefaultCompressionLevel)
        {
            if (uncompressedBytes is null) throw new ArgumentNullException(nameof(uncompressedBytes));
            WriteZstdFrame(stream, compressionLevel, uncompressedBytes.LongLength, frame => frame.Write(uncompressedBytes));
        }

        /// <summary>
        /// Compress bytes using Zstandard
        /// </summary>
        /// <param name="uncompressedBytes">Bytes to compress</param>
        /// <param name="compressionLevel">Zstandard compression level (1-22, default: 3)</param>
        /// <returns>Compressed byte array</returns>
        public static byte[] Compress(byte[] uncompressedBytes, int compressionLevel = DefaultCompressionLevel)
        {
            using var compressor = new Compressor(compressionLevel);
            return compressor.Wrap(uncompressedBytes).ToArray();
        }

        /// <summary>The most bytes a Zstd frame header takes: magic, descriptor, window, dictionary
        /// id and content size.</summary>
        internal const int ZstdFrameHeaderMaxBytes = 18;

        /// <summary>
        /// The decompressed size the Zstd frames of <paramref name="frames"/>, from where it stands to
        /// its end, declare between them: what bounds a streamed payload's header before a byte of it
        /// is believed. Every frame Shorokoo writes declares its size (<see cref="WriteZstdFrame(Stream, int, long, Action{Stream})"/>),
        /// and a tool that writes a payload in several frames — pzstd, say — declares each one's, so
        /// bytes that are not Zstd frames each declaring its size are refused, with the exception
        /// <paramref name="malformed"/> makes of the reason, as is a frame declaring more than its
        /// blocks hold. A skippable frame holds no payload and adds nothing.
        ///
        /// <para>Each frame is walked by its header and its blocks' headers, three bytes a block, the
        /// blocks themselves sought past where the stream can seek and read past where it cannot: no
        /// byte is decompressed, and a file is read through at most once.</para>
        /// </summary>
        internal static long DeclaredZstdContentSize(Stream frames, Func<string, Exception> malformed)
            => ZstdContentSize(frames, malformed, requireDeclared: true);

        /// <summary>
        /// The most bytes the Zstd frames of <paramref name="frames"/>, from where it stands to its
        /// end, decompress to, walked as <see cref="DeclaredZstdContentSize"/> walks them: each
        /// frame's declared size, or for a frame declaring none the most its blocks hold — a raw or
        /// RLE block its stated size, a compressed one <see cref="ZstdMaxBlockBytes"/>. What a
        /// streamed payload's fields declare is held to it before anything they claim is allocated.
        /// </summary>
        internal static long ZstdContentSizeBound(Stream frames, Func<string, Exception> malformed)
            => ZstdContentSize(frames, malformed, requireDeclared: false);

        /// <summary>The most bytes one Zstd block decompresses to.</summary>
        internal const int ZstdMaxBlockBytes = 1 << 17;

        private static long ZstdContentSize(Stream frames, Func<string, Exception> malformed, bool requireDeclared)
        {
            Span<byte> field = stackalloc byte[8];
            void Read(Span<byte> bytes)
            {
                if (frames.ReadAtLeast(bytes, bytes.Length, throwOnEndOfStream: false) < bytes.Length)
                    throw malformed("its Zstd frames are cut short");
            }
            void Pass(long count)
            {
                if (frames.CanSeek)
                {
                    if (count > frames.Length - frames.Position) throw malformed("its Zstd frames are cut short");
                    frames.Seek(count, SeekOrigin.Current);
                    return;
                }
                var discard = new byte[(int)Math.Min(count, 1 << 16)];
                for (long left = count; left > 0; left -= discard.Length)
                    Read(discard.AsSpan(0, (int)Math.Min(left, discard.Length)));
            }

            long total = 0;
            for (int read = 0; ; read++)
            {
                int got = frames.ReadAtLeast(field[..4], 4, throwOnEndOfStream: false);
                if (got == 0 && read > 0) return total;
                if (got < 4) throw malformed(read == 0 ? "its bytes are not a Zstd frame" : "its Zstd frames are cut short");
                uint magic = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(field);
                if ((magic & 0xFFFFFFF0) == 0x184D2A50)
                {
                    Read(field[..4]);
                    Pass(System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(field));
                    read--;
                    continue;
                }
                if (magic != 0xFD2FB528)
                    throw malformed(read == 0 ? "its bytes are not a Zstd frame" : "it holds bytes after its Zstd frames that are not one");

                Read(field[..1]);
                int descriptor = field[0];
                if ((descriptor & 0x08) != 0) throw malformed("a Zstd frame header of it is malformed");
                bool singleSegment = (descriptor & 0x20) != 0;
                Pass((singleSegment ? 0 : 1) + (descriptor & 3) switch { 0 => 0, 1 => 1, 2 => 2, _ => 4 });
                int sizeBytes = (descriptor >> 6) switch { 0 => singleSegment ? 1 : 0, 1 => 2, 2 => 4, _ => 8 };
                if (sizeBytes == 0 && requireDeclared)
                    throw malformed("a Zstd frame of it declares no decompressed size, which every frame written here declares");
                field.Clear();
                Read(field[..sizeBytes]);
                ulong? declared = sizeBytes == 0 ? null
                    : System.Buffers.Binary.BinaryPrimitives.ReadUInt64LittleEndian(field) + (sizeBytes == 2 ? 256UL : 0);

                ulong held = 0;
                while (true)
                {
                    Read(field[..3]);
                    int header = field[0] | field[1] << 8 | field[2] << 16;
                    int type = header >> 1 & 3;
                    if (type == 3) throw malformed("a Zstd block of it is of a reserved type");
                    Pass(type == 1 ? 1 : header >> 3);
                    held += type == 2 ? ZstdMaxBlockBytes : (ulong)(header >> 3);
                    if ((header & 1) != 0) break;
                }
                if ((descriptor & 0x04) != 0) Pass(4);

                if (declared > held)
                    throw malformed("a Zstd frame of it declares more bytes than its blocks hold");
                ulong size = declared ?? held;
                if (size > (ulong)(long.MaxValue - total))
                    throw malformed("its Zstd frames declare more bytes than a payload holds");
                total += (long)size;
            }
        }

        /// <summary>
        /// Writes what <paramref name="produce"/> writes to <paramref name="destination"/> as one
        /// Zstd frame at <paramref name="level"/> whose header declares its decompressed size, so a
        /// reader can hold the payload's own header to it before allocating what that header claims
        /// (<see cref="DeclaredZstdContentSize"/>). <paramref name="produce"/> runs twice: once into
        /// a stream that only counts, to learn the size, then into the compressor. The frame is a
        /// pure function of the bytes produced and the level, so writing it again writes it
        /// identically.
        /// </summary>
        internal static void WriteZstdFrame(Stream destination, int level, Action<Stream> produce)
        {
            var counter = new LengthCountingStream();
            produce(counter);
            WriteZstdFrame(destination, level, counter.Length, produce);
        }

        /// <summary>
        /// <see cref="WriteZstdFrame(Stream, int, Action{Stream})"/> for a producer whose length is
        /// known: <paramref name="produce"/> runs once, straight into the compressor, and must write
        /// exactly <paramref name="length"/> bytes — the compressor refuses the frame otherwise.
        /// </summary>
        internal static void WriteZstdFrame(Stream destination, int level, long length, Action<Stream> produce)
        {
            using var frame = new PledgedZstdFrameStream(destination, level, length);
            produce(frame);
            frame.Finish();
        }

        /// <summary>A write-only stream that keeps nothing and only counts, told a payload's length
        /// rather than given it wherever the writer can (<see cref="ILengthOnlyStream"/>).</summary>
        private sealed class LengthCountingStream : Stream, ILengthOnlyStream
        {
            private long _length;
            public bool IsLengthOnly => true;
            public void Advance(long count) => _length += count;
            public override void Write(ReadOnlySpan<byte> buffer) => _length += buffer.Length;
            public override void Write(byte[] buffer, int offset, int count) => _length += count;
            public override long Length => _length;
            public override bool CanRead => false;
            public override bool CanSeek => false;
            public override bool CanWrite => true;
            public override long Position { get => _length; set => throw new NotSupportedException(); }
            public override void Flush() { }
            public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
            public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
            public override void SetLength(long value) => throw new NotSupportedException();
        }

        /// <summary>
        /// A write-only stream compressing into one Zstd frame whose size is pledged up front, which
        /// is what puts the size in the frame header — the library's own compression stream offers
        /// no pledge. The frame is closed by <see cref="Finish"/>, not by disposal: a producer that
        /// failed part-way has written less than was pledged, and closing the frame then would only
        /// replace its failure with the compressor's.
        /// </summary>
        private sealed unsafe class PledgedZstdFrameStream : Stream
        {
            private readonly Stream _destination;
            private readonly byte[] _out = new byte[(int)ZstdSharp.Unsafe.Methods.ZSTD_CStreamOutSize()];
            private ZstdSharp.Unsafe.ZSTD_CCtx_s* _cctx;

            public PledgedZstdFrameStream(Stream destination, int level, long pledgedSize)
            {
                _destination = destination;
                _cctx = ZstdSharp.Unsafe.Methods.ZSTD_createCCtx();
                if (_cctx is null) throw new OutOfMemoryException("Zstd could not allocate a compression context.");
                try
                {
                    Check(ZstdSharp.Unsafe.Methods.ZSTD_CCtx_setParameter(
                        _cctx, ZstdSharp.Unsafe.ZSTD_cParameter.ZSTD_c_compressionLevel, level));
                    Check(ZstdSharp.Unsafe.Methods.ZSTD_CCtx_setPledgedSrcSize(_cctx, (ulong)pledgedSize));
                }
                catch
                {
                    Free();
                    throw;
                }
            }

            public override void Write(ReadOnlySpan<byte> buffer)
            {
                fixed (byte* src = buffer)
                {
                    var input = new ZstdSharp.Unsafe.ZSTD_inBuffer_s { src = src, size = (nuint)buffer.Length, pos = 0 };
                    while (input.pos < input.size)
                        Compress(&input, ZstdSharp.Unsafe.ZSTD_EndDirective.ZSTD_e_continue);
                }
            }

            public override void Write(byte[] buffer, int offset, int count) => Write(buffer.AsSpan(offset, count));
            public override void WriteByte(byte value) => Write([value]);

            /// <summary>Closes the frame, writing whatever the compressor still holds.</summary>
            public void Finish()
            {
                var input = new ZstdSharp.Unsafe.ZSTD_inBuffer_s { src = null, size = 0, pos = 0 };
                while (Compress(&input, ZstdSharp.Unsafe.ZSTD_EndDirective.ZSTD_e_end) != 0) { }
            }

            private nuint Compress(ZstdSharp.Unsafe.ZSTD_inBuffer_s* input, ZstdSharp.Unsafe.ZSTD_EndDirective directive)
            {
                ObjectDisposedException.ThrowIf(_cctx is null, this);
                nuint remaining;
                int produced;
                fixed (byte* dst = _out)
                {
                    var output = new ZstdSharp.Unsafe.ZSTD_outBuffer_s { dst = dst, size = (nuint)_out.Length, pos = 0 };
                    remaining = Check(ZstdSharp.Unsafe.Methods.ZSTD_compressStream2(_cctx, &output, input, directive));
                    produced = (int)output.pos;
                }
                _destination.Write(_out, 0, produced);
                return remaining;
            }

            private static nuint Check(nuint result)
                => ZstdSharp.Unsafe.Methods.ZSTD_isError(result)
                    ? throw new InvalidOperationException(
                        $"Zstd compression failed: {ZstdSharp.Unsafe.Methods.ZSTD_getErrorName(result)}")
                    : result;

            private void Free()
            {
                if (_cctx is null) return;
                ZstdSharp.Unsafe.Methods.ZSTD_freeCCtx(_cctx);
                _cctx = null;
            }

            protected override void Dispose(bool disposing)
            {
                Free();
                base.Dispose(disposing);
            }

            ~PledgedZstdFrameStream() => Free();

            public override bool CanRead => false;
            public override bool CanSeek => false;
            public override bool CanWrite => true;
            public override long Length => throw new NotSupportedException();
            public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
            public override void Flush() { }
            public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
            public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
            public override void SetLength(long value) => throw new NotSupportedException();
        }

        #endregion

        #region File Format Detection

        /// <summary>
        /// Determines if a file path is for a compressed safetensor file
        /// </summary>
        public static bool IsCompressedSafeTensor(string filePath)
            => filePath.EndsWith(CompressedSafeTensorExtension, StringComparison.OrdinalIgnoreCase);

        // Note: there is deliberately no IsCompressedArchitecture(path) helper. A .srk file's
        // compression is declared in its header, never implied by the .srk/.zsrk extension, so
        // an extension-based predicate would misrepresent the format contract. Read the header
        // with SrkFileFormat.TryReadHeaderFromFile when the compression must be known.

        #endregion
    }
}

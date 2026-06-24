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
using ZstdSharp;

namespace Shorokoo.Core.Utils
{
    /// <summary>
    /// Utility class for loading and saving compressed Shorokoo data formats.
    /// Supports:
    /// - .zsafetensor: Zstandard-compressed safetensors files
    /// - .zsrk: Zstandard-compressed Shorokoo architecture (.bin) files
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
            var decompressedBytes = DecompressFile(filePath);
            var tensors = SafeTensorLoader.ParseSafeTensorBytes(decompressedBytes);
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
            var decompressedBytes = DecompressFile(filePath);
            return SafeTensorLoader.ParseSafeTensorBytes(decompressedBytes);
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
        /// Save tensors to a compressed SafeTensor file (.zsafetensor)
        /// </summary>
        /// <param name="filePath">Path for the output .zsafetensor file</param>
        /// <param name="tensors">List of SafeTensor objects to save</param>
        /// <param name="globalMetadata">Optional global metadata to include</param>
        /// <param name="compressionLevel">Zstandard compression level (1-22, default: 3)</param>
        public static void SaveCompressedSafeTensors(string filePath, List<SafeTensor> tensors, Dictionary<string, object>? globalMetadata = null, int compressionLevel = DefaultCompressionLevel)
        {
            // First save to an uncompressed memory stream
            using var uncompressedStream = new MemoryStream();
            SafeTensorLoader.SaveSafeTensorsToStream(uncompressedStream, tensors, globalMetadata);
            var uncompressedBytes = uncompressedStream.ToArray();

            // Compress and write to file
            CompressToFile(filePath, uncompressedBytes, compressionLevel);
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

        #region Compressed Architecture Loading

        /// <summary>
        /// Load a compressed Shorokoo architecture file (.zsrk)
        /// </summary>
        /// <param name="filePath">Path to the .zsrk file</param>
        /// <returns><see cref="FastComputationGraph"/> loaded from the compressed architecture file</returns>
        public static FastComputationGraph LoadCompressedArchitecture(string filePath)
        {
            var decompressedBytes = DecompressFile(filePath);
            return LoadFastGraphFromBinary(decompressedBytes);
        }

        /// <summary>
        /// Load a compressed Shorokoo architecture file (.zsrk) from a stream
        /// </summary>
        /// <param name="stream">Stream containing the compressed .zsrk data</param>
        /// <returns><see cref="FastComputationGraph"/> loaded from the compressed architecture</returns>
        public static FastComputationGraph LoadCompressedArchitectureFromStream(Stream stream)
        {
            var decompressedBytes = DecompressStream(stream);
            return LoadFastGraphFromBinary(decompressedBytes);
        }

        /// <summary>
        /// Serialize <paramref name="graph"/> to ONNX bytes via
        /// <see cref="FastOnnxModelBuilder"/>, optionally Zstd-compressing the result.
        /// .zsrk archives produced by <see cref="SaveCompressedArchitecture"/> wrap an
        /// already-compressed payload (with <paramref name="compressed"/>=true) inside
        /// another Zstd layer; that outer layer is added by
        /// <see cref="CompressToFile"/>.
        /// </summary>
        public static byte[] SaveFastGraphToBinary(FastComputationGraph graph, bool compressed = true)
        {
            using var memoryStream = new MemoryStream();
            var model = FastOnnxModelBuilder.BuildOnnxModel(graph);
            Serializer.Serialize(memoryStream, model);
            var data = memoryStream.ToArray();
            return compressed ? Compress(data) : data;
        }

        /// <summary>
        /// Inverse of <see cref="SaveFastGraphToBinary"/>: deserialize ONNX bytes (with
        /// optional Zstd decompression) into a <see cref="FastComputationGraph"/> via
        /// <see cref="OnnxModelImporter"/>.
        /// </summary>
        public static FastComputationGraph LoadFastGraphFromBinary(byte[] data, bool isCompressed = true)
        {
            if (isCompressed)
                data = Decompress(data);
            return OnnxModelImporter.FromOnnxModelToFastGraph(data);
        }

        /// <summary>
        /// Save <paramref name="graph"/> directly to <paramref name="filename"/> as a
        /// single-Zstd-wrapped ONNX payload. Inverse of
        /// <see cref="LoadFastGraphFromFile"/>. Note: distinct on-disk format from
        /// <see cref="SaveCompressedArchitecture"/>, which adds an additional outer
        /// Zstd layer.
        /// </summary>
        public static string SaveFastGraphToFile(string filename, FastComputationGraph graph, bool compressed = true, bool overrideExtension = true)
        {
            if (File.Exists(filename))
                File.Delete(filename);

            filename = Path.GetFullPath(filename);

            if (overrideExtension)
                filename = Path.ChangeExtension(filename,
                    compressed ? CompressedArchitectureExtension : UncompressedArchitectureExtension);

            string? directoryPath = Path.GetDirectoryName(filename);
            if (directoryPath is not null)
                Directory.CreateDirectory(directoryPath);

            File.WriteAllBytes(filename, SaveFastGraphToBinary(graph, compressed));
            return filename;
        }

        /// <summary>
        /// Load a <see cref="FastComputationGraph"/> from a file written by
        /// <see cref="SaveFastGraphToFile"/>. Compression is auto-detected from the
        /// file extension when <paramref name="isCompressed"/> is <c>null</c>.
        /// </summary>
        public static FastComputationGraph LoadFastGraphFromFile(string filename, bool? isCompressed = null)
        {
            var actualIsCompressed = isCompressed ?? IsCompressedArchitecture(filename);
            var allData = File.ReadAllBytes(filename);
            ThrowIfGitLfsPointer(filename, allData);
            return LoadFastGraphFromBinary(allData, actualIsCompressed);
        }

        /// <summary>
        /// Generates a text listing of all node names and tensor names found in a compressed
        /// architecture file (.zsrk), by deserializing into intermediate JSON objects rather
        /// than a full <see cref="FastComputationGraph"/>.
        ///
        /// The returned string lists all node names first, followed by an empty line, then all
        /// tensor names. Names are listed in the order they are first encountered when scanning
        /// the JSON representation of the file.
        /// </summary>
        /// <param name="filePath">Path to the .zsrk file</param>
        /// <returns>Formatted text listing of node names and tensor names</returns>
        public static string GetNodeAndTensorNameListing(string filePath)
        {
            var decompressedBytes = DecompressFile(filePath);

            // Deserialize to IR.ModelProto — do NOT go all the way to FastComputationGraph
            ModelProto model;
            using (var ms = new MemoryStream(decompressedBytes))
            {
                model = Serializer.Deserialize<ModelProto>(ms);
            }

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
        /// <param name="filePath">Path to the .zsrk file to convert.</param>
        /// <returns>Pretty-printed JSON string representing the ModelProto.</returns>
        public static string ToJson(string filePath)
        {
            var decompressedBytes = DecompressFile(filePath);

            ModelProto model;
            using (var ms = new MemoryStream(decompressedBytes))
            {
                model = Serializer.Deserialize<ModelProto>(ms);
            }

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
        public static string SaveAsJson(string sourcePath, string? targetPath = null)
        {
            targetPath ??= Path.ChangeExtension(sourcePath, JsonArchitectureExtension);
            var json = ToJson(sourcePath);
            File.WriteAllText(targetPath, json);
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

        #region Compressed Architecture Saving

        /// <summary>
        /// Save a <see cref="FastComputationGraph"/> to a compressed architecture file (.zsrk)
        /// </summary>
        /// <param name="filePath">Path for the output .zsrk file</param>
        /// <param name="graph"><see cref="FastComputationGraph"/> to save</param>
        /// <param name="compressionLevel">Zstandard compression level (1-22, default: 3)</param>
        public static void SaveCompressedArchitecture(string filePath, FastComputationGraph graph, int compressionLevel = DefaultCompressionLevel)
        {
            var uncompressedBytes = SaveFastGraphToBinary(graph);
            CompressToFile(filePath, uncompressedBytes, compressionLevel);
        }

        /// <summary>
        /// Save a <see cref="FastComputationGraph"/> to a compressed architecture stream (.zsrk)
        /// </summary>
        /// <param name="stream">Stream to write the compressed data to</param>
        /// <param name="graph"><see cref="FastComputationGraph"/> to save</param>
        /// <param name="compressionLevel">Zstandard compression level (1-22, default: 3)</param>
        public static void SaveCompressedArchitectureToStream(Stream stream, FastComputationGraph graph, int compressionLevel = DefaultCompressionLevel)
        {
            var uncompressedBytes = SaveFastGraphToBinary(graph);
            CompressToStream(stream, uncompressedBytes, compressionLevel);
        }

        #endregion

        #region Generic Compression Utilities

        /// <summary>
        /// Decompress a Zstandard-compressed file
        /// </summary>
        /// <param name="filePath">Path to the compressed file</param>
        /// <returns>Decompressed byte array</returns>
        public static byte[] DecompressFile(string filePath)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException($"Compressed file not found: {filePath}");

            var compressedBytes = File.ReadAllBytes(filePath);
            ThrowIfGitLfsPointer(filePath, compressedBytes);
            return Decompress(compressedBytes);
        }

        /// <summary>
        /// Magic prefix that marks a file as a Git LFS pointer (rather than the real binary it stands in for).
        /// See https://github.com/git-lfs/git-lfs/blob/main/docs/spec.md
        /// </summary>
        private const string GitLfsPointerPrefix = "version https://git-lfs.github.com/spec/v1";

        /// <summary>
        /// Returns true if <paramref name="fileBytes"/> look like a Git LFS pointer rather than real content.
        /// Pointer files are short UTF-8 text beginning with a fixed version line.
        /// </summary>
        public static bool IsGitLfsPointer(byte[] fileBytes)
        {
            if (fileBytes == null || fileBytes.Length < GitLfsPointerPrefix.Length)
                return false;
            // LFS pointer files are small; avoid scanning large binaries.
            if (fileBytes.Length > 1024)
                return false;
            for (int i = 0; i < GitLfsPointerPrefix.Length; i++)
            {
                if (fileBytes[i] != (byte)GitLfsPointerPrefix[i])
                    return false;
            }
            return true;
        }

        /// <summary>
        /// Throws a clear, actionable error if the bytes are a Git LFS pointer instead of real content.
        /// </summary>
        public static void ThrowIfGitLfsPointer(string filePath, byte[] fileBytes)
        {
            if (!IsGitLfsPointer(fileBytes))
                return;
            throw new InvalidDataException(
                $"'{filePath}' is a Git LFS pointer, not the actual file content. " +
                "Install git-lfs and run 'git lfs install && git lfs pull' to download the real file.");
        }

        /// <summary>
        /// Decompress a Zstandard-compressed stream
        /// </summary>
        /// <param name="stream">Stream containing compressed data</param>
        /// <returns>Decompressed byte array</returns>
        public static byte[] DecompressStream(Stream stream)
        {
            using var memoryStream = new MemoryStream();
            stream.CopyTo(memoryStream);
            var compressedBytes = memoryStream.ToArray();
            return Decompress(compressedBytes);
        }

        /// <summary>
        /// Decompress Zstandard-compressed bytes
        /// </summary>
        /// <param name="compressedBytes">Compressed byte array</param>
        /// <returns>Decompressed byte array</returns>
        public static byte[] Decompress(byte[] compressedBytes)
        {
            using var decompressor = new Decompressor();
            return decompressor.Unwrap(compressedBytes).ToArray();
        }

        /// <summary>
        /// Compress bytes using Zstandard and write to a file
        /// </summary>
        /// <param name="filePath">Path for the output file</param>
        /// <param name="uncompressedBytes">Bytes to compress</param>
        /// <param name="compressionLevel">Zstandard compression level (1-22, default: 3)</param>
        public static void CompressToFile(string filePath, byte[] uncompressedBytes, int compressionLevel = DefaultCompressionLevel)
        {
            var compressedBytes = Compress(uncompressedBytes, compressionLevel);
            File.WriteAllBytes(filePath, compressedBytes);
        }

        /// <summary>
        /// Compress bytes using Zstandard and write to a stream
        /// </summary>
        /// <param name="stream">Stream to write the compressed data to</param>
        /// <param name="uncompressedBytes">Bytes to compress</param>
        /// <param name="compressionLevel">Zstandard compression level (1-22, default: 3)</param>
        public static void CompressToStream(Stream stream, byte[] uncompressedBytes, int compressionLevel = DefaultCompressionLevel)
        {
            var compressedBytes = Compress(uncompressedBytes, compressionLevel);
            stream.Write(compressedBytes, 0, compressedBytes.Length);
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

        #endregion

        #region File Format Detection

        /// <summary>
        /// Determines if a file path is for a compressed safetensor file
        /// </summary>
        public static bool IsCompressedSafeTensor(string filePath)
            => filePath.EndsWith(CompressedSafeTensorExtension, StringComparison.OrdinalIgnoreCase);

        /// <summary>
        /// Determines if a file path is for a compressed architecture file
        /// </summary>
        public static bool IsCompressedArchitecture(string filePath)
            => filePath.EndsWith(CompressedArchitectureExtension, StringComparison.OrdinalIgnoreCase);

        #endregion
    }
}

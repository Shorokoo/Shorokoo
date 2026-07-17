using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Modules;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes.OnnxNodes;
using static Shorokoo.Globals;
using Shorokoo.Core.Inference.Abstractions;

namespace Shorokoo.Onnx
{
    /// <summary>
    /// Loads SafeTensor files into various Shorokoo data structures.
    /// Supports both single-file checkpoints and the Hugging Face multi-file
    /// (sharded) convention: shard files named
    /// <c>model-00001-of-000NN.safetensors</c> plus a
    /// <c>model.safetensors.index.json</c> manifest whose <c>weight_map</c> maps
    /// tensor names to shard file names.
    /// </summary>
    public static class SafeTensorLoader
    {
        /// <summary>
        /// Default maximum shard size (1 GB) for sharded saving. Deliberately below
        /// the Hugging Face convention of 5 GB: the loader currently materializes
        /// each file as one byte[], so files of 2 GB or more cannot be read back.
        /// Streaming shard I/O through a fixed-size buffer (potentially enabling
        /// direct-to-GPU loading) is tracked in
        /// https://github.com/Shorokoo/Shorokoo/issues/48; restore the 5 GB
        /// convention once that lands.
        /// </summary>
        public const long DefaultMaxShardSizeBytes = 1_000_000_000L;

        /// <summary>
        /// File-name suffix that marks a sharded-checkpoint index manifest.
        /// </summary>
        public const string ShardIndexSuffix = ".index.json";

        /// <summary>
        /// Load a SafeTensor file that contains a single tensor into TensorData
        /// </summary>
        /// <param name="filePath">Path to the SafeTensor file</param>
        /// <returns>TensorData containing the single tensor</returns>
        /// <exception cref="InvalidOperationException">Thrown if file contains zero or multiple tensors</exception>
        public static TensorData LoadSingleTensor(string filePath)
        {
            var tensors = LoadSafeTensors(filePath);

            if (tensors.Count == 0)
                throw new InvalidOperationException($"SafeTensor file '{filePath}' contains no tensors");

            if (tensors.Count > 1)
                throw new InvalidOperationException($"SafeTensor file '{filePath}' contains {tensors.Count} tensors, expected exactly 1");

            return tensors.First().Data;
        }

        /// <summary>
        /// Load a SafeTensor file into a ModelParamSet
        /// </summary>
        /// <param name="filePath">Path to the SafeTensor file</param>
        /// <param name="paramType">Classification applied to every loaded tensor (trainable vs state param)</param>
        /// <returns>ModelParamSet containing all tensors from the file</returns>
        public static ModelParamList LoadModelParamSet(string filePath, ModelParamType paramType = ModelParamType.TrainableParam)
        {
            var tensors = LoadSafeTensors(filePath);
            var paramDict = tensors.ToDictionary(t => t.Name, t => t.Data);
            return new ModelParamList(paramDict, paramType);
        }

        /// <summary>
        /// Load a SafeTensor file into a Dictionary of tensor names to TensorData
        /// </summary>
        /// <param name="filePath">Path to the SafeTensor file, shard index, or checkpoint directory</param>
        /// <returns>Dictionary mapping tensor names to TensorData</returns>
        public static Dictionary<string, TensorData> LoadTensorDictionary(string filePath)
        {
            var tensors = LoadSafeTensors(filePath);
            return tensors.ToDictionary(t => t.Name, t => t.Data);
        }

        /// <summary>
        /// Load only the named tensors into a Dictionary of tensor names to TensorData.
        /// For a sharded checkpoint, only the shard files that contain a requested
        /// tensor are opened.
        /// </summary>
        /// <param name="filePath">Path to the SafeTensor file, shard index, or checkpoint directory</param>
        /// <param name="tensorNames">Names of the tensors to load; unknown names fail loudly</param>
        /// <returns>Dictionary mapping the requested tensor names to TensorData</returns>
        public static Dictionary<string, TensorData> LoadTensorDictionary(string filePath, IReadOnlyCollection<string> tensorNames)
        {
            var tensors = LoadSafeTensors(filePath, tensorNames);
            return tensors.ToDictionary(t => t.Name, t => t.Data);
        }

        /// <summary>
        /// Load a SafeTensor checkpoint into a List of SafeTensor objects with full metadata.
        /// The path may be a single <c>.safetensors</c> file, a sharded-checkpoint
        /// index manifest (<c>*.index.json</c>), or a directory containing one such
        /// manifest; sharded checkpoints load as the union of their shards.
        /// </summary>
        /// <param name="filePath">Path to the SafeTensor file, shard index, or checkpoint directory</param>
        /// <returns>List of SafeTensor objects containing tensor data and metadata</returns>
        public static List<SafeTensor> LoadSafeTensors(string filePath)
            => LoadSafeTensors(filePath, tensorNames: null);

        /// <summary>
        /// Load a SafeTensor checkpoint, optionally restricted to a named subset of
        /// tensors. For a sharded checkpoint, only the shard files that contain a
        /// requested tensor are opened.
        /// </summary>
        /// <param name="filePath">Path to the SafeTensor file, shard index, or checkpoint directory</param>
        /// <param name="tensorNames">Names of the tensors to load, or null for all; unknown names fail loudly</param>
        /// <returns>List of SafeTensor objects containing tensor data and metadata</returns>
        public static List<SafeTensor> LoadSafeTensors(string filePath, IReadOnlyCollection<string>? tensorNames)
        {
            if (TryResolveShardIndex(filePath, out var indexPath))
                return LoadShardedSafeTensors(indexPath, tensorNames);

            if (!File.Exists(filePath))
                throw new FileNotFoundException($"SafeTensor file not found: {filePath}");

            var fileBytes = File.ReadAllBytes(filePath);
            var tensors = ParseSafeTensorFile(fileBytes);

            if (tensorNames is null)
                return tensors;

            var present = new HashSet<string>(tensors.Select(t => t.Name));
            foreach (var name in tensorNames)
                if (!present.Contains(name))
                    throw new KeyNotFoundException($"Tensor '{name}' not found in SafeTensor file '{filePath}'.");

            var requested = new HashSet<string>(tensorNames);
            return tensors.Where(t => requested.Contains(t.Name)).ToList();
        }

        /// <summary>
        /// Resolves <paramref name="path"/> to a sharded-checkpoint index manifest:
        /// either the path is the manifest itself (<c>*.index.json</c>) or a
        /// directory containing exactly one <c>*.safetensors.index.json</c>.
        /// </summary>
        private static bool TryResolveShardIndex(string path, out string indexPath)
        {
            if (Directory.Exists(path))
            {
                string[] candidates = [.. Directory.GetFiles(path, "*.safetensors" + ShardIndexSuffix).OrderBy(p => p, StringComparer.Ordinal)];
                if (candidates.Length == 0)
                    throw new FileNotFoundException($"Directory '{path}' contains no *.safetensors{ShardIndexSuffix} shard index.");
                if (candidates.Length > 1)
                    throw new InvalidOperationException(
                        $"Directory '{path}' contains {candidates.Length} shard indexes " +
                        $"({string.Join(", ", candidates.Select(Path.GetFileName))}); pass the index file path explicitly.");
                indexPath = candidates[0];
                return true;
            }

            if (path.EndsWith(ShardIndexSuffix, StringComparison.OrdinalIgnoreCase))
            {
                if (!File.Exists(path))
                    throw new FileNotFoundException($"SafeTensor shard index not found: {path}");
                indexPath = path;
                return true;
            }

            indexPath = string.Empty;
            return false;
        }

        /// <summary>
        /// Load a sharded checkpoint through its index manifest. Shards are opened
        /// lazily: when <paramref name="tensorNames"/> restricts the load, shard
        /// files containing none of the requested tensors are never touched.
        /// </summary>
        private static List<SafeTensor> LoadShardedSafeTensors(string indexPath, IReadOnlyCollection<string>? tensorNames)
        {
            var indexBytes = File.ReadAllBytes(indexPath);

            // Parse weight_map preserving entry order; duplicate keys in the JSON
            // (the same tensor mapped twice) are surfaced by JsonDocument and rejected.
            var weightMap = new List<(string Name, string Shard)>();
            var owningShard = new Dictionary<string, string>();
            using (var doc = JsonDocument.Parse(indexBytes))
            {
                if (!doc.RootElement.TryGetProperty("weight_map", out var mapElement) || mapElement.ValueKind != JsonValueKind.Object)
                    throw new InvalidOperationException($"SafeTensor shard index '{indexPath}' has no 'weight_map' object.");

                foreach (var entry in mapElement.EnumerateObject())
                {
                    if (entry.Value.ValueKind != JsonValueKind.String)
                        throw new InvalidOperationException($"weight_map entry for tensor '{entry.Name}' in '{indexPath}' is not a shard file name.");
                    var shard = entry.Value.GetString()!;
                    if (!owningShard.TryAdd(entry.Name, shard))
                        throw new InvalidOperationException(
                            $"Duplicate tensor '{entry.Name}' in the weight_map of '{indexPath}' " +
                            $"(mapped to both '{owningShard[entry.Name]}' and '{shard}').");
                    weightMap.Add((entry.Name, shard));
                }
            }

            var requested = tensorNames is null ? null : new HashSet<string>(tensorNames);
            if (requested != null)
                foreach (var name in requested)
                    if (!owningShard.ContainsKey(name))
                        throw new KeyNotFoundException($"Tensor '{name}' is not listed in the weight_map of '{indexPath}'.");

            // Group tensor names by shard, preserving weight_map order.
            var shardOrder = new List<string>();
            var mappedByShard = new Dictionary<string, List<string>>();
            foreach (var (name, shard) in weightMap)
            {
                if (!mappedByShard.TryGetValue(shard, out var names))
                {
                    names = [];
                    mappedByShard[shard] = names;
                    shardOrder.Add(shard);
                }
                names.Add(name);
            }

            var indexDir = Path.GetDirectoryName(Path.GetFullPath(indexPath))!;
            var loaded = new Dictionary<string, SafeTensor>();
            foreach (var shard in shardOrder)
            {
                var mapped = mappedByShard[shard];
                if (requested != null && !mapped.Any(requested.Contains))
                    continue; // no requested tensor lives in this shard — do not touch the file

                var shardPath = Path.Combine(indexDir, shard);
                if (!File.Exists(shardPath))
                    throw new FileNotFoundException(
                        $"Shard file '{shard}' (holding tensor '{mapped[0]}') referenced by '{indexPath}' not found at '{shardPath}'.");

                var shardBytes = File.ReadAllBytes(shardPath);
                var parsed = ParseSafeTensorFile(shardBytes);

                foreach (var tensor in parsed)
                {
                    if (!owningShard.TryGetValue(tensor.Name, out var owner))
                        throw new InvalidOperationException(
                            $"Tensor '{tensor.Name}' found in shard '{shard}' is not listed in the weight_map of '{indexPath}'.");
                    if (!string.Equals(owner, shard, StringComparison.Ordinal))
                        throw new InvalidOperationException(
                            $"Duplicate tensor '{tensor.Name}': found in shard '{shard}' but the weight_map of '{indexPath}' assigns it to '{owner}'.");
                    loaded[tensor.Name] = tensor;
                }

                foreach (var name in mapped)
                    if (!loaded.ContainsKey(name))
                        throw new InvalidOperationException(
                            $"Tensor '{name}' is listed in the weight_map of '{indexPath}' as stored in shard '{shard}' but is missing from that shard.");
            }

            // Return in weight_map order, restricted to the requested subset.
            var result = new List<SafeTensor>();
            foreach (var (name, _) in weightMap)
                if ((requested is null || requested.Contains(name)) && loaded.TryGetValue(name, out var tensor))
                    result.Add(tensor);
            return result;
        }

        /// <summary>
        /// Save SafeTensor objects to a file in SafeTensors format
        /// (8-byte header length, JSON header, raw tensor data).
        /// When <paramref name="maxShardSizeBytes"/> is set and the tensors do not
        /// fit in a single shard, the checkpoint is written in the Hugging Face
        /// multi-file convention instead: shard files named
        /// <c>{name}-00001-of-000NN.safetensors</c> next to <paramref name="filePath"/>
        /// plus a <c>{name}.safetensors.index.json</c> manifest with a
        /// <c>weight_map</c> and total-size metadata. Each shard is an individually
        /// valid safetensors file carrying <paramref name="globalMetadata"/>.
        /// </summary>
        /// <param name="filePath">Path for the output .safetensors file; shard and index names are derived from it</param>
        /// <param name="tensors">List of SafeTensor objects to save</param>
        /// <param name="globalMetadata">Optional global metadata to include</param>
        /// <param name="maxShardSizeBytes">
        /// Opt-in maximum shard size in bytes (default:
        /// <see cref="DefaultMaxShardSizeBytes"/>, 1 GB). Null (the default) always
        /// writes a single file; when the total tensor size is at or below the
        /// threshold the single-file output is byte-for-byte identical to a save
        /// without it.
        /// </param>
        public static void SaveSafeTensors(string filePath, List<SafeTensor> tensors, Dictionary<string, object>? globalMetadata = null, long? maxShardSizeBytes = null)
        {
            if (maxShardSizeBytes is not null)
            {
                if (maxShardSizeBytes <= 0)
                    throw new ArgumentOutOfRangeException(nameof(maxShardSizeBytes), "Maximum shard size must be positive.");
                if (TrySaveShardedSafeTensors(filePath, tensors, globalMetadata, maxShardSizeBytes.Value))
                    return;
            }

            using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None);
            SaveSafeTensorsToStream(fs, tensors, globalMetadata);
        }

        /// <summary>
        /// Writes a sharded checkpoint when the tensors overflow a single shard of
        /// <paramref name="maxShardSizeBytes"/>; returns false (writing nothing) when
        /// they fit in one shard so the caller falls back to the standard single file.
        /// Tensors are packed greedily into shards in list order.
        /// </summary>
        private static bool TrySaveShardedSafeTensors(string filePath, List<SafeTensor> tensors, Dictionary<string, object>? globalMetadata, long maxShardSizeBytes)
        {
            if (tensors == null)
                throw new ArgumentNullException(nameof(tensors));

            if (tensors.Count == 0)
                throw new ArgumentException("Cannot save an empty SafeTensor list.", nameof(tensors));

            var sizes = new long[tensors.Count];
            long totalSize = 0L;
            for (int i = 0; i < tensors.Count; i++)
            {
                var st = tensors[i];
                if (st == null)
                    throw new InvalidOperationException("SafeTensor list contains a null entry.");
                if (st.Data == null)
                    throw new InvalidOperationException($"SafeTensor '{st.Name}' has no Data.");
                sizes[i] = st.Data.AccessRawMemory().Length;
                totalSize += sizes[i];
            }

            // Greedy packing in list order: start a new shard when the next tensor
            // would overflow the current one. A tensor larger than the limit gets a
            // shard of its own.
            var shardRanges = new List<(int Start, int Count)>();
            int rangeStart = 0;
            long currentSize = 0L;
            for (int i = 0; i < tensors.Count; i++)
            {
                if (i > rangeStart && currentSize + sizes[i] > maxShardSizeBytes)
                {
                    shardRanges.Add((rangeStart, i - rangeStart));
                    rangeStart = i;
                    currentSize = 0L;
                }
                currentSize += sizes[i];
            }
            shardRanges.Add((rangeStart, tensors.Count - rangeStart));

            if (shardRanges.Count == 1)
                return false; // fits in one shard — standard single-file output

            var fullPath = Path.GetFullPath(filePath);
            var directory = Path.GetDirectoryName(fullPath)!;
            var extension = Path.GetExtension(fullPath);
            if (extension.Length == 0)
                extension = ".safetensors";
            var stem = Path.GetFileNameWithoutExtension(fullPath);

            string ShardName(int s) => $"{stem}-{s + 1:D5}-of-{shardRanges.Count:D5}{extension}";

            // Map every tensor to its shard (and reject duplicates) before writing
            // any file, so a failure never leaves a partial checkpoint behind.
            var weightMap = new Dictionary<string, string>();
            for (int s = 0; s < shardRanges.Count; s++)
            {
                var (start, count) = shardRanges[s];
                for (int i = start; i < start + count; i++)
                    if (!weightMap.TryAdd(tensors[i].Name, ShardName(s)))
                        throw new InvalidOperationException($"Duplicate tensor name '{tensors[i].Name}'; sharded saving requires unique tensor names.");
            }

            for (int s = 0; s < shardRanges.Count; s++)
            {
                var (start, count) = shardRanges[s];
                SaveSafeTensors(Path.Combine(directory, ShardName(s)), tensors.GetRange(start, count), globalMetadata);
            }

            var index = new Dictionary<string, object>
            {
                ["metadata"] = new Dictionary<string, object> { ["total_size"] = totalSize },
                ["weight_map"] = weightMap,
            };
            File.WriteAllText(Path.Combine(directory, stem + extension + ShardIndexSuffix), JsonSerializer.Serialize(index));
            return true;
        }

        /// <summary>
        /// Save SafeTensor objects to a stream
        /// </summary>
        /// <param name="stream">Stream to write to</param>
        /// <param name="tensors">List of SafeTensor objects to save</param>
        /// <param name="globalMetadata">Optional global metadata to include</param>
        public static void SaveSafeTensorsToStream(Stream stream, List<SafeTensor> tensors, Dictionary<string, object>? globalMetadata = null)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));

            if (!stream.CanWrite)
                throw new ArgumentException("Stream must be writable.", nameof(stream));

            if (tensors == null)
                throw new ArgumentNullException(nameof(tensors));

            if (tensors.Count == 0)
                throw new ArgumentException("Cannot save an empty SafeTensor list.", nameof(tensors));

            // Build header object and collect raw tensor byte blobs
            var header = new Dictionary<string, object>();
            var tensorBlobs = new List<byte[]>(tensors.Count);

            long currentOffset = 0L;

            foreach (var st in tensors)
            {
                if (st == null)
                    throw new InvalidOperationException("SafeTensor list contains a null entry.");

                if (string.IsNullOrWhiteSpace(st.Name))
                    throw new InvalidOperationException("SafeTensor has no valid Name.");

                // An empty shape is the valid SafeTensors encoding of a rank-0 scalar
                // (product of an empty shape = 1 element); only a null shape is invalid.
                if (st.Shape == null)
                    throw new InvalidOperationException($"SafeTensor '{st.Name}' has no valid Shape.");

                if (st.Data == null)
                    throw new InvalidOperationException($"SafeTensor '{st.Name}' has no Data.");

                if (string.IsNullOrWhiteSpace(st.DataType))
                    throw new InvalidOperationException($"SafeTensor '{st.Name}' has no valid DType.");

                var shape = st.Shape;
                var dtype = st.DataType.ToUpperInvariant();

                // Flatten and convert tensor to raw bytes
                var blob = st.Data.AccessRawMemory().ToArray();
                tensorBlobs.Add(blob);

                long startOffset = currentOffset;
                long endOffset = startOffset + blob.Length;
                currentOffset = endOffset;

                // Per-tensor metadata according to SafeTensors spec
                var tensorMeta = new Dictionary<string, object>
                {
                    ["dtype"] = dtype,
                    ["shape"] = shape,
                    ["data_offsets"] = new long[] { startOffset, endOffset }
                };

                // If you have extra metadata on SafeTensor, merge it here.
                if (st.Metadata != null)
                    foreach (var kv in st.Metadata)
                        tensorMeta[kv.Key] = kv.Value;

                header[st.Name] = tensorMeta;
            }

            if (globalMetadata is not null)
                header["__metadata__"] = globalMetadata;

            // Serialize header to JSON UTF-8
            var headerJson = JsonSerializer.Serialize(header);
            var headerBytes = System.Text.Encoding.UTF8.GetBytes(headerJson);
            long headerLength = headerBytes.LongLength;

            // Compose file: [8-byte little-endian header length][header JSON][tensor binary]
            var lengthBytes = BitConverter.GetBytes(headerLength);
            stream.Write(lengthBytes, 0, lengthBytes.Length);
            stream.Write(headerBytes, 0, headerBytes.Length);

            foreach (var blob in tensorBlobs)
                stream.Write(blob, 0, blob.Length);
        }

        /// <summary>
        /// Parse SafeTensor bytes and return list of SafeTensor objects.
        /// This is a public entry point for parsing in-memory safetensor data.
        /// </summary>
        /// <param name="fileBytes">Raw bytes of the SafeTensor file</param>
        /// <returns>List of SafeTensor objects</returns>
        public static List<SafeTensor> ParseSafeTensorBytes(byte[] fileBytes)
        {
            return ParseSafeTensorFile(fileBytes);
        }

        /// <summary>
        /// Convert a Shorokoo DType to the SafeTensor dtype string format
        /// </summary>
        /// <param name="dtype">Shorokoo DType</param>
        /// <returns>SafeTensor dtype string (e.g., "F32", "I64")</returns>
        public static string DTypeToSafeTensorDType(DType dtype)
        {
            if (dtype == DType.Bool) return "BOOL";
            if (dtype == DType.Int8) return "I8";
            if (dtype == DType.Int16) return "I16";
            if (dtype == DType.Int32) return "I32";
            if (dtype == DType.Int64) return "I64";
            if (dtype == DType.UInt8) return "U8";
            if (dtype == DType.UInt16) return "U16";
            if (dtype == DType.UInt32) return "U32";
            if (dtype == DType.UInt64) return "U64";
            if (dtype == DType.Float32) return "F32";
            if (dtype == DType.Float64) return "F64";
            if (dtype == DType.Float16) return "F16";
            if (dtype == DType.BFloat16) return "BF16";
            throw new NotSupportedException($"Unsupported DType for SafeTensor format: {dtype}");
        }

        /// <summary>
        /// Parse SafeTensor file format and return list of SafeTensor objects
        /// </summary>
        /// <param name="fileBytes">Raw bytes of the SafeTensor file</param>
        /// <returns>List of SafeTensor objects</returns>
        private static List<SafeTensor> ParseSafeTensorFile(byte[] fileBytes)
        {
            if (fileBytes.Length < 8)
                throw new InvalidOperationException("File too short to be a valid SafeTensor file");

            // Read header length (first 8 bytes, little-endian)
            long headerLength = BitConverter.ToInt64(fileBytes, 0);

            if (headerLength <= 0 || headerLength > fileBytes.Length - 8)
                throw new InvalidOperationException($"Invalid header length: {headerLength}");

            // Read header JSON
            var headerBytes = new byte[headerLength];
            Array.Copy(fileBytes, 8, headerBytes, 0, (int)headerLength);
            var headerJson = System.Text.Encoding.UTF8.GetString(headerBytes);

            // Parse the JSON to get tensor metadata
            var metadata = JsonSerializer.Deserialize<Dictionary<string, object>>(headerJson);

            if (metadata == null)
                throw new InvalidOperationException("Failed to parse SafeTensor header JSON");

            var result = new List<SafeTensor>();
            long dataOffset = 8 + headerLength; // Start of tensor data

            // Extract tensor information from metadata
            foreach (var kvp in metadata)
            {
                if (kvp.Key == "__metadata__") continue; // Skip metadata section

                var tensorName = kvp.Key;
                var tensorMeta = kvp.Value;

                try
                {
                    var safeTensor = ParseTensorMetadata(tensorName, tensorMeta, fileBytes, dataOffset);
                    result.Add(safeTensor);
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException($"Failed to parse tensor '{tensorName}': {ex.Message}", ex);
                }
            }

            return result;
        }

        /// <summary>
        /// Parse metadata for a single tensor and create SafeTensor object
        /// </summary>
        private static SafeTensor ParseTensorMetadata(string tensorName, object tensorMeta, byte[] fileBytes, long dataOffset)
        {
            try
            {
                // Parse the tensor metadata
                var metaJson = JsonSerializer.Serialize(tensorMeta);
                var metaDict = JsonSerializer.Deserialize<Dictionary<string, object>>(metaJson);

                if (metaDict == null)
                    throw new InvalidOperationException("Failed to parse tensor metadata");

                // Extract shape
                var shape = ExtractShape(metaDict);

                // Extract data_offsets
                var (startOffset, endOffset) = ExtractDataOffsets(metaDict);

                // Extract dtype
                var dtype = ExtractDataType(metaDict);

                // Extract tensor data
                var dataSize = (int)(endOffset - startOffset);
                var tensorData = new byte[dataSize];
                var actualStart = dataOffset + startOffset;

                if (actualStart + dataSize <= fileBytes.Length)
                {
                    Array.Copy(fileBytes, actualStart, tensorData, 0, dataSize);
                }
                else
                {
                    throw new InvalidOperationException($"Tensor data extends beyond file bounds for tensor '{tensorName}'");
                }

                // Create TensorData based on dtype using the working Globals.TensorData methods
                var tensorDataObj = CreateTensorFromRawBytes(dtype, shape, tensorData);

                // Extract any additional metadata
                var additionalMetadata = metaDict.Where(kvp =>
                    kvp.Key != "shape" &&
                    kvp.Key != "data_offsets" &&
                    kvp.Key != "dtype")
                    .ToDictionary(kvp => kvp.Key, kvp => kvp.Value);

                return new SafeTensor(tensorName, tensorDataObj, dtype, shape, additionalMetadata);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to parse tensor metadata for '{tensorName}': {ex.Message}", ex.InnerException ?? ex);
            }
        }

        /// <summary>
        /// Extract shape array from metadata dictionary
        /// </summary>
        private static long[] ExtractShape(Dictionary<string, object> metaDict)
        {
            if (metaDict.TryGetValue("shape", out var shapeObj))
            {
                var shapeJson = JsonSerializer.Serialize(shapeObj);
                var shapeParsed = JsonSerializer.Deserialize<long[]>(shapeJson);
                if (shapeParsed != null)
                    return shapeParsed;
            }
            return new long[] { 1 }; // Default fallback
        }

        /// <summary>
        /// Extract data offsets from metadata dictionary
        /// </summary>
        private static (long startOffset, long endOffset) ExtractDataOffsets(Dictionary<string, object> metaDict)
        {
            if (metaDict.TryGetValue("data_offsets", out var offsetsObj))
            {
                var offsetsJson = JsonSerializer.Serialize(offsetsObj);
                var offsetsParsed = JsonSerializer.Deserialize<long[]>(offsetsJson);
                if (offsetsParsed != null && offsetsParsed.Length >= 2)
                {
                    return (offsetsParsed[0], offsetsParsed[1]);
                }
            }
            return (0L, 4L); // Default 1 float
        }

        /// <summary>
        /// Extract data type from metadata dictionary
        /// </summary>
        private static string ExtractDataType(Dictionary<string, object> metaDict)
        {
            if (metaDict.TryGetValue("dtype", out var dtypeObj))
            {
                return dtypeObj.ToString() ?? "F32";
            }
            return "F32"; // Default
        }

        /// <summary>
        /// Create TensorData from raw bytes by converting to typed arrays and using working Globals.TensorData methods
        /// </summary>
        private static TensorData CreateTensorFromRawBytes(string safeTensorDType, long[] shape, byte[] rawData)
        {
            return safeTensorDType.ToUpperInvariant() switch
            {
                "BOOL" => (TensorData)Globals.TensorData(shape, ConvertRawBytes<bool>(rawData)),
                "I8" => (TensorData)Globals.TensorData(shape, ConvertRawBytes<sbyte>(rawData)),
                "I16" => (TensorData)Globals.TensorData(shape, ConvertRawBytes<short>(rawData)),
                "I32" => (TensorData)Globals.TensorData(shape, ConvertRawBytes<int>(rawData)),
                "I64" => (TensorData)Globals.TensorData(shape, ConvertRawBytes<long>(rawData)),
                "U8" => (TensorData)Globals.TensorData(shape, ConvertRawBytes<byte>(rawData)),
                "U16" => (TensorData)Globals.TensorData(shape, ConvertRawBytes<ushort>(rawData)),
                "U32" => (TensorData)Globals.TensorData(shape, ConvertRawBytes<uint>(rawData)),
                "U64" => (TensorData)Globals.TensorData(shape, ConvertRawBytes<ulong>(rawData)),
                "F32" => (TensorData)Globals.TensorData(shape, ConvertRawBytes<float>(rawData)),
                "F64" => (TensorData)Globals.TensorData(shape, ConvertRawBytes<double>(rawData)),
                // F16/BF16 payloads are raw little-endian IEEE half / bfloat16 bit
                // patterns (2 bytes per element) — exactly the in-memory layout of the
                // ushort-backed Float16/BFloat16 structs, so the same memcpy path works.
                "F16" => (TensorData)Globals.TensorData(shape, ConvertRawBytes<Float16>(rawData)),
                "BF16" => (TensorData)Globals.TensorData(shape, ConvertRawBytes<BFloat16>(rawData)),
                _ => throw new NotSupportedException(
                    $"Unsupported SafeTensor data type: {safeTensorDType}. " +
                    "Supported formats: BOOL, I8, I16, I32, I64, U8, U16, U32, U64, F16, BF16, F32, F64.")
            };
        }

        /// <summary>
        /// Convert raw bytes to typed array
        /// </summary>
        private static unsafe T[] ConvertRawBytes<T>(byte[] rawData) where T : unmanaged
        {
            var elementSize = sizeof(T);
            var elementCount = rawData.Length / elementSize;
            var result = new T[elementCount];

            fixed (byte* srcPtr = rawData)
            fixed (T* dstPtr = result)
            {
                Buffer.MemoryCopy(srcPtr, dstPtr, rawData.Length, rawData.Length);
            }

            return result;
        }
    }
}
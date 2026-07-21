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
    /// Loads SafeTensor files into various Shorokoo data structures
    /// </summary>
    public static class SafeTensorLoader
    {
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
        /// <param name="filePath">Path to the SafeTensor file</param>
        /// <returns>Dictionary mapping tensor names to TensorData</returns>
        public static Dictionary<string, TensorData> LoadTensorDictionary(string filePath)
        {
            var tensors = LoadSafeTensors(filePath);
            return tensors.ToDictionary(t => t.Name, t => t.Data);
        }

        /// <summary>
        /// Load a SafeTensor file into a List of SafeTensor objects with full metadata
        /// </summary>
        /// <param name="filePath">Path to the SafeTensor file</param>
        /// <returns>List of SafeTensor objects containing tensor data and metadata</returns>
        public static List<SafeTensor> LoadSafeTensors(string filePath)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException($"SafeTensor file not found: {filePath}");

            var fileBytes = File.ReadAllBytes(filePath);
            return ParseSafeTensorFile(fileBytes, filePath);
        }

        /// <summary>
        /// Save SafeTensor objects to a file in SafeTensors format
        /// (8-byte header length, JSON header, raw tensor data)
        /// </summary>
        public static void SaveSafeTensors(string filePath, List<SafeTensor> tensors, Dictionary<string, object>? globalMetadata = null)
        {
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
                // ASSUMPTION: there is an AdditionalMetadata (Dictionary<string, object>) property.
                // If your SafeTensor type uses a different name, adjust this block accordingly.
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
            using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None);

            var lengthBytes = BitConverter.GetBytes(headerLength); // little-endian on all common platforms
            fs.Write(lengthBytes, 0, lengthBytes.Length);
            fs.Write(headerBytes, 0, headerBytes.Length);

            foreach (var blob in tensorBlobs)
                fs.Write(blob, 0, blob.Length);
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
            return ParseSafeTensorFile(fileBytes, "<in-memory SafeTensor data>");
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
        /// Parse SafeTensor file format and return list of SafeTensor objects. The declared
        /// header length and every tensor's data_offsets range are validated against the actual
        /// byte count up front, so a truncated file (interrupted download/copy, disk full, …)
        /// fails with a <see cref="ModelException"/> naming truncation and the declared vs.
        /// actual sizes instead of an incidental parse error.
        /// </summary>
        /// <param name="fileBytes">Raw bytes of the SafeTensor file</param>
        /// <param name="origin">Name used in error messages, typically the file path</param>
        /// <returns>List of SafeTensor objects</returns>
        private static List<SafeTensor> ParseSafeTensorFile(byte[] fileBytes, string origin)
        {
            if (fileBytes.Length < 8)
                throw new ModelException(ErrorCodes.ST001, $"SafeTensor file '{origin}'",
                    $"the file is only {fileBytes.Length} byte(s) — too short to hold the 8-byte " +
                    "SafeTensors header-length field. The file is truncated or not a SafeTensors file.");

            // Read header length (first 8 bytes, little-endian)
            long headerLength = BitConverter.ToInt64(fileBytes, 0);

            if (headerLength <= 0)
                throw new InvalidOperationException($"Invalid header length: {headerLength}");

            if (headerLength > fileBytes.Length - 8)
                throw new ModelException(ErrorCodes.ST002, $"SafeTensor file '{origin}'",
                    $"truncated SafeTensor file — the header declares {headerLength} bytes of JSON header, " +
                    $"but only {fileBytes.Length - 8} byte(s) follow the length field (the file has " +
                    $"{fileBytes.Length} bytes). The file was likely cut short by an interrupted download or copy.");

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
                    var safeTensor = ParseTensorMetadata(tensorName, tensorMeta, fileBytes, dataOffset, origin);
                    result.Add(safeTensor);
                }
                catch (Exception ex) when (ex is not ShorokooException)
                {
                    throw new InvalidOperationException($"Failed to parse tensor '{tensorName}': {ex.Message}", ex);
                }
            }

            return result;
        }

        /// <summary>
        /// Parse metadata for a single tensor and create SafeTensor object
        /// </summary>
        private static SafeTensor ParseTensorMetadata(
            string tensorName, object tensorMeta, byte[] fileBytes, long dataOffset, string origin)
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

                if (startOffset < 0 || endOffset < startOffset)
                    throw new InvalidOperationException(
                        $"Tensor '{tensorName}' has invalid data_offsets [{startOffset}, {endOffset})");

                if (dataOffset + endOffset > fileBytes.Length)
                    throw new ModelException(ErrorCodes.ST003, $"SafeTensor file '{origin}'",
                        $"truncated SafeTensor file — tensor '{tensorName}' declares data_offsets " +
                        $"[{startOffset}, {endOffset}), which requires the file to hold " +
                        $"{dataOffset + endOffset} bytes, but the file has {fileBytes.Length} bytes. " +
                        "The file was likely cut short by an interrupted download or copy.");

                // Extract tensor data
                var dataSize = (int)(endOffset - startOffset);
                var tensorData = new byte[dataSize];
                Array.Copy(fileBytes, dataOffset + startOffset, tensorData, 0, dataSize);

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
            catch (Exception ex) when (ex is not ShorokooException)
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
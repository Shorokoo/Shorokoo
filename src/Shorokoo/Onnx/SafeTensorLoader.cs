using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Utils;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Modules;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes.OnnxNodes;
using static Shorokoo.Globals;
using Shorokoo.Core.Backends;
using Shorokoo.Runtime;

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
        /// Load a SafeTensor file into a List of SafeTensor objects with full metadata. Tensors come
        /// back in the order their bytes are laid out in the file, which is the order they were
        /// written — not the order the JSON header happens to list them in.
        /// </summary>
        /// <param name="filePath">Path to the SafeTensor file</param>
        /// <returns>List of SafeTensor objects containing tensor data and metadata, in file order</returns>
        public static List<SafeTensor> LoadSafeTensors(string filePath)
            => LoadSafeTensors(filePath, (_, _) => ComputeContext.Host);

        /// <summary>
        /// <see cref="LoadSafeTensors(string)"/>, putting each tensor where
        /// <paramref name="placement"/> names for it: the file is read once, forward, each tensor's
        /// bytes going straight into the memory it lives in — so host memory holds one bounded
        /// buffer for a tensor loaded onto a device, never the file (Shorokoo/Shorokoo#436). A tensor
        /// <paramref name="placement"/> names no memory for is passed over and not returned.
        /// </summary>
        internal static List<SafeTensor> LoadSafeTensors(string filePath, Func<string, long, ComputeContext?> placement)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException($"SafeTensor file not found: {filePath}");

            using var file = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read,
                bufferSize: 1 << 16, FileOptions.SequentialScan);
            return ReadSafeTensors(file, file.Length, placement, filePath);
        }

        /// <summary>
        /// Save SafeTensor objects to a file in SafeTensors format
        /// (8-byte header length, JSON header, raw tensor data). The write is atomic: the file
        /// is staged in a <c>.tmp-</c> sibling and committed by rename, so a failed or
        /// interrupted save leaves any previous file at <paramref name="filePath"/> untouched.
        /// The target's directory must already exist.
        /// </summary>
        public static void SaveSafeTensors(string filePath, List<SafeTensor> tensors, Dictionary<string, object>? globalMetadata = null)
        {
            if (tensors == null)
                throw new ArgumentNullException(nameof(tensors));

            if (tensors.Count == 0)
                throw new ArgumentException("Cannot save an empty SafeTensor list.", nameof(tensors));

            AtomicFileWriter.WriteFile(
                filePath, stream => SaveSafeTensorsToStream(stream, tensors, globalMetadata));
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

            // Build the header from each tensor's byte LENGTH, taken off its own storage. Nothing is
            // copied here: the payload is written straight out of each tensor's storage in the second
            // pass below, so saving a parameter set costs no second copy of it in managed memory —
            // which at checkpoint sizes is both the allocation and the collection of a duplicate of
            // the whole model (Shorokoo/Shorokoo#48, #338).
            var header = new Dictionary<string, object>();

            long currentOffset = 0L;

            for (int i = 0; i < tensors.Count; i++)
            {
                var st = tensors[i];
                if (st == null)
                    throw new InvalidOperationException("SafeTensor list contains a null entry.");

                if (string.IsNullOrWhiteSpace(st.Name))
                    throw new InvalidOperationException("SafeTensor has no valid Name.");

                // An empty shape is the valid SafeTensors encoding of a rank-0 scalar
                // (product of an empty shape = 1 element); only a null shape is invalid.
                if (st.Shape == null)
                    throw new InvalidOperationException($"SafeTensor '{st.Name}' has no valid Shape.");

                if (string.IsNullOrWhiteSpace(st.DataType))
                    throw new InvalidOperationException($"SafeTensor '{st.Name}' has no valid DType.");

                var shape = st.Shape;
                var dtype = st.DataType.ToUpperInvariant();

                // The tensor's storage as raw bytes — measured, not read, wherever it is.
                long blobLength = st.ByteLength;

                long startOffset = currentOffset;
                long endOffset = startOffset + blobLength;
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

            // A stream that only counts what is written through it has no use for the payload, and
            // reading a device-resident tensor to count it would bring the whole state off the card
            // for nothing: it is told the payload's length instead.
            if (stream is ILengthOnlyStream { IsLengthOnly: true } counter)
            {
                counter.Advance(currentOffset);
                return;
            }

            // Second pass: each tensor's payload goes from its own storage into the stream, in the
            // order the header's offsets were accumulated. Re-reading the storage is sound because a
            // tensor's byte length is fixed by the value it wraps and SafeTensor.Data is get-only, so
            // the payload cannot disagree with the offsets already written. A tensor in a device's
            // memory is streamed through one bounded host buffer rather than copied whole.
            for (int i = 0; i < tensors.Count; i++)
                tensors[i].WriteTo(stream);
        }

        /// <summary>
        /// Parse SafeTensor bytes and return list of SafeTensor objects, in the order their bytes are
        /// laid out (the order they were written), not the order the JSON header lists them in.
        /// This is a public entry point for parsing in-memory safetensor data.
        /// </summary>
        /// <param name="fileBytes">Raw bytes of the SafeTensor file</param>
        /// <returns>List of SafeTensor objects, in file order</returns>
        public static List<SafeTensor> ParseSafeTensorBytes(byte[] fileBytes)
        {
            return ParseSafeTensorFile(fileBytes, "<in-memory SafeTensor data>");
        }

        /// <summary>
        /// Parse SafeTensor bytes, naming <paramref name="origin"/> (typically the source file
        /// path) in any error — use this when the bytes were read or decompressed from a file,
        /// so truncation/corruption diagnostics point at the file the caller passed.
        /// </summary>
        /// <param name="fileBytes">Raw bytes of the SafeTensor file</param>
        /// <param name="origin">Name used in error messages, typically the file path</param>
        /// <returns>List of SafeTensor objects</returns>
        public static List<SafeTensor> ParseSafeTensorBytes(byte[] fileBytes, string origin)
        {
            return ParseSafeTensorFile(fileBytes, origin);
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
        /// Parse SafeTensor file format and return list of SafeTensor objects, in host memory —
        /// <see cref="ReadSafeTensors"/> over the bytes, whose length is known up front.
        /// </summary>
        private static List<SafeTensor> ParseSafeTensorFile(byte[] fileBytes, string origin)
        {
            using var source = new MemoryStream(fileBytes, writable: false);
            return ReadSafeTensors(source, fileBytes.Length, (_, _) => ComputeContext.Host, origin);
        }

        /// <summary>
        /// Reads a SafeTensors payload from <paramref name="source"/> in one forward pass, putting
        /// each tensor in the memory <paramref name="placement"/> names for it: host memory
        /// (<see cref="ComputeContext.Host"/>), which the bytes are read into directly, or a
        /// context's device memory, which they reach through one bounded buffer rather than a host
        /// copy of the tensor (Shorokoo/Shorokoo#436). Tensors come back in the order their bytes
        /// are laid out, which is the order they were written — not the order the JSON header lists
        /// them in. Nothing past the last tensor's bytes is read. A tensor
        /// <paramref name="placement"/> names no memory for (null) is passed over: its bytes are read
        /// past, never into memory, and it is not returned.
        ///
        /// <para><paramref name="available"/> is the payload's length, known up front, and lets a
        /// truncated file (interrupted download/copy, disk full, …) be refused before a byte of data
        /// is read, with a <see cref="ModelException"/> naming the declared and actual sizes: no
        /// header claiming more than the payload holds is believed. A decompressing stream passes the
        /// size its frames declare. On any failure the tensors already read are
        /// deleted.</para>
        /// </summary>
        internal static List<SafeTensor> ReadSafeTensors(
            Stream source, long available, Func<string, long, ComputeContext?> placement, string origin)
        {
            var lengthField = new byte[8];
            int got = source.ReadAtLeast(lengthField, 8, throwOnEndOfStream: false);
            if (got < 8)
                throw new ModelException(ErrorCodes.ST001, $"SafeTensor file '{origin}'",
                    $"the file is only {got} byte(s) — too short to hold the 8-byte " +
                    "SafeTensors header-length field. The file is truncated or not a SafeTensors file.");

            long headerLength = BitConverter.ToInt64(lengthField, 0);
            if (headerLength <= 0)
                throw new InvalidOperationException($"Invalid header length: {headerLength}");
            if (headerLength > available - 8)
                throw new ModelException(ErrorCodes.ST002, $"SafeTensor file '{origin}'",
                    $"truncated SafeTensor file — the header declares {headerLength} bytes of JSON header, " +
                    $"but only {available - 8} byte(s) follow the length field (the file has " +
                    $"{available} bytes). The file was likely cut short by an interrupted download or copy.");
            if (headerLength > int.MaxValue)
                throw new ModelException(ErrorCodes.ST002, $"SafeTensor file '{origin}'",
                    $"the header declares {headerLength} bytes of JSON header, more than any SafeTensors " +
                    "header holds. The file is corrupt or not a SafeTensors file.");

            var headerBytes = new byte[headerLength];
            try
            {
                source.ReadExactly(headerBytes);
            }
            catch (EndOfStreamException e)
            {
                throw new ModelException(ErrorCodes.ST002, $"SafeTensor file '{origin}'",
                    $"truncated SafeTensor file — the header declares {headerLength} bytes of JSON header, " +
                    "but the data ends before them. The file was likely cut short by an interrupted " +
                    "download or copy.", e);
            }
            var metadata = JsonSerializer.Deserialize<Dictionary<string, object>>(System.Text.Encoding.UTF8.GetString(headerBytes))
                ?? throw new InvalidOperationException("Failed to parse SafeTensor header JSON");

            long dataOffset = 8 + headerLength;
            var entries = new List<HeaderEntry>();
            foreach (var kvp in metadata)
            {
                if (kvp.Key == "__metadata__") continue;
                try
                {
                    entries.Add(ParseHeaderEntry(kvp.Key, kvp.Value, dataOffset, available, origin));
                }
                catch (Exception ex) when (ex is not ShorokooException)
                {
                    throw new InvalidOperationException($"Failed to parse tensor '{kvp.Key}': {ex.Message}", ex);
                }
            }

            // OrderBy, not List.Sort: the sort has to be stable. A zero-element tensor occupies no
            // bytes, so it shares a start offset with whatever follows it, and an unstable sort would
            // order those two arbitrarily.
            var tensors = new List<SafeTensor>(entries.Count);
            try
            {
                long position = 0;
                foreach (var entry in entries.OrderBy(e => e.Start))
                {
                    var onto = placement(entry.Name, entry.Elements);
                    // Occupying no bytes, a zero-byte tensor overlaps nothing wherever its offsets
                    // fall -- inside another tensor's range, or at a start a longer tensor shares --
                    // and reading it consumes nothing, so the position stays where it is.
                    if (entry.Start == entry.End)
                    {
                        if (onto is not null)
                            tensors.Add(new SafeTensor(entry.Name,
                                onto.ReadTensor(new Shape(entry.Shape), entry.DType, source),
                                entry.DTypeName, entry.Shape, entry.Metadata));
                        continue;
                    }
                    if (entry.Start < position)
                        throw new InvalidOperationException(
                            $"Tensor '{entry.Name}' has data_offsets [{entry.Start}, {entry.End}), which overlap " +
                            "the bytes of the tensor before it; SafeTensors tensors do not share bytes.");
                    if (onto is null)
                    {
                        Skip(source, entry.End - position, entry.Name, origin);
                        position = entry.End;
                        continue;
                    }
                    Skip(source, entry.Start - position, entry.Name, origin);
                    TensorData data;
                    try
                    {
                        data = onto.ReadTensor(new Shape(entry.Shape), entry.DType, source);
                    }
                    catch (EndOfStreamException e)
                    {
                        throw Truncated(entry.Name, origin, e);
                    }
                    tensors.Add(new SafeTensor(entry.Name, data, entry.DTypeName, entry.Shape, entry.Metadata));
                    position = entry.End;
                }
            }
            catch
            {
                foreach (var t in tensors) t.Data.Delete();
                throw;
            }
            return tensors;
        }

        private sealed record HeaderEntry(
            string Name, string DTypeName, DType DType, long[] Shape, long Elements, long Start, long End,
            Dictionary<string, object> Metadata);

        /// <summary>One tensor's header entry, checked against what the payload can hold.</summary>
        private static HeaderEntry ParseHeaderEntry(
            string tensorName, object tensorMeta, long dataOffset, long available, string origin)
        {
            var metaDict = JsonSerializer.Deserialize<Dictionary<string, object>>(JsonSerializer.Serialize(tensorMeta))
                ?? throw new InvalidOperationException("Failed to parse tensor metadata");
            var shape = ExtractShape(metaDict);
            var (startOffset, endOffset) = ExtractDataOffsets(metaDict);
            var dtypeName = ExtractDataType(metaDict);

            if (startOffset < 0 || endOffset < startOffset)
                throw new InvalidOperationException(
                    $"Tensor '{tensorName}' has invalid data_offsets [{startOffset}, {endOffset})");

            // Unsigned sum: both terms are non-negative longs, so this cannot overflow the way a
            // signed dataOffset + endOffset could for an absurd corrupt endOffset.
            ulong requiredBytes = (ulong)dataOffset + (ulong)endOffset;
            if (requiredBytes > (ulong)available)
                throw new ModelException(ErrorCodes.ST003, $"SafeTensor file '{origin}'",
                    $"truncated SafeTensor file — tensor '{tensorName}' declares data_offsets " +
                    $"[{startOffset}, {endOffset}), which requires the file to hold " +
                    $"{requiredBytes} bytes, but the file has {available} bytes. " +
                    "The file was likely cut short by an interrupted download or copy.");

            var dtype = SafeTensorDTypeToDType(dtypeName);
            long elements = 1;
            foreach (var d in shape)
            {
                if (d < 0)
                    throw new InvalidOperationException(
                        $"Tensor '{tensorName}' has a negative dimension in shape [{string.Join(", ", shape)}]");
                elements = checked(elements * d);
            }
            long expected = checked(elements * (TensorData.StorageBits(dtype) / 8));
            if (endOffset - startOffset != expected)
                throw new InvalidOperationException(
                    $"Tensor '{tensorName}' has data_offsets [{startOffset}, {endOffset}) covering " +
                    $"{endOffset - startOffset} bytes, but a {dtypeName} tensor of shape " +
                    $"[{string.Join(", ", shape)}] takes {expected}.");

            var additionalMetadata = metaDict
                .Where(kvp => kvp.Key != "shape" && kvp.Key != "data_offsets" && kvp.Key != "dtype")
                .ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
            return new HeaderEntry(tensorName, dtypeName, dtype, shape, elements, startOffset, endOffset, additionalMetadata);
        }

        /// <summary>Reads past <paramref name="count"/> bytes no tensor claims.</summary>
        private static void Skip(Stream source, long count, string nextTensor, string origin)
        {
            if (count == 0) return;
            var scratch = new byte[(int)Math.Min(count, 64 * 1024)];
            try
            {
                for (long left = count; left > 0;)
                {
                    int n = (int)Math.Min(left, scratch.Length);
                    source.ReadExactly(scratch, 0, n);
                    left -= n;
                }
            }
            catch (EndOfStreamException e)
            {
                throw Truncated(nextTensor, origin, e);
            }
        }

        private static ModelException Truncated(string tensorName, string origin, Exception inner)
            => new(ErrorCodes.ST003, $"SafeTensor file '{origin}'",
                $"truncated SafeTensor file — the data ends before the bytes of tensor '{tensorName}'. " +
                "The file was likely cut short by an interrupted download or copy.", inner);

        /// <summary>
        /// Extract shape array from metadata dictionary. An empty array is the valid encoding of
        /// a rank-0 scalar; only a missing or unparsable field is rejected — silently defaulting
        /// would let a corrupt header load garbage.
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
            throw new InvalidOperationException(
                "Tensor metadata is missing a valid 'shape' field — the header is corrupt or not SafeTensors.");
        }

        /// <summary>
        /// Extract data offsets from metadata dictionary. A missing or unparsable field is
        /// rejected — silently defaulting would make the truncation checks validate fabricated
        /// offsets and load garbage from a corrupt header.
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
            throw new InvalidOperationException(
                "Tensor metadata is missing a valid 'data_offsets' field (expected [begin, end]) — " +
                "the header is corrupt or not SafeTensors.");
        }

        /// <summary>
        /// Extract data type from metadata dictionary; a missing field is rejected rather than
        /// silently defaulted (the payload would be reinterpreted under the wrong dtype).
        /// </summary>
        private static string ExtractDataType(Dictionary<string, object> metaDict)
        {
            if (metaDict.TryGetValue("dtype", out var dtypeObj) &&
                dtypeObj.ToString() is { Length: > 0 } dtype)
            {
                return dtype;
            }
            throw new InvalidOperationException(
                "Tensor metadata is missing a valid 'dtype' field — the header is corrupt or not SafeTensors.");
        }
        /// <summary>The Shorokoo element type a SafeTensors dtype string names.</summary>
        private static DType SafeTensorDTypeToDType(string safeTensorDType)
            => safeTensorDType.ToUpperInvariant() switch
            {
                "BOOL" => DType.Bool,
                "I8" => DType.Int8,
                "I16" => DType.Int16,
                "I32" => DType.Int32,
                "I64" => DType.Int64,
                "U8" => DType.UInt8,
                "U16" => DType.UInt16,
                "U32" => DType.UInt32,
                "U64" => DType.UInt64,
                "F32" => DType.Float32,
                "F64" => DType.Float64,
                // F16/BF16 payloads are raw little-endian IEEE half / bfloat16 bit patterns (2 bytes
                // per element) — exactly the in-memory layout of the ushort-backed Float16/BFloat16
                // structs.
                "F16" => DType.Float16,
                "BF16" => DType.BFloat16,
                _ => throw new NotSupportedException(
                    $"Unsupported SafeTensor data type: {safeTensorDType}. " +
                    "Supported formats: BOOL, I8, I16, I32, I64, U8, U16, U32, U64, F16, BF16, F32, F64.")
            };
    }
}

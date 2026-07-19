using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using Shorokoo.Core.Factory.IR;

namespace Shorokoo.Onnx
{
    /// <summary>
    /// Read-side support for the standard ONNX external-data mechanism: initializer
    /// bytes stored in a side file next to the model, referenced from
    /// <see cref="TensorProto.ExternalDatas"/> (<c>location</c>/<c>offset</c>/<c>length</c>
    /// entries) with <see cref="TensorProto.data_location"/> set to
    /// <see cref="TensorProto.DataLocation.External"/>.
    ///
    /// <para>
    /// <see cref="LoadIntoModel"/> materializes every external tensor in a freshly
    /// deserialized <see cref="ModelProto"/> into its inline <see cref="TensorProto.RawData"/>
    /// form, so the rest of the import pipeline never sees an external tensor. All dtypes
    /// the inline path supports are supported externally — the side file carries exactly
    /// the little-endian raw bytes that <c>raw_data</c> would have carried.
    /// </para>
    ///
    /// <para>
    /// Failure modes are loud and name the tensor and the file: a <c>location</c> that
    /// escapes the model's directory (absolute path or <c>..</c> traversal), a missing
    /// side file, an out-of-range or unparsable <c>offset</c>/<c>length</c>, a
    /// <c>length</c> that contradicts the tensor's shape/dtype, and an external-data
    /// model loaded from a stream/bytes with no base directory all throw
    /// <see cref="ModelException"/> (no silent zero-fill).
    /// </para>
    /// </summary>
    internal static class OnnxExternalData
    {
        internal const string LocationKey = "location";
        internal const string OffsetKey = "offset";
        internal const string LengthKey = "length";

        /// <summary>
        /// Materializes every <c>data_location=EXTERNAL</c> tensor in <paramref name="model"/>
        /// into inline <see cref="TensorProto.RawData"/>, resolving <c>location</c> keys
        /// against <paramref name="baseDirectory"/> (the model file's directory).
        /// <paramref name="baseDirectory"/> may be null when the model came from a
        /// stream/bytes — then any external tensor fails loudly.
        /// </summary>
        internal static void LoadIntoModel(ModelProto model, string? baseDirectory)
        {
            // One open stream per distinct side file: several tensors commonly slice
            // into the same file at different offsets.
            Dictionary<string, FileStream>? openFiles = null;
            try
            {
                foreach (var tensor in EnumerateAllTensors(model))
                {
                    if (tensor.data_location != TensorProto.DataLocation.External)
                        continue;
                    openFiles ??= new Dictionary<string, FileStream>(StringComparer.Ordinal);
                    MaterializeExternalTensor(tensor, baseDirectory, openFiles);
                }
            }
            finally
            {
                if (openFiles is not null)
                    foreach (var fs in openFiles.Values)
                        fs.Dispose();
            }
        }

        /// <summary>
        /// Walks every <see cref="TensorProto"/> reachable from the model: graph
        /// initializers, sparse initializers, tensor-valued node attributes, and all of
        /// those recursively through subgraph attributes and function bodies.
        /// </summary>
        internal static IEnumerable<TensorProto> EnumerateAllTensors(ModelProto model)
        {
            if (model.Graph is not null)
                foreach (var t in EnumerateGraphTensors(model.Graph))
                    yield return t;

            foreach (var fn in model.Functions)
                foreach (var node in fn.Nodes)
                    foreach (var t in EnumerateNodeTensors(node))
                        yield return t;
        }

        private static IEnumerable<TensorProto> EnumerateGraphTensors(GraphProto graph)
        {
            foreach (var init in graph.Initializers)
                yield return init;

            foreach (var sparse in graph.SparseInitializers)
            {
                if (sparse.Values is not null) yield return sparse.Values;
                if (sparse.Indices is not null) yield return sparse.Indices;
            }

            foreach (var node in graph.Nodes)
                foreach (var t in EnumerateNodeTensors(node))
                    yield return t;
        }

        private static IEnumerable<TensorProto> EnumerateNodeTensors(NodeProto node)
        {
            foreach (var attr in node.Attributes)
            {
                if (attr.T is not null)
                    yield return attr.T;
                foreach (var t in attr.Tensors)
                    yield return t;
                if (attr.G is not null)
                    foreach (var t in EnumerateGraphTensors(attr.G))
                        yield return t;
                foreach (var g in attr.Graphs)
                    foreach (var t in EnumerateGraphTensors(g))
                        yield return t;
            }
        }

        private static void MaterializeExternalTensor(
            TensorProto tensor, string? baseDirectory, Dictionary<string, FileStream> openFiles)
        {
            var tensorName = string.IsNullOrEmpty(tensor.Name) ? "<unnamed>" : tensor.Name;
            var tensorInfo = $"tensor '{tensorName}'";

            if (baseDirectory is null)
                throw new ModelException(ErrorCodes.XD001, tensorInfo,
                    "the tensor's data is stored externally (data_location=EXTERNAL), but the model was " +
                    "loaded from a stream/bytes with no base directory to resolve external files against. " +
                    "Load the model from a file path, or pass the externalDataDirectory argument of " +
                    "OnnxModelImporter.FromOnnxModelToFastGraph.");

            string? location = null;
            long offset = 0;
            long? length = null;
            foreach (var entry in tensor.ExternalDatas)
            {
                switch (entry.Key)
                {
                    case LocationKey:
                        location = entry.Value;
                        break;
                    case OffsetKey:
                        if (!long.TryParse(entry.Value, NumberStyles.Integer, CultureInfo.InvariantCulture, out offset))
                            throw new ModelException(ErrorCodes.XD005, tensorInfo,
                                $"the external_data 'offset' entry '{entry.Value}' is not a valid integer.");
                        break;
                    case LengthKey:
                        if (!long.TryParse(entry.Value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsedLength))
                            throw new ModelException(ErrorCodes.XD005, tensorInfo,
                                $"the external_data 'length' entry '{entry.Value}' is not a valid integer.");
                        length = parsedLength;
                        break;
                    // "checksum" and any custom keys are ignored, per the ONNX spec.
                }
            }

            if (string.IsNullOrEmpty(location))
                throw new ModelException(ErrorCodes.XD002, tensorInfo,
                    "data_location is EXTERNAL but the external_data list has no 'location' entry.");

            var baseFull = Path.GetFullPath(baseDirectory);
            if (Path.IsPathRooted(location))
                throw new ModelException(ErrorCodes.XD003, tensorInfo,
                    $"the external data location '{location}' is an absolute path; external data must " +
                    "live inside the model's directory.");
            var resolved = Path.GetFullPath(Path.Combine(baseFull, location));
            var basePrefix = Path.EndsInDirectorySeparator(baseFull)
                ? baseFull
                : baseFull + Path.DirectorySeparatorChar;
            if (!resolved.StartsWith(basePrefix, StringComparison.Ordinal))
                throw new ModelException(ErrorCodes.XD003, tensorInfo,
                    $"the external data location '{location}' escapes the model's directory " +
                    $"('{baseFull}'); external data must live inside it.");

            if (!openFiles.TryGetValue(resolved, out var fs))
            {
                if (!File.Exists(resolved))
                    throw new ModelException(ErrorCodes.XD004, tensorInfo,
                        $"the external data file '{resolved}' does not exist.");
                fs = File.OpenRead(resolved);
                openFiles[resolved] = fs;
            }

            long fileLength = fs.Length;
            long expected = TryGetExpectedByteLength(tensor);

            if (offset < 0)
                throw new ModelException(ErrorCodes.XD005, tensorInfo,
                    $"the external_data offset {offset} into file '{resolved}' is negative.");
            if (length is < 0)
                throw new ModelException(ErrorCodes.XD005, tensorInfo,
                    $"the external_data length {length} for file '{resolved}' is negative.");

            if (length is long l && expected >= 0 && l != expected)
                throw new ModelException(ErrorCodes.XD006, tensorInfo,
                    $"the external_data length {l} (file '{resolved}') does not match the " +
                    $"{expected} bytes implied by the tensor's shape and dtype.");

            if (offset > fileLength)
                throw new ModelException(ErrorCodes.XD005, tensorInfo,
                    $"the external_data offset {offset} is out of range for file '{resolved}' " +
                    $"({fileLength} bytes).");

            // Per the ONNX spec, a missing length means "the rest of the file"; when the
            // dtype/shape pin down the exact byte count we use that instead, so a
            // trailing-slack side file still validates. The range check is written
            // subtraction-style: with offset within the file and both operands
            // non-negative it cannot overflow, whereas `offset + readLength` could wrap
            // for near-long.MaxValue values from a malformed model and slip past the
            // check into an opaque allocation failure.
            long readLength = length ?? (expected >= 0 ? expected : fileLength - offset);
            if (readLength > fileLength - offset)
                throw new ModelException(ErrorCodes.XD005, tensorInfo,
                    $"the external data range [offset {offset}, length {readLength}] is out of range " +
                    $"for file '{resolved}' ({fileLength} bytes).");

            var bytes = new byte[readLength];
            fs.Seek(offset, SeekOrigin.Begin);
            fs.ReadExactly(bytes);

            tensor.RawData = bytes;
            tensor.ExternalDatas.Clear();
            tensor.Resetdata_location();
        }

        /// <summary>
        /// The byte count implied by the tensor's dtype and dims, or -1 when the dtype
        /// has no fixed per-element width (e.g. String) so the count cannot be derived.
        /// </summary>
        internal static long TryGetExpectedByteLength(TensorProto tensor)
        {
            var dtype = (DType)tensor.data_type;
            int bits;
            try
            {
                bits = dtype.EncodingBitCount;
            }
            catch (UnsupportedDTypeException)
            {
                return -1;
            }

            long count = 1;
            if (tensor.Dims is not null)
                foreach (var d in tensor.Dims)
                    count *= d;
            return count * bits / 8;
        }
    }
}

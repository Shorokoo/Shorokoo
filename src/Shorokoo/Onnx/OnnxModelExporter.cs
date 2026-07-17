using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Shorokoo.Core.Factory.IR;

namespace Shorokoo.Onnx
{
    /// <summary>
    /// Options for <see cref="OnnxModelExporter.SaveWithExternalData"/>.
    /// </summary>
    public sealed class OnnxExternalDataOptions
    {
        /// <summary>
        /// Initializers whose raw data is at least this many bytes are written to the
        /// side file; smaller ones stay inline in the .onnx. Default 1024 (the
        /// conventional ONNX threshold).
        /// </summary>
        public long SizeThreshold { get; init; } = 1024;

        /// <summary>
        /// Each externalized tensor's data starts at a multiple of this many bytes in
        /// the side file (zero-padded), so the file can be memory-mapped page-aligned.
        /// Default 4096.
        /// </summary>
        public int Alignment { get; init; } = 4096;
    }

    /// <summary>
    /// Saves an ONNX <see cref="ModelProto"/> to disk, either self-contained
    /// (<see cref="Save(ModelProto, string)"/>, all initializer bytes inline — the default
    /// export shape, unchanged from serializing the proto yourself) or with large
    /// initializers moved to a single side file
    /// (<see cref="SaveWithExternalData"/>, the standard ONNX external-data layout),
    /// which removes protobuf's 2 GB message ceiling on model size.
    /// </summary>
    public static class OnnxModelExporter
    {
        /// <summary>
        /// Protobuf encodes every message with 32-bit lengths, so a self-contained model
        /// whose tensor payloads reach 2 GB cannot be serialized at all.
        /// </summary>
        internal const long MaxSelfContainedTensorBytes = int.MaxValue;

        /// <summary>
        /// Saves the model self-contained (all initializer bytes inline). Byte-identical
        /// to <c>ProtoBuf.Serializer.Serialize(File.Create(path), model)</c>, plus a clear
        /// error — instead of a protobuf failure — when the model's tensor data exceeds
        /// the 2 GB protobuf ceiling, suggesting <see cref="SaveWithExternalData"/>.
        /// </summary>
        public static void Save(ModelProto model, string filePath)
            => Save(model, filePath, MaxSelfContainedTensorBytes);

        internal static void Save(ModelProto model, string filePath, long maxTensorBytes)
        {
            if (model is null) throw new ArgumentNullException(nameof(model));

            long totalTensorBytes = OnnxExternalData.EnumerateAllTensors(model)
                .Sum(t => (long)(t.RawData?.Length ?? 0));
            if (totalTensorBytes > maxTensorBytes)
                throw new ModelException(ErrorCodes.XD007, $"model '{filePath}'",
                    $"the model's tensor data totals {totalTensorBytes:N0} bytes, which exceeds the " +
                    $"{maxTensorBytes:N0}-byte protobuf message ceiling for a self-contained .onnx file. " +
                    "Use OnnxModelExporter.SaveWithExternalData to store large initializers in a side file.");

            using var fs = File.Create(filePath);
            ProtoBuf.Serializer.Serialize(fs, model);
        }

        /// <summary>
        /// Saves the model with every top-level graph initializer at or above
        /// <see cref="OnnxExternalDataOptions.SizeThreshold"/> bytes written to a single
        /// side file named <c>{model file name}.data</c> in the model's directory
        /// (e.g. <c>model.onnx</c> → <c>model.onnx.data</c>), in the standard ONNX
        /// external-data layout (<c>data_location=EXTERNAL</c> plus
        /// <c>location</c>/<c>offset</c>/<c>length</c> entries). Small initializers stay
        /// inline. Tensor data is written in initializer order, each aligned to
        /// <see cref="OnnxExternalDataOptions.Alignment"/>, so the same model produces
        /// byte-identical <c>.onnx</c> + <c>.onnx.data</c> pairs across runs. When no
        /// initializer reaches the threshold, no side file is written and the output
        /// equals <see cref="Save(ModelProto, string)"/>.
        /// The passed <paramref name="model"/> is left unmodified (externalized tensors
        /// are restored to their inline form before returning).
        /// </summary>
        public static void SaveWithExternalData(
            ModelProto model, string filePath, OnnxExternalDataOptions? options = null)
        {
            if (model is null) throw new ArgumentNullException(nameof(model));
            options ??= new OnnxExternalDataOptions();
            if (options.SizeThreshold < 0)
                throw new ArgumentOutOfRangeException(nameof(options), options.SizeThreshold,
                    "SizeThreshold must be non-negative.");
            if (options.Alignment < 1)
                throw new ArgumentOutOfRangeException(nameof(options), options.Alignment,
                    "Alignment must be at least 1.");

            var fullPath = Path.GetFullPath(filePath);
            var dataFileName = Path.GetFileName(fullPath) + ".data";
            var dataPath = fullPath + ".data";

            var externalized = (model.Graph?.Initializers ?? [])
                .Where(t => t.RawData is not null
                         && t.RawData.LongLength >= options.SizeThreshold
                         && t.data_location != TensorProto.DataLocation.External)
                .ToList();

            if (externalized.Count == 0)
            {
                Save(model, filePath);
                return;
            }

            var savedRawData = externalized.Select(t => t.RawData).ToList();
            try
            {
                using (var dataStream = File.Create(dataPath))
                {
                    foreach (var tensor in externalized)
                    {
                        long padding = (options.Alignment - dataStream.Position % options.Alignment)
                            % options.Alignment;
                        if (padding > 0)
                            dataStream.Write(new byte[padding]);

                        long offset = dataStream.Position;
                        var raw = tensor.RawData!;
                        dataStream.Write(raw);

                        tensor.RawData = null!;
                        tensor.data_location = TensorProto.DataLocation.External;
                        tensor.ExternalDatas.Add(new StringStringEntryProto
                        { Key = OnnxExternalData.LocationKey, Value = dataFileName });
                        tensor.ExternalDatas.Add(new StringStringEntryProto
                        { Key = OnnxExternalData.OffsetKey, Value = offset.ToString(CultureInfo.InvariantCulture) });
                        tensor.ExternalDatas.Add(new StringStringEntryProto
                        { Key = OnnxExternalData.LengthKey, Value = raw.LongLength.ToString(CultureInfo.InvariantCulture) });
                    }
                }

                using var fs = File.Create(fullPath);
                ProtoBuf.Serializer.Serialize(fs, model);
            }
            finally
            {
                // Restore the caller's proto to its self-contained form.
                for (int i = 0; i < externalized.Count; i++)
                {
                    externalized[i].RawData = savedRawData[i];
                    externalized[i].ExternalDatas.Clear();
                    externalized[i].Resetdata_location();
                }
            }
        }
    }
}

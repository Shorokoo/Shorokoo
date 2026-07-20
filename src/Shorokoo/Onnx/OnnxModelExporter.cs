using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Core.Utils;
using Shorokoo.Graph;

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
                .Sum(TensorPayloadBytes);
            if (totalTensorBytes > maxTensorBytes)
                throw new ModelException(ErrorCodes.XD007, $"model '{filePath}'",
                    $"the model's tensor data totals {totalTensorBytes:N0} bytes, which exceeds the " +
                    $"{maxTensorBytes:N0}-byte protobuf message ceiling for a self-contained .onnx file. " +
                    "Use OnnxModelExporter.SaveWithExternalData to store large initializers in a side file.");

            using var fs = File.Create(filePath);
            ProtoBuf.Serializer.Serialize(fs, model);
        }

        /// <summary>
        /// The tensor's serialized payload size: raw_data plus every typed data field
        /// (a tensor may carry its bytes in float_data/int32_data/int64_data/... instead
        /// of raw_data — e.g. the RNG key vector initializer uses int64_data), so the
        /// 2 GB pre-check cannot be bypassed by typed-field storage.
        /// </summary>
        private static long TensorPayloadBytes(TensorProto t)
        {
            long total = t.RawData?.LongLength ?? 0;
            total += (long)(t.FloatDatas?.Length ?? 0) * sizeof(float);
            total += (long)(t.Int32Datas?.Length ?? 0) * sizeof(int);
            total += (long)(t.Int64Datas?.Length ?? 0) * sizeof(long);
            total += (long)(t.DoubleDatas?.Length ?? 0) * sizeof(double);
            total += (long)(t.Uint64Datas?.Length ?? 0) * sizeof(ulong);
            foreach (var s in t.StringDatas)
                total += s?.LongLength ?? 0;
            return total;
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
        /// initializer reaches the threshold, no side file is written (a stale one from
        /// a previous save of the same path is removed) and the output equals
        /// <see cref="Save(ModelProto, string)"/>.
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

            // Concrete models only: externalization covers exactly the top-level graph
            // initializers, which is complete when — and only when — all weights live
            // there. Shorokoo-written models carry the graph-kind metadata tag; only
            // untagged (foreign) protos fall back to op-scan classification.
            var stage = SrkFileFormat.TryReadKindTag(model) ?? SrkFileFormat.DetectStage(model);
            if (stage != GraphKind.ConcreteModel)
                throw new ModelException(ErrorCodes.XD008, $"model '{filePath}'",
                    $"SaveWithExternalData requires a '{SrkFileFormat.StageName(GraphKind.ConcreteModel)}' " +
                    $"graph, but this model is a '{SrkFileFormat.StageName(stage)}'. It externalizes only " +
                    "top-level graph initializers — complete exactly when every weight lives there, i.e. " +
                    "for a concrete model. Lower the graph first (ToConcreteArchitecture -> ToConcreteModel), " +
                    "build the ONNX model from that, then save it.");

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
                // A side file from a previous external save of the same path would now
                // be orphaned (nothing in the fresh self-contained .onnx references it);
                // remove it so the directory reflects exactly this save.
                File.Delete(dataPath);
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
            catch
            {
                // Don't leave a partial side file behind on a failed save; cleanup is
                // best-effort — the original exception is the one that matters.
                try { File.Delete(dataPath); } catch { }
                throw;
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

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.OnnxNodes;
using RandN.Distributions.UnitInterval;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Modules;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Onnx;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Factory.IR;
using IR = Shorokoo.Core.Factory.IR;

namespace Shorokoo.Onnx
{
    /// <summary>
    /// Imports serialized ONNX models into <see cref="ComputationGraph"/>s
    /// (deserializes the ModelProto, materializes any external tensor data, then builds
    /// the graph via <c>OnnxModelReader</c>). Imported data carries no kind stamp, so
    /// the returned graph's <see cref="ComputationGraph.Kind"/> is classified by
    /// op-scanning (<see cref="Shorokoo.Core.Utils.SrkFileFormat.DetectStage"/>) — the
    /// sanctioned fallback for foreign/headerless data.
    ///
    /// <para>
    /// Models using the standard ONNX external-data mechanism (initializer bytes in a
    /// side file referenced from <c>TensorProto.external_data</c>) load transparently
    /// from a file path — <c>location</c> keys resolve against the model file's
    /// directory. When loading from a stream/bytes, pass
    /// <c>externalDataDirectory</c>; without it, a model that requires external data
    /// fails with a clear error (never a silent zero-fill).
    /// </para>
    /// </summary>
    public static class OnnxModelImporter
    {
        /// <summary>
        /// Imports an ONNX model from a stream. <paramref name="externalDataDirectory"/>
        /// is the directory external-data <c>location</c> keys resolve against; when
        /// null, a model requiring external data fails loudly.
        /// </summary>
        public static ComputationGraph FromOnnxModel(Stream inputStream, string? externalDataDirectory = null)
            => Wrap(FromOnnxModelToInternalGraph(inputStream, externalDataDirectory));

        /// <summary>
        /// Imports an ONNX model from a file path. External tensor data (if any)
        /// resolves against the file's directory.
        /// </summary>
        public static ComputationGraph FromOnnxModel(string filePath)
            => Wrap(FromOnnxModelToInternalGraph(filePath));

        /// <summary>
        /// Imports an ONNX model from in-memory bytes. <paramref name="externalDataDirectory"/>
        /// is the directory external-data <c>location</c> keys resolve against; when
        /// null, a model requiring external data fails loudly.
        /// </summary>
        public static ComputationGraph FromOnnxModel(byte[] rawData, string? externalDataDirectory = null)
            => Wrap(FromOnnxModelToInternalGraph(rawData, externalDataDirectory));

        private static ComputationGraph Wrap(InternalComputationGraph graph)
            => new(graph, Shorokoo.Core.Utils.SrkFileFormat.DetectStage(graph));

        /// <summary>Internal-graph form of <see cref="FromOnnxModel(Stream, string?)"/>.</summary>
        internal static InternalComputationGraph FromOnnxModelToInternalGraph(Stream inputStream, string? externalDataDirectory = null)
        {
            var model = ProtoBuf.Serializer.Deserialize<IR.ModelProto>(inputStream);
            OnnxExternalData.LoadIntoModel(model, externalDataDirectory);
            var reader = new OnnxModelReader(model);
            return reader.BuildInternalComputationGraph();
        }

        /// <summary>Internal-graph form of <see cref="FromOnnxModel(string)"/>.</summary>
        internal static InternalComputationGraph FromOnnxModelToInternalGraph(string filePath)
        {
            using var fileReaderStream = File.OpenRead(filePath);
            return FromOnnxModelToInternalGraph(
                fileReaderStream,
                Path.GetDirectoryName(Path.GetFullPath(filePath)));
        }

        /// <summary>Internal-graph form of <see cref="FromOnnxModel(byte[], string?)"/>.</summary>
        internal static InternalComputationGraph FromOnnxModelToInternalGraph(byte[] rawData, string? externalDataDirectory = null)
        {
            using var stream = new MemoryStream(rawData);
            return FromOnnxModelToInternalGraph(stream, externalDataDirectory);
        }
    }
}

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
    /// Imports serialized ONNX models into <see cref="FastComputationGraph"/>s
    /// (deserializes the ModelProto, materializes any external tensor data, then builds
    /// the graph via <c>OnnxModelReader</c>).
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
        public static FastComputationGraph FromOnnxModelToFastGraph(Stream inputStream, string? externalDataDirectory = null)
        {
            var model = ProtoBuf.Serializer.Deserialize<IR.ModelProto>(inputStream);
            OnnxExternalData.LoadIntoModel(model, externalDataDirectory);
            var reader = new OnnxModelReader(model);
            return reader.BuildFastComputationGraph();
        }

        /// <summary>
        /// Imports an ONNX model from a file path. External tensor data (if any)
        /// resolves against the file's directory.
        /// </summary>
        public static FastComputationGraph FromOnnxModelToFastGraph(string filePath)
        {
            using var fileReaderStream = File.OpenRead(filePath);
            return FromOnnxModelToFastGraph(
                fileReaderStream,
                Path.GetDirectoryName(Path.GetFullPath(filePath)));
        }

        /// <summary>
        /// Imports an ONNX model from in-memory bytes. <paramref name="externalDataDirectory"/>
        /// is the directory external-data <c>location</c> keys resolve against; when
        /// null, a model requiring external data fails loudly.
        /// </summary>
        public static FastComputationGraph FromOnnxModelToFastGraph(byte[] rawData, string? externalDataDirectory = null)
        {
            using var stream = new MemoryStream(rawData);
            return FromOnnxModelToFastGraph(stream, externalDataDirectory);
        }
    }
}

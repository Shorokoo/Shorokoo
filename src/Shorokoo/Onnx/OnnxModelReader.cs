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
    /// (deserializes the ModelProto, then builds the graph via <c>OnnxModelReader</c>).
    /// </summary>
    public static class OnnxModelImporter
    {
        /// <summary>Imports an ONNX model from a stream.</summary>
        public static FastComputationGraph FromOnnxModelToFastGraph(Stream inputStream)
        {
            var model = ProtoBuf.Serializer.Deserialize<IR.ModelProto>(inputStream);
            var reader = new OnnxModelReader(model);
            return reader.BuildFastComputationGraph();
        }

        /// <summary>Imports an ONNX model from a file path.</summary>
        public static FastComputationGraph FromOnnxModelToFastGraph(string filePath)
        {
            using var fileReaderStream = File.OpenRead(filePath);
            return FromOnnxModelToFastGraph(fileReaderStream);
        }

        /// <summary>Imports an ONNX model from in-memory bytes.</summary>
        public static FastComputationGraph FromOnnxModelToFastGraph(byte[] rawData)
        {
            using var stream = new MemoryStream(rawData);
            return FromOnnxModelToFastGraph(stream);
        }
    }
}

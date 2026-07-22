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
    /// the graph via <c>OnnxModelReader</c>). Models written by Shorokoo carry a
    /// graph-kind metadata tag, so the returned graph's
    /// <see cref="ComputationGraph.Kind"/> reloads as the kind it was saved with;
    /// untagged (foreign) models are classified by op-scanning
    /// (<see cref="Shorokoo.Core.Utils.SrkFileFormat.DetectStage(InternalComputationGraph)"/>) — the
    /// sanctioned fallback for data that arrives without a stamp.
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
            => Wrap(FromOnnxModelWithKindTag(inputStream, externalDataDirectory));

        /// <summary>
        /// Imports an ONNX model from a file path. External tensor data (if any)
        /// resolves against the file's directory.
        /// </summary>
        public static ComputationGraph FromOnnxModel(string filePath)
        {
            using var fileReaderStream = File.OpenRead(filePath);
            return Wrap(FromOnnxModelWithKindTag(
                fileReaderStream, Path.GetDirectoryName(Path.GetFullPath(filePath))));
        }

        /// <summary>
        /// Imports an ONNX model from in-memory bytes. <paramref name="externalDataDirectory"/>
        /// is the directory external-data <c>location</c> keys resolve against; when
        /// null, a model requiring external data fails loudly.
        /// </summary>
        public static ComputationGraph FromOnnxModel(byte[] rawData, string? externalDataDirectory = null)
        {
            using var stream = new MemoryStream(rawData);
            return Wrap(FromOnnxModelWithKindTag(stream, externalDataDirectory));
        }

        private static ComputationGraph Wrap((InternalComputationGraph Graph, Shorokoo.Graph.GraphKind? TaggedKind) import)
            => new(import.Graph,
                import.TaggedKind ?? Shorokoo.Core.Utils.SrkFileFormat.DetectStage(import.Graph));

        /// <summary>
        /// Internal-graph import that also surfaces the graph-kind metadata tag the
        /// Shorokoo ONNX builders stamp into the model (null for untagged/foreign
        /// models). A tag that is structurally impossible for the imported content
        /// (per <see cref="Shorokoo.Core.Utils.SrkFileFormat.DescribeKindViolation"/>)
        /// fails loudly — the file is corrupt or written by an incompatible tool.
        /// </summary>
        internal static (InternalComputationGraph Graph, Shorokoo.Graph.GraphKind? TaggedKind)
            FromOnnxModelWithKindTag(Stream inputStream, string? externalDataDirectory = null)
        {
            var model = ProtoBuf.Serializer.Deserialize<IR.ModelProto>(inputStream);
            OnnxExternalData.LoadIntoModel(model, externalDataDirectory);
            var taggedKind = Shorokoo.Core.Utils.SrkFileFormat.TryReadKindTag(model);
            var reader = new OnnxModelReader(model);
            var graph = reader.BuildInternalComputationGraph();

            RestoreSignatureIONames(model, graph);

            if (taggedKind is { } kind &&
                Shorokoo.Core.Utils.SrkFileFormat.DescribeKindViolation(graph, kind) is { } violation)
                throw new InvalidDataException(
                    $"the model's '{Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames.ShrkMetaGraphKind}' " +
                    $"metadata declares kind '{Shorokoo.Core.Utils.SrkFileFormat.StageName(kind)}', but the " +
                    $"graph content is incompatible: {violation} " +
                    "The file is corrupt or was written by an incompatible tool.");

            return (graph, taggedKind);
        }

        /// <summary>
        /// Restores the graph's human-readable signature I/O names from the
        /// <c>shrk_input_names</c> / <c>shrk_output_names</c> model metadata written by
        /// internal-dialect exports (whose graph-I/O ValueInfos must keep raw
        /// <c>N{k}_T{s}</c> tensor ids). Applied only when a list parses and its length
        /// matches the reconstructed graph's I/O count; otherwise (foreign models, files
        /// written before the tag existed) the proto names stand.
        /// </summary>
        private static void RestoreSignatureIONames(IR.ModelProto model, InternalComputationGraph graph)
        {
            foreach (var prop in model.MetadataProps)
            {
                var target = prop.Key switch
                {
                    OnnxOpAttributeNames.ShrkMetaInputNames => graph.InputUniqueNames,
                    OnnxOpAttributeNames.ShrkMetaOutputNames => graph.OutputUniqueNames,
                    _ => null,
                };
                if (target is null) continue;

                List<string?>? names;
                try
                {
                    names = System.Text.Json.JsonSerializer.Deserialize<List<string?>>(prop.Value);
                }
                catch (System.Text.Json.JsonException)
                {
                    continue;
                }
                if (names is null || names.Count != target.Count) continue;

                target.Clear();
                target.AddRange(names);
            }
        }

        /// <summary>Internal-graph form of <see cref="FromOnnxModel(Stream, string?)"/>.</summary>
        internal static InternalComputationGraph FromOnnxModelToInternalGraph(Stream inputStream, string? externalDataDirectory = null)
            => FromOnnxModelWithKindTag(inputStream, externalDataDirectory).Graph;

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

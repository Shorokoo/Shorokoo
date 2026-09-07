using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using Shorokoo.Core.Factory.CSharpFactory;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;

namespace Shorokoo.Graph
{
    /// <summary>
    /// A point of the <c>ToConcreteArchitecture</c> lowering pipeline at which a snapshot of the
    /// graph can be requested, listed in the order the pipeline reaches them. Every value names a
    /// pass that runs, so every requested point writes its file — but the pipeline has stages with
    /// no point of their own, so this is not an enumeration of it. The names are historical and do
    /// not always match the stage names a <see cref="BuildProgress"/> sink reports.
    /// </summary>
    public enum GraphCreationPoint
    {
        /// <summary>After local ModelIds are assigned. A graph built from modules already carries
        /// them, so this snapshot is normally the graph as passed in — the baseline to diff the
        /// later points against.</summary>
        AfterApplyIdentifierTemplates,

        /// <summary>After every sub-module and function is inlined.</summary>
        AfterInlineAllModulesAndFunctions,

        /// <summary>After the model-global RNG execution counter (<c>RngExecutionCounter</c>) is
        /// wired into every runtime random feed.</summary>
        AfterInjectRngExecutionCounter,

        /// <summary>After parameter references become id-refs.</summary>
        AfterConvertToIdRefModelParams,

        /// <summary>After model structs, model hyperparameters and model ids are unpacked.</summary>
        AfterUnpackModelStruct,

        /// <summary>After tensor structs are unpacked into their individual tensors.</summary>
        AfterUnpackTensorStructs,

        /// <summary>After the id-refs are resolved into the trainable parameters themselves.
        /// Taken before the unhinted-input check, so a build that fails it still leaves the
        /// snapshot that explains why.</summary>
        AfterProcessTrainableParameters,

        /// <summary>After the first simplification pass: constant folding, loop unrolling and
        /// branch selection.</summary>
        AfterFirstSimplify,

        /// <summary>After Shorokoo's operator variants (e.g. <c>SHRK_CONV</c>) are lowered to their
        /// standard ONNX counterparts.</summary>
        AfterLowerAttributeTensorOps,

        /// <summary>After autodiff expansion.</summary>
        AfterExpandAutoGrad,

        /// <summary>The final concrete architecture graph, as returned.</summary>
        FinalGraph
    }

    /// <summary>
    /// Saves a snapshot of the graph, as compilable C#, at each requested point of the
    /// <c>ToConcreteArchitecture</c> lowering pipeline. A point with no entry writes nothing.
    /// </summary>
    public class DebugRequests
    {
        private readonly ImmutableDictionary<GraphCreationPoint, string> _debugPoints;

        /// <summary>
        /// Constructor accepting a list of tuples (GraphCreationPoint, filepath)
        /// </summary>
        public DebugRequests(IEnumerable<(GraphCreationPoint point, string filepath)> debugPoints)
        {
            var builder = ImmutableDictionary.CreateBuilder<GraphCreationPoint, string>();
            foreach (var (point, filepath) in debugPoints)
            {
                builder[point] = filepath;
            }
            _debugPoints = builder.ToImmutable();
        }

        /// <summary>
        /// Constructor accepting a dictionary
        /// </summary>
        public DebugRequests(IDictionary<GraphCreationPoint, string> debugPoints)
        {
            _debugPoints = debugPoints.ToImmutableDictionary();
        }

        /// <summary>
        /// Writes <paramref name="graph"/> as C# to the file requested for <paramref name="point"/>,
        /// creating the directory and replacing any existing file. Does nothing when the point was
        /// not requested.
        /// </summary>
        public void PrintDebug(InternalComputationGraph graph, GraphCreationPoint point)
        {
            if (_debugPoints.TryGetValue(point, out var filepath))
            {
                var directory = Path.GetDirectoryName(filepath);
                if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
                {
                    Directory.CreateDirectory(directory);
                }

                if (File.Exists(filepath))
                    File.Delete(filepath);

                var code = new CSharpModelBuilder().BuildFullGraph(graph, "CodeModel");
                using var textWriter = File.CreateText(filepath);
                textWriter.WriteLine(code);
            }
        }
    }
}

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
    /// Enum representing different points where VirtualGraph is constructed during ToConcreteArchitecture processing
    /// </summary>
    public enum GraphCreationPoint
    {
        AfterInlineAllModulesAndFunctions,
        AfterProcessTrainableParameters,
        AfterProcessAllModelHyperparamRefs,
        AfterProcessModelSequences,
        AfterProcessAccessibleModuleSetHyperparams,
        AfterUnrollModuleLoop,
        AfterSimplify,
        AfterSimplifyTrainableParamInitializers,
        AfterLowerStateUpdateNodes,
        AfterFirstSimplify,
        AfterExpandAutoGrad,
        AfterSecondSimplify,
        FinalGraph
    }

    /// <summary>
    /// Class to handle debug output requests for VirtualGraph instances during ToConcreteArchitecture processing
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

        public void PrintDebug(FastComputationGraph graph, GraphCreationPoint point)
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

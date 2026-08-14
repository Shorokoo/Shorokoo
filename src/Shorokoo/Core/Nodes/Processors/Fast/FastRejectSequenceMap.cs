using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Graph;
using System;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Rejects an imported <c>SequenceMap</c>. The generic OPEN/CLOSE definitions let one
    /// through structurally, so without this pass it imports silently and exports
    /// body-less — a corrupted model rather than a failure. An op Shorokoo cannot execute
    /// has to say so at the import that introduced it.
    /// </summary>
    /// <remarks>
    /// Scans one graph only, and does not descend into function bodies: a body is not
    /// reachable as node data (<see cref="Shorokoo.Core.Function.OriginalFastGraph"/>
    /// thaws a fresh copy on every access), so a recursive walk would thaw every
    /// reachable function on every import just to read one op code. The ONNX reader
    /// instead applies this to each function body as it builds it — while that body is
    /// still the mutable graph in hand — and to the top-level graph, which together cover
    /// every node the import produces.
    /// </remarks>
    internal static class FastRejectSequenceMap
    {
        public static void Process(InternalComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            foreach (var node in graph.Nodes)
            {
                if (node.OpCode != OpCodes.SEQUENCE_MAP_OPEN) continue;
                throw new NotSupportedException(
                    $"ONNX import: the 'SequenceMap' operator (node '{node.FriendlyName}') is not supported. " +
                    "Lowering SequenceMap to a Loop requires whole-graph type inference: its additional " +
                    "inputs are mapped per-element when sequence-typed but broadcast when tensor-typed " +
                    "(indistinguishable without inferring the element types), and the per-output " +
                    "accumulator sequences need a typed SequenceEmpty seed. The ONNX Runtime execution " +
                    "backend has no SequenceMap kernel either. Workaround: rewrite the model as an " +
                    "explicit Loop over SequenceLength using SequenceAt/SequenceInsert (in Shorokoo, " +
                    "build it with LoopAPI) — that form is fully supported. " +
                    "See Documentation/limitations.md.");
            }
        }
    }
}

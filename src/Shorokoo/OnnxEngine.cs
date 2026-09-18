using Shorokoo;
using Shorokoo.Runtime;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Core.Utils;

namespace Shorokoo
{
    /// <summary>
    /// Convenience evaluator: builds a computation graph from the given output
    /// variables and executes it in a fresh <see cref="ComputeContext"/>.
    /// </summary>
    public static class OnnxEngine
    {
        /// <summary>
        /// Evaluates the given output variables and returns their values, in order. The outputs
        /// must come from a concretized graph: a <c>[Module]</c>'s output is refused with the
        /// lowering hint rather than failing deep inside ONNX Runtime.
        /// </summary>
        public static TensorData[] Eval(Variable[] outputs)
        {

            var graph = new Shorokoo.Graph.InternalComputationGraph([], [.. outputs]);
            graph.RequireRunnableOps("OnnxEngine.Eval");

            // Detaching, and disposed: what this hands back outlives the call, and a context
            // nothing disposes would keep every result of every Eval on its books for good. The
            // results are also literals a graph may be built from, and an operator attribute
            // refuses a tensor that still belongs to a context.
            using var ctx = new ComputeContext(detachesOutputs: true);
            return ctx.Execute(graph).Select(x => x.ToTensorData()).ToArray();
        }

        /// <summary>Evaluates two or more output variables and returns their values, in order.</summary>
        public static TensorData[] Eval(Variable output1, Variable output2, params Variable[] outputs)
        {
            var allOutputs = new[] { output1, output2 }.Concat(outputs).ToArray();
            return Eval(allOutputs);
        }

        /// <summary>Evaluates a single output variable and returns its value.</summary>
        public static TensorData Eval(Variable output)
        {
            var allOutputs = new[] { output };
            return Eval(allOutputs)[0];
        }

    }
}
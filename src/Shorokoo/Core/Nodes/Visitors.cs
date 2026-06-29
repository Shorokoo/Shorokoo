using Shorokoo;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Factory.IR;
using System.Diagnostics;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using System.Collections.Immutable;
using Shorokoo.Core.Nodes.Processors.Helpers;

namespace Shorokoo.Core.Nodes
{
    internal static class Visitors
    {
        public static IEnumerable<Node> TopologicalOrder(IEnumerable<Variable> inputs, IEnumerable<Variable> outputs)
        {
            // A dedicated first-class traversal would be faster; this generic visitor is fast enough in practice.
            return Visitors.ReversePreOrder(inputs, outputs).Select(x => x.OwningNode).Distinct().Reverse();
        }

        /// <summary>
        /// Visits the graph starting at outputs and stoping at inputs.
        /// Essentially visits the graph in a reverse pre-order.
        /// Basically, flip the graph upside down. Add a virtual root node
        /// then visit the graph in pre-order.
        /// 
        /// This is the most natural ordering as it is efficient and simple
        /// to implement with a unique deterministic ordering.
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="outputs"></param>
        /// <returns></returns>
        public static IEnumerable<Variable> ReversePreOrder(IEnumerable<Variable> inputs, IEnumerable<Variable> outputs)
        {
            var inputSet = inputs.ToHashSet();
            var scheduledTensors = new HashSet<Variable>();
            Stack<Variable> tensorStack = new Stack<Variable>(outputs.Reverse());
            while (tensorStack.Count > 0)
            {
                Variable tensor = tensorStack.Pop();
                yield return tensor;

                if (inputSet.Contains(tensor))
                    continue;

                foreach (var inputTensor in tensor.OwningNode.Inputs.NotNulls().Reverse())
                {
                    if (!scheduledTensors.Contains(inputTensor))
                    {
                        scheduledTensors.Add(inputTensor);
                        tensorStack.Push(inputTensor);
                    }
                }

                if (tensor.OwningNode.IsCloseNode)
                {
                    var closeNode = tensor.OwningNode;
                    var connectingTensor = closeNode.ConnectingTensor.AssertNotNull();
                    if (!scheduledTensors.Contains(connectingTensor))
                    {
                        scheduledTensors.Add(connectingTensor);
                        tensorStack.Push(connectingTensor);
                    }
                }
            }
        }

        private static IEnumerable<NodeProto> getAllSubgraphNodes(NodeProto node)
        {
            foreach (var attribute in node.Attributes)
            {
                foreach (var graph in attribute.Graphs.Append(attribute.G))
                {
                    if (graph is null) 
                        continue;

                    foreach (var subgraphNode in graph.Nodes)
                    {
                        foreach (var subsubgraphNode in getAllSubgraphNodes(subgraphNode))
                            yield return subsubgraphNode;

                        yield return subgraphNode;
                    }
                }    
            }
        }

        /// <summary>
        /// Gets all the inputs for a node, including implied inputs from graphs found in attributes.
        /// </summary>
        /// <param name="node"></param>
        /// <returns></returns>
        private static List<string> getAllInputs(NodeProto node)
        {
            var allNodes = getAllSubgraphNodes(node).Append(node).ToList();
            var allOutputs = allNodes.SelectMany(x => x.Outputs).ToHashSet();
            var allInputs = allNodes.SelectMany(x => x.Inputs).ToHashSet();

            allInputs.ExceptWith(allOutputs);

            return allInputs.ToList();
        }

        /// <summary>
        /// Verifies that <paramref name="nodes"/> are listed in topological order:
        /// for every node N, every input tensor name (including those introduced by
        /// subgraph attributes) is either produced by an earlier node in the
        /// sequence, or supplied externally (i.e. produced by no node in
        /// <paramref name="nodes"/>).
        /// </summary>
        public static bool IsTopologicallyOrdered(IEnumerable<NodeProto> nodes)
        {
            var nodeList = nodes as IList<NodeProto> ?? nodes.ToList();

            var allOutputs = new HashSet<string>();
            foreach (var n in nodeList)
                foreach (var o in n.Outputs)
                    allOutputs.Add(o);

            var produced = new HashSet<string>();
            foreach (var n in nodeList)
            {
                foreach (var input in getAllInputs(n))
                {
                    if (allOutputs.Contains(input) && !produced.Contains(input))
                        return false;
                }
                foreach (var o in n.Outputs)
                    produced.Add(o);
            }
            return true;
        }
    }
}

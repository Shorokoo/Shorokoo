using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Runtime;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.Processors.AutoGrad;
using Shorokoo.Core.Nodes.Processors.Fast;
using Shorokoo.Core.Inference;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using System.Collections.Generic;

namespace Shorokoo.Core.Nodes.Processors.Training
{
    /// <summary>
    /// Resolves every <c>LOOP_OPEN</c> node's iteration-count input to a literal
    /// <c>CONSTANT</c> node so the surrounding autograd pipeline can unroll the
    /// loop. Mutates <c>graph</c> in place; returns silently if no
    /// <c>LOOP_OPEN</c> node has a non-constant iter input.
    ///
    /// <para>
    /// Evaluation strategy: try the <see cref="QuickExecutionEngine"/> first on a
    /// cloned, pruned resolver graph, and fall back to
    /// <see cref="ComputeContext.Execute(InternalComputationGraph)"/> (ONNX Runtime)
    /// for any iter-count key QEE couldn't produce. The QEE-first path keeps the
    /// common case (Add / Sub / Mul of constants, Shape-of-input-arithmetic, etc.)
    /// entirely in pure C# and avoids the per-call cost of building an ORT
    /// <c>InferenceSession</c>; the ORT fallback covers ops QEE doesn't model.
    /// </para>
    /// </summary>
    internal static class FastFoldLoopIterationCountsToConstantsProcessor
    {
        public static void Process(InternalComputationGraph graph, ComputeContext compute)
        {
            var nodesByKey = new Dictionary<FastNodeKey, FastNode>(graph.Nodes.Count);
            foreach (var n in graph.Nodes) nodesByKey[n.Key] = n;

            var iterCountKeys = new List<FastTensorKey>();
            var seen = new HashSet<FastTensorKey>();
            foreach (var node in graph.Nodes)
            {
                if (node.OpCode != OpCodes.LOOP_OPEN) continue;
                if (!node.FullInputs.TryGetValue("", out var inputs) || inputs.Count == 0) continue;
                var iterCountKey = inputs[0];
                if (iterCountKey is null || iterCountKey.Value.IsEmpty) continue;
                if (nodesByKey.TryGetValue(iterCountKey.Value.FastNodeKey, out var producer)
                    && producer.OpCode == OpCodes.CONSTANT)
                    continue;
                if (seen.Add(iterCountKey.Value))
                    iterCountKeys.Add(iterCountKey.Value);
            }

            if (iterCountKeys.Count == 0)
                return;

            // Clone the input graph as a resolver: keep all nodes, retarget outputs to the
            // iter-count tensors, drop graph inputs (iter-count expressions must be
            // self-contained constant computations), then sweep nodes that no longer feed
            // any output.
            var resolverGraph = graph.Clone();
            resolverGraph.Inputs = new List<FastTensorKey>();
            resolverGraph.InputUniqueNames = new List<string?>();
            resolverGraph.Outputs = new List<FastTensorKey>(iterCountKeys);
            resolverGraph.OutputUniqueNames = new List<string?>(new string?[iterCountKeys.Count]);
            resolverGraph.OutputRankOverrides = null;
            FastProcessorHelper.RemoveUnreachableNodes(resolverGraph);

            var resolvedData = ResolveIterCountValues(resolverGraph, iterCountKeys, compute);

            var constantAttrDefs = Definitions.NodeDefinitions[OpCodes.CONSTANT].AttributeDefs;
            var remap = new Dictionary<FastTensorKey, FastTensorKey>(iterCountKeys.Count);
            var newConstantNodes = new List<FastNode>(iterCountKeys.Count);
            for (int i = 0; i < iterCountKeys.Count; i++)
            {
                var iterKey = iterCountKeys[i];
                var tensorData = resolvedData[i];

                var nodeKey = FastNodeKey.New();
                var attrs = OnnxCSharpAttributes.FromCSharpVals(
                    new Dictionary<string, object?> { [OnnxOpAttributeNames.AttrValue] = tensorData },
                    constantAttrDefs);
                var constantOutputKey = new FastTensorKey(nodeKey, 0);
                newConstantNodes.Add(new FastNode
                {
                    Key = nodeKey,
                    OpCode = OpCodes.CONSTANT,
                    Attributes = attrs,
                    FullOutputs = { [""] = new List<FastTensorKey?> { constantOutputKey } },
                });
                remap[iterKey] = constantOutputKey;
            }

            // Rewire every input slot of every remaining node, replacing keys that match
            // a former iter-count tensor with the new CONSTANT output.
            foreach (var node in graph.Nodes)
            {
                foreach (var (groupName, slots) in node.FullInputs)
                {
                    for (int i = 0; i < slots.Count; i++)
                    {
                        var k = slots[i];
                        if (k is FastTensorKey tk && remap.TryGetValue(tk, out var newKey))
                            slots[i] = newKey;
                    }
                }
            }

            // Rewire graph outputs in case any output is itself a former iter-count tensor.
            for (int i = 0; i < graph.Outputs.Count; i++)
                if (remap.TryGetValue(graph.Outputs[i], out var newKey))
                    graph.Outputs[i] = newKey;

            // CONSTANT has no inputs, so the new nodes are topologically valid at the
            // front of the node list.
            graph.Nodes.InsertRange(0, newConstantNodes);

            FastProcessorHelper.RemoveUnreachableNodes(graph);
        }

        /// <summary>
        /// Resolves the iter-count tensors by running QEE on the pruned resolver graph
        /// first; if any required key is missing from the QEE store (op not modelled,
        /// per-op throw caught by QEE, or shape-only result), runs the resolver graph
        /// through ORT via <paramref name="compute"/> as a fallback for the unresolved
        /// keys.
        /// </summary>
        private static TensorData[] ResolveIterCountValues(
            InternalComputationGraph resolverGraph,
            List<FastTensorKey> iterCountKeys,
            ComputeContext compute)
        {
            var resolved = new TensorData?[iterCountKeys.Count];

            try
            {
                var qeeStore = new QuickExecutionEngine().Run(resolverGraph);
                for (int i = 0; i < iterCountKeys.Count; i++)
                {
                    if (qeeStore.TryGetValue(iterCountKeys[i], out var rt)
                        && rt is RuntimeTensor plain
                        && TensorDataConverter.ToTensorData(plain) is { } td)
                    {
                        resolved[i] = td;
                    }
                }
            }
            catch
            {
                // QEE itself catches per-op exceptions, so reaching here means a
                // structural failure (e.g. bad initial-input mapping). Leave every
                // slot null and let the ORT fallback below cover all of them.
            }

            bool needsFallback = false;
            for (int i = 0; i < resolved.Length; i++)
                if (resolved[i] is null) { needsFallback = true; break; }

            if (needsFallback)
            {
                var ortResolved = compute.Execute(resolverGraph);
                for (int i = 0; i < iterCountKeys.Count; i++)
                    if (resolved[i] is null)
                        resolved[i] = ortResolved[i].ToTensorData();
            }

            var result = new TensorData[iterCountKeys.Count];
            for (int i = 0; i < iterCountKeys.Count; i++)
                result[i] = resolved[i]!;
            return result;
        }
    }
}

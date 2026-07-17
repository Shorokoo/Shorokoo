using System;
using Shorokoo.Core.Graph;
using System.Collections.Generic;
using System.Linq;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Graph;

namespace Shorokoo.Graph
{
    /// <summary>
    /// State-update helpers on <see cref="FastComputationGraph"/>: the
    /// <c>GetStateParamDataNodes</c> / <c>GetStateUpdateOutputCount</c> /
    /// <c>WithUpdatedStates</c> surface used to execute a Fast graph with state.
    /// </summary>
    public partial class FastComputationGraph
    {
        /// <summary>
        /// All MODEL_PARAM_DATA nodes whose IsTrainable attribute is false (i.e. state params).
        /// The RngSeed parameter (the model's RNG identity, at reserved ModelId [0]) is
        /// excluded: it is non-trainable data but not per-step state — it has no state update,
        /// and including it would break the positional pairing with state-update outputs.
        /// </summary>
        public List<FastNode> GetStateParamDataNodes()
        {
            var list = new List<FastNode>();
            foreach (var node in this.Nodes)
            {
                if (node.OpCode != InternalOpCodes.MODEL_PARAM_DATA) continue;
                if (node.IdentifierTemplate ==
                        Core.Nodes.Processors.Fast.FastWireRngKeyDerivation.RngSeedIdentifierTemplate)
                    continue;
                var isTrainable = node.Attributes.GetBoolVal(OnnxOpAttributeNames.ShrkAttrIsTrainable) ?? false;
                if (!isTrainable) list.Add(node);
            }
            return list;
        }

        /// <summary>
        /// Number of STATE_UPDATE_LINK nodes in the graph.
        /// </summary>
        public int GetStateUpdateOutputCount()
        {
            int count = 0;
            foreach (var node in this.Nodes)
                if (node.OpCode == InternalOpCodes.STATE_UPDATE_LINK)
                    count++;
            return count;
        }

        /// <summary>
        /// Returns a new <see cref="FastComputationGraph"/> with each state-param node's
        /// <c>ShrkAttrTensorData</c> replaced by the corresponding entry in
        /// <paramref name="tensorDatas"/>.
        /// </summary>
        public FastComputationGraph WithUpdatedStates(TensorData[] tensorDatas)
        {
            var stateParamNodes = GetStateParamDataNodes();

            if (stateParamNodes.Count == 0) return this;

            if (stateParamNodes.Count != tensorDatas.Length)
                throw new InvalidOperationException(
                    $"WithUpdatedStates: Expected {stateParamNodes.Count} state update values, but got {tensorDatas.Length}");

            var indexInGraph = new Dictionary<FastNodeKey, int>();
            for (int i = 0; i < this.Nodes.Count; i++)
                indexInGraph[this.Nodes[i].Key] = i;

            var copy = this.Clone();

            for (int s = 0; s < stateParamNodes.Count; s++)
            {
                var originalNode = stateParamNodes[s];
                var idx = indexInGraph[originalNode.Key];
                var clonedNode = copy.Nodes[idx];
                clonedNode.Attributes = clonedNode.Attributes.SetAttributes(
                    (OnnxOpAttributeNames.ShrkAttrTensorData, (object?)tensorDatas[s]));
            }

            return copy;
        }
    }
}

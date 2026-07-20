using Shorokoo.Core;
using Shorokoo.Core.Graph;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using System.Collections.Generic;
using System.Linq;
using Shorokoo.Core.Nodes.Processors.AutoGrad;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Shared mutable state for <see cref="FastUnpackModelStruct"/>'s native pass. Collects
    /// the two unpack tables, the output-key remap, the node-by-key index, the new-node
    /// accumulator and the dead-node set so that each per-op handler works against the
    /// same state without threading nine parameters through every call.
    /// </summary>
    internal sealed class FastModelStructContext
    {
        /// <summary>Maps a Model tensor key to its field tensor keys
        /// [iterationIndices, hp0, hp1, ..., hpN, modelId].</summary>
        public Dictionary<FastTensorKey, List<FastTensorKey?>> UnpackedStructs { get; } = new();

        /// <summary>Maps a Sequence-of-Model tensor key to one parallel sequence tensor key
        /// per struct field, in the same order as <see cref="UnpackedStructs"/> entries.</summary>
        public Dictionary<FastTensorKey, List<FastTensorKey>> UnpackedStructSequences { get; } = new();

        /// <summary>Simple output remap: keys that are exactly equal to a single replacement
        /// tensor (e.g. MODEL_HYPERPARAM output → struct field, LOOP_OPEN slot shift for a
        /// non-Model loop variable).</summary>
        public Dictionary<FastTensorKey, FastTensorKey> Remap { get; } = new();

        /// <summary>Nodes whose role was subsumed by generated replacements and that must be
        /// dropped from the graph at the end of the pass.</summary>
        public HashSet<FastNodeKey> NodesToRemove { get; } = new();

        /// <summary>Nodes produced by handlers, appended to the graph at end-of-pass.</summary>
        public List<FastNode> NewNodes { get; } = new();

        /// <summary>FastNodeKey → FastNode index, refreshed as new nodes are produced so
        /// that <see cref="ResolveModelKey"/> can chase identity chains into newly inserted
        /// nodes too.</summary>
        public Dictionary<FastNodeKey, FastNode> NodeByKey { get; }

        /// <summary>LOOP_OPEN.Key → original (pre-expansion) variadic-input count. The
        /// matching LOOP_CLOSE reads this to recover the loopVar / scanVar split in its own
        /// pre-mutated body inputs (since we've already rewritten the open node's inputs by
        /// the time the close visits).</summary>
        private readonly Dictionary<FastNodeKey, int> _loopOpenOriginalVariadicCounts = new();

        public FastModelStructContext(InternalComputationGraph graph)
        {
            NodeByKey = FastProcessorHelper.BuildNodeByKey(graph);
        }

        public void RecordNewNode(FastNode node)
        {
            NewNodes.Add(node);
            NodeByKey[node.Key] = node;
        }

        public void SetLoopOpenOriginalVariadicCount(FastNodeKey openNodeKey, int count)
            => _loopOpenOriginalVariadicCounts[openNodeKey] = count;

        public int GetLoopOpenOriginalVariadicCount(FastNodeKey openNodeKey)
            => _loopOpenOriginalVariadicCounts[openNodeKey];

        /// <summary>
        /// Follows <see cref="OpCodes.IDENTITY"/> chains until reaching a key that is
        /// already registered in <see cref="UnpackedStructs"/> or
        /// <see cref="UnpackedStructSequences"/>, or until the chain terminates. Used when a
        /// handler encounters a Model-typed input that may have been aliased through one or
        /// more IDENTITY passes.
        /// </summary>
        public FastTensorKey ResolveModelKey(FastTensorKey key)
        {
            var visited = new HashSet<FastTensorKey>();
            while (visited.Add(key))
            {
                if (UnpackedStructs.ContainsKey(key) || UnpackedStructSequences.ContainsKey(key))
                    return key;

                if (!NodeByKey.TryGetValue(key.FastNodeKey, out var node))
                    break;

                if (node.OpCode == OpCodes.IDENTITY)
                {
                    var input = node.Inputs[0];
                    if (input is null) break;
                    key = input.Value;
                }
                else
                {
                    break;
                }
            }
            return key;
        }
    }

}

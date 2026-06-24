using System.Collections.Generic;
using Shorokoo.Core.Graph;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using Shorokoo.Core.Nodes.Processors.AutoGrad;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Shared helper used by <see cref="FastAddIdentityForOuterScopeValues"/> and
    /// <see cref="FastPrepForOnnx"/>: wraps every input slot of every IF_CLOSE
    /// and LOOP_CLOSE node in a freshly-inserted Identity. The Identity nodes
    /// are placed just before their close in the topological order, and the
    /// close's input slots are rewritten to reference the Identity outputs.
    /// </summary>
    internal static class FastIdentityWrapping
    {
        public static void WrapCloseInputs(FastComputationGraph graph)
        {
            // We're inserting nodes into graph.Nodes mid-iteration, so collect
            // the (closeIndex → list-of-new-identities) pairs first and apply
            // the inserts in a second pass (in reverse order so earlier inserts
            // don't shift later indices).
            var inserts = new List<(int closeIndex, List<FastNode> identities)>();
            var identityAttrDefs = Definitions.NodeDefinitions[OpCodes.IDENTITY].AttributeDefs;

            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                var node = graph.Nodes[i];
                if (node.OpCode != OpCodes.IF_CLOSE && node.OpCode != OpCodes.LOOP_CLOSE)
                    continue;

                var newIdentities = new List<FastNode>();
                foreach (var groupKey in new List<string>(node.FullInputs.Keys))
                {
                    var slot = node.FullInputs[groupKey];
                    for (int s = 0; s < slot.Count; s++)
                    {
                        var input = slot[s];
                        if (input is null || input.Value.IsEmpty) continue;

                        var idKey = FastNodeKey.New();
                        var idOutputKey = new FastTensorKey(idKey, 0);
                        var identity = new FastNode
                        {
                            Key = idKey,
                            OpCode = OpCodes.IDENTITY,
                            Attributes = OnnxCSharpAttributes.FromCSharpVals(
                                new Dictionary<string, object?>(),
                                identityAttrDefs),
                            FriendlyName = null,
                            FullInputs = new Dictionary<string, List<FastTensorKey?>>
                            {
                                [""] = new List<FastTensorKey?> { input },
                            },
                            FullOutputs = new Dictionary<string, List<FastTensorKey?>>
                            {
                                [""] = new List<FastTensorKey?> { idOutputKey },
                            },
                        };
                        newIdentities.Add(identity);
                        slot[s] = idOutputKey;
                    }
                }

                if (newIdentities.Count > 0)
                    inserts.Add((i, newIdentities));
            }

            // Apply inserts in reverse so prior insertions don't shift later
            // indices in graph.Nodes.
            for (int idx = inserts.Count - 1; idx >= 0; idx--)
            {
                var (closeIndex, identities) = inserts[idx];
                graph.Nodes.InsertRange(closeIndex, identities);
            }
        }
    }
}

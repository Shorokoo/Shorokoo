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
    /// The two places a freshly-inserted Identity buys a name that ONNX needs and the
    /// graph does not otherwise have. <see cref="WrapCloseInputs"/> wraps every input
    /// slot of every IF_CLOSE and LOOP_CLOSE node, for
    /// <see cref="FastAddIdentityForOuterScopeValues"/> and <see cref="FastPrepForOnnx"/>;
    /// <see cref="WrapAliasedOutputs"/> wraps a graph output that names something no node
    /// of its own produces, for function emission in
    /// <see cref="Shorokoo.Core.Factory.FastOnnxModelBuilder"/>.
    /// </summary>
    internal static class FastIdentityWrapping
    {
        public static void WrapCloseInputs(InternalComputationGraph graph)
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

        /// <summary>
        /// Ensures every graph output is produced by a node of its own, rewiring through a
        /// freshly appended Identity any output that instead names one of the graph's own
        /// inputs. An <c>Inline</c> body that hands an argument straight back
        /// (<c>Inline(Scalar&lt;float32&gt; value) =&gt; value</c>) produces exactly that shape,
        /// and ONNX has no way to express it: a FunctionProto whose output name is one of its
        /// input names has no node producing the output, and ORT rejects the model when it
        /// builds the function's schema. An output repeating an earlier one is wrapped by the
        /// same rule — reachable only through the internal dialect, which keeps multi-output
        /// module functions, but it costs nothing to be right about.
        /// </summary>
        public static void WrapAliasedOutputs(InternalComputationGraph graph)
        {
            var claimed = new HashSet<FastTensorKey>(graph.Inputs);
            var identityAttrDefs = Definitions.NodeDefinitions[OpCodes.IDENTITY].AttributeDefs;

            for (int i = 0; i < graph.Outputs.Count; i++)
            {
                var output = graph.Outputs[i];
                if (output.IsEmpty || claimed.Add(output)) continue;

                var idKey = FastNodeKey.New();
                var idOutputKey = new FastTensorKey(idKey, 0);
                // Appended at the very end, where every scope is closed, so the linear order
                // stays valid. The value being wrapped is a formal parameter (or an output
                // already emitted), so it is in scope there.
                graph.Nodes.Add(new FastNode
                {
                    Key = idKey,
                    OpCode = OpCodes.IDENTITY,
                    Attributes = OnnxCSharpAttributes.FromCSharpVals(
                        new Dictionary<string, object?>(),
                        identityAttrDefs),
                    FriendlyName = null,
                    FullInputs = new Dictionary<string, List<FastTensorKey?>>
                    {
                        [""] = new List<FastTensorKey?> { output },
                    },
                    FullOutputs = new Dictionary<string, List<FastTensorKey?>>
                    {
                        [""] = new List<FastTensorKey?> { idOutputKey },
                    },
                });
                graph.Outputs[i] = idOutputKey;
                claimed.Add(idOutputKey);
            }
        }
    }
}

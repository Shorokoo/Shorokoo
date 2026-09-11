using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using System;
using System.Collections.Generic;
using System.Linq;
using Shorokoo.Core.Nodes.Processors.AutoGrad;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Fast version of the state-update lowering pass. Operates directly on a
    /// <see cref="InternalComputationGraph"/>:
    /// <list type="number">
    ///   <item>Each <c>STATE_UPDATE_LINK(originalState, updatedState)</c> is rewritten to
    ///         <c>IDENTITY(updatedState)</c>.</item>
    ///   <item>Each <c>WITH_STATE_DEPS(mainOutput, ...stateDeps)</c> is rewritten to
    ///         <c>IDENTITY(mainOutput)</c>. The state-dep inputs are dropped.</item>
    ///   <item>One tensor per state parameter — the value its last update produced — is appended
    ///         to <see cref="InternalComputationGraph.Outputs"/>, in the order
    ///         <c>GetStateParamDataNodes</c> lists the parameters, since the executor pairs the
    ///         two positionally.</item>
    /// </list>
    /// Tensor keys are preserved by mutating nodes in place, so consumers of the lowered
    /// nodes' outputs do not need any rewriting.
    /// </summary>
    internal static class FastLowerStateUpdateNodes
    {
        public static void Process(InternalComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            var hasStateNodes = graph.Nodes.Any(n =>
                n.OpCode == InternalOpCodes.WITH_STATE_DEPS ||
                n.OpCode == InternalOpCodes.STATE_UPDATE_LINK);

            if (!hasStateNodes) return;

            var identityAttrDefs = Definitions.NodeDefinitions[OpCodes.IDENTITY].AttributeDefs;
            var emptyIdentityAttrs = OnnxCSharpAttributes.FromCSharpVals(
                new Dictionary<string, object?>(), identityAttrDefs);

            // Read the chains off the graph before rewriting: once a link is an IDENTITY there is
            // nothing left to tell one call's link from the parameter it was chained to.
            var newStateOutputs = FinalLinkPerStateParam(graph)
                .Select(link => link.Outputs[0]
                    ?? throw new InvalidOperationException("STATE_UPDATE_LINK has no output."))
                .ToList();

            foreach (var node in graph.Nodes)
            {
                if (node.OpCode == InternalOpCodes.STATE_UPDATE_LINK)
                {
                    // STATE_UPDATE_LINK(originalState, updatedState) → IDENTITY(updatedState)
                    var inputs = node.Inputs;
                    var updatedState = inputs[1]
                        ?? throw new InvalidOperationException(
                            "STATE_UPDATE_LINK has null updatedState input.");

                    node.OpCode = OpCodes.IDENTITY;
                    node.Attributes = emptyIdentityAttrs;
                    node.FullInputs = new Dictionary<string, List<FastTensorKey?>>
                    {
                        [""] = new List<FastTensorKey?> { updatedState }
                    };
                }
                else if (node.OpCode == InternalOpCodes.WITH_STATE_DEPS)
                {
                    // WITH_STATE_DEPS(mainOutput, ...stateDeps) → IDENTITY(mainOutput)
                    var inputs = node.Inputs;
                    var mainOutput = inputs[0]
                        ?? throw new InvalidOperationException(
                            "WITH_STATE_DEPS has null mainOutput input.");

                    node.OpCode = OpCodes.IDENTITY;
                    node.Attributes = emptyIdentityAttrs;
                    node.FullInputs = new Dictionary<string, List<FastTensorKey?>>
                    {
                        [""] = new List<FastTensorKey?> { mainOutput }
                    };
                }
            }

            foreach (var key in newStateOutputs)
            {
                graph.Outputs.Add(key);
                graph.OutputUniqueNames.Add(null);
            }
        }

        /// <summary>
        /// The link holding each state parameter's end-of-step value, ordered as
        /// <c>GetStateParamDataNodes</c> orders the parameters.
        ///
        /// <para>A parameter called at several sites has one link per call, chained by
        /// <see cref="FastChainStateUpdatesAcrossCallSites"/> so that each call's link reads the
        /// previous one. Only the last link in a chain holds the value the step ends with, and the
        /// executor takes exactly one value per parameter, so a link that another link reads is not
        /// an output.</para>
        /// </summary>
        internal static List<FastNode> FinalLinkPerStateParam(InternalComputationGraph graph)
        {
            var nodeByKey = FastProcessorHelper.BuildNodeByKey(graph);

            var linkByOutput = new Dictionary<FastTensorKey, FastNode>();
            foreach (var node in graph.Nodes)
                if (node.OpCode == InternalOpCodes.STATE_UPDATE_LINK
                    && node.Outputs.Count > 0 && node.Outputs[0] is FastTensorKey outKey)
                    linkByOutput[outKey] = node;

            // Node order puts a chain's earlier links first, so the root each link inherits is
            // already known by the time the next one is reached.
            var rootOfLink = new Dictionary<FastNodeKey, FastTensorKey>();
            var lastLinkOfRoot = new Dictionary<FastTensorKey, FastNode>();
            foreach (var node in graph.Nodes)
            {
                if (node.OpCode != InternalOpCodes.STATE_UPDATE_LINK) continue;
                if (node.Inputs.Count == 0 || node.Inputs[0] is not FastTensorKey original) continue;
                if (ResolveThroughIdentities(original, nodeByKey) is not FastTensorKey resolved) continue;

                var root = linkByOutput.TryGetValue(resolved, out var previous)
                           && rootOfLink.TryGetValue(previous.Key, out var previousRoot)
                    ? previousRoot
                    : resolved;
                rootOfLink[node.Key] = root;
                lastLinkOfRoot[root] = node;
            }

            var finalLinks = new List<FastNode>();
            foreach (var param in graph.GetStateParamDataNodes())
                if (lastLinkOfRoot.TryGetValue(new FastTensorKey(param.Key, 0), out var link))
                    finalLinks.Add(link);
            return finalLinks;
        }

        /// <summary>The value behind an IDENTITY chain, which inlining leaves around a spliced
        /// parameter.</summary>
        private static FastTensorKey? ResolveThroughIdentities(
            FastTensorKey key, Dictionary<FastNodeKey, FastNode> nodeByKey)
        {
            var current = key;
            var visited = new HashSet<FastTensorKey>();
            while (visited.Add(current))
            {
                if (!nodeByKey.TryGetValue(current.FastNodeKey, out var node)) return current;
                if (node.OpCode != OpCodes.IDENTITY) return current;
                if (node.Inputs.Count == 0 || node.Inputs[0] is not FastTensorKey inner) return current;
                current = inner;
            }
            return null;
        }
    }
}

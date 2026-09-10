using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Makes a module-owned state update apply once per call, rather than once per module.
    ///
    /// <para>Concretization collapses two call sites of one model handle onto one state
    /// parameter — correctly, since they share it — but each call keeps its own
    /// <c>STATE_UPDATE_LINK</c>, and each reads the parameter as the graph gave it. So the second
    /// call computes its update from the initial value rather than from the first call's, and the
    /// two updates do not compose (Shorokoo/Shorokoo#306). This pass points a later call's reads
    /// at the earlier call's link.</para>
    ///
    /// <para><b>Why the link's output rather than the updated value.</b> A link lowers one way
    /// where state persists across executions (<c>Identity(updatedState)</c>, see
    /// <see cref="FastLowerStateUpdateNodes"/>) and the other for one-shot inference
    /// (<c>Identity(originalState)</c>, see <see cref="FastLowerStateUpdateLinksForInference"/>).
    /// Reading through the link keeps both honest with one rewrite: where state persists the
    /// updates compose, and in one-shot inference — where the update computation is dropped — every
    /// call still sees the value it was given.</para>
    ///
    /// <para><b>What bounds a call.</b> <c>WITH_STATE_DEPS</c> is emitted once per call site that
    /// owns state, listing that call's links, so the ones naming a given parameter mark that
    /// parameter's call sites in order. Node order supplies the sequence. Position alone does not:
    /// a body reads its parameter after its own link as often as before it — the link is an
    /// annotation, not an assignment — so a read is attributed to a call by which marker it falls
    /// under, never by whether it follows a link.</para>
    ///
    /// <para><b>Calls in the arms of an IfElse.</b> Node order is not a running order there: the
    /// arms are alternatives, so composing across them would credit an update that never happened.
    /// Each arm is chained on its own from the value the parameter held entering the branch, both
    /// arms' results are threaded out through the <c>IF_CLOSE</c> as one more output pair, and a
    /// module-scope link takes whichever arm ran (Shorokoo/Shorokoo#308). An arm that does not call
    /// the model hands the incoming value straight back, so a call in one arm alone updates the
    /// state only when that arm runs.</para>
    /// </summary>
    internal static class FastChainStateUpdatesAcrossCallSites
    {
        /// <summary>One call site of a state parameter: its link, the marker that closes it, and
        /// the arm it belongs to (null at module scope).</summary>
        private readonly record struct CallSite(FastNode Link, FastNode Marker, ArmKey? Arm);

        /// <summary>An IfElse arm: the branch node and which side of it.</summary>
        private readonly record struct ArmKey(FastNodeKey IfClose, string Branch);

        public static void Process(InternalComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            var nodeByKey = FastProcessorHelper.BuildNodeByKey(graph);
            var positionOf = new Dictionary<FastNodeKey, int>(graph.Nodes.Count);
            for (int i = 0; i < graph.Nodes.Count; i++) positionOf[graph.Nodes[i].Key] = i;

            // Links per state parameter, in node order. A link reaches its parameter through the
            // Identity chain inlining leaves behind, so the walk has to follow it — a direct key
            // match finds nothing and would leave every parameter silently un-chained.
            var linksByParam = new Dictionary<FastNodeKey, List<FastNode>>();
            foreach (var node in graph.Nodes)
            {
                if (node.OpCode != InternalOpCodes.STATE_UPDATE_LINK) continue;
                if (node.Inputs.Count == 0 || node.Inputs[0] is not FastTensorKey originalKey) continue;
                if (ResolveThroughIdentities(originalKey, nodeByKey) is not FastTensorKey paramKey) continue;
                if (!linksByParam.TryGetValue(paramKey.FastNodeKey, out var list))
                    linksByParam[paramKey.FastNodeKey] = list = [];
                list.Add(node);
            }

            if (linksByParam.Count == 0) return;

            var enclosingScope = ComputeEnclosingScope(graph);
            var armOfNode = ClassifyBranches(graph, positionOf);

            // Range rewrites and node insertions are collected first and applied after, so every
            // position taken here stays the one the node still has.
            var rewrites = new List<(int fromPos, int toPos, FastTensorKey param, FastTensorKey replacement)>();
            var insertions = new List<(int atPos, FastNode node)>();
            var branchLinks = new List<FastTensorKey>();

            foreach (var (paramNodeKey, links) in linksByParam)
            {
                var sites = CallSitesOf(graph, links, positionOf, enclosingScope, armOfNode, nodeByKey);
                if (sites is null) continue;                        // shape this pass does not order
                if (sites.Count < 2 && sites.All(s => s.Arm is null)) continue;   // nothing to compose

                Thread(graph, new FastTensorKey(paramNodeKey, 0), sites, positionOf, nodeByKey,
                       rewrites, insertions, branchLinks);
            }

            foreach (var (from, to, param, replacement) in rewrites)
                for (int i = from; i <= to && i < graph.Nodes.Count; i++)
                    foreach (var (_, inputs) in graph.Nodes[i].FullInputs)
                        for (int j = 0; j < inputs.Count; j++)
                        {
                            // Through the Identity chain, as the link walk above: a read wrapped in
                            // one that was spliced ahead of this call is still a read of the
                            // parameter, and missing it leaves this call on the stale value.
                            if (inputs[j] is not FastTensorKey k) continue;
                            if (ResolveThroughIdentities(k, nodeByKey) is FastTensorKey read
                                && read.Equals(param))
                                inputs[j] = replacement;
                        }

            foreach (var (at, node) in insertions.OrderByDescending(x => x.atPos))
                graph.Nodes.Insert(at, node);

            RootBranchLinks(graph, branchLinks);
        }

        /// <summary>
        /// Keeps the links an IfElse threading created reachable. Nothing reads the value a state
        /// update produces — the state lowerings collect it — so a link nobody depends on is swept
        /// as dead, taking the branch output it selects with it. The same <c>WITH_STATE_DEPS</c>
        /// marker a module body puts on its outputs says so here.
        /// </summary>
        private static void RootBranchLinks(InternalComputationGraph graph, List<FastTensorKey> branchLinks)
        {
            if (branchLinks.Count == 0 || graph.Outputs.Count == 0) return;

            var key = FastNodeKey.New();
            var output = new FastTensorKey(key, 0);
            graph.Nodes.Add(new FastNode
            {
                Key = key,
                OpCode = InternalOpCodes.WITH_STATE_DEPS,
                Attributes = OnnxCSharpAttributes.FromCSharpVals(
                    new Dictionary<string, object?>(),
                    Definitions.NodeDefinitions[InternalOpCodes.WITH_STATE_DEPS].AttributeDefs),
                FullInputs = { [""] = [graph.Outputs[0], .. branchLinks.Select(k => (FastTensorKey?)k)] },
                FullOutputs = { [""] = [output] },
            });
            graph.Outputs[0] = output;
        }

        /// <summary>
        /// This parameter's call sites in node order, or null when the shape is one this pass does
        /// not order: a call inside a loop body (the body repeats, so its calls are not a sequence
        /// — and a rolled loop carrying state is refused where it matters, when the graph is
        /// prepared for training), or an IfElse nested inside another scope.
        /// </summary>
        private static List<CallSite>? CallSitesOf(
            InternalComputationGraph graph,
            List<FastNode> links,
            Dictionary<FastNodeKey, int> positionOf,
            Dictionary<FastNodeKey, FastNodeKey?> enclosingScope,
            Dictionary<FastNodeKey, ArmKey> armOfNode,
            Dictionary<FastNodeKey, FastNode> nodeByKey)
        {
            var linkKeys = links.Select(l => l.Outputs[0]!.Value).ToHashSet();
            var markerByLink = new Dictionary<FastNodeKey, FastNode>();
            foreach (var node in graph.Nodes)
            {
                if (node.OpCode != InternalOpCodes.WITH_STATE_DEPS) continue;
                foreach (var dep in node.Inputs.Skip(1))
                    if (dep is FastTensorKey k && linkKeys.Contains(k) && !markerByLink.ContainsKey(k.FastNodeKey))
                        markerByLink[k.FastNodeKey] = node;
            }

            var sites = new List<CallSite>(links.Count);
            foreach (var link in links)
            {
                if (!markerByLink.TryGetValue(link.Key, out var marker)) return null;

                var scope = enclosingScope[link.Key];
                if (scope is null) { sites.Add(new CallSite(link, marker, null)); continue; }

                // Inside a scope: only an IfElse arm whose branch is itself at module scope.
                if (!armOfNode.TryGetValue(link.Key, out var arm)) return null;
                if (!nodeByKey.TryGetValue(arm.IfClose, out var close)) return null;
                if (enclosingScope[close.Key] is not null) return null;
                sites.Add(new CallSite(link, marker, arm));
            }
            return sites;
        }

        /// <summary>
        /// Walks the parameter's call sites in node order, giving each the value the parameter
        /// holds when that call starts, and threading an IfElse's arms back together after it.
        /// </summary>
        private static void Thread(
            InternalComputationGraph graph,
            FastTensorKey param,
            List<CallSite> sites,
            Dictionary<FastNodeKey, int> positionOf,
            Dictionary<FastNodeKey, FastNode> nodeByKey,
            List<(int fromPos, int toPos, FastTensorKey param, FastTensorKey replacement)> rewrites,
            List<(int atPos, FastNode node)> insertions,
            List<FastTensorKey> branchLinks)
        {
            var current = param;
            var armValue = new Dictionary<ArmKey, FastTensorKey>();
            var armsOfClose = new Dictionary<FastNodeKey, FastTensorKey>();   // the value entering each branch
            int previousMarker = -1;

            for (int i = 0; i < sites.Count; i++)
            {
                var site = sites[i];

                if (site.Arm is ArmKey arm)
                {
                    if (!armsOfClose.ContainsKey(arm.IfClose)) armsOfClose[arm.IfClose] = current;
                    var incoming = armValue.TryGetValue(arm, out var held) ? held : armsOfClose[arm.IfClose];
                    rewrites.Add((previousMarker + 1, positionOf[site.Marker.Key], param, incoming));
                    armValue[arm] = site.Link.Outputs[0]!.Value;
                }
                else
                {
                    rewrites.Add((previousMarker + 1, positionOf[site.Marker.Key], param, current));
                    current = site.Link.Outputs[0]!.Value;
                }
                previousMarker = positionOf[site.Marker.Key];

                // Close out a branch once its last call site is behind us.
                bool lastOfThisClose = site.Arm is ArmKey a
                    && (i + 1 == sites.Count || sites[i + 1].Arm?.IfClose != a.IfClose);
                if (!lastOfThisClose) continue;

                var ifClose = nodeByKey[((ArmKey)site.Arm!).IfClose];
                current = CloseBranch(graph, ifClose, armsOfClose[ifClose.Key], armValue,
                                      positionOf, insertions);
                branchLinks.Add(current);
            }
        }

        /// <summary>
        /// Threads a parameter through one IfElse: each arm's final value becomes one more branch
        /// output, and a link at the branch's own scope takes whichever arm ran. An arm with no
        /// call hands the incoming value back, through an Identity built inside it — a branch
        /// output has to be produced in the branch.
        /// </summary>
        private static FastTensorKey CloseBranch(
            InternalComputationGraph graph,
            FastNode ifClose,
            FastTensorKey incoming,
            Dictionary<ArmKey, FastTensorKey> armValue,
            Dictionary<FastNodeKey, int> positionOf,
            List<(int atPos, FastNode node)> insertions)
        {
            int closePos = positionOf[ifClose.Key];
            var identityAttrs = OnnxCSharpAttributes.FromCSharpVals(
                new Dictionary<string, object?>(), Definitions.NodeDefinitions[OpCodes.IDENTITY].AttributeDefs);

            FastTensorKey ArmFinal(string branch)
            {
                if (armValue.TryGetValue(new ArmKey(ifClose.Key, branch), out var v)) return v;
                var key = FastNodeKey.New();
                var output = new FastTensorKey(key, 0);
                insertions.Add((closePos, new FastNode
                {
                    Key = key,
                    OpCode = OpCodes.IDENTITY,
                    Attributes = identityAttrs,
                    FullInputs = { [""] = [incoming] },
                    FullOutputs = { [""] = [output] },
                }));
                return output;
            }

            var thenFinal = ArmFinal(OnnxOpAttributeNames.AttrThenBranch);
            var elseFinal = ArmFinal(OnnxOpAttributeNames.AttrElseBranch);

            ifClose.FullInputs[OnnxOpAttributeNames.AttrThenBranch].Add(thenFinal);
            ifClose.FullInputs[OnnxOpAttributeNames.AttrElseBranch].Add(elseFinal);
            var outputs = ifClose.FullOutputs.Single().Value;
            var selected = new FastTensorKey(ifClose.Key, outputs.Count);
            outputs.Add(selected);

            var linkKey = FastNodeKey.New();
            var linkOutput = new FastTensorKey(linkKey, 0);
            insertions.Add((closePos + 1, new FastNode
            {
                Key = linkKey,
                OpCode = InternalOpCodes.STATE_UPDATE_LINK,
                Attributes = OnnxCSharpAttributes.FromCSharpVals(
                    new Dictionary<string, object?>(),
                    Definitions.NodeDefinitions[InternalOpCodes.STATE_UPDATE_LINK].AttributeDefs),
                FullInputs = { [""] = [incoming, selected] },
                FullOutputs = { [""] = [linkOutput] },
            }));
            return linkOutput;
        }

        /// <summary>The innermost open node enclosing each node, or null at module scope.</summary>
        private static Dictionary<FastNodeKey, FastNodeKey?> ComputeEnclosingScope(InternalComputationGraph graph)
        {
            var enclosing = new Dictionary<FastNodeKey, FastNodeKey?>(graph.Nodes.Count);
            var open = new Stack<FastNodeKey>();
            foreach (var node in graph.Nodes)
            {
                if (node.IsCloseNode() && open.Count > 0) open.Pop();
                enclosing[node.Key] = open.Count > 0 ? open.Peek() : null;
                if (node.IsOpenNode()) open.Push(node.Key);
            }
            return enclosing;
        }

        /// <summary>
        /// Which arm of which IfElse each node belongs to. The arms share one scope — node order
        /// does not separate them — so membership is read off what each branch's own inputs reach.
        /// </summary>
        private static Dictionary<FastNodeKey, ArmKey> ClassifyBranches(
            InternalComputationGraph graph, Dictionary<FastNodeKey, int> positionOf)
        {
            var producerOf = new Dictionary<FastTensorKey, FastNode>();
            foreach (var node in graph.Nodes)
                foreach (var (_, outs) in node.FullOutputs)
                    foreach (var ok in outs)
                        if (ok is not null && !ok.Value.IsEmpty) producerOf[ok.Value] = node;

            var armOf = new Dictionary<FastNodeKey, ArmKey>();
            foreach (var close in graph.Nodes)
            {
                if (close.OpCode != OpCodes.IF_CLOSE) continue;
                int openPos = close.GraphOpenNodeKey is FastNodeKey openKey && positionOf.TryGetValue(openKey, out var op)
                    ? op : -1;
                foreach (var branch in (string[])[OnnxOpAttributeNames.AttrThenBranch, OnnxOpAttributeNames.AttrElseBranch])
                {
                    if (!close.FullInputs.TryGetValue(branch, out var roots)) continue;
                    var seen = new HashSet<FastNodeKey>();
                    var worklist = new Stack<FastNode>();
                    foreach (var root in roots)
                        if (root is FastTensorKey k && producerOf.TryGetValue(k, out var producer)
                            && seen.Add(producer.Key))
                            worklist.Push(producer);

                    while (worklist.Count > 0)
                    {
                        var node = worklist.Pop();
                        // Stop at the branch's edge: a node from before the If belongs to neither arm.
                        if (positionOf[node.Key] <= openPos) continue;
                        armOf[node.Key] = new ArmKey(close.Key, branch);
                        foreach (var (_, ins) in node.FullInputs)
                            foreach (var ik in ins)
                                if (ik is FastTensorKey k && producerOf.TryGetValue(k, out var producer)
                                    && seen.Add(producer.Key))
                                    worklist.Push(producer);
                    }
                }
            }
            return armOf;
        }

        /// <summary>
        /// The value behind an Identity chain. Inlining wraps a parameter in one at every splice,
        /// so a link names its parameter only at the end of the chain.
        /// </summary>
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

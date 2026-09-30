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
    /// <para><b>Calls in the arms of an IfElse.</b> The arms are alternatives, so composing across
    /// them would credit an update that never happened. Each arm is chained on its own from the
    /// value the parameter held entering the branch, both arms' results are threaded out through
    /// the <c>IF_CLOSE</c> as one more output pair, and a link at the branch's own scope takes
    /// whichever arm ran (Shorokoo/Shorokoo#308). An arm that does not call the model hands the
    /// incoming value straight back, so a call in one arm alone updates the state only when that
    /// arm runs.</para>
    ///
    /// <para><b>Calls in a loop body.</b> A body is one call site however many trips it runs: every
    /// trip starts from the value the parameter held entering the loop, the calls within a trip
    /// compose in order, and the update is the one the last trip that ran made. The loop is unrolled
    /// by now, with each trip's reads of the parameter cloned in call order and each link naming its
    /// trips (<see cref="OnnxOpAttributeNames.ShrkAttrLoopTrips"/>), so the calls thread like any
    /// others except that a new trip starts over from the value entering its loop. Where whether a
    /// trip runs is decided at run time, a trip that does not run starts from where the previous
    /// one ended and its updates hand back what they read, so it changes nothing.</para>
    ///
    /// <para>A branch expression is an ordinary argument, so its nodes are traced <em>before</em>
    /// the <c>IF_OPEN</c> that selects them and only move inside it at ONNX build time (see
    /// <see cref="FastIfBranchScoper"/>). Which arm a call belongs to is therefore read off what
    /// the <c>IF_CLOSE</c>'s branch inputs reach, not off where the call sits: a value both arms
    /// read belongs to neither and is the parameter's ordinary, unconditional history.</para>
    /// </summary>
    internal static class FastChainStateUpdatesAcrossCallSites
    {
        /// <summary>One call site of a state parameter: its link, the marker that closes it, and
        /// the arm it belongs to (null at module scope).</summary>
        private readonly record struct CallSite(FastNode Link, FastNode Marker, IfArm? Arm);

        /// <param name="graph">The graph, rewritten in place.</param>
        /// <param name="onlyLoopsUnrolledSince">Order only the parameters an unrolled loop's links
        /// still name trips for — those a loop kept rolled the last time this ran, and which a
        /// later simplify has unrolled. The others are ordered already; a link is left naming its
        /// trips only until it is.</param>
        public static void Process(InternalComputationGraph graph, bool onlyLoopsUnrolledSince = false)
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
            var armsOfNode = FastIfArms.Classify(graph);

            // Range rewrites and node insertions are collected first and applied after, so every
            // position taken here stays the one the node still has.
            var rewrites = new List<(int fromPos, int toPos, FastTensorKey param, FastTensorKey replacement)>();
            var insertions = new List<(int atPos, FastNode node)>();
            var branchLinks = new List<FastTensorKey>();

            foreach (var (paramNodeKey, links) in linksByParam)
            {
                if (onlyLoopsUnrolledSince && links.All(l => TripsOf(l).Count == 0)) continue;
                var sites = CallSitesOf(graph, links, enclosingScope, armsOfNode, nodeByKey);
                if (sites is null) continue;                        // shape this pass does not order
                if (sites.Count < 2 && sites.All(s => s.Arm is null && TripsOf(s.Link).All(t => t.Ran is null)))
                    continue;                                       // nothing to compose

                Thread(graph, new FastTensorKey(paramNodeKey, 0), sites, positionOf, nodeByKey,
                       rewrites, insertions, branchLinks);
            }

            ApplyRewrites(graph, rewrites, nodeByKey);

            // From the back, so earlier positions stay put; of two at one position the one added
            // first ends up first.
            foreach (var (at, node, _) in insertions.Select((x, order) => (x.atPos, x.node, order))
                                                    .OrderByDescending(x => x.atPos).ThenByDescending(x => x.order))
                graph.Nodes.Insert(at, node);

            RootBranchLinks(graph, branchLinks);
        }

        /// <summary>Points every read of a parameter within each node range at its replacement.</summary>
        private static void ApplyRewrites(
            InternalComputationGraph graph,
            List<(int fromPos, int toPos, FastTensorKey param, FastTensorKey replacement)> rewrites,
            Dictionary<FastNodeKey, FastNode> nodeByKey)
        {
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
        }

        /// <summary>The unrolled loop trips a link belongs to, outermost-unrolled last, each with the
        /// flag saying whether it ran when that is decided at run time.</summary>
        private static List<(long Loop, long Trip, FastTensorKey? Ran)> TripsOf(FastNode link)
        {
            var trips = new List<(long, long, FastTensorKey?)>();
            if (link.Attributes.GetLongsVal(OnnxOpAttributeNames.ShrkAttrLoopTrips) is not long[] vals) return trips;
            for (int i = 0; i + 2 < vals.Length; i += 3)
                trips.Add((vals[i], vals[i + 1], vals[i + 2] >= 0 ? link.Inputs[(int)vals[i + 2]] : null));
            return trips;
        }

        /// <summary>The <c>WITH_STATE_DEPS</c> that closes each link's call: the first to name it.</summary>
        private static Dictionary<FastNodeKey, FastNode> MarkerByLink(
            InternalComputationGraph graph, IEnumerable<FastNode> links)
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
            return markerByLink;
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
            graph.InsertAtBodyEnd(new FastNode
            {
                Key = key,
                OpCode = InternalOpCodes.WITH_STATE_DEPS,
                Attributes = OnnxCSharpAttributes.FromCSharpVals(
                    new Dictionary<string, object?>(),
                    Definitions.NodeDefinitions[InternalOpCodes.WITH_STATE_DEPS].AttributeDefs),
                FullInputs = { [""] = [graph.Outputs[0], .. branchLinks.Select(k => (FastTensorKey?)k)] },
                FullOutputs = { [""] = [output] },
            });
            graph.RetargetOutput(0, output);
        }

        /// <summary>
        /// This parameter's call sites in node order, or null when the shape is one this pass does
        /// not order: a call inside a loop body, whose calls repeat rather than run in sequence —
        /// and a rolled loop carrying state is refused where it matters, when the graph is prepared
        /// for training.
        /// </summary>
        private static List<CallSite>? CallSitesOf(
            InternalComputationGraph graph,
            List<FastNode> links,
            Dictionary<FastNodeKey, FastNodeKey?> enclosingScope,
            Dictionary<FastNodeKey, List<IfArm>> armsOfNode,
            Dictionary<FastNodeKey, FastNode> nodeByKey)
        {
            var markerByLink = MarkerByLink(graph, links);

            var sites = new List<CallSite>(links.Count);
            foreach (var link in links)
            {
                if (!markerByLink.TryGetValue(link.Key, out var marker)) return null;
                if (enclosingScope[link.Key] is not null) return null;   // inside a loop body

                var arms = armsOfNode.TryGetValue(link.Key, out var found) ? found : [];
                if (arms.Count == 0) { sites.Add(new CallSite(link, marker, null)); continue; }
                if (arms.Count > 1)
                    throw new InvalidOperationException(
                        "FastChainStateUpdatesAcrossCallSites: a state update sits inside nested IfElse "
                        + "branches, so which of them decides whether it happened takes reasoning this "
                        + "pass does not do. Chaining it as an ordinary call would apply an update the "
                        + "condition did not select. Call the model once outside the branches, or give "
                        + "each branch its own model.");
                if (!nodeByKey.TryGetValue(arms[0].IfClose, out var close)) return null;
                if (enclosingScope[close.Key] is not null) return null;   // the IfElse is inside a loop
                sites.Add(new CallSite(link, marker, arms[0]));
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
            var loops = new Dictionary<long, (FastTensorKey Entry, long Trip, int Order)>();
            var armValue = new Dictionary<IfArm, FastTensorKey>();
            var armsOfClose = new Dictionary<FastNodeKey, FastTensorKey>();   // the value entering each branch
            FastNodeKey? pendingClose = null;                                 // a branch still being read
            int previousMarker = -1;

            for (int i = 0; i < sites.Count; i++)
            {
                var site = sites[i];
                var trips = TripsOf(site.Link);
                current = EnterTrips(graph, param, trips, loops, current,
                                     previousMarker + 1, positionOf[site.Marker.Key], nodeByKey, insertions);

                if (site.Arm is IfArm arm)
                {
                    if (!armsOfClose.ContainsKey(arm.IfClose)) armsOfClose[arm.IfClose] = current;
                    pendingClose = arm.IfClose;
                    var incoming = armValue.TryGetValue(arm, out var held) ? held : armsOfClose[arm.IfClose];
                    rewrites.Add((previousMarker + 1, positionOf[site.Marker.Key], param, incoming));
                    GateOnTripsRan(site.Link, trips, incoming, positionOf, insertions);
                    armValue[arm] = site.Link.Outputs[0]!.Value;
                }
                else
                {
                    // A call no branch chooses between, made after one a branch does, would have to
                    // take its turn before a value that only exists at the IF_CLOSE following it.
                    if (pendingClose is not null)
                        throw new InvalidOperationException(
                            "FastChainStateUpdatesAcrossCallSites: a call this IfElse does not choose "
                            + "between is made after one it does, so the parameter would have to carry "
                            + "the branch's answer before the branch produces it. Make the calls the "
                            + "branch does not choose between before the ones it does.");
                    rewrites.Add((previousMarker + 1, positionOf[site.Marker.Key], param, current));
                    GateOnTripsRan(site.Link, trips, current, positionOf, insertions);
                    current = site.Link.Outputs[0]!.Value;
                }
                previousMarker = positionOf[site.Marker.Key];

                // Close out a branch once its last call site is behind us.
                bool lastOfThisClose = site.Arm is IfArm a
                    && (i + 1 == sites.Count || sites[i + 1].Arm?.IfClose != a.IfClose);
                if (!lastOfThisClose) continue;

                var closing = (IfArm)site.Arm!;
                var ifClose = nodeByKey[closing.IfClose];
                current = CloseBranch(graph, ifClose, closing.Condition, armsOfClose[ifClose.Key],
                                      armValue, positionOf, insertions);
                branchLinks.Add(current);
                pendingClose = null;
            }
        }

        /// <summary>
        /// The value a call site starts from, given the unrolled loop trips it belongs to. The first
        /// call of a loop records what the parameter holds entering it; a call in a later trip of a
        /// loop already entered starts that trip over from there — from the outermost such loop,
        /// whose trip forgets the loops nested in it. Where the trip may not run, it starts from
        /// where the previous trip ended instead when it does not: a link at the first read of the
        /// range selects which.
        /// </summary>
        private static FastTensorKey EnterTrips(
            InternalComputationGraph graph,
            FastTensorKey param,
            List<(long Loop, long Trip, FastTensorKey? Ran)> trips,
            Dictionary<long, (FastTensorKey Entry, long Trip, int Order)> loops,
            FastTensorKey current,
            int fromPos, int toPos,
            Dictionary<FastNodeKey, FastNode> nodeByKey,
            List<(int atPos, FastNode node)> insertions)
        {
            (long Loop, long Trip, FastTensorKey? Ran)? restart = null;
            int restartOrder = int.MaxValue;
            foreach (var t in trips)
                if (loops.TryGetValue(t.Loop, out var l) && l.Trip != t.Trip && l.Order < restartOrder)
                    (restart, restartOrder) = (t, l.Order);

            if (restart is { } r)
            {
                foreach (var nested in loops.Where(kv => kv.Value.Order > restartOrder).Select(kv => kv.Key).ToList())
                    loops.Remove(nested);
                var entry = loops[r.Loop].Entry;
                loops[r.Loop] = loops[r.Loop] with { Trip = r.Trip };
                current = r.Ran is FastTensorKey ran
                    ? StartTripIfItRan(graph, param, ran, entry, current, fromPos, toPos, nodeByKey, insertions)
                    : entry;
            }

            foreach (var t in trips)
                if (!loops.ContainsKey(t.Loop))
                    loops[t.Loop] = (current, t.Trip, loops.Count == 0 ? 0 : loops.Values.Max(l => l.Order) + 1);
            return current;
        }

        /// <summary>
        /// <c>STATE_UPDATE_LINK(previous, Where(ran, entry, previous))</c>, placed at the first read
        /// of the parameter in the trip's first range: the value entering the loop when the trip
        /// runs, the previous trip's when it does not.
        /// </summary>
        private static FastTensorKey StartTripIfItRan(
            InternalComputationGraph graph, FastTensorKey param, FastTensorKey ran,
            FastTensorKey entry, FastTensorKey previous, int fromPos, int toPos,
            Dictionary<FastNodeKey, FastNode> nodeByKey, List<(int atPos, FastNode node)> insertions)
        {
            int at = toPos;
            for (int i = fromPos; i <= toPos && i < graph.Nodes.Count; i++)
                if (graph.Nodes[i].Inputs.Any(k => k is FastTensorKey key
                        && ResolveThroughIdentities(key, nodeByKey) is FastTensorKey read && read.Equals(param)))
                {
                    at = i;
                    break;
                }

            var selected = Where(ran, entry, previous, at, insertions);
            var linkKey = FastNodeKey.New();
            insertions.Add((at, new FastNode
            {
                Key = linkKey,
                OpCode = InternalOpCodes.STATE_UPDATE_LINK,
                Attributes = OnnxCSharpAttributes.FromCSharpVals(
                    new Dictionary<string, object?>(),
                    Definitions.NodeDefinitions[InternalOpCodes.STATE_UPDATE_LINK].AttributeDefs),
                FullInputs = { [""] = [previous, selected] },
                FullOutputs = { [""] = [new FastTensorKey(linkKey, 0)] },
            }));
            return new FastTensorKey(linkKey, 0);
        }

        /// <summary>
        /// Makes a link's update hand back what it read on a trip that does not run — a trip the
        /// continue condition skipped makes no update — and drops the trips the link named, now
        /// that they are ordered.
        /// </summary>
        private static void GateOnTripsRan(
            FastNode link, List<(long Loop, long Trip, FastTensorKey? Ran)> trips, FastTensorKey read,
            Dictionary<FastNodeKey, int> positionOf, List<(int atPos, FastNode node)> insertions)
        {
            var updated = link.Inputs[1]!.Value;
            foreach (var t in trips)
                if (t.Ran is FastTensorKey ran)
                    updated = Where(ran, updated, read, positionOf[link.Key], insertions);
            link.FullInputs[""] = [link.Inputs[0], updated];
            link.Attributes = OnnxCSharpAttributes.FromCSharpVals(
                new Dictionary<string, object?>(),
                Definitions.NodeDefinitions[InternalOpCodes.STATE_UPDATE_LINK].AttributeDefs);
        }

        private static FastTensorKey Where(
            FastTensorKey condition, FastTensorKey whenTrue, FastTensorKey whenFalse, int at,
            List<(int atPos, FastNode node)> insertions)
        {
            var key = FastNodeKey.New();
            insertions.Add((at, new FastNode
            {
                Key = key,
                OpCode = OpCodes.WHERE,
                Attributes = OnnxCSharpAttributes.FromCSharpVals(
                    new Dictionary<string, object?>(), Definitions.NodeDefinitions[OpCodes.WHERE].AttributeDefs),
                FullInputs = { [""] = [condition, whenTrue, whenFalse] },
                FullOutputs = { [""] = [new FastTensorKey(key, 0)] },
            }));
            return new FastTensorKey(key, 0);
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
            FastTensorKey condition,
            FastTensorKey incoming,
            Dictionary<IfArm, FastTensorKey> armValue,
            Dictionary<FastNodeKey, int> positionOf,
            List<(int atPos, FastNode node)> insertions)
        {
            int closePos = positionOf[ifClose.Key];
            var identityAttrs = OnnxCSharpAttributes.FromCSharpVals(
                new Dictionary<string, object?>(), Definitions.NodeDefinitions[OpCodes.IDENTITY].AttributeDefs);

            FastTensorKey ArmFinal(bool isThen)
            {
                if (armValue.TryGetValue(new IfArm(ifClose.Key, condition, isThen), out var v)) return v;
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

            var thenFinal = ArmFinal(isThen: true);
            var elseFinal = ArmFinal(isThen: false);

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

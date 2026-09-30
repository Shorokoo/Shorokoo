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
    /// <para><b>Calls in a rolled loop's body.</b> A loop whose trip count is not known when this
    /// runs is still standing, with the calls in its body once. The same rule applies: the first
    /// call in the body reads the value entering the loop on every trip, the calls after it compose
    /// in order, and the value the last of them produces is carried out of the loop as one more
    /// loop variable, so the one after the <c>LOOP_CLOSE</c> is the last trip's, or the value
    /// entering the loop when no trip runs. A link at the loop's own scope takes it, as for an
    /// IfElse. A rolled loop nested in another is carried out of the inner one into the outer
    /// one's body first, whose trips each start the inner loop over; an unrolled loop in a rolled
    /// one's body, or a rolled one in an unrolled one's, orders by both rules. An IfElse whose arm
    /// holds a rolled loop threads the loop's value into the arm; an IfElse in a rolled loop's body
    /// is left unordered.</para>
    ///
    /// <para>A branch expression is an ordinary argument, so its nodes are traced <em>before</em>
    /// the <c>IF_OPEN</c> that selects them and only move inside it at ONNX build time (see
    /// <see cref="FastIfBranchScoper"/>). Which arm a call belongs to is therefore read off what
    /// the <c>IF_CLOSE</c>'s branch inputs reach, not off where the call sits: a value both arms
    /// read, or one read outside the branch too, belongs to neither and is the parameter's
    /// ordinary, unconditional history (see <see cref="FastIfArms"/>).</para>
    /// </summary>
    internal static class FastChainStateUpdatesAcrossCallSites
    {
        /// <summary>One call site of a state parameter: its link, the marker that closes it, the
        /// arm it belongs to (null at module scope) and the rolled loops whose bodies it is in,
        /// outermost first.</summary>
        private readonly record struct CallSite(FastNode Link, FastNode Marker, IfArm? Arm, List<FastNode> Loops);

        /// <param name="graph">The graph, rewritten in place.</param>
        /// <param name="onlyLoopsUnrolledSince">Order only the parameters none of whose links this
        /// pass has ordered (<see cref="OnnxOpAttributeNames.ShrkAttrStateOrdered"/>) and whose
        /// unrolled loops' links name trips — those this pass left alone the last time it ran, and
        /// which a later simplify has unrolled. A loop ordered while still rolled carries its state
        /// out as a loop variable, which unrolling keeps as it is.</param>
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
            var scopeLinks = new List<FastTensorKey>();

            foreach (var (paramNodeKey, links) in linksByParam)
            {
                if (onlyLoopsUnrolledSince && (links.Any(IsOrdered) || links.All(l => TripsOf(l).Count == 0)))
                    continue;
                var sites = CallSitesOf(graph, links, enclosingScope, armsOfNode, nodeByKey);
                if (sites is null) continue;                        // shape this pass does not order
                if (sites.Count < 2 && sites.All(s => s.Arm is null && s.Loops.Count == 0
                                                      && TripsOf(s.Link).All(t => t.Ran is null && t.Trip == t.Count - 1)))
                {
                    foreach (var link in links) link.Attributes = OrderedLinkAttributes();
                    continue;                                       // nothing to compose
                }

                Thread(graph, new FastTensorKey(paramNodeKey, 0), sites, positionOf, nodeByKey,
                       rewrites, insertions, scopeLinks);
            }

            ApplyRewrites(graph, rewrites, nodeByKey);

            // From the back, so earlier positions stay put; of two at one position the one added
            // first ends up first.
            foreach (var (at, node, _) in insertions.Select((x, order) => (x.atPos, x.node, order))
                                                    .OrderByDescending(x => x.atPos).ThenByDescending(x => x.order))
                graph.Nodes.Insert(at, node);

            RootScopeLinks(graph, scopeLinks);
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

        /// <summary>Whether this pass has ordered the link already.</summary>
        internal static bool IsOrdered(FastNode link)
            => link.Attributes.GetBoolVal(OnnxOpAttributeNames.ShrkAttrStateOrdered) == true;

        /// <summary>A link this pass has ordered: marked so, and naming no trips any more.</summary>
        private static OnnxCSharpAttributes OrderedLinkAttributes()
            => OnnxCSharpAttributes.FromCSharpVals(
                new Dictionary<string, object?> { [OnnxOpAttributeNames.ShrkAttrStateOrdered] = true },
                Definitions.NodeDefinitions[InternalOpCodes.STATE_UPDATE_LINK].AttributeDefs);

        /// <summary>The unrolled loop trips a link belongs to, outermost-unrolled last, each with the
        /// flag saying whether it ran when that is decided at run time.</summary>
        private static List<(long Loop, long Trip, long Count, FastTensorKey? Ran)> TripsOf(FastNode link)
        {
            var trips = new List<(long, long, long, FastTensorKey?)>();
            if (link.Attributes.GetLongsVal(OnnxOpAttributeNames.ShrkAttrLoopTrips) is not long[] vals) return trips;
            for (int i = 0; i + 3 < vals.Length; i += 4)
                trips.Add((vals[i], vals[i + 1], vals[i + 2], vals[i + 3] >= 0 ? link.Inputs[(int)vals[i + 3]] : null));
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
        /// Keeps the links threading an IfElse or a rolled loop created reachable. Nothing reads the
        /// value a state update produces — the state lowerings collect it — so a link nobody
        /// depends on is swept as dead, taking the branch or loop output it takes with it. The same
        /// <c>WITH_STATE_DEPS</c> marker a module body puts on its outputs says so here.
        /// </summary>
        private static void RootScopeLinks(InternalComputationGraph graph, List<FastTensorKey> scopeLinks)
        {
            if (scopeLinks.Count == 0 || graph.Outputs.Count == 0) return;

            var key = FastNodeKey.New();
            var output = new FastTensorKey(key, 0);
            graph.InsertAtBodyEnd(new FastNode
            {
                Key = key,
                OpCode = InternalOpCodes.WITH_STATE_DEPS,
                Attributes = OnnxCSharpAttributes.FromCSharpVals(
                    new Dictionary<string, object?>(),
                    Definitions.NodeDefinitions[InternalOpCodes.WITH_STATE_DEPS].AttributeDefs),
                FullInputs = { [""] = [graph.Outputs[0], .. scopeLinks.Select(k => (FastTensorKey?)k)] },
                FullOutputs = { [""] = [output] },
            });
            graph.RetargetOutput(0, output);
        }

        /// <summary>
        /// This parameter's call sites in node order, or null when the shape is one this pass does
        /// not order: a call in an IfElse inside a rolled loop's body.
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
                var arms = armsOfNode.TryGetValue(link.Key, out var found) ? found : [];

                if (enclosingScope[link.Key] is FastNodeKey scope)
                {
                    if (enclosingScope[marker.Key] != scope || arms.Count > 1) return null;
                    var loops = new List<FastNode>();
                    for (FastNodeKey? s = scope; s is FastNodeKey open; s = enclosingScope[open])
                    {
                        if (!nodeByKey.TryGetValue(open, out var loop) || loop.OpCode != OpCodes.LOOP_OPEN) return null;
                        loops.Insert(0, loop);
                    }
                    // An IfElse whose arm holds the whole loop orders like one holding a call; one
                    // inside the loop's body is not ordered here.
                    if (arms.Count == 1 && (!nodeByKey.TryGetValue(arms[0].IfClose, out var armClose)
                                            || enclosingScope[armClose.Key] is not null))
                        return null;
                    sites.Add(new CallSite(link, marker, arms.Count == 1 ? arms[0] : null, loops));
                    continue;
                }

                if (arms.Count == 0) { sites.Add(new CallSite(link, marker, null, [])); continue; }
                if (arms.Count > 1)
                    throw new InvalidOperationException(
                        "FastChainStateUpdatesAcrossCallSites: a state update sits inside nested IfElse "
                        + "branches, so which of them decides whether it happened takes reasoning this "
                        + "pass does not do. Chaining it as an ordinary call would apply an update the "
                        + "condition did not select. Call the model once outside the branches, or give "
                        + "each branch its own model.");
                if (!nodeByKey.TryGetValue(arms[0].IfClose, out var close)) return null;
                if (enclosingScope[close.Key] is not null) return null;   // the IfElse is inside a loop
                sites.Add(new CallSite(link, marker, arms[0], []));
            }
            return sites;
        }

        /// <summary>
        /// Walks the parameter's call sites in node order, giving each the value the parameter
        /// holds when that call starts, and threading an IfElse's arms back together after it and
        /// a rolled loop's body out of it.
        /// </summary>
        private static void Thread(
            InternalComputationGraph graph,
            FastTensorKey param,
            List<CallSite> sites,
            Dictionary<FastNodeKey, int> positionOf,
            Dictionary<FastNodeKey, FastNode> nodeByKey,
            List<(int fromPos, int toPos, FastTensorKey param, FastTensorKey replacement)> rewrites,
            List<(int atPos, FastNode node)> insertions,
            List<FastTensorKey> scopeLinks)
        {
            var current = param;
            var loopEntries = new List<FastTensorKey>();                      // the value entering each rolled loop open
            var loops = new Dictionary<long, UnrolledLoop>();
            var armValue = new Dictionary<IfArm, FastTensorKey>();
            var armsOfClose = new Dictionary<FastNodeKey, FastTensorKey>();   // the value entering each branch
            FastNodeKey? pendingClose = null;                                 // a branch still being read
            int previousMarker = -1;

            for (int i = 0; i < sites.Count; i++)
            {
                var site = sites[i];
                var trips = TripsOf(site.Link);

                // The value the call starts from: the arm's own where a branch chooses it, the
                // parameter's running value where none does — either one started over where the
                // call begins a new trip of an unrolled loop.
                FastTensorKey start;
                if (site.Arm is IfArm arm)
                {
                    if (!armsOfClose.ContainsKey(arm.IfClose)) armsOfClose[arm.IfClose] = current;
                    pendingClose = arm.IfClose;
                    start = armValue.TryGetValue(arm, out var held) ? held : armsOfClose[arm.IfClose];
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
                    start = current;
                }
                int firstRolledEntered = loopEntries.Count < site.Loops.Count
                    ? positionOf[site.Loops[loopEntries.Count].Key] : int.MaxValue;
                start = EnterTrips(graph, param, trips, loops, start, previousMarker + 1,
                                   Math.Min(positionOf[site.Marker.Key], firstRolledEntered), nodeByKey, insertions);
                while (loopEntries.Count < site.Loops.Count) loopEntries.Add(start);
                rewrites.Add((previousMarker + 1, positionOf[site.Marker.Key], param, start));
                GateOnTripsRan(site.Link, trips, start, positionOf, insertions);
                var after = site.Link.Outputs[0]!.Value;
                previousMarker = positionOf[site.Marker.Key];

                // Leave the rolled loops the next call is not in, innermost first.
                int stay = i + 1 == sites.Count ? 0 : SharedPrefix(site.Loops, sites[i + 1].Loops);
                int leftAt = positionOf[site.Marker.Key];
                for (int l = site.Loops.Count - 1; l >= stay; l--)
                {
                    after = CloseLoop(graph, site.Loops[l], loopEntries[l], after, positionOf, insertions);
                    leftAt = Math.Max(leftAt, positionOf[ClosingOf(graph, site.Loops[l]).Key]);
                    loopEntries.RemoveAt(l);
                    if (l == 0 && site.Arm is null) scopeLinks.Add(after);
                }

                // Leave the unrolled loops the next call is not in. A loop's update is its last
                // trip's, so one whose last trip made no call — an IfElse on the iteration folded
                // it away — hands back the value entering it.
                var nextLoops = i + 1 == sites.Count ? [] : TripsOf(sites[i + 1].Link).Select(t => t.Loop).ToHashSet();
                foreach (var (id, left) in loops.Where(kv => !nextLoops.Contains(kv.Key)).OrderByDescending(kv => kv.Value.Order).ToList())
                {
                    loops.Remove(id);
                    if (left.Trip < left.Count - 1 && !left.MayNotRun)
                    {
                        after = HandBack(after, left.Entry, leftAt + 1, insertions);
                        if (site.Arm is null) scopeLinks.Add(after);
                    }
                }
                if (site.Arm is IfArm siteArm) armValue[siteArm] = after;
                else current = after;
                if (stay > 0) continue;

                // Close out a branch once its last call site is behind us.
                bool lastOfThisClose = site.Arm is IfArm a
                    && (i + 1 == sites.Count || sites[i + 1].Arm?.IfClose != a.IfClose);
                if (!lastOfThisClose) continue;

                var closing = (IfArm)site.Arm!;
                var ifClose = nodeByKey[closing.IfClose];
                current = CloseBranch(graph, ifClose, closing.Condition, armsOfClose[ifClose.Key],
                                      armValue, positionOf, insertions);
                scopeLinks.Add(current);
                pendingClose = null;
            }
        }

        /// <summary>An unrolled loop the calls are in: the value entering it, the trip its latest call
        /// is in, when it was entered relative to the others, its trip count, and whether a trip of it
        /// may not run.</summary>
        private readonly record struct UnrolledLoop(FastTensorKey Entry, long Trip, int Order, long Count, bool MayNotRun);

        /// <summary><c>STATE_UPDATE_LINK(from, to)</c>: the parameter holds <paramref name="to"/> from
        /// here on.</summary>
        private static FastTensorKey HandBack(
            FastTensorKey from, FastTensorKey to, int at, List<(int atPos, FastNode node)> insertions)
        {
            var key = FastNodeKey.New();
            insertions.Add((at, new FastNode
            {
                Key = key,
                OpCode = InternalOpCodes.STATE_UPDATE_LINK,
                Attributes = OrderedLinkAttributes(),
                FullInputs = { [""] = [from, to] },
                FullOutputs = { [""] = [new FastTensorKey(key, 0)] },
            }));
            return new FastTensorKey(key, 0);
        }

        private static FastNode ClosingOf(InternalComputationGraph graph, FastNode loopOpen)
            => graph.Nodes.Single(n => n.OpCode == OpCodes.LOOP_CLOSE && n.GraphOpenNodeKey == loopOpen.Key);

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
            List<(long Loop, long Trip, long Count, FastTensorKey? Ran)> trips,
            Dictionary<long, UnrolledLoop> loops,
            FastTensorKey current,
            int fromPos, int toPos,
            Dictionary<FastNodeKey, FastNode> nodeByKey,
            List<(int atPos, FastNode node)> insertions)
        {
            (long Loop, long Trip, long Count, FastTensorKey? Ran)? restart = null;
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
                    loops[t.Loop] = new UnrolledLoop(current, t.Trip, loops.Count == 0 ? 0 : loops.Values.Max(l => l.Order) + 1,
                                                     t.Count, t.Ran is not null);
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
                Attributes = OrderedLinkAttributes(),
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
            FastNode link, List<(long Loop, long Trip, long Count, FastTensorKey? Ran)> trips, FastTensorKey read,
            Dictionary<FastNodeKey, int> positionOf, List<(int atPos, FastNode node)> insertions)
        {
            var updated = link.Inputs[1]!.Value;
            foreach (var t in trips)
                if (t.Ran is FastTensorKey ran)
                    updated = Where(ran, updated, read, positionOf[link.Key], insertions);
            link.FullInputs[""] = [link.Inputs[0], updated];
            link.Attributes = OrderedLinkAttributes();
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
                Attributes = OrderedLinkAttributes(),
                FullInputs = { [""] = [incoming, selected] },
                FullOutputs = { [""] = [linkOutput] },
            }));
            return linkOutput;
        }

        /// <summary>
        /// Carries a parameter out of a rolled loop: the value its body's last call produces becomes
        /// one more loop variable, started from the value entering the loop, and a link after the
        /// <c>LOOP_CLOSE</c> takes the value it ends with. The body never reads the variable — each
        /// trip starts from the value entering the loop — so it is the last trip's value that comes
        /// out, and the entering one when no trip runs.
        /// </summary>
        private static FastTensorKey CloseLoop(
            InternalComputationGraph graph,
            FastNode loopOpen,
            FastTensorKey entering,
            FastTensorKey bodyFinal,
            Dictionary<FastNodeKey, int> positionOf,
            List<(int atPos, FastNode node)> insertions)
        {
            var loopClose = graph.Nodes.Single(n => n.OpCode == OpCodes.LOOP_CLOSE && n.GraphOpenNodeKey == loopOpen.Key);

            // LOOP_OPEN: [maxIter, cond, ...inits] -> [iterIndex, vestigialTrue, ...loopVars].
            // LOOP_CLOSE: [cond, ...bodyOuts, ...scanInputs] -> [...finals, ...scans].
            var openInputs = loopOpen.FullInputs[""];
            int loopVars = Math.Max(0, openInputs.Count - 2);
            var openOutputs = loopOpen.FullOutputs.Single().Value;
            var closeInputs = loopClose.FullInputs.Single().Value;
            var closeOutputs = loopClose.FullOutputs.Single().Value;

            // The open lists a loop variable two slots after the close does.
            int slot = Math.Max(FirstFreeOutputIndex(loopClose.Key, closeOutputs),
                                FirstFreeOutputIndex(loopOpen.Key, openOutputs) - 2);
            openInputs.Add(entering);
            openOutputs.Add(new FastTensorKey(loopOpen.Key, slot + 2));
            closeInputs.Insert(1 + loopVars, bodyFinal);
            var final = new FastTensorKey(loopClose.Key, slot);
            closeOutputs.Insert(loopVars, final);

            var linkKey = FastNodeKey.New();
            insertions.Add((positionOf[loopClose.Key] + 1, new FastNode
            {
                Key = linkKey,
                OpCode = InternalOpCodes.STATE_UPDATE_LINK,
                Attributes = OrderedLinkAttributes(),
                FullInputs = { [""] = [entering, final] },
                FullOutputs = { [""] = [new FastTensorKey(linkKey, 0)] },
            }));
            return new FastTensorKey(linkKey, 0);
        }

        private static int SharedPrefix(List<FastNode> a, List<FastNode> b)
        {
            int n = 0;
            while (n < a.Count && n < b.Count && a[n] == b[n]) n++;
            return n;
        }

        private static int FirstFreeOutputIndex(FastNodeKey nodeKey, List<FastTensorKey?> outputs)
        {
            int next = 0;
            foreach (var o in outputs)
                if (o is FastTensorKey k && !k.IsEmpty && k.FastNodeKey.Equals(nodeKey) && k.OutputIndex >= next)
                    next = k.OutputIndex + 1;
            return next;
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

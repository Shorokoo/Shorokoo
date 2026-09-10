using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Factory;
using Shorokoo.Modules;
using System;
using System.Collections.Generic;
using Shorokoo.Core.Nodes.Processors.AutoGrad;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Implementation of <see cref="InternalComputationGraph.ConfigureScopes"/>.
    ///
    /// Scope membership in the Fast pipeline is positional: a non-boundary node
    /// is "in scope S" iff its index in <see cref="InternalComputationGraph.Nodes"/>
    /// falls between S's <c>OPEN</c> and <c>CLOSE</c>. This class assigns each
    /// node a per-scope MustIn / MustOut / Free status from the graph's data
    /// flow, places it inside every scope that does not force it out, then
    /// rewrites <see cref="InternalComputationGraph.Nodes"/> to honor the decisions
    /// while preserving topological order.
    /// </summary>
    internal static class FastScopeConfigurator
    {
        private enum ScopeKind { Loop, If }
        private enum NodeStatus { MustIn, MustOut, Free }

        private sealed class Scope
        {
            public int Id;
            public int OpenIdx;
            public int CloseIdx;
            public FastNode OpenNode = null!;
            public FastNode CloseNode = null!;
            public ScopeKind Kind;
            public int? Parent;
        }

        public static void Configure(InternalComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            System.Diagnostics.Debug.Assert(graph.IsLinearOrderValid(),
                "FastScopeConfigurator.Configure: graph must already be in valid linear order on entry. " +
                "This pass resizes scopes; it does not fix invalid ones.");

            var scopes = CollectScopes(graph);
            if (scopes.Count == 0) return;

            var (forwardReach, backwardReach) = BuildReachSets(graph, scopes);
            var (thenReach, elseReach) = BuildIfBranchReachSets(graph, scopes);
            var inAByStructure = BuildContainmentTable(scopes);

            var status = ClassifyNodes(
                graph, scopes, forwardReach, backwardReach, thenReach, elseReach, inAByStructure);
            var decision = DecidePlacement(graph, scopes, status);

            graph.Nodes = Reorder(graph, scopes, decision);
            graph.Nodes = ReorderIfBranches(graph, scopes, thenReach, elseReach);
            System.Diagnostics.Debug.Assert(graph.IsLinearOrderValid(), "graph.IsLinearOrderValid()");
        }

        // -- Step 1 -----------------------------------------------------------

        private static List<Scope> CollectScopes(InternalComputationGraph graph)
        {
            var scopes = new List<Scope>();
            var stack = new Stack<int>();
            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                var n = graph.Nodes[i];
                if (FastOpsetResolver.IsOpenOpCode(n.OpCode))
                {
                    var s = new Scope
                    {
                        Id = scopes.Count,
                        OpenIdx = i,
                        OpenNode = n,
                        Kind = (n.OpCode == OpCodes.IF_OPEN) ? ScopeKind.If : ScopeKind.Loop,
                        Parent = stack.Count > 0 ? stack.Peek() : (int?)null,
                    };
                    scopes.Add(s);
                    stack.Push(s.Id);
                }
                else if (FastOpsetResolver.IsCloseOpCode(n.OpCode))
                {
                    if (stack.Count == 0)
                        throw new InvalidOperationException(
                            $"Unmatched close node at index {i} ({n.OpCode}).");
                    var topId = stack.Pop();
                    var top = scopes[topId];
                    if (top.OpenNode.Key != n.GraphOpenNodeKey)
                        throw new InvalidOperationException(
                            $"Non-nesting open/close at index {i}: scopes overlap.");
                    top.CloseIdx = i;
                    top.CloseNode = n;
                }
            }
            if (stack.Count > 0)
                throw new InvalidOperationException("Unmatched open node — scope is missing its close.");
            return scopes;
        }

        // -- Step 2 -----------------------------------------------------------

        private static (Dictionary<int, HashSet<FastNodeKey>> fwd,
                        Dictionary<int, HashSet<FastNodeKey>> bwd)
            BuildReachSets(InternalComputationGraph graph, List<Scope> scopes)
        {
            var producer = new Dictionary<FastTensorKey, FastNode>();
            foreach (var n in graph.Nodes)
                foreach (var grp in n.FullOutputs)
                    foreach (var key in grp.Value)
                        if (key is FastTensorKey tk && !tk.IsEmpty)
                            producer[tk] = n;

            var consumers = new Dictionary<FastNodeKey, List<FastNode>>();
            foreach (var n in graph.Nodes)
                foreach (var grp in n.FullInputs)
                    foreach (var key in grp.Value)
                        if (key is FastTensorKey tk && !tk.IsEmpty &&
                            producer.TryGetValue(tk, out var p))
                        {
                            if (!consumers.TryGetValue(p.Key, out var list))
                                consumers[p.Key] = list = new List<FastNode>();
                            list.Add(n);
                        }

            var fwd = new Dictionary<int, HashSet<FastNodeKey>>();
            var bwd = new Dictionary<int, HashSet<FastNodeKey>>();
            foreach (var s in scopes)
            {
                // Forward and backward BFS for scope S must each stop at any
                // boundary node that takes the walk OUTSIDE S. That set is the
                // OPEN and CLOSE of every scope that is NOT a descendant of S
                // (S itself, ancestors of S, and unrelated scopes). Descendant
                // scope boundaries are inside S, so we walk through them. The
                // walk's start node (S.OPEN for fwd, S.CLOSE for bwd) is
                // dropped from its respective stop set.
                var stops = BuildScopeStops(s, scopes);
                var fwdStops = new HashSet<FastNodeKey>(stops); fwdStops.Remove(s.OpenNode.Key);
                var bwdStops = new HashSet<FastNodeKey>(stops); bwdStops.Remove(s.CloseNode.Key);
                fwd[s.Id] = ForwardBfs(s.OpenNode, fwdStops, consumers);
                bwd[s.Id] = BackwardBfs(s.CloseNode, bwdStops, producer);
            }
            return (fwd, bwd);
        }

        private static HashSet<FastNodeKey> BuildScopeStops(Scope s, List<Scope> scopes)
        {
            var stops = new HashSet<FastNodeKey>();
            foreach (var x in scopes)
            {
                bool xNestedInS = x.OpenIdx > s.OpenIdx && x.CloseIdx < s.CloseIdx;
                if (xNestedInS) continue;
                stops.Add(x.OpenNode.Key);
                stops.Add(x.CloseNode.Key);
            }
            return stops;
        }

        // Forward BFS from <paramref name="start"/> following producer→consumer
        // edges. Nodes in <paramref name="boundaries"/> are reachable but not
        // expanded — anything strictly downstream of those boundaries has
        // exited the scope and is not "inside" it.
        private static HashSet<FastNodeKey> ForwardBfs(
            FastNode start, HashSet<FastNodeKey> boundaries,
            Dictionary<FastNodeKey, List<FastNode>> consumers)
        {
            var visited = new HashSet<FastNodeKey> { start.Key };
            var queue = new Queue<FastNode>();
            queue.Enqueue(start);
            while (queue.Count > 0)
            {
                var n = queue.Dequeue();
                if (boundaries.Contains(n.Key)) continue;
                if (consumers.TryGetValue(n.Key, out var cs))
                    foreach (var c in cs)
                        if (visited.Add(c.Key)) queue.Enqueue(c);
            }
            return visited;
        }

        // Backward BFS from <paramref name="start"/> following consumer→producer
        // edges. <paramref name="boundary"/> is treated as a source: included in
        // the result if reached, but its inputs are not expanded. For the scope
        // configurator this is the matching OPEN node — anything strictly
        // upstream of OPEN is outside the scope.
        private static HashSet<FastNodeKey> BackwardBfs(
            FastNode start, HashSet<FastNodeKey> boundaries,
            Dictionary<FastTensorKey, FastNode> producer)
        {
            var visited = new HashSet<FastNodeKey> { start.Key };
            var queue = new Queue<FastNode>();
            queue.Enqueue(start);
            while (queue.Count > 0)
            {
                var n = queue.Dequeue();
                if (boundaries.Contains(n.Key)) continue;
                foreach (var grp in n.FullInputs)
                    foreach (var key in grp.Value)
                        if (key is FastTensorKey tk && !tk.IsEmpty &&
                            producer.TryGetValue(tk, out var p) &&
                            visited.Add(p.Key))
                            queue.Enqueue(p);
            }
            return visited;
        }

        // -- Step 2b ----------------------------------------------------------
        // For each IF scope, a separate backward-reach set per branch. A node is
        // in thenReach[S] iff its data flows into one of close_S's then-branch
        // input slots; elseReach[S] similarly. LOOP scopes get empty sets here.
        private static (Dictionary<int, HashSet<FastNodeKey>> thenReach,
                        Dictionary<int, HashSet<FastNodeKey>> elseReach)
            BuildIfBranchReachSets(InternalComputationGraph graph, List<Scope> scopes)
        {
            var producer = new Dictionary<FastTensorKey, FastNode>();
            foreach (var n in graph.Nodes)
                foreach (var grp in n.FullOutputs)
                    foreach (var key in grp.Value)
                        if (key is FastTensorKey tk && !tk.IsEmpty)
                            producer[tk] = n;

            var thenReach = new Dictionary<int, HashSet<FastNodeKey>>();
            var elseReach = new Dictionary<int, HashSet<FastNodeKey>>();
            foreach (var s in scopes)
            {
                if (s.Kind != ScopeKind.If)
                {
                    thenReach[s.Id] = new HashSet<FastNodeKey>();
                    elseReach[s.Id] = new HashSet<FastNodeKey>();
                    continue;
                }
                var bwdStops = BuildScopeStops(s, scopes);
                bwdStops.Remove(s.CloseNode.Key);
                thenReach[s.Id] = BackwardBfsFromBranchInputs(
                    s.CloseNode, bwdStops, OnnxOpAttributeNames.AttrThenBranch, producer);
                elseReach[s.Id] = BackwardBfsFromBranchInputs(
                    s.CloseNode, bwdStops, OnnxOpAttributeNames.AttrElseBranch, producer);
            }
            return (thenReach, elseReach);
        }

        // Backward BFS from the branch-keyed inputs of an IF_CLOSE; stops at the
        // same OPEN-node boundary set as <see cref="BackwardBfs"/> — the IF's
        // own OPEN, ancestors' OPENs, and unrelated scopes' OPENs. Walking past
        // any of those leads to nodes outside the IF scope.
        private static HashSet<FastNodeKey> BackwardBfsFromBranchInputs(
            FastNode ifClose, HashSet<FastNodeKey> boundaries, string branchAttr,
            Dictionary<FastTensorKey, FastNode> producer)
        {
            var visited = new HashSet<FastNodeKey>();
            var queue = new Queue<FastNode>();
            if (ifClose.FullInputs.TryGetValue(branchAttr, out var inputs))
            {
                foreach (var key in inputs)
                {
                    if (key is FastTensorKey tk && !tk.IsEmpty &&
                        producer.TryGetValue(tk, out var p) &&
                        visited.Add(p.Key))
                        queue.Enqueue(p);
                }
            }
            while (queue.Count > 0)
            {
                var n = queue.Dequeue();
                if (boundaries.Contains(n.Key)) continue;
                foreach (var grp in n.FullInputs)
                    foreach (var key in grp.Value)
                        if (key is FastTensorKey tk && !tk.IsEmpty &&
                            producer.TryGetValue(tk, out var p) &&
                            visited.Add(p.Key))
                            queue.Enqueue(p);
            }
            return visited;
        }

        // -- Step 3 -----------------------------------------------------------

        // inAByStructure[a, b] = scope b is structurally nested inside scope a.
        private static bool[,] BuildContainmentTable(List<Scope> scopes)
        {
            int n = scopes.Count;
            var t = new bool[n, n];
            for (int a = 0; a < n; a++)
                for (int b = 0; b < n; b++)
                    t[a, b] = (scopes[b].OpenIdx > scopes[a].OpenIdx &&
                               scopes[b].CloseIdx < scopes[a].CloseIdx);
            return t;
        }

        // -- Step 4 -----------------------------------------------------------

        private static Dictionary<FastNodeKey, NodeStatus[]> ClassifyNodes(
            InternalComputationGraph graph,
            List<Scope> scopes,
            Dictionary<int, HashSet<FastNodeKey>> fwd,
            Dictionary<int, HashSet<FastNodeKey>> bwd,
            Dictionary<int, HashSet<FastNodeKey>> thenReach,
            Dictionary<int, HashSet<FastNodeKey>> elseReach,
            bool[,] inAByStructure)
        {
            var status = new Dictionary<FastNodeKey, NodeStatus[]>();
            foreach (var node in graph.Nodes)
            {
                if (IsScopeBoundary(node)) continue;

                var arr = new NodeStatus[scopes.Count];
                for (int s = 0; s < scopes.Count; s++)
                {
                    var sc = scopes[s];
                    bool reachesFromOpen = fwd[s].Contains(node.Key) && node.Key != sc.OpenNode.Key;
                    bool reachesClose = bwd[s].Contains(node.Key);

                    if (reachesFromOpen)
                    {
                        // Node consumes (transitively) something OPEN_S produces.
                        // It must be inside S — its inputs only exist there. If
                        // it doesn't feed CLOSE_S the chain is dead inside S,
                        // which is fine: ONNX subgraphs accept dead nodes as
                        // long as the declared subgraph outputs are produced.
                        arr[s] = NodeStatus.MustIn;
                        continue;
                    }
                    if (!reachesClose)
                    {
                        arr[s] = NodeStatus.MustOut;
                        continue;
                    }

                    // IF-branch rule: a node feeding both then- and else-branch chains
                    // of the same IF must sit outside the IF.
                    if (sc.Kind == ScopeKind.If &&
                        thenReach[s].Contains(node.Key) &&
                        elseReach[s].Contains(node.Key))
                    {
                        arr[s] = NodeStatus.MustOut;
                        continue;
                    }

                    // Constraint 3: forced out if N reaches another scope's close
                    // where that other scope is unrelated to S in the scope tree
                    // (neither nested in S nor an ancestor of S). Ancestor-scope
                    // closes are the natural exit path and don't trigger this.
                    bool forcedOut = false;
                    for (int sPrime = 0; sPrime < scopes.Count; sPrime++)
                    {
                        if (sPrime == s) continue;
                        if (bwd[sPrime].Contains(node.Key) &&
                            !inAByStructure[s, sPrime] &&
                            !inAByStructure[sPrime, s])
                        {
                            forcedOut = true;
                            break;
                        }
                    }
                    arr[s] = forcedOut ? NodeStatus.MustOut : NodeStatus.Free;
                }
                status[node.Key] = arr;
            }
            return status;
        }

        private static bool IsScopeBoundary(FastNode n) =>
            FastOpsetResolver.IsOpenOpCode(n.OpCode) || FastOpsetResolver.IsCloseOpCode(n.OpCode);

        // -- Step 5 -----------------------------------------------------------

        /// <summary>
        /// Turns the per-scope statuses into a per-scope in/out decision. A node goes inside
        /// every scope that does not force it out — the widest placement its data flow allows,
        /// and the only one an emitter can use: a node left outside a scope it could have been
        /// in is a node that runs when that scope does not.
        ///
        /// <para>Nesting consistency then needs one direction only. A node inside S must also be
        /// inside every ancestor of S, so a MustOut anywhere up the chain takes it out of S too,
        /// however deep the nesting. The converse needs no pass: an ancestor that does not force
        /// the node out is already in by the rule above. Only a scope that genuinely requires the
        /// node (MustIn) can contradict an enclosing MustOut, and that is a graph leaking a body
        /// value rather than a placement to settle — settling it by throwing is what refused
        /// nestings that were merely deeper than two (Shorokoo/Shorokoo#270).</para>
        /// </summary>
        private static Dictionary<FastNodeKey, bool[]> DecidePlacement(
            InternalComputationGraph graph,
            List<Scope> scopes,
            Dictionary<FastNodeKey, NodeStatus[]> status)
        {
            var decision = new Dictionary<FastNodeKey, bool[]>();
            foreach (var node in graph.Nodes)
            {
                if (IsScopeBoundary(node)) continue;

                var st = status[node.Key];
                var dec = new bool[scopes.Count];
                for (int s = 0; s < scopes.Count; s++)
                    dec[s] = st[s] != NodeStatus.MustOut;

                for (int s = 0; s < scopes.Count; s++)
                {
                    if (!dec[s] || !AnyAncestorForcesOut(scopes, st, s)) continue;
                    if (st[s] == NodeStatus.MustIn)
                        throw new InvalidOperationException(
                            $"Node {node.OpCode} (Key={node.Key}) is required inside " +
                            $"{scopes[s].Kind} scope #{s} but forced out of an enclosing scope.");
                    dec[s] = false;
                }

                decision[node.Key] = dec;
            }
            return decision;
        }

        private static bool AnyAncestorForcesOut(List<Scope> scopes, NodeStatus[] st, int s)
        {
            int? p = scopes[s].Parent;
            while (p is int pid)
            {
                if (st[pid] == NodeStatus.MustOut) return true;
                p = scopes[pid].Parent;
            }
            return false;
        }

        // -- Step 6 -----------------------------------------------------------

        private static List<FastNode> Reorder(
            InternalComputationGraph graph, List<Scope> scopes,
            Dictionary<FastNodeKey, bool[]> decision)
        {
            var result = new List<FastNode>(graph.Nodes.Count);
            // Active scopes whose open has been emitted but close has not, oldest-first.
            var active = new List<(int id, int posInResult)>();

            foreach (var n in graph.Nodes)
            {
                if (FastOpsetResolver.IsOpenOpCode(n.OpCode))
                {
                    int scopeId = ScopeIdForBoundary(n, scopes, isOpen: true);
                    active.Add((scopeId, result.Count));
                    result.Add(n);
                    continue;
                }
                if (FastOpsetResolver.IsCloseOpCode(n.OpCode))
                {
                    int scopeId = ScopeIdForBoundary(n, scopes, isOpen: false);
                    result.Add(n);
                    for (int i = active.Count - 1; i >= 0; i--)
                        if (active[i].id == scopeId) active.RemoveAt(i);
                    continue;
                }

                if (!decision.TryGetValue(n.Key, out var dec))
                {
                    result.Add(n);
                    continue;
                }

                int hoistTo = -1;
                foreach (var (id, pos) in active)
                {
                    if (!dec[id]) { hoistTo = pos; break; }
                }
                if (hoistTo < 0)
                {
                    result.Add(n);
                }
                else
                {
                    result.Insert(hoistTo, n);
                    for (int i = 0; i < active.Count; i++)
                        if (active[i].posInResult >= hoistTo)
                            active[i] = (active[i].id, active[i].posInResult + 1);
                }
            }
            return result;
        }

        private static int ScopeIdForBoundary(FastNode n, List<Scope> scopes, bool isOpen)
        {
            for (int i = 0; i < scopes.Count; i++)
            {
                var boundary = isOpen ? scopes[i].OpenNode : scopes[i].CloseNode;
                if (boundary.Key == n.Key) return i;
            }
            throw new InvalidOperationException(
                $"Boundary node {n.OpCode} not associated with any scope.");
        }

        // -- Step 7 — Pass 2: branch ordering inside IF scopes ---------------
        // Within each IF scope's positional range, reorder the body so that
        // everything belonging to the then-branch precedes everything belonging
        // to the else-branch. Topological order within each branch is preserved
        // by stable partitioning.
        //
        // The unit of movement is a whole nested OPEN…CLOSE band, never its
        // individual nodes. Reordering node by node tore a nested scope apart,
        // because a nested OPEN produces no tensor and so is reached from
        // neither branch: it sorted into the leftover bucket and landed after
        // the body and CLOSE it opens (Shorokoo/Shorokoo#270).
        private static List<FastNode> ReorderIfBranches(
            InternalComputationGraph graph, List<Scope> scopes,
            Dictionary<int, HashSet<FastNodeKey>> thenReach,
            Dictionary<int, HashSet<FastNodeKey>> elseReach)
        {
            // Process IF scopes innermost-first so each pass operates on already-
            // settled inner ranges.
            var ifScopeIds = new List<int>();
            foreach (var s in scopes) if (s.Kind == ScopeKind.If) ifScopeIds.Add(s.Id);
            ifScopeIds.Sort((a, b) =>
                ScopeDepth(scopes, b).CompareTo(ScopeDepth(scopes, a)));  // deepest first

            var nodes = new List<FastNode>(graph.Nodes);
            foreach (var sId in ifScopeIds)
            {
                int openIdx = IndexOfNodeKey(nodes, scopes[sId].OpenNode.Key);
                int closeIdx = IndexOfNodeKey(nodes, scopes[sId].CloseNode.Key);
                if (openIdx < 0 || closeIdx < 0 || closeIdx <= openIdx + 1) continue;

                var thenSet = thenReach[sId];
                var elseSet = elseReach[sId];

                var thenBody = new List<FastNode>();
                var elseBody = new List<FastNode>();
                var neither = new List<FastNode>();

                foreach (var band in BandsWithin(nodes, openIdx, closeIdx))
                {
                    bool inThen = false, inElse = false;
                    foreach (var n in band)
                    {
                        if (thenSet.Contains(n.Key)) inThen = true;
                        if (elseSet.Contains(n.Key)) inElse = true;
                    }
                    if (inThen && !inElse) thenBody.AddRange(band);
                    else if (inElse && !inThen) elseBody.AddRange(band);
                    else neither.AddRange(band);
                }

                nodes.RemoveRange(openIdx + 1, closeIdx - openIdx - 1);
                int insertAt = openIdx + 1;
                foreach (var n in thenBody) nodes.Insert(insertAt++, n);
                foreach (var n in elseBody) nodes.Insert(insertAt++, n);
                foreach (var n in neither) nodes.Insert(insertAt++, n);
            }
            return nodes;
        }

        /// <summary>
        /// The direct contents of <c>(openIdx, closeIdx)</c>, each a single node or a whole
        /// nested OPEN…CLOSE band, in order.
        /// </summary>
        private static List<List<FastNode>> BandsWithin(List<FastNode> nodes, int openIdx, int closeIdx)
        {
            var bands = new List<List<FastNode>>();
            int i = openIdx + 1;
            while (i < closeIdx)
            {
                if (!FastOpsetResolver.IsOpenOpCode(nodes[i].OpCode))
                {
                    bands.Add([nodes[i]]);
                    i++;
                    continue;
                }

                int depth = 0;
                int j = i;
                for (; j < closeIdx; j++)
                {
                    if (FastOpsetResolver.IsOpenOpCode(nodes[j].OpCode)) depth++;
                    else if (FastOpsetResolver.IsCloseOpCode(nodes[j].OpCode) && --depth == 0) break;
                }
                bands.Add(nodes.GetRange(i, Math.Min(j, closeIdx - 1) - i + 1));
                i = j + 1;
            }
            return bands;
        }

        private static int ScopeDepth(List<Scope> scopes, int id)
        {
            int d = 0;
            int? p = scopes[id].Parent;
            while (p is int pid) { d++; p = scopes[pid].Parent; }
            return d;
        }

        private static int IndexOfNodeKey(List<FastNode> nodes, FastNodeKey key)
        {
            for (int i = 0; i < nodes.Count; i++)
                if (nodes[i].Key == key) return i;
            return -1;
        }
    }
}

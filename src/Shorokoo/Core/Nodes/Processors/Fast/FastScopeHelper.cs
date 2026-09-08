using Shorokoo.Graph;
using Shorokoo.Core.Graph;
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
    /// Primitives for expanding and shrinking loop scopes in a
    /// <see cref="InternalComputationGraph"/>. Scope membership in the Fast pipeline is
    /// positional: every node whose index in <see cref="InternalComputationGraph.Nodes"/>
    /// falls between a <c>LOOP_OPEN</c> and its paired <c>LOOP_CLOSE</c> is treated as
    /// a body node by <see cref="Shorokoo.Core.Inference.QuickExecutionEngine"/>'s
    /// linear loop-back model. Shrinking a scope moves nodes that positionally fall
    /// inside it but do not actually depend on the loop's body outputs to just before
    /// the <c>LOOP_OPEN</c>, reducing the re-execution range on loop-back and
    /// eliminating spurious per-iteration <c>History</c> accumulation on QEE-tracked
    /// tensors.
    ///
    /// The data-flow notion used here mirrors the legacy CG-side hoisting
    /// primitive. A tensor is
    /// "loop-dependent" if it is produced by a <c>LOOP_OPEN</c> or by any node that
    /// transitively consumes a loop-dependent tensor. A node is "loop-invariant" at
    /// a given scope if none of its inputs is loop-dependent.
    /// </summary>
    internal static class FastScopeHelper
    {
        /// <summary>
        /// Describes a single <c>LOOP_OPEN</c> / <c>LOOP_CLOSE</c> pair and its
        /// positions in <see cref="InternalComputationGraph.Nodes"/>. Returned from
        /// <see cref="TryResolveLoopScope"/> so callers can inspect a scope without
        /// themselves knowing the linear layout convention.
        /// </summary>
        public readonly struct FastLoopScope
        {
            public FastNode OpenNode { get; }
            public FastNode CloseNode { get; }
            public int OpenIdx { get; }
            public int CloseIdx { get; }

            public FastLoopScope(FastNode openNode, FastNode closeNode, int openIdx, int closeIdx)
            {
                OpenNode = openNode;
                CloseNode = closeNode;
                OpenIdx = openIdx;
                CloseIdx = closeIdx;
            }
        }

        /// <summary>
        /// Returns every <see cref="FastTensorKey"/> whose value depends on some
        /// <c>LOOP_OPEN</c>'s body outputs — directly, or transitively through any chain
        /// of nodes. Output keys produced by a <c>LOOP_OPEN</c> are included, as are those
        /// produced by any close node.
        ///
        /// <para>The close nodes are included unconditionally because this set is what
        /// <see cref="ShrinkAllScopes"/> uses to decide what may leave a scope, and a close
        /// node never leaves one. Its outputs are produced inside its scope, so a consumer of
        /// one cannot be hoisted out past it — even when the close's own inputs are all
        /// loop-invariant, which is exactly what a nested loop carrying an unvarying value
        /// looks like. Judging such a consumer by its inputs alone moved it above the nested
        /// loop that produces what it reads.</para>
        /// </summary>
        public static HashSet<FastTensorKey> BuildLoopDependentTensors(InternalComputationGraph graph)
        {
            var loopDependent = new HashSet<FastTensorKey>();
            foreach (var node in graph.Nodes)
            {
                if (node.OpCode == OpCodes.LOOP_OPEN
                    || node.OpCode == OpCodes.LOOP_CLOSE
                    || node.OpCode == OpCodes.IF_CLOSE)
                {
                    AddAllOutputs(node, loopDependent);
                    continue;
                }

                if (HasLoopDependentInput(node, loopDependent))
                    AddAllOutputs(node, loopDependent);
            }
            return loopDependent;
        }

        /// <summary>
        /// Returns every <see cref="FastTensorKey"/> whose value a consumer <em>may</em> read
        /// differently on each pass of a loop it sits inside — the keys reached from some loop's
        /// iteration index, trip count or carries. It is a reachability answer, not a value one: a
        /// key derived from an iteration index is returned even if the arithmetic happens to give
        /// the same value every time, since that is not knowable here.
        ///
        /// <para>This is narrower than <see cref="BuildLoopDependentTensors"/>, which marks
        /// everything descended from a <c>LOOP_OPEN</c> and so keeps propagating past the
        /// <c>LOOP_CLOSE</c> into the rest of the graph. That is what <see cref="ShrinkAllScopes"/>
        /// wants (it only ever asks about nodes inside a scope), but a loop's <em>result</em> is
        /// computed once and does not vary, so a caller asking "does this differ per iteration?"
        /// about a key anywhere in the graph needs this one instead. A <c>LOOP_CLOSE</c> therefore
        /// sheds the variation of the loop it closes, keeping only an enclosing loop's.</para>
        ///
        /// <para>A close node also folds in its paired open's inputs, resolved through
        /// <see cref="FastNode.GraphOpenNodeKey"/> — dependence that arrives that way is invisible to
        /// a walk over data inputs. For an <c>IF</c> that pairing carries the whole of it: the
        /// condition is an input of the <c>IF_OPEN</c>, which has no outputs, so a branch value
        /// selected by a per-iteration condition varies per iteration even when both branches are
        /// constants the loop never touches.</para>
        ///
        /// <para>Two things make the loop shedding subtle, and both are handled per slot rather than
        /// per node. First, an enclosing loop reaches a nested loop through the inner
        /// <c>LOOP_OPEN</c>'s inputs — its trip count and carry initializers — and not through the
        /// <c>LOOP_CLOSE</c>'s, which is why the close folds in its paired open's inputs as well as
        /// its own (the initializers are also what a zero-trip loop returns outright). Second, the
        /// carries of one loop are independent: a carry that varies with an enclosing loop must not
        /// mark its siblings, so the slot correspondence
        /// <c>open.Inputs[2 + i]</c> → <c>open.Outputs[2 + i]</c> → <c>close.Inputs[1 + i]</c> →
        /// <c>close.Outputs[i]</c> is followed for each carry, with the trip count and the
        /// continue-condition folded into every one of them since they govern the whole loop. A
        /// node whose slots do not match that layout falls back to one depth for all its outputs.</para>
        /// </summary>
        public static HashSet<FastTensorKey> BuildPerIterationTensors(InternalComputationGraph graph)
        {
            // Key → the shallowest loop depth across whose iterations the key may vary (scope
            // membership is positional, so depth is just the open/close nesting count).
            var varyDepth = new Dictionary<FastTensorKey, int>();
            var openNodes = new Stack<FastNode>();
            var openByKey = new Dictionary<FastNodeKey, FastNode>();

            foreach (var node in graph.Nodes)
            {
                if (node.OpCode == OpCodes.LOOP_OPEN)
                {
                    openNodes.Push(node);
                    openByKey[node.Key] = node;
                    RecordLoopOpen(node, openNodes.Count, varyDepth);
                }
                else if (node.OpCode == OpCodes.LOOP_CLOSE && openNodes.Count > 0)
                {
                    RecordLoopClose(node, openNodes.Pop(), openNodes.Count + 1, varyDepth);
                }
                else
                {
                    if (node.OpCode == OpCodes.IF_OPEN) openByKey[node.Key] = node;

                    // A close node's value depends on its paired open's inputs as well as its own.
                    // For an IF that is the whole of the dependence that matters here: the condition
                    // lives on the IF_OPEN, which has no outputs, so a branch value picked by a
                    // per-iteration condition is per-iteration even when both branches are constant.
                    var pairedOpenInputs = node.GraphOpenNodeKey is FastNodeKey pairedKey
                                           && !pairedKey.IsEmpty
                                           && openByKey.TryGetValue(pairedKey, out var pairedOpen)
                        ? pairedOpen.Inputs
                        : [];

                    RecordAllOutputs(node, Shallowest(
                        MinVaryDepth(node.Inputs, varyDepth, shallowerThan: int.MaxValue),
                        MinVaryDepth(pairedOpenInputs, varyDepth, shallowerThan: int.MaxValue)), varyDepth);
                }
            }

            return [.. varyDepth.Keys];
        }

        /// <summary>
        /// Depths for a <c>LOOP_OPEN</c>'s outputs <c>[iterationIndex, vestigialTrue, ...carries]</c>
        /// at nesting <paramref name="depth"/>: everything the body sees varies with this loop, and
        /// on top of that with any enclosing loop reaching it through the trip count, the initial
        /// condition, or that carry's own initializer.
        /// </summary>
        private static void RecordLoopOpen(FastNode open, int depth, Dictionary<FastTensorKey, int> varyDepth)
        {
            var inputs = open.Inputs;
            var outputs = open.Outputs;
            int nCarries = Math.Max(0, inputs.Count - 2);
            // The trip count and initial condition govern every output.
            int governing = MinVaryDepth(inputs.Take(Math.Min(2, inputs.Count)), varyDepth, depth);

            if (outputs.Count != 2 + nCarries)
            {
                RecordAllOutputs(open, Shallowest(governing, depth), varyDepth);
                return;
            }

            RecordDepth(outputs[0], Shallowest(governing, depth), varyDepth);
            RecordDepth(outputs[1], Shallowest(governing, depth), varyDepth);
            for (int i = 0; i < nCarries; i++)
                RecordDepth(outputs[2 + i],
                    Shallowest(Shallowest(governing, VaryDepthOf(inputs[2 + i], varyDepth, depth)), depth),
                    varyDepth);
        }

        /// <summary>
        /// Depths for a <c>LOOP_CLOSE</c>'s outputs <c>[...finalCarries, ...scanOutputs]</c>, shedding
        /// the depth of the loop it closes. Each final carry keeps only what an enclosing loop
        /// contributes through the matching end-of-body value, that carry's initializer on the paired
        /// <paramref name="open"/>, or the trip count / conditions that govern the loop as a whole.
        /// </summary>
        private static void RecordLoopClose(
            FastNode close, FastNode open, int depth, Dictionary<FastTensorKey, int> varyDepth)
        {
            var closeInputs = close.Inputs;
            var closeOutputs = close.Outputs;
            var openInputs = open.Inputs;
            int nCarries = Math.Max(0, openInputs.Count - 2);
            int nScans = closeOutputs.Count - nCarries;

            // Anything that changes how many iterations run changes every output.
            int governing = Shallowest(
                MinVaryDepth(openInputs.Take(Math.Min(2, openInputs.Count)), varyDepth, depth),
                VaryDepthOf(closeInputs.Count > 0 ? closeInputs[0] : null, varyDepth, depth));

            bool slotsMatch = nScans >= 0
                && closeInputs.Count == 1 + nCarries + nScans
                && close.GraphOpenNodeKey is FastNodeKey openKey && openKey == open.Key;
            if (!slotsMatch)
            {
                RecordAllOutputs(close, Shallowest(
                    Shallowest(governing, MinVaryDepth(closeInputs, varyDepth, depth)),
                    MinVaryDepth(openInputs, varyDepth, depth)), varyDepth);
                return;
            }

            for (int i = 0; i < nCarries; i++)
                RecordDepth(closeOutputs[i], Shallowest(
                    Shallowest(governing, VaryDepthOf(closeInputs[1 + i], varyDepth, depth)),
                    VaryDepthOf(openInputs[2 + i], varyDepth, depth)), varyDepth);

            for (int j = 0; j < nScans; j++)
                RecordDepth(closeOutputs[nCarries + j], Shallowest(
                    governing, VaryDepthOf(closeInputs[1 + nCarries + j], varyDepth, depth)), varyDepth);
        }

        /// <summary>The shallower of two depths, where 0 means "does not vary".</summary>
        private static int Shallowest(int a, int b) => a == 0 ? b : b == 0 ? a : Math.Min(a, b);

        /// <summary>The depth <paramref name="key"/> varies at, counting only depths shallower than
        /// <paramref name="shallowerThan"/>; 0 when it does not vary or is missing.</summary>
        private static int VaryDepthOf(
            FastTensorKey? key, Dictionary<FastTensorKey, int> varyDepth, int shallowerThan)
            => key is FastTensorKey tk && !tk.IsEmpty
               && varyDepth.TryGetValue(tk, out var d) && d < shallowerThan ? d : 0;

        /// <summary>Shallowest depth any of <paramref name="keys"/> varies at, counting only depths
        /// shallower than <paramref name="shallowerThan"/>; 0 when none varies.</summary>
        private static int MinVaryDepth(
            IEnumerable<FastTensorKey?> keys, Dictionary<FastTensorKey, int> varyDepth, int shallowerThan)
        {
            int min = 0;
            foreach (var key in keys)
                min = Shallowest(min, VaryDepthOf(key, varyDepth, shallowerThan));
            return min;
        }

        /// <summary>Records <paramref name="depth"/> for every output of <paramref name="node"/>,
        /// or nothing when it is 0 (the node's outputs do not vary).</summary>
        private static void RecordAllOutputs(FastNode node, int depth, Dictionary<FastTensorKey, int> varyDepth)
        {
            if (depth == 0) return;
            foreach (var kvp in node.FullOutputs)
                foreach (var key in kvp.Value)
                    RecordDepth(key, depth, varyDepth);
        }

        /// <summary>Records <paramref name="depth"/> for one key, keeping the shallowest seen.</summary>
        private static void RecordDepth(FastTensorKey? key, int depth, Dictionary<FastTensorKey, int> varyDepth)
        {
            if (depth == 0 || key is not FastTensorKey tk || tk.IsEmpty) return;
            varyDepth[tk] = varyDepth.TryGetValue(tk, out var existing) ? Math.Min(existing, depth) : depth;
        }

        /// <summary>
        /// Resolves a <c>LOOP_CLOSE</c> node to its paired <c>LOOP_OPEN</c> and returns
        /// their positions in <see cref="InternalComputationGraph.Nodes"/>. Returns null if
        /// <paramref name="closeNode"/> is not a <c>LOOP_CLOSE</c>, if its
        /// <see cref="FastNode.GraphOpenNodeKey"/> is missing, or if the paired
        /// <c>LOOP_OPEN</c> is not present in the graph.
        /// </summary>
        public static FastLoopScope? TryResolveLoopScope(InternalComputationGraph graph, FastNode closeNode)
        {
            if (closeNode.OpCode != OpCodes.LOOP_CLOSE) return null;
            if (closeNode.GraphOpenNodeKey is not FastNodeKey openKey || openKey.IsEmpty) return null;

            int openIdx = -1;
            int closeIdx = -1;
            FastNode? openNode = null;
            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                var n = graph.Nodes[i];
                if (n.Key == openKey) { openNode = n; openIdx = i; }
                else if (ReferenceEquals(n, closeNode)) { closeIdx = i; }
                if (openIdx >= 0 && closeIdx >= 0) break;
            }
            if (openNode is null || openIdx < 0 || closeIdx < 0) return null;
            return new FastLoopScope(openNode, closeNode, openIdx, closeIdx);
        }

        /// <summary>
        /// Shrinks every loop scope in <paramref name="graph"/>: any node whose
        /// positional index falls between a <c>LOOP_OPEN</c> and its paired
        /// <c>LOOP_CLOSE</c> but whose inputs include no loop-dependent tensor is moved
        /// to just before the outermost currently-active <c>LOOP_OPEN</c>. The relative
        /// order of nodes is otherwise preserved. Mutates
        /// <see cref="InternalComputationGraph.Nodes"/> in place.
        ///
        /// <para>Scope boundaries never move, whatever their inputs look like: a close node's
        /// value belongs to the scope that produced it, and hoisting an <c>IF_OPEN</c> out of an
        /// enclosing loop would strand its <c>IF_CLOSE</c> inside — which is why an if whose
        /// condition is loop-invariant used to leave the graph in an order the pipeline
        /// rejects.</para>
        ///
        /// <para>Two further things stay put, both because loop-invariant by dataflow does not
        /// mean the same value every pass. A node inside an <c>IF</c> body runs only when its
        /// branch is taken, so lifting one out runs it unconditionally — the lazy
        /// <c>IfElse(cond, Func, Func)</c> overload exists precisely so a branch may hold an
        /// operation that is invalid off-branch. And a draw has no inputs to be dependent on,
        /// yet a second execution of one is a second sample: hoisting it out gives every
        /// iteration the one value, which is the wrong answer #262 describes reached by another
        /// route. Both hoists leave the node order valid, so only these refusals catch them.</para>
        ///
        /// This is the Fast-pipeline equivalent of the legacy CG-side
        /// hoisting primitive that ran as part of ComputationGraph
        /// construction. QEE consumers benefit because
        /// hoisted nodes no longer appear inside the positional loop body and so:
        ///   - they are not re-executed on loop-back, and
        ///   - their results are not tagged with per-iteration <c>IterationIndices</c> /
        ///     <c>History</c> (which simplifies <c>IntData</c> availability checks for
        ///     downstream consumers).
        /// </summary>
        public static void ShrinkAllScopes(InternalComputationGraph graph)
        {
            // Asserted on entry as well as exit: this pass shrinks scopes, it does not fix
            // invalid ones, so a violation introduced upstream must not read as one introduced
            // here — the same contract FastScopeConfigurator.Configure states.
            System.Diagnostics.Debug.Assert(graph.IsLinearOrderValid(), "graph.IsLinearOrderValid()");

            var loopDependent = BuildLoopDependentTensors(graph);

            var result = new List<FastNode>(graph.Nodes.Count);
            // FastNodeKey of each active (not-yet-closed) LOOP_OPEN → its position in result.
            var openPositions = new Dictionary<FastNodeKey, int>();
            // Position of each active non-loop scope's OPEN, innermost last. A hoist may not
            // cross one, so only a loop opened after the innermost of these is a target.
            var barrierPositions = new List<int>();

            foreach (var node in graph.Nodes)
            {
                if (node.OpCode == OpCodes.LOOP_OPEN)
                {
                    openPositions[node.Key] = result.Count;
                    result.Add(node);
                    continue;
                }

                // A scope boundary never moves: hoisting an IF_OPEN out of an enclosing loop
                // would strand its IF_CLOSE inside, and a close node's value belongs to the
                // scope that produced it. Classified through FastOpsetResolver so that
                // SEQUENCE_MAP, which the linear-order validator scopes too, is covered.
                if (Shorokoo.Core.Factory.FastOpsetResolver.IsOpenOpCode(node.OpCode))
                {
                    barrierPositions.Add(result.Count);
                    result.Add(node);
                    continue;
                }

                if (Shorokoo.Core.Factory.FastOpsetResolver.IsCloseOpCode(node.OpCode))
                {
                    if (node.OpCode == OpCodes.LOOP_CLOSE &&
                        node.GraphOpenNodeKey is FastNodeKey openKey && !openKey.IsEmpty)
                        openPositions.Remove(openKey);
                    else if (barrierPositions.Count > 0)
                        barrierPositions.RemoveAt(barrierPositions.Count - 1);

                    result.Add(node);
                    continue;
                }

                if (Shorokoo.Core.AutoDiffCheckpointing.Rematerializer.IsDeterministicOpCode(node.OpCode) &&
                    !HasLoopDependentInput(node, loopDependent) &&
                    TryFindHoistPosition(openPositions, barrierPositions, out int hoistPos))
                {
                    result.Insert(hoistPos, node);
                    foreach (var k in openPositions.Keys.ToList())
                        if (openPositions[k] >= hoistPos)
                            openPositions[k] = openPositions[k] + 1;
                    for (int i = 0; i < barrierPositions.Count; i++)
                        if (barrierPositions[i] >= hoistPos)
                            barrierPositions[i] = barrierPositions[i] + 1;
                    continue;
                }

                result.Add(node);
            }

            graph.Nodes = result;
            System.Diagnostics.Debug.Assert(graph.IsLinearOrderValid(), "graph.IsLinearOrderValid()");
        }

        /// <summary>
        /// Position a loop-invariant node hoists to: just before the outermost active
        /// <c>LOOP_OPEN</c> that was opened inside the innermost active non-loop scope, so the
        /// hoist shrinks every loop it can without leaving a branch. Returns false when no loop
        /// scope is active, or when every active one encloses that barrier rather than sitting
        /// inside it — a loop opened <em>inside</em> a branch still shrinks; the hoist just stops
        /// at that branch's OPEN.
        /// </summary>
        private static bool TryFindHoistPosition(
            Dictionary<FastNodeKey, int> openPositions, List<int> barrierPositions, out int position)
        {
            int innermostBarrier = barrierPositions.Count > 0 ? barrierPositions[^1] : -1;

            position = int.MaxValue;
            foreach (var p in openPositions.Values)
                if (p > innermostBarrier && p < position) position = p;

            if (position != int.MaxValue) return true;
            position = -1;
            return false;
        }

        private static bool HasLoopDependentInput(FastNode node, HashSet<FastTensorKey> loopDependent)
        {
            foreach (var kvp in node.FullInputs)
                foreach (var key in kvp.Value)
                    if (key is FastTensorKey tk && !tk.IsEmpty && loopDependent.Contains(tk))
                        return true;
            return false;
        }

        private static void AddAllOutputs(FastNode node, HashSet<FastTensorKey> loopDependent)
        {
            foreach (var kvp in node.FullOutputs)
                foreach (var key in kvp.Value)
                    if (key is FastTensorKey tk && !tk.IsEmpty)
                        loopDependent.Add(tk);
        }
    }
}

using Shorokoo.Core.AutoDiffCheckpointing;
using Shorokoo.Core.Graph;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using System.Collections.Generic;
using System.Linq;

namespace Shorokoo.Core.AutoDiffCheckpointing;

/// <summary>
/// Reorders a <see cref="FastComputationGraph"/>'s topological order to minimize peak memory usage.
///
/// The default topological order is determined by node creation order, which often
/// produces suboptimal memory usage: tensors produced early but consumed late create
/// long "lifetimes" that inflate peak memory. By reordering nodes (while respecting
/// dependencies), we can produce tensors closer to when they're needed and free them
/// sooner.
///
/// Algorithm: greedy memory-aware topological sort. At each step, we pick the
/// ready node (all inputs computed) that maximizes net memory freed:
///   score = (memory freed by consuming last-use inputs) - (memory added by outputs)
///
/// Structural constraints: close nodes (LOOP_CLOSE / IF_CLOSE) cannot be scheduled
/// before their matching open node (linked via <see cref="FastNode.GraphOpenNodeKey"/>).
/// </summary>
internal class MemoryAwareScheduler
{
    /// <summary>
    /// Reorders the given graph's nodes to minimize peak memory, based on the
    /// provided shape information for tensor sizes.
    /// </summary>
    /// <returns>A new <see cref="FastComputationGraph"/> with the same nodes in a memory-optimized order.</returns>
    public FastComputationGraph Reorder(FastComputationGraph graph, ShapeInferenceResult shapeInfo)
    {
        var nodes = graph.Nodes;
        if (nodes.Count <= 2)
            return graph;

        var reordered = MemoryAwareTopologicalSort(nodes, shapeInfo);

        // The scope-aware scheduler can fail to find a memory-minimizing linear order
        // for some valid graph shapes — notably the back-prop-through-time scope of a
        // recurrent op (RNN/LSTM/GRU), whose scope structure the greedy scope model
        // can stall on. Reordering is a pure optimization and the INPUT order is
        // already a valid topological order, so fall back to it (returning the same
        // graph reference, which the optimizer reads as "no improvement") rather than
        // throwing and blocking the whole training graph. A genuinely malformed graph
        // (real cycle) is still surfaced later by execution/compilation.
        if (reordered is null)
            return graph;

        // Check if the order actually changed
        bool changed = false;
        for (int i = 0; i < nodes.Count; i++)
        {
            if (!ReferenceEquals(nodes[i], reordered[i]))
            {
                changed = true;
                break;
            }
        }

        if (!changed)
            return graph;

        // Build a fresh graph that reuses everything except the node order.
        var copy = graph.Clone();
        copy.Nodes = reordered.ToList();
        System.Diagnostics.Debug.Assert(copy.IsLinearOrderValid(), "copy.IsLinearOrderValid()");
        return copy;
    }

    /// <summary>
    /// Performs a memory-aware topological sort using a greedy heuristic.
    ///
    /// Maintains a set of "ready" nodes (all dependencies satisfied) and at each step
    /// picks the node that provides the best net memory impact:
    ///   score = sum(sizes of inputs that are last-consumed by this node)
    ///         - sum(sizes of output tensors produced by this node)
    ///
    /// <para>
    /// Subgraph scopes (LOOP_OPEN..LOOP_CLOSE, IF_OPEN..IF_CLOSE) are honored as
    /// regions a valid linear order can't cross: each non-boundary node has an
    /// owning OPEN, and only nodes belonging to the current scope (or the
    /// matching CLOSE) are eligible at each step. Scheduling an OPEN descends
    /// into its scope; scheduling its CLOSE pops back out. Without this the
    /// scheduler could hoist a body node out of an IF subgraph (or vice versa)
    /// and produce a graph that fails <c>IsLinearOrderValid</c>.
    /// </para>
    /// </summary>
    private List<FastNode>? MemoryAwareTopologicalSort(
        IList<FastNode> nodes,
        ShapeInferenceResult shapeInfo)
    {
        // Pre-compute tensor memory sizes
        var tensorMemory = new Dictionary<FastTensorKey, long>();
        foreach (var node in nodes)
        {
            foreach (var output in node.Outputs)
            {
                if (output is null) continue;
                var info = shapeInfo.GetTensorInfo(output.Value);
                tensorMemory[output.Value] = info?.MemoryBytes ?? 0;
            }
        }

        // Build consumer counts: how many nodes consume each tensor
        var consumerCount = new Dictionary<FastTensorKey, int>();
        foreach (var node in nodes)
        {
            foreach (var input in node.Inputs)
            {
                if (input is null) continue;
                consumerCount.TryGetValue(input.Value, out var c);
                consumerCount[input.Value] = c + 1;
            }
        }

        // Pre-compute each node's owning scope (the innermost enclosing OPEN's key).
        // Convention: OPEN belongs to its PARENT scope (it's the scope's boundary).
        // CLOSE belongs to ITS OWN scope (so scheduling it from within the scope pops back out).
        // All other nodes belong to whichever scope is currently active in the original linear order.
        // Also: for each OPEN, capture the set of tensor keys its body consumes but doesn't
        // produce — those are the scope's "external prerequisites" that must be available
        // before we descend into the scope, otherwise body nodes would stall on unmet deps.
        var nodeOwnerScope = new Dictionary<FastNode, FastNodeKey?>();
        var openBodyExternals = new Dictionary<FastNode, HashSet<FastTensorKey>>();
        {
            var scanStack = new Stack<(FastNode open, HashSet<FastTensorKey> produced, HashSet<FastTensorKey> consumed)>();
            foreach (var node in nodes)
            {
                if (node.IsOpenNode())
                {
                    nodeOwnerScope[node] = scanStack.Count > 0 ? scanStack.Peek().open.Key : (FastNodeKey?)null;
                    // Record the OPEN as part of its parent scope's body too.
                    if (scanStack.Count > 0)
                    {
                        var parent = scanStack.Peek();
                        foreach (var input in node.Inputs)
                            if (input is not null) parent.consumed.Add(input.Value);
                        foreach (var output in node.Outputs)
                            if (output is not null) parent.produced.Add(output.Value);
                    }
                    scanStack.Push((node, new HashSet<FastTensorKey>(), new HashSet<FastTensorKey>()));
                }
                else if (node.IsCloseNode())
                {
                    nodeOwnerScope[node] = scanStack.Count > 0 ? scanStack.Peek().open.Key : (FastNodeKey?)null;
                    if (scanStack.Count > 0)
                    {
                        var (openNode, produced, consumed) = scanStack.Pop();
                        // CLOSE is part of its own scope's body for produced/consumed bookkeeping.
                        foreach (var input in node.Inputs)
                            if (input is not null) consumed.Add(input.Value);
                        foreach (var output in node.Outputs)
                            if (output is not null) produced.Add(output.Value);
                        // External = consumed but not produced anywhere in the body.
                        consumed.ExceptWith(produced);
                        openBodyExternals[openNode] = consumed;
                        // Also feed the close's outputs back to the parent scope as produced.
                        if (scanStack.Count > 0)
                        {
                            var parent = scanStack.Peek();
                            foreach (var output in node.Outputs)
                                if (output is not null) parent.produced.Add(output.Value);
                            // The close consumes from inside its own scope (those are not the
                            // parent's concern). Parent only sees the close's outputs.
                        }
                    }
                }
                else
                {
                    nodeOwnerScope[node] = scanStack.Count > 0 ? scanStack.Peek().open.Key : (FastNodeKey?)null;
                    if (scanStack.Count > 0)
                    {
                        var current = scanStack.Peek();
                        foreach (var input in node.Inputs)
                            if (input is not null) current.consumed.Add(input.Value);
                        foreach (var output in node.Outputs)
                            if (output is not null) current.produced.Add(output.Value);
                    }
                }
            }
        }

        // Track what's available: starts EMPTY, populated by each scheduled node's outputs.
        // Crucially we don't seed with graph.Inputs even though those tensor keys are
        // "available externally" — every graph-input key is also the output of a
        // MODEL_TENSOR_INPUT node in graph.Nodes, and downstream code that compiles the
        // reordered graph (FastComputationGraphConverter) requires strict producer-then-
        // consumer ordering: each consumer's input tensors must already appear as some
        // earlier node's output in the linear order. Seeding with graph.Inputs would let
        // a consumer be picked before its MODEL_TENSOR_INPUT producer, producing a graph
        // that passes IsLinearOrderValid (which treats graph inputs as always-available)
        // but blows up at compile time.
        var available = new HashSet<FastTensorKey>();

        bool DepsMet(FastNode n)
        {
            foreach (var input in n.Inputs)
                if (input is not null && !available.Contains(input.Value))
                    return false;
            return true;
        }

        // Runtime scope stack mirrored as the schedule unfolds.
        var runtimeScopeStack = new Stack<FastNodeKey>();
        FastNodeKey? CurrentScope() => runtimeScopeStack.Count > 0 ? runtimeScopeStack.Peek() : (FastNodeKey?)null;

        // Remaining consumer count (decremented as nodes are scheduled), used by the
        // memory heuristic to recognize last-consumer events.
        var remainingConsumers = new Dictionary<FastTensorKey, int>(consumerCount);

        var pending = new HashSet<FastNode>(nodes);
        var scheduled = new List<FastNode>(nodes.Count);

        while (pending.Count > 0)
        {
            var eligible = new List<FastNode>();
            var currentScope = CurrentScope();
            foreach (var n in pending)
            {
                if (!Equals(nodeOwnerScope[n], currentScope)) continue;
                if (!DepsMet(n)) continue;
                // For an OPEN, also require all of its body's external inputs to be
                // available — otherwise descending into the scope would leave body
                // nodes with unmet deps from outside.
                if (n.IsOpenNode() && openBodyExternals.TryGetValue(n, out var externals))
                {
                    bool externalsReady = true;
                    foreach (var ext in externals)
                    {
                        if (!available.Contains(ext)) { externalsReady = false; break; }
                    }
                    if (!externalsReady) continue;
                }
                eligible.Add(n);
            }

            if (eligible.Count == 0)
                // No node at the current scope has its deps met — the greedy scope
                // model can't make progress on this graph shape (e.g. a recurrent
                // BPTT scope). Signal "can't schedule" so Reorder falls back to the
                // input order rather than throwing; the input is a valid topological
                // order, so correctness is preserved (only the memory reorder is skipped).
                return null;

            FastNode best = PickBestNode(eligible, remainingConsumers, tensorMemory);

            pending.Remove(best);
            scheduled.Add(best);

            // Decrement consumer counts for inputs
            foreach (var input in best.Inputs)
            {
                if (input is null) continue;
                if (remainingConsumers.ContainsKey(input.Value))
                    remainingConsumers[input.Value]--;
            }

            // Mark this node's outputs as available for downstream consumers.
            foreach (var output in best.Outputs)
            {
                if (output is null) continue;
                available.Add(output.Value);
            }

            // Update runtime scope stack so the next iteration's eligibility check
            // matches the new active scope.
            if (best.IsOpenNode())
            {
                runtimeScopeStack.Push(best.Key);
            }
            else if (best.IsCloseNode())
            {
                if (runtimeScopeStack.Count > 0) runtimeScopeStack.Pop();
            }
        }

        return scheduled;
    }

    /// <summary>
    /// Picks the best node from the ready set based on memory impact scoring.
    /// </summary>
    private static FastNode PickBestNode(
        IEnumerable<FastNode> readySet,
        Dictionary<FastTensorKey, int> remainingConsumers,
        Dictionary<FastTensorKey, long> tensorMemory)
    {
        FastNode? best = null;
        long bestScore = long.MinValue;
        long bestOutputSize = long.MaxValue;

        foreach (var node in readySet)
        {
            // Memory freed: inputs where this node is the last remaining consumer
            long memoryFreed = 0;
            foreach (var input in node.Inputs)
            {
                if (input is null) continue;
                if (remainingConsumers.TryGetValue(input.Value, out var remaining) && remaining == 1)
                    memoryFreed += tensorMemory.GetValueOrDefault(input.Value, 0);
            }

            // Memory added: output tensor sizes
            long memoryAdded = 0;
            foreach (var output in node.Outputs)
            {
                if (output is null) continue;
                memoryAdded += tensorMemory.GetValueOrDefault(output.Value, 0);
            }

            long score = memoryFreed - memoryAdded;

            // Pick highest score; break ties by preferring smaller output allocation
            if (score > bestScore || (score == bestScore && memoryAdded < bestOutputSize))
            {
                best = node;
                bestScore = score;
                bestOutputSize = memoryAdded;
            }
        }

        return best!;
    }
}

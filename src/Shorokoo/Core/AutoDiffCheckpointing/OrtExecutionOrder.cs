using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Graph;
using System.Collections.Generic;
using System.Linq;

namespace Shorokoo.Core.AutoDiffCheckpointing;

/// <summary>
/// The order ONNX Runtime actually executes a graph in, and its inverse.
///
/// <para>ORT ignores the order NodeProtos appear in. Its <c>GraphViewer</c> recomputes a
/// topological order: a reverse depth-first search from the leaf nodes (nodes with no
/// consumer) taken in node-index order, where each node's producers are sorted by node index
/// and pushed on a stack — so the <b>highest</b>-index producer is explored first — and nodes
/// are emitted in post-order. The sequential executor then runs that order. Node index is
/// the position among emitted NodeProtos, which is this graph's <see cref="InternalComputationGraph.Nodes"/>
/// order with the non-emitted nodes skipped, so the only influence the linear order has on
/// what runs is the tie-break among independent siblings — and it is inverted: among
/// two ready branches ORT runs the one that appears LATER first.</para>
///
/// <para><see cref="Compute"/> reproduces that traversal. It was verified against the
/// profiled kernel sequence of three lowered training-step graphs with graph optimizations
/// disabled (440/440, 392/392 and 1452/1452 kernels at the same position). With
/// optimizations enabled ORT fuses and folds nodes, so the sequence is approximate there,
/// and the ONNX emitter's scope configuration (<c>ConfigureScopes</c>) can move nodes into a
/// Loop/If body before emission, so graphs with scopes are modelled at the granularity of
/// this graph's own scopes.</para>
///
/// <para>Nodes ORT never schedules as kernels — model inputs, parameter data, <c>Constant</c>
/// (folded to an initializer at load), and the Loop body's own inputs — are resident before
/// the level they belong to runs, so they are placed at the start of that level. A scope
/// (OPEN..CLOSE) is one ORT node (the Loop/If kernel, positioned at its CLOSE); its body is
/// a subgraph ORT orders by the same rule, recursively.</para>
/// </summary>
internal static class OrtExecutionOrder
{
    /// <summary>The indices of <paramref name="nodes"/> in the order ORT executes them.</summary>
    public static int[] Compute(IList<FastNode> nodes)
    {
        var result = new List<int>(nodes.Count);
        EmitLevel(nodes, MatchScopes(nodes), 0, nodes.Count, rank: null, result);
        return result.ToArray();
    }

    /// <summary>
    /// A linear order of <paramref name="nodes"/> that makes ORT's traversal follow the
    /// sibling preferences of <paramref name="preferred"/> (a valid linear order of the same
    /// nodes): wherever ORT chooses among independent producers, the one that comes first in
    /// <paramref name="preferred"/> gets the higher index and therefore runs first. The result
    /// is itself a valid linear order (producers first, scopes contiguous).
    ///
    /// <para>This is the only lever the linear order has on ORT. A preferred sequence is not
    /// reproduced verbatim when it interleaves independent chains (a depth-first executor
    /// finishes one producer subtree before starting the next) or when a producer is shared
    /// between siblings; the caller must score the result under
    /// <see cref="EvaluationOrder.OrtOrder"/> rather than assume the preference was honoured.</para>
    /// </summary>
    public static List<FastNode> Realize(IList<FastNode> nodes, IList<FastNode> preferred)
    {
        var index = new Dictionary<FastNode, int>(nodes.Count);
        for (int i = 0; i < nodes.Count; i++) index[nodes[i]] = i;
        var rank = new int[nodes.Count];
        for (int p = 0; p < preferred.Count; p++) rank[index[preferred[p]]] = p;

        var result = new List<int>(nodes.Count);
        EmitLevel(nodes, MatchScopes(nodes), 0, nodes.Count, rank, result);
        return result.Select(i => nodes[i]).ToList();
    }

    /// <summary>
    /// How much of the linear order survives at runtime: the fraction of consecutive pairs in
    /// <paramref name="ortOrder"/> (kernels only) whose node indices ascend. 1 means ORT runs
    /// the linear order verbatim; a graph whose sibling branches ORT visits in the opposite
    /// order scores lower.
    /// </summary>
    public static double Fidelity(IList<FastNode> nodes, int[] ortOrder)
    {
        int pairs = 0, ascending = 0, previous = -1;
        foreach (var idx in ortOrder)
        {
            if (!IsKernel(nodes[idx])) continue;
            if (previous >= 0) { pairs++; if (idx > previous) ascending++; }
            previous = idx;
        }
        return pairs == 0 ? 1.0 : (double)ascending / pairs;
    }

    private static bool IsPreResident(FastNode node)
        => node.IsModelInput() || node.IsModelParamData()
        || node.OpCode is OpCodes.CONSTANT or OpCodes.LOOP_FAKE_INPUT or OpCodes.LOOP_INDEX_VARIABLE or OpCodes.LOOP_SCAN_VARIABLE;

    private static bool IsKernel(FastNode node) => !IsPreResident(node) && !node.IsOpenNode();

    /// <summary>closeIdx per OPEN index (-1 elsewhere), or null when the scopes do not nest.</summary>
    private static int[]? MatchScopes(IList<FastNode> nodes)
    {
        var closeOf = new int[nodes.Count];
        System.Array.Fill(closeOf, -1);
        var open = new Stack<int>();
        for (int i = 0; i < nodes.Count; i++)
        {
            if (nodes[i].IsOpenNode()) open.Push(i);
            else if (nodes[i].IsCloseNode())
            {
                if (open.Count == 0 || nodes[open.Peek()].Key != nodes[i].GraphOpenNodeKey) return null;
                closeOf[open.Pop()] = i;
            }
        }
        return open.Count == 0 ? closeOf : null;
    }

    /// <summary>
    /// Orders the members of one scope level — plain nodes and nested scopes — in
    /// <c>[lo, hi)</c>, appending node indices to <paramref name="result"/>. With
    /// <paramref name="rank"/> null the traversal key is ORT's node index; otherwise it is the
    /// preferred rank, which yields the linear order <see cref="Realize"/> describes.
    /// </summary>
    private static void EmitLevel(IList<FastNode> nodes, int[]? closeOf, int lo, int hi, int[]? rank, List<int> result)
    {
        var members = new List<(int Idx, int Close)>();
        for (int i = lo; i < hi; i++)
        {
            if (closeOf is not null && closeOf[i] >= 0) { members.Add((i, closeOf[i])); i = closeOf[i]; }
            else members.Add((i, -1));
        }
        int count = members.Count;

        long Key(int m) => rank is null
            ? (members[m].Close >= 0 ? members[m].Close : members[m].Idx)
            : rank[members[m].Idx];

        var producer = new Dictionary<FastTensorKey, int>();
        for (int m = 0; m < count; m++)
        {
            var (idx, close) = members[m];
            for (int i = idx; i <= (close >= 0 ? close : idx); i++)
                foreach (var output in nodes[i].Outputs)
                    if (output is not null) producer[output.Value] = m;
        }

        var preResident = new bool[count];
        for (int m = 0; m < count; m++)
            preResident[m] = members[m].Close < 0 && IsPreResident(nodes[members[m].Idx]);

        var deps = new HashSet<int>[count];
        var hasConsumer = new bool[count];
        for (int m = 0; m < count; m++)
        {
            deps[m] = new HashSet<int>();
            var (idx, close) = members[m];
            for (int i = idx; i <= (close >= 0 ? close : idx); i++)
                foreach (var input in nodes[i].Inputs)
                    if (input is not null && producer.TryGetValue(input.Value, out var p) && p != m && !preResident[p])
                        deps[m].Add(p);
            foreach (var p in deps[m]) hasConsumer[p] = true;
        }

        void EmitMember(int m)
        {
            var (idx, close) = members[m];
            result.Add(idx);
            if (close < 0) return;
            EmitLevel(nodes, closeOf, idx + 1, close, rank, result);
            result.Add(close);
        }

        foreach (var m in Enumerable.Range(0, count).Where(m => preResident[m]).OrderBy(Key))
            EmitMember(m);

        var visited = new bool[count];
        var stack = new List<(int Member, bool Leave)>();
        foreach (var leaf in Enumerable.Range(0, count).Where(m => !preResident[m] && !hasConsumer[m]).OrderBy(Key))
            stack.Add((leaf, false));

        while (stack.Count > 0)
        {
            var (m, leave) = stack[^1];
            stack.RemoveAt(stack.Count - 1);
            if (leave) { EmitMember(m); continue; }
            if (visited[m]) continue;
            visited[m] = true;
            stack.Add((m, true));
            foreach (var p in deps[m].OrderBy(Key))
                if (!visited[p]) stack.Add((p, false));
        }

        for (int m = 0; m < count; m++)
            if (!preResident[m] && !visited[m]) { visited[m] = true; EmitMember(m); }
    }
}

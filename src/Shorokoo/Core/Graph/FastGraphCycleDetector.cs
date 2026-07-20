using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.OnnxNodes;
using System.Collections.Generic;
using System.Linq;
using Shorokoo.Core.Nodes.Processors.Helpers;

namespace Shorokoo.Core.Graph
{
    /// <summary>
    /// Cycle detection helpers for <see cref="InternalComputationGraph"/>. A ComputationGraph is a
    /// DAG by construction; a cycle always indicates a bug in a graph transformation. These
    /// helpers are meant to be sprinkled through the pipeline to bisect the stage that
    /// introduced a bad edge.
    /// </summary>
    internal static class FastGraphCycleDetector
    {
        /// <summary>
        /// Finds one cycle in <paramref name="graph"/>, if any. Returns the node keys in the
        /// cycle in traversal order (with the first node repeated at the end to close the loop),
        /// or null if the graph is acyclic.
        /// </summary>
        public static List<FastNodeKey>? FindCycle(InternalComputationGraph graph)
        {
            var producer = new Dictionary<FastTensorKey, FastNode>();
            foreach (var node in graph.Nodes)
                foreach (var output in node.Outputs)
                    if (output is FastTensorKey tk && !tk.IsEmpty)
                        producer[tk] = node;

            // 0 = unvisited, 1 = on current DFS path, 2 = fully explored
            var color = new Dictionary<FastNode, int>(graph.Nodes.Count);
            foreach (var n in graph.Nodes) color[n] = 0;

            foreach (var root in graph.Nodes)
            {
                if (color[root] != 0) continue;

                var path = new List<FastNode>();
                var iterators = new Stack<IEnumerator<FastNode>>();

                color[root] = 1;
                path.Add(root);
                iterators.Push(ParentsOf(root, producer).GetEnumerator());

                while (iterators.Count > 0)
                {
                    var it = iterators.Peek();
                    if (it.MoveNext())
                    {
                        var parent = it.Current;
                        var pc = color[parent];
                        if (pc == 1)
                        {
                            // Back edge: cycle is parent -> ... -> path.Last() -> parent
                            var startIdx = path.IndexOf(parent);
                            var cycle = path.Skip(startIdx).Select(n => n.Key).ToList();
                            cycle.Add(parent.Key);
                            return cycle;
                        }
                        if (pc == 0)
                        {
                            color[parent] = 1;
                            path.Add(parent);
                            iterators.Push(ParentsOf(parent, producer).GetEnumerator());
                        }
                        // else already fully explored; skip
                    }
                    else
                    {
                        iterators.Pop();
                        var done = path[path.Count - 1];
                        path.RemoveAt(path.Count - 1);
                        color[done] = 2;
                    }
                }
            }

            return null;
        }

        /// <summary>
        /// Throws if <paramref name="graph"/> contains a cycle. The exception message includes
        /// the offending node keys and the <paramref name="context"/> label to make it easy to
        /// identify which pipeline stage introduced the bad edge.
        /// </summary>
        public static void AssertAcyclic(InternalComputationGraph graph, string context)
        {
            var cycle = FindCycle(graph);
            if (cycle is null) return;

            var detail = string.Join(" -> ", cycle);
            throw new System.InvalidOperationException(
                $"Circular reference detected in InternalComputationGraph at [{context}]. " +
                $"Cycle: {detail}");
        }

        private static IEnumerable<FastNode> ParentsOf(FastNode n, Dictionary<FastTensorKey, FastNode> producer)
        {
            var seen = new HashSet<FastNode>();
            foreach (var inp in n.Inputs)
            {
                if (inp is not FastTensorKey tk || tk.IsEmpty) continue;
                if (!producer.TryGetValue(tk, out var p)) continue;
                if (seen.Add(p)) yield return p;
            }
        }
    }
}

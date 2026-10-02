using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory.IR;
using Shorokoo.PythonTranslation;

namespace Shorokoo.PyTorch;

/// <summary>
/// Which nodes of a graph a torch run computing no gradient writes over one of their operands: an
/// element-wise operator reading, in the position it writes each element, a value of memory of its
/// own that nothing reads after it — what a run would let go of right after the node — so that the
/// node allocates nothing for its result and the run never holds the two at once.
///
/// <para>A translation runs the top-level graph's nodes one after the other in the order it writes
/// them (<see cref="OnnxToPythonTranslator.ShapesFirst"/>), so "after" is that order. The operand
/// is the output of a node whose translation always computes into memory of its own
/// (<see cref="PlacementMemory.TorchFresh"/>), and so is anything handed back over it — a view
/// (<see cref="PlacementMemory.PyTorch"/>), or a value a node holding a subgraph that reads it hands
/// back — each read by no node after the writer, nor after the run. Whether torch can write the
/// result there — of the operand's type and shape, the run computing no gradient — the run decides
/// as it writes, and computes the result as it would otherwise where not.</para>
/// </summary>
internal static class TorchInPlace
{
    /// <summary>The nodes of <paramref name="graph"/>'s top level that may be written over an
    /// operand, by their output's name, with the operand's slot; none for a training step whose
    /// gradient torch computes, which keeps what the forward pass made.</summary>
    internal static IReadOnlyDictionary<string, int> Plan(GraphProto graph)
    {
        var plan = new Dictionary<string, int>(StringComparer.Ordinal);
        if (graph.Nodes.Any(n => n.Domain == TrainingFormats.AutoGradDomain)) return plan;
        var order = OnnxToPythonTranslator.ShapesFirst(graph.Nodes).ToList();
        var producer = new Dictionary<string, (NodeProto Node, int Output)>(StringComparer.Ordinal);
        var readers = new Dictionary<string, List<int>>(StringComparer.Ordinal);
        var views = new Dictionary<string, List<string>>(StringComparer.Ordinal);
        var outputs = graph.Outputs.Select(o => o.Name).ToHashSet(StringComparer.Ordinal);
        void Read(string value, int at)
        {
            if (!readers.TryGetValue(value, out var list)) readers[value] = list = [];
            list.Add(at);
        }
        void View(string of, string view)
        {
            if (!views.TryGetValue(of, out var list)) views[of] = list = [];
            list.Add(view);
        }
        for (int k = 0; k < order.Count; k++)
        {
            var node = order[k];
            for (int o = 0; o < node.Outputs.Count; o++)
                if (node.Outputs[o].Length > 0) producer[node.Outputs[o]] = (node, o);
            if (OutputAliasProof.ReadsOnlyAShape(node)) continue;
            for (int i = 0; i < node.Inputs.Count; i++)
            {
                var input = node.Inputs[i];
                if (input.Length == 0) continue;
                Read(input, k);
                for (int o = 0; o < node.Outputs.Count; o++)
                    if (node.Outputs[o].Length > 0 && PlacementMemory.PyTorch.SharesUnlessPlaced(node, i, o)) View(input, node.Outputs[o]);
            }
            if (!OutputAliasProof.HoldsSubgraph(node)) continue;
            var captured = new HashSet<string>(StringComparer.Ordinal);
            foreach (var attribute in node.Attributes)
            {
                if (attribute.G is { } subgraph) OutputAliasProof.GraphIndex.ReferencedFrom(subgraph, captured);
                foreach (var each in attribute.Graphs) OutputAliasProof.GraphIndex.ReferencedFrom(each, captured);
            }
            foreach (var value in captured)
            {
                Read(value, k);
                foreach (var output in node.Outputs.Where(o => o.Length > 0)) View(value, output);
            }
        }

        for (int k = 0; k < order.Count; k++)
        {
            var writer = order[k];
            if (!Writes(writer) || writer.Outputs.Count(o => o.Length > 0) != 1 || writer.Outputs[0].Length == 0) continue;
            foreach (var slot in Slots(writer))
            {
                var operand = writer.Inputs[slot];
                if (operand.Length == 0 || !Fresh(operand, producer)) continue;
                var memory = Memory(operand, views);
                if (memory.Any(outputs.Contains)) continue;
                if (memory.Any(value => readers.GetValueOrDefault(value, []).Any(at => at > k || (at == k && value != operand)))) continue;
                if (writer.Inputs.Where((input, i) => i != slot && input.Length > 0).Any(input => input != operand && memory.Contains(input))) continue;
                plan[writer.Outputs[0]] = slot;
                break;
            }
        }
        return plan;
    }

    /// <summary>Whether torch can write <paramref name="node"/>'s result over an operand it reads
    /// in the position it writes each element: the element-wise operators torch has a form of that
    /// writes into a given tensor, a <c>Clip</c> and a <c>Gelu</c>.</summary>
    private static bool Writes(NodeProto node)
        => OutputAliasProof.IsStandard(node)
           && ((PlacementMemory.TorchElementWise.Contains(node.OpType)
                && (PlacementProof.InPlaceUnary.Contains(node.OpType) ? node.Inputs.Count == 1 : node.Inputs.Count == 2))
               || node.OpType is "Clip" or "Gelu");

    /// <summary>The operand slots <paramref name="node"/> reads in the position it writes.</summary>
    private static IEnumerable<int> Slots(NodeProto node)
        => node.OpType is "Clip" or "Gelu" || PlacementProof.InPlaceUnary.Contains(node.OpType) ? [0] : [0, 1];

    /// <summary>Whether <paramref name="value"/> is memory of its own: written by a node of the
    /// graph whose translation always computes afresh.</summary>
    private static bool Fresh(string value, Dictionary<string, (NodeProto Node, int Output)> producer)
        => producer.TryGetValue(value, out var made)
           && PlacementMemory.TorchFresh.Contains(made.Node.OpType)
           && OutputAliasProof.IsStandard(made.Node)
           && !Enumerable.Range(0, made.Node.Inputs.Count).Any(i => PlacementMemory.PyTorch.SharesUnlessPlaced(made.Node, i, made.Output));

    /// <summary><paramref name="value"/> and everything handed back over its memory.</summary>
    private static HashSet<string> Memory(string value, Dictionary<string, List<string>> views)
    {
        var memory = new HashSet<string>(StringComparer.Ordinal) { value };
        var pending = new Queue<string>([value]);
        while (pending.TryDequeue(out var current))
            foreach (var view in views.GetValueOrDefault(current, []))
                if (memory.Add(view)) pending.Enqueue(view);
        return memory;
    }
}

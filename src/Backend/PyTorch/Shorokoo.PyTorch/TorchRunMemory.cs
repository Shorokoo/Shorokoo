using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory.IR;
using Shorokoo.PythonTranslation;

namespace Shorokoo.PyTorch;

/// <summary>
/// The most a run of a model's translation holds at once beyond its inputs, as the translation lays
/// values out: each value takes its bytes when the node writing it runs, in the order the
/// translation runs the nodes (<see cref="OnnxToPythonTranslator.RunOrder"/>), and gives them back
/// after the last node reading it — torch's caching allocator has them again at once. A value
/// handed back over an input (<see cref="PlacementMemory.PyTorch"/>: a slice, a reshape, a
/// transpose, an expansion) is that input's memory; a node written over an operand dying there
/// (<see cref="TorchInPlace"/>) takes the operand's memory; an output written into the input it is
/// paired with (<see cref="OutputAlias"/>) takes that input's. The inputs, the initializers and the
/// constants are held before the run begins and are not counted.
/// </summary>
internal static class TorchRunMemory
{
    /// <summary>
    /// The peak of a run of <paramref name="model"/>, whose inputs state their shapes in full, with
    /// the outputs of <paramref name="aliases"/> written into their inputs, which the run consumes;
    /// null where a value's shape cannot be told, which leaves the model's bytes unknown.
    /// </summary>
    internal static long? Peak(ModelProto model, IReadOnlyList<OutputAlias> aliases)
    {
        if (model.Graph is not { } graph) return null;
        var given = new Dictionary<string, (long[] Shape, int ElementType)>(StringComparer.Ordinal);
        foreach (var input in graph.Inputs)
        {
            if (input.Type?.TensorType is not { Shape: { } shape } tensor) return null;
            if (shape.Dims.Any(d => d.DimValue <= 0 && d.DimParam is { Length: > 0 })) return null;
            given[input.Name] = ([.. shape.Dims.Select(d => d.DimValue)], tensor.ElemType);
        }
        Dictionary<string, PlacementShapes.Value> shapes;
        try
        {
            shapes = PlacementShapes.Evaluate(graph, given);
        }
        catch (ArgumentException)
        {
            return null;
        }
        if (graph.Nodes.Any(n => n.Outputs.Any(o => o.Length > 0 && !shapes.ContainsKey(o)))) return null;

        var order = OnnxToPythonTranslator.RunOrder(graph.Nodes);
        var position = new Dictionary<NodeProto, int>(ReferenceEqualityComparer.Instance);
        for (int k = 0; k < order.Count; k++) position[order[k]] = k;
        var held = graph.Inputs.Select(i => i.Name).Concat(graph.Initializers.Select(i => i.Name)).ToHashSet(StringComparer.Ordinal);
        var consumed = aliases.Select(a => a.Input).ToHashSet(StringComparer.Ordinal);
        var intoInput = aliases.ToDictionary(a => a.Output, a => a.Input, StringComparer.Ordinal);
        var over = TorchInPlace.Plan(graph);

        // Every value's memory: its own, or the one it is handed back over.
        var root = new Dictionary<string, string>(StringComparer.Ordinal);
        string RootOf(string value) => root.TryGetValue(value, out var r) && r != value ? root[value] = RootOf(r) : value;
        foreach (var node in order)
            for (int o = 0; o < node.Outputs.Count; o++)
            {
                var output = node.Outputs[o];
                if (output.Length == 0) continue;
                for (int i = 0; i < node.Inputs.Count; i++)
                    if (node.Inputs[i].Length > 0 && PlacementMemory.PyTorch.SharesUnlessPlaced(node, i, o))
                    {
                        root[output] = RootOf(node.Inputs[i]);
                        break;
                    }
            }
        var last = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var output in graph.Outputs) last[RootOf(output.Name)] = int.MaxValue;
        foreach (var node in order)
            foreach (var input in node.Inputs.Where(i => i.Length > 0))
            {
                var r = RootOf(input);
                last[r] = Math.Max(last.GetValueOrDefault(r, -1), position[node]);
            }

        var alive = new Dictionary<string, long>(StringComparer.Ordinal);
        long live = 0, peak = 0;
        for (int k = 0; k < order.Count; k++)
        {
            var node = order[k];
            foreach (var output in node.Outputs.Where(o => o.Length > 0))
            {
                if (RootOf(output) != output || node.OpType == "Constant") continue;
                if (intoInput.TryGetValue(output, out var input) && held.Contains(input) && WritesInto(node))
                {
                    root[output] = input;
                    continue;
                }
                var bytes = Math.Max(shapes[output].Bytes, 0);
                if (over.TryGetValue(output, out var slot) && node.Inputs[slot].Length > 0)
                {
                    var operand = RootOf(node.Inputs[slot]);
                    var dies = last.GetValueOrDefault(operand, -1) == k;
                    if (dies && alive.TryGetValue(operand, out var size) && size == bytes)
                    {
                        alive.Remove(operand);
                        alive[output] = bytes;
                        root[operand] = output;
                        continue;
                    }
                    if (dies && consumed.Contains(operand) && shapes.TryGetValue(operand, out var fed) && fed.Bytes == bytes)
                    {
                        root[output] = operand;
                        continue;
                    }
                }
                alive[output] = bytes;
                live += bytes;
            }
            peak = Math.Max(peak, live);
            foreach (var value in node.Inputs.Concat(node.Outputs).Where(v => v.Length > 0).Distinct())
            {
                var r = RootOf(value);
                if (alive.TryGetValue(r, out var size) && last.GetValueOrDefault(r, k) <= k)
                {
                    alive.Remove(r);
                    live -= size;
                }
            }
        }
        return peak;
    }

    /// <summary>Whether the translation writes <paramref name="node"/>'s output into the input it
    /// is paired with: a two-input <c>Add</c>, <c>Sub</c>, <c>Mul</c> or <c>Div</c>, torch's
    /// <c>out=</c> forms of which the translation's output aliasing calls.</summary>
    private static bool WritesInto(NodeProto node)
        => OutputAliasProof.IsStandard(node) && node.OpType is "Add" or "Sub" or "Mul" or "Div" && node.Inputs.Count == 2;
}

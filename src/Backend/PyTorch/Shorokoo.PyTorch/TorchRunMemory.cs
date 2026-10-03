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
/// transpose, an expansion, a reduction of no axis) is that input's memory — a reduction only where
/// it is of its input's size, since one reducing an axis computes a value of its own, and a reshape
/// only where torch can view what it reshapes, by its strides, since it copies it otherwise; a node written over an operand dying there
/// (<see cref="TorchInPlace"/>) takes the operand's memory; an output written into the input it is
/// paired with (<see cref="OutputAlias"/>) takes that input's. A kernel's own temporaries are held
/// for the node's length (<see cref="Scratch"/>). The inputs, the initializers and the constants are
/// held before the run begins and are not counted.
/// </summary>
internal static class TorchRunMemory
{
    /// <summary>
    /// The peak of a run of <paramref name="model"/>, whose inputs state their shapes in full, with
    /// the outputs of <paramref name="aliases"/> written into their inputs, which the run consumes,
    /// on the host where <paramref name="onHost"/> and on a card otherwise; null where a value's
    /// shape cannot be told, which leaves the model's bytes unknown.
    /// </summary>
    internal static long? Peak(ModelProto model, IReadOnlyList<OutputAlias> aliases, bool onHost = true)
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

        // Every value's memory: its own, or the one it is handed back over -- a reshape only where the
        // strides of what it reshapes allow a view (Strides), as torch's reshape copies otherwise.
        var root = new Dictionary<string, string>(StringComparer.Ordinal);
        string RootOf(string value) => root.TryGetValue(value, out var r) && r != value ? root[value] = RootOf(r) : value;
        var strides = new Dictionary<string, long[]>(StringComparer.Ordinal);
        long[]? StridesOf(string value) => strides.TryGetValue(value, out var known) ? known
            : shapes.TryGetValue(value, out var v) ? Contiguous(v.Shape) : null;
        foreach (var node in order)
            for (int o = 0; o < node.Outputs.Count; o++)
            {
                var output = node.Outputs[o];
                if (output.Length == 0) continue;
                for (int i = 0; i < node.Inputs.Count; i++)
                    if (node.Inputs[i].Length > 0 && PlacementMemory.PyTorch.SharesUnlessPlaced(node, i, o)
                        && !(node.OpType.StartsWith("Reduce", StringComparison.Ordinal)
                             && shapes.TryGetValue(node.Inputs[i], out var reduced) && reduced.Bytes != shapes[output].Bytes))
                    {
                        var viewed = ViewStrides(node, StridesOf(node.Inputs[i]), shapes.GetValueOrDefault(node.Inputs[i]), shapes.GetValueOrDefault(output));
                        if (viewed is null && node.OpType is "Reshape" or "Flatten" or "Squeeze" or "Unsqueeze") break;
                        if (viewed is not null) strides[output] = viewed;
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

        var producer = new Dictionary<string, NodeProto>(StringComparer.Ordinal);
        foreach (var node in order)
            foreach (var output in node.Outputs.Where(o => o.Length > 0))
                producer[output] = node;
        var alive = new Dictionary<string, long>(StringComparer.Ordinal);
        long live = 0, peak = 0;
        for (int k = 0; k < order.Count; k++)
        {
            var node = order[k];
            var writtenOver = false;
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
                        writtenOver = true;
                        continue;
                    }
                    if (dies && consumed.Contains(operand) && shapes.TryGetValue(operand, out var fed) && fed.Bytes == bytes)
                    {
                        root[output] = operand;
                        writtenOver = true;
                        continue;
                    }
                }
                alive[output] = bytes;
                live += bytes;
            }
            peak = Math.Max(peak, live + Scratch(node, shapes, producer, writtenOver, onHost));
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

    /// <summary>The strides of a value of <paramref name="shape"/> laid out row by row.</summary>
    private static long[] Contiguous(long[] shape)
    {
        var strides = new long[shape.Length];
        long step = 1;
        for (int d = shape.Length - 1; d >= 0; d--)
        {
            strides[d] = step;
            step *= Math.Max(shape[d], 1);
        }
        return strides;
    }

    /// <summary>
    /// The strides of <paramref name="node"/>'s output as a view of its input of
    /// <paramref name="input"/>'s shape and <paramref name="inputStrides"/>: permuted by a transpose,
    /// none along an axis an expansion broadcasts, the input's own for every other view — and for a
    /// reshape the strides torch gives the view, or null where torch's reshape cannot view the input
    /// and copies it.
    /// </summary>
    private static long[]? ViewStrides(NodeProto node, long[]? inputStrides, PlacementShapes.Value? input, PlacementShapes.Value? output)
    {
        if (inputStrides is null || input is null || output is null) return null;
        switch (node.OpType)
        {
            case "Transpose":
            {
                var perm = node.Attributes.FirstOrDefault(a => a.Name == "perm")?.Ints is { Length: > 0 } given
                    ? given : [.. Enumerable.Range(0, inputStrides.Length).Reverse().Select(d => (long)d)];
                return perm.Length == inputStrides.Length ? [.. perm.Select(d => inputStrides[d])] : null;
            }
            case "Expand":
            {
                var strides = new long[output.Shape.Length];
                var lead = output.Shape.Length - input.Shape.Length;
                for (int d = 0; d < strides.Length; d++)
                    strides[d] = d < lead || (input.Shape[d - lead] == 1 && output.Shape[d] != 1) ? 0 : inputStrides[d - lead];
                return strides;
            }
            case "Reshape" or "Flatten" or "Squeeze" or "Unsqueeze":
                return ComputeStride(input.Shape, inputStrides, output.Shape);
            default:
                return input.Shape.SequenceEqual(output.Shape) ? inputStrides : Contiguous(output.Shape);
        }
    }

    /// <summary>The strides torch gives a view of a tensor of <paramref name="shape"/> and
    /// <paramref name="strides"/> reshaped to <paramref name="target"/>, or null where it cannot
    /// view it (ATen's <c>computeStride</c>).</summary>
    private static long[]? ComputeStride(long[] shape, long[] strides, long[] target)
    {
        if (shape.Length == 0) return [.. target.Select(_ => 1L)];
        if (shape.Aggregate(1L, (a, d) => a * d) == 0) return shape.SequenceEqual(target) ? strides : Contiguous(target);
        var result = new long[target.Length];
        var viewD = target.Length - 1;
        var chunkBaseStride = strides[^1];
        long tensorNumel = 1, viewNumel = 1;
        for (int tensorD = shape.Length - 1; tensorD >= 0; tensorD--)
        {
            tensorNumel *= shape[tensorD];
            if (tensorD == 0 || (shape[tensorD - 1] != 1 && strides[tensorD - 1] != tensorNumel * chunkBaseStride))
            {
                while (viewD >= 0 && (viewNumel < tensorNumel || target[viewD] == 1))
                {
                    result[viewD] = viewNumel * chunkBaseStride;
                    viewNumel *= target[viewD];
                    viewD--;
                }
                if (viewNumel != tensorNumel) return null;
                if (tensorD > 0)
                {
                    chunkBaseStride = strides[tensorD - 1];
                    tensorNumel = 1;
                    viewNumel = 1;
                }
            }
        }
        return viewD == -1 ? result : null;
    }

    /// <summary>
    /// What the translation's kernel for <paramref name="node"/> holds of its own while it runs,
    /// measured operator by operator through torch's profiler: a <c>LayerNormalization</c> its
    /// centered values' squares where it is written over its operand
    /// (<paramref name="writtenOver"/>), and otherwise its centered values and one value more, each of
    /// its input's size; on the host, where oneDNN computes in a layout of its own, a convolution a
    /// copy of its output, a convolution of two transposed views — the gradient of another's weights,
    /// which the translation computes as torch does — a copy of each of the values they view, and a
    /// transposed convolution a copy of its input.
    /// </summary>
    internal static long Scratch(NodeProto node, IReadOnlyDictionary<string, PlacementShapes.Value> shapes,
        IReadOnlyDictionary<string, NodeProto> producer, bool writtenOver, bool onHost)
    {
        long BytesOf(string name) => name.Length > 0 && shapes.TryGetValue(name, out var value) ? Math.Max(value.Bytes, 0) : 0;
        if (!OutputAliasProof.IsStandard(node) || node.Inputs.Count == 0) return 0;
        var input = BytesOf(node.Inputs[0]);
        switch (node.OpType)
        {
            case "LayerNormalization":
                return writtenOver ? input : 2 * input;
            case "Conv" when onHost:
            {
                string? Transposed(int slot) => slot < node.Inputs.Count && producer.TryGetValue(node.Inputs[slot], out var made)
                    && made.OpType == "Transpose" && made.Attributes.FirstOrDefault(a => a.Name == "perm")?.Ints is [1, 0, ..] ? made.Inputs[0] : null;
                return Transposed(0) is { } activation && Transposed(1) is { } gradient
                    ? BytesOf(activation) + BytesOf(gradient)
                    : node.Outputs.Count > 0 ? BytesOf(node.Outputs[0]) : 0;
            }
            case "ConvTranspose" when onHost:
                return input;
            default:
                return 0;
        }
    }

    /// <summary>Whether the translation writes <paramref name="node"/>'s output into the input it
    /// is paired with: a two-input <c>Add</c>, <c>Sub</c>, <c>Mul</c> or <c>Div</c>, torch's
    /// <c>out=</c> forms of which the translation's output aliasing calls.</summary>
    private static bool WritesInto(NodeProto node)
        => OutputAliasProof.IsStandard(node) && node.OpType is "Add" or "Sub" or "Mul" or "Div" && node.Inputs.Count == 2;
}

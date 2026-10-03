using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory.IR;
using Shorokoo.PythonTranslation;

namespace Shorokoo.PyTorch;

/// <summary>
/// The most a run of a model's translation holds at once beyond its inputs, as the translation lays
/// values out: each value takes its bytes when the node writing it runs, in the order the
/// translation runs the nodes (<see cref="OnnxToPythonTranslator.RunOrder"/>), and gives them back
/// after the last node reading it — torch's caching allocator has them again at once. A value
/// handed back over an input (<see cref="Views"/>: a slice, a reshape, a transpose, an expansion, a
/// reduction of no axis) is that input's memory — a reduction only where
/// it is of its input's size, since one reducing an axis computes a value of its own, and a reshape
/// only where torch can view what it reshapes, by its strides, since it copies it otherwise — an
/// element-wise result lying as its operands do (<see cref="TorchStrides.OfElementwise"/>). A node
/// written over an operand dying there (<see cref="TorchInPlace"/>) takes the operand's memory,
/// where the operand is contiguous, as torch writes over no other; an output written into the input
/// it is paired with (<see cref="OutputAlias"/>) takes that input's. A kernel's own temporaries are held
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
        TorchUnrolledRun.Run? run;
        try
        {
            run = TorchUnrolledRun.Of(graph, given);
        }
        catch (ArgumentException)
        {
            return null;
        }
        if (run is null) return null;
        var (order, shapes) = (run.Order, run.Shapes);
        if (order.Any(n => n.Outputs.Any(o => o.Length > 0 && !shapes.ContainsKey(o)))) return null;

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
            : shapes.TryGetValue(value, out var v) ? TorchStrides.Contiguous(v.Shape) : null;
        bool Contiguous(string value) => value.Length > 0 && shapes.TryGetValue(value, out var v) && StridesOf(value) is { } s && TorchStrides.IsContiguous(v.Shape, s);
        foreach (var node in order)
            for (int o = 0; o < node.Outputs.Count; o++)
            {
                var output = node.Outputs[o];
                if (output.Length == 0) continue;
                for (int i = 0; i < node.Inputs.Count; i++)
                    if (node.Inputs[i].Length > 0 && Views(node, i, o, shapes)
                        && !(node.OpType.StartsWith("Reduce", StringComparison.Ordinal)
                             && shapes.TryGetValue(node.Inputs[i], out var reduced) && reduced.Bytes != shapes[output].Bytes))
                    {
                        var viewed = StridesOf(node.Inputs[i]) is { } inputStrides && shapes.TryGetValue(node.Inputs[i], out var viewedShape) && shapes.TryGetValue(output, out var viewShape)
                            ? TorchStrides.OfView(node.OpType, node.Attributes.FirstOrDefault(a => a.Name == "perm")?.Ints, viewedShape.Shape, inputStrides, viewShape.Shape)
                            : null;
                        if (viewed is null && node.OpType is "Reshape" or "Flatten" or "Squeeze" or "Unsqueeze") break;
                        if (viewed is not null) strides[output] = viewed;
                        root[output] = RootOf(node.Inputs[i]);
                        break;
                    }
                // An element-wise result of memory of its own lies as its operands do, or as the
                // operand it is written over, which torch writes over only where it is contiguous.
                if (root.ContainsKey(output) || intoInput.ContainsKey(output) || !TorchStrides.LaidOutAsOperands.Contains(node.OpType)
                    || !OutputAliasProof.IsStandard(node) || !shapes.TryGetValue(output, out var made)) continue;
                if (over.TryGetValue(output, out var target) && Contiguous(node.Inputs[target])
                    && shapes[node.Inputs[target]] is var written && written.ElementType == made.ElementType && written.Shape.SequenceEqual(made.Shape)) continue;
                var operands = node.Inputs.Where(i => i.Length > 0 && shapes.ContainsKey(i) && StridesOf(i) is not null)
                    .Select(i => ((IReadOnlyList<long>)shapes[i].Shape, (IReadOnlyList<long>)StridesOf(i)!)).ToList();
                var laid = TorchStrides.OfElementwise(operands, made.Shape);
                if (!TorchStrides.IsContiguous(made.Shape, laid)) strides[output] = laid;
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
                if (over.TryGetValue(output, out var slot) && node.Inputs[slot].Length > 0 && Contiguous(node.Inputs[slot]))
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
            peak = Math.Max(peak, live + Scratch(node, shapes, producer, StridesOf, writtenOver, onHost));
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

    /// <summary>
    /// What the translation's kernel for <paramref name="node"/> holds of its own while it runs,
    /// measured operator by operator through torch's profiler: a <c>LayerNormalization</c> its
    /// centered values' squares where it is written over its operand
    /// (<paramref name="writtenOver"/>), and otherwise its centered values and one value more, each of
    /// its input's size; on the host, where oneDNN computes in a layout of its own, a convolution a
    /// copy of its output, a convolution of two transposed views — the gradient of another's weights,
    /// which the translation computes as torch does — a copy of each of the values they view, and a
    /// transposed convolution a copy of its input; a <c>MatMul</c> the copies it makes of operands it
    /// cannot read where they lie (<see cref="TorchStrides.MatMulCopies"/>); a recurrent layer — an
    /// <c>LSTM</c>, <c>GRU</c> or <c>RNN</c> — one value of its hidden states' size, the list of each
    /// step's or each direction's outputs it stacks into the next.
    /// </summary>
    internal static long Scratch(NodeProto node, IReadOnlyDictionary<string, PlacementShapes.Value> shapes,
        IReadOnlyDictionary<string, NodeProto> producer, Func<string, long[]?> stridesOf, bool writtenOver, bool onHost)
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
            case "LSTM" or "GRU" or "RNN" when shapes.TryGetValue(node.Inputs[0], out var x) && x.Shape.Length == 3 && node.Inputs.Count > 2
                    && shapes.TryGetValue(node.Inputs[2], out var r) && r.Shape.Length == 3:
            {
                var batchFirst = (node.Attributes.FirstOrDefault(at => at.Name == "layout")?.I ?? 0) != 0;
                return x.Shape[batchFirst ? 1 : 0] * x.Shape[batchFirst ? 0 : 1] * r.Shape[0] * r.Shape[2] * PlacementShapes.ElementBytes(x.ElementType);
            }
            case "MatMul" when node.Inputs.Count == 2 && shapes.TryGetValue(node.Inputs[0], out var a) && shapes.TryGetValue(node.Inputs[1], out var b)
                    && stridesOf(node.Inputs[0]) is { } aStrides && stridesOf(node.Inputs[1]) is { } bStrides:
                return TorchStrides.MatMulCopies(a.Shape, aStrides, b.Shape, bStrides, !onHost) * PlacementShapes.ElementBytes(a.ElementType);
            default:
                return 0;
        }
    }

    /// <summary>
    /// Whether the translation hands <paramref name="node"/>'s output <paramref name="output"/> back
    /// over its input <paramref name="input"/>: a slice, a reshape, a transpose, an expansion or a
    /// split of its first input, a cast to the type it has, a one-input sum, mean, maximum or minimum, a
    /// clip without bounds, a reduction — of no axis, as the caller tells by size — and a branch, which
    /// may hand back what it captured (<see cref="PlacementMemory.PyTorch"/>), as anything the model
    /// does not know may. Every other operator computes a result of its own.
    /// </summary>
    private static bool Views(NodeProto node, int input, int output, IReadOnlyDictionary<string, PlacementShapes.Value> shapes)
    {
        if (!OutputAliasProof.IsStandard(node)) return PlacementMemory.PyTorch.SharesUnlessPlaced(node, input, output);
        return node.OpType switch
        {
            "Slice" or "Identity" or "Reshape" or "Squeeze" or "Unsqueeze" or "Flatten" or "Transpose" or "Expand" or "Split" => input == 0,
            "Cast" => input == 0 && shapes.TryGetValue(node.Inputs[0], out var from) && shapes.TryGetValue(node.Outputs[output], out var to)
                      && from.ElementType == to.ElementType,
            "Max" or "Min" or "Sum" or "Mean" or "Clip" or "If" => PlacementMemory.PyTorch.SharesUnlessPlaced(node, input, output),
            _ => node.OpType.StartsWith("Reduce", StringComparison.Ordinal) && PlacementMemory.PyTorch.SharesUnlessPlaced(node, input, output),
        };
    }

    /// <summary>Whether the translation writes <paramref name="node"/>'s output into the input it
    /// is paired with: a two-input <c>Add</c>, <c>Sub</c>, <c>Mul</c> or <c>Div</c>, torch's
    /// <c>out=</c> forms of which the translation's output aliasing calls.</summary>
    private static bool WritesInto(NodeProto node)
        => OutputAliasProof.IsStandard(node) && node.OpType is "Add" or "Sub" or "Mul" or "Div" && node.Inputs.Count == 2;
}

using System.Collections.Concurrent;
using Shorokoo.Core.Factory.IR;

namespace Shorokoo.Core.Backends;

/// <summary>
/// The check a run makes, before it is fed, of the integer inputs of its graph that index
/// something: a <c>Gather</c>'s indices into its table, and the class targets of a
/// <c>SoftmaxCrossEntropyLoss</c> or <c>NegativeLogLikelihoodLoss</c>. An input reaches what it
/// indexes as it is fed, or through the ops that only reshape or retype it (<c>Identity</c>,
/// <c>Cast</c> between integer types, <c>Reshape</c>, <c>Squeeze</c>, <c>Unsqueeze</c>,
/// <c>Flatten</c>). A <c>Gather</c> index lies in <c>[-rows, rows - 1]</c>, ONNX's range, where a
/// negative index counts from the end; a class target lies in <c>[0, classes - 1]</c> or is the
/// loss's <c>ignore_index</c>. An element outside is refused with an
/// <see cref="IndexOutOfRangeInputException"/> naming the input, the element and the range, on
/// every backend alike: left to the backend, one throws from inside a node the user never wrote,
/// another reads a zero row and trains on.
///
/// <para>It reads what the host holds and nothing else: an input in a device's memory is not
/// copied back to be checked, and an index the graph computes is not checked. The extent of what
/// an input indexes comes from the shapes the run is fed — straight from a fed table or an
/// initializer, and otherwise evaluated over the part of the graph that computes it
/// (<see cref="PlacementShapes"/>), once per set of input shapes. An extent nothing here can
/// evaluate leaves its input unchecked.</para>
/// </summary>
internal sealed class IndexInputCheck
{
    private const int Int32 = 6, Int64 = 7;

    // How many sets of input shapes the extents are kept evaluated for.
    private const int EvaluatedShapes = 16;

    private static readonly HashSet<string> PassThrough = new(StringComparer.Ordinal)
    {
        "Identity", "Cast", "Reshape", "Squeeze", "Unsqueeze", "Flatten",
    };

    /// <summary>The input <paramref name="Input"/> (by the run's name for it) indexes axis
    /// <paramref name="Axis"/> of the value <paramref name="Indexed"/>: a table's rows, or a loss's
    /// classes where <paramref name="ClassTarget"/>.</summary>
    private sealed record Use(string Input, string Indexed, long Axis, bool ClassTarget, long? IgnoreIndex, string Op);

    private readonly Use[] _uses;

    // The run's name of each graph input, by the graph's name for it.
    private readonly Dictionary<string, string> _originalByGraphInput;

    // Each initializer's dims a use indexes.
    private readonly Dictionary<string, long[]> _initializerDims;

    // The part of the graph that computes what a use indexes, where that is neither an input nor
    // an initializer, with every large tensor in it reduced to its shape; the graph inputs it reads,
    // and what it evaluated per set of their shapes.
    private readonly GraphProto? _extents;
    private readonly (string GraphName, string Original, int ElementType)[] _extentInputs;
    private readonly ConcurrentDictionary<string, Dictionary<string, PlacementShapes.Value>> _evaluated = new(StringComparer.Ordinal);

    private IndexInputCheck(Use[] uses, Dictionary<string, string> originalByGraphInput,
        Dictionary<string, long[]> initializerDims, GraphProto? extents,
        (string GraphName, string Original, int ElementType)[] extentInputs)
    {
        _uses = uses;
        _originalByGraphInput = originalByGraphInput;
        _initializerDims = initializerDims;
        _extents = extents;
        _extentInputs = extentInputs;
    }

    /// <summary>
    /// The check for runs of <paramref name="graph"/>, whose first inputs are the run's
    /// <paramref name="originalInputNames"/> by position; null where no input of it indexes anything.
    /// </summary>
    internal static IndexInputCheck? Of(GraphProto graph, IReadOnlyList<string> originalInputNames)
    {
        var consumers = new Dictionary<string, List<(NodeProto Node, int Slot)>>(StringComparer.Ordinal);
        foreach (var node in graph.Nodes)
            for (int slot = 0; slot < node.Inputs.Count; slot++)
            {
                var name = node.Inputs[slot];
                if (name.Length == 0) continue;
                if (!consumers.TryGetValue(name, out var list)) consumers[name] = list = [];
                list.Add((node, slot));
            }

        var originalByGraphInput = new Dictionary<string, string>(StringComparer.Ordinal);
        List<Use> uses = [];
        for (int i = 0; i < originalInputNames.Count && i < graph.Inputs.Count; i++)
        {
            var input = graph.Inputs[i];
            originalByGraphInput[input.Name] = originalInputNames[i];
            if (input.Type?.TensorType?.ElemType is not (Int32 or Int64)) continue;

            var seen = new HashSet<string>(StringComparer.Ordinal);
            var pending = new Stack<string>([input.Name]);
            while (pending.TryPop(out var value))
            {
                if (!seen.Add(value) || !consumers.TryGetValue(value, out var reads)) continue;
                foreach (var (node, slot) in reads)
                {
                    if (node.Domain is not ("" or "ai.onnx")) continue;
                    if (slot == 0 && PassThrough.Contains(node.OpType)
                        && (node.OpType != "Cast" || Attr(node, "to") is Int32 or Int64))
                        pending.Push(node.Outputs[0]);
                    else if (slot == 1 && node.OpType == "Gather")
                        uses.Add(new Use(originalInputNames[i], node.Inputs[0], Attr(node, "axis") ?? 0,
                            ClassTarget: false, IgnoreIndex: null, node.OpType));
                    else if (slot == 1 && node.OpType is "SoftmaxCrossEntropyLoss" or "NegativeLogLikelihoodLoss")
                        uses.Add(new Use(originalInputNames[i], node.Inputs[0], 1,
                            ClassTarget: true, Attr(node, "ignore_index"), node.OpType));
                }
            }
        }
        if (uses.Count == 0) return null;

        var initializers = graph.Initializers.ToDictionary(t => t.Name, StringComparer.Ordinal);
        var initializerDims = new Dictionary<string, long[]>(StringComparer.Ordinal);
        List<string> computed = [];
        foreach (var use in uses)
        {
            if (originalByGraphInput.ContainsKey(use.Indexed)) continue;
            if (initializers.TryGetValue(use.Indexed, out var initializer)) initializerDims[use.Indexed] = initializer.Dims ?? [];
            else computed.Add(use.Indexed);
        }

        var (extents, extentInputs) = computed.Count == 0 ? (null, []) : ExtentsGraph(graph, computed, initializers, originalByGraphInput);
        // A class target first: its range is the tighter, and a gradient's Gather of the same
        // target over the loss's class weights would otherwise answer for it.
        return new IndexInputCheck([.. uses.Distinct().OrderBy(u => !u.ClassTarget)], originalByGraphInput,
            initializerDims, extents, extentInputs);
    }

    /// <summary>
    /// Refuses <paramref name="inputs"/> where one the host holds has an element outside the range
    /// of what it indexes.
    /// </summary>
    /// <exception cref="IndexOutOfRangeInputException">An element is outside its range.</exception>
    internal void Check(IReadOnlyList<NamedModelParam> inputs)
    {
        var fed = new Dictionary<string, NamedModelParam>(StringComparer.Ordinal);
        foreach (var input in inputs) fed.TryAdd(input.ParamName, input);
        Dictionary<string, PlacementShapes.Value>? evaluated = null;

        foreach (var use in _uses)
        {
            if (!fed.TryGetValue(use.Input, out var param) || HostTensor(param) is not { } tensor) continue;
            var shape = ShapeOf(use.Indexed, fed, ref evaluated);
            if (shape is null) continue;
            var axis = use.Axis < 0 ? use.Axis + shape.Length : use.Axis;
            if (axis < 0 || axis >= shape.Length) continue;

            var extent = shape[axis];
            var (minimum, maximum) = use.ClassTarget ? (0L, extent - 1) : (-extent, extent - 1);
            var (at, value) = FirstOutside(tensor, minimum, maximum, use.IgnoreIndex);
            if (at < 0) continue;

            var indexes = use.ClassTarget
                ? $"the {extent} classes of a {use.Op}"
                : $"the {extent} entries along axis {axis} of a {use.Op}'s table";
            throw new IndexOutOfRangeInputException(param.Label ?? $"input '{param.ParamName}'",
                PositionOf(at, tensor.Shape.Dims), value, minimum, maximum, use.IgnoreIndex, indexes);
        }
    }

    private static TensorData? HostTensor(NamedModelParam param)
        => param is TensorDataModelParam && param.ToTensorData() is { IsDisposed: false, IsHostResident: true } tensor
           && (tensor.DType.IsSameElementTypeAs(DType.Int64) || tensor.DType.IsSameElementTypeAs(DType.Int32))
            ? tensor
            : null;

    private long[]? ShapeOf(string value, Dictionary<string, NamedModelParam> fed,
        ref Dictionary<string, PlacementShapes.Value>? evaluated)
    {
        if (_originalByGraphInput.TryGetValue(value, out var original))
            return fed.TryGetValue(original, out var param) && param is TensorDataModelParam
                ? param.ToTensorData().Shape.Dims
                : null;
        if (_initializerDims.TryGetValue(value, out var dims)) return dims;
        if (_extents is null) return null;
        evaluated ??= Evaluated(fed);
        return evaluated.TryGetValue(value, out var made) ? made.Shape : null;
    }

    /// <summary>The values of the extents graph for the shapes <paramref name="fed"/> come in,
    /// evaluated the first time a run comes with them.</summary>
    private Dictionary<string, PlacementShapes.Value> Evaluated(Dictionary<string, NamedModelParam> fed)
    {
        var given = new Dictionary<string, (long[] Shape, int ElementType)>(StringComparer.Ordinal);
        foreach (var (graphName, original, elementType) in _extentInputs)
            if (fed.TryGetValue(original, out var param) && param is TensorDataModelParam)
                given[graphName] = (param.ToTensorData().Shape.Dims, elementType);
        var key = string.Join(";", _extentInputs.Select(i => given.TryGetValue(i.GraphName, out var g) ? string.Join(",", g.Shape) : "?"));
        if (_evaluated.TryGetValue(key, out var values)) return values;
        // A graph fed ever-changing shapes keeps the latest few rather than every one it was fed.
        if (_evaluated.Count >= EvaluatedShapes) _evaluated.Clear();
        return _evaluated.GetOrAdd(key, _ => PlacementShapes.Evaluate(_extents!, given));
    }

    /// <summary>The flat index and value of the first element of <paramref name="tensor"/> outside
    /// <c>[minimum, maximum]</c> and not <paramref name="ignoreIndex"/>, or -1.</summary>
    private static (long At, long Value) FirstOutside(TensorData tensor, long minimum, long maximum, long? ignoreIndex)
        => tensor.Reading(() =>
        {
            if (tensor.DType.IsSameElementTypeAs(DType.Int64))
            {
                var elements = tensor.AccessMemory<long>();
                for (int i = 0; i < elements.Length; i++)
                    if ((elements[i] < minimum || elements[i] > maximum) && elements[i] != ignoreIndex) return (i, elements[i]);
            }
            else
            {
                var elements = tensor.AccessMemory<int>();
                for (int i = 0; i < elements.Length; i++)
                    if ((elements[i] < minimum || elements[i] > maximum) && elements[i] != ignoreIndex) return (i, (long)elements[i]);
            }
            return (-1L, 0L);
        });

    private static long[] PositionOf(long at, long[] dims)
    {
        var position = new long[dims.Length];
        for (int d = dims.Length - 1; d >= 0; d--)
        {
            position[d] = dims[d] == 0 ? 0 : at % dims[d];
            at = dims[d] == 0 ? at : at / dims[d];
        }
        return position;
    }

    /// <summary>
    /// The nodes of <paramref name="graph"/> that compute <paramref name="values"/>, in order, over
    /// its inputs and the initializers they read, each tensor of more elements than a shape is made
    /// of reduced to its shape so that nothing of a weight is kept; and the graph inputs they read.
    /// </summary>
    private static (GraphProto, (string, string, int)[]) ExtentsGraph(
        GraphProto graph, List<string> values, Dictionary<string, TensorProto> initializers,
        Dictionary<string, string> originalByGraphInput)
    {
        var producer = new Dictionary<string, NodeProto>(StringComparer.Ordinal);
        foreach (var node in graph.Nodes)
            foreach (var output in node.Outputs)
                if (output.Length > 0) producer[output] = node;

        var needed = new HashSet<NodeProto>(ReferenceEqualityComparer.Instance);
        var read = new HashSet<string>(StringComparer.Ordinal);
        var pending = new Stack<string>(values);
        while (pending.TryPop(out var value))
        {
            if (!read.Add(value) || !producer.TryGetValue(value, out var node) || !needed.Add(node)) continue;
            foreach (var input in node.Inputs)
                if (input.Length > 0) pending.Push(input);
        }

        var extents = new GraphProto { Name = graph.Name };
        foreach (var node in graph.Nodes)
            if (needed.Contains(node)) extents.Nodes.Add(ShapeOnly(node));
        foreach (var name in read)
            if (initializers.TryGetValue(name, out var initializer)) extents.Initializers.Add(ShapeOnly(initializer));
        extents.Inputs.AddRange(graph.Inputs);
        extents.ValueInfoes.AddRange(graph.ValueInfoes);
        extents.Outputs.AddRange(graph.Outputs);

        var inputs = graph.Inputs
            .Where(i => read.Contains(i.Name) && originalByGraphInput.ContainsKey(i.Name))
            .Select(i => (i.Name, originalByGraphInput[i.Name], i.Type?.TensorType?.ElemType ?? 0))
            .ToArray();
        return (extents, inputs);
    }

    private static TensorProto ShapeOnly(TensorProto tensor)
        => (tensor.Dims ?? []).Aggregate(1L, (a, d) => a * d) <= PlacementShapes.SmallInts
            ? tensor
            : new TensorProto { Name = tensor.Name, Dims = tensor.Dims ?? [], data_type = tensor.data_type };

    private static NodeProto ShapeOnly(NodeProto node)
    {
        if (node.OpType != "Constant" || !node.Attributes.Any(a => a.T is { } t && !ReferenceEquals(ShapeOnly(t), t)))
            return node;
        var copy = new NodeProto { Name = node.Name, OpType = node.OpType, Domain = node.Domain };
        copy.Inputs.AddRange(node.Inputs);
        copy.Outputs.AddRange(node.Outputs);
        foreach (var attribute in node.Attributes)
            copy.Attributes.Add(attribute.T is null ? attribute
                : new AttributeProto { Name = attribute.Name, Type = attribute.Type, T = ShapeOnly(attribute.T) });
        return copy;
    }

    private static long? Attr(NodeProto node, string name)
        => node.Attributes.FirstOrDefault(a => a.Name == name) is { } a ? a.I : null;
}

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
/// copied back to be checked, and an index the graph computes is not checked. Each input is
/// scanned once per run for its smallest and largest element, whatever it indexes. The extent of
/// what it indexes is a fed table's or an initializer's dimension, or else one evaluated from the
/// shapes the run is fed over the part of the graph that computes it (<see cref="PlacementShapes"/>),
/// counting only what that infers and never a shape the graph states. An evaluated extent is kept
/// and evaluated again only for a run whose elements it does not take. An extent that cannot be
/// evaluated, the first time it is asked for, leaves its input unchecked from then on.</para>
/// </summary>
internal sealed class IndexInputCheck
{
    private const int Int32 = 6, Int64 = 7;

    private static readonly HashSet<string> PassThrough = new(StringComparer.Ordinal)
    {
        "Identity", "Cast", "Reshape", "Squeeze", "Unsqueeze", "Flatten",
    };

    /// <summary>The input <paramref name="Input"/> (by the run's name for it) indexes axis
    /// <paramref name="Axis"/> of the value <paramref name="Indexed"/>: a table's entries, or a
    /// loss's classes where <paramref name="ClassTarget"/>.</summary>
    private sealed record Use(string Input, string Indexed, long Axis, bool ClassTarget, long? IgnoreIndex);

    private readonly Use[] _uses;

    // The run's name of each graph input, by the graph's name for it.
    private readonly Dictionary<string, string> _originalByGraphInput;

    // Each initializer's dims a use indexes.
    private readonly Dictionary<string, long[]> _initializerDims;

    // The part of the graph that computes what a use indexes, where that is neither an input nor an
    // initializer, with every large tensor in it reduced to its shape; and the graph inputs it reads.
    private readonly GraphProto? _extents;
    private readonly (string GraphName, string Original, int ElementType)[] _extentInputs;

    // Per use, the extent last evaluated for it: 0 where none was yet, Unevaluable where none can be.
    private readonly long[] _evaluatedExtent;
    private const long Unevaluable = -1;

    private IndexInputCheck(Use[] uses, Dictionary<string, string> originalByGraphInput,
        Dictionary<string, long[]> initializerDims, GraphProto? extents,
        (string GraphName, string Original, int ElementType)[] extentInputs)
    {
        _uses = uses;
        _originalByGraphInput = originalByGraphInput;
        _initializerDims = initializerDims;
        _extents = extents;
        _extentInputs = extentInputs;
        _evaluatedExtent = new long[uses.Length];
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
                    if (slot == 0 && node.Outputs.Count > 0 && PassThrough.Contains(node.OpType)
                        && (node.OpType != "Cast" || Attr(node, "to") is Int32 or Int64))
                        pending.Push(node.Outputs[0]);
                    else if (slot == 1 && node.OpType == "Gather")
                        uses.Add(new Use(originalInputNames[i], node.Inputs[0], Attr(node, "axis") ?? 0,
                            ClassTarget: false, IgnoreIndex: null));
                    else if (slot == 1 && node.OpType is "SoftmaxCrossEntropyLoss" or "NegativeLogLikelihoodLoss")
                        uses.Add(new Use(originalInputNames[i], node.Inputs[0], 1,
                            ClassTarget: true, Attr(node, "ignore_index")));
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
        var scanned = new Dictionary<(string Input, long? IgnoreIndex), (long Min, long Max)?>();
        Dictionary<string, PlacementShapes.Value>? evaluated = null;

        for (int u = 0; u < _uses.Length; u++)
        {
            var use = _uses[u];
            if (!fed.TryGetValue(use.Input, out var param) || HostTensor(param) is not { } tensor) continue;
            var key = (use.Input, use.IgnoreIndex);
            if (!scanned.TryGetValue(key, out var span)) scanned[key] = span = MinMax(tensor, use.IgnoreIndex);
            if (span is not var (min, max)) continue;

            long extent, axis;
            if (DirectShape(use.Indexed, fed) is { } shape)
            {
                if (Axis(use, shape.Length) is not long direct) continue;
                (extent, axis) = (shape[direct], direct);
            }
            else
            {
                var known = Volatile.Read(ref _evaluatedExtent[u]);
                if (known < 0 || known > 0 && Within(use, known, min, max)) continue;
                evaluated ??= Evaluated(fed);
                if (evaluated?.TryGetValue(use.Indexed, out var value) != true
                    || Axis(use, value!.Shape.Length) is not long computed)
                {
                    Volatile.Write(ref _evaluatedExtent[u], Unevaluable);
                    continue;
                }
                (extent, axis) = (value.Shape[computed], computed);
                Volatile.Write(ref _evaluatedExtent[u], extent);
            }
            if (Within(use, extent, min, max)) continue;

            var (minimum, maximum) = Range(use, extent);
            var (at, offending) = FirstOutside(tensor, minimum, maximum, use.IgnoreIndex);
            var indexes = use.ClassTarget
                ? $"the {extent} classes of the loss"
                : $"the {extent} entries along axis {axis} of a Gather's table";
            throw new IndexOutOfRangeInputException(param.Label ?? $"input '{param.ParamName}'",
                PositionOf(at, tensor.Shape.Dims), offending, minimum, maximum, use.IgnoreIndex, indexes);
        }
    }

    private static (long Minimum, long Maximum) Range(Use use, long extent)
        => use.ClassTarget ? (0L, extent - 1) : (-extent, extent - 1);

    private static bool Within(Use use, long extent, long min, long max)
        => Range(use, extent) is var (minimum, maximum) && min >= minimum && max <= maximum;

    private static long? Axis(Use use, int rank)
        => (use.Axis < 0 ? use.Axis + rank : use.Axis) is var axis && axis >= 0 && axis < rank ? axis : null;

    private static TensorData? HostTensor(NamedModelParam param)
        => param is TensorDataModelParam && param.ToTensorData() is { IsDisposed: false, IsHostResident: true } tensor
           && (tensor.DType.IsSameElementTypeAs(DType.Int64) || tensor.DType.IsSameElementTypeAs(DType.Int32))
            ? tensor
            : null;

    /// <summary>The shape of <paramref name="value"/> where it is a graph input the run is fed or an
    /// initializer; null otherwise.</summary>
    private long[]? DirectShape(string value, Dictionary<string, NamedModelParam> fed)
    {
        if (_initializerDims.TryGetValue(value, out var dims)) return dims;
        return _originalByGraphInput.TryGetValue(value, out var original)
               && fed.TryGetValue(original, out var param) && param is TensorDataModelParam
            ? param.ToTensorData().Shape.Dims
            : null;
    }

    /// <summary>The values of the extents graph for the shapes <paramref name="fed"/> come in, or
    /// null where they cannot be evaluated.</summary>
    private Dictionary<string, PlacementShapes.Value>? Evaluated(Dictionary<string, NamedModelParam> fed)
    {
        if (_extents is null) return null;
        var given = new Dictionary<string, (long[] Shape, int ElementType)>(StringComparer.Ordinal);
        foreach (var (graphName, original, elementType) in _extentInputs)
            if (fed.TryGetValue(original, out var param) && param is TensorDataModelParam)
                given[graphName] = (param.ToTensorData().Shape.Dims, elementType);
        try
        {
            return PlacementShapes.Evaluate(_extents, given);
        }
        catch (Exception)
        {
            // An extent that cannot be evaluated leaves its input unchecked: the check never fails
            // a run the backend would take.
            return null;
        }
    }

    /// <summary>The smallest and largest element of <paramref name="tensor"/> that is not
    /// <paramref name="ignoreIndex"/>, or null where there is none.</summary>
    private static (long Min, long Max)? MinMax(TensorData tensor, long? ignoreIndex)
        => tensor.Reading<(long Min, long Max)?>(() =>
        {
            long min = long.MaxValue, max = long.MinValue;
            if (tensor.DType.IsSameElementTypeAs(DType.Int64))
            {
                foreach (var element in tensor.AccessMemory<long>())
                    if (element != ignoreIndex) (min, max) = (Math.Min(min, element), Math.Max(max, element));
            }
            else
            {
                foreach (long element in tensor.AccessMemory<int>())
                    if (element != ignoreIndex) (min, max) = (Math.Min(min, element), Math.Max(max, element));
            }
            GC.KeepAlive(tensor);
            return min <= max ? (min, max) : null;
        });

    /// <summary>The flat index and value of the first element of <paramref name="tensor"/> outside
    /// <c>[minimum, maximum]</c> and not <paramref name="ignoreIndex"/>.</summary>
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
            GC.KeepAlive(tensor);
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
    /// its inputs and the initializers they read, and the graph inputs they read. Each tensor of more
    /// elements than a shape is made of is reduced to its shape, so that nothing of a weight is kept;
    /// a node holding a subgraph is left out, its outputs then unknown; and no shape the graph states
    /// for a value is carried, so that only what is inferred counts.
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
            if (!read.Add(value) || !producer.TryGetValue(value, out var node)
                || node.Attributes.Any(a => a.G is not null || a.Graphs is { Count: > 0 }) || !needed.Add(node)) continue;
            foreach (var input in node.Inputs)
                if (input.Length > 0) pending.Push(input);
        }

        var extents = new GraphProto { Name = graph.Name };
        foreach (var node in graph.Nodes)
            if (needed.Contains(node)) extents.Nodes.Add(ShapeOnly(node));
        foreach (var name in read)
            if (initializers.TryGetValue(name, out var initializer)) extents.Initializers.Add(ShapeOnly(initializer));
        extents.Inputs.AddRange(graph.Inputs.Where(i => read.Contains(i.Name)));

        var inputs = extents.Inputs
            .Where(i => originalByGraphInput.ContainsKey(i.Name))
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
        if (!node.Attributes.Any(a => a.T is { } t && !ReferenceEquals(ShapeOnly(t), t)))
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

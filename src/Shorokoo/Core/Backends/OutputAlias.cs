using Shorokoo.Core.Factory.IR;

namespace Shorokoo.Core.Backends;

/// <summary>
/// An output a run may write straight into the memory of one of its inputs — <see cref="Input"/> —
/// instead of into memory of its own: output aliasing, which is how a run reuses the memory of an
/// input it consumed. ONNX Runtime keeps every input of a run until the run ends, so a consumed
/// input's memory cannot come back part-way through to be used for something else; binding an
/// output into it is the one way that memory does work in the same run.
///
/// <para>That is correct only when nothing reads the input after the output is written. A lowering
/// proposes the pairs it proves so over its own graph (<see cref="OutputAliasProof"/>) — the
/// training rig proposes each updated state output with the state input it replaces — and a
/// backend binds a pair only when the input was consumed by the run, and the memory, shape and
/// element type of the two agree. Names are the model's own: the graph input and graph output as
/// the session names them.</para>
/// </summary>
/// <param name="Output">The graph output written into the input's memory.</param>
/// <param name="Input">The graph input whose memory the output is written into.</param>
public readonly record struct OutputAlias(string Output, string Input);

/// <summary>
/// Which outputs of an ONNX graph may be written into the memory of which of its inputs — the proof
/// behind output aliasing (<see cref="OutputAlias"/>). It answers for the graph it is given, so a
/// backend that rewrites a graph before running it asks again of the graph it runs.
///
/// <para>An output <c>O</c>, written by the node <c>P</c>, may be written into the memory of an
/// input <c>I</c> when, in the graph as given:</para>
/// <list type="bullet">
/// <item><c>O</c> is a graph output written by a node of the graph itself, and not by one holding a
/// subgraph; <c>I</c> is a graph input that is not an initializer, and neither it nor any view of
/// it is a graph output.</item>
/// <item>Where the graph states their types, the two have one element type — and, where it states
/// both shapes in full, one shape.</item>
/// <item>Every node reading <c>I</c>'s memory has run before <c>P</c> writes: it is an ancestor of
/// <c>P</c>. A node reads <c>I</c>'s memory when it reads <c>I</c> or a view of it — an output a
/// runtime may hand back in the memory of one of the node's inputs: that of an <c>Identity</c>,
/// <c>Reshape</c>, <c>Squeeze</c>, <c>Unsqueeze</c>, <c>Flatten</c> and a few others over their first
/// input, and the running mean and variance of a <c>BatchNormalization</c> over the mean and
/// variance it was given. Ancestry counts only the edges a runtime cannot fold away: an edge into
/// the standard <c>Shape</c> or <c>Size</c> — which read no memory and are no readers — orders
/// nothing once the shape is known when the session is built, so it does not count.</item>
/// <item><c>P</c> itself reads <c>I</c> only as the first operand of a two-input <c>Add</c>,
/// <c>Sub</c>, <c>Mul</c> or <c>Div</c>, which reads each element before writing the same element
/// of its output — the in-place form ONNX Runtime uses for these operators itself. With <c>O</c>
/// and <c>I</c> of one shape, that operand is not broadcast.</item>
/// <item>No node holding a subgraph refers to <c>I</c> or a view of it: what a subgraph hands back
/// may be the memory it was given.</item>
/// </list>
/// <para>Each input backs at most one output, and each output at most one input — the first
/// candidate proved takes both. Everything else is refused: an aliasing this cannot prove is not
/// made.</para>
/// </summary>
public static class OutputAliasProof
{
    /// <summary>
    /// The candidates the serialized ONNX model <paramref name="model"/> proves, in the order they
    /// were given — see the class for the rule.
    /// </summary>
    /// <exception cref="ArgumentNullException">An argument is null.</exception>
    public static IReadOnlyList<OutputAlias> Prove(byte[] model, IEnumerable<OutputAlias> candidates)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(candidates);
        ModelProto parsed;
        using (var stream = new MemoryStream(model, writable: false))
            parsed = ProtoBuf.Serializer.Deserialize<ModelProto>(stream);
        return parsed.Graph is { } graph ? Prove(graph, candidates) : [];
    }

    /// <summary>The candidates <paramref name="graph"/> proves, in the order they were
    /// given.</summary>
    internal static IReadOnlyList<OutputAlias> Prove(GraphProto graph, IEnumerable<OutputAlias> candidates)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(candidates);
        var index = new GraphIndex(graph);
        var proven = new List<OutputAlias>();
        var inputs = new HashSet<string>(StringComparer.Ordinal);
        var outputs = new HashSet<string>(StringComparer.Ordinal);
        foreach (var candidate in candidates)
        {
            if (inputs.Contains(candidate.Input) || outputs.Contains(candidate.Output)) continue;
            if (!index.Proves(candidate)) continue;
            inputs.Add(candidate.Input);
            outputs.Add(candidate.Output);
            proven.Add(candidate);
        }
        return proven;
    }

    /// <summary>
    /// Whether a runtime may hand back <paramref name="node"/>'s input at position
    /// <paramref name="input"/> as its output at position <paramref name="output"/>, the two being
    /// one buffer: whoever reads that output reads the input. By position, because that is how
    /// ONNX Runtime declares it on a kernel, and binds it even where the input is a graph input.
    /// Its CPU and CUDA kernels in both the standard and Microsoft domains declare the pairs below:
    /// an output 0 over input 0, <c>BatchNormalization</c> in training mode writing its running mean
    /// and variance into the buffers of the mean and variance it was given, and, where the runtime
    /// is built with NCCL, <c>AllReduce</c> handing back each of its inputs. <c>Dropout</c>, an
    /// identity outside training, is counted as one too, and a sequence built by
    /// <c>SequenceConstruct</c> or <c>SequenceInsert</c> may hold the memory of any tensor it was
    /// given. Named whatever their domain, which only ever widens what counts as a reader.
    /// </summary>
    private static bool Shares(NodeProto node, int input, int output) => node.OpType switch
    {
        "Identity" or "Reshape" or "Squeeze" or "Unsqueeze" or "Flatten" or "Dropout" or "ExpandDims"
            or "Optional" or "OptionalGetElement" => (input, output) is (0, 0),
        "BatchNormalization" => (input, output) is (3, 1) or (4, 2),
        "SequenceConstruct" or "SequenceInsert" => output == 0,
        "AllReduce" => input == output,
        _ => false,
    };

    // Whether the node reads a tensor's shape and none of its memory: the standard Shape and Size,
    // and nothing else of those names -- an operator of another domain may read whatever it likes.
    private static bool ReadsOnlyAShape(NodeProto node) => IsStandard(node) && node.OpType is "Shape" or "Size";

    // The element-wise operators whose first operand the output may be written over: each element
    // is read before the same element of the output is written, and ONNX Runtime writes these in
    // place itself.
    private static readonly HashSet<string> InPlace = new(StringComparer.Ordinal) { "Add", "Sub", "Mul", "Div" };

    private static bool IsStandard(NodeProto node) => node.Domain is "" or "ai.onnx";

    private static bool HoldsSubgraph(NodeProto node)
        => node.Attributes.Any(a => a.G is not null || a.Graphs.Count > 0);

    /// <summary>One graph, indexed for the questions the proof asks of it.</summary>
    private sealed class GraphIndex
    {
        private readonly List<NodeProto> _nodes;
        private readonly Dictionary<string, int> _producer = new(StringComparer.Ordinal);
        private readonly Dictionary<string, List<int>> _consumers = new(StringComparer.Ordinal);
        private readonly HashSet<string> _referencedBySubgraphs = new(StringComparer.Ordinal);
        private readonly HashSet<string> _inputs = new(StringComparer.Ordinal);
        private readonly HashSet<string> _initializers = new(StringComparer.Ordinal);
        private readonly Dictionary<string, int> _outputs = new(StringComparer.Ordinal);
        private readonly Dictionary<string, TypeProto> _types = new(StringComparer.Ordinal);

        // A stamp per node for the ancestry walks, so each walk marks what it visited without
        // clearing an array first.
        private readonly int[] _visited;
        private int _stamp;

        internal GraphIndex(GraphProto graph)
        {
            _nodes = graph.Nodes;
            _visited = new int[_nodes.Count];
            foreach (var input in graph.Inputs) _inputs.Add(input.Name);
            foreach (var initializer in graph.Initializers) _initializers.Add(initializer.Name);
            foreach (var output in graph.Outputs)
                _outputs[output.Name] = _outputs.TryGetValue(output.Name, out var seen) ? seen + 1 : 1;
            foreach (var info in graph.Inputs.Concat(graph.Outputs).Concat(graph.ValueInfoes))
                if (info.Type is { } type) _types.TryAdd(info.Name, type);

            for (int n = 0; n < _nodes.Count; n++)
            {
                var node = _nodes[n];
                foreach (var output in node.Outputs)
                    if (output.Length > 0) _producer[output] = n;
                foreach (var input in node.Inputs)
                {
                    if (input.Length == 0) continue;
                    if (!_consumers.TryGetValue(input, out var readers)) _consumers[input] = readers = [];
                    readers.Add(n);
                }
                foreach (var attribute in node.Attributes)
                {
                    if (attribute.G is { } subgraph) ReferencedFrom(subgraph, _referencedBySubgraphs);
                    foreach (var each in attribute.Graphs) ReferencedFrom(each, _referencedBySubgraphs);
                }
            }
        }

        /// <summary>Whether this graph proves <paramref name="alias"/>.</summary>
        internal bool Proves(OutputAlias alias)
        {
            var (input, output) = (alias.Input, alias.Output);
            if (!_inputs.Contains(input) || _initializers.Contains(input)) return false;
            if (!_outputs.TryGetValue(output, out var listed) || listed != 1 || _inputs.Contains(output))
                return false;
            if (!_producer.TryGetValue(output, out var writer)) return false;
            var p = _nodes[writer];
            if (HoldsSubgraph(p) || !TypesAgree(input, output)) return false;

            // The input and every view of it, and the nodes that read any of them.
            var views = new HashSet<string>(StringComparer.Ordinal) { input };
            var readers = new HashSet<int>();
            var pending = new Queue<string>();
            pending.Enqueue(input);
            while (pending.TryDequeue(out var name))
            {
                if (_outputs.ContainsKey(name) || _referencedBySubgraphs.Contains(name)) return false;
                if (!_consumers.TryGetValue(name, out var consumers)) continue;
                foreach (var n in consumers)
                {
                    var node = _nodes[n];
                    if (ReadsOnlyAShape(node)) continue;
                    readers.Add(n);
                    for (int i = 0; i < node.Inputs.Count; i++)
                    {
                        if (node.Inputs[i] != name) continue;
                        for (int o = 0; o < node.Outputs.Count; o++)
                            if (node.Outputs[o].Length > 0 && Shares(node, i, o) && views.Add(node.Outputs[o]))
                                pending.Enqueue(node.Outputs[o]);
                    }
                }
            }

            if (readers.Remove(writer) && !WritesInPlace(p, views)) return false;
            return readers.Count == 0 || AllAncestorsOf(writer, readers);
        }

        /// <summary>Whether <paramref name="p"/> reads the input — through one of
        /// <paramref name="views"/> — only as it may while writing over it.</summary>
        private static bool WritesInPlace(NodeProto p, HashSet<string> views)
            => IsStandard(p) && InPlace.Contains(p.OpType)
               && p.Inputs.Count == 2 && p.Outputs.Count == 1
               && views.Contains(p.Inputs[0]) && !views.Contains(p.Inputs[1]);

        /// <summary>
        /// Whether every one of <paramref name="readers"/> is an ancestor of node
        /// <paramref name="writer"/>, walking back from it along the edges that order execution
        /// whatever a runtime folds: explicit inputs, except those of a node that reads only a
        /// shape.
        /// </summary>
        private bool AllAncestorsOf(int writer, HashSet<int> readers)
        {
            var stamp = ++_stamp;
            var missing = readers.Count;
            var pending = new Stack<int>();
            _visited[writer] = stamp;
            pending.Push(writer);
            while (pending.TryPop(out var n))
            {
                var node = _nodes[n];
                if (ReadsOnlyAShape(node)) continue;
                foreach (var input in node.Inputs)
                {
                    if (input.Length == 0 || !_producer.TryGetValue(input, out var producer)) continue;
                    if (_visited[producer] == stamp) continue;
                    _visited[producer] = stamp;
                    if (readers.Contains(producer) && --missing == 0) return true;
                    pending.Push(producer);
                }
            }
            return false;
        }

        /// <summary>
        /// Whether what the graph states of the two's types agrees: both tensors where it says what
        /// either is, of one element type where it states both, and of one shape where it states
        /// both in full. A type the graph leaves unstated agrees with anything — the backend checks
        /// the value it would bind against the output before binding it.
        /// </summary>
        private bool TypesAgree(string input, string output)
        {
            var inTensor = _types.TryGetValue(input, out var inType) ? inType.TensorType : null;
            var outTensor = _types.TryGetValue(output, out var outType) ? outType.TensorType : null;
            if (IsOtherThanTensor(inType) || IsOtherThanTensor(outType)) return false;
            if (inTensor is null || outTensor is null) return true;
            if (inTensor.ElemType != 0 && outTensor.ElemType != 0 && inTensor.ElemType != outTensor.ElemType)
                return false;
            return ConcreteShape(inTensor.Shape) is not { } inShape
                   || ConcreteShape(outTensor.Shape) is not { } outShape
                   || inShape.SequenceEqual(outShape);
        }

        /// <summary>Whether <paramref name="type"/> says the value is a sequence, a map, an
        /// optional or a sparse tensor — anything but a dense tensor, which is all an output can be
        /// written over.</summary>
        private static bool IsOtherThanTensor(TypeProto? type)
            => type is not null
               && (type.SequenceType is not null || type.MapType is not null
                   || type.OptionalType is not null || type.SparseTensorType is not null);

        /// <summary>A shape's dims where every one is a known positive number, or null.</summary>
        private static long[]? ConcreteShape(TensorShapeProto? shape)
        {
            if (shape is null) return null;
            var dims = new long[shape.Dims.Count];
            for (int i = 0; i < dims.Length; i++)
            {
                var dim = shape.Dims[i];
                if (dim.DimValue <= 0 || dim.DimParam.Length > 0) return null;
                dims[i] = dim.DimValue;
            }
            return dims;
        }

        /// <summary>Adds to <paramref name="found"/> every name <paramref name="subgraph"/> — and any
        /// subgraph inside it — reads from outside itself.</summary>
        private static void ReferencedFrom(GraphProto subgraph, HashSet<string> found)
        {
            var defined = new HashSet<string>(StringComparer.Ordinal);
            foreach (var input in subgraph.Inputs) defined.Add(input.Name);
            foreach (var initializer in subgraph.Initializers) defined.Add(initializer.Name);
            foreach (var node in subgraph.Nodes)
                foreach (var output in node.Outputs) defined.Add(output);
            // A subgraph output may be an outer name handed straight back, which reads it too.
            var inner = new HashSet<string>(subgraph.Outputs.Select(o => o.Name), StringComparer.Ordinal);
            foreach (var node in subgraph.Nodes)
            {
                foreach (var input in node.Inputs) inner.Add(input);
                foreach (var attribute in node.Attributes)
                {
                    if (attribute.G is { } nested) ReferencedFrom(nested, inner);
                    foreach (var each in attribute.Graphs) ReferencedFrom(each, inner);
                }
            }
            foreach (var name in inner)
                if (name.Length > 0 && !defined.Contains(name)) found.Add(name);
        }
    }
}

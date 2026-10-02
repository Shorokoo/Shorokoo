using Shorokoo.Core.Factory.IR;

namespace Shorokoo.Core.Backends;

/// <summary>
/// A value of a run placed at a byte range of a <b>block</b> — the memory of an input the run
/// consumed: <see cref="Value"/> is written there, at <see cref="Offset"/>, its
/// <see cref="Bytes"/> long, inside the memory input <see cref="Block"/> was fed.
/// </summary>
internal readonly record struct Placement(string Value, string Block, long Offset, long Bytes)
{
    internal long End => Offset + Bytes;
}

/// <summary>
/// How a backend lays a run's values out in memory, as far as placing them goes: which outputs of
/// an operator are its input's memory whatever happens (<see cref="AlwaysShares"/>, never placed),
/// which are its input's memory unless the value is placed elsewhere (<see cref="SharesUnlessPlaced"/>
/// — a view the backend hands back over its input, which a placed value is instead written into its
/// range), and which operators it can write into a given range at all (<see cref="Writes"/>).
/// </summary>
internal sealed record PlacementMemory(
    Func<NodeProto, int, int, bool> AlwaysShares,
    Func<NodeProto, int, int, bool> SharesUnlessPlaced,
    Func<NodeProto, int, bool> Writes)
{
    /// <summary>ONNX Runtime: what it may hand back over a kernel's input
    /// (<see cref="OutputAliasProof.Shares"/>) is that input's memory, and every other value of the
    /// graph is memory of its own — a <c>Slice</c> copies. A binding can take any value.</summary>
    internal static PlacementMemory OnnxRuntime { get; } = new(OutputAliasProof.Shares, OutputAliasProof.Shares, static (_, _) => true);

    /// <summary>
    /// PyTorch run eagerly from a translation of the graph. An output of anything but the operators
    /// known to compute into memory of their own (<see cref="TorchFresh"/>) may be its input's memory
    /// — a slice, a reshape, a transpose, an expansion, a split, a cast to the type it has, a
    /// one-input <c>Max</c>, a <c>Clip</c> without bounds, a branch handing back a value it captured
    /// — unless it is placed, and then written into its range. It writes into a given range what it
    /// writes there allocating nothing a plain run does not allocate at that node too, and holding
    /// nothing once the node is done, so that a placed run never holds more than a plain one: the
    /// element-wise operators of <see cref="TorchElementWise"/> and <c>MatMul</c> through torch's
    /// <c>out=</c> forms, and the operators of <see cref="TorchComposed"/> step by step in the range
    /// (each with the temporaries a plain run makes for it), over floating-point values; a fill; a
    /// concatenation part by part; and a copy of what a view operator reads. A convolution is
    /// written there by cuDNN on a card, and elsewhere computed and copied in, as an operator whose
    /// translation cannot write into the range is.
    /// </summary>
    internal static PlacementMemory PyTorch { get; } = new(static (_, _, _) => false, TorchShares, TorchWrites);

    /// <summary>The operators whose translation always computes into memory of its own, never
    /// handing back an input or a view of one; <c>Max</c>, <c>Min</c> and <c>Sum</c> only of two
    /// inputs or more, since one of one input is that input, and <c>Clip</c> only with a
    /// bound.</summary>
    internal static readonly HashSet<string> TorchFresh = new(StringComparer.Ordinal)
    {
        "Shape", "Size", "Add", "Sub", "Mul", "Div", "Pow", "Mod", "Neg", "Abs", "Sign", "Reciprocal", "Sqrt",
        "Exp", "Log", "Sin", "Cos", "Tan", "Asin", "Acos", "Atan", "Sinh", "Cosh", "Tanh", "Asinh", "Acosh",
        "Atanh", "Erf", "Sigmoid", "Relu", "Mean", "MatMul", "Gemm", "ConstantOfShape", "Concat", "Where",
        "Softmax", "LogSoftmax", "Gelu", "Conv", "LayerNormalization", "BatchNormalization",
    };

    /// <summary>The operators PyTorch's translation computes as a function that, handed a range,
    /// writes its first output there step by step; placed only where that output is the one
    /// used.</summary>
    internal static readonly HashSet<string> TorchComposed = new(StringComparer.Ordinal)
    {
        "Softmax", "LogSoftmax", "Gelu", "Clip", "Gemm", "LayerNormalization", "BatchNormalization", "Conv",
    };

    /// <summary>The element-wise operators PyTorch writes into a given range of a floating-point
    /// value as the translation computes them: those of one input, and those of two. A floating-point
    /// <c>MatMul</c> of two matrices or stacks of them is written there too.</summary>
    internal static readonly HashSet<string> TorchElementWise = new(StringComparer.Ordinal)
    {
        "Neg", "Abs", "Sigmoid", "Relu", "Exp", "Log", "Sqrt", "Tanh", "Sin", "Cos", "Tan", "Asin", "Acos",
        "Atan", "Sinh", "Cosh", "Asinh", "Acosh", "Atanh", "Reciprocal", "Floor", "Ceil", "Round", "Erf",
        "Add", "Sub", "Mul", "Div", "Pow", "Max", "Min", "Sum",
    };

    /// <summary>The operators whose output PyTorch writes into a given range as a copy of what the
    /// operator reads: the views, and what is a view unless placed.</summary>
    internal static readonly HashSet<string> TorchCopies = new(StringComparer.Ordinal)
    {
        "Slice", "Identity", "Reshape", "Squeeze", "Unsqueeze", "Flatten", "Transpose", "Expand",
    };

    private static bool TorchShares(NodeProto node, int input, int output)
    {
        if (!OutputAliasProof.IsStandard(node)) return true;
        if (node.OpType is "Max" or "Min" or "Sum") return node.Inputs.Count(i => i.Length > 0) < 2;
        if (node.OpType == "Clip")
            return node.Inputs.Skip(1).All(i => i.Length == 0) && node.Attributes.All(a => a.Name is not ("min" or "max"));
        return !TorchFresh.Contains(node.OpType);
    }

    private static bool TorchWrites(NodeProto node, int elementType)
    {
        if (!OutputAliasProof.IsStandard(node) || node.Outputs.Count(o => o.Length > 0) != 1) return false;
        var floating = elementType is 1 or 10 or 11 or 16;
        if (TorchElementWise.Contains(node.OpType) || node.OpType == "MatMul")
            return floating && (PlacementProof.InPlaceUnary.Contains(node.OpType) ? node.Inputs.Count == 1 : node.Inputs.Count == 2);
        if (TorchComposed.Contains(node.OpType))
            return floating && node.Outputs[0].Length > 0 && !TorchShares(node, 0, 0);
        return node.OpType is "ConstantOfShape" or "Concat" || TorchCopies.Contains(node.OpType);
    }
}

/// <summary>
/// Which values of an ONNX graph may be written into which byte ranges of the memory of the inputs a
/// run consumes, for one run: the proof behind placement, and a planner that proposes placements for
/// it to prove. It answers for the graph it is given — the graph the backend runs, after its own
/// rewrites — with the shapes the run's inputs come in, and the backend's memory layout
/// (<see cref="PlacementMemory"/>).
///
/// <para>A placement puts value V, written by node P, at a range of a block. It is proved when:</para>
/// <list type="bullet">
/// <item><b>V is a value of its own</b>: a fixed-width tensor written by a node of the graph the
/// backend can write into a range — not a graph input, not an initializer (a constant a backend
/// folded), not the output of a node holding a subgraph, and not an output the backend always hands
/// back as its input's memory; its shape is known, and it lies inside its block at exactly its size,
/// at a 256-byte boundary.</item>
/// <item><b>Its block's occupants are ordered.</b> A block's occupants are the input's own content
/// — read by every node reading the input or a view of it, each over the range it reads (the range
/// the view covers, where it is one contiguous run, or the whole block) — and the values placed in
/// it. Of any two occupants whose ranges overlap, one comes first: every node reading its overlapped
/// bytes, other than the second's writer, runs before the second's writer, its own writer does
/// too, and it is not read after the run — a graph output, or anything a graph output is a view of,
/// never is overwritten.</item>
/// <item><b>A writer reads what it overwrites only where it writes it.</b> Where V's writer reads an
/// occupant V overwrites, it reads each overlapped byte in the position it writes that byte: an
/// element-wise operator of one input, or of two, over an operand of V's own shape at V's own
/// offset; a <c>Slice</c> written at the offset it reads; a <c>Concat</c> part written where the
/// concatenation puts it.</item>
/// </list>
/// <para>What runs before a node depends on the backend. One that runs the graph's nodes one at a
/// time in the order they are listed — ONNX Runtime's sequential executor, over the graph the session
/// wrote out — runs before a node every node listed before it (<c>runsInOrder</c>), so a proof holds
/// for that order alone. Otherwise a node's ancestors run before it, ancestry following the edges
/// that order execution whatever a runtime folds, as <see cref="OutputAliasProof"/> does: explicit
/// inputs, except those of a node that reads only a shape, and the outer values a node's subgraphs
/// read. Such a proof depends on no order a runtime picks among independent nodes, and holds under
/// a sequential order, a parallel executor, and a card's single stream alike.</para>
/// </summary>
internal sealed class PlacementProof
{
    /// <summary>The most nodes a graph may have for placement to be considered: ancestry is kept
    /// as a bit set per node.</summary>
    internal const int MostNodes = 20000;

    /// <summary>The boundary every placement starts at: what a card's own allocations are aligned
    /// to, which kernels reading wide vectors rely on.</summary>
    internal const long Alignment = 256;

    /// <summary>The smallest value the planner places: one under a mebibyte is left to the backend,
    /// whose own reuse serves it, and placing it costs more than it saves.</summary>
    internal const long Smallest = 1L << 20;

    /// <summary>The most of a block the outputs placed in it may leave idle once the run is over,
    /// where the block lives whole as long as its outputs.</summary>
    internal const long IdleOutputBytes = 1L << 20;

    // The reader standing for whatever reads a value once the run is over: it runs after every node.
    private const int AfterTheRun = int.MaxValue;

    // Element-wise operators reading each element of an operand of the output's shape in the position
    // they write it, in one pass: the standard ones of one input, and of exactly two.
    internal static readonly HashSet<string> InPlaceUnary = new(StringComparer.Ordinal)
    {
        "Abs", "Neg", "Sigmoid", "Relu", "Exp", "Log", "Sqrt", "Tanh", "Sin", "Cos", "Tan", "Asin", "Acos",
        "Atan", "Sinh", "Cosh", "Asinh", "Acosh", "Atanh", "Reciprocal", "Floor", "Ceil", "Round", "Sign",
        "Erf", "Softplus", "Softsign", "Elu", "Selu", "LeakyRelu", "ThresholdedRelu", "HardSigmoid",
        "HardSwish", "Celu", "Not", "BitwiseNot", "Gelu",
    };

    internal static readonly HashSet<string> InPlaceBinary = new(StringComparer.Ordinal)
    {
        "Add", "Sub", "Mul", "Div", "Pow", "Max", "Min", "Sum", "Mean", "PRelu", "BitwiseAnd", "BitwiseOr",
        "BitwiseXor", "And", "Or", "Xor",
    };

    private readonly List<NodeProto> _nodes;
    private readonly Dictionary<string, int> _producer = new(StringComparer.Ordinal);
    private readonly Dictionary<string, List<(int Node, int Slot)>> _consumers = new(StringComparer.Ordinal);
    private readonly Dictionary<string, List<int>> _capturedBy = new(StringComparer.Ordinal);
    private readonly Dictionary<string, int> _outputs = new(StringComparer.Ordinal);
    private readonly HashSet<string> _readAfterRun;
    private readonly HashSet<string> _inputs = new(StringComparer.Ordinal);
    private readonly HashSet<string> _initializers = new(StringComparer.Ordinal);
    private readonly IReadOnlyDictionary<string, long> _blocks;
    private readonly IReadOnlyDictionary<string, PlacementShapes.Value> _shapes;
    private readonly PlacementMemory _memory;
    private readonly ulong[][] _ancestors;
    private readonly bool _runsInOrder;

    /// <summary>
    /// The proof over <paramref name="graph"/> for one run: <paramref name="blocks"/> names each
    /// input the run consumed — the memory it may place values in — with its size in bytes, and
    /// <paramref name="shapes"/> holds every value's shape for the run
    /// (<see cref="PlacementShapes.Evaluate"/>). <paramref name="readAfterRun"/> names the graph
    /// outputs the caller reads once the run is over, each of which nothing may overwrite; null for
    /// every one of them. A graph output not named there is one a binding takes for the run alone,
    /// as a placed value made an output only so it can be bound to its range.
    /// <paramref name="memory"/> is how the backend lays values out; ONNX Runtime's where null.
    /// <paramref name="runsInOrder"/> says the backend runs the graph's nodes one at a time in the
    /// order they are listed — on each of its streams, where every node that reads or writes a
    /// block is on one — so that a node listed earlier has run before a later one starts; where
    /// it does not, a node runs before another only where the graph's edges make it.
    /// </summary>
    /// <exception cref="ArgumentException">The graph has more than <see cref="MostNodes"/> nodes,
    /// or is not in topological order.</exception>
    internal PlacementProof(
        GraphProto graph, IReadOnlyDictionary<string, long> blocks, IReadOnlyDictionary<string, PlacementShapes.Value> shapes,
        IReadOnlySet<string>? readAfterRun = null, PlacementMemory? memory = null, bool runsInOrder = false)
    {
        _nodes = graph.Nodes;
        _runsInOrder = runsInOrder;
        if (_nodes.Count > MostNodes) throw new ArgumentException($"The graph has more than {MostNodes} nodes.", nameof(graph));
        _blocks = blocks;
        _shapes = shapes;
        _memory = memory ?? PlacementMemory.OnnxRuntime;
        foreach (var input in graph.Inputs) _inputs.Add(input.Name);
        foreach (var initializer in graph.Initializers) _initializers.Add(initializer.Name);
        foreach (var output in graph.Outputs) _outputs[output.Name] = _outputs.GetValueOrDefault(output.Name) + 1;
        _readAfterRun = readAfterRun is null
            ? new HashSet<string>(_outputs.Keys, StringComparer.Ordinal)
            : new HashSet<string>(readAfterRun.Where(_outputs.ContainsKey), StringComparer.Ordinal);

        var captured = new Dictionary<int, HashSet<string>>();
        for (int n = 0; n < _nodes.Count; n++)
        {
            var node = _nodes[n];
            foreach (var output in node.Outputs)
                if (output.Length > 0) _producer[output] = n;
            for (int i = 0; i < node.Inputs.Count; i++)
            {
                var input = node.Inputs[i];
                if (input.Length == 0) continue;
                if (!_consumers.TryGetValue(input, out var readers)) _consumers[input] = readers = [];
                readers.Add((n, i));
            }
            if (OutputAliasProof.HoldsSubgraph(node))
            {
                var names = new HashSet<string>(StringComparer.Ordinal);
                foreach (var attribute in node.Attributes)
                {
                    if (attribute.G is { } subgraph) OutputAliasProof.GraphIndex.ReferencedFrom(subgraph, names);
                    foreach (var each in attribute.Graphs) OutputAliasProof.GraphIndex.ReferencedFrom(each, names);
                }
                captured[n] = names;
                foreach (var name in names)
                {
                    if (!_capturedBy.TryGetValue(name, out var holders)) _capturedBy[name] = holders = [];
                    holders.Add(n);
                }
            }
        }
        _ancestors = Ancestry(captured);
    }

    // ---- ancestry ----

    /// <summary>Per node, the nodes that run before it whatever a runtime folds, as a bit set:
    /// computed in node order, which must be topological.</summary>
    private ulong[][] Ancestry(Dictionary<int, HashSet<string>> captured)
    {
        var words = (_nodes.Count + 63) / 64;
        var ancestors = new ulong[_nodes.Count][];
        for (int n = 0; n < _nodes.Count; n++)
        {
            var bits = new ulong[words];
            var node = _nodes[n];
            if (!OutputAliasProof.ReadsOnlyAShape(node))
            {
                IEnumerable<string> reads = node.Inputs;
                if (captured.TryGetValue(n, out var names)) reads = reads.Concat(names);
                foreach (var input in reads)
                {
                    if (input.Length == 0 || !_producer.TryGetValue(input, out var p)) continue;
                    if (p >= n) throw new ArgumentException("The graph is not in topological order.");
                    var from = ancestors[p];
                    for (int w = 0; w < words; w++) bits[w] |= from[w];
                    bits[p >> 6] |= 1UL << (p & 63);
                }
            }
            ancestors[n] = bits;
        }
        return ancestors;
    }

    /// <summary>Whether node <paramref name="before"/> has run before node <paramref name="node"/>
    /// starts: listed first, where the backend runs the nodes in their order, and otherwise an
    /// ancestor of it, which runs first whatever order a runtime picks.</summary>
    private bool Precedes(int before, int node)
        => _runsInOrder ? before < node : (_ancestors[node][before >> 6] & (1UL << (before & 63))) != 0;

    // ---- memory chains ----

    /// <summary>Where a value in an occupant's memory chain lies within the occupant: from
    /// <see cref="Shift"/> bytes into it, <see cref="Bytes"/> long, as one contiguous run — or, where
    /// it is not <see cref="Contiguous"/>, somewhere within the occupant's range.</summary>
    private readonly record struct Member(long Shift, long Bytes, bool Contiguous);

    /// <summary>
    /// <paramref name="root"/>, <paramref name="bytes"/> long, and every view of it the backend hands
    /// back over its memory — an output it always shares, or one it shares unless that output is in
    /// <paramref name="placed"/> — with where each lies within it.
    /// </summary>
    private Dictionary<string, Member> ChainOf(string root, long bytes, IReadOnlySet<string> placed)
    {
        var chain = new Dictionary<string, Member>(StringComparer.Ordinal) { [root] = new(0, bytes, true) };
        var pending = new Queue<string>();
        pending.Enqueue(root);
        while (pending.TryDequeue(out var current))
        {
            var parent = chain[current];
            // A node holding a subgraph that reads the value may hand it back as one of its outputs,
            // where the backend does that: somewhere within the occupant.
            if (_capturedBy.TryGetValue(current, out var holders))
                foreach (var n in holders)
                    foreach (var (output, o) in _nodes[n].Outputs.Select((output, o) => (output, o)))
                        if (output.Length > 0 && !chain.ContainsKey(output)
                            && (_memory.AlwaysShares(_nodes[n], -1, o) || (_memory.SharesUnlessPlaced(_nodes[n], -1, o) && !placed.Contains(output))))
                        {
                            chain[output] = new Member(0, bytes, false);
                            pending.Enqueue(output);
                        }
            if (!_consumers.TryGetValue(current, out var consumers)) continue;
            foreach (var (n, slot) in consumers)
            {
                var node = _nodes[n];
                for (int o = 0; o < node.Outputs.Count; o++)
                {
                    var output = node.Outputs[o];
                    if (output.Length == 0 || chain.ContainsKey(output)) continue;
                    if (!_memory.AlwaysShares(node, slot, o) && (!_memory.SharesUnlessPlaced(node, slot, o) || placed.Contains(output)))
                        continue;
                    chain[output] = ViewLayout(node, slot, o, current, parent, bytes);
                    pending.Enqueue(output);
                }
            }
        }
        return chain;
    }

    /// <summary>Where output <paramref name="o"/> of a view node lies within the occupant, given
    /// where its input <paramref name="parentName"/> lies.</summary>
    private Member ViewLayout(NodeProto node, int slot, int o, string parentName, Member parent, long occupantBytes)
    {
        var whole = new Member(0, occupantBytes, false);
        if (!parent.Contiguous) return whole;
        var output = node.Outputs[o];
        var size = ElementBytesOf(parentName);
        switch (node.OpType)
        {
            case "Identity" or "Reshape" or "Squeeze" or "Unsqueeze" or "Flatten" or "Dropout" or "ExpandDims"
                or "Optional" or "OptionalGetElement" when slot == 0 && o == 0:
                return parent;
            case "BatchNormalization":
                return parent;
            case "Slice" when slot == 0 && SliceRun(node, parentName) is { } run && size > 0:
                return new Member(parent.Shift + run.First * size, run.Count * size, true);
            case "Split" when slot == 0 && size > 0 && _shapes.TryGetValue(parentName, out var data):
            {
                var axisAttribute = node.Attributes.FirstOrDefault(a => a.Name == "axis");
                var rank = data.Shape.Length;
                var axis = (int)(axisAttribute is null ? 0 : axisAttribute.I < 0 ? axisAttribute.I + rank : axisAttribute.I);
                if (axis < 0 || axis >= rank || data.Shape[..axis].Any(d => d != 1)) return whole;
                long before = 0;
                for (int k = 0; k < o; k++)
                {
                    if (!_shapes.TryGetValue(node.Outputs[k], out var part)) return whole;
                    before += part.Elements;
                }
                return _shapes.TryGetValue(output, out var mine)
                    ? new Member(parent.Shift + before * size, mine.Elements * size, true)
                    : whole;
            }
            default:
                return whole;
        }
    }

    /// <summary>
    /// Every read of the memory behind <paramref name="chain"/> — at <paramref name="offset"/> in its
    /// block, <paramref name="bytes"/> long: per reading node, the range of the block it reads. A
    /// <c>Shape</c> or <c>Size</c> reads nothing; a <c>Slice</c> taking one contiguous run of a
    /// contiguous value reads that run; anything else reads all of the value it reads, or the whole
    /// occupant where that value is not one contiguous run of it; a node whose subgraph refers to it
    /// reads the whole occupant. A value read after the run is read there by
    /// <see cref="AfterTheRun"/>, which no node precedes.
    /// </summary>
    private List<(int Node, long Start, long End)> ReadsOf(Dictionary<string, Member> chain, long offset, long bytes)
    {
        var reads = new List<(int, long, long)>();
        foreach (var (name, member) in chain)
        {
            var (start, end) = member.Contiguous ? (offset + member.Shift, offset + member.Shift + member.Bytes) : (offset, offset + bytes);
            if (_readAfterRun.Contains(name)) reads.Add((AfterTheRun, start, end));
            if (_consumers.TryGetValue(name, out var consumers))
                foreach (var (n, slot) in consumers)
                {
                    var node = _nodes[n];
                    if (OutputAliasProof.ReadsOnlyAShape(node)) continue;
                    if (member.Contiguous && node.OpType == "Slice" && OutputAliasProof.IsStandard(node) && slot == 0
                        && SliceRun(node, name) is { } run && ElementBytesOf(name) is var size and > 0)
                    {
                        reads.Add((n, start + run.First * size, start + (run.First + run.Count) * size));
                        continue;
                    }
                    reads.Add((n, start, end));
                }
            if (_capturedBy.TryGetValue(name, out var holders))
                foreach (var n in holders) reads.Add((n, offset, offset + bytes));
        }
        return reads;
    }

    /// <summary>The contiguous run, in elements, a <c>Slice</c> node takes of
    /// <paramref name="data"/>, or null.</summary>
    private (long First, long Count)? SliceRun(NodeProto node, string data)
    {
        if (!_shapes.TryGetValue(data, out var shape)) return null;
        PlacementShapes.Value?[] read = [.. node.Inputs.Select(name => name.Length > 0 && _shapes.TryGetValue(name, out var v) ? v : null)];
        return PlacementShapes.SliceOf(node, read)?.Contiguous(shape.Shape);
    }

    private int ElementBytesOf(string name)
        => _shapes.TryGetValue(name, out var value) ? PlacementShapes.ElementBytes(value.ElementType) : 0;

    // ---- occupants ----

    /// <summary>What occupies a range of a block for a while: an input's own content (no writer)
    /// or a placed value, the memory chain behind it, and every read of it.</summary>
    private sealed record Occupant(
        Placement Placement, int Writer, Dictionary<string, Member> Chain, List<(int Node, long Start, long End)> Reads);

    private Occupant Content(string block, IReadOnlySet<string> placed)
    {
        var bytes = _blocks[block];
        var chain = ChainOf(block, bytes, placed);
        return new Occupant(new Placement(block, block, 0, bytes), -1, chain, ReadsOf(chain, 0, bytes));
    }

    private Occupant Placed(Placement placement, IReadOnlySet<string> placed)
    {
        var chain = ChainOf(placement.Value, placement.Bytes, placed);
        return new Occupant(placement, _producer[placement.Value], chain, ReadsOf(chain, placement.Offset, placement.Bytes));
    }

    /// <summary>
    /// Why <paramref name="value"/> cannot be placed at all, or null where it may be: it must be a
    /// fixed-width tensor of known shape written by a node of the graph that holds no subgraph and
    /// that the backend can write into a range, as that node's own memory rather than its input's.
    /// </summary>
    internal string? Unplaceable(string value)
    {
        if (_inputs.Contains(value)) return "a graph input";
        if (_initializers.Contains(value)) return "an initializer";
        if (!_producer.TryGetValue(value, out var p)) return "written by no node";
        if (_outputs.GetValueOrDefault(value) > 1) return "a graph output listed twice";
        var node = _nodes[p];
        if (OutputAliasProof.HoldsSubgraph(node)) return "written by a node holding a subgraph";
        var o = node.Outputs.IndexOf(value);
        for (int i = 0; i < node.Inputs.Count; i++)
            if (_memory.AlwaysShares(node, i, o)) return "a view of its writer's input";
        if (!_shapes.TryGetValue(value, out var shape)) return "of unknown shape";
        if (shape.Bytes <= 0) return "of no fixed-width element type";
        if (!_memory.Writes(node, shape.ElementType)) return "written by an operator the backend cannot write into a range";
        return null;
    }

    /// <summary>Whether occupant <paramref name="first"/> comes before <paramref name="second"/>
    /// on the bytes their ranges share.</summary>
    private bool Before(Occupant first, Occupant second) => ConflictOf(first, second, out _) == Conflict.None;

    /// <summary>Why one occupant cannot come before another on the bytes they share.</summary>
    private enum Conflict { None, ContentLast, WriterOrder, ReadAfterRun, ReaderOrder, WriterReads }

    /// <summary>
    /// What stands in the way of occupant <paramref name="first"/> coming before
    /// <paramref name="second"/> on the bytes their ranges share — <see cref="Conflict.None"/>
    /// where nothing does — with <paramref name="node"/> the node it names, or -1.
    /// </summary>
    private Conflict ConflictOf(Occupant first, Occupant second, out int node)
    {
        node = -1;
        var w = second.Writer;
        if (w < 0) return Conflict.ContentLast;
        if (first.Writer >= 0 && (first.Writer == w || !Precedes(first.Writer, w)))
        {
            node = first.Writer;
            return Conflict.WriterOrder;
        }
        var lo = Math.Max(first.Placement.Offset, second.Placement.Offset);
        var hi = Math.Min(first.Placement.End, second.Placement.End);
        foreach (var (reader, start, end) in first.Reads)
        {
            if (end <= lo || start >= hi) continue;
            if (reader == w) continue;
            if (reader == AfterTheRun) return Conflict.ReadAfterRun;
            if (!Precedes(reader, w))
            {
                node = reader;
                return Conflict.ReaderOrder;
            }
        }
        foreach (var (start, end, identical) in WriterReads(w, first, second.Placement))
            if (end > lo && start < hi && !identical)
            {
                node = w;
                return Conflict.WriterReads;
            }
        return Conflict.None;
    }

    /// <summary>
    /// Why <paramref name="value"/> is not placed beside <paramref name="placements"/>, for a
    /// measurement to read: why it cannot be placed at all, or, for each of the first ranges the
    /// planner would try, the occupant in its way and how — in neither order can the two share the
    /// range.
    /// </summary>
    internal string WhyNot(string value, IReadOnlyList<Placement> placements)
    {
        if (Unplaceable(value) is { } why) return why;
        var occupants = Occupants(placements);
        var placed = placements.Select(p => p.Value).Append(value).ToHashSet(StringComparer.Ordinal);
        var reasons = new List<string>();
        foreach (var option in Options(value, occupants).Take(4))
        {
            var at = $"{option.Block}+{option.Offset}";
            if (!Fits(option))
            {
                reasons.Add($"{at}: outside its block or off a boundary");
                continue;
            }
            var candidate = Placed(option, placed);
            var other = occupants.FirstOrDefault(o => !Compatible(o, candidate));
            if (other is null)
            {
                reasons.Add($"{at}: fits");
                continue;
            }
            reasons.Add($"{at}: over {other.Placement.Value}, {Describe(ConflictOf(other, candidate, out var n1), n1)}; "
                        + $"before it, {Describe(ConflictOf(candidate, other, out var n2), n2)}");
        }
        return reasons.Count == 0 ? "no range to try" : string.Join(" | ", reasons);
    }

    private string Describe(Conflict conflict, int node)
    {
        var named = node >= 0 ? $"{_nodes[node].OpType} '{_nodes[node].Name}'" : "";
        return conflict switch
        {
            Conflict.ContentLast => "an input's own content comes first",
            Conflict.WriterOrder => $"written by {named}, which does not run first",
            Conflict.ReadAfterRun => "read after the run",
            Conflict.ReaderOrder => $"read by {named}, which does not run first",
            Conflict.WriterReads => $"its writer {named} reads it where it does not write it",
            _ => "nothing",
        };
    }

    /// <summary>Whether two occupants of one block can share it: their ranges do not overlap, or one
    /// comes first.</summary>
    private bool Compatible(Occupant a, Occupant b)
        => a.Placement.Block != b.Placement.Block
           || a.Placement.End <= b.Placement.Offset || b.Placement.End <= a.Placement.Offset
           || Before(a, b) || Before(b, a);

    /// <summary>
    /// Each read node <paramref name="writer"/> makes of the memory behind <paramref name="occupant"/>
    /// while writing <paramref name="written"/>: the range of the block it reads, and whether it
    /// reads it only in the positions it writes.
    /// </summary>
    private IEnumerable<(long Start, long End, bool Identical)> WriterReads(int writer, Occupant occupant, Placement written)
    {
        var node = _nodes[writer];
        var standard = OutputAliasProof.IsStandard(node);
        _shapes.TryGetValue(written.Value, out var outShape);
        var outBytes = outShape is null ? 0 : PlacementShapes.ElementBytes(outShape.ElementType);
        for (int slot = 0; slot < node.Inputs.Count; slot++)
        {
            var name = node.Inputs[slot];
            if (name.Length == 0 || !occupant.Chain.TryGetValue(name, out var member)) continue;
            if (!member.Contiguous)
            {
                yield return (occupant.Placement.Offset, occupant.Placement.End, false);
                continue;
            }
            var start = occupant.Placement.Offset + member.Shift;
            var end = start + member.Bytes;
            var identical = false;
            if (standard && _shapes.TryGetValue(name, out var inShape) && outShape is not null)
            {
                var inBytes = PlacementShapes.ElementBytes(inShape.ElementType);
                var sameElements = inBytes == outBytes && inBytes > 0 && inShape.Shape.SequenceEqual(outShape.Shape);
                if ((InPlaceUnary.Contains(node.OpType) && slot == 0 && node.Outputs.Count == 1)
                    || (node.OpType == "Clip" && slot == 0)
                    || (InPlaceBinary.Contains(node.OpType) && node.Inputs.Count == 2))
                    identical = sameElements && written.Offset == start;
                else if (IsWholeCopy(node, slot))
                    identical = inBytes == outBytes && inBytes > 0 && inShape.Elements == outShape.Elements && written.Offset == start;
                else if (node.OpType == "Slice" && slot == 0 && SliceRun(node, name) is { } run && inBytes > 0)
                {
                    start += run.First * inBytes;
                    end = start + run.Count * inBytes;
                    identical = inBytes == outBytes && written.Offset == start;
                }
                else if (node.OpType == "Concat" && ConcatPart(node, slot, outShape) is { } part && inBytes == outBytes)
                    identical = written.Offset + part * outBytes == start;
            }
            else if (IsFusedSum(node, slot, written.Value) && _shapes.TryGetValue(name, out var summed) && outShape is not null)
                identical = PlacementShapes.ElementBytes(summed.ElementType) == outBytes && outBytes > 0
                            && summed.Shape.SequenceEqual(outShape.Shape) && written.Offset == start;
            yield return (start, end, identical);
        }
    }

    /// <summary>
    /// Whether <paramref name="value"/> is the sum a fused layer normalization writes beside its
    /// result — output 3 of ONNX Runtime's <c>SkipLayerNormalization</c> and
    /// <c>SkipSimplifiedLayerNormalization</c>, its input plus its skip (plus a bias) — and
    /// <paramref name="slot"/> one of the two it adds element for element: the kernels, on the host
    /// and the card, read each element of input and skip before writing that element of the sum,
    /// and read neither again, so the sum may be written over either where it lies.
    /// </summary>
    private static bool IsFusedSum(NodeProto node, int slot, string value)
        => node.Domain == "com.microsoft" && node.OpType is "SkipLayerNormalization" or "SkipSimplifiedLayerNormalization"
           && slot is 0 or 1 && node.Outputs.Count > 3 && node.Outputs[3] == value;

    /// <summary>Whether <paramref name="node"/> writes its output element for element as it reads
    /// its input <paramref name="slot"/>, in the same order: a view operator that keeps every
    /// element where it is, which a backend that places it writes as a copy.</summary>
    private static bool IsWholeCopy(NodeProto node, int slot)
        => slot == 0 && node.OpType is "Identity" or "Reshape" or "Squeeze" or "Unsqueeze" or "Flatten";

    /// <summary>Where part <paramref name="slot"/> of a <c>Concat</c> lands in its output, in
    /// elements, when every part lands as one contiguous run — every dimension outside the axis
    /// is one; null otherwise.</summary>
    private long? ConcatPart(NodeProto node, int slot, PlacementShapes.Value output)
    {
        var axisAttribute = node.Attributes.FirstOrDefault(a => a.Name == "axis");
        if (axisAttribute is null) return null;
        var rank = output.Shape.Length;
        var axis = (int)(axisAttribute.I < 0 ? axisAttribute.I + rank : axisAttribute.I);
        if (axis < 0 || axis >= rank) return null;
        for (int d = 0; d < axis; d++)
            if (output.Shape[d] != 1) return null;
        long before = 0;
        for (int i = 0; i < slot; i++)
        {
            if (!_shapes.TryGetValue(node.Inputs[i], out var part)) return null;
            before += part.Elements;
        }
        return before;
    }

    // ---- proving ----

    /// <summary>
    /// The occupants of the blocks with <paramref name="placements"/> in them: each block's own
    /// content, then each placement, every memory chain as the backend lays it out with those
    /// values placed.
    /// </summary>
    private List<Occupant> Occupants(IReadOnlyList<Placement> placements, IReadOnlySet<string>? cut = null)
    {
        var placed = placements.Select(p => p.Value).ToHashSet(StringComparer.Ordinal);
        if (cut is not null) placed.UnionWith(cut);
        return [.. _blocks.Keys.Select(b => Content(b, placed)), .. placements.Select(p => Placed(p, placed))];
    }

    /// <summary>Whether <paramref name="value"/> is a view its writer hands back unless it is
    /// placed, so placing it changes the memory chain of what it views.</summary>
    internal bool LeavesAChain(string value)
    {
        var node = _nodes[_producer[value]];
        var o = node.Outputs.IndexOf(value);
        for (int i = 0; i < node.Inputs.Count; i++)
            if (_memory.SharesUnlessPlaced(node, i, o)) return true;
        return false;
    }

    /// <summary>
    /// <paramref name="occupants"/> — those of <paramref name="accepted"/>, laid out with the values
    /// of <paramref name="cut"/> out of every memory chain too — with <paramref name="candidate"/>
    /// placed as well, where every pair of them is compatible; null where the candidate does not fit.
    /// </summary>
    private List<Occupant>? With(List<Occupant> occupants, List<Placement> accepted, Placement candidate, IReadOnlySet<string>? cut = null)
    {
        if (!Fits(candidate)) return null;
        if (!LeavesAChain(candidate.Value) || cut?.Contains(candidate.Value) == true)
        {
            var placed = accepted.Select(p => p.Value).Append(candidate.Value).ToHashSet(StringComparer.Ordinal);
            if (cut is not null) placed.UnionWith(cut);
            var occupant = Placed(candidate, placed);
            return occupants.All(other => Compatible(other, occupant)) ? [.. occupants, occupant] : null;
        }
        // Placing a view writes it into its own range, which takes it out of the memory chain it was
        // part of: every occupant is laid out again, and every pair checked.
        var rebuilt = Occupants([.. accepted, candidate], cut);
        for (int i = 0; i < rebuilt.Count; i++)
            for (int j = i + 1; j < rebuilt.Count; j++)
                if (!Compatible(rebuilt[i], rebuilt[j])) return null;
        return rebuilt;
    }

    /// <summary>
    /// The placements of <paramref name="placements"/> this graph proves, in the order given: each is
    /// kept where it is a value that may be placed (<see cref="Unplaceable"/>), lies at its exact size
    /// inside a block at an aligned offset, and every occupant kept so far — the blocks' own contents
    /// first — is compatible with it.
    /// </summary>
    internal IReadOnlyList<Placement> Prove(IEnumerable<Placement> placements)
    {
        var kept = new List<Placement>();
        var occupants = Occupants(kept);
        var values = new HashSet<string>(StringComparer.Ordinal);
        foreach (var placement in placements)
        {
            if (!values.Add(placement.Value)) continue;
            if (With(occupants, kept, placement) is not { } next)
            {
                values.Remove(placement.Value);
                continue;
            }
            occupants = next;
            kept.Add(placement);
        }
        return kept;
    }

    private bool Fits(Placement placement)
        => _blocks.TryGetValue(placement.Block, out var blockBytes)
           && Unplaceable(placement.Value) is null
           && _shapes[placement.Value].Bytes == placement.Bytes
           && placement.Offset >= 0 && placement.End <= blockBytes
           && placement.Offset % Alignment == 0
           && !_blocks.ContainsKey(placement.Value);

    // ---- the memory a run holds ----

    /// <summary>
    /// The most the run holds at once beyond its inputs and initializers, as modelled from the graph
    /// with <paramref name="placements"/> in their blocks: each node, in the graph's order, takes
    /// memory for every output it makes that is neither placed nor what the backend hands back
    /// over an input, and lets go of each once the last node reading it — or anything handed back
    /// over it — has run; what is read after the run is held to its end. A value of unknown shape
    /// counts nothing.
    /// </summary>
    internal long ModelledPeak(IEnumerable<Placement> placements)
    {
        var placed = placements.Select(p => p.Value).ToHashSet(StringComparer.Ordinal);
        var root = new Dictionary<string, string>(StringComparer.Ordinal);
        var bytes = new Dictionary<string, long>(StringComparer.Ordinal);
        string RootOf(string name) => root.TryGetValue(name, out var r) ? r : name;
        for (int n = 0; n < _nodes.Count; n++)
        {
            var node = _nodes[n];
            for (int o = 0; o < node.Outputs.Count; o++)
            {
                var output = node.Outputs[o];
                if (output.Length == 0) continue;
                var over = Enumerable.Range(0, node.Inputs.Count)
                    .FirstOrDefault(i => node.Inputs[i].Length > 0 && _memory.AlwaysShares(node, i, o), -1);
                if (over >= 0)
                {
                    root[output] = RootOf(node.Inputs[over]);
                    continue;
                }
                root[output] = output;
                bytes[output] = placed.Contains(output) || !_shapes.TryGetValue(output, out var value) || value.Bytes < 0 ? 0 : value.Bytes;
            }
        }
        var last = new Dictionary<string, int>(StringComparer.Ordinal);
        void Read(string name, int n)
        {
            var r = RootOf(name);
            last[r] = Math.Max(last.GetValueOrDefault(r, -1), n);
        }
        for (int n = 0; n < _nodes.Count; n++)
            foreach (var input in _nodes[n].Inputs)
                if (input.Length > 0) Read(input, n);
        foreach (var (name, holders) in _capturedBy)
            foreach (var n in holders) Read(name, n);
        foreach (var output in _outputs.Keys) Read(output, AfterTheRun);

        var freed = new Dictionary<int, long>();
        long live = 0, peak = 0;
        for (int n = 0; n < _nodes.Count; n++)
        {
            foreach (var output in _nodes[n].Outputs)
            {
                if (output.Length == 0 || RootOf(output) != output || bytes[output] == 0) continue;
                live += bytes[output];
                var until = last.GetValueOrDefault(output, n);
                if (until != AfterTheRun) freed[until] = freed.GetValueOrDefault(until) + bytes[output];
            }
            peak = Math.Max(peak, live);
            live -= freed.GetValueOrDefault(n);
        }
        return peak;
    }

    // ---- planning ----

    /// <summary>
    /// Placements this graph proves, chosen to need as little memory beyond the blocks as it can:
    /// the outputs read after the run first, largest first, since nothing may overwrite one and where
    /// one goes constrains everything else, then every other value in node order. Each value of at
    /// least <paramref name="smallest"/> bytes tries the ranges its writer writes in place — an
    /// element-wise operand's, a slice's source, a concatenation part's slot — then the first aligned
    /// range that fits. An output placed in a block whose outputs would leave more than
    /// <paramref name="idleOutputBytes"/> of it unused after the run is not placed: the block lives as
    /// long as its outputs — unless it is one of <paramref name="givingBack"/>, whose memory no output
    /// stands on goes back as the run ends.
    ///
    /// <para>An output that is a view unless placed is planned as though every such output were
    /// placed — handed over as a view of a block, it would be copied out of it anyway — and a value
    /// that is a view unless placed and is not read after the run is not planned at all: as a view it
    /// takes no memory. The plan is proved over the placements it ends with, so an output planned that
    /// way that found no room refuses whatever relied on its being placed.</para>
    /// </summary>
    internal IReadOnlyList<Placement> Plan(long smallest, long idleOutputBytes, IReadOnlySet<string>? givingBack = null)
    {
        if (_blocks.Count == 0) return [];
        var candidates = new List<string>();
        for (int n = 0; n < _nodes.Count; n++)
            foreach (var value in _nodes[n].Outputs)
                if (value.Length > 0 && Unplaceable(value) is null && _shapes[value].Bytes >= smallest
                    && (_readAfterRun.Contains(value) || !LeavesAChain(value)))
                    candidates.Add(value);
        var outputs = candidates.Where(_readAfterRun.Contains).OrderByDescending(v => _shapes[v].Bytes).ToList();
        var others = candidates.Where(v => !_readAfterRun.Contains(v)).ToList();
        var cut = outputs.Where(LeavesAChain).ToHashSet(StringComparer.Ordinal);

        var placed = new List<Placement>();
        var occupants = Occupants(placed, cut);
        void Try(string value, IReadOnlySet<string>? planCut)
        {
            foreach (var placement in Options(value, occupants))
            {
                if (With(occupants, placed, placement, planCut) is not { } next) continue;
                occupants = next;
                placed.Add(placement);
                return;
            }
        }
        foreach (var value in outputs) Try(value, cut);

        // An output whose block would keep more than idleOutputBytes idle beside the outputs placed
        // in it goes back to the backend.
        var idle = placed.Where(p => _readAfterRun.Contains(p.Value)).GroupBy(p => p.Block)
            .Where(g => givingBack?.Contains(g.Key) != true && _blocks[g.Key] - g.Sum(p => p.Bytes) > idleOutputBytes)
            .Select(g => g.Key).ToHashSet();
        placed.RemoveAll(p => idle.Contains(p.Block));
        placed = [.. ProveViewsFirst(placed)];
        occupants = Occupants(placed);

        foreach (var value in others) Try(value, null);
        return ProveViewsFirst(placed);
    }

    /// <summary><see cref="Prove"/> over <paramref name="placements"/> with the views among them
    /// first: placing a view takes it out of the memory it views, which can only free what the
    /// placements after it need, and a placement planned on that is proved after it.</summary>
    private IReadOnlyList<Placement> ProveViewsFirst(List<Placement> placements)
        => Prove(placements.OrderBy(p => LeavesAChain(p.Value) ? 0 : 1));

    /// <summary>Where <paramref name="value"/> may go, best first: the ranges its writer writes in
    /// place, then every aligned range starting at a block's start or right after an
    /// occupant.</summary>
    private IEnumerable<Placement> Options(string value, List<Occupant> occupants)
    {
        var bytes = _shapes[value].Bytes;
        var writer = _nodes[_producer[value]];
        var seen = new HashSet<(string, long)>();
        foreach (var occupant in occupants)
        {
            for (int slot = 0; slot < writer.Inputs.Count; slot++)
            {
                var name = writer.Inputs[slot];
                if (name.Length == 0 || !occupant.Chain.TryGetValue(name, out var member) || !member.Contiguous) continue;
                var at = InPlaceOffset(writer, slot, name, occupant.Placement.Offset + member.Shift, value);
                if (at is { } offset && offset >= 0 && offset + bytes <= _blocks[occupant.Placement.Block]
                    && seen.Add((occupant.Placement.Block, offset)))
                    yield return new Placement(value, occupant.Placement.Block, offset, bytes);
            }
        }
        foreach (var (block, blockBytes) in _blocks)
        {
            var starts = new SortedSet<long> { 0 };
            foreach (var occupant in occupants)
                if (occupant.Placement.Block == block && occupant.Writer >= 0)
                    starts.Add((occupant.Placement.End + Alignment - 1) / Alignment * Alignment);
            foreach (var at in starts)
                if (at + bytes <= blockBytes && seen.Add((block, at)))
                    yield return new Placement(value, block, at, bytes);
        }
    }

    /// <summary>The offset at which <paramref name="writer"/> would write <paramref name="value"/>
    /// in the position it reads input <paramref name="name"/> at slot <paramref name="slot"/>, which
    /// lies at <paramref name="start"/> in its block; null where it reads it no such way.</summary>
    private long? InPlaceOffset(NodeProto writer, int slot, string name, long start, string value)
    {
        if (!_shapes.TryGetValue(name, out var inShape)) return null;
        var outShape = _shapes[value];
        var size = PlacementShapes.ElementBytes(inShape.ElementType);
        if (size != PlacementShapes.ElementBytes(outShape.ElementType)) return null;
        if (IsFusedSum(writer, slot, value)) return inShape.Shape.SequenceEqual(outShape.Shape) ? start : null;
        if (!OutputAliasProof.IsStandard(writer)) return null;
        if ((InPlaceUnary.Contains(writer.OpType) && slot == 0) || (writer.OpType == "Clip" && slot == 0)
            || (InPlaceBinary.Contains(writer.OpType) && writer.Inputs.Count == 2))
            return inShape.Shape.SequenceEqual(outShape.Shape) ? start : null;
        if (IsWholeCopy(writer, slot))
            return inShape.Elements == outShape.Elements ? start : null;
        if (writer.OpType == "Slice" && slot == 0 && SliceRun(writer, name) is { } run)
            return start + run.First * size;
        if (writer.OpType == "Concat" && ConcatPart(writer, slot, outShape) is { } part)
            return start - part * size;
        return null;
    }
}

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
    Func<NodeProto, bool> Writes)
{
    /// <summary>ONNX Runtime: what it may hand back over a kernel's input
    /// (<see cref="OutputAliasProof.Shares"/>) is that input's memory, and every other value of the
    /// graph is memory of its own — a <c>Slice</c> copies. A binding can take any value.</summary>
    internal static PlacementMemory OnnxRuntime { get; } = new(OutputAliasProof.Shares, OutputAliasProof.Shares, static _ => true);

    /// <summary>
    /// PyTorch run eagerly from a translation: a slice, a reshape, a transpose, an expansion and a
    /// split are views of their input unless placed — then written into their range — and so is
    /// whatever a runtime may hand back over its input. It writes into a given range the operators
    /// it has a form for that writes into a tensor: element-wise ones, a fill, a slice, a
    /// concatenation, and the copies a view makes.
    /// </summary>
    internal static PlacementMemory PyTorch { get; } = new(static (_, _, _) => false, TorchShares, TorchWrites);

    private static bool TorchShares(NodeProto node, int input, int output)
        => OutputAliasProof.Shares(node, input, output)
           || (OutputAliasProof.IsStandard(node) && input == 0 && node.OpType switch
           {
               "Slice" or "Reshape" or "Squeeze" or "Unsqueeze" or "Flatten" or "Transpose" or "Expand"
                   or "Identity" or "Dropout" => output == 0,
               "Split" => true,
               _ => false,
           });

    private static bool TorchWrites(NodeProto node)
        => OutputAliasProof.IsStandard(node)
           && (PlacementProof.InPlaceUnary.Contains(node.OpType) || PlacementProof.InPlaceBinary.Contains(node.OpType)
               || node.OpType is "ConstantOfShape" or "Slice" or "Concat" or "Clip" or "Identity" or "Reshape"
                   or "Squeeze" or "Unsqueeze" or "Flatten");
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
/// bytes, other than the second's writer, is an ancestor of the second's writer, its own writer is
/// too, and it is not read after the run — a graph output, or anything a graph output is a view of,
/// never is overwritten.</item>
/// <item><b>A writer reads what it overwrites only where it writes it.</b> Where V's writer reads an
/// occupant V overwrites, it reads each overlapped byte in the position it writes that byte: an
/// element-wise operator of one input, or of two, over an operand of V's own shape at V's own
/// offset; a <c>Slice</c> written at the offset it reads; a <c>Concat</c> part written where the
/// concatenation puts it.</item>
/// </list>
/// <para>Ancestry follows the edges that order execution whatever a runtime folds, as
/// <see cref="OutputAliasProof"/> does: explicit inputs, except those of a node that reads only a
/// shape, and the outer values a node's subgraphs read. Nothing depends on the order a runtime
/// picks among independent nodes, so a proof holds under a sequential order, a parallel executor,
/// and a card's single stream alike.</para>
/// </summary>
internal sealed class PlacementProof
{
    /// <summary>The most nodes a graph may have for placement to be considered: ancestry is kept
    /// as a bit set per node.</summary>
    internal const int MostNodes = 20000;

    /// <summary>The boundary every placement starts at: what a card's own allocations are aligned
    /// to, which kernels reading wide vectors rely on.</summary>
    internal const long Alignment = 256;

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

    /// <summary>
    /// The proof over <paramref name="graph"/> for one run: <paramref name="blocks"/> names each
    /// input the run consumed — the memory it may place values in — with its size in bytes, and
    /// <paramref name="shapes"/> holds every value's shape for the run
    /// (<see cref="PlacementShapes.Evaluate"/>). <paramref name="readAfterRun"/> names the graph
    /// outputs the caller reads once the run is over, each of which nothing may overwrite; null for
    /// every one of them. A graph output not named there is one a binding takes for the run alone,
    /// as a placed value made an output only so it can be bound to its range.
    /// <paramref name="memory"/> is how the backend lays values out; ONNX Runtime's where null.
    /// </summary>
    /// <exception cref="ArgumentException">The graph has more than <see cref="MostNodes"/> nodes,
    /// or is not in topological order.</exception>
    internal PlacementProof(
        GraphProto graph, IReadOnlyDictionary<string, long> blocks, IReadOnlyDictionary<string, PlacementShapes.Value> shapes,
        IReadOnlySet<string>? readAfterRun = null, PlacementMemory? memory = null)
    {
        _nodes = graph.Nodes;
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

    /// <summary>Whether node <paramref name="before"/> runs before node <paramref name="node"/>
    /// whatever order a runtime picks.</summary>
    private bool Precedes(int before, int node) => (_ancestors[node][before >> 6] & (1UL << (before & 63))) != 0;

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
            if (!_consumers.TryGetValue(current, out var consumers)) continue;
            var parent = chain[current];
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
    /// reads the whole occupant.
    /// </summary>
    private List<(int Node, long Start, long End)> ReadsOf(Dictionary<string, Member> chain, long offset, long bytes)
    {
        var reads = new List<(int, long, long)>();
        foreach (var (name, member) in chain)
        {
            var (start, end) = member.Contiguous ? (offset + member.Shift, offset + member.Shift + member.Bytes) : (offset, offset + bytes);
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

    /// <summary>Whether anything in <paramref name="chain"/> is read after the run.</summary>
    private bool ReadAfterRun(Dictionary<string, Member> chain) => chain.Keys.Any(_readAfterRun.Contains);

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
    /// or a placed value, the memory chain behind it, every read of it, and whether it is read
    /// after the run.</summary>
    private sealed record Occupant(
        Placement Placement, int Writer, Dictionary<string, Member> Chain, List<(int Node, long Start, long End)> Reads, bool AfterRun);

    private Occupant Content(string block, IReadOnlySet<string> placed)
    {
        var bytes = _blocks[block];
        var chain = ChainOf(block, bytes, placed);
        return new Occupant(new Placement(block, block, 0, bytes), -1, chain, ReadsOf(chain, 0, bytes), ReadAfterRun(chain));
    }

    private Occupant Placed(Placement placement, IReadOnlySet<string> placed)
    {
        var chain = ChainOf(placement.Value, placement.Bytes, placed);
        return new Occupant(placement, _producer[placement.Value], chain, ReadsOf(chain, placement.Offset, placement.Bytes), ReadAfterRun(chain));
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
        if (!_memory.Writes(node)) return "written by an operator the backend cannot write into a range";
        if (!_shapes.TryGetValue(value, out var shape)) return "of unknown shape";
        if (shape.Bytes <= 0) return "of no fixed-width element type";
        return null;
    }

    /// <summary>Whether occupant <paramref name="first"/> comes before <paramref name="second"/>
    /// on the bytes their ranges share.</summary>
    private bool Before(Occupant first, Occupant second)
    {
        if (first.AfterRun) return false;
        var w = second.Writer;
        if (w < 0) return false;
        if (first.Writer >= 0 && (first.Writer == w || !Precedes(first.Writer, w))) return false;
        var lo = Math.Max(first.Placement.Offset, second.Placement.Offset);
        var hi = Math.Min(first.Placement.End, second.Placement.End);
        foreach (var (reader, start, end) in first.Reads)
        {
            if (end <= lo || start >= hi) continue;
            if (reader == w) continue;
            if (!Precedes(reader, w)) return false;
        }
        foreach (var (start, end, identical) in WriterReads(w, first, second.Placement))
            if (end > lo && start < hi && !identical) return false;
        return true;
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
                else if (node.OpType == "Slice" && slot == 0 && SliceRun(node, name) is { } run && inBytes > 0)
                {
                    start += run.First * inBytes;
                    end = start + run.Count * inBytes;
                    identical = inBytes == outBytes && written.Offset == start;
                }
                else if (node.OpType == "Concat" && ConcatPart(node, slot, outShape) is { } part && inBytes == outBytes)
                    identical = written.Offset + part * outBytes == start;
            }
            yield return (start, end, identical);
        }
    }

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
    private List<Occupant> Occupants(IReadOnlyList<Placement> placements)
    {
        var placed = placements.Select(p => p.Value).ToHashSet(StringComparer.Ordinal);
        return [.. _blocks.Keys.Select(b => Content(b, placed)), .. placements.Select(p => Placed(p, placed))];
    }

    /// <summary>Whether <paramref name="value"/> is a view its writer hands back unless it is
    /// placed, so placing it changes the memory chain of what it views.</summary>
    private bool LeavesAChain(string value)
    {
        var node = _nodes[_producer[value]];
        var o = node.Outputs.IndexOf(value);
        for (int i = 0; i < node.Inputs.Count; i++)
            if (_memory.SharesUnlessPlaced(node, i, o)) return true;
        return false;
    }

    /// <summary>
    /// <paramref name="occupants"/> — those of <paramref name="accepted"/> — with
    /// <paramref name="candidate"/> placed too, where every pair of them is compatible; null where
    /// the candidate does not fit.
    /// </summary>
    private List<Occupant>? With(List<Occupant> occupants, List<Placement> accepted, Placement candidate)
    {
        if (!Fits(candidate)) return null;
        if (!LeavesAChain(candidate.Value))
        {
            var occupant = Placed(candidate, accepted.Select(p => p.Value).Append(candidate.Value).ToHashSet(StringComparer.Ordinal));
            return occupants.All(other => Compatible(other, occupant)) ? [.. occupants, occupant] : null;
        }
        // Placing a view writes it into its own range, which takes it out of the memory chain it was
        // part of: every occupant is laid out again, and every pair checked.
        var rebuilt = Occupants([.. accepted, candidate]);
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

    // ---- planning ----

    /// <summary>
    /// Placements this graph proves, chosen to need as little memory beyond the blocks as it can:
    /// the outputs read after the run first, largest first, since nothing may overwrite one and where
    /// one goes constrains everything else, then every other value in node order. Each value of at
    /// least <paramref name="smallest"/> bytes tries the ranges its writer writes in place — an
    /// element-wise operand's, a slice's source, a concatenation part's slot — then the first aligned
    /// range that fits. An output placed in a block whose outputs would leave more than
    /// <paramref name="idleOutputBytes"/> of it unused after the run is not placed: the block lives as
    /// long as its outputs.
    /// </summary>
    internal IReadOnlyList<Placement> Plan(long smallest, long idleOutputBytes)
    {
        if (_blocks.Count == 0) return [];
        var candidates = new List<string>();
        for (int n = 0; n < _nodes.Count; n++)
            foreach (var value in _nodes[n].Outputs)
                if (value.Length > 0 && Unplaceable(value) is null && _shapes[value].Bytes >= smallest)
                    candidates.Add(value);
        var outputs = candidates.Where(_readAfterRun.Contains).OrderByDescending(v => _shapes[v].Bytes).ToList();
        var others = candidates.Where(v => !_readAfterRun.Contains(v)).ToList();

        var placed = new List<Placement>();
        var occupants = Occupants(placed);
        void Try(string value)
        {
            foreach (var placement in Options(value, occupants))
            {
                if (With(occupants, placed, placement) is not { } next) continue;
                occupants = next;
                placed.Add(placement);
                return;
            }
        }
        foreach (var value in outputs) Try(value);

        // An output that would keep more of its block idle than it uses goes back to the backend.
        var idle = placed.Where(p => _readAfterRun.Contains(p.Value)).GroupBy(p => p.Block)
            .Where(g => _blocks[g.Key] - g.Sum(p => p.Bytes) > idleOutputBytes).Select(g => g.Key).ToHashSet();
        if (idle.Count > 0)
        {
            placed.RemoveAll(p => idle.Contains(p.Block));
            occupants = Occupants(placed);
        }

        foreach (var value in others) Try(value);
        return Prove(placed);
    }

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
        if (!OutputAliasProof.IsStandard(writer) || !_shapes.TryGetValue(name, out var inShape)) return null;
        var outShape = _shapes[value];
        var size = PlacementShapes.ElementBytes(inShape.ElementType);
        if (size != PlacementShapes.ElementBytes(outShape.ElementType)) return null;
        if ((InPlaceUnary.Contains(writer.OpType) && slot == 0) || (writer.OpType == "Clip" && slot == 0)
            || (InPlaceBinary.Contains(writer.OpType) && writer.Inputs.Count == 2))
            return inShape.Shape.SequenceEqual(outShape.Shape) ? start : null;
        if (writer.OpType == "Slice" && slot == 0 && SliceRun(writer, name) is { } run)
            return start + run.First * size;
        if (writer.OpType == "Concat" && ConcatPart(writer, slot, outShape) is { } part)
            return start - part * size;
        return null;
    }
}

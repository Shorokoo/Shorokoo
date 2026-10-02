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
/// Which values of an ONNX graph may be written into which byte ranges of the memory of the inputs a
/// run consumes, for one run: the proof behind placement, and a planner that proposes placements for
/// it to prove. It answers for the graph it is given — the graph the backend runs, after its own
/// rewrites — with the shapes the run's inputs come in.
///
/// <para>A placement puts value V, written by node P, at a range of a block. It is proved when:</para>
/// <list type="bullet">
/// <item><b>V is a value of its own</b>: a fixed-width tensor written by a node of the graph — not a
/// graph input, not an initializer (a constant a backend folded), not the output of a node holding a
/// subgraph, and not an output a runtime may hand back as a view of the node's input
/// (<see cref="OutputAliasProof.Shares"/>); its shape is known, and it lies inside its block at
/// exactly its size.</item>
/// <item><b>Its block's occupants are ordered.</b> A block's occupants are the input's own content
/// — read by every node reading the input or a view of it, each over the range it reads (the whole
/// block, or the contiguous run a <c>Slice</c> takes) — and the values placed in it. Of any two
/// occupants whose ranges overlap, one comes first: every node reading its overlapped bytes, other
/// than the second's writer, is an ancestor of the second's writer, its own writer is too, and it is
/// not read after the run — a graph output, or anything a graph output is a view of, never is
/// overwritten.</item>
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

    private readonly List<NodeProto> _nodes;
    private readonly Dictionary<string, int> _producer = new(StringComparer.Ordinal);
    private readonly Dictionary<string, List<(int Node, int Slot)>> _consumers = new(StringComparer.Ordinal);
    private readonly Dictionary<string, List<int>> _capturedBy = new(StringComparer.Ordinal);
    private readonly Dictionary<string, int> _outputs = new(StringComparer.Ordinal);
    private readonly HashSet<string> _inputs = new(StringComparer.Ordinal);
    private readonly HashSet<string> _initializers = new(StringComparer.Ordinal);
    private readonly IReadOnlyDictionary<string, long> _blocks;
    private readonly IReadOnlyDictionary<string, PlacementShapes.Value> _shapes;
    private readonly ulong[][] _ancestors;

    /// <summary>
    /// The proof over <paramref name="graph"/> for one run: <paramref name="blocks"/> names each
    /// input the run consumed — the memory it may place values in — with its size in bytes, and
    /// <paramref name="shapes"/> holds every value's shape for the run
    /// (<see cref="PlacementShapes.Evaluate"/>).
    /// </summary>
    /// <exception cref="ArgumentException">The graph has more than <see cref="MostNodes"/> nodes,
    /// or is not in topological order.</exception>
    internal PlacementProof(
        GraphProto graph, IReadOnlyDictionary<string, long> blocks, IReadOnlyDictionary<string, PlacementShapes.Value> shapes)
    {
        _nodes = graph.Nodes;
        if (_nodes.Count > MostNodes) throw new ArgumentException($"The graph has more than {MostNodes} nodes.", nameof(graph));
        _blocks = blocks;
        _shapes = shapes;
        foreach (var input in graph.Inputs) _inputs.Add(input.Name);
        foreach (var initializer in graph.Initializers) _initializers.Add(initializer.Name);
        foreach (var output in graph.Outputs) _outputs[output.Name] = _outputs.GetValueOrDefault(output.Name) + 1;

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

    // ---- what reads what ----

    /// <summary>A value and every view of it a runtime may hand back over its memory.</summary>
    private HashSet<string> ChainOf(string name)
    {
        var chain = new HashSet<string>(StringComparer.Ordinal) { name };
        var pending = new Queue<string>();
        pending.Enqueue(name);
        while (pending.TryDequeue(out var current))
        {
            if (!_consumers.TryGetValue(current, out var consumers)) continue;
            foreach (var (n, slot) in consumers)
            {
                var node = _nodes[n];
                for (int o = 0; o < node.Outputs.Count; o++)
                    if (node.Outputs[o].Length > 0 && OutputAliasProof.Shares(node, slot, o) && chain.Add(node.Outputs[o]))
                        pending.Enqueue(node.Outputs[o]);
            }
        }
        return chain;
    }

    /// <summary>
    /// Every read of the memory behind <paramref name="chain"/> — a value and its views, at
    /// <paramref name="offset"/> in its block, <paramref name="bytes"/> long: per reading node, the
    /// range of the block it reads. A <c>Shape</c> or <c>Size</c> reads nothing; a <c>Slice</c>
    /// taking one contiguous run reads that run; anything else reads it all, a node whose subgraph
    /// refers to it included.
    /// </summary>
    private List<(int Node, long Start, long End)> ReadsOf(HashSet<string> chain, long offset, long bytes)
    {
        var reads = new List<(int, long, long)>();
        foreach (var name in chain)
        {
            if (_consumers.TryGetValue(name, out var consumers))
                foreach (var (n, slot) in consumers)
                {
                    var node = _nodes[n];
                    if (OutputAliasProof.ReadsOnlyAShape(node)) continue;
                    if (node.OpType == "Slice" && OutputAliasProof.IsStandard(node) && slot == 0
                        && SliceRun(node, name) is { } run && ElementBytesOf(name) is var size and > 0)
                    {
                        reads.Add((n, offset + run.First * size, offset + (run.First + run.Count) * size));
                        continue;
                    }
                    reads.Add((n, offset, offset + bytes));
                }
            if (_capturedBy.TryGetValue(name, out var holders))
                foreach (var n in holders) reads.Add((n, offset, offset + bytes));
        }
        return reads;
    }

    /// <summary>Whether anything in <paramref name="chain"/> is read after the run: a graph
    /// output.</summary>
    private bool ReadAfterRun(HashSet<string> chain) => chain.Any(_outputs.ContainsKey);

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
        Placement Placement, int Writer, HashSet<string> Chain, List<(int Node, long Start, long End)> Reads, bool AfterRun);

    private Occupant Content(string block)
    {
        var bytes = _blocks[block];
        var chain = ChainOf(block);
        return new Occupant(new Placement(block, block, 0, bytes), -1, chain, ReadsOf(chain, 0, bytes), ReadAfterRun(chain));
    }

    private Occupant Placed(Placement placement)
    {
        var chain = ChainOf(placement.Value);
        return new Occupant(placement, _producer[placement.Value], chain, ReadsOf(chain, placement.Offset, placement.Bytes), ReadAfterRun(chain));
    }

    /// <summary>
    /// Why <paramref name="value"/> cannot be placed at all, or null where it may be: it must be a
    /// fixed-width tensor of known shape written by a node of the graph that holds no subgraph, as
    /// that node's own memory rather than a view of an input.
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
            if (OutputAliasProof.Shares(node, i, o)) return "a view of its writer's input";
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

    // Element-wise operators reading each element of an operand of the output's shape in the position
    // they write it, in one pass: the standard ones of one input, and of exactly two.
    private static readonly HashSet<string> Unary = new(StringComparer.Ordinal)
    {
        "Abs", "Neg", "Sigmoid", "Relu", "Exp", "Log", "Sqrt", "Tanh", "Sin", "Cos", "Tan", "Asin", "Acos",
        "Atan", "Sinh", "Cosh", "Asinh", "Acosh", "Atanh", "Reciprocal", "Floor", "Ceil", "Round", "Sign",
        "Erf", "Softplus", "Softsign", "Elu", "Selu", "LeakyRelu", "ThresholdedRelu", "HardSigmoid",
        "HardSwish", "Celu", "Not", "BitwiseNot", "Gelu",
    };

    private static readonly HashSet<string> Binary = new(StringComparer.Ordinal)
    {
        "Add", "Sub", "Mul", "Div", "Pow", "Max", "Min", "Sum", "Mean", "PRelu", "BitwiseAnd", "BitwiseOr",
        "BitwiseXor", "And", "Or", "Xor",
    };

    /// <summary>
    /// Each read node <paramref name="writer"/> makes of the memory behind <paramref name="occupant"/>
    /// while writing <paramref name="written"/>: the range of the block it reads, and whether it
    /// reads it only in the positions it writes.
    /// </summary>
    private IEnumerable<(long Start, long End, bool Identical)> WriterReads(int writer, Occupant occupant, Placement written)
    {
        var node = _nodes[writer];
        var standard = OutputAliasProof.IsStandard(node);
        var output = written.Value;
        _shapes.TryGetValue(output, out var outShape);
        var outBytes = outShape is null ? 0 : PlacementShapes.ElementBytes(outShape.ElementType);
        for (int slot = 0; slot < node.Inputs.Count; slot++)
        {
            var name = node.Inputs[slot];
            if (name.Length == 0 || !occupant.Chain.Contains(name)) continue;
            var start = occupant.Placement.Offset;
            var end = occupant.Placement.End;
            var identical = false;
            if (standard && _shapes.TryGetValue(name, out var inShape) && outShape is not null)
            {
                var inBytes = PlacementShapes.ElementBytes(inShape.ElementType);
                var sameElements = inBytes == outBytes && inBytes > 0 && inShape.Shape.SequenceEqual(outShape.Shape);
                if ((Unary.Contains(node.OpType) && slot == 0 && node.Outputs.Count == 1)
                    || (node.OpType == "Clip" && slot == 0)
                    || (Binary.Contains(node.OpType) && node.Inputs.Count == 2))
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
    /// The placements of <paramref name="placements"/> this graph proves, in the order given: each is
    /// kept where it is a value that may be placed (<see cref="Unplaceable"/>), lies at its exact size
    /// inside a block, and every occupant kept so far — the blocks' own contents first — is
    /// compatible with it.
    /// </summary>
    internal IReadOnlyList<Placement> Prove(IEnumerable<Placement> placements)
    {
        var occupants = _blocks.Keys.Select(Content).ToList();
        var kept = new List<Placement>();
        var values = new HashSet<string>(StringComparer.Ordinal);
        foreach (var placement in placements)
        {
            if (!values.Add(placement.Value)) continue;
            if (!Fits(placement)) { values.Remove(placement.Value); continue; }
            var occupant = Placed(placement);
            if (!occupants.All(other => Compatible(other, occupant))) { values.Remove(placement.Value); continue; }
            occupants.Add(occupant);
            kept.Add(placement);
        }
        return kept;
    }

    private bool Fits(Placement placement)
        => _blocks.TryGetValue(placement.Block, out var blockBytes)
           && Unplaceable(placement.Value) is null
           && _shapes[placement.Value].Bytes == placement.Bytes
           && placement.Offset >= 0 && placement.End <= blockBytes
           && !_blocks.ContainsKey(placement.Value);

    // ---- planning ----

    /// <summary>
    /// Placements this graph proves, chosen to need as little memory beyond the blocks as it can:
    /// the graph outputs first, largest first, since nothing may overwrite one and where one goes
    /// constrains everything else, then every other value in node order. Each value of at least
    /// <paramref name="smallest"/> bytes tries the ranges its writer writes in place — an
    /// element-wise operand's, a slice's source, a concatenation part's slot — then the first range
    /// that fits. An output placed in a block whose outputs would leave more than
    /// <paramref name="idleOutputBytes"/> of it unused after the run is not placed: the block lives
    /// as long as its outputs.
    /// </summary>
    internal IReadOnlyList<Placement> Plan(long smallest, long idleOutputBytes)
    {
        if (_blocks.Count == 0) return [];
        var candidates = new List<string>();
        for (int n = 0; n < _nodes.Count; n++)
            foreach (var value in _nodes[n].Outputs)
                if (value.Length > 0 && Unplaceable(value) is null && _shapes[value].Bytes >= smallest)
                    candidates.Add(value);
        var outputs = candidates.Where(_outputs.ContainsKey).OrderByDescending(v => _shapes[v].Bytes).ToList();
        var others = candidates.Where(v => !_outputs.ContainsKey(v)).ToList();

        var occupants = _blocks.Keys.Select(Content).ToList();
        var placed = new List<Placement>();
        void Try(string value)
        {
            foreach (var placement in Options(value, occupants))
            {
                var occupant = Placed(placement);
                if (!occupants.All(other => Compatible(other, occupant))) continue;
                occupants.Add(occupant);
                placed.Add(placement);
                return;
            }
        }
        foreach (var value in outputs) Try(value);

        // An output that would keep more of its block idle than it uses goes back to the backend.
        var idle = placed.Where(p => _outputs.ContainsKey(p.Value)).GroupBy(p => p.Block)
            .Where(g => _blocks[g.Key] - g.Sum(p => p.Bytes) > idleOutputBytes).Select(g => g.Key).ToHashSet();
        if (idle.Count > 0)
        {
            placed.RemoveAll(p => idle.Contains(p.Block));
            occupants = [.. _blocks.Keys.Select(Content), .. placed.Select(Placed)];
        }

        foreach (var value in others) Try(value);
        return Prove(placed);
    }

    /// <summary>Where <paramref name="value"/> may go, best first: the ranges its writer writes in
    /// place, then every range starting at a block's start or right after an occupant.</summary>
    private IEnumerable<Placement> Options(string value, List<Occupant> occupants)
    {
        var bytes = _shapes[value].Bytes;
        var writer = _nodes[_producer[value]];
        var seen = new HashSet<(string, long)>();
        foreach (var occupant in occupants)
        {
            var hypothetical = new Placement(value, occupant.Placement.Block, 0, bytes);
            foreach (var slot in Enumerable.Range(0, writer.Inputs.Count))
            {
                var name = writer.Inputs[slot];
                if (name.Length == 0 || !occupant.Chain.Contains(name)) continue;
                long? offset = InPlaceOffset(writer, slot, name, occupant, value);
                if (offset is { } at && at >= 0 && at + bytes <= _blocks[occupant.Placement.Block] && seen.Add((occupant.Placement.Block, at)))
                    yield return hypothetical with { Offset = at };
            }
        }
        foreach (var (block, blockBytes) in _blocks)
        {
            var starts = new SortedSet<long> { 0 };
            foreach (var occupant in occupants)
                if (occupant.Placement.Block == block && occupant.Writer >= 0) starts.Add(occupant.Placement.End);
            foreach (var at in starts)
                if (at + bytes <= blockBytes && seen.Add((block, at)))
                    yield return new Placement(value, block, at, bytes);
        }
    }

    /// <summary>The offset at which <paramref name="writer"/> would write <paramref name="value"/>
    /// in the position it reads input <paramref name="name"/> at slot <paramref name="slot"/>, behind
    /// <paramref name="occupant"/>; null where it reads it no such way.</summary>
    private long? InPlaceOffset(NodeProto writer, int slot, string name, Occupant occupant, string value)
    {
        if (!OutputAliasProof.IsStandard(writer) || !_shapes.TryGetValue(name, out var inShape)) return null;
        var outShape = _shapes[value];
        var size = PlacementShapes.ElementBytes(inShape.ElementType);
        if (size != PlacementShapes.ElementBytes(outShape.ElementType)) return null;
        var start = occupant.Placement.Offset;
        if ((Unary.Contains(writer.OpType) && slot == 0) || (writer.OpType == "Clip" && slot == 0)
            || (Binary.Contains(writer.OpType) && writer.Inputs.Count == 2))
            return inShape.Shape.SequenceEqual(outShape.Shape) ? start : null;
        if (writer.OpType == "Slice" && slot == 0 && SliceRun(writer, name) is { } run)
            return start + run.First * size;
        if (writer.OpType == "Concat" && ConcatPart(writer, slot, outShape) is { } part)
            return start - part * size;
        return null;
    }
}

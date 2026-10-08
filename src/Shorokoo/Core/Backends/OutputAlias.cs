using Shorokoo.Core.Factory.IR;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Graph;

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
/// A state pair Shorokoo's own graph would prove but for the order some of its readers run in: see
/// <see cref="OutputAliasProof.Unordered"/>. Positions are into the graph's nodes.
/// </summary>
/// <param name="Pair">The pair, by position: output <c>Output</c> of the graph, input <c>Input</c>.</param>
/// <param name="Writer">The node writing the output.</param>
/// <param name="Readers">Each reader of the input that is not an ancestor of the writer: one node
/// (<c>First</c> equal to <c>Last</c>), or an opaque region of them — a scope, from its <c>OPEN</c>
/// to its <c>CLOSE</c>.</param>
internal sealed record UnorderedStatePair(
    (int Output, int Input) Pair, int Writer, IReadOnlyList<(int First, int Last)> Readers);

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
/// variance it was given. Ancestry follows a node's inputs and the outer values its subgraphs
/// read, which a runtime has computed before it runs the node, and counts only the edges a runtime
/// cannot fold away: an edge into the standard <c>Shape</c> or <c>Size</c> — which read no memory
/// and are no readers — orders nothing once the shape is known when the session is built, so it
/// does not count.</item>
/// <item><c>P</c> itself reads <c>I</c> only as the first operand of a two-input <c>Add</c>,
/// <c>Sub</c>, <c>Mul</c> or <c>Div</c>, which reads each element before writing the same element
/// of its output — the in-place form ONNX Runtime uses for these operators itself. With <c>O</c>
/// and <c>I</c> of one shape, that operand is not broadcast. Or <c>P</c> is Shorokoo's fused
/// optimizer update (<c>ai.shorokoo</c> <c>AdamUpdate</c>), <c>O</c> its parameter or moment output
/// and <c>I</c> read only as the input of the same position, which it updates element by element
/// in the same way.</item>
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
    /// <exception cref="ProtoBuf.ProtoException"><paramref name="model"/> is not a model protobuf reads:
    /// malformed, or nested deeper than it reads.</exception>
    public static IReadOnlyList<OutputAlias> Prove(byte[] model, IEnumerable<OutputAlias> candidates)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(candidates);
        ModelProto parsed;
        using (var stream = new MemoryStream(model, writable: false))
            parsed = Shorokoo.Onnx.OnnxProtobuf.ReadModel(stream);
        return parsed.Graph is { } graph ? Prove(graph, candidates) : [];
    }

    /// <summary>
    /// The candidates <paramref name="graph"/> proves, in the order they were given: the answer
    /// <see cref="Prove(byte[], IEnumerable{OutputAlias})"/> gives for a model already parsed, so a
    /// caller that reads more of the model than the proof does parses it once.
    /// </summary>
    /// <exception cref="ArgumentNullException">An argument is null.</exception>
    public static IReadOnlyList<OutputAlias> Prove(GraphProto graph, IEnumerable<OutputAlias> candidates)
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
    /// The pairs of <paramref name="candidates"/> — by position, output <c>Output</c> into input
    /// <c>Input</c> — that Shorokoo's own <paramref name="graph"/> proves, in the order they were
    /// given: the rule above, asked of the ONNX graph <paramref name="graph"/> is emitted as, so a
    /// pass rewriting the graph can see which of its pairs a rewrite keeps without emitting it. A
    /// candidate naming a position the graph does not have is no candidate.
    ///
    /// <para>The projection keeps what the rule reads — each node's op type, the values it reads
    /// and writes, the graph's inputs and outputs — and leaves out types, which it states nowhere;
    /// a caller holding shapes compares them itself. What it cannot give an ONNX meaning is made
    /// opaque, never lenient: a scope (<c>OPEN</c>..<c>CLOSE</c>, the body of a <c>Loop</c> or
    /// <c>If</c>) and a Shorokoo-internal or function node each become one node holding a subgraph
    /// that reads whatever the region reads. So a value such a region reads is refused as an
    /// input, and one it writes as an output, exactly as the rule refuses what a subgraph touches;
    /// and the region still orders what follows it.</para>
    ///
    /// <para>This graph is the one a backend receives, not the one it runs, and the proof counts
    /// the readers a backend's folding can create out of the input. Folding away work that
    /// depended on the input alone — two transposes that cancel, a reshape or cast undone by
    /// another — turns a node which read the result into a reader of the input itself. Here, then,
    /// a node also reads the input when it reads a value computed from the input, its views and
    /// constants alone (the outputs of <c>Shape</c> and <c>Size</c> count as constants, as a
    /// backend folds them where shapes are known), other than through the output's writer; each
    /// such node must be an ancestor of the writer like any other reader, and a subgraph referring
    /// to such a value refuses the pair. That covers the readers such folding makes. It does not
    /// cover every rewrite a backend makes — a fusion can place the node it creates where no node
    /// of this graph stood, as a <c>MatMul</c> and the <c>Add</c> after it fused into one
    /// <c>Gemm</c> at the <c>Add</c> — so this proof is the model's estimate, and the backend's own
    /// proof over the graph it runs (<see cref="Prove(GraphProto, IEnumerable{OutputAlias})"/>)
    /// decides what is written in place. The writer's own reads of such values are not held
    /// against it: a fold that turns one into a read of the input as its second operand is refused
    /// by that backend proof, never written in place.</para>
    /// </summary>
    internal static IReadOnlyList<(int Output, int Input)> Prove(
        InternalComputationGraph graph, IReadOnlyList<(int Output, int Input)> candidates)
        => [.. Classify(graph, candidates, provedOnly: true).Select(pair => pair.Pair)];

    /// <summary>
    /// The pairs of <paramref name="candidates"/> Shorokoo's own <paramref name="graph"/> would
    /// prove — as <see cref="Prove(InternalComputationGraph, IReadOnlyList{ValueTuple{int, int}})"/>
    /// proves them, counting the readers folding can make — were each of its readers an ancestor
    /// of its writer, in the order they were given, with the readers that are not. A pair with no
    /// such reader is proved; the others are what ordering the writer after those readers would
    /// prove (<see cref="Shorokoo.Core.AutoDiffCheckpointing.StateReadOrdering"/>). Each input
    /// backs at most one pair, and each output at most one, the first candidate listed taking both.
    /// </summary>
    internal static IReadOnlyList<UnorderedStatePair> Unordered(
        InternalComputationGraph graph, IReadOnlyList<(int Output, int Input)> candidates)
        => Classify(graph, candidates, provedOnly: false);

    private static List<UnorderedStatePair> Classify(
        InternalComputationGraph graph, IReadOnlyList<(int Output, int Input)> candidates, bool provedOnly)
    {
        if (candidates.Count == 0) return [];
        var (proto, spans) = Project(graph);
        var index = new GraphIndex(proto);
        var takenInputs = new HashSet<int>();
        var takenOutputs = new HashSet<int>();
        var found = new List<UnorderedStatePair>();
        foreach (var (output, input) in candidates)
        {
            if (output < 0 || output >= proto.Outputs.Count || input < 0 || input >= proto.Inputs.Count) continue;
            if (takenInputs.Contains(input) || takenOutputs.Contains(output)) continue;
            var alias = new OutputAlias(proto.Outputs[output].Name, proto.Inputs[input].Name);
            if (index.Unordered(alias, foldable: true) is not { } unordered) continue;
            if (provedOnly && unordered.Readers.Count > 0) continue;
            takenInputs.Add(input);
            takenOutputs.Add(output);
            found.Add(new UnorderedStatePair(
                (output, input), spans[unordered.Writer].First, [.. unordered.Readers.Select(r => spans[r])]));
        }
        return found;
    }

    /// <summary>
    /// <paramref name="graph"/> as the ONNX structure the rule reads (see
    /// <see cref="Prove(InternalComputationGraph, IReadOnlyList{ValueTuple{int, int}})"/>), with,
    /// for each node of it, the positions in <paramref name="graph"/>'s nodes it stands for: one
    /// node, or the whole of an opaque region.
    /// </summary>
    private static (GraphProto Proto, List<(int First, int Last)> Spans) Project(InternalComputationGraph graph)
    {
        var names = new Dictionary<FastTensorKey, string>();
        string Name(FastTensorKey? key)
        {
            if (key is not { } k) return "";
            if (!names.TryGetValue(k, out var name))
                names[k] = name = names.Count.ToString(System.Globalization.CultureInfo.InvariantCulture);
            return name;
        }

        var proto = new GraphProto();
        var spans = new List<(int First, int Last)>();
        foreach (var input in graph.Inputs) proto.Inputs.Add(new ValueInfoProto { Name = Name(input) });
        foreach (var output in graph.Outputs) proto.Outputs.Add(new ValueInfoProto { Name = Name(output) });

        var nodes = graph.Nodes;
        var end = graph.BodyEnd;
        for (int i = graph.InputCount; i < end; i++)
        {
            var node = nodes[i];
            var last = i;
            if (node.IsOpenNode())
                for (int depth = 0; last < end; last++)
                {
                    if (nodes[last].IsOpenNode()) depth++;
                    else if (nodes[last].IsCloseNode() && --depth == 0) break;
                }
            else if (node.TargetFunction is null
                     && Definitions.VanillaOpNames.Contains(node.OpCode))
            {
                var projected = new NodeProto { OpType = node.OpCode };
                foreach (var input in Flattened(node.FullInputs)) projected.Inputs.Add(Name(input));
                foreach (var output in Flattened(node.FullOutputs)) projected.Outputs.Add(Name(output));
                proto.Nodes.Add(projected);
                spans.Add((i, i));
                continue;
            }

            // Opaque: the region's outputs on the outer node, and everything it reads from outside
            // itself on the one node of a subgraph it holds.
            last = Math.Min(last, end - 1);
            var region = new NodeProto { OpType = node.OpCode };
            var reader = new NodeProto { OpType = node.OpCode };
            for (int r = i; r <= last; r++)
                foreach (var output in Flattened(nodes[r].FullOutputs))
                    if (output is not null) region.Outputs.Add(Name(output));
            var written = new HashSet<string>(region.Outputs, StringComparer.Ordinal);
            for (int r = i; r <= last; r++)
                foreach (var input in Flattened(nodes[r].FullInputs))
                    if (input is not null && !written.Contains(Name(input))) reader.Inputs.Add(Name(input));
            var body = new GraphProto();
            body.Nodes.Add(reader);
            region.Attributes.Add(new AttributeProto { Name = "body", Type = AttributeProto.AttributeType.Graph, G = body });
            proto.Nodes.Add(region);
            spans.Add((i, last));
            i = last;
        }
        return (proto, spans);
    }

    /// <summary>A node's values in <see cref="FastNode.Inputs"/> order — its slots by ordinal
    /// name — without sorting the one slot most nodes have.</summary>
    private static IEnumerable<FastTensorKey?> Flattened(Dictionary<string, List<FastTensorKey?>> slots)
        => slots.Count == 1 ? slots.Values.First()
            : slots.OrderBy(slot => slot.Key, StringComparer.Ordinal).SelectMany(slot => slot.Value);

    /// <summary>
    /// Whether a runtime may hand back <paramref name="node"/>'s input at position
    /// <paramref name="input"/> as its output at position <paramref name="output"/>, the two being
    /// one buffer: whoever reads that output reads the input. By position, because that is how
    /// ONNX Runtime declares it on a kernel, and binds it even where the input is a graph input.
    /// Its CPU and CUDA kernels in both the standard and Microsoft domains declare the pairs below:
    /// an output 0 over input 0, <c>BatchNormalization</c> in training mode writing its running mean
    /// and variance into the buffers of the mean and variance it was given, and, where the runtime
    /// is built with NCCL, <c>AllReduce</c> handing back each of its inputs. <c>Dropout</c>, an
    /// identity outside training, is counted as one too; a sequence built by
    /// <c>SequenceConstruct</c> or <c>SequenceInsert</c> may hold the memory of any tensor it was
    /// given, and what <c>SequenceErase</c> leaves of a sequence the memory that sequence held.
    /// Named whatever their domain, which only ever widens what counts as a reader.
    /// </summary>
    internal static bool Shares(NodeProto node, int input, int output) => node.OpType switch
    {
        "Identity" or "Reshape" or "Squeeze" or "Unsqueeze" or "Flatten" or "Dropout" or "ExpandDims"
            or "Optional" or "OptionalGetElement" => (input, output) is (0, 0),
        "BatchNormalization" => (input, output) is (3, 1) or (4, 2),
        "SequenceConstruct" or "SequenceInsert" => output == 0,
        "SequenceErase" => (input, output) is (0, 0),
        "AllReduce" => input == output,
        _ => false,
    };

    // Whether the node reads a tensor's shape and none of its memory: the standard Shape and Size,
    // and nothing else of those names -- an operator of another domain may read whatever it likes.
    internal static bool ReadsOnlyAShape(NodeProto node) => IsStandard(node) && node.OpType is "Shape" or "Size";

    // The element-wise operators whose first operand the output may be written over: each element
    // is read before the same element of the output is written, and ONNX Runtime writes these in
    // place itself.
    private static readonly HashSet<string> InPlace = new(StringComparer.Ordinal) { "Add", "Sub", "Mul", "Div" };

    internal static bool IsStandard(NodeProto node) => node.Domain is "" or "ai.onnx";

    // Shorokoo's own fused optimizer update, the AdamUpdate of the ai.shorokoo domain that the ONNX
    // Runtime backend writes an Adam or AdamW update as: it writes its outputs 0, 1 and 2 (the
    // parameter and the two moments) over its inputs 0, 1 and 2, element by element, each element
    // read before the same element is written, and reads its other inputs whole.
    private const int UpdatedInPlace = 3;

    private static bool IsUpdate(NodeProto node) => node.Domain == "ai.shorokoo" && node.OpType == "AdamUpdate";

    internal static bool HoldsSubgraph(NodeProto node)
        => node.Attributes.Any(a => a.G is not null || a.Graphs.Count > 0);

    /// <summary>One graph, indexed for the questions the proof asks of it.</summary>
    internal sealed class GraphIndex
    {
        private readonly List<NodeProto> _nodes;
        private readonly Dictionary<string, int> _producer = new(StringComparer.Ordinal);
        private readonly Dictionary<string, List<int>> _consumers = new(StringComparer.Ordinal);
        private readonly HashSet<string> _referencedBySubgraphs = new(StringComparer.Ordinal);
        private readonly HashSet<string> _inputs = new(StringComparer.Ordinal);
        private readonly HashSet<string> _initializers = new(StringComparer.Ordinal);
        private readonly Dictionary<string, int> _outputs = new(StringComparer.Ordinal);
        private readonly Dictionary<string, TypeProto> _types = new(StringComparer.Ordinal);

        // Per node, the outer values its subgraphs read other than through a shape.
        private readonly Dictionary<int, HashSet<string>> _captured = [];

        // A stamp per node for the ancestry walks, so each walk marks what it visited without
        // clearing an array first.
        private readonly int[] _visited;
        private int _stamp;

        // What Constants() found, once it has been asked.
        private HashSet<string>? _constants;

        // What OrderingProducers() found, once it has been asked.
        private int[][]? _orderingProducers;

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
                if (HoldsSubgraph(node))
                {
                    var captured = new HashSet<string>(StringComparer.Ordinal);
                    foreach (var attribute in node.Attributes)
                    {
                        if (attribute.G is { } subgraph) ReferencedFrom(subgraph, captured, orderingOnly: true);
                        foreach (var each in attribute.Graphs) ReferencedFrom(each, captured, orderingOnly: true);
                    }
                    _captured[n] = captured;
                }
            }
        }

        /// <summary>Whether this graph proves <paramref name="alias"/>.</summary>
        internal bool Proves(OutputAlias alias) => Unordered(alias, foldable: false) is { Readers.Count: 0 };

        /// <summary>
        /// Null where this graph refuses <paramref name="alias"/> whatever order its nodes run in;
        /// otherwise the node writing the output, and the readers of the input that are not its
        /// ancestors — none where the graph proves the pair. With <paramref name="foldable"/>, a
        /// node reading a value computed from the input and constants alone is a reader too (see
        /// <see cref="AddFoldableReaders"/>).
        /// </summary>
        internal (int Writer, List<int> Readers)? Unordered(OutputAlias alias, bool foldable)
        {
            var (input, output) = (alias.Input, alias.Output);
            if (!_inputs.Contains(input) || _initializers.Contains(input)) return null;
            if (!_outputs.TryGetValue(output, out var listed) || listed != 1 || _inputs.Contains(output))
                return null;
            if (!_producer.TryGetValue(output, out var writer)) return null;
            var p = _nodes[writer];
            if (HoldsSubgraph(p) || !TypesAgree(input, output)) return null;

            // The input and every view of it, and the nodes that read any of them.
            var views = new HashSet<string>(StringComparer.Ordinal) { input };
            var readers = new HashSet<int>();
            var pending = new Queue<string>();
            pending.Enqueue(input);
            while (pending.TryDequeue(out var name))
            {
                if (_outputs.ContainsKey(name) || _referencedBySubgraphs.Contains(name)) return null;
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

            if (readers.Remove(writer) && !WritesInPlace(p, views, output)) return null;
            if (foldable && !AddFoldableReaders(views, writer, readers)) return null;
            return (writer, NotAncestorsOf(writer, readers));
        }

        /// <summary>
        /// Adds to <paramref name="readers"/> every node other than <paramref name="writer"/>
        /// reading a value computed from <paramref name="views"/> — the input and its views — and
        /// constants alone: what a backend can fold into a read of the input. Such a value is one
        /// written by a node whose every input is one of them or a constant (<see cref="Constants"/>),
        /// the writer's own outputs excepted. False where a subgraph refers to one of them, which
        /// the pair cannot survive as it cannot survive a subgraph referring to a view.
        /// </summary>
        private bool AddFoldableReaders(HashSet<string> views, int writer, HashSet<int> readers)
        {
            var constants = Constants();
            var derived = new HashSet<string>(views, StringComparer.Ordinal);
            var pending = new Queue<string>(views);
            bool DerivedOrConstant(string name) => name.Length == 0 || derived.Contains(name) || constants.Contains(name);
            while (pending.TryDequeue(out var name))
            {
                if (_referencedBySubgraphs.Contains(name)) return false;
                if (!_consumers.TryGetValue(name, out var consumers)) continue;
                foreach (var n in consumers)
                {
                    var node = _nodes[n];
                    if (n == writer || ReadsOnlyAShape(node)) continue;
                    readers.Add(n);
                    if (!node.Inputs.All(DerivedOrConstant)) continue;
                    if (_captured.TryGetValue(n, out var captured) && !captured.All(DerivedOrConstant)) continue;
                    foreach (var produced in node.Outputs)
                        if (produced.Length > 0 && derived.Add(produced)) pending.Enqueue(produced);
                }
            }
            return true;
        }

        /// <summary>
        /// The values a backend can know before the run begins: the initializers, the outputs of
        /// the standard <c>Shape</c> and <c>Size</c>, which it folds where shapes are known, and
        /// whatever a node computes from those alone — a <c>Constant</c> and any other node reading
        /// nothing among them. Counting a value constant that is not only widens what
        /// <see cref="AddFoldableReaders"/> counts as a reader.
        /// </summary>
        private HashSet<string> Constants()
        {
            if (_constants is not null) return _constants;
            var constants = new HashSet<string>(_initializers, StringComparer.Ordinal);
            for (var changed = true; changed;)
            {
                changed = false;
                for (int n = 0; n < _nodes.Count; n++)
                {
                    var node = _nodes[n];
                    var known = ReadsOnlyAShape(node)
                        || (node.Inputs.All(i => i.Length == 0 || constants.Contains(i))
                            && (!_captured.TryGetValue(n, out var captured) || captured.All(constants.Contains)));
                    if (!known) continue;
                    foreach (var output in node.Outputs)
                        if (output.Length > 0 && constants.Add(output)) changed = true;
                }
            }
            return _constants = constants;
        }

        /// <summary>Whether <paramref name="p"/> reads the input — through one of
        /// <paramref name="views"/> — only as it may while writing over it.</summary>
        private static bool WritesInPlace(NodeProto p, HashSet<string> views, string output)
        {
            if (IsStandard(p))
                return InPlace.Contains(p.OpType)
                       && p.Inputs.Count == 2 && p.Outputs.Count == 1
                       && views.Contains(p.Inputs[0]) && !views.Contains(p.Inputs[1]);
            if (!IsUpdate(p)) return false;
            var k = p.Outputs.IndexOf(output);
            if (k < 0 || k >= UpdatedInPlace) return false;
            for (int i = 0; i < p.Inputs.Count; i++)
                if (views.Contains(p.Inputs[i]) != (i == k)) return false;
            return true;
        }

        /// <summary>
        /// Those of <paramref name="readers"/> that are not ancestors of node
        /// <paramref name="writer"/>, in node order, walking back from it along the edges that
        /// order execution whatever a runtime folds: explicit inputs, except those of a node that
        /// reads only a shape, and the outer values a node's subgraphs read other than through a
        /// shape.
        /// </summary>
        private List<int> NotAncestorsOf(int writer, HashSet<int> readers)
        {
            if (readers.Count == 0) return [];
            var producers = OrderingProducers();
            var stamp = ++_stamp;
            var missing = readers.Count;
            var pending = new Stack<int>();
            _visited[writer] = stamp;
            pending.Push(writer);
            while (missing > 0 && pending.TryPop(out var n))
                foreach (var producer in producers[n])
                {
                    if (_visited[producer] == stamp) continue;
                    _visited[producer] = stamp;
                    if (readers.Contains(producer)) missing--;
                    pending.Push(producer);
                }
            return [.. readers.Where(r => _visited[r] != stamp).Order()];
        }

        /// <summary>
        /// Per node, the nodes writing what it reads along the edges that order execution whatever
        /// a runtime folds: its explicit inputs, and the outer values its subgraphs read other than
        /// through a shape — none for a node that reads only a shape. Built once, when first asked.
        /// </summary>
        private int[][] OrderingProducers()
        {
            if (_orderingProducers is not null) return _orderingProducers;
            var producers = new int[_nodes.Count][];
            for (int n = 0; n < _nodes.Count; n++)
            {
                var node = _nodes[n];
                if (ReadsOnlyAShape(node)) { producers[n] = []; continue; }
                var captured = _captured.TryGetValue(n, out var names) ? names : [];
                var found = new List<int>();
                foreach (var input in node.Inputs.Concat(captured))
                    if (input.Length > 0 && _producer.TryGetValue(input, out var producer)) found.Add(producer);
                producers[n] = [.. found];
            }
            return _orderingProducers = producers;
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
        /// subgraph inside it — reads from outside itself; with <paramref name="orderingOnly"/>,
        /// only those it reads other than through the standard <c>Shape</c> or <c>Size</c>.</summary>
        internal static void ReferencedFrom(GraphProto subgraph, HashSet<string> found, bool orderingOnly = false)
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
                if (!orderingOnly || !ReadsOnlyAShape(node))
                    foreach (var input in node.Inputs) inner.Add(input);
                foreach (var attribute in node.Attributes)
                {
                    if (attribute.G is { } nested) ReferencedFrom(nested, inner, orderingOnly);
                    foreach (var each in attribute.Graphs) ReferencedFrom(each, inner, orderingOnly);
                }
            }
            foreach (var name in inner)
                if (name.Length > 0 && !defined.Contains(name)) found.Add(name);
        }
    }
}

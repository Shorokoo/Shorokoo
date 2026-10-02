using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory.IR;

namespace Shorokoo.PythonTranslation;

/// <summary>
/// An output a session may write into the memory of one of its inputs (see <see cref="OutputAlias"/>),
/// as the translation arranged it: the pair, the input's position among <c>main</c>'s parameters,
/// and whether the translation writes the output there itself — which it does where the node
/// producing it is a two-input <c>Add</c>, <c>Sub</c>, <c>Mul</c> or <c>Div</c>, the one kind of
/// operator torch can be told to write into memory it is handed.
/// </summary>
internal sealed record AliasSlot(string Output, string Input, int InputIndex, bool WrittenByTheGraph);

/// <summary>
/// A value of the top-level graph a run writes into a range of memory it is handed rather than into
/// memory of its own: <see cref="Slot"/> numbers the range among the run's placements.
/// <see cref="OwnMemory"/> marks a value whose operator may hand back its input's memory, which a
/// run that does not write it into its range must copy into memory of its own all the same — the
/// placement was proved with it out of its input's memory.
/// </summary>
internal readonly record struct PlacedValue(int Slot, bool OwnMemory);

/// <summary>
/// What the translation adds to every model for the runs of the session built from it, apart from
/// what the model computes: a point to stop at before each node, and the writes of an output into a
/// consumed input's memory. Both are calls the support package's <c>runtime</c> puts in the model's
/// namespace (<c>_stop</c>, <c>_alias_write</c>), so the translation imports nothing for them.
/// </summary>
internal sealed partial class OnnxToPythonTranslator
{
    // The operators whose outputs the support package always computes into memory of their own,
    // never handing back an input or a view of one: an output of anything else may be an input's
    // memory under another name, as a torch view or as the very same tensor. Deliberately short --
    // an operator missing from it costs a runtime check, one wrongly in it a corrupted read.
    private static readonly HashSet<string> FreshOutputs = new(StringComparer.Ordinal)
    {
        "Add", "Sub", "Mul", "Div", "MatMul", "Gemm", "Shape", "Size",
    };

    // The domain and operator a native training step's gradient node is handed over as. torch
    // autograd hands back gradients of their own.
    private const string TrainingDomain = "ai.shorokoo.training";

    private static readonly Dictionary<string, string> Writers = new(StringComparer.Ordinal)
    {
        ["Add"] = "add", ["Sub"] = "sub", ["Mul"] = "mul", ["Div"] = "div",
    };

    private AliasPlan? _aliasPlan;
    private IReadOnlyDictionary<string, PlacedValue>? _placed;
    private IReadOnlyDictionary<string, int>? _over;
    private Scope? _mainScope;

    /// <summary>
    /// Translates <paramref name="model"/>, arranging for the runs of its session to write the outputs
    /// <paramref name="outputAliases"/> names into the memory of the inputs it pairs them with, where
    /// the graph proves that sound (<see cref="OutputAliasProof"/>). The graph run is the graph handed
    /// over — nothing is rewritten — so the proof over it stands.
    /// </summary>
    /// <exception cref="NotSupportedException">The model uses something the backend cannot run: the
    /// dialect's own exception.</exception>
    public static TranslatedModel Translate(ModelProto model, IReadOnlyList<OutputAlias> outputAliases, PythonDialect dialect)
        => Translate(model, outputAliases, dialect, null);

    /// <summary>
    /// <see cref="Translate(ModelProto, IReadOnlyList{OutputAlias}, PythonDialect)"/>, with each value
    /// of the top-level graph <paramref name="placed"/> names written into the range of its slot
    /// where a run hands one over (the support package's <c>_into</c>): a value whose operator is
    /// translated as one plain call of a support function. Which ranges are safe to write is the
    /// caller's to prove (<see cref="PlacementProof"/>); a run that hands over no range for a slot
    /// computes the value as the plain translation does.
    /// </summary>
    /// <para>A node of the top-level graph <paramref name="over"/> names, with the operand slot it
    /// gives, is written over that operand where a run computes without gradients (the support
    /// package's <c>_over</c>): an element-wise operator whose operand nothing reads after it, which
    /// the caller has made sure of.</para>
    internal static TranslatedModel Translate(
        ModelProto model, IReadOnlyList<OutputAlias> outputAliases, PythonDialect dialect,
        IReadOnlyDictionary<string, PlacedValue>? placed, IReadOnlyDictionary<string, int>? over = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(outputAliases);
        ArgumentNullException.ThrowIfNull(dialect);
        var graph = model.Graph ?? throw dialect.Unsupported(
            UnsupportedReason.UnsupportedModel, null, null, "The model has no graph.");
        var plan = AliasPlan.For(graph, outputAliases);
        var translator = new OnnxToPythonTranslator(dialect) { _aliasPlan = plan, _placed = placed, _over = over };
        var translated = translator.Run(model, graph);
        return translated with { Aliases = plan?.Slots(translated.InputNames) ?? [] };
    }

    /// <summary>
    /// <paramref name="expression"/>, the call of <paramref name="function"/> a node of the top-level
    /// graph evaluates to, as a write into its range where the node's one output is placed: the same
    /// call made through <c>_into</c>, which writes the result into the range a run hands over for
    /// the slot and returns it there, or computes it as the call does where none is. A call returning
    /// a tuple of the node's outputs is placed where its first output is the one the node uses.
    /// </summary>
    private string Placed(NodeProto node, Scope scope, string? function, string expression)
    {
        if (_placed is null || !ReferenceEquals(scope, _mainScope) || function is null
            || !expression.StartsWith(function + "(", StringComparison.Ordinal)
            || node.Outputs.Count(o => o.Length > 0) != 1 || node.Outputs[0].Length == 0
            || !_placed.TryGetValue(node.Outputs[0], out var placed))
            return expression;
        return $"_into({placed.Slot}, {(placed.OwnMemory ? "True" : "False")}, {function}, {expression[(function.Length + 1)..]}";
    }

    /// <summary>
    /// <paramref name="expression"/>, the call of <paramref name="function"/> a node of the top-level
    /// graph evaluates to, as a write over the operand <see cref="_over"/> names for the node's one
    /// output: the same call made through <c>_over</c>, which writes the result over that operand
    /// where torch can and the run computes no gradient, and computes it as the call does
    /// elsewhere.
    /// </summary>
    private string Over(NodeProto node, Scope scope, string? function, string expression)
    {
        if (_over is null || !ReferenceEquals(scope, _mainScope) || function is null
            || !expression.StartsWith(function + "(", StringComparison.Ordinal)
            || node.Outputs.Count(o => o.Length > 0) != 1 || node.Outputs[0].Length == 0
            || _placed?.ContainsKey(node.Outputs[0]) == true
            || !_over.TryGetValue(node.Outputs[0], out var slot) || slot >= node.Inputs.Count || node.Inputs[slot].Length == 0)
            return expression;
        return $"_over({scope.Lookup(node.Inputs[slot], node)}, {function}, {expression[(function.Length + 1)..]}";
    }

    /// <summary>
    /// Writes what goes before a node's statement: the point a cancelled run stops at. One per node,
    /// in every graph and function, so that a run stops between nodes wherever it is — a loop body
    /// included, once per iteration.
    /// </summary>
    private void BeforeNode(NodeProto node)
    {
        if (Dialect.StopPoints) Line("_stop()");
    }

    /// <summary>
    /// Where <paramref name="node"/> writes an output this session may put into a consumed input's
    /// memory, writes the call that does so and returns true; the caller then writes the node's own
    /// statement guarded, to run only where that call declined — on a run that did not consume the
    /// input, say. Otherwise writes nothing and returns false.
    ///
    /// <para>The call is handed the values this graph reads after the node that could be the input's
    /// memory under another name, so it can refuse where one is: the proof behind the pair knows
    /// ONNX Runtime's views, and torch makes more of them (<c>Transpose</c>, <c>Slice</c>, an
    /// <c>Expand</c> or a <c>Cast</c> that changes nothing).</para>
    /// </summary>
    private bool AliasedWrite(NodeProto node, Scope scope, IReadOnlyList<string> targets)
    {
        if (_aliasPlan?.WriteBy(node) is not { } write || targets.Count != 1) return false;
        string[] live;
        string first, second;
        try
        {
            first = scope.Lookup(node.Inputs[0], node);
            second = scope.Lookup(node.Inputs[1], node);
            live = [.. _aliasPlan.LiveAfter(write, name => IsBound(scope, name)).Select(name => scope.Lookup(name, node))];
        }
        catch (NotSupportedException)
        {
            // A value this could not name from here: then it cannot check it either, so it does not write.
            return false;
        }
        var tuple = live.Length == 0 ? "()" : $"({string.Join(", ", live)},)";
        Line($"{targets[0]} = _alias_write({write.Slot}, \"{Writers[node.OpType]}\", {first}, {second}, {tuple})");
        return true;
    }

    /// <summary>Whether <paramref name="name"/> has been given a value by the statements written so
    /// far, where <paramref name="scope"/> can see it.</summary>
    private static bool IsBound(Scope scope, string name)
    {
        for (var each = scope; each is not null; each = each.Parent)
            if (each.IsBoundHere(name)) return true;
        return false;
    }

    /// <summary>The outputs of the top-level graph a session may write into consumed inputs, and what
    /// the translation needs to know to do it: which node writes each, and what could still read the
    /// input's memory once it has.</summary>
    private sealed class AliasPlan
    {
        private readonly GraphProto _graph;
        private readonly List<(OutputAlias Alias, NodeProto? Writer)> _pairs;
        private readonly Dictionary<NodeProto, PlannedWrite> _writes = new(ReferenceEqualityComparer.Instance);
        private readonly HashSet<string> _initializers = new(StringComparer.Ordinal);

        // What each top-level node reads, in graph order: its inputs, and every name a subgraph it
        // holds reads from outside itself.
        private readonly List<(NodeProto Node, HashSet<string> Reads)> _reads = [];

        private AliasPlan(GraphProto graph, List<(OutputAlias Alias, NodeProto? Writer)> pairs)
        {
            _graph = graph;
            _pairs = pairs;
            foreach (var node in graph.Nodes)
            {
                var reads = new HashSet<string>(node.Inputs.Where(input => input.Length > 0), StringComparer.Ordinal);
                foreach (var attribute in node.Attributes)
                {
                    if (attribute.G is { } body) ReferencedFrom(body, reads);
                    foreach (var each in attribute.Graphs) ReferencedFrom(each, reads);
                }
                _reads.Add((node, reads));
            }
            foreach (var initializer in graph.Initializers) _initializers.Add(initializer.Name);
            for (int slot = 0; slot < pairs.Count; slot++)
                if (pairs[slot].Writer is { } writer && IsWriter(writer))
                    _writes[writer] = new PlannedWrite(slot, pairs[slot].Alias.Input, writer);
        }

        /// <summary>The plan for the pairs of <paramref name="aliases"/> the graph proves, or null
        /// where it proves none.</summary>
        public static AliasPlan? For(GraphProto graph, IReadOnlyList<OutputAlias> aliases)
        {
            if (aliases.Count == 0) return null;
            var proved = OutputAliasProof.Prove(graph, aliases);
            if (proved.Count == 0) return null;
            var producers = new Dictionary<string, NodeProto>(StringComparer.Ordinal);
            foreach (var node in graph.Nodes)
                foreach (var output in node.Outputs)
                    if (output.Length > 0) producers[output] = node;
            var strings = graph.Outputs
                .Where(o => o.Type?.TensorType?.ElemType == (int)TensorProto.DataType.String)
                .Select(o => o.Name).ToHashSet(StringComparer.Ordinal);
            List<(OutputAlias Alias, NodeProto? Writer)> pairs =
            [
                .. proved.Where(a => !strings.Contains(a.Output))
                    .Select(a => (a, producers.GetValueOrDefault(a.Output))),
            ];
            return pairs.Count == 0 ? null : new AliasPlan(graph, pairs);
        }

        /// <summary>Whether <paramref name="node"/> is one torch can be told to write into memory
        /// it is handed: a standard two-input <c>Add</c>, <c>Sub</c>, <c>Mul</c> or <c>Div</c>.</summary>
        private static bool IsWriter(NodeProto node)
            => node.Domain is "" or "ai.onnx" && Writers.ContainsKey(node.OpType)
               && node.Inputs.Count == 2 && node.Outputs.Count == 1
               && node.Inputs[0].Length > 0 && node.Inputs[1].Length > 0;

        public PlannedWrite? WriteBy(NodeProto node) => _writes.GetValueOrDefault(node);

        /// <summary>
        /// The values already made when <paramref name="write"/>'s node runs that something still to
        /// come reads — a node not yet written, a subgraph one of those holds, or the graph's own
        /// outputs — and that could be the input's memory: the input itself, and anything made from
        /// it by an operator not known to make memory of its own. What has been written is read off
        /// <paramref name="isBound"/>, the names the statements so far have given values, so that the
        /// answer holds whatever order the nodes are written in.
        /// </summary>
        public IEnumerable<string> LiveAfter(PlannedWrite write, Func<string, bool> isBound)
        {
            var mayShare = MayShare(write.Input);
            var read = new HashSet<string>(StringComparer.Ordinal);
            foreach (var (node, reads) in _reads)
                if (!ReferenceEquals(node, write.Writer) && !node.Outputs.Where(o => o.Length > 0).All(isBound))
                    read.UnionWith(reads);
            foreach (var output in _graph.Outputs) read.Add(output.Name);
            return read.Where(name => name.Length > 0 && mayShare.Contains(name) && !_initializers.Contains(name)
                                      && !write.Writer.Outputs.Contains(name) && isBound(name))
                .Order(StringComparer.Ordinal);
        }

        /// <summary><paramref name="input"/>, and every value made from it — or from one of those —
        /// by an operator not known to make memory of its own.</summary>
        private HashSet<string> MayShare(string input)
        {
            // One pass in graph order is the whole closure: a graph's nodes come after what they read.
            var found = new HashSet<string>(StringComparer.Ordinal) { input };
            foreach (var (node, reads) in _reads)
            {
                if (IsFresh(node) || !reads.Overlaps(found)) continue;
                foreach (var output in node.Outputs)
                    if (output.Length > 0) found.Add(output);
            }
            return found;
        }

        private static bool IsFresh(NodeProto node)
            => node.Domain is "" or "ai.onnx" ? FreshOutputs.Contains(node.OpType) : node.Domain == TrainingDomain;

        /// <summary>Adds to <paramref name="found"/> every name <paramref name="subgraph"/>, or a
        /// subgraph inside it, reads from outside itself.</summary>
        internal static void ReferencedFrom(GraphProto subgraph, HashSet<string> found)
        {
            var defined = new HashSet<string>(StringComparer.Ordinal);
            foreach (var input in subgraph.Inputs) defined.Add(input.Name);
            foreach (var initializer in subgraph.Initializers) defined.Add(initializer.Name);
            foreach (var node in subgraph.Nodes)
                foreach (var output in node.Outputs) defined.Add(output);
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

        /// <summary>The slots of the pairs whose input is one of <c>main</c>'s parameters.</summary>
        public IReadOnlyList<AliasSlot> Slots(string[] inputNames)
        {
            var slots = new List<AliasSlot>(_pairs.Count);
            for (int slot = 0; slot < _pairs.Count; slot++)
            {
                var (alias, writer) = _pairs[slot];
                slots.Add(new AliasSlot(alias.Output, alias.Input, Array.IndexOf(inputNames, alias.Input),
                    writer is not null && _writes.ContainsKey(writer)));
            }
            return slots;
        }
    }

    /// <summary>A write the translation makes: which slot, into which input, by which node.</summary>
    private sealed record PlannedWrite(int Slot, string Input, NodeProto Writer);
}

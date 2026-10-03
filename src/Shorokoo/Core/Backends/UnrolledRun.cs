using Shorokoo.Core.Factory.IR;

namespace Shorokoo.Core.Backends;

/// <summary>
/// A run of a graph as the nodes a backend runs, one after the other, with the shape of every
/// value they make: the top-level graph's nodes in the backend's order, each <c>Loop</c> unrolled
/// where it stands — its body once per iteration, in the backend's order, its values renamed per
/// iteration (<see cref="Run.Iterated"/>) — and each sequence resolved into the tensors it holds,
/// as a backend keeps a sequence: a Python list of tensors in a translation, a list of tensors in
/// ONNX Runtime.
///
/// <para>A loop unrolls where its trip count is known from the shapes the run is fed and its
/// condition from what its body computes: a run takes <c>trip_count</c> iterations, or fewer where
/// the condition turns false. Each iteration's carried values are the last one's; a scan output is
/// the iteration's values stacked, a <c>Concat</c> of each unsqueezed, read once the loop ends. A
/// sequence holds no memory of its own: inserting into one, erasing from one or reading one element
/// of it makes no tensor, and an element read from one is that tensor (an <c>Identity</c>). What a
/// sequence holds lives as long as the sequence does, so every node reading a sequence reads each
/// of its elements too — a <see cref="HoldOpType"/> node of the <see cref="HoldDomain"/> domain,
/// with no outputs, where the node reading it stands — and the outer values a loop's body reads are
/// read once more where the loop ends, as the frame around the loop holds them until it is
/// done.</para>
///
/// <para>Null where a loop's trip count or condition cannot be told, or a sequence reaches a node
/// other than the sequence operators and <c>Identity</c>.</para>
/// </summary>
internal static class UnrolledRun
{
    /// <summary>The domain of the nodes that read what a list holds.</summary>
    internal const string HoldDomain = "shorokoo.run";

    /// <summary>The operator of the nodes that read what a list holds.</summary>
    internal const string HoldOpType = "Hold";

    /// <summary>The nodes a run runs, in order, the shapes of the values they make, and those of the
    /// values made for a loop — in its body, or of its outputs where it ends.</summary>
    internal sealed record Run(IReadOnlyList<NodeProto> Order, Dictionary<string, PlacementShapes.Value> Shapes,
        IReadOnlySet<string> Iterated);

    /// <summary>The run of <paramref name="graph"/> fed <paramref name="inputs"/>, a graph's nodes
    /// run in the order <paramref name="runOrder"/> puts them in, or null where it cannot be
    /// told.</summary>
    internal static Run? Of(GraphProto graph, IReadOnlyDictionary<string, (long[] Shape, int ElementType)> inputs,
        Func<IReadOnlyList<NodeProto>, IReadOnlyList<NodeProto>> runOrder)
    {
        var order = runOrder(graph.Nodes);
        if (!graph.Nodes.Any(Unrolls))
            return new Run(order, PlacementShapes.Evaluate(graph, inputs), new HashSet<string>(StringComparer.Ordinal));
        var (values, symbols) = PlacementShapes.Start(graph, inputs);
        var stated = new Dictionary<string, ValueInfoProto>(StringComparer.Ordinal);
        foreach (var info in graph.ValueInfoes.Concat(graph.Outputs)) stated.TryAdd(info.Name, info);
        var unrolling = new Unrolling(values, symbols, stated, runOrder);
        if (!unrolling.Expand(order, new Dictionary<string, string>(StringComparer.Ordinal), renameOutputs: false)) return null;
        foreach (var output in graph.Outputs)
            if (unrolling.Lists.TryGetValue(output.Name, out var held)) unrolling.Hold(held);
        return new Run(unrolling.Order, values, unrolling.Iterated);
    }

    /// <summary>Whether <paramref name="node"/> is one a run of the graph is unrolled for.</summary>
    private static bool Unrolls(NodeProto node)
        => OutputAliasProof.IsStandard(node) && (node.OpType == "Loop" || node.OpType.StartsWith("Sequence", StringComparison.Ordinal)
            || node.OpType is "ConcatFromSequence" or "SplitToSequence");

    private sealed class Unrolling(Dictionary<string, PlacementShapes.Value> values, Dictionary<string, long> symbols,
        IReadOnlyDictionary<string, ValueInfoProto> stated, Func<IReadOnlyList<NodeProto>, IReadOnlyList<NodeProto>> runOrder)
    {
        private int _renamed;
        private int _looping;

        internal List<NodeProto> Order { get; } = [];

        /// <summary>The values made for a loop: in its body, or of its outputs where it ends.</summary>
        internal HashSet<string> Iterated { get; } = new(StringComparer.Ordinal);

        /// <summary>Every sequence made so far, by name: the tensors it holds, in order.</summary>
        internal Dictionary<string, List<string>> Lists { get; } = new(StringComparer.Ordinal);

        internal void Hold(IEnumerable<string> held)
        {
            var node = new NodeProto { OpType = HoldOpType, Domain = HoldDomain };
            node.Inputs.AddRange(held.Where(h => h.Length > 0));
            if (node.Inputs.Count > 0) Order.Add(node);
        }

        private void Emit(NodeProto node)
        {
            PlacementShapes.Step(node, values, symbols, stated);
            Order.Add(node);
            if (_looping > 0) Iterated.UnionWith(node.Outputs.Where(o => o.Length > 0));
        }

        private string Fresh(string name) => $"{name}§{_renamed++}";

        private static NodeProto Node(string op, IEnumerable<string> inputs, string output)
        {
            var node = new NodeProto { OpType = op };
            node.Inputs.AddRange(inputs);
            node.Outputs.Add(output);
            return node;
        }

        private static NodeProto Constant(string output, int elementType, long value)
        {
            var tensor = new TensorProto { data_type = elementType, Dims = [] };
            if (elementType == (int)TensorProto.DataType.Int64) tensor.Int64Datas = [value];
            else tensor.Int32Datas = [(int)value];
            var node = new NodeProto { OpType = "Constant" };
            node.Outputs.Add(output);
            node.Attributes.Add(new AttributeProto { Name = "value", Type = AttributeProto.AttributeType.Tensor, T = tensor });
            return node;
        }

        private long? Scalar(string name) => name.Length > 0 && values.TryGetValue(name, out var v) && v.Ints is [var only] ? only : null;

        /// <summary>
        /// Appends <paramref name="nodes"/>, run in that order, to the run: each reading through
        /// <paramref name="rename"/> what a name stands for here, and — where
        /// <paramref name="renameOutputs"/> — making its outputs under fresh names it adds there.
        /// False where a node cannot be unrolled.
        /// </summary>
        internal bool Expand(IReadOnlyList<NodeProto> nodes, Dictionary<string, string> rename, bool renameOutputs)
        {
            foreach (var original in nodes)
            {
                string In(string name) => name.Length > 0 && rename.TryGetValue(name, out var to) ? to : name;
                List<string> inputs = [.. original.Inputs.Select(In)];
                List<string> outputs = [.. original.Outputs.Select(o => o.Length == 0 ? "" : renameOutputs ? rename[o] = Fresh(o) : o)];
                var op = OutputAliasProof.IsStandard(original) ? original.OpType : "";
                List<string>? ListOf(int slot) => slot < inputs.Count && Lists.TryGetValue(inputs[slot], out var list) ? list : null;
                switch (op)
                {
                    case "SequenceEmpty":
                        Lists[outputs[0]] = [];
                        continue;
                    case "SequenceConstruct":
                        Lists[outputs[0]] = [.. inputs];
                        continue;
                    case "SequenceInsert" or "SequenceErase":
                    {
                        if (ListOf(0) is not { } list) return false;
                        var made = new List<string>(list);
                        var position = op == "SequenceInsert" ? (inputs.Count > 2 && inputs[2].Length > 0 ? Scalar(inputs[2]) : made.Count)
                            : (inputs.Count > 1 && inputs[1].Length > 0 ? Scalar(inputs[1]) : made.Count - 1);
                        if (position is not { } at) return false;
                        if (at < 0) at += made.Count;
                        if (op == "SequenceInsert")
                        {
                            if (at < 0 || at > made.Count) return false;
                            made.Insert((int)at, inputs[1]);
                        }
                        else
                        {
                            if (at < 0 || at >= made.Count) return false;
                            made.RemoveAt((int)at);
                        }
                        Hold([.. list, .. op == "SequenceInsert" ? [inputs[1]] : Array.Empty<string>()]);
                        Lists[outputs[0]] = made;
                        continue;
                    }
                    case "SequenceAt":
                    {
                        if (ListOf(0) is not { } list || Scalar(inputs[1]) is not { } at) return false;
                        if (at < 0) at += list.Count;
                        if (at < 0 || at >= list.Count) return false;
                        Hold(list);
                        Emit(Node("Identity", [list[(int)at]], outputs[0]));
                        continue;
                    }
                    case "SequenceLength":
                    {
                        if (ListOf(0) is not { } list) return false;
                        Hold(list);
                        Emit(Constant(outputs[0], (int)TensorProto.DataType.Int64, list.Count));
                        continue;
                    }
                    case "ConcatFromSequence":
                    {
                        if (ListOf(0) is not { } list || list.Count == 0) return false;
                        var axis = original.Attributes.FirstOrDefault(a => a.Name == "axis")?.I ?? 0;
                        var stack = (original.Attributes.FirstOrDefault(a => a.Name == "new_axis")?.I ?? 0) != 0;
                        var parts = list;
                        if (stack)
                        {
                            var axes = Fresh("axes");
                            Emit(ConstantInts(axes, axis));
                            parts = [];
                            foreach (var element in list)
                            {
                                var unsqueezed = Fresh(element);
                                Emit(Node("Unsqueeze", [element, axes], unsqueezed));
                                parts.Add(unsqueezed);
                            }
                        }
                        var concat = Node("Concat", parts, outputs[0]);
                        concat.Attributes.Add(new AttributeProto { Name = "axis", Type = AttributeProto.AttributeType.Int, I = axis });
                        Emit(concat);
                        continue;
                    }
                    case "SplitToSequence":
                        return false;
                    case "Identity" when ListOf(0) is { } list:
                        Lists[outputs[0]] = list;
                        continue;
                    case "Loop":
                        if (!Loop(original, inputs, outputs)) return false;
                        continue;
                }
                if (inputs.Any(Lists.ContainsKey)) return false;
                var clone = new NodeProto { OpType = original.OpType, Domain = original.Domain };
                clone.Inputs.AddRange(inputs);
                clone.Outputs.AddRange(outputs);
                clone.Attributes.AddRange(original.Attributes);
                Emit(clone);
            }
            return true;
        }

        private static NodeProto ConstantInts(string output, params long[] ints)
        {
            var node = new NodeProto { OpType = "Constant" };
            node.Outputs.Add(output);
            node.Attributes.Add(new AttributeProto
            {
                Name = "value", Type = AttributeProto.AttributeType.Tensor,
                T = new TensorProto { data_type = (int)TensorProto.DataType.Int64, Dims = [ints.Length], Int64Datas = ints },
            });
            return node;
        }

        /// <summary>Unrolls a <c>Loop</c> reading <paramref name="inputs"/> and making
        /// <paramref name="outputs"/>; false where its trip count or condition cannot be told.</summary>
        private bool Loop(NodeProto loop, List<string> inputs, List<string> outputs)
        {
            _looping++;
            try
            {
                return Unroll(loop, inputs, outputs);
            }
            finally
            {
                _looping--;
            }
        }

        private bool Unroll(NodeProto loop, List<string> inputs, List<string> outputs)
        {
            if (loop.Attributes.FirstOrDefault(a => a.Name == "body")?.G is not { } body) return false;
            if (inputs.Count == 0 || inputs[0].Length == 0 || Scalar(inputs[0]) is not { } trips) return false;
            var carried = inputs.Count - 2;
            if (carried < 0 || body.Inputs.Count != carried + 2 || body.Outputs.Count < carried + 1) return false;
            var keepGoing = inputs.Count > 1 && inputs[1].Length > 0 ? Scalar(inputs[1]) : 1;
            if (keepGoing is null) return false;
            var defined = body.Inputs.Select(i => i.Name).Concat(body.Initializers.Select(i => i.Name))
                .Concat(body.Nodes.SelectMany(n => n.Outputs)).ToHashSet(StringComparer.Ordinal);
            var captured = body.Nodes.SelectMany(n => n.Inputs).Where(i => i.Length > 0 && !defined.Contains(i)).Distinct().ToList();
            var bodyOrder = runOrder(body.Nodes);
            var current = inputs.Skip(2).ToList();
            var scans = Enumerable.Range(0, body.Outputs.Count - 1 - carried).Select(_ => new List<string>()).ToList();
            var condition = inputs.Count > 1 && inputs[1].Length > 0 ? inputs[1] : "";
            for (long iteration = 0; iteration < trips && keepGoing != 0; iteration++)
            {
                var rename = new Dictionary<string, string>(StringComparer.Ordinal);
                var counter = Fresh(body.Inputs[0].Name);
                Emit(Constant(counter, (int)TensorProto.DataType.Int64, iteration));
                rename[body.Inputs[0].Name] = counter;
                if (condition.Length == 0)
                {
                    condition = Fresh(body.Inputs[1].Name);
                    Emit(Constant(condition, (int)TensorProto.DataType.Bool, 1));
                }
                rename[body.Inputs[1].Name] = condition;
                for (int i = 0; i < carried; i++) rename[body.Inputs[2 + i].Name] = current[i];
                foreach (var initializer in body.Initializers)
                    if (PlacementShapes.FromTensor(initializer) is { } value)
                        values[rename[initializer.Name] = Fresh(initializer.Name)] = value;
                if (!Expand(bodyOrder, rename, renameOutputs: true)) return false;
                string Out(int slot) => rename.TryGetValue(body.Outputs[slot].Name, out var to) ? to : body.Outputs[slot].Name;
                condition = Out(0);
                keepGoing = Scalar(condition);
                if (keepGoing is null) return false;
                for (int i = 0; i < carried; i++) current[i] = Out(1 + i);
                for (int s = 0; s < scans.Count; s++) scans[s].Add(Out(1 + carried + s));
            }
            Hold([.. captured.SelectMany(c => Lists.TryGetValue(c, out var list) ? list : [c])]);
            for (int i = 0; i < carried && i < outputs.Count; i++)
            {
                if (outputs[i].Length == 0) continue;
                if (Lists.TryGetValue(current[i], out var list)) Lists[outputs[i]] = list;
                else Emit(Node("Identity", [current[i]], outputs[i]));
            }
            for (int s = 0; s < scans.Count && carried + s < outputs.Count; s++)
            {
                if (outputs[carried + s].Length == 0) continue;
                if (scans[s].Count == 0) return false;
                var axes = Fresh("axes");
                Emit(ConstantInts(axes, 0));
                List<string> parts = [];
                foreach (var element in scans[s])
                {
                    var unsqueezed = Fresh(element);
                    Emit(Node("Unsqueeze", [element, axes], unsqueezed));
                    parts.Add(unsqueezed);
                }
                var stacked = Node("Concat", parts, outputs[carried + s]);
                stacked.Attributes.Add(new AttributeProto { Name = "axis", Type = AttributeProto.AttributeType.Int, I = 0 });
                Emit(stacked);
            }
            return true;
        }
    }
}

using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory.IR;

namespace Shorokoo.Tests.Benchmarks;

/// <summary>
/// What a lowered training step holds at its peak, modelled from its graph: each value takes its
/// bytes when the node writing it runs and gives them back once the last node reading it — or
/// anything a backend hands back over it — has run, in the order a backend runs the nodes. The
/// step's inputs (the state and the batch) are held before it begins and are not counted; the
/// state outputs are written over the state inputs (output aliasing) and take nothing; a
/// <c>Constant</c> takes nothing (it is folded into the model).
///
/// <para>Each value is put to the part of the step that writes it: the <b>forward</b> pass (the
/// loss and everything it is computed from), the <b>parameter gradients</b> (a value of the
/// backward pass the optimizer reads, written from values of other shapes — a weight's product of
/// activation and output gradient, a bias's sum), the rest of the <b>backward</b> pass (everything
/// a parameter gradient is computed from that the forward pass is not: activation gradients and
/// their temporaries), and the <b>update</b> (the optimizer's arithmetic: everything else, which
/// reads the optimizer's state or a parameter gradient and writes the new state).</para>
/// </summary>
internal sealed class StepAnatomy
{
    internal enum Part { Forward, Backward, ParameterGradient, Update, Batch }

    internal sealed record Snapshot(int Position, long Bytes, IReadOnlyDictionary<Part, long> ByPart, IReadOnlyList<(string Value, string Op, Part Part, long Bytes, string Until)> Largest);

    private readonly GraphProto _graph;
    private readonly Dictionary<string, PlacementShapes.Value> _shapes;
    private readonly Dictionary<string, int> _producer = new(StringComparer.Ordinal);
    private readonly Dictionary<NodeProto, int> _index = new(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<string, List<int>> _consumers = new(StringComparer.Ordinal);
    private readonly HashSet<string> _inputs;
    private readonly Part[] _parts;
    private readonly int _state;

    internal long StateBytes { get; }
    internal long ParameterBytes { get; }
    internal long BatchBytes { get; }
    internal long UnknownValues { get; }

    /// <param name="graph">The step as the backend is handed it.</param>
    /// <param name="parameters">How many of its leading inputs are parameters.</param>
    /// <param name="state">How many of its leading inputs, and as many leading outputs, are the
    /// state the step updates (parameters, model state, optimizer state, in that order); the inputs
    /// after them are the batch.</param>
    /// <param name="optimizerState">How many of the state inputs, last among them, are the
    /// optimizer's.</param>
    internal StepAnatomy(GraphProto graph, int parameters, int state, int optimizerState)
    {
        _graph = graph;
        _state = state;
        var given = graph.Inputs.ToDictionary(i => i.Name,
            i => (i.Type.TensorType.Shape.Dims.Select(d => d.DimValue).ToArray(), i.Type.TensorType.ElemType), StringComparer.Ordinal);
        _shapes = PlacementShapes.Evaluate(graph, given);
        _inputs = graph.Inputs.Select(i => i.Name).ToHashSet(StringComparer.Ordinal);
        for (int n = 0; n < graph.Nodes.Count; n++)
        {
            _index[graph.Nodes[n]] = n;
            foreach (var output in graph.Nodes[n].Outputs.Where(o => o.Length > 0)) _producer[output] = n;
            foreach (var input in graph.Nodes[n].Inputs.Where(i => i.Length > 0))
            {
                if (!_consumers.TryGetValue(input, out var list)) _consumers[input] = list = [];
                list.Add(n);
            }
        }
        long BytesOf(string name) => _shapes.TryGetValue(name, out var v) && v.Bytes > 0 ? v.Bytes : 0;
        StateBytes = graph.Inputs.Take(state).Sum(i => BytesOf(i.Name));
        ParameterBytes = graph.Inputs.Take(parameters).Sum(i => BytesOf(i.Name));
        BatchBytes = graph.Inputs.Skip(state).Sum(i => BytesOf(i.Name));
        UnknownValues = graph.Nodes.SelectMany(n => n.Outputs).Count(o => o.Length > 0 && !_shapes.ContainsKey(o));

        var count = graph.Nodes.Count;
        var forward = Ancestors(graph.Outputs.Skip(state).Select(o => o.Name));
        var optimizerInputs = graph.Inputs.Skip(state - optimizerState).Take(optimizerState).Select(i => i.Name);
        var fromOptimizer = Descendants(optimizerInputs);
        var fromForward = Descendants(graph.Inputs.Skip(state).Select(i => i.Name).Concat(forward.SelectMany(n => graph.Nodes[n].Outputs)));
        var optimizerOutputs = Ancestors(graph.Outputs.Skip(state - optimizerState).Take(optimizerState).Select(o => o.Name));
        var parameterShapes = graph.Inputs.Take(parameters).Select(i => string.Join("x", _shapes[i.Name].Shape)).ToHashSet();
        var gradients = new HashSet<int>();
        for (int n = 0; n < count; n++)
        {
            if (forward.Contains(n) || fromOptimizer.Contains(n) || !fromForward.Contains(n) || !optimizerOutputs.Contains(n)) continue;
            var node = graph.Nodes[n];
            if (node.OpType == "Constant" || node.Outputs.Count(o => o.Length > 0) != 1) continue;
            var output = node.Outputs.First(o => o.Length > 0);
            if (!_shapes.TryGetValue(output, out var shape) || !parameterShapes.Contains(string.Join("x", shape.Shape))) continue;
            if (node.Inputs.Any(i => i.Length > 0 && _shapes.TryGetValue(i, out var s) && !s.Shape.SequenceEqual(shape.Shape) && s.Elements > 1
                                     && (_inputs.Contains(i) ? !optimizerInputs.Contains(i) : _producer.TryGetValue(i, out var pi) && fromForward.Contains(pi))))
                gradients.Add(n);
        }
        var backward = Ancestors(gradients.SelectMany(n => graph.Nodes[n].Outputs));
        _parts = new Part[count];
        for (int n = 0; n < count; n++)
            _parts[n] = forward.Contains(n) ? Part.Forward
                : gradients.Contains(n) ? Part.ParameterGradient
                : backward.Contains(n) && !fromOptimizer.Contains(n) ? Part.Backward
                : Part.Update;
    }

    private HashSet<int> Ancestors(IEnumerable<string> values)
    {
        var seen = new HashSet<int>();
        var pending = new Stack<string>(values);
        while (pending.TryPop(out var value))
            if (_producer.TryGetValue(value, out var n) && seen.Add(n))
                foreach (var input in _graph.Nodes[n].Inputs.Where(i => i.Length > 0)) pending.Push(input);
        return seen;
    }

    private HashSet<int> Descendants(IEnumerable<string> values)
    {
        var seen = new HashSet<int>();
        var pending = new Stack<string>(values);
        while (pending.TryPop(out var value))
            if (_consumers.TryGetValue(value, out var readers))
                foreach (var n in readers)
                    if (seen.Add(n))
                        foreach (var output in _graph.Nodes[n].Outputs.Where(o => o.Length > 0)) pending.Push(output);
        return seen;
    }

    /// <summary>The order ONNX Runtime's sequential executor runs the nodes in.</summary>
    internal int[] OrtOrder() => [.. Shorokoo.PythonTranslation.OnnxToPythonTranslator.InOnnxRuntimeOrder(_graph.Nodes).Select(n => _index[n])];

    /// <summary>
    /// What the step holds at its peak, run in <paramref name="order"/>, under a backend that hands
    /// an output back over its input where <paramref name="shares"/> says so and writes an output
    /// over an input it reads where <paramref name="inPlace"/> says so — the input dying there, of
    /// the output's size.
    /// </summary>
    internal Snapshot Peak(int[] order, Func<NodeProto, int, int, bool> shares, Func<NodeProto, int, bool>? inPlace = null, bool batchFreed = false,
        bool shapeReadsHold = true)
    {
        var position = new int[_graph.Nodes.Count];
        for (int k = 0; k < order.Length; k++) position[order[k]] = k;
        var root = new Dictionary<string, string>(StringComparer.Ordinal);
        string RootOf(string v) => root.TryGetValue(v, out var r) && r != v ? root[v] = RootOf(r) : v;
        var last = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var output in _graph.Outputs) last[output.Name] = int.MaxValue;
        var bytes = new Dictionary<string, long>(StringComparer.Ordinal);
        var parts = new Dictionary<string, Part>(StringComparer.Ordinal);
        var stateOutputs = _graph.Outputs.Take(_state).Select(o => o.Name).ToHashSet(StringComparer.Ordinal);

        // Views first: a value handed back over an input is that input's memory.
        foreach (var n in order)
        {
            var node = _graph.Nodes[n];
            for (int o = 0; o < node.Outputs.Count; o++)
            {
                var output = node.Outputs[o];
                if (output.Length == 0) continue;
                var over = Enumerable.Range(0, node.Inputs.Count).FirstOrDefault(i => node.Inputs[i].Length > 0 && shares(node, i, o), -1);
                if (over >= 0) root[output] = RootOf(node.Inputs[over]);
            }
        }
        foreach (var n in order)
            foreach (var input in _graph.Nodes[n].Inputs.Where(i => i.Length > 0 && (shapeReadsHold || !OutputAliasProof.ReadsOnlyAShape(_graph.Nodes[n]))))
            {
                var r = RootOf(input);
                last[r] = Math.Max(last.GetValueOrDefault(r, -1), position[n]);
            }

        long live = 0, peak = -1;
        int peakAt = 0;
        var alive = new Dictionary<string, long>(StringComparer.Ordinal);
        if (batchFreed)
            foreach (var input in _graph.Inputs.Skip(_state))
                if (_shapes.TryGetValue(input.Name, out var v) && v.Bytes > 0 && last.ContainsKey(input.Name))
                {
                    alive[input.Name] = v.Bytes;
                    parts[input.Name] = Part.Batch;
                    live += v.Bytes;
                }
        var held = live;
        Dictionary<string, long> atPeak = [];
        for (int k = 0; k < order.Length; k++)
        {
            var node = _graph.Nodes[order[k]];
            var reused = false;
            foreach (var output in node.Outputs.Where(o => o.Length > 0))
            {
                if (RootOf(output) != output || node.OpType == "Constant" || stateOutputs.Contains(output)) continue;
                var size = _shapes.TryGetValue(output, out var v) && v.Bytes > 0 ? v.Bytes : 0;
                if (size == 0) continue;
                if (!reused && inPlace is not null)
                {
                    var operand = Enumerable.Range(0, node.Inputs.Count).FirstOrDefault(i => node.Inputs[i].Length > 0 && inPlace(node, i)
                        && alive.ContainsKey(RootOf(node.Inputs[i])) && alive[RootOf(node.Inputs[i])] == size && last[RootOf(node.Inputs[i])] == k, -1);
                    if (operand >= 0)
                    {
                        var from = RootOf(node.Inputs[operand]);
                        alive.Remove(from);
                        root[from] = output;
                        alive[output] = size;
                        parts[output] = _parts[order[k]];
                        reused = true;
                        continue;
                    }
                }
                alive[output] = size;
                parts[output] = _parts[order[k]];
                live += size;
            }
            if (live > peak)
            {
                peak = live;
                peakAt = k;
                atPeak = new Dictionary<string, long>(alive, StringComparer.Ordinal);
            }
            foreach (var input in node.Inputs.Where(i => i.Length > 0).Concat(node.Outputs.Where(o => o.Length > 0)).Distinct())
            {
                var r = RootOf(input);
                if (alive.TryGetValue(r, out var size) && last.GetValueOrDefault(r, k) <= k)
                {
                    alive.Remove(r);
                    live -= size;
                }
            }
        }
        var byPart = Enum.GetValues<Part>().ToDictionary(p => p, p => atPeak.Where(a => parts[a.Key] == p).Sum(a => a.Value));
        var largest = atPeak.OrderByDescending(a => a.Value).Take(8)
            .Select(a => (a.Key, _producer.TryGetValue(a.Key, out var p) ? _graph.Nodes[p].OpType : "input", parts[a.Key], a.Value,
                last.TryGetValue(a.Key, out var until) && until != int.MaxValue && until < order.Length ? $"{until - peakAt} later, {At(order, until)}" : "after the run"))
            .ToList();
        return new Snapshot(peakAt, peak - held, byPart, largest);
    }

    /// <summary>The operators whose outputs are of unknown shape though every input they read is known, with
    /// what they read: where shape evaluation stops.</summary>
    internal string UnknownRoots => string.Join(" | ", _graph.Nodes
        .Where(n => n.Outputs.Any(o => o.Length > 0 && !_shapes.ContainsKey(o)) && n.Inputs.All(i => i.Length == 0 || _shapes.ContainsKey(i)))
        .Select(n => $"{n.OpType}({string.Join("; ", n.Inputs.Select(i => i.Length == 0 ? "-" : $"{string.Join("x", _shapes[i].Shape)}:{_shapes[i].ElementType}{(_shapes[i].Ints is { } v ? "=" + string.Join(",", v.Take(6)) : "")}"))})")
        .GroupBy(x => x).Select(g => $"{g.Key} x{g.Count()}").Take(15))
        + " -- " + string.Join(" | ", _graph.Nodes
            .Where(n => n.Outputs.Any(o => o.Length > 0 && !_shapes.ContainsKey(o)) && n.Inputs.All(i => i.Length == 0 || _shapes.ContainsKey(i)))
            .Take(2).SelectMany(n => n.Inputs.Where(i => i.Length > 0 && _shapes[i].Ints is null && _shapes[i].Elements <= 8).Select(i => Explain(i, 4))));

    /// <summary>How <paramref name="value"/> is computed, with what is known of each value, a few
    /// levels down.</summary>
    private string Explain(string value, int depth)
    {
        var known = _shapes.TryGetValue(value, out var v) ? $"{string.Join("x", v.Shape)}:{v.ElementType}{(v.Ints is { } i ? "=" + string.Join(",", i.Take(6)) : "")}" : "?";
        if (depth == 0 || !_producer.TryGetValue(value, out var n)) return known;
        var node = _graph.Nodes[n];
        return $"{node.OpType}({string.Join("; ", node.Inputs.Where(x => x.Length > 0).Select(x => Explain(x, depth - 1)))})->{known}";
    }

    /// <summary>The part each node of the step belongs to, counted.</summary>
    internal IReadOnlyDictionary<Part, int> NodesByPart => Enum.GetValues<Part>().ToDictionary(p => p, p => _parts.Count(x => x == p));

    /// <summary>The node an order runs at position <paramref name="position"/>, and the part it
    /// belongs to.</summary>
    internal string At(int[] order, int position) => $"{_graph.Nodes[order[position]].OpType} ({_parts[order[position]]})";
}

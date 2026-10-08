using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory.IR;

namespace Shorokoo.OnnxRuntime;

/// <summary>
/// Writes each Adam or AdamW parameter update of a training step's model as one
/// <see cref="AdamUpdate"/> node, an operator the native library adds to ONNX Runtime's CPU
/// provider (<c>native/shorokoo_ort_ops.cpp</c>), in place of the chain of element-wise operators
/// the optimizer is built of.
///
/// <para><b>Why.</b> Written out, the update is twelve operators over the parameter's size —
/// thirteen with a weight decay — and ONNX Runtime runs each as a pass that streams tensors of
/// that size through memory: about 29 parameter-sized reads and writes per element, where the
/// update needs 7 (the parameter, its gradient and both moments read, the parameter and the
/// moments written). Those passes are bound by memory, not arithmetic, so written out the update
/// makes a training step cost in proportion to the parameter count: measured on the host, a model
/// with 2.5 times another's parameters at the same arithmetic steps 13–15% slower that way, and
/// 5–6% slower fused (Shorokoo/Shorokoo#508). The fused operator makes the one pass.</para>
///
/// <para><b>Same result.</b> It computes what the chain computes, each operation in float32 and in
/// the chain's order, the coefficients the chain computes from the hyperparameters fed to it as
/// they are; so a step gives the same parameters and moments to the bit with it as without. A chain
/// is rewritten only where every part of it is as below, the moments and the parameter are inputs
/// of the step of one stated float32 shape, the gradient is a float32 value of that shape too —
/// as the graph states its type, or as its shape follows from the dimensions the graph states for
/// its inputs — and every coefficient is a single value of no greater rank:</para>
/// <code>
///   m' = beta1 * m + c1 * g
///   v' = beta2 * v + (c2 * g) * g
///   p' = p [* decay] - m' / (sqrt(v') + eps) * step
/// </code>
/// <para>Each intermediate value is read by the chain alone; <c>m'</c> and <c>v'</c> may be read
/// elsewhere too, as the step's state outputs are. Anything else is left as written.</para>
///
/// <para><b>Where.</b> Only in a training step's session on the host
/// (<see cref="Core.Backends.ShorokooGraphOptimization.TrainingStep"/>), of a backend that runs the
/// CPU provider alone. A CUDA session has no such kernel and keeps the chain.</para>
/// </summary>
internal static class OrtFusedUpdates
{
    /// <summary>The domain of the operators the native library registers.</summary>
    internal const string Domain = "ai.shorokoo";

    /// <summary>The fused update's operator type.</summary>
    internal const string AdamUpdate = "AdamUpdate";

    /// <summary>Fuses <paramref name="model"/>'s Adam and AdamW updates in place, importing the
    /// operators' domain where it fused any; whether it did.</summary>
    internal static bool Fuse(ModelProto model)
    {
        if (model.Graph is not { } graph || !Fuse(graph)) return false;
        if (!model.OpsetImports.Any(o => o.Domain == Domain))
            model.OpsetImports.Add(new OperatorSetIdProto { Domain = Domain, Version = 1 });
        return true;
    }

    /// <summary>Whether <paramref name="node"/> is a fused update.</summary>
    internal static bool IsAdamUpdate(NodeProto node) => node.Domain == Domain && node.OpType == AdamUpdate;

    /// <summary>Rewrites <paramref name="graph"/>'s updates in place; whether it rewrote any.</summary>
    private static bool Fuse(GraphProto graph)
    {
        var index = new Index(graph);
        var removed = new HashSet<int>();
        var added = new List<(int After, NodeProto Node)>();
        for (int n = 0; n < graph.Nodes.Count; n++)
            if (index.Match(n, removed) is { } match)
            {
                removed.UnionWith(match.Chain);
                added.Add((match.After, match.Node));
            }
        if (added.Count == 0) return false;

        var nodes = new List<NodeProto>(graph.Nodes.Count);
        foreach (var (after, node) in added)
            if (after < 0) nodes.Add(node);
        for (int n = 0; n < graph.Nodes.Count; n++)
        {
            if (!removed.Contains(n)) nodes.Add(graph.Nodes[n]);
            foreach (var (after, node) in added)
                if (after == n) nodes.Add(node);
        }
        graph.Nodes.Clear();
        graph.Nodes.AddRange(nodes);
        return true;
    }

    /// <summary>A fused update: the node, the nodes it replaces, and the node it goes after (-1 for
    /// the start of the graph).</summary>
    private sealed record Fused(NodeProto Node, int[] Chain, int After);

    /// <summary>The graph, indexed for matching.</summary>
    private sealed class Index
    {
        private readonly GraphProto _graph;
        private readonly Dictionary<string, int> _producer = new(StringComparer.Ordinal);
        private readonly Dictionary<string, List<int>> _consumers = new(StringComparer.Ordinal);
        private readonly HashSet<string> _outputs = new(StringComparer.Ordinal);
        private readonly Dictionary<string, ValueInfoProto> _inputs = new(StringComparer.Ordinal);

        // Every value whose type the graph states: its inputs, outputs and value infos.
        private readonly Dictionary<string, ValueInfoProto> _stated = new(StringComparer.Ordinal);

        // The shape and element type of each value that follows from the dimensions the graph states
        // for its inputs; worked out the first time a value of no stated type is asked about.
        private Dictionary<string, PlacementShapes.Value>? _inferred;

        // Per value known to hold a single element, its rank.
        private readonly Dictionary<string, int> _single = new(StringComparer.Ordinal);

        internal Index(GraphProto graph)
        {
            _graph = graph;
            foreach (var output in graph.Outputs) _outputs.Add(output.Name);
            foreach (var info in graph.Inputs.Concat(graph.Outputs).Concat(graph.ValueInfoes)) _stated.TryAdd(info.Name, info);
            foreach (var input in graph.Inputs)
            {
                _inputs[input.Name] = input;
                if (StatedDims(input) is { } dims && dims.All(d => d.DimValue == 1 && string.IsNullOrEmpty(d.DimParam)))
                    _single[input.Name] = dims.Count;
            }
            foreach (var initializer in graph.Initializers)
            {
                _inputs.Remove(initializer.Name);
                if ((initializer.Dims ?? []).All(d => d == 1)) _single[initializer.Name] = initializer.Dims?.Length ?? 0;
                else _single.Remove(initializer.Name);
            }
            for (int n = 0; n < graph.Nodes.Count; n++)
            {
                var node = graph.Nodes[n];
                foreach (var output in node.Outputs)
                    if (output.Length > 0) _producer[output] = n;
                foreach (var input in node.Inputs)
                {
                    if (input.Length == 0) continue;
                    if (!_consumers.TryGetValue(input, out var readers)) _consumers[input] = readers = [];
                    readers.Add(n);
                }
                if (SingleRank(node) is { } rank) _single[node.Outputs[0]] = rank;
            }
        }

        /// <summary>The rank of <paramref name="node"/>'s one output where it is known to hold a
        /// single element: a constant of one element, or an element-wise operator over such values.</summary>
        private int? SingleRank(NodeProto node)
        {
            if (!Standard(node) || node.Outputs.Count != 1) return null;
            if (node.OpType == "Constant")
            {
                if (node.Attributes.Count != 1) return null;
                var attribute = node.Attributes[0];
                if (attribute.Name is "value_float" or "value_int") return 0;
                if (attribute.Name == "value" && attribute.T is { } t && (t.Dims ?? []).All(d => d == 1)) return t.Dims?.Length ?? 0;
                return null;
            }
            if (node.OpType is not ("Add" or "Sub" or "Mul" or "Div" or "Pow" or "Sqrt" or "Neg" or "Reciprocal"
                or "Exp" or "Log" or "Abs" or "Identity" or "Cast") || node.Inputs.Count == 0)
                return null;
            int most = 0;
            foreach (var input in node.Inputs)
            {
                if (!_single.TryGetValue(input, out var rank)) return null;
                most = Math.Max(most, rank);
            }
            return most;
        }

        /// <summary>The fused update whose parameter output is <paramref name="sub"/>'s, where the
        /// node is the last of an update chain none of whose nodes is in <paramref name="taken"/>.</summary>
        internal Fused? Match(int sub, HashSet<int> taken)
        {
            var s = _graph.Nodes[sub];
            if (!Is(s, "Sub", 2)) return null;
            var chain = new List<int> { sub };

            // p' = X - U, with U = Q * step, Q = m' / D and D = sqrt(v') + eps.
            if (!Scaled(s.Inputs[1], sub, chain, out var q, out var step)
                || !Producer(q, "Div", 2, chain[^1], chain, out var divAt))
                return null;
            var (mNew, denominator) = (_graph.Nodes[divAt].Inputs[0], _graph.Nodes[divAt].Inputs[1]);
            if (!Producer(denominator, "Add", 2, divAt, chain, out var denominatorAt)) return null;
            string? vNew = null, eps = null;
            foreach (var (a, b) in Pairs(_graph.Nodes[denominatorAt]))
                if (_single.ContainsKey(b) && Producer(a, "Sqrt", 1, denominatorAt, chain, out var rootAt))
                {
                    (vNew, eps) = (_graph.Nodes[rootAt].Inputs[0], b);
                    break;
                }
            if (vNew is null) return null;

            // v' = beta2 * v + (c2 * g) * g, its Add read by the chain and as the step's state.
            if (!_producer.TryGetValue(vNew, out var vAt) || !Is(_graph.Nodes[vAt], "Add", 2)) return null;
            chain.Add(vAt);
            string? v = null, beta2 = null, g = null, c2 = null;
            foreach (var (a, b) in Pairs(_graph.Nodes[vAt]))
            {
                var mark = chain.Count;
                if (Squared(b, vAt, chain, out var gg, out var cc) && Scaled(a, vAt, chain, out var vv, out var bb))
                {
                    (v, beta2, g, c2) = (vv, bb, gg, cc);
                    break;
                }
                chain.RemoveRange(mark, chain.Count - mark);
            }
            if (v is null || g is null) return null;

            // m' = beta1 * m + c1 * g, likewise.
            if (!_producer.TryGetValue(mNew, out var mAt) || !Is(_graph.Nodes[mAt], "Add", 2)) return null;
            chain.Add(mAt);
            string? m = null, beta1 = null, c1 = null;
            foreach (var (a, b) in Pairs(_graph.Nodes[mAt]))
            {
                var mark = chain.Count;
                if (Scaled(b, mAt, chain, out var gg, out var cc) && gg == g && Scaled(a, mAt, chain, out var mm, out var bb))
                {
                    (m, beta1, c1) = (mm, bb, cc);
                    break;
                }
                chain.RemoveRange(mark, chain.Count - mark);
            }
            if (m is null) return null;

            // X = p, or p * decay.
            string p = s.Inputs[0];
            string? decay = null;
            if (!_inputs.ContainsKey(p))
            {
                if (!Scaled(p, sub, chain, out var pp, out var dd)) return null;
                (p, decay) = (pp, dd);
            }

            // The state is three distinct inputs of one stated float32 shape, the gradient none of
            // them and a float32 value of that shape too, and every coefficient a single value of no
            // greater rank.
            if (!_inputs.TryGetValue(p, out var pInfo) || !_inputs.TryGetValue(m, out var mInfo) || !_inputs.TryGetValue(v, out var vInfo))
                return null;
            if (new HashSet<string>(StringComparer.Ordinal) { p, m, v, g }.Count != 4 || _single.ContainsKey(g)) return null;
            if (StatedDims(pInfo) is not { } dims || !IsFloat(pInfo) || !IsFloat(mInfo) || !IsFloat(vInfo)
                || !SameDims(dims, StatedDims(mInfo)) || !SameDims(dims, StatedDims(vInfo))
                || !IsFloatOf(g, dims))
                return null;
            string[] coefficients = decay is null ? [beta1!, c1!, beta2!, c2!, eps!, step] : [beta1!, c1!, beta2!, c2!, eps!, step, decay];
            if (coefficients.Any(c => !_single.TryGetValue(c, out var rank) || rank > dims.Count)) return null;
            if (chain.Any(taken.Contains) || chain.Distinct().Count() != chain.Count) return null;

            // After everything it reads, and before everything that reads what it writes.
            string[] inputs = [p, m, v, g, .. coefficients];
            var after = inputs.Select(i => _producer.TryGetValue(i, out var at) ? at : -1).Max();
            string[] written = [s.Outputs[0], mNew, vNew];
            var inChain = chain.ToHashSet();
            foreach (var output in written)
                foreach (var reader in _consumers.GetValueOrDefault(output) ?? [])
                    if (!inChain.Contains(reader) && reader <= after) return null;

            var node = new NodeProto { OpType = AdamUpdate, Domain = Domain, Name = AdamUpdate + "_" + s.Outputs[0] };
            node.Inputs.AddRange(inputs);
            node.Outputs.AddRange(written);
            return new Fused(node, [.. chain], after);
        }

        /// <summary>Whether <paramref name="value"/> is a tensor times a single value, written by a
        /// node only <paramref name="reader"/> reads: the tensor and the single value, the node
        /// added to <paramref name="chain"/>.</summary>
        private bool Scaled(string value, int reader, List<int> chain, out string tensor, out string single)
        {
            (tensor, single) = ("", "");
            if (!Producer(value, "Mul", 2, reader, chain, out var at)) return false;
            foreach (var (a, b) in Pairs(_graph.Nodes[at]))
                if (_single.ContainsKey(b) && !_single.ContainsKey(a))
                {
                    (tensor, single) = (a, b);
                    return true;
                }
            chain.RemoveAt(chain.Count - 1);
            return false;
        }

        /// <summary>Whether <paramref name="value"/> is <c>(c * g) * g</c>, its nodes read by the
        /// chain alone: <c>g</c> and <c>c</c>, the nodes added to <paramref name="chain"/>.</summary>
        private bool Squared(string value, int reader, List<int> chain, out string g, out string c)
        {
            (g, c) = ("", "");
            if (!Producer(value, "Mul", 2, reader, chain, out var outer)) return false;
            foreach (var (a, b) in Pairs(_graph.Nodes[outer]))
            {
                if (!Scaled(a, outer, chain, out var gg, out var cc)) continue;
                if (gg == b)
                {
                    (g, c) = (gg, cc);
                    return true;
                }
                chain.RemoveAt(chain.Count - 1);
            }
            chain.RemoveAt(chain.Count - 1);
            return false;
        }

        /// <summary>Whether <paramref name="value"/> is the output of a standard <paramref name="op"/>
        /// of <paramref name="arity"/> inputs, no output of the graph, and read by
        /// <paramref name="reader"/> alone: that node, added to <paramref name="chain"/>.</summary>
        private bool Producer(string value, string op, int arity, int reader, List<int> chain, out int at)
        {
            if (!_producer.TryGetValue(value, out at) || !Is(_graph.Nodes[at], op, arity) || _outputs.Contains(value)
                || !_consumers.TryGetValue(value, out var readers) || readers.Count != 1 || readers[0] != reader)
                return false;
            chain.Add(at);
            return true;
        }

        private static IEnumerable<(string, string)> Pairs(NodeProto node)
        {
            yield return (node.Inputs[0], node.Inputs[1]);
            yield return (node.Inputs[1], node.Inputs[0]);
        }

        private static bool Is(NodeProto node, string op, int arity)
            => Standard(node) && node.OpType == op && node.Inputs.Count == arity && node.Outputs.Count == 1
               && node.Attributes.Count == 0 && node.Inputs.All(i => i.Length > 0);

        private static bool Standard(NodeProto node) => node.Domain is null or "" or "ai.onnx";

        private static List<TensorShapeProto.Dimension>? StatedDims(ValueInfoProto info)
            => info.Type?.TensorType?.Shape?.Dims;

        private static bool IsFloat(ValueInfoProto info) => info.Type?.TensorType?.ElemType == 1;

        /// <summary>Whether <paramref name="value"/> is a float32 value of <paramref name="dims"/>: as
        /// the graph states its type, or, where it states none, as its shape follows from the
        /// dimensions the graph states for its inputs.</summary>
        private bool IsFloatOf(string value, List<TensorShapeProto.Dimension> dims)
        {
            if (_stated.TryGetValue(value, out var info) && info.Type?.TensorType is not null)
                return IsFloat(info) && SameDims(dims, StatedDims(info));
            if (dims.Any(d => d.DimValue <= 0)) return false;
            _inferred ??= PlacementShapes.Evaluate(_graph, _graph.Inputs
                .Where(i => i.Type?.TensorType is { Shape: { } shape } && shape.Dims.All(d => d.DimValue > 0))
                .ToDictionary(i => i.Name, i => (i.Type.TensorType.Shape.Dims.Select(d => d.DimValue).ToArray(), i.Type.TensorType.ElemType),
                    StringComparer.Ordinal));
            return _inferred.TryGetValue(value, out var inferred) && inferred.ElementType == 1
                && inferred.Shape.AsSpan().SequenceEqual(dims.Select(d => d.DimValue).ToArray());
        }

        private static bool SameDims(List<TensorShapeProto.Dimension> a, List<TensorShapeProto.Dimension>? b)
            => b is not null && a.Count == b.Count && a.Zip(b).All(d =>
                d.First.DimValue > 0 ? d.First.DimValue == d.Second.DimValue && string.IsNullOrEmpty(d.Second.DimParam)
                : !string.IsNullOrEmpty(d.First.DimParam) && d.First.DimParam == d.Second.DimParam);
    }
}

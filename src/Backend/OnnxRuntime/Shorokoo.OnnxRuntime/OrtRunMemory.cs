using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory.IR;

namespace Shorokoo.OnnxRuntime;

/// <summary>
/// The most a run holds at once beyond its inputs and initializers, as ONNX Runtime runs it: over
/// the graph a session writes out (<c>OptimizedModelFilePath</c>), its fusions and rewrites made,
/// in that graph's order, which is the order a session runs its nodes in (see
/// <see cref="OrtBackend.Configure"/>), each loop's body run once per iteration in its own order and
/// each sequence the tensors it holds (<see cref="UnrolledRun"/>) — on a card, a loop's carried
/// sequences copied out where it ends, which holds them twice there, as measured at three sizes of the
/// LSTM benchmark's step — laid out as its allocation plan lays a run out:
///
/// <list type="bullet">
/// <item>a node takes a buffer for each output it makes; an output a kernel hands back over its input
/// (<see cref="OutputAliasProof.Shares"/>) is that input's buffer, and an activation that runs in place
/// writes over its input where that dies there;</item>
/// <item>a buffer whose every value has died is kept for the next output of its exact shape and type,
/// the most recently let go of first, and is occupied from its first value's birth to its last one's
/// death; one that no later output takes is given back as its value dies, and so is one whose value's
/// shape the graph does not fix (<see cref="ShapesNotFixed(GraphProto)"/>), which takes a buffer of its own, and
/// one made in a loop's body, which the body's own frame gives back as its iteration ends;</item>
/// <item>an output the run writes into the input it is paired with takes nothing, a constant — which
/// a session holds as it holds an initializer — takes nothing, and every other output is held to the
/// run's end;</item>
/// <item>a kernel's own scratch for the node's length: on the host as <see cref="HostScratch"/> says,
/// on a card as <see cref="CardScratch"/> does.</item>
/// </list>
/// </summary>
internal static class OrtRunMemory
{
    /// <summary>The activations ONNX Runtime's kernels write over their input where it dies there:
    /// they register to run in place.</summary>
    private static readonly HashSet<string> InPlaceActivations = new(StringComparer.Ordinal)
    {
        "Relu", "Sigmoid", "Tanh", "Elu", "LeakyRelu", "HardSigmoid", "Selu", "Softplus", "Softsign", "ThresholdedRelu",
    };

    /// <summary>
    /// The peak of a run of <paramref name="graph"/>, the graph ONNX Runtime runs, fed
    /// <paramref name="inputs"/> (each input's shape and element type), with the outputs of
    /// <paramref name="aliases"/> written into their inputs; on the host where
    /// <paramref name="onHost"/>, with its kernels' scratch. Null where a value's shape cannot be
    /// told, which leaves its bytes unknown.
    /// </summary>
    internal static long? Peak(GraphProto graph, IReadOnlyDictionary<string, (long[] Shape, int ElementType)> inputs,
        IReadOnlyList<OutputAlias> aliases, bool onHost)
    {
        UnrolledRun.Run? run;
        try
        {
            run = UnrolledRun.Of(graph, inputs, written => written, copiesListsAtLoopEnd: !onHost);
        }
        catch (ArgumentException)
        {
            return null;
        }
        if (run is null) return null;
        var (nodes, shapes) = (run.Order, run.Shapes);
        if (nodes.Any(n => n.Outputs.Any(o => o.Length > 0 && !shapes.ContainsKey(o)))) return null;

        var root = new Dictionary<string, string>(StringComparer.Ordinal);
        string RootOf(string v) => root.TryGetValue(v, out var r) && r != v ? root[v] = RootOf(r) : v;
        foreach (var node in nodes)
            for (int o = 0; o < node.Outputs.Count; o++)
                for (int i = 0; i < node.Inputs.Count; i++)
                    if (node.Outputs[o].Length > 0 && node.Inputs[i].Length > 0 && OutputAliasProof.Shares(node, i, o))
                    {
                        root[node.Outputs[o]] = RootOf(node.Inputs[i]);
                        break;
                    }
        var intoInputs = aliases.Select(a => a.Output).ToHashSet(StringComparer.Ordinal);
        var held = graph.Outputs.Select(o => o.Name).Where(o => !intoInputs.Contains(o)).Select(RootOf).ToHashSet(StringComparer.Ordinal);
        var lastRead = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int k = 0; k < nodes.Count; k++)
            foreach (var v in nodes[k].Inputs.Where(v => v.Length > 0))
                lastRead[RootOf(v)] = k;

        var buffers = new List<(long Bytes, string Shape, int Start, int End)>();
        var bufferOf = new Dictionary<string, int>(StringComparer.Ordinal);
        var users = new List<int>();
        var free = new List<int>();
        var scratch = new long[nodes.Count];
        var unfixed = ShapesNotFixed(nodes, graph.Initializers.Select(i => i.Name));
        unfixed.UnionWith(run.Iterated);
        for (int k = 0; k < nodes.Count; k++)
        {
            var node = nodes[k];
            scratch[k] = onHost ? HostScratch(node, shapes) : CardScratch(node, shapes);
            foreach (var o in node.Outputs.Where(o => o.Length > 0))
            {
                if (RootOf(o) != o || intoInputs.Contains(o) || shapes[o].Bytes <= 0 || node.OpType == "Constant") continue;
                var value = shapes[o];
                var shape = $"{value.ElementType}[{string.Join(",", value.Shape)}]";
                if (InPlaceActivations.Contains(node.OpType) && OutputAliasProof.IsStandard(node)
                    && node.Inputs.Count > 0 && node.Inputs[0] is { Length: > 0 } operand
                    && bufferOf.TryGetValue(RootOf(operand), out var over) && users[over] == 1
                    && lastRead.GetValueOrDefault(RootOf(operand), -1) == k && buffers[over].Shape == shape)
                {
                    bufferOf[o] = over;
                    users[over]++;
                    continue;
                }
                var taken = unfixed.Contains(o) ? -1 : free.FindLastIndex(b => buffers[b].Shape == shape);
                if (taken >= 0)
                {
                    var id = free[taken];
                    free.RemoveAt(taken);
                    buffers[id] = buffers[id] with { End = int.MaxValue };
                    bufferOf[o] = id;
                    users[id] = 1;
                    continue;
                }
                buffers.Add((value.Bytes, shape, k, int.MaxValue));
                users.Add(1);
                bufferOf[o] = buffers.Count - 1;
            }
            foreach (var v in node.Inputs.Concat(node.Outputs).Where(v => v.Length > 0).Select(RootOf).Distinct())
            {
                if (!bufferOf.TryGetValue(v, out var id) || held.Contains(v) || lastRead.GetValueOrDefault(v, k) > k) continue;
                bufferOf.Remove(v);
                if (--users[id] > 0) continue;
                buffers[id] = buffers[id] with { End = k };
                if (!unfixed.Contains(v)) free.Add(id);
            }
        }

        var delta = new long[nodes.Count + 1];
        foreach (var (bytes, _, start, end) in buffers)
        {
            delta[start] += bytes;
            delta[Math.Min(end, nodes.Count - 1) + 1] -= bytes;
        }
        long occupied = 0, peak = 0;
        for (int k = 0; k < nodes.Count; k++)
        {
            occupied += delta[k];
            peak = Math.Max(peak, occupied + scratch[k]);
        }
        return peak;
    }

    /// <summary>
    /// The values of <paramref name="graph"/> whose shape the graph alone does not fix: the outputs
    /// of an operator whose shape follows from the contents of an input that is not an initializer
    /// — a reshape to a computed shape, a reduction over computed axes, a slice of computed bounds —
    /// or from the contents of its data (a compression, a loop). ONNX Runtime's allocation plan
    /// reuses a buffer only for a value its own inference gives a shape, and hands one on only from
    /// such a value: these take a buffer of their own, and give it back as they die.
    /// </summary>
    internal static HashSet<string> ShapesNotFixed(GraphProto graph)
        => ShapesNotFixed(graph.Nodes, graph.Initializers.Select(i => i.Name));

    /// <summary><see cref="ShapesNotFixed(GraphProto)"/> over <paramref name="nodes"/>, whose
    /// initializers are <paramref name="initializers"/>.</summary>
    private static HashSet<string> ShapesNotFixed(IEnumerable<NodeProto> nodes, IEnumerable<string> initializers)
    {
        var constant = initializers.ToHashSet(StringComparer.Ordinal);
        var unfixed = new HashSet<string>(StringComparer.Ordinal);
        foreach (var node in nodes)
        {
            int[] read = node.OpType switch
            {
                "Reshape" or "Expand" or "Tile" or "Pad" or "Split" or "TopK" or "OneHot" => [1],
                "ReduceSum" or "ReduceMean" or "ReduceMax" or "ReduceMin" or "ReduceProd" or "ReduceL1" or "ReduceL2"
                    or "ReduceLogSum" or "ReduceLogSumExp" or "ReduceSumSquare" or "Squeeze" or "Unsqueeze" => [1],
                "Slice" => [1, 2, 3, 4],
                "ConstantOfShape" => [0],
                "Range" => [0, 1, 2],
                "Resize" or "Upsample" => [1, 2, 3],
                _ => [],
            };
            if (node.OpType is "Compress" or "NonZero" or "Unique" or "NonMaxSuppression" or "Loop" or "If" or "Scan"
                || read.Any(i => i < node.Inputs.Count && node.Inputs[i].Length > 0 && !constant.Contains(node.Inputs[i])))
                foreach (var output in node.Outputs.Where(o => o.Length > 0)) unfixed.Add(output);
        }
        return unfixed;
    }

    /// <summary>
    /// What a kernel of ONNX Runtime's CUDA provider takes for its own use while
    /// <paramref name="node"/> runs, measured kernel by kernel through Shorokoo's allocator in strict
    /// float32: a reduction over axes that are neither the leading nor the trailing ones of its
    /// input — those of more than one element — a workspace of its input's size; one over leading or
    /// trailing axes, which the provider sums as the rows or columns of a matrix, none.
    /// </summary>
    internal static long CardScratch(NodeProto node, IReadOnlyDictionary<string, PlacementShapes.Value> shapes)
    {
        if (!OutputAliasProof.IsStandard(node) || !node.OpType.StartsWith("Reduce", StringComparison.Ordinal)
            || node.Inputs.Count == 0 || !shapes.TryGetValue(node.Inputs[0], out var input)) return 0;
        var rank = input.Shape.Length;
        long[]? axes = node.Inputs.Count > 1 && node.Inputs[1].Length > 0
            ? shapes.TryGetValue(node.Inputs[1], out var given) ? given.Ints : null
            : node.Attributes.FirstOrDefault(a => a.Name == "axes")?.Ints;
        if (axes is null || axes.Length == 0) return 0;
        var reduced = axes.Select(a => (int)((a % rank + rank) % rank)).ToHashSet();
        var counted = Enumerable.Range(0, rank).Where(d => input.Shape[d] != 1).ToList();
        var flags = counted.Select(reduced.Contains).ToList();
        var leading = flags.SkipWhile(f => f).All(f => !f);
        var trailing = flags.SkipWhile(f => !f).All(f => f);
        return leading || trailing ? 0 : Math.Max(input.Bytes, 0);
    }

    /// <summary>
    /// What a kernel of ONNX Runtime's CPU provider takes for its own use while <paramref name="node"/>
    /// runs, measured kernel by kernel through Shorokoo's allocator:
    /// <list type="bullet">
    /// <item><c>Where</c>: each operand selected into a value of the output's size before the two are
    /// merged — two of them;</item>
    /// <item>a convolution (<c>Conv</c>, the fused <c>FusedConv</c>): its columns expanded for a matrix
    /// product, the whole of one image's where the output is small (under 64 positions), and
    /// otherwise 64 KiB for each image computed at once, up to 64 of them — or for each segment of
    /// 256 output positions of a single image; nothing for a pointwise one;</item>
    /// <item><c>ConvTranspose</c>: one image's columns;</item>
    /// <item><c>LSTM</c>: its gates for every step, four values of the output's size, and five of a
    /// step's hidden state.</item>
    /// </list>
    /// </summary>
    internal static long HostScratch(NodeProto node, IReadOnlyDictionary<string, PlacementShapes.Value> shapes)
    {
        PlacementShapes.Value? Of(int input) => input < node.Inputs.Count && node.Inputs[input] is { Length: > 0 } name && shapes.TryGetValue(name, out var v) ? v : null;
        var output = node.Outputs.Count > 0 && shapes.TryGetValue(node.Outputs[0], out var y) ? y : null;
        if (output is null) return 0;
        long element = output.Elements > 0 ? output.Bytes / output.Elements : 4;
        long[] Ints(string name) => node.Attributes.FirstOrDefault(a => a.Name == name)?.Ints?.ToArray() ?? [];
        long Group() => node.Attributes.FirstOrDefault(a => a.Name == "group")?.I is long g and > 0 ? g : 1;
        switch (node.OpType)
        {
            case "Where" when OutputAliasProof.IsStandard(node):
                return 2 * output.Bytes;
            case "Conv" or "FusedConv" when Of(0) is { } x && Of(1) is { } w && x.Shape.Length >= 3 && w.Shape.Length == x.Shape.Length:
            {
                var kernel = w.Shape[2..];
                long[] strides = Ints("strides"), pads = Ints("pads");
                if (kernel.All(d => d == 1) && strides.All(s => s == 1) && pads.All(p => p == 0)) return 0;
                var columns = w.Shape[1] * kernel.Aggregate(1L, (a, d) => a * d);
                var positions = output.Shape[2..].Aggregate(1L, (a, d) => a * d);
                if (positions < 64) return positions * columns * element;
                var images = x.Shape[0] * Group();
                var threads = images > 1 ? Math.Min(images, 64) : Math.Clamp(positions / 256, 1, 64);
                return threads * 16384 * element;
            }
            case "ConvTranspose" when Of(0) is { } x && Of(1) is { } w && x.Shape.Length >= 3 && w.Shape.Length == x.Shape.Length:
                return w.Shape[1] * w.Shape[2..].Aggregate(1L, (a, d) => a * d) * x.Shape[2..].Aggregate(1L, (a, d) => a * d) * element;
            case "LSTM" when Of(0) is { } x && x.Shape.Length == 3:
            {
                var hidden = node.Attributes.FirstOrDefault(a => a.Name == "hidden_size")?.I ?? 0;
                var directions = node.Attributes.FirstOrDefault(a => a.Name == "direction")?.S is { } d && System.Text.Encoding.UTF8.GetString(d) == "bidirectional" ? 2 : 1;
                return (4 * x.Shape[0] + 5) * x.Shape[1] * hidden * directions * element;
            }
            default:
                return 0;
        }
    }
}

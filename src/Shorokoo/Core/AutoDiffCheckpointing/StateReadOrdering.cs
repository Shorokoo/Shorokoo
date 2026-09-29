using Shorokoo.Core.Backends;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Graph;
using System.Collections.Generic;
using System.Linq;

namespace Shorokoo.Core.AutoDiffCheckpointing;

/// <summary>
/// Orders the writer of each updated state output after every reader of the state it replaces, so
/// that the output can be written into that state's memory (<see cref="OutputAlias"/>) whatever
/// the backend folds before it runs the step.
///
/// <para><b>Why the order matters.</b> A state pair is written in place only when every node
/// reading the input has run before the output's writer (<see cref="OutputAliasProof"/>). A
/// backend proves that again over the graph it runs, after its own rewrites, and those rewrites
/// create readers: a Linear's backward multiplies the gradient by the transpose of the forward's
/// transpose of its weight, a backend cancels the pair, and the product — the gradient flowing to
/// the layer below — reads the weight itself. Reading the weight there is the standard backward;
/// what loses the pair is that nothing orders the weight's update after that product. So the
/// readers to order are the nodes reading the input or a value computed from it and constants
/// alone — its transposes, reshapes, casts and the like — which is every reader folding can make
/// (see <see cref="OutputAliasProof.Unordered"/>, which finds them). Tied weights, and any other
/// state read in several places, are the same case: each place is a reader.</para>
///
/// <para><b>How it is ordered.</b> ONNX has no control edges, so the order is a data dependency,
/// and one that costs O(1) whatever the state's size. For each reader it keeps, the ordering takes
/// an empty slice of the reader's output — <c>Reshape(Slice(out, [0], [0], [0]), [-1])</c>, the
/// first zero rows along its first axis (a scalar is first reshaped to one element), so zero
/// elements whatever the output's shape, a zero-element output included, and a slice reads only
/// what it keeps — casts it to the anchor's element type, and concatenates it onto a small value
/// the writer depends on (the anchor):
/// <c>Reshape(Concat(Reshape(anchor, [-1]), empty…), shape(anchor))</c>. Concatenating
/// zero elements is exactly the anchor, bit for bit, for every value including NaN and
/// infinity; nothing is multiplied or added. The one consumer on the writer's side that read the
/// anchor reads the concatenation instead, so it — and the writer after it — waits for every
/// reader. The anchor is found walking back from the writer along the edges that order execution,
/// among nodes no reader depends on: the first floating-point input of at most
/// <see cref="MaxAnchorElements"/> elements (<see cref="Anchorable"/>), preferring one that is not
/// a constant, whose value a backend would otherwise fold.
/// For AdamW that is the decay factor of <c>param * (1 - lr * weightDecay)</c> or the step size;
/// for SGD, the learning rate. A backend folds none of it: a <c>Slice</c> of a value computed at
/// run time is not a constant, and a <c>Concat</c> with a non-constant input is kept — ONNX
/// Runtime keeps the dependency through its training-step optimizations, and writes every pair of
/// a step so ordered in place.</para>
///
/// <para><b>What is left alone.</b> A pair one of whose readers depends on the writer cannot be
/// ordered — the reader needs the update — and a pair with no small anchor outside every reader's
/// ancestry is left rather than routed through a state-sized operand, which would cost a pass
/// over it; a reader that is a scope, or writes no tensor it can be sliced from, is left too.
/// Such a pair keeps whatever order it had, and whether it is written in place is for the proof
/// to say. A graph holding a scope is handed back as it came, as the rest of the memory-aware
/// pass hands it back.</para>
///
/// <para>The ordering is a candidate, not a given: <see cref="MemoryAwareGraphOptimizer"/> scores
/// it against the graph as it came and keeps the better. It pays for itself when the step writes
/// its state in place, since each pair it keeps is a state-sized buffer the run does not allocate;
/// it costs a few kernels and can hold a reader's output a little longer. Where the step writes
/// nothing in place the ordering buys nothing and is not tried.</para>
/// </summary>
internal static class StateReadOrdering
{
    /// <summary>The most elements an anchor may have: the concatenation copies it once.</summary>
    internal const long MaxAnchorElements = 64;

    /// <summary>
    /// <paramref name="graph"/> with the writer of each pair of <paramref name="pairs"/> it can
    /// order placed after every reader of that pair's input, and shape information covering every
    /// node the ordering added — or the very same graph and shape information where there is
    /// nothing to order.
    /// </summary>
    public static (InternalComputationGraph Graph, ShapeInferenceResult ShapeInfo) Apply(
        InternalComputationGraph graph, ShapeInferenceResult shapeInfo, IReadOnlyList<(int Output, int Input)> pairs)
    {
        if (pairs.Count == 0 || graph.Nodes.Any(n => n.IsOpenNode())) return (graph, shapeInfo);
        var unordered = OutputAliasProof.Unordered(graph, pairs).Where(p => p.Readers.Count > 0).ToList();
        if (unordered.Count == 0) return (graph, shapeInfo);

        var copy = graph.Clone();
        var nodes = copy.Nodes;
        var inputs = copy.Inputs;
        var dag = new Dag(nodes);
        var shapes = shapeInfo.TensorInfos.ToBuilder();
        var constants = new Constants(shapes);
        var insertBefore = new Dictionary<FastNode, List<FastNode>>();
        var ordered = false;

        foreach (var pair in unordered)
        {
            if (pair.Readers.Any(r => r.First != r.Last)) continue;
            var writer = nodes[pair.Writer];
            var readers = dag.Unordered(writer, pair.Readers.Select(r => nodes[r.First]));
            if (readers is null || readers.Count == 0) continue;
            var outputs = readers.Select(reader => SliceableOutput(reader, shapes)).ToList();
            if (outputs.Any(o => o is null)) continue;
            if (Anchor(dag, writer, readers, inputs[pair.Pair.Input], shapes) is not var (consumer, slot, index, anchor, anchorInfo))
                continue;
            var slices = new List<FastTensorKey?>(readers.Count + 1);
            var added = new List<FastNode>();
            foreach (var (output, outputInfo) in outputs.Select(o => o!.Value))
            {
                var dims = outputInfo.Shape.Dims;
                var sliced = dims.Length > 0 ? output
                    : Add(added, shapes, OpCodes.RESHAPE, [], [output, constants.Vector(-1L)], new TensorShapeInfo(new Shape(1L), outputInfo.DType, null));
                long[] emptyDims = dims.Length > 0 ? [0L, .. dims.Skip(1)] : [0L];
                var empty = Add(added, shapes, OpCodes.SLICE, [], [sliced, constants.Vector(0L), constants.Vector(0L), constants.Vector(0L)],
                    new TensorShapeInfo(new Shape(emptyDims), outputInfo.DType, null));
                if (emptyDims.Length > 1)
                    empty = Add(added, shapes, OpCodes.RESHAPE, [], [empty, constants.Vector(-1L)], new TensorShapeInfo(new Shape(0L), outputInfo.DType, null));
                slices.Add(outputInfo.DType == anchorInfo.DType ? empty
                    : Add(added, shapes, OpCodes.CAST, new() { [OnnxOpAttributeNames.AttrTo] = anchorInfo.DType }, [empty],
                        new TensorShapeInfo(new Shape(0L), anchorInfo.DType, null)));
            }

            var rank = anchorInfo.Shape.Dims.Length;
            var flatInfo = new TensorShapeInfo(new Shape(anchorInfo.ElementCount), anchorInfo.DType, null);
            var flatAnchor = rank == 1 ? anchor
                : Add(added, shapes, OpCodes.RESHAPE, [], [anchor, constants.Vector(-1L)], flatInfo);
            slices.Insert(0, flatAnchor);
            var joined = Add(added, shapes, OpCodes.CONCAT, new() { [OnnxOpAttributeNames.AttrAxis] = 0L }, slices, flatInfo);
            var reshaped = rank == 1 ? joined
                : Add(added, shapes, OpCodes.RESHAPE, [], [joined, constants.Vector(anchorInfo.Shape.Dims)], anchorInfo);

            consumer.FullInputs[slot][index] = reshaped;
            (insertBefore.TryGetValue(consumer, out var before) ? before : insertBefore[consumer] = []).AddRange(added);
            dag.Added(added);
            ordered = true;
        }
        if (!ordered) return (graph, shapeInfo);

        // The new nodes go just before the consumer they feed, and the body is then sorted again,
        // keeping every node's place where its producers allow it: a reader the ordering hangs the
        // consumer on may come after it in the order it came in.
        var body = new List<FastNode>(nodes.Count + insertBefore.Values.Sum(l => l.Count));
        foreach (var node in nodes.Take(copy.BodyEnd))
        {
            if (insertBefore.TryGetValue(node, out var before)) body.AddRange(before);
            body.Add(node);
        }
        body.InsertRange(copy.InputCount, constants.Nodes);
        copy.Nodes = [.. StableTopologicalOrder(body, copy.InputCount), .. nodes.Skip(copy.BodyEnd)];
        copy.MoveOutputsToEnd();
        return (copy, new ShapeInferenceResult(shapes.ToImmutable()));
    }

    /// <summary>
    /// The anchor for <paramref name="writer"/>: walking back from it, breadth first, along the
    /// edges that order execution and among the nodes none of <paramref name="readers"/> depends
    /// on, the first input of a standard ONNX node that <see cref="Anchorable"/> admits — one no
    /// constant writes if there is one — other than <paramref name="state"/> itself. Null where
    /// there is none.
    /// </summary>
    private static (FastNode Consumer, string Slot, int Index, FastTensorKey Anchor, TensorShapeInfo Info)? Anchor(
        Dag dag, FastNode writer, IReadOnlyList<FastNode> readers, FastTensorKey state,
        IDictionary<FastTensorKey, TensorShapeInfo> shapes)
    {
        var excluded = dag.AncestorsOf(readers);
        (FastNode, string, int, FastTensorKey, TensorShapeInfo)? constant = null;
        var seen = new HashSet<FastNode> { writer };
        var pending = new Queue<FastNode>();
        pending.Enqueue(writer);
        while (pending.TryDequeue(out var node))
        {
            if (Dag.ReadsOnlyAShape(node)) continue;
            if (node.TargetFunction is null && Definitions.VanillaOpNames.Contains(node.OpCode))
                foreach (var (slot, keys) in node.FullInputs.OrderBy(s => s.Key, System.StringComparer.Ordinal))
                    for (int i = 0; i < keys.Count; i++)
                    {
                        if (keys[i] is not { } key || key.Equals(state)) continue;
                        if (!shapes.TryGetValue(key, out var info) || !Anchorable(info)) continue;
                        if (!dag.WrittenByConstant(key)) return (node, slot, i, key, info);
                        constant ??= (node, slot, i, key, info);
                    }
            foreach (var producer in dag.Producers(node))
                if (!excluded.Contains(producer) && seen.Add(producer)) pending.Enqueue(producer);
        }
        return constant;
    }

    /// <summary>Whether a value can anchor: a floating-point value — never a shape, an index or
    /// a bound, which a kernel may read on the host or a backend fold at load — of at least one
    /// and at most <see cref="MaxAnchorElements"/> elements, its shape known.</summary>
    private static bool Anchorable(TensorShapeInfo info)
        => info.ElementCount is >= 1 and <= MaxAnchorElements && info.Shape.Dims.All(d => d >= 0)
           && FloatingPointTypes.Contains(info.DType);

    private static readonly HashSet<DType> FloatingPointTypes = [DType.Float16, DType.BFloat16, DType.Float32, DType.Float64];

    private static readonly HashSet<DType> CastableTypes =
    [
        DType.Float16, DType.BFloat16, DType.Float32, DType.Float64,
        DType.Int8, DType.Int16, DType.Int32, DType.Int64,
        DType.UInt8, DType.UInt16, DType.UInt32, DType.UInt64, DType.Bool,
    ];

    /// <summary>An output of <paramref name="reader"/> an empty slice can be taken of and cast:
    /// the first whose shape is known and whose type <c>Cast</c> takes.</summary>
    private static (FastTensorKey Output, TensorShapeInfo Info)? SliceableOutput(
        FastNode reader, IDictionary<FastTensorKey, TensorShapeInfo> shapes)
    {
        foreach (var output in reader.Outputs)
            if (output is { } key && shapes.TryGetValue(key, out var info)
                && info.Shape.Dims.All(d => d >= 0) && CastableTypes.Contains(info.DType))
                return (key, info);
        return null;
    }

    /// <summary>Appends to <paramref name="added"/> a one-output node of <paramref name="opCode"/>
    /// over <paramref name="inputs"/>, recording its output's shape; its output.</summary>
    private static FastTensorKey Add(
        List<FastNode> added, IDictionary<FastTensorKey, TensorShapeInfo> shapes, string opCode,
        Dictionary<string, object?> attributes, List<FastTensorKey?> inputs, TensorShapeInfo info)
    {
        var node = NewNode(opCode, attributes, inputs);
        added.Add(node);
        var output = node.FullOutputs[""][0]!.Value;
        shapes[output] = info;
        return output;
    }

    private static FastNode NewNode(string opCode, Dictionary<string, object?> attributes, List<FastTensorKey?> inputs)
    {
        var key = FastNodeKey.New();
        return new FastNode
        {
            Key = key,
            OpCode = opCode,
            Attributes = OnnxCSharpAttributes.FromCSharpVals(attributes, Definitions.NodeDefinitions[opCode].AttributeDefs),
            FullInputs = { [""] = inputs },
            FullOutputs = { [""] = [new FastTensorKey(key, 0)] },
        };
    }

    /// <summary>
    /// <paramref name="body"/> in an order where every node follows its producers, each node as
    /// early as its place in <paramref name="body"/> and its producers allow: the first
    /// <paramref name="fixedPrefix"/> nodes — the graph's inputs — stay first.
    /// </summary>
    private static List<FastNode> StableTopologicalOrder(List<FastNode> body, int fixedPrefix)
    {
        var producer = new Dictionary<FastTensorKey, int>();
        for (int i = 0; i < body.Count; i++)
            foreach (var output in body[i].Outputs)
                if (output is { } key) producer[key] = i;
        var waiting = new int[body.Count];
        var consumers = new List<int>[body.Count];
        for (int i = 0; i < body.Count; i++) consumers[i] = [];
        for (int i = 0; i < body.Count; i++)
        {
            foreach (var input in body[i].Inputs.Distinct())
                if (input is { } key && producer.TryGetValue(key, out var p) && p != i)
                {
                    waiting[i]++;
                    consumers[p].Add(i);
                }
        }
        var order = new List<FastNode>(body.Count);
        var ready = new PriorityQueue<int, int>();
        for (int i = 0; i < body.Count; i++)
            if (waiting[i] == 0) ready.Enqueue(i, i < fixedPrefix ? i - fixedPrefix : i);
        while (ready.TryDequeue(out var next, out _))
        {
            order.Add(body[next]);
            foreach (var c in consumers[next])
                if (--waiting[c] == 0) ready.Enqueue(c, c);
        }
        return order;
    }

    /// <summary>
    /// The graph's dependencies as the ordering reads and extends them: each node's producers, by
    /// the edges that order execution whatever a backend folds, and by every edge.
    /// </summary>
    private sealed class Dag
    {
        private readonly Dictionary<FastTensorKey, FastNode> _producer = [];

        public Dag(IEnumerable<FastNode> nodes)
        {
            foreach (var node in nodes) Record(node);
        }

        private void Record(FastNode node)
        {
            foreach (var output in node.Outputs)
                if (output is { } key) _producer[key] = node;
        }

        /// <summary>Records the nodes the ordering <paramref name="added"/>.</summary>
        public void Added(IEnumerable<FastNode> added)
        {
            foreach (var node in added) Record(node);
        }

        /// <summary>Whether a <c>Constant</c> writes <paramref name="key"/>.</summary>
        public bool WrittenByConstant(FastTensorKey key)
            => _producer.TryGetValue(key, out var p) && p.OpCode == OpCodes.CONSTANT;

        /// <summary>The standard <c>Shape</c> and <c>Size</c>, which read no memory and whose
        /// input orders nothing once a backend knows the shape.</summary>
        public static bool ReadsOnlyAShape(FastNode node) => node.OpCode is OpCodes.SHAPE or OpCodes.SIZE;

        /// <summary>The nodes writing what <paramref name="node"/> reads.</summary>
        public IEnumerable<FastNode> Producers(FastNode node)
        {
            foreach (var input in node.Inputs)
                if (input is { } key && _producer.TryGetValue(key, out var p)) yield return p;
        }

        /// <summary><paramref name="from"/> and every node they depend on, by every edge.</summary>
        public HashSet<FastNode> AncestorsOf(IEnumerable<FastNode> from)
        {
            var seen = new HashSet<FastNode>();
            var pending = new Stack<FastNode>(from);
            while (pending.TryPop(out var node))
                if (seen.Add(node))
                    foreach (var p in Producers(node)) pending.Push(p);
            return seen;
        }

        /// <summary>
        /// Of <paramref name="readers"/>, those not already ancestors of <paramref name="writer"/>
        /// by the edges that order execution, less those that are such ancestors of another of them
        /// — ordering that one orders them too. Null where one of them depends on the writer, by
        /// any edge: that reader needs the update, and cannot run before it.
        /// </summary>
        public List<FastNode>? Unordered(FastNode writer, IEnumerable<FastNode> readers)
        {
            var candidates = readers.Distinct().ToList();
            if (AncestorsOf(candidates).Contains(writer)) return null;
            var ordered = OrderingAncestors([writer]);
            candidates.RemoveAll(ordered.Contains);
            var covered = OrderingAncestors(candidates.SelectMany(Producers));
            return [.. candidates.Where(r => !covered.Contains(r))];
        }

        /// <summary><paramref name="from"/> and every node they depend on by the edges that order
        /// execution: none through the input of a <c>Shape</c> or <c>Size</c>.</summary>
        private HashSet<FastNode> OrderingAncestors(IEnumerable<FastNode> from)
        {
            var seen = new HashSet<FastNode>();
            var pending = new Stack<FastNode>(from);
            while (pending.TryPop(out var node))
                if (seen.Add(node) && !ReadsOnlyAShape(node))
                    foreach (var p in Producers(node)) pending.Push(p);
            return seen;
        }
    }

    /// <summary>The <c>int64</c> vectors the ordering reads — slice bounds and reshape targets —
    /// one <c>Constant</c> each, shared by every pair it orders.</summary>
    private sealed class Constants(IDictionary<FastTensorKey, TensorShapeInfo> shapes)
    {
        private readonly Dictionary<string, FastTensorKey> _made = [];

        public List<FastNode> Nodes { get; } = [];

        public FastTensorKey Vector(params long[] values)
        {
            var name = string.Join(",", values);
            if (_made.TryGetValue(name, out var made)) return made;
            var bytes = new byte[values.Length * sizeof(long)];
            System.Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
            var data = TensorAttribute.Create(new Shape((long)values.Length), DType.Int64, bytes);
            var node = NewNode(OpCodes.CONSTANT, new() { [OnnxOpAttributeNames.AttrValue] = data }, []);
            node.FullInputs.Clear();
            Nodes.Add(node);
            var key = node.FullOutputs[""][0]!.Value;
            shapes[key] = new TensorShapeInfo(new Shape((long)values.Length), DType.Int64, data);
            return _made[name] = key;
        }
    }
}

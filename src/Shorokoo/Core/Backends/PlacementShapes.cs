using Shorokoo.Core.Factory.IR;

namespace Shorokoo.Core.Backends;

/// <summary>
/// What a placement needs to know of an ONNX graph's values for one run: each value's shape and
/// element type, evaluated from the shapes the run's inputs come in, and the contents of the small
/// integer values a graph computes its shapes from — the output of a <c>Shape</c>, the half of a
/// dimension a <c>Div</c> makes of it, a slice's bounds.
///
/// <para>It knows the operators shapes are made of and the common ones values are made of. A value
/// whose operator it does not know takes the shape the graph states for it, where that shape is
/// fully known once the inputs' named dimensions are; otherwise it is unknown, and a placement
/// never involves a value whose shape is unknown. Nothing here guesses: an attribute or input it
/// cannot read leaves the value unknown.</para>
/// </summary>
internal static class PlacementShapes
{
    /// <summary>The largest integer value whose contents are carried along (<see cref="Value.Ints"/>).</summary>
    internal const int SmallInts = 64;

    /// <summary>A value's shape, its ONNX element type, and its contents where it is a small integer
    /// value whose contents are known.</summary>
    internal sealed record Value(long[] Shape, int ElementType, long[]? Ints)
    {
        internal long Elements => Shape.Aggregate(1L, (a, d) => a * d);

        /// <summary>The value's size in bytes, or -1 for an element type of no fixed width.</summary>
        internal long Bytes => ElementBytes(ElementType) is var size and > 0 ? Elements * size : -1;
    }

    /// <summary>The width of an element of ONNX element type <paramref name="type"/> in bytes; zero
    /// for one of no fixed whole-byte width (strings, four-bit types).</summary>
    internal static int ElementBytes(int type) => type switch
    {
        1 or 6 or 12 => 4,
        2 or 3 or 9 => 1,
        4 or 5 or 10 or 16 => 2,
        7 or 11 or 13 or 14 => 8,
        15 => 16,
        >= 17 and <= 20 => 1,
        _ => 0,
    };

    private const int Float = 1, Int32 = 6, Int64 = 7, Bool = 9, UInt32 = 12, UInt64 = 13;

    /// <summary>Whether values of ONNX element type <paramref name="type"/> carry small integer
    /// contents here: the integer types and bool. An unsigned value is carried only while it reads
    /// the same as a signed one.</summary>
    private static bool Integral(int type) => type is 2 or 3 or 4 or 5 or Int32 or Int64 or Bool or UInt32 or UInt64;

    /// <summary>
    /// Every value of <paramref name="graph"/> whose shape follows from <paramref name="inputs"/> —
    /// the shape and ONNX element type each input comes in, by name — by name.
    /// </summary>
    internal static Dictionary<string, Value> Evaluate(
        GraphProto graph, IReadOnlyDictionary<string, (long[] Shape, int ElementType)> inputs)
    {
        var values = new Dictionary<string, Value>(StringComparer.Ordinal);
        var symbols = new Dictionary<string, long>(StringComparer.Ordinal);
        foreach (var initializer in graph.Initializers)
            if (FromTensor(initializer) is { } value) values[initializer.Name] = value;
        foreach (var input in graph.Inputs)
        {
            if (inputs.TryGetValue(input.Name, out var given))
            {
                values[input.Name] = new Value(given.Shape, given.ElementType, null);
                if (input.Type?.TensorType?.Shape is { } declared && declared.Dims.Count == given.Shape.Length)
                    for (int i = 0; i < given.Shape.Length; i++)
                        if (declared.Dims[i].DimParam is { Length: > 0 } name) symbols.TryAdd(name, given.Shape[i]);
            }
            else if (!values.ContainsKey(input.Name) && Stated(input, symbols) is { } stated)
                values[input.Name] = stated;
        }

        var statedTypes = new Dictionary<string, ValueInfoProto>(StringComparer.Ordinal);
        foreach (var info in graph.ValueInfoes.Concat(graph.Outputs)) statedTypes.TryAdd(info.Name, info);

        foreach (var node in graph.Nodes)
        {
            Value?[] read = [.. node.Inputs.Select(name => name.Length > 0 && values.TryGetValue(name, out var v) ? v : null)];
            Value?[] made;
            try
            {
                made = Infer(node, read);
            }
            catch (Exception exception) when (exception is ArgumentException or InvalidOperationException
                                                  or IndexOutOfRangeException or OverflowException or DivideByZeroException)
            {
                made = [];
            }
            for (int o = 0; o < node.Outputs.Count; o++)
            {
                var name = node.Outputs[o];
                if (name.Length == 0) continue;
                var value = o < made.Length ? made[o] : null;
                if (value is null && statedTypes.TryGetValue(name, out var info)) value = Stated(info, symbols);
                if (value is not null && value.Shape.All(d => d >= 0)) values[name] = value;
            }
        }
        return values;
    }

    /// <summary>The shape <paramref name="info"/> states, where every dimension is a number or a
    /// name an input bound; null otherwise.</summary>
    private static Value? Stated(ValueInfoProto info, Dictionary<string, long> symbols)
    {
        if (info.Type?.TensorType is not { } tensor || tensor.Shape is not { } shape || tensor.ElemType == 0) return null;
        var dims = new long[shape.Dims.Count];
        for (int i = 0; i < dims.Length; i++)
        {
            var dim = shape.Dims[i];
            if (dim.DimParam is { Length: > 0 } name)
            {
                if (!symbols.TryGetValue(name, out dims[i])) return null;
            }
            else if (dim.ShouldSerializeDimValue() && dim.DimValue >= 0) dims[i] = dim.DimValue;
            else return null;
        }
        return new Value(dims, tensor.ElemType, null);
    }

    /// <summary>An initializer or a constant tensor as a value, with its contents where it is a
    /// small integer one.</summary>
    internal static Value? FromTensor(TensorProto tensor)
    {
        var dims = tensor.Dims ?? [];
        var count = dims.Aggregate(1L, (a, d) => a * d);
        long[]? ints = null;
        var type = tensor.data_type;
        if (count <= SmallInts && Integral(type))
        {
            var size = ElementBytes(type);
            if (type == Int64 && tensor.Int64Datas is { Length: > 0 } l && l.Length == count) ints = [.. l];
            else if (type is UInt64 or UInt32 && tensor.Uint64Datas is { Length: > 0 } u && u.Length == count)
                ints = u.All(v => v <= long.MaxValue) ? [.. u.Select(v => (long)v)] : null;
            else if (type is not (Int64 or UInt64 or UInt32) && tensor.Int32Datas is { Length: > 0 } i32 && i32.Length == count) ints = [.. i32.Select(v => (long)v)];
            else if (tensor.RawData is { } raw && raw.Length == count * size)
                ints = [.. Enumerable.Range(0, (int)count).Select(k => size switch
                {
                    8 => BitConverter.ToInt64(raw, k * 8),
                    4 => type == UInt32 ? BitConverter.ToUInt32(raw, k * 4) : BitConverter.ToInt32(raw, k * 4),
                    2 => type == 4 ? BitConverter.ToUInt16(raw, k * 2) : BitConverter.ToInt16(raw, k * 2),
                    _ => type is 2 or Bool ? raw[k] : (sbyte)raw[k],
                })];
            else if (count == 0) ints = [];
            if (type == UInt64 && ints is not null && ints.Any(v => v < 0)) ints = null;
        }
        return new Value([.. dims], type, ints);
    }

    // ---- operators ----

    private static readonly HashSet<string> SameShape = new(StringComparer.Ordinal)
    {
        "Abs", "Neg", "Sigmoid", "Relu", "Exp", "Log", "Sqrt", "Tanh", "Sin", "Cos", "Tan", "Asin", "Acos",
        "Atan", "Sinh", "Cosh", "Asinh", "Acosh", "Atanh", "Reciprocal", "Floor", "Ceil", "Round", "Sign",
        "Erf", "Softplus", "Softsign", "Elu", "Selu", "LeakyRelu", "ThresholdedRelu", "HardSigmoid",
        "HardSwish", "Celu", "Mish", "Gelu", "FastGelu", "QuickGelu", "BiasGelu", "Clip", "Identity", "Softmax",
        "LogSoftmax", "Hardmax", "Not", "BitwiseNot", "Shrink", "LpNormalization", "CumSum", "Trilu",
        "InstanceNormalization", "LayerNormalization", "SimplifiedLayerNormalization",
        "SkipLayerNormalization", "SkipSimplifiedLayerNormalization", "BatchNormalization", "Dropout",
        "BiasDropout", "SoftmaxCrossEntropyLossGrad", "ReluGrad", "SigmoidGrad", "TanhGrad", "GeluGrad",
        "FastGeluGrad", "SoftmaxGrad", "SoftmaxGrad_13", "LogSoftmaxGrad", "LogSoftmaxGrad_13",
    };

    private static readonly HashSet<string> Broadcasting = new(StringComparer.Ordinal)
    {
        "Add", "Sub", "Mul", "Div", "Pow", "Mod", "Max", "Min", "Sum", "Mean", "BitShift", "BitwiseAnd",
        "BitwiseOr", "BitwiseXor", "PRelu", "And", "Or", "Xor", "Equal", "Less", "Greater", "LessOrEqual",
        "GreaterOrEqual", "Where",
    };

    private static readonly HashSet<string> Comparing = new(StringComparer.Ordinal)
    {
        "And", "Or", "Xor", "Equal", "Less", "Greater", "LessOrEqual", "GreaterOrEqual",
    };

    private static readonly HashSet<string> Reducing = new(StringComparer.Ordinal)
    {
        "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd", "ReduceL1", "ReduceL2",
        "ReduceSumSquare", "ReduceLogSum", "ReduceLogSumExp",
    };

    private static Value?[] Infer(NodeProto node, Value?[] read)
    {
        var op = node.OpType;
        Value? In(int i) => i < read.Length ? read[i] : null;
        var x = In(0);

        if (op == "Constant")
            return [ConstantOf(node)];
        if (op == "Shape")
        {
            if (x is null) return [];
            var rank = x.Shape.Length;
            var start = Clamp(Attr(node, "start") ?? 0, rank);
            var end = Clamp(Attr(node, "end") ?? rank, rank);
            long[] dims = end > start ? x.Shape[(int)start..(int)end] : [];
            return [new Value([dims.Length], Int64, dims)];
        }
        if (op == "Size")
            return x is null ? [] : [new Value([], Int64, [x.Elements])];
        if (SameShape.Contains(op))
        {
            if (x is null) return [];
            var type = op is "Not" ? Bool : x.ElementType;
            var ints = op switch
            {
                "Neg" when x.Ints is { } v => v.Select(e => -e).ToArray(),
                "Abs" when x.Ints is { } v => v.Select(Math.Abs).ToArray(),
                "Not" when x.Ints is { } v => v.Select(e => e == 0 ? 1L : 0L).ToArray(),
                "Identity" => x.Ints,
                _ => null,
            };
            var first = new Value(x.Shape, type, ints);
            return op switch
            {
                "Dropout" or "BiasDropout" => [first, new Value(x.Shape, Bool, null)],
                _ => [first],
            };
        }
        if (op is "IsNaN" or "IsInf")
            return x is null ? [] : [new Value(x.Shape, Bool, null)];
        if (Broadcasting.Contains(op))
        {
            if (read.Length == 0 || read.Any(v => v is null)) return [];
            var shape = read[0]!.Shape;
            for (int i = 1; i < read.Length; i++) shape = Broadcast(shape, read[i]!.Shape);
            var type = Comparing.Contains(op) ? Bool : op == "Where" ? read[1]!.ElementType : read[0]!.ElementType;
            return [new Value(shape, type, op == "Where" ? IntWhere(read) : IntArithmetic(op, read))];
        }
        if (op == "Cast")
            return x is null || Attr(node, "to") is not { } to ? []
                : [new Value(x.Shape, (int)to, Integral((int)to) && Integral(x.ElementType) && (x.Ints?.All(v => v >= 0) == true || to is Int64 or Int32)
                    ? (to == Bool ? x.Ints?.Select(v => v != 0 ? 1L : 0L).ToArray() : x.Ints)
                    : null)];
        if (op == "CastLike")
            return x is null || In(1) is not { } like ? [] : [new Value(x.Shape, like.ElementType, null)];
        if (op == "ConstantOfShape")
        {
            if (x?.Ints is not { } dims) return [];
            var value = node.Attributes.FirstOrDefault(a => a.Name == "value")?.T;
            var type = value?.data_type ?? Float;
            var count = dims.Aggregate(1L, (a, d) => a * d);
            long[]? ints = null;
            if (type is Int64 or Int32 && count <= SmallInts && value is not null && FromTensor(value)?.Ints is [var fill])
                ints = [.. Enumerable.Repeat(fill, (int)count)];
            return [new Value(dims, type, ints)];
        }
        if (op == "Unsqueeze")
        {
            if (x is null || (Ints(node, "axes") ?? In(1)?.Ints) is not { } axes) return [];
            var rank = x.Shape.Length + axes.Length;
            var normalized = axes.Select(a => a < 0 ? a + rank : a).ToHashSet();
            var dims = new List<long>();
            var next = 0;
            for (int d = 0; d < rank; d++) dims.Add(normalized.Contains(d) ? 1 : x.Shape[next++]);
            return [new Value([.. dims], x.ElementType, x.Ints)];
        }
        if (op == "Squeeze")
        {
            if (x is null) return [];
            var axes = Ints(node, "axes") ?? In(1)?.Ints;
            var rank = x.Shape.Length;
            var drop = axes is null
                ? Enumerable.Range(0, rank).Where(d => x.Shape[d] == 1).ToHashSet()
                : axes.Select(a => (int)(a < 0 ? a + rank : a)).ToHashSet();
            return [new Value([.. x.Shape.Where((_, d) => !drop.Contains(d))], x.ElementType, x.Ints)];
        }
        if (op == "Concat")
        {
            if (read.Any(v => v is null) || read.Length == 0 || Attr(node, "axis") is not { } axisAttr) return [];
            var rank = read[0]!.Shape.Length;
            var axis = (int)(axisAttr < 0 ? axisAttr + rank : axisAttr);
            var dims = (long[])read[0]!.Shape.Clone();
            dims[axis] = read.Sum(v => v!.Shape[axis]);
            long[]? ints = read.All(v => v!.Ints is not null) && dims.Aggregate(1L, (a, d) => a * d) <= SmallInts && rank == 1
                ? [.. read.SelectMany(v => v!.Ints!)]
                : null;
            return [new Value(dims, read[0]!.ElementType, ints)];
        }
        if (op == "Slice")
        {
            if (x is null || SliceOf(node, read) is not { } slice) return [];
            return [new Value(slice.Shape, x.ElementType, slice.Ints)];
        }
        if (op == "Gather")
        {
            if (x is null || In(1) is not { } indices) return [];
            var rank = x.Shape.Length;
            var axisAttr = Attr(node, "axis") ?? 0;
            var axis = (int)(axisAttr < 0 ? axisAttr + rank : axisAttr);
            long[] dims = [.. x.Shape[..axis], .. indices.Shape, .. x.Shape[(axis + 1)..]];
            long[]? ints = null;
            if (rank == 1 && x.Ints is { } data && indices.Ints is { } at)
                ints = [.. at.Select(i => data[(int)(i < 0 ? i + data.Length : i)])];
            return [new Value(dims, x.ElementType, ints)];
        }
        if (op == "Reshape")
        {
            if (x is null || In(1)?.Ints is not { } target) return [];
            var allowZero = (Attr(node, "allowzero") ?? 0) != 0;
            var dims = new long[target.Length];
            var infer = -1;
            for (int d = 0; d < dims.Length; d++)
            {
                if (target[d] == 0 && !allowZero) dims[d] = x.Shape[d];
                else if (target[d] == -1) { infer = d; dims[d] = 1; }
                else dims[d] = target[d];
            }
            if (infer >= 0)
            {
                var known = dims.Aggregate(1L, (a, d) => a * d);
                dims[infer] = known == 0 ? 0 : x.Elements / known;
            }
            return dims.Aggregate(1L, (a, d) => a * d) == x.Elements ? [new Value(dims, x.ElementType, x.Ints)] : [];
        }
        if (op == "Flatten")
        {
            if (x is null) return [];
            var rank = x.Shape.Length;
            var axisAttr = Attr(node, "axis") ?? 1;
            var axis = (int)(axisAttr < 0 ? axisAttr + rank : axisAttr);
            var outer = x.Shape[..axis].Aggregate(1L, (a, d) => a * d);
            return [new Value([outer, x.Elements / Math.Max(outer, 1)], x.ElementType, null)];
        }
        if (op == "Expand")
        {
            if (x is null || In(1)?.Ints is not { } target) return [];
            return [new Value(Broadcast(x.Shape, target), x.ElementType, null)];
        }
        if (op == "Transpose")
        {
            if (x is null) return [];
            var perm = Ints(node, "perm") ?? [.. Enumerable.Range(0, x.Shape.Length).Reverse().Select(d => (long)d)];
            return [new Value([.. perm.Select(p => x.Shape[(int)p])], x.ElementType, null)];
        }
        if (op is "MatMul" or "FusedMatMul")
        {
            if (x is null || In(1) is not { } y || x.Shape.Length == 0 || y.Shape.Length == 0) return [];
            var a = x.Shape.Length == 1 ? [1, x.Shape[0]] : x.Shape;
            var b = y.Shape.Length == 1 ? [y.Shape[0], 1] : y.Shape;
            if (op == "FusedMatMul")
            {
                if ((Attr(node, "transA") ?? 0) != 0) a = [.. a[..^2], a[^1], a[^2]];
                if ((Attr(node, "transB") ?? 0) != 0) b = [.. b[..^2], b[^1], b[^2]];
            }
            var batch = Broadcast(a[..^2], b[..^2]);
            var dims = new List<long>(batch);
            if (x.Shape.Length > 1) dims.Add(a[^2]);
            if (y.Shape.Length > 1) dims.Add(b[^1]);
            return [new Value([.. dims], x.ElementType, null)];
        }
        if (op == "Gemm")
        {
            if (x is null || In(1) is not { } y || x.Shape.Length != 2 || y.Shape.Length != 2) return [];
            var m = (Attr(node, "transA") ?? 0) != 0 ? x.Shape[1] : x.Shape[0];
            var n = (Attr(node, "transB") ?? 0) != 0 ? y.Shape[0] : y.Shape[1];
            return [new Value([m, n], x.ElementType, null)];
        }
        if (Reducing.Contains(op))
        {
            if (x is null) return [];
            var axes = Ints(node, "axes") ?? In(1)?.Ints;
            if (axes is null && read.Length > 1 && !string.IsNullOrEmpty(node.Inputs[1])) return [];
            var keep = (Attr(node, "keepdims") ?? 1) != 0;
            var rank = x.Shape.Length;
            var all = axes is null || axes.Length == 0;
            if (all && (Attr(node, "noop_with_empty_axes") ?? 0) != 0) return [x];
            var reduced = all ? Enumerable.Range(0, rank).ToHashSet() : axes!.Select(a => (int)(a < 0 ? a + rank : a)).ToHashSet();
            var dims = new List<long>();
            for (int d = 0; d < rank; d++)
                if (!reduced.Contains(d)) dims.Add(x.Shape[d]);
                else if (keep) dims.Add(1);
            long[]? ints = rank == 1 && reduced.Contains(0) && x.Ints is { Length: > 0 } v ? op switch
            {
                "ReduceSum" => [v.Sum()],
                "ReduceProd" => [v.Aggregate(1L, (a, b) => a * b)],
                "ReduceMax" => [v.Max()],
                "ReduceMin" => [v.Min()],
                _ => null,
            } : null;
            return [new Value([.. dims], x.ElementType, ints)];
        }
        if (op is "ArgMax" or "ArgMin")
        {
            if (x is null) return [];
            var rank = x.Shape.Length;
            var axisAttr = Attr(node, "axis") ?? 0;
            var axis = (int)(axisAttr < 0 ? axisAttr + rank : axisAttr);
            var keep = (Attr(node, "keepdims") ?? 1) != 0;
            var dims = x.Shape.Select((d, i) => i == axis ? (keep ? 1L : -1L) : d).Where(d => d >= 0).ToArray();
            return [new Value(dims, Int64, null)];
        }
        if (op == "Split")
        {
            if (x is null) return [];
            var rank = x.Shape.Length;
            var axisAttr = Attr(node, "axis") ?? 0;
            var axis = (int)(axisAttr < 0 ? axisAttr + rank : axisAttr);
            var sizes = Ints(node, "split") ?? In(1)?.Ints;
            var count = node.Outputs.Count;
            if (sizes is null)
            {
                var length = x.Shape[axis];
                var each = (length + count - 1) / count;
                sizes = [.. Enumerable.Range(0, count).Select(k => Math.Max(0, Math.Min(each, length - k * each)))];
            }
            return [.. sizes.Select(size =>
            {
                var dims = (long[])x.Shape.Clone();
                dims[axis] = size;
                return (Value?)new Value(dims, x.ElementType, null);
            })];
        }
        if (op == "Range")
        {
            if (x?.Ints is not [var start] || In(1)?.Ints is not [var limit] || In(2)?.Ints is not [var delta] || delta == 0) return [];
            var count = Math.Max(delta > 0 ? (limit - start + delta - 1) / delta : (start - limit - delta - 1) / -delta, 0);
            long[]? ints = count <= SmallInts ? [.. Enumerable.Range(0, (int)count).Select(k => start + k * delta)] : null;
            return [new Value([count], x.ElementType, ints)];
        }
        if (op is "Conv" or "MaxPool" or "AveragePool" or "LpPool")
        {
            if (x is null || x.Shape.Length < 3) return [];
            var spatial = x.Shape.Length - 2;
            long channels;
            long[] kernel;
            if (op == "Conv")
            {
                if (In(1) is not { } w || w.Shape.Length != x.Shape.Length) return [];
                channels = w.Shape[0];
                kernel = Ints(node, "kernel_shape") ?? w.Shape[2..];
            }
            else
            {
                channels = x.Shape[1];
                if (Ints(node, "kernel_shape") is not { } k) return [];
                kernel = k;
            }
            if (kernel.Length != spatial) return [];
            var strides = Ints(node, "strides") is { Length: > 0 } st ? st : Enumerable.Repeat(1L, spatial).ToArray();
            var dilations = Ints(node, "dilations") is { Length: > 0 } dl ? dl : Enumerable.Repeat(1L, spatial).ToArray();
            var pads = Ints(node, "pads") is { Length: > 0 } pd ? pd : new long[2 * spatial];
            var autoPad = node.Attributes.FirstOrDefault(a => a.Name == "auto_pad")?.S is { Length: > 0 } bytes
                ? System.Text.Encoding.UTF8.GetString(bytes) : "NOTSET";
            var ceil = (Attr(node, "ceil_mode") ?? 0) != 0;
            if (strides.Length != spatial || dilations.Length != spatial || pads.Length != 2 * spatial) return [];
            var dims = new List<long> { x.Shape[0], channels };
            for (int d = 0; d < spatial; d++)
            {
                var size = x.Shape[d + 2];
                var extent = dilations[d] * (kernel[d] - 1) + 1;
                long outSize = autoPad switch
                {
                    "SAME_UPPER" or "SAME_LOWER" => (size + strides[d] - 1) / strides[d],
                    "VALID" => (size - extent) / strides[d] + 1,
                    "NOTSET" => ceil
                        ? (size + pads[d] + pads[d + spatial] - extent + strides[d] - 1) / strides[d] + 1
                        : (size + pads[d] + pads[d + spatial] - extent) / strides[d] + 1,
                    _ => -1,
                };
                if (outSize < 0) return [];
                dims.Add(outSize);
            }
            var output = new Value([.. dims], x.ElementType, null);
            return op == "MaxPool" ? [output, new Value([.. dims], Int64, null)] : [output];
        }
        if (op is "GlobalAveragePool" or "GlobalMaxPool" or "GlobalLpPool")
            return x is null || x.Shape.Length < 3 ? [] : [new Value([x.Shape[0], x.Shape[1], .. Enumerable.Repeat(1L, x.Shape.Length - 2)], x.ElementType, null)];
        if (op is "LSTM" or "GRU" or "RNN")
        {
            if (x is null || x.Shape.Length != 3 || Attr(node, "hidden_size") is not { } hidden) return [];
            var layout = Attr(node, "layout") ?? 0;
            var direction = node.Attributes.FirstOrDefault(a => a.Name == "direction")?.S is { Length: > 0 } d
                ? System.Text.Encoding.UTF8.GetString(d) : "forward";
            long directions = direction == "bidirectional" ? 2 : 1;
            var (sequence, batch) = layout == 0 ? (x.Shape[0], x.Shape[1]) : (x.Shape[1], x.Shape[0]);
            var y = layout == 0
                ? new Value([sequence, directions, batch, hidden], x.ElementType, null)
                : new Value([batch, sequence, directions, hidden], x.ElementType, null);
            var last = layout == 0
                ? new Value([directions, batch, hidden], x.ElementType, null)
                : new Value([batch, directions, hidden], x.ElementType, null);
            return op == "LSTM" ? [y, last, last] : [y, last];
        }
        if (op == "Tile")
        {
            if (x is null || In(1)?.Ints is not { } repeats || repeats.Length != x.Shape.Length) return [];
            return [new Value([.. x.Shape.Select((d, i) => d * repeats[i])], x.ElementType, null)];
        }
        return [];
    }

    /// <summary>
    /// What a <c>Slice</c> node takes of its data input: per axis, the first element, the step and
    /// the count, and the shape and small contents of what it makes. Null where a bound cannot be
    /// read.
    /// </summary>
    internal static SliceTaken? SliceOf(NodeProto node, Value?[] read)
    {
        if (read.Length == 0 || read[0] is not { } x) return null;
        var rank = x.Shape.Length;
        long[]? starts, ends, axes, steps;
        if (read.Length > 1 || node.Inputs.Count > 1)
        {
            starts = Read(read, 1);
            ends = Read(read, 2);
            axes = node.Inputs.Count > 3 && node.Inputs[3].Length > 0 ? Read(read, 3) : [.. Enumerable.Range(0, starts?.Length ?? 0).Select(a => (long)a)];
            steps = node.Inputs.Count > 4 && node.Inputs[4].Length > 0 ? Read(read, 4) : [.. Enumerable.Repeat(1L, starts?.Length ?? 0)];
        }
        else
        {
            starts = Ints(node, "starts");
            ends = Ints(node, "ends");
            axes = Ints(node, "axes") ?? [.. Enumerable.Range(0, starts?.Length ?? 0).Select(a => (long)a)];
            steps = [.. Enumerable.Repeat(1L, starts?.Length ?? 0)];
        }
        if (starts is null || ends is null || axes is null || steps is null) return null;
        if (starts.Length != ends.Length || starts.Length != axes.Length || starts.Length != steps.Length) return null;

        var first = new long[rank];
        var step = Enumerable.Repeat(1L, rank).ToArray();
        var count = (long[])x.Shape.Clone();
        for (int k = 0; k < starts.Length; k++)
        {
            var axis = (int)(axes[k] < 0 ? axes[k] + rank : axes[k]);
            var dim = x.Shape[axis];
            var s = steps[k];
            if (s == 0) return null;
            long begin = starts[k] < 0 ? starts[k] + dim : starts[k];
            long end = ends[k] < 0 ? ends[k] + dim : ends[k];
            if (s > 0)
            {
                begin = Math.Clamp(begin, 0, dim);
                end = Math.Clamp(end, 0, dim);
                count[axis] = end > begin ? (end - begin + s - 1) / s : 0;
            }
            else
            {
                begin = Math.Clamp(begin, 0, dim - 1);
                end = Math.Clamp(end, -1, dim - 1);
                count[axis] = begin > end ? (begin - end - s - 1) / -s : 0;
            }
            first[axis] = begin;
            step[axis] = s;
        }
        long[]? ints = null;
        if (rank == 1 && x.Ints is { } data)
            ints = [.. Enumerable.Range(0, (int)count[0]).Select(k => data[(int)(first[0] + k * step[0])])];
        return new SliceTaken(first, step, count, ints);
    }

    /// <summary>What a <c>Slice</c> takes: per axis, the first element, the step, and the count —
    /// which is also the shape of what it makes — and that value's small contents.</summary>
    internal sealed record SliceTaken(long[] First, long[] Step, long[] Shape, long[]? Ints)
    {
        /// <summary>
        /// Where what the slice takes lies in its data, as one contiguous run of elements: the
        /// first element's offset and the count, both in elements; null where it is not one run —
        /// where a step is not one, or a dimension inside the outermost one sliced is not taken
        /// whole.
        /// </summary>
        internal (long First, long Count)? Contiguous(long[] dataShape)
        {
            if (Step.Any(s => s != 1)) return null;
            var rank = dataShape.Length;
            // The outermost axis taken partly; every axis inside it must be whole, every axis
            // outside it must take a single index.
            var outer = -1;
            for (int d = 0; d < rank; d++)
                if (Shape[d] != dataShape[d]) { outer = d; break; }
            if (outer < 0) return (0, Shape.Aggregate(1L, (a, b) => a * b));
            for (int d = outer + 1; d < rank; d++)
                if (Shape[d] != dataShape[d]) return null;
            for (int d = 0; d < outer; d++)
                if (Shape[d] != 1 && dataShape[d] != 1) return null;
            long offset = 0, stride = 1;
            for (int d = rank - 1; d >= 0; d--)
            {
                offset += First[d] * stride;
                stride *= dataShape[d];
            }
            return (offset, Shape.Aggregate(1L, (a, b) => a * b));
        }
    }

    // ---- helpers ----

    private static long[]? Read(Value?[] read, int index) => index < read.Length ? read[index]?.Ints : null;

    private static long Clamp(long index, int rank) => Math.Clamp(index < 0 ? index + rank : index, 0, rank);

    private static long? Attr(NodeProto node, string name)
        => node.Attributes.FirstOrDefault(a => a.Name == name) is { } a ? a.I : null;

    private static long[]? Ints(NodeProto node, string name)
        => node.Attributes.FirstOrDefault(a => a.Name == name) is { } a ? a.Ints ?? [] : null;

    private static Value? ConstantOf(NodeProto node)
    {
        foreach (var attribute in node.Attributes)
        {
            switch (attribute.Name)
            {
                case "value" when attribute.T is { } tensor: return FromTensor(tensor);
                case "value_int": return new Value([], Int64, [attribute.I]);
                case "value_ints": return new Value([attribute.Ints?.Length ?? 0], Int64, attribute.Ints ?? []);
                case "value_float": return new Value([], Float, null);
                case "value_floats": return new Value([attribute.Floats?.Length ?? 0], Float, null);
            }
        }
        return null;
    }

    /// <summary>The shape two shapes broadcast to, numpy-style.</summary>
    /// <exception cref="ArgumentException">They do not broadcast.</exception>
    internal static long[] Broadcast(long[] a, long[] b)
    {
        var rank = Math.Max(a.Length, b.Length);
        var dims = new long[rank];
        for (int i = 0; i < rank; i++)
        {
            var da = i < rank - a.Length ? 1 : a[i - (rank - a.Length)];
            var db = i < rank - b.Length ? 1 : b[i - (rank - b.Length)];
            if (da != db && da != 1 && db != 1) throw new ArgumentException("shapes do not broadcast");
            dims[i] = da == 1 ? db : da;
        }
        return dims;
    }

    /// <summary>The contents of a <c>Where</c> over small integer values whose contents are known,
    /// broadcasting a single value; null otherwise.</summary>
    private static long[]? IntWhere(Value?[] read)
    {
        if (read.Length != 3 || read[0]?.Ints is not { } c || read[1]?.Ints is not { } a || read[2]?.Ints is not { } b) return null;
        if (!Integral(read[1]!.ElementType)) return null;
        var n = Math.Max(c.Length, Math.Max(a.Length, b.Length));
        if ((c.Length != n && c.Length != 1) || (a.Length != n && a.Length != 1) || (b.Length != n && b.Length != 1)) return null;
        if (c.Length == 0 || a.Length == 0 || b.Length == 0) return [];
        var result = new long[n];
        for (int i = 0; i < n; i++)
            result[i] = c[c.Length == 1 ? 0 : i] != 0 ? a[a.Length == 1 ? 0 : i] : b[b.Length == 1 ? 0 : i];
        return result;
    }

    /// <summary>The contents of an integer operator over small integer values whose contents are
    /// known, broadcasting a single value; null otherwise.</summary>
    private static long[]? IntArithmetic(string op, Value?[] read)
    {
        if (read.Length != 2 || read[0]?.Ints is not { } a || read[1]?.Ints is not { } b) return null;
        if (!Integral(read[0]!.ElementType)) return null;
        if (a.Length != b.Length && a.Length != 1 && b.Length != 1) return null;
        var n = Math.Max(a.Length, b.Length);
        if (a.Length == 0 || b.Length == 0) return [];
        var result = new long[n];
        for (int i = 0; i < n; i++)
        {
            var l = a[a.Length == 1 ? 0 : i];
            var r = b[b.Length == 1 ? 0 : i];
            switch (op)
            {
                case "Add": result[i] = l + r; break;
                case "Sub": result[i] = l - r; break;
                case "Mul": result[i] = l * r; break;
                // Truncating, as ONNX Runtime's integer kernels divide.
                case "Div" when r != 0: result[i] = l / r; break;
                case "Mod" when r != 0: result[i] = l % r; break;
                case "Max": result[i] = Math.Max(l, r); break;
                case "Min": result[i] = Math.Min(l, r); break;
                case "Equal": result[i] = l == r ? 1 : 0; break;
                case "Less": result[i] = l < r ? 1 : 0; break;
                case "Greater": result[i] = l > r ? 1 : 0; break;
                case "LessOrEqual": result[i] = l <= r ? 1 : 0; break;
                case "GreaterOrEqual": result[i] = l >= r ? 1 : 0; break;
                case "And": result[i] = l != 0 && r != 0 ? 1 : 0; break;
                case "Or": result[i] = l != 0 || r != 0 ? 1 : 0; break;
                case "Xor": result[i] = (l != 0) != (r != 0) ? 1 : 0; break;
                default: return null;
            }
        }
        // An unsigned value that would wrap reads differently from a signed one: not carried.
        return read[0]!.ElementType is 2 or 4 or UInt32 or UInt64 && result.Any(v => v < 0) ? null : result;
    }
}

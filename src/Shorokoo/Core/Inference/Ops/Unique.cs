using System.Collections.Immutable;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// QEE kernel for ONNX <c>Unique</c>. Four outputs: (y, indices, inverse_indices, counts).
/// The unique-element count is data-dependent, so without concrete input values the
/// affected extents degrade to rank-only (+ MaxShape bounds) — never a guessed dim.
/// With concrete values and no <c>axis</c> (the flatten form) all four outputs are
/// computed exactly, honoring <c>sorted</c> (default 1; sorted=0 keeps first-occurrence
/// order, matching ORT). The axis form stays shape-only (unique-slice comparison is
/// rarely needed for constant folding).
/// </summary>
internal sealed class UniqueOp : QuickOp
{
    public override string OpCode => OpCodes.UNIQUE;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var dtype = x?.DType ?? DType.Float32;
        var axisAttr = attrs.GetLongVal(OnnxOpAttributeNames.AttrAxis);
        var sorted = attrs.GetAttributeObj(OnnxOpAttributeNames.AttrSorted) as bool? ?? true;

        if (x?.Shape is null)
        {
            var rankY = axisAttr is null ? 1 : x?.Rank;
            return [
                RuntimeTensorFactory.CreateRankOnly(dtype, rankY),
                RuntimeTensorFactory.CreateRankOnly(DType.Int64, 1),
                RuntimeTensorFactory.CreateRankOnly(DType.Int64, 1),
                RuntimeTensorFactory.CreateRankOnly(DType.Int64, 1),
            ];
        }

        var dims = x.Shape.Dims;
        long numel = 1;
        foreach (var d in dims) numel = d < 0 ? -1 : (numel < 0 ? -1 : numel * d);

        if (axisAttr is null)
        {
            // Flatten form. inverse_indices length equals the element count (exact when
            // known); y/indices/counts are bounded by it.
            var inverse = numel >= 0
                ? RuntimeTensorFactory.Create(DType.Int64, new Shape([numel]))
                : RuntimeTensorFactory.CreateRankOnly(DType.Int64, 1);

            if (numel >= 0 && RuntimeTensorFactory.ShouldStoreData(x.Shape, maxDataElements)
                && (x.FloatData is not null || x.IntData is not null))
            {
                return ComputeFlattenValues(x, dtype, sorted, (int)numel);
            }

            var bound = numel >= 0 ? new Shape([numel]) : null;
            return [
                RuntimeTensorFactory.CreateRankOnly(dtype, 1) with { MaxShape = bound },
                RuntimeTensorFactory.CreateRankOnly(DType.Int64, 1) with { MaxShape = bound },
                inverse,
                RuntimeTensorFactory.CreateRankOnly(DType.Int64, 1) with { MaxShape = bound },
            ];
        }

        // Axis form: y keeps the input rank with a data-dependent extent at `axis`;
        // inverse_indices is exactly [dims[axis]]; indices/counts are 1-D bounded by it.
        var rank = dims.Length;
        var axis = (int)(axisAttr.Value < 0 ? axisAttr.Value + rank : axisAttr.Value);
        if (axis < 0 || axis >= rank)
        {
            return [
                RuntimeTensorFactory.CreateRankOnly(dtype, rank),
                RuntimeTensorFactory.CreateRankOnly(DType.Int64, 1),
                RuntimeTensorFactory.CreateRankOnly(DType.Int64, 1),
                RuntimeTensorFactory.CreateRankOnly(DType.Int64, 1),
            ];
        }

        var axisLen = dims[axis];
        var axisBound = axisLen >= 0 ? new Shape([axisLen]) : null;
        return [
            RuntimeTensorFactory.CreateRankOnly(dtype, rank) with { MaxShape = x.Shape },
            RuntimeTensorFactory.CreateRankOnly(DType.Int64, 1) with { MaxShape = axisBound },
            axisLen >= 0
                ? RuntimeTensorFactory.Create(DType.Int64, new Shape([axisLen]))
                : RuntimeTensorFactory.CreateRankOnly(DType.Int64, 1),
            RuntimeTensorFactory.CreateRankOnly(DType.Int64, 1) with { MaxShape = axisBound },
        ];
    }

    private static RuntimeTensor[] ComputeFlattenValues(RuntimeTensor x, DType dtype, bool sorted, int numel)
    {
        // Keyed on the storage type itself: a double view neither distinguishes int64 values
        // above 2^53 nor orders unsigned lanes, which share the signed long buffer.
        if (x.FloatData is { } fd)
            return Gather(numel, i => fd[i], Comparer<float>.Default, dtype, sorted,
                (y, order) => y with { FloatData = ImmutableArray.CreateRange(order) });

        var id = x.IntData!.Value;
        return Gather(numel, i => id[i], IntSemantics.Comparer(DTypeHelpers.IsUnsignedInt(dtype)), dtype, sorted,
            (y, order) => y with { IntData = ImmutableArray.CreateRange(order) });
    }

    /// <summary>
    /// The four Unique outputs over any key type: distinct values (sorted, or in
    /// first-occurrence order), their first indices, the inverse map, and the occurrence counts.
    /// </summary>
    private static RuntimeTensor[] Gather<T>(int numel, Func<int, T> get, IComparer<T> comparer,
        DType dtype, bool sorted, Func<RuntimeTensor, List<T>, RuntimeTensor> withValues)
        where T : notnull
    {
        var firstIndex = new Dictionary<T, long>();
        var order = new List<T>();
        var counts = new Dictionary<T, long>();
        for (int i = 0; i < numel; i++)
        {
            var v = get(i);
            if (!firstIndex.ContainsKey(v))
            {
                firstIndex[v] = i;
                order.Add(v);
                counts[v] = 0;
            }
            counts[v]++;
        }
        if (sorted) order.Sort(comparer);

        var inverseMap = new Dictionary<T, long>();
        for (int u = 0; u < order.Count; u++) inverseMap[order[u]] = u;

        var idxArr = new long[order.Count];
        var cntArr = new long[order.Count];
        for (int u = 0; u < order.Count; u++)
        {
            idxArr[u] = firstIndex[order[u]];
            cntArr[u] = counts[order[u]];
        }
        var invArr = new long[numel];
        for (int i = 0; i < numel; i++) invArr[i] = inverseMap[get(i)];

        var yShape = new Shape([(long)order.Count]);
        var y = withValues(RuntimeTensorFactory.Create(dtype, yShape), order);

        return [
            y,
            RuntimeTensorFactory.Create(DType.Int64, yShape) with { IntData = ImmutableArray.Create(idxArr) },
            RuntimeTensorFactory.Create(DType.Int64, new Shape([(long)numel])) with { IntData = ImmutableArray.Create(invArr) },
            RuntimeTensorFactory.Create(DType.Int64, yShape) with { IntData = ImmutableArray.Create(cntArr) },
        ];
    }
}

using System.Collections.Immutable;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// QEE kernel for ONNX <c>ScatterElements</c>: output is a copy of <c>data</c> with the
/// entries addressed by <c>indices</c> (along <c>axis</c>, negative indices count from the
/// end) replaced by — or combined with, per the <c>reduction</c> attribute (none / add /
/// mul / max / min) — the matching <c>updates</c> entries.
/// </summary>
internal sealed class ScatterElementsOp : QuickOp
{
    public override string OpCode => OpCodes.SCATTER_ELEMENTS;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs[0];
        var indices = inputs.Length > 1 ? inputs[1] : null;
        var updates = inputs.Length > 2 ? inputs[2] : null;
        var dtype = x?.DType ?? DType.Float32;
        var rt = RuntimeTensorFactory.Create(dtype, x?.Shape);
        if (x?.Shape is null || !RuntimeTensorFactory.ShouldStoreData(x.Shape, maxDataElements))
            return [rt];
        if (indices?.Shape is null || indices.IntData is not { } idxData) return [rt];

        var inDims = x.Shape.Dims;
        var idxDims = indices.Shape.Dims;
        var rank = inDims.Length;
        if (idxDims.Length != rank) return [rt]; // invalid per spec

        var axis = (int)AttrAccess.GetLong(attrs, OnnxOpAttributeNames.AttrAxis, 0);
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) return [rt];
        long axisSize = inDims[axis];

        var reduction = ScatterReduction.Resolve(attrs);
        if (reduction is null) return [rt]; // unrecognized reduction — don't guess

        var inStrides = new long[rank];
        long s = 1;
        for (int d = rank - 1; d >= 0; d--) { inStrides[d] = s; s *= inDims[d]; }

        long idxCount = indices.Shape.Count;
        var idx = new long[rank];

        if (x.FloatData is { } fd && updates?.FloatData is { } uf && uf.Length >= idxCount)
        {
            var buf = fd.ToArray();
            for (long flat = 0; flat < idxCount; flat++)
            {
                long ix = idxData[(int)flat];
                if (ix < 0) ix += axisSize;
                if (ix < 0 || ix >= axisSize) return [rt];
                long dst = 0;
                for (int d = 0; d < rank; d++) dst += (d == axis ? ix : idx[d]) * inStrides[d];
                buf[(int)dst] = ScatterReduction.ApplyFloat(reduction.Value, buf[(int)dst], uf[(int)flat]);
                Advance(idx, idxDims);
            }
            return [rt with { FloatData = ImmutableArray.Create(buf) }];
        }
        if (x.IntData is { } xd && updates?.IntData is { } ui && ui.Length >= idxCount)
        {
            var buf = xd.ToArray();
            for (long flat = 0; flat < idxCount; flat++)
            {
                long ix = idxData[(int)flat];
                if (ix < 0) ix += axisSize;
                if (ix < 0 || ix >= axisSize) return [rt];
                long dst = 0;
                for (int d = 0; d < rank; d++) dst += (d == axis ? ix : idx[d]) * inStrides[d];
                buf[(int)dst] = ScatterReduction.ApplyInt(reduction.Value, buf[(int)dst], ui[(int)flat]);
                Advance(idx, idxDims);
            }
            return [rt with { IntData = ImmutableArray.Create(buf) }];
        }
        return [rt];
    }

    private static void Advance(long[] idx, long[] dims)
    {
        for (int d = dims.Length - 1; d >= 0; d--)
        {
            idx[d]++;
            if (idx[d] < dims[d]) break;
            idx[d] = 0;
        }
    }
}

/// <summary>
/// Shared tolerant resolution + application of the <c>reduction</c> attribute used by
/// <see cref="ScatterElementsOp"/> and <see cref="ScatterNDOp"/>. Accepts the in-framework
/// <see cref="ScatterNDReduction"/> enum or the wire-form lowercase string; null when an
/// unknown value is supplied (callers then skip value computation).
/// </summary>
internal static class ScatterReduction
{
    public static ScatterNDReduction? Resolve(OnnxCSharpAttributes attrs)
    {
        if (!attrs.IsAttributeDefined(OnnxOpAttributeNames.AttrReduction)) return ScatterNDReduction.None;
        var obj = attrs.GetAttributeObj(OnnxOpAttributeNames.AttrReduction);
        return obj switch
        {
            null => ScatterNDReduction.None,
            ScatterNDReduction r => r,
            string s when s.Equals("none", StringComparison.OrdinalIgnoreCase) => ScatterNDReduction.None,
            string s when s.Equals("add", StringComparison.OrdinalIgnoreCase) => ScatterNDReduction.Add,
            string s when s.Equals("mul", StringComparison.OrdinalIgnoreCase) => ScatterNDReduction.Mul,
            string s when s.Equals("max", StringComparison.OrdinalIgnoreCase) => ScatterNDReduction.Max,
            string s when s.Equals("min", StringComparison.OrdinalIgnoreCase) => ScatterNDReduction.Min,
            _ => null,
        };
    }

    public static float ApplyFloat(ScatterNDReduction reduction, float current, float update) => reduction switch
    {
        ScatterNDReduction.Add => current + update,
        ScatterNDReduction.Mul => current * update,
        ScatterNDReduction.Max => Math.Max(current, update),
        ScatterNDReduction.Min => Math.Min(current, update),
        _ => update,
    };

    public static long ApplyInt(ScatterNDReduction reduction, long current, long update) => reduction switch
    {
        ScatterNDReduction.Add => current + update,
        ScatterNDReduction.Mul => current * update,
        ScatterNDReduction.Max => Math.Max(current, update),
        ScatterNDReduction.Min => Math.Min(current, update),
        _ => update,
    };
}

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

internal sealed class CastOp : QuickOp
{
    public override string OpCode => OpCodes.CAST;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs[0];
        var toDType = attrs.GetDTypeVal(OnnxOpAttributeNames.AttrTo) ?? x?.DType ?? DType.Float32;
        var rt = new RuntimeTensor
        {
            DType = toDType,
            Shape = x?.Shape,
            MaxShape = x?.MaxShape ?? x?.Shape,
            Rank = x?.Rank,
            MaxRank = x?.MaxRank ?? x?.Rank,
        };

        if (x is null || !RuntimeTensorFactory.ShouldStoreData(rt.Shape, maxDataElements))
            return [rt];

        return [WithConvertedData(rt, x, toDType)];
    }

    /// <summary>
    /// Returns <paramref name="rt"/> with <paramref name="x"/>'s data converted to the
    /// storage category of <paramref name="toDType"/>. Shared with <see cref="CastLikeOp"/>.
    ///
    /// QEE value-conversion semantics (the ONNX Cast spec leaves integer overflow
    /// implementation-defined, so this documents what QEE does rather than chasing
    /// exact ORT parity):
    /// <list type="bullet">
    ///   <item>float→int truncates toward zero; out-of-range values saturate at
    ///         <c>long.MinValue</c>/<c>long.MaxValue</c> and NaN becomes 0 (.NET's
    ///         saturating float→integer conversion). Narrower int targets (int8…int32,
    ///         unsigned) are NOT wrapped/clamped to their own range — QEE stores all
    ///         integer data as 64-bit.</item>
    ///   <item>x→bool is <c>x != 0</c>; bool→numeric is 1/0.</item>
    ///   <item>Float16/BFloat16 targets reuse the float32 storage unchanged — the
    ///         stored values do not model f16/bf16 rounding (real rounding happens in
    ///         the ORT execution path and in TensorDataConversion's constant paths).</item>
    ///   <item>Every DType pair propagates the dtype (Compute always stamps the `to`
    ///         attr); pairs outside the Float/Int/Bool storage categories — String,
    ///         Complex64/128, Int4/UInt4 — keep shape/dtype but carry no data. Those
    ///         dtypes are documented-unsupported and throw UnsupportedDTypeException
    ///         (or, for Int4/UInt4, DT001/DT002 in DType.ToIVarType) wherever typed
    ///         materialization is attempted.</item>
    /// </list>
    /// </summary>
    internal static RuntimeTensor WithConvertedData(RuntimeTensor rt, RuntimeTensor x, DType toDType)
    {
        var targetCategory = DTypeHelpers.Categorize(toDType);
        if (targetCategory == DTypeCategory.Float)
        {
            if (x.FloatData is { } fd) return rt with { FloatData = fd };
            if (x.IntData is { } id) return rt with { FloatData = ImmutableArray.CreateRange(id.Select(v => (float)v)) };
            if (x.BoolData is { } bd) return rt with { FloatData = ImmutableArray.CreateRange(bd.Select(v => v ? 1f : 0f)) };
        }
        else if (targetCategory == DTypeCategory.Int)
        {
            if (x.IntData is { } id) return rt with { IntData = id };
            if (x.FloatData is { } fd) return rt with { IntData = ImmutableArray.CreateRange(fd.Select(v => (long)v)) };
            if (x.BoolData is { } bd) return rt with { IntData = ImmutableArray.CreateRange(bd.Select(v => v ? 1L : 0L)) };
        }
        else if (targetCategory == DTypeCategory.Bool)
        {
            if (x.BoolData is { } bd) return rt with { BoolData = bd };
            if (x.FloatData is { } fd) return rt with { BoolData = ImmutableArray.CreateRange(fd.Select(v => v != 0)) };
            if (x.IntData is { } id) return rt with { BoolData = ImmutableArray.CreateRange(id.Select(v => v != 0)) };
        }
        return rt;
    }
}

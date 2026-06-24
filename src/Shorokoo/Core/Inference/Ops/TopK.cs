using System.Collections.Immutable;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// QEE kernel for ONNX <c>TopK</c>: outputs (Values, Indices), each with the input shape
/// except dim[axis] = k. When the K input is connected but its value is unknown the
/// outputs degrade to rank-only with the input shape as MaxShape (k ≤ dim) — never a
/// guessed extent. With concrete inputs the values are computed honoring
/// <c>largest</c>/<c>sorted</c> (ties resolved to the lower index, matching ORT;
/// sorted=0 still returns sorted order, which the spec leaves unspecified).
/// </summary>
internal sealed class TopKOp : QuickOp
{
    public override string OpCode => OpCodes.TOPK;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs[0];
        var k = inputs.Length > 1 ? inputs[1] : null;
        var dtype = x?.DType ?? DType.Float32;

        if (x?.Shape is null)
            return [
                RuntimeTensorFactory.CreateRankOnly(dtype, x?.Rank),
                RuntimeTensorFactory.CreateRankOnly(DType.Int64, x?.Rank),
            ];

        var dims = x.Shape.Dims;
        var rank = dims.Length;
        var axisAttr = attrs.GetLongVal(OnnxOpAttributeNames.AttrAxis) ?? -1;
        var axis = (int)(axisAttr < 0 ? axisAttr + rank : axisAttr);
        if (axis < 0 || axis >= rank)
            return [
                RuntimeTensorFactory.CreateRankOnly(dtype, rank),
                RuntimeTensorFactory.CreateRankOnly(DType.Int64, rank),
            ];

        // K is a required input; its concrete value gates exact shape claims.
        long? kVal = k?.IntData is { Length: > 0 } kv ? kv[0] : null;
        if (kVal is null || kVal < 0 || (dims[axis] >= 0 && kVal > dims[axis]))
            return [
                RuntimeTensorFactory.CreateRankOnly(dtype, rank) with { MaxShape = x.Shape },
                RuntimeTensorFactory.CreateRankOnly(DType.Int64, rank) with { MaxShape = x.Shape },
            ];

        var outDims = dims.ToArray();
        outDims[axis] = kVal.Value;
        var shape = new Shape(outDims);
        var values = RuntimeTensorFactory.Create(dtype, shape);
        var indices = RuntimeTensorFactory.Create(DType.Int64, shape);

        if (!RuntimeTensorFactory.ShouldStoreData(shape, maxDataElements)
            || dims.Any(d => d < 0)
            || (x.FloatData is null && x.IntData is null))
            return [values, indices];

        var largest = attrs.GetAttributeObj(OnnxOpAttributeNames.AttrLargest) as bool? ?? true;
        var isFloat = x.FloatData is not null;
        Func<int, double> get = isFloat
            ? i => x.FloatData!.Value[i]
            : i => x.IntData!.Value[i];

        long axisLen = dims[axis];
        long inner = 1;
        for (int d = axis + 1; d < rank; d++) inner *= dims[d];
        long outer = 1;
        for (int d = 0; d < axis; d++) outer *= dims[d];

        int kInt = (int)kVal.Value;
        var outCount = (int)shape.Count;
        var valF = isFloat ? new float[outCount] : null;
        var valI = isFloat ? null : new long[outCount];
        var idxOut = new long[outCount];

        for (long o = 0; o < outer; o++)
        {
            for (long inIdx = 0; inIdx < inner; inIdx++)
            {
                var sliceIdx = Enumerable.Range(0, (int)axisLen)
                    .OrderBy(i => largest ? -get((int)(o * axisLen * inner + i * inner + inIdx)) : get((int)(o * axisLen * inner + i * inner + inIdx)))
                    .ThenBy(i => i)
                    .Take(kInt)
                    .ToArray();
                for (int r = 0; r < kInt; r++)
                {
                    var src = (int)(o * axisLen * inner + sliceIdx[r] * inner + inIdx);
                    var dst = (int)(o * kInt * inner + r * inner + inIdx);
                    if (isFloat) valF![dst] = x.FloatData!.Value[src];
                    else valI![dst] = x.IntData!.Value[src];
                    idxOut[dst] = sliceIdx[r];
                }
            }
        }

        values = isFloat
            ? values with { FloatData = ImmutableArray.Create(valF!) }
            : values with { IntData = ImmutableArray.Create(valI!) };
        return [values, indices with { IntData = ImmutableArray.Create(idxOut) }];
    }
}

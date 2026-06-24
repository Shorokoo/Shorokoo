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
/// Shared QEE implementation for ONNX <c>ArgMax</c> / <c>ArgMin</c>: int64 output, shape
/// from <c>axis</c>+<c>keepdims</c>, and a concrete value path honoring
/// <c>select_last_index</c> (ties pick the last occurrence when set, first otherwise).
/// </summary>
internal abstract class ArgExtremeOpBase : QuickOp
{
    /// <summary>True when <paramref name="candidate"/> strictly beats <paramref name="best"/>
    /// (greater for ArgMax, less for ArgMin).</summary>
    protected abstract bool Beats(float candidate, float best);

    protected abstract bool BeatsInt(long candidate, long best);

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs[0];
        var axis = (int)AttrAccess.GetLong(attrs, OnnxOpAttributeNames.AttrAxis, 0);
        var keepDims = AttrAccess.GetBool(attrs, OnnxOpAttributeNames.AttrKeepdims, true);
        var selectLastIndex = AttrAccess.GetBool(attrs, OnnxOpAttributeNames.AttrSelectLastIndex, false);
        if (x?.Shape is null) return [RuntimeTensorFactory.Create(DType.Int64, null)];

        var inDims = x.Shape.Dims;
        if (axis < 0) axis += inDims.Length;
        if (axis < 0 || axis >= inDims.Length) return [RuntimeTensorFactory.Create(DType.Int64, null)];

        var dims = inDims.ToList();
        if (keepDims) dims[axis] = 1;
        else dims.RemoveAt(axis);
        var outShape = new Shape(dims.ToArray());
        var rt = RuntimeTensorFactory.Create(DType.Int64, outShape);

        if (!RuntimeTensorFactory.ShouldStoreData(outShape, maxDataElements)) return [rt];
        if (x.FloatData is null && x.IntData is null) return [rt];

        long axisDim = inDims[axis];
        if (axisDim <= 0) return [rt];
        long innerCount = 1;
        for (int d = axis + 1; d < inDims.Length; d++) innerCount *= inDims[d];
        long outerCount = 1;
        for (int d = 0; d < axis; d++) outerCount *= inDims[d];

        var buf = new long[outerCount * innerCount];
        long pos = 0;
        for (long outer = 0; outer < outerCount; outer++)
        {
            long outerOff = outer * axisDim * innerCount;
            for (long inner = 0; inner < innerCount; inner++)
            {
                long bestIdx = 0;
                if (x.FloatData is { } fd)
                {
                    float best = fd[(int)(outerOff + inner)];
                    for (long k = 1; k < axisDim; k++)
                    {
                        float v = fd[(int)(outerOff + k * innerCount + inner)];
                        if (Beats(v, best) || (selectLastIndex && v == best)) { best = v; bestIdx = k; }
                    }
                }
                else
                {
                    var id = x.IntData!.Value;
                    long best = id[(int)(outerOff + inner)];
                    for (long k = 1; k < axisDim; k++)
                    {
                        long v = id[(int)(outerOff + k * innerCount + inner)];
                        if (BeatsInt(v, best) || (selectLastIndex && v == best)) { best = v; bestIdx = k; }
                    }
                }
                buf[pos++] = bestIdx;
            }
        }
        return [rt with { IntData = ImmutableArray.Create(buf) }];
    }
}

internal sealed class ArgMaxOp : ArgExtremeOpBase
{
    public override string OpCode => OpCodes.ARG_MAX;
    protected override bool Beats(float candidate, float best) => candidate > best;
    protected override bool BeatsInt(long candidate, long best) => candidate > best;
}

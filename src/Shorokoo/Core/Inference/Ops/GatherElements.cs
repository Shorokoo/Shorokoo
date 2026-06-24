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
/// QEE kernel for ONNX <c>GatherElements</c>: the output has the indices' shape, the data's
/// dtype, and <c>out[i…] = data[i…with axis-coord replaced by indices[i…]]</c> (negative
/// indices count from the end of the axis).
/// </summary>
internal sealed class GatherElementsOp : QuickOp
{
    public override string OpCode => OpCodes.GATHER_ELEMENTS;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        // GatherElements output shape equals the indices shape.
        var x = inputs[0];
        var indices = inputs.Length > 1 ? inputs[1] : null;
        var dtype = x?.DType ?? DType.Float32;
        var rt = RuntimeTensorFactory.Create(dtype, indices?.Shape);
        if (x?.Shape is null || indices?.Shape is null) return [rt];

        if (!RuntimeTensorFactory.ShouldStoreData(indices.Shape, maxDataElements)) return [rt];
        if (indices.IntData is not { } idxData) return [rt];
        if (x.FloatData is null && x.IntData is null && x.BoolData is null) return [rt];

        var inDims = x.Shape.Dims;
        var outDims = indices.Shape.Dims;
        var rank = inDims.Length;
        if (outDims.Length != rank) return [rt]; // invalid per spec — don't guess

        var axis = (int)AttrAccess.GetLong(attrs, OnnxOpAttributeNames.AttrAxis, 0);
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) return [rt];
        long axisSize = inDims[axis];

        var inStrides = new long[rank];
        long s = 1;
        for (int d = rank - 1; d >= 0; d--) { inStrides[d] = s; s *= inDims[d]; }

        long outCount = indices.Shape.Count;
        long[]? intBuf = x.IntData is not null ? new long[outCount] : null;
        float[]? floatBuf = x.FloatData is not null ? new float[outCount] : null;
        bool[]? boolBuf = x.BoolData is not null ? new bool[outCount] : null;
        var idx = new long[rank];
        for (long flat = 0; flat < outCount; flat++)
        {
            long ix = idxData[(int)flat];
            if (ix < 0) ix += axisSize;
            if (ix < 0 || ix >= axisSize) return [rt]; // out-of-range index — drop values
            long src = 0;
            for (int d = 0; d < rank; d++) src += (d == axis ? ix : idx[d]) * inStrides[d];
            if (intBuf is not null) intBuf[flat] = x.IntData!.Value[(int)src];
            else if (floatBuf is not null) floatBuf[flat] = x.FloatData!.Value[(int)src];
            else if (boolBuf is not null) boolBuf[flat] = x.BoolData!.Value[(int)src];
            for (int d = rank - 1; d >= 0; d--)
            {
                idx[d]++;
                if (idx[d] < outDims[d]) break;
                idx[d] = 0;
            }
        }
        if (intBuf is not null) return [rt with { IntData = ImmutableArray.Create(intBuf) }];
        if (floatBuf is not null) return [rt with { FloatData = ImmutableArray.Create(floatBuf) }];
        if (boolBuf is not null) return [rt with { BoolData = ImmutableArray.Create(boolBuf) }];
        return [rt];
    }
}

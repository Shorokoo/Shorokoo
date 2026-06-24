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
/// QEE kernel for ONNX <c>GatherND</c>. Output shape is
/// <c>indices.shape[:-1] + data.shape[batch_dims + indices.shape[-1]:]</c>; each k-tuple in
/// the last indices dim addresses a slice of <c>data</c> (after the shared batch dims).
/// </summary>
internal sealed class GatherNDOp : QuickOp
{
    public override string OpCode => OpCodes.GATHER_ND;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs[0];
        var indices = inputs[1];
        var batchDims = (int)(attrs.GetLongVal(OnnxOpAttributeNames.AttrBatchDims) ?? 0);
        var dtype = x?.DType ?? DType.Float32;

        if (x?.Shape is null || indices?.Shape is null)
            return [RuntimeTensorFactory.Create(dtype, null)];

        var r = x.Shape.Dims.Length;
        var q = indices.Shape.Dims.Length;
        if (q < 1 || batchDims < 0 || batchDims >= q)
            return [RuntimeTensorFactory.Create(dtype, null)];
        var k = (int)indices.Shape.Dims[q - 1];
        if (k < 1 || batchDims + k > r)
            return [RuntimeTensorFactory.Create(dtype, null)];
        // Output rank = q + r - k - 1 - batchDims
        var outDims = new List<long>();
        for (int i = 0; i < batchDims; i++) outDims.Add(x.Shape.Dims[i]);
        for (int i = batchDims; i < q - 1; i++) outDims.Add(indices.Shape.Dims[i]);
        for (int i = batchDims + k; i < r; i++) outDims.Add(x.Shape.Dims[i]);
        var outShape = new Shape(outDims.ToArray());
        var rt = RuntimeTensorFactory.Create(dtype, outShape);

        if (!RuntimeTensorFactory.ShouldStoreData(outShape, maxDataElements)) return [rt];
        if (indices.IntData is not { } idxData) return [rt];
        if (x.FloatData is null && x.IntData is null && x.BoolData is null) return [rt];

        var inDims = x.Shape.Dims;
        var inStrides = new long[r];
        long s = 1;
        for (int d = r - 1; d >= 0; d--) { inStrides[d] = s; s *= inDims[d]; }

        // Per batch-and-tuple slice: copy data.shape[batchDims+k:] contiguous elements.
        long sliceLen = 1;
        for (int d = batchDims + k; d < r; d++) sliceLen *= inDims[d];
        long batchCount = 1;
        for (int d = 0; d < batchDims; d++) batchCount *= inDims[d];
        long tupleCount = 1;
        for (int d = batchDims; d < q - 1; d++) tupleCount *= indices.Shape.Dims[d];

        long outCount = outShape.Count;
        long[]? intBuf = x.IntData is not null ? new long[outCount] : null;
        float[]? floatBuf = x.FloatData is not null ? new float[outCount] : null;
        bool[]? boolBuf = x.BoolData is not null ? new bool[outCount] : null;

        // The batch coords are shared between data and indices: batch b is the b-th
        // combination of the leading batchDims dims (identical in both tensors for valid
        // models), so data offsets use inStrides over those dims.
        var batchIdx = new long[Math.Max(batchDims, 1)];
        long outPos = 0;
        for (long b = 0; b < batchCount; b++)
        {
            long batchOff = 0;
            for (int d = 0; d < batchDims; d++) batchOff += batchIdx[d] * inStrides[d];
            for (long t = 0; t < tupleCount; t++)
            {
                long tupleBase = (b * tupleCount + t) * k;
                long srcBase = batchOff;
                for (int j = 0; j < k; j++)
                {
                    long ix = idxData[(int)(tupleBase + j)];
                    long dimSize = inDims[batchDims + j];
                    if (ix < 0) ix += dimSize;
                    if (ix < 0 || ix >= dimSize) return [rt]; // out-of-range — drop values
                    srcBase += ix * inStrides[batchDims + j];
                }
                for (long e = 0; e < sliceLen; e++)
                {
                    long src = srcBase + e;
                    if (intBuf is not null) intBuf[outPos] = x.IntData!.Value[(int)src];
                    else if (floatBuf is not null) floatBuf[outPos] = x.FloatData!.Value[(int)src];
                    else if (boolBuf is not null) boolBuf[outPos] = x.BoolData!.Value[(int)src];
                    outPos++;
                }
            }
            for (int d = batchDims - 1; d >= 0; d--)
            {
                batchIdx[d]++;
                if (batchIdx[d] < inDims[d]) break;
                batchIdx[d] = 0;
            }
        }

        if (intBuf is not null) return [rt with { IntData = ImmutableArray.Create(intBuf) }];
        if (floatBuf is not null) return [rt with { FloatData = ImmutableArray.Create(floatBuf) }];
        if (boolBuf is not null) return [rt with { BoolData = ImmutableArray.Create(boolBuf) }];
        return [rt];
    }
}

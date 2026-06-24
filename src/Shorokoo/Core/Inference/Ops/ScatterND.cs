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
/// QEE kernel for ONNX <c>ScatterND</c>: output is a copy of <c>data</c>; each k-tuple in
/// the last dim of <c>indices</c> addresses a slice of shape <c>data.shape[k:]</c> that is
/// replaced by — or combined with, per <c>reduction</c> — the matching <c>updates</c> slice.
/// </summary>
internal sealed class ScatterNDOp : QuickOp
{
    public override string OpCode => OpCodes.SCATTER_ND;

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
        var r = inDims.Length;
        var q = indices.Shape.Dims.Length;
        if (q < 1) return [rt];
        var k = (int)indices.Shape.Dims[q - 1];
        if (k < 1 || k > r) return [rt];

        var reduction = ScatterReduction.Resolve(attrs);
        if (reduction is null) return [rt];

        var inStrides = new long[r];
        long s = 1;
        for (int d = r - 1; d >= 0; d--) { inStrides[d] = s; s *= inDims[d]; }

        long sliceLen = 1;
        for (int d = k; d < r; d++) sliceLen *= inDims[d];
        long tupleCount = idxData.Length / Math.Max(k, 1);

        if (x.FloatData is { } fd && updates?.FloatData is { } uf && uf.Length >= tupleCount * sliceLen)
        {
            var buf = fd.ToArray();
            for (long t = 0; t < tupleCount; t++)
            {
                long dstBase = 0;
                for (int j = 0; j < k; j++)
                {
                    long ix = idxData[(int)(t * k + j)];
                    if (ix < 0) ix += inDims[j];
                    if (ix < 0 || ix >= inDims[j]) return [rt];
                    dstBase += ix * inStrides[j];
                }
                for (long e = 0; e < sliceLen; e++)
                    buf[(int)(dstBase + e)] = ScatterReduction.ApplyFloat(
                        reduction.Value, buf[(int)(dstBase + e)], uf[(int)(t * sliceLen + e)]);
            }
            return [rt with { FloatData = ImmutableArray.Create(buf) }];
        }
        if (x.IntData is { } xd && updates?.IntData is { } ui && ui.Length >= tupleCount * sliceLen)
        {
            var buf = xd.ToArray();
            for (long t = 0; t < tupleCount; t++)
            {
                long dstBase = 0;
                for (int j = 0; j < k; j++)
                {
                    long ix = idxData[(int)(t * k + j)];
                    if (ix < 0) ix += inDims[j];
                    if (ix < 0 || ix >= inDims[j]) return [rt];
                    dstBase += ix * inStrides[j];
                }
                for (long e = 0; e < sliceLen; e++)
                    buf[(int)(dstBase + e)] = ScatterReduction.ApplyInt(
                        reduction.Value, buf[(int)(dstBase + e)], ui[(int)(t * sliceLen + e)]);
            }
            return [rt with { IntData = ImmutableArray.Create(buf) }];
        }
        return [rt];
    }
}

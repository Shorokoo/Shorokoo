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

internal sealed class ExpandOp : QuickOp
{
    public override string OpCode => OpCodes.EXPAND;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs[0];
        var dtype = x?.DType ?? DType.Float32;
        if (inputs.Length <= 1 || inputs[1]?.IntData is not { } shapeVals)
            return [RuntimeTensorFactory.Create(dtype, null)];
        var target = new Shape(shapeVals.ToArray());
        var result = ShapeHelpers.Broadcast(x?.Shape, target) ?? target;
        var rt = RuntimeTensorFactory.Create(dtype, result);

        if (x?.Shape is null || !RuntimeTensorFactory.ShouldStoreData(result, maxDataElements))
            return [rt];

        if (x.FloatData is { } fd)
            return [rt with { FloatData = ImmutableArray.Create(BroadcastFloat(fd, x.Shape, result)) }];
        if (x.IntData is { } id)
            return [rt with { IntData = ImmutableArray.Create(BroadcastInt(id, x.Shape, result)) }];
        if (x.BoolData is { } bd)
            return [rt with { BoolData = ImmutableArray.Create(BroadcastBool(bd, x.Shape, result)) }];
        return [rt];
    }

    private static float[] BroadcastFloat(ImmutableArray<float> src, Shape srcShape, Shape outShape)
    {
        var dst = new float[outShape.Count];
        var strides = BroadcastStrides(srcShape.Dims, outShape.Dims);
        var idx = new int[outShape.Dims.Length];
        for (int flat = 0; flat < dst.Length; flat++)
        {
            int sp = 0;
            for (int d = 0; d < idx.Length; d++) sp += idx[d] * strides[d];
            dst[flat] = src[sp];
            for (int d = idx.Length - 1; d >= 0; d--)
            {
                if (++idx[d] < outShape.Dims[d]) break;
                idx[d] = 0;
            }
        }
        return dst;
    }

    private static long[] BroadcastInt(ImmutableArray<long> src, Shape srcShape, Shape outShape)
    {
        var dst = new long[outShape.Count];
        var strides = BroadcastStrides(srcShape.Dims, outShape.Dims);
        var idx = new int[outShape.Dims.Length];
        for (int flat = 0; flat < dst.Length; flat++)
        {
            int sp = 0;
            for (int d = 0; d < idx.Length; d++) sp += idx[d] * strides[d];
            dst[flat] = src[sp];
            for (int d = idx.Length - 1; d >= 0; d--)
            {
                if (++idx[d] < outShape.Dims[d]) break;
                idx[d] = 0;
            }
        }
        return dst;
    }

    private static bool[] BroadcastBool(ImmutableArray<bool> src, Shape srcShape, Shape outShape)
    {
        var dst = new bool[outShape.Count];
        var strides = BroadcastStrides(srcShape.Dims, outShape.Dims);
        var idx = new int[outShape.Dims.Length];
        for (int flat = 0; flat < dst.Length; flat++)
        {
            int sp = 0;
            for (int d = 0; d < idx.Length; d++) sp += idx[d] * strides[d];
            dst[flat] = src[sp];
            for (int d = idx.Length - 1; d >= 0; d--)
            {
                if (++idx[d] < outShape.Dims[d]) break;
                idx[d] = 0;
            }
        }
        return dst;
    }

    private static int[] BroadcastStrides(long[] srcDims, long[] outDims)
    {
        var strides = new int[outDims.Length];
        int stride = 1;
        int srcOffset = outDims.Length - srcDims.Length;
        for (int d = srcDims.Length - 1; d >= 0; d--)
        {
            var odIdx = d + srcOffset;
            strides[odIdx] = srcDims[d] == 1 ? 0 : stride;
            stride *= (int)srcDims[d];
        }
        return strides;
    }
}

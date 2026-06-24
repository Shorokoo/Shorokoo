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

internal sealed class AndOp : QuickOp
{
    public override string OpCode => OpCodes.AND;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var a = inputs[0];
        var b = inputs[1];
        var shape = ShapeHelpers.Broadcast(a?.Shape, b?.Shape);
        var rt = RuntimeTensorFactory.Create(DType.Bool, shape);
        if (shape is not null && RuntimeTensorFactory.ShouldStoreData(shape, maxDataElements)
            && a?.BoolData is { } aData && b?.BoolData is { } bData)
        {
            var dst = new bool[shape.Count];
            var aDims = a.Shape!.Dims;
            var bDims = b.Shape!.Dims;
            var rank = shape.Dims.Length;
            var idx = new int[rank];
            var aStride = Stride(aDims, shape.Dims);
            var bStride = Stride(bDims, shape.Dims);
            for (int i = 0; i < dst.Length; i++)
            {
                int ia = 0, ib = 0;
                for (int d = 0; d < rank; d++) { ia += idx[d] * aStride[d]; ib += idx[d] * bStride[d]; }
                dst[i] = aData[ia] && bData[ib];
                for (int d = rank - 1; d >= 0; d--) { if (++idx[d] < shape.Dims[d]) break; idx[d] = 0; }
            }
            return [rt with { BoolData = ImmutableArray.Create(dst) }];
        }
        return [rt];
    }

    private static int[] Stride(long[] src, long[] outDims)
    {
        var s = new int[outDims.Length];
        int stride = 1;
        int off = outDims.Length - src.Length;
        for (int d = src.Length - 1; d >= 0; d--) { s[d + off] = src[d] == 1 ? 0 : stride; stride *= (int)src[d]; }
        return s;
    }
}

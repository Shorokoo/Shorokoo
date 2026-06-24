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

internal sealed class TileOp : QuickOp
{
    public override string OpCode => OpCodes.TILE;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs[0];
        var dtype = x?.DType ?? DType.Float32;
        if (x?.Shape is null || inputs.Length <= 1 || inputs[1]?.IntData is not { } repeats)
            return [RuntimeTensorFactory.Create(dtype, null)];
        if (repeats.Any(r => r < 0))
            return [RuntimeTensorFactory.Create(dtype, null)];

        var inDims = x.Shape.Dims;
        var rank = inDims.Length;
        var outDims = inDims.ToArray();
        for (int d = 0; d < rank && d < repeats.Length; d++)
            outDims[d] *= repeats[d];
        var outShape = new Shape(outDims);
        var rt = RuntimeTensorFactory.Create(dtype, outShape);

        if (!RuntimeTensorFactory.ShouldStoreData(outShape, maxDataElements)) return [rt];
        if (x.FloatData is null && x.IntData is null && x.BoolData is null) return [rt];

        // Value path: in-coord = out-coord mod input dim, per dimension.
        var inStrides = new long[rank];
        long s = 1;
        for (int d = rank - 1; d >= 0; d--) { inStrides[d] = s; s *= inDims[d]; }

        long outCount = outShape.Count;
        var idx = new long[rank];
        long[]? intBuf = x.IntData is not null ? new long[outCount] : null;
        float[]? floatBuf = x.FloatData is not null ? new float[outCount] : null;
        bool[]? boolBuf = x.BoolData is not null ? new bool[outCount] : null;
        for (long flat = 0; flat < outCount; flat++)
        {
            long src = 0;
            for (int d = 0; d < rank; d++) src += (idx[d] % inDims[d]) * inStrides[d];
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
        if (intBuf is not null) return [rt with { IntData = System.Collections.Immutable.ImmutableArray.Create(intBuf) }];
        if (floatBuf is not null) return [rt with { FloatData = System.Collections.Immutable.ImmutableArray.Create(floatBuf) }];
        if (boolBuf is not null) return [rt with { BoolData = System.Collections.Immutable.ImmutableArray.Create(boolBuf) }];
        return [rt];
    }
}

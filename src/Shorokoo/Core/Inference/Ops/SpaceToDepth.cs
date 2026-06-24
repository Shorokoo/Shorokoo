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
/// QEE kernel for ONNX <c>SpaceToDepth</c>: rearranges <c>[N, C, H*B, W*B]</c>
/// into <c>[N, C*B², H, W]</c> by partitioning HxW into B-sized blocks and packing each
/// block into the channel dimension (with a concrete value path for small tensors).
/// </summary>
internal sealed class SpaceToDepthOp : QuickOp
{
    public override string OpCode => OpCodes.SPACE_TO_DEPTH;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var dtype = x?.DType ?? DType.Float32;
        if (x?.Shape is null) return [RuntimeTensorFactory.Create(dtype, null)];

        var dims = x.Shape.Dims;
        if (dims.Length != 4) return [RuntimeTensorFactory.Create(dtype, null)];

        // Accept both "blocksize" and "block_size" spellings (ONNX uses "blocksize").
        var bs = attrs.GetLongVal(OnnxOpAttributeNames.AttrBlocksize)
              ?? attrs.GetLongVal(OnnxOpAttributeNames.AttrBlockSize)
              ?? 2;
        if (bs <= 0 || dims[2] % bs != 0 || dims[3] % bs != 0)
            return [RuntimeTensorFactory.Create(dtype, null)];

        var outDims = new long[4];
        outDims[0] = dims[0];
        outDims[1] = dims[1] * bs * bs;
        outDims[2] = dims[2] / bs;
        outDims[3] = dims[3] / bs;
        var outShape = new Shape(outDims);
        var rt = RuntimeTensorFactory.Create(dtype, outShape);

        if (!RuntimeTensorFactory.ShouldStoreData(outShape, maxDataElements)) return [rt];
        if (x.FloatData is null && x.IntData is null && x.BoolData is null) return [rt];

        // Value path: out[n, (b1*B + b2)*C + c, h, w] = in[n, c, h*B + b1, w*B + b2]
        // (the inverse of DepthToSpace's DCR layout).
        long cIn = dims[1], hIn = dims[2], wIn = dims[3];
        long outCount = outShape.Count;
        long[]? intBuf = x.IntData is not null ? new long[outCount] : null;
        float[]? floatBuf = x.FloatData is not null ? new float[outCount] : null;
        bool[]? boolBuf = x.BoolData is not null ? new bool[outCount] : null;
        long pos = 0;
        for (long n = 0; n < outDims[0]; n++)
            for (long oc = 0; oc < outDims[1]; oc++)
                for (long h = 0; h < outDims[2]; h++)
                    for (long w = 0; w < outDims[3]; w++)
                    {
                        long c = oc % cIn;
                        long b2 = (oc / cIn) % bs;
                        long b1 = oc / (cIn * bs);
                        long src = ((n * cIn + c) * hIn + (h * bs + b1)) * wIn + (w * bs + b2);
                        if (intBuf is not null) intBuf[pos] = x.IntData!.Value[(int)src];
                        else if (floatBuf is not null) floatBuf[pos] = x.FloatData!.Value[(int)src];
                        else if (boolBuf is not null) boolBuf[pos] = x.BoolData!.Value[(int)src];
                        pos++;
                    }

        if (intBuf is not null) return [rt with { IntData = System.Collections.Immutable.ImmutableArray.Create(intBuf) }];
        if (floatBuf is not null) return [rt with { FloatData = System.Collections.Immutable.ImmutableArray.Create(floatBuf) }];
        if (boolBuf is not null) return [rt with { BoolData = System.Collections.Immutable.ImmutableArray.Create(boolBuf) }];
        return [rt];
    }
}

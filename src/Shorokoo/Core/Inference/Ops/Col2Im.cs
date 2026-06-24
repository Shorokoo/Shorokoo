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
/// Shape inference for ONNX <c>Col2Im</c>. Output shape is <c>[N, C, *image_shape...]</c>:
/// the batch comes from <c>input.shape[0]</c>, channels are
/// <c>input.shape[1] / prod(block_shape)</c> (the block sizes come from the
/// <c>block_shape</c> INPUT, so they may be unknown at QEE time), and the spatial dims are
/// the <c>image_shape</c> input's values. The dilations/pads/strides attributes only affect
/// which columns map to which pixels, not the output shape. When the batch or channel count
/// can't be derived (input shape unknown, block_shape values unknown, or a non-divisible
/// channel count) the shape degrades to unknown — with the rank still pinned to
/// <c>2 + len(image_shape)</c> when image_shape values are known. Values are not computed
/// (overlap-add reconstruction; shape-only is enough for downstream propagation).
/// </summary>
internal sealed class Col2ImOp : QuickOp
{
    public override string OpCode => OpCodes.COL2IM;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var input = inputs.Length > 0 ? inputs[0] : null;
        var imageShape = inputs.Length > 1 ? inputs[1] : null;
        var blockShape = inputs.Length > 2 ? inputs[2] : null;
        var dtype = input?.DType ?? DType.Float32;

        if (imageShape?.IntData is not { } imageDims)
            return [RuntimeTensorFactory.CreateRankOnly(dtype, null)];

        int outRank = 2 + imageDims.Length;
        if (input?.Shape is null || input.Shape.Dims.Length != 3)
            return [RuntimeTensorFactory.CreateRankOnly(dtype, outRank)];

        var batch = input.Shape.Dims[0];
        long channels = -1;
        if (blockShape?.IntData is { } blockDims)
        {
            long blockProduct = 1;
            for (int i = 0; i < blockDims.Length; i++) blockProduct *= blockDims[i];
            if (blockProduct > 0 && input.Shape.Dims[1] % blockProduct == 0)
                channels = input.Shape.Dims[1] / blockProduct;
        }
        if (batch < 0 || channels < 0)
            return [RuntimeTensorFactory.CreateRankOnly(dtype, outRank)];

        var outDims = new long[outRank];
        outDims[0] = batch;
        outDims[1] = channels;
        for (int i = 0; i < imageDims.Length; i++)
        {
            if (imageDims[i] < 0) return [RuntimeTensorFactory.CreateRankOnly(dtype, outRank)];
            outDims[2 + i] = imageDims[i];
        }
        return [RuntimeTensorFactory.Create(dtype, new Shape(outDims))];
    }
}

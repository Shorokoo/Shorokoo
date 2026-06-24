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
/// Shape inference for ONNX <c>ImageDecoder</c>. Output is a 3-D uint8 image with shape
/// <c>[H, W, C]</c>; H and W are data-dependent (read from the encoded bytes), so only the
/// rank (3) is reported — per-dim placeholders would leak into Shape-op value chains. The
/// channel count implied by <c>pixel_format</c> (3 for RGB/BGR, 4 for RGBA/BGRA, 1 for
/// Grayscale) is not representable without per-dim unknowns, so it is intentionally dropped.
/// </summary>
internal sealed class ImageDecoderOp : QuickOp
{
    public override string OpCode => OpCodes.IMAGE_DECODER;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        return [RuntimeTensorFactory.CreateRankOnly(DType.UInt8, 3)];
    }
}

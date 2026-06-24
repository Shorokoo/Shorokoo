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
/// Shape inference for ONNX <c>RoiAlign</c>. Output is
/// <c>[num_rois, C, output_height, output_width]</c> (output_height/width default to 1 per
/// spec). The mode / coordinate_transformation_mode / sampling_ratio / spatial_scale
/// attributes only affect values. When num_rois or C are unknown the shape degrades to
/// unknown (rank 4) instead of carrying placeholder dims.
/// </summary>
internal sealed class RoiAlignOp : QuickOp
{
    public override string OpCode => OpCodes.ROI_ALIGN;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var rois = inputs.Length > 1 ? inputs[1] : null;
        var dtype = x?.DType ?? DType.Float32;

        var outH = attrs.GetLongVal(OnnxOpAttributeNames.AttrOutputHeight) ?? 1;
        var outW = attrs.GetLongVal(OnnxOpAttributeNames.AttrOutputWidth) ?? 1;

        long numRois = rois?.Shape?.Dims is { Length: > 0 } rDims ? rDims[0] : -1;
        long channels = x?.Shape?.Dims is { Length: > 1 } xDims ? xDims[1] : -1;
        if (numRois < 0 || channels < 0 || outH < 0 || outW < 0)
            return [RuntimeTensorFactory.CreateRankOnly(dtype, 4)];

        return [RuntimeTensorFactory.Create(dtype, new Shape(new[] { numRois, channels, outH, outW }))];
    }
}

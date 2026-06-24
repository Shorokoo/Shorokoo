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
/// Shape inference for ONNX <c>DeformConv</c>. Output spatial dims follow the same
/// kernel/stride/pad/dilation arithmetic as <see cref="ConvOp"/>; the batch dim comes
/// from <c>X</c> and the channel dim from <c>W</c>'s first dim.
/// </summary>
internal sealed class DeformConvOp : QuickOp
{
    public override string OpCode => OpCodes.DEFORM_CONV;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var w = inputs.Length > 1 ? inputs[1] : null;
        var dtype = x?.DType ?? DType.Float32;
        if (x?.Shape is null || w?.Shape is null) return [RuntimeTensorFactory.Create(dtype, null)];

        var xDims = x.Shape.Dims;
        var wDims = w.Shape.Dims;
        var spatialRank = xDims.Length - 2;
        var strides = attrs.GetIntsVal(OnnxOpAttributeNames.AttrStrides);
        var pads = attrs.GetIntsVal(OnnxOpAttributeNames.AttrPads);
        var dilations = attrs.GetIntsVal(OnnxOpAttributeNames.AttrDilations);

        var outDims = new long[xDims.Length];
        outDims[0] = xDims[0];
        outDims[1] = wDims[0];
        for (int d = 0; d < spatialRank; d++)
        {
            var inSize = xDims[d + 2];
            var kSize = wDims[d + 2];
            var stride = strides is not null && d < strides.Length ? strides[d] : 1;
            var dilation = dilations is not null && d < dilations.Length ? dilations[d] : 1;
            var padBegin = pads is not null && d < pads.Length ? pads[d] : 0;
            var padEnd = pads is not null && d + spatialRank < pads.Length ? pads[d + spatialRank] : 0;
            var effectiveK = (kSize - 1) * dilation + 1;
            outDims[d + 2] = (inSize + padBegin + padEnd - effectiveK) / stride + 1;
        }
        return [RuntimeTensorFactory.Create(dtype, new Shape(outDims))];
    }
}

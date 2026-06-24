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
/// Shape inference for ONNX <c>QLinearConv</c>. Same output-shape arithmetic as
/// <see cref="ConvOp"/>; the quantization inputs (scales, zero points, bias) don't
/// affect output dims. Output dtype matches input <c>x</c>.
/// </summary>
internal sealed class QLinearConvOp : QuickOp
{
    public override string OpCode => OpCodes.QLINEAR_CONV;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        // QLinearConv inputs: [x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, B?]
        var x = inputs.Length > 0 ? inputs[0] : null;
        var w = inputs.Length > 3 ? inputs[3] : null;
        var yZeroPoint = inputs.Length > 7 ? inputs[7] : null;
        var dtype = yZeroPoint?.DType ?? x?.DType ?? DType.UInt8;
        if (x?.Shape is null || w?.Shape is null) return [RuntimeTensorFactory.Create(dtype, null)];

        var xDims = x.Shape.Dims;
        var wDims = w.Shape.Dims;
        var spatialRank = xDims.Length - 2;
        var strides = attrs.GetIntsVal(OnnxOpAttributeNames.AttrStrides);
        var pads = attrs.GetIntsVal(OnnxOpAttributeNames.AttrPads);
        var dilations = attrs.GetIntsVal(OnnxOpAttributeNames.AttrDilations);
        var autoPad = attrs.GetAttributeObj(OnnxOpAttributeNames.AttrAutoPad) as AutoPad?
                      ?? AutoPad.NotSet;

        var outDims = new long[xDims.Length];
        outDims[0] = xDims[0];
        outDims[1] = wDims[0];
        for (int d = 0; d < spatialRank; d++)
        {
            var inSize = xDims[d + 2];
            var kSize = wDims[d + 2];
            var stride = strides is not null && d < strides.Length ? strides[d] : 1;
            var dilation = dilations is not null && d < dilations.Length ? dilations[d] : 1;
            if (autoPad == AutoPad.SameUpper || autoPad == AutoPad.SameLower)
            {
                outDims[d + 2] = (inSize + stride - 1) / stride;
            }
            else
            {
                long padBegin = 0, padEnd = 0;
                if (autoPad != AutoPad.Valid)
                {
                    padBegin = pads is not null && d < pads.Length ? pads[d] : 0;
                    padEnd = pads is not null && d + spatialRank < pads.Length ? pads[d + spatialRank] : 0;
                }
                var effectiveK = (kSize - 1) * dilation + 1;
                outDims[d + 2] = (inSize + padBegin + padEnd - effectiveK) / stride + 1;
            }
        }
        return [RuntimeTensorFactory.Create(dtype, new Shape(outDims))];
    }
}

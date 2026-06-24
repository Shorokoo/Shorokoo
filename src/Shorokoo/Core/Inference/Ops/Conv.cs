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

internal sealed class ConvOp : QuickOp
{
    public override string OpCode => OpCodes.CONV;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs[0];
        var w = inputs[1];
        var dtype = x?.DType ?? DType.Float32;
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
                // ONNX SAME_UPPER/SAME_LOWER: output spatial = ceil(input / stride).
                outDims[d + 2] = (inSize + stride - 1) / stride;
            }
            else
            {
                // VALID is "no padding"; NotSet uses the explicit pads array.
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

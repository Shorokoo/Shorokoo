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

internal sealed class ConvTransposeOp : QuickOp
{
    public override string OpCode => OpCodes.CONV_TRANSPOSE;

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
        var outputPadding = attrs.GetIntsVal(OnnxOpAttributeNames.AttrOutputPadding);
        var outputShape = attrs.GetIntsVal(OnnxOpAttributeNames.AttrOutputShape);
        var group = attrs.GetLongVal(OnnxOpAttributeNames.AttrGroup) ?? 1;
        var autoPad = attrs.GetAttributeObj(OnnxOpAttributeNames.AttrAutoPad) as AutoPad?
                      ?? AutoPad.NotSet;

        var outDims = new long[xDims.Length];
        outDims[0] = xDims[0];
        outDims[1] = wDims[1] * group;
        for (int d = 0; d < spatialRank; d++)
        {
            // The output_shape attribute overrides the arithmetic outright (the pads are then
            // auto-generated to match). Tolerate both the spec's spatial-rank form and a
            // full-rank [N, C, ...spatial] form by reading entries back from the tail.
            if (outputShape is not null && outputShape.Length >= spatialRank)
            {
                outDims[d + 2] = outputShape[outputShape.Length - spatialRank + d];
                continue;
            }

            var stride = strides is not null && d < strides.Length ? strides[d] : 1;
            if (autoPad == AutoPad.SameUpper || autoPad == AutoPad.SameLower)
            {
                // SAME_*: pads are computed so output = input * stride.
                outDims[d + 2] = xDims[d + 2] * stride;
                continue;
            }

            // VALID is "no padding"; NotSet uses the explicit pads array.
            long padBegin = 0, padEnd = 0;
            if (autoPad != AutoPad.Valid)
            {
                padBegin = pads is not null && d < pads.Length ? pads[d] : 0;
                padEnd = pads is not null && d + spatialRank < pads.Length ? pads[d + spatialRank] : 0;
            }
            var dilation = dilations is not null && d < dilations.Length ? dilations[d] : 1;
            var effectiveK = (wDims[d + 2] - 1) * dilation + 1;
            var outPad = outputPadding is not null && d < outputPadding.Length ? outputPadding[d] : 0;
            outDims[d + 2] = stride * (xDims[d + 2] - 1) + outPad + effectiveK - padBegin - padEnd;
        }
        return [RuntimeTensorFactory.Create(dtype, new Shape(outDims))];
    }
}

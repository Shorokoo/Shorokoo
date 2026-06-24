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
/// Shape inference for Shorokoo's <c>SHRK_CONV</c> (the Conv variant whose geometry —
/// pads, strides, dilations, kernel_shape, group — flows as tensor inputs rather than
/// static attributes). Same per-spatial-dim arithmetic as <see cref="ConvOp"/>.
/// </summary>
internal sealed class ShrkConvOp : QuickOp
{
    public override string OpCode => InternalOpCodes.SHRK_CONV;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        // Inputs: [x, w, b, pads, strides, dilations, kernel_shape, group]
        var x = inputs.Length > 0 ? inputs[0] : null;
        var w = inputs.Length > 1 ? inputs[1] : null;
        var padsT = inputs.Length > 3 ? inputs[3] : null;
        var stridesT = inputs.Length > 4 ? inputs[4] : null;
        var dilationsT = inputs.Length > 5 ? inputs[5] : null;
        var dtype = x?.DType ?? DType.Float32;
        if (x?.Shape is null || w?.Shape is null) return [RuntimeTensorFactory.Create(dtype, null)];

        var xDims = x.Shape.Dims;
        var wDims = w.Shape.Dims;
        var spatialRank = xDims.Length - 2;
        var pads = padsT?.IntData?.ToArray();
        var strides = stridesT?.IntData?.ToArray();
        var dilations = dilationsT?.IntData?.ToArray();
        var autoPad = attrs.GetAttributeObj(OnnxOpAttributeNames.AttrAutoPad) as AutoPad?
                      ?? AutoPad.NotSet;

        // Geometry inputs that are connected but value-unknown at QEE time make the spatial
        // dims unknowable — degrade to rank-only instead of assuming default geometry
        // (which would produce a wrong concrete shape). Absent inputs keep the defaults.
        if ((padsT is not null && pads is null && autoPad is AutoPad.NotSet)
            || (stridesT is not null && strides is null)
            || (dilationsT is not null && dilations is null)
            || (strides is not null && strides.Any(s => s <= 0))
            || (dilations is not null && dilations.Any(d => d <= 0)))
            return [RuntimeTensorFactory.Create(dtype, null) with
            {
                Rank = xDims.Length,
                MaxRank = xDims.Length,
            }];

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
                if (autoPad != AutoPad.Valid && pads is not null)
                {
                    padBegin = d < pads.Length ? pads[d] : 0;
                    padEnd = d + spatialRank < pads.Length ? pads[d + spatialRank] : 0;
                }
                var effectiveK = (kSize - 1) * dilation + 1;
                outDims[d + 2] = (inSize + padBegin + padEnd - effectiveK) / stride + 1;
            }
        }
        return [RuntimeTensorFactory.Create(dtype, new Shape(outDims))];
    }
}

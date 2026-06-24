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
/// Shape inference for ONNX <c>MaxUnpool</c>. Output spatial dims are taken from the
/// <c>output_shape</c> input when provided; otherwise computed by the inverse of MaxPool
/// (<c>stride * (in - 1) + kernel - 2 * pad</c>). Batch and channel dims pass through.
/// </summary>
internal sealed class MaxUnpoolOp : QuickOp
{
    public override string OpCode => OpCodes.MAX_UNPOOL;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var outputShape = inputs.Length > 2 ? inputs[2] : null;
        var dtype = x?.DType ?? DType.Float32;
        if (x?.Shape is null) return [RuntimeTensorFactory.Create(dtype, null)];

        if (outputShape is not null)
        {
            // The optional output_shape input overrides the inverse-pool arithmetic outright.
            // When it's connected but its values aren't known at QEE time, the output dims are
            // unknown — falling back to the formula could disagree with the actual shape.
            if (outputShape.IntData is { } sd && sd.Length > 0)
                return [RuntimeTensorFactory.Create(dtype, new Shape(sd.ToArray()))];
            return [RuntimeTensorFactory.Create(dtype, null)];
        }

        var xDims = x.Shape.Dims;
        var kernelShape = attrs.GetIntsVal(OnnxOpAttributeNames.AttrKernelShape);
        var strides = attrs.GetIntsVal(OnnxOpAttributeNames.AttrStrides);
        var pads = attrs.GetIntsVal(OnnxOpAttributeNames.AttrPads);

        var spatialRank = xDims.Length - 2;
        var outDims = new long[xDims.Length];
        outDims[0] = xDims[0];
        outDims[1] = xDims[1];
        for (int d = 0; d < spatialRank; d++)
        {
            var stride = strides is not null && d < strides.Length ? strides[d] : 1;
            var kSize = kernelShape is not null && d < kernelShape.Length ? kernelShape[d] : 1;
            var padBegin = pads is not null && d < pads.Length ? pads[d] : 0;
            var padEnd = pads is not null && d + spatialRank < pads.Length ? pads[d + spatialRank] : 0;
            outDims[d + 2] = stride * (xDims[d + 2] - 1) + kSize - padBegin - padEnd;
        }
        return [RuntimeTensorFactory.Create(dtype, new Shape(outDims))];
    }
}

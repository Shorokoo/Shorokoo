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

internal sealed class MaxPoolOp : QuickOp
{
    public override string OpCode => OpCodes.MAX_POOL;

    // MaxPool produces up to 2 outputs: (values, indices). We always return both; the engine
    // drops the indices tensor if the graph didn't declare that output.
    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs[0];
        var dtype = x?.DType ?? DType.Float32;
        if (x?.Shape is null)
            return [RuntimeTensorFactory.Create(dtype, null), RuntimeTensorFactory.Create(DType.Int64, null)];

        var xDims = x.Shape.Dims;
        var kernelShape = attrs.GetIntsVal(OnnxOpAttributeNames.AttrKernelShape);
        var strides = attrs.GetIntsVal(OnnxOpAttributeNames.AttrStrides);
        var pads = attrs.GetIntsVal(OnnxOpAttributeNames.AttrPads);
        var dilations = attrs.GetIntsVal(OnnxOpAttributeNames.AttrDilations);
        var autoPad = attrs.GetAttributeObj(OnnxOpAttributeNames.AttrAutoPad) as AutoPad?
                      ?? AutoPad.NotSet;
        var ceilMode = AttrAccess.GetBool(attrs, OnnxOpAttributeNames.AttrCeilMode);
        var spatialRank = xDims.Length - 2;
        var outDims = new long[xDims.Length];
        outDims[0] = xDims[0]; outDims[1] = xDims[1];
        for (int d = 0; d < spatialRank; d++)
        {
            var kSize = kernelShape is not null && d < kernelShape.Length ? kernelShape[d] : 1;
            var stride = strides is not null && d < strides.Length ? strides[d] : 1;
            var dilation = dilations is not null && d < dilations.Length ? dilations[d] : 1;
            var padBegin = pads is not null && d < pads.Length ? pads[d] : 0;
            var padEnd = pads is not null && d + spatialRank < pads.Length ? pads[d + spatialRank] : 0;
            outDims[d + 2] = ShapeHelpers.PoolOutputSize(
                xDims[d + 2], kSize, stride, dilation, padBegin, padEnd, autoPad, ceilMode);
        }
        var shape = new Shape(outDims);
        return [
            RuntimeTensorFactory.Create(dtype, shape),
            RuntimeTensorFactory.Create(DType.Int64, shape),
        ];
    }
}

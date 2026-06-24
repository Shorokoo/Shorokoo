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
/// Shape inference for ONNX <c>MaxRoiPool</c>. Output is <c>[num_rois, C, *pooled_shape...]</c>:
/// channels come from <c>X</c>, batch from the <c>rois</c> tensor's leading dim, and the
/// pooled spatial dims are listed in the <c>pooled_shape</c> attribute.
/// </summary>
internal sealed class MaxRoiPoolOp : QuickOp
{
    public override string OpCode => OpCodes.MAX_ROI_POOL;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var rois = inputs.Length > 1 ? inputs[1] : null;
        var dtype = x?.DType ?? DType.Float32;
        if (x?.Shape is null) return [RuntimeTensorFactory.Create(dtype, null)];

        var pooled = attrs.GetIntsVal(OnnxOpAttributeNames.AttrPooledShape);
        if (pooled is null) return [RuntimeTensorFactory.Create(dtype, null)];

        var xDims = x.Shape.Dims;
        long numRois = rois?.Shape?.Dims is { Length: > 0 } rDims ? rDims[0] : -1;
        var outDims = new long[2 + pooled.Length];
        outDims[0] = numRois;
        outDims[1] = xDims.Length > 1 ? xDims[1] : -1;
        for (int i = 0; i < pooled.Length; i++) outDims[2 + i] = pooled[i];
        return [RuntimeTensorFactory.Create(dtype, new Shape(outDims))];
    }
}

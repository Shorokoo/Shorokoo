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
/// Shape inference for ONNX <c>GridSample</c>. Input <c>X</c> is <c>[N, C, *in_spatial...]</c>
/// and <c>grid</c> is <c>[N, *out_spatial..., D]</c>. Output is <c>[N, C, *out_spatial...]</c>:
/// the spatial dims come from <c>grid</c>, while batch and channel come from <c>X</c>.
/// </summary>
internal sealed class GridSampleOp : QuickOp
{
    public override string OpCode => OpCodes.GRID_SAMPLE;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var grid = inputs.Length > 1 ? inputs[1] : null;
        var dtype = x?.DType ?? DType.Float32;
        if (x?.Shape is null || grid?.Shape is null) return [RuntimeTensorFactory.Create(dtype, null)];

        var xDims = x.Shape.Dims;
        var gDims = grid.Shape.Dims;
        if (xDims.Length < 3 || gDims.Length < 3) return [RuntimeTensorFactory.Create(dtype, null)];

        var spatialRank = gDims.Length - 2; // drop batch and trailing coord dim
        var outDims = new long[2 + spatialRank];
        outDims[0] = xDims[0];
        outDims[1] = xDims[1];
        for (int i = 0; i < spatialRank; i++) outDims[2 + i] = gDims[1 + i];
        return [RuntimeTensorFactory.Create(dtype, new Shape(outDims))];
    }
}

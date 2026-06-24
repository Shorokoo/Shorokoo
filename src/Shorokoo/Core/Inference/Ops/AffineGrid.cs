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
/// Shape inference for ONNX <c>AffineGrid</c>. Output is <c>[N, *spatial..., D]</c>
/// where the spatial dims and <c>D</c> are read from the <c>size</c> tensor input.
/// Layout for the 2-D case: <c>[N, H, W, 2]</c>; for 3-D: <c>[N, D, H, W, 3]</c>.
/// </summary>
internal sealed class AffineGridOp : QuickOp
{
    public override string OpCode => OpCodes.AFFINE_GRID;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var theta = inputs.Length > 0 ? inputs[0] : null;
        var size = inputs.Length > 1 ? inputs[1] : null;
        var dtype = theta?.DType ?? DType.Float32;

        if (size?.IntData is not { } sd || sd.Length < 4)
            return [RuntimeTensorFactory.Create(dtype, null)];

        // size = [N, C, H, W] (2-D) or [N, C, D, H, W] (3-D); we drop C and append the
        // spatial dim count.
        var spatialRank = sd.Length - 2;
        var outDims = new long[2 + spatialRank];
        outDims[0] = sd[0];
        for (int i = 0; i < spatialRank; i++) outDims[1 + i] = sd[2 + i];
        outDims[outDims.Length - 1] = spatialRank;
        return [RuntimeTensorFactory.Create(dtype, new Shape(outDims))];
    }
}

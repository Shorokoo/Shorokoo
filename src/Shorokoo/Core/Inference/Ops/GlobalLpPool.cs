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
/// Shape inference for ONNX <c>GlobalLpPool</c>. Output collapses every spatial dim to 1,
/// matching the standard global-pool layout <c>[N, C, 1, 1, ...]</c>.
/// </summary>
internal sealed class GlobalLpPoolOp : QuickOp
{
    public override string OpCode => OpCodes.GLOBAL_LP_POOL;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var dtype = x?.DType ?? DType.Float32;
        if (x?.Shape is null) return [RuntimeTensorFactory.Create(dtype, null)];

        var xDims = x.Shape.Dims;
        var outDims = new long[xDims.Length];
        outDims[0] = xDims[0]; outDims[1] = xDims[1];
        for (int d = 2; d < xDims.Length; d++) outDims[d] = 1;
        return [RuntimeTensorFactory.Create(dtype, new Shape(outDims))];
    }
}

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

internal sealed class GlobalMaxPoolOp : QuickOp
{
    public override string OpCode => OpCodes.GLOBAL_MAX_POOL;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs[0];
        var dtype = x?.DType ?? DType.Float32;
        if (x?.Shape is null) return [RuntimeTensorFactory.Create(dtype, null)];
        var outDims = x.Shape.Dims.ToArray();
        for (int d = 2; d < outDims.Length; d++) outDims[d] = 1;
        return [RuntimeTensorFactory.Create(dtype, new Shape(outDims))];
    }
}

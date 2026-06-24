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

internal sealed class ShrkRandomNormalOp : QuickOp
{
    public override string OpCode => InternalOpCodes.SHRK_RANDOM_NORMAL;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var shapeInput = inputs[0];
        var dtype = DType.Float32;
        Shape? shape = shapeInput?.IntData is { } s && s.All(d => d >= 0) ? new Shape(s.ToArray()) : null;
        var rt = RuntimeTensorFactory.Create(dtype, shape);
        // Shape values unknown but the shape input's own 1-D extent gives the output rank.
        if (shape is null && shapeInput?.Shape?.Dims is { Length: 1 } sd)
            rt = rt with { Rank = (int)sd[0], MaxRank = (int)sd[0] };
        return [rt];
    }
}

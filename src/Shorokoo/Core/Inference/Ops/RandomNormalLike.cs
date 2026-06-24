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

internal sealed class RandomNormalLikeOp : QuickOp
{
    public override string OpCode => OpCodes.RANDOM_NORMAL_LIKE;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs[0];
        var dtype = attrs.GetDTypeVal(OnnxOpAttributeNames.AttrDtype) ?? x?.DType ?? DType.Float32;
        return [RuntimeTensorFactory.Create(dtype, x?.Shape)];
    }
}

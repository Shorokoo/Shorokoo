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
/// Shape inference for ONNX <c>Bernoulli</c>. Output has the same shape as input; dtype
/// optionally overridden by the <c>dtype</c> attribute (defaults to input dtype).
/// </summary>
internal sealed class BernoulliOp : QuickOp
{
    public override string OpCode => OpCodes.BERNOULLI;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var dtype = attrs.GetDTypeVal(OnnxOpAttributeNames.AttrDtype) ?? x?.DType ?? DType.Float32;
        return [RuntimeTensorFactory.Create(dtype, x?.Shape)];
    }
}

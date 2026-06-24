using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// Shape inference for ONNX <c>RegexFullMatch</c>. Output is a bool tensor with the same
/// shape as the string input.
/// </summary>
internal sealed class RegexFullMatchOp : QuickOp
{
    public override string OpCode => OpCodes.REGEX_FULL_MATCH;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        return [RuntimeTensorFactory.Create(DType.Bool, x?.Shape)];
    }
}

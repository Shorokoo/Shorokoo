using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using Shorokoo;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// Shape inference for ONNX <c>StringConcat</c>. Element-wise string concatenation with
/// ONNX broadcasting; dtype is the string dtype.
/// </summary>
internal sealed class StringConcatOp : QuickOp
{
    public override string OpCode => OpCodes.STRING_CONCAT;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var a = inputs.Length > 0 ? inputs[0] : null;
        var b = inputs.Length > 1 ? inputs[1] : null;
        var dtype = a?.DType ?? b?.DType ?? DType.String;
        var shape = ShapeHelpers.Broadcast(a?.Shape, b?.Shape);
        return [RuntimeTensorFactory.Create(dtype, shape)];
    }
}

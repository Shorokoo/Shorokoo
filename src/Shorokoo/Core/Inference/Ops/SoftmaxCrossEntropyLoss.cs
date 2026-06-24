using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// Shape inference for ONNX <c>SoftmaxCrossEntropyLoss</c>. Two outputs:
///   loss: scalar when reduction is "mean"/"sum" (the default), otherwise the labels' shape;
///   log_prob: the scores' shape, same dtype as scores.
/// </summary>
internal sealed class SoftmaxCrossEntropyLossOp : QuickOp
{
    public override string OpCode => OpCodes.SOFTMAX_CROSS_ENTROPY_LOSS;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var scores = inputs.Length > 0 ? inputs[0] : null;
        var labels = inputs.Length > 1 ? inputs[1] : null;
        var dtype = scores?.DType ?? DType.Float32;

        var reduction = attrs.GetAttributeObj(OnnxOpAttributeNames.AttrReduction) as string ?? "mean";
        Shape? lossShape = reduction == "none"
            ? labels?.Shape
            : new Shape(Array.Empty<long>());

        return [
            RuntimeTensorFactory.Create(dtype, lossShape),
            RuntimeTensorFactory.Create(dtype, scores?.Shape),
        ];
    }
}

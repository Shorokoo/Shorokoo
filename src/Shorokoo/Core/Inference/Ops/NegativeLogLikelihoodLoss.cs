using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// Shape inference for ONNX <c>NegativeLogLikelihoodLoss</c>. Input is <c>[N, C, d1, …]</c>;
/// target is <c>[N, d1, …]</c>. With <c>reduction = "none"</c> the output has the target's
/// shape; with <c>"mean"</c> or <c>"sum"</c> the output is a scalar.
/// </summary>
internal sealed class NegativeLogLikelihoodLossOp : QuickOp
{
    public override string OpCode => OpCodes.NEGATIVE_LOG_LIKELIHOOD_LOSS;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var input = inputs.Length > 0 ? inputs[0] : null;
        var target = inputs.Length > 1 ? inputs[1] : null;
        var dtype = input?.DType ?? DType.Float32;

        var reduction = attrs.GetAttributeObj(OnnxOpAttributeNames.AttrReduction) as string ?? "mean";
        if (reduction == "none")
            return [RuntimeTensorFactory.Create(dtype, target?.Shape)];

        return [RuntimeTensorFactory.Create(dtype, new Shape(Array.Empty<long>()))];
    }
}

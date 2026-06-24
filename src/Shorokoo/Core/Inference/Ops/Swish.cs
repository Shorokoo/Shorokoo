using System.Collections.Immutable;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Inference.Helpers;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// QEE kernel for ONNX <c>Swish</c> (opset 24+): elementwise
/// <c>y = x * sigmoid(alpha * x)</c> with <c>alpha</c> defaulting to 1.
/// Output shape and dtype match the input; values are computed for small float
/// tensors. NOTE: ONNX Runtime 1.26 has no Swish kernel on any execution
/// provider, so the QEE is the only execution path for this op.
/// </summary>
internal sealed class SwishOp : QuickOp
{
    public override string OpCode => OpCodes.SWISH;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var alpha = attrs.GetFloatVal(OnnxOpAttributeNames.AttrAlpha) ?? 1.0f;
        var dtype = x?.DType ?? DType.Float32;
        var rt = RuntimeTensorFactory.Create(dtype, x?.Shape);
        if (x?.FloatData is { } fd && RuntimeTensorFactory.ShouldStoreData(rt.Shape, maxDataElements))
        {
            var d = new float[fd.Length];
            for (int i = 0; i < d.Length; i++)
            {
                var v = fd[i];
                d[i] = v * (1f / (1f + MathF.Exp(-alpha * v)));
            }
            return [rt with { FloatData = ImmutableArray.Create(d) }];
        }
        return [rt];
    }
}

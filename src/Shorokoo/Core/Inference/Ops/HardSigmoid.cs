using System.Collections.Immutable;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// Elementwise <c>HardSigmoid(x) = max(0, min(1, alpha·x + beta))</c>. Alpha defaults to
/// 0.2, beta to 0.5 per the ONNX spec.
/// </summary>
internal sealed class HardSigmoidOp : QuickOp
{
    public override string OpCode => OpCodes.HARD_SIGMOID;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var dtype = x?.DType ?? DType.Float32;
        var rt = RuntimeTensorFactory.Create(dtype, x?.Shape);

        if (x?.FloatData is { } fd && RuntimeTensorFactory.ShouldStoreData(x.Shape, maxDataElements))
        {
            var alpha = attrs.GetFloatVal(OnnxOpAttributeNames.AttrAlpha) ?? 0.2f;
            var beta = attrs.GetFloatVal(OnnxOpAttributeNames.AttrBeta) ?? 0.5f;
            var buf = new float[fd.Length];
            for (int i = 0; i < buf.Length; i++)
            {
                var t = alpha * fd[i] + beta;
                if (t < 0f) t = 0f;
                else if (t > 1f) t = 1f;
                buf[i] = t;
            }
            return [rt with { FloatData = ImmutableArray.Create(buf) }];
        }
        return [rt];
    }
}

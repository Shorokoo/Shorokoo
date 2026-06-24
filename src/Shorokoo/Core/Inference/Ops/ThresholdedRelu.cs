using System.Collections.Immutable;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// Elementwise <c>ThresholdedRelu(x) = x if x &gt; alpha, else 0</c>. Alpha defaults to 1.0.
/// </summary>
internal sealed class ThresholdedReluOp : QuickOp
{
    public override string OpCode => OpCodes.THRESHOLDED_RELU;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var dtype = x?.DType ?? DType.Float32;
        var rt = RuntimeTensorFactory.Create(dtype, x?.Shape);

        if (x?.FloatData is { } fd && RuntimeTensorFactory.ShouldStoreData(x.Shape, maxDataElements))
        {
            var alpha = attrs.GetFloatVal(OnnxOpAttributeNames.AttrAlpha) ?? 1.0f;
            var buf = new float[fd.Length];
            for (int i = 0; i < buf.Length; i++) buf[i] = fd[i] > alpha ? fd[i] : 0f;
            return [rt with { FloatData = ImmutableArray.Create(buf) }];
        }
        return [rt];
    }
}

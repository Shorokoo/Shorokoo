using System.Collections.Immutable;
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

internal sealed class SeluOp : QuickOp
{
    public override string OpCode => OpCodes.SELU;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs[0];
        var alpha = attrs.GetFloatVal(OnnxOpAttributeNames.AttrAlpha) ?? 1.67326319217681884765625f;
        var gamma = attrs.GetFloatVal(OnnxOpAttributeNames.AttrGamma) ?? 1.0507010221481323242187f;
        var dtype = x?.DType ?? DType.Float32;
        var rt = RuntimeTensorFactory.Create(dtype, x?.Shape);
        if (x?.FloatData is { } fd && RuntimeTensorFactory.ShouldStoreData(rt.Shape, maxDataElements))
        {
            var d = new float[fd.Length];
            for (int i = 0; i < d.Length; i++)
            {
                var v = fd[i];
                d[i] = gamma * (v > 0 ? v : alpha * (MathF.Exp(v) - 1f));
            }
            return [rt with { FloatData = ImmutableArray.Create(d) }];
        }
        return [rt];
    }
}

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

internal sealed class GeluOp : QuickOp
{
    public override string OpCode => OpCodes.GELU;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs[0];
        var dtype = x?.DType ?? DType.Float32;
        var rt = RuntimeTensorFactory.Create(dtype, x?.Shape);
        if (x?.FloatData is { } fd && RuntimeTensorFactory.ShouldStoreData(rt.Shape, maxDataElements))
        {
            // approximate defaults to "none" per spec → exact erf-based Gelu. The op used
            // to compute the tanh approximation unconditionally.
            var useTanh = ResolveApproximate(attrs);
            var d = new float[fd.Length];
            for (int i = 0; i < d.Length; i++)
            {
                var v = fd[i];
                d[i] = useTanh
                    ? 0.5f * v * (1f + MathF.Tanh(0.7978845608f * (v + 0.044715f * v * v * v)))
                    : 0.5f * v * (1f + ErfOp.Erf(v / 1.4142135624f));
            }
            return [rt with { FloatData = ImmutableArray.Create(d) }];
        }
        return [rt];
    }

    private static bool ResolveApproximate(OnnxCSharpAttributes attrs)
    {
        if (!attrs.IsAttributeDefined(OnnxOpAttributeNames.AttrApproximate)) return false;
        var obj = attrs.GetAttributeObj(OnnxOpAttributeNames.AttrApproximate);
        return obj switch
        {
            GeluApproximate g => g == GeluApproximate.Tanh,
            string s => s.Equals("tanh", StringComparison.OrdinalIgnoreCase),
            _ => false,
        };
    }
}

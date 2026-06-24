using System.Collections.Immutable;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// Elementwise <c>Shrink</c>: <c>y = x - bias if x &gt; lambd, x + bias if x &lt; -lambd, else 0</c>.
/// Defaults: bias = 0, lambd = 0.5.
/// </summary>
internal sealed class ShrinkOp : QuickOp
{
    public override string OpCode => OpCodes.SHRINK;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var dtype = x?.DType ?? DType.Float32;
        var rt = RuntimeTensorFactory.Create(dtype, x?.Shape);

        var bias = attrs.GetFloatVal(OnnxOpAttributeNames.AttrBias) ?? 0f;
        var lambd = attrs.GetFloatVal(OnnxOpAttributeNames.AttrLambd) ?? 0.5f;

        if (x?.FloatData is { } fd && RuntimeTensorFactory.ShouldStoreData(x.Shape, maxDataElements))
        {
            var buf = new float[fd.Length];
            for (int i = 0; i < buf.Length; i++)
            {
                var v = fd[i];
                buf[i] = v > lambd ? v - bias : v < -lambd ? v + bias : 0f;
            }
            return [rt with { FloatData = ImmutableArray.Create(buf) }];
        }
        if (x?.IntData is { } id && RuntimeTensorFactory.ShouldStoreData(x.Shape, maxDataElements))
        {
            var biasL = (long)bias;
            var lambdL = (long)lambd;
            var buf = new long[id.Length];
            for (int i = 0; i < buf.Length; i++)
            {
                var v = id[i];
                buf[i] = v > lambdL ? v - biasL : v < -lambdL ? v + biasL : 0L;
            }
            return [rt with { IntData = ImmutableArray.Create(buf) }];
        }
        return [rt];
    }
}

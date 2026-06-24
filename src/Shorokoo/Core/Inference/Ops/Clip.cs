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

internal sealed class ClipOp : QuickOp
{
    public override string OpCode => OpCodes.CLIP;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs[0];
        var min = inputs.Length > 1 ? inputs[1] : null;
        var max = inputs.Length > 2 ? inputs[2] : null;
        var dtype = x?.DType ?? DType.Float32;
        var rt = RuntimeTensorFactory.Create(dtype, x?.Shape);
        if (x is null || !RuntimeTensorFactory.ShouldStoreData(rt.Shape, maxDataElements))
            return [rt];

        if (x.FloatData is { } fd)
        {
            // A connected min/max input whose value is unknown at QEE time blocks value
            // computation entirely (the old code silently treated it as ±inf).
            bool minKnown = min is null || min.FloatData is { Length: > 0 };
            bool maxKnown = max is null || max.FloatData is { Length: > 0 };
            if (!minKnown || !maxKnown) return [rt];

            float lo = min?.FloatData is { Length: > 0 } lf ? lf[0] : float.NegativeInfinity;
            float hi = max?.FloatData is { Length: > 0 } hf ? hf[0] : float.PositiveInfinity;
            var d = new float[fd.Length];
            // Manual clamp: Math.Clamp throws when lo > hi, while ONNX (numpy-style)
            // lets max win in that case.
            for (int i = 0; i < d.Length; i++)
            {
                var v = fd[i];
                if (v < lo) v = lo;
                if (v > hi) v = hi;
                d[i] = v;
            }
            return [rt with { FloatData = ImmutableArray.Create(d) }];
        }
        if (x.IntData is { } id)
        {
            bool minKnown = min is null || min.IntData is { Length: > 0 };
            bool maxKnown = max is null || max.IntData is { Length: > 0 };
            if (!minKnown || !maxKnown) return [rt];

            long lo = min?.IntData is { Length: > 0 } li ? li[0] : long.MinValue;
            long hi = max?.IntData is { Length: > 0 } hf ? hf[0] : long.MaxValue;
            var d = new long[id.Length];
            for (int i = 0; i < d.Length; i++)
            {
                var v = id[i];
                if (v < lo) v = lo;
                if (v > hi) v = hi;
                d[i] = v;
            }
            return [rt with { IntData = ImmutableArray.Create(d) }];
        }
        return [rt];
    }
}

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

internal sealed class SqueezeOp : QuickOp
{
    public override string OpCode => OpCodes.SQUEEZE;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs[0];
        var dtype = x?.DType ?? DType.Float32;
        if (x?.Shape is null) return [RuntimeTensorFactory.Create(dtype, null)];

        var dims = x.Shape.Dims.ToList();
        long[]? axes = null;
        var axesInput = inputs.Length > 1 ? inputs[1] : null;
        if (axesInput?.IntData is { } idxData) axes = idxData.ToArray();
        else if (axesInput is not null)
            // axes connected but unknown at QEE time: which size-1 dims get removed is
            // unknowable (it can be any subset), so degrade to an unknown shape.
            return [RuntimeTensorFactory.Create(dtype, null)];
        if (axes is null) axes = AttrAccess.GetLongs(attrs, OnnxOpAttributeNames.AttrAxes);

        if (axes is null)
        {
            dims.RemoveAll(d => d == 1);
        }
        else
        {
            foreach (var ax in axes.OrderByDescending(a => a))
            {
                var idx = ax < 0 ? ax + dims.Count : ax;
                if (idx >= 0 && idx < dims.Count && dims[(int)idx] == 1)
                    dims.RemoveAt((int)idx);
            }
        }
        return [BuildResult(rt: RuntimeTensorFactory.Create(dtype, new Shape(dims.ToArray())), src: x, maxDataElements)];
    }

    private static RuntimeTensor BuildResult(RuntimeTensor rt, RuntimeTensor src, int maxDataElements)
    {
        // Squeeze only removes size-1 dims — data layout is unchanged, so propagate concrete data.
        if (RuntimeTensorFactory.ShouldStoreData(rt.Shape, maxDataElements))
        {
            if (src.IntData is { } id) return rt with { IntData = id };
            if (src.FloatData is { } fd) return rt with { FloatData = fd };
            if (src.BoolData is { } bd) return rt with { BoolData = bd };
        }
        return rt;
    }
}

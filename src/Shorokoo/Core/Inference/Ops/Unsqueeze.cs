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

internal sealed class UnsqueezeOp : QuickOp
{
    public override string OpCode => OpCodes.UNSQUEEZE;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs[0];
        var dtype = x?.DType ?? DType.Float32;
        long[]? axes = null;
        if (inputs.Length > 1 && inputs[1]?.IntData is { } idxData) axes = idxData.ToArray();
        if (axes is null) axes = AttrAccess.GetLongs(attrs, OnnxOpAttributeNames.AttrAxes);
        if (x?.Shape is null || axes is null)
            return [RuntimeTensorFactory.Create(dtype, null)];

        var dims = x.Shape.Dims.ToList();
        var newRank = dims.Count + axes.Length;
        var sorted = axes.Select(a => a < 0 ? a + newRank : a).OrderBy(a => a).ToArray();
        foreach (var ax in sorted) dims.Insert((int)ax, 1L);
        var rt = RuntimeTensorFactory.Create(dtype, new Shape(dims.ToArray()));
        // Unsqueeze only inserts size-1 dims — data layout is unchanged, so propagate concrete data.
        if (RuntimeTensorFactory.ShouldStoreData(rt.Shape, maxDataElements))
        {
            if (x.IntData is { } id) return [rt with { IntData = id }];
            if (x.FloatData is { } fd) return [rt with { FloatData = fd }];
            if (x.BoolData is { } bd) return [rt with { BoolData = bd }];
        }
        return [rt];
    }
}

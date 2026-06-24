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

internal sealed class FlattenOp : QuickOp
{
    public override string OpCode => OpCodes.FLATTEN;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs[0];
        var axis = (int)(attrs.GetLongVal(OnnxOpAttributeNames.AttrAxis) ?? 1);
        var dtype = x?.DType ?? DType.Float32;
        if (x?.Shape is null)
            return [RuntimeTensorFactory.Create(dtype, null)];

        var rank = x.Shape.Dims.Length;
        if (axis < 0) axis += rank;
        if (axis < 0 || axis > rank) return [RuntimeTensorFactory.Create(dtype, null)];
        long d0 = 1, d1 = 1;
        for (int i = 0; i < axis; i++) d0 *= x.Shape.Dims[i];
        for (int i = axis; i < rank; i++) d1 *= x.Shape.Dims[i];
        var rt = RuntimeTensorFactory.Create(dtype, new Shape(new[] { d0, d1 }));
        // Flatten is a pure reshape — the data layout is unchanged, so propagate it.
        if (RuntimeTensorFactory.ShouldStoreData(rt.Shape, maxDataElements))
        {
            if (x.IntData is { } id) return [rt with { IntData = id }];
            if (x.FloatData is { } fd) return [rt with { FloatData = fd }];
            if (x.BoolData is { } bd) return [rt with { BoolData = bd }];
        }
        return [rt];
    }
}

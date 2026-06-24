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

internal sealed class OrOp : QuickOp
{
    public override string OpCode => OpCodes.OR;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var a = inputs.Length > 0 ? inputs[0] : null;
        var b = inputs.Length > 1 ? inputs[1] : null;
        var shape = ShapeHelpers.Broadcast(a?.Shape, b?.Shape);
        var rt = RuntimeTensorFactory.Create(DType.Bool, shape);

        if (shape is not null
            && RuntimeTensorFactory.ShouldStoreData(shape, maxDataElements)
            && a?.BoolData is { } ad
            && b?.BoolData is { } bd
            && a.Shape is not null
            && b.Shape is not null)
        {
            var data = ElementwiseBroadcast.Bool(ad, a.Shape, bd, b.Shape, shape, (x, y) => x || y);
            return [rt with { BoolData = ImmutableArray.Create(data) }];
        }

        return [rt];
    }
}

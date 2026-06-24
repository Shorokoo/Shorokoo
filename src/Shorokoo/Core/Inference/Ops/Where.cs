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

internal sealed class WhereOp : QuickOp
{
    public override string OpCode => OpCodes.WHERE;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var cond = inputs.Length > 0 ? inputs[0] : null;
        var then = inputs.Length > 1 ? inputs[1] : null;
        var @else = inputs.Length > 2 ? inputs[2] : null;

        var shape = ShapeHelpers.Broadcast(cond?.Shape, then?.Shape, @else?.Shape);
        var dtype = then?.DType ?? @else?.DType ?? DType.Float32;
        var rt = RuntimeTensorFactory.Create(dtype, shape);

        // Concrete-data paths for bool/int/float so the liveness analysis in
        // FastListAllSpecificModelIdsUsed (which uses Where to gate mask branches on a
        // constant-folded IF condition) gets concrete bits back.
        if (shape is not null
            && RuntimeTensorFactory.ShouldStoreData(shape, maxDataElements)
            && cond?.BoolData is { } condData
            && cond.Shape is not null
            && then?.Shape is not null
            && @else?.Shape is not null)
        {
            if (then.FloatData is { } tf && @else.FloatData is { } ef)
            {
                var data = ElementwiseBroadcast.FloatWhere(
                    condData, cond.Shape,
                    tf, then.Shape,
                    ef, @else.Shape, shape);
                return [rt with { FloatData = ImmutableArray.Create(data) }];
            }
            if (then.IntData is { } ti && @else.IntData is { } ei)
            {
                var data = ElementwiseBroadcast.IntWhere(
                    condData, cond.Shape,
                    ti, then.Shape,
                    ei, @else.Shape, shape);
                return [rt with { IntData = ImmutableArray.Create(data) }];
            }
            if (dtype == DType.Bool && then.BoolData is { } tb && @else.BoolData is { } eb)
            {
                var data = ElementwiseBroadcast.BoolWhere(
                    condData, cond.Shape,
                    tb, then.Shape,
                    eb, @else.Shape, shape);
                return [rt with { BoolData = ImmutableArray.Create(data) }];
            }
        }

        return [rt];
    }
}

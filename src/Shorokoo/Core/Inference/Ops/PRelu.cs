using System.Collections.Immutable;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// ONNX <c>PRelu</c>: <c>y = x &lt; 0 ? slope * x : x</c>. The slope broadcasts
/// UNIDIRECTIONALLY to x, so for valid models the broadcast of the two shapes equals
/// x's shape; dtype matches x. Values are computed when both inputs carry data.
/// </summary>
internal sealed class PReluOp : QuickOp
{
    public override string OpCode => OpCodes.P_RELU;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var slope = inputs.Length > 1 ? inputs[1] : null;
        var dtype = x?.DType ?? slope?.DType ?? DType.Float32;
        var shape = ShapeHelpers.Broadcast(x?.Shape, slope?.Shape);
        var rt = RuntimeTensorFactory.Create(dtype, shape);

        if (shape is not null && RuntimeTensorFactory.ShouldStoreData(shape, maxDataElements)
            && x is not null && slope is not null)
        {
            if (x.FloatData is { } xf && slope.FloatData is { } sf)
                return [rt with { FloatData = ImmutableArray.Create(ElementwiseBroadcast.Float(
                    xf, x.Shape!, sf, slope.Shape!, shape, (v, s) => v < 0 ? s * v : v)) }];
            if (x.IntData is { } xi && slope.IntData is { } si)
                return [rt with { IntData = ImmutableArray.Create(ElementwiseBroadcast.Int(
                    xi, x.Shape!, si, slope.Shape!, shape, (v, s) => v < 0 ? s * v : v)) }];
        }
        return [rt];
    }
}

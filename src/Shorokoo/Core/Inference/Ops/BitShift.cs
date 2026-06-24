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

/// <summary>
/// QEE kernel for ONNX <c>BitShift</c>. Bitwise shift with broadcasting; direction comes
/// from the <c>direction</c> attribute (defaults to LEFT per ONNX).
/// </summary>
internal sealed class BitShiftOp : QuickOp
{
    public override string OpCode => OpCodes.BIT_SHIFT;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var a = inputs.Length > 0 ? inputs[0] : null;
        var b = inputs.Length > 1 ? inputs[1] : null;
        var dtype = a?.DType ?? b?.DType ?? DType.Int64;
        var shape = ShapeHelpers.Broadcast(a?.Shape, b?.Shape);

        ImmutableArray<long>? iData = null;
        if (shape is not null && RuntimeTensorFactory.ShouldStoreData(shape, maxDataElements)
            && a?.IntData is { } ai && b?.IntData is { } bi)
        {
            var direction = ResolveDirection(attrs);
            iData = ImmutableArray.Create(ElementwiseBroadcast.Int(ai, a.Shape!, bi, b.Shape!, shape,
                direction == BitShiftDirection.Left
                    ? (long x, long y) => x << (int)y
                    : (long x, long y) => (long)((ulong)x >> (int)y)));
        }
        return [RuntimeTensorFactory.Create(dtype, shape) with { IntData = iData }];
    }

    private static BitShiftDirection ResolveDirection(OnnxCSharpAttributes attrs)
    {
        if (!attrs.IsAttributeDefined(OnnxOpAttributeNames.AttrDirection)) return BitShiftDirection.Left;
        var obj = attrs.GetAttributeObj(OnnxOpAttributeNames.AttrDirection);
        return obj switch
        {
            BitShiftDirection d => d,
            string s when s.Equals("RIGHT", System.StringComparison.OrdinalIgnoreCase) => BitShiftDirection.Right,
            _ => BitShiftDirection.Left,
        };
    }
}

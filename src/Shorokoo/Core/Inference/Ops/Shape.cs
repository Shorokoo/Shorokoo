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

internal sealed class ShapeOp : QuickOp
{
    public override string OpCode => OpCodes.SHAPE;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs[0];
        var start = (int)(attrs.GetLongVal(OnnxOpAttributeNames.AttrStart) ?? 0);
        var endAttr = attrs.GetLongVal(OnnxOpAttributeNames.AttrEnd);

        if (x?.Shape is null)
            return [RuntimeTensorFactory.Create(DType.Int64, null)];

        var dims = x.Shape.Dims;
        var s = start < 0 ? start + dims.Length : start;
        var e = endAttr.HasValue ? (int)(endAttr.Value < 0 ? endAttr.Value + dims.Length : endAttr.Value) : dims.Length;
        s = Math.Clamp(s, 0, dims.Length);
        e = Math.Clamp(e, s, dims.Length);
        var slice = new long[e - s];
        Array.Copy(dims, s, slice, 0, e - s);
        return [RuntimeTensorFactory.Create(DType.Int64, new Shape(new long[] { slice.Length }))
            with { IntData = ImmutableArray.Create(slice) }];
    }
}

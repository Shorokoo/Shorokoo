using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Inference.Helpers;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// QEE implementation of the raw-bits runtime feed SHRK_RANDOM_BITS (before it is lowered to
/// the keyed draw): shape propagation like the float feeds, but the output dtype is the unsigned
/// width from the shrk_dtype attribute. The bit values themselves come from the lowered,
/// width-specialized "bits" function, not from this op.
/// </summary>
internal sealed class ShrkRandomBitsOp : QuickOp
{
    public override string OpCode => InternalOpCodes.SHRK_RANDOM_BITS;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var shapeInput = inputs[0];
        var dtype = attrs.GetDTypeVal(OnnxOpAttributeNames.ShrkAttrDtype) ?? DType.UInt32;
        Shape? shape = shapeInput?.IntData is { } s && s.All(d => d >= 0) ? new Shape(s.ToArray()) : null;
        var rt = RuntimeTensorFactory.Create(dtype, shape);
        // Shape values unknown but the shape input's own 1-D extent gives the output rank.
        if (shape is null && shapeInput?.Shape?.Dims is { Length: 1 } sd)
            rt = rt with { Rank = (int)sd[0], MaxRank = (int)sd[0] };
        return [rt];
    }
}

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

namespace Shorokoo.Core.Inference.Ops;

// SHRK_RNG_SPLIT has deliberately NO QEE op: RNG is graph-only (issue #136). A split is
// computed by executing the lowered graph — its algorithm `split` FUNCTION_INVOKE runs on
// ORT (whose session-build constant folding collapses constant key chains to literal keys),
// exactly as the RngSeed-rooted seed-derivation chain already resolves. The host never
// computes a split.

/// <summary>
/// QEE implementation of SHRK_RNG_UNIFORM / SHRK_RNG_NORMAL: shape/rank propagation only
/// (like the unkeyed random ops). Values are deliberately not computed in QEE: the normal
/// transform uses float transcendentals whose ULP behavior may differ from the execution
/// backend, and QEE-vs-backend comparisons must not flake. RNG is graph-only (#136): no QEE
/// op computes a draw or a split — values come from executing the lowered graph.
/// </summary>
internal abstract class ShrkRngDrawOpBase : QuickOp
{
    /// <summary>The draw's output dtype (uniform/normal are float32; bits is the shrk_dtype width).</summary>
    protected virtual DType OutputDType(OnnxCSharpAttributes attrs) => DType.Float32;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var shapeInput = inputs[2];
        Shape? shape = shapeInput?.IntData is { } s && s.All(d => d >= 0) ? new Shape(s.ToArray()) : null;
        var rt = RuntimeTensorFactory.Create(OutputDType(attrs), shape);
        // Shape values unknown but the shape input's own 1-D extent gives the output rank.
        if (shape is null && shapeInput?.Shape?.Dims is { Length: 1 } sd)
            rt = rt with { Rank = (int)sd[0], MaxRank = (int)sd[0] };
        return [rt];
    }
}

internal sealed class ShrkRngUniformOp : ShrkRngDrawOpBase
{
    public override string OpCode => InternalOpCodes.SHRK_RNG_UNIFORM;
}

internal sealed class ShrkRngNormalOp : ShrkRngDrawOpBase
{
    public override string OpCode => InternalOpCodes.SHRK_RNG_NORMAL;
}

/// <summary>
/// QEE implementation of SHRK_RNG_BITS: shape/rank propagation like the float draws, but the
/// output dtype is the unsigned width from the shrk_dtype attribute. Bit values themselves come
/// from the lowered, width-specialized "bits" function (integer/bit-exact), not from this op.
/// </summary>
internal sealed class ShrkRngBitsOp : ShrkRngDrawOpBase
{
    public override string OpCode => InternalOpCodes.SHRK_RNG_BITS;
    protected override DType OutputDType(OnnxCSharpAttributes attrs)
        => attrs.GetDTypeVal(OnnxOpAttributeNames.ShrkAttrDtype)
           ?? throw new InvalidOperationException("SHRK_RNG_BITS is missing its shrk_dtype (output width) attribute.");
}

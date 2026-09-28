using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

using static OnnxOp;
using static OnnxOpAttributeNames;
using static OpCodes;

/// <summary>
/// A reduction with <c>noop_with_empty_axes</c> 1 and no axes, or an empty axes tensor, which the
/// spec defines as its input passed through and ONNX Runtime's kernels do not compute as such.
/// Rewritten as <c>Identity</c> when the axes are absent or a constant empty tensor, and as an
/// <c>If</c> on the axes' element count otherwise. A <c>Constant</c> input that is not empty
/// keeps the call as it stands.
/// (Shorokoo/Shorokoo#409)
/// </summary>
internal sealed class ReduceNoopEmptyAxesWorkaround : KernelWorkaround
{
    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([REDUCE_L1, REDUCE_L2, REDUCE_LOG_SUM, REDUCE_LOG_SUM_EXP, REDUCE_MAX, REDUCE_MEAN, REDUCE_MIN, REDUCE_PROD, REDUCE_SUM, REDUCE_SUM_SQUARE], StringComparer.Ordinal);

    public override bool Applies(WorkaroundSite site)
        => site.Attributes.GetBoolVal(AttrNoopWithEmptyAxes) == true
           && (!site.IsPresent(1) || site.ConstantOf(1) is not { } axes || axes.Shape.Dims.Contains(0))
           && Reductions.InputMayBeEmpty(site);

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
    {
        var x = inputs[0]!;
        if (inputs[1] is not { } axes || site.ConstantOf(1) is not null) return [Identity(x, null)];
        return [Ops.IfElse((Scalar<bit>)Equal(Size(axes), Globals.Scalar(0L)),
            Identity(x, null), Reductions.Rebuild(site, x, axes, noOp: false))];
    }
}

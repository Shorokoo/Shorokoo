using Shorokoo.Core.Nodes;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

/// <summary>
/// A reduction with <c>noop_with_empty_axes</c> 1 and no axes, or an empty axes tensor, which the
/// spec defines as its input passed through and ONNX Runtime's kernels do not compute as such.
/// Rewritten as <c>Identity</c> when the axes are absent or a constant empty tensor, and as an
/// <c>If</c> on the axes' element count otherwise.
/// (Shorokoo/Shorokoo#409)
/// </summary>
internal sealed class ReduceNoopEmptyAxesWorkaround : KernelWorkaround
{
    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([REDUCE_L1, REDUCE_L2, REDUCE_LOG_SUM, REDUCE_LOG_SUM_EXP, REDUCE_MAX, REDUCE_MEAN, REDUCE_MIN, REDUCE_PROD, REDUCE_SUM, REDUCE_SUM_SQUARE], StringComparer.Ordinal);

    public override bool Applies(WorkaroundSite site) => false;

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
        => throw new NotSupportedException($"{Name} rewrites no call.");
}

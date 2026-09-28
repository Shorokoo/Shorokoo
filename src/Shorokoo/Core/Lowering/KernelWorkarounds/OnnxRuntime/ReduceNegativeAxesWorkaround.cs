using Shorokoo.Core.Nodes;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

/// <summary>
/// A reduction over negative axes of an empty input, which ONNX Runtime's kernels do not
/// normalise as the spec does. Rewritten with the axes made non-negative: as a constant when the
/// axes are constant and the input's rank known, and in the graph otherwise.
/// (Shorokoo/Shorokoo#422)
/// </summary>
internal sealed class ReduceNegativeAxesWorkaround : KernelWorkaround
{
    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([REDUCE_L1, REDUCE_L2, REDUCE_LOG_SUM, REDUCE_LOG_SUM_EXP, REDUCE_MAX, REDUCE_MEAN, REDUCE_MIN, REDUCE_PROD, REDUCE_SUM, REDUCE_SUM_SQUARE], StringComparer.Ordinal);

    public override bool Applies(WorkaroundSite site) => false;

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
        => throw new NotSupportedException($"{Name} rewrites no call.");
}

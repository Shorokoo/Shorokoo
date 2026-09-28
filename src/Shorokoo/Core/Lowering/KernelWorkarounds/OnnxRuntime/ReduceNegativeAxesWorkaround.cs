using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

using static OnnxOp;
using static OpCodes;

/// <summary>
/// A reduction over negative axes of an empty input, which ONNX Runtime's kernels do not
/// normalise as the spec does. Rewritten with the axes made non-negative: as a constant when the
/// axes are constant and the input's rank known, and in the graph otherwise. Constant axes with
/// no negative entry, and a <c>Constant</c> input that is not empty, keep the call as it stands.
/// (Shorokoo/Shorokoo#422)
/// </summary>
internal sealed class ReduceNegativeAxesWorkaround : KernelWorkaround
{
    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([REDUCE_L1, REDUCE_L2, REDUCE_LOG_SUM, REDUCE_LOG_SUM_EXP, REDUCE_MAX, REDUCE_MEAN, REDUCE_MIN, REDUCE_PROD, REDUCE_SUM, REDUCE_SUM_SQUARE], StringComparer.Ordinal);

    public override bool Applies(WorkaroundSite site)
        => site.IsPresent(1)
           && (site.ConstantOf(1) is not { } axes || axes.Elements<long>().ToArray().Any(a => a < 0))
           && Reductions.InputMayBeEmpty(site);

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
    {
        var x = inputs[0]!;
        var axes = inputs[1]!;
        var rank = site.RankOf(0);
        if (rank is { } r && site.ConstantOf(1) is { } constant)
        {
            long[] normalised = [.. constant.Elements<long>().ToArray().Select(a => a < 0 ? a + r : a)];
            return [Reductions.Rebuild(site, x, Globals.Vector(normalised))];
        }
        Variable extent = rank is { } known ? Globals.Scalar((long)known) : Size(Shape(x));
        return [Reductions.Rebuild(site, x, Where(Less(axes, Globals.Scalar(0L)), Add(axes, extent), axes))];
    }
}

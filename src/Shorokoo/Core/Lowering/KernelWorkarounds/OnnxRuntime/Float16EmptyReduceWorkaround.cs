using Shorokoo.Core.Nodes;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

/// <summary>
/// A float16 <c>ReduceSumSquare</c> or <c>ReduceL1</c> over no or empty axes, whose ONNX Runtime
/// kernel fails on an empty input. Rewritten as an <c>If</c> on the input's element count that
/// reduces an empty input in float32 and casts the result back, so the failing kernel never sees
/// one.
/// (Shorokoo/Shorokoo#411)
/// </summary>
internal sealed class Float16EmptyReduceWorkaround : KernelWorkaround
{
    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([REDUCE_SUM_SQUARE, REDUCE_L1], StringComparer.Ordinal);

    public override bool Applies(WorkaroundSite site) => false;

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
        => throw new NotSupportedException($"{Name} rewrites no call.");
}

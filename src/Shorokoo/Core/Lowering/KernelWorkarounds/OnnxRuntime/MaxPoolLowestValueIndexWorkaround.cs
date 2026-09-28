using Shorokoo.Core.Nodes;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

/// <summary>
/// An int8 or uint8 <c>MaxPool</c> whose indices are read, which ONNX Runtime gives an index of -1
/// for a window holding only the type's lowest value. Rewritten as the same pool over the input
/// cast to float32, its values cast back.
/// (Shorokoo/Shorokoo#420)
/// </summary>
internal sealed class MaxPoolLowestValueIndexWorkaround : KernelWorkaround
{
    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([MAX_POOL], StringComparer.Ordinal);

    public override bool Applies(WorkaroundSite site) => false;

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
        => throw new NotSupportedException($"{Name} rewrites no call.");
}

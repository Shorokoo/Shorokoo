using Shorokoo.Core.Nodes;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

/// <summary>
/// A <c>Where</c> over int8, int16, uint16, uint32, uint64 or bool values, which ONNX Runtime has no
/// kernel for. Rewritten through a type it has one for, or, for bool, as logical operators.
/// (Shorokoo/Shorokoo#423)
/// </summary>
internal sealed class WhereTypesWorkaround : KernelWorkaround
{
    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([WHERE], StringComparer.Ordinal);

    public override bool Applies(WorkaroundSite site) => false;

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
        => throw new NotSupportedException($"{Name} rewrites no call.");
}

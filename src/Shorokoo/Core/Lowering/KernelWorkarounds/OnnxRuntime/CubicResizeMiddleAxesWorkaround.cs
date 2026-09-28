using Shorokoo.Core.Nodes;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

/// <summary>
/// A cubic rank-4 <c>tf_crop_and_resize</c> <c>Resize</c> scaling axis 1 or 2 with axes 0 and 3 at
/// scale 1, which ONNX Runtime's kernel reads with a layout rule of its own. Rewritten as the same
/// resize between transposes that bring the scaled axes last.
/// (Shorokoo/Shorokoo#421)
/// </summary>
internal sealed class CubicResizeMiddleAxesWorkaround : KernelWorkaround
{
    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([RESIZE], StringComparer.Ordinal);

    public override bool Applies(WorkaroundSite site) => false;

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
        => throw new NotSupportedException($"{Name} rewrites no call.");
}

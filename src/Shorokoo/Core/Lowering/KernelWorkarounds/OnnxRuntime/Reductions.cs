using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

using static OnnxOpAttributeNames;

/// <summary>What the reduction workarounds share.</summary>
internal static class Reductions
{
    /// <summary>The site's reduction over <paramref name="data"/> and <paramref name="axes"/>,
    /// with the site's <c>keepdims</c> and, unless <paramref name="noOp"/> overrides it, its
    /// <c>noop_with_empty_axes</c>.</summary>
    public static Variable Rebuild(WorkaroundSite site, Variable data, Variable? axes, bool? noOp = null)
        => NodeBuilder.BuildNodeSingleOut(site.OpCode, [data, axes],
            [(AttrKeepdims, site.Attributes.GetBoolVal(AttrKeepdims)),
             (AttrNoopWithEmptyAxes, noOp ?? site.Attributes.GetBoolVal(AttrNoopWithEmptyAxes))]);

    /// <summary>Whether input slot 0 may be empty: it is not a <c>Constant</c>, or it is an empty
    /// one.</summary>
    public static bool InputMayBeEmpty(WorkaroundSite site)
        => site.ConstantOf(0) is not { } constant || constant.Shape.Dims.Contains(0);
}

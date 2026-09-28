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

    /// <summary>Whether <paramref name="x"/> holds no element, as the product of its shape: ONNX
    /// Runtime folds that to a constant when the model states the dimensions, and never folds
    /// <c>Size</c>.</summary>
    public static Scalar<bit> IsEmpty(Variable x)
        => (Scalar<bit>)OnnxOp.Equal(ElementCount(x), Globals.Scalar(0L));

    /// <summary>The number of elements of <paramref name="x"/>, as the product of its shape.</summary>
    public static Variable ElementCount(Variable x)
        => OnnxOp.ReduceProd(OnnxOp.Shape(x), keepdims: false);

    /// <summary>Whether input slot 0 may be empty: it is not a <c>Constant</c>, or it is an empty
    /// one.</summary>
    public static bool InputMayBeEmpty(WorkaroundSite site)
        => site.ConstantShapeOf(0) is not { } shape || shape.Dims.Contains(0);
}

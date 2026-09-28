using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

using static OnnxOp;
using static OnnxOpAttributeNames;
using static OpCodes;

/// <summary>
/// A reduction with <c>noop_with_empty_axes</c> 1 and no axes, or an empty axes tensor, reduces no
/// axis: each element is the reduction of its own one-element group. ONNX Runtime's kernels compute
/// that over a nonempty input, and over an empty one reduce every axis instead; the result there is
/// the input itself, an empty tensor of its shape and type.
///
/// <para>When the model states every input's dimensions, the call becomes an <c>If</c> on the input
/// and the axes both being empty, each counted as the product of its shape: the plain call where
/// they are not, the input where they are. ONNX Runtime folds that <c>If</c> away when it builds
/// the session wherever those shapes follow from the stated dimensions, and runs it where one
/// depends on the data. Otherwise the
/// call stays one call to the same kernel on the same data, with <c>keepdims</c> 0 and the axes it
/// reduces put back by an <c>Unsqueeze</c>; only when the input and the axes are both empty is the
/// input viewed with a trailing axis of one, and that axis alone reduced. That choice is shape
/// arithmetic and the views share the tensor's memory, so there is no branch and no copy. A scalar
/// input, never empty, and a <c>Constant</c> input that is not empty keep the call as it stands; an
/// empty <c>Constant</c> input with no axes or constant empty ones becomes an <c>Identity</c>.</para>
/// (Shorokoo/Shorokoo#409)
/// </summary>
internal sealed class ReduceNoopEmptyAxesWorkaround : KernelWorkaround
{
    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([REDUCE_L1, REDUCE_L2, REDUCE_LOG_SUM, REDUCE_LOG_SUM_EXP, REDUCE_MAX, REDUCE_MEAN, REDUCE_MIN, REDUCE_PROD, REDUCE_SUM, REDUCE_SUM_SQUARE], StringComparer.Ordinal);

    public override bool Applies(WorkaroundSite site)
        => site.Attributes.GetBoolVal(AttrNoopWithEmptyAxes) == true
           && site.RankOf(0) != 0
           && (!site.IsPresent(1) || site.ConstantOf(1) is not { } axes || axes.Shape.Dims.Contains(0))
           && Reductions.InputMayBeEmpty(site);

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
    {
        var x = inputs[0]!;
        var axes = site.IsPresent(1) && site.ConstantOf(1) is null ? inputs[1] : null;
        if (axes is null && site.ConstantOf(0) is not null) return [Identity(x, null)];
        var shape = Shape(x);

        if (site.ShapesAreConcrete)
        {
            var elements = Reductions.ElementCount(x);
            if (axes is not null) elements = Add(elements, Reductions.ElementCount(axes));
            return [Ops.IfElse((Scalar<bit>)Equal(elements, Globals.Scalar(0L)),
                Identity(x, null), Reductions.Rebuild(site, x, inputs[1]))];
        }

        Variable rank = site.RankOf(0) is { } known ? Globals.Vector((long)known) : Shape(shape);
        Variable empty = ReduceProd(shape, keepdims: true);
        if (axes is not null) empty = Add(Shape(axes), empty);
        var unitAxis = Slice(rank, empty, Globals.Vector(1L));
        var reduced = NodeBuilder.BuildNodeSingleOut(site.OpCode,
            [Unsqueeze(x, unitAxis), axes is null ? unitAxis : Concat([axes, unitAxis], 0)],
            [(AttrKeepdims, false), (AttrNoopWithEmptyAxes, true)]);
        return [axes is null || site.Attributes.GetBoolVal(AttrKeepdims) == false ? reduced : Unsqueeze(reduced, axes)];
    }
}

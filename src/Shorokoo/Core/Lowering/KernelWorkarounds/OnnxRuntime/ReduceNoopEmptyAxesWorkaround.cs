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
/// <para>The call stays one call to the same kernel on the same data, with <c>keepdims</c> 0 and
/// the axes it reduces put back by an <c>Unsqueeze</c>; only when the input and the axes are both
/// empty is the input viewed with a trailing axis of one, and that axis alone reduced. That choice
/// is shape arithmetic and the views share the tensor's memory, so there is no branch and no copy,
/// and ONNX Runtime computes the choice when it builds the session wherever the shapes follow from
/// the dimensions the model states. An <c>If</c> holding the call in a branch would fail to build
/// some of the sessions in which ONNX Runtime folds it, among them one over an input it computes
/// from those dimensions alone, such as a tensor of ones of an input's shape. A scalar
/// input, never empty, and a <c>Constant</c> input that is not empty keep the call as it stands; an
/// empty <c>Constant</c> input with no axes or constant empty ones becomes an <c>Identity</c>.</para>
/// (Shorokoo/Shorokoo#409)
/// </summary>
internal class ReduceNoopEmptyAxesWorkaround : KernelWorkaround
{
    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([REDUCE_L1, REDUCE_L2, REDUCE_LOG_SUM, REDUCE_LOG_SUM_EXP, REDUCE_MAX, REDUCE_MEAN, REDUCE_MIN, REDUCE_PROD, REDUCE_SUM, REDUCE_SUM_SQUARE], StringComparer.Ordinal);

    /// <summary>Whether a call of <paramref name="opCode"/> is rewritten whatever its input, a
    /// nonempty one and a scalar included, and views its input with the trailing axis whenever its
    /// axes are empty.</summary>
    protected virtual bool OverEveryInput(string opCode) => false;

    public override bool Applies(WorkaroundSite site)
        => site.Attributes.GetBoolVal(AttrNoopWithEmptyAxes) == true
           && (OverEveryInput(site.OpCode) || site.RankOf(0) != 0 && Reductions.InputMayBeEmpty(site))
           && (!site.IsPresent(1) || site.ConstantShapeOf(1) is not { } axes || axes.Dims.Contains(0));

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
    {
        var x = inputs[0]!;
        var axes = site.IsPresent(1) && site.ConstantShapeOf(1) is null ? inputs[1] : null;
        if (axes is null && site.ConstantShapeOf(0) is { } constant && constant.Dims.Contains(0)) return [Identity(x, null)];

        var shape = OverEveryInput(site.OpCode) ? null : Shape(x);
        Variable rank = site.RankOf(0) is { } known ? Globals.Vector((long)known) : Shape(shape ?? Shape(x));
        if (shape is null)
        {
            if (axes is null) return [Squeeze(Kept(site, x, rank, rank), rank)];
            var trailing = Slice(rank, Shape(axes), Globals.Vector(1L));
            var reducedAxes = Concat([axes, trailing], 0);
            var kept = Kept(site, x, trailing, reducedAxes);
            return [site.Attributes.GetBoolVal(AttrKeepdims) == false
                ? Squeeze(kept, reducedAxes)
                : Reshape(kept, Slice(Shape(kept), Globals.Vector(0L), rank), allowZero: true)];
        }

        Variable empty = ReduceProd(shape, keepdims: true);
        if (axes is not null) empty = Add(Shape(axes), empty);
        var unitAxis = Slice(rank, empty, Globals.Vector(1L));
        var reduced = NodeBuilder.BuildNodeSingleOut(site.OpCode,
            [Unsqueeze(x, unitAxis), axes is null ? unitAxis : Concat([axes, unitAxis], 0)],
            [(AttrKeepdims, false), (AttrNoopWithEmptyAxes, true)]);
        return [axes is null || site.Attributes.GetBoolVal(AttrKeepdims) == false ? reduced : Unsqueeze(reduced, axes)];
    }

    /// <summary>The site's reduction, with <c>keepdims</c> 1, of <paramref name="x"/> viewed with a
    /// trailing axis of one at <paramref name="trailing"/>, over <paramref name="axes"/>.</summary>
    private static Variable Kept(WorkaroundSite site, Variable x, Variable trailing, Variable axes)
        => NodeBuilder.BuildNodeSingleOut(site.OpCode, [Unsqueeze(x, trailing), axes],
            [(AttrKeepdims, true), (AttrNoopWithEmptyAxes, true)]);
}

/// <summary>
/// <see cref="ReduceNoopEmptyAxesWorkaround"/> with <c>ReduceL1</c>, <c>ReduceL2</c>,
/// <c>ReduceSumSquare</c> and <c>ReduceLogSum</c> rewritten over every input (Shorokoo/Shorokoo#537).
///
/// <para>ONNX Runtime's CUDA execution provider passes the input through unchanged when a reduction
/// reduces no axis, which is each element's own group for the reductions whose one-element group is
/// the element itself, and wrong for these four: the absolute value, the square and the logarithm.
/// So whenever their axes are empty — absent, a constant empty tensor, or one the run computes empty
/// — the input is viewed with a trailing axis of one and reduced over it with <c>keepdims</c> 1, which
/// computes each element's group, and the trailing axis is dropped again: by a <c>Squeeze</c> of it,
/// and for axes the run computes, which are reduced together with a trailing axis that is there only
/// when they are empty, by a <c>Squeeze</c> of all of them or, under <c>keepdims</c> 1, a
/// <c>Reshape</c> to the input's rank. No call has <c>keepdims</c> 0, so none meets the provider's
/// refusal of it over an empty axis. That is shape arithmetic and views, with no branch and no copy,
/// over a nonempty input, a scalar and a <c>Constant</c> alike; only an empty <c>Constant</c> with no axes or constant empty ones, the
/// input itself, becomes an <c>Identity</c>. Every other reduction is rewritten as
/// <see cref="ReduceNoopEmptyAxesWorkaround"/> does.</para>
///
/// <para>Part of the CUDA set only (<see cref="Shorokoo.Core.Backends.KernelWorkaroundSets.OnnxRuntimeCuda"/>),
/// in that workaround's place: the CPU provider computes each element's group.</para>
/// </summary>
internal sealed class CudaReduceNoopAxesWorkaround : ReduceNoopEmptyAxesWorkaround
{
    protected override bool OverEveryInput(string opCode) => opCode is REDUCE_L1 or REDUCE_L2 or REDUCE_SUM_SQUARE or REDUCE_LOG_SUM;
}

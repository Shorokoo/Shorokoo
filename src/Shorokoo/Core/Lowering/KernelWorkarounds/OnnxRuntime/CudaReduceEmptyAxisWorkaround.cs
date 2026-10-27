using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

using static OnnxOp;
using static OnnxOpAttributeNames;
using static OpCodes;

/// <summary>
/// A reduction with <c>keepdims</c> 0 of an input that may be empty, rewritten as the same
/// reduction with <c>keepdims</c> 1 and the reduced axes squeezed out (Shorokoo/Shorokoo#536).
///
/// <para>ONNX Runtime's CUDA execution provider refuses every reduction with <c>keepdims</c> 0 over
/// an axis of extent 0, where the spec gives the empty group's identity. With <c>keepdims</c> 1 it
/// computes that identity, so the reduced axes are kept and then removed by a <c>Squeeze</c>, a
/// view of the result. Constant axes are squeezed as they stand, and a reduction of every axis — no
/// axes, or an empty axes tensor with <c>noop_with_empty_axes</c> 0 — squeezes every axis, all of
/// them of extent 1. Axes the run computes are squeezed as computed, which reduces every axis when
/// they come out empty, as the reduction then does; with <c>noop_with_empty_axes</c> 1, where empty
/// axes reduce nothing, a trailing axis of 1 is put on and squeezed with them, so the squeeze never
/// removes more than they name. A call that reduces nothing, a scalar input and a <c>Constant</c>
/// input that is not empty keep the call as it stands.</para>
///
/// <para>Part of the CUDA set only (<see cref="Shorokoo.Core.Backends.KernelWorkaroundSets.OnnxRuntimeCuda"/>),
/// after every rewrite of <see cref="Shorokoo.Core.Backends.KernelWorkaroundSets.OnnxRuntime"/>,
/// so the reductions those build are rewritten too: the CPU provider computes the call as
/// written.</para>
/// </summary>
internal sealed class CudaReduceEmptyAxisWorkaround : KernelWorkaround
{
    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([REDUCE_L1, REDUCE_L2, REDUCE_LOG_SUM, REDUCE_LOG_SUM_EXP, REDUCE_MAX, REDUCE_MEAN, REDUCE_MIN, REDUCE_PROD, REDUCE_SUM, REDUCE_SUM_SQUARE], StringComparer.Ordinal);

    public override bool Applies(WorkaroundSite site)
        => site.Attributes.GetBoolVal(AttrKeepdims) == false
           && site.RankOf(0) != 0
           && !(NoOp(site) && (!site.IsPresent(1) || site.ConstantShapeOf(1) is { } axes && axes.Dims.Contains(0)))
           && Reductions.InputMayBeEmpty(site);

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
    {
        var x = inputs[0]!;
        var axes = inputs[1];
        var kept = NodeBuilder.BuildNodeSingleOut(site.OpCode, [x, axes],
            [(AttrKeepdims, true), (AttrNoopWithEmptyAxes, site.Attributes.GetBoolVal(AttrNoopWithEmptyAxes))]);
        if (axes is null || site.ConstantShapeOf(1) is { Dims: [0] }) return [Squeeze(kept, null)];
        if (!NoOp(site) || site.ConstantShapeOf(1) is not null) return [Squeeze(kept, axes)];

        Variable rank = site.RankOf(0) is { } known ? Globals.Scalar((long)known) : Size(Shape(x));
        var trailing = Unsqueeze(rank, Globals.Vector(0L));
        var counted = Where(Less(axes, Globals.Scalar(0L)), Add(axes, rank), axes);
        return [Squeeze(Unsqueeze(kept, trailing), Concat([counted, trailing], 0))];
    }

    private static bool NoOp(WorkaroundSite site) => site.Attributes.GetBoolVal(AttrNoopWithEmptyAxes) == true;
}

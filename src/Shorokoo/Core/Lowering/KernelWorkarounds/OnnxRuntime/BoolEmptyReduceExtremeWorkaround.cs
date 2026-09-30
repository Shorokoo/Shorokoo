using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

using static OnnxOpAttributeNames;
using static OpCodes;

/// <summary>
/// <c>ReduceMax</c> or <c>ReduceMin</c> over a boolean input, rewritten so that an empty group
/// takes the spec's value, false for ReduceMax and true for ReduceMin (Shorokoo/Shorokoo#382).
///
/// <para>ONNX Runtime's CPU kernels throw when a reduced axis of a boolean input has extent 0. The
/// call never reaches the boolean kernel: ReduceMax becomes the uint8 ReduceMax of the input cast
/// to uint8, cast back, and ReduceMin the negation of that over the negated input. An empty uint8
/// group gives 0, which casts to false, so both are exact on every input, with no branch. A call is
/// left as it stands for a scalar input or a <c>Constant</c> one that is not empty, and when
/// nothing is reduced (no axes with <c>noop_with_empty_axes</c> set).</para>
/// </summary>
internal sealed class BoolEmptyReduceExtremeWorkaround : KernelWorkaround
{
    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([REDUCE_MAX, REDUCE_MIN], StringComparer.Ordinal);

    public override bool Applies(WorkaroundSite site)
    {
        if (!site.DTypeOf(0).IsSameElementTypeAs(DType.Bool)) return false;
        if (site.RankOf(0) == 0) return false;
        if (!site.IsPresent(1) && site.Attributes.GetBoolVal(AttrNoopWithEmptyAxes) == true) return false;
        return Reductions.InputMayBeEmpty(site);
    }

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
    {
        var input = inputs[0]!;
        var axes = inputs[1];
        var keepDims = site.Attributes.GetBoolVal(AttrKeepdims);
        var noOp = site.Attributes.GetBoolVal(AttrNoopWithEmptyAxes);

        Variable Any(Variable bits)
            => OnnxOp.Cast(OnnxOp.ReduceMax(OnnxOp.Cast(bits, null, DType.UInt8), axes, keepDims, noOp), null, DType.Bool);
        return [site.OpCode == REDUCE_MAX ? Any(input) : OnnxOp.Not(Any(OnnxOp.Not(input)))];
    }
}

using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

using static OnnxOp;
using static OnnxOpAttributeNames;
using static OpCodes;

/// <summary>
/// A float16 <c>ReduceSumSquare</c>, <c>ReduceL1</c> or <c>ReduceLogSum</c> without an axes
/// input, whose ONNX Runtime kernel crashes the process on an empty input. Rewritten as an
/// <c>If</c> on the input's element count, the product of its shape, that reduces an empty input
/// in float32 and casts the result back, so the crashing kernel never sees one; ONNX Runtime folds
/// the <c>If</c> away when the model states the input's dimensions. An empty <c>Constant</c> input
/// takes the float32 form alone, and a scalar input or a nonempty <c>Constant</c> one keeps the
/// call as it stands.
/// (Shorokoo/Shorokoo#411)
/// </summary>
internal sealed class Float16EmptyReduceWorkaround : KernelWorkaround
{
    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([REDUCE_SUM_SQUARE, REDUCE_L1, REDUCE_LOG_SUM], StringComparer.Ordinal);

    public override bool Applies(WorkaroundSite site)
        => site.DTypeOf(0).IsSameElementTypeAs(DType.Float16)
           && site.RankOf(0) != 0
           && !site.IsPresent(1)
           && site.Attributes.GetBoolVal(AttrNoopWithEmptyAxes) != true
           && Reductions.InputMayBeEmpty(site);

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
    {
        var x = inputs[0]!;
        var inFloat32 = Cast(Reductions.Rebuild(site, Cast(x, null, DType.Float32), null), null, DType.Float16);
        if (site.ConstantOf(0) is not null) return [inFloat32];
        return [Ops.IfElse(Reductions.IsEmpty(x), inFloat32, Reductions.Rebuild(site, x, null))];
    }
}

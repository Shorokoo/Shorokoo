using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

using static OnnxOp;
using static OnnxOpAttributeNames;
using static OpCodes;

/// <summary>
/// An int8 or uint8 <c>MaxPool</c> whose indices are read, rewritten so that a window holding only
/// the element type's lowest value has the spec's index: the position of its first maximum
/// (Shorokoo/Shorokoo#420).
///
/// <para>ONNX Runtime's pooling kernel starts each window's maximum at the type's lowest value and
/// takes an element only when it is strictly greater, so such a window keeps the position it
/// starts from — -1 along every spatial axis, which its index arithmetic turns into -1 for a
/// one-dimensional pool and into some other value, possibly the position of another element, once
/// there is a second spatial axis or a second channel.</para>
///
/// <para>The rewrite is the same pool over the input cast to float32, where every value is above
/// the lowest, its values cast back. A window the pool takes wholly from its padding is pooled over
/// float32 padding, below every int8 or uint8 value; its value is raised to the type's lowest,
/// which the pool without indices gives it, before the cast back.</para>
/// </summary>
internal sealed class MaxPoolLowestValueIndexWorkaround : KernelWorkaround
{
    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([MAX_POOL], StringComparer.Ordinal);

    public override bool Applies(WorkaroundSite site)
        => site.IsOutputUsed(1)
           && (site.DTypeOf(0).IsSameElementTypeAs(DType.Int8) || site.DTypeOf(0).IsSameElementTypeAs(DType.UInt8));

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
    {
        var a = site.Attributes;
        var x = inputs[0]!;
        var dtype = site.DTypeOf(0);
        var (wide, indices) = MaxPoolWithIndices(Cast(x, null, DType.Float32),
            a.GetEnumVal<AutoPad>(AttrAutoPad), a.GetBoolVal(AttrCeilMode), a.GetLongsVal(AttrDilations),
            a.GetLongsVal(AttrKernelShape), a.GetLongsVal(AttrPads), a.GetLongVal(AttrStorageOrder), a.GetLongsVal(AttrStrides));
        var lowest = Constant(dtype.IsSameElementTypeAs(DType.Int8) ? (float)sbyte.MinValue : 0f);
        return [Cast(Max(wide, lowest), null, dtype), indices];
    }
}

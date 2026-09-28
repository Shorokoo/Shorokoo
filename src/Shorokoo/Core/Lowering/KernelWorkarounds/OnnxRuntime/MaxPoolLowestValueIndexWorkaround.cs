using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

using static OnnxOp;
using static OnnxOpAttributeNames;
using static OpCodes;

/// <summary>
/// A <c>MaxPool</c> whose indices are read, rewritten so that a window whose maximum is at or below
/// the element type's lowest finite value has the spec's index: the position of its first maximum
/// (Shorokoo/Shorokoo#420).
///
/// <para>ONNX Runtime's pooling kernel starts each window's maximum at the type's lowest finite
/// value and takes an element only when it is strictly greater, so such a window keeps the
/// position it starts from — -1 along every spatial axis, which its index arithmetic turns into -1
/// for a one-dimensional pool and into some other value, possibly the position of another element,
/// once there is a second spatial axis or a second channel. That is a window holding only the
/// lowest value for int8 and uint8, only the lowest finite value and -inf for float32 and
/// float64, whose value the kernel also gives as the lowest finite one, and only -inf for
/// float16, which it pools as float32.</para>
///
/// <para>An int8 or uint8 pool is the same pool over the input cast to float32, where every value
/// is above the lowest, its values cast back. A floating-point pool keeps its own values and
/// indices save for the windows whose maximum it gives at or below the lowest finite value, whose
/// value and index are taken from the same pool over a float32 tensor holding 1 where the input is
/// at least the lowest finite value and 0 elsewhere: the first 1 of such a window is its first
/// element at the lowest finite value, its first element when all of them are -inf, and the value
/// is the lowest finite one or -inf accordingly.</para>
/// </summary>
internal sealed class MaxPoolLowestValueIndexWorkaround : KernelWorkaround
{
    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([MAX_POOL], StringComparer.Ordinal);

    public override bool Applies(WorkaroundSite site)
        => site.IsOutputUsed(1) && (IsSmallInteger(site.DTypeOf(0)) || LowestFinite(site.DTypeOf(0)) is not null);

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
    {
        var a = site.Attributes;
        var x = inputs[0]!;
        (Variable y, Variable indices) Pool(Variable input) => MaxPoolWithIndices(input,
            a.GetEnumVal<AutoPad>(AttrAutoPad), a.GetBoolVal(AttrCeilMode), a.GetLongsVal(AttrDilations),
            a.GetLongsVal(AttrKernelShape), a.GetLongsVal(AttrPads), a.GetLongVal(AttrStorageOrder), a.GetLongsVal(AttrStrides));

        var dtype = site.DTypeOf(0);
        if (IsSmallInteger(dtype))
        {
            var (wide, wideIndices) = Pool(Cast(x, null, DType.Float32));
            return [Cast(wide, null, dtype), wideIndices];
        }

        var lowest = LowestFinite(dtype)!(x);
        var (y, indices) = Pool(x);
        var (atLeastLowest, firstAtLeastLowest) = Pool(Cast(GreaterOrEqual(x, lowest), null, DType.Float32));
        var unfound = LessOrEqual(y, lowest);
        var value = Where(Greater(atLeastLowest, Constant(0f)), lowest, CastLike(Constant(float.NegativeInfinity), x, null));
        return [Where(unfound, value, y), Where(unfound, firstAtLeastLowest, indices)];
    }

    private static bool IsSmallInteger(DType dtype)
        => dtype.IsSameElementTypeAs(DType.Int8) || dtype.IsSameElementTypeAs(DType.UInt8);

    /// <summary>Builds the lowest finite value of the floating-point type <paramref name="dtype"/>
    /// as a constant of the type of the tensor it is given; null for every other type.</summary>
    private static Func<Variable, Variable>? LowestFinite(DType dtype)
    {
        if (dtype.IsSameElementTypeAs(DType.Float16)) return like => CastLike(Constant(-65504f), like, null);
        if (dtype.IsSameElementTypeAs(DType.Float32)) return _ => Constant(float.MinValue);
        if (dtype.IsSameElementTypeAs(DType.Float64)) return _ => Globals.Scalar(double.MinValue);
        return null;
    }
}

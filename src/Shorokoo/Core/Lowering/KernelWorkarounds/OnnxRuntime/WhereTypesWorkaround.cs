using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

using static OnnxOp;
using static OpCodes;

/// <summary>
/// A <c>Where</c> over int8, int16, uint16, uint32, uint64, bfloat16 or bool values, which ONNX
/// Runtime has no kernel for (Shorokoo/Shorokoo#423).
///
/// <para>A bool call becomes <c>Or(And(c, x), And(Not(c), y))</c>, which broadcasts its three
/// operands as <c>Where</c> does. Every other call selects through a type ONNX Runtime has a kernel
/// for and casts the result back: int32 for int8, int16 and uint16; int64 for uint32 and uint64,
/// whose casts to and from int64 keep every bit; float32 for bfloat16, which holds every bfloat16
/// value (a NaN stays a NaN). The workaround comes last in the set, so it also covers a
/// <c>Where</c> an earlier workaround emits.</para>
/// </summary>
internal sealed class WhereTypesWorkaround : KernelWorkaround
{
    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([WHERE], StringComparer.Ordinal);

    public override bool Applies(WorkaroundSite site)
        => site.DTypeOf(1).IsSameElementTypeAs(DType.Bool) || Carrier(site.DTypeOf(1)) is not null;

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
    {
        var (condition, x, y) = (inputs[0]!, inputs[1]!, inputs[2]!);
        var dtype = site.DTypeOf(1);
        if (dtype.IsSameElementTypeAs(DType.Bool))
            return [Or(And(condition, x), And(Not(condition), y))];

        var carrier = Carrier(dtype)!;
        return [Cast(Where(condition, Cast(x, null, carrier), Cast(y, null, carrier)), null, dtype)];
    }

    /// <summary>The type a <c>Where</c> over <paramref name="dtype"/> selects through; null when
    /// ONNX Runtime has a kernel for <paramref name="dtype"/> itself.</summary>
    private static DType? Carrier(DType dtype)
    {
        if (dtype.IsSameElementTypeAs(DType.Int8) || dtype.IsSameElementTypeAs(DType.Int16)
            || dtype.IsSameElementTypeAs(DType.UInt16))
            return DType.Int32;
        if (dtype.IsSameElementTypeAs(DType.UInt32) || dtype.IsSameElementTypeAs(DType.UInt64))
            return DType.Int64;
        if (dtype.IsSameElementTypeAs(DType.BFloat16)) return DType.Float32;
        return null;
    }
}

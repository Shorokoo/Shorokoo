using System.Numerics;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

using static OnnxOp;
using static OpCodes;

/// <summary>
/// An int64 or int32 <c>Range</c>, rewritten so that its element count is exact when
/// <c>limit - start</c> is beyond what ONNX Runtime's kernel counts exactly (Shorokoo/Shorokoo#447,
/// Shorokoo/Shorokoo#450).
///
/// <para>The kernel takes <c>limit - start</c> in the input type, which wraps for a span beyond
/// it, and counts the elements as <c>ceil((limit - start) / delta)</c> in double precision, which
/// rounds an int64 span beyond 2^53. Either can give a wrong count: <c>Range(0, 2^62 + 1, 2^61)</c>
/// gives two elements where the spec gives three, <c>Range(long.MaxValue - 2, long.MinValue + 2,
/// 1)</c> gives five where the spec gives none, and the int32 <c>Range(int.MinValue, int.MaxValue,
/// 2^30)</c> gives none where the spec gives four. A difference that neither wraps nor exceeds 2^53
/// is exact, and the double quotient is never rounded onto an integer, so the count is exact. An
/// int16 <c>Range</c> is counted from a difference in a wider type, and every call of another type
/// counts a span a double holds; both are left alone.</para>
///
/// <para>A call whose three inputs are <c>Constant</c>s is left as it stands where their span is
/// below 2^53 for int64, or within int32 for int32. So is a call whose <c>delta</c> is a
/// <c>Constant</c> 1 or -1 and whose <c>start</c> is a <c>Constant</c> of a magnitude below 2^62
/// for int64, or 0 for int32, as in <c>Range(0, n, 1)</c>. For int32, <c>limit - 0</c> never
/// wraps, so the count is exact. For int64 the count is exact for a span below 2^53; for any other
/// span the kernel's count is 0 or at least 2^53, the wrapped difference included, since it is
/// then more than 2^62. A span of 2^53 or more has a spec count of 0 or one no tensor holds, so
/// the kernel refuses such a call or gives no elements, and never gives a wrong element.</para>
///
/// <para>Every other call counts in uint64, an int32 call on its inputs cast to int64 and its
/// elements cast back: the span and the magnitude of <c>delta</c> are taken as unsigned, which
/// holds every span of two int64s, and the count is the exact ceiling
/// <c>(span - 1) / |delta| + 1</c>, or 0 where <c>limit</c> does not lie beyond <c>start</c> in
/// the direction of <c>delta</c>. A count above 2^62, which no tensor holds, is taken as 2^62,
/// which ONNX Runtime refuses to allocate. The elements are <c>Range(0, count, 1) · delta +
/// start</c>, whose count ONNX Runtime computes exactly, since a count a tensor can hold is below
/// 2^53; each element lies between <c>start</c> and <c>limit</c>, so the int64 product and sum
/// give it exactly, and an int32 call's elements are int32s. A <c>delta</c> of 0 reaches that
/// <c>Range</c> as its step, which ONNX Runtime refuses as it refuses the call as written.</para>
/// </summary>
internal sealed class IntegerRangeCountWorkaround : KernelWorkaround
{
    private static readonly BigInteger ExactSpan = BigInteger.One << 53;

    private const long SmallStart = 1L << 62;

    private const ulong RefusedCount = 1UL << 62;

    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([RANGE], StringComparer.Ordinal);

    public override bool Applies(WorkaroundSite site)
    {
        var wide = site.DTypeOf(0).IsSameElementTypeAs(DType.Int64);
        if (!wide && !site.DTypeOf(0).IsSameElementTypeAs(DType.Int32)) return false;
        var (start, limit, delta) = (ConstantOf(site, 0, wide), ConstantOf(site, 1, wide), ConstantOf(site, 2, wide));
        if (start is { } s && limit is { } l && delta is not null)
        {
            var span = (BigInteger)l - s;
            return wide ? BigInteger.Abs(span) >= ExactSpan : span < int.MinValue || span > int.MaxValue;
        }
        return !(delta is 1L or -1L && (wide ? start is > -SmallStart and < SmallStart : start == 0L));
    }

    private static long? ConstantOf(WorkaroundSite site, int input, bool wide)
        => site.ConstantShapeOf(input) is null ? null
            : wide ? site.ConstantOf(input)!.Elements<long>()[0] : site.ConstantOf(input)!.Elements<int>()[0];

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
    {
        var (start, limit, delta) = (inputs[0]!, inputs[1]!, inputs[2]!);
        if (site.DTypeOf(0).IsSameElementTypeAs(DType.Int64)) return [ExactRange(start, limit, delta)];
        static Variable Wide(Variable v) => Cast(v, null, DType.Int64);
        return [Cast(ExactRange(Wide(start), Wide(limit), Wide(delta)), null, DType.Int32)];
    }

    private static Variable ExactRange(Variable start, Variable limit, Variable delta)
    {
        var zero = Globals.Scalar(0L);
        var descending = Less(delta, zero);
        var reaches = Or(And(Greater(delta, zero), Greater(limit, start)), And(descending, Less(limit, start)));

        var span = Cast(Where(descending, Sub(start, limit), Sub(limit, start)), null, DType.UInt64);
        var magnitude = Cast(Where(descending, Sub(zero, delta), delta), null, DType.UInt64);
        // A delta of 0 divides by 1 here and is refused by the Range below.
        var divisor = Add(magnitude, Cast(Equal(delta, zero), null, DType.UInt64));
        var exact = Add(Div(Sub(span, Globals.Scalar(1UL)), divisor), Globals.Scalar(1UL));
        var count = Cast(Min([exact, Globals.Scalar(RefusedCount)]), null, DType.Int64);

        var step = Cast(Not(Equal(delta, zero)), null, DType.Int64);
        var positions = Range(zero, Where(reaches, count, zero), step);
        return Add(Mul(positions, delta), start);
    }
}

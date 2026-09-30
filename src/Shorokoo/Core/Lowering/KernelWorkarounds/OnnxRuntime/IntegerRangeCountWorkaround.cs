using System.Numerics;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

using static OnnxOp;
using static OpCodes;

/// <summary>
/// An integer <c>Range</c>, rewritten so that its element count is exact where ONNX Runtime cannot
/// count it exactly (Shorokoo/Shorokoo#447, Shorokoo/Shorokoo#450): an int64 one its kernel
/// miscounts, and an int32 one of three constants whose span leaves int32, which its shape
/// inference miscounts.
///
/// <para>The kernel converts <c>start</c> and <c>limit</c> to double precision, and counts the
/// elements as <c>ceil((limit - start) / delta)</c> in double precision. An int64 beyond 2^53 is
/// rounded on the way, so a <c>Range</c> that starts or ends there can have a wrong count, however
/// short it is: <c>Range(2^62, 2^62 + 2, 1)</c> gives no element where the spec gives two, and
/// <c>Range(2^62, 2^62 + 1000, 1)</c> gives 1024. Where <c>start</c> and <c>limit</c> both lie
/// within 2^53 of 0 and are less than 2^53 apart, the count is exact, since the double quotient is
/// never rounded onto an integer.</para>
///
/// <para>An int32 <c>Range</c> the kernel counts exactly, from values a double holds exactly. Where
/// all three inputs are constants, though, ONNX Runtime also infers the output's length when it
/// builds the session, and that count goes wrong when <c>limit - start</c> leaves int32:
/// <c>Range(int.MinValue, int.MaxValue, 2^30)</c> has four elements, the length inferred is eight,
/// and a node that reuses the output's buffer fails the run. Such a call is rewritten; every other
/// int32 call, and every other element type, is left as it stands.</para>
///
/// <para>A call is left as it stands where the kernel's count is right: three <c>Constant</c>s
/// within those bounds, and a call whose <c>delta</c> is a <c>Constant</c> 1 or -1 and whose
/// <c>start</c> or <c>limit</c> is a <c>Constant</c> within 2^52 of 0, as in <c>Range(0, n, 1)</c>,
/// the form ordinary code builds for a position or index, which so pays nothing. With a unit step
/// the count is the span itself: within 2^53 it is counted exactly, and beyond that it is more
/// elements than a tensor holds, which the kernel refuses.</para>
///
/// <para>Every other call counts in uint64: the span and the magnitude of <c>delta</c> are taken as
/// unsigned, which holds every span of two int64s, and the count is the exact ceiling
/// <c>(span - 1) / |delta| + 1</c>, or 0 where <c>limit</c> does not lie beyond <c>start</c> in
/// the direction of <c>delta</c>. A count above 2^62, which no tensor holds, is taken as 2^62,
/// which ONNX Runtime refuses to allocate. The elements are <c>Range(0, count, 1) · delta +
/// start</c>, whose count ONNX Runtime computes exactly, since a count a tensor can hold is below
/// 2^53; each element lies between <c>start</c> and <c>limit</c>, so the int64 product and sum
/// give it exactly. A <c>delta</c> of 0 reaches that <c>Range</c> as its step, which ONNX Runtime
/// refuses as it refuses the call as written.</para>
/// </summary>
internal sealed class IntegerRangeCountWorkaround : KernelWorkaround
{
    private static readonly BigInteger ExactSpan = BigInteger.One << 53;

    private static readonly BigInteger UnitEnd = BigInteger.One << 52;

    private const ulong RefusedCount = 1UL << 62;

    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([RANGE], StringComparer.Ordinal);

    public override bool Applies(WorkaroundSite site)
    {
        if (site.DTypeOf(0).IsSameElementTypeAs(DType.Int32))
            return ConstantOf(site, 0) is { } first && ConstantOf(site, 1) is { } last && ConstantOf(site, 2) is not null
                && ((long)last - first is < int.MinValue or > int.MaxValue);
        if (!site.DTypeOf(0).IsSameElementTypeAs(DType.Int64)) return false;
        if (ConstantOf(site, 2) is not { } delta) return true;
        var (start, limit) = (ConstantOf(site, 0), ConstantOf(site, 1));
        if (start is { } s && limit is { } l)
            return !(Within(s, ExactSpan) && Within(l, ExactSpan) && BigInteger.Abs((BigInteger)l - s) < ExactSpan);
        return !(delta is 1L or -1L && (start is { } a && Within(a, UnitEnd) || limit is { } b && Within(b, UnitEnd)));
    }

    private static bool Within(long value, BigInteger bound) => BigInteger.Abs(value) <= bound;

    private static long? ConstantOf(WorkaroundSite site, int input)
        => site.ConstantShapeOf(input) is null ? null
            : site.DTypeOf(0).IsSameElementTypeAs(DType.Int64) ? site.ConstantOf(input)!.Elements<long>()[0]
            : site.ConstantOf(input)!.Elements<int>()[0];

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

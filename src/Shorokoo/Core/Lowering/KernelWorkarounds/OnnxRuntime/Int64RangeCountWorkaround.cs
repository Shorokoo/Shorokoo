using System.Numerics;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

using static OnnxOp;
using static OpCodes;

/// <summary>
/// An int64 <c>Range</c>, rewritten so that its element count is exact when <c>limit - start</c>
/// is beyond 2^53 (Shorokoo/Shorokoo#447).
///
/// <para>ONNX Runtime's kernel counts the elements as <c>ceil((limit - start) / delta)</c> in
/// double precision, which rounds a span beyond 2^53 and can drop elements:
/// <c>Range(0, 2^62 + 1, 2^61)</c> gives two where the spec gives three. Within 2^53 the double
/// quotient is never rounded onto an integer, so the count is exact. A call whose three inputs are
/// <c>Constant</c>s spanning less than 2^53 is left as it stands, as is every call of another
/// type, whose span a double holds. So is a call whose <c>delta</c> is a <c>Constant</c> 1 or -1,
/// as in <c>Range(0, n, 1)</c>: its count is the span itself, so a count the kernel gets wrong is
/// one of 2^53 elements or more, which no tensor holds.</para>
///
/// <para>Every other call counts in uint64: the span and the magnitude of <c>delta</c> are taken
/// as unsigned, which holds every span of two int64s, and the count is the exact ceiling
/// <c>(span - 1) / |delta| + 1</c>, or 0 where <c>limit</c> does not lie beyond <c>start</c> in
/// the direction of <c>delta</c>. The elements are <c>Range(0, count, 1) · delta + start</c>, whose
/// count ONNX Runtime computes exactly, since a count a tensor can hold is below 2^53; each element
/// lies between <c>start</c> and <c>limit</c>, so the int64 product and sum give it exactly. A
/// <c>delta</c> of 0 reaches that <c>Range</c> as its step, which ONNX Runtime refuses as it
/// refuses the call as written.</para>
/// </summary>
internal sealed class Int64RangeCountWorkaround : KernelWorkaround
{
    private static readonly BigInteger ExactSpan = BigInteger.One << 53;

    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([RANGE], StringComparer.Ordinal);

    public override bool Applies(WorkaroundSite site)
    {
        if (!site.DTypeOf(0).IsSameElementTypeAs(DType.Int64)) return false;
        if (site.ConstantShapeOf(2) is not null && site.ConstantOf(2)!.Elements<long>()[0] is 1L or -1L) return false;
        if (site.ConstantShapeOf(0) is null || site.ConstantShapeOf(1) is null || site.ConstantShapeOf(2) is null) return true;
        var start = site.ConstantOf(0)!.Elements<long>()[0];
        var limit = site.ConstantOf(1)!.Elements<long>()[0];
        return BigInteger.Abs((BigInteger)limit - start) >= ExactSpan;
    }

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
    {
        var (start, limit, delta) = (inputs[0]!, inputs[1]!, inputs[2]!);
        var zero = Globals.Scalar(0L);
        var descending = Less(delta, zero);
        var reaches = Or(And(Greater(delta, zero), Greater(limit, start)), And(descending, Less(limit, start)));

        var span = Cast(Where(descending, Sub(start, limit), Sub(limit, start)), null, DType.UInt64);
        var magnitude = Cast(Where(descending, Sub(zero, delta), delta), null, DType.UInt64);
        // A delta of 0 divides by 1 here and is refused by the Range below.
        var divisor = Add(magnitude, Cast(Equal(delta, zero), null, DType.UInt64));
        var count = Add(Cast(Div(Sub(span, Globals.Scalar(1UL)), divisor), null, DType.Int64), Globals.Scalar(1L));

        var step = Cast(Not(Equal(delta, zero)), null, DType.Int64);
        var positions = Range(zero, Where(reaches, count, zero), step);
        return [Add(Mul(positions, delta), start)];
    }
}

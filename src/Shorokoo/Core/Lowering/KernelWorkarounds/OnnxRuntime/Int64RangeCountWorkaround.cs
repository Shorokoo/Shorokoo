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
/// <para>ONNX Runtime's kernel takes <c>limit - start</c> in int64, which wraps for a span beyond
/// int64, and counts the elements as <c>ceil((limit - start) / delta)</c> in double precision,
/// which rounds a span beyond 2^53. Either can give a wrong count: <c>Range(0, 2^62 + 1, 2^61)</c>
/// gives two elements where the spec gives three, and
/// <c>Range(long.MaxValue - 2, long.MinValue + 2, 1)</c> gives five where the spec gives none.
/// Within 2^53 the difference is exact and the double quotient is never rounded onto an integer,
/// so the count is exact.</para>
///
/// <para>A call whose three inputs are <c>Constant</c>s spanning less than 2^53 is left as it
/// stands. So is a call whose <c>delta</c> is a <c>Constant</c> 1 or -1 and whose <c>start</c> is
/// a <c>Constant</c> of magnitude below 2^62, as in <c>Range(0, n, 1)</c>: its count is exact for
/// a span below 2^53, and for any other span the kernel's count is 0 or at least 2^53, the
/// wrapped difference included, since it is then more than 2^62. A span of 2^53 or more has a
/// spec count of 0 or one no tensor holds, so the kernel refuses such a call or gives no elements,
/// and never gives a wrong element. Every call of another type is left alone.</para>
///
/// <para>Every other call counts in uint64: the span and the magnitude of <c>delta</c> are taken
/// as unsigned, which holds every span of two int64s, and the count is the exact ceiling
/// <c>(span - 1) / |delta| + 1</c>, or 0 where <c>limit</c> does not lie beyond <c>start</c> in
/// the direction of <c>delta</c>. A count above 2^62, which no tensor holds, is taken as 2^62,
/// which ONNX Runtime refuses to allocate. The elements are <c>Range(0, count, 1) · delta +
/// start</c>, whose count ONNX Runtime computes exactly, since a count a tensor can hold is below
/// 2^53; each element lies between <c>start</c> and <c>limit</c>, so the int64 product and sum
/// give it exactly. A <c>delta</c> of 0 reaches that <c>Range</c> as its step, which ONNX Runtime
/// refuses as it refuses the call as written.</para>
/// </summary>
internal sealed class Int64RangeCountWorkaround : KernelWorkaround
{
    private static readonly BigInteger ExactSpan = BigInteger.One << 53;

    private const long SmallStart = 1L << 62;

    private const ulong RefusedCount = 1UL << 62;

    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([RANGE], StringComparer.Ordinal);

    public override bool Applies(WorkaroundSite site)
    {
        if (!site.DTypeOf(0).IsSameElementTypeAs(DType.Int64)) return false;
        long? start = site.ConstantShapeOf(0) is null ? null : site.ConstantOf(0)!.Elements<long>()[0];
        long? limit = site.ConstantShapeOf(1) is null ? null : site.ConstantOf(1)!.Elements<long>()[0];
        long? delta = site.ConstantShapeOf(2) is null ? null : site.ConstantOf(2)!.Elements<long>()[0];
        if (start is { } s && limit is { } l && delta is not null) return BigInteger.Abs((BigInteger)l - s) >= ExactSpan;
        return !(delta is 1L or -1L && start is > -SmallStart and < SmallStart);
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
        var exact = Add(Div(Sub(span, Globals.Scalar(1UL)), divisor), Globals.Scalar(1UL));
        var count = Cast(Min([exact, Globals.Scalar(RefusedCount)]), null, DType.Int64);

        var step = Cast(Not(Equal(delta, zero)), null, DType.Int64);
        var positions = Range(zero, Where(reaches, count, zero), step);
        return [Add(Mul(positions, delta), start)];
    }
}

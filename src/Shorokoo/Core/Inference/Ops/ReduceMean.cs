using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Inference.Ops;

internal sealed class ReduceMeanOp : ReduceOpBase
{
    public override string OpCode => OpCodes.REDUCE_MEAN;
    protected override float Reduce(IEnumerable<float> values) => values.Average();
    // Integer mean truncates like ORT: sum and divide in the integer domain, at the DECLARED
    // width. The sum may overflow that width in the 64-bit buffer harmlessly — truncation
    // commutes with addition — but the divide does not commute, so the accumulator has to
    // re-enter the width first. Without that, a sum that overflowed int32 divides a value int32
    // cannot hold, and the folded constant disagrees with the backend.
    protected override long ReduceInt(IEnumerable<long> values, DType dtype)
    {
        long s = 0, n = 0;
        foreach (var v in values) unchecked { s += v; n++; }
        return n == 0 ? 0 : IntSemantics.NarrowToWidth(dtype, s) / n;
    }

    protected override ulong ReduceUInt(IEnumerable<ulong> values, DType dtype)
    {
        ulong s = 0, n = 0;
        foreach (var v in values) unchecked { s += v; n++; }
        return n == 0 ? 0 : IntSemantics.U(IntSemantics.NarrowToWidth(dtype, IntSemantics.S(s))) / n;
    }
}

using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Inference.Ops;

internal sealed class ReduceSumOp : ReduceOpBase
{
    public override string OpCode => OpCodes.REDUCE_SUM;
    protected override float Reduce(IEnumerable<float> values) => values.Sum();
    // Exact integer accumulation (the float-roundtrip default loses precision past 2^24).
    protected override long ReduceInt(IEnumerable<long> values, DType dtype) { long s = 0; foreach (var v in values) s += v; return s; }
    protected override ulong ReduceUInt(IEnumerable<ulong> values, DType dtype) { ulong s = 0; foreach (var v in values) unchecked { s += v; } return s; }
}

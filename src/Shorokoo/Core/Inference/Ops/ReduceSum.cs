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
    protected override long ReduceInt(IEnumerable<long> values) { long s = 0; foreach (var v in values) s += v; return s; }
}

using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Inference.Ops;

internal sealed class ReduceMeanOp : ReduceOpBase
{
    public override string OpCode => OpCodes.REDUCE_MEAN;
    protected override float Reduce(IEnumerable<float> values) => values.Average();
    // Integer mean truncates like ORT (sum and divide in the integer domain).
    protected override long ReduceInt(IEnumerable<long> values)
    {
        long s = 0, n = 0;
        foreach (var v in values) { s += v; n++; }
        return n == 0 ? 0 : s / n;
    }
}

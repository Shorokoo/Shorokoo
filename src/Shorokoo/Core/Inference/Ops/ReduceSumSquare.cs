using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Inference.Ops;

internal sealed class ReduceSumSquareOp : ReduceOpBase
{
    public override string OpCode => OpCodes.REDUCE_SUM_SQUARE;
    protected override float Reduce(IEnumerable<float> values) => values.Select(v => v * v).Sum();
    protected override long ReduceInt(IEnumerable<long> values) { long s = 0; foreach (var v in values) s += v * v; return s; }
}

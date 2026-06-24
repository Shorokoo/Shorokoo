using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Inference.Ops;

internal sealed class ReduceL1Op : ReduceOpBase
{
    public override string OpCode => OpCodes.REDUCE_L1;
    protected override float Reduce(IEnumerable<float> values) => values.Select(MathF.Abs).Sum();
    protected override long ReduceInt(IEnumerable<long> values) { long s = 0; foreach (var v in values) s += Math.Abs(v); return s; }
}

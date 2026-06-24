using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Inference.Ops;

internal sealed class ReduceMaxOp : ReduceOpBase
{
    public override string OpCode => OpCodes.REDUCE_MAX;
    protected override float Reduce(IEnumerable<float> values) => values.Max();
    protected override long ReduceInt(IEnumerable<long> values) => values.Max();
}

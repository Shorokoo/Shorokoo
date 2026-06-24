using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Inference.Ops;

internal sealed class ReduceLogSumOp : ReduceOpBase
{
    public override string OpCode => OpCodes.REDUCE_LOG_SUM;
    protected override float Reduce(IEnumerable<float> values) => MathF.Log(values.Sum());
}

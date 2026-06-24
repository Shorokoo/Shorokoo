using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Inference.Ops;

internal sealed class ReduceL2Op : ReduceOpBase
{
    public override string OpCode => OpCodes.REDUCE_L2;
    protected override float Reduce(IEnumerable<float> values) => MathF.Sqrt(values.Select(v => v * v).Sum());
}

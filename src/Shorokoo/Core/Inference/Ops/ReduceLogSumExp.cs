using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Inference.Ops;

internal sealed class ReduceLogSumExpOp : ReduceOpBase
{
    public override string OpCode => OpCodes.REDUCE_LOG_SUM_EXP;
    protected override float Reduce(IEnumerable<float> values)
    {
        var arr = values.ToArray();
        var max = arr.Length == 0 ? 0 : arr.Max();
        float sum = 0;
        foreach (var v in arr) sum += MathF.Exp(v - max);
        return max + MathF.Log(sum);
    }
}

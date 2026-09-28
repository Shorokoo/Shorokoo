using Shorokoo.Core.Interpreter.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Interpreter.Ops;

internal sealed class ReduceLogSumExpOp : ReduceOpBase
{
    public override string OpCode => OpCodes.REDUCE_LOG_SUM_EXP;
    protected override float Reduce(IEnumerable<float> values)
    {
        // Shifting by the largest element keeps exp from overflowing; an infinite one is not
        // shifted by, so a group whose largest element is +inf or -inf gives that infinity.
        var arr = values.ToArray();
        var max = arr.Length == 0 ? 0 : arr.Max();
        var shift = float.IsFinite(max) ? max : 0;
        float sum = 0;
        foreach (var v in arr) sum += MathF.Exp(v - shift);
        return shift + MathF.Log(sum);
    }
}

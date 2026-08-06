using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Inference.Ops;

internal sealed class ReduceSumSquareOp : ReduceOpBase
{
    public override string OpCode => OpCodes.REDUCE_SUM_SQUARE;
    protected override float Reduce(IEnumerable<float> values) => values.Select(v => v * v).Sum();
    protected override long ReduceInt(IEnumerable<long> values, DType dtype) { long s = 0; foreach (var v in values) s += v * v; return s; }
    protected override ulong ReduceUInt(IEnumerable<ulong> values, DType dtype) { ulong s = 0; foreach (var v in values) unchecked { s += v * v; } return s; }
}

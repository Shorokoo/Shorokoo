using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Inference.Ops;

internal sealed class ReduceL1Op : ReduceOpBase
{
    public override string OpCode => OpCodes.REDUCE_L1;
    protected override float Reduce(IEnumerable<float> values) => values.Select(MathF.Abs).Sum();
    protected override long ReduceInt(IEnumerable<long> values, DType dtype) { long s = 0; foreach (var v in values) s += Math.Abs(v); return s; }
    // Unsigned lanes are already their own magnitude.
    protected override ulong ReduceUInt(IEnumerable<ulong> values, DType dtype) { ulong s = 0; foreach (var v in values) unchecked { s += v; } return s; }
}

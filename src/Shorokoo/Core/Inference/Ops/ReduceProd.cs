using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Inference.Ops;

internal sealed class ReduceProdOp : ReduceOpBase
{
    public override string OpCode => OpCodes.REDUCE_PROD;
    protected override float Reduce(IEnumerable<float> values) { float r = 1; foreach (var v in values) r *= v; return r; }
    // Exact integer product — critical for Shape → ReduceProd element-count chains.
    protected override long ReduceInt(IEnumerable<long> values) { long r = 1; foreach (var v in values) r *= v; return r; }
}

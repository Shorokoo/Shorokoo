using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Inference.Ops;

internal sealed class NegOp : UnaryNumericOp
{
    public override string OpCode => OpCodes.NEG;
    protected override float ApplyFloat(float x) => -x;
    protected override long ApplyInt(long x) => -x;
}

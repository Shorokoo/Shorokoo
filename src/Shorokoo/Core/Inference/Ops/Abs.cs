using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Inference.Ops;

internal sealed class AbsOp : UnaryNumericOp
{
    public override string OpCode => OpCodes.ABS;
    protected override float ApplyFloat(float x) => MathF.Abs(x);
    protected override long ApplyInt(long x) => Math.Abs(x);
}

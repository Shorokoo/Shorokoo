using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Inference.Ops;

internal sealed class ExpOp : UnaryFloatOp
{
    public override string OpCode => OpCodes.EXP;
    protected override float Apply(float x) => MathF.Exp(x);
}

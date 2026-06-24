using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Inference.Ops;

internal sealed class CoshOp : UnaryFloatOp
{
    public override string OpCode => OpCodes.COSH;
    protected override float Apply(float x) => MathF.Cosh(x);
}

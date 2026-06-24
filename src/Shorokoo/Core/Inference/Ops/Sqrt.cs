using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Inference.Ops;

internal sealed class SqrtOp : UnaryFloatOp
{
    public override string OpCode => OpCodes.SQRT;
    protected override float Apply(float x) => MathF.Sqrt(x);
}

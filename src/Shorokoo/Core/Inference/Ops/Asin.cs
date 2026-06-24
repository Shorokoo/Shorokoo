using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Inference.Ops;

internal sealed class AsinOp : UnaryFloatOp
{
    public override string OpCode => OpCodes.ASIN;
    protected override float Apply(float x) => MathF.Asin(x);
}

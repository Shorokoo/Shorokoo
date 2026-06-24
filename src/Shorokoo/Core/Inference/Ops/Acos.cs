using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Inference.Ops;

internal sealed class AcosOp : UnaryFloatOp
{
    public override string OpCode => OpCodes.ACOS;
    protected override float Apply(float x) => MathF.Acos(x);
}

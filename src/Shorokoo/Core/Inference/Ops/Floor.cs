using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Inference.Ops;

internal sealed class FloorOp : UnaryFloatOp
{
    public override string OpCode => OpCodes.FLOOR;
    protected override float Apply(float x) => MathF.Floor(x);
}

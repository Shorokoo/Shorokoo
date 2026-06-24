using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Inference.Ops;

internal sealed class AsinhOp : UnaryFloatOp
{
    public override string OpCode => OpCodes.ASINH;
    protected override float Apply(float x) => MathF.Asinh(x);
}

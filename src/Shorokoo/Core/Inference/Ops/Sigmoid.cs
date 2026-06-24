using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Inference.Ops;

internal sealed class SigmoidOp : UnaryFloatOp
{
    public override string OpCode => OpCodes.SIGMOID;
    protected override float Apply(float x) => 1f / (1f + MathF.Exp(-x));
}

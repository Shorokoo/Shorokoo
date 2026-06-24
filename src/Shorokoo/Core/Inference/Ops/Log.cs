using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Inference.Ops;

internal sealed class LogOp : UnaryFloatOp
{
    public override string OpCode => OpCodes.LOG;
    protected override float Apply(float x) => MathF.Log(x);
}

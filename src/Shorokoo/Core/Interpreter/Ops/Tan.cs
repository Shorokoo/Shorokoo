using Shorokoo.Core.Interpreter.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Interpreter.Ops;

internal sealed class TanOp : UnaryFloatOp
{
    public override string OpCode => OpCodes.TAN;
    protected override float Apply(float x) => MathF.Tan(x);
}

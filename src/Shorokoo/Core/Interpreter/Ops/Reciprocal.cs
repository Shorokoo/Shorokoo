using Shorokoo.Core.Interpreter.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Interpreter.Ops;

internal sealed class ReciprocalOp : UnaryFloatOp
{
    public override string OpCode => OpCodes.RECIPROCAL;
    protected override float Apply(float x) => 1f / x;
}

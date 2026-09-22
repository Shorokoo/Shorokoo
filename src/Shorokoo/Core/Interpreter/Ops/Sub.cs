using Shorokoo.Core.Interpreter.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Interpreter.Ops;

internal sealed class SubOp : BinaryNumericOp
{
    public override string OpCode => OpCodes.SUB;
    protected override float ApplyFloat(float a, float b) => a - b;
    protected override long ApplyInt(long a, long b) => a - b;
}

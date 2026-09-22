using Shorokoo.Core.Interpreter.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Interpreter.Ops;

internal sealed class BitwiseXorOp : BinaryNumericOp
{
    public override string OpCode => OpCodes.BITWISE_XOR;
    protected override long ApplyInt(long a, long b) => a ^ b;
}

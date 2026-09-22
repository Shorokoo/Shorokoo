using Shorokoo.Core.Interpreter.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Interpreter.Ops;

/// <summary>Elementwise unary <c>Softplus(x) = ln(1 + exp(x))</c>.</summary>
internal sealed class SoftplusOp : UnaryFloatOp
{
    public override string OpCode => OpCodes.SOFTPLUS;
    protected override float Apply(float x) => MathF.Log(1f + MathF.Exp(x));
}

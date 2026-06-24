using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// Elementwise unary <c>Mish(x) = x · tanh(softplus(x))</c>. Shape and dtype passthrough.
/// </summary>
internal sealed class MishOp : UnaryFloatOp
{
    public override string OpCode => OpCodes.MISH;
    protected override float Apply(float x) => x * MathF.Tanh(MathF.Log(1f + MathF.Exp(x)));
}

using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>Elementwise unary <c>Softsign(x) = x / (1 + |x|)</c>.</summary>
internal sealed class SoftsignOp : UnaryFloatOp
{
    public override string OpCode => OpCodes.SOFTSIGN;
    protected override float Apply(float x) => x / (1f + MathF.Abs(x));
}

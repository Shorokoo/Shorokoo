using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// Elementwise unary <c>HardSwish(x) = x · max(0, min(1, x/6 + 0.5))</c>.
/// </summary>
internal sealed class HardSwishOp : UnaryFloatOp
{
    public override string OpCode => OpCodes.HARD_SWISH;
    protected override float Apply(float x)
    {
        var t = x / 6f + 0.5f;
        if (t < 0f) t = 0f;
        else if (t > 1f) t = 1f;
        return x * t;
    }
}

using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Inference.Ops;

internal sealed class ErfOp : UnaryFloatOp
{
    public override string OpCode => OpCodes.ERF;

    protected override float Apply(float x) => Erf(x);

    // Abramowitz & Stegun 7.1.26 approximation (max error ~1.5e-7). Shared with
    // GeluOp's exact (approximate="none") path.
    internal static float Erf(float x)
    {
        const float a1 = 0.254829592f;
        const float a2 = -0.284496736f;
        const float a3 = 1.421413741f;
        const float a4 = -1.453152027f;
        const float a5 = 1.061405429f;
        const float p = 0.3275911f;
        int sign = x < 0 ? -1 : 1;
        x = MathF.Abs(x);
        float t = 1.0f / (1.0f + p * x);
        float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * MathF.Exp(-x * x);
        return sign * y;
    }
}

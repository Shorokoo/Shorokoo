using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Inference.Ops;

internal sealed class PowOp : BinaryNumericOp
{
    public override string OpCode => OpCodes.POW;
    protected override float ApplyFloat(float a, float b) => MathF.Pow(a, b);

    // Exact integer exponentiation for non-negative exponents (the previous
    // (long)Math.Pow(a, b) loses precision once the result needs more than 53 bits).
    // Negative exponents fall back to the double path: |a|>1 truncates to 0, a=±1
    // stays ±1, matching ORT's cast-from-double behavior.
    protected override long ApplyInt(long a, long b)
    {
        if (b < 0) return (long)Math.Pow(a, b);
        long result = 1;
        long baseVal = a;
        long exp = b;
        unchecked
        {
            while (exp > 0)
            {
                if ((exp & 1) == 1) result *= baseVal;
                baseVal *= baseVal;
                exp >>= 1;
            }
        }
        return result;
    }
}

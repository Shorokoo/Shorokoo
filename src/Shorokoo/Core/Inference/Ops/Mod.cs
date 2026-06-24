using System.Collections.Immutable;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// ONNX <c>Mod</c> with the <c>fmod</c> attribute (default 0). fmod=0 follows numpy.mod:
/// the result takes the sign of the DIVISOR (Python <c>%</c>); fmod=1 follows C fmod: the
/// result takes the sign of the DIVIDEND (C# <c>%</c>). The op used to apply C# <c>%</c>
/// unconditionally, which is wrong for the default integer mod with mixed signs.
/// </summary>
internal sealed class ModOp : QuickOp
{
    public override string OpCode => OpCodes.MOD;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var a = inputs.Length > 0 ? inputs[0] : null;
        var b = inputs.Length > 1 ? inputs[1] : null;
        var dtype = a?.DType ?? b?.DType ?? DType.Float32;
        var shape = ShapeHelpers.Broadcast(a?.Shape, b?.Shape);

        ImmutableArray<float>? fData = null;
        ImmutableArray<long>? iData = null;
        if (shape is not null && RuntimeTensorFactory.ShouldStoreData(shape, maxDataElements)
            && a is not null && b is not null)
        {
            var fmod = AttrAccess.GetBool(attrs, OnnxOpAttributeNames.AttrFmod, false);
            if (a.FloatData is { } af && b.FloatData is { } bf)
                fData = ImmutableArray.Create(ElementwiseBroadcast.Float(af, a.Shape!, bf, b.Shape!, shape,
                    fmod ? FmodFloat : ModFloat));
            else if (a.IntData is { } ai && b.IntData is { } bi)
                iData = ImmutableArray.Create(ElementwiseBroadcast.Int(ai, a.Shape!, bi, b.Shape!, shape,
                    fmod ? FmodInt : ModInt));
        }

        return [RuntimeTensorFactory.Create(dtype, shape) with { FloatData = fData, IntData = iData }];
    }

    // C fmod: sign of dividend; x % 0f is NaN in C#, matching fmod(x, 0).
    private static float FmodFloat(float a, float b) => a % b;

    // numpy.mod: sign of divisor. (fmod=0 with float tensors is disallowed by the spec,
    // but compute the consistent value anyway.)
    private static float ModFloat(float a, float b)
    {
        var r = a % b;
        if (r != 0 && (r < 0) != (b < 0)) r += b;
        return r;
    }

    // Integer division by zero is undefined per spec; return 0 instead of throwing.
    private static long FmodInt(long a, long b) => b == 0 ? 0 : a % b;

    private static long ModInt(long a, long b)
    {
        if (b == 0) return 0;
        var r = a % b;
        if (r != 0 && (r < 0) != (b < 0)) r += b;
        return r;
    }
}

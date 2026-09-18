using Shorokoo.Core.Nodes.NodeDefinitions;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;

namespace Shorokoo.Core.Lowering;

/// <summary>
/// <c>Softsign(x) = x / (1 + |x|)</c>, as Abs/Add/Div over a <c>1</c> carrying x's own dtype.
///
/// <para>The constant is emitted rather than built: a <c>Constant</c> holding float32 <c>1</c>,
/// then a <c>CastLike</c> against the input, so the addition's operands share a dtype whatever
/// the input's is. That keeps <see cref="IOpEmitter{T}"/> down to the single <c>Emit</c> method
/// — there is nothing a lowering needs that emitting an operator cannot express.</para>
///
/// <para>This changes nothing about what is exported: a <c>Softsign</c> node stays a
/// <c>Softsign</c> node in the graph and in the ONNX written from it. The decomposition exists
/// only for an engine that has no <c>Softsign</c> of its own.</para>
/// </summary>
internal sealed class SoftsignLowering : OpLowering
{
    public override string OpCode => SOFTSIGN;

    public override T[] Lower<T>(IOpEmitter<T> emitter, T?[] inputs, OnnxCSharpAttributes attributes)
        where T : class
    {
        var x = inputs[0];
        var one = emitter.Emit(CONSTANT, [], [(AttrValueFloat, 1.0f)], 1)[0];
        var oneLikeX = emitter.Emit(CAST_LIKE, [one, x], [], 1)[0];
        var magnitude = emitter.Emit(ABS, [x], [], 1)[0];
        var denominator = emitter.Emit(ADD, [oneLikeX, magnitude], [], 1)[0];
        return emitter.Emit(DIV, [x, denominator], [], 1);
    }
}

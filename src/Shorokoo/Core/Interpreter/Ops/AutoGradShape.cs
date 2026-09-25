using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Interpreter.Helpers;

namespace Shorokoo.Core.Interpreter.Ops;

/// <summary>
/// The shape rule for an <c>AUTO_GRAD</c> node left standing in a training step whose gradient the
/// execution backend computes (<see cref="TrainingFormats.OnnxAutoGrad"/>): gradient <i>i</i> has
/// the dtype, shape and rank of the tensor it differentiates with respect to, input <i>i + 1</i>
/// (input 0 is the loss). No values: a gradient is known only once the backend has run.
///
/// <para>Deliberately <b>not</b> discovered by <see cref="OpRegistry"/> — its constructor is not
/// public — and installed only around the one shape inference of such a step, through the
/// thread-scoped <see cref="OpRegistry.Override"/>. Every other engine run sees no rule for
/// <c>AUTO_GRAD</c>, exactly as before, so the default training path's inference cannot change.</para>
/// </summary>
internal sealed class AutoGradShapeOp : QuickOp
{
    /// <summary>The one instance, for <see cref="OpRegistry.Override"/>.</summary>
    internal static AutoGradShapeOp Instance { get; } = new();

    private AutoGradShapeOp()
    {
    }

    public override string OpCode => InternalOpCodes.AUTO_GRAD;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var grads = new RuntimeTensor[Math.Max(0, inputs.Length - 1)];
        for (int i = 0; i < grads.Length; i++)
        {
            var wrt = inputs[i + 1];
            grads[i] = new RuntimeTensor
            {
                DType = wrt?.DType ?? DType.Invalid,
                Shape = wrt?.Shape,
                MaxShape = wrt?.MaxShape ?? wrt?.Shape,
                Rank = wrt?.Rank,
                MaxRank = wrt?.MaxRank ?? wrt?.Rank,
            };
        }
        return grads;
    }
}

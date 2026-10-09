using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

using static OnnxOp;
using static OnnxOpAttributeNames;
using static OpCodes;

/// <summary>
/// <c>SoftmaxCrossEntropyLoss</c> handed to ONNX Runtime as the <c>LogSoftmax</c> over the class
/// axis and the <c>NegativeLogLikelihoodLoss</c> of it that the spec defines it as.
///
/// <para>ONNX Runtime has no kernel for the operator, on the host or on a card, and runs the spec's
/// function body in its place. That body reshapes the scores to <c>[N, C, -1]</c> and transposes
/// them to <c>[N, D, C]</c> before the <c>LogSoftmax</c> and back after it. For the common
/// <c>[N, C]</c> scores the transposes move nothing, yet each is a full copy of the scores: on a
/// card two passes over a tensor the size of the logits, each as long as an elementwise pass,
/// every step. And over an empty batch ONNX Runtime refuses the reshape, whose <c>-1</c> it cannot
/// infer from no elements (Shorokoo/Shorokoo#550). <c>LogSoftmax</c> takes its axis as it stands,
/// so the call computes the same log-probabilities over axis 1 directly, and the loss of them is the
/// same <c>NegativeLogLikelihoodLoss</c> with the call's weight, ignore index and reduction.</para>
/// </summary>
internal sealed class CrossEntropyAsLogSoftmaxWorkaround : KernelWorkaround
{
    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([SOFTMAX_CROSS_ENTROPY_LOSS], StringComparer.Ordinal);

    public override bool Applies(WorkaroundSite site) => true;

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
    {
        var logProb = LogSoftmax(inputs[0]!, axis: 1);
        var loss = NegativeLogLikelihoodLoss(logProb, inputs[1]!, inputs.Length > 2 ? inputs[2] : null,
            site.Attributes.GetAttributeObj(AttrIgnoreIndex) as long?, site.Attributes.GetAttributeObj(AttrReduction) as string);
        return site.OutputCount > 1 ? [loss, site.IsOutputPresent(1) ? logProb : null] : [loss];
    }
}

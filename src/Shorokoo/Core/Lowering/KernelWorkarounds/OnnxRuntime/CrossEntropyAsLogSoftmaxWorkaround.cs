using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

using static OnnxOp;
using static OnnxOpAttributeNames;
using static OpCodes;

/// <summary>
/// <c>SoftmaxCrossEntropyLoss</c> handed to ONNX Runtime's CUDA execution provider as the
/// <c>LogSoftmax</c> over the class axis and the <c>NegativeLogLikelihoodLoss</c> of it that the
/// spec defines it as.
///
/// <para>The provider has no kernel for the operator, so ONNX Runtime runs the spec's function body
/// in its place, and that body views the scores as <c>[N, C, D]</c> and transposes them to
/// <c>[N, D, C]</c> before the <c>LogSoftmax</c> and back after it. For the common
/// <c>[N, C]</c> scores the transposes move nothing, yet each is a full copy of the scores on the
/// card: two passes over a tensor the size of the logits, each as long as an elementwise pass,
/// every step. <c>LogSoftmax</c> takes its axis as it stands, so the call computes the same
/// log-probabilities over axis 1 directly, and the loss of them is the same
/// <c>NegativeLogLikelihoodLoss</c> with the call's weight, ignore index and reduction.</para>
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

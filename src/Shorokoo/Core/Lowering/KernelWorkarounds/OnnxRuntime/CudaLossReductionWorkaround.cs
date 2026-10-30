using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

using static OnnxOp;
using static OnnxOpAttributeNames;
using static OpCodes;

/// <summary>
/// A <c>NegativeLogLikelihoodLoss</c> that sums or averages its losses over targets that may be
/// empty, or averages <c>float16</c> losses, written as the loss of each target and its sum,
/// divided for the mean by the sum of the weights of the targets not ignored
/// (Shorokoo/Shorokoo#536).
///
/// <para>ONNX Runtime has no kernel for the operator and runs the spec's function body in its
/// place, whose reductions take <c>keepdims</c> 0; over an empty batch the CUDA execution provider
/// refuses them, where the spec gives a sum of 0. Written out here, the reductions are Shorokoo's
/// own and <see cref="CudaReduceEmptyAxisWorkaround"/>, after this, keeps their axes. The weight
/// of a target is its class's weight, 1 without a weight input, and 0 for an ignored target, as
/// the spec says; without a weight or an ignore index the mean divides by the number of targets,
/// counted on the card as a sum of ones rather than read off the shape, which would be copied there.
/// A call whose input's dimensions follow from the stated ones and are none of them 0
/// (<see cref="WorkaroundSite.ShapeOf"/>) and one with reduction <c>none</c> keep the call as it
/// stands.</para>
///
/// <para>A <c>float16</c> mean is rewritten the same way whatever its targets' shape, with the
/// losses, their weights and the division in <c>float32</c> and the mean cast back
/// (Shorokoo/Shorokoo#553): the function body sums and divides in <c>float16</c>, whose range
/// ends at 65504, so over more targets than that its count of them, and a sum of losses past it,
/// is infinite, and the mean comes out as 0 or NaN. Only the losses of the targets, one per
/// target, are cast, not the scores; each loss is the one the call computes, and the mean of them
/// is rounded to <c>float16</c> once.</para>
///
/// <para>Part of the CUDA set only (<see cref="Shorokoo.Core.Backends.KernelWorkaroundSets.OnnxRuntimeCuda"/>),
/// after the cross-entropy rewrite of the shared set, which builds such calls: the CPU provider
/// computes the function body as the spec does.</para>
/// </summary>
internal sealed class CudaLossReductionWorkaround : KernelWorkaround
{
    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([NEGATIVE_LOG_LIKELIHOOD_LOSS], StringComparer.Ordinal);

    public override bool ReadsShapes => true;

    public override bool Applies(WorkaroundSite site)
        => Reduction(site) is not "none"
           && (HalfMean(site)
               || (!(site.ShapeOf(1) is { } targets && !targets.Dims.Contains(0)) && Reductions.InputMayBeEmpty(site)));

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
    {
        var (x, target, weight) = (inputs[0]!, inputs[1]!, inputs.Length > 2 ? inputs[2] : null);
        var ignore = site.Attributes.GetAttributeObj(AttrIgnoreIndex) as long?;
        var losses = NegativeLogLikelihoodLoss(x, target, weight, ignore, "none");
        var half = HalfMean(site);
        Variable Summed(Variable v) => ReduceSum(half ? Cast(v, null, DType.Float32) : v, null, false, null);
        var sum = Summed(losses);
        if (Reduction(site) == "sum") return [sum];

        Variable? kept = ignore is { } index ? Not(Equal(target, CastLike(Globals.Scalar(index), target, null))) : null;
        Variable? weights = weight is null
            ? kept is null ? null : CastLike(kept, x, null)
            : kept is null ? Gather(weight, target, 0)
            : Where(kept, Gather(weight, Where(kept, target, CastLike(Globals.Scalar(0L), target, null)), 0), CastLike(Globals.Scalar(0f), x, null));
        var mean = Div(sum, Summed(weights ?? CastLike(Equal(target, target), x, null)));
        return [half ? Cast(mean, null, site.DTypeOf(0)) : mean];
    }

    private static bool HalfMean(WorkaroundSite site)
        => Reduction(site) == "mean" && site.DTypeOf(0).IsSameElementTypeAs(DType.Float16);

    private static string Reduction(WorkaroundSite site)
        => (site.Attributes.GetAttributeObj(AttrReduction) as string ?? "mean").ToLowerInvariant();
}

using Shorokoo.Modules.Initializers;
using Shorokoo.Modules.Layers;

namespace Shorokoo.Tests.Modules;

// ---------------------------------------------------------------------------
// Self-checking / training-rig [Module]s for the classic layers added on top
// of the baseline NN library: Conv3d (NCDHW) and BatchNorm1d ([N, C]).
//
// Conv3d is value-checked the same way as NNConv2dMatchesStaticConv: a dynamic
// SHRK_CONV geometry must equal a hand-built static-attribute NN.Conv with the
// same geometry and identically-seeded KaimingUniform weights, so it runs
// through AutoTest.AdvancedTestGraph (returns Scalar<bit>).
//
// BatchNorm1d carries Globals.StateUpdate links (STATE_UPDATE_LINK is not an
// executable ORT op in the plain inference pipeline), so — like BatchNorm2d —
// it is exercised via TrainingRig-based tests through the NNTinyBatchNorm1d*
// models below, not AutoTest.
// ---------------------------------------------------------------------------

/// <summary>Conv3d (dynamic SHRK_CONV geometry, NCDHW) must equal the static-attribute NN.Conv with identical geometry and weights.</summary>
[Module]
public partial class NNConv3dMatchesStaticConv
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var outChannels = Scalar(3L);
        var y = Conv3d.Call(outChannels, Scalar(3L), Scalar(2L), Scalar(1L), Scalar(1L), Scalar(1L), Scalar(true), x);

        Scalar<int64> inChannels = x.ShapeTensor()[1];
        var wRef = KaimingUniform.Init([outChannels, inChannels, Scalar(3L), Scalar(3L), Scalar(3L)]);
        var yRef = NN.Conv(x, wRef, VectorFill(outChannels, 0f), AutoPad.NotSet,
            dilations: [1L, 1L, 1L], group: 1L, kernelShape: [3L, 3L, 3L],
            pads: [1L, 1L, 1L, 1L, 1L, 1L], strides: [2L, 2L, 2L]);

        var diff = (y - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return diff < Scalar(1e-3f) * (Scalar(1f) + yRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar());
    }
}

/// <summary>BatchNorm1d in training mode (momentum 0.9, eps 1e-5) over a [N, C] input, followed by per-channel mean: [N, C] → [C].</summary>
[Module]
public partial class NNBatchNorm1dTrainGradModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var y = BatchNorm1d.Model(Scalar(0.9f), Scalar(1e-5f), Scalar(true)).Call(input);
        Vector<int64> batchAxis = [Scalar(0L)];
        return y.Reduce(ReduceKind.Mean, batchAxis, keepDims: false);
    }
}

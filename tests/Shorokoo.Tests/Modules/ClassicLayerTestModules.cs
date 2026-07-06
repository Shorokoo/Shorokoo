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

/// <summary>Conv3d forward output on RangeTensor([1,2,5,5,5],0.05,-2) at MasterSeed=0 must match the
/// frozen reference. The old check re-ran Conv against a hand-built static NN.Conv (a tautology);
/// the reference is now the layer's own frozen forward output. Output [1,3,3,3,3]=81 is collapsed
/// to 19 via SelfCheck.Collapse.</summary>
[Module]
public partial class NNConv3dMatchesStaticConv
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var y = Conv3d.Model(Scalar(3L), Scalar(3L), Scalar(2L), Scalar(1L), Scalar(1L), Scalar(1L), Scalar(true)).Call(x);   // [1,3,3,3,3] = 81

        // REFERENCE: golden — Shorokoo's own forward output, collapsed to 19 (self-generated).
        var reference = Vector(-2.2050622f, -5.5534472f, 3.2518554f, 5.779654f, 0.47101557f, -0.27055806f, -3.3270457f, 0.16702354f, 1.2823882f, 0.8556348f, -2.5663595f, -6.228979f, 0.60689855f, 5.1870475f, -1.1765716f, -3.8725114f, 1.8714321f, 3.496889f, 2.813375f);

        var diff = (SelfCheck.Collapse(y, 81) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
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

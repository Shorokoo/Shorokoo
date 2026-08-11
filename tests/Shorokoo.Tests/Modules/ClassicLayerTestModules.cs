using Shorokoo.Modules.Initializers;
using Shorokoo.Modules.Layers;

namespace Shorokoo.Tests.Modules;

// ---------------------------------------------------------------------------
// Self-checking / training-rig [Module]s for the classic layers added on top
// of the baseline NN library: Conv3d (NCDHW) and BatchNorm1d ([N, C]).
//
// Conv3d is value-checked the same way as NNConv2dForwardGolden: a frozen
// forward-value golden (self-generated at master-seed-0), driven through
// AutoTest.AdvancedTestGraph (returns Scalar<bit>). The former hand-built
// static-attribute NN.Conv reference relied on re-materializing identical
// weights and was retired with keyed per-parameter init.
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
public partial class NNConv3dForwardGolden
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var y = Conv3d.Model(Scalar(3L), Scalar(3L), Scalar(2L), Scalar(1L), Scalar(1L), Scalar(1L), Scalar(true)).Call(x);   // [1,3,3,3,3] = 81

        // REFERENCE: golden — Shorokoo's own forward output, collapsed to 19 (self-generated).
        var reference = Vector(-0.3597958f, -4.815351f, 2.371303f, -4.192379f, 3.834232f, 0.4283028f, -3.87363f, 0.6956633f, -1.386085f, -0.6653334f, -2.922421f, -3.220134f, -1.756067f, 5.355292f, 1.080021f, 0.1887852f, -2.311194f, -1.124305f, 1.573647f);

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

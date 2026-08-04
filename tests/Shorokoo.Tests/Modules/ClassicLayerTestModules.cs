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
        var reference = Vector(-0.16921827f, -1.9305707f, -2.197832f, 6.4505725f, -0.35098982f, -0.4969166f, 1.7714214f, 0.39273822f, -1.8290322f, -0.26773334f, 1.6262126f, 0.9653335f, -1.2342063f, -2.2644093f, -0.1615113f, -0.0067960024f, 2.0993922f, 0.57456166f, -0.56375825f);

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

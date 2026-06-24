using Shorokoo.Modules.Initializers;
using Shorokoo.Modules.Losses;

namespace Shorokoo.Tests.Modules;

// ---------------------------------------------------------------------------
// Self-checking [Module]s for KLDivLoss and the extra initializers
// (TruncatedNormal / LeCunNormal). Each module returns a single Scalar<bit> so
// AutoTest.AdvancedTestGraph treats it as a self-checking graph (the bit must
// be true), keeping the xUnit tests one-liners — mirroring
// NNLossClosedFormChecks in NNLibraryTestModules.cs.
//
// The initializer checks exploit that the library initializers are
// seeded/deterministic: TruncatedNormal clamps every draw to [-2, 2] (so the
// max abs value is bounded by 2), and LeCunNormal of shape [in, out] has
// empirical std ≈ sqrt(1 / fanIn) for a largish sample.
// ---------------------------------------------------------------------------

/// <summary>
/// KLDivLoss closed forms (predictions = log q, targets = p), batchmean over N=1:
///   p = q = [0.5, 0.5] (log q = [ln 0.5, ln 0.5]) → KL = 0;
///   p = [1, 0], q = [0.5, 0.5] → KL = 1·(ln 1 − ln 0.5) + 0 = ln 2 ≈ 0.6931
///     (the p = 0 term is dropped by the Where guard).
/// The runtime input is unused; values are hardcoded so the check is exact.
/// </summary>
[Module]
public partial class KLDivClosedForm
{
    public static Scalar<bit> Inline(Tensor<float32> dummy)
    {
        var ln2 = Scalar(0.69314718f);

        // p = q: KL = 0.
        var logqUniform = Tensor(new long[] { 1L, 2L }, -0.69314718f, -0.69314718f);
        var pUniform = Tensor(new long[] { 1L, 2L }, 0.5f, 0.5f);
        var pen = (KLDivLoss.Inline(logqUniform, pUniform) - Scalar(0f)).Abs();

        // p = [1, 0] against uniform q: KL = ln 2.
        var pOneHot = Tensor(new long[] { 1L, 2L }, 1f, 0f);
        pen = pen + (KLDivLoss.Inline(logqUniform, pOneHot) - ln2).Abs();

        return pen < Scalar(1e-4f);
    }
}

/// <summary>
/// Property checks for the extra initializers (built via <c>.Init</c> at shape
/// [64, 64], fanIn = 64):
///   TruncatedNormal: max abs sample ≤ 2 (clamp bound);
///   LeCunNormal: empirical std ≈ sqrt(1/64) = 0.125 within a loose tolerance.
/// The runtime input is unused; shapes are hardcoded for a stable sample std.
/// </summary>
[Module]
public partial class InitializerProps
{
    public static Scalar<bit> Inline(Tensor<float32> dummy)
    {
        var trunc = TruncatedNormal.Init([Scalar(64L), Scalar(64L)]);
        var maxAbs = trunc.Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        var truncOk = (maxAbs <= Scalar(2f + 1e-4f)).IfElse(Scalar(1L), Scalar(0L));

        var lecun = LeCunNormal.Init([Scalar(64L), Scalar(64L)]);
        var mean = lecun.Reduce(ReduceKind.Mean, keepDims: false).Scalar();
        var meanSq = (lecun * lecun).Reduce(ReduceKind.Mean, keepDims: false).Scalar();
        var std = (meanSq - mean * mean).Sqrt();
        var lecunOk = ((std - Scalar(0.125f)).Abs() <= Scalar(0.04f)).IfElse(Scalar(1L), Scalar(0L));

        return truncOk + lecunOk > Scalar(1L); // both must hold
    }
}

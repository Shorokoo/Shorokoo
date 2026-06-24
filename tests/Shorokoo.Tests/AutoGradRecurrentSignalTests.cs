namespace Shorokoo.Tests;

/// <summary>
/// Phase 4 AD-B3 tests for the recurrent / signal / misc gradient completion batch:
/// reverse-direction RNN/GRU/LSTM, the DFT onesided + inverse adjoints, the STFT
/// overlap-add adjoint (formerly a silent ZERO-STUB), the generic 2-D/3-D AffineGrid
/// gradient, training-mode BatchNormalization, the tanh-approximation Gelu derivative,
/// and the new AD003 attribute-envelope guards. Follows the same one-liner pattern as
/// <see cref="AutoGradEngineTests"/>: each scenario drives
/// <see cref="AutoTest.AdvancedTestGraph{TModule}"/> against a self-checking module from
/// <c>Modules/AutoGradRecurrentSignalModules.cs</c>; for the throwing scenarios the
/// <c>AutoDiffNotSupportedException</c> (AD003) surfaces from the AUTO_GRAD lowering
/// during concretization — out of the <c>AdvancedTestGraph</c> call itself, before any
/// execution backend runs.
/// </summary>
[Trait("Domain", "AutoDiff")]
[Trait("Purpose", "Coverage")]
public class AutoGradRecurrentSignalTests
{
    /// <summary>
    /// Gelu approximate="tanh": the gradient must use the tanh-approximation derivative
    /// (the FD probes the tanh forward, so the old always-erf gradient fails here).
    /// Probed away from 0 where the two derivative forms differ measurably.
    /// </summary>
    [Fact]
    public void TestAutoGradGeluTanhGradient()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGeluTanhCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 0.7f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGeluTanhCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], -1.3f)]));
    }

    /// <summary>
    /// Reverse-direction RNN / GRU / LSTM gradients (implemented this batch by reducing
    /// the reverse scan to the forward BPTT on time-flipped x/dY). The losses consume
    /// both the full Y sequence and the final state(s), FD-checked against ORT's own
    /// reverse forward.
    /// </summary>
    [Fact]
    public void TestAutoGradRecurrentReverseDirection()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradRnnReverseCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 0.3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGruReverseCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 0.3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradLstmReverseCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 0.3f)]));
    }

    /// <summary>
    /// AD003 guards for the recurrent attribute envelope: bidirectional scans, the clip
    /// attribute, and LSTM peephole weights must fail loudly at lowering instead of
    /// producing a silently wrong gradient subgraph.
    /// </summary>
    [Fact]
    public void TestAutoGradRnnBidirectionalThrows()
    {
        var ex = Assert.Throws<AutoDiffNotSupportedException>(() =>
            AutoTest.AdvancedTestGraph<AutoGradRnnBidirectionalThrowCheck>(
                hyperparamInputs: [],
                runtimeInputs: [TensorData(DType.Float32, [], 0.3f)]));
        Assert.Equal(ErrorCodes.AD003, ex.ErrorCode);
        Assert.Contains("bidirectional", ex.Message);
    }

    [Fact]
    public void TestAutoGradGruClipThrows()
    {
        var ex = Assert.Throws<AutoDiffNotSupportedException>(() =>
            AutoTest.AdvancedTestGraph<AutoGradGruClipThrowCheck>(
                hyperparamInputs: [],
                runtimeInputs: [TensorData(DType.Float32, [], 0.3f)]));
        Assert.Equal(ErrorCodes.AD003, ex.ErrorCode);
        Assert.Contains("clip", ex.Message);
    }

    [Fact]
    public void TestAutoGradLstmPeepholeThrows()
    {
        var ex = Assert.Throws<AutoDiffNotSupportedException>(() =>
            AutoTest.AdvancedTestGraph<AutoGradLstmPeepholeThrowCheck>(
                hyperparamInputs: [],
                runtimeInputs: [TensorData(DType.Float32, [], 0.3f)]));
        Assert.Equal(ErrorCodes.AD003, ex.ErrorCode);
        Assert.Contains("peephole", ex.Message);
    }

    /// <summary>
    /// DeformConv's silent ZERO-STUB (null gradients → frozen parameters) is replaced
    /// with a loud AD003 at lowering.
    /// </summary>
    [Fact]
    public void TestAutoGradDeformConvThrows()
    {
        var ex = Assert.Throws<AutoDiffNotSupportedException>(() =>
            AutoTest.AdvancedTestGraph<AutoGradDeformConvThrowCheck>(
                hyperparamInputs: [],
                runtimeInputs: [TensorData(DType.Float32, [], 0.5f)]));
        Assert.Equal(ErrorCodes.AD003, ex.ErrorCode);
        Assert.Contains("DeformConv", ex.Message);
    }

    /// <summary>
    /// DFT onesided=1 gradient (RFFT adjoint = zero-pad the kept bins + full-DFT
    /// adjoint) and the inverse=1 path, FD-checked with per-element weighted losses.
    /// </summary>
    [Fact]
    public void TestAutoGradDftOnesidedAndInverse()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradDftOnesidedCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [1L, 4L, 1L])]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradDftInverseCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [1L, 4L, 1L])]));
    }

    /// <summary>
    /// STFT overlap-add adjoint (formerly a ZERO-STUB that silently froze everything
    /// upstream of an STFT): signal gradient with overlapping frames, window gradient,
    /// and the windowless frame_length-driven path.
    /// </summary>
    [Fact]
    public void TestAutoGradStftGradients()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradStftSignalCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [1L, 8L, 1L])]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradStftWindowCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradStftNoWindowCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [1L, 10L, 1L])]));
    }

    /// <summary>
    /// AffineGrid 3-D gradient (the rewritten gradient recovers the base grid from the
    /// op itself, so 2-D and 3-D share one path; 2-D stays covered by
    /// AutoGradAffineGridMultiBatchCheck in the ops suite).
    /// </summary>
    [Fact]
    public void TestAutoGradAffineGrid3D()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradAffineGrid3DCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1.2f)]));
    }

    /// <summary>
    /// Training-mode BatchNormalization gradient (implemented this batch — was an AD-B2
    /// AD003 guard): dx carries the batch-statistics backprop terms (FD check), and
    /// dscale/dbias use the batch-stat x̂ (in-graph closed forms).
    /// </summary>
    [Fact]
    public void TestAutoGradBatchNormTrainingMode()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradBatchNormTrainingInputCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradBatchNormTrainingScaleBiasCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f), TensorData(DType.Float32, [], 1f)]));
    }
}

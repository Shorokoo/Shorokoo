using static Shorokoo.Tests.AutoGradOpsRunners;

namespace Shorokoo.Tests;

/// <summary>
/// Recurrent / signal / misc gradients: reverse-direction RNN/GRU/LSTM, DFT onesided +
/// inverse adjoints, the STFT overlap-add adjoint, the 3-D AffineGrid gradient,
/// training-mode BatchNormalization, the tanh-approximation Gelu derivative, and the
/// AD003 attribute-envelope guards. Each row drives a self-checking module from
/// <c>Modules/AutoGradRecurrentSignalModules.cs</c> through
/// <see cref="AutoTest.AdvancedTestGraph{TModule}"/>; for the throwing rows the AD003
/// surfaces from the AUTO_GRAD lowering during concretization, out of the
/// <c>AdvancedTestGraph</c> call itself.
/// </summary>
[Trait("Domain", "AutoDiff")]
[Trait("Purpose", "Coverage")]
public class AutoGradRecurrentSignalTests
{
    private static void Throws<TModule>(string expectedWord)
    {
        var ex = Assert.Throws<AutoDiffNotSupportedException>(() =>
            AutoTest.AdvancedTestGraph<TModule>(
                [], [TensorData(DType.Float32, [], 0.3f)]));
        Assert.Equal(ErrorCodes.AD003, ex.ErrorCode);
        Assert.Contains(expectedWord, ex.Message);
    }

    [Fact]
    public void TestAutoGradGeluTanhReverseRecurrentAndBatchNormTrainingGradients()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGeluTanhCheck>(
            [], [TensorData(DType.Float32, [2L], 0.7f, -1.3f)]));
        Run<AutoGradRnnReverseCheck>(0.3f);
        Run<AutoGradGruReverseCheck>(0.3f);
        Run<AutoGradLstmReverseCheck>(0.3f);
        Run<AutoGradBatchNormTrainingInputCheck>(0.5f);
        Run<AutoGradBatchNormTrainingScaleBiasCheck>(2f, 1f);
    }

    [Fact]
    public void TestAutoGradDftStftAndAffineGrid3DGradients()
    {
        RunSmall<AutoGradDftOnesidedCheck>([1L, 4L, 1L]);
        RunSmall<AutoGradDftInverseCheck>([1L, 4L, 1L]);
        RunSmall<AutoGradStftSignalCheck>([1L, 8L, 1L]);
        Run<AutoGradStftWindowCheck>(0.5f);
        RunSmall<AutoGradStftNoWindowCheck>([1L, 10L, 1L]);
        Run<AutoGradAffineGrid3DCheck>(1.2f);
    }

    [Fact]
    public void TestAutoGradUnsupportedAttributeEnvelopesThrowAD003()
    {
        Throws<AutoGradRnnBidirectionalThrowCheck>("bidirectional");
        Throws<AutoGradGruClipThrowCheck>("clip");
        Throws<AutoGradLstmPeepholeThrowCheck>("peephole");
        Throws<AutoGradDeformConvThrowCheck>("DeformConv");
    }
}

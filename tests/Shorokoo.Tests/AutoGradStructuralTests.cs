namespace Shorokoo.Tests;

/// <summary>
/// Gradient correctness for the structural autodiff family (Conv / ConvTranspose /
/// MaxPool / AveragePool / GlobalAveragePool / BatchNorm / LayerNorm / GroupNorm /
/// InstanceNorm / Concat / Split / Sum / Min / Max / Mean / Dropout). Each row drives a
/// self-checking module from <c>Modules/AutoGradStructuralModules.cs</c> through
/// <see cref="AutoTest.AdvancedTestGraph{TModule}"/>; the module's <c>Inline</c> verifies
/// the analytical gradient in-graph and returns <c>Scalar&lt;bit&gt;</c>.
/// </summary>
[Trait("Domain", "AutoDiff")]
[Trait("Purpose", "Coverage")]
public class AutoGradStructuralTests
{
    private static void Run<TModule>(params float[] scalars) =>
        Assert.True(AutoTest.AdvancedTestGraph<TModule>(
            [], [.. scalars.Select(v => TensorData(DType.Float32, [], v))]));

    private static void RunNoQee<TModule>(params float[] scalars) =>
        Assert.True(AutoTest.AdvancedTestGraph<TModule>(
            [], [.. scalars.Select(v => TensorData(DType.Float32, [], v))],
            testQuickEngineExecution: false));

    private static void RunSmall<TModule>(long[] shape) =>
        Assert.True(AutoTest.AdvancedTestGraph<TModule>(
            [], [TensorDataWithSmallVals(DType.Float32, shape)]));

    [Fact]
    public void TestAutoGradStructuralConvAndConvTransposeGradients()
    {
        RunSmall<AutoGradStructConvStridePadCheck>([1L, 2L, 5L, 5L]);
        RunSmall<AutoGradStructConvDilationCheck>([1L, 2L, 7L, 7L]);
        RunSmall<AutoGradStructConvWeightStride2Check>([1L, 2L, 5L, 5L]);
        RunSmall<AutoGradStructConvGroupedInputCheck>([1L, 4L, 5L, 5L]);
        RunSmall<AutoGradStructConvGroupedWeightCheck>([1L, 4L, 5L, 5L]);
        RunSmall<AutoGradStructConvGroupedWeightStridePadCheck>([1L, 4L, 5L, 5L]);
        RunSmall<AutoGradStructConvTransposeWeightCheck>([1L, 3L, 5L, 5L]);
        RunSmall<AutoGradStructConvTransposeStride2Check>([1L, 2L, 4L, 4L]);
        RunSmall<AutoGradStructConvTransposeWeightStride2Check>([1L, 2L, 4L, 4L]);
        RunSmall<AutoGradStructConvTransposeGroupedWeightCheck>([1L, 4L, 5L, 5L]);
    }

    [Fact]
    public void TestAutoGradStructuralPoolingAndNormalizationGradients()
    {
        Run<AutoGradStructMaxPoolOverlapCheck>(1f);
        Run<AutoGradStructMaxPoolPadCheck>(1f);
        RunNoQee<AutoGradStructMaxPoolCeilCheck>(1f);
        RunNoQee<AutoGradStructMaxPoolDilationCheck>(1f);
        RunNoQee<AutoGradStructAvgPoolDilationCheck>(1f);
        Run<AutoGradStructGlobalAvgPool5DCheck>(3f);
        Run<AutoGradStructBatchNormScaleBiasCheck>(2f, 1f);
        Run<AutoGradStructLayerNormScaleBiasCheck>(1.5f, -0.5f);
        Run<AutoGradStructLayerNormAxis1Check>(0.5f);
        Run<AutoGradStructGroupNormScaleBiasCheck>(2f, 1f);
        Run<AutoGradStructInstanceNormScaleBiasCheck>(2f, 1f);
    }

    [Fact]
    public void TestAutoGradStructuralConcatSplitVariadicAndDropoutGradients()
    {
        Run<AutoGradStructConcatNegAxisCheck>(3f, 7f);
        Run<AutoGradStructSplitNegAxisCheck>(2f);
        RunSmall<AutoGradStructSplitPartialUseCheck>([4L]);
        Run<AutoGradStructSumBroadcastCheck>(3f, 5f);
        Run<AutoGradStructMeanBroadcastCheck>(3f, 5f);
        Run<AutoGradStructMaxBroadcastCheck>(2f, 5f);
        Run<AutoGradStructMinBroadcastCheck>(7f, 2f);
        Run<AutoGradStructDropoutRatioInputCheck>(2f);
    }
}

namespace Shorokoo.Tests;

/// <summary>
/// Phase 4 AD-B1 gradient-correctness tests for the structural autodiff family
/// (Conv / ConvTranspose / MaxPool / AveragePool / GlobalAveragePool /
/// BatchNorm / LayerNorm / GroupNorm / InstanceNorm / Concat / Split /
/// Sum / Min / Max / Mean / Dropout). Follows the same one-liner pattern as
/// <see cref="AutoGradOpsCoverageTests"/>: each test drives
/// <see cref="AutoTest.AdvancedTestGraph{TModule}"/> against a self-checking
/// module from <c>Modules/AutoGradStructuralModules.cs</c> whose <c>Inline</c>
/// verifies the analytical gradient in-graph and returns <c>Scalar&lt;bit&gt;</c>.
/// </summary>
[Trait("Domain", "AutoDiff")]
[Trait("Purpose", "Coverage")]
public class AutoGradStructuralTests
{
    [Fact]
    public void TestAutoGradStructuralConvCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradStructConvStridePadCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [1L, 2L, 5L, 5L])]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradStructConvDilationCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [1L, 2L, 7L, 7L])]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradStructConvWeightStride2Check>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [1L, 2L, 5L, 5L])]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradStructConvGroupedInputCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [1L, 4L, 5L, 5L])]));
    }

    [Fact]
    public void TestAutoGradStructuralConvTransposeCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradStructConvTransposeWeightCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [1L, 3L, 5L, 5L])]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradStructConvTransposeStride2Check>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [1L, 2L, 4L, 4L])]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradStructConvTransposeWeightStride2Check>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [1L, 2L, 4L, 4L])]));
    }

    /// <summary>
    /// Grouped (group != 1) weight gradients for Conv and ConvTranspose —
    /// historically returned as null by <c>ConvGradient</c>/<c>ConvTransposeGradient</c>
    /// and lowered to zeros (silently freezing grouped/depthwise kernels in
    /// training); now implemented via the per-group swapped-roles Conv + Concat
    /// in <c>GroupedConvWeightGradient</c>.
    /// </summary>
    [Fact]
    public void TestAutoGradStructuralConvGroupedWeight()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradStructConvGroupedWeightCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [1L, 4L, 5L, 5L])]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradStructConvGroupedWeightStridePadCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [1L, 4L, 5L, 5L])]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradStructConvTransposeGroupedWeightCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [1L, 4L, 5L, 5L])]));
    }

    [Fact]
    public void TestAutoGradStructuralPoolingCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradStructMaxPoolOverlapCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradStructMaxPoolPadCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
        // ceil_mode / dilations: the QEE MaxPool shape inference doesn't model
        // either attribute yet, so only the ONNX/CS roundtrips execute these.
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradStructMaxPoolCeilCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1f)],
            testQuickEngineExecution: false));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradStructMaxPoolDilationCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1f)],
            testQuickEngineExecution: false));
        // Col2Im gradient path (matches the other overlapping-AvgPool tests).
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradStructAvgPoolDilationCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1f)],
            testQuickEngineExecution: false));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradStructGlobalAvgPool5DCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
    }

    [Fact]
    public void TestAutoGradStructuralNormalizationCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradStructBatchNormScaleBiasCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f), TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradStructLayerNormScaleBiasCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1.5f), TensorData(DType.Float32, [], -0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradStructLayerNormAxis1Check>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradStructGroupNormScaleBiasCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f), TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradStructInstanceNormScaleBiasCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f), TensorData(DType.Float32, [], 1f)]));
    }

    [Fact]
    public void TestAutoGradStructuralConcatSplitCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradStructConcatNegAxisCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f), TensorData(DType.Float32, [], 7f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradStructSplitNegAxisCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradStructSplitPartialUseCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [4L])]));
    }

    [Fact]
    public void TestAutoGradStructuralVariadicAndDropoutCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradStructSumBroadcastCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f), TensorData(DType.Float32, [], 5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradStructMeanBroadcastCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f), TensorData(DType.Float32, [], 5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradStructMaxBroadcastCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f), TensorData(DType.Float32, [], 5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradStructMinBroadcastCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 7f), TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradStructDropoutRatioInputCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
    }
}

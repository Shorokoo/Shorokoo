using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Runtime;
using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;

using static Shorokoo.Tests.NNLibraryFixtures;
using static Shorokoo.Tests.NNLibraryTrainingFixtures;

namespace Shorokoo.Tests;

/// <summary>
/// Inputs for the baseline NN library coverage (Shorokoo.Modules Initializers / Layers / Losses):
/// each [Fact] drives self-checking modules from NNLibraryTestModules.cs through
/// AutoTest.AdvancedTestGraph (ONNX roundtrip, CS codegen, QEE); the value correctness and its
/// closed forms live inside those modules. BatchNorm is covered by the rig-based BatchNorm classes
/// below instead (its StateUpdate links are not executable in the plain inference pipeline).
/// </summary>
internal static class NNLibraryFixtures
{
    /// <summary>[i * scale + offset for i in 0..N) as a float32 TensorData.</summary>
    internal static TensorData RangeTensor(long[] dims, float scale = 1f, float offset = 0f)
    {
        long total = 1;
        foreach (var d in dims) total *= d;
        return TensorData(DType.Float32, dims,
            Enumerable.Range(0, (int)total).Select(i => (object)(i * scale + offset)).ToArray());
    }

    /// <summary>[i * scale + offset + curv * i² for i in 0..N): a QUADRATIC ramp. A linear
    /// <see cref="RangeTensor"/> is degenerate for a frozen norm reference — mean-subtraction
    /// annihilates the per-slice offset, so every slice standardizes identically and the golden
    /// becomes invariant under an internal N/C transpose. The i² term keeps the slices distinct.</summary>
    internal static TensorData CurvedTensor(long[] dims, float scale, float offset, float curv)
    {
        long total = 1;
        foreach (var d in dims) total *= d;
        return TensorData(DType.Float32, dims,
            Enumerable.Range(0, (int)total).Select(i => (object)(i * scale + offset + curv * i * i)).ToArray());
    }
}

[Trait("Domain", "Modules")]
[Trait("Purpose", "Coverage")]
public class NNLibraryLayerAndInitializerCoverageTests
{
    [Fact]
    public void TestLinearConvAndGeneralizedConvCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<NNLinearMatchesPyTorch>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L], 0.5f, -1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNConv2dForwardGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 5L, 5L], 0.1f, -2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNConv1dForwardGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 7L], 0.25f, -1.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNConvTranspose2dForwardGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 3L, 3L], 0.3f, -2f)]));

        Assert.True(AutoTest.AdvancedTestGraph<ConvNonSquareKernelGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 7L, 9L], 0.05f, -2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<ConvPerAxisStrideDilationGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 7L, 7L], 0.05f, -1.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<ConvAsymmetricPadGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 5L, 5L], 0.1f, -2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<ConvAutoPadGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 6L, 6L], 0.07f, -1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<ConvGroupsGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 4L, 5L, 5L], 0.05f, -2.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<ConvNoBiasGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 5L, 5L], 0.1f, -2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<ConvAliasesGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 5L, 5L], 0.1f, -2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<ConvScalarOverloadGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 5L, 5L], 0.1f, -2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<ConvTransposeOutputPaddingGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 3L, 3L], 0.3f, -2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<ConvTransposeOutputShapeGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 3L, 3L], 0.3f, -2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<ConvTranspose1dGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 4L], 0.25f, -1.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<ConvTranspose3dGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 2L, 2L, 2L], 0.2f, -1f)]));

        // padding_mode: the reflect/edge/wrap Pad is non-differentiable and has no QEE values.
        Assert.True(AutoTest.AdvancedTestGraph<ConvPaddingModesGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 5L, 5L], 0.1f, -2f)],
            testCsRoundtrip: false, testQuickEngineExecution: false));
        Assert.True(AutoTest.AdvancedTestGraph<ConvCausalGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 7L], 0.25f, -1.5f)],
            testCsRoundtrip: false, testQuickEngineExecution: false));
    }

    [Fact]
    public void TestBilinearForwardClosedFormUseBiasAndBatchBroadcast()
    {
        Assert.True(AutoTest.AdvancedTestGraph<NNBilinearForwardGolden>(
            hyperparamInputs: [],
            runtimeInputs: [RangeTensor([2L, 3L], 0.5f, -1f), RangeTensor([2L, 4L], 0.3f, -0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNBilinearUseBiasGoldens>(
            hyperparamInputs: [],
            runtimeInputs: [RangeTensor([2L, 3L], 0.5f, -1f), RangeTensor([2L, 4L], 0.3f, -0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNBilinearBatchBroadcasts>(
            hyperparamInputs: [],
            runtimeInputs: [RangeTensor([2L, 2L, 3L], 0.5f, -1f), RangeTensor([2L, 2L, 4L], 0.3f, -0.5f)]));
    }

    [Fact]
    public void TestLayerGroupInstanceNormCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<NNLayerNormNormalizes>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 4L], 1.5f, -3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNGroupNormNormalizes>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 4L, 3L, 3L], 0.7f, -10f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNInstanceNorm2dNormalizes>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 4L, 4L], 0.4f, -3f)]));

        Assert.True(AutoTest.AdvancedTestGraph<NNInstanceNormRank3Normalizes>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 5L], 0.4f, -3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNInstanceNormRank4Normalizes>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 4L, 4L], 0.4f, -3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNInstanceNormRank5Normalizes>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 2L, 2L, 2L], 0.4f, -3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNGroupNormRank3Normalizes>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 4L, 5L], 0.5f, -8f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNGroupNormRank4Normalizes>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 4L, 3L, 3L], 0.7f, -10f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNGroupNormRank5Normalizes>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 4L, 2L, 2L, 2L], 0.5f, -8f)]));

        Assert.True(AutoTest.AdvancedTestGraph<NNInstanceNormAffineFalseGolden>(
            hyperparamInputs: [], runtimeInputs: [CurvedTensor([2L, 3L, 4L, 4L], 0.4f, -3f, 0.05f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNGroupNormAffineFalseGolden>(
            hyperparamInputs: [], runtimeInputs: [CurvedTensor([2L, 4L, 3L, 3L], 0.7f, -10f, 0.05f)]));

        Assert.True(AutoTest.AdvancedTestGraph<NNGroupNormG1MatchesLayerNorm>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 4L, 4L], 0.4f, -3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNGroupNormGCMatchesInstanceNormRank4>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 4L, 4L], 0.4f, -3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNGroupNormGCMatchesInstanceNormRank3>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 5L], 0.4f, -3f)]));

        Assert.True(AutoTest.AdvancedTestGraph<NNInstanceNorm1dAliasEquiv>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 5L], 0.4f, -3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNInstanceNorm2dAliasEquiv>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 4L, 4L], 0.4f, -3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNInstanceNorm3dAliasEquiv>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 2L, 2L, 2L], 0.4f, -3f)]));
    }

    [Fact]
    public void TestPoolingHelpersAndNNStaticWrapperOpsCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<NNPoolingHelpersChecks>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 4L, 4L], 0.5f, -4f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNPool1d3dClosedForm>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 4L, 4L], 0.5f, -4f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNLpPoolClosedFormAndGlobal>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 4L, 4L], 0.5f, -4f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNFullWindowEqualsGlobal>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 4L, 4L], 0.5f, -4f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNPoolScalarPerAxisAliasEquiv>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 4L, 4L], 0.5f, -4f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNPoolPerAxisGeometryMatchesCoreOp>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 6L, 5L], 0.1f, -3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNMaxUnpoolRoundTrip>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 4L, 4L], 0.5f, -4f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNAvgPoolCountIncludePadToggle>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 4L, 4L], 0.5f, -4f)]));

        Assert.True(AutoTest.AdvancedTestGraph<NNStaticWrapperWindowEyeDetCheck>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 2L], 1f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNStaticWrapperPoolMathCheck>(
            hyperparamInputs: [],
            runtimeInputs: [
                RangeTensor([1L, 2L, 4L, 4L], 0.5f, -4f),
                TensorData(DType.Int64, [3L], 7L, 8L, 9L),
                TensorData(DType.Int64, [3L], 2L, 3L, 4L)]));
    }

    /// <summary>Constant (the #440 op-name-collision regression guard), Orthogonal Gram ≈ I,
    /// configurable UniformRange / NormalDist sample statistics, and the configurable-gain
    /// Xavier/Kaiming sample std (which excludes the §4.1 √6-double-bake value).</summary>
    [Fact]
    public void TestInitializerCoverage()
    {
        var dummy = RangeTensor([2L, 3L], 0.5f, -1f);

        Assert.True(AutoTest.AdvancedTestGraph<NNConstantInitFillsValue>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNConstantInitRank1Negative>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNConstantInitMatchesZerosOnes>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNOrthogonalSquareGramIsIdentity>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNOrthogonalTallGramIsIdentity>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNOrthogonalWideGramIsIdentity>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNUniformRangeInRange>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNNormalDistMoments>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNXavierKaimingGainStd>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
    }
}

[Trait("Domain", "Modules")]
[Trait("Purpose", "Coverage")]
public class NNLibraryDropoutAndEmbeddingCoverageTests
{
    [Fact]
    public void TestDropoutEmbeddingBagKnobsAndActivationCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<NNDropoutChecks>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 8L], 0.5f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNDropoutRatioOneAllZeros>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 8L], 0.5f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNEmbeddingForwardGolden>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Int64, [3L], 0L, 1L, 0L)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNLeakyReLUAndELUClosedForm>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [7L], -3f, -1f, -0.5f, 0f, 0.5f, 1f, 3f)]));

        Assert.True(AutoTest.AdvancedTestGraph<NNEmbeddingPaddingIdxZeros>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Int64, [5L], 0L, 1L, 2L, 2L, 3L)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNEmbeddingMaxNormClampsL2>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Int64, [2L], 0L, 3L)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNEmbeddingNormTypeL1VsL2>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Int64, [1L], 0L)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNEmbeddingInitChoice>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Int64, [3L], 0L, 2L, 4L)]));

        Assert.True(AutoTest.AdvancedTestGraph<NNEmbeddingBagSumGolden>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Int64, [2L, 3L], 0L, 1L, 2L, 1L, 3L, 0L)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNEmbeddingBagMeanGolden>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Int64, [2L, 3L], 0L, 1L, 2L, 1L, 3L, 0L)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNEmbeddingBagMaxGolden>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Int64, [2L, 3L], 0L, 1L, 2L, 1L, 3L, 0L)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNEmbeddingBagPaddingIdxSumExact>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Int64, [2L, 3L], 0L, 2L, 1L, 2L, 3L, 0L)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNEmbeddingBagInitChoice>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Int64, [2L, 3L], 0L, 1L, 2L, 1L, 3L, 0L)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNEmbeddingBagShapeCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Int64, [2L, 3L], 0L, 1L, 2L, 1L, 3L, 0L)]));
    }

    [Fact]
    public void TestSpatialAlphaAndFeatureAlphaDropoutCoverage()
    {
        // SpatialDropout train mode: 0-or-scale·x with a channel-uniform mask, ranks 3/4/5.
        Assert.True(AutoTest.AdvancedTestGraph<NNSpatialDropoutChannelWise>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 2L, 2L], 0.5f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNSpatialDropoutSurvivorScale75>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 2L, 2L], 0.5f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNSpatialDropoutChannelWiseRank3>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 4L], 0.5f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNSpatialDropoutChannelWiseRank5>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 2L, 2L, 2L], 0.5f, 1f)]));

        // SpatialDropout eval identity (ratios 0.5/0.9), rank 2/3/4/5, and the Dropout1d/2d/3d aliases.
        Assert.True(AutoTest.AdvancedTestGraph<NNSpatialDropoutEvalIdentity>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 2L, 2L], 0.5f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNSpatialDropoutEvalIdentityAnyRank>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 4L], 0.5f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNSpatialDropoutEvalIdentityAnyRank>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 4L, 4L], 0.5f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNSpatialDropoutEvalIdentityAnyRank>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 2L, 2L, 2L], 0.5f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNSpatialDropoutRank2Degenerate>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L], 0.5f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNDropout1dAliasEquiv>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 4L], 0.5f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNDropout2dAliasEquiv>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 4L, 4L], 0.5f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNDropout3dAliasEquiv>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 2L, 2L, 2L], 0.5f, 1f)]));

        // AlphaDropout: the two-value per-element invariant at ratio 0.5 and 0.25 (a,b track the
        // ratio), moment preservation over [64,64], and eval identity.
        Assert.True(AutoTest.AdvancedTestGraph<NNAlphaDropoutPerElementInvariant>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 8L], 0.5f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNAlphaDropoutPerElementInvariantRatio25>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 8L], 0.5f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNAlphaDropoutMomentPreservation>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([64L, 64L], 0.001f, -2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNAlphaDropoutEvalIdentity>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 8L], 0.5f, 1f)]));

        // FeatureAlphaDropout: channel uniformity at rank 3/4/5, plus eval identity rank 2/3/4/5.
        Assert.True(AutoTest.AdvancedTestGraph<NNFeatureAlphaDropoutChannelUniform>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 2L, 2L], 0.5f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNFeatureAlphaDropoutChannelUniformRank3>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 4L], 0.5f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNFeatureAlphaDropoutChannelUniformRank5>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 2L, 2L, 2L], 0.5f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNFeatureAlphaDropoutEvalIdentityAnyRank>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 4L], 0.5f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNFeatureAlphaDropoutEvalIdentityAnyRank>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 4L, 4L], 0.5f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNFeatureAlphaDropoutEvalIdentityAnyRank>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 2L, 2L, 2L], 0.5f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNFeatureAlphaDropoutEvalIdentityAnyRank>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L], 0.5f, 1f)]));
    }
}

[Trait("Domain", "Modules")]
[Trait("Purpose", "Coverage")]
public class NNLibraryLossCoverageTests
{
    [Fact]
    public void TestLossClosedFormsEdgeCasesAndKnobsCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<NNLossEdgeCaseChecks>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [2L], 0.5f, 2f)]));

        var dummy = TensorData(DType.Float32, [2L], 1f, 3f);
        Assert.True(AutoTest.AdvancedTestGraph<NNLossClosedFormChecks>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNLogCoshLossChecks>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNPoissonNLLLossChecks>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNHingeLossChecks>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNSquaredHingeLossChecks>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNBinaryFocalLossChecks>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNCrossEntropyReductionWeightIgnoreChecks>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNCrossEntropyLabelSmoothingChecks>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNCrossEntropyLabelSmoothWeightIgnoreChecks>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNNLLLossWeightIgnoreChecks>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNBCEWithLogitsPosWeightChecks>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNSmoothL1BetaChecks>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNRegressionReductionChecks>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
    }

    [Fact]
    public void TestTripletMarginAndCosineEmbeddingLossCoverage()
    {
        var dummy = TensorData(DType.Float32, [2L], 1f, 3f);
        Assert.True(AutoTest.AdvancedTestGraph<NNTripletMarginClosedFormChecks>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNTripletMarginSwapMarginPChecks>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNTripletMarginReductionChecks>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNTripletMarginWithDistanceChecks>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNCosineEmbeddingClosedFormChecks>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNCosineEmbeddingMarginGatingChecks>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNCosineEmbeddingWhereSplitChecks>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNCosineEmbeddingReductionChecks>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNCosineSimilarityHelperChecks>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
    }

    [Fact]
    public void TestEveryLossFamilyReducedNoneThrows()
    {
        var pred = Tensor([1L, 2L], 0f, 0f);
        var tgt = Vector(0L);
        var fpred = Tensor([2L], 1f, 3f);
        var ftgt = Tensor([2L], 0f, 1f);
        var a = Tensor([1L, 2L], 0f, 0f);
        var p = Tensor([1L, 2L], 1f, 0f);
        var n = Tensor([1L, 2L], 0f, 2f);
        var x1 = Tensor([1L, 2L], 1f, 0f);
        var x2 = Tensor([1L, 2L], 0f, 1f);
        var y = Tensor([1L], 1f);

        static void Throws(Action call) => Assert.Throws<ArgumentException>(call);

        Throws(() => CrossEntropyLoss.Reduced(pred, tgt, reduction: LossReduction.None));
        Throws(() => NLLLoss.Reduced(pred, tgt, reduction: LossReduction.None));
        Throws(() => BCEWithLogitsLoss.Reduced(fpred, ftgt, reduction: LossReduction.None));
        Throws(() => SmoothL1Loss.Reduced(1f, fpred, ftgt, reduction: LossReduction.None));
        Throws(() => HuberLoss.Reduced(Scalar(1f), fpred, ftgt, reduction: LossReduction.None));
        Throws(() => L1Loss.Reduced(fpred, ftgt, reduction: LossReduction.None));
        Throws(() => L2Loss.Reduced(fpred, ftgt, reduction: LossReduction.None));
        Throws(() => TripletMarginLoss.Reduced(Scalar(1f), Scalar(2f), Scalar(1e-6f), Scalar(false),
            a, p, n, reduction: LossReduction.None));
        Throws(() => TripletMarginWithDistance.Reduced(
            (u, v) => { var d = u - v; return (d * d).Reduce(ReduceKind.Sum, [Scalar(-1L)], keepDims: false); },
            1f, false, a, p, n, reduction: LossReduction.None));
        Throws(() => CosineEmbeddingLoss.Reduced(Scalar(0f), Scalar(1e-8f), x1, x2, y,
            reduction: LossReduction.None));
    }
}

[Trait("Domain", "Modules")]
[Trait("Purpose", "Coverage")]
public class NNLibraryRecurrentLayerCoverageTests
{
    /// <summary>Rnn / Lstm / Gru forward goldens: baseline + batchFirst, the single-step gate
    /// anchors, bias on/off, numLayers:2, Reverse + Bidirectional, the GRU linearBeforeReset
    /// both-forms crux, and the hN == y[-1] state contract.</summary>
    [Fact]
    public void TestRecurrentLayerForwardValueCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<RnnBaselineForwardTanhGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<RnnBatchFirstGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 4L, 3L], 0.1f, -1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<RnnSingleStepAnchorTanh>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 3L], 0.2f, -0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<RnnReluForwardGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<RnnNoBiasGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<RnnNumLayersStackGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<RnnReverseGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<RnnBidirectionalGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<RnnStateContractForwardSingleLayer>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));

        Assert.True(AutoTest.AdvancedTestGraph<LstmBaselineForwardGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<LstmBatchFirstGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 4L, 3L], 0.1f, -1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<LstmSingleStepGateAnchor>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 3L], 0.2f, -0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<LstmNoBiasGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<LstmNumLayersStackGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<LstmReverseGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<LstmBidirectionalGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<LstmStateContractForwardSingleLayer>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));

        Assert.True(AutoTest.AdvancedTestGraph<GruBaselineForwardGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<GruBatchFirstGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 4L, 3L], 0.1f, -1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<GruLinearBeforeResetBothForms>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<GruSingleStepGateAnchor>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 3L], 0.2f, -0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<GruNoBiasGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<GruNumLayersStackGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<GruReverseGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<GruBidirectionalGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<GruStateContractForwardSingleLayer>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));
    }
}

[Trait("Domain", "Modules")]
[Trait("Purpose", "Coverage")]
public class NNLibraryRecurrentCellCoverageTests
{
    /// <summary>RNNCell / LSTMCell / GRUCell single-step goldens with NONZERO previous state (so R
    /// is exercised): the tanh and relu closed forms, the LSTM i/o/f/c and GRU z/r/h gate packing,
    /// both linearBeforeReset forms, bias:false, the [N,H] shape contract, and the two-step
    /// hand-unrolled STATE THREADING golden.</summary>
    [Fact]
    public void TestRecurrentCellForwardValueCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<RnnCellClosedFormTanh>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 3L], 0.2f, -0.3f)]));
        // Wider positive ramp than the tanh anchor: the [-0.3, 0.1] ramp drives every relu
        // pre-activation negative under the seed-0 draw, freezing a vacuous all-zero golden.
        Assert.True(AutoTest.AdvancedTestGraph<RnnCellClosedFormRelu>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 3L], 0.5f, 0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<RnnCellSingleStepGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L], 0.1f, -0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<RnnCellNoBiasGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L], 0.1f, -0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<RnnCellStateThreading>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 2L, 3L], 0.1f, -0.6f)]));

        Assert.True(AutoTest.AdvancedTestGraph<LstmCellClosedFormGateAnchor>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 3L], 0.2f, -0.3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<LstmCellSingleStepGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L], 0.1f, -0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<LstmCellNoBiasGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L], 0.1f, -0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<LstmCellStateThreading>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 2L, 3L], 0.1f, -0.6f)]));

        Assert.True(AutoTest.AdvancedTestGraph<GruCellClosedFormLbrTrue>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 3L], 0.2f, -0.3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<GruCellClosedFormLbrFalse>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 3L], 0.2f, -0.3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<GruCellSingleStepGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L], 0.1f, -0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<GruCellNoBiasGolden>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L], 0.1f, -0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<GruCellStateThreading>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 2L, 3L], 0.1f, -0.6f)]));
    }
}

/// <summary>
/// Shared rig plumbing for the training-side coverage of the baseline NN library: batch/struct
/// construction, the fixed conv batch, the analytic multi-step TrainStep driver and its
/// struct comparison.
/// </summary>
internal static class NNLibraryTrainingFixtures
{
    internal static TensorDataStruct MakeStruct(string structName, params (string name, TensorData data)[] fields)
    {
        TensorStructFieldDef[] defs =
            [.. fields.Select(f => new TensorStructFieldDef(
                f.name, DataStructure.Tensor, f.data.Shape.Dims.Length, f.data.DType))];
        return new TensorDataStruct(new TensorStructDef(defs, structName),
            fields.ToDictionary(f => f.name, f => (IData)f.data));
    }

    internal static TensorDataStruct MakeBatch(string fieldName, string structName, TensorData data)
        => MakeStruct(structName, (fieldName, data));

    internal static float[] Floats(IData data) => ((TensorData<float32>)data).AccessMemory().ToArray();

    internal static bool AnyParamMoved(TrainingRig rig, TrainingCheckpoint before, TrainingCheckpoint after)
        => rig.TrainableParamStructDef.Fields.Any(f =>
            Floats(before.TrainableParams.Fields[f.Name])
                .Zip(Floats(after.TrainableParams.Fields[f.Name]))
                .Any(p => MathF.Abs(p.First - p.Second) > 1e-7f));

    // 4 samples of [1,4,4]: class 0 lights the left two columns, class 1 the right two,
    // at two different intensities.
    internal static (TensorData input, TensorData target) MakeTinyConvBatch()
    {
        var vals = new float[4 * 16];
        for (int s = 0; s < 4; s++)
        {
            float intensity = s < 2 ? 1f : 0.6f;
            bool rightHalf = (s % 2) == 1;
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++)
                    vals[s * 16 + r * 4 + c] = (rightHalf ? c >= 2 : c < 2) ? intensity : 0f;
        }
        long[] classes = [0L, 1L, 0L, 1L];
        return (TensorData([4L, 1L, 4L, 4L], vals), TensorData([4L], classes));
    }

    /// <summary>Runs <paramref name="steps"/> TrainSteps of model + L2Loss + optimizer on one
    /// fixed batch and returns the final checkpoint, so each analytic check is a one-liner.</summary>
    internal static TrainingCheckpoint TrainAnalytic(
        ComputationGraph modelGraph, ComputationGraph optimizerGraph, Hyperparameter[] hypers,
        long[] inShape, float[] input, long[] outShape, float[] target, int steps)
    {
        var rig = TrainingRig.FromScratch(modelGraph, L2Loss.ComputationGraph, optimizerGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData(inShape, input))],
            hypers);
        var ckpt = rig.CreateInitialCheckpoint();
        for (int i = 0; i < steps; i++)
            ckpt = rig.TrainStep(ckpt,
                MakeBatch("input", "AnalyticIn", TensorData(inShape, input)),
                MakeBatch("targets", "AnalyticTg", TensorData(outShape, target)));
        return ckpt;
    }

    /// <summary>Asserts the struct's fields, flattened in definition order, equal
    /// <paramref name="expected"/> within <paramref name="tol"/>.</summary>
    internal static void AssertStructIs(TensorDataStruct s, float[] expected, float tol)
    {
        var flat = s.Definition.Fields.SelectMany(f => Floats(s.Fields[f.Name])).ToArray();
        Assert.Equal(expected.Length, flat.Length);
        for (int i = 0; i < flat.Length; i++)
            Assert.True(MathF.Abs(flat[i] - expected[i]) <= tol);
    }

    /// <summary>[i * scale + offset for i in 0..total).</summary>
    internal static float[] Ramp(long total, float scale = 1f, float offset = 0f)
        => Enumerable.Range(0, (int)total).Select(i => i * scale + offset).ToArray();
}

/// <summary>
/// Optimizer rig construction, per-optimizer step numerics and end-to-end convergence runs.
/// </summary>
[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class NNLibraryOptimizerTrainingCoverageTests
{
    private static void CoverFromScratch(
        ComputationGraph modelGraph,
        ComputationGraph lossGraph,
        ComputationGraph optimizerGraph,
        long[] inputShape,
        params Hyperparameter[] hyperparams)
    {
        long totalElements = 1;
        foreach (var d in inputShape) totalElements *= d;
        var sampleInput = new TensorDataModelParam(
            "input", ModelParamType.InputParam,
            TensorData(inputShape, new float[totalElements]));

        var rig = TrainingRig.FromScratch(modelGraph, lossGraph, optimizerGraph,
            [sampleInput], hyperparams);

        var checkpoint = rig.CreateInitialCheckpoint();
        Assert.NotEmpty(rig.TrainableParamStructDef.Fields);
        Assert.NotNull(checkpoint.TrainableParams);
    }

    // --- Wide-regression convergence fixture -------------------------------------------------
    // 32 samples, 2 features in, 400 outputs out (matches NNWideRegressionModel). Deterministic
    // and perfectly realizable: Y = X·Wtᵀ + bt, so the global-min L2 loss is exactly 0. The many
    // outputs concentrate the random-init starting loss into the narrow band asserted below.
    private const int WideN = 32, WideF = 2, WideO = 400;

    private static (TensorData input, TensorData target) MakeWideRegressionData()
    {
        var x = new float[WideN * WideF];
        for (int n = 0; n < WideN; n++)
            for (int f = 0; f < WideF; f++)
                x[n * WideF + f] = (float)Math.Sin(1.0 + n * 0.7 + f * 1.3);

        var y = new float[WideN * WideO];
        for (int n = 0; n < WideN; n++)
            for (int o = 0; o < WideO; o++)
            {
                double acc = 0.25 * Math.Sin(0.5 + o * 0.11);                       // bias bt[o]
                for (int f = 0; f < WideF; f++)
                    acc += Math.Sin(1.0 + n * 0.7 + f * 1.3) * (0.5 * Math.Sin(2.0 + o * 0.3 + f * 0.9));
                y[n * WideO + o] = (float)acc;
            }

        return (TensorData([(long)WideN, WideF], x), TensorData([(long)WideN, WideO], y));
    }

    // The starting loss of NNWideRegressionModel under its seeded KaimingUniform init is a
    // platform-stable quantity: across 150k simulated draws it is mean 1.184, σ 0.046, so the
    // central 1-in-100-billion (6.81σ) range is [0.871, 1.497]. Asserted a hair wider to absorb
    // float-vs-double and slight right-skew.
    private static void AssertWideStartLossInBand(float startLoss) =>
        Assert.True(startLoss is >= 0.85f and <= 1.52f);

    /// <summary>Builds a rig for <paramref name="optimizerGraph"/>, asserts the trainable-param
    /// and optimizer-state structs are non-empty, then runs ONE TrainStep and asserts the loss is
    /// finite and the param actually moved — exercising the optimizer's state threading end to end.</summary>
    private static void CoverTrainStepMovesParam(
        ComputationGraph optimizerGraph, params Hyperparameter[] hyperparams)
    {
        float[] input = [1f, 2f, 3f, 4f];
        float[] target = [0f, 0f, 0f, 0f];
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, optimizerGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData([4L], input))],
            hyperparams);

        var initial = rig.CreateInitialCheckpoint();
        Assert.NotEmpty(rig.TrainableParamStructDef.Fields);

        string wName = rig.TrainableParamStructDef.Fields[0].Name;
        float w0 = Floats(initial.TrainableParams.Fields[wName])[0];

        var step = rig.TrainStep(initial,
            MakeBatch("input", "ModelInput", TensorData([4L], input)),
            MakeBatch("targets", "Target", TensorData([4L], target)));
        float w1 = Floats(step.TrainableParams.Fields[wName])[0];

        Assert.True(float.IsFinite(step.Loss!.Value));
        Assert.NotEmpty(step.OptimizerState.Fields);
        Assert.True(MathF.Abs(w1 - w0) > 1e-7f);
    }

    /// <summary>Like <see cref="CoverTrainStepMovesParam"/> but for a rank-≥2 trainable param —
    /// the Adafactor/LAMB rank-agnosticism gate (reduce-all RMS / ‖·‖ scalars over a non-scalar param).</summary>
    private static void CoverTrainStepMovesNonScalarParam(
        ComputationGraph modelGraph, ComputationGraph optimizerGraph,
        long[] inShape, float[] input, float[] target,
        params Hyperparameter[] hyperparams)
    {
        var rig = TrainingRig.FromScratch(
            modelGraph, L2Loss.ComputationGraph, optimizerGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData(inShape, input))],
            hyperparams);

        var initial = rig.CreateInitialCheckpoint();
        Assert.NotEmpty(rig.TrainableParamStructDef.Fields);

        string wName = rig.TrainableParamStructDef.Fields[0].Name;
        float[] w0 = Floats(initial.TrainableParams.Fields[wName]);

        var step = rig.TrainStep(initial,
            MakeBatch("input", "ModelInput", TensorData(inShape, input)),
            MakeBatch("targets", "Target", TensorData(inShape, target)));
        float[] w1 = Floats(step.TrainableParams.Fields[wName]);

        Assert.True(float.IsFinite(step.Loss!.Value));
        Assert.NotEmpty(step.OptimizerState.Fields);
        Assert.True(w0.Length >= 2);
        Assert.True(w0.Zip(w1).Any(p => MathF.Abs(p.First - p.Second) > 1e-7f));
    }

    /// <summary>Rig construction + one TrainStep for every optimizer: the per-param state structs
    /// are non-empty, the loss is finite, the param moves, and the state threads. Gates NAdam's two
    /// scalar states (step + muProduct), RAdam's runtime <c>Where</c> through the autodiff/scheduler,
    /// and the Adafactor / LAMB reduce-all scalars over a rank-2 [2,2] param.</summary>
    [Fact]
    public void TestOptimizerRigCoverage()
    {
        CoverFromScratch(ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            AdamOptimizer.ComputationGraph, [4L], 0.001f, 0.9f, 0.999f, 1e-8f);
        CoverFromScratch(ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            RMSpropOptimizer.ComputationGraph, [4L], 0.01f, 0.99f, 1e-8f, 0.0f);
        CoverFromScratch(ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            AdagradOptimizer.ComputationGraph, [4L], 0.01f, 1e-10f);
        CoverFromScratch(ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            AdamaxOptimizer.ComputationGraph, [4L], 0.002f, 0.9f, 0.999f, 1e-8f);
        CoverFromScratch(ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            NAdamOptimizer.ComputationGraph, [4L], 0.002f, 0.9f, 0.999f, 1e-8f, 0.004f);
        CoverFromScratch(ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            RAdamOptimizer.ComputationGraph, [4L], 0.001f, 0.9f, 0.999f, 1e-8f);
        CoverFromScratch(ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            AdadeltaOptimizer.ComputationGraph, [4L], 1.0f, 0.9f, 1e-6f);
        CoverFromScratch(ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            LionOptimizer.ComputationGraph, [4L], 0.0001f, 0.9f, 0.99f, 0.0f);
        CoverFromScratch(ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            AdafactorOptimizer.ComputationGraph, [4L], 0.01f, -0.8f, 1e-30f, 1e-3f, 1.0f, 0.0f);
        CoverFromScratch(ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            LambOptimizer.ComputationGraph, [4L], 0.001f, 0.9f, 0.999f, 1e-6f, 0.01f);

        CoverTrainStepMovesParam(AdamaxOptimizer.ComputationGraph, 0.002f, 0.9f, 0.999f, 1e-8f);
        CoverTrainStepMovesParam(NAdamOptimizer.ComputationGraph, 0.002f, 0.9f, 0.999f, 1e-8f, 0.004f);
        CoverTrainStepMovesParam(RAdamOptimizer.ComputationGraph, 0.001f, 0.9f, 0.999f, 1e-8f);
        CoverTrainStepMovesParam(AdadeltaOptimizer.ComputationGraph, 1.0f, 0.9f, 1e-6f);
        CoverTrainStepMovesParam(LionOptimizer.ComputationGraph, 0.0001f, 0.9f, 0.99f, 0.0f);
        CoverTrainStepMovesParam(AdafactorOptimizer.ComputationGraph, 0.01f, -0.8f, 1e-30f, 1e-3f, 1.0f, 0.0f);
        CoverTrainStepMovesParam(LambOptimizer.ComputationGraph, 0.001f, 0.9f, 0.999f, 1e-6f, 0.01f);

        CoverTrainStepMovesNonScalarParam(AnalyticMatMulModel.ComputationGraph, AdafactorOptimizer.ComputationGraph,
            [2L, 2L], [1f, 2f, 3f, 4f], [0f, 0f, 0f, 0f], 0.01f, -0.8f, 1e-30f, 1e-3f, 1.0f, 0.0f);
        CoverTrainStepMovesNonScalarParam(AnalyticMatMulModel.ComputationGraph, LambOptimizer.ComputationGraph,
            [2L, 2L], [1f, 2f, 3f, 4f], [0f, 0f, 0f, 0f], 0.01f, 0.9f, 0.999f, 1e-6f, 0.01f);
    }

    /// <summary>End-to-end convergence. Adam and LAMB on the wide, perfectly realizable
    /// <c>[32,2] → [32,400]</c> regression fixture must start inside the platform-invariant
    /// 1-in-100-billion random-init band and fall below the ABSOLUTE 1e-2 target in 150 steps;
    /// the tiny conv net + CrossEntropy + SGDMomentum loss must decrease over 15 steps.</summary>
    [Fact]
    public void TestAdamLambAndSgdMomentumConverge()
    {
        var (wideInput, wideTarget) = MakeWideRegressionData();

        float[] WideLosses(ComputationGraph optimizerGraph, Hyperparameter[] hypers)
        {
            var rig = TrainingRig.FromScratch(
                NNWideRegressionModel.ComputationGraph, L2Loss.ComputationGraph, optimizerGraph,
                [new TensorDataModelParam("input", ModelParamType.InputParam, wideInput)], hypers);
            var inputBatch = MakeBatch("input", "ModelInput", wideInput);
            var targetBatch = MakeBatch("targets", "Target", wideTarget);
            var ckpt = rig.CreateInitialCheckpoint();
            var losses = new float[150];
            for (int i = 0; i < losses.Length; i++)
            {
                ckpt = rig.TrainStep(ckpt, inputBatch, targetBatch);
                losses[i] = ckpt.Loss!.Value;
            }
            return losses;
        }

        float[][] runs =
        [
            WideLosses(AdamOptimizer.ComputationGraph, [0.05f, 0.9f, 0.999f, 1e-8f]),
            WideLosses(LambOptimizer.ComputationGraph, [0.05f, 0.9f, 0.999f, 1e-6f, 0.01f]),
        ];
        foreach (var losses in runs)
        {
            Assert.All(losses, l => Assert.True(float.IsFinite(l)));
            AssertWideStartLossInBand(losses[0]);
            Assert.True(losses[^1] < 1e-2f);
        }

        var (convInput, convTarget) = MakeTinyConvBatch();
        var convRig = TrainingRig.FromScratch(
            NNTinyConvClassifier.ComputationGraph, CrossEntropyLoss.ComputationGraph,
            SGDMomentumOptimizer.ComputationGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, convInput)],
            0.2f, 0.9f);
        var convInBatch = MakeBatch("input", "ModelInput", convInput);
        var convTgBatch = MakeBatch("targets", "Target", convTarget);
        var convCkpt = convRig.CreateInitialCheckpoint();
        var convLosses = new float[15];
        for (int i = 0; i < convLosses.Length; i++)
        {
            convCkpt = convRig.TrainStep(convCkpt, convInBatch, convTgBatch);
            convLosses[i] = convCkpt.Loss!.Value;
        }
        Assert.All(convLosses, l => Assert.True(float.IsFinite(l)));
        Assert.True(convLosses[^1] < convLosses[0]);
    }

    // -----------------------------------------------------------------------
    //  Analytic value checks. Every expectation is hand-computed from the fixtures' constant
    //  initializers; gradients are inferred through real TrainStep execution (SGD:
    //  w' = w − lr·grad, L2Loss = mean over ALL output elements so dL/dyᵢ = 2(yᵢ−tᵢ)/N).
    // -----------------------------------------------------------------------

    /// <summary>Autodiff gradient values (reverse-broadcast sum-reduction, Relu masking, MatMul's
    /// xᵀ·gUp, gradient ACCUMULATION when a param is consumed twice, routing through
    /// Reshape→Transpose→Reshape, and Slice) plus the core optimizer step values on y = w·x, w₀=1,
    /// x=[1], t=[0] ⇒ grad = 2w: Adam's bias-correction timestep, SGD-momentum's velocity carry,
    /// AdamW's decoupled-decay-then-uncorrected step, and Adam's first step being ≈ lr regardless of
    /// the gradient magnitude (uncorrected it would be ≈ 3.16·lr).</summary>
    [Fact]
    public void TestAutodiffGradientAndCoreOptimizerValuesAnalytic()
    {
        // w=0.5, x=[1,2,3,4], t=0: dL/dyᵢ=yᵢ/2 → grad_w = Σ(yᵢ/2)·xᵢ = 7.5 → w' = 0.5 − 0.1·7.5
        AssertStructIs(TrainAnalytic(AnalyticBroadcastMulModel.ComputationGraph, SGDOptimizer.ComputationGraph, [0.1f],
            [4L], [1f, 2f, 3f, 4f], [4L], [0f, 0f, 0f, 0f], 1).TrainableParams, [-0.25f], 1e-5f);
        // b=0.5: y=[1.5,2.5,3.5,4.5] → grad_b = Σ yᵢ/2 = 6 → b' = 0.5 − 0.6
        AssertStructIs(TrainAnalytic(AnalyticBroadcastAddModel.ComputationGraph, SGDOptimizer.ComputationGraph, [0.1f],
            [4L], [1f, 2f, 3f, 4f], [4L], [0f, 0f, 0f, 0f], 1).TrainableParams, [-0.1f], 1e-5f);
        // w=[1,2,3,4], x=[1,−1,1,−1]: pre=[1,−2,3,−4], mask=[1,0,1,0] → grad=[0.5,0,1.5,0], lr=1
        AssertStructIs(TrainAnalytic(AnalyticReluModel.ComputationGraph, SGDOptimizer.ComputationGraph, [1f],
            [4L], [1f, -1f, 1f, -1f], [4L], [0f, 0f, 0f, 0f], 1).TrainableParams, [0.5f, 2f, 1.5f, 4f], 1e-5f);
        // x=[[1,2],[3,4]], W=[[1,2],[3,4]]: y=[[7,10],[15,22]], grad_W = xᵀ·(y/2) = [[26,38],[37,54]], lr=0.01
        AssertStructIs(TrainAnalytic(AnalyticMatMulModel.ComputationGraph, SGDOptimizer.ComputationGraph, [0.01f],
            [2L, 2L], [1f, 2f, 3f, 4f], [2L, 2L], [0f, 0f, 0f, 0f], 1).TrainableParams, [0.74f, 1.62f, 2.63f, 3.46f], 1e-5f);
        // w=0.5 used TWICE (w·x + w): grad_w = Σ(yᵢ/2)·(xᵢ+1) = 13.5 → w' = 0.5 − 1.35
        // (mul-path-only would give −0.5; add-path-only +0.15 — accumulation is pinned exactly)
        AssertStructIs(TrainAnalytic(AnalyticDoubleUseModel.ComputationGraph, SGDOptimizer.ComputationGraph, [0.1f],
            [4L], [1f, 2f, 3f, 4f], [4L], [0f, 0f, 0f, 0f], 1).TrainableParams, [-0.85f], 1e-5f);
        // w=[1,2,3,4] permuted to [w0,w2,w1,w3], x=[1,2,4,8]: dL/dwp=[0.5,6,16,128] routes back
        // through the inverse permutation → grad_w=[0.5,16,6,128], lr=0.01
        AssertStructIs(TrainAnalytic(AnalyticPermuteModel.ComputationGraph, SGDOptimizer.ComputationGraph, [0.01f],
            [4L], [1f, 2f, 4f, 8f], [4L], [0f, 0f, 0f, 0f], 1).TrainableParams, [0.995f, 1.84f, 2.94f, 2.72f], 1e-5f);
        // y = w[0:2]·x, x=[2,3]: grad = [y₀·x₀, y₁·x₁, 0, 0] = [4,18,0,0] → w'=[0.6,0.2,3,4], lr=0.1
        AssertStructIs(TrainAnalytic(AnalyticSliceParamModel.ComputationGraph, SGDOptimizer.ComputationGraph, [0.1f],
            [2L], [2f, 3f], [2L], [0f, 0f], 1).TrainableParams, [0.6f, 0.2f, 3f, 4f], 1e-5f);

        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, AdamOptimizer.ComputationGraph,
            [0.1f, 0.9f, 0.999f, 1e-8f], [1L], [1f], [1L], [0f], 2).TrainableParams, [0.8004123f], 2e-4f);
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, SGDMomentumOptimizer.ComputationGraph,
            [0.1f, 0.9f], [1L], [1f], [1L], [0f], 2).TrainableParams, [0.46f], 1e-5f);
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, AdamWOptimizer.ComputationGraph,
            [0.1f, 0.9f, 0.999f, 1e-8f, 0.1f], [1L], [1f], [1L], [0f], 1).TrainableParams, [0.6737722f], 1e-4f);

        // Adam bias correction: grad is large (15) yet the first step moves the weight by ≈ lr.
        const float lr = 0.001f;
        float[] biasIn = [1f, 2f, 3f, 4f];
        var biasRig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, AdamOptimizer.ComputationGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData([4L], biasIn))],
            lr, 0.9f, 0.999f, 1e-8f);
        var biasInitial = biasRig.CreateInitialCheckpoint();
        string biasW = biasRig.TrainableParamStructDef.Fields[0].Name;
        float bw0 = Floats(biasInitial.TrainableParams.Fields[biasW])[0];
        var biasStep = biasRig.TrainStep(biasInitial,
            MakeBatch("input", "ModelInput", TensorData([4L], biasIn)),
            MakeBatch("targets", "Target", TensorData([4L], new float[4])));
        Assert.True(MathF.Abs((bw0 - Floats(biasStep.TrainableParams.Fields[biasW])[0]) - lr) < 5e-5f);
        Assert.NotEmpty(biasStep.OptimizerState.Fields);
    }

    /// <summary>Per-step values for Adamax / NAdam / RAdam / Adadelta / Lion / Adafactor / LAMB on
    /// y = w·x, w₀=1, x=[1], t=[0] ⇒ grad = 2w, all RE-DERIVED in double precision (adadelta's
    /// design.md w₁ is 2× too large). Sharp discriminators: RAdam's step-1 UN-ADAPTED branch
    /// (ρ_t=1 ≤ 5) lands at 0.8, Lion's 4-step w₄ = 0.0 fails any β1↔β2 swap, and LAMB's 0.81/0.729
    /// fail a dropped trust ratio (plain Adam would give 0.8004123 at step 2). The LAMB zero-guard
    /// (target == pred ⇒ ‖u‖ = 0) must leave w₁ = 1.0 exactly, not NaN.</summary>
    [Fact]
    public void TestExtendedOptimizerStepValuesAnalytic()
    {
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, AdamaxOptimizer.ComputationGraph,
            [0.1f, 0.9f, 0.999f, 1e-8f], [1L], [1f], [1L], [0f], 1).TrainableParams, [0.9f], 1e-5f);
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, AdamaxOptimizer.ComputationGraph,
            [0.1f, 0.9f, 0.999f, 1e-8f], [1L], [1f], [1L], [0f], 2).TrainableParams, [0.80516833f], 1e-5f);

        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, NAdamOptimizer.ComputationGraph,
            [0.1f, 0.9f, 0.999f, 1e-8f, 0.004f], [1L], [1f], [1L], [0f], 1).TrainableParams, [0.89435482f], 1e-5f);
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, NAdamOptimizer.ComputationGraph,
            [0.1f, 0.9f, 0.999f, 1e-8f, 0.004f], [1L], [1f], [1L], [0f], 2).TrainableParams, [0.81997307f], 1e-5f);

        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, RAdamOptimizer.ComputationGraph,
            [0.1f, 0.9f, 0.999f, 1e-8f], [1L], [1f], [1L], [0f], 1).TrainableParams, [0.8f], 1e-5f);
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, RAdamOptimizer.ComputationGraph,
            [0.1f, 0.9f, 0.999f, 1e-8f], [1L], [1f], [1L], [0f], 2).TrainableParams, [0.62105263f], 1e-5f);

        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, AdadeltaOptimizer.ComputationGraph,
            [1.0f, 0.9f, 1e-6f], [1L], [1f], [1L], [0f], 1).TrainableParams, [0.99683773f], 1e-5f);
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, AdadeltaOptimizer.ComputationGraph,
            [1.0f, 0.9f, 1e-6f], [1L], [1f], [1L], [0f], 2).TrainableParams, [0.99359817f], 1e-5f);

        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, LionOptimizer.ComputationGraph,
            [0.1f, 0.9f, 0.99f, 0.0f], [1L], [1f], [1L], [0f], 1).TrainableParams, [0.9f], 1e-5f);
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, LionOptimizer.ComputationGraph,
            [0.1f, 0.9f, 0.99f, 0.0f], [1L], [1f], [1L], [0f], 2).TrainableParams, [0.8f], 1e-5f);
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, LionOptimizer.ComputationGraph,
            [0.1f, 0.9f, 0.99f, 1.0f], [1L], [1f], [1L], [0f], 1).TrainableParams, [0.8f], 1e-5f);
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, LionOptimizer.ComputationGraph,
            [0.5f, 0.9f, 0.99f, 0.0f], [1L], [1f], [1L], [0f], 4).TrainableParams, [0.0f], 1e-5f);

        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, AdafactorOptimizer.ComputationGraph,
            [0.01f, -0.8f, 1e-30f, 1e-3f, 1.0f, 0.0f], [1L], [1f], [1L], [0f], 1).TrainableParams, [0.99f], 1e-5f);
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, AdafactorOptimizer.ComputationGraph,
            [0.01f, -0.8f, 1e-30f, 1e-3f, 1.0f, 0.0f], [1L], [1f], [1L], [0f], 2).TrainableParams, [0.98014250f], 1e-5f);
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, AdafactorOptimizer.ComputationGraph,
            [0.01f, -0.8f, 1e-30f, 1e-3f, 0.5f, 0.0f], [1L], [1f], [1L], [0f], 1).TrainableParams, [0.995f], 1e-5f);
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, AdafactorOptimizer.ComputationGraph,
            [0.01f, -0.8f, 1e-30f, 1e-3f, 1.0f, 0.5f], [1L], [1f], [1L], [0f], 1).TrainableParams, [0.985f], 1e-5f);

        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, LambOptimizer.ComputationGraph,
            [0.1f, 0.9f, 0.999f, 1e-6f, 0.0f], [1L], [1f], [1L], [0f], 1).TrainableParams, [0.9f], 1e-4f);
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, LambOptimizer.ComputationGraph,
            [0.1f, 0.9f, 0.999f, 1e-6f, 0.0f], [1L], [1f], [1L], [0f], 2).TrainableParams, [0.81f], 1e-4f);
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, LambOptimizer.ComputationGraph,
            [0.1f, 0.9f, 0.999f, 1e-6f, 0.0f], [1L], [1f], [1L], [0f], 3).TrainableParams, [0.729f], 1e-4f);

        var zeroGuard = TrainAnalytic(AnalyticScalarWModel.ComputationGraph, LambOptimizer.ComputationGraph,
            [0.1f, 0.9f, 0.999f, 1e-6f, 0.0f], [1L], [1f], [1L], [1f], 1).TrainableParams;
        AssertStructIs(zeroGuard, [1.0f], 1e-6f);
        Assert.True(float.IsFinite(Floats(zeroGuard.Fields[zeroGuard.Definition.Fields[0].Name])[0]));
    }

    /// <summary>LAMB weight decay on/off, MULTI-STEP and MULTI-ELEMENT: the 1-element vehicle hides
    /// the WD term (trust·u = ‖w‖·sign(u) = w cancels it), so this uses AnalyticReluModel
    /// (w₀ = [1,2,3,4], all-positive x so every relu mask is 1). With a genuine multi-element
    /// ‖w‖/‖u‖ the single scalar trust ratio no longer cancels WD, so ‖w‖_on &lt; ‖w‖_off
    /// (≈ 4.455 vs ≈ 4.541 at 2 steps) and every element stays finite.</summary>
    [Fact]
    public void TestLambWeightDecayDivergesMultiStep()
    {
        const int steps = 2;

        var ckptOff = TrainAnalytic(AnalyticReluModel.ComputationGraph, LambOptimizer.ComputationGraph,
            [0.1f, 0.9f, 0.999f, 1e-6f, 0.0f], [4L], [1f, 1f, 1f, 1f], [4L], [0f, 0f, 0f, 0f], steps).TrainableParams;
        var ckptOn = TrainAnalytic(AnalyticReluModel.ComputationGraph, LambOptimizer.ComputationGraph,
            [0.1f, 0.9f, 0.999f, 1e-6f, 0.5f], [4L], [1f, 1f, 1f, 1f], [4L], [0f, 0f, 0f, 0f], steps).TrainableParams;

        string fName = ckptOff.Definition.Fields[0].Name;
        float[] wOff = Floats(ckptOff.Fields[fName]);
        float[] wOn = Floats(ckptOn.Fields[fName]);

        Assert.Equal(4, wOff.Length);
        Assert.All(wOff, x => Assert.True(float.IsFinite(x)));
        Assert.All(wOn, x => Assert.True(float.IsFinite(x)));
        Assert.True(MathF.Sqrt(wOn.Sum(x => x * x)) < MathF.Sqrt(wOff.Sum(x => x * x)));
    }

    /// <summary>Schedules.Linear(0.2, 0.1, 1) applies lr(0) at the FIRST step and lr(1) at the second
    /// (w₀=1, t=3 ⇒ w₂ = 2.04; both off-by-one variants give 2.28 / 2.072); training-mode BatchNorm2d
    /// follows the documented ONNX EMA with BIASED variance ([0.25, 1.025] for batch [1,2,3,4] at
    /// momentum 0.9, isolated with lr=0); and ToConcreteModel binds W/b by ToShorokooIdString name so
    /// the library Linear gives exactly x·Wᵀ + b = [13, 27].</summary>
    [Fact]
    public void TestScheduleBatchNormStateAndWeightBindingAnalytic()
    {
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, SGDOptimizer.ComputationGraph,
            [Schedules.Linear(0.2f, 0.1f, 1)], [1L], [1f], [1L], [3f], 2).TrainableParams, [2.04f], 1e-5f);
        AssertStructIs(TrainAnalytic(AnalyticBatchNormModel.ComputationGraph, SGDOptimizer.ComputationGraph,
            [0f], [1L, 1L, 2L, 2L], [1f, 2f, 3f, 4f], [1L, 1L, 2L, 2L], [0f, 0f, 0f, 0f], 1).ModelState,
            [0.25f, 1.025f], 1e-4f);

        var g = AnalyticBindLinearModel.ComputationGraph;
        var x = TensorData([1L, 2L], 1f, 1f);
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([x]));
        var infos = arch.GetConcreteModelParamInfos().ParamInfos;
        Assert.Equal(2, infos.Length);
        var weights = new ModelParamList(
            [
                Tuple.Create(infos[0].ToShorokooIdString(), (TensorData)TensorData([2L, 2L], 1f, 2f, 3f, 4f)),
                Tuple.Create(infos[1].ToShorokooIdString(), (TensorData)TensorData([2L], 10f, 20f)),
            ],
            ModelParamType.TrainableParam);
        var y = new ComputeContext().Execute(arch.ToConcreteModel(weights), x)[0]
            .ToTensorData().As<float32>().AccessMemory<float>().ToArray();
        Assert.Equal(2, y.Length);
        Assert.True(MathF.Abs(y[0] - 13f) < 1e-5f && MathF.Abs(y[1] - 27f) < 1e-5f);
    }
}

/// <summary>
/// Layer trainability through the rig: Bilinear / Embedding / EmbeddingBag / triplet + cosine
/// embeddings, the generalized Convolution.Conv corner and the configurable CrossEntropy variants.
/// </summary>
[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class NNLibraryLayerTrainingCoverageTests
{
    /// <summary>One TrainStep of <paramref name="model"/> + <paramref name="loss"/> +
    /// <paramref name="optimizer"/> on a fixed batch: finite loss, threaded optimizer state (when the
    /// optimizer has any), and the first rank-<paramref name="paramRank"/> trainable param moved.</summary>
    private static void AssertTrainStepMovesParam(
        ComputationGraph model, ComputationGraph loss, ComputationGraph optimizer,
        (string name, long[] shape, float[] data)[] inputs, long[] targetShape,
        int paramRank, bool statefulOptimizer, Hyperparameter[] hypers)
    {
        NamedModelParam[] sample =
            [.. inputs.Select(i => new TensorDataModelParam(
                i.name, ModelParamType.InputParam, TensorData(i.shape, i.data)))];
        var rig = TrainingRig.FromScratch(model, loss, optimizer, sample, hypers);

        var initial = rig.CreateInitialCheckpoint();
        Assert.NotEmpty(rig.TrainableParamStructDef.Fields);

        string? wName = null;
        foreach (var f in rig.TrainableParamStructDef.Fields)
            if (initial.TrainableParams.Fields[f.Name] is TensorData td && td.Shape.Dims.Length == paramRank)
            { wName = f.Name; break; }
        Assert.NotNull(wName);
        float[] w0 = Floats(initial.TrainableParams.Fields[wName]);
        Assert.True(w0.Length >= 2);

        long targetTotal = 1;
        foreach (var d in targetShape) targetTotal *= d;
        var step = rig.TrainStep(initial,
            MakeStruct("ModelInput",
                [.. inputs.Select(i => (i.name, (TensorData)TensorData(i.shape, i.data)))]),
            MakeBatch("targets", "Target", TensorData(targetShape, new float[targetTotal])));
        float[] w1 = Floats(step.TrainableParams.Fields[wName]);

        Assert.True(float.IsFinite(step.Loss!.Value));
        if (statefulOptimizer) Assert.NotEmpty(step.OptimizerState.Fields);
        Assert.True(w0.Zip(w1).Any(p => MathF.Abs(p.First - p.Second) > 1e-7f));
    }

    private static void AssertTinyConvRigTrainStepFlows(ComputationGraph modelGraph, ComputationGraph lossGraph)
    {
        var (inputData, targetData) = MakeTinyConvBatch();
        var rig = TrainingRig.FromScratch(
            modelGraph, lossGraph, SGDMomentumOptimizer.ComputationGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, inputData)],
            0.2f, 0.9f);

        Assert.NotEmpty(rig.TrainableParamStructDef.Fields);
        var initial = rig.CreateInitialCheckpoint();
        var step = rig.TrainStep(initial,
            MakeBatch("input", "ModelInput", inputData),
            MakeBatch("targets", "Target", targetData));

        Assert.True(float.IsFinite(step.Loss!.Value));
        Assert.True(AnyParamMoved(rig, initial, step));
    }

    /// <summary>Layer trainability: the rank-3 Bilinear A (the first module-level Einsum-autodiff
    /// exercise), the paddingIdx-masked Embedding table, the EmbeddingBag Gather+Reduce table, and
    /// the shared Linear embeddings behind the triplet-margin and cosine-embedding losses (the
    /// "loss-is-the-model-tail" recipe) each move under one TrainStep.</summary>
    [Fact]
    public void TestLayerRigTrainStepsMoveWeights()
    {
        Hyperparameter[] adam = [0.01f, 0.9f, 0.999f, 1e-8f];

        AssertTrainStepMovesParam(
            BilinearRigModel.ComputationGraph, L2Loss.ComputationGraph, AdamOptimizer.ComputationGraph,
            [("x1", [2L, 3L], [0.5f, -1f, 0.25f, 1f, -0.5f, 0.75f]),
             ("x2", [2L, 4L], [0.3f, -0.5f, 0.2f, -0.1f, 0.4f, 0.6f, -0.2f, 0.8f])],
            [2L], 3, true, adam);

        AssertTrainStepMovesParam(
            EmbeddingPaddingRigModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [("x", [3L], [0.5f, -1f, 0.25f])], [3L], 2, false, [0.1f]);

        AssertTrainStepMovesParam(
            EmbeddingBagRigModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [("x", [2L], [0.5f, -1f])], [2L], 2, false, [0.1f]);

        // [3N, D] = [6, 2]: rows 0-1 anchor, 2-3 positive (near), 4-5 negative (far).
        AssertTrainStepMovesParam(
            NNTripletEmbeddingRigModel.ComputationGraph, NNIdentityScalarLoss.ComputationGraph,
            AdamOptimizer.ComputationGraph,
            [("x", [6L, 2L], [0.5f, 1.0f, -0.5f, -1.0f, 0.6f, 1.1f, -0.4f, -0.9f, 2.0f, -2.0f, 2.5f, 3.0f])],
            [1L], 2, true, adam);

        // [2N, D] = [4, 2]: rows 0-1 are x1, rows 2-3 are x2.
        AssertTrainStepMovesParam(
            NNCosineEmbeddingRigModel.ComputationGraph, NNIdentityScalarLoss.ComputationGraph,
            AdamOptimizer.ComputationGraph,
            [("x", [4L, 2L], [0.5f, 1.0f, -0.5f, -1.0f, 0.6f, 1.1f, -0.4f, -0.9f])],
            [1L], 2, true, adam);
    }

    /// <summary>The generalized Convolution.Conv differentiable corner (groups:1, explicit pads,
    /// Zeros mode) trains, and the configurable CrossEntropy variants — baked
    /// <c>ignoreIndex:7 + reduction:Sum</c> and a baked-constant class <c>weight=[2,1]</c> — still
    /// satisfy the rig's (predictions, targets) → scalar contract and pass gradient through.</summary>
    [Fact]
    public void TestGeneralizedConvAndConfigurableCrossEntropyRigTrainStepFlows()
    {
        AssertTinyConvRigTrainStepFlows(ConvGeneralizedTrainModel.ComputationGraph, CrossEntropyLoss.ComputationGraph);
        AssertTinyConvRigTrainStepFlows(NNTinyConvClassifier.ComputationGraph, NNCrossEntropyIgnoreSumLoss.ComputationGraph);
        AssertTinyConvRigTrainStepFlows(NNTinyConvClassifier.ComputationGraph, NNCrossEntropyBakedWeightLoss.ComputationGraph);
    }
}

// -----------------------------------------------------------------------
//  Generalized rank-generic BatchNorm coverage. Every BatchNorm graph carries StateUpdate
//  links, so ALL of these run through the rig (not AutoTest) — even the "pure" eval
//  closed-form checks. Closed-form / alias-equivalence models output (y − reference); a zero
//  target makes the L2 loss the mean squared elementwise deviation, so loss ≈ 0 pins exact
//  per-element equality.
// -----------------------------------------------------------------------

/// <summary>BatchNorm eval-path closed forms and aliases, plus the affine on/off param-count gate
/// shared with InstanceNorm and GroupNorm.</summary>
[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class NNLibraryBatchNormEvalTrainingCoverageTests
{
    /// <summary>One L2 + SGD TrainStep of a residual model against a zero target; the returned
    /// checkpoint's Loss is the mean squared deviation from the model's reference.</summary>
    private static TrainingCheckpoint RunResidualStep(
        ComputationGraph modelGraph, long[] inShape, float[] input, long[] outShape, float lr = 0f)
    {
        var inputData = TensorData(inShape, input);
        var rig = TrainingRig.FromScratch(modelGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, inputData)], lr);
        long outTotal = 1;
        foreach (var d in outShape) outTotal *= d;
        var targetData = TensorData(outShape, new float[outTotal]);
        return rig.TrainStep(rig.CreateInitialCheckpoint(),
            MakeBatch("input", "ModelInput", inputData),
            MakeBatch("targets", "Target", targetData));
    }

    private static int TrainableFieldCount(ComputationGraph modelGraph, TensorData inputData)
        => TrainingRig.FromScratch(modelGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, inputData)], 0.5f)
            .TrainableParamStructDef.Fields.Length;

    /// <summary>BatchNorm eval path: the closed form y = x/sqrt(1+eps) at ranks 2/3/4/5 (incl. the
    /// rank-3 [N,C,L] the old BatchNorm1d rejected and the new rank-5 path); the BatchNorm2d/1d/3d
    /// aliases equal the generic call bit-for-bit; and gamma and beta both take gradient through
    /// eval while the running stats stay untouched.</summary>
    [Fact]
    public void TestBatchNormEvalClosedFormsAndAliases()
    {
        Assert.True(RunResidualStep(NNBatchNormEvalRank2ClosedForm.ComputationGraph,
            [2L, 3L], Ramp(6, 0.5f, -1f), [2L, 3L]).Loss!.Value < 1e-8f);
        Assert.True(RunResidualStep(NNBatchNormEvalRank3ClosedForm.ComputationGraph,
            [2L, 3L, 4L], Ramp(24, 0.25f, -2f), [2L, 3L, 4L]).Loss!.Value < 1e-8f);
        Assert.True(RunResidualStep(NNBatchNormEvalRank4ClosedForm.ComputationGraph,
            [2L, 3L, 4L, 4L], Ramp(96, 0.1f, -3f), [2L, 3L, 4L, 4L]).Loss!.Value < 1e-8f);
        Assert.True(RunResidualStep(NNBatchNormEvalRank5ClosedForm.ComputationGraph,
            [2L, 3L, 2L, 2L, 2L], Ramp(48, 0.2f, -2f), [2L, 3L, 2L, 2L, 2L]).Loss!.Value < 1e-8f);

        Assert.True(RunResidualStep(NNBatchNorm2dAliasEquiv.ComputationGraph,
            [2L, 3L, 4L, 4L], Ramp(96, 0.1f, -3f), [2L, 3L, 4L, 4L]).Loss!.Value < 1e-10f);
        Assert.True(RunResidualStep(NNBatchNorm1dAliasEquivRank2.ComputationGraph,
            [2L, 3L], Ramp(6, 0.5f, -1f), [2L, 3L]).Loss!.Value < 1e-10f);
        Assert.True(RunResidualStep(NNBatchNorm1dAliasEquivRank3.ComputationGraph,
            [2L, 3L, 4L], Ramp(24, 0.25f, -2f), [2L, 3L, 4L]).Loss!.Value < 1e-10f);
        Assert.True(RunResidualStep(NNBatchNorm3dAliasEquiv.ComputationGraph,
            [2L, 3L, 2L, 2L, 2L], Ramp(48, 0.2f, -2f), [2L, 3L, 2L, 2L, 2L]).Loss!.Value < 1e-10f);

        // Eval closed form through the rig's loss (pred_c = mean_{n,h,w}(x)/sqrt(1+eps)), gamma AND
        // beta both moving, and the running stats untouched by an eval pass.
        var vals = Enumerable.Range(0, 24).Select(i => (float)i).ToArray();
        var inputData = TensorData([2L, 3L, 2L, 2L], vals);
        float[] targetVals = [1f, 2f, 3f];
        var rig = TrainingRig.FromScratch(
            NNBatchNormEvalGradModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, inputData)], 0.5f);
        var initial = rig.CreateInitialCheckpoint();
        var step = rig.TrainStep(initial,
            MakeBatch("input", "ModelInput", inputData),
            MakeBatch("targets", "Target", TensorData([3L], targetVals)));

        float invStd = 1f / MathF.Sqrt(1f + 1e-5f);
        float expectedLoss = 0f;
        for (int c = 0; c < 3; c++)
        {
            float sum = 0f;
            for (int n = 0; n < 2; n++)
                for (int s = 0; s < 4; s++)
                    sum += vals[n * 12 + c * 4 + s];
            float pred = sum / 8f * invStd;
            expectedLoss += (pred - targetVals[c]) * (pred - targetVals[c]);
        }
        expectedLoss /= 3f;
        Assert.True(MathF.Abs(step.Loss!.Value - expectedLoss) < 1e-3f);

        Assert.Equal(2, rig.TrainableParamStructDef.Fields.Length);
        foreach (var field in rig.TrainableParamStructDef.Fields)
            Assert.True(Floats(initial.TrainableParams.Fields[field.Name])
                .Zip(Floats(step.TrainableParams.Fields[field.Name]))
                .Any(p => MathF.Abs(p.First - p.Second) > 1e-7f));

        foreach (var field in rig.ModelStateDef.Fields)
            Assert.True(Floats(initial.ModelState.Fields[field.Name])
                .Zip(Floats(step.ModelState.Fields[field.Name]))
                .All(p => MathF.Abs(p.First - p.Second) < 1e-7f));
    }

    /// <summary>The affine bit gates whether γ/β receive gradient: with affine:false the eval output
    /// is the bare normalizer x/sqrt(1+eps), and γ/β sit on the dead branch and are pruned — so an
    /// otherwise-identical model exposes only the upstream scalar weight. Checked for BatchNorm
    /// (1 vs 2 params) and for InstanceNorm and GroupNorm (1 vs 3).</summary>
    [Fact]
    public void TestBatchNormInstanceAndGroupNormAffineOnOff()
    {
        Assert.True(RunResidualStep(NNBatchNormAffineFalseClosedForm.ComputationGraph,
            [2L, 3L, 2L, 2L], Ramp(24, 0.5f, -3f), [2L, 3L, 2L, 2L]).Loss!.Value < 1e-8f);

        var bnInput = TensorData([2L, 3L, 2L, 2L], Ramp(24));
        Assert.Equal(1, TrainableFieldCount(NNBatchNormAffineFalseEvalGradModel.ComputationGraph, bnInput));
        Assert.Equal(2, TrainableFieldCount(NNBatchNormEvalGradModel.ComputationGraph, bnInput));

        var normInput = TensorData([2L, 4L, 3L, 3L], Ramp(72));
        Assert.Equal(1, TrainableFieldCount(NNInstanceNormAffineFalseParamModel.ComputationGraph, normInput));
        Assert.Equal(3, TrainableFieldCount(NNInstanceNormAffineTrueParamModel.ComputationGraph, normInput));
        Assert.Equal(1, TrainableFieldCount(NNGroupNormAffineFalseParamModel.ComputationGraph, normInput));
        Assert.Equal(3, TrainableFieldCount(NNGroupNormAffineTrueParamModel.ComputationGraph, normInput));
    }
}

/// <summary>BatchNorm train-path normalization, EMA running stats and track/selection, plus the
/// Dropout-family gradient-flow paths.</summary>
[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class NNLibraryBatchNormTrainAndDropoutCoverageTests
{
    /// <summary>One train step: the per-channel-mean output is ~0 (loss ~0 vs zero targets) and
    /// every ModelState field moves.</summary>
    private static void AssertTrainNormalizesAndMovesState(
        ComputationGraph modelGraph, long[] inShape, float[] input, long[] outShape)
    {
        var inputData = TensorData(inShape, input);
        long outTotal = 1;
        foreach (var d in outShape) outTotal *= d;
        var rig = TrainingRig.FromScratch(modelGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, inputData)], 0.1f);
        var initial = rig.CreateInitialCheckpoint();
        var step = rig.TrainStep(initial,
            MakeBatch("input", "ModelInput", inputData),
            MakeBatch("targets", "Target", TensorData(outShape, new float[outTotal])));

        Assert.True(step.Loss!.Value < 1e-6f);
        Assert.NotEmpty(rig.ModelStateDef.Fields);
        foreach (var field in rig.ModelStateDef.Fields)
            Assert.True(Floats(initial.ModelState.Fields[field.Name])
                .Zip(Floats(step.ModelState.Fields[field.Name]))
                .Any(p => MathF.Abs(p.First - p.Second) > 1e-7f));
    }

    /// <summary>BatchNorm train path at ranks 2/3/4/5: the batch-normalized output has ~zero
    /// per-channel mean and the pass EMA-updates both running stats; plus the analytic EMA on input
    /// [1,2,3,4] (mean 2.5, BIASED var 1.25) pinning the ONNX/Keras momentum sense at 0.9 ⇒
    /// [0.25, 1.025] and 0.5 ⇒ [1.25, 1.125], the rank-generic {0,2} reduction, and the rank-2
    /// batch-axis-only reduction on [2,1] values [1,3] ⇒ [0.2, 1.0].</summary>
    [Fact]
    public void TestBatchNormTrainNormalizationAndEmaAnalytic()
    {
        AssertTrainNormalizesAndMovesState(NNBatchNormTrainGradModel.ComputationGraph,
            [2L, 3L, 2L, 2L], Enumerable.Range(0, 24).Select(i => (float)i).ToArray(), [3L]);
        AssertTrainNormalizesAndMovesState(NNBatchNormTrainRank2Model.ComputationGraph,
            [4L, 3L], Ramp(12, 1f, -5f), [3L]);
        AssertTrainNormalizesAndMovesState(NNBatchNormTrainRank3Model.ComputationGraph,
            [2L, 3L, 4L], Ramp(24, 1f, -5f), [3L]);
        AssertTrainNormalizesAndMovesState(NNBatchNormTrainRank5Model.ComputationGraph,
            [2L, 3L, 2L, 2L, 2L], Ramp(48, 1f, -10f), [3L]);

        AssertStructIs(TrainAnalytic(NNBatchNormAnalyticMomentum09Model.ComputationGraph, SGDOptimizer.ComputationGraph,
            [0f], [1L, 1L, 2L, 2L], [1f, 2f, 3f, 4f], [1L, 1L, 2L, 2L], [0f, 0f, 0f, 0f], 1).ModelState,
            [0.25f, 1.025f], 1e-4f);
        AssertStructIs(TrainAnalytic(NNBatchNormAnalyticMomentum05Model.ComputationGraph, SGDOptimizer.ComputationGraph,
            [0f], [1L, 1L, 2L, 2L], [1f, 2f, 3f, 4f], [1L, 1L, 2L, 2L], [0f, 0f, 0f, 0f], 1).ModelState,
            [1.25f, 1.125f], 1e-4f);
        AssertStructIs(TrainAnalytic(NNBatchNormAnalyticMomentum09Model.ComputationGraph, SGDOptimizer.ComputationGraph,
            [0f], [1L, 1L, 4L], [1f, 2f, 3f, 4f], [1L, 1L, 4L], [0f, 0f, 0f, 0f], 1).ModelState,
            [0.25f, 1.025f], 1e-4f);
        AssertStructIs(TrainAnalytic(NNBatchNormAnalyticRank2Model.ComputationGraph, SGDOptimizer.ComputationGraph,
            [0f], [2L, 1L], [1f, 3f], [2L, 1L], [0f, 0f], 1).ModelState,
            [0.2f, 1.0f], 1e-4f);
    }

    // Helpers for the track/selection checks: train-then-eval with cross-rig ModelState injection.
    // Both train and eval models use the generic BatchNorm with the same channel count, so their
    // ModelState field layout is identical and the moved running stats can be injected into a fresh
    // eval checkpoint; the eval output is read indirectly through the L2 loss against a
    // hand-computed closed-form target.

    private static (float loss, float[] state) RunTrainLossAndState(
        ComputationGraph modelGraph, long[] inShape, float[] input, long[] outShape)
    {
        long outTotal = 1;
        foreach (var d in outShape) outTotal *= d;
        var inputData = TensorData(inShape, input);
        var rig = TrainingRig.FromScratch(modelGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, inputData)], 0f);
        var step = rig.TrainStep(rig.CreateInitialCheckpoint(),
            MakeBatch("input", "ModelInput", inputData),
            MakeBatch("targets", "Target", TensorData(outShape, new float[outTotal])));
        var state = rig.ModelStateDef.Fields.SelectMany(f => Floats(step.ModelState.Fields[f.Name])).ToArray();
        return (step.Loss!.Value, state);
    }

    /// <summary>Runs an eval-mode BN model with <paramref name="injectedState"/> as its ModelState,
    /// reading the eval output through the L2 loss against <paramref name="matchTarget"/> (≈0 when it
    /// matches) and <paramref name="mismatchTarget"/> (high when the two paths genuinely differ), and
    /// reports whether the eval pass moved ModelState (it must not).</summary>
    private static (float matchLoss, float mismatchLoss, bool stateMoved) EvalLossAgainstTargets(
        ComputationGraph modelGraph, long[] inShape, float[] input, float[] injectedState,
        float[] matchTarget, float[] mismatchTarget)
    {
        var inputData = TensorData(inShape, input);

        float LossAgainst(float[] target, out float[] stateAfter)
        {
            var rig = TrainingRig.FromScratch(modelGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
                [new TensorDataModelParam("input", ModelParamType.InputParam, inputData)], 0f);
            var fresh = rig.CreateInitialCheckpoint();
            var injected = InjectModelState(fresh.ModelState, injectedState);
            var ckpt = new TrainingCheckpoint(fresh.TrainableParams, injected, fresh.OptimizerState, fresh.Step);
            var step = rig.TrainStep(ckpt,
                MakeBatch("input", "ModelInput", inputData),
                MakeBatch("targets", "Target", TensorData(inShape, target)));
            stateAfter = rig.ModelStateDef.Fields.SelectMany(f => Floats(step.ModelState.Fields[f.Name])).ToArray();
            return step.Loss!.Value;
        }

        float matchLoss = LossAgainst(matchTarget, out var afterMatch);
        float mismatchLoss = LossAgainst(mismatchTarget, out _);
        bool stateMoved = injectedState.Zip(afterMatch).Any(p => MathF.Abs(p.First - p.Second) > 1e-6f);
        return (matchLoss, mismatchLoss, stateMoved);
    }

    /// <summary>Builds a ModelState copy of <paramref name="template"/> with field i filled with the
    /// i-th value of <paramref name="values"/> (one running-stat scalar per field).</summary>
    private static TensorDataStruct InjectModelState(TensorDataStruct template, float[] values)
    {
        var fields = new Dictionary<string, IData>();
        int i = 0;
        foreach (var f in template.Definition.Fields)
        {
            var existing = (TensorData)template.Fields[f.Name];
            int count = (int)existing.Shape.Dims.Aggregate(1L, (a, b) => a * b);
            var filled = Enumerable.Repeat(values[i], count).ToArray();
            fields[f.Name] = TensorData(existing.Shape.Dims, filled);
            i++;
        }
        return new TensorDataStruct(template.Definition, fields);
    }

    /// <summary>track_running_stats on/off and train-vs-eval stat selection. After a train step moves
    /// the running stats to [0.25, 1.025], a track:true eval normalizes with those MOVED stats while a
    /// track:false eval normalizes with the eval BATCH stats — each matching its own closed form
    /// (loss ≈ 0) and NOT the other's — and neither eval pass touches ModelState.</summary>
    [Fact]
    public void TestBatchNormTrackRunningStatsAndTrainVsEvalSelection()
    {
        const float eps = 1e-5f;
        float[] input = [1f, 2f, 3f, 4f];   // [1,1,2,2]: mean 2.5, biased var 1.25

        var (trainLoss, movedState) = RunTrainLossAndState(
            NNBatchNormTrainGradModel.ComputationGraph, [1L, 1L, 2L, 2L], input, [1L]);
        Assert.True(trainLoss < 1e-5f);
        Assert.Equal(2, movedState.Length);
        Assert.True(MathF.Abs(movedState[0] - 0.25f) < 1e-4f && MathF.Abs(movedState[1] - 1.025f) < 1e-4f);

        // The full-output train model must move the running stats to the same [0.25, 1.025].
        var fullState = RunTrainLossAndState(NNBatchNormTrainFullModel.ComputationGraph,
            [1L, 1L, 2L, 2L], input, [1L, 1L, 2L, 2L]).state;
        Assert.Equal(2, fullState.Length);
        Assert.True(MathF.Abs(fullState[0] - 0.25f) < 1e-4f && MathF.Abs(fullState[1] - 1.025f) < 1e-4f);

        float runMean = movedState[0], runVar = movedState[1];
        float runInvStd = 1f / MathF.Sqrt(runVar + eps);

        // Same input: eval (track:true) uses the running stats, NOT the zero-mean batch stats.
        var sameEvalExpected = input.Select(v => (v - runMean) * runInvStd).ToArray();
        var sameBatchExpected = input.Select(v => (v - 2.5f) / MathF.Sqrt(1.25f + eps)).ToArray();
        Assert.True(sameEvalExpected.Zip(sameBatchExpected).Sum(p => MathF.Abs(p.First - p.Second)) > 1e-2f);
        var sameEval = EvalLossAgainstTargets(NNBatchNormEvalTrackTrueFullModel.ComputationGraph,
            [1L, 1L, 2L, 2L], input, movedState, sameEvalExpected, sameBatchExpected);
        Assert.True(sameEval.matchLoss < 1e-4f);
        Assert.True(sameEval.mismatchLoss > 1e-2f);

        // A DIFFERENT eval batch so the batch stats differ from the running stats.
        float[] evalInput = [2f, 4f, 6f, 8f];   // mean 5, biased var 5
        var trackTrueExpected = evalInput.Select(v => (v - runMean) * runInvStd).ToArray();
        var trackFalseExpected = evalInput.Select(v => (v - 5f) / MathF.Sqrt(5f + eps)).ToArray();
        Assert.True(trackTrueExpected.Zip(trackFalseExpected).Sum(p => MathF.Abs(p.First - p.Second)) > 1e-2f);

        var trackTrue = EvalLossAgainstTargets(NNBatchNormEvalTrackTrueFullModel.ComputationGraph,
            [1L, 1L, 2L, 2L], evalInput, movedState, trackTrueExpected, trackFalseExpected);
        Assert.True(trackTrue.matchLoss < 1e-4f);
        Assert.True(trackTrue.mismatchLoss > 1e-2f);
        Assert.False(trackTrue.stateMoved);

        var trackFalse = EvalLossAgainstTargets(NNBatchNormEvalTrackFalseFullModel.ComputationGraph,
            [1L, 1L, 2L, 2L], evalInput, movedState, trackFalseExpected, trackTrueExpected);
        Assert.True(trackFalse.matchLoss < 1e-4f);
        Assert.True(trackFalse.mismatchLoss > 1e-2f);
        Assert.False(trackFalse.stateMoved);
    }

    /// <summary>Eval mode is the exact identity, so the loss equals the no-dropout closed form
    /// mean(x²) = 7.5 and the upstream scalar weight takes the full gradient
    /// w1 = 1 − lr·2·mean(x²) = −0.5.</summary>
    private static void AssertDropoutEvalIsIdentity(ComputationGraph modelGraph, long[] shape)
    {
        float[] xs = [1f, 2f, 3f, 4f];
        var inputData = TensorData(shape, xs);
        var rig = TrainingRig.FromScratch(modelGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, inputData)], 0.1f);
        var step = rig.TrainStep(rig.CreateInitialCheckpoint(),
            MakeBatch("input", "ModelInput", inputData),
            MakeBatch("targets", "Target", TensorData(shape, new float[4])));

        Assert.True(MathF.Abs(step.Loss!.Value - 7.5f) < 1e-4f);
        string wName = rig.TrainableParamStructDef.Fields[0].Name;
        Assert.True(MathF.Abs(Floats(step.TrainableParams.Fields[wName])[0] + 0.5f) < 1e-3f);
    }

    /// <summary>Train mode: the RNG mask forbids an exact value, so only the finite loss and a moved
    /// param (gradient reached the upstream weight through the mask) are asserted.</summary>
    private static void AssertDropoutTrainStepMoves(ComputationGraph modelGraph, long[] shape)
    {
        float[] xs = [1f, 2f, 3f, 4f];
        var inputData = TensorData(shape, xs);
        var rig = TrainingRig.FromScratch(modelGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, inputData)], 0.1f);
        var initial = rig.CreateInitialCheckpoint();
        Assert.NotEmpty(rig.TrainableParamStructDef.Fields);

        string wName = rig.TrainableParamStructDef.Fields[0].Name;
        float w0 = Floats(initial.TrainableParams.Fields[wName])[0];
        var step = rig.TrainStep(initial,
            MakeBatch("input", "ModelInput", inputData),
            MakeBatch("targets", "Target", TensorData(shape, new float[4])));

        Assert.True(float.IsFinite(step.Loss!.Value));
        Assert.True(MathF.Abs(Floats(step.TrainableParams.Fields[wName])[0] - w0) > 1e-7f);
    }

    /// <summary>Dropout / SpatialDropout / AlphaDropout eval-mode identity anchors (loss 7.5,
    /// w1 = −0.5) and the SpatialDropout / AlphaDropout / FeatureAlphaDropout train-mode gradient
    /// paths through the channel-broadcast or elementwise affine + mask.</summary>
    [Fact]
    public void TestDropoutFamilyRigTrainSteps()
    {
        AssertDropoutEvalIsIdentity(NNDropoutEvalGradModel.ComputationGraph, [4L]);
        AssertDropoutEvalIsIdentity(NNSpatialDropoutEvalGradModel.ComputationGraph, [1L, 2L, 2L]);
        AssertDropoutEvalIsIdentity(NNAlphaDropoutEvalGradModel.ComputationGraph, [1L, 2L, 2L]);

        AssertDropoutTrainStepMoves(NNSpatialDropoutTrainGradModel.ComputationGraph, [1L, 2L, 2L]);
        AssertDropoutTrainStepMoves(NNAlphaDropoutTrainGradModel.ComputationGraph, [1L, 2L, 2L]);
        AssertDropoutTrainStepMoves(NNFeatureAlphaDropoutTrainGradModel.ComputationGraph, [1L, 2L, 2L]);
    }
}

/// <summary>Recurrent forward goldens, the BPTT-unsupported guards and recurrent rig trainability.</summary>
[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class NNLibraryRecurrentTrainingCoverageTests
{
    private static TensorData ProbeScalar() => TensorData(DType.Float32, [], 0.3f);

    /// <summary>Frozen forward goldens over a probed-scalar sequence for Recurrent.RNN / LSTM / GRU
    /// and the RNNCell / LSTMCell / GRUCell single-step forms (forward, single-layer, layout=0,
    /// default activations). Forward values only — gradient coverage is at the op level in
    /// AutoGradOpsTests and end to end in <see cref="TestRecurrentTrainStepFlows"/>.</summary>
    [Fact]
    public void TestRecurrentForwardGoldens()
    {
        Assert.True(AutoTest.AdvancedTestGraph<RnnForwardTanhGolden>(
            hyperparamInputs: [], runtimeInputs: [ProbeScalar()]));
        Assert.True(AutoTest.AdvancedTestGraph<LstmForwardGolden>(
            hyperparamInputs: [], runtimeInputs: [ProbeScalar()]));
        Assert.True(AutoTest.AdvancedTestGraph<GruForwardGolden>(
            hyperparamInputs: [], runtimeInputs: [ProbeScalar()]));
        Assert.True(AutoTest.AdvancedTestGraph<RnnCellForwardTanhGolden>(
            hyperparamInputs: [], runtimeInputs: [ProbeScalar()]));
        Assert.True(AutoTest.AdvancedTestGraph<LstmCellForwardGolden>(
            hyperparamInputs: [], runtimeInputs: [ProbeScalar()]));
        Assert.True(AutoTest.AdvancedTestGraph<GruCellForwardGolden>(
            hyperparamInputs: [], runtimeInputs: [ProbeScalar()]));
    }

    /// <summary>Back-propagating through a relu RNN / relu RNNCell (non-default activation) or a
    /// bidirectional RNN / LSTM / GRU (unimplemented BPTT) must throw AD003 at AUTO_GRAD lowering,
    /// before any backend runs. These modes are inference-grade only.</summary>
    [Fact]
    public void TestRecurrentBpttThrows()
    {
        static void ThrowsAd003(Action call, string? messageWord = null)
        {
            var ex = Assert.Throws<AutoDiffNotSupportedException>(call);
            Assert.Equal(ErrorCodes.AD003, ex.ErrorCode);
            if (messageWord is not null) Assert.Contains(messageWord, ex.Message);
        }

        ThrowsAd003(() => AutoTest.AdvancedTestGraph<RnnReluBpttThrowCheck>(
            hyperparamInputs: [], runtimeInputs: [ProbeScalar()]));
        ThrowsAd003(() => AutoTest.AdvancedTestGraph<RnnCellReluBpttThrowCheck>(
            hyperparamInputs: [], runtimeInputs: [ProbeScalar()]));
        ThrowsAd003(() => AutoTest.AdvancedTestGraph<RnnBidirectionalBpttThrowCheck>(
            hyperparamInputs: [], runtimeInputs: [ProbeScalar()]), "bidirectional");
        ThrowsAd003(() => AutoTest.AdvancedTestGraph<LstmBidirectionalBpttThrowCheck>(
            hyperparamInputs: [], runtimeInputs: [ProbeScalar()]), "bidirectional");
        ThrowsAd003(() => AutoTest.AdvancedTestGraph<GruBidirectionalBpttThrowCheck>(
            hyperparamInputs: [], runtimeInputs: [ProbeScalar()]), "bidirectional");
    }

    /// <summary>One TrainStep of a recurrent model + loss + optimizer: the rig builds (so the path
    /// differentiates), the loss is finite, and ≥1 owned W/R/bias param moves.</summary>
    private static void AssertRecurrentRigTrainStepFlows(
        ComputationGraph modelGraph, ComputationGraph lossGraph, ComputationGraph optimizerGraph,
        long[] inShape, float[] input, TensorData targetData, params Hyperparameter[] hypers)
    {
        var inputData = TensorData(inShape, input);
        var rig = TrainingRig.FromScratch(modelGraph, lossGraph, optimizerGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, inputData)], hypers);

        Assert.NotEmpty(rig.TrainableParamStructDef.Fields);
        var initial = rig.CreateInitialCheckpoint();
        var step = rig.TrainStep(initial,
            MakeBatch("input", "ModelInput", inputData),
            MakeBatch("targets", "Target", targetData));

        Assert.True(float.IsFinite(step.Loss!.Value));
        Assert.True(AnyParamMoved(rig, initial, step));
    }

    /// <summary>Recurrent trainability: forward single-layer LSTM and GRU + CrossEntropy +
    /// SGDMomentum on [L=3,N=4,in=2] (the #440 MemoryAwareScheduler fallback that lets RNN/LSTM/GRU
    /// BPTT scopes build), and hand-unrolled 2-step RNNCell / LSTMCell / GRUCell loops + L2 + SGD on
    /// [L=2,N=4,in=2] (the cell path differentiating through a USER loop).</summary>
    [Fact]
    public void TestRecurrentTrainStepFlows()
    {
        var seqVals = Enumerable.Range(0, 3 * 4 * 2).Select(i => 0.1f * i - 1f).ToArray();
        long[] classes = [0L, 1L, 0L, 1L];
        var seqTarget = TensorData([4L], classes);

        AssertRecurrentRigTrainStepFlows(LstmForwardTrainModel.ComputationGraph,
            CrossEntropyLoss.ComputationGraph, SGDMomentumOptimizer.ComputationGraph,
            [3L, 4L, 2L], seqVals, seqTarget, 0.2f, 0.9f);
        AssertRecurrentRigTrainStepFlows(GruForwardTrainModel.ComputationGraph,
            CrossEntropyLoss.ComputationGraph, SGDMomentumOptimizer.ComputationGraph,
            [3L, 4L, 2L], seqVals, seqTarget, 0.2f, 0.9f);

        var cellVals = Enumerable.Range(0, 2 * 4 * 2).Select(i => 0.1f * i - 0.7f).ToArray();
        var cellTarget = TensorData([4L, 2L], new float[8]);
        AssertRecurrentRigTrainStepFlows(RnnCellTrainModel.ComputationGraph,
            L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph, [2L, 4L, 2L], cellVals, cellTarget, 0.1f);
        AssertRecurrentRigTrainStepFlows(LstmCellTrainModel.ComputationGraph,
            L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph, [2L, 4L, 2L], cellVals, cellTarget, 0.1f);
        AssertRecurrentRigTrainStepFlows(GruCellTrainModel.ComputationGraph,
            L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph, [2L, 4L, 2L], cellVals, cellTarget, 0.1f);
    }
}

using Shorokoo.Core.Nodes.Processors.Helpers;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage-purpose tests for AutoGrad ops, following the same one-liner pattern as
/// <see cref="Shorokoo.Tests.Modules.CoverageTests.ModulesCoverageTests"/>: each test
/// drives <see cref="AutoTest.AdvancedTestGraph{TModule}"/> against a module whose
/// <c>Inline</c> method embeds the AutoGrad operator under test. The AutoTester does
/// the work — ONNX roundtrip, CS roundtrip, QuickExecutionEngine validation — so the
/// uncovered code paths in AutoGrad lowering get exercised end-to-end.
/// </summary>
[Trait("Domain", "AutoDiff")]
[Trait("Purpose", "Coverage")]
public class AutoGradOpsCoverageTests
{
    // Each [Fact] groups the AdvancedTestGraph calls that previously lived in many one-liner
    // tests within the same category. The grouping keeps a single test under ~5s while still
    // routing every gradient op module through the AutoTester (ONNX+CS roundtrips + QEE).

    [Fact]
    public void TestAutoGradInitialMiscCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradScalarSquare>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradPairProduct>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f), TensorData(DType.Float32, [], 4f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSubBinaryCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 5f), TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSubChainedCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 5f), TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradDivCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 6f), TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradNegCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradNegChainedCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f), TensorData(DType.Float32, [], 4f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradPowCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f), TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradAbsCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradAbsCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], -3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradReciprocalCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradModSumCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 5f), TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradModSumCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 7f), TensorData(DType.Float32, [], 4f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradModWithDownstreamCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 5f), TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradAbsMulChainCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], -3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradPowSqrtIdentityCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
    }

    [Fact]
    public void TestAutoGradActivationsCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradReluCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradReluCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], -2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradReluChainedCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSigmoidCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSigmoidCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradLeakyReluCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradLeakyReluCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], -2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGeluCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGeluCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradEluCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradEluCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], -1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSeluCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSeluCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], -1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradCeluCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradCeluCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], -1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSigmoidExpChainCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradLeakyReluExpChainCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 0f)]));

        // Batch29 elementwise activations: HardSigmoid, HardSwish, Mish, Softplus, Softsign,
        // ThresholdedRelu, Shrink — exercise both smooth branches per piecewise op.
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradHardSigmoidCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradHardSwishCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradHardSwishCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], -1.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradMishCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 0.7f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradMishCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], -0.7f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSoftplusCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSoftplusCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], -0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSoftsignCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSoftsignCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], -0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradThresholdedReluCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradThresholdedReluCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 0.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradShrinkCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradShrinkCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], -1.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradShrinkCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 0.1f)]));
    }

    [Fact]
    public void TestAutoGradTrigonometricCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSinCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], MathF.PI / 4f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradCosCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], MathF.PI / 4f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradTanCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradAsinCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradAcosCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradAtanCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSinhCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradCoshCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradAsinhCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradAcoshCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradAtanhCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradTanhCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradTanhCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSinCosChainCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], MathF.PI / 6f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSinhCoshChainCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradAtanExpChainCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 0.5f)]));
    }

    [Fact]
    public void TestAutoGradMathCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradExpCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradExpChainedCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradLogCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSqrtCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 4f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSqrtDivChainCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 9f), TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradLogExpIdentityCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradErfCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSignCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradCeilCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3.2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradFloorCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3.7f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradErfMulChainCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradClipCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradClipCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], -2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradClipCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 15f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradCumSumCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradCumSumReverseCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradCumSumWithScaleCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradDet2x2IdentityCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradDet2x2DiagonalCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradDet2x2ChainRuleCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradCastLikeCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 5f), TensorData(DType.Float32, [], 0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradCastLikeWithScaleCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f), TensorData(DType.Float32, [], 0f)]));
    }

    [Fact]
    public void TestAutoGradReductionsCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradReduceProdCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradReduceSumSquareCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradReduceLogSumExpCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradReduceL1Check>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradReduceL2Check>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradReduceLogSumCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradReduceMaxCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradReduceMinCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
    }

    [Fact]
    public void TestAutoGradOptionalCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradOptionalWrapUnwrap2x3Check>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradOptionalWrapUnwrapScalarCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 5.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradOptionalWrapUnwrap3x4Check>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradOptionalWrapUnwrapWithScaleCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1.5f)]));
    }

    [Fact]
    public void TestAutoGradSequenceCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradConcat2Check>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f), TensorData(DType.Float32, [], 7f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradConcatWithScaleCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f), TensorData(DType.Float32, [], 7f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradConcat3Check>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1f), TensorData(DType.Float32, [], 2f), TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradConcatWithActivationCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f), TensorData(DType.Float32, [], -2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSplit2OutputsCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSplit3OutputsCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSplitWithScaleCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSplitConcatRoundTripCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
    }

    [Fact]
    public void TestAutoGradIfElseCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradIfTrueConditionCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f), TensorData(DType.Float32, [], 5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradIfFalseConditionCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f), TensorData(DType.Float32, [], 5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradIfSharedInputTrueCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradIfSharedInputFalseCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradIfWithDownstreamOpsCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f), TensorData(DType.Float32, [], 5f), TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradIfWithUpstreamOpsCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f), TensorData(DType.Float32, [], 5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradIfMultiOutputPartiallyUsedCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f), TensorData(DType.Float32, [], 5f)]));
    }

    [Fact]
    public void TestAutoGradPoolingCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGlobalAveragePoolCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGlobalAveragePoolWithScaleCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGlobalMaxPoolCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGlobalAveragePoolExpChainCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGlobalLpPoolP2Check>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGlobalLpPoolP1Check>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGlobalLpPoolP2WithScaleCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradLpPoolP2Check>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradLpPoolP1Check>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradLpPoolP2WithScaleCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradMaxUnpoolCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradMaxUnpoolWithScaleCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradMaxUnpoolChainCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
    }

    [Fact]
    public void TestAutoGradOtherPart1Coverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradDropoutInferenceCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradDropoutWithScaleCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradDropoutWithDownstreamCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSum2InputsCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f), TensorData(DType.Float32, [], 5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSum3InputsCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f), TensorData(DType.Float32, [], 7f), TensorData(DType.Float32, [], 4f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSumWithScaleCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f), TensorData(DType.Float32, [], 5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSumTensorInputsCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f), TensorData(DType.Float32, [], 5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradMean2InputsCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f), TensorData(DType.Float32, [], 5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradMean3InputsCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f), TensorData(DType.Float32, [], 7f), TensorData(DType.Float32, [], 4f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradMeanWithScaleCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f), TensorData(DType.Float32, [], 4f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradMeanTensorInputsCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f), TensorData(DType.Float32, [], 5f)]));
    }

    [Fact]
    public void TestAutoGradOtherPart2Coverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradMax2InputsCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f), TensorData(DType.Float32, [], 5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradMax3InputsCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f), TensorData(DType.Float32, [], 7f), TensorData(DType.Float32, [], 4f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradMaxWithScaleCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f), TensorData(DType.Float32, [], 5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradMaxTensorInputsCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f), TensorData(DType.Float32, [], 5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradMin2InputsCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f), TensorData(DType.Float32, [], 5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradMin3InputsCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 5f), TensorData(DType.Float32, [], 2f), TensorData(DType.Float32, [], 4f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradMinWithScaleCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 8f), TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradMinTensorInputsCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 7f), TensorData(DType.Float32, [], 2f)]));
    }

    [Fact]
    public void TestAutoGradMatrixCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradEinsumMatmulBasicCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradEinsumTransposeCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 5.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradEinsumImplicitModeCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradEinsumFreeIndexCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGemmBasicCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGemmWithAlphaCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGemmWithBetaAndCCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGemmTransACheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGemmTransBCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradMatMulKnownRankCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradMatMulUnknownRankBatchedCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradReduceSumExplicitAxesKeepdimsTrueCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradReduceSumExplicitAxesKeepdimsFalseCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradReduceMeanExplicitAxesCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
    }

    [Fact]
    public void TestAutoGradAffineGridCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradAffineGridMultiBatchCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradReshapePassthroughCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradDeadParam>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 7f), TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradMatMulReduce>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [4L, 3L])]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradTrainableParam>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [4L, 3L])]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSliceWithAxes>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [3L, 4L])]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradConv>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [1L, 3L, 5L, 5L])]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradConvWeight>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [1L, 3L, 5L, 5L])]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradConvTranspose>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [1L, 3L, 5L, 5L])]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradReduceMeanAllAxes>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [3L, 4L])]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSoftmax>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [2L, 4L])]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradTransposePerm>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [2L, 3L, 4L])]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradPadAxes>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [3L, 4L])]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradTile>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [2L, 3L])]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradAvgPool>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [1L, 2L, 4L, 4L])]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradAvgPoolOverlap>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [1L, 2L, 5L, 5L])],
            testQuickEngineExecution: false));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradAvgPoolPadInclude>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [1L, 2L, 4L, 4L])],
            testQuickEngineExecution: false));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradAvgPoolPadExclude>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [1L, 2L, 4L, 4L])],
            testQuickEngineExecution: false));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradAvgPoolSameUpper>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [1L, 2L, 5L, 5L])],
            testQuickEngineExecution: false));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradAvgPoolSameLower>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [1L, 2L, 5L, 5L])],
            testQuickEngineExecution: false));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradMaxPool>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [1L, 2L, 4L, 4L])]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGemmTrans>(
            hyperparamInputs: [],
            runtimeInputs: [
                TensorDataWithSmallVals(DType.Float32, [3L, 2L]),
                TensorDataWithSmallVals(DType.Float32, [4L, 3L])]));
    }

    [Fact]
    public void TestAutoGradNormalizationPart1Coverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradBatchNormSimpleCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradBatchNormWithScaleCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradBatchNorm3DCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradBatchNormExpChainCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGroupNormBasicCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGroupNormScaleCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
    }

    [Fact]
    public void TestAutoGradNormalizationPart2Coverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGroupNormNonConstInputCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGroupNorm2GroupsCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradInstanceNormBasicCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradInstanceNormWithScaleCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
    }

    [Fact]
    public void TestAutoGradNormalizationPart3Coverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradLpNormL2BasicCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradLpNormL2AsymmetricCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradLpNormL1BasicCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));

        // Batch30 normalization-flavored additions: LayerNorm, MVN, LogSoftmax, PRelu.
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradLayerNormalizationCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradMeanVarianceNormalizationCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradLogSoftmaxCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradPReluCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1.5f)]));
    }

    [Fact]
    public void TestAutoGradIndexingPart1Coverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGatherElementsAxis0Check>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGatherElementsAxis1Check>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGatherElementsWithScaleCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradScatterElementsAddCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradScatterElementsNoneCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradScatterElementsWithScaleCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradScatterNDAddCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradScatterNDNoneCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradScatterNDWithScaleCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradScatterNDReluChainCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGatherAxis0Check>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 4f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGatherDuplicateIndicesCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 4f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGatherAllIndicesCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
    }

    [Fact]
    public void TestAutoGradIndexingPart2Coverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGatherAxis0MultiDimIndicesCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGatherNonZeroAxisOneDimIndicesCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGatherNonZeroAxisOneDimIndicesUnknownRankCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGatherNonZeroAxisMultiDimIndicesCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 4f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGatherNDCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 4f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGatherNDDuplicateIndicesCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGatherNDWithScaleCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradWhereTrueBranchCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f), TensorData(DType.Float32, [], 7f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradWhereFalseBranchCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f), TensorData(DType.Float32, [], 7f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradUniqueSingleElementCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradUniqueAllSameCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradUniqueWithAxisCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradUniqueDistinctCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
    }

    [Fact]
    public void TestAutoGradShapePart1Coverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradTransposeCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 4f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradFlattenCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSqueezeCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 4f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradUnsqueezeCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradExpandCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradPadConstantCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradPadConstantWithMultiplyCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradPad2DCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 7f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradPadWithSigmoidCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradResizeNearestSumLossWithScalesCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradResizeNearestSizesCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradResizeNearestChainedCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradResizeNearestMultipleInputsCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1f), TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSliceCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSliceMultipleElementsCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
    }

    [Fact]
    public void TestAutoGradShapePart2Coverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSliceWithScaleCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSpaceToDepthBasicCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSpaceToDepthWithScaleCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSpaceToDepthMultiChannelCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradDepthToSpaceDCRCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradDepthToSpaceCRDCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradDepthToSpaceWithScaleCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradTile1DCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradTileWithScaleCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradTile2DCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 4f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradTriluUpperCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradTriluLowerCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradTriluUpperWithKCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradTriluWithScaleCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradUpsampleNearestSumLossCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
    }

    [Fact]
    public void TestAutoGradShapePart3Coverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradUpsampleNearestChainedCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradUpsampleNearestMultipleInputsCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1f), TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradCenterCropPadCropCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradCenterCropPadPadCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradCenterCropPadSameSizeCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradCenterCropPadWithAxesCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradReverseSequenceBasicCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradReverseSequencePartialReverseCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradReverseSequenceAllSameCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradReverseSequenceBatchAxis1Check>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradCol2ImBasicNoOverlapCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradCol2ImWithOverlapCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradCol2ImWithPaddingCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradCol2Im1x1BlockCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 4f)]));
    }

    [Fact]
    public void TestAutoGradSequenceOpsCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSequenceConstructAtExtractFirstCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f), TensorData(DType.Float32, [], 7f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSequenceConstructAtExtractSecondCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f), TensorData(DType.Float32, [], 7f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSequenceConstructAtWithScaleCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 5f), TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradConcatFromSequenceBasicCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f), TensorData(DType.Float32, [], 7f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradConcatFromSequenceNewAxisCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f), TensorData(DType.Float32, [], 7f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradConcatFromSequenceWithScaleCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f), TensorData(DType.Float32, [], 7f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSequenceConstructAtThreeInputsCheck>(
            hyperparamInputs: [],
            runtimeInputs: [
                TensorData(DType.Float32, [], 1f),
                TensorData(DType.Float32, [], 2f),
                TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSequenceInsertAtPositionCheck>(
            hyperparamInputs: [],
            runtimeInputs: [
                TensorData(DType.Float32, [], 3f),
                TensorData(DType.Float32, [], 7f),
                TensorData(DType.Float32, [], 5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSequenceInsertAppendNullPositionCheck>(
            hyperparamInputs: [],
            runtimeInputs: [
                TensorData(DType.Float32, [], 3f),
                TensorData(DType.Float32, [], 7f),
                TensorData(DType.Float32, [], 5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSequenceEraseElementCheck>(
            hyperparamInputs: [],
            runtimeInputs: [
                TensorData(DType.Float32, [], 3f),
                TensorData(DType.Float32, [], 7f),
                TensorData(DType.Float32, [], 5f)]));
    }

    [Fact]
    public void TestAutoGradCompressCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradCompressNoAxisCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradCompressWithAxisCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradCompressWithAxisZeroCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 5f)]));
    }

    [Fact]
    public void TestAutoGradDftRoundtripCoverage() =>
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradDftRoundtripCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));

    [Fact]
    public void TestAutoGradConstantOfShapeCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradConstantOfShapeNullShapeGradientCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
    }

    [Fact]
    public void TestAutoGradDftDefaultAxisCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradDftDefaultAxisCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGridSampleMultiChannelCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
    }

    [Fact]
    public void TestAutoGradTopKCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradTopK1DLargestK1Check>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 5.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradTopK1DLargestK2Check>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 5.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradTopKNotSelectedCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradTopK2DAxis1Check>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 7.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradTopKSmallestK1Check>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.5f)]));
    }

    [Fact]
    public void TestAutoGradMaxRoiPoolCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradMaxRoiPoolPositiveCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradMaxRoiPoolGradientShapeCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradMaxRoiPoolMultiChannelCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradMaxRoiPoolMultipleRoisCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradMaxRoiPoolForwardSumCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
    }

    [Fact]
    public void TestAutoGradRoiAlignCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradRoiAlignPositiveCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradRoiAlignSpatialScaleCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradRoiAlignMultiChannelCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradRoiAlignMultipleRoisCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradRoiAlignOutputHalfPixelCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradRoiAlignForwardSumCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 1f)]));
    }

    [Fact]
    public void TestAutoGradGRUPart1Coverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGruXSeqLen1Check>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGruXSeqLen2Check>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGruXSeqLen3Check>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGruWCheck>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGruRCheck>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.2f)]));
    }

    [Fact]
    public void TestAutoGradGRUPart2Coverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGruBCheck>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.05f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGruH0Check>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGruLinearBeforeResetCheck>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGruFullSequenceOutputCheck>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.3f)]));
    }

    [Fact]
    public void TestAutoGradRNNCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradRnnXSeqLen1Check>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradRnnXSeqLen2Check>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradRnnXSeqLen3Check>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradRnnWCheck>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradRnnRCheck>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradRnnBCheck>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.05f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradRnnH0Check>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradRnnFullSequenceOutputCheck>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.3f)]));
    }

    [Fact]
    public void TestAutoGradLSTMPart1Coverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradLstmXSeqLen1Check>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradLstmXSeqLen2Check>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradLstmXSeqLen3Check>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.3f)]));
    }

    [Fact]
    public void TestAutoGradLSTMPart2Coverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradLstmWCheck>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradLstmRCheck>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradLstmBCheck>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.05f)]));
    }

    [Fact]
    public void TestAutoGradLSTMPart3Coverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradLstmH0Check>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradLstmC0Check>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.15f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradLstmFullSequenceOutputCheck>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.3f)]));
    }

    [Fact]
    public void TestAutoGradLrnCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradLrnBasicCheck>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 2.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradLrnNumericalCheck>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 2.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradLrnSmallAlphaCheck>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 3.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradLrnHighBetaCheck>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 1.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradLrnWithScaleCheck>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 2.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradLrnMultiChannelCh1Check>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 1.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradLrnMultiChannelCh2Check>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 2.0f)]));
    }

    [Fact]
    public void TestAutoGradAlignCornersFalseCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradAffineGridAlignCornersFalseCheck>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGridSampleAlignCornersFalseCheck>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 1.0f)]));
    }

    [Fact]
    public void TestAutoGradCoverageViaRuntimeInputCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradCastRoundTripCheck>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 2.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradIfRuntimeConditionTrueCheck>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 2.0f), TensorData(DType.Float32, [], 3.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradIfRuntimeConditionFalseCheck>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], -1.0f), TensorData(DType.Float32, [], 3.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradDftWithDftLengthCheck>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 3.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradConstantOfShapeRuntimeShapeCheck>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 2.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSeqAtRuntimeIdxCheck>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 3.0f), TensorData(DType.Float32, [], 7.0f), TensorData(DType.Float32, [], 0.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSeqInsertEraseRuntimeIdxCheck>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 3.0f), TensorData(DType.Float32, [], 7.0f), TensorData(DType.Float32, [], 5.0f), TensorData(DType.Float32, [], 0.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSeqInsertAppendCheck>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 3.0f), TensorData(DType.Float32, [], 7.0f), TensorData(DType.Float32, [], 1.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradIfMultiOutputRuntimeCondPartiallyUsedCheck>(hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 2.0f), TensorData(DType.Float32, [], 3.0f)]));
    }

    // ===================================================================
    //  Coverage for the AutoDiffs.Batch28 non-differentiable stubs. Each
    //  Module routes a runtime input through a stub op so the analytical
    //  gradient is zero — the [AutoDiff] entries make the autograd engine
    //  treat these ops as the correct "gradient is zero" rather than
    //  throwing NotImplementedException when an imported graph hits one.
    //  See AutoGradStubTestModules.cs for the module bodies and rationale
    //  on which Batch28 entries aren't reachable from a [Module] graph.
    // ===================================================================

    [Fact]
    public void TestAutoGradIntegerAndBooleanStubsCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradBitwiseStubCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [1L], 5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradBitShiftStubCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [1L], 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradBooleanStubCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [3L], 1f, -2f, 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradComparisonStubCheck>(
            hyperparamInputs: [], runtimeInputs: [
                TensorData(DType.Float32, [3L], 2f, 3f, 4f),
                TensorData(DType.Float32, [3L], 3f, 2f, 4f)]));
    }

    [Fact]
    public void TestAutoGradIndexAndShapeStubsCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradEyeLikeStubCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [3L, 3L],
                1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradShapeStubCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [2L, 3L],
                1f, 2f, 3f, 4f, 5f, 6f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradArgMaxArgMinStubCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [3L], 1f, 3f, 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradNonZeroStubCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [3L], 0f, 1f, 0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradRangeStubCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0f)]));
    }

    [Fact]
    public void TestAutoGradStochasticStubsCoverage()
    {
        // Random ops are non-reproducible across ORT vs. CS roundtrip, so we skip those two
        // round-trips and only run the original ORT forward + the gradient lowering.
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradBernoulliStubCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [3L], 0.1f, 0.5f, 0.9f)],
            testOnnxRoundtrip: false, testCsRoundtrip: false));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradRandomLikeStubCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [3L], 0f, 0f, 0f)],
            testOnnxRoundtrip: false, testCsRoundtrip: false));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradBlackmanWindowStubCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [3L], 1f, 2f, 3f)]));
        // Batch30 additions: HammingWindow + HannWindow share the same length-driven pattern.
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradWindowsStubCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [3L], 1f, 2f, 3f)]));
    }

    [Fact]
    public void TestAutoGradQuantizationStubsCoverage()
    {
        // Quantization-family ops register null gradients (Batch30 dispatcher entries).
        // Each module routes a float input through the quantization op via a multiply-by-zero
        // mask so the analytical gradient on the unmasked Σ(x) path stays at 1.
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradQuantizeLinearStubCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [3L], 1f, 2f, 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradDynamicQuantizeLinearStubCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [3L], 1f, 2f, 3f)]));
    }

    [Fact]
    public void TestAutoGradSequenceAndOptionalStubsCoverage()
    {
        // Sequence/optional non-float-output ops: SequenceLength + OptionalHasElement.
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSequenceLengthStubCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [3L], 1f, 2f, 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradOptionalHasElementStubCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [3L], 1f, 2f, 3f)]));
    }

    [Fact]
    public void TestAutoGradDetectionStubsCoverage()
    {
        // NonMaxSuppression: index output through float boxes/scores.
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradNonMaxSuppressionStubCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [4L], 0.1f, 0.2f, 0.3f, 0.4f)]));
        // DeformConv: AD-B3 replaced the silent ZERO-STUB (null grads → frozen params)
        // with an AD003 guard, and the guard fires even for the zero-valued gradient
        // this module routes through its masked (×0) side branch — no silent zeros.
        var ex = Assert.Throws<AutoDiffNotSupportedException>(() =>
            AutoTest.AdvancedTestGraph<AutoGradDeformConvStubCheck>(
                hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [16L],
                    1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f)]));
        Assert.Equal(ErrorCodes.AD003, ex.ErrorCode);
        Assert.Contains("DeformConv", ex.Message);
    }

    [Fact]
    public void TestAutoGradLossesPart1Coverage()
    {
        // Batch30 variadic loss gradients — Part 1 covers the default ("sum"-reduction)
        // path for NLLLoss + SoftmaxCrossEntropyLoss plus the log_prob-grad branch of SCEL.
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradNegativeLogLikelihoodLossCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 1.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSoftmaxCrossEntropyLossCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSoftmaxCrossEntropyLossLogProbCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        // Only log_prob is consumed — exercises the lossGrad-null branch.
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSoftmaxCrossEntropyLossOnlyLogProbCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
    }

    [Fact]
    public void TestAutoGradLossesPart2Coverage()
    {
        // Part 2 sweeps the remaining reduction modes ("mean", "none"), per-class weight
        // input, and ignore_index masking for both NLLLoss and SCEL so the per-branch
        // gradient code in Batch30 is fully exercised.
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradNegativeLogLikelihoodLossMeanCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 1.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradNegativeLogLikelihoodLossNoneCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 1.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradNegativeLogLikelihoodLossWeightCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 1.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradNegativeLogLikelihoodLossIgnoreIndexCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 1.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSoftmaxCrossEntropyLossMeanCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSoftmaxCrossEntropyLossNoneCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSoftmaxCrossEntropyLossWeightIgnoreCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 2f)]));
    }

    [Fact]
    public void TestAutoGradSplitToSequenceCoverage()
    {
        // Batch30 SplitToSequence variadic gradient: split-then-recombine identity.
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSplitToSequenceCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.5f)]));
    }

    [Fact]
    public void TestAutoGradSignalStubsCoverage()
    {
        // STFT: signal-processing forward op with a stubbed gradient.
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradSTFTStubCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [4L], 1f, 2f, 3f, 4f)]));
    }

}

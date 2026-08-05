namespace Shorokoo.Tests;

/// <summary>
/// Each row drives <see cref="AutoTest.AdvancedTestGraph{TModule}"/> against a module from
/// <c>Modules/AutoGradTestModules.cs</c> / <c>Modules/AutoGradStubTestModules.cs</c> whose
/// <c>Inline</c> embeds the AutoGrad operator under test and verifies its gradient in-graph;
/// the AutoTester adds the ONNX roundtrip, CS roundtrip and QuickExecutionEngine runs.
/// </summary>
[Trait("Domain", "AutoDiff")]
[Trait("Purpose", "Coverage")]
public class AutoGradOpsCoverageTests
{
    private static void Run<TModule>(params float[] scalars) =>
        Assert.True(AutoTest.AdvancedTestGraph<TModule>(
            [], [.. scalars.Select(v => TensorData(DType.Float32, [], v))]));

    private static void RunTensor<TModule>(long[] dims, params float[] vals) =>
        Assert.True(AutoTest.AdvancedTestGraph<TModule>(
            [], [TensorData(DType.Float32, dims, [.. vals.Select(v => (object)v)])]));

    private static void RunSmall<TModule>(long[] shape) =>
        Assert.True(AutoTest.AdvancedTestGraph<TModule>(
            [], [TensorDataWithSmallVals(DType.Float32, shape)]));

    private static void RunSmallNoQee<TModule>(long[] shape) =>
        Assert.True(AutoTest.AdvancedTestGraph<TModule>(
            [], [TensorDataWithSmallVals(DType.Float32, shape)],
            testQuickEngineExecution: false));

    [Fact]
    public void TestAutoGradArithmeticAndTrigonometricGradients()
    {
        Run<AutoGradScalarSquare>(2.5f);
        Run<AutoGradPairProduct>(3f, 4f);
        Run<AutoGradSubBinaryCheck>(5f, 3f);
        Run<AutoGradSubChainedCheck>(5f, 3f);
        Run<AutoGradDivCheck>(6f, 3f);
        Run<AutoGradNegCheck>(5f);
        Run<AutoGradNegChainedCheck>(3f, 4f);
        Run<AutoGradPowCheck>(2f, 3f);
        RunTensor<AutoGradAbsCheck>([2L], 3f, -3f);
        Run<AutoGradReciprocalCheck>(2f);
        Run<AutoGradModSumCheck>(5f, 3f);
        Run<AutoGradModSumCheck>(7f, 4f);
        Run<AutoGradModWithDownstreamCheck>(5f, 3f);
        Run<AutoGradAbsMulChainCheck>(-3f);
        Run<AutoGradPowSqrtIdentityCheck>(3f);
        Run<AutoGradSinCheck>(MathF.PI / 4f);
        Run<AutoGradCosCheck>(MathF.PI / 4f);
        Run<AutoGradTanCheck>(0.5f);
        Run<AutoGradAsinCheck>(0.5f);
        Run<AutoGradAcosCheck>(0.5f);
        Run<AutoGradAtanCheck>(1.0f);
        Run<AutoGradSinhCheck>(1.0f);
        Run<AutoGradCoshCheck>(1.0f);
        Run<AutoGradAsinhCheck>(1.0f);
        Run<AutoGradAcoshCheck>(2.0f);
        Run<AutoGradAtanhCheck>(0.5f);
        RunTensor<AutoGradTanhCheck>([2L], 0f, 1f);
        Run<AutoGradSinCosChainCheck>(MathF.PI / 6f);
        Run<AutoGradSinhCoshChainCheck>(1.0f);
        Run<AutoGradAtanExpChainCheck>(0.5f);
    }

    [Fact]
    public void TestAutoGradActivationGradients()
    {
        RunTensor<AutoGradReluCheck>([2L], 3f, -2f);
        Run<AutoGradReluChainedCheck>(3f);
        RunTensor<AutoGradSigmoidCheck>([2L], 0f, 2f);
        RunTensor<AutoGradLeakyReluCheck>([2L], 3f, -2f);
        RunTensor<AutoGradGeluCheck>([2L], 0f, 1f);
        RunTensor<AutoGradEluCheck>([2L], 2f, -1f);
        RunTensor<AutoGradSeluCheck>([2L], 2f, -1f);
        RunTensor<AutoGradCeluCheck>([2L], 2f, -1f);
        Run<AutoGradSigmoidExpChainCheck>(0f);
        Run<AutoGradLeakyReluExpChainCheck>(0f);
        Run<AutoGradHardSigmoidCheck>(0.5f);
        RunTensor<AutoGradHardSwishCheck>([2L], 1.5f, -1.5f);
        RunTensor<AutoGradMishCheck>([2L], 0.7f, -0.7f);
        RunTensor<AutoGradSoftplusCheck>([2L], 0.5f, -0.5f);
        RunTensor<AutoGradSoftsignCheck>([2L], 0.5f, -0.5f);
        RunTensor<AutoGradThresholdedReluCheck>([2L], 1.0f, 0.0f);
        RunTensor<AutoGradShrinkCheck>([3L], 1.0f, -1.0f, 0.1f);
    }

    [Fact]
    public void TestAutoGradMathClipAndCumSumGradients()
    {
        Run<AutoGradExpCheck>(1f);
        Run<AutoGradExpChainedCheck>(1f);
        Run<AutoGradLogCheck>(3f);
        Run<AutoGradSqrtCheck>(4f);
        Run<AutoGradSqrtDivChainCheck>(9f, 2f);
        Run<AutoGradLogExpIdentityCheck>(2f);
        Run<AutoGradErfCheck>(0.5f);
        Run<AutoGradSignCheck>(3f);
        Run<AutoGradCeilCheck>(3.2f);
        Run<AutoGradFloorCheck>(3.7f);
        Run<AutoGradErfMulChainCheck>(0.5f);
        RunTensor<AutoGradClipCheck>([3L], 5f, -2f, 15f);
        Run<AutoGradCumSumCheck>(2f);
        Run<AutoGradCumSumReverseCheck>(3f);
        Run<AutoGradCumSumWithScaleCheck>(1f);
        Run<AutoGradDet2x2IdentityCheck>(3f);
        Run<AutoGradDet2x2DiagonalCheck>(3f);
        Run<AutoGradDet2x2ChainRuleCheck>(5f);
        Run<AutoGradCastLikeCheck>(5f, 0f);
        Run<AutoGradCastLikeWithScaleCheck>(2f, 0f);
    }

    [Fact]
    public void TestAutoGradReductionOptionalAndCompressGradients()
    {
        Run<AutoGradReduceProdCheck>(3f);
        Run<AutoGradReduceSumSquareCheck>(3f);
        Run<AutoGradReduceLogSumExpCheck>(1f);
        Run<AutoGradReduceL1Check>(3f);
        Run<AutoGradReduceL2Check>(3f);
        Run<AutoGradReduceLogSumCheck>(3f);
        Run<AutoGradReduceMaxCheck>(3f);
        Run<AutoGradReduceMinCheck>(3f);
        Run<AutoGradOptionalWrapUnwrap2x3Check>(3.0f);
        Run<AutoGradOptionalWrapUnwrapScalarCheck>(5.0f);
        Run<AutoGradOptionalWrapUnwrap3x4Check>(2.0f);
        Run<AutoGradOptionalWrapUnwrapWithScaleCheck>(1.5f);
        Run<AutoGradCompressNoAxisCheck>(3f);
        Run<AutoGradCompressWithAxisCheck>(2f);
        Run<AutoGradCompressWithAxisZeroCheck>(5f);
    }

    [Fact]
    public void TestAutoGradConcatSplitAndIfElseGradients()
    {
        Run<AutoGradConcat2Check>(3f, 7f);
        Run<AutoGradConcatWithScaleCheck>(3f, 7f);
        Run<AutoGradConcat3Check>(1f, 2f, 3f);
        Run<AutoGradConcatWithActivationCheck>(3f, -2f);
        Run<AutoGradSplit2OutputsCheck>(3f);
        Run<AutoGradSplit3OutputsCheck>(5f);
        Run<AutoGradSplitWithScaleCheck>(1f);
        Run<AutoGradSplitConcatRoundTripCheck>(2f);
        Run<AutoGradIfTrueConditionCheck>(3f, 5f);
        Run<AutoGradIfFalseConditionCheck>(3f, 5f);
        Run<AutoGradIfSharedInputTrueCheck>(3f);
        Run<AutoGradIfSharedInputFalseCheck>(3f);
        Run<AutoGradIfWithDownstreamOpsCheck>(3f, 5f, 1f);
        Run<AutoGradIfWithUpstreamOpsCheck>(3f, 5f);
        Run<AutoGradIfMultiOutputPartiallyUsedCheck>(3f, 5f);
    }

    [Fact]
    public void TestAutoGradPoolingAndMaxRoiPoolGradients()
    {
        Run<AutoGradGlobalAveragePoolCheck>(3f);
        Run<AutoGradGlobalAveragePoolWithScaleCheck>(5f);
        Run<AutoGradGlobalMaxPoolCheck>(3f);
        Run<AutoGradGlobalAveragePoolExpChainCheck>(1f);
        Run<AutoGradGlobalLpPoolP2Check>(3f);
        Run<AutoGradGlobalLpPoolP1Check>(3f);
        Run<AutoGradGlobalLpPoolP2WithScaleCheck>(5f);
        Run<AutoGradLpPoolP2Check>(3f);
        Run<AutoGradLpPoolP1Check>(3f);
        Run<AutoGradLpPoolP2WithScaleCheck>(5f);
        Run<AutoGradMaxUnpoolCheck>(3f);
        Run<AutoGradMaxUnpoolWithScaleCheck>(5f);
        Run<AutoGradMaxUnpoolChainCheck>(1f);
        Run<AutoGradMaxRoiPoolPositiveCheck>(1f);
        Run<AutoGradMaxRoiPoolGradientShapeCheck>(1f);
        Run<AutoGradMaxRoiPoolMultiChannelCheck>(1f);
        Run<AutoGradMaxRoiPoolMultipleRoisCheck>(1f);
        Run<AutoGradMaxRoiPoolForwardSumCheck>(1f);
    }

    [Fact]
    public void TestAutoGradVariadicAndDropoutGradients()
    {
        Run<AutoGradDropoutInferenceCheck>(2f);
        Run<AutoGradDropoutWithScaleCheck>(5f);
        Run<AutoGradDropoutWithDownstreamCheck>(3f);
        Run<AutoGradSum2InputsCheck>(3f, 5f);
        Run<AutoGradSum3InputsCheck>(2f, 7f, 4f);
        Run<AutoGradSumWithScaleCheck>(2f, 5f);
        Run<AutoGradSumTensorInputsCheck>(3f, 5f);
        Run<AutoGradMean2InputsCheck>(3f, 5f);
        Run<AutoGradMean3InputsCheck>(2f, 7f, 4f);
        Run<AutoGradMeanWithScaleCheck>(2f, 4f);
        Run<AutoGradMeanTensorInputsCheck>(3f, 5f);
        Run<AutoGradMax2InputsCheck>(3f, 5f);
        Run<AutoGradMax3InputsCheck>(2f, 7f, 4f);
        Run<AutoGradMaxWithScaleCheck>(2f, 5f);
        Run<AutoGradMaxTensorInputsCheck>(3f, 5f);
        Run<AutoGradMin2InputsCheck>(3f, 5f);
        Run<AutoGradMin3InputsCheck>(5f, 2f, 4f);
        Run<AutoGradMinWithScaleCheck>(8f, 3f);
        Run<AutoGradMinTensorInputsCheck>(7f, 2f);
    }

    [Fact]
    public void TestAutoGradMatrixAndRoiAlignGradients()
    {
        Run<AutoGradEinsumMatmulBasicCheck>(3.0f);
        Run<AutoGradEinsumTransposeCheck>(5.0f);
        Run<AutoGradEinsumImplicitModeCheck>(2.0f);
        Run<AutoGradEinsumFreeIndexCheck>(2.5f);
        Run<AutoGradGemmBasicCheck>(3f);
        Run<AutoGradGemmWithAlphaCheck>(2f);
        Run<AutoGradGemmWithBetaAndCCheck>(2f);
        Run<AutoGradGemmTransACheck>(3f);
        Run<AutoGradGemmTransBCheck>(2f);
        Run<AutoGradMatMulKnownRankCheck>(2f);
        Run<AutoGradMatMulUnknownRankBatchedCheck>(2f);
        Run<AutoGradReduceSumExplicitAxesKeepdimsTrueCheck>(2f);
        Run<AutoGradReduceSumExplicitAxesKeepdimsFalseCheck>(2f);
        Run<AutoGradReduceMeanExplicitAxesCheck>(2f);
        Run<AutoGradRoiAlignPositiveCheck>(1f);
        Run<AutoGradRoiAlignSpatialScaleCheck>(1f);
        Run<AutoGradRoiAlignMultiChannelCheck>(1f);
        Run<AutoGradRoiAlignMultipleRoisCheck>(1f);
        Run<AutoGradRoiAlignOutputHalfPixelCheck>(1f);
        Run<AutoGradRoiAlignForwardSumCheck>(1f);
    }

    [Fact]
    public void TestAutoGradConvPoolAndTensorInputGradients()
    {
        Run<AutoGradAffineGridMultiBatchCheck>(0.5f);
        Run<AutoGradReshapePassthroughCheck>(3f);
        Run<AutoGradDeadParam>(7f, 2f);
        RunSmall<AutoGradMatMulReduce>([4L, 3L]);
        RunSmall<AutoGradTrainableParam>([4L, 3L]);
        RunSmall<AutoGradSliceWithAxes>([3L, 4L]);
        RunSmall<AutoGradConv>([1L, 3L, 5L, 5L]);
        RunSmall<AutoGradConvWeight>([1L, 3L, 5L, 5L]);
        RunSmall<AutoGradConvTranspose>([1L, 3L, 5L, 5L]);
        RunSmall<AutoGradReduceMeanAllAxes>([3L, 4L]);
        RunSmall<AutoGradSoftmax>([2L, 4L]);
        RunSmall<AutoGradTransposePerm>([2L, 3L, 4L]);
        RunSmall<AutoGradPadAxes>([3L, 4L]);
        RunSmall<AutoGradTile>([2L, 3L]);
        RunSmall<AutoGradAvgPool>([1L, 2L, 4L, 4L]);
        RunSmallNoQee<AutoGradAvgPoolOverlap>([1L, 2L, 5L, 5L]);
        RunSmallNoQee<AutoGradAvgPoolPadInclude>([1L, 2L, 4L, 4L]);
        RunSmallNoQee<AutoGradAvgPoolPadExclude>([1L, 2L, 4L, 4L]);
        RunSmallNoQee<AutoGradAvgPoolSameUpper>([1L, 2L, 5L, 5L]);
        RunSmallNoQee<AutoGradAvgPoolSameLower>([1L, 2L, 5L, 5L]);
        RunSmall<AutoGradMaxPool>([1L, 2L, 4L, 4L]);
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradGemmTrans>(
            [], [TensorDataWithSmallVals(DType.Float32, [3L, 2L]),
                 TensorDataWithSmallVals(DType.Float32, [4L, 3L])]));
    }

    [Fact]
    public void TestAutoGradNormalizationGradients()
    {
        Run<AutoGradBatchNormSimpleCheck>(3f);
        Run<AutoGradBatchNormWithScaleCheck>(2f);
        Run<AutoGradBatchNorm3DCheck>(1f);
        Run<AutoGradBatchNormExpChainCheck>(1f);
        Run<AutoGradGroupNormBasicCheck>(3f);
        Run<AutoGradGroupNormScaleCheck>(1f);
        Run<AutoGradGroupNormNonConstInputCheck>(3f);
        Run<AutoGradGroupNorm2GroupsCheck>(2f);
        Run<AutoGradInstanceNormBasicCheck>(2f);
        Run<AutoGradInstanceNormWithScaleCheck>(3f);
        Run<AutoGradLpNormL2BasicCheck>(3f);
        Run<AutoGradLpNormL2AsymmetricCheck>(3f);
        Run<AutoGradLpNormL1BasicCheck>(3f);
        Run<AutoGradLayerNormalizationCheck>(2f);
        Run<AutoGradMeanVarianceNormalizationCheck>(2f);
        Run<AutoGradLogSoftmaxCheck>(1f);
        Run<AutoGradPReluCheck>(1.5f);
    }

    [Fact]
    public void TestAutoGradGatherScatterAndTopKGradients()
    {
        Run<AutoGradGatherElementsAxis0Check>(2f);
        Run<AutoGradGatherElementsAxis1Check>(1f);
        Run<AutoGradGatherElementsWithScaleCheck>(3f);
        Run<AutoGradScatterElementsAddCheck>(3f);
        Run<AutoGradScatterElementsNoneCheck>(2f);
        Run<AutoGradScatterElementsWithScaleCheck>(1f);
        Run<AutoGradScatterNDAddCheck>(3f);
        Run<AutoGradScatterNDNoneCheck>(2f);
        Run<AutoGradScatterNDWithScaleCheck>(1f);
        Run<AutoGradScatterNDReluChainCheck>(2f);
        Run<AutoGradGatherAxis0Check>(4f);
        Run<AutoGradGatherDuplicateIndicesCheck>(4f);
        Run<AutoGradGatherAllIndicesCheck>(2f);
        Run<AutoGradTopK1DLargestK1Check>(5.0f);
        Run<AutoGradTopK1DLargestK2Check>(5.0f);
        Run<AutoGradTopKNotSelectedCheck>(0.5f);
        Run<AutoGradTopK2DAxis1Check>(7.0f);
        Run<AutoGradTopKSmallestK1Check>(0.5f);
    }

    [Fact]
    public void TestAutoGradGatherNDWhereAndUniqueGradients()
    {
        Run<AutoGradGatherAxis0MultiDimIndicesCheck>(2f);
        Run<AutoGradGatherNonZeroAxisOneDimIndicesCheck>(3f);
        Run<AutoGradGatherNonZeroAxisOneDimIndicesUnknownRankCheck>(3f);
        Run<AutoGradGatherNonZeroAxisMultiDimIndicesCheck>(4f);
        Run<AutoGradGatherNDCheck>(4f);
        Run<AutoGradGatherNDDuplicateIndicesCheck>(2f);
        Run<AutoGradGatherNDWithScaleCheck>(3f);
        Run<AutoGradWhereTrueBranchCheck>(3f, 7f);
        Run<AutoGradWhereFalseBranchCheck>(3f, 7f);
        Run<AutoGradUniqueSingleElementCheck>(3f);
        Run<AutoGradUniqueAllSameCheck>(2f);
        Run<AutoGradUniqueWithAxisCheck>(1f);
        Run<AutoGradUniqueDistinctCheck>(2f);
    }

    [Fact]
    public void TestAutoGradReshapePadResizeSliceTileAndTriluGradients()
    {
        Run<AutoGradTransposeCheck>(4f);
        Run<AutoGradFlattenCheck>(3f);
        Run<AutoGradSqueezeCheck>(4f);
        Run<AutoGradUnsqueezeCheck>(5f);
        Run<AutoGradExpandCheck>(3f);
        Run<AutoGradPadConstantCheck>(5f);
        Run<AutoGradPadConstantWithMultiplyCheck>(5f);
        Run<AutoGradPad2DCheck>(7f);
        Run<AutoGradPadWithSigmoidCheck>(0f);
        Run<AutoGradResizeNearestSumLossWithScalesCheck>(1f);
        Run<AutoGradResizeNearestSizesCheck>(2f);
        Run<AutoGradResizeNearestChainedCheck>(1f);
        Run<AutoGradResizeNearestMultipleInputsCheck>(1f, 2f);
        Run<AutoGradSliceCheck>(5f);
        Run<AutoGradSliceMultipleElementsCheck>(3f);
        Run<AutoGradSliceWithScaleCheck>(2f);
        Run<AutoGradSpaceToDepthBasicCheck>(3f);
        Run<AutoGradSpaceToDepthWithScaleCheck>(2f);
        Run<AutoGradSpaceToDepthMultiChannelCheck>(1f);
        Run<AutoGradDepthToSpaceDCRCheck>(3f);
        Run<AutoGradDepthToSpaceCRDCheck>(2f);
        Run<AutoGradDepthToSpaceWithScaleCheck>(1f);
        Run<AutoGradTile1DCheck>(5f);
        Run<AutoGradTileWithScaleCheck>(2f);
        Run<AutoGradTile2DCheck>(4f);
        Run<AutoGradTriluUpperCheck>(5f);
        Run<AutoGradTriluLowerCheck>(5f);
        Run<AutoGradTriluUpperWithKCheck>(2f);
        Run<AutoGradTriluWithScaleCheck>(3f);
        Run<AutoGradUpsampleNearestSumLossCheck>(1f);
    }

    [Fact]
    public void TestAutoGradUpsampleCropReverseAndCol2ImGradients()
    {
        Run<AutoGradUpsampleNearestChainedCheck>(1f);
        Run<AutoGradUpsampleNearestMultipleInputsCheck>(1f, 2f);
        Run<AutoGradCenterCropPadCropCheck>(3f);
        Run<AutoGradCenterCropPadPadCheck>(2f);
        Run<AutoGradCenterCropPadSameSizeCheck>(5f);
        Run<AutoGradCenterCropPadWithAxesCheck>(1f);
        Run<AutoGradReverseSequenceBasicCheck>(3f);
        Run<AutoGradReverseSequencePartialReverseCheck>(2f);
        Run<AutoGradReverseSequenceAllSameCheck>(5f);
        Run<AutoGradReverseSequenceBatchAxis1Check>(1f);
        Run<AutoGradCol2ImBasicNoOverlapCheck>(2f);
        Run<AutoGradCol2ImWithOverlapCheck>(1.5f);
        Run<AutoGradCol2ImWithPaddingCheck>(1f);
        Run<AutoGradCol2Im1x1BlockCheck>(4f);
    }

    [Fact]
    public void TestAutoGradSequenceOpsDftAndConstantOfShapeGradients()
    {
        Run<AutoGradSequenceConstructAtExtractFirstCheck>(3f, 7f);
        Run<AutoGradSequenceConstructAtExtractSecondCheck>(3f, 7f);
        Run<AutoGradSequenceConstructAtWithScaleCheck>(5f, 2f);
        Run<AutoGradConcatFromSequenceBasicCheck>(3f, 7f);
        Run<AutoGradConcatFromSequenceNewAxisCheck>(3f, 7f);
        Run<AutoGradConcatFromSequenceWithScaleCheck>(3f, 7f);
        Run<AutoGradSequenceConstructAtThreeInputsCheck>(1f, 2f, 3f);
        Run<AutoGradSequenceInsertAtPositionCheck>(3f, 7f, 5f);
        Run<AutoGradSequenceInsertAppendNullPositionCheck>(3f, 7f, 5f);
        Run<AutoGradSequenceEraseElementCheck>(3f, 7f, 5f);
        Run<AutoGradSplitToSequenceCheck>(0.5f);
        Run<AutoGradDftRoundtripCheck>(2f);
        Run<AutoGradDftDefaultAxisCheck>(2f);
        Run<AutoGradGridSampleMultiChannelCheck>(1f);
        Run<AutoGradConstantOfShapeNullShapeGradientCheck>(2f);
    }

    [Fact]
    public void TestAutoGradGruGradients()
    {
        Run<AutoGradGruXSeqLen1Check>(0.3f);
        Run<AutoGradGruXSeqLen2Check>(0.3f);
        Run<AutoGradGruXSeqLen3Check>(0.3f);
        Run<AutoGradGruWCheck>(0.1f);
        Run<AutoGradGruRCheck>(0.2f);
        Run<AutoGradGruBCheck>(0.05f);
        Run<AutoGradGruH0Check>(0.2f);
        Run<AutoGradGruLinearBeforeResetCheck>(0.3f);
        Run<AutoGradGruFullSequenceOutputCheck>(0.3f);
    }

    [Fact]
    public void TestAutoGradLstmGradients()
    {
        Run<AutoGradLstmXSeqLen1Check>(0.3f);
        Run<AutoGradLstmXSeqLen2Check>(0.3f);
        Run<AutoGradLstmXSeqLen3Check>(0.3f);
        Run<AutoGradLstmWCheck>(0.1f);
        Run<AutoGradLstmRCheck>(0.2f);
        Run<AutoGradLstmBCheck>(0.05f);
        Run<AutoGradLstmH0Check>(0.2f);
        Run<AutoGradLstmC0Check>(0.15f);
        Run<AutoGradLstmFullSequenceOutputCheck>(0.3f);
    }

    [Fact]
    public void TestAutoGradRnnLrnAndAlignCornersGradients()
    {
        Run<AutoGradRnnXSeqLen1Check>(0.3f);
        Run<AutoGradRnnXSeqLen2Check>(0.3f);
        Run<AutoGradRnnXSeqLen3Check>(0.3f);
        Run<AutoGradRnnWCheck>(0.1f);
        Run<AutoGradRnnRCheck>(0.2f);
        Run<AutoGradRnnBCheck>(0.05f);
        Run<AutoGradRnnH0Check>(0.2f);
        Run<AutoGradRnnFullSequenceOutputCheck>(0.3f);
        Run<AutoGradLrnBasicCheck>(2.0f);
        Run<AutoGradLrnNumericalCheck>(2.0f);
        Run<AutoGradLrnSmallAlphaCheck>(3.0f);
        Run<AutoGradLrnHighBetaCheck>(1.5f);
        Run<AutoGradLrnWithScaleCheck>(2.0f);
        Run<AutoGradLrnMultiChannelCh1Check>(1.0f);
        Run<AutoGradLrnMultiChannelCh2Check>(2.0f);
        Run<AutoGradAffineGridAlignCornersFalseCheck>(0.5f);
        Run<AutoGradGridSampleAlignCornersFalseCheck>(1.0f);
    }

    [Fact]
    public void TestAutoGradLossGradients()
    {
        Run<AutoGradNegativeLogLikelihoodLossCheck>(1.5f);
        Run<AutoGradNegativeLogLikelihoodLossMeanCheck>(1.5f);
        Run<AutoGradNegativeLogLikelihoodLossNoneCheck>(1.5f);
        Run<AutoGradNegativeLogLikelihoodLossWeightCheck>(1.5f);
        Run<AutoGradNegativeLogLikelihoodLossIgnoreIndexCheck>(1.5f);
        Run<AutoGradSoftmaxCrossEntropyLossCheck>(2f);
        Run<AutoGradSoftmaxCrossEntropyLossLogProbCheck>(2f);
        Run<AutoGradSoftmaxCrossEntropyLossOnlyLogProbCheck>(2f);
        Run<AutoGradSoftmaxCrossEntropyLossMeanCheck>(2f);
        Run<AutoGradSoftmaxCrossEntropyLossNoneCheck>(2f);
        Run<AutoGradSoftmaxCrossEntropyLossWeightIgnoreCheck>(2f);
    }

    [Fact]
    public void TestAutoGradRuntimeInputDrivenGradients()
    {
        Run<AutoGradCastRoundTripCheck>(2.0f);
        Run<AutoGradIfRuntimeConditionTrueCheck>(2.0f, 3.0f);
        Run<AutoGradIfRuntimeConditionFalseCheck>(-1.0f, 3.0f);
        Run<AutoGradDftWithDftLengthCheck>(3.0f);
        Run<AutoGradConstantOfShapeRuntimeShapeCheck>(2.0f);
        Run<AutoGradSeqAtRuntimeIdxCheck>(3.0f, 7.0f, 0.0f);
        Run<AutoGradSeqInsertEraseRuntimeIdxCheck>(3.0f, 7.0f, 5.0f, 0.0f);
        Run<AutoGradSeqInsertAppendCheck>(3.0f, 7.0f, 1.0f);
        Run<AutoGradIfMultiOutputRuntimeCondPartiallyUsedCheck>(2.0f, 3.0f);
    }

    [Fact]
    public void TestAutoGradNonDifferentiableAndStochasticStubGradients()
    {
        RunTensor<AutoGradBitwiseStubCheck>([1L], 5f);
        RunTensor<AutoGradBitShiftStubCheck>([1L], 3f);
        RunTensor<AutoGradBooleanStubCheck>([3L], 1f, -2f, 3f);
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradComparisonStubCheck>(
            [], [TensorData(DType.Float32, [3L], 2f, 3f, 4f),
                 TensorData(DType.Float32, [3L], 3f, 2f, 4f)]));
        RunTensor<AutoGradEyeLikeStubCheck>([3L, 3L], 1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f);
        RunTensor<AutoGradShapeStubCheck>([2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f);
        RunTensor<AutoGradArgMaxArgMinStubCheck>([3L], 1f, 3f, 2f);
        RunTensor<AutoGradNonZeroStubCheck>([3L], 0f, 1f, 0f);
        Run<AutoGradRangeStubCheck>(0f);
        RunTensor<AutoGradBlackmanWindowStubCheck>([3L], 1f, 2f, 3f);
        RunTensor<AutoGradWindowsStubCheck>([3L], 1f, 2f, 3f);
        RunTensor<AutoGradQuantizeLinearStubCheck>([3L], 1f, 2f, 3f);
        RunTensor<AutoGradDynamicQuantizeLinearStubCheck>([3L], 1f, 2f, 3f);
        RunTensor<AutoGradSequenceLengthStubCheck>([3L], 1f, 2f, 3f);
        RunTensor<AutoGradOptionalHasElementStubCheck>([3L], 1f, 2f, 3f);
        RunTensor<AutoGradNonMaxSuppressionStubCheck>([4L], 0.1f, 0.2f, 0.3f, 0.4f);
        RunTensor<AutoGradSTFTStubCheck>([4L], 1f, 2f, 3f, 4f);
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradBernoulliStubCheck>(
            [], [TensorData(DType.Float32, [3L], 0.1f, 0.5f, 0.9f)],
            testOnnxRoundtrip: false, testCsRoundtrip: false));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradRandomLikeStubCheck>(
            [], [TensorData(DType.Float32, [3L], 0f, 0f, 0f)],
            testOnnxRoundtrip: false, testCsRoundtrip: false));
    }
}

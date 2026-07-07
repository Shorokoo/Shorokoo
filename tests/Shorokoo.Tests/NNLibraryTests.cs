using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Runtime;
using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage tests for the baseline NN library (Shorokoo.Modules Initializers /
/// Layers / Losses), in the <see cref="Shorokoo.Tests.Modules.CoverageTests"/>
/// one-liner style: each [Fact] drives self-checking modules from
/// NNLibraryTestModules.cs through <see cref="AutoTest.AdvancedTestGraph{TModule}"/>
/// (ONNX roundtrip, CS codegen, QEE). Value correctness lives inside the
/// modules (they return Scalar&lt;bit&gt;); see the module docs for the exact
/// closed forms checked. BatchNorm2d is covered by the rig-based
/// <see cref="NNLibraryTrainingCoverageTests"/> instead (its StateUpdate links
/// are not executable in the plain inference pipeline).
/// </summary>
[Trait("Domain", "Modules")]
[Trait("Purpose", "Coverage")]
public class NNLibraryCoverageTests
{
    /// <summary>[i * scale + offset for i in 0..N) as a float32 TensorData — varied, non-degenerate values for the norm/conv checks.</summary>
    private static TensorData RangeTensor(long[] dims, float scale = 1f, float offset = 0f)
    {
        long total = 1;
        foreach (var d in dims) total *= d;
        return TensorData(DType.Float32, dims,
            Enumerable.Range(0, (int)total).Select(i => (object)(i * scale + offset)).ToArray());
    }

    /// <summary>[i * scale + offset + curv * i² for i in 0..N) as a float32 TensorData — a QUADRATIC
    /// ramp. A linear <see cref="RangeTensor"/> is degenerate for a frozen norm reference: every
    /// per-(sample, channel)/group slice is a contiguous arithmetic run differing only by an offset,
    /// and normalization's mean-subtraction annihilates that offset, so all slices standardize to the
    /// IDENTICAL pattern — the output (and its collapsed golden) is then invariant under an internal
    /// N/C transpose, which the reference can't catch. The i² term gives each slice a position-dependent
    /// slope (the cross term 2·curv·base·k survives mean-subtraction), so slices standardize distinctly
    /// and a transpose moves the output. Deterministic/portable (integer i², one float multiply).</summary>
    private static TensorData CurvedTensor(long[] dims, float scale, float offset, float curv)
    {
        long total = 1;
        foreach (var d in dims) total *= d;
        return TensorData(DType.Float32, dims,
            Enumerable.Range(0, (int)total).Select(i => (object)(i * scale + offset + curv * i * i)).ToArray());
    }

    [Fact]
    public void TestLinearAndConvLayersCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<NNLinearMatchesManualMatMul>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L], 0.5f, -1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNConv2dMatchesStaticConv>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 5L, 5L], 0.1f, -2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNConv1dMatchesStaticConv>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 7L], 0.25f, -1.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNConvTranspose2dMatchesStatic>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 3L, 3L], 0.3f, -2f)]));
    }

    /// <summary>
    /// Bilinear (src/Shorokoo.Modules/Layers/Bilinear.cs, PyTorch nn.Bilinear) forward-value
    /// coverage, design §7.1–7.3. Each self-checking [Module] (two graph inputs x1, x2) compares
    /// Bilinear.Call(...) against an INDEPENDENT manual bilinear form — a broadcast Mul + ReduceSum
    /// Σ_{i,j} x1·A·x2 (NOT a second einsum, so a bug in the "ni,kij,nj->nk" equation cannot hide
    /// behind itself), re-materializing the SAME seeded RecurrentUniform weight A [out,in1,in2] and
    /// bias b [out] (both U(±1/√in1)) so the random draws cancel exactly, the NNLinearMatchesManualMatMul
    /// idiom. Covers §7.1 closed form (useBias:true), §7.2 useBias gating (false == no-bias reference,
    /// and true − false == exactly b — guarding the IfElse branch; the bias being seeded-uniform, NOT
    /// Zeros, is load-bearing), and §7.3 batch broadcasting ([B,T,in_k] → output shape [B,T,out]
    /// asserted == [2,2,2], plus the per-row reference). The rig train-step (§7.4, the first
    /// Einsum-autodiff module-level check that the rank-3 A moves) is in
    /// <see cref="NNLibraryTrainingCoverageTests.TestBilinearTrainStepMovesWeight"/>.
    /// </summary>
    [Fact]
    public void TestBilinearForwardValueCoverage()
    {
        // §7.1 closed form (useBias:true): x1 [2,3], x2 [2,4].
        Assert.True(AutoTest.AdvancedTestGraph<NNBilinearMatchesManualForm>(
            hyperparamInputs: [],
            runtimeInputs: [RangeTensor([2L, 3L], 0.5f, -1f), RangeTensor([2L, 4L], 0.3f, -0.5f)]));

        // §7.2 useBias gating (false == no-bias ref; true − false == bias b).
        Assert.True(AutoTest.AdvancedTestGraph<NNBilinearUseBiasFalseAndDiff>(
            hyperparamInputs: [],
            runtimeInputs: [RangeTensor([2L, 3L], 0.5f, -1f), RangeTensor([2L, 4L], 0.3f, -0.5f)]));

        // §7.3 batch broadcasting: x1 [2,2,3], x2 [2,2,4] → [2,2,2].
        Assert.True(AutoTest.AdvancedTestGraph<NNBilinearBatchBroadcasts>(
            hyperparamInputs: [],
            runtimeInputs: [RangeTensor([2L, 2L, 3L], 0.5f, -1f), RangeTensor([2L, 2L, 4L], 0.3f, -0.5f)]));
    }

    /// <summary>
    /// Generalized Convolution helper (src/Shorokoo.Modules/Layers/Convolution.cs) forward
    /// coverage, design §7 groups 1–9 (the differentiable, QEE-safe Zeros-mode checks):
    /// non-square kernel (§7-1), per-axis stride/dilation (§7-2), asymmetric pad (§7-3),
    /// auto_pad SAME_UPPER/VALID (§7-4), groups/depthwise (§7-5), ConvTranspose output_padding/
    /// output_shape and the 1d/3d rank aliases (§7-7), alias + scalar-overload equivalence
    /// (§7-8), and bias on/off (§7-9). Each module compares its (collapsed) forward output
    /// against an inlined frozen golden reference (self-generated at master-seed-0 init) on the
    /// ORT backend. (The non-differentiable padding_mode checks live in
    /// TestConvPaddingModesCoverage; the trainability smoke in NNLibraryTrainingCoverageTests.)
    /// </summary>
    [Fact]
    public void TestGeneralizedConvLayersCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<ConvNonSquareKernelMatchesStatic>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 7L, 9L], 0.05f, -2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<ConvPerAxisStrideDilationMatchesStatic>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 7L, 7L], 0.05f, -1.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<ConvAsymmetricPadMatchesStatic>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 5L, 5L], 0.1f, -2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<ConvAutoPadMatchesStatic>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 6L, 6L], 0.07f, -1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<ConvGroupsMatchesStatic>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 4L, 5L, 5L], 0.05f, -2.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<ConvBiasOnOffMatchesZeroBias>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 5L, 5L], 0.1f, -2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<ConvAliasAndScalarEquivalence>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 5L, 5L], 0.1f, -2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<ConvScalarBroadcastMatchesPerAxis>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 5L, 5L], 0.1f, -2f)]));

        // ConvTranspose §7-7: output_padding, output_shape, and the 1d/3d rank aliases.
        Assert.True(AutoTest.AdvancedTestGraph<ConvTransposeOutputPaddingMatchesStatic>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 3L, 3L], 0.3f, -2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<ConvTransposeOutputShapeMatchesStatic>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 3L, 3L], 0.3f, -2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<ConvTranspose1dMatchesStatic>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 4L], 0.25f, -1.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<ConvTranspose3dMatchesStatic>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 2L, 2L, 2L], 0.2f, -1f)]));
    }

    /// <summary>
    /// Generalized Convolution padding_mode coverage (design §7-6): Reflect / Replicate /
    /// Circular each match a hand-built x.Pad(&lt;PadMode&gt;, pads, axes:spatial) + NN.Conv(pads:0)
    /// reference, and Causal (1D) matches a left-Pad((k-1)*dilation) + VALID conv. Forward only:
    /// the reflect/edge/wrap Pad these compose is non-differentiable and has NO QEE values, which
    /// breaks the QEE DType-resolution and the CS roundtrip, so both are disabled for these
    /// checks (the ORT/ONNX forward value is still validated). Causal's Pad is constant-mode and
    /// would survive QEE, but it is grouped here and run with the same flags for simplicity.
    /// </summary>
    [Fact]
    public void TestConvPaddingModesCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<ConvPaddingModeMatchesHandPad>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 5L, 5L], 0.1f, -2f)],
            testCsRoundtrip: false, testQuickEngineExecution: false));
        Assert.True(AutoTest.AdvancedTestGraph<ConvCausalMatchesLeftPadValid>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 7L], 0.25f, -1.5f)],
            testCsRoundtrip: false, testQuickEngineExecution: false));
    }

    [Fact]
    public void TestNormalizationLayersCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<NNLayerNormNormalizes>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 4L], 1.5f, -3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNGroupNormNormalizes>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 4L, 3L, 3L], 0.7f, -10f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNInstanceNorm2dNormalizes>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 4L, 4L], 0.4f, -3f)]));
    }

    /// <summary>
    /// Generalized rank-generic InstanceNorm + GroupNorm value/equivalence coverage
    /// (design §7 modes 1, 2a, 3, 4, 5). All state-free, so driven through
    /// AutoTest.AdvancedTestGraph exactly like the existing norm checks:
    /// <list type="bullet">
    ///   <item>§7-1 per-region zero-mean/unit-var for InstanceNorm and GroupNorm(G=2)
    ///         at ranks 3 ([2,3,5]), 4 ([2,3,4,4]) and 5 ([2,3,2,2,2]);</item>
    ///   <item>§7-2a affine:false output == hand-built x̂ reference (InstanceNorm + GroupNorm);</item>
    ///   <item>§7-3 GroupNorm(G=1) ≡ LayerNorm-over-CHW on rank-4;</item>
    ///   <item>§7-4 GroupNorm(G=C) ≡ InstanceNorm on rank-4 and rank-3;</item>
    ///   <item>§7-5 InstanceNorm{1,2,3}d aliases == the generic InstanceNorm (affine off).</item>
    /// </list>
    /// Inputs use the shared RangeTensor non-degenerate scale/offset. Biased variance is
    /// used uniformly, so the G=1 / G=C equivalences are exact up to float error. The
    /// affine param-count discrimination (§7-2b) is a rig check in
    /// <see cref="NNLibraryTrainingCoverageTests.TestInstanceGroupNormAffineOnOff"/>.
    /// </summary>
    [Fact]
    public void TestInstanceGroupNormCoverage()
    {
        // §7-1 InstanceNorm per-region normalization, ranks 3/4/5.
        Assert.True(AutoTest.AdvancedTestGraph<NNInstanceNormRank3Normalizes>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 5L], 0.4f, -3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNInstanceNormRank4Normalizes>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 4L, 4L], 0.4f, -3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNInstanceNormRank5Normalizes>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 2L, 2L, 2L], 0.4f, -3f)]));

        // §7-1 GroupNorm(G=2) per-region normalization, ranks 3/4/5.
        Assert.True(AutoTest.AdvancedTestGraph<NNGroupNormRank3Normalizes>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 4L, 5L], 0.5f, -8f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNGroupNormRank4Normalizes>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 4L, 3L, 3L], 0.7f, -10f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNGroupNormRank5Normalizes>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 4L, 2L, 2L, 2L], 0.5f, -8f)]));

        // §7-2a affine:false == frozen forward reference. Distinct-valued (quadratic) input so the
        // per-slice patterns differ — a linear ramp makes every slice standardize identically, leaving
        // the reference blind to an internal N/C transpose (all slices interchangeable).
        Assert.True(AutoTest.AdvancedTestGraph<NNInstanceNormAffineFalseMatchesManual>(
            hyperparamInputs: [], runtimeInputs: [CurvedTensor([2L, 3L, 4L, 4L], 0.4f, -3f, 0.05f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNGroupNormAffineFalseMatchesManual>(
            hyperparamInputs: [], runtimeInputs: [CurvedTensor([2L, 4L, 3L, 3L], 0.7f, -10f, 0.05f)]));

        // §7-3 GroupNorm(G=1) ≡ LayerNorm-over-CHW (rank-4).
        Assert.True(AutoTest.AdvancedTestGraph<NNGroupNormG1MatchesLayerNorm>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 4L, 4L], 0.4f, -3f)]));

        // §7-4 GroupNorm(G=C) ≡ InstanceNorm (rank-4 and rank-3).
        Assert.True(AutoTest.AdvancedTestGraph<NNGroupNormGCMatchesInstanceNormRank4>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 4L, 4L], 0.4f, -3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNGroupNormGCMatchesInstanceNormRank3>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 5L], 0.4f, -3f)]));

        // §7-5 InstanceNorm{1,2,3}d alias equivalence (bit-for-bit; aliases forward).
        Assert.True(AutoTest.AdvancedTestGraph<NNInstanceNorm1dAliasEquiv>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 5L], 0.4f, -3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNInstanceNorm2dAliasEquiv>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 4L, 4L], 0.4f, -3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNInstanceNorm3dAliasEquiv>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 2L, 2L, 2L], 0.4f, -3f)]));
    }

    [Fact]
    public void TestDropoutEmbeddingActivationCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<NNDropoutChecks>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 8L], 0.5f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNEmbeddingMatchesGather>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Int64, [3L], 0L, 1L, 0L)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNLeakyReLUAndELUClosedForm>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [7L], -3f, -1f, -0.5f, 0f, 0.5f, 1f, 3f)]));
    }

    /// <summary>
    /// Embedding knob coverage (paddingIdx / maxNorm / normType / init choice,
    /// src/Shorokoo.Modules/Layers/Embedding.cs — embedding-knobs design §8 cases
    /// 2–7). Each self-checking module reproduces the masked / clamped lookup by
    /// hand against the seeded Normal / XavierUniform weight and returns a
    /// Scalar&lt;bit&gt; driven through AutoTest.AdvancedTestGraph. (§8-1 baseline
    /// regression — NNEmbeddingMatchesGather, default knobs == plain Gather — runs
    /// in TestDropoutEmbeddingActivationCoverage above; §8-8 train-step is the
    /// rig-based TestEmbeddingPaddingTrainStepMovesWeight.)
    /// </summary>
    [Fact]
    public void TestEmbeddingKnobsCoverage()
    {
        // §8-2/§8-3 paddingIdx zeroes pad rows; off-sentinel -1 is a no-op.
        Assert.True(AutoTest.AdvancedTestGraph<NNEmbeddingPaddingIdxZeros>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Int64, [5L], 0L, 1L, 2L, 2L, 3L)]));
        // §8-4/§8-6 maxNorm L2 shrink-only clamp; big cap / off-sentinel are no-ops.
        Assert.True(AutoTest.AdvancedTestGraph<NNEmbeddingMaxNormClampsL2>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Int64, [2L], 0L, 3L)]));
        // §8-5 normType honored: L1 vs L2 clamp the chosen-p norm and DIFFER.
        Assert.True(AutoTest.AdvancedTestGraph<NNEmbeddingNormTypeL1VsL2>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Int64, [1L], 0L)]));
    }

    /// <summary>§8-7 init choice via the static EmbeddingHelpers.Embed (Xavier; default == Normal).</summary>
    [Fact]
    public void TestEmbeddingInitChoiceCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<NNEmbeddingInitChoice>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Int64, [3L], 0L, 2L, 4L)]));
    }

    /// <summary>
    /// EmbeddingBag (src/Shorokoo.Modules/Layers/Embedding.cs — embedding-bag design §8)
    /// forward-value coverage. EmbeddingBag.Bag(indices [B,L], V, D, mode) is
    /// Embedding(indices).Reduce(mode, axis=1) → [B, D]; each self-checking [Module] compares
    /// Bag's output against an inlined frozen golden reference (self-generated at master-seed-0
    /// init). V=5, D=4, indices [2,3] with distinct ids 0,1,2 / 1,3,0 so
    /// Sum ≠ Mean ≠ Max are all non-trivial:
    /// <list type="bullet">
    ///   <item>§8-1 Sum / Mean / Max — one frozen golden per mode (the load-bearing per-mode
    ///         correctness check);</item>
    ///   <item>§8-2 shape [B,L]=[2,3] → [B,D]=[2,4] (ShapeTensor[0]==2, [1]==4);</item>
    ///   <item>§8-3 paddingIdx:2 zeroes the pad rows for Sum (EXACT): the masked AND unmasked
    ///         Sums both fold into the golden, so an ignored paddingIdx fails;</item>
    ///   <item>§8-4 init choice via the embeddingInit selector (Xavier AND the Normal default,
    ///         both folded into the golden).</item>
    /// </list>
    /// The §8-5 train-step is the rig-based
    /// <see cref="NNLibraryTrainingCoverageTests.TestEmbeddingBagTrainStepMovesWeight"/>.
    /// </summary>
    [Fact]
    public void TestEmbeddingBagCoverage()
    {
        // §8-1 Sum / Mean / Max vs the independent Gather→Reduce reference.
        Assert.True(AutoTest.AdvancedTestGraph<NNEmbeddingBagSumMatchesGatherReduce>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Int64, [2L, 3L], 0L, 1L, 2L, 1L, 3L, 0L)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNEmbeddingBagMeanMatchesGatherReduce>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Int64, [2L, 3L], 0L, 1L, 2L, 1L, 3L, 0L)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNEmbeddingBagMaxMatchesGatherReduce>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Int64, [2L, 3L], 0L, 1L, 2L, 1L, 3L, 0L)]));

        // §8-3 paddingIdx:2 zeroes pad rows for Sum (EXACT) — bags contain the pad id 2.
        Assert.True(AutoTest.AdvancedTestGraph<NNEmbeddingBagPaddingIdxSumExact>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Int64, [2L, 3L], 0L, 2L, 1L, 2L, 3L, 0L)]));

        // §8-4 init choice (Xavier selector; default == Normal).
        Assert.True(AutoTest.AdvancedTestGraph<NNEmbeddingBagInitChoice>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Int64, [2L, 3L], 0L, 1L, 2L, 1L, 3L, 0L)]));
    }

    /// <summary>§8-2 EmbeddingBag output shape [2,3] → [2,4]. Weight-independent (checks only the
    /// reduced-away bag axis and the [batch, embeddingDim] output rank).</summary>
    [Fact]
    public void TestEmbeddingBagShapeCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<NNEmbeddingBagShapeCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Int64, [2L, 3L], 0L, 1L, 2L, 1L, 3L, 0L)]));
    }

    /// <summary>
    /// SpatialDropout (channel-wise dropout, src/Shorokoo.Modules/Layers/Dropout.cs)
    /// TRAIN-MODE behavior — the defining property of the unit. Mask-AGNOSTIC self-checks
    /// (ONNX Dropout's random draw is QEE-value-blocked, computed on the ORT backend inside
    /// AdvancedTestGraph): each element must be 0 or survivor-scale·x, AND the per-channel
    /// mask m=y/x must be CONSTANT across the spatial axes (channel uniformity — what
    /// distinguishes channel-wise from elementwise dropout). x is strictly positive so y/x
    /// is a clean mask. Covers (A) rank-4 channel-wise behavior (0-or-2x + uniformity),
    /// (B) ratio-0.75 survivor scaling (0-or-4x), and (D-train) channel uniformity at rank 3
    /// ([N,C,1] mask) and rank 5 ([N,C,1,1,1] mask).
    ///
    /// REGRESSION GUARD (#440): asserts the corrected training-mode behavior. ORT's
    /// constant-folding used to collapse a training-mode Dropout feeding a downstream op (the
    /// channel-broadcast Mul `x * Dropout(ones,…)`) to the identity all-ones mask, making
    /// training-mode SpatialDropout a silent NO-OP (y == x) that never drops a channel. The fix
    /// (HasTrainingModeDropout disables graph optimizations for such models) is verified here:
    /// the channel-wise mask must genuinely drop. Adds no unique line/branch coverage (the rig +
    /// eval dropout tests already execute these paths) — kept as the only assertion of correct
    /// channel-wise drop, so the bug cannot silently regress.
    /// </summary>
    [Fact]
    public void TestSpatialDropoutTrainModeChannelWise()
    {
        // A. Channel-wise behavior on [N,C,H,W] (train, mask-agnostic): 0-or-2x + channel uniformity.
        Assert.True(AutoTest.AdvancedTestGraph<NNSpatialDropoutChannelWise>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 2L, 2L], 0.5f, 1f)]));

        // B. Survivor scaling 1/(1-0.75)=4 (train, mask-agnostic): 0-or-4x.
        Assert.True(AutoTest.AdvancedTestGraph<NNSpatialDropoutSurvivorScale75>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 2L, 2L], 0.5f, 1f)]));

        // D. Train-mode channel uniformity at rank 3 ([N,C,1] mask) and rank 5 ([N,C,1,1,1] mask).
        Assert.True(AutoTest.AdvancedTestGraph<NNSpatialDropoutChannelWiseRank3>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 4L], 0.5f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNSpatialDropoutChannelWiseRank5>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 2L, 2L, 2L], 0.5f, 1f)]));
    }

    /// <summary>
    /// SpatialDropout EVAL-mode (exact identity, QEE-computable) + alias coverage. Eval is
    /// correct (the bug pinned by <see cref="TestSpatialDropoutTrainModeChannelWise"/> only
    /// affects training mode), so these pass. Covers (C) eval identity at ratios 0.5 and 0.9,
    /// (D-eval) rank-genericity — eval identity at rank 3/4/5 + the rank-2 degenerate path,
    /// and (E) Dropout1d/2d/3d alias equivalence (bit-for-bit vs SpatialDropout in eval mode).
    /// </summary>
    [Fact]
    public void TestSpatialDropoutEvalAndAliases()
    {
        // C. Eval-mode identity at ratios 0.5 and 0.9 (exact, ratio-independent).
        Assert.True(AutoTest.AdvancedTestGraph<NNSpatialDropoutEvalIdentity>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 2L, 2L], 0.5f, 1f)]));

        // D. Rank-genericity — eval identity at rank 3 (1d), 4 (2d), 5 (3d).
        Assert.True(AutoTest.AdvancedTestGraph<NNSpatialDropoutEvalIdentityAnyRank>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 4L], 0.5f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNSpatialDropoutEvalIdentityAnyRank>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 4L, 4L], 0.5f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNSpatialDropoutEvalIdentityAnyRank>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 2L, 2L, 2L], 0.5f, 1f)]));
        // D. Rank-2 degenerate path (eval identity).
        Assert.True(AutoTest.AdvancedTestGraph<NNSpatialDropoutRank2Degenerate>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L], 0.5f, 1f)]));

        // E. Dropout1d/2d/3d alias equivalence (eval mode, bit-for-bit).
        Assert.True(AutoTest.AdvancedTestGraph<NNDropout1dAliasEquiv>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 4L], 0.5f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNDropout2dAliasEquiv>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 4L, 4L], 0.5f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNDropout3dAliasEquiv>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 2L, 2L, 2L], 0.5f, 1f)]));
    }

    /// <summary>
    /// AlphaDropout (SELU-paired elementwise dropout, src/Shorokoo.Modules/Layers/Dropout.cs,
    /// Klambauer et al. 2017) coverage. The load-bearing check is the TRAIN-mode per-element
    /// invariant (A, mask-AGNOSTIC, evaluated on the ORT backend's real random draw): for a
    /// fixed ratio every output element is one of exactly TWO known closed forms — kept →
    /// a·x+b, dropped → a·α'+b (a constant) — so (y−(a·x+b))·(y−(a·α'+b))==0 elementwise
    /// whatever the mask (the AlphaDropout analog of NNDropoutChecks' y·(y−2x)==0). Run at
    /// ratio 0.5 (a=0.8864053, b=0.7791938) AND ratio 0.25 (a=0.8672579, b=0.3811814) to pin
    /// that a,b TRACK the ratio. (B) Moment preservation over a large [64,64] tensor — an
    /// APPROXIMATE statistical bound (preserved only in expectation, #74004). (C) Eval-mode
    /// exact identity at ratios 0.5 and 0.9 (QEE-computable). x is built STRICTLY POSITIVE so
    /// kept values never collide with the negative dropped constant.
    /// </summary>
    [Fact]
    public void TestAlphaDropoutCoverage()
    {
        // A. Train-mode per-element two-value invariant at ratio 0.5 and 0.25 (mask-agnostic).
        Assert.True(AutoTest.AdvancedTestGraph<NNAlphaDropoutPerElementInvariant>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 8L], 0.5f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNAlphaDropoutPerElementInvariantRatio25>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 8L], 0.5f, 1f)]));

        // B. Moment preservation over a large tensor (approximate; in-expectation, #74004).
        Assert.True(AutoTest.AdvancedTestGraph<NNAlphaDropoutMomentPreservation>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([64L, 64L], 0.001f, -2f)]));

        // C. Eval-mode exact identity at ratios 0.5 and 0.9 (ratio-independent).
        Assert.True(AutoTest.AdvancedTestGraph<NNAlphaDropoutEvalIdentity>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 8L], 0.5f, 1f)]));
    }

    /// <summary>
    /// FeatureAlphaDropout (channel-wise SELU-paired dropout, src/Shorokoo.Modules/Layers/
    /// Dropout.cs, Klambauer et al. 2017 + Tompson et al. 2015) coverage. The distinguishing
    /// property is CHANNEL UNIFORMITY (D, train mode, mask-AGNOSTIC): the drop decision is per
    /// (sample,channel), so the per-element is-kept indicator must be CONSTANT across the
    /// spatial axes within each (n,c), AND the two-value per-element invariant holds. This
    /// PASSES for FeatureAlphaDropout and FAILS for elementwise AlphaDropout. Covers rank-4
    /// ([N,C,1,1] mask) plus rank-3 ([N,C,1]) and rank-5 ([N,C,1,1,1]) to exercise the
    /// rank-derived ones-run. (E) Rank-genericity — eval-mode exact identity at rank 3/4/5 +
    /// the rank-2 [N,C] degenerate path. x is strictly positive so kept (a·x+b > 0) never
    /// collides with the negative dropped constant a·α'+b.
    /// </summary>
    [Fact]
    public void TestFeatureAlphaDropoutCoverage()
    {
        // D. Channel uniformity at rank 4 ([N,C,1,1] mask), rank 3 ([N,C,1]), rank 5 ([N,C,1,1,1]).
        Assert.True(AutoTest.AdvancedTestGraph<NNFeatureAlphaDropoutChannelUniform>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 2L, 2L], 0.5f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNFeatureAlphaDropoutChannelUniformRank3>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 4L], 0.5f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNFeatureAlphaDropoutChannelUniformRank5>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 2L, 2L, 2L], 0.5f, 1f)]));

        // E. Rank-genericity — eval-mode exact identity at rank 3/4/5 + rank-2 degenerate.
        Assert.True(AutoTest.AdvancedTestGraph<NNFeatureAlphaDropoutEvalIdentityAnyRank>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 4L], 0.5f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNFeatureAlphaDropoutEvalIdentityAnyRank>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 4L, 4L], 0.5f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNFeatureAlphaDropoutEvalIdentityAnyRank>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L, 2L, 2L, 2L], 0.5f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNFeatureAlphaDropoutEvalIdentityAnyRank>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L], 0.5f, 1f)]));
    }

    [Fact]
    public void TestPoolingHelpersCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<NNPoolingHelpersChecks>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 4L, 4L], 0.5f, -4f)]));
    }

    /// <summary>
    /// Generalized Pooling helper coverage (src/Shorokoo.Modules/Layers/Pooling.cs),
    /// pooling-generalization design §7 cases 1–8. Each self-checking [Module] returns a
    /// Scalar&lt;bit&gt; verified on the ORT backend inside AutoTest.AdvancedTestGraph (pool ops
    /// have NO QEE values — only shape/dtype resolves there), exactly like NNPoolingHelpersChecks:
    /// <list type="bullet">
    ///   <item>§7-1 1D/3D closed forms: MaxPool1d([2]) on [1,3,2,4]→[3,4], AvgPool1d→[2,3];
    ///         full-window MaxPool3d/AvgPool3d on a [1,1,2,2,2] cube of 1..8 → max 8 / mean 4.5;</item>
    ///   <item>§7-2 LpPool p=2 == √(Σx²): LpPool1d([2]) on [3,4]→√25=5, and full-window
    ///         LpPool2d == GlobalLpPool;</item>
    ///   <item>§7-3 full-window MaxPool2d/AvgPool2d/LpPool2d == GlobalMax/Avg/LpPool (relative-L1);</item>
    ///   <item>§7-4 scalar↔per-axis alias equivalence (MaxPool2d(c,2)==MaxPool2d(c,[2,2]),
    ///         Avg/Lp likewise) + per-rank MaxPool1d([2])==MaxPool([2]);</item>
    ///   <item>§7-5 asymmetric geometry (kernel [3,2], stride [2,1], padding [1,0]) == a hand-built
    ///         NN.MaxPool with the same attrs;</item>
    ///   <item>§7-6 MaxUnpool round-trip: (vals,idx)=MaxPoolWithIndices(x,[2,2]) then
    ///         MaxUnpool(...,outputShape:x.shape) ⇒ x's shape, GlobalMax reinstated, sum(u)==sum(vals);</item>
    ///   <item>§7-7 count_include_pad toggle: AvgPool2d([2,2],padding:[1,1]) false==[[1,2],[3,4]]
    ///         vs true==[[.25,.5],[.75,1]], and the two DIFFER at the padded border.</item>
    /// </list>
    /// The closed-form / scalar-overload / count-pad modules build the pool input as an in-module
    /// constant (exact hand math; the scalar overload also needs a build-time-known rank) and fold a
    /// 0*x touch; the global-equivalence / geometry / unpool modules pool the runtime input directly.
    /// No QEE/CS-roundtrip flags are disabled — every module's graph (incl. MaxUnpool, which resolves
    /// shape from the outputShape input) survives QEE DType resolution and the CS roundtrip.
    /// </summary>
    [Fact]
    public void TestGeneralizedPoolingHelpersCoverage()
    {
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
    }

    [Fact]
    public void TestNNStaticWrapperOpsCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<NNStaticWrapperWindowEyeDetCheck>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 2L], 1f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<NNStaticWrapperPoolMathCheck>(
            hyperparamInputs: [],
            runtimeInputs: [
                RangeTensor([1L, 2L, 4L, 4L], 0.5f, -4f),
                TensorData(DType.Int64, [3L], 7L, 8L, 9L),
                TensorData(DType.Int64, [3L], 2L, 3L, 4L)]));
    }

    /// <summary>Loss edge semantics from the 2026-06-12 analytic campaign (the closed
    /// forms at ordinary points are in <see cref="TestLossClosedFormCoverage"/>):
    /// SmoothL1's quadratic vs linear regions, BCE's clamp at p ∈ {0,1}, BCEWithLogits'
    /// numerical stability at logits ±100, and CrossEntropy's log-softmax at
    /// non-uniform logits. Exact expectations live in NNLossEdgeCaseChecks.</summary>
    [Fact]
    public void TestLossEdgeCasesCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<NNLossEdgeCaseChecks>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [2L], 0.5f, 2f)]));
    }

    [Fact]
    public void TestLossClosedFormCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<NNLossClosedFormChecks>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [2L], 1f, 3f)]));
    }

    /// <summary>
    /// LogCoshLoss + PoissonNLLLoss closed-form / knob / numerical-stability
    /// coverage. Each self-checking [Module] (NNLogCoshLossChecks /
    /// NNPoissonNLLLossChecks) compares the loss on in-module constant inputs to
    /// hand-derived (double-precision) expectations via the NaN-safe
    /// Within/IfElse(1,0) ok-counting idiom, exactly like NNLossEdgeCaseChecks:
    /// <list type="bullet">
    ///   <item>LogCosh: Inline(mean)=0.21689042, Reduced(..,Sum)=0.43378083,
    ///         PerElement=[0,0.43378083], mid d=2 → 1.3250027, and the d=100
    ///         stability edge (≈99.30685, finite — no overflow/NaN);</item>
    ///   <item>PoissonNLL: Inline(mean)=0.85914091, Reduced(..,Sum)=1.71828183,
    ///         PerElement=[1,0.71828183], logInput=false mean=0.80685281, and the
    ///         full=true Stirling on target=[0,1,3] → [1,1,2.7640816] with the
    ///         target=0 element asserted FINITE (the clamped-0·log NaN-safety gate).</item>
    /// </list>
    /// A dummy [2] runtime input drives AutoTest.AdvancedTestGraph; the asserted
    /// math is entirely on the in-module constants.
    /// </summary>
    [Fact]
    public void TestLogCoshPoissonNLLLossCoverage()
    {
        var dummy = TensorData(DType.Float32, [2L], 1f, 3f);
        Assert.True(AutoTest.AdvancedTestGraph<NNLogCoshLossChecks>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNPoissonNLLLossChecks>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
    }

    /// <summary>
    /// Hinge / SquaredHinge / BinaryFocal closed-form + mode + knob coverage.
    /// Each self-checking [Module] (NNHingeLossChecks / NNSquaredHingeLossChecks /
    /// NNBinaryFocalLossChecks) compares the loss on in-module constant inputs to
    /// hand-derived (double-precision) expectations via the NaN-safe
    /// Within/IfElse(1,0) ok-counting idiom:
    /// <list type="bullet">
    ///   <item>Hinge (targets ±1): Inline(mean)=0.66666667, Reduced(..,Sum)=2.0,
    ///         PerElement=[0.5,1.5,0], negative-class pred=[−2]/target=[−1]→0, and the
    ///         exact-margin edge pred=[1]/target=[1]→0;</item>
    ///   <item>SquaredHinge (targets ±1): Inline(mean)=0.83333333, Reduced(..,Sum)=2.5,
    ///         PerElement=[0.25,2.25,0], and the Keras cross-check
    ///         target=[−1,1,1]/pred=[0.6,−0.7,−0.5]→mean 2.56666667;</item>
    ///   <item>BinaryFocal (logits, targets {0,1}): Inline defaults (α=0.25,γ=2) at
    ///         logit=0,t=1→0.04332170, γ=0→α-weighted BCE 0.17328680, t=0→0.12997010
    ///         (α_t=0.75), and the α=−1 sentinel (no α-weighting)→0.17328680.</item>
    /// </list>
    /// A dummy [2] runtime input drives AutoTest.AdvancedTestGraph; the asserted
    /// math is entirely on the in-module constants.
    /// </summary>
    [Fact]
    public void TestHingeSquaredHingeFocalLossCoverage()
    {
        var dummy = TensorData(DType.Float32, [2L], 1f, 3f);
        Assert.True(AutoTest.AdvancedTestGraph<NNHingeLossChecks>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNSquaredHingeLossChecks>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNBinaryFocalLossChecks>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
    }

    /// <summary>
    /// Loss configurability-knob closed forms (loss-knobs design §7): every knob
    /// covered with a hand-computed constant inside a self-checking [Module]
    /// (Scalar&lt;bit&gt; ok-counting, NaN-safe). CE reduction/weight/ignore_index;
    /// CE label_smoothing (incl. the 3-way labelSmoothing+weight+ignoreIndex
    /// interaction, design O2); NLL weight/ignore_index; BCEWithLogits pos_weight;
    /// SmoothL1 beta (cross-checked against Huber(δ=β)/β); L1/L2/Huber reductions.
    /// All driven through AutoTest.AdvancedTestGraph with a dummy runtime input
    /// (the asserted math is on in-module constants). The Reduced(..,None)-throws
    /// case is the separate <see cref="TestLossReducedNoneThrows"/> [Fact].
    /// </summary>
    [Fact]
    public void TestLossKnobClosedFormCoverage()
    {
        var dummy = TensorData(DType.Float32, [2L], 1f, 3f);
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

    /// <summary>
    /// Loss-knobs design §6.2/§7: <c>Reduced(..., LossReduction.None)</c> throws
    /// (None returns a per-element tensor, not a scalar — callers must use
    /// <c>PerElement</c>). Pinned for every loss family's <c>Reduced</c> overload.
    /// </summary>
    [Fact]
    public void TestLossReducedNoneThrows()
    {
        var pred = Tensor([1L, 2L], 0f, 0f);
        var tgt = Vector(0L);
        var fpred = Tensor([2L], 1f, 3f);
        var ftgt = Tensor([2L], 0f, 1f);

        Assert.Throws<ArgumentException>(() =>
            CrossEntropyLoss.Reduced(pred, tgt, reduction: LossReduction.None));
        Assert.Throws<ArgumentException>(() =>
            NLLLoss.Reduced(pred, tgt, reduction: LossReduction.None));
        Assert.Throws<ArgumentException>(() =>
            BCEWithLogitsLoss.Reduced(fpred, ftgt, reduction: LossReduction.None));
        Assert.Throws<ArgumentException>(() =>
            SmoothL1Loss.Reduced(1f, fpred, ftgt, reduction: LossReduction.None));
        Assert.Throws<ArgumentException>(() =>
            HuberLoss.Reduced(Scalar(1f), fpred, ftgt, reduction: LossReduction.None));
        Assert.Throws<ArgumentException>(() =>
            L1Loss.Reduced(fpred, ftgt, reduction: LossReduction.None));
        Assert.Throws<ArgumentException>(() =>
            L2Loss.Reduced(fpred, ftgt, reduction: LossReduction.None));
    }

    /// <summary>
    /// TripletMarginLoss + TripletMarginWithDistance closed-form / knob coverage
    /// (triplet-margin-loss design §9). Each self-checking [Module]
    /// (NNTripletMarginClosedFormChecks / NNTripletMarginSwapMarginPChecks /
    /// NNTripletMarginReductionChecks / NNTripletMarginWithDistanceChecks) compares
    /// the loss on in-module constant anchor/positive/negative tensors to
    /// hand-derived (double-precision) expectations via the NaN-safe
    /// Within/IfElse(1,0) ok-counting idiom; the load-bearing closed form is also
    /// cross-checked against an INDEPENDENT raw-op Euclidean reference. Covers:
    /// the §9.1 [N=3,D=3] closed form ([0,0.5,2] per-triplet, mean 0.8333333,
    /// sum 2.5); the Balntas swap raising the loss 0→2 (the min + bit-gate); margin
    /// and p sweeps (p=1 vs p=2 ⇒ different loss); the Mean==Inline /
    /// Sum==ΣPerElement reduction equivalences; and the TripletMarginWithDistance
    /// Func wiring (custom squared-L2 closed form, euclid-Func == built-in p=2, and
    /// swap on the custom distance). The Reduced(..,None)-throws case is the
    /// separate <see cref="TestTripletMarginReducedNoneThrows"/> [Fact].
    /// </summary>
    [Fact]
    public void TestTripletMarginLossCoverage()
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
    }

    /// <summary>
    /// TripletMarginLoss / TripletMarginWithDistance <c>Reduced(..., None)</c>
    /// throws <see cref="ArgumentException"/> (None returns a per-triplet tensor —
    /// callers must use <c>PerElement</c>), matching every other loss family's
    /// guard (triplet-margin-loss design §9.4). C#-level [Fact] (a thrown
    /// exception, not a graph check).
    /// </summary>
    [Fact]
    public void TestTripletMarginReducedNoneThrows()
    {
        var a = Tensor([1L, 2L], 0f, 0f);
        var p = Tensor([1L, 2L], 1f, 0f);
        var n = Tensor([1L, 2L], 0f, 2f);

        Assert.Throws<ArgumentException>(() =>
            TripletMarginLoss.Reduced(Scalar(1f), Scalar(2f), Scalar(1e-6f), Scalar(false),
                a, p, n, reduction: LossReduction.None));
        Assert.Throws<ArgumentException>(() =>
            TripletMarginWithDistance.Reduced(
                (x, y) => { var d = x - y; return (d * d).Reduce(ReduceKind.Sum, [Scalar(-1L)], keepDims: false); },
                1f, false, a, p, n, reduction: LossReduction.None));
    }

    /// <summary>
    /// CosineEmbeddingLoss + CosineSimilarity closed-form / knob coverage
    /// (cosine-embedding-loss design §9). Each self-checking [Module]
    /// (NNCosineEmbeddingClosedFormChecks / NNCosineEmbeddingMarginGatingChecks /
    /// NNCosineEmbeddingWhereSplitChecks / NNCosineEmbeddingReductionChecks /
    /// NNCosineSimilarityHelperChecks) compares the loss/helper on in-module constant
    /// x1/x2/y tensors to hand-derived expectations via the NaN-safe
    /// Within/IfElse(1,0) ok-counting idiom; the load-bearing closed form is also
    /// cross-checked against an INDEPENDENT raw-op cosine + where-split reference.
    /// Covers: the §9.1 [N=6,D=2] closed form over BOTH y branches (PerElement
    /// [0,1,2,0.29289322,1,0.70710678], mean 0.83333334, sum 5); margin gating
    /// (the y=−1 arm shifts by the margin, the y=+1 arm is margin-independent); the
    /// where(y==1,…) data-dependent split under a y-flip ([0,0]→[1,1]); the
    /// Mean==Inline / Sum==ΣPerElement reduction equivalences; and the
    /// CosineSimilarity helper (exact cosines, scale-invariance cos(k·x1,x2)==cos,
    /// and the eps-guard zero-vector → finite 0). The Reduced(..,None)-throws case is
    /// the separate <see cref="TestCosineEmbeddingReducedNoneThrows"/> [Fact].
    /// </summary>
    [Fact]
    public void TestCosineEmbeddingLossCoverage()
    {
        var dummy = TensorData(DType.Float32, [2L], 1f, 3f);
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

    /// <summary>
    /// CosineEmbeddingLoss <c>Reduced(..., None)</c> throws
    /// <see cref="ArgumentException"/> (None returns a per-sample tensor — callers
    /// must use <c>PerElement</c>), matching every other loss family's guard
    /// (cosine-embedding-loss design §9.4). C#-level [Fact] (a thrown exception, not
    /// a graph check).
    /// </summary>
    [Fact]
    public void TestCosineEmbeddingReducedNoneThrows()
    {
        var x1 = Tensor([1L, 2L], 1f, 0f);
        var x2 = Tensor([1L, 2L], 0f, 1f);
        var y = Tensor([1L], 1f);

        Assert.Throws<ArgumentException>(() =>
            CosineEmbeddingLoss.Reduced(Scalar(0f), Scalar(1e-8f), x1, x2, y,
                reduction: LossReduction.None));
    }

    /// <summary>
    /// Recurrent.RNN (vanilla/Elman) forward-value coverage (rnn/design.md §7). Each
    /// self-checking [Module] compares Recurrent.RNN(x, …)'s (collapsed) y⊕hN against an
    /// inlined frozen golden reference (self-generated at master-seed-0 init) on the ORT
    /// backend (RNN has NO QEE step values — design note [2]):
    /// <list type="bullet">
    ///   <item>§7-2 core-op match (forward, tanh): y AND hN, on [L,N,in] and (batchFirst) [N,L,in];</item>
    ///   <item>§7-1 single-step recurrence anchor (L=1, h_0=0): y[0]==tanh(W·x_0+b), hN==y[0];</item>
    ///   <item>§7-3 relu forward-value match (BPTT-throws is a separate Training [Fact]);</item>
    ///   <item>§7-4 bias on/off ⇒ no-B / concat(bias,zeros) op;</item>
    ///   <item>§7-5 numLayers:2 ⇒ a hand-built 2-op stack (y + [2,N,H] hN);</item>
    ///   <item>§7-6 Reverse (trainable) and Bidirectional (forward only) ⇒ the matching op,
    ///         with bidi y last axis == 2H and hN == [2,N,H];</item>
    ///   <item>§7-7 batchFirst transpose equivalence (independent of the op reference);</item>
    ///   <item>§7-8 state contract: hN == y[-1] == the op's Y_h, shape [D·numLayers,N,H].</item>
    /// </list>
    /// All shapes have N=2 (batch ≥ 2) so the batch axis is non-degenerate.
    /// </summary>
    [Fact]
    public void TestRecurrentRnnForwardValueCoverage()
    {
        // §7-2 core op (forward, tanh): seq-first and batch-first inputs.
        Assert.True(AutoTest.AdvancedTestGraph<RnnMatchesCoreOpForwardTanh>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<RnnMatchesCoreOpBatchFirst>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 4L, 3L], 0.1f, -1f)]));

        // §7-1 single-step recurrence anchor (L=1, h_0=0).
        Assert.True(AutoTest.AdvancedTestGraph<RnnSingleStepAnchorTanh>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 3L], 0.2f, -0.5f)]));

        // §7-3 relu forward-value match.
        Assert.True(AutoTest.AdvancedTestGraph<RnnReluMatchesCoreOpForward>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));

        // §7-4 bias on/off.
        Assert.True(AutoTest.AdvancedTestGraph<RnnBiasOnOffMatchesCoreOp>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));

        // §7-5 numLayers:2 stacking.
        Assert.True(AutoTest.AdvancedTestGraph<RnnNumLayersStackMatchesCoreOp>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));

        // §7-6 Reverse (trainable) + Bidirectional (forward only).
        Assert.True(AutoTest.AdvancedTestGraph<RnnReverseMatchesCoreOp>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<RnnBidirectionalMatchesCoreOp>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));

        // §7-7 batchFirst transpose equivalence.
        Assert.True(AutoTest.AdvancedTestGraph<RnnBatchFirstEquivalence>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 4L, 3L], 0.1f, -1f)]));

        // §7-8 state contract.
        Assert.True(AutoTest.AdvancedTestGraph<RnnStateContractForwardSingleLayer>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));
    }

    /// <summary>
    /// Recurrent.LSTM forward-value coverage (lstm/design.md §7). Mirrors
    /// TestRecurrentRnnForwardValueCoverage exactly: each self-checking [Module] compares
    /// Recurrent.LSTM(x, …)'s (collapsed) y⊕hN⊕cN against an inlined frozen golden reference
    /// (self-generated at master-seed-0 init) on the ORT backend (LSTM has NO QEE step
    /// values):
    /// <list type="bullet">
    ///   <item>§7-1 core-op match (forward): y, hN, cN, on [L,N,in] and (batchFirst) [N,L,in];</item>
    ///   <item>§7-2 single-step gate anchor (L=1, h_0=c_0=0): closed-form i/o/c̃ from the i,o,f,c
    ///         gate blocks ⇒ C_1=i⊙c̃, H_1=o⊙tanh(C_1) — pins the gate packing order;</item>
    ///   <item>§7-3 bias on/off ⇒ no-B / concat(bias,zeros) op;</item>
    ///   <item>§7-4 numLayers:2 ⇒ a hand-built 2-op stack (y + [2,N,H] hN + [2,N,H] cN);</item>
    ///   <item>§7-5 Reverse (trainable) and Bidirectional (forward only) ⇒ the matching op,
    ///         with bidi y last axis == 2H and hN/cN == [2,N,H];</item>
    ///   <item>§7-6 batchFirst transpose equivalence (independent of the op reference);</item>
    ///   <item>§7-7 state contract: hN == y[-1] == the op's Y_h, cN == the op's Y_c, shape [D·numLayers,N,H].</item>
    /// </list>
    /// All shapes have N=2 (batch ≥ 2) so the batch axis is non-degenerate.
    /// </summary>
    [Fact]
    public void TestRecurrentLstmForwardValueCoverage()
    {
        // §7-1 core op (forward): seq-first and batch-first inputs.
        Assert.True(AutoTest.AdvancedTestGraph<LstmMatchesCoreOpForward>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<LstmMatchesCoreOpBatchFirst>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 4L, 3L], 0.1f, -1f)]));

        // §7-2 single-step gate anchor (L=1, h_0=c_0=0) — pins the i,o,f,c gate packing.
        Assert.True(AutoTest.AdvancedTestGraph<LstmSingleStepGateAnchor>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 3L], 0.2f, -0.5f)]));

        // §7-3 bias on/off.
        Assert.True(AutoTest.AdvancedTestGraph<LstmBiasOnOffMatchesCoreOp>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));

        // §7-4 numLayers:2 stacking.
        Assert.True(AutoTest.AdvancedTestGraph<LstmNumLayersStackMatchesCoreOp>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));

        // §7-5 Reverse (trainable) + Bidirectional (forward only).
        Assert.True(AutoTest.AdvancedTestGraph<LstmReverseMatchesCoreOp>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<LstmBidirectionalMatchesCoreOp>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));

        // §7-6 batchFirst transpose equivalence.
        Assert.True(AutoTest.AdvancedTestGraph<LstmBatchFirstEquivalence>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 4L, 3L], 0.1f, -1f)]));

        // §7-7 state contract.
        Assert.True(AutoTest.AdvancedTestGraph<LstmStateContractForwardSingleLayer>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));
    }

    /// <summary>
    /// Recurrent.GRU forward-value coverage (gru/design.md §7). Mirrors
    /// TestRecurrentLstmForwardValueCoverage exactly (plus the GRU-specific linearBeforeReset
    /// both-forms check): each self-checking [Module] compares Recurrent.GRU(x, …)'s (collapsed)
    /// y⊕hN against an inlined frozen golden reference (self-generated at master-seed-0 init)
    /// on the ORT backend (GRU has NO QEE step values):
    /// <list type="bullet">
    ///   <item>§7-1 core-op match (forward): y, hN, on [L,N,in] and (batchFirst) [N,L,in];</item>
    ///   <item>§7-2 linearBeforeReset BOTH forms: true vs false DIFFER, and each matches its own-form
    ///         op (the reset-after default + that the bit is honored — the GRU numeric crux);</item>
    ///   <item>§7-3 single-step gate anchor (L=1, h_0=0): closed-form z/ĥ from the z,r,h gate blocks
    ///         ⇒ H_1=(1−z)⊙ĥ — pins the gate packing order;</item>
    ///   <item>§7-4 bias on/off ⇒ no-B / concat(bias,zeros) op;</item>
    ///   <item>§7-5 numLayers:2 ⇒ a hand-built 2-op stack (y + [2,N,H] hN);</item>
    ///   <item>§7-6 Reverse (trainable) and Bidirectional (forward only) ⇒ the matching op,
    ///         with bidi y last axis == 2H and hN == [2,N,H];</item>
    ///   <item>§7-7 batchFirst transpose equivalence (independent of the op reference);</item>
    ///   <item>§7-8 state contract: hN == y[-1] == the op's Y_h, shape [D·numLayers,N,H].</item>
    /// </list>
    /// All shapes have N=2 (batch ≥ 2) so the batch axis is non-degenerate.
    /// </summary>
    [Fact]
    public void TestRecurrentGruForwardValueCoverage()
    {
        // §7-1 core op (forward): seq-first and batch-first inputs.
        Assert.True(AutoTest.AdvancedTestGraph<GruMatchesCoreOpForward>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<GruMatchesCoreOpBatchFirst>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 4L, 3L], 0.1f, -1f)]));

        // §7-2 linearBeforeReset BOTH forms (the GRU numeric crux) — true vs false differ, each matches its op form.
        Assert.True(AutoTest.AdvancedTestGraph<GruLinearBeforeResetBothForms>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));

        // §7-3 single-step gate anchor (L=1, h_0=0) — pins the z,r,h gate packing.
        Assert.True(AutoTest.AdvancedTestGraph<GruSingleStepGateAnchor>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 2L, 3L], 0.2f, -0.5f)]));

        // §7-4 bias on/off.
        Assert.True(AutoTest.AdvancedTestGraph<GruBiasOnOffMatchesCoreOp>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));

        // §7-5 numLayers:2 stacking.
        Assert.True(AutoTest.AdvancedTestGraph<GruNumLayersStackMatchesCoreOp>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));

        // §7-6 Reverse (trainable) + Bidirectional (forward only).
        Assert.True(AutoTest.AdvancedTestGraph<GruReverseMatchesCoreOp>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<GruBidirectionalMatchesCoreOp>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));

        // §7-7 batchFirst transpose equivalence.
        Assert.True(AutoTest.AdvancedTestGraph<GruBatchFirstEquivalence>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 4L, 3L], 0.1f, -1f)]));

        // §7-8 state contract.
        Assert.True(AutoTest.AdvancedTestGraph<GruStateContractForwardSingleLayer>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([4L, 2L, 3L], 0.1f, -1f)]));
    }

    /// <summary>
    /// Single-step recurrent CELL forward-value coverage (recurrent-cells/design.md §7) for
    /// Recurrent.RNNCell / LSTMCell / GRUCell. Each self-checking [Module] runs the cell with
    /// nonzero previous state(s) threaded through initial_h(/initial_c) and compares the
    /// (collapsed) outputs against an inlined frozen golden reference (self-generated at
    /// master-seed-0 init) on the ORT backend (cells have NO QEE step values). Covers, per
    /// cell:
    /// <list type="bullet">
    ///   <item>§7-1 closed-form anchor with NONZERO h (so R is exercised): RNNCell tanh AND relu;
    ///         LSTMCell i,o,f,c gate algebra (pins the gate packing); GRUCell both lbr forms;</item>
    ///   <item>§7-2/§7-3 cell ≡ seq=1 reference op + the [N,H] (num_dir-stripped) shape contract;</item>
    ///   <item>§7-4 bias on/off ⇒ no-B / concat(bias,zeros) seq=1 op;</item>
    ///   <item>§7-5 STATE THREADING (the defining test): two hand-unrolled cell steps from h_0=0
    ///         equal the full Recurrent.RNN/LSTM/GRU over the length-2 sequence — the cell really is
    ///         one step of the scan.</item>
    /// </list>
    /// The §7-6 FD grad checks, §7-7 rig smoke and §7-8 relu-cell BPTT throw are Training [Fact]s.
    /// </summary>
    [Fact]
    public void TestRecurrentCellForwardValueCoverage()
    {
        // ---- RNNCell ----
        // §7-1 closed-form anchors (H=2, N=1, nonzero h): tanh and relu.
        Assert.True(AutoTest.AdvancedTestGraph<RnnCellClosedFormTanh>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 3L], 0.2f, -0.3f)]));
        // Wider positive ramp than the tanh anchor: under the seed-0 weight draw the
        // [-0.3, 0.1] ramp drives every relu pre-activation negative, freezing an all-zero
        // (vacuous) golden — this input keeps at least one unit alive.
        Assert.True(AutoTest.AdvancedTestGraph<RnnCellClosedFormRelu>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 3L], 0.5f, 0.5f)]));
        // §7-2/§7-3 cell ≡ seq=1 op + shape; §7-4 bias on/off; §7-5 state threading.
        Assert.True(AutoTest.AdvancedTestGraph<RnnCellMatchesSeq1Op>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L], 0.1f, -0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<RnnCellBiasOnOff>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L], 0.1f, -0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<RnnCellStateThreading>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 2L, 3L], 0.1f, -0.6f)]));

        // ---- LSTMCell ----
        // §7-1 closed-form i,o,f,c gate anchor (nonzero h AND c — pins the gate packing).
        Assert.True(AutoTest.AdvancedTestGraph<LstmCellClosedFormGateAnchor>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 3L], 0.2f, -0.3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<LstmCellMatchesSeq1Op>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L], 0.1f, -0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<LstmCellBiasOnOff>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L], 0.1f, -0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<LstmCellStateThreading>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 2L, 3L], 0.1f, -0.6f)]));

        // ---- GRUCell ----
        // §7-1 closed-form anchors, BOTH lbr forms (lbr:false differs from lbr:true and matches its op).
        Assert.True(AutoTest.AdvancedTestGraph<GruCellClosedFormLbrTrue>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 3L], 0.2f, -0.3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<GruCellClosedFormLbrFalse>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([1L, 3L], 0.2f, -0.3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<GruCellMatchesSeq1Op>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L], 0.1f, -0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<GruCellBiasOnOff>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 3L], 0.1f, -0.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<GruCellStateThreading>(
            hyperparamInputs: [], runtimeInputs: [RangeTensor([2L, 2L, 3L], 0.1f, -0.6f)]));
    }

    /// <summary>
    /// Constant initializer coverage (constant-init design §7). Each self-checking [Module]
    /// materializes the seeded init graph and asserts on the produced constant via
    /// AutoTest.AdvancedTestGraph, exactly like NNLinearMatchesManualMatMul exercises
    /// KaimingUniform.Init: Constant fills every element with the supplied scalar (a [2,3]
    /// param == 7); is shape/rank-agnostic and reproduces negatives (a rank-1 [4] bias ==
    /// −2.5, Reduce(Sum) == −10); and specializes Zeros/Ones (Constant(0)==Zeros,
    /// Constant(1)==Ones, relative-L1 ≈ 0). A non-degenerate RangeTensor runtime input drives
    /// AutoTest (the params are input-independent; each module folds a 0*x touch).
    /// <para><b>REGRESSION GUARD (#440):</b> building the
    /// <c>Constant.Init([dims], Scalar(value))</c> graph into an ONNX model used to fail —
    /// the <c>Constant</c> initializer emitted a function whose name collided with the built-in
    /// ONNX <c>Constant</c> op. Fixed in #440 by OnnxFunctionName.Encode (prefixing colliding
    /// function names). This asserts the initializer now materializes correct values end-to-end.
    /// Adds no unique line/branch coverage (export/import/concretize tests already execute these
    /// paths) — kept as the only assertion that an op-named initializer produces correct values,
    /// so the collision cannot silently regress.</para>
    /// </summary>
    [Fact]
    public void TestConstantInitFillsCorrectValues()
    {
        var dummy = RangeTensor([2L, 3L], 0.5f, -1f);

        Assert.True(AutoTest.AdvancedTestGraph<NNConstantInitFillsValue>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNConstantInitRank1Negative>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNConstantInitMatchesZerosOnes>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
    }

    /// <summary>
    /// Orthogonal initializer coverage (orthogonal-init design §7) — the Björck/Newton–Schulz
    /// approximation's convergence quality. Each self-checking [Module] materializes
    /// <c>Q = Orthogonal.Init([...])</c>, forms the Gram matrix, and asserts it ≈ I (identity
    /// built via NN.EyeLike, as NNStaticWrapperWindowEyeDetCheck does): a [4,4] square
    /// (Qᵀ·Q ≈ I_4), a [4,2] tall (columns orthonormal, Qᵀ·Q ≈ I_2) and a [2,4] wide (rows
    /// orthonormal, Q·Qᵀ ≈ I_2 — exercising the r&lt;c branch). The asserted tolerances are the
    /// EMPIRICALLY OBSERVED converged max|G − I| for the seed-19 matrices: square &lt; 1e-2
    /// (observed 6.46e-3), tall/wide &lt; 1e-5 (observed ~1e-7 — machine-eps). 15 Björck steps
    /// converge acceptably (the square case to ~6.5e-3, tall/wide to machine precision). A
    /// non-degenerate RangeTensor runtime input drives AutoTest (the params are
    /// input-independent; each module folds a 0*x touch).
    /// </summary>
    [Fact]
    public void TestOrthogonalInitCoverage()
    {
        var dummy = RangeTensor([2L, 3L], 0.5f, -1f);

        Assert.True(AutoTest.AdvancedTestGraph<NNOrthogonalSquareGramIsIdentity>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNOrthogonalTallGramIsIdentity>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNOrthogonalWideGramIsIdentity>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
    }

    /// <summary>
    /// Configurable UniformRange + NormalDist initializer coverage (configurable-uniform-normal
    /// design §7). Each self-checking [Module] materializes a large seeded sample via
    /// <c>&lt;Init&gt;.Init([...], Scalar(...), Scalar(...))</c> and asserts on its empirical
    /// statistics through AutoTest.AdvancedTestGraph, exactly like NNConstantInitFillsValue
    /// exercises Constant.Init:
    /// <list type="bullet">
    ///   <item>UniformRange: a [1000] sample over (2,5) lies wholly in [2,5], spans the range
    ///         (min&lt;2.2, max&gt;4.8) and has the midpoint mean (≈3.5±0.1); plus a symmetric
    ///         (−1,1) sample in [−1,1] with mean≈0. Observed: min 2.00047 / max 4.99732 /
    ///         mean 3.48868, and −0.99969 / 0.99821 / −0.00754.</item>
    ///   <item>NormalDist: a [10000] sample over (10,0.5) has empirical mean≈10 and std≈0.5
    ///         (both within 0.05), plus a (0,2) sample with mean≈0 and std≈2. Observed:
    ///         mean 10.00530 / std 0.49979, and mean 0.02117 / std 1.99934.</item>
    /// </list>
    /// A non-degenerate RangeTensor runtime input drives AutoTest (the params are
    /// input-independent; each module folds a 0*x touch).
    /// </summary>
    [Fact]
    public void TestUniformRangeNormalDistInitCoverage()
    {
        var dummy = RangeTensor([2L, 3L], 0.5f, -1f);

        Assert.True(AutoTest.AdvancedTestGraph<NNUniformRangeInRange>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
        Assert.True(AutoTest.AdvancedTestGraph<NNNormalDistMoments>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
    }

    /// <summary>
    /// Configurable-gain Xavier/Kaiming initializer coverage (configurable-gain design §7).
    /// The four <c>*Gain</c> classes are materialized via
    /// <c>&lt;Init&gt;.Init([64,64], Scalar(gain))</c> and checked through their empirical
    /// sample std (<c>sqrt(mean(w²) − mean(w)²)</c>), exactly like NNNormalDistMoments exercises
    /// NormalDist. On a SQUARE [64,64] shape (fanIn = fanOut = 64) all four collapse to the same
    /// closed form, so at gain=2 every one has std ≈ 0.25 (Xavier-uniform bound 2·√(6/128)=0.43301,
    /// /√3 = 0.25; Xavier-normal 2·√(2/128)=0.25; Kaiming-uniform 2·√(3/64)=0.43301, /√3 = 0.25;
    /// Kaiming-normal 2·√(1/64)=0.25). The tight ±0.015 band (observed 0.2482–0.2527 for the seeded
    /// 4096-sample draws) EXCLUDES the √6-double-bake value 0.354 (a buggy Kaiming using √(6/fanIn)
    /// would give 2·√(6/64)/√3 = 0.354), so this discriminates the §4.1 double-bake trap. The
    /// gain=1 Xavier (std ≈ 0.125, observed 0.126) reproduces the baked XavierUniform default, and
    /// the gain=√2 Kaiming (std ≈ 0.1767, observed 0.1765) reproduces the baked KaimingUniform —
    /// pinning the √2-equivalence. Driven through AutoTest.AdvancedTestGraph with a non-degenerate
    /// RangeTensor runtime input (the params are input-independent; the module folds a 0*x touch).
    /// </summary>
    [Fact]
    public void TestXavierKaimingGainInitCoverage()
    {
        var dummy = RangeTensor([2L, 3L], 0.5f, -1f);
        Assert.True(AutoTest.AdvancedTestGraph<NNXavierKaimingGainStd>(
            hyperparamInputs: [], runtimeInputs: [dummy]));
    }
}

/// <summary>
/// Training-rig coverage for the baseline NN library, in the
/// <see cref="TrainingRigCoverageTests"/> style: rig-construction one-liners
/// for the new optimizers (Adam/RMSprop/Adagrad), end-to-end convergence runs
/// (Linear + L2 + Adam regression; tiny conv net + CrossEntropy + SGDMomentum
/// classification), Adam bias-correction numerics, and gradient-flow /
/// state-update checks for the BatchNorm2d and Dropout eval paths.
/// </summary>
[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class NNLibraryTrainingCoverageTests
{
    /// <summary>Same rig-construction smoke helper as <see cref="TrainingRigCoverageTests"/>.</summary>
    private static void CoverFromScratch(
        FastComputationGraph modelGraph,
        FastComputationGraph lossGraph,
        FastComputationGraph optimizerGraph,
        long[] inputShape,
        params HyperValue[] hyperparams)
    {
        long totalElements = 1;
        foreach (var d in inputShape) totalElements *= d;
        var sampleInput = new TensorDataModelParam(
            "input", ModelParamType.InputParam,
            TensorData(inputShape, new float[totalElements]));

        var rig = TrainingRig.FromScratch(modelGraph, lossGraph, optimizerGraph,
            new NamedModelParam[] { sampleInput }, hyperparams);

        var checkpoint = rig.CreateDefaultCheckpoint();
        Assert.NotEmpty(rig.TrainableParamStructDef.Fields);
        Assert.NotNull(checkpoint.TrainableParams);
    }

    private static TensorDataStruct MakeBatch(string fieldName, string structName, TensorData data)
    {
        var def = new TensorStructDef(
            new[] { new TensorStructFieldDef(fieldName, DataStructure.Tensor, data.Shape.Dims.Length, data.DType) },
            structName);
        return new TensorDataStruct(def, new Dictionary<string, IData> { { fieldName, data } });
    }

    private static float[] Floats(IData data) => ((TensorData<float32>)data).AccessMemory().ToArray();

    // --- Wide-regression convergence fixture -------------------------------------------------
    // 32 samples, 2 features in, 400 outputs out (matches NNWideRegressionModel). Deterministic
    // and perfectly realizable: Y = X·Wtᵀ + bt, so the global-min L2 loss is exactly 0. The many
    // outputs concentrate the random-init starting loss into the narrow band asserted below. These
    // exact formulas are mirrored by the Monte-Carlo characterization that set the band/target.
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
    // central 1-in-100-billion (6.81σ) range is [0.871, 1.497]. We assert a hair wider to absorb
    // float-vs-double and slight right-skew; this still flags any gross init/model regression
    // (a starting loss off by even ~30% lands outside).
    private static void AssertWideStartLossInBand(float startLoss) =>
        Assert.True(startLoss is >= 0.85f and <= 1.52f,
            $"starting loss must fall in the platform-independent 1-in-100-billion init band " +
            $"[0.85, 1.52] (mean 1.184 ± 6.81σ); got {startLoss}");

    /// <summary>
    /// Rig-construction coverage for the three new optimizers (per-param state
    /// counts: Adam 3 — m/v/step; RMSprop 2 — squareAvg/momentumBuffer;
    /// Adagrad 1 — accumulator), mirroring TestNonDefaultOptimizersCoverage.
    /// </summary>
    [Fact]
    public void TestNewOptimizersRigCoverage()
    {
        CoverFromScratch(ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            AdamOptimizer.ComputationGraph, [4L], 0.001f, 0.9f, 0.999f, 1e-8f);
        CoverFromScratch(ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            RMSpropOptimizer.ComputationGraph, [4L], 0.01f, 0.99f, 1e-8f, 0.0f);
        CoverFromScratch(ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            AdagradOptimizer.ComputationGraph, [4L], 0.01f, 1e-10f);
    }

    /// <summary>
    /// Bilinear rig train-step smoke (design §7.4) — the FIRST module-level exercise of Einsum's
    /// autodiff, the load-bearing integration check. Builds a TrainingRig.FromScratch around
    /// BilinearRigModel (two graph inputs x1 [N,in1], x2 [N,in2] → Bilinear → per-row mean [N]) with
    /// L2Loss + Adam, runs ONE TrainStep, and asserts (a) the loss is finite, and (b) the rank-3 A
    /// weight [out,in1,in2] actually MOVED (|w1 − w0| &gt; 1e-7 for ≥1 element) — i.e. the
    /// EinsumGradient flows back into the kij operand. Two InputParams are seeded à la the multi-input
    /// attention decoder rig (TransformerDecoderMeanPoolModel).
    /// </summary>
    [Fact]
    public void TestBilinearTrainStepMovesWeight()
    {
        long[] x1Shape = [2L, 3L];   // [N, in1]
        long[] x2Shape = [2L, 4L];   // [N, in2]
        long[] outShape = [2L];      // per-row mean over out → [N]
        var x1Data = new float[] { 0.5f, -1f, 0.25f, 1f, -0.5f, 0.75f };
        var x2Data = new float[] { 0.3f, -0.5f, 0.2f, -0.1f, 0.4f, 0.6f, -0.2f, 0.8f };

        var sample = new NamedModelParam[]
        {
            new TensorDataModelParam("x1", ModelParamType.InputParam, TensorData(x1Shape, x1Data)),
            new TensorDataModelParam("x2", ModelParamType.InputParam, TensorData(x2Shape, x2Data)),
        };

        var rig = TrainingRig.FromScratch(
            BilinearRigModel.ComputationGraph, L2Loss.ComputationGraph, AdamOptimizer.ComputationGraph,
            sample, 0.01f, 0.9f, 0.999f, 1e-8f);

        var initial = rig.CreateDefaultCheckpoint();
        Assert.NotEmpty(rig.TrainableParamStructDef.Fields);

        // Locate the rank-3 A weight field [out, in1, in2] (the bias is rank 1).
        string? aName = null;
        foreach (var f in rig.TrainableParamStructDef.Fields)
            if (initial.TrainableParams.Fields[f.Name] is TensorData td && td.Shape.Dims.Length == 3)
            { aName = f.Name; break; }
        Assert.NotNull(aName);
        float[] a0 = Floats(initial.TrainableParams.Fields[aName]);
        Assert.True(a0.Length >= 2, $"expected the rank-3 A weight to have ≥2 elements; got {a0.Length}");

        var modelInputDef = new TensorStructDef(
            new[]
            {
                new TensorStructFieldDef("x1", DataStructure.Tensor, 2, DType.Float32),
                new TensorStructFieldDef("x2", DataStructure.Tensor, 2, DType.Float32),
            }, "ModelInput");
        var inputBatch = new TensorDataStruct(modelInputDef,
            new Dictionary<string, IData>
            {
                { "x1", TensorData(x1Shape, x1Data) },
                { "x2", TensorData(x2Shape, x2Data) },
            });
        var targetBatch = MakeBatch("targets", "Target", TensorData(outShape, new float[2]));

        var ctx = new ComputeContext();
        var compiled = ctx.Compile(rig.TrainingStepPureGraph);
        var step = rig.TrainStep(initial, inputBatch, targetBatch, compiled);
        float[] a1 = Floats(step.Checkpoint.TrainableParams.Fields[aName]);

        Assert.True(float.IsFinite(step.Loss), $"TrainStep loss must be finite; got {step.Loss}");
        Assert.NotEmpty(step.Checkpoint.OptimizerState.Fields);
        bool moved = false;
        for (int i = 0; i < a0.Length; i++)
            if (MathF.Abs(a1[i] - a0[i]) > 1e-7f) { moved = true; break; }
        Assert.True(moved, "the rank-3 Bilinear A weight must move (Einsum autodiff must flow into the kij operand)");
    }

    /// <summary>
    /// §8-8 Embedding paddingIdx train-step: a TrainingRig.FromScratch around
    /// EmbeddingPaddingRigModel (constant indices [0,1,2] over a trainable weight
    /// [4,3], paddingIdx:2 → last gathered row masked to zero; per-row-mean + the
    /// float input x → [3]) with L2Loss + SGD, runs ONE TrainStep, and asserts (a)
    /// the loss is finite and (b) the trainable embedding weight MOVED (≥1 element)
    /// — i.e. the masked lookup is differentiable and trains (the pad-row gradient
    /// is routed to zero by the output mask, the non-pad rows carry the signal).
    /// </summary>
    [Fact]
    public void TestEmbeddingPaddingTrainStepMovesWeight()
    {
        long[] xShape = [3L];
        var xData = new float[] { 0.5f, -1f, 0.25f };

        var sample = new NamedModelParam[]
        {
            new TensorDataModelParam("x", ModelParamType.InputParam, TensorData(xShape, xData)),
        };

        var rig = TrainingRig.FromScratch(
            EmbeddingPaddingRigModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, 0.1f);

        var initial = rig.CreateDefaultCheckpoint();
        Assert.NotEmpty(rig.TrainableParamStructDef.Fields);

        // Locate the embedding weight field [4, 3] (the only trainable param).
        string wName = rig.TrainableParamStructDef.Fields[0].Name;
        float[] w0 = Floats(initial.TrainableParams.Fields[wName]);
        Assert.True(w0.Length >= 2, $"expected the embedding weight to have ≥2 elements; got {w0.Length}");

        var inputBatch = MakeBatch("x", "ModelInput", TensorData(xShape, xData));
        var targetBatch = MakeBatch("targets", "Target", TensorData(xShape, new float[3]));

        var ctx = new ComputeContext();
        var compiled = ctx.Compile(rig.TrainingStepPureGraph);
        var step = rig.TrainStep(initial, inputBatch, targetBatch, compiled);
        float[] w1 = Floats(step.Checkpoint.TrainableParams.Fields[wName]);

        Assert.True(float.IsFinite(step.Loss), $"Embedding TrainStep loss must be finite; got {step.Loss}");
        bool moved = false;
        for (int i = 0; i < w0.Length; i++)
            if (MathF.Abs(w1[i] - w0[i]) > 1e-7f) { moved = true; break; }
        Assert.True(moved, "the trainable Embedding weight must move (the masked lookup must be differentiable)");
    }

    /// <summary>
    /// §8-5 EmbeddingBag train-step: a TrainingRig.FromScratch around EmbeddingBagRigModel
    /// (constant bags [[0,1],[2,3]] over a trainable table [5,3], BagMode.Sum → [2,3];
    /// per-row-mean + the float input x → [2]) with L2Loss + SGD, runs ONE TrainStep, and
    /// asserts (a) the loss is finite and (b) the owned trainable embedding table MOVED (≥1
    /// element) — i.e. the bag lookup (Gather + Reduce) is differentiable and trains end-to-end
    /// through the static helper, exactly like the Recurrent helper-owned weights. Mirrors
    /// TestEmbeddingPaddingTrainStepMovesWeight.
    /// </summary>
    [Fact]
    public void TestEmbeddingBagTrainStepMovesWeight()
    {
        long[] xShape = [2L];
        var xData = new float[] { 0.5f, -1f };

        var sample = new NamedModelParam[]
        {
            new TensorDataModelParam("x", ModelParamType.InputParam, TensorData(xShape, xData)),
        };

        var rig = TrainingRig.FromScratch(
            EmbeddingBagRigModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            sample, 0.1f);

        var initial = rig.CreateDefaultCheckpoint();
        Assert.NotEmpty(rig.TrainableParamStructDef.Fields);

        // Locate the embedding table field [5, 3] (the only trainable param).
        string wName = rig.TrainableParamStructDef.Fields[0].Name;
        float[] w0 = Floats(initial.TrainableParams.Fields[wName]);
        Assert.True(w0.Length >= 2, $"expected the embedding table to have ≥2 elements; got {w0.Length}");

        var inputBatch = MakeBatch("x", "ModelInput", TensorData(xShape, xData));
        var targetBatch = MakeBatch("targets", "Target", TensorData(xShape, new float[2]));

        var ctx = new ComputeContext();
        var compiled = ctx.Compile(rig.TrainingStepPureGraph);
        var step = rig.TrainStep(initial, inputBatch, targetBatch, compiled);
        float[] w1 = Floats(step.Checkpoint.TrainableParams.Fields[wName]);

        Assert.True(float.IsFinite(step.Loss), $"EmbeddingBag TrainStep loss must be finite; got {step.Loss}");
        bool moved = false;
        for (int i = 0; i < w0.Length; i++)
            if (MathF.Abs(w1[i] - w0[i]) > 1e-7f) { moved = true; break; }
        Assert.True(moved, "the trainable EmbeddingBag table must move (the bag lookup must be differentiable)");
    }

    /// <summary>
    /// TripletMarginLoss rig-trainability via the design §5 "loss-is-the-model-tail"
    /// recipe (triplet-margin-loss design §9.5). A TrainingRig.FromScratch around
    /// NNTripletEmbeddingRigModel (a [6,2]=[3N,D] input split into anchor/positive/
    /// negative blocks, a SHARED trainable Linear embedding on each, returning the
    /// scalar triplet loss with swap=true) + the NNIdentityScalarLoss adapter (loss
    /// already computed in the model) + Adam, runs ONE TrainStep and asserts (a) the
    /// loss is finite and (b) the trainable Linear embedding weight MOVED (≥1
    /// element) — i.e. the gradient flows through Pow/Sum/Min(swap)/Relu/IfElse back
    /// into the embedding. The input is a separable triplet (positives near anchors,
    /// negatives far) so the triplet loss is a real, nonzero objective. Mirrors
    /// TestEmbeddingPaddingTrainStepMovesWeight.
    /// </summary>
    [Fact]
    public void TestTripletMarginTrainStepMovesWeight()
    {
        // [3N, D] = [6, 2]: rows 0-1 anchor, 2-3 positive, 4-5 negative.
        long[] xShape = [6L, 2L];
        var xData = new float[]
        {
            0.5f, 1.0f,   -0.5f, -1.0f,   // anchors
            0.6f, 1.1f,   -0.4f, -0.9f,   // positives (near the anchors)
            2.0f, -2.0f,   2.5f,  3.0f,   // negatives (far)
        };

        var sample = new NamedModelParam[]
        {
            new TensorDataModelParam("x", ModelParamType.InputParam, TensorData(xShape, xData)),
        };

        var rig = TrainingRig.FromScratch(
            NNTripletEmbeddingRigModel.ComputationGraph, NNIdentityScalarLoss.ComputationGraph,
            AdamOptimizer.ComputationGraph, sample, 0.01f, 0.9f, 0.999f, 1e-8f);

        var initial = rig.CreateDefaultCheckpoint();
        Assert.NotEmpty(rig.TrainableParamStructDef.Fields);

        // Locate the Linear embedding weight (the rank-2 [out, in] = [2, 2] param).
        string? wName = null;
        foreach (var f in rig.TrainableParamStructDef.Fields)
            if (initial.TrainableParams.Fields[f.Name] is TensorData td && td.Shape.Dims.Length == 2)
            { wName = f.Name; break; }
        Assert.NotNull(wName);
        float[] w0 = Floats(initial.TrainableParams.Fields[wName]);
        Assert.True(w0.Length >= 2, $"expected the embedding weight to have ≥2 elements; got {w0.Length}");

        var inputBatch = MakeBatch("x", "ModelInput", TensorData(xShape, xData));
        var targetBatch = MakeBatch("targets", "Target", TensorData([1L], new float[1]));

        var ctx = new ComputeContext();
        var compiled = ctx.Compile(rig.TrainingStepPureGraph);
        var step = rig.TrainStep(initial, inputBatch, targetBatch, compiled);
        float[] w1 = Floats(step.Checkpoint.TrainableParams.Fields[wName]);

        Assert.True(float.IsFinite(step.Loss), $"Triplet TrainStep loss must be finite; got {step.Loss}");
        Assert.NotEmpty(step.Checkpoint.OptimizerState.Fields);
        bool moved = false;
        for (int i = 0; i < w0.Length; i++)
            if (MathF.Abs(w1[i] - w0[i]) > 1e-7f) { moved = true; break; }
        Assert.True(moved,
            "the trainable embedding weight must move (triplet-loss gradient must flow through Pow/Sum/Min/Relu/IfElse)");
    }

    /// <summary>
    /// CosineEmbeddingLoss rig-trainability via the design §5 "loss-is-the-model-tail"
    /// recipe (cosine-embedding-loss design §9.6, Recipe A fallback). A
    /// TrainingRig.FromScratch around NNCosineEmbeddingRigModel (a [4,2]=[2N,D] input
    /// split into x1/x2 blocks, a SHARED trainable Linear embedding on each, returning
    /// the scalar cosine loss with y=−1 and margin=−1 so the hinge relu(cos−margin) is
    /// ALWAYS active) + the NNIdentityScalarLoss adapter + Adam, runs ONE TrainStep
    /// and asserts (a) the loss is finite and (b) the trainable Linear embedding weight
    /// MOVED (≥1 element) — i.e. the gradient flows through Reduce(Sum/L2)/Clip/Where/
    /// Relu/division back into the embedding. Mirrors TestTripletMarginTrainStepMovesWeight.
    /// </summary>
    [Fact]
    public void TestCosineEmbeddingTrainStepMovesWeight()
    {
        // [2N, D] = [4, 2]: rows 0-1 are x1, rows 2-3 are x2.
        long[] xShape = [4L, 2L];
        var xData = new float[]
        {
            0.5f, 1.0f,   -0.5f, -1.0f,   // x1
            0.6f, 1.1f,   -0.4f, -0.9f,   // x2
        };

        var sample = new NamedModelParam[]
        {
            new TensorDataModelParam("x", ModelParamType.InputParam, TensorData(xShape, xData)),
        };

        var rig = TrainingRig.FromScratch(
            NNCosineEmbeddingRigModel.ComputationGraph, NNIdentityScalarLoss.ComputationGraph,
            AdamOptimizer.ComputationGraph, sample, 0.01f, 0.9f, 0.999f, 1e-8f);

        var initial = rig.CreateDefaultCheckpoint();
        Assert.NotEmpty(rig.TrainableParamStructDef.Fields);

        // Locate the Linear embedding weight (the rank-2 [out, in] = [2, 2] param).
        string? wName = null;
        foreach (var f in rig.TrainableParamStructDef.Fields)
            if (initial.TrainableParams.Fields[f.Name] is TensorData td && td.Shape.Dims.Length == 2)
            { wName = f.Name; break; }
        Assert.NotNull(wName);
        float[] w0 = Floats(initial.TrainableParams.Fields[wName]);
        Assert.True(w0.Length >= 2, $"expected the embedding weight to have ≥2 elements; got {w0.Length}");

        var inputBatch = MakeBatch("x", "ModelInput", TensorData(xShape, xData));
        var targetBatch = MakeBatch("targets", "Target", TensorData([1L], new float[1]));

        var ctx = new ComputeContext();
        var compiled = ctx.Compile(rig.TrainingStepPureGraph);
        var step = rig.TrainStep(initial, inputBatch, targetBatch, compiled);
        float[] w1 = Floats(step.Checkpoint.TrainableParams.Fields[wName]);

        Assert.True(float.IsFinite(step.Loss), $"Cosine TrainStep loss must be finite; got {step.Loss}");
        Assert.NotEmpty(step.Checkpoint.OptimizerState.Fields);
        bool moved = false;
        for (int i = 0; i < w0.Length; i++)
            if (MathF.Abs(w1[i] - w0[i]) > 1e-7f) { moved = true; break; }
        Assert.True(moved,
            "the trainable embedding weight must move (cosine-loss gradient must flow through Reduce(Sum/L2)/Clip/Where/Relu/division)");
    }

    /// <summary>Builds a rig for <paramref name="optimizerGraph"/>, asserts the trainable-param
    /// and optimizer-state structs are non-empty, then runs ONE TrainStep on a fixed batch and
    /// asserts the loss is finite and at least one param actually moved — exercising the optimizer's
    /// state threading (e.g. NAdam's two scalar states step + muProduct, RAdam's Where path through
    /// the autodiff/scheduler) end-to-end, not just construction.</summary>
    private static void CoverTrainStepMovesParam(
        FastComputationGraph optimizerGraph, params HyperValue[] hyperparams)
    {
        var input = new float[] { 1f, 2f, 3f, 4f };
        var target = new float[] { 0f, 0f, 0f, 0f };
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, optimizerGraph,
            new NamedModelParam[]
            {
                new TensorDataModelParam("input", ModelParamType.InputParam, TensorData([4L], input)),
            },
            hyperparams);

        var initial = rig.CreateDefaultCheckpoint();
        Assert.NotEmpty(rig.TrainableParamStructDef.Fields);

        var ctx = new ComputeContext();
        var compiled = ctx.Compile(rig.TrainingStepPureGraph);
        string wName = rig.TrainableParamStructDef.Fields[0].Name;
        float w0 = Floats(initial.TrainableParams.Fields[wName])[0];

        var step = rig.TrainStep(initial,
            MakeBatch("input", "ModelInput", TensorData([4L], input)),
            MakeBatch("targets", "Target", TensorData([4L], target)),
            compiled);
        float w1 = Floats(step.Checkpoint.TrainableParams.Fields[wName])[0];

        Assert.True(float.IsFinite(step.Loss), $"TrainStep loss must be finite; got {step.Loss}");
        Assert.NotEmpty(step.Checkpoint.OptimizerState.Fields); // state threaded between steps
        Assert.True(MathF.Abs(w1 - w0) > 1e-7f, $"param must move; w0={w0}, w1={w1}");
    }

    /// <summary>Like <see cref="CoverTrainStepMovesParam"/> but for a rank-≥2 trainable param:
    /// runs ONE TrainStep of <paramref name="modelGraph"/> + L2Loss + the optimizer on a fixed
    /// <paramref name="inShape"/> batch and asserts the loss is finite and at least one element of
    /// the (multi-dim) param actually moved. This is the Adafactor rank-agnosticism gate — it
    /// proves the reduce-all RMS(θ)/RMS(U) scalars work over a non-scalar param.</summary>
    private static void CoverTrainStepMovesNonScalarParam(
        FastComputationGraph modelGraph, FastComputationGraph optimizerGraph,
        long[] inShape, float[] input, float[] target,
        params HyperValue[] hyperparams)
    {
        var rig = TrainingRig.FromScratch(
            modelGraph, L2Loss.ComputationGraph, optimizerGraph,
            new NamedModelParam[]
            {
                new TensorDataModelParam("input", ModelParamType.InputParam, TensorData(inShape, input)),
            },
            hyperparams);

        var initial = rig.CreateDefaultCheckpoint();
        Assert.NotEmpty(rig.TrainableParamStructDef.Fields);

        var ctx = new ComputeContext();
        var compiled = ctx.Compile(rig.TrainingStepPureGraph);
        string wName = rig.TrainableParamStructDef.Fields[0].Name;
        float[] w0 = Floats(initial.TrainableParams.Fields[wName]);

        var step = rig.TrainStep(initial,
            MakeBatch("input", "ModelInput", TensorData(inShape, input)),
            MakeBatch("targets", "Target", TensorData(inShape, target)),
            compiled);
        float[] w1 = Floats(step.Checkpoint.TrainableParams.Fields[wName]);

        Assert.True(float.IsFinite(step.Loss), $"TrainStep loss must be finite; got {step.Loss}");
        Assert.NotEmpty(step.Checkpoint.OptimizerState.Fields);
        Assert.True(w0.Length >= 2, $"expected a rank-≥2 param with ≥2 elements; got {w0.Length}");
        bool moved = false;
        for (int i = 0; i < w0.Length; i++)
            if (MathF.Abs(w1[i] - w0[i]) > 1e-7f) { moved = true; break; }
        Assert.True(moved, "at least one element of the multi-dim param must move");
    }

    /// <summary>Rig-construction + one-TrainStep smoke for the four just-landed optimizers
    /// (per-param state: Adamax 3 — m/u/step; NAdam 4 — m/v/step/muProduct; RAdam 3 — m/v/step;
    /// Adadelta 2 — squareAvg/accDelta). The TrainStep gates the harder rig threading:
    /// NAdam's two scalar states (step + muProduct, the latter seeded at 1 via OptimizerScalarOnes)
    /// and RAdam's runtime scalar <c>Where</c> through the autodiff/scheduler. Right hyper counts:
    /// Adamax 4, NAdam 5, RAdam 4, Adadelta 3.</summary>
    [Trait("Domain", "Training")]
    [Trait("Purpose", "Coverage")]
    [Fact]
    public void TestLandedOptimizersRigCoverage()
    {
        // Construction-only smoke (state structs non-empty).
        CoverFromScratch(ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            AdamaxOptimizer.ComputationGraph, [4L], 0.002f, 0.9f, 0.999f, 1e-8f);
        CoverFromScratch(ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            NAdamOptimizer.ComputationGraph, [4L], 0.002f, 0.9f, 0.999f, 1e-8f, 0.004f);
        CoverFromScratch(ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            RAdamOptimizer.ComputationGraph, [4L], 0.001f, 0.9f, 0.999f, 1e-8f);
        CoverFromScratch(ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            AdadeltaOptimizer.ComputationGraph, [4L], 1.0f, 0.9f, 1e-6f);

        // One-TrainStep smoke (finite loss + param moves + state threaded). NAdam and RAdam
        // are the gate for the two-scalar-state and Where-path rig threading respectively.
        CoverTrainStepMovesParam(AdamaxOptimizer.ComputationGraph, 0.002f, 0.9f, 0.999f, 1e-8f);
        CoverTrainStepMovesParam(NAdamOptimizer.ComputationGraph, 0.002f, 0.9f, 0.999f, 1e-8f, 0.004f);
        CoverTrainStepMovesParam(RAdamOptimizer.ComputationGraph, 0.001f, 0.9f, 0.999f, 1e-8f);
        CoverTrainStepMovesParam(AdadeltaOptimizer.ComputationGraph, 1.0f, 0.9f, 1e-6f);
    }

    /// <summary>
    /// Adam bias correction numerics: at t=1, m_hat = grad and v_hat = grad^2
    /// exactly, so the first step moves the weight by lr * g / (|g| + eps) ≈ lr
    /// regardless of the gradient magnitude. Without bias correction the step
    /// would be ≈ 3.16 * lr (0.1 / sqrt(0.001)), so this discriminates sharply.
    /// </summary>
    [Fact]
    public void TestAdamBiasCorrectionFirstStepSize()
    {
        const float lr = 0.001f;
        var rig = TrainingRig.FromScratch(
            ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, AdamOptimizer.ComputationGraph,
            new NamedModelParam[]
            {
                new TensorDataModelParam("input", ModelParamType.InputParam,
                    TensorData([4L], new float[] { 1f, 2f, 3f, 4f })),
            },
            lr, 0.9f, 0.999f, 1e-8f);

        var initial = rig.CreateDefaultCheckpoint();
        var ctx = new ComputeContext();
        var compiled = ctx.Compile(rig.TrainingStepPureGraph);

        var inputBatch = MakeBatch("input", "ModelInput", TensorData([4L], new float[] { 1f, 2f, 3f, 4f }));
        var targetBatch = MakeBatch("targets", "Target", TensorData([4L], new float[] { 0f, 0f, 0f, 0f }));

        string wName = rig.TrainableParamStructDef.Fields[0].Name;
        float w0 = Floats(initial.TrainableParams.Fields[wName])[0];
        var step = rig.TrainStep(initial, inputBatch, targetBatch, compiled);
        float w1 = Floats(step.Checkpoint.TrainableParams.Fields[wName])[0];

        // Gradient is large (15), so |Δw| must be ≈ lr to within eps effects.
        Assert.True(MathF.Abs((w0 - w1) - lr) < 5e-5f,
            $"bias-corrected first Adam step must be ≈ lr={lr}; got Δw={w0 - w1}");
        Assert.NotEmpty(step.Checkpoint.OptimizerState.Fields); // m/v/step state flows
    }

    /// <summary>
    /// End-to-end (a): Linear + L2 regression with Adam on the wide
    /// (<c>[32,2] → [32,400]</c>), perfectly realizable fixture
    /// (<see cref="MakeWideRegressionData"/>). The pass conditions are deliberately
    /// <em>platform-invariant</em>, unlike the old per-step-monotonicity + ratio checks
    /// that were tuned to one machine's lucky random init (see the git history of this
    /// test): the starting loss is bounded to the narrow band that random init produces
    /// across <em>any</em> platform's seeded RNG, and the final loss is checked against
    /// an <em>absolute</em> target rather than a ratio of the (platform-varying) start.
    ///
    /// <para>Both bounds are derived from a Monte-Carlo characterization of this exact
    /// model + data (800 weights, init <c>U(±√3)</c>): the 400 independent output rows
    /// concentrate the random-init starting loss to mean ≈ 1.184, σ ≈ 0.046, so the
    /// 6.81σ band — the central 1-in-100-billion range — is [0.871, 1.497]; and because
    /// the problem is convex with a realizable global minimum of 0, every init converges
    /// to that floor (≈1e-6 by step 150 at lr=0.05), independent of platform.</para>
    /// </summary>
    [Fact]
    public void TestLinearRegressionWithAdamConverges()
    {
        var (inputData, targetData) = MakeWideRegressionData();

        var rig = TrainingRig.FromScratch(
            NNWideRegressionModel.ComputationGraph, L2Loss.ComputationGraph, AdamOptimizer.ComputationGraph,
            new NamedModelParam[] { new TensorDataModelParam("input", ModelParamType.InputParam, inputData) },
            0.05f, 0.9f, 0.999f, 1e-8f);

        var ctx = new ComputeContext();
        var compiled = ctx.Compile(rig.TrainingStepPureGraph);
        var inputBatch = MakeBatch("input", "ModelInput", inputData);
        var targetBatch = MakeBatch("targets", "Target", targetData);

        var ckpt = rig.CreateDefaultCheckpoint();
        var losses = new List<float>();
        for (int i = 0; i < 150; i++)
        {
            var step = rig.TrainStep(ckpt, inputBatch, targetBatch, compiled);
            losses.Add(step.Loss);
            ckpt = step.Checkpoint;
        }

        Assert.All(losses, l => Assert.True(float.IsFinite(l)));
        AssertWideStartLossInBand(losses[0]);
        // Converged to the realizable global minimum (0): an ABSOLUTE target, not a ratio
        // of the platform-dependent starting loss. Simulated end loss ≈ 1e-6; 1e-2 is a
        // ~4-order margin that swallows any cross-platform / optimizer-FP variation.
        Assert.True(losses[^1] < 1e-2f,
            $"Adam should converge below the absolute target 1e-2 in 150 steps; got {losses[^1]}");
    }

    /// <summary>
    /// End-to-end (b): tiny conv net (Conv2d → ReLU → GlobalAvgPool) with
    /// CrossEntropyLoss + SGDMomentum on a fixed 4-sample batch (class 0 =
    /// left-half active, class 1 = right-half active). The loss must decrease.
    /// </summary>
    [Fact]
    public void TestTinyConvNetCrossEntropySgdMomentumConverges()
    {
        // 4 samples of [1, 4, 4]: rows of each 4x4 image; class 0 lights the
        // left two columns, class 1 the right two, at two different intensities.
        var vals = new float[4 * 16];
        for (int s = 0; s < 4; s++)
        {
            float intensity = s < 2 ? 1f : 0.6f;
            bool rightHalf = (s % 2) == 1;
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++)
                    vals[s * 16 + r * 4 + c] = (rightHalf ? c >= 2 : c < 2) ? intensity : 0f;
        }
        var inputData = TensorData([4L, 1L, 4L, 4L], vals);
        var targetData = TensorData([4L], new long[] { 0L, 1L, 0L, 1L });

        var rig = TrainingRig.FromScratch(
            NNTinyConvClassifier.ComputationGraph, CrossEntropyLoss.ComputationGraph,
            SGDMomentumOptimizer.ComputationGraph,
            new NamedModelParam[] { new TensorDataModelParam("input", ModelParamType.InputParam, inputData) },
            0.2f, 0.9f);

        var ctx = new ComputeContext();
        var compiled = ctx.Compile(rig.TrainingStepPureGraph);
        var inputBatch = MakeBatch("input", "ModelInput", inputData);
        var targetBatch = MakeBatch("targets", "Target", targetData);

        var ckpt = rig.CreateDefaultCheckpoint();
        var losses = new List<float>();
        for (int i = 0; i < 15; i++)
        {
            var step = rig.TrainStep(ckpt, inputBatch, targetBatch, compiled);
            losses.Add(step.Loss);
            ckpt = step.Checkpoint;
        }

        Assert.All(losses, l => Assert.True(float.IsFinite(l)));
        Assert.True(losses[^1] < losses[0],
            $"cross-entropy loss should decrease over 15 SGDMomentum steps; got {losses[0]} → {losses[^1]}");
    }

    /// <summary>
    /// Generalized Convolution trainability smoke (design §7-10): a tiny groups:1,
    /// explicit-pad, Zeros-mode Convolution.Conv (the supported differentiable corner — 2-D
    /// weight gradient, explicit pads, zeros mode) → ReLU → GlobalAvgPool classifier builds a
    /// rig via TrainingRig.FromScratch and one CrossEntropy + SGDMomentum TrainStep yields a
    /// finite loss while moving at least one trainable param (so a gradient flowed through the
    /// generalized conv path). Mirrors TestTinyConvNetCrossEntropySgdMomentumConverges.
    /// </summary>
    [Fact]
    public void TestGeneralizedConvTrainStepFlows()
    {
        // 4 samples of [1, 4, 4]: class 0 lights the left two columns, class 1 the right two.
        var vals = new float[4 * 16];
        for (int s = 0; s < 4; s++)
        {
            float intensity = s < 2 ? 1f : 0.6f;
            bool rightHalf = (s % 2) == 1;
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++)
                    vals[s * 16 + r * 4 + c] = (rightHalf ? c >= 2 : c < 2) ? intensity : 0f;
        }
        var inputData = TensorData([4L, 1L, 4L, 4L], vals);
        var targetData = TensorData([4L], new long[] { 0L, 1L, 0L, 1L });

        var rig = TrainingRig.FromScratch(
            ConvGeneralizedTrainModel.ComputationGraph, CrossEntropyLoss.ComputationGraph,
            SGDMomentumOptimizer.ComputationGraph,
            new NamedModelParam[] { new TensorDataModelParam("input", ModelParamType.InputParam, inputData) },
            0.2f, 0.9f);

        // Building the rig is itself meaningful coverage: the generalized conv differentiates.
        Assert.NotEmpty(rig.TrainableParamStructDef.Fields);

        var ctx = new ComputeContext();
        var compiled = ctx.Compile(rig.TrainingStepPureGraph);
        var initial = rig.CreateDefaultCheckpoint();
        var step = rig.TrainStep(initial,
            MakeBatch("input", "ModelInput", inputData),
            MakeBatch("targets", "Target", targetData),
            compiled);

        Assert.True(float.IsFinite(step.Loss), $"loss must be finite; got {step.Loss}");

        // At least one trainable param must move (gradient flowed through the generalized conv).
        bool anyMoved = false;
        foreach (var field in rig.TrainableParamStructDef.Fields)
        {
            var before = Floats(initial.TrainableParams.Fields[field.Name]);
            var after = Floats(step.Checkpoint.TrainableParams.Fields[field.Name]);
            if (before.Zip(after).Any(p => MathF.Abs(p.First - p.Second) > 1e-7f)) { anyMoved = true; break; }
        }
        Assert.True(anyMoved, "no trainable param moved — no gradient flowed through the generalized Conv path");
    }

    /// <summary>
    /// BatchNorm2d eval path: (1) the forward value matches the closed form
    /// (running stats at init: y = x / sqrt(1 + eps), then per-channel mean),
    /// checked through the rig's loss; (2) gamma AND beta both receive
    /// non-zero gradients through the eval path (every trainable field moves);
    /// (3) eval passes leave the running statistics untouched.
    /// </summary>
    [Fact]
    public void TestBatchNormEvalGradientFlowAndClosedForm()
    {
        var vals = Enumerable.Range(0, 24).Select(i => (float)i).ToArray();
        var inputData = TensorData([2L, 3L, 2L, 2L], vals);
        var targetVals = new float[] { 1f, 2f, 3f };
        var targetData = TensorData([3L], targetVals);

        var rig = TrainingRig.FromScratch(
            NNBatchNormEvalGradModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            new NamedModelParam[] { new TensorDataModelParam("input", ModelParamType.InputParam, inputData) },
            0.5f);

        var ctx = new ComputeContext();
        var compiled = ctx.Compile(rig.TrainingStepPureGraph);
        var initial = rig.CreateDefaultCheckpoint();
        var step = rig.TrainStep(initial,
            MakeBatch("input", "ModelInput", inputData),
            MakeBatch("targets", "Target", targetData),
            compiled);

        // Closed-form loss: pred_c = mean_{n,h,w}(x) / sqrt(1 + eps).
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
        Assert.True(MathF.Abs(step.Loss - expectedLoss) < 1e-3f,
            $"BatchNorm2d eval output must match closed form; expected loss {expectedLoss}, got {step.Loss}");

        // Both gamma and beta must receive non-zero gradients through the eval path.
        Assert.Equal(2, rig.TrainableParamStructDef.Fields.Length);
        foreach (var field in rig.TrainableParamStructDef.Fields)
        {
            var before = Floats(initial.TrainableParams.Fields[field.Name]);
            var after = Floats(step.Checkpoint.TrainableParams.Fields[field.Name]);
            Assert.True(before.Zip(after).Any(p => MathF.Abs(p.First - p.Second) > 1e-7f),
                $"trainable param '{field.Name}' did not move — no gradient flowed through the BN eval path");
        }

        // Eval mode must not touch the running statistics.
        foreach (var field in rig.ModelStateDef.Fields)
        {
            var before = Floats(initial.ModelState.Fields[field.Name]);
            var after = Floats(step.Checkpoint.ModelState.Fields[field.Name]);
            Assert.True(before.Zip(after).All(p => MathF.Abs(p.First - p.Second) < 1e-7f),
                $"running stat '{field.Name}' changed during an eval-mode pass");
        }
    }

    /// <summary>
    /// BatchNorm2d training path: the normalized output has ~zero per-channel
    /// mean, so against zero targets the loss is ~0; and the training pass
    /// EMA-updates the running statistics (both state fields move).
    /// </summary>
    [Fact]
    public void TestBatchNormTrainModeNormalizesAndUpdatesRunningStats()
    {
        var vals = Enumerable.Range(0, 24).Select(i => (float)i).ToArray();
        var inputData = TensorData([2L, 3L, 2L, 2L], vals);
        var targetData = TensorData([3L], new float[] { 0f, 0f, 0f });

        var rig = TrainingRig.FromScratch(
            NNBatchNormTrainGradModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            new NamedModelParam[] { new TensorDataModelParam("input", ModelParamType.InputParam, inputData) },
            0.1f);

        var ctx = new ComputeContext();
        var compiled = ctx.Compile(rig.TrainingStepPureGraph);
        var initial = rig.CreateDefaultCheckpoint();
        var step = rig.TrainStep(initial,
            MakeBatch("input", "ModelInput", inputData),
            MakeBatch("targets", "Target", targetData),
            compiled);

        // Per-channel mean of the batch-normalized output is 0 (gamma=1, beta=0).
        Assert.True(step.Loss < 1e-6f,
            $"training-mode BN output must have ~zero per-channel mean; got loss {step.Loss}");

        // The training pass must EMA-update both running statistics.
        Assert.NotEmpty(rig.ModelStateDef.Fields);
        foreach (var field in rig.ModelStateDef.Fields)
        {
            var before = Floats(initial.ModelState.Fields[field.Name]);
            var after = Floats(step.Checkpoint.ModelState.Fields[field.Name]);
            Assert.True(before.Zip(after).Any(p => MathF.Abs(p.First - p.Second) > 1e-7f),
                $"running stat '{field.Name}' was not updated by a training-mode pass");
        }
    }

    // -----------------------------------------------------------------------
    //  Generalized rank-generic BatchNorm coverage (design §7 groups A–G).
    //  Every BatchNorm graph carries StateUpdate links, so ALL of these run
    //  through the rig (not AutoTest) — even the "pure" eval closed-form checks.
    //  Closed-form / alias-equivalence models output (y − reference); a zero
    //  target makes the L2 loss the mean squared elementwise deviation, so
    //  loss ≈ 0 pins exact per-element equality. (See the module docs in
    //  NNLibraryTestModules.cs for what each model computes.)
    // -----------------------------------------------------------------------

    /// <summary>Runs a single L2 + SGD TrainStep of <paramref name="modelGraph"/> on one fixed
    /// batch and returns the resulting step (Loss + Checkpoint). Used by the closed-form /
    /// alias-equivalence BatchNorm checks whose models emit a residual against a zero target.</summary>
    private static TrainingStepResult RunResidualStep(
        FastComputationGraph modelGraph, long[] inShape, float[] input, long[] outShape, float lr = 0f)
    {
        var inputData = TensorData(inShape, input);
        var rig = TrainingRig.FromScratch(modelGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            new NamedModelParam[] { new TensorDataModelParam("input", ModelParamType.InputParam, inputData) }, lr);
        var ctx = new ComputeContext();
        var compiled = ctx.Compile(rig.TrainingStepPureGraph);
        long outTotal = 1;
        foreach (var d in outShape) outTotal *= d;
        var targetData = TensorData(outShape, new float[outTotal]);
        return rig.TrainStep(rig.CreateDefaultCheckpoint(),
            MakeBatch("input", "ModelInput", inputData),
            MakeBatch("targets", "Target", targetData),
            compiled);
    }

    /// <summary>[i * scale + offset for i in 0..total) — varied, non-degenerate inputs.</summary>
    private static float[] Ramp(long total, float scale = 1f, float offset = 0f)
        => Enumerable.Range(0, (int)total).Select(i => i * scale + offset).ToArray();

    /// <summary>
    /// Group A — rank generality (eval path). At init (running 0/1, gamma=1, beta=0)
    /// the eval output equals x / sqrt(1 + eps) for EVERY element at ranks 2, 3, 4, 5
    /// ([N,C], [N,C,L], [N,C,H,W], [N,C,D,H,W]). The residual model y − x/sqrt(1+eps)
    /// against a zero target gives loss ≈ 0 only if every element matches. Pins the
    /// lifted rank-2-only regression AND the new rank-3 [N,C,L] and rank-5 paths.
    /// </summary>
    [Fact]
    public void TestBatchNormRankGeneralityEvalClosedForm()
    {
        // rank 2 [2,3]
        Assert.True(RunResidualStep(NNBatchNormEvalRank2ClosedForm.ComputationGraph,
            [2L, 3L], Ramp(6, 0.5f, -1f), [2L, 3L]).Loss < 1e-8f);
        // rank 3 [2,3,4] — the form the old BatchNorm1d rejected
        Assert.True(RunResidualStep(NNBatchNormEvalRank3ClosedForm.ComputationGraph,
            [2L, 3L, 4L], Ramp(24, 0.25f, -2f), [2L, 3L, 4L]).Loss < 1e-8f);
        // rank 4 [2,3,4,4]
        Assert.True(RunResidualStep(NNBatchNormEvalRank4ClosedForm.ComputationGraph,
            [2L, 3L, 4L, 4L], Ramp(96, 0.1f, -3f), [2L, 3L, 4L, 4L]).Loss < 1e-8f);
        // rank 5 [2,3,2,2,2] — the new rank-5 path
        Assert.True(RunResidualStep(NNBatchNormEvalRank5ClosedForm.ComputationGraph,
            [2L, 3L, 2L, 2L, 2L], Ramp(48, 0.2f, -2f), [2L, 3L, 2L, 2L, 2L]).Loss < 1e-8f);
    }

    /// <summary>
    /// Group G — alias equivalence. BatchNorm2d/1d/3d.Call must equal
    /// BatchNorm.Call(.., affine:true, track:true) bit-for-bit: BatchNorm2d on
    /// [2,3,4,4]; BatchNorm1d on BOTH [N,C] and [N,C,L] (the latter formerly
    /// rejected); BatchNorm3d on [N,C,D,H,W]. The (alias − generic) residual model
    /// against a zero target gives loss ≈ 0 only if they are identical.
    /// </summary>
    [Fact]
    public void TestBatchNormAliasEquivalence()
    {
        Assert.True(RunResidualStep(NNBatchNorm2dAliasEquiv.ComputationGraph,
            [2L, 3L, 4L, 4L], Ramp(96, 0.1f, -3f), [2L, 3L, 4L, 4L]).Loss < 1e-10f);
        Assert.True(RunResidualStep(NNBatchNorm1dAliasEquivRank2.ComputationGraph,
            [2L, 3L], Ramp(6, 0.5f, -1f), [2L, 3L]).Loss < 1e-10f);
        Assert.True(RunResidualStep(NNBatchNorm1dAliasEquivRank3.ComputationGraph,
            [2L, 3L, 4L], Ramp(24, 0.25f, -2f), [2L, 3L, 4L]).Loss < 1e-10f);
        Assert.True(RunResidualStep(NNBatchNorm3dAliasEquiv.ComputationGraph,
            [2L, 3L, 2L, 2L, 2L], Ramp(48, 0.2f, -2f), [2L, 3L, 2L, 2L, 2L]).Loss < 1e-10f);
    }

    /// <summary>
    /// Group D — affine on/off. (1) With affine:false, the eval output equals the bare
    /// normalizer x/sqrt(1+eps) (no gamma/beta) — residual loss ≈ 0. (2) The affine bit
    /// gates whether gamma/beta receive gradient: with affine:TRUE the BN exposes 2
    /// trainable params (gamma + beta), both of which move under a train step (proven in
    /// TestBatchNormEvalGradientFlowAndClosedForm); with affine:FALSE gamma/beta sit on the
    /// dead branch and receive NO gradient, so the only trainable param of an otherwise
    /// identical model is the upstream scalar weight (gamma/beta contribute none).
    /// </summary>
    [Fact]
    public void TestBatchNormAffineOnOff()
    {
        // affine:false eval output == bare x/sqrt(1+eps).
        Assert.True(RunResidualStep(NNBatchNormAffineFalseClosedForm.ComputationGraph,
            [2L, 3L, 2L, 2L], Ramp(24, 0.5f, -3f), [2L, 3L, 2L, 2L]).Loss < 1e-8f);

        // affine:false ⇒ gamma/beta receive no gradient ⇒ the model's only live trainable
        // param is the scalar pre-weight (the affine:true NNBatchNormEvalGradModel has 2).
        var inputData = TensorData([2L, 3L, 2L, 2L], Ramp(24));
        var rig = TrainingRig.FromScratch(
            NNBatchNormAffineFalseEvalGradModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            new NamedModelParam[] { new TensorDataModelParam("input", ModelParamType.InputParam, inputData) }, 0.5f);
        Assert.Equal(1, rig.TrainableParamStructDef.Fields.Length);

        // Cross-check: the affine:TRUE eval model exposes exactly gamma + beta (2 params).
        var affineTrueRig = TrainingRig.FromScratch(
            NNBatchNormEvalGradModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            new NamedModelParam[] { new TensorDataModelParam("input", ModelParamType.InputParam, inputData) }, 0.5f);
        Assert.Equal(2, affineTrueRig.TrainableParamStructDef.Fields.Length);
    }

    /// <summary>
    /// Design §7-2(b) — InstanceNorm/GroupNorm affine param-count discrimination, mirroring
    /// <see cref="TestBatchNormAffineOnOff"/>. The affine bit gates whether γ/β receive gradient:
    /// with affine:FALSE they sit on the dead branch and are pruned, so an otherwise-identical
    /// model's ONLY trainable param is the upstream scalar pre-weight (1 field); with affine:TRUE
    /// γ and β survive, so the model exposes 3 trainable params (scalar weight + γ + β). Checked
    /// for both InstanceNorm and GroupNorm. (These two norms are state-free, so the rig is only
    /// needed for this param-count count — the value/equivalence checks run via AutoTest in
    /// <see cref="NNLibraryCoverageTests.TestInstanceGroupNormCoverage"/>.)
    /// </summary>
    [Fact]
    public void TestInstanceGroupNormAffineOnOff()
    {
        var inputData = TensorData([2L, 4L, 3L, 3L], Ramp(72));

        NamedModelParam[] Inputs() => new NamedModelParam[]
            { new TensorDataModelParam("input", ModelParamType.InputParam, inputData) };

        // InstanceNorm: affine:false ⇒ 1 trainable param (scalar weight); affine:true ⇒ 3 (weight + γ + β).
        var inAffineFalse = TrainingRig.FromScratch(
            NNInstanceNormAffineFalseParamModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            Inputs(), 0.5f);
        Assert.Equal(1, inAffineFalse.TrainableParamStructDef.Fields.Length);

        var inAffineTrue = TrainingRig.FromScratch(
            NNInstanceNormAffineTrueParamModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            Inputs(), 0.5f);
        Assert.Equal(3, inAffineTrue.TrainableParamStructDef.Fields.Length);

        // GroupNorm: same discrimination.
        var gnAffineFalse = TrainingRig.FromScratch(
            NNGroupNormAffineFalseParamModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            Inputs(), 0.5f);
        Assert.Equal(1, gnAffineFalse.TrainableParamStructDef.Fields.Length);

        var gnAffineTrue = TrainingRig.FromScratch(
            NNGroupNormAffineTrueParamModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            Inputs(), 0.5f);
        Assert.Equal(3, gnAffineTrue.TrainableParamStructDef.Fields.Length);
    }

    /// <summary>
    /// Group B — train-path normalization + EMA at rank 3 ([N,C,L]) and rank 5
    /// ([N,C,D,H,W]). Per-channel mean of the batch-normalized output ≈ 0 ⇒ loss ≈ 0
    /// against zero targets; and a training pass moves BOTH ModelState fields.
    /// </summary>
    [Fact]
    public void TestBatchNormTrainNormalizationRank3And5()
    {
        AssertTrainNormalizesAndMovesState(NNBatchNormTrainRank3Model.ComputationGraph,
            [2L, 3L, 4L], Ramp(24, 1f, -5f), [3L]);
        AssertTrainNormalizesAndMovesState(NNBatchNormTrainRank5Model.ComputationGraph,
            [2L, 3L, 2L, 2L, 2L], Ramp(48, 1f, -10f), [3L]);
    }

    /// <summary>
    /// Group B at rank 2 — train-path normalization on [N,C] (reduceAxes={0}, empty
    /// spatial range, the original BatchNorm1d MLP/tabular use case). Per-channel mean of
    /// the batch-normalized output ≈ 0 ⇒ loss ≈ 0 vs zero targets, and a training pass
    /// moves BOTH ModelState fields. The existing rank-2 coverage only exercised the eval
    /// path (running stats 0/1), never this batch-statistics reduction.
    /// </summary>
    [Trait("Domain", "Training")]
    [Trait("Purpose", "Coverage")]
    [Fact]
    public void TestBatchNormTrainNormalizationRank2()
    {
        AssertTrainNormalizesAndMovesState(NNBatchNormTrainRank2Model.ComputationGraph,
            [4L, 3L], Ramp(12, 1f, -5f), [3L]);
    }

    /// <summary>One train step: per-channel-mean output is ~0 (loss ~0 vs zero targets) and
    /// every ModelState field moves. Shared by the rank-3 / rank-5 group-B checks.</summary>
    private static void AssertTrainNormalizesAndMovesState(
        FastComputationGraph modelGraph, long[] inShape, float[] input, long[] outShape)
    {
        var inputData = TensorData(inShape, input);
        long outTotal = 1; foreach (var d in outShape) outTotal *= d;
        var rig = TrainingRig.FromScratch(modelGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            new NamedModelParam[] { new TensorDataModelParam("input", ModelParamType.InputParam, inputData) }, 0.1f);
        var ctx = new ComputeContext();
        var compiled = ctx.Compile(rig.TrainingStepPureGraph);
        var initial = rig.CreateDefaultCheckpoint();
        var step = rig.TrainStep(initial,
            MakeBatch("input", "ModelInput", inputData),
            MakeBatch("targets", "Target", TensorData(outShape, new float[outTotal])),
            compiled);

        Assert.True(step.Loss < 1e-6f,
            $"training-mode BN output must have ~zero per-channel mean; got loss {step.Loss}");
        Assert.NotEmpty(rig.ModelStateDef.Fields);
        foreach (var field in rig.ModelStateDef.Fields)
        {
            var before = Floats(initial.ModelState.Fields[field.Name]);
            var after = Floats(step.Checkpoint.ModelState.Fields[field.Name]);
            Assert.True(before.Zip(after).Any(p => MathF.Abs(p.First - p.Second) > 1e-7f),
                $"running stat '{field.Name}' was not updated by a training-mode pass");
        }
    }

    /// <summary>
    /// Group C — EMA correctness + momentum value + biased variance (analytic, generic
    /// module, lr=0 to isolate ModelState). Input [1,2,3,4] (one channel: mean 2.5,
    /// BIASED var 1.25): momentum 0.9 ⇒ ModelState [0·0.9+2.5·0.1, 1·0.9+1.25·0.1] =
    /// [0.25, 1.025]; momentum 0.5 ⇒ [0·0.5+2.5·0.5, 1·0.5+1.25·0.5] = [1.25, 1.125].
    /// A rank-3 [1,1,4] case (same channel moments) proves the generalized reduction
    /// computes the same batch stats. Pins the ONNX/Keras momentum sense and biased EMA.
    /// </summary>
    [Fact]
    public void TestBatchNormEMAMomentumAnalytic()
    {
        // rank-4 [1,1,2,2], momentum 0.9 ⇒ [0.25, 1.025]
        AssertStructIs(TrainAnalytic(NNBatchNormAnalyticMomentum09Model.ComputationGraph, SGDOptimizer.ComputationGraph,
            [0f], [1L, 1L, 2L, 2L], [1f, 2f, 3f, 4f], [1L, 1L, 2L, 2L], [0f, 0f, 0f, 0f], 1).ModelState,
            [0.25f, 1.025f], 1e-4f);
        // rank-4 [1,1,2,2], momentum 0.5 ⇒ [1.25, 1.125]
        AssertStructIs(TrainAnalytic(NNBatchNormAnalyticMomentum05Model.ComputationGraph, SGDOptimizer.ComputationGraph,
            [0f], [1L, 1L, 2L, 2L], [1f, 2f, 3f, 4f], [1L, 1L, 2L, 2L], [0f, 0f, 0f, 0f], 1).ModelState,
            [1.25f, 1.125f], 1e-4f);
        // rank-3 [1,1,4] (same channel moments), momentum 0.9 ⇒ [0.25, 1.025] —
        // proves the rank-generic {0,2} reduction matches the hand-computed moments.
        AssertStructIs(TrainAnalytic(NNBatchNormAnalyticMomentum09Model.ComputationGraph, SGDOptimizer.ComputationGraph,
            [0f], [1L, 1L, 4L], [1f, 2f, 3f, 4f], [1L, 1L, 4L], [0f, 0f, 0f, 0f], 1).ModelState,
            [0.25f, 1.025f], 1e-4f);
    }

    /// <summary>
    /// Group C at rank 2 — EMA + biased variance on the rank-2 [N,C] TRAIN path
    /// (reduceAxes={0}: reduce over the batch axis only, empty spatial range,
    /// paramShape=[1,C] — the original BatchNorm1d MLP/tabular use case). Input [2,1]
    /// values [1,3] (N=2, C=1): batch mean 2, BIASED var ((1−2)²+(3−2)²)/2 = 1; momentum
    /// 0.9, lr=0 (isolates ModelState) ⇒ [0·0.9+2·0.1, 1·0.9+1·0.1] = [0.2, 1.0]. Pins
    /// the rank-2 batch-axis-only reduction + biased var + momentum, which the existing
    /// rank-2 eval test (running stats 0/1) never exercises.
    /// </summary>
    [Trait("Domain", "Training")]
    [Trait("Purpose", "Coverage")]
    [Fact]
    public void TestBatchNormRank2TrainAnalytic()
    {
        AssertStructIs(TrainAnalytic(NNBatchNormAnalyticRank2Model.ComputationGraph, SGDOptimizer.ComputationGraph,
            [0f], [2L, 1L], [1f, 3f], [2L, 1L], [0f, 0f], 1).ModelState,
            [0.2f, 1.0f], 1e-4f);
    }

    /// <summary>
    /// Group E — track_running_stats on/off. After one train step moves the running
    /// stats (to [0.25, 1.025] for the [1,2,3,4] channel), two eval passes on the SAME
    /// eval batch with the moved stats injected: (track:true) normalizes with the MOVED
    /// running stats — (x − 0.25)/sqrt(1.025+eps); (track:false) normalizes with the eval
    /// BATCH stats — ~zero per-channel mean. The eval output is read via the L2 loss
    /// against each path's hand-computed closed-form target (loss ≈ 0 ⇒ match; a high
    /// loss against the OTHER path's target ⇒ the two genuinely differ). Both eval passes
    /// must leave ModelState untouched (StateUpdate gated by training).
    /// </summary>
    [Fact]
    public void TestBatchNormTrackRunningStatsOnOff()
    {
        const float eps = 1e-5f;
        // Train one step to move the running stats to [0.25, 1.025].
        var movedState = RunTrainAndGetState(NNBatchNormTrainFullModel.ComputationGraph,
            [1L, 1L, 2L, 2L], new float[] { 1f, 2f, 3f, 4f });
        Assert.Equal(2, movedState.Length);
        Assert.True(MathF.Abs(movedState[0] - 0.25f) < 1e-4f && MathF.Abs(movedState[1] - 1.025f) < 1e-4f,
            $"sanity: train step must move running stats to [0.25, 1.025]; got [{movedState[0]}, {movedState[1]}]");
        float runMean = movedState[0], runVar = movedState[1];

        // Eval on a DIFFERENT batch so the batch stats differ from the running stats.
        var evalInput = new float[] { 2f, 4f, 6f, 8f }; // mean 5, biased var 5
        float runInvStd = 1f / MathF.Sqrt(runVar + eps);
        var trackTrueExpected = evalInput.Select(v => (v - runMean) * runInvStd).ToArray();
        float batchInvStd = 1f / MathF.Sqrt(5f + eps);
        var trackFalseExpected = evalInput.Select(v => (v - 5f) * batchInvStd).ToArray();

        // The two closed forms must differ (else the test is vacuous).
        float refDiff = trackTrueExpected.Zip(trackFalseExpected).Sum(p => MathF.Abs(p.First - p.Second));
        Assert.True(refDiff > 1e-2f, "track:true and track:false closed forms must differ");

        // track:true eval output matches its closed form (loss ≈ 0) but NOT the other's (loss high).
        var trackTrue = EvalLossAgainstTargets(NNBatchNormEvalTrackTrueFullModel.ComputationGraph,
            [1L, 1L, 2L, 2L], evalInput, movedState, trackTrueExpected, trackFalseExpected);
        Assert.True(trackTrue.matchLoss < 1e-4f, $"track:true eval must use moved running stats; got loss {trackTrue.matchLoss}");
        Assert.True(trackTrue.mismatchLoss > 1e-2f, "track:true eval must NOT equal the batch-stat closed form");
        Assert.False(trackTrue.stateMoved, "track:true eval must leave ModelState untouched");

        // track:false eval output matches the eval-batch closed form (~zero mean) but NOT the running-stat one.
        var trackFalse = EvalLossAgainstTargets(NNBatchNormEvalTrackFalseFullModel.ComputationGraph,
            [1L, 1L, 2L, 2L], evalInput, movedState, trackFalseExpected, trackTrueExpected);
        Assert.True(trackFalse.matchLoss < 1e-4f, $"track:false eval must use eval-batch stats; got loss {trackFalse.matchLoss}");
        Assert.True(trackFalse.mismatchLoss > 1e-2f, "track:false eval must NOT equal the running-stat closed form");
        Assert.False(trackFalse.stateMoved, "track:false eval must leave ModelState untouched");
    }

    /// <summary>
    /// Group F — train vs eval stat selection. On the fixed input [1,2,3,4]: the train
    /// pass is batch-normalized (~zero per-channel mean ⇒ loss ≈ 0 vs zero target), while
    /// a track:true eval pass with the running stats moved by the train step uses the
    /// EMA'd running stats (output = (x − 0.25)/sqrt(1.025+eps)). The eval output matches
    /// that running-stat closed form (loss ≈ 0) but NOT the zero-mean batch form, so the
    /// two paths differ.
    /// </summary>
    [Fact]
    public void TestBatchNormTrainVsEvalStatSelection()
    {
        const float eps = 1e-5f;
        var input = new float[] { 1f, 2f, 3f, 4f }; // [1,1,2,2]: mean 2.5, biased var 1.25

        // Train step via a per-channel-MEAN model: the batch-normalized output has ~zero
        // per-channel mean ⇒ loss ≈ 0 vs a zero target; and the running stats move to
        // [0.25, 1.025] (reduction {0,2,3} on a single-channel [1,1,2,2] input).
        var (trainLoss, movedState) = RunTrainLossAndState(
            NNBatchNormTrainGradModel.ComputationGraph, [1L, 1L, 2L, 2L], input, [1L]);
        Assert.True(trainLoss < 1e-5f, $"train output must be ~zero per-channel mean (loss vs 0); got {trainLoss}");
        Assert.True(MathF.Abs(movedState[0] - 0.25f) < 1e-4f && MathF.Abs(movedState[1] - 1.025f) < 1e-4f,
            $"train step must move running stats to [0.25, 1.025]; got [{movedState[0]}, {movedState[1]}]");
        float runMean = movedState[0], runVar = movedState[1];

        // Eval (track:true) on the SAME input with the moved stats: uses the running stats,
        // NOT the (zero-mean) batch stats. The full-output eval model is read via the L2
        // loss against the running-stat closed form (match) and the batch form (mismatch).
        float runInvStd = 1f / MathF.Sqrt(runVar + eps);
        var evalExpected = input.Select(v => (v - runMean) * runInvStd).ToArray();
        float batchInvStd = 1f / MathF.Sqrt(1.25f + eps);
        var batchExpected = input.Select(v => (v - 2.5f) * batchInvStd).ToArray();

        Assert.True(evalExpected.Zip(batchExpected).Sum(p => MathF.Abs(p.First - p.Second)) > 1e-2f,
            "running-stat and batch-stat closed forms must differ");

        var eval = EvalLossAgainstTargets(NNBatchNormEvalTrackTrueFullModel.ComputationGraph,
            [1L, 1L, 2L, 2L], input, movedState, evalExpected, batchExpected);
        Assert.True(eval.matchLoss < 1e-4f, $"eval must use the post-step running stats; got loss {eval.matchLoss}");
        Assert.True(eval.mismatchLoss > 1e-2f, "eval (running stats) must differ from the train (batch) output");
    }

    // -- Helpers for groups E/F: train-then-eval with cross-rig ModelState injection.
    //    Both train and eval models use the generic BatchNorm with the same channel
    //    count, so their ModelState field layout is identical and the moved running
    //    stats can be injected into a fresh eval checkpoint. The eval output is read
    //    indirectly through the L2 loss against a hand-computed closed-form target.

    /// <summary>One train step of a full-output BN model (output shape == input shape);
    /// returns the moved ModelState (flattened).</summary>
    private static float[] RunTrainAndGetState(FastComputationGraph modelGraph, long[] inShape, float[] input)
        => RunTrainLossAndState(modelGraph, inShape, input, inShape).state;

    /// <summary>One train step of a BN model against a ZERO target of shape <paramref name="outShape"/>;
    /// returns (loss, moved ModelState). For a per-channel-mean model, loss ≈ 0 ⇒ the batch-normalized
    /// output is ~zero per-channel mean.</summary>
    private static (float loss, float[] state) RunTrainLossAndState(
        FastComputationGraph modelGraph, long[] inShape, float[] input, long[] outShape)
    {
        long outTotal = 1; foreach (var d in outShape) outTotal *= d;
        var inputData = TensorData(inShape, input);
        var rig = TrainingRig.FromScratch(modelGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            new NamedModelParam[] { new TensorDataModelParam("input", ModelParamType.InputParam, inputData) }, 0f);
        var ctx = new ComputeContext();
        var compiled = ctx.Compile(rig.TrainingStepPureGraph);
        var step = rig.TrainStep(rig.CreateDefaultCheckpoint(),
            MakeBatch("input", "ModelInput", inputData),
            MakeBatch("targets", "Target", TensorData(outShape, new float[outTotal])),
            compiled);
        var state = rig.ModelStateDef.Fields.SelectMany(f => Floats(step.Checkpoint.ModelState.Fields[f.Name])).ToArray();
        return (step.Loss, state);
    }

    /// <summary>Runs an eval-mode BN model with <paramref name="injectedState"/> as its ModelState
    /// (via a fresh rig sharing the generic-BatchNorm state layout), reading the eval output
    /// indirectly: the L2 loss against <paramref name="matchTarget"/> (the path's expected closed
    /// form, ≈0 when it matches) and against <paramref name="mismatchTarget"/> (the OTHER path's
    /// closed form, high when they genuinely differ). Also reports whether the eval pass changed
    /// ModelState (it must NOT, since StateUpdate is gated by training).</summary>
    private static (float matchLoss, float mismatchLoss, bool stateMoved) EvalLossAgainstTargets(
        FastComputationGraph modelGraph, long[] inShape, float[] input, float[] injectedState,
        float[] matchTarget, float[] mismatchTarget)
    {
        var inputData = TensorData(inShape, input);

        float LossAgainst(float[] target, out float[] stateAfter)
        {
            var rig = TrainingRig.FromScratch(modelGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
                new NamedModelParam[] { new TensorDataModelParam("input", ModelParamType.InputParam, inputData) }, 0f);
            var compiled = new ComputeContext().Compile(rig.TrainingStepPureGraph);
            var fresh = rig.CreateDefaultCheckpoint();
            var injected = InjectModelState(fresh.ModelState, injectedState);
            var ckpt = new TrainingCheckpoint(fresh.TrainableParams, injected, fresh.OptimizerState, fresh.Step);
            var step = rig.TrainStep(ckpt,
                MakeBatch("input", "ModelInput", inputData),
                MakeBatch("targets", "Target", TensorData(inShape, target)),
                compiled);
            stateAfter = rig.ModelStateDef.Fields.SelectMany(f => Floats(step.Checkpoint.ModelState.Fields[f.Name])).ToArray();
            return step.Loss;
        }

        float matchLoss = LossAgainst(matchTarget, out var afterMatch);
        float mismatchLoss = LossAgainst(mismatchTarget, out _);
        bool stateMoved = injectedState.Zip(afterMatch).Any(p => MathF.Abs(p.First - p.Second) > 1e-6f);
        return (matchLoss, mismatchLoss, stateMoved);
    }

    /// <summary>Builds a ModelState struct copy of <paramref name="template"/> with field i set to
    /// the i-th value of <paramref name="values"/> (one running-stat scalar per field — valid for
    /// the single-channel analytic cases here).</summary>
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

    /// <summary>
    /// Dropout eval path: the layer is the exact identity in the training
    /// pipeline (loss equals the no-dropout closed form mean(x^2) = 7.5), and
    /// the upstream scalar weight receives the full gradient
    /// (w1 = 1 - lr * 2 * mean(x^2) = -0.5).
    /// </summary>
    [Fact]
    public void TestDropoutEvalGradientFlow()
    {
        var inputData = TensorData([4L], new float[] { 1f, 2f, 3f, 4f });
        var targetData = TensorData([4L], new float[] { 0f, 0f, 0f, 0f });

        var rig = TrainingRig.FromScratch(
            NNDropoutEvalGradModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            new NamedModelParam[] { new TensorDataModelParam("input", ModelParamType.InputParam, inputData) },
            0.1f);

        var ctx = new ComputeContext();
        var compiled = ctx.Compile(rig.TrainingStepPureGraph);
        var initial = rig.CreateDefaultCheckpoint();
        var step = rig.TrainStep(initial,
            MakeBatch("input", "ModelInput", inputData),
            MakeBatch("targets", "Target", targetData),
            compiled);

        Assert.True(MathF.Abs(step.Loss - 7.5f) < 1e-4f,
            $"eval-mode Dropout must be the identity (expected loss 7.5); got {step.Loss}");

        string wName = rig.TrainableParamStructDef.Fields[0].Name;
        float w1 = Floats(step.Checkpoint.TrainableParams.Fields[wName])[0];
        Assert.True(MathF.Abs(w1 - (-0.5f)) < 1e-3f,
            $"gradient must flow through eval-mode Dropout (expected w1 ≈ -0.5); got {w1}");
    }

    /// <summary>
    /// SpatialDropout rig train-step smoke (parallels <see cref="TestDropoutEvalGradientFlow"/>).
    /// (a) EVAL mode: the channel-wise layer is the exact identity, so w·SpatialDropout_eval(x)
    /// == w·x — the loss equals the no-dropout closed form mean(x²) = 7.5 (same 4 values, now a
    /// spatial [1,2,2] tensor: N=1, C=2, L=2) and the upstream scalar weight gets the full gradient
    /// w1 = 1 − lr·2·mean(x²) = −0.5, a deterministic value-checkable anchor.
    /// (b) TRAIN mode: a gradient flows through the channel-broadcast Mul + forward mask, so the
    /// loss is finite and ≥1 trainable param moves; no exact value (the mask is RNG).
    /// </summary>
    [Trait("Domain", "Training")]
    [Trait("Purpose", "Coverage")]
    [Fact]
    public void TestSpatialDropoutRigTrainStep()
    {
        // 4 values [1,2,3,4] as a spatial [1,2,2] tensor: mean(x²) = (1+4+9+16)/4 = 7.5.
        var inputData = TensorData([1L, 2L, 2L], new float[] { 1f, 2f, 3f, 4f });
        var targetData = TensorData([1L, 2L, 2L], new float[] { 0f, 0f, 0f, 0f });

        // (a) Eval-mode value anchor: identity ⇒ loss 7.5, w1 = 1 − 0.1·2·7.5 = −0.5.
        {
            var rig = TrainingRig.FromScratch(
                NNSpatialDropoutEvalGradModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
                new NamedModelParam[] { new TensorDataModelParam("input", ModelParamType.InputParam, inputData) },
                0.1f);

            var ctx = new ComputeContext();
            var compiled = ctx.Compile(rig.TrainingStepPureGraph);
            var initial = rig.CreateDefaultCheckpoint();
            var step = rig.TrainStep(initial,
                MakeBatch("input", "ModelInput", inputData),
                MakeBatch("targets", "Target", targetData),
                compiled);

            Assert.True(MathF.Abs(step.Loss - 7.5f) < 1e-4f,
                $"eval-mode SpatialDropout must be the identity (expected loss 7.5); got {step.Loss}");

            string wName = rig.TrainableParamStructDef.Fields[0].Name;
            float w1 = Floats(step.Checkpoint.TrainableParams.Fields[wName])[0];
            Assert.True(MathF.Abs(w1 - (-0.5f)) < 1e-3f,
                $"gradient must flow through eval-mode SpatialDropout (expected w1 ≈ -0.5); got {w1}");
        }

        // (b) Train-mode smoke: finite loss + ≥1 param moves (gradient through the channel mask). RNG ⇒ no exact value.
        {
            var rig = TrainingRig.FromScratch(
                NNSpatialDropoutTrainGradModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
                new NamedModelParam[] { new TensorDataModelParam("input", ModelParamType.InputParam, inputData) },
                0.1f);

            var ctx = new ComputeContext();
            var compiled = ctx.Compile(rig.TrainingStepPureGraph);
            var initial = rig.CreateDefaultCheckpoint();
            Assert.NotEmpty(rig.TrainableParamStructDef.Fields);

            string wName = rig.TrainableParamStructDef.Fields[0].Name;
            float w0 = Floats(initial.TrainableParams.Fields[wName])[0];

            var step = rig.TrainStep(initial,
                MakeBatch("input", "ModelInput", inputData),
                MakeBatch("targets", "Target", targetData),
                compiled);
            float w1 = Floats(step.Checkpoint.TrainableParams.Fields[wName])[0];

            Assert.True(float.IsFinite(step.Loss), $"train-mode SpatialDropout TrainStep loss must be finite; got {step.Loss}");
            Assert.True(MathF.Abs(w1 - w0) > 1e-7f,
                $"gradient must flow through train-mode SpatialDropout channel mask (param must move); w0={w0}, w1={w1}");
        }
    }

    /// <summary>
    /// AlphaDropout + FeatureAlphaDropout rig train-step smoke (parallels
    /// <see cref="TestDropoutEvalGradientFlow"/> / <see cref="TestSpatialDropoutRigTrainStep"/>).
    /// (a) AlphaDropout EVAL mode: the SELU-paired layer is the exact identity, so
    /// w·AlphaDropout_eval(x) == w·x — the loss equals the no-dropout closed form mean(x²) = 7.5
    /// and the upstream scalar weight gets the full gradient w1 = 1 − lr·2·mean(x²) = −0.5, a
    /// deterministic value-checkable anchor.
    /// (b) AlphaDropout TRAIN mode: a gradient flows through the affine renorm + elementwise
    /// forward mask, so the loss is finite and ≥1 trainable param moves; no exact value (RNG mask).
    /// (c) FeatureAlphaDropout TRAIN mode: same finite-loss + param-moves smoke through the
    /// channel-broadcast affine + mask (input is a spatial [1,2,2] = N=1,C=2,L=2 tensor).
    /// </summary>
    [Trait("Domain", "Training")]
    [Trait("Purpose", "Coverage")]
    [Fact]
    public void TestAlphaDropoutRigTrainStep()
    {
        // 4 values [1,2,3,4] as a spatial [1,2,2] tensor: mean(x²) = (1+4+9+16)/4 = 7.5.
        var inputData = TensorData([1L, 2L, 2L], new float[] { 1f, 2f, 3f, 4f });
        var targetData = TensorData([1L, 2L, 2L], new float[] { 0f, 0f, 0f, 0f });

        // (a) AlphaDropout eval-mode value anchor: identity ⇒ loss 7.5, w1 = 1 − 0.1·2·7.5 = −0.5.
        {
            var rig = TrainingRig.FromScratch(
                NNAlphaDropoutEvalGradModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
                new NamedModelParam[] { new TensorDataModelParam("input", ModelParamType.InputParam, inputData) },
                0.1f);

            var ctx = new ComputeContext();
            var compiled = ctx.Compile(rig.TrainingStepPureGraph);
            var initial = rig.CreateDefaultCheckpoint();
            var step = rig.TrainStep(initial,
                MakeBatch("input", "ModelInput", inputData),
                MakeBatch("targets", "Target", targetData),
                compiled);

            Assert.True(MathF.Abs(step.Loss - 7.5f) < 1e-4f,
                $"eval-mode AlphaDropout must be the identity (expected loss 7.5); got {step.Loss}");

            string wName = rig.TrainableParamStructDef.Fields[0].Name;
            float w1 = Floats(step.Checkpoint.TrainableParams.Fields[wName])[0];
            Assert.True(MathF.Abs(w1 - (-0.5f)) < 1e-3f,
                $"gradient must flow through eval-mode AlphaDropout (expected w1 ≈ -0.5); got {w1}");
        }

        // (b) AlphaDropout train-mode smoke: finite loss + ≥1 param moves (gradient through affine + mask).
        {
            var rig = TrainingRig.FromScratch(
                NNAlphaDropoutTrainGradModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
                new NamedModelParam[] { new TensorDataModelParam("input", ModelParamType.InputParam, inputData) },
                0.1f);

            var ctx = new ComputeContext();
            var compiled = ctx.Compile(rig.TrainingStepPureGraph);
            var initial = rig.CreateDefaultCheckpoint();
            string wName = rig.TrainableParamStructDef.Fields[0].Name;
            float w0 = Floats(initial.TrainableParams.Fields[wName])[0];

            var step = rig.TrainStep(initial,
                MakeBatch("input", "ModelInput", inputData),
                MakeBatch("targets", "Target", targetData),
                compiled);
            float w1 = Floats(step.Checkpoint.TrainableParams.Fields[wName])[0];

            Assert.True(float.IsFinite(step.Loss), $"train-mode AlphaDropout TrainStep loss must be finite; got {step.Loss}");
            Assert.True(MathF.Abs(w1 - w0) > 1e-7f,
                $"gradient must flow through train-mode AlphaDropout affine + mask (param must move); w0={w0}, w1={w1}");
        }

        // (c) FeatureAlphaDropout train-mode smoke: finite loss + ≥1 param moves (channel-broadcast affine + mask).
        {
            var rig = TrainingRig.FromScratch(
                NNFeatureAlphaDropoutTrainGradModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
                new NamedModelParam[] { new TensorDataModelParam("input", ModelParamType.InputParam, inputData) },
                0.1f);

            var ctx = new ComputeContext();
            var compiled = ctx.Compile(rig.TrainingStepPureGraph);
            var initial = rig.CreateDefaultCheckpoint();
            string wName = rig.TrainableParamStructDef.Fields[0].Name;
            float w0 = Floats(initial.TrainableParams.Fields[wName])[0];

            var step = rig.TrainStep(initial,
                MakeBatch("input", "ModelInput", inputData),
                MakeBatch("targets", "Target", targetData),
                compiled);
            float w1 = Floats(step.Checkpoint.TrainableParams.Fields[wName])[0];

            Assert.True(float.IsFinite(step.Loss), $"train-mode FeatureAlphaDropout TrainStep loss must be finite; got {step.Loss}");
            Assert.True(MathF.Abs(w1 - w0) > 1e-7f,
                $"gradient must flow through train-mode FeatureAlphaDropout affine + channel mask (param must move); w0={w0}, w1={w1}");
        }
    }

    /// <summary>
    /// Loss-knobs design §7 "Rig-path tests": the configurable losses stay usable
    /// through the TrainingRig 2-input/scalar contract when their build-time knobs
    /// are baked. (a) CrossEntropyLoss with baked <c>ignoreIndex:7</c> +
    /// <c>reduction:Sum</c> (an attribute + a scalar reduction — no extra input,
    /// output stays scalar) and (b) CrossEntropyLoss with a baked-constant class
    /// <c>weight=[2,1]</c> (the documented weight-via-rig recipe — the weight,
    /// normally a third graph input, is fixed as a graph constant inside the
    /// wrapper). Each is handed to <see cref="TrainingRig.FromScratch"/> with the
    /// tiny conv classifier model; one CrossEntropy + SGDMomentum TrainStep must
    /// produce a finite loss and move at least one trainable param (a gradient
    /// flowed through the knobbed loss). Mirrors
    /// <see cref="TestTinyConvNetCrossEntropySgdMomentumConverges"/> /
    /// <see cref="TestGeneralizedConvTrainStepFlows"/>.
    /// </summary>
    [Fact]
    public void TestConfigurableCrossEntropyRigContract()
    {
        // 4 samples of [1, 4, 4]: class 0 lights the left two columns, class 1 the right two.
        var vals = new float[4 * 16];
        for (int s = 0; s < 4; s++)
        {
            float intensity = s < 2 ? 1f : 0.6f;
            bool rightHalf = (s % 2) == 1;
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++)
                    vals[s * 16 + r * 4 + c] = (rightHalf ? c >= 2 : c < 2) ? intensity : 0f;
        }
        var inputData = TensorData([4L, 1L, 4L, 4L], vals);
        var targetData = TensorData([4L], new long[] { 0L, 1L, 0L, 1L });

        void AssertRigTrainsThroughLoss(FastComputationGraph lossGraph)
        {
            var rig = TrainingRig.FromScratch(
                NNTinyConvClassifier.ComputationGraph, lossGraph, SGDMomentumOptimizer.ComputationGraph,
                new NamedModelParam[] { new TensorDataModelParam("input", ModelParamType.InputParam, inputData) },
                0.2f, 0.9f);

            // Satisfying the 2-input rig contract is itself the coverage: the loss
            // built to a (predictions, targets) → scalar graph the rig could bind.
            Assert.NotEmpty(rig.TrainableParamStructDef.Fields);

            var ctx = new ComputeContext();
            var compiled = ctx.Compile(rig.TrainingStepPureGraph);
            var initial = rig.CreateDefaultCheckpoint();
            var step = rig.TrainStep(initial,
                MakeBatch("input", "ModelInput", inputData),
                MakeBatch("targets", "Target", targetData),
                compiled);

            Assert.True(float.IsFinite(step.Loss), $"loss must be finite; got {step.Loss}");

            bool anyMoved = false;
            foreach (var field in rig.TrainableParamStructDef.Fields)
            {
                var before = Floats(initial.TrainableParams.Fields[field.Name]);
                var after = Floats(step.Checkpoint.TrainableParams.Fields[field.Name]);
                if (before.Zip(after).Any(p => MathF.Abs(p.First - p.Second) > 1e-7f)) { anyMoved = true; break; }
            }
            Assert.True(anyMoved, "no trainable param moved — no gradient flowed through the configurable CE loss");
        }

        // (a) baked ignoreIndex:7 + reduction:Sum; (b) baked-constant weight=[2,1].
        AssertRigTrainsThroughLoss(NNCrossEntropyIgnoreSumLoss.ComputationGraph);
        AssertRigTrainsThroughLoss(NNCrossEntropyBakedWeightLoss.ComputationGraph);
    }

    // -----------------------------------------------------------------------
    //  Analytic value checks promoted from the 2026-06-12 framework behavior
    //  test campaign, covering "Autodiff" / "Optimizers" / "Schedules & rig" /
    //  "State threading" / "Weight binding". Every expectation below is hand-computed from the
    //  fixtures' constant initializers; gradients are inferred through real
    //  TrainStep execution (SGD: w' = w − lr·grad, L2Loss = mean over ALL
    //  output elements so dL/dyᵢ = 2(yᵢ−tᵢ)/N).
    // -----------------------------------------------------------------------

    /// <summary>Runs <paramref name="steps"/> TrainSteps of model + L2Loss + optimizer
    /// on one fixed batch and returns the final checkpoint, so each analytic check is a
    /// one-liner asserting exact hand-computed post-step values.</summary>
    private static TrainingCheckpoint TrainAnalytic(
        FastComputationGraph modelGraph, FastComputationGraph optimizerGraph, HyperValue[] hypers,
        long[] inShape, float[] input, long[] outShape, float[] target, int steps)
    {
        var rig = TrainingRig.FromScratch(modelGraph, L2Loss.ComputationGraph, optimizerGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData(inShape, input))],
            hypers);
        var ctx = new ComputeContext();
        var compiled = ctx.Compile(rig.TrainingStepPureGraph);
        var ckpt = rig.CreateDefaultCheckpoint();
        for (int i = 0; i < steps; i++)
            ckpt = rig.TrainStep(ckpt,
                MakeBatch("input", "AnalyticIn", TensorData(inShape, input)),
                MakeBatch("targets", "AnalyticTg", TensorData(outShape, target)),
                compiled).Checkpoint;
        return ckpt;
    }

    /// <summary>Asserts the struct's fields, flattened in definition order, equal
    /// <paramref name="expected"/> within <paramref name="tol"/>.</summary>
    private static void AssertStructIs(TensorDataStruct s, float[] expected, float tol)
    {
        var flat = s.Definition.Fields.SelectMany(f => Floats(s.Fields[f.Name])).ToArray();
        Assert.Equal(expected.Length, flat.Length);
        for (int i = 0; i < flat.Length; i++)
            Assert.True(MathF.Abs(flat[i] - expected[i]) <= tol,
                $"element [{i}]: got {flat[i]}, want {expected[i]} (±{tol})");
    }

    /// <summary>Autodiff gradient values, inferred from one SGD step against hand-derived
    /// gradients (per-line math in the comments): reverse-broadcast sum-reduction for
    /// w[1]·x[4] and x[4]+b[1], Relu masking, MatMul's xᵀ·gUp, gradient ACCUMULATION when
    /// a param is consumed twice (w·x + w), routing through Reshape→Transpose→Reshape,
    /// and Slice (only sliced elements receive gradient).</summary>
    [Fact]
    public void TestAutodiffGradientValuesAnalytic()
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
    }

    /// <summary>Optimizer per-step values vs hand-computed updates (model y = w·x, w₀=1,
    /// x=[1], t=[0] → grad = 2w). Adam step 2 pins the bias-correction timestep advancing
    /// (m̂₂ = 0.36/0.19, v̂₂ = 0.007236/0.001999 → w₂ = 0.8004123); SGD-momentum pins the
    /// velocity threading (v₁=2 → w₁=0.8; v₂ = 0.9·2+1.6 = 3.4 → w₂ = 0.46); AdamW pins
    /// the decoupled decay then UNCORRECTED step (w₁ = 1·(1−0.01) − 0.1·0.2/√0.004 =
    /// 0.6737722 — matching the campaign's recorded value).</summary>
    [Fact]
    public void TestOptimizerStepValuesAnalytic()
    {
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, AdamOptimizer.ComputationGraph,
            [0.1f, 0.9f, 0.999f, 1e-8f], [1L], [1f], [1L], [0f], 2).TrainableParams, [0.8004123f], 2e-4f);
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, SGDMomentumOptimizer.ComputationGraph,
            [0.1f, 0.9f], [1L], [1f], [1L], [0f], 2).TrainableParams, [0.46f], 1e-5f);
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, AdamWOptimizer.ComputationGraph,
            [0.1f, 0.9f, 0.999f, 1e-8f, 0.1f], [1L], [1f], [1L], [0f], 1).TrainableParams, [0.6737722f], 1e-4f);
    }

    /// <summary>Per-step values for the four just-landed optimizers vs hand-derived updates
    /// (same model y = w·x, w₀=1, x=[1], t=[0] → grad = 2w; expected values RE-DERIVED here,
    /// not taken from the design docs — adadelta/design.md's w₁ is wrong, see below). Each is a
    /// one-liner asserting the exact post-step weight.
    /// <list type="bullet">
    /// <item>Adamax (lr 0.1, β 0.9/0.999, eps 1e-8): m₁=0.2, u₁=max(0,2+1e-8)=2,
    ///   w₁ = 1 − (0.1/(1−0.9))·0.2/2 = 1 − 0.1 = 0.9; step2 = 0.80516833 (pins the bias-correction
    ///   timestep and the running-max u carry).</item>
    /// <item>NAdam (lr 0.1, β 0.9/0.999, eps 1e-8, momentumDecay 0.004): step1 = 0.89435482,
    ///   step2 = 0.81997307 (pins both scalar states — step and the running muProduct).</item>
    /// <item>RAdam (lr 0.1, β 0.9/0.999, eps 1e-8): step1 takes the UN-ADAPTED branch (ρ_t=1 ≤ 5)
    ///   ⇒ m̂₁ = 0.2/0.1 = 2, w₁ = 1 − 0.1·2 = 0.8; step2 (ρ_t≈2.0, still ≤5) = 0.62105263. A wrongly
    ///   always-adaptive impl lands well off 0.8 — sharp discriminator for the Where selection.</item>
    /// <item>Adadelta (lr 1.0, rho 0.9, eps 1e-6): newSq=0.1·2²=0.4, delta=√(0+1e-6)/√(0.4+1e-6)·2
    ///   ≈ 0.00316228, w₁ = 1 − 1.0·0.00316228 ≈ 0.99683773 (NOT design.md's 0.9936757, which is 2×
    ///   too large); step2 = 0.99359817 pins the previous-vs-current accDelta carry.</item>
    /// </list></summary>
    [Trait("Domain", "Training")]
    [Trait("Purpose", "Coverage")]
    [Fact]
    public void TestNewOptimizerStepValuesAnalytic()
    {
        // Adamax: step1 = 0.9, step2 = 0.80516833.
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, AdamaxOptimizer.ComputationGraph,
            [0.1f, 0.9f, 0.999f, 1e-8f], [1L], [1f], [1L], [0f], 1).TrainableParams, [0.9f], 1e-5f);
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, AdamaxOptimizer.ComputationGraph,
            [0.1f, 0.9f, 0.999f, 1e-8f], [1L], [1f], [1L], [0f], 2).TrainableParams, [0.80516833f], 1e-5f);

        // NAdam: step1 = 0.89435482, step2 = 0.81997307.
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, NAdamOptimizer.ComputationGraph,
            [0.1f, 0.9f, 0.999f, 1e-8f, 0.004f], [1L], [1f], [1L], [0f], 1).TrainableParams, [0.89435482f], 1e-5f);
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, NAdamOptimizer.ComputationGraph,
            [0.1f, 0.9f, 0.999f, 1e-8f, 0.004f], [1L], [1f], [1L], [0f], 2).TrainableParams, [0.81997307f], 1e-5f);

        // RAdam: step1 = 0.8 (un-adapted branch, ρ_t=1 ≤ 5), step2 = 0.62105263 (still un-adapted).
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, RAdamOptimizer.ComputationGraph,
            [0.1f, 0.9f, 0.999f, 1e-8f], [1L], [1f], [1L], [0f], 1).TrainableParams, [0.8f], 1e-5f);
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, RAdamOptimizer.ComputationGraph,
            [0.1f, 0.9f, 0.999f, 1e-8f], [1L], [1f], [1L], [0f], 2).TrainableParams, [0.62105263f], 1e-5f);

        // Adadelta: step1 = 0.99683773 (RE-DERIVED; design.md's 0.9936757 is wrong), step2 = 0.99359817.
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, AdadeltaOptimizer.ComputationGraph,
            [1.0f, 0.9f, 1e-6f], [1L], [1f], [1L], [0f], 1).TrainableParams, [0.99683773f], 1e-5f);
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, AdadeltaOptimizer.ComputationGraph,
            [1.0f, 0.9f, 1e-6f], [1L], [1f], [1L], [0f], 2).TrainableParams, [0.99359817f], 1e-5f);
    }

    /// <summary>Lion (EvoLved Sign Momentum) per-step values vs hand-derived updates (model
    /// y = w·x, w₀ = 1, x = [1], t per test → grad = 2(w·x − t)·x; values RE-DERIVED here in
    /// double precision, NOT taken from the design doc). Positional hypers: lr, β1, β2, wd (4).
    /// <list type="bullet">
    /// <item><b>Sign step + EMA carry (lr 0.1, β 0.9/0.99, wd 0, t 0).</b> step1:
    ///   update = sign(0.9·0 + 0.1·2) = sign(0.2) = +1 ⇒ w₁ = 1 − 0.1·1 = 0.9; m₁ = 0.99·0 + 0.01·2 = 0.02.
    ///   step2: g = 1.8, update = sign(0.9·0.02 + 0.1·1.8) = sign(0.198) = +1 ⇒ w₂ = 0.8. The unit-magnitude
    ///   sign step pins lr itself as the per-step move (independent of ‖g‖).</item>
    /// <item><b>Decoupled weight decay (lr 0.1, wd 1.0, t 0).</b> single step:
    ///   w₁ = 1·(1 − 0.1·1) − 0.1·sign(0.2) = 0.9 − 0.1 = 0.8 (AdamW-style decoupled WD folded in).</item>
    /// <item><b>β1↔β2-swap discriminator (lr 0.5, β 0.9/0.99, wd 0, t 0, 4 steps).</b> The first
    ///   three steps drive w 1 → 0.5 → 0.0 → −0.5 with grad still ≥ 0 (all sign(+) updates). At step 4
    ///   the gradient flips negative (g = 2·(−0.5) = −1). Under the CORRECT roles (β1 = 0.9 blends the
    ///   per-step update; β2 = 0.99 decays the carried EMA m, leaving m₃ = 0.0295 small) the blend is
    ///   0.9·0.0295 + 0.1·(−1) = −0.0734 &lt; 0 ⇒ update = −1 ⇒ w₄ = 0.0. If β1/β2 were SWAPPED, the EMA
    ///   (now decayed by 0.9) grows to m₃ = 0.252 and the update-blend (now 0.99) weights it heavily:
    ///   0.99·0.252 + 0.01·(−1) = +0.239 &gt; 0 ⇒ update = +1 ⇒ w₄ = −1.0 — a different, wrong result. So
    ///   asserting w₄ = 0.0 fails any impl that swaps the two betas (the classic Lion-vs-Adam confusion).</item>
    /// </list></summary>
    [Trait("Domain", "Training")]
    [Trait("Purpose", "Coverage")]
    [Fact]
    public void TestLionStepValuesAnalytic()
    {
        // Sign step + EMA carry: step1 = 0.9, step2 = 0.8 (lr 0.1, wd 0).
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, LionOptimizer.ComputationGraph,
            [0.1f, 0.9f, 0.99f, 0.0f], [1L], [1f], [1L], [0f], 1).TrainableParams, [0.9f], 1e-5f);
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, LionOptimizer.ComputationGraph,
            [0.1f, 0.9f, 0.99f, 0.0f], [1L], [1f], [1L], [0f], 2).TrainableParams, [0.8f], 1e-5f);

        // Decoupled weight-decay branch: w₁ = 0.9 − 0.1 = 0.8 (lr 0.1, wd 1.0).
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, LionOptimizer.ComputationGraph,
            [0.1f, 0.9f, 0.99f, 1.0f], [1L], [1f], [1L], [0f], 1).TrainableParams, [0.8f], 1e-5f);

        // β1↔β2-swap discriminator: w₄ = 0.0 with correct roles; a swap lands at −1.0. lr 0.5, 4 steps.
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, LionOptimizer.ComputationGraph,
            [0.5f, 0.9f, 0.99f, 0.0f], [1L], [1f], [1L], [0f], 4).TrainableParams, [0.0f], 1e-5f);
    }

    /// <summary>Adafactor (non-factored variant) per-step values vs hand-derived updates (model
    /// y = w·x, w₀ = 1, x = [1], t = [0] → grad = 2w; values RE-DERIVED here in double precision).
    /// Positional hypers: lr, β2Decay, ε1, ε2, clip, wd (6). RMS(θ)/RMS(U) reduce over the single
    /// element, so RMS = |·| here; the rank-agnosticism over a real ≥2-D param is gated separately
    /// in <see cref="TestLandedOptimizersExtraRigCoverage"/>.
    /// <list type="bullet">
    /// <item><b>step1 (defaults lr 0.01, β2Decay −0.8, ε1 1e-30, ε2 1e-3, clip 1.0, wd 0).</b> t = 1 ⇒
    ///   β2t = 1 − 1^(−0.8) = 0 ⇒ V = g² = 4; U = 2/√4 = 1; RMS(U) = 1; clip = 1 ⇒ Û = 1; ρ = min(0.01, 1) =
    ///   0.01; RMS(θ) = 1; α = max(1e-3, 1)·0.01 = 0.01 ⇒ w₁ = 1 − 0.01·1 = 0.99.</item>
    /// <item><b>2-step (pins the increasing-decay EMA carry).</b> step2: β2t = 1 − 2^(−0.8) = 0.4256508;
    ///   g = 1.98; V₂ = 0.4256508·4 + 0.5743492·(1.98²) = 3.9542818; U = 1.98/√V₂ = 0.9957066; ρ = 0.01;
    ///   α = 0.99·0.01 = 0.0099 ⇒ w₂ = 0.99 − 0.0099·0.9957066 = 0.98014250.</item>
    /// <item><b>clipThreshold 0.5 (pins RMS update clipping).</b> RMS(U) = 1 with d = 0.5 ⇒
    ///   Û = U / max(1, 1/0.5) = U/2, halving the step ⇒ w₁ = 1 − 0.01·0.5 = 0.995.</item>
    /// <item><b>weightDecay 0.5 (decoupled WD branch).</b> decayed = 1·(1 − 0.01·0.5) = 0.995;
    ///   then − α·Û = − 0.01·1 ⇒ w₁ = 0.995 − 0.01 = 0.985.</item>
    /// </list></summary>
    [Trait("Domain", "Training")]
    [Trait("Purpose", "Coverage")]
    [Fact]
    public void TestAdafactorStepValuesAnalytic()
    {
        // step1 defaults: w₁ = 0.99.
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, AdafactorOptimizer.ComputationGraph,
            [0.01f, -0.8f, 1e-30f, 1e-3f, 1.0f, 0.0f], [1L], [1f], [1L], [0f], 1).TrainableParams, [0.99f], 1e-5f);

        // 2-step: w₂ = 0.98014250 (β2t = 1 − t^τ EMA carry).
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, AdafactorOptimizer.ComputationGraph,
            [0.01f, -0.8f, 1e-30f, 1e-3f, 1.0f, 0.0f], [1L], [1f], [1L], [0f], 2).TrainableParams, [0.98014250f], 1e-5f);

        // clipThreshold = 0.5: RMS clipping halves the step ⇒ w₁ = 0.995.
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, AdafactorOptimizer.ComputationGraph,
            [0.01f, -0.8f, 1e-30f, 1e-3f, 0.5f, 0.0f], [1L], [1f], [1L], [0f], 1).TrainableParams, [0.995f], 1e-5f);

        // weightDecay = 0.5 (decoupled): w₁ = 0.995 − 0.01 = 0.985.
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, AdafactorOptimizer.ComputationGraph,
            [0.01f, -0.8f, 1e-30f, 1e-3f, 1.0f, 0.5f], [1L], [1f], [1L], [0f], 1).TrainableParams, [0.985f], 1e-5f);
    }

    /// <summary>Rig-construction + one-TrainStep smoke for the two just-landed optimizers
    /// (per-param state: Lion 1 — m only, no v, no step; Adafactor 2 — full param-shaped v + scalar
    /// step). Positional hyper counts: Lion 4, Adafactor 6. The scalar TrainStep gates basic state
    /// threading; the Adafactor rank-≥2 TrainStep (AnalyticMatMulModel, a [2,2] weight) is the key
    /// rank-agnosticism gate — it proves the reduce-all RMS(θ)/RMS(U) scalars work over a non-scalar
    /// param (finite loss, ≥1 element moves), not just the [1]-shaped param the analytic tests use.</summary>
    [Trait("Domain", "Training")]
    [Trait("Purpose", "Coverage")]
    [Fact]
    public void TestLandedOptimizersExtraRigCoverage()
    {
        // Construction-only smoke (state structs non-empty).
        CoverFromScratch(ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            LionOptimizer.ComputationGraph, [4L], 0.0001f, 0.9f, 0.99f, 0.0f);
        CoverFromScratch(ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            AdafactorOptimizer.ComputationGraph, [4L], 0.01f, -0.8f, 1e-30f, 1e-3f, 1.0f, 0.0f);

        // One-TrainStep smoke on the scalar model (finite loss + param moves + state threaded).
        CoverTrainStepMovesParam(LionOptimizer.ComputationGraph, 0.0001f, 0.9f, 0.99f, 0.0f);
        CoverTrainStepMovesParam(AdafactorOptimizer.ComputationGraph, 0.01f, -0.8f, 1e-30f, 1e-3f, 1.0f, 0.0f);

        // Adafactor rank-agnosticism gate: one TrainStep on a [2,2] weight (matmul model).
        CoverTrainStepMovesNonScalarParam(AnalyticMatMulModel.ComputationGraph, AdafactorOptimizer.ComputationGraph,
            [2L, 2L], [1f, 2f, 3f, 4f], [0f, 0f, 0f, 0f], 0.01f, -0.8f, 1e-30f, 1e-3f, 1.0f, 0.0f);
    }

    /// <summary>LAMB (You et al. 2019) per-step values vs hand-derived updates, RE-DERIVED in
    /// double precision against the ACTUAL analytic vehicle (NOT the design doc's grad=15 number,
    /// which is for a different fixture). The vehicle is <c>AnalyticScalarWModel</c> = y = w·x with
    /// w₀ = 1, x = [1], target = [0] ⇒ L2Loss = mean((w·x − t)²) = w² ⇒ <b>grad = 2w</b> (one
    /// element, so ‖w‖ = |w| and ‖u‖ = |u|). Positional hypers: lr, β1, β2, ε, wd (5).
    /// <list type="bullet">
    /// <item><b>Closed-form step, w₀ = 1 (lr 0.1, β 0.9/0.999, ε 1e-6, wd 0).</b> t = 1, grad = 2:
    ///   m₁ = 0.1·2 = 0.2; v₁ = 0.001·4 = 0.004; m̂ = 0.2/0.1 = 2; v̂ = 0.004/0.001 = 4;
    ///   r = 2/(√4 + 1e-6) ≈ 0.99999950; u = r (wd 0); ‖w‖ = 1, ‖u‖ ≈ 1.0 ⇒ trust = 1/‖u‖, and
    ///   trust·u = ‖w‖·sign(u) = 1 exactly ⇒ <b>w₁ = 1 − 0.1·1 = 0.9</b> (the t=1 r≈sign(g)
    ///   bias-correction collapse, same w₁ as the design's number but reached via grad = 2, not 15).</item>
    /// <item><b>Trust-ratio discriminator (multi-step, same hypers, wd 0).</b> The fixture pins
    ///   w₀ = 1, so a single-step trust is numerically 1; instead carry the state across steps. For
    ///   a positive scalar, trust·u = ‖w‖·sign(u) = w EXACTLY each step (the trust ratio normalizes
    ///   the update norm to ‖w‖), so the LAMB step is lr·w independent of the gradient magnitude ⇒
    ///   wₙ = (1 − lr)ⁿ = 0.9ⁿ: <b>w₂ = 0.81</b>, <b>w₃ = 0.729</b>. This is the sharp discriminator:
    ///   a bug that DROPS the trust ratio degenerates to plain Adam, which gives 0.8004123 at step 2
    ///   (the exact value the Adam analytic test asserts), NOT 0.81 — so 0.81/0.729 prove the trust
    ///   ratio is genuinely computed AND threaded through the m/v/step state.</item>
    /// <item><b>Zero-guard (grad = 0 ⇒ ‖u‖ = 0 fallback, NO NaN).</b> Same model but target = [1]
    ///   with w₀ = 1 ⇒ pred = 1 = target ⇒ grad = 2·(1−1) = 0 ⇒ m = v = 0 ⇒ r = 0 ⇒ u = 0 (wd 0)
    ///   ⇒ ‖u‖ = 0 ⇒ the inner Where fires ⇒ trust = 1 ⇒ step = 0 ⇒ <b>w₁ = 1.0 unchanged</b>, no
    ///   0/0 NaN. The asserted exact 1.0 (not NaN) is the nested-Where zero-guard gate.</item>
    /// </list>
    /// The scalar single-element vehicle CANNOT expose the weight-decay term: at t=1 (and while
    /// grad and w keep the same sign) trust·u = ‖w‖·sign(u) = w regardless of wd, so WD-on/off
    /// coincide on a 1-element tensor at every small step. The WD divergence is therefore gated by
    /// the MULTI-element test below.</summary>
    [Trait("Domain", "Training")]
    [Trait("Purpose", "Coverage")]
    [Fact]
    public void TestLambStepValuesAnalytic()
    {
        // Closed-form, w₀ = 1, wd 0: w₁ = 0.9.
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, LambOptimizer.ComputationGraph,
            [0.1f, 0.9f, 0.999f, 1e-6f, 0.0f], [1L], [1f], [1L], [0f], 1).TrainableParams, [0.9f], 1e-4f);

        // Trust-ratio discriminator (multi-step): wₙ = 0.9ⁿ because trust·u = ‖w‖ each step. A bug
        // dropping the trust ratio degenerates to plain Adam (0.8004123 at step 2), NOT 0.81/0.729.
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, LambOptimizer.ComputationGraph,
            [0.1f, 0.9f, 0.999f, 1e-6f, 0.0f], [1L], [1f], [1L], [0f], 2).TrainableParams, [0.81f], 1e-4f);
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, LambOptimizer.ComputationGraph,
            [0.1f, 0.9f, 0.999f, 1e-6f, 0.0f], [1L], [1f], [1L], [0f], 3).TrainableParams, [0.729f], 1e-4f);

        // Zero-guard: target == pred ⇒ grad = 0 ⇒ ‖u‖ = 0 ⇒ trust falls back to 1 ⇒ step 0,
        // w₁ = 1.0 EXACTLY (no 0/0 NaN). w₀ = 1, x = [1], target = [1].
        var z = TrainAnalytic(AnalyticScalarWModel.ComputationGraph, LambOptimizer.ComputationGraph,
            [0.1f, 0.9f, 0.999f, 1e-6f, 0.0f], [1L], [1f], [1L], [1f], 1).TrainableParams;
        AssertStructIs(z, [1.0f], 1e-6f);
        float zW = Floats(z.Fields[z.Definition.Fields[0].Name])[0];
        Assert.True(float.IsFinite(zW), $"zero-guard must avoid 0/0 NaN; got w₁ = {zW}");
    }

    /// <summary>LAMB weight-decay on/off divergence — MULTI-STEP, MULTI-ELEMENT (design §7.4).
    /// The 1-element analytic vehicle hides the WD term (trust·u = ‖w‖·sign(u) = w cancels WD at
    /// every small step), so this uses <c>AnalyticReluModel</c> (y = relu(w ⊙ x), w₀ = [1,2,3,4])
    /// with all-positive x = [1,1,1,1] (every relu mask = 1, so all four elements receive
    /// gradient). With a genuine multi-element ‖w‖/‖u‖, the SINGLE scalar trust ratio broadcasts
    /// across elements whose per-element w/u ratios differ, so the WD term no longer cancels:
    /// wd > 0 inflates ‖u‖ and adds λ·w into u, shrinking ‖w‖ MORE than wd = 0. Runs 2 steps for
    /// each and asserts ‖w‖_on &lt; ‖w‖_off (the WD direction) and every element finite (no NaN).
    /// Re-derived in double precision: at 2 steps ‖w‖_off ≈ 4.541, ‖w‖_on ≈ 4.455 (on &lt; off).</summary>
    [Trait("Domain", "Training")]
    [Trait("Purpose", "Coverage")]
    [Fact]
    public void TestLambWeightDecayDivergesMultiStep()
    {
        const int steps = 2;

        // Resolve the trainable-param field generically (definition order) rather than coupling to
        // the generated field name; AnalyticReluModel has a single [4]-shaped weight.
        var ckptOff = TrainAnalytic(AnalyticReluModel.ComputationGraph, LambOptimizer.ComputationGraph,
            [0.1f, 0.9f, 0.999f, 1e-6f, 0.0f], [4L], [1f, 1f, 1f, 1f], [4L], [0f, 0f, 0f, 0f], steps).TrainableParams;
        var ckptOn = TrainAnalytic(AnalyticReluModel.ComputationGraph, LambOptimizer.ComputationGraph,
            [0.1f, 0.9f, 0.999f, 1e-6f, 0.5f], [4L], [1f, 1f, 1f, 1f], [4L], [0f, 0f, 0f, 0f], steps).TrainableParams;

        string fName = ckptOff.Definition.Fields[0].Name;
        float[] wOff = Floats(ckptOff.Fields[fName]);
        float[] wOn = Floats(ckptOn.Fields[fName]);

        Assert.Equal(4, wOff.Length);
        foreach (var x in wOff) Assert.True(float.IsFinite(x), $"WD-off weight must be finite; got {x}");
        foreach (var x in wOn) Assert.True(float.IsFinite(x), $"WD-on weight must be finite; got {x}");

        float normOff = MathF.Sqrt(wOff.Sum(x => x * x));
        float normOn = MathF.Sqrt(wOn.Sum(x => x * x));
        Assert.True(normOn < normOff,
            $"weight decay (wd>0) must shrink ‖w‖ more than wd=0 by step {steps}; " +
            $"got ‖w‖_on={normOn}, ‖w‖_off={normOff}");
    }

    /// <summary>Rig-construction + one-TrainStep smoke for LAMB (per-param state: 3 — m/v/step,
    /// Adam's footprint; positional hyper count 5: lr/β1/β2/ε/wd). The scalar TrainStep gates
    /// basic m/v/step threading; the rank-≥2 TrainStep on AnalyticMatMulModel (a [2,2] weight) is
    /// the key gate — it proves the reduce-all-to-scalar ‖w‖/‖u‖ trust ratio works over a genuine
    /// MULTI-element tensor (finite loss, ≥1 element moves), which the [1]-shaped analytic vehicle
    /// cannot exercise (the L2-vs-RMS norm distinction and a real multi-element trust ratio).</summary>
    [Trait("Domain", "Training")]
    [Trait("Purpose", "Coverage")]
    [Fact]
    public void TestLambRigCoverage()
    {
        // Construction-only smoke (m/v/step state struct non-empty).
        CoverFromScratch(ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph,
            LambOptimizer.ComputationGraph, [4L], 0.001f, 0.9f, 0.999f, 1e-6f, 0.01f);

        // One-TrainStep smoke on the scalar model (finite loss + param moves + state threaded).
        CoverTrainStepMovesParam(LambOptimizer.ComputationGraph, 0.001f, 0.9f, 0.999f, 1e-6f, 0.01f);

        // Rank-agnosticism gate: one TrainStep on a [2,2] weight (matmul model) — the multi-element
        // ‖w‖/‖u‖ trust ratio over a non-scalar param.
        CoverTrainStepMovesNonScalarParam(AnalyticMatMulModel.ComputationGraph, LambOptimizer.ComputationGraph,
            [2L, 2L], [1f, 2f, 3f, 4f], [0f, 0f, 0f, 0f], 0.01f, 0.9f, 0.999f, 1e-6f, 0.01f);
    }

    /// <summary>End-to-end: Linear + L2 regression with LAMB on a fixed, perfectly realizable batch
    /// (the wide <c>[32,2] → [32,400]</c> realizable fixture), mirroring
    /// <see cref="TestLinearRegressionWithAdamConverges"/>. This is the multi-step, multi-element
    /// check that the trust ratio drives real optimization on a non-scalar parameter and exposes the
    /// WD term (state carry breaks the t=1 r≈sign(g) collapse). The pass conditions are
    /// platform-invariant (see the Adam sibling): the starting loss is bounded to the same narrow
    /// 1-in-100-billion random-init band, and the final loss is checked against an ABSOLUTE target
    /// rather than a ratio of the platform-varying start. LAMB carries a non-zero weight decay
    /// (0.01), so its converged L2 loss settles at a small positive floor (shrunken optimum), not 0 —
    /// the target is set above that floor with margin.</summary>
    [Trait("Domain", "Training")]
    [Trait("Purpose", "Coverage")]
    [Fact]
    public void TestLinearRegressionWithLambConverges()
    {
        var (inputData, targetData) = MakeWideRegressionData();

        var rig = TrainingRig.FromScratch(
            NNWideRegressionModel.ComputationGraph, L2Loss.ComputationGraph, LambOptimizer.ComputationGraph,
            new NamedModelParam[] { new TensorDataModelParam("input", ModelParamType.InputParam, inputData) },
            0.05f, 0.9f, 0.999f, 1e-6f, 0.01f);

        var ctx = new ComputeContext();
        var compiled = ctx.Compile(rig.TrainingStepPureGraph);
        var inputBatch = MakeBatch("input", "ModelInput", inputData);
        var targetBatch = MakeBatch("targets", "Target", targetData);

        var ckpt = rig.CreateDefaultCheckpoint();
        var losses = new List<float>();
        for (int i = 0; i < 150; i++)
        {
            var step = rig.TrainStep(ckpt, inputBatch, targetBatch, compiled);
            losses.Add(step.Loss);
            ckpt = step.Checkpoint;
        }

        Assert.All(losses, l => Assert.True(float.IsFinite(l)));
        AssertWideStartLossInBand(losses[0]);
        // Converged near the (weight-decay-shrunken) minimum: an ABSOLUTE target, not a ratio
        // of the platform-dependent starting loss.
        Assert.True(losses[^1] < 1e-2f,
            $"LAMB should converge below the absolute target 1e-2 in 150 steps; got {losses[^1]}");
    }

    /// <summary>Schedule step indexing and BatchNorm state threading, exactly:
    /// (a) Schedules.Linear(0.2, 0.1, 1) must apply lr(0)=0.2 at the FIRST step and
    /// lr(1)=0.1 at the second (w₀=1, t=3, grad=2(w−3): w₁ = 1+0.2·4 = 1.8,
    /// w₂ = 1.8+0.1·2.4 = 2.04 — both off-by-one variants give 2.28 / 2.072 instead);
    /// (b) training-mode BatchNorm2d's running stats follow the documented ONNX EMA
    /// (running·m + batch·(1−m), BIASED variance): batch [1,2,3,4] (mean 2.5,
    /// var 1.25), m=0.9 → ModelState = [0·0.9+2.5·0.1, 1·0.9+1.25·0.1] = [0.25, 1.025],
    /// isolated via lr=0.</summary>
    [Fact]
    public void TestScheduleAndBatchNormStateAnalytic()
    {
        AssertStructIs(TrainAnalytic(AnalyticScalarWModel.ComputationGraph, SGDOptimizer.ComputationGraph,
            [Schedules.Linear(0.2f, 0.1f, 1)], [1L], [1f], [1L], [3f], 2).TrainableParams, [2.04f], 1e-5f);
        AssertStructIs(TrainAnalytic(AnalyticBatchNormModel.ComputationGraph, SGDOptimizer.ComputationGraph,
            [0f], [1L, 1L, 2L, 2L], [1f, 2f, 3f, 4f], [1L, 1L, 2L, 2L], [0f, 0f, 0f, 0f], 1).ModelState,
            [0.25f, 1.025f], 1e-4f);
    }

    /// <summary>Weight binding (campaign §3): GetConcreteModelParamInfos exposes the
    /// params by ToShorokooIdString name, and ToConcreteModel(weights) binds by that
    /// name — known W=[[1,2],[3,4]] ([out,in] layout) and b=[10,20] through the library
    /// Linear (including its both-branches IfElse bias path) give exactly x·Wᵀ + b =
    /// [13, 27] for x=[[1,1]].</summary>
    [Fact]
    public void TestWeightBindingAnalytic()
    {
        var g = AnalyticBindLinearModel.ComputationGraph;
        var x = TensorData([1L, 2L], 1f, 1f);
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([x]));
        var infos = arch.GetConcreteModelParamInfos().ParamInfos;
        Assert.Equal(2, infos.Length);
        var weights = new ModelParamList(
            new[]
            {
                Tuple.Create(infos[0].ToShorokooIdString(), (TensorData)TensorData([2L, 2L], 1f, 2f, 3f, 4f)),
                Tuple.Create(infos[1].ToShorokooIdString(), (TensorData)TensorData([2L], 10f, 20f)),
            },
            ModelParamType.TrainableParam);
        var y = new ComputeContext().Execute(arch.ToConcreteModel(weights), x)[0]
            .ToTensorData().As<float32>().AccessMemory<float>().ToArray();
        Assert.Equal(2, y.Length);
        Assert.True(MathF.Abs(y[0] - 13f) < 1e-5f && MathF.Abs(y[1] - 27f) < 1e-5f,
            $"bound Linear must give x·Wᵀ + b = [13, 27]; got [{y[0]}, {y[1]}]");
    }

    /// <summary>
    /// Recurrent.RNN §7-9 trainable-corner gradient (forward, tanh, single-layer, layout=0 — the
    /// only autodiff-supported configuration, design note [3]). The analytic AutoGrad gradient of
    /// the loss ΣY + Σh_n is FD-checked against a two-sided directional derivative on ORT's own
    /// forward, mirroring TestAutoGradRecurrentReverseDirection / AutoGradRnnReverseCheck — the
    /// proven AutoGrad-graph gradient path. This is the positive trainable-gradient coverage; the
    /// end-to-end recurrent-rig training is covered by the LSTM/GRU rig train-step tests (the
    /// shared RNN-BPTT scheduler stall they exercise was fixed in #440).
    /// </summary>
    [Fact]
    public void TestRecurrentRnnForwardTanhGradient()
    {
        Assert.True(AutoTest.AdvancedTestGraph<RnnForwardTanhGolden>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.3f)]));
    }

    /// <summary>
    /// Recurrent.RNN §7-3 / §7-6 BPTT guards: back-propagation through a relu RNN and through a
    /// bidirectional RNN must each throw AD003 at AUTO_GRAD lowering (relu is a non-default
    /// activation; bidirectional BPTT is unimplemented — design note [3]). The exception surfaces
    /// from the AdvancedTestGraph call itself during concretization, before any backend runs —
    /// mirroring AutoGradRnnBidirectionalThrowCheck. These two modes are inference-grade only.
    /// </summary>
    [Fact]
    public void TestRecurrentRnnReluAndBidirectionalBpttThrow()
    {
        var relu = Assert.Throws<AutoDiffNotSupportedException>(() =>
            AutoTest.AdvancedTestGraph<RnnReluBpttThrowCheck>(
                hyperparamInputs: [],
                runtimeInputs: [TensorData(DType.Float32, [], 0.3f)]));
        Assert.Equal(ErrorCodes.AD003, relu.ErrorCode);

        var bidi = Assert.Throws<AutoDiffNotSupportedException>(() =>
            AutoTest.AdvancedTestGraph<RnnBidirectionalBpttThrowCheck>(
                hyperparamInputs: [],
                runtimeInputs: [TensorData(DType.Float32, [], 0.3f)]));
        Assert.Equal(ErrorCodes.AD003, bidi.ErrorCode);
        Assert.Contains("bidirectional", bidi.Message);
    }

    /// <summary>
    /// Recurrent.LSTM §7-8 trainable-corner gradient (forward, single-layer, layout=0, default
    /// activations — a supported autodiff configuration, lstm/design.md §5.2). The analytic AutoGrad
    /// gradient of the loss ΣY + Σh_n + Σc_n is FD-checked against a two-sided directional derivative
    /// on ORT's own forward, mirroring TestRecurrentRnnForwardTanhGradient / AutoGradLstmReverseCheck.
    /// This is the positive trainable-gradient coverage; the end-to-end TrainingRig smoke is
    /// TestRecurrentLstmForwardTrainStepFlows below.
    /// </summary>
    [Fact]
    public void TestRecurrentLstmForwardGradient()
    {
        Assert.True(AutoTest.AdvancedTestGraph<LstmForwardGolden>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.3f)]));
    }

    /// <summary>
    /// Recurrent.LSTM §7-8 trainability smoke (the trainable corner): a forward, single-layer
    /// Recurrent.LSTM model + CrossEntropy + SGDMomentum trains end-to-end through TrainingRig
    /// FromScratch / TrainStep — the rig builds, one step gives a finite loss, and ≥1 owned
    /// W/R/bias param moves — exercising the recurrent-rig path (the #440 MemoryAwareScheduler
    /// fallback that lets RNN/LSTM/GRU BPTT scopes build). Input [L=3, N=4, in=2].
    /// </summary>
    [Fact]
    public void TestRecurrentLstmForwardTrainStepFlows()
    {
        var vals = Enumerable.Range(0, 3 * 4 * 2).Select(i => 0.1f * i - 1f).ToArray();
        var inputData = TensorData([3L, 4L, 2L], vals);
        var targetData = TensorData([4L], new long[] { 0L, 1L, 0L, 1L });

        var rig = TrainingRig.FromScratch(
            LstmForwardTrainModel.ComputationGraph, CrossEntropyLoss.ComputationGraph,
            SGDMomentumOptimizer.ComputationGraph,
            new NamedModelParam[] { new TensorDataModelParam("input", ModelParamType.InputParam, inputData) },
            0.2f, 0.9f);

        // Building the rig is itself coverage: the forward LSTM differentiates (W/R/bias).
        Assert.NotEmpty(rig.TrainableParamStructDef.Fields);

        var ctx = new ComputeContext();
        var compiled = ctx.Compile(rig.TrainingStepPureGraph);
        var initial = rig.CreateDefaultCheckpoint();
        var step = rig.TrainStep(initial,
            MakeBatch("input", "ModelInput", inputData),
            MakeBatch("targets", "Target", targetData),
            compiled);

        Assert.True(float.IsFinite(step.Loss), $"loss must be finite; got {step.Loss}");

        bool anyMoved = false;
        foreach (var field in rig.TrainableParamStructDef.Fields)
        {
            var before = Floats(initial.TrainableParams.Fields[field.Name]);
            var after = Floats(step.Checkpoint.TrainableParams.Fields[field.Name]);
            if (before.Zip(after).Any(p => MathF.Abs(p.First - p.Second) > 1e-7f)) { anyMoved = true; break; }
        }
        Assert.True(anyMoved, "no trainable param moved — no gradient flowed through the Recurrent.LSTM path");
    }

    /// <summary>
    /// Recurrent.LSTM §7-5 BPTT guard: back-propagation through a bidirectional LSTM must throw AD003
    /// at AUTO_GRAD lowering (bidirectional BPTT is unimplemented — lstm/design.md §5.2). The
    /// exception surfaces from the AdvancedTestGraph call itself during concretization, before any
    /// backend runs — mirroring TestRecurrentRnnReluAndBidirectionalBpttThrow. Bidirectional LSTM is
    /// inference-grade only.
    /// </summary>
    [Fact]
    public void TestRecurrentLstmBidirectionalBpttThrow()
    {
        var bidi = Assert.Throws<AutoDiffNotSupportedException>(() =>
            AutoTest.AdvancedTestGraph<LstmBidirectionalBpttThrowCheck>(
                hyperparamInputs: [],
                runtimeInputs: [TensorData(DType.Float32, [], 0.3f)]));
        Assert.Equal(ErrorCodes.AD003, bidi.ErrorCode);
        Assert.Contains("bidirectional", bidi.Message);
    }

    /// <summary>
    /// Recurrent.GRU §7-9 trainable-corner gradient (forward, single-layer, layout=0, default
    /// activations, linearBeforeReset:true — a supported autodiff configuration, gru/design.md §5).
    /// The analytic AutoGrad gradient of the loss ΣY + Σh_n is FD-checked against a two-sided
    /// directional derivative on ORT's own forward, mirroring TestRecurrentLstmForwardGradient /
    /// AutoGradGruReverseCheck. This is the positive trainable-gradient coverage; the end-to-end
    /// TrainingRig smoke is TestRecurrentGruForwardTrainStepFlows below.
    /// </summary>
    [Fact]
    public void TestRecurrentGruForwardGradient()
    {
        Assert.True(AutoTest.AdvancedTestGraph<GruForwardGolden>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.3f)]));
    }

    /// <summary>
    /// Recurrent.GRU §7-9 trainability smoke (the trainable corner): a forward, single-layer,
    /// linearBeforeReset:true Recurrent.GRU model + CrossEntropy + SGDMomentum trains end-to-end
    /// through TrainingRig FromScratch / TrainStep — the rig builds, one step gives a finite loss,
    /// and ≥1 owned W/R/bias param moves. Mirrors TestRecurrentLstmForwardTrainStepFlows (works via
    /// the recurrent-rig scheduler fix). Input [L=3, N=4, in=2].
    /// </summary>
    [Fact]
    public void TestRecurrentGruForwardTrainStepFlows()
    {
        var vals = Enumerable.Range(0, 3 * 4 * 2).Select(i => 0.1f * i - 1f).ToArray();
        var inputData = TensorData([3L, 4L, 2L], vals);
        var targetData = TensorData([4L], new long[] { 0L, 1L, 0L, 1L });

        var rig = TrainingRig.FromScratch(
            GruForwardTrainModel.ComputationGraph, CrossEntropyLoss.ComputationGraph,
            SGDMomentumOptimizer.ComputationGraph,
            new NamedModelParam[] { new TensorDataModelParam("input", ModelParamType.InputParam, inputData) },
            0.2f, 0.9f);

        // Building the rig is itself coverage: the forward GRU differentiates (W/R/bias).
        Assert.NotEmpty(rig.TrainableParamStructDef.Fields);

        var ctx = new ComputeContext();
        var compiled = ctx.Compile(rig.TrainingStepPureGraph);
        var initial = rig.CreateDefaultCheckpoint();
        var step = rig.TrainStep(initial,
            MakeBatch("input", "ModelInput", inputData),
            MakeBatch("targets", "Target", targetData),
            compiled);

        Assert.True(float.IsFinite(step.Loss), $"loss must be finite; got {step.Loss}");

        bool anyMoved = false;
        foreach (var field in rig.TrainableParamStructDef.Fields)
        {
            var before = Floats(initial.TrainableParams.Fields[field.Name]);
            var after = Floats(step.Checkpoint.TrainableParams.Fields[field.Name]);
            if (before.Zip(after).Any(p => MathF.Abs(p.First - p.Second) > 1e-7f)) { anyMoved = true; break; }
        }
        Assert.True(anyMoved, "no trainable param moved — no gradient flowed through the Recurrent.GRU path");
    }

    /// <summary>
    /// Recurrent.GRU §7-6 BPTT guard: back-propagation through a bidirectional GRU must throw AD003
    /// at AUTO_GRAD lowering (bidirectional BPTT is unimplemented — gru/design.md §5). The exception
    /// surfaces from the AdvancedTestGraph call itself during concretization, before any backend runs —
    /// mirroring TestRecurrentLstmBidirectionalBpttThrow. Bidirectional GRU is inference-grade only.
    /// </summary>
    [Fact]
    public void TestRecurrentGruBidirectionalBpttThrow()
    {
        var bidi = Assert.Throws<AutoDiffNotSupportedException>(() =>
            AutoTest.AdvancedTestGraph<GruBidirectionalBpttThrowCheck>(
                hyperparamInputs: [],
                runtimeInputs: [TensorData(DType.Float32, [], 0.3f)]));
        Assert.Equal(ErrorCodes.AD003, bidi.ErrorCode);
        Assert.Contains("bidirectional", bidi.Message);
    }

    /// <summary>
    /// Single-step recurrent CELL §7-6 trainable-corner FD grad checks (recurrent-cells/design.md §7).
    /// The analytic AutoGrad gradient of a loss Σh' (+ Σc' for LSTM) over a single cell step is FD-checked
    /// against a two-sided directional derivative on ORT's own forward, mirroring RnnForwardTanhGolden /
    /// LstmForwardGolden / GruForwardGolden. Both x AND the previous-state input(s) depend on the
    /// probed scalar, so the gradient threads through the cell's distinguishing h(/c) input as well as x —
    /// the trainable corner (forward-tanh RNNCell, LSTMCell, and BOTH lbr forms of GRUCell).
    /// </summary>
    [Fact]
    public void TestRecurrentCellForwardGradients()
    {
        Assert.True(AutoTest.AdvancedTestGraph<RnnCellForwardTanhGolden>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<LstmCellForwardGolden>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<GruCellForwardGolden>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 0.3f)]));
    }

    /// <summary>
    /// Single-step recurrent CELL §7-7 trainability rig smoke. A small hand-unrolled 2-step cell loop
    /// (RNNCell tanh / LSTMCell / GRUCell) + L2Loss + SGD trains end-to-end through TrainingRig
    /// FromScratch / TrainStep — the rig builds, one step gives a finite loss, and ≥1 owned W/R/bias param
    /// moves, proving the cell path differentiates end-to-end through a USER loop (the cell's whole point)
    /// and the owned params train. Mirrors TestRecurrentLstmForwardTrainStepFlows. Input [L=2, N=4, in=2];
    /// the target is a zero [N=4, 2] tensor for the L2 regression.
    /// </summary>
    [Fact]
    public void TestRecurrentCellTrainStepFlows()
    {
        CoverCellTrainStep(RnnCellTrainModel.ComputationGraph);
        CoverCellTrainStep(LstmCellTrainModel.ComputationGraph);
        CoverCellTrainStep(GruCellTrainModel.ComputationGraph);
    }

    /// <summary>Builds a TrainingRig over a 2-step cell-loop model + L2Loss + SGD, runs one TrainStep on a
    /// fixed [L=2,N=4,in=2] batch with a zero [N=4,2] target, and asserts a finite loss and ≥1 owned
    /// W/R/bias param moved. The cell models own their W/R/bias via RecurrentUniform inside the user loop.</summary>
    private static void CoverCellTrainStep(FastComputationGraph modelGraph)
    {
        var inVals = Enumerable.Range(0, 2 * 4 * 2).Select(i => 0.1f * i - 0.7f).ToArray();
        var inputData = TensorData([2L, 4L, 2L], inVals);
        var targetData = TensorData([4L, 2L], new float[8]);   // zero target for L2 regression

        var rig = TrainingRig.FromScratch(
            modelGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            new NamedModelParam[] { new TensorDataModelParam("input", ModelParamType.InputParam, inputData) },
            0.1f);

        // Building the rig is itself coverage: the cell loop differentiates (W/R/bias).
        Assert.NotEmpty(rig.TrainableParamStructDef.Fields);

        var ctx = new ComputeContext();
        var compiled = ctx.Compile(rig.TrainingStepPureGraph);
        var initial = rig.CreateDefaultCheckpoint();
        var step = rig.TrainStep(initial,
            MakeBatch("input", "ModelInput", inputData),
            MakeBatch("targets", "Target", targetData),
            compiled);

        Assert.True(float.IsFinite(step.Loss), $"loss must be finite; got {step.Loss}");

        bool anyMoved = false;
        foreach (var field in rig.TrainableParamStructDef.Fields)
        {
            var before = Floats(initial.TrainableParams.Fields[field.Name]);
            var after = Floats(step.Checkpoint.TrainableParams.Fields[field.Name]);
            if (before.Zip(after).Any(p => MathF.Abs(p.First - p.Second) > 1e-7f)) { anyMoved = true; break; }
        }
        Assert.True(anyMoved, "no trainable param moved — no gradient flowed through the recurrent cell path");
    }

    /// <summary>
    /// Single-step recurrent CELL §7-8 BPTT guard: back-propagation through Recurrent.RNNCell(Relu) must
    /// throw AD003 at AUTO_GRAD lowering (relu is a non-default activation; cell BPTT unsupported — the
    /// same loud, documented limit the full Recurrent.RNN(Relu) carries). The exception surfaces from the
    /// AdvancedTestGraph call itself during concretization, before any backend runs — mirroring
    /// TestRecurrentRnnReluAndBidirectionalBpttThrow. The relu cell builds and infers but won't train.
    /// </summary>
    [Fact]
    public void TestRecurrentCellReluBpttThrow()
    {
        var relu = Assert.Throws<AutoDiffNotSupportedException>(() =>
            AutoTest.AdvancedTestGraph<RnnCellReluBpttThrowCheck>(
                hyperparamInputs: [],
                runtimeInputs: [TensorData(DType.Float32, [], 0.3f)]));
        Assert.Equal(ErrorCodes.AD003, relu.ErrorCode);
    }
}

using System.Reflection;
using Shorokoo.Runtime;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Inference;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage-purpose tests that drive <see cref="AutoTest.AdvancedTestGraph{TModule}"/>
/// against modules built around groups of QuickExecutionEngine op handlers. Each module
/// chains several related ops so a single Coverage test widens QEE coverage across many
/// branches that the AutoGrad-focused Coverage suite never reaches (ArgMax/ArgMin,
/// EyeLike, Random*, SequenceSlice/Concat, OptionalHasElement, the Constant attribute
/// branches, the Loop placeholder ops, the integer / bitwise / activation paths, and the
/// per-dtype branches in TensorDataConverter).
/// </summary>
[Trait("Domain", "Inference")]
[Trait("Purpose", "Coverage")]
public class QeeOpsCoverageTests
{
    /// <summary>
    /// QEE-only one-liner for modules whose graphs include Shorokoo-internal ops
    /// (#SequenceSlice#, #SequenceConcat#, #LoopFakeInput#, #LoopScanVariable#,
    /// #LoopIndexVariable#) that have no ONNX op-set registration. AutoTest's leading
    /// ComputeContext.Execute loads ORT and fails on these even with both roundtrip
    /// flags off, so we lower the module exactly like AdvancedTestGraph but only run
    /// the QuickExecutionEngine validation pass.
    /// </summary>
    private static bool QeeOnly<TModule>(TensorData[] runtimeInputs)
    {
        var prop = typeof(TModule).GetProperty("ComputationGraph", BindingFlags.Public | BindingFlags.Static)
            ?? throw new InvalidOperationException($"{typeof(TModule).FullName} has no public static ComputationGraph property");
        var moduleGraph = (FastComputationGraph)prop.GetValue(null)!;
        var concreteArch = moduleGraph.ToConcreteArchitecture(moduleGraph.FromOrderedInputs([.. runtimeInputs]));
        var concreteModel = concreteArch.ToConcreteModel();
        var qee = new QuickExecutionEngine();
        var store = runtimeInputs.Length == 0 ? qee.Run(concreteModel) : qee.Run(concreteModel, runtimeInputs);
        foreach (var outKey in concreteModel.Outputs)
            if (!store.TryGetValue(outKey, out var rt) || rt.DType == DType.Invalid) return false;
        return true;
    }

    private static readonly TensorData IntVec3A = TensorData(DType.Int64, [3L], 1L, 2L, 3L);
    private static readonly TensorData IntVec3B = TensorData(DType.Int64, [3L], 2L, 2L, 2L);
    private static readonly TensorData FloatMat3x2 = TensorData(DType.Float32, [3L, 2L], 1f, 2f, 3f, 4f, 5f, 6f);
    private static readonly TensorData FloatMat3x3 = TensorData(DType.Float32, [3L, 3L], 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f);
    private static readonly TensorData FloatVec3 = TensorData(DType.Float32, [3L], -1f, 0.5f, 2f);
    private static readonly TensorData FloatScalar = TensorData(DType.Float32, [], 1f);

    [Fact]
    public void TestQeeNumericAndBitwiseOpsCoverage()
    {
        var aBin = TensorData(DType.Int64, [3L], 0b1100L, 0b1010L, 0b1111L);
        var bBin = TensorData(DType.Int64, [3L], 0b1010L, 0b0101L, 0b0011L);
        Assert.True(AutoTest.AdvancedTestGraph<QeeIntUnaryOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Int64, [3L], -2L, 0L, 5L)]));
        Assert.True(AutoTest.AdvancedTestGraph<QeeIntCompareOpsCheck>(hyperparamInputs: [], runtimeInputs: [IntVec3A, IntVec3B]));
        Assert.True(AutoTest.AdvancedTestGraph<QeeIntBinaryOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Int64, [3L], 2L, 3L, 4L), TensorData(DType.Int64, [3L], 1L, 2L, 3L)]));
        Assert.True(AutoTest.AdvancedTestGraph<QeeBitwiseOpsCheck>(hyperparamInputs: [], runtimeInputs: [aBin, bBin]));
    }

    [Fact]
    public void TestQeeArgAndActivationOpsCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<QeeArgOpsCheck>(hyperparamInputs: [], runtimeInputs: [FloatMat3x2]));
        Assert.True(AutoTest.AdvancedTestGraph<QeeActivationsCheck>(hyperparamInputs: [], runtimeInputs: [FloatVec3]));
        Assert.True(AutoTest.AdvancedTestGraph<QeeMiscFloatBoolOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [
                TensorData(DType.Float32, [3L], 1.2f, 2.7f, -0.4f),
                TensorData(DType.Bool, [3L], true, false, true),
                TensorData(DType.Bool, [3L], true, true, false)]));
    }

    [Fact]
    public void TestQeeShapeProducerOpsCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<QeeEyeLikeOpsCheck>(hyperparamInputs: [], runtimeInputs: [FloatMat3x3]));
        Assert.True(AutoTest.AdvancedTestGraph<QeeConstantOpsCheck>(hyperparamInputs: [], runtimeInputs: []));
        Assert.True(AutoTest.AdvancedTestGraph<QeeRandomOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [2L, 3L], 0f, 0f, 0f, 0f, 0f, 0f)],
            testCsRoundtrip: false));
        Assert.True(AutoTest.AdvancedTestGraph<QeeShrkRandomOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Int64, [2L], 2L, 3L)]));
    }

    [Fact]
    public void TestQeeCastAndOptionalOpsCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<QeeCastToBoolOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [
                TensorData(DType.Float32, [3L], 0f, 1.5f, -2.0f),
                TensorData(DType.Int64, [3L], 0L, 1L, 2L),
                TensorData(DType.Bool, [3L], true, false, true)]));
        Assert.True(AutoTest.AdvancedTestGraph<QeeOptionalHasElementCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [3L], 1f, 2f, 3f)]));
    }

    /// <summary>
    /// Shorokoo-internal Sequence/Loop placeholder ops. ORT can't run #SequenceSlice#,
    /// #SequenceConcat#, #LoopFakeInput#, #LoopScanVariable#, or #LoopIndexVariable#, so
    /// this goes through the QeeOnly helper.
    /// </summary>
    [Fact]
    public void TestQeeInternalSequenceLoopOpsCoverage() =>
        Assert.True(QeeOnly<QeeInternalSequenceLoopOpsCheck>([
            TensorData(DType.Float32, [2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f),
            TensorData(DType.Float32, [], 1f),
            TensorData(DType.Float32, [], 2f),
            TensorData(DType.Float32, [], 3f)]));

    /// <summary>
    /// Routes inputs of every non-{float32,int64} dtype through QEE so each per-dtype
    /// branch of TensorDataConverter.ToRuntimeTensor fires. Split into two modules
    /// because the source generator caps modules at 8 inputs.
    /// </summary>
    [Fact]
    public void TestQeeDtypeIdentityOpsCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<QeeDtypeIdentitySignedOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [
                TensorData(DType.Float64, [3L], 1.0, 2.0, 3.0),
                TensorData(DType.Int32, [3L], 1, 2, 3),
                TensorData(DType.Int16, [3L], (short)1, (short)2, (short)3),
                TensorData(DType.Int8, [3L], (sbyte)1, (sbyte)2, (sbyte)3)]));
        Assert.True(AutoTest.AdvancedTestGraph<QeeDtypeIdentityUnsignedOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [
                TensorData(DType.UInt8, [3L], (byte)1, (byte)2, (byte)3),
                TensorData(DType.UInt16, [3L], (ushort)1, (ushort)2, (ushort)3),
                TensorData(DType.UInt32, [3L], (uint)1, (uint)2, (uint)3),
                TensorData(DType.UInt64, [3L], (ulong)1, (ulong)2, (ulong)3),
                TensorData(DType.Bool, [3L], true, false, true)]));
    }

    // ===================================================================
    //  Coverage tests for the QEE op handlers added by the AutoGrad/QEE
    //  expansion batch (src/Shorokoo/Core/Inference/Ops/<Op>.cs). Each
    //  [Fact] groups several Modules whose forward graphs collectively
    //  drive every branch of one related family of ops.
    // ===================================================================

    private static readonly TensorData NchwImage1x1x4x4 = TensorData(
        DType.Float32, [1L, 1L, 4L, 4L],
        1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f,
        9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f);

    private static readonly TensorData FloatMat4x4 = TensorData(
        DType.Float32, [4L, 4L],
        1f, 0f, 0f, 0f, 0f, 2f, 0f, 0f,
        0f, 0f, 3f, 0f, 0f, 0f, 0f, 4f);

    [Fact]
    public void TestQeeShapeTransformOpsCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<QeeSpaceDepthOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [NchwImage1x1x4x4]));
        Assert.True(AutoTest.AdvancedTestGraph<QeeCenterCropPadOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [3L, 3L],
                1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f)]));
        Assert.True(AutoTest.AdvancedTestGraph<QeeUpsampleOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [NchwImage1x1x4x4]));
        Assert.True(AutoTest.AdvancedTestGraph<QeeCol2ImOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [1L, 4L, 4L],
                1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f)]));
    }

    [Fact]
    public void TestQeeReductionAndCompressOpsCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<QeeCumSumVariantsOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [3L], 1f, 2f, 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<QeeReverseSequenceOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [
                TensorData(DType.Float32, [2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f),
                TensorData(DType.Int64, [2L], 2L, 3L)]));
        Assert.True(AutoTest.AdvancedTestGraph<QeeCompressOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [
                TensorData(DType.Float32, [3L, 2L], 1f, 2f, 3f, 4f, 5f, 6f),
                TensorData(DType.Bool, [3L], true, false, true)]));
    }

    [Fact]
    public void TestQeeLinearAlgebraOpsCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<QeeDetEinsumOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [FloatMat4x4]));
        Assert.True(AutoTest.AdvancedTestGraph<QeeEinsumImplicitOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [
                TensorData(DType.Float32, [2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f),
                TensorData(DType.Float32, [3L, 2L], 1f, 0f, 0f, 1f, 1f, 1f)]));
    }

    [Fact]
    public void TestQeeUniqueAndNonMaxOpsCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<QeeUniqueOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [4L],
                1f, 2f, 2f, 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<QeeUniqueFlatOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [2L, 2L],
                1f, 2f, 2f, 3f)]));
        Assert.True(AutoTest.AdvancedTestGraph<QeeNonMaxSuppressionOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [
                TensorData(DType.Float32, [1L, 4L, 4L],
                    0f, 0f, 1f, 1f, 0.1f, 0.1f, 1.1f, 1.1f,
                    0f, 2f, 1f, 3f, 2f, 2f, 3f, 3f),
                TensorData(DType.Float32, [1L, 1L, 4L], 0.9f, 0.8f, 0.7f, 0.6f)],
            testCsRoundtrip: false));
    }

    [Fact]
    public void TestQeeAffineGridAndRoiOpsCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<QeeAffineGridSampleOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [
                TensorData(DType.Float32, [1L, 2L, 3L], 1f, 0f, 0f, 0f, 1f, 0f),
                NchwImage1x1x4x4]));
        // MaxUnpool: indices are int64 per spec (the MAX_UNPOOL definition was fixed in the
        // Phase 4 QEE-A1 audit batch); kept on the QeeOnly path to preserve this test's
        // original QEE-shape-inference focus.
        Assert.True(QeeOnly<QeeMaxUnpoolOpsCheck>([
            TensorData(DType.Float32, [1L, 1L, 2L, 2L], 6f, 8f, 14f, 16f),
            TensorData(DType.Int64, [1L, 1L, 2L, 2L], 5L, 7L, 13L, 15L)]));
    }

    [Fact]
    public void TestQeePoolingVariantsAndShrkConvOpsCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<QeePoolingVariantsOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [NchwImage1x1x4x4]));
        Assert.True(AutoTest.AdvancedTestGraph<QeeShrkConvOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [
                NchwImage1x1x4x4,
                TensorData(DType.Float32, [1L, 1L, 2L, 2L], 1f, 0f, 0f, 1f),
                TensorData(DType.Float32, [1L], 0f)]));
    }

    // Coverage for the QEE shape-inference branches of Rnn.cs / Gru.cs / Lstm.cs
    // comes from the existing AutoDiff-domain recurrent suite (TestAutoGradRnn*,
    // TestAutoGradGru*, TestAutoGradLstm* in AutoGradOpsTests.cs), which builds and
    // executes the full recurrent forward and backward — direct QEE-only tests on
    // sparsely-populated RNN/GRU/LSTM modules trip the source-generator/lowering
    // pipeline because too many optional inputs come through unbound.

    [Fact]
    public void TestQeeQuantizationOpsCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<QeeDequantizeLinearOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [
                TensorData(DType.Int8, [2L, 2L], (sbyte)10, (sbyte)20, (sbyte)30, (sbyte)40),
                TensorData(DType.Float32, [2L], 0.5f, 0.25f),
                TensorData(DType.Int8, [2L], (sbyte)0, (sbyte)1)]));
        Assert.True(AutoTest.AdvancedTestGraph<QeeDynamicQuantizeLinearOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [], 3.5f)]));
        Assert.True(AutoTest.AdvancedTestGraph<QeeMatMulIntegerOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [
                TensorData(DType.Int8, [2L, 3L], (sbyte)1, (sbyte)2, (sbyte)3, (sbyte)4, (sbyte)5, (sbyte)6),
                TensorData(DType.Int8, [3L, 2L], (sbyte)1, (sbyte)0, (sbyte)0, (sbyte)1, (sbyte)1, (sbyte)1),
                TensorData(DType.Int8, [], (sbyte)0),
                TensorData(DType.Int8, [], (sbyte)0)]));
        Assert.True(AutoTest.AdvancedTestGraph<QeeConvIntegerOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [
                TensorData(DType.Int8, [1L, 1L, 3L, 3L],
                    (sbyte)1, (sbyte)2, (sbyte)3, (sbyte)4, (sbyte)5,
                    (sbyte)6, (sbyte)7, (sbyte)8, (sbyte)9),
                TensorData(DType.Int8, [1L, 1L, 2L, 2L], (sbyte)1, (sbyte)0, (sbyte)0, (sbyte)1),
                TensorData(DType.Int8, [], (sbyte)0),
                TensorData(DType.Int8, [], (sbyte)0)]));
    }

    [Fact]
    public void TestQeeMiscOpsCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<QeeBernoulliOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [
                TensorData(DType.Float32, [3L], 0.1f, 0.5f, 0.9f)],
            testCsRoundtrip: false));
        Assert.True(AutoTest.AdvancedTestGraph<QeeBitShiftOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [
                TensorData(DType.UInt64, [3L], (ulong)1, (ulong)2, (ulong)3),
                TensorData(DType.UInt64, [3L], (ulong)1, (ulong)1, (ulong)2)]));
        Assert.True(AutoTest.AdvancedTestGraph<QeeDeformConvOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [
                NchwImage1x1x4x4,
                TensorData(DType.Float32, [1L, 1L, 2L, 2L], 1f, 0f, 0f, 1f),
                TensorDataWithDefaultVals(DType.Float32, [1L, 8L, 3L, 3L]),
                TensorData(DType.Float32, [1L], 0f)],
            testOnnxRoundtrip: false, testCsRoundtrip: false));
    }

    // ===================================================================
    //  Coverage tests for the opset-21 QEE op handlers added in the
    //  Inference/Ops batch — modeled on the V2-style coverage tests above.
    //  Each [Fact] groups several QeeOpsTestModulesV3 modules so a single
    //  test drives the QEE Compute path of several related ops.
    // ===================================================================

    private static readonly TensorData FloatMat2x3 = TensorData(
        DType.Float32, [2L, 3L],
        -1f, 0.5f, 2f,
        0.1f, -0.3f, 1.5f);

    [Fact]
    public void TestQeeNewActivationsCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<QeeNewActivationsCheck>(
            hyperparamInputs: [], runtimeInputs: [FloatVec3]));
        Assert.True(AutoTest.AdvancedTestGraph<QeeIsInfNaNCheck>(
            hyperparamInputs: [], runtimeInputs: [
                TensorData(DType.Float32, [4L], 0f, float.PositiveInfinity, float.NegativeInfinity, float.NaN)]));
        Assert.True(AutoTest.AdvancedTestGraph<QeeRoundShrinkSizeCheck>(
            hyperparamInputs: [], runtimeInputs: [
                TensorData(DType.Float32, [4L], -1.5f, 0.4f, 1.6f, 2.5f)]));
    }

    [Fact]
    public void TestQeeNormalizationOpsCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<QeeNormalizationVariantsCheck>(
            hyperparamInputs: [], runtimeInputs: [FloatMat2x3]));
        Assert.True(AutoTest.AdvancedTestGraph<QeeLayerNormalizationCheck>(
            hyperparamInputs: [], runtimeInputs: [
                FloatMat2x3,
                TensorData(DType.Float32, [3L], 1f, 1f, 1f),
                TensorData(DType.Float32, [3L], 0f, 0f, 0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<QeePReluCheck>(
            hyperparamInputs: [], runtimeInputs: [
                FloatMat2x3,
                TensorData(DType.Float32, [3L], 0.1f, 0.2f, 0.3f)]));
    }

    [Fact]
    public void TestQeeNewLossOpsCoverage()
    {
        // SoftmaxCrossEntropyLoss/NegativeLogLikelihoodLoss: ORT requires the labels' shape to
        // match the input's batch axes; we use a [2, 3] logits with [2] int64 labels.
        Assert.True(AutoTest.AdvancedTestGraph<QeeNewLossOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [
                FloatMat2x3,
                TensorData(DType.Int64, [2L], 0L, 2L)]));
    }

    [Fact]
    public void TestQeeNewSignalOpsCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<QeeNewWindowOpsCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Int64, [], 16L)]));
        Assert.True(AutoTest.AdvancedTestGraph<QeeSTFTCheck>(
            hyperparamInputs: [], runtimeInputs: [
                TensorData(DType.Float32, [1L, 16L, 1L],
                    0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f,
                    8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f),
                TensorData(DType.Int64, [], 4L),
                TensorData(DType.Float32, [4L], 1f, 1f, 1f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<QeeMelWeightMatrixCheck>(
            hyperparamInputs: [], runtimeInputs: [
                TensorData(DType.Int64, [], 8L),
                TensorData(DType.Int64, [], 16L),
                TensorData(DType.Int64, [], 16000L),
                TensorData(DType.Float32, [], 0f),
                TensorData(DType.Float32, [], 8000f)]));
    }

    [Fact]
    public void TestQeeNewMiscOpsCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<QeeOneHotCheck>(
            hyperparamInputs: [], runtimeInputs: [
                TensorData(DType.Int64, [3L], 0L, 1L, 2L),
                TensorData(DType.Int64, [], 3L),
                TensorData(DType.Float32, [2L], 0f, 1f)]));
        // Multinomial — non-deterministic; skip the C# roundtrip since the values won't match.
        Assert.True(AutoTest.AdvancedTestGraph<QeeMultinomialCheck>(
            hyperparamInputs: [], runtimeInputs: [
                TensorData(DType.Float32, [2L, 3L], 0.1f, 0.4f, 0.5f, 0.3f, 0.3f, 0.4f)],
            testCsRoundtrip: false));
    }

    [Fact]
    public void TestQeeNewQuantizationOpsCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<QeeQuantizeLinearCheck>(
            hyperparamInputs: [], runtimeInputs: [
                TensorData(DType.Float32, [2L, 2L], 1.0f, 2.0f, 3.0f, 4.0f),
                TensorData(DType.Float32, [2L], 0.5f, 0.25f),
                TensorData(DType.Int8, [2L], (sbyte)0, (sbyte)1)]));
        Assert.True(AutoTest.AdvancedTestGraph<QeeQLinearMatMulCheck>(
            hyperparamInputs: [], runtimeInputs: [
                TensorData(DType.Int8, [2L, 3L], (sbyte)1, (sbyte)2, (sbyte)3, (sbyte)4, (sbyte)5, (sbyte)6),
                TensorData(DType.Float32, [], 0.5f),
                TensorData(DType.Int8, [], (sbyte)0),
                TensorData(DType.Int8, [3L, 2L], (sbyte)1, (sbyte)0, (sbyte)0, (sbyte)1, (sbyte)1, (sbyte)1),
                TensorData(DType.Float32, [], 0.25f),
                TensorData(DType.Int8, [], (sbyte)0),
                TensorData(DType.Float32, [], 0.5f),
                TensorData(DType.Int8, [], (sbyte)0)]));
        Assert.True(AutoTest.AdvancedTestGraph<QeeQLinearConvCheck>(
            hyperparamInputs: [], runtimeInputs: [
                TensorData(DType.Int8, [1L, 1L, 3L, 3L],
                    (sbyte)1, (sbyte)2, (sbyte)3, (sbyte)4, (sbyte)5,
                    (sbyte)6, (sbyte)7, (sbyte)8, (sbyte)9),
                TensorData(DType.Float32, [], 0.5f),
                TensorData(DType.Int8, [], (sbyte)0),
                TensorData(DType.Int8, [1L, 1L, 2L, 2L], (sbyte)1, (sbyte)0, (sbyte)0, (sbyte)1),
                TensorData(DType.Float32, [], 0.25f),
                TensorData(DType.Int8, [], (sbyte)0),
                TensorData(DType.Float32, [], 0.5f),
                TensorData(DType.Int8, [], (sbyte)0)]));
    }

    /// <summary>SplitToSequence + SequenceAt roundtrip. The sequence output isn't
    /// representable in plain TensorData, so this uses the QeeOnly path.</summary>
    [Fact]
    public void TestQeeNewSequenceOpsCoverage() =>
        Assert.True(QeeOnly<QeeSplitToSequenceCheck>([FloatMat2x3]));

    /// <summary>ImageDecoder shape inference only — a real PNG/JPEG byte stream is
    /// not supplied so ORT would fail to execute this node.</summary>
    [Fact]
    public void TestQeeImageDecoderCoverage() =>
        Assert.True(QeeOnly<QeeImageDecoderCheck>([
            TensorData(DType.UInt8, [4L], (byte)0, (byte)0, (byte)0, (byte)0)]));

    [Fact]
    public void TestQeeTfIdfVectorizerCoverage() =>
        Assert.True(AutoTest.AdvancedTestGraph<QeeTfIdfVectorizerCheck>(
            hyperparamInputs: [], runtimeInputs: [
                TensorData(DType.Int64, [4L], 1L, 2L, 3L, 4L)]));

    /// <summary>
    /// QEE coverage for every <see cref="DType.String"/>-input op: StringConcat,
    /// StringNormalizer, StringSplit, RegexFullMatch. Routed through the QeeOnly helper
    /// because variable-length UTF-8 string tensors don't roundtrip through the
    /// AdvancedTestGraph result-byte comparator (the outputs have no flat byte buffer
    /// to span over). Grouped into two modules so one [Fact] drives all four handlers'
    /// Compute branches, mirroring the V2/V3 module-density convention.
    /// </summary>
    [Fact]
    public void TestQeeStringOpsCoverage()
    {
        Assert.True(QeeOnly<QeeStringConcatRegexCheck>([
            TensorData([3L], "foo", "bar", "baz"),
            TensorData([3L], "1", "2", "3")]));
        Assert.True(QeeOnly<QeeStringNormalizerSplitCheck>([
            TensorData([2L], "Hello World", "the quick brown fox")]));
    }
}

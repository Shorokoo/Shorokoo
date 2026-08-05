using System.Reflection;
using Shorokoo.Runtime;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Inference;
using Shorokoo.Core.Inference.Helpers;

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
        var moduleGraph = ((ComputationGraph)prop.GetValue(null)!).ToInternal();
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

/// <summary>
/// A <c>uint32</c> constant that has to survive host-side constant folding as a
/// <c>uint32</c>: the constant sub-chain (a <c>uint64</c> literal narrowed to
/// <c>uint32</c>) folds, and the folded value then feeds an <c>Add</c> whose other
/// operand is genuinely runtime-valued, so the Add's type constraint sees the folded
/// constant's dtype directly.
/// </summary>
[Module]
public partial class QeeFoldedUnsignedConstant
{
    public static Tensor<uint32> Inline(Tensor<float32> x)
    {
        var runtime = OnnxOp.Range(Scalar(0L), x.ShapeTensor().Reduce(ReduceKind.Prod), Scalar(1L))
            .int64().Cast<uint32>();
        return runtime + Scalar(7UL).Cast<uint32>();
    }
}

/// <summary>
/// The QuickExecutionEngine stores every integer width in one <c>long</c> buffer, so an
/// integer tensor's actual width lives ONLY in <see cref="RuntimeTensor.DType"/>. These
/// tests pin that materializing a runtime tensor back to <see cref="TensorData"/> keeps
/// that width instead of retyping every integer tensor as <c>int64</c> — which silently
/// corrupts host constant folding: a folded <c>uint32</c> constant comes back typed
/// <c>int64</c> and then violates its consumer's type constraint.
/// </summary>
[Trait("Domain", "Inference")]
[Trait("Purpose", "Coverage")]
public class QeeIntegerDTypeRoundTripTests
{
    [Fact]
    public void TestRuntimeTensorRoundTripKeepsIntegerWidth()
    {
        (DType dtype, object[] vals)[] cases = [
            (DType.Int8,   [(sbyte)-3, (sbyte)7]),
            (DType.Int16,  [(short)-300, (short)700]),
            (DType.Int32,  [-70000, 70000]),
            (DType.Int64,  [-5_000_000_000L, 5_000_000_000L]),
            (DType.UInt8,  [(byte)3, (byte)250]),
            (DType.UInt16, [(ushort)7, (ushort)65000]),
            (DType.UInt32, [7u, 4_000_000_000u]),
            (DType.UInt64, [7UL, 18_000_000_000_000_000_000UL]),
        ];
        foreach (var (dtype, vals) in cases)
        {
            var td = TensorData(dtype, [2L], vals);
            var rt = TensorDataConverter.ToRuntimeTensor(td, maxElements: 16);
            var back = TensorDataConverter.ToTensorData(rt);
            Assert.NotNull(back);
            Assert.Equal(dtype, back!.DType);
            Assert.Equal(td.AccessRawMemory().ToArray(), back.AccessRawMemory().ToArray());
        }
    }

    [Fact]
    public void TestFoldedUnsignedConstantKeepsItsDType()
    {
        // End-to-end consequence: building this model host-folds the uint32 constant and
        // then re-derives the Add's output type from its inputs. If folding retyped the
        // constant to int64 the Add's type constraint is violated at graph construction.
        var g = ((ComputationGraph)typeof(QeeFoldedUnsignedConstant)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var input = TensorData([2L, 2L], 0f, 0f, 0f, 0f);
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([input])).ToConcreteModel();
        var outputs = ComputeContext.Default.Execute(concrete, input);
        Assert.Equal(DType.UInt32, outputs[0].ToTensorData().DType);
        Assert.Equal((uint[])[7u, 8u, 9u, 10u],
            outputs[0].ToTensorData().As<uint32>().AccessMemory().ToArray());
    }
}

/// <summary>
/// A <c>uint32</c> chain that host-constant-folds only in part: a fully-constant Threefry
/// bijection produces the key words, which then key a second bijection over a
/// <b>runtime-shaped</b> counter. Only the first bijection (and the constant-only head of the
/// second) folds host-side, so the folded values re-enter the graph and every later op reads
/// them — including a right shift, which is where a value carrying bits above its declared
/// 32-bit width becomes visible.
/// </summary>
[Module]
public partial class QeePartiallyFoldedUInt32Chain
{
    public static Tensor<uint32> Inline(Tensor<float32> x)
    {
        var (a0, a1) = Shorokoo.Core.Rng.RuntimeRng.Bijection(
            Scalar(0u), Scalar(0u), Scalar(123u), Scalar(456u));
        var c0 = OnnxOp.Range(Scalar(0L), x.ShapeTensor().Reduce(ReduceKind.Prod), Scalar(1L))
            .int64().Cast<uint32>();
        var (b0, _) = Shorokoo.Core.Rng.RuntimeRng.Bijection(c0, Scalar(0u), a0, a1);
        return b0;
    }
}

/// <summary>
/// The QuickExecutionEngine keeps every integer width in one <c>long</c> buffer. These tests
/// pin that an op's result is narrowed to its tensor's <b>declared</b> width, not left carrying
/// the extra bits an overflowing 64-bit computation produced. Leaked high bits are invisible to
/// the next add (which is exact mod 2^32 either way) but not to a right shift, which pulls them
/// straight down into the result — so a folded uint32 chain silently computed wrong values.
/// </summary>
[Trait("Domain", "Inference")]
[Trait("Purpose", "Coverage")]
public class QeeIntegerWidthTests
{
    [Fact]
    public void TestOverflowingUnsignedAddDoesNotLeakIntoALaterShift()
    {
        // (2^32 - 1) + 1 is 0 in uint32, so >> 4 is 0 and the result is just the element index.
        // Carrying the sum as 2^32 in QEE's 64-bit buffer instead makes the shift yield 2^28.
        var g = ((ComputationGraph)typeof(QeeOverflowThenShift)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var input = TensorData([2L, 2L], 0f, 0f, 0f, 0f);
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([input])).ToConcreteModel();
        var got = ComputeContext.Default.Execute(concrete, input)[0]
            .ToTensorData().As<uint32>().AccessMemory().ToArray();
        Assert.Equal((uint[])[0u, 1u, 2u, 3u], got);
    }

    [Fact]
    public void TestFloat64ConstantIsNotFoldedThroughTheFloat32Buffer()
    {
        // QEE's float buffer is float32, so a float64 constant cannot survive a host fold intact.
        // Materializing it anyway is a choice between two wrongs — retype it (breaks the
        // consumer's type constraint) or stamp Float64 on float32-rounded values (silently wrong,
        // and 1e300 becomes Infinity). It must decline to fold instead.
        var g = ((ComputationGraph)typeof(QeeFoldedFloat64Constant)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var input = TensorData([2L, 2L], 1f, 2f, 3f, 4f);
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([input])).ToConcreteModel();
        var td = ComputeContext.Default.Execute(concrete, input)[0].ToTensorData();
        Assert.Equal(DType.Float64, td.DType);
        Assert.Equal((double[])[7.0, 8.0, 9.0, 10.0], td.As<float64>().AccessMemory().ToArray());

        // The discriminating half: a value the float32 buffer cannot hold. Folding it through
        // that buffer would return Infinity; declining to fold keeps it exact.
        var wide = ((ComputationGraph)typeof(QeeWideFloat64Constant)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var wideConcrete = wide.ToConcreteArchitecture(wide.FromOrderedInputs([input])).ToConcreteModel();
        var wideTd = ComputeContext.Default.Execute(wideConcrete, input)[0].ToTensorData();
        Assert.All(wideTd.As<float64>().AccessMemory().ToArray(), v => Assert.True(double.IsFinite(v)));
        Assert.Equal(1e300, wideTd.As<float64>().AccessMemory().ToArray()[0]);
    }

    // QEE may decline to fold (no data — the backend computes it instead); it must never fold wrong.
    private static void Qee<TModule>(DType dtype, params ulong[] expected)
    {
        var g = ((ComputationGraph)typeof(TModule)
            .GetProperty("ComputationGraph", BindingFlags.Public | BindingFlags.Static)!.GetValue(null)!).ToInternal();
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([])).ToConcreteModel();
        var rt = new QuickExecutionEngine().Run(concrete)[concrete.Outputs[0]];
        Assert.Equal(dtype, rt.DType);
        if (rt is RuntimeTensor { IntData: { } d })
            Assert.Equal(expected, d.Select(v => unchecked((ulong)v)));
    }

    [Fact]
    public void TestUInt32OpsWrapAtTheWidthBoundary()
    {
        Qee<QeeU32Add>(DType.UInt32, 0, 1, 2147483650, 4294967294, 1);
        Qee<QeeU32Sub>(DType.UInt32, 4294967295, 4294967294, 4294967295, 0);
        Qee<QeeU32Mul>(DType.UInt32, 0, 4294967294, 0, 1);
        Qee<QeeU32Shift>(DType.UInt32, 2147483648, 65536, 4294967294, 65535, 1, 0);
    }

    [Fact]
    public void TestUInt64OpsWrapAboveLongMaxValue()
    {
        Qee<QeeU64Add>(DType.UInt64, 0, 1, 9223372036854775808, 18446744073709551614);
        Qee<QeeU64Sub>(DType.UInt64, 18446744073709551615, 18446744073709551614, 9223372036854775809, 0);
        Qee<QeeU64Mul>(DType.UInt64, 0, 18446744073709551614, 0, 1);
        Qee<QeeU64Shift>(DType.UInt64, 9223372036854775808, 4294967296, 4294967295, 1);
        Qee<QeeU64Bitwise>(DType.UInt64,
            9223372036854775808, 9223372036854775808, 4294967295,
            9223372036854775807, 0, 18446744069414584320,
            18446744073709551615, 9223372036854775808, 18446744073709551615);
        Qee<QeeU64Cast>(DType.UInt64, 4294967295, 0, 7);
    }

    [Fact]
    public void TestPartiallyFoldedUInt32ChainMatchesTheHostGenerator()
    {
        // Threefry is the real-world instance: RuntimeRng's rotate is a shift pair, so a leaked
        // bit lands inside the result. Compared against the independent host generator.
        var g = ((ComputationGraph)typeof(QeePartiallyFoldedUInt32Chain)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var input = TensorData([2L, 2L], 0f, 0f, 0f, 0f);
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([input])).ToConcreteModel();
        var got = ComputeContext.Default.Execute(concrete, input)[0]
            .ToTensorData().As<uint32>().AccessMemory().ToArray();

        var (a0, a1) = Shorokoo.Core.Rng.Threefry2x32.Bijection(0u, 0u, 123u, 456u);
        var want = System.Linq.Enumerable.Range(0, 4)
            .Select(i => Shorokoo.Core.Rng.Threefry2x32.Bijection((uint)i, 0u, a0, a1).Item1).ToArray();
        Assert.Equal(want, got);
    }
}

/// <summary>
/// A uint32 add that overflows, then a right shift of the sum — with the result added to a
/// runtime-valued tensor so the constant sub-chain is actually FORCED through the host folder.
/// (A fully-constant graph folds nothing: FastFoldConstants only materializes a constant that a
/// non-constant node consumes, so the whole chain would fall through to ORT and pin nothing.)
/// </summary>
[Module]
public partial class QeeOverflowThenShift
{
    public static Tensor<uint32> Inline(Tensor<float32> x)
    {
        var folded = OnnxOp.BitShift(Scalar(4_294_967_295u) + Scalar(1u), Scalar(4u),
            BitShiftDirection.Right).uint32();
        var runtime = OnnxOp.Range(Scalar(0L), x.ShapeTensor().Reduce(ReduceKind.Prod), Scalar(1L))
            .int64().Cast<uint32>();
        return runtime + folded;
    }
}

/// <summary>Same shape, but with a constant beyond float32's range: 1e300 survives only if the
/// chain is never routed through QEE's float32 buffer.</summary>
[Module]
public partial class QeeWideFloat64Constant
{
    public static Tensor<float64> Inline(Tensor<float32> x)
    {
        var runtime = x.Reshape([Scalar(-1L)]).Cast<float64>() * Scalar(0.0);
        return runtime + (Scalar(1e300) * Scalar(1.0));
    }
}

/// <summary>A constant float64 sub-chain feeding a runtime-valued float64 tensor. QEE's float
/// buffer is float32, so it must NOT materialize this as a folded constant — neither retyped to
/// Float32 (which violates the consumer's type constraint) nor stamped Float64 over float32-rounded
/// values. Refusing to fold leaves the real ops for a backend that computes at genuine float64.</summary>
[Module]
public partial class QeeFoldedFloat64Constant
{
    public static Tensor<float64> Inline(Tensor<float32> x)
    {
        var runtime = x.Reshape([Scalar(-1L)]).Cast<float64>();
        return runtime + (Scalar(3.0) * Scalar(2.0));
    }
}

/// <summary>A uint64 divide whose operands are constants, consumed by a runtime-valued tensor so
/// the divide is forced through host constant folding. The dividend is above long.MaxValue.</summary>
[Module]
public partial class QeeUInt64SignedDivide
{
    public static Tensor<uint64> Inline(Tensor<float32> x)
    {
        var runtime = OnnxOp.Range(Scalar(0L), x.ShapeTensor().Reduce(ReduceKind.Prod), Scalar(1L))
            .int64().Cast<uint64>();
        var folded = OnnxOp.Div(Scalar(9223372036854775808UL), Scalar(2UL)).uint64();
        return runtime + folded;
    }
}

/// <summary>A uint64 modulo whose operands are constants, forced through host constant folding the
/// same way as <see cref="QeeUInt64SignedDivide"/>. Both dividends are above long.MaxValue.</summary>
[Module]
public partial class QeeUInt64SignedModulo
{
    public static Tensor<uint64> Inline(Tensor<float32> x)
    {
        var runtime = OnnxOp.Range(Scalar(0L), x.ShapeTensor().Reduce(ReduceKind.Prod), Scalar(1L))
            .int64().Cast<uint64>();
        // 2^63 % 1000 == 808 unsigned; the signed floored modulo of -2^63 gives 192 instead.
        var folded = OnnxOp.Mod(Scalar(9223372036854775808UL), Scalar(1000UL)).uint64();
        return runtime + folded;
    }
}

/// <summary>A uint64 divide whose dividend is <c>ulong.MaxValue</c> — the all-ones bit pattern,
/// which signed division reads as <c>-1</c> and so collapses to 0 for any divisor &gt; 1.</summary>
[Module]
public partial class QeeUInt64SignedDivideMaxValue
{
    public static Tensor<uint64> Inline(Tensor<float32> x)
    {
        var runtime = OnnxOp.Range(Scalar(0L), x.ShapeTensor().Reduce(ReduceKind.Prod), Scalar(1L))
            .int64().Cast<uint64>();
        var folded = OnnxOp.Div(Scalar(18446744073709551615UL), Scalar(3UL)).uint64();
        return runtime + folded;
    }
}

/// <summary>
/// QEE holds every integer width in one <c>long</c> buffer, so a <c>uint64</c> above
/// <c>long.MaxValue</c> is a negative bit-pattern long — and Div/Mod/Less/Greater/Sign/Abs read it
/// with signed C# operators. Host constant folding runs those kernels and bakes the result into the
/// graph, so the wrong value is persisted, not merely displayed.
/// </summary>
[Trait("Domain", "Inference")]
[Trait("Purpose", "Coverage")]
public class QeeUInt64SignedOperatorTests
{
    // Skipped against https://github.com/Shorokoo/Shorokoo/issues/141 — QEE's uint64 kernels use
    // signed operators. Self-checking: deleting the Skip flips this green the moment #141 is fixed.
    [Fact(Skip = "QEE uint64 kernels use signed operators — Shorokoo/Shorokoo#141")]
    public void TestFoldedUInt64DivideUsesUnsignedSemantics()
    {
        var g = ((ComputationGraph)typeof(QeeUInt64SignedDivide)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var input = TensorData([2L, 2L], 0f, 0f, 0f, 0f);
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([input])).ToConcreteModel();
        var got = ComputeContext.Default.Execute(concrete, input)[0]
            .ToTensorData().As<uint64>().AccessMemory().ToArray();

        // 2^63 / 2 == 2^62, plus the element index. Signed division of the bit pattern gives
        // -2^62, i.e. 2^64 - 2^62 = 13835058055282163712.
        const ulong half = 4611686018427387904UL;   // 2^62
        Assert.Equal((ulong[])[half, half + 1, half + 2, half + 3], got);
    }

    // Same fault, Mod rather than Div — pinned separately because #133's bits packing is specified
    // as `(word / 2^(W*l)) mod 2^W`, so a literal implementation reaches BOTH operators.
    [Fact(Skip = "QEE uint64 kernels use signed operators — Shorokoo/Shorokoo#141")]
    public void TestFoldedUInt64ModuloUsesUnsignedSemantics()
    {
        var g = ((ComputationGraph)typeof(QeeUInt64SignedModulo)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var input = TensorData([2L, 2L], 0f, 0f, 0f, 0f);
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([input])).ToConcreteModel();
        var got = ComputeContext.Default.Execute(concrete, input)[0]
            .ToTensorData().As<uint64>().AccessMemory().ToArray();

        // 2^63 % 1000 == 808, plus the element index. Signed reads the bit pattern as -2^63, and
        // ONNX Mod (fmod=0) is FLOORED rather than truncated, so it returns 1000 - 808 == 192 —
        // a plausible-looking small remainder, which is what makes this one easy to miss.
        const ulong rem = 808UL;
        Assert.Equal((ulong[])[rem, rem + 1, rem + 2, rem + 3], got);
    }

    // The all-ones dividend: signed division reads ulong.MaxValue as -1, so ANY divisor > 1
    // collapses the result to 0 — the most destructive shape of this bug, since it survives every
    // "is it roughly right?" eyeball check.
    [Fact(Skip = "QEE uint64 kernels use signed operators — Shorokoo/Shorokoo#141")]
    public void TestFoldedUInt64DivideOfMaxValueUsesUnsignedSemantics()
    {
        var g = ((ComputationGraph)typeof(QeeUInt64SignedDivideMaxValue)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var input = TensorData([2L, 2L], 0f, 0f, 0f, 0f);
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([input])).ToConcreteModel();
        var got = ComputeContext.Default.Execute(concrete, input)[0]
            .ToTensorData().As<uint64>().AccessMemory().ToArray();

        // (2^64 - 1) / 3 == 6148914691236517205, plus the element index. Signed gives -1 / 3 == 0.
        const ulong third = 6148914691236517205UL;
        Assert.Equal((ulong[])[third, third + 1, third + 2, third + 3], got);
    }
}

[Module] public partial class QeeU32Add { public static Tensor<uint32> Inline()
    => Vector(4294967295u, 0u, 2147483648u, 4294967295u, 4294967295u) + Vector(1u, 1u, 2u, 4294967295u, 2u); }

[Module] public partial class QeeU32Sub { public static Tensor<uint32> Inline()
    => Vector(0u, 0u, 1u, 4294967295u) - Vector(1u, 2u, 2u, 4294967295u); }

[Module] public partial class QeeU32Mul { public static Tensor<uint32> Inline()
    => Vector(2147483648u, 4294967295u, 65536u, 4294967295u) * Vector(2u, 2u, 65536u, 4294967295u); }

[Module] public partial class QeeU32Shift { public static Tensor<uint32> Inline()
    => (Tensor<uint32>)OnnxOp.Concat([
        OnnxOp.BitShift(Vector(1u, 1u, 4294967295u), Vector(31u, 16u, 1u), BitShiftDirection.Left),
        OnnxOp.BitShift(Vector(4294967295u, 2147483648u, 1u), Vector(16u, 31u, 1u), BitShiftDirection.Right)], axis: 0); }

[Module] public partial class QeeU64Add { public static Tensor<uint64> Inline()
    => Vector(18446744073709551615UL, 0UL, 9223372036854775808UL, 18446744073709551615UL)
     + Vector(1UL, 1UL, 0UL, 18446744073709551615UL); }

[Module] public partial class QeeU64Sub { public static Tensor<uint64> Inline()
    => Vector(0UL, 0UL, 9223372036854775808UL, 18446744073709551615UL)
     - Vector(1UL, 2UL, 18446744073709551615UL, 18446744073709551615UL); }

[Module] public partial class QeeU64Mul { public static Tensor<uint64> Inline()
    => Vector(9223372036854775808UL, 18446744073709551615UL, 4294967296UL, 18446744073709551615UL)
     * Vector(2UL, 2UL, 4294967296UL, 18446744073709551615UL); }

[Module] public partial class QeeU64Shift { public static Tensor<uint64> Inline()
    => (Tensor<uint64>)OnnxOp.Concat([
        OnnxOp.BitShift(Vector(1UL, 1UL), Vector(63UL, 32UL), BitShiftDirection.Left),
        OnnxOp.BitShift(Vector(18446744073709551615UL, 9223372036854775808UL), Vector(32UL, 63UL),
            BitShiftDirection.Right)], axis: 0); }

[Module] public partial class QeeU64Bitwise { public static Tensor<uint64> Inline()
{
    var hi = Vector(18446744073709551615UL, 9223372036854775808UL, 18446744073709551615UL);
    var lo = Vector(9223372036854775808UL, 9223372036854775808UL, 4294967295UL);
    return (Tensor<uint64>)OnnxOp.Concat(
        [OnnxOp.BitwiseAnd(hi, lo), OnnxOp.BitwiseXor(hi, lo), OnnxOp.BitwiseOr(hi, lo)], axis: 0);
} }

[Module] public partial class QeeU64Cast { public static Tensor<uint64> Inline()
    => Vector(18446744073709551615UL, 9223372036854775808UL, 4294967303UL).Cast<uint32>().Cast<uint64>(); }

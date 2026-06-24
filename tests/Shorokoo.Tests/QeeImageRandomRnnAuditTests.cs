using System.Reflection;
using Shorokoo.Runtime;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Inference;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage tests for the Phase 4 QEE-A5 audit batch (image/geometry, random/generator, and
/// recurrent families, ONNX opset 21). Each module in QeeImageRandomRnnAuditModules.cs is
/// self-checking (single Scalar&lt;bit&gt;) and is driven by
/// <see cref="AutoTest.AdvancedTestGraph{TModule}"/> (validates expectations against real
/// ONNX Runtime) plus, where the bit is QEE-computable, a QeeSelfCheck pass (validates that
/// QuickExecutionEngine's own shape/value inference reproduces the expectations). Outputs
/// whose shapes legitimately stay unknown at QEE time (NonMaxSuppression's data-dependent n,
/// ImageDecoder's data-dependent H/W, Constant string tensors) are asserted by direct
/// <see cref="RuntimeTensor"/> inspection instead — the audit contract is that they degrade
/// to a null shape with the correct rank/dtype, never to guessed or negative dims.
/// </summary>
[Trait("Domain", "Inference")]
[Trait("Purpose", "Coverage")]
public class QeeImageRandomRnnAuditTests
{
    /// <summary>Same QEE-only strong self-check as QeePoolConvAuditTests: lowers the module
    /// like AdvancedTestGraph, runs only the QuickExecutionEngine, and requires the single
    /// Scalar&lt;bit&gt; output to be concretely computed as true.</summary>
    private static bool QeeSelfCheck<TModule>(TensorData[] runtimeInputs)
    {
        var store = QeeRun<TModule>(runtimeInputs, out var concreteModel);
        foreach (var outKey in concreteModel.Outputs)
        {
            if (!store.TryGetValue(outKey, out var rt)) return false;
            if (rt is not RuntimeTensor { BoolData: { Length: 1 } bits } plain
                || plain.DType != DType.Bool || !bits[0])
                return false;
        }
        return true;
    }

    /// <summary>Runs a module through QEE only and returns its output runtime tensors in
    /// declaration order — used to inspect outputs whose shapes are intentionally unknown
    /// (rank-only degradation) and string tensors that don't fit the bit-check pattern.</summary>
    private static IRuntimeTensor[] QeeOutputs<TModule>(TensorData[] runtimeInputs)
    {
        var store = QeeRun<TModule>(runtimeInputs, out var concreteModel);
        return [.. concreteModel.Outputs.Select(k =>
            store.TryGetValue(k, out var rt) ? rt : throw new InvalidOperationException($"missing output {k}"))];
    }

    private static Dictionary<FastTensorKey, IRuntimeTensor> QeeRun<TModule>(
        TensorData[] runtimeInputs, out FastComputationGraph concreteModel)
    {
        var prop = typeof(TModule).GetProperty("ComputationGraph", BindingFlags.Public | BindingFlags.Static)
            ?? throw new InvalidOperationException($"{typeof(TModule).FullName} has no public static ComputationGraph property");
        var moduleGraph = (FastComputationGraph)prop.GetValue(null)!;
        var concreteArch = moduleGraph.ToConcreteArchitecture(moduleGraph.FromOrderedInputs([.. runtimeInputs]));
        concreteModel = concreteArch.ToConcreteModel();
        var qee = new QuickExecutionEngine();
        return runtimeInputs.Length == 0 ? qee.Run(concreteModel) : qee.Run(concreteModel, runtimeInputs);
    }

    // ----- image / geometry ------------------------------------------------

    [Fact]
    public void TestQeeResizeShapeAudit()
    {
        var x = TensorDataWithDefaultVals(DType.Float32, [1L, 1L, 8L, 8L]);
        Assert.True(AutoTest.AdvancedTestGraph<QeeResizeShapeAuditCheck>(
            hyperparamInputs: [], runtimeInputs: [x]));
        Assert.True(QeeSelfCheck<QeeResizeShapeAuditCheck>([x]));
        // Negative axes (spec opset 18+) — QEE-only: ORT 1.25.1's Resize kernel rejects
        // negative axes ("Scale value should be greater than 0").
        Assert.True(QeeSelfCheck<QeeResizeNegativeAxesAuditCheck>([x]));
    }

    [Fact]
    public void TestQeeUpsampleAffineGridSampleShapeAudit()
    {
        var x = TensorDataWithDefaultVals(DType.Float32, [1L, 2L, 4L, 4L]);
        Assert.True(AutoTest.AdvancedTestGraph<QeeUpsampleAffineGridSampleAuditCheck>(
            hyperparamInputs: [], runtimeInputs: [x]));
        Assert.True(QeeSelfCheck<QeeUpsampleAffineGridSampleAuditCheck>([x]));
    }

    [Fact]
    public void TestQeeAffineGridSample5DShapeAudit()
    {
        var x5 = TensorDataWithDefaultVals(DType.Float32, [1L, 1L, 3L, 4L, 4L]);
        Assert.True(AutoTest.AdvancedTestGraph<QeeAffineGridSample5DAuditCheck>(
            hyperparamInputs: [], runtimeInputs: [x5]));
        Assert.True(QeeSelfCheck<QeeAffineGridSample5DAuditCheck>([x5]));
    }

    [Fact]
    public void TestQeeRoiAlignShapeAudit()
    {
        TensorData[] inputs = [
            TensorDataWithDefaultVals(DType.Float32, [1L, 2L, 8L, 8L]),
            TensorData(DType.Float32, [3L, 4L],
                0f, 0f, 4f, 4f,
                1f, 1f, 6f, 6f,
                2f, 2f, 7f, 7f),
            TensorData(DType.Int64, [3L], 0L, 0L, 0L)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeRoiAlignShapeAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeRoiAlignShapeAuditCheck>(inputs));
    }

    private static readonly TensorData NmsBoxes = TensorData(DType.Float32, [1L, 4L, 4L],
        // corner format [y1, x1, y2, x2]; 0/1 overlap (IoU ≈ 0.6), 2/3 overlap.
        0.0f, 0.0f, 1.0f, 1.0f,
        0.0f, 0.1f, 1.0f, 1.1f,
        5.0f, 5.0f, 6.0f, 6.0f,
        5.0f, 5.1f, 6.0f, 6.1f);

    private static readonly TensorData NmsScores = TensorData(DType.Float32, [1L, 1L, 4L],
        0.9f, 0.8f, 0.7f, 0.6f);

    /// <summary>The data-dependent [n,3] output: ORT validates the expected n per attribute
    /// combo; QEE degrades to an unknown shape (rank 2, int64) with the
    /// batches*classes*min(spatial, max_boxes) MaxShape cap — asserted by inspection.</summary>
    [Fact]
    public void TestQeeNonMaxSuppressionShapeAudit()
    {
        Assert.True(AutoTest.AdvancedTestGraph<QeeNmsOrtShapeAuditCheck>(
            hyperparamInputs: [], runtimeInputs: [NmsBoxes, NmsScores]));

        var nms = Assert.IsType<RuntimeTensor>(QeeOutputs<QeeNmsRankOnlyCheck>([NmsBoxes, NmsScores]).Single());
        Assert.Equal(DType.Int64, nms.DType);
        Assert.Null(nms.Shape);
        Assert.Equal(2, nms.Rank);
        Assert.Equal(2, nms.MaxRank);
        Assert.NotNull(nms.MaxShape);
        Assert.Equal([2L, 3L], nms.MaxShape!.Dims); // 1 batch * 1 class * min(4, max_boxes=2)
    }

    /// <summary>max_output_boxes_per_class absent → spec default 0 → exactly [0,3]; this
    /// branch is concrete, so both ORT and QEE verify it.</summary>
    [Fact]
    public void TestQeeNonMaxSuppressionEmptyDefaultAudit()
    {
        Assert.True(AutoTest.AdvancedTestGraph<QeeNmsEmptyAuditCheck>(
            hyperparamInputs: [], runtimeInputs: [NmsBoxes, NmsScores]));
        Assert.True(QeeSelfCheck<QeeNmsEmptyAuditCheck>([NmsBoxes, NmsScores]));
    }

    [Fact]
    public void TestQeeCol2ImCenterCropPadAudit()
    {
        TensorData[] inputs = [
            TensorDataWithDefaultVals(DType.Float32, [1L, 8L, 12L]),
            TensorData(DType.Float32, [3L, 5L],
                1f, 2f, 3f, 4f, 5f,
                6f, 7f, 8f, 9f, 10f,
                11f, 12f, 13f, 14f, 15f)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeCol2ImCenterCropPadAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeCol2ImCenterCropPadAuditCheck>(inputs));
    }

    /// <summary>ImageDecoder: data-dependent H/W → QEE must degrade to a rank-3 uint8
    /// tensor with NO guessed dims (it used to emit [-1,-1,channels]). A real encoded
    /// stream isn't supplied, so ORT execution is skipped (as in TestQeeImageDecoderCoverage).</summary>
    [Fact]
    public void TestQeeImageDecoderRankOnlyAudit()
    {
        var img = Assert.IsType<RuntimeTensor>(QeeOutputs<QeeImageDecoderCheck>([
            TensorData(DType.UInt8, [4L], (byte)0, (byte)0, (byte)0, (byte)0)]).Single());
        Assert.Equal(DType.UInt8, img.DType);
        Assert.Null(img.Shape);
        Assert.Equal(3, img.Rank);
        Assert.Equal(3, img.MaxRank);
    }

    // ----- random / generator ----------------------------------------------

    [Fact]
    public void TestQeeRandomFamilyShapeDtypeAudit()
    {
        TensorData[] inputs = [
            TensorData(DType.Float32, [2L, 3L], 0.1f, 0.5f, 0.9f, 0.3f, 0.7f, 0.2f),
            TensorData(DType.Float32, [2L, 4L], 0.1f, 0.4f, 0.3f, 0.2f, 0.25f, 0.25f, 0.25f, 0.25f)];
        // Random values won't survive the C# codegen roundtrip — same flag as QeeRandomOpsCheck.
        Assert.True(AutoTest.AdvancedTestGraph<QeeRandomFamilyAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs, testCsRoundtrip: false));
        Assert.True(QeeSelfCheck<QeeRandomFamilyAuditCheck>(inputs));
    }

    /// <summary>Seeded determinism is defined by the spec (fixed seed → reproducible
    /// stream); ORT-only since QEE never computes random values.</summary>
    [Fact]
    public void TestQeeRandomSeededDeterminism() =>
        Assert.True(AutoTest.AdvancedTestGraph<QeeRandomSeededDeterminismCheck>(
            hyperparamInputs: [], runtimeInputs: [], testCsRoundtrip: false));

    [Fact]
    public void TestQeeRangeConstantOfShapeAudit()
    {
        Assert.True(AutoTest.AdvancedTestGraph<QeeRangeConstantOfShapeAuditCheck>(
            hyperparamInputs: [], runtimeInputs: []));
        Assert.True(QeeSelfCheck<QeeRangeConstantOfShapeAuditCheck>([]));
    }

    /// <summary>Constant value_string / value_strings (QEE used to Debug.Assert and emit an
    /// Invalid tensor for these). Inspected directly — string tensors don't fit the
    /// arithmetic bit-check.</summary>
    [Fact]
    public void TestQeeConstantStringAudit()
    {
        var outputs = QeeOutputs<QeeConstantStringCheck>([]);
        var cs = Assert.IsType<RuntimeTensor>(outputs[0]);
        Assert.Equal(DType.String, cs.DType);
        Assert.Equal(System.Array.Empty<long>(), cs.Shape!.Dims);
        Assert.Equal(new[] { "hello" }, cs.StringData!.Value.ToArray());
        var css = Assert.IsType<RuntimeTensor>(outputs[1]);
        Assert.Equal(DType.String, css.DType);
        Assert.Equal(new[] { 3L }, css.Shape!.Dims);
        Assert.Equal(new[] { "a", "b", "c" }, css.StringData!.Value.ToArray());
    }

    // ----- recurrent family (shape-only: QEE computes no recurrent values) --

    private static readonly TensorData RecurrentX =
        TensorDataWithDefaultVals(DType.Float32, [4L, 2L, 3L]);

    [Fact]
    public void TestQeeRnnShapeAudit()
    {
        Assert.True(AutoTest.AdvancedTestGraph<QeeRnnShapeAuditCheck>(
            hyperparamInputs: [], runtimeInputs: [RecurrentX]));
        Assert.True(QeeSelfCheck<QeeRnnShapeAuditCheck>([RecurrentX]));
    }

    /// <summary>layout=1 and the hidden_size-from-W inference are QEE-only: ORT's CPU
    /// recurrent kernels mandate the hidden_size attribute and reject layout=1 outright
    /// ("Batchwise recurrent operations (layout == 1) are not supported").</summary>
    [Fact]
    public void TestQeeRecurrentLayoutAndHiddenInferenceAudit() =>
        Assert.True(QeeSelfCheck<QeeRecurrentQeeOnlyShapeAuditCheck>([RecurrentX]));

    [Fact]
    public void TestQeeGruShapeAudit()
    {
        Assert.True(AutoTest.AdvancedTestGraph<QeeGruShapeAuditCheck>(
            hyperparamInputs: [], runtimeInputs: [RecurrentX]));
        Assert.True(QeeSelfCheck<QeeGruShapeAuditCheck>([RecurrentX]));
    }

    [Fact]
    public void TestQeeLstmShapeAudit()
    {
        TensorData[] inputs = [
            RecurrentX,
            TensorData(DType.Int32, [2L], 4, 4)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeLstmShapeAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeLstmShapeAuditCheck>(inputs));
    }
}

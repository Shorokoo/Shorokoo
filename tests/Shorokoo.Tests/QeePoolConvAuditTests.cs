using System.Reflection;
using Shorokoo.Runtime;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Inference;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage tests for the Phase 4 QEE-A1 audit batch (pooling &amp; convolution family,
/// ONNX opset 21). Each module in QeePoolConvAuditModules.cs is self-checking: it compares
/// the ShapeTensor() of every op result against the spec-expected dims and returns a single
/// Scalar&lt;bit&gt;. Each module is driven twice:
/// <list type="bullet">
///   <item><see cref="AutoTest.AdvancedTestGraph{TModule}"/> — validates the expected dims
///         against real ONNX Runtime execution (plus the ONNX/CS roundtrips).</item>
///   <item><see cref="QeeSelfCheck{TModule}"/> — runs the same graph through
///         <see cref="QuickExecutionEngine"/> only and asserts the self-check bit computed
///         from QEE's own inferred shapes is true, so a QEE shape-inference regression
///         fails the test even though AdvancedTestGraph's QEE pass only checks dtypes.</item>
/// </list>
/// </summary>
[Trait("Domain", "Inference")]
[Trait("Purpose", "Coverage")]
public class QeePoolConvAuditTests
{
    /// <summary>
    /// QEE-only strong self-check: lowers the module exactly like AdvancedTestGraph, runs
    /// only the QuickExecutionEngine, and requires the module's single Scalar&lt;bit&gt;
    /// output to be concretely computed as true. Because the modules derive the bit from
    /// Shape ops over the audited op results, the bit is false whenever QEE infers a wrong
    /// output dim. Also used (like QeeOnly in QeeOpsTests.cs) for modules ORT rejects —
    /// MaxUnpool's float-typed indices input.
    /// </summary>
    private static bool QeeSelfCheck<TModule>(TensorData[] runtimeInputs)
    {
        var prop = typeof(TModule).GetProperty("ComputationGraph", BindingFlags.Public | BindingFlags.Static)
            ?? throw new InvalidOperationException($"{typeof(TModule).FullName} has no public static ComputationGraph property");
        var moduleGraph = ((ComputationGraph)prop.GetValue(null)!).ToInternal();
        var concreteArch = moduleGraph.ToConcreteArchitecture(moduleGraph.FromOrderedInputs([.. runtimeInputs]));
        var concreteModel = concreteArch.ToConcreteModel();
        var qee = new QuickExecutionEngine();
        var store = runtimeInputs.Length == 0 ? qee.Run(concreteModel) : qee.Run(concreteModel, runtimeInputs);
        foreach (var outKey in concreteModel.Outputs)
        {
            if (!store.TryGetValue(outKey, out var rt)) return false;
            if (rt is not RuntimeTensor { BoolData: { Length: 1 } bits } plain
                || plain.DType != DType.Bool || !bits[0])
                return false;
        }
        return true;
    }

    private static readonly TensorData Image1x1x10x10 =
        TensorDataWithDefaultVals(DType.Float32, [1L, 1L, 10L, 10L]);

    private static readonly TensorData Image1x3x10x10 =
        TensorDataWithDefaultVals(DType.Float32, [1L, 3L, 10L, 10L]);

    [Fact]
    public void TestQeeMaxPoolShapeAudit()
    {
        Assert.True(AutoTest.AdvancedTestGraph<QeeMaxPoolShapeAuditCheck>(
            hyperparamInputs: [], runtimeInputs: [Image1x1x10x10]));
        Assert.True(QeeSelfCheck<QeeMaxPoolShapeAuditCheck>([Image1x1x10x10]));
    }

    [Fact]
    public void TestQeeAveragePoolShapeAudit()
    {
        Assert.True(AutoTest.AdvancedTestGraph<QeeAveragePoolShapeAuditCheck>(
            hyperparamInputs: [], runtimeInputs: [Image1x1x10x10]));
        Assert.True(QeeSelfCheck<QeeAveragePoolShapeAuditCheck>([Image1x1x10x10]));
    }

    [Fact]
    public void TestQeeLpPoolAndGlobalPoolShapeAudit()
    {
        Assert.True(AutoTest.AdvancedTestGraph<QeeLpPoolGlobalPoolShapeAuditCheck>(
            hyperparamInputs: [], runtimeInputs: [Image1x3x10x10]));
        Assert.True(QeeSelfCheck<QeeLpPoolGlobalPoolShapeAuditCheck>([Image1x3x10x10]));
    }

    [Fact]
    public void TestQeeConvShapeAudit()
    {
        var x = TensorDataWithDefaultVals(DType.Float32, [1L, 4L, 9L, 9L]);
        Assert.True(AutoTest.AdvancedTestGraph<QeeConvShapeAuditCheck>(
            hyperparamInputs: [], runtimeInputs: [x]));
        Assert.True(QeeSelfCheck<QeeConvShapeAuditCheck>([x]));
    }

    [Fact]
    public void TestQeeConvTransposeShapeAudit()
    {
        var x = TensorDataWithDefaultVals(DType.Float32, [1L, 2L, 5L, 5L]);
        Assert.True(AutoTest.AdvancedTestGraph<QeeConvTransposeShapeAuditCheck>(
            hyperparamInputs: [], runtimeInputs: [x]));
        Assert.True(QeeSelfCheck<QeeConvTransposeShapeAuditCheck>([x]));
    }

    [Fact]
    public void TestQeeQuantizedConvShapeAudit()
    {
        TensorData[] inputs = [
            TensorDataWithDefaultVals(DType.Int8, [1L, 1L, 7L, 7L]),
            TensorDataWithDefaultVals(DType.Int8, [1L, 1L, 3L, 3L]),
            TensorData(DType.Int8, [], (sbyte)0),
            TensorData(DType.Int8, [], (sbyte)0),
            TensorData(DType.Float32, [], 0.5f),
            TensorData(DType.Float32, [], 0.25f),
            TensorData(DType.Float32, [], 0.5f),
            TensorData(DType.Int8, [], (sbyte)0)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeQuantizedConvShapeAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeQuantizedConvShapeAuditCheck>(inputs));
    }

    [Fact]
    public void TestQeeMaxRoiPoolShapeAudit()
    {
        TensorData[] inputs = [
            TensorDataWithDefaultVals(DType.Float32, [1L, 2L, 8L, 8L]),
            TensorData(DType.Float32, [2L, 5L],
                0f, 0f, 0f, 7f, 7f,
                0f, 1f, 1f, 6f, 6f)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeMaxRoiPoolShapeAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeMaxRoiPoolShapeAuditCheck>(inputs));
    }

    /// <summary>DeformConv — same roundtrip flags as the existing
    /// <c>TestQeeMiscOpsCoverage</c> DeformConv test (no ONNX/CS roundtrip).</summary>
    [Fact]
    public void TestQeeDeformConvShapeAudit()
    {
        TensorData[] inputs = [
            TensorDataWithDefaultVals(DType.Float32, [1L, 1L, 4L, 4L]),
            TensorData(DType.Float32, [1L, 1L, 2L, 2L], 1f, 0f, 0f, 1f),
            TensorDataWithDefaultVals(DType.Float32, [1L, 8L, 3L, 2L]),
            TensorData(DType.Float32, [1L], 0f)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeDeformConvShapeAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs,
            testOnnxRoundtrip: false, testCsRoundtrip: false));
        Assert.True(QeeSelfCheck<QeeDeformConvShapeAuditCheck>(inputs));
    }

    /// <summary>MaxUnpool — validates both the inverse-pool formula and the
    /// output_shape-input override. Runs under ORT too now that MAX_POOL/MAX_UNPOOL
    /// type the indices as int64 per spec (fixed in this audit batch).</summary>
    [Fact]
    public void TestQeeMaxUnpoolShapeAudit()
    {
        TensorData[] inputs = [
            TensorData(DType.Float32, [1L, 1L, 2L, 2L], 6f, 8f, 14f, 16f),
            TensorData(DType.Int64, [1L, 1L, 2L, 2L], 0L, 2L, 5L, 7L)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeMaxUnpoolShapeAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeMaxUnpoolShapeAuditCheck>(inputs));
    }
}

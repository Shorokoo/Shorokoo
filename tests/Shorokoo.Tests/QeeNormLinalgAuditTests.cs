using System.Reflection;
using Shorokoo.Runtime;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Inference;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage tests for the Phase 4 QEE-A4 audit batch (normalization, softmax,
/// linear-algebra &amp; quantization family, ONNX opset 21). Each module in
/// QeeNormLinalgAuditModules.cs is self-checking on VALUES (where QEE computes them)
/// and inferred SHAPES (via ShapeTensor): it compares every audited op's output against
/// spec-expected constants and returns a single Scalar&lt;bit&gt;. Each module is driven
/// twice:
/// <list type="bullet">
///   <item><see cref="AutoTest.AdvancedTestGraph{TModule}"/> — validates the expected
///         values/shapes against real ONNX Runtime execution (plus the ONNX/CS
///         roundtrips).</item>
///   <item><see cref="QeeSelfCheck{TModule}"/> — runs the same graph through
///         <see cref="QuickExecutionEngine"/> only and asserts the self-check bit is
///         concretely computed as true, so a wrong (or missing) QEE concrete value or
///         inferred shape for any audited op fails the test even though
///         AdvancedTestGraph's QEE pass only checks dtypes.</item>
/// </list>
/// </summary>
[Trait("Domain", "Inference")]
[Trait("Purpose", "Coverage")]
public class QeeNormLinalgAuditTests
{
    /// <summary>
    /// QEE-only strong self-check (same pattern as QeePoolConvAuditTests /
    /// QeeReductionShapeAuditTests): lowers the module exactly like AdvancedTestGraph,
    /// runs only the QuickExecutionEngine, and requires the module's single
    /// Scalar&lt;bit&gt; output to be concretely computed as true.
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

    [Fact]
    public void TestQeeNormalizationAudit()
    {
        TensorData[] inputs = [TensorData(DType.Float32, [2L, 2L], 1f, 2f, 3f, 4f)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeNormalizationAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeNormalizationAuditCheck>(inputs));
    }

    [Fact]
    public void TestQeeSoftmaxFamilyValueAudit()
    {
        TensorData[] inputs = [TensorData(DType.Float32, [2L, 3L], 1f, 2f, 3f, 3f, 2f, 1f)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeSoftmaxFamilyValueAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeSoftmaxFamilyValueAuditCheck>(inputs));
    }

    [Fact]
    public void TestQeeLossDropoutAudit()
    {
        TensorData[] inputs = [
            TensorData(DType.Float32, [2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f),
            TensorData(DType.Int64, [2L], 0L, 2L)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeLossDropoutAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeLossDropoutAuditCheck>(inputs));
    }

    [Fact]
    public void TestQeeMatMulGemmValueAudit()
    {
        TensorData[] inputs = [TensorData(DType.Float32, [2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeMatMulGemmValueAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeMatMulGemmValueAuditCheck>(inputs));
    }

    [Fact]
    public void TestQeeEinsumDetAudit()
    {
        TensorData[] inputs = [TensorData(DType.Float32, [2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeEinsumDetAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeEinsumDetAuditCheck>(inputs));
    }

    [Fact]
    public void TestQeeQuantizationValueAudit()
    {
        TensorData[] inputs = [TensorData(DType.Float32, [2L, 2L], 1.25f, -0.5f, 0.6f, 3.1f)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeQuantizationValueAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeQuantizationValueAuditCheck>(inputs));
    }
}

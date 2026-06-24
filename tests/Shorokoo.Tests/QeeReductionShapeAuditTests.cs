using System.Reflection;
using Shorokoo.Runtime;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Inference;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage tests for the Phase 4 QEE-A3 audit batch (reductions + shape/data-movement
/// family, ONNX opset 21). Each module in QeeReductionShapeAuditModules.cs is
/// self-checking on VALUES (and inferred shapes via ShapeTensor): it compares every
/// audited op's output against spec-expected constants and returns a single
/// Scalar&lt;bit&gt;. Each module is driven twice:
/// <list type="bullet">
///   <item><see cref="AutoTest.AdvancedTestGraph{TModule}"/> — validates the expected
///         values against real ONNX Runtime execution (plus the ONNX/CS roundtrips).</item>
///   <item><see cref="QeeSelfCheck{TModule}"/> — runs the same graph through
///         <see cref="QuickExecutionEngine"/> only and asserts the self-check bit is
///         concretely computed as true, so a wrong (or missing) QEE concrete value or
///         inferred shape for any audited op fails the test even though
///         AdvancedTestGraph's QEE pass only checks dtypes.</item>
/// </list>
/// </summary>
[Trait("Domain", "Inference")]
[Trait("Purpose", "Coverage")]
public class QeeReductionShapeAuditTests
{
    /// <summary>
    /// QEE-only strong self-check (same pattern as QeePoolConvAuditTests /
    /// QeeElementwiseAuditTests): lowers the module exactly like AdvancedTestGraph, runs
    /// only the QuickExecutionEngine, and requires the module's single Scalar&lt;bit&gt;
    /// output to be concretely computed as true.
    /// </summary>
    private static bool QeeSelfCheck<TModule>(TensorData[] runtimeInputs)
    {
        var prop = typeof(TModule).GetProperty("ComputationGraph", BindingFlags.Public | BindingFlags.Static)
            ?? throw new InvalidOperationException($"{typeof(TModule).FullName} has no public static ComputationGraph property");
        var moduleGraph = (FastComputationGraph)prop.GetValue(null)!;
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
    public void TestQeeReduceValueAudit()
    {
        TensorData[] inputs = [
            TensorData(DType.Float32, [2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f),
            TensorData(DType.Int64, [2L, 3L], 1L, -2L, 3L, 4L, 5L, -6L)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeReduceValueAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeReduceValueAuditCheck>(inputs));
    }

    [Fact]
    public void TestQeeArgCumSumValueAudit()
    {
        TensorData[] inputs = [TensorData(DType.Float32, [2L, 3L], 1f, 3f, 3f, 2f, 0f, 2f)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeArgCumSumValueAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeArgCumSumValueAuditCheck>(inputs));
    }

    [Fact]
    public void TestQeeReshapeFamilyValueAudit()
    {
        TensorData[] inputs = [TensorData(DType.Float32, [2L, 3L, 4L],
            Enumerable.Range(0, 24).Select(i => (object)(float)i).ToArray())];
        Assert.True(AutoTest.AdvancedTestGraph<QeeReshapeFamilyValueAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeReshapeFamilyValueAuditCheck>(inputs));
    }

    [Fact]
    public void TestQeeSliceGatherValueAudit()
    {
        TensorData[] inputs = [TensorData(DType.Float32, [3L, 4L],
            0f, 1f, 2f, 3f, 10f, 11f, 12f, 13f, 20f, 21f, 22f, 23f)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeSliceGatherValueAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeSliceGatherValueAuditCheck>(inputs));
    }

    [Fact]
    public void TestQeeScatterPadValueAudit()
    {
        TensorData[] inputs = [TensorData(DType.Float32, [2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeScatterPadValueAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeScatterPadValueAuditCheck>(inputs));
    }

    [Fact]
    public void TestQeeSplitConcatTileSpaceValueAudit()
    {
        TensorData[] inputs = [TensorData(DType.Float32, [7L], 1f, 2f, 3f, 4f, 5f, 6f, 7f)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeSplitConcatTileSpaceValueAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeSplitConcatTileSpaceValueAuditCheck>(inputs));
    }

    [Fact]
    public void TestQeeOneHotTriluNonZeroValueAudit()
    {
        TensorData[] inputs = [TensorData(DType.Int64, [4L], 1L, 3L, -2L, 5L)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeOneHotTriluNonZeroValueAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeOneHotTriluNonZeroValueAuditCheck>(inputs));
    }
}

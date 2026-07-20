using System.Reflection;
using Shorokoo.Core.Inference;
using Shorokoo.Core.Graph;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// Phase-4 follow-up coverage for TopK and Unique, which the family audit batches
/// missed (surfaced while generating Documentation/operator-support.md). Value cases run under
/// both ORT (AdvancedTestGraph) and the QuickExecutionEngine (QeeSelfCheck); the
/// Unique axis form is data-dependent, so its shape case is ORT-validated only.
/// </summary>
[Trait("Domain", "Inference")]
[Trait("Purpose", "Coverage")]
public class QeeTopKUniqueAuditTests
{
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
    public void TestQeeTopKUniqueValueAudit()
    {
        TensorData[] inputs = [TensorData(DType.Float32, [2L, 4L], 3f, 1f, 4f, 1f, 5f, 9f, 2f, 6f)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeTopKUniqueValueAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeTopKUniqueValueAuditCheck>(inputs));
    }

    [Fact]
    public void TestQeeTopKUniqueShapeAudit()
    {
        TensorData[] inputs = [TensorData(DType.Float32, [3L, 2L], 1f, 2f, 1f, 2f, 3f, 4f)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeTopKUniqueShapeAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
    }
}

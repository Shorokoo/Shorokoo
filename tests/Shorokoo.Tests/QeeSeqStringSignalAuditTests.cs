using System.Reflection;
using Shorokoo.Runtime;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Inference;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage tests for the Phase 4 QEE-A6 audit batch (sequence, optional, string, signal
/// &amp; control-flow family, ONNX opset 21). Each module in
/// QeeSeqStringSignalAuditModules.cs is self-checking on VALUES (where QEE computes them)
/// and inferred SHAPES (via ShapeTensor) against spec-expected constants and returns a
/// Scalar&lt;bit&gt;. ORT-runnable modules are driven twice:
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
/// Modules built on Shorokoo-internal op codes or @string runtime inputs (no ORT-comparable
/// data path) go through <see cref="QeeStrictCheck{TModule}"/> only — the QeeOnly pattern
/// from QeeOpsTests.cs hardened to require every scalar-bool output to be concretely TRUE
/// (other outputs only need a valid dtype, for the data-dependent/unknown-shape cases).
/// </summary>
[Trait("Domain", "Inference")]
[Trait("Purpose", "Coverage")]
public class QeeSeqStringSignalAuditTests
{
    /// <summary>
    /// QEE-only strong self-check (same pattern as QeeNormLinalgAuditTests): lowers the
    /// module exactly like AdvancedTestGraph, runs only the QuickExecutionEngine, and
    /// requires the module's single Scalar&lt;bit&gt; output to be concretely true.
    /// </summary>
    private static bool QeeSelfCheck<TModule>(TensorData[] runtimeInputs)
    {
        var store = RunQee<TModule>(runtimeInputs, out var outputs);
        foreach (var outKey in outputs)
        {
            if (!store.TryGetValue(outKey, out var rt)) return false;
            if (rt is not RuntimeTensor { BoolData: { Length: 1 } bits } plain
                || plain.DType != DType.Bool || !bits[0])
                return false;
        }
        return true;
    }

    /// <summary>
    /// QeeOnly pattern (QeeOpsTests.cs) hardened for self-checking modules whose graphs
    /// can't run under ORT: every scalar-bool output must be concretely computed TRUE;
    /// every other output (data-dependent/unknown-shape tensors returned raw) just needs a
    /// valid dtype.
    /// </summary>
    private static bool QeeStrictCheck<TModule>(TensorData[] runtimeInputs)
    {
        var store = RunQee<TModule>(runtimeInputs, out var outputs);
        foreach (var outKey in outputs)
        {
            if (!store.TryGetValue(outKey, out var rt) || rt.DType == DType.Invalid) return false;
            if (rt is RuntimeTensor bitOut && bitOut.DType == DType.Bool
                && (bitOut.BoolData is not { Length: 1 } bits || !bits[0]))
                return false;
        }
        return true;
    }

    private static Dictionary<FastTensorKey, IRuntimeTensor> RunQee<TModule>(
        TensorData[] runtimeInputs, out IReadOnlyList<FastTensorKey> outputs)
    {
        var prop = typeof(TModule).GetProperty("ComputationGraph", BindingFlags.Public | BindingFlags.Static)
            ?? throw new InvalidOperationException($"{typeof(TModule).FullName} has no public static ComputationGraph property");
        var moduleGraph = ((ComputationGraph)prop.GetValue(null)!).ToInternal();
        var concreteArch = moduleGraph.ToConcreteArchitecture(moduleGraph.FromOrderedInputs([.. runtimeInputs]));
        var concreteModel = concreteArch.ToConcreteModel();
        var qee = new QuickExecutionEngine();
        var store = runtimeInputs.Length == 0 ? qee.Run(concreteModel) : qee.Run(concreteModel, runtimeInputs);
        outputs = concreteModel.Outputs;
        return store;
    }

    private static readonly TensorData FloatMat2x3 = TensorData(
        DType.Float32, [2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f);

    [Fact]
    public void TestQeeSequenceCoreAudit()
    {
        TensorData[] inputs = [FloatMat2x3, TensorData(DType.Int64, [], 0L)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeSequenceCoreAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeSequenceCoreAuditCheck>(inputs));
    }

    [Fact]
    public void TestQeeSplitToSeqConcatAudit()
    {
        TensorData[] inputs = [FloatMat2x3, TensorData(DType.Int64, [], 0L)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeSplitToSeqConcatAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeSplitToSeqConcatAuditCheck>(inputs));
        // Spec: keepdims is IGNORED when split is given. ORT deviates (applies it and
        // crashes on non-1 chunk extents), so this conformance check is QEE-only.
        Assert.True(QeeSelfCheck<QeeSplitKeepdimsInteractionAuditCheck>(inputs));
    }

    [Fact]
    public void TestQeeReverseSequenceAudit()
    {
        TensorData[] inputs = [TensorData(DType.Float32, [4L, 4L],
            1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeReverseSequenceAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeReverseSequenceAuditCheck>(inputs));
    }

    [Fact]
    public void TestQeeOptionalAudit()
    {
        TensorData[] inputs = [TensorData(DType.Float32, [3L], 1f, 2f, 3f)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeOptionalAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeOptionalAuditCheck>(inputs));
    }

    [Fact]
    public void TestQeeWindowValueAudit()
    {
        TensorData[] inputs = [TensorData(DType.Int64, [], 8L)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeWindowValueAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeWindowValueAuditCheck>(inputs));
    }

    [Fact]
    public void TestQeeDftStftMelAudit()
    {
        TensorData[] inputs = [
            TensorData(DType.Float32, [1L, 4L, 1L], 1f, 2f, 3f, 4f),
            TensorData(DType.Float32, [1L, 16L, 1L],
                0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f),
            TensorData(DType.Int64, [], 4L),
            TensorData(DType.Float32, [4L], 1f, 1f, 1f, 1f)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeDftStftMelAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeDftStftMelAuditCheck>(inputs));
    }

    [Fact]
    public void TestQeeTfIdfShapeAudit()
    {
        TensorData[] inputs = [TensorData(DType.Int64, [4L], 1L, 2L, 3L, 4L)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeTfIdfShapeAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeTfIdfShapeAuditCheck>(inputs));
    }

    /// <summary>@string runtime tensors carry no data into QEE and have no flat byte buffer
    /// for AdvancedTestGraph's result comparator (see TestQeeStringOpsCoverage in
    /// QeeOpsTests.cs), so this is QEE-only: shape/dtype checks via the strict bit.</summary>
    [Fact]
    public void TestQeeStringOpsAudit() =>
        Assert.True(QeeStrictCheck<QeeStringOpsAuditCheck>([
            TensorData([2L], "Hello World", "the quick fox"),
            TensorData([2L], "A", "B")]));

    /// <summary>Shorokoo-internal lowering ops (no ONNX registration → no ORT execution),
    /// audited for type/shape propagation + pass-through values via the strict bit.</summary>
    [Fact]
    public void TestQeeInternalControlFlowAudit() =>
        Assert.True(QeeStrictCheck<QeeInternalControlFlowAuditCheck>([
            FloatMat2x3,
            TensorData(DType.Float32, [1L, 1L, 4L, 4L],
                1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f,
                9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f),
            TensorData(DType.Float32, [1L, 1L, 2L, 2L], 1f, 0f, 0f, 1f),
            TensorData(DType.Float32, [1L], 0f)]));
}

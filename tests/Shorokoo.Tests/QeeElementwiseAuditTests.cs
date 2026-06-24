using System.Reflection;
using Shorokoo.Runtime;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Inference;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage tests for the Phase 4 QEE-A2 audit batch (elementwise, comparison, logical
/// &amp; bitwise family, ONNX opset 21). Each module in QeeElementwiseAuditModules.cs is
/// self-checking on VALUES: it compares every audited op's output against spec-expected
/// constants and returns a single Scalar&lt;bit&gt;. Each module is driven twice:
/// <list type="bullet">
///   <item><see cref="AutoTest.AdvancedTestGraph{TModule}"/> — validates the expected
///         values against real ONNX Runtime execution (plus the ONNX/CS roundtrips).</item>
///   <item><see cref="QeeSelfCheck{TModule}"/> — runs the same graph through
///         <see cref="QuickExecutionEngine"/> only and asserts the self-check bit is
///         concretely computed as true, so a wrong (or missing) QEE concrete value for
///         any audited op fails the test even though AdvancedTestGraph's QEE pass only
///         checks dtypes.</item>
/// </list>
/// </summary>
[Trait("Domain", "Inference")]
[Trait("Purpose", "Coverage")]
public class QeeElementwiseAuditTests
{
    /// <summary>
    /// QEE-only strong self-check (same pattern as QeePoolConvAuditTests): lowers the
    /// module exactly like AdvancedTestGraph, runs only the QuickExecutionEngine, and
    /// requires the module's single Scalar&lt;bit&gt; output to be concretely computed
    /// as true. Because the modules derive the bit from the audited ops' VALUES, the
    /// bit is false (or never computed, which also fails) whenever QEE computes a wrong
    /// value or stops propagating data along the chain.
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
    public void TestQeeTrigExpLogValueAudit()
    {
        TensorData[] inputs = [
            TensorData(DType.Float32, [3L], 0.5f, -0.25f, 0.75f),
            TensorData(DType.Float32, [3L], 1f, 2f, 4f)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeTrigExpLogValueAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeTrigExpLogValueAuditCheck>(inputs));
    }

    [Fact]
    public void TestQeeUnaryRoundingValueAudit()
    {
        TensorData[] inputs = [TensorData(DType.Float32, [5L], -1.5f, -0.5f, 0.5f, 1.5f, 2.5f)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeUnaryRoundingValueAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeUnaryRoundingValueAuditCheck>(inputs));
    }

    [Fact]
    public void TestQeeActivationValueAudit()
    {
        TensorData[] inputs = [
            TensorData(DType.Float32, [5L], -2f, -0.5f, 0f, 0.5f, 2f),
            TensorData(DType.Float32, [4L], -2.7f, -1f, 0.5f, 2.7f)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeActivationValueAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeActivationValueAuditCheck>(inputs));
    }

    [Fact]
    public void TestQeeBinaryArithValueAudit()
    {
        TensorData[] inputs = [
            TensorData(DType.Float32, [3L], 7.5f, -5.5f, 9.25f),
            TensorData(DType.Float32, [3L], 2f, 3f, -4f),
            TensorData(DType.Int64, [3L], 7L, -7L, 9L),
            TensorData(DType.Int64, [3L], 2L, 2L, -4L)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeBinaryArithValueAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeBinaryArithValueAuditCheck>(inputs));
    }

    [Fact]
    public void TestQeeCompareLogicValueAudit()
    {
        TensorData[] inputs = [
            TensorData(DType.Float32, [3L], 1f, 2f, 3f),
            TensorData(DType.Float32, [3L], 2f, 2f, 2f),
            TensorData(DType.Bool, [4L], true, false, true, false),
            TensorData(DType.Bool, [4L], true, true, false, false)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeCompareLogicValueAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeCompareLogicValueAuditCheck>(inputs));
    }

    [Fact]
    public void TestQeeBitwiseValueAudit()
    {
        TensorData[] inputs = [
            TensorData(DType.Int64, [3L], 12L, 10L, 15L),
            TensorData(DType.Int64, [3L], 10L, 5L, 3L)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeBitwiseValueAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeBitwiseValueAuditCheck>(inputs));
    }

    [Fact]
    public void TestQeeMiscElementwiseValueAudit()
    {
        TensorData[] inputs = [
            TensorData(DType.Float32, [4L], 1f, -1f, 0f, 2f),
            TensorData(DType.Float32, [4L], 0f, 0f, 0f, 1f),
            TensorData(DType.Bool, [4L], true, false, true, false)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeMiscElementwiseValueAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeMiscElementwiseValueAuditCheck>(inputs));
    }

    /// <summary>Where with bool then/else values — ORT's CPU EP has no bool-typed Where
    /// kernel ([ErrorCode:NotImplemented] Where(16)), so this only runs the QEE pass.</summary>
    [Fact]
    public void TestQeeWhereBoolValueAudit() =>
        Assert.True(QeeSelfCheck<QeeWhereBoolValueAuditCheck>(
            [TensorData(DType.Bool, [4L], true, false, true, false)]));

    /// <summary>Full reverse Slice (negative step, ends=INT_MIN). Pinned in AD-B3: the
    /// QEE kernel clamped the negative-step exclusive ends to 0 instead of −1, dropping
    /// the first element (output [2] instead of [3]); ORT computed the spec value. Fixed
    /// in the same batch (step-sign-aware clamping in SliceOp), so both passes agree now.</summary>
    [Fact]
    public void TestQeeSliceReverseValueAudit()
    {
        TensorData[] inputs = [TensorData(DType.Float32, [3L], 1f, 2f, 3f)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeSliceReverseValueAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeSliceReverseValueAuditCheck>(inputs));
    }
}

/// <summary>
/// Bug pins for wrapper-API value bugs in Scalar.cs / Vector.cs, discovered during the
/// Phase 7 documentation sweep. Each test runs a self-checking module (see
/// QeeElementwiseAuditModules.cs) through <see cref="QuickExecutionEngine"/> and asserts
/// the value the public wrapper SHOULD produce. These stay failing until the product
/// bugs are fixed:
/// <list type="bullet">
///   <item>Scalar&lt;T&gt;.operator &lt;&lt;(Scalar&lt;T&gt;, PrimitiveParam) delegates to
///         &gt;&gt;, so a left shift by a primitive constant right-shifts.</item>
///   <item>Scalar&lt;T&gt;.Min/Max(params Scalar&lt;T&gt;[] others) ignore <c>others</c>.</item>
///   <item>Vector&lt;T&gt;.Min/Max(params Tensor&lt;T&gt;[] others) ignore <c>others</c>.</item>
/// </list>
/// </summary>
[Trait("Domain", "Core")]
public class ScalarVectorWrapperBugPinTests
{
    /// <summary>Same QEE-only strong self-check as <see cref="QeeElementwiseAuditTests"/>:
    /// lowers the module, runs QuickExecutionEngine, and requires the single
    /// Scalar&lt;bit&gt; output to be concretely computed as true.</summary>
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

    /// <summary>Pins the Scalar left-shift-by-primitive bug (computes a right shift).</summary>
    [Fact]
    public void TestScalarShiftLeftByPrimitiveShiftsLeft() =>
        Assert.True(QeeSelfCheck<ScalarShiftLeftPrimitiveBugPinCheck>(
            [TensorData(DType.Int64, [], 4L)]));

    /// <summary>Pins the Scalar Min/Max bug (the params 'others' argument is ignored).</summary>
    [Fact]
    public void TestScalarMinMaxForwardOthers() =>
        Assert.True(QeeSelfCheck<ScalarMinMaxOthersBugPinCheck>(
            [TensorData(DType.Float32, [], 5f), TensorData(DType.Float32, [], 2f)]));

    /// <summary>Pins the Vector Min/Max bug (the params 'others' argument is ignored).</summary>
    [Fact]
    public void TestVectorMinMaxForwardOthers() =>
        Assert.True(QeeSelfCheck<VectorMinMaxOthersBugPinCheck>(
            [TensorData(DType.Float32, [2L], 1f, 5f), TensorData(DType.Float32, [2L], 3f, 2f)]));
}


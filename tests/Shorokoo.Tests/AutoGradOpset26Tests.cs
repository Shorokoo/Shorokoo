using System.Reflection;
using Shorokoo.Core.Inference;
using Shorokoo.Core.Graph;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// Gradient coverage for the opset 22-26 op batch (see <c>AutoDiffs.Batch31.cs</c>).
/// The decomposable ops (Swish @24, RMSNormalization @23) lower inline to opset-21
/// primitives, so their gradients flow through those primitives and are exercised by
/// FD / closed-form self-checking modules from <c>Modules/AutoGradOpset26Modules.cs</c>
/// (following the one-liner pattern of <see cref="AutoGradStructuralTests"/>). The Swish
/// check runs QEE-only because ORT 1.26 registers no Swish kernel — but Swish lowers to
/// Mul/Sigmoid, so the loaded graph contains no Swish node; it is QEE-only here only to
/// match the audit-module style.
/// The ops with no opset-21 equivalent (Attention, RotaryEmbedding, TensorScatter,
/// BitCast, CumProd) throw <c>NotImplementedException</c> from their <see cref="OnnxOp"/>
/// entry point, before any graph — including a gradient path — can be built, so those
/// tests assert the authoring throw directly. Their op definitions and AD003 autodiff
/// guards are retained for when a runtime supports them; a proper non-fused Attention
/// gradient is deferred to the core project.
/// </summary>
[Trait("Domain", "AutoDiff")]
[Trait("Purpose", "Coverage")]
public class AutoGradOpset26Tests
{
    /// <summary>
    /// QEE-only strong self-check (same lowering as AdvancedTestGraph, QEE-only
    /// execution; pattern from QeeOpset26AuditTests) for graphs ORT 1.26 cannot
    /// load (Swish has no registered kernel).
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

    /// <summary>Swish gradient (s + a·x·s·(1−s)), default + explicit alpha, FD
    /// self-check. QEE-only: ORT 1.26 has no Swish kernel, so the forward loss
    /// graph itself cannot run under ORT.</summary>
    [Fact]
    public void TestAutoGradSwishCoverage()
    {
        Assert.True(QeeSelfCheck<AutoGradSwishCheck>(
            [TensorData(DType.Float32, [5L], -2f, -1f, 0.5f, 1f, 2f)]));
    }

    /// <summary>CumProd cannot be authored under the single-opset-21 export (no opset-21
    /// equivalent), so the entry point throws before any graph — including a gradient path —
    /// can be built. The CUM_PROD op definition and its autodiff rule are retained for when a
    /// runtime supports the op.</summary>
    [Fact]
    public void TestAutoGradCumProdNotImplemented()
    {
        var x = Globals.InputTensor<float32>(defaultName: "x", rank: 1);
        Assert.Throws<System.NotImplementedException>(() => OnnxOp.CumProd(x, Globals.Scalar(0L)));
    }

    /// <summary>RMSNormalization gradients: dX (default axis −1, explicit epsilon)
    /// and dScale (explicit positive axis, trainable scale param), both FD
    /// self-checks.</summary>
    [Fact]
    public void TestAutoGradRmsNormCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradRmsNormInputCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [2L, 3L], 0.5f, -1f, 2f, 1.5f, -0.5f, 1f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradRmsNormScaleCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [2L, 3L], 0.5f, -1f, 2f, 1.5f, -0.5f, 1f)]));
    }

    /// <summary>BitCast cannot be authored under the single-opset-21 export (a bit
    /// reinterpretation has no opset-21 primitive), so the entry point throws before any graph —
    /// including a gradient path — can be built. The BIT_CAST op definition and its (zero-class)
    /// autodiff rule are retained for when a runtime supports the op.</summary>
    [Fact]
    public void TestAutoGradBitCastNotImplemented()
    {
        var x = Globals.InputTensor<float32>(defaultName: "x", rank: 1);
        Assert.Throws<System.NotImplementedException>(() => OnnxOp.BitCast(x, DType.Int32));
    }

    /// <summary>Attention cannot be authored under the single-opset-21 export (no opset-21
    /// equivalent), so the entry point throws before any graph — including a gradient path —
    /// can be built. The ATTENTION op definition and its AD003 autodiff guard are retained; a
    /// proper non-fused gradient is deferred to the core project.</summary>
    [Fact]
    public void TestAutoGradAttentionNotImplemented()
    {
        var q = Globals.InputTensor<float32>(defaultName: "q", rank: 4);
        var k = Globals.InputTensor<float32>(defaultName: "k", rank: 4);
        var v = Globals.InputTensor<float32>(defaultName: "v", rank: 4);
        Assert.Throws<System.NotImplementedException>(() => OnnxOp.Attention(q, k, v));
    }

    /// <summary>RotaryEmbedding cannot be authored under the single-opset-21 export, so the entry
    /// point throws before any graph — including a gradient path — can be built. The
    /// ROTARY_EMBEDDING op definition and its AD003 autodiff guard are retained.</summary>
    [Fact]
    public void TestAutoGradRotaryEmbeddingNotImplemented()
    {
        var x = Globals.InputTensor<float32>(defaultName: "x", rank: 4);
        var cos = Globals.InputTensor<float32>(defaultName: "cos", rank: 2);
        var sin = Globals.InputTensor<float32>(defaultName: "sin", rank: 2);
        Assert.Throws<System.NotImplementedException>(() => OnnxOp.RotaryEmbedding(x, cos, sin));
    }

    /// <summary>TensorScatter cannot be authored under the single-opset-21 export, so the entry
    /// point throws before any graph — including a gradient path — can be built. The
    /// TENSOR_SCATTER op definition and its AD003 autodiff guard are retained.</summary>
    [Fact]
    public void TestAutoGradTensorScatterNotImplemented()
    {
        var p = Globals.InputTensor<float32>(defaultName: "p", rank: 4);
        var update = Globals.InputTensor<float32>(defaultName: "u", rank: 4);
        Assert.Throws<System.NotImplementedException>(() => OnnxOp.TensorScatter(p, update));
    }
}

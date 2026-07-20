using System.Reflection;
using Shorokoo.Core.Inference;
using Shorokoo.Core.Graph;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage for the opset 22-26 op batch under Shorokoo's single-opset-21 export.
/// The decomposable ops (Swish @24, RMSNormalization @23) are lowered inline to
/// opset-21 primitives by their <see cref="OnnxOp"/> entry points, so their value
/// audits run normally (Swish QEE-only because ORT 1.26 registers no Swish kernel;
/// RMSNormalization under both ORT and QEE). The ops with no opset-21 equivalent —
/// Attention / AttentionWithKVCache / RotaryEmbedding (opset 23), TensorScatter
/// (opset 24), BitCast / CumProd (opset 26) — cannot be emitted into an opset-21
/// model, so their entry points throw <see cref="System.NotImplementedException"/>
/// at authoring time (a faithful lowering / proper non-fused autodiff is deferred to
/// the core project). Their op definitions and QEE
/// kernels are retained for when a runtime supports them, so these audits assert the
/// throw rather than a value.
/// </summary>
[Trait("Domain", "Inference")]
[Trait("Purpose", "Coverage")]
public class QeeOpset26AuditTests
{
    /// <summary>
    /// QEE-only strong self-check (same pattern as QeeTopKUniqueAuditTests): lowers
    /// the module exactly like AdvancedTestGraph, runs only the QuickExecutionEngine,
    /// and requires the module's single Scalar&lt;bit&gt; output to be concretely true.
    /// </summary>
    private static bool QeeSelfCheck<TModule>(TensorData[] runtimeInputs)
    {
        var prop = typeof(TModule).GetProperty("ComputationGraph", BindingFlags.Public | BindingFlags.Static)
            ?? throw new InvalidOperationException($"{typeof(TModule).FullName} has no public static ComputationGraph property");
        var moduleGraph = (InternalComputationGraph)prop.GetValue(null)!;
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

    /// <summary>ORT 1.26 registers no Swish kernel on any execution provider
    /// (checked against rel-1.26.0 OperatorKernels.md), so Swish graphs are
    /// QEE-executable only — the value audit runs through QeeSelfCheck.</summary>
    [Fact]
    public void TestQeeSwishValueAudit()
    {
        TensorData[] inputs = [TensorData(DType.Float32, [5L], -2f, -1f, 0f, 1f, 2f)];
        Assert.True(QeeSelfCheck<QeeSwishValueAuditCheck>(inputs));
    }

    [Fact]
    public void TestQeeCumProdNotImplemented()
    {
        // CumProd (opset 26) has no opset-21 equivalent (a faithful general decomposition needs a
        // Scan multiply body), so Shorokoo's single-opset-21 export throws rather than emit it. The
        // CUM_PROD op definition and QEE kernel are retained for when a runtime supports it.
        var x = Globals.InputTensor<float32>(defaultName: "x", rank: 1);
        Assert.Throws<System.NotImplementedException>(() => OnnxOp.CumProd(x, Globals.Scalar(0L)));
    }

    [Fact]
    public void TestQeeRmsNormValueAudit()
    {
        TensorData[] inputs = [TensorData(DType.Float32, [4L], 1f, 2f, 3f, 4f)];
        Assert.True(AutoTest.AdvancedTestGraph<QeeRmsNormValueAuditCheck>(
            hyperparamInputs: [], runtimeInputs: inputs));
        Assert.True(QeeSelfCheck<QeeRmsNormValueAuditCheck>(inputs));
    }

    [Fact]
    public void TestQeeBitCastNotImplemented()
    {
        // BitCast (opset 26) reinterprets bit patterns — no opset-21 primitive does that — so
        // Shorokoo's single-opset-21 export throws rather than emit it. The BIT_CAST op definition
        // and QEE kernel are retained for when a runtime supports it.
        var x = Globals.InputTensor<float32>(defaultName: "x", rank: 1);
        Assert.Throws<System.NotImplementedException>(() => OnnxOp.BitCast(x, DType.Int32));
    }

    /// <summary>Attention (opset 23) has no opset-21 equivalent, so the single-opset-21
    /// export throws at authoring rather than emit it. The ATTENTION op definition and
    /// QEE kernel are retained for when a runtime supports it.</summary>
    [Fact]
    public void TestQeeAttentionNotImplemented()
    {
        var q = Globals.InputTensor<float32>(defaultName: "q", rank: 4);
        var k = Globals.InputTensor<float32>(defaultName: "k", rank: 4);
        var v = Globals.InputTensor<float32>(defaultName: "v", rank: 4);
        Assert.Throws<System.NotImplementedException>(() => OnnxOp.Attention(q, k, v));
    }

    /// <summary>The KV-cache (multi-output) Attention form throws for the same reason as
    /// <see cref="TestQeeAttentionNotImplemented"/>.</summary>
    [Fact]
    public void TestQeeAttentionKvCacheNotImplemented()
    {
        var q = Globals.InputTensor<float32>(defaultName: "q", rank: 4);
        var k = Globals.InputTensor<float32>(defaultName: "k", rank: 4);
        var v = Globals.InputTensor<float32>(defaultName: "v", rank: 4);
        Assert.Throws<System.NotImplementedException>(() => OnnxOp.AttentionWithKVCache(q, k, v));
    }

    /// <summary>RotaryEmbedding (opset 23) has no opset-21 equivalent, so the single-opset-21
    /// export throws at authoring. The ROTARY_EMBEDDING op definition and QEE kernel are
    /// retained for when a runtime supports it.</summary>
    [Fact]
    public void TestQeeRotaryEmbeddingNotImplemented()
    {
        var x = Globals.InputTensor<float32>(defaultName: "x", rank: 4);
        var cos = Globals.InputTensor<float32>(defaultName: "cos", rank: 2);
        var sin = Globals.InputTensor<float32>(defaultName: "sin", rank: 2);
        Assert.Throws<System.NotImplementedException>(() => OnnxOp.RotaryEmbedding(x, cos, sin));
    }

    /// <summary>TensorScatter (opset 24) has no opset-21 equivalent, so the single-opset-21
    /// export throws at authoring. The TENSOR_SCATTER op definition and (shape-only) QEE
    /// kernel are retained for when a runtime supports it.</summary>
    [Fact]
    public void TestQeeTensorScatterNotImplemented()
    {
        var p = Globals.InputTensor<float32>(defaultName: "p", rank: 4);
        var update = Globals.InputTensor<float32>(defaultName: "u", rank: 4);
        Assert.Throws<System.NotImplementedException>(() => OnnxOp.TensorScatter(p, update));
    }
}

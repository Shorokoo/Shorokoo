using Shorokoo.Core.Inference;
using Shorokoo.Core.Graph;
using static Shorokoo.Tests.Utils.QeeAudit;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage for the opset 22-26 op batch under Shorokoo's single-opset-21 export. The
/// decomposable ops (Swish @24, RMSNormalization @23) are lowered inline to opset-21
/// primitives by their <see cref="OnnxOp"/> entry points, so their value audits run
/// normally (Swish QEE-only because ORT 1.26 registers no Swish kernel). The ops with no
/// opset-21 equivalent — Attention / AttentionWithKVCache / RotaryEmbedding (opset 23),
/// TensorScatter (opset 24), BitCast / CumProd (opset 26) — cannot be emitted into an
/// opset-21 model, so their entry points throw at authoring time; their op definitions and
/// QEE kernels are retained for when a runtime supports them.
/// </summary>
[Trait("Domain", "Inference")]
[Trait("Purpose", "Coverage")]
public class QeeOpset26AuditTests
{
    [Fact]
    public void TestQeeSwishAndRmsNormValueAudits()
    {
        Assert.True(QeeAudit.QeeOnly<QeeSwishValueAuditCheck>(F32([5L], -2f, -1f, 0f, 1f, 2f)));
        Assert.True(QeeAudit.Check<QeeRmsNormValueAuditCheck>(F32([4L], 1f, 2f, 3f, 4f)));
    }

    [Fact]
    public void TestOpsWithoutOpset21EquivalentThrowAtAuthoring()
    {
        var x1 = Globals.InputTensor<float32>(defaultName: "x", rank: 1);
        var q = Globals.InputTensor<float32>(defaultName: "q", rank: 4);
        var k = Globals.InputTensor<float32>(defaultName: "k", rank: 4);
        var v = Globals.InputTensor<float32>(defaultName: "v", rank: 4);
        var cos = Globals.InputTensor<float32>(defaultName: "cos", rank: 2);
        var sin = Globals.InputTensor<float32>(defaultName: "sin", rank: 2);
        var update = Globals.InputTensor<float32>(defaultName: "u", rank: 4);

        Assert.Throws<System.NotImplementedException>(() => OnnxOp.CumProd(x1, Globals.Scalar(0L)));
        Assert.Throws<System.NotImplementedException>(() => OnnxOp.BitCast(x1, DType.Int32));
        Assert.Throws<System.NotImplementedException>(() => OnnxOp.Attention(q, k, v));
        Assert.Throws<System.NotImplementedException>(() => OnnxOp.AttentionWithKVCache(q, k, v));
        Assert.Throws<System.NotImplementedException>(() => OnnxOp.RotaryEmbedding(q, cos, sin));
        Assert.Throws<System.NotImplementedException>(() => OnnxOp.TensorScatter(q, update));
    }
}

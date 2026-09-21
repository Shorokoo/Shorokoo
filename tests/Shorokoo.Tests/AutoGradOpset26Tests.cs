using Shorokoo.Core.Inference;
using static Shorokoo.Tests.Utils.QeeAudit;

namespace Shorokoo.Tests;

/// <summary>
/// Gradient coverage for the decomposable ops of the opset 22-26 batch (see
/// <c>AutoDiffs.Batch31.cs</c>): Swish @24 and RMSNormalization @23 lower inline to
/// opset-21 primitives, so their gradients flow through those primitives and are exercised
/// by the finite-difference self-checking modules in <c>Modules/AutoGradOpset26Modules.cs</c>.
/// The Swish check is QEE-only only to match the audit-module style — its lowered graph
/// carries no Swish node and loads anywhere. TensorScatter @24 has no gradient rule either,
/// and is not meant to: the autodiff pass lowers it before it walks backwards, so what is
/// differentiated is the mask and gather of its registered lowering. The batch's
/// non-decomposable ops (Attention, RotaryEmbedding, BitCast, CumProd) throw
/// from their <c>OnnxOp</c> entry point before any graph — gradient path included — exists,
/// so no autodiff code runs for them; that authoring throw is pinned in
/// <c>QeeOpset26AuditTests.TestOpsWithoutOpset21EquivalentThrowAtAuthoring</c>.
/// </summary>
[Trait("Domain", "AutoDiff")]
[Trait("Purpose", "Coverage")]
public class AutoGradOpset26Tests
{
    [Fact]
    public void TestAutoGradSwishAndRmsNormFiniteDifferenceChecks()
    {
        Assert.True(QeeAudit.QeeOnly<AutoGradSwishCheck>(F32([5L], -2f, -1f, 0.5f, 1f, 2f)));
        Assert.True(QeeAudit.OrtOnly<AutoGradRmsNormInputCheck>(
            F32([2L, 3L], 0.5f, -1f, 2f, 1.5f, -0.5f, 1f)));
        Assert.True(QeeAudit.OrtOnly<AutoGradRmsNormScaleCheck>(
            F32([2L, 3L], 0.5f, -1f, 2f, 1.5f, -0.5f, 1f)));
    }

    [Fact]
    public void TestAutoGradTensorScatterLoweredGradientChecks()
    {
        var past = F32([2L, 3L], 1f, 2f, 3f, 4f, 5f, 6f);
        Assert.True(QeeAudit.Check<AutoGradTensorScatterLinearGradientCheck>(
            past, F32([2L, 1L], 9f, 9f),
            F32([2L, 3L], 0f, 2f, 3f, 4f, 5f, 0f), F32([2L, 1L], 1f, 6f)));
        Assert.True(QeeAudit.Check<AutoGradTensorScatterCircularGradientCheck>(
            past, F32([2L, 2L], 9f, 9f, 9f, 9f),
            F32([2L, 3L], 0f, 2f, 0f, 4f, 0f, 0f), F32([2L, 2L], 3f, 1f, 5f, 6f)));
    }
}

using System.IO;
using System.Linq;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Core.Inference;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Onnx;
using static Shorokoo.Tests.Utils.QeeAudit;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage for the opset 22-26 op batch under Shorokoo's single-opset-21 export. The
/// decomposable ops (Swish @24, RMSNormalization @23) are lowered inline to opset-21
/// primitives by their <see cref="OnnxOp"/> entry points, so their value audits run
/// normally (the Swish audit is QEE-only only to match the audit-module style — its lowered
/// graph carries no Swish node and loads anywhere). TensorScatter (@24) is decomposed by
/// its registered lowering instead, which is a different arrangement: the node is built,
/// run and differentiated as itself, and only the exported file carries the decomposition.
/// The ops with no opset-21 equivalent — Attention / AttentionWithKVCache /
/// RotaryEmbedding (opset 23), BitCast / CumProd (opset 26) — cannot be emitted into an
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

        Assert.Throws<System.NotImplementedException>(() => OnnxOp.CumProd(x1, Globals.Scalar(0L)));
        Assert.Throws<System.NotImplementedException>(() => OnnxOp.BitCast(x1, DType.Int32));
        Assert.Throws<System.NotImplementedException>(() => OnnxOp.Attention(q, k, v));
        Assert.Throws<System.NotImplementedException>(() => OnnxOp.AttentionWithKVCache(q, k, v));
        Assert.Throws<System.NotImplementedException>(() => OnnxOp.RotaryEmbedding(q, cos, sin));
    }

    [Fact]
    public void TestQeeTensorScatterValueAudits()
    {
        var q = Globals.InputTensor<float32>(defaultName: "q", rank: 4);
        Assert.Throws<System.ArgumentOutOfRangeException>(() => OnnxOp.TensorScatter(q, q, axis: 0L));

        Assert.True(QeeAudit.Check<QeeTensorScatterValueAuditCheck>(
            F32([2L, 3L, 2L], 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f),
            F32([2L, 4L], 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f)));
        Assert.True(QeeAudit.Check<QeeTensorScatterSpecExampleAuditCheck>(
            F32([2L, 1L, 4L, 5L],
                1f, 2f, 3f, 4f, 5f, 5f, 6f, 7f, 8f, 9f, 8f, 7f, 6f, 5f, 4f, 4f, 3f, 2f, 1f, 0f,
                1f, 2f, 3f, 4f, 5f, 5f, 6f, 7f, 8f, 9f, 8f, 7f, 6f, 5f, 4f, 4f, 3f, 2f, 1f, 0f)));
    }

    // TensorScatter is on the export list and no other: the graph the caller built keeps its
    // node, and only the written file — which must stay at opset 21, the one opset Shorokoo
    // emits — carries the decomposition.
    [Fact]
    public void TestTensorScatterIsDecomposedOnlyOnTheWayOut()
    {
        var dir = Directory.CreateDirectory(
            Path.Combine(Path.GetTempPath(), $"ShorokooTensorScatter_{System.Guid.NewGuid():N}")).FullName;
        try
        {
            var past = TensorData(DType.Float32, [2L, 3L, 2L], 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f);
            var wide = TensorData(DType.Float32, [2L, 4L], 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f);
            var g = QeeTensorScatterValueAuditCheck.ComputationGraph;
            var built = g.ToConcreteArchitecture(g.FromOrderedInputs([past, wide])).ToConcreteModel();
            var concrete = built.ToInternal();
            var exported = FastOnnxModelBuilder.BuildInternalOnnxModel(
                concrete, prepForOnnx: true, inputDims: [[2L, 3L, 2L], [2L, 4L]]);

            Assert.Contains(concrete.Nodes, n => n.OpCode == OpCodes.TENSOR_SCATTER);
            Assert.DoesNotContain(exported.Graph.Nodes, n => n.OpType == OpCodes.TENSOR_SCATTER);
            Assert.Contains(exported.Graph.Nodes, n => n.OpType == OpCodes.GATHER_ELEMENTS);
            Assert.Contains(
                FastOnnxModelBuilder.BuildInternalOnnxModel(concrete, applyExecutionLowerings: false).Graph.Nodes,
                n => n.OpType == OpCodes.TENSOR_SCATTER);

            var path = Path.Combine(dir, "scatter.onnx");
            Persistence.ExportOnnx(built, path);
            using var fs = File.OpenRead(path);
            Assert.Equal(21L, ProtoBuf.Serializer.Deserialize<ModelProto>(fs)
                .OpsetImports.Single(o => o.Domain == "").Version);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}

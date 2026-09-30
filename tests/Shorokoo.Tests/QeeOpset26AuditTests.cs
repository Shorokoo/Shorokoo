using System;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Core.Interpreter;
using Shorokoo.Core.Backends;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Onnx;
using Shorokoo.Runtime;
using Ort = Microsoft.ML.OnnxRuntime;
using static Shorokoo.Tests.Utils.QeeAudit;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage for the opset 22-26 op batch under Shorokoo's single-opset-21 export. The
/// decomposable ops (Swish @24, RMSNormalization @23) are lowered inline to opset-21
/// primitives by their <see cref="OnnxOp"/> entry points, so their value audits run
/// normally (the Swish audit is QEE-only only to match the audit-module style — its lowered
/// graph carries no Swish node and loads anywhere). TensorScatter (@24) is decomposed by
/// its registered lowering instead, which is a different arrangement: the node is built
/// and kept as itself, and every model written from the graph carries the decomposition.
/// The ops with no opset-21 equivalent — Attention / AttentionWithKVCache /
/// RotaryEmbedding (opset 23), BitCast / CumProd (opset 26) — cannot be emitted into an
/// opset-21 model, so their entry points throw at authoring time.
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

    [Fact]
    public void TestTensorScatterIsDecomposedInEveryWrittenModelAndKeptInTheGraph()
    {
        var dir = Directory.CreateDirectory(
            Path.Combine(Path.GetTempPath(), $"ShorokooTensorScatter_{System.Guid.NewGuid():N}")).FullName;
        try
        {
            var past = TensorData(DType.Float32, [2L, 3L, 2L], 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f);
            var wide = TensorData(DType.Float32, [2L, 4L], 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f);
            var g = QeeTensorScatterValueAuditCheck.ComputationGraph;
            var built = g.ToConcreteArchitecture([past, wide]).ToConcreteModel();
            var concrete = built.ToInternal();
            var exported = FastOnnxModelBuilder.BuildInternalOnnxModel(
                concrete, prepForOnnx: true, inputDims: [[2L, 3L, 2L], [2L, 4L]]);

            Assert.Contains(concrete.Nodes, n => n.OpCode == OpCodes.TENSOR_SCATTER);
            Assert.DoesNotContain(exported.Graph.Nodes, n => n.OpType == OpCodes.TENSOR_SCATTER);
            Assert.Contains(exported.Graph.Nodes, n => n.OpType == OpCodes.GATHER_ELEMENTS);
            Assert.DoesNotContain(
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

    [Fact]
    public void TestAnExportListedOperatorThatCannotBeDecomposedFailsTheExport()
    {
        var past = Globals.InputTensor<float32>(defaultName: "past", rank: 3);
        var update = Globals.InputTensor<float32>(defaultName: "update", rank: 3);
        ImmutableArray<Variable> inputs = [past, update];
        var graph = new InternalComputationGraph(inputs, [
            NodeBuilder.BuildNodeSingleOut(OpCodes.TENSOR_SCATTER, [past, update, null],
                [(OnnxOpAttributeNames.AttrAxis, 0L), (OnnxOpAttributeNames.AttrMode, null)])]);

        Assert.Contains(OpCodes.TENSOR_SCATTER, Assert.Throws<InvalidOperationException>(
            () => FastOnnxModelBuilder.BuildInternalOnnxModel(
                graph, prepForOnnx: true, inputDims: [[2L, 3L, 2L], [2L, 1L, 2L]])).Message);
        Assert.Contains(OpCodes.TENSOR_SCATTER, Assert.Throws<InvalidOperationException>(
            () => FastOnnxModelBuilder.BuildInternalOnnxModel(graph, applyExecutionLowerings: false)).Message);
    }

    // TensorScatter is the one lowered operator with a reference implementation to hand: ONNX
    // Runtime's CPU provider registers a native kernel for it at opset 24. The reference runs one
    // hand-built opset-24 TensorScatter node straight on ONNX Runtime, the decomposition runs
    // through Shorokoo, and the two are compared byte for byte. These are data movements, so no
    // tolerance is involved.
    private static (long[] Dims, byte[] Bytes) Scatter(bool fused, DType type,
        long[] past, long[] update, long[]? writeIndices, long? axis, TensorScatterMode? mode)
    {
        TensorData[] feed = writeIndices is null
            ? [Ramp(type, past, 1), Ramp(type, update, 100)]
            : [Ramp(type, past, 1), Ramp(type, update, 100),
               TensorData([(long)writeIndices.Length], writeIndices)];
        if (fused) return OnOrt(FusedScatter(type, feed.Length, axis, mode), feed);

        var cache = Globals.InputTensor(type, "past", rank: past.Length);
        var window = Globals.InputTensor(type, "update", rank: update.Length);
        var starts = writeIndices is null ? null : Globals.InputTensor(DType.Int64, "starts", rank: 1);
        ImmutableArray<Variable> inputs = starts is null ? [cache, window] : [cache, window, starts];
        var graph = new InternalComputationGraph(
            inputs, [OnnxOp.TensorScatter(cache, window, starts, axis, mode)]);
        var present = ComputeContext.Default.Execute(graph, [.. feed])[0].ToTensorData();
        return (present.Shape.Dims, present.AccessRawMemory().ToArray());
    }

    private static readonly string[] ScatterInputs = ["past", "update", "starts"];

    private static byte[] FusedScatter(DType type, int inputCount, long? axis, TensorScatterMode? mode)
    {
        TypeProto Typed(DType t) => new() { TensorType = new TypeProto.Tensor { ElemType = t.ProtoTypeNum } };
        var g = new GraphProto { Name = "scatter" };
        for (int i = 0; i < inputCount; i++)
            g.Inputs.Add(new ValueInfoProto { Name = ScatterInputs[i], Type = Typed(i < 2 ? type : DType.Int64) });
        var node = new NodeProto { OpType = OpCodes.TENSOR_SCATTER, Name = "scatter" };
        node.Inputs.AddRange(ScatterInputs.Take(inputCount));
        node.Outputs.Add("present");
        if (axis is { } a)
            node.Attributes.Add(new AttributeProto { Name = OnnxOpAttributeNames.AttrAxis, Type = AttributeProto.AttributeType.Int, I = a });
        if (mode is { } m)
            node.Attributes.Add(new AttributeProto { Name = OnnxOpAttributeNames.AttrMode, Type = AttributeProto.AttributeType.String,
                S = System.Text.Encoding.UTF8.GetBytes(m == TensorScatterMode.Circular ? "circular" : "linear") });
        g.Nodes.Add(node);
        g.Outputs.Add(new ValueInfoProto { Name = "present", Type = Typed(type) });
        var model = new ModelProto { IrVersion = 10, Graph = g };
        model.OpsetImports.Add(new OperatorSetIdProto { Domain = "", Version = 24 });
        using var ms = new MemoryStream();
        ProtoBuf.Serializer.Serialize(ms, model);
        return ms.ToArray();
    }

    private static (long[] Dims, byte[] Bytes) OnOrt(byte[] model, TensorData[] feed)
    {
        using var session = new Ort.InferenceSession(model);
        using var runOptions = new Ort.RunOptions();
        var values = feed.Select(t =>
        {
            var value = Ort.OrtValue.CreateAllocatedTensorValue(
                Ort.OrtAllocator.DefaultInstance, (Ort.Tensors.TensorElementType)t.DType.ProtoTypeNum, t.Shape.Dims);
            t.AccessRawMemory().CopyTo(value.GetTensorMutableRawData());
            return value;
        }).ToArray();
        try
        {
            using var outputs = session.Run(runOptions, ScatterInputs.Take(feed.Length).ToArray(), values, ["present"]);
            return (outputs[0].GetTensorTypeAndShape().Shape, outputs[0].GetTensorMutableRawData().ToArray());
        }
        finally
        {
            foreach (var value in values) value.Dispose();
        }
    }

    private static bool Matches(DType type, long[] past, long[] update,
        long[]? writeIndices = null, long? axis = null, TensorScatterMode? mode = null)
    {
        var fused = Scatter(true, type, past, update, writeIndices, axis, mode);
        var lowered = Scatter(false, type, past, update, writeIndices, axis, mode);
        return fused.Dims.SequenceEqual(lowered.Dims) && fused.Bytes.SequenceEqual(lowered.Bytes);
    }

    private static bool OrtRefuses(long[] past, long[] update,
        long[]? writeIndices = null, long? axis = null, TensorScatterMode? mode = null)
    {
        try
        {
            Scatter(true, DType.Float32, past, update, writeIndices, axis, mode);
            return false;
        }
        catch (Microsoft.ML.OnnxRuntime.OnnxRuntimeException ex)
        {
            return ex.Message.Contains(OpCodes.TENSOR_SCATTER, StringComparison.Ordinal);
        }
    }

    private static TensorData Ramp(DType type, long[] dims, int start)
    {
        int[] v = [.. Enumerable.Range(start, (int)dims.Aggregate(1L, (n, d) => n * d))];
        if (type == DType.Float64) return TensorData(dims, [.. v.Select(x => (double)x)]);
        if (type == DType.Float16) return TensorData(dims, [.. v.Select(x => (Float16)(float)x)]);
        if (type == DType.BFloat16) return TensorData(dims, [.. v.Select(x => (BFloat16)(float)x)]);
        if (type == DType.Int64) return TensorData(dims, [.. v.Select(x => (long)x)]);
        if (type == DType.Int32) return TensorData(dims, v);
        if (type == DType.Int16) return TensorData(dims, [.. v.Select(x => (short)x)]);
        if (type == DType.Int8) return TensorData(dims, [.. v.Select(x => (sbyte)x)]);
        if (type == DType.UInt64) return TensorData(dims, [.. v.Select(x => (ulong)x)]);
        if (type == DType.UInt32) return TensorData(dims, [.. v.Select(x => (uint)x)]);
        if (type == DType.UInt16) return TensorData(dims, [.. v.Select(x => (ushort)x)]);
        if (type == DType.UInt8) return TensorData(dims, [.. v.Select(x => (byte)x)]);
        if (type == DType.Bool) return TensorData(dims, [.. v.Select(x => (x & 1) == 0)]);
        return TensorData(dims, [.. v.Select(x => (float)x)]);
    }

    [Fact]
    public void TestTensorScatterLoweringMatchesTheOrtKernelAcrossAxesAndRanks()
    {
        Assert.True(Matches(DType.Float32, [2, 4], [2, 2], [1, 2], axis: -1L));
        Assert.True(Matches(DType.Float32, [2, 4], [2, 2], [3, 2], axis: 1L, mode: TensorScatterMode.Circular));
        Assert.True(Matches(DType.Float32, [2, 3, 2], [2, 1, 2], [0, 2]));
        Assert.True(Matches(DType.Float32, [2, 3, 2], [2, 1, 2], [2, 0], axis: 1L));
        Assert.True(Matches(DType.Float32, [2, 2, 3], [2, 2, 1], [0, 2], axis: -1L));
        Assert.True(Matches(DType.Float32, [2, 2, 3], [2, 2, 2], [1, 0], axis: 2L));
        Assert.True(Matches(DType.Float32, [2, 1, 4, 5], [2, 1, 2, 5], [1, 2]));
        Assert.True(Matches(DType.Float32, [2, 1, 4, 5], [2, 1, 1, 5], [3, 0], axis: 2L));
        Assert.True(Matches(DType.Float32, [2, 1, 4, 5], [2, 1, 4, 5], [0, 0], axis: 1L));
        Assert.True(Matches(DType.Float32, [2, 1, 4, 5], [2, 1, 4, 5], [0, 0], axis: -3L));
        Assert.True(Matches(DType.Float32, [2, 1, 4, 5], [2, 1, 4, 2], [3, 0], axis: -1L));
        Assert.True(Matches(DType.Float32, [2, 1, 4, 5], [2, 1, 4, 2], [3, 0], axis: 3L));
    }

    [Fact]
    public void TestTensorScatterLoweringMatchesTheOrtKernelAcrossModesWindowsAndWriteIndices()
    {
        Assert.True(Matches(DType.Float32, [2, 3, 2], [2, 1, 2]));
        Assert.True(Matches(DType.Float32, [2, 3, 2], [2, 1, 2], [0, 2], mode: TensorScatterMode.Linear));
        Assert.True(Matches(DType.Float32, [2, 3, 2], [2, 2, 2], [1, 0]));
        Assert.True(Matches(DType.Float32, [2, 3, 2], [2, 2, 2], [1, 1]));
        Assert.True(Matches(DType.Float32, [2, 3, 2], [2, 3, 2], [0, 0]));
        Assert.True(Matches(DType.Float32, [2, 3, 2], [2, 3, 2], [1, 2], mode: TensorScatterMode.Circular));
        Assert.True(Matches(DType.Float32, [2, 3, 2], [2, 2, 2], [2, 1], mode: TensorScatterMode.Circular));
        Assert.True(Matches(DType.Float32, [2, 3, 2], [2, 2, 2], [3, 0], mode: TensorScatterMode.Circular));
        Assert.True(Matches(DType.Float32, [2, 3, 2], [2, 2, 2], [1000, 1001], mode: TensorScatterMode.Circular));
        Assert.True(Matches(DType.Float32, [2, 3, 2], [2, 2, 2], mode: TensorScatterMode.Circular));
        Assert.True(Matches(DType.Float32, [3, 2, 2], [3, 1, 2], [0, 1, 1]));
        Assert.True(Matches(DType.Float32, [3, 2, 2], [3, 2, 2], [0, 1, 3], mode: TensorScatterMode.Circular));
    }

    [Fact]
    public void TestTensorScatterLoweringMatchesTheOrtKernelAcrossElementTypes()
    {
        Assert.True(Matches(DType.Float64, [2, 3, 2], [2, 2, 2], [1, 0]));
        Assert.True(Matches(DType.Float32, [2, 3, 2], [2, 2, 2], [2, 1], mode: TensorScatterMode.Circular));
        Assert.True(Matches(DType.Float16, [2, 3, 2], [2, 2, 2], [1, 0]));
        Assert.True(Matches(DType.BFloat16, [2, 3, 2], [2, 2, 2], [2, 1], mode: TensorScatterMode.Circular));
        Assert.True(Matches(DType.Int64, [2, 3, 2], [2, 2, 2], [1, 0]));
        Assert.True(Matches(DType.Int32, [2, 3, 2], [2, 2, 2], [2, 1], mode: TensorScatterMode.Circular));
        Assert.True(Matches(DType.Int16, [2, 3, 2], [2, 2, 2], [1, 0]));
        Assert.True(Matches(DType.Int8, [2, 3, 2], [2, 2, 2], [2, 1], mode: TensorScatterMode.Circular));
        Assert.True(Matches(DType.UInt64, [2, 3, 2], [2, 2, 2], [1, 0]));
        Assert.True(Matches(DType.UInt32, [2, 3, 2], [2, 2, 2], [2, 1], mode: TensorScatterMode.Circular));
        Assert.True(Matches(DType.UInt16, [2, 3, 2], [2, 2, 2], [1, 0]));
        Assert.True(Matches(DType.UInt8, [2, 3, 2], [2, 2, 2], [2, 1], mode: TensorScatterMode.Circular));
        Assert.True(Matches(DType.Bool, [2, 3, 2], [2, 2, 2], [1, 0]));
    }

    [Fact]
    public void TestTensorScatterLoweringMatchesTheOrtKernelOnEmptyAndSingletonShapes()
    {
        Assert.True(Matches(DType.Float32, [2, 1], [2, 1], [0, 0], axis: 1L));
        Assert.True(Matches(DType.Float32, [2, 1, 3], [2, 1, 3], [0, 0]));
        Assert.True(Matches(DType.Float32, [0, 3, 2], [0, 1, 2], []));
        Assert.True(Matches(DType.Float32, [0, 3, 2], [0, 1, 2]));
        Assert.True(Matches(DType.Float32, [2, 3, 2], [2, 0, 2], [0, 0]));
        Assert.True(Matches(DType.Float32, [2, 3, 2], [2, 0, 2]));
        Assert.True(Matches(DType.Float32, [2, 3, 2], [2, 0, 2], [1, 2], mode: TensorScatterMode.Circular));
        Assert.True(Matches(DType.Float32, [2, 0, 2], [2, 0, 2], [0, 0]));
        Assert.True(Matches(DType.Float32, [2, 0, 2], [2, 0, 2], [0, 0], mode: TensorScatterMode.Circular));
        Assert.True(Matches(DType.Float32, [2, 3, 0], [2, 1, 0], [0, 1]));
        Assert.True(Matches(DType.Float32, [2, 3, 0], [2, 3, 0], [0, 0], axis: 2L));
    }

    // The far side of the same comparison: every shape the lowering does not cover is one the
    // reference kernel refuses outright, so nothing here is valid input handled differently.
    // A rank-2 cache at the default axis is on this list because -2 normalizes to 0.
    [Fact]
    public void TestTheOrtKernelRefusesEveryInputTheLoweringLeavesUndefined()
    {
        Assert.True(OrtRefuses([2, 3, 2], [2, 4, 2], [0, 0]));
        Assert.True(OrtRefuses([2, 3, 2], [2, 4, 2], [0, 0], mode: TensorScatterMode.Circular));
        Assert.True(OrtRefuses([2, 3, 2], [2, 2, 2], [2, 2]));
        Assert.True(OrtRefuses([2, 3, 2], [2, 1, 2], [3, 3]));
        Assert.True(OrtRefuses([2, 3, 2], [2, 1, 2], [-1, 0]));
        Assert.True(OrtRefuses([2, 3, 2], [2, 1, 2], [-1, 0], mode: TensorScatterMode.Circular));
        Assert.True(OrtRefuses([2, 3, 2], [2, 1, 2], [0]));
        Assert.True(OrtRefuses([2, 4], [2, 2], [1, 1]));
        Assert.True(OrtRefuses([2, 3, 2], [2, 1, 2], [0, 0], axis: 3L));
        Assert.True(OrtRefuses([2, 3, 2], [2, 1, 2], [0, 0], axis: -4L));
    }
}

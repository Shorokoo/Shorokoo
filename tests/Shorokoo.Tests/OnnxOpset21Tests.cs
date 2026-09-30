using System.Collections.Immutable;
using System.IO;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Utils;
using Shorokoo.Graph;
using Shorokoo.Runtime;
using Shorokoo.Tests.Modules;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Tests.OnnxProtoBuilders;

namespace Shorokoo.Tests;

/// <summary>
/// ONNX opset 21 is the one opset Shorokoo reads and writes: import refuses a model at any other
/// opset, or carrying an operator or attribute opset 21 does not define, and every model it
/// writes — an exported file, a session's model, a <c>.srk</c> payload — is stamped 21.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class OnnxOpset21Tests
{
    private const int FloatElem = 1;

    private static AttributeProto IntAttr(string name, long value)
        => new() { Name = name, Type = AttributeProto.AttributeType.Int, I = value };

    private static AttributeProto StringAttr(string name, string value)
        => new() { Name = name, Type = AttributeProto.AttributeType.String, S = System.Text.Encoding.UTF8.GetBytes(value) };

    private static ModelProto NodeModel(string opType, AttributeProto[] attributes, params (string Domain, long Version)[] opsets)
    {
        var g = new GraphProto { Name = "g" };
        g.Inputs.Add(TensorInfo("x", FloatElem, 4));
        var node = new NodeProto { OpType = opType, Name = "n0" };
        node.Inputs.Add("x");
        node.Outputs.Add("y");
        node.Attributes.AddRange(attributes);
        g.Nodes.Add(node);
        g.Outputs.Add(TensorInfo("y", FloatElem, 4));
        var model = new ModelProto { IrVersion = 10, Graph = g };
        foreach (var (domain, version) in opsets)
            model.OpsetImports.Add(new OperatorSetIdProto { Domain = domain, Version = version });
        return model;
    }

    private static ModelProto ReluModel(params (string Domain, long Version)[] opsets) => NodeModel("Relu", [], opsets);

    private static ModelProto Opset21(string opType, params AttributeProto[] attributes) => NodeModel(opType, attributes, ("", 21));

    private static ModelProto WithFunctionAt(long version)
    {
        var model = ReluModel(("", 21), ("Functions", 1));
        var fn = new FunctionProto { Name = "f", Domain = "Functions" };
        fn.OpsetImports.Add(new OperatorSetIdProto { Domain = "", Version = version });
        model.Functions.Add(fn);
        return model;
    }

    private static string Refusal(System.Action import)
    {
        var ex = Assert.Throws<ModelException>(import);
        Assert.Equal(ErrorCodes.FW060, ex.ErrorCode);
        return ex.Message;
    }

    private static bool Refused(ModelProto model) => Refusal(() => Import(model)) is not null;

    [Fact]
    public void TestImportRefusesModelsNotStampedAtOpset21()
    {
        Assert.NotNull(Import(ReluModel(("", 21))));
        Assert.NotNull(Import(ReluModel(("ai.onnx", 21))));
        Assert.NotNull(Import(ReluModel(("Functions", 1), ("", 21))));
        Assert.True(Refused(ReluModel(("", 20))));
        Assert.True(Refused(ReluModel(("", 22))));
        Assert.True(Refused(ReluModel(("ai.onnx", 24))));
        Assert.True(Refused(ReluModel(("", 26))));
        Assert.True(Refused(ReluModel(("", 27))));
        Assert.True(Refused(ReluModel(("Functions", 21))));
        Assert.True(Refused(WithFunctionAt(22)));
        Assert.Matches("opset 23.*opset 21", Refusal(() => Import(ReluModel(("", 23)))));
    }

    [Fact]
    public void TestImportRefusesAttributesOpset21DoesNotDefine()
    {
        Assert.NotNull(Import(Opset21("Cast", IntAttr("to", FloatElem))));
        Assert.True(Refused(Opset21("Cast", IntAttr("to", FloatElem), StringAttr("round_mode", "up"))));
        Assert.True(Refused(Opset21("CastLike", StringAttr("round_mode", "up"))));
        Assert.True(Refused(Opset21("DequantizeLinear", IntAttr("output_dtype", FloatElem))));
        Assert.True(Refused(Opset21("QuantizeLinear", IntAttr("precision", 0))));
    }

    [Fact]
    public void TestImportRefusesOperatorsOpset21DoesNotDefine()
    {
        Assert.True(Refused(Opset21("Swish")));
        Assert.True(Refused(Opset21("RMSNormalization")));
        Assert.True(Refused(Opset21("RotaryEmbedding")));
        Assert.True(Refused(Opset21("Attention")));
        Assert.True(Refused(Opset21("TensorScatter")));
        Assert.True(Refused(Opset21("BitCast", IntAttr("to", 6))));
        Assert.True(Refused(Opset21("CumProd")));
    }

    private static TensorData[] Samples() =>
    [
        TensorData(DType.Float32, [2L, 3L, 2L], 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f),
        TensorData(DType.Float32, [2L, 4L], 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f),
    ];

    private static long[] Stamps(ModelProto model)
        => [.. model.OpsetImports.Concat(model.Functions.SelectMany(f => f.OpsetImports))
            .Where(o => o.Domain is "" or "ai.onnx").Select(o => o.Version).Distinct()];

    private static ModelProto Payload(byte[] srk)
        => ProtoBuf.Serializer.Deserialize<ModelProto>(new MemoryStream(SrkFileFormat.Read(srk).OnnxBytes));

    private static byte[] Restamped(byte[] srk, long version)
    {
        var model = Payload(srk);
        model.OpsetImports.Single(o => o.Domain == "").Version = version;
        using var ms = new MemoryStream();
        ProtoBuf.Serializer.Serialize(ms, model);
        return SrkFileFormat.Write(ms.ToArray(), GraphKind.Module, false, 0, model.IrVersion, [new("", version)]);
    }

    private static NodeProto[] AllNodes(ModelProto model)
        => [.. model.Graph.Nodes, .. model.Functions.SelectMany(f => f.Nodes)];

    private static bool Passes(ComputationGraph architecture)
        => ComputeContext.Default.Execute(architecture.ToConcreteModel(), [.. Samples()])[0]
            .ToTensorData().As<bit>().CopyMemory()[0];

    [Fact]
    public void TestEveryWrittenModelIsStampedAtOpset21()
    {
        var module = QeeTensorScatterValueAuditCheck.ComputationGraph;
        var concrete = module.ToConcreteArchitecture(Samples()).ToConcreteModel();
        var srk = CompressedFormatUtils.SaveFastGraphToBinary(FCLayer.ComputationGraph);
        var dir = Directory.CreateDirectory(
            Path.Combine(Path.GetTempPath(), $"ShorokooOpset21_{System.Guid.NewGuid():N}")).FullName;
        try
        {
            var path = Path.Combine(dir, "model.onnx");
            Persistence.ExportOnnx(concrete, path);
            using (var fs = File.OpenRead(path))
                Assert.Equal([21L], Stamps(ProtoBuf.Serializer.Deserialize<ModelProto>(fs)));
            Assert.Equal([21L], Stamps(FastOnnxModelBuilder.BuildInternalOnnxModel(concrete.ToInternal(), prepForOnnx: true)));
            Assert.NotEmpty(Payload(srk).Functions);
            Assert.Equal([21L], Stamps(Payload(srk)));
            Assert.Equal(21L, SrkFileFormat.Read(srk).Header.Producer!.Opsets![""]);

            var foreign = Path.Combine(dir, "foreign.onnx");
            using (var fs = File.Create(foreign))
                ProtoBuf.Serializer.Serialize(fs, ReluModel(("", 23)));
            Refusal(() => Persistence.ImportOnnx(foreign));
            Refusal(() => CompressedFormatUtils.LoadFastGraphFromBinary(Restamped(srk, 24)));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void TestASavedGraphCarryingTensorScatterStampsOpset21AndReloadsComputingTheSameValues()
    {
        var module = QeeTensorScatterValueAuditCheck.ComputationGraph;
        var architecture = module.ToConcreteArchitecture(Samples());
        var moduleSrk = CompressedFormatUtils.SaveFastGraphToBinary(module);
        var architectureSrk = CompressedFormatUtils.SaveFastGraphToBinary(architecture);

        Assert.True(Passes(architecture));
        Assert.Contains(AllNodes(Payload(moduleSrk)), n => n.OpType == OpCodes.GATHER_ELEMENTS);
        Assert.DoesNotContain(AllNodes(Payload(moduleSrk)), n => n.OpType == OpCodes.TENSOR_SCATTER);
        Assert.DoesNotContain(AllNodes(Payload(architectureSrk)), n => n.OpType == OpCodes.TENSOR_SCATTER);
        Assert.Equal([21L], Stamps(Payload(moduleSrk)));
        Assert.True(Passes(CompressedFormatUtils.LoadFastGraphFromBinary(moduleSrk).ToConcreteArchitecture(Samples())));
        Assert.True(Passes(CompressedFormatUtils.LoadFastGraphFromBinary(architectureSrk)));
    }

    private static bool EmissionRefused(Variable input, Variable output)
    {
        ImmutableArray<Variable> inputs = [input];
        var graph = new InternalComputationGraph(inputs, [output]);
        Refusal(() => FastOnnxModelBuilder.BuildOnnxModel(graph));
        Refusal(() => FastOnnxModelBuilder.BuildInternalOnnxModel(graph, prepForOnnx: true));
        Refusal(() => FastOnnxModelBuilder.BuildInternalOnnxModel(graph, applyExecutionLowerings: false, emitInputsAsNodes: true));
        return true;
    }

    [Fact]
    public void TestEmissionRefusesANodeOpset21DoesNotDefine()
    {
        var x = Globals.InputTensor<float32>(defaultName: "x", rank: 1);
        var q = Globals.InputTensor<uint8>(defaultName: "q", rank: 1);
        Assert.True(EmissionRefused(x, NodeBuilder.BuildNodeSingleOut(OpCodes.SWISH, [x], [(AttrAlpha, null)])));
        Assert.True(EmissionRefused(x, NodeBuilder.BuildNodeSingleOut(OpCodes.CAST, [x],
            [(AttrTo, DType.Float16), (AttrRoundMode, "up")])));
        Assert.True(EmissionRefused(q, NodeBuilder.BuildNodeSingleOut(OpCodes.DEQUANTIZE_LINEAR, [q, Globals.Scalar(0.5f), null],
            [(AttrAxis, null), (AttrBlockSize, null), (AttrOutputDtype, 1L)])));
    }
}

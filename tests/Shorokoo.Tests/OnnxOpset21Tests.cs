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
    private const int Int64Elem = 7;
    private const int BoolElem = 9;

    private static AttributeProto IntAttr(string name, long value)
        => new() { Name = name, Type = AttributeProto.AttributeType.Int, I = value };

    private static AttributeProto StringAttr(string name, string value)
        => new() { Name = name, Type = AttributeProto.AttributeType.String, S = System.Text.Encoding.UTF8.GetBytes(value) };

    private static NodeProto Node(string opType, string input, string output, params AttributeProto[] attributes)
    {
        var node = new NodeProto { OpType = opType, Name = output + "_node" };
        node.Inputs.Add(input);
        node.Outputs.Add(output);
        node.Attributes.AddRange(attributes);
        return node;
    }

    private static ModelProto GraphModel(GraphProto g, params (string Domain, long Version)[] opsets)
    {
        var model = new ModelProto { IrVersion = 10, Graph = g };
        foreach (var (domain, version) in opsets)
            model.OpsetImports.Add(new OperatorSetIdProto { Domain = domain, Version = version });
        return model;
    }

    private static ModelProto NodeModel(string opType, AttributeProto[] attributes, params (string Domain, long Version)[] opsets)
    {
        var g = new GraphProto { Name = "g" };
        g.Inputs.Add(TensorInfo("x", FloatElem, 4));
        var node = Node(opType, "x", "y", attributes);
        node.Name = "n0";
        g.Nodes.Add(node);
        g.Outputs.Add(TensorInfo("y", FloatElem, 4));
        return GraphModel(g, opsets);
    }

    private static ModelProto ReluModel(params (string Domain, long Version)[] opsets) => NodeModel("Relu", [], opsets);

    private static ModelProto Opset21(string opType, params AttributeProto[] attributes) => NodeModel(opType, attributes, ("", 21));

    private static ModelProto Unnamed(ModelProto model)
    {
        model.Graph.Nodes[0].Name = "";
        return model;
    }

    private static ModelProto WithFunctionAt(long? version, string opType = "Relu")
    {
        var model = ReluModel(("", 21), ("Functions", 1));
        var fn = new FunctionProto { Name = "f", Domain = "Functions" };
        if (version is { } v)
            fn.OpsetImports.Add(new OperatorSetIdProto { Domain = "", Version = v });
        fn.Inputs.Add("fx");
        fn.Outputs.Add("fy");
        fn.ValueInfoes.Add(TensorInfo("fx", FloatElem, 4));
        fn.ValueInfoes.Add(TensorInfo("fy", FloatElem, 4));
        fn.Nodes.Add(Node(opType, "fx", "fy"));
        model.Functions.Add(fn);
        return model;
    }

    private static ModelProto WithMissingExternalData(long version)
    {
        var model = NodeModel("Relu", [], ("", version));
        var w = new TensorProto { Name = "w", data_type = FloatElem, Dims = [4L], data_location = TensorProto.DataLocation.External };
        w.ExternalDatas.Add(new StringStringEntryProto { Key = "location", Value = $"missing_{System.Guid.NewGuid():N}.bin" });
        model.Graph.Initializers.Add(w);
        return model;
    }

    private static GraphProto Branch(string opType, string output)
    {
        var g = new GraphProto { Name = output };
        g.Nodes.Add(Node(opType, "x", output));
        g.Outputs.Add(TensorInfo(output, FloatElem, 4));
        return g;
    }

    private static ModelProto InIf(string opType)
    {
        var g = new GraphProto { Name = "g" };
        g.Inputs.Add(TensorInfo("x", FloatElem, 4));
        g.Inputs.Add(TensorInfo("c", BoolElem));
        var node = new NodeProto { OpType = "If", Name = "if0" };
        node.Inputs.Add("c");
        node.Outputs.Add("y");
        node.Attributes.Add(new AttributeProto { Name = "then_branch", Type = AttributeProto.AttributeType.Graph, G = Branch(opType, "t") });
        node.Attributes.Add(new AttributeProto { Name = "else_branch", Type = AttributeProto.AttributeType.Graph, G = Branch("Relu", "e") });
        g.Nodes.Add(node);
        g.Outputs.Add(TensorInfo("y", FloatElem, 4));
        return GraphModel(g, ("", 21));
    }

    private static ModelProto InLoop(string opType)
    {
        var body = new GraphProto { Name = "body" };
        body.Inputs.Add(TensorInfo("i", Int64Elem));
        body.Inputs.Add(TensorInfo("c_in", BoolElem));
        body.Inputs.Add(TensorInfo("s_in", FloatElem, 4));
        body.Nodes.Add(Node("Identity", "c_in", "c_out"));
        body.Nodes.Add(Node(opType, "s_in", "s_out"));
        body.Outputs.Add(TensorInfo("c_out", BoolElem));
        body.Outputs.Add(TensorInfo("s_out", FloatElem, 4));
        var g = new GraphProto { Name = "g" };
        g.Inputs.Add(TensorInfo("x", FloatElem, 4));
        g.Inputs.Add(TensorInfo("c", BoolElem));
        var node = new NodeProto { OpType = "Loop", Name = "loop0" };
        node.Inputs.AddRange(["", "c", "x"]);
        node.Outputs.Add("y");
        node.Attributes.Add(new AttributeProto { Name = "body", Type = AttributeProto.AttributeType.Graph, G = body });
        g.Nodes.Add(node);
        g.Outputs.Add(TensorInfo("y", FloatElem, 4));
        return GraphModel(g, ("", 21));
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
        Assert.Matches("opset 23.*opset 21", Refusal(() => Import(ReluModel(("", 23)))));
    }

    [Fact]
    public void TestImportChecksEveryDefaultDomainStampBeforeReadingTheModel()
    {
        Assert.NotNull(Import(ReluModel(("", 21), ("ai.onnx", 21))));
        Assert.NotNull(Import(WithFunctionAt(21)));
        Assert.NotNull(Import(WithFunctionAt(null)));
        Assert.True(Refused(ReluModel(("", 21), ("ai.onnx", 22))));
        Assert.True(Refused(ReluModel(("ai.onnx", 22), ("", 21))));
        Assert.True(Refused(WithFunctionAt(22)));
        Assert.True(Refused(WithMissingExternalData(22)));
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
        Assert.Contains("node 'n0' ('Attention') is an operator", Refusal(() => Import(Opset21("Attention"))));
        Assert.True(Refused(Opset21("TensorScatter")));
        Assert.True(Refused(Opset21("BitCast", IntAttr("to", 6))));
        Assert.True(Refused(Opset21("CumProd")));
        Assert.Contains("'Swish' is an operator", Refusal(() => Import(Unnamed(Opset21("Swish")))));
    }

    [Fact]
    public void TestImportRefusesAnOperatorOpset21DoesNotDefineWhereverItSits()
    {
        Assert.True(Refused(WithFunctionAt(21, "Swish")));
        Assert.True(Refused(InIf("Swish")));
        Assert.True(Refused(InLoop("Swish")));
        Assert.NotNull(Import(InIf("Relu")));
        Assert.NotNull(Import(InLoop("Relu")));
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

    private static bool Passes(ComputationGraph architecture) => ModelPasses(architecture.ToConcreteModel());

    private static bool ModelPasses(ComputationGraph model)
        => ComputeContext.Default.Execute(model, [.. Samples()])[0].ToTensorData().As<bit>().CopyMemory()[0];

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

        var skpt = Path.Combine(Path.GetTempPath(), $"ShorokooOpset21_{System.Guid.NewGuid():N}.skpt");
        try
        {
            Persistence.From(QeeTensorScatterParamAuditCheck.ComputationGraph.ToConcreteArchitecture(Samples()).ToConcreteModel())
                .WithModel().WithWeights().Save(skpt);
            Assert.Equal([21L], Stamps(Payload(SkptModelEntry(skpt))));
            Assert.DoesNotContain(AllNodes(Payload(SkptModelEntry(skpt))), n => n.OpType == OpCodes.TENSOR_SCATTER);
            Assert.True(ModelPasses(Persistence.Load(skpt)));
        }
        finally
        {
            File.Delete(skpt);
        }
    }

    private static byte[] SkptModelEntry(string path)
    {
        using var zip = System.IO.Compression.ZipFile.OpenRead(path);
        using var entry = zip.GetEntry(SkptFileFormat.ModelEntryPath)!.Open();
        using var ms = new MemoryStream();
        entry.CopyTo(ms);
        return ms.ToArray();
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

    private static bool Emits(Variable input, Variable output)
    {
        ImmutableArray<Variable> inputs = [input];
        var graph = new InternalComputationGraph(inputs, [output]);
        return FastOnnxModelBuilder.BuildOnnxModel(graph) is not null
            && FastOnnxModelBuilder.BuildInternalOnnxModel(graph, prepForOnnx: true) is not null
            && FastOnnxModelBuilder.BuildInternalOnnxModel(graph, applyExecutionLowerings: false, emitInputsAsNodes: true) is not null;
    }

    private static bool BuildRefused(System.Func<Variable> build) => Refusal(() => build()) is not null;

    [Fact]
    public void TestNodeBuilderRefusesAttributesOpset21DoesNotDefine()
    {
        var x = Globals.InputTensor<float32>(defaultName: "x", rank: 1);
        var q = Globals.InputTensor<uint8>(defaultName: "q", rank: 1);
        Assert.True(BuildRefused(() => NodeBuilder.BuildNodeSingleOut(OpCodes.DEQUANTIZE_LINEAR, [q, Globals.Scalar(0.5f), null],
            [(AttrAxis, null), (AttrBlockSize, null), (AttrOutputDtype, 10L)])));
        Assert.True(BuildRefused(() => NodeBuilder.BuildNodeSingleOut(OpCodes.QUANTIZE_LINEAR, [x, Globals.Scalar(0.5f), null],
            [(AttrPrecision, 1L)])));
        Assert.True(BuildRefused(() => NodeBuilder.BuildNodeSingleOut(OpCodes.CAST, [x], [(AttrTo, DType.Float16), (AttrRoundMode, "up")])));
        Assert.True(BuildRefused(() => NodeBuilder.BuildNodeSingleOut(OpCodes.CAST_LIKE, [x, q], [(AttrRoundMode, "up")])));
    }

    [Fact]
    public void TestEmissionRefusesANodeOpset21DoesNotDefine()
    {
        var x = Globals.InputTensor<float32>(defaultName: "x", rank: 1);
        Assert.True(EmissionRefused(x, NodeBuilder.BuildNodeSingleOut(OpCodes.SWISH, [x], [(AttrAlpha, null)])));
        Assert.True(EmissionRefused(x, NodeBuilder.BuildNodeSingleOut(OpCodes.RMS_NORMALIZATION, [x, x], [])));
        Assert.True(EmissionRefused(x, NodeBuilder.BuildNodeSingleOut(OpCodes.CUM_PROD, [x, Globals.Scalar(0L)], [])));
        Assert.True(Refusal(() => CompressedFormatUtils.SaveFastGraphToBinary(RawSwishParamLayer.ComputationGraph)) is not null);
        Assert.True(Emits(x, NodeBuilder.BuildNodeSingleOut(OpCodes.CAST, [x], [(AttrTo, DType.Float16)])));
    }
}

[TrainableParamInitializer]
public static partial class RawSwishInit
{
    public static Tensor<float32> Inline(Vector<int64> shape)
        => NodeBuilder.BuildNodeSingleOut(OpCodes.SWISH, [Globals.TensorFill(shape, 1f)], [(AttrAlpha, null)]);
}

[Module]
public partial class RawSwishParamLayer
{
    public static Tensor<float32> Inline(Tensor<float32> input) => input * RawSwishInit.Init(input.ShapeTensor());
}

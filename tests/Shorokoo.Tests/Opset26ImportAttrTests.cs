using System.IO;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Core.Inference;
using Shorokoo.Core.Graph;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// Import tolerance for the new optional attributes that opsets 22-26 added to
/// EXISTING ops (the only non-dtype changes in the whole 21→26 respec):
/// DequantizeLinear-23 <c>output_dtype</c>, QuantizeLinear-23 <c>precision</c>,
/// Cast-24 <c>round_mode</c>. Models setting them must import without error,
/// resolve the right output dtypes, and execute.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class Opset26ImportAttrTests
{
    private const int FloatElem = 1;   // TensorProto.DataType.FLOAT
    private const int UInt8Elem = 2;   // UINT8

    private static ValueInfoProto TensorInfo(string name, int elemType, params long[] dims)
    {
        var shape = new TensorShapeProto();
        foreach (var d in dims)
            shape.Dims.Add(new TensorShapeProto.Dimension { DimValue = d });
        return new ValueInfoProto
        {
            Name = name,
            Type = new TypeProto
            {
                TensorType = new TypeProto.Tensor { ElemType = elemType, Shape = shape },
            },
        };
    }

    private static AttributeProto IntAttr(string name, long value)
        => new AttributeProto { Name = name, Type = AttributeProto.AttributeType.Int, I = value };

    private static AttributeProto StringAttr(string name, string value)
        => new AttributeProto { Name = name, Type = AttributeProto.AttributeType.String, S = System.Text.Encoding.UTF8.GetBytes(value) };

    private static TensorProto Init(string name, int elemType, long[] dims, byte[] raw)
        => new TensorProto { Name = name, data_type = elemType, Dims = dims, RawData = raw };

    private static InternalComputationGraph Import(ModelProto model)
    {
        using var ms = new MemoryStream();
        ProtoBuf.Serializer.Serialize(ms, model);
        return OnnxModelImporter.FromOnnxModelToFastGraph(ms.ToArray());
    }

    private static ModelProto WrapModel(GraphProto graph, long opset)
    {
        var model = new ModelProto { IrVersion = 10, Graph = graph };
        model.OpsetImports.Add(new OperatorSetIdProto { Domain = "", Version = opset });
        return model;
    }

    [Fact]
    public void TestDequantizeLinearOutputDtypeImports()
    {
        // x: uint8 [4], scale: float scalar, output_dtype = FLOAT (1).
        var g = new GraphProto { Name = "dq" };
        g.Inputs.Add(TensorInfo("x", UInt8Elem, 4));
        g.Initializers.Add(Init("scale", FloatElem, [], System.BitConverter.GetBytes(0.5f)));
        var n = new NodeProto { OpType = "DequantizeLinear", Name = "dq0" };
        n.Inputs.AddRange(new[] { "x", "scale" });
        n.Outputs.Add("y");
        n.Attributes.Add(IntAttr("output_dtype", 1));
        g.Nodes.Add(n);
        g.Outputs.Add(TensorInfo("y", FloatElem, 4));

        var fast = Import(WrapModel(g, 23));
        var results = ComputeContext.Default.Execute(fast,
            new IData[] { TensorData(DType.UInt8, [4L], (byte)2, (byte)4, (byte)6, (byte)8) });
        var td = (TensorData<float32>)results[0].ToTensorData();
        Assert.Equal(new[] { 1f, 2f, 3f, 4f }, td.AccessMemory().ToArray());

        // QEE dtype propagation honors the attribute as well.
        var qee = new QuickExecutionEngine();
        var store = qee.Run(fast, new TensorData[] { TensorData(DType.UInt8, [4L], (byte)2, (byte)4, (byte)6, (byte)8) });
        Assert.Equal(DType.Float32, store[fast.Outputs[0]].DType);
    }

    [Fact]
    public void TestQuantizeLinearPrecisionImports()
    {
        var g = new GraphProto { Name = "q" };
        g.Inputs.Add(TensorInfo("x", FloatElem, 4));
        g.Initializers.Add(Init("scale", FloatElem, [], System.BitConverter.GetBytes(0.5f)));
        var n = new NodeProto { OpType = "QuantizeLinear", Name = "q0" };
        n.Inputs.AddRange(new[] { "x", "scale" });
        n.Outputs.Add("y");
        n.Attributes.Add(IntAttr("precision", 0)); // default-valued, must be tolerated
        g.Nodes.Add(n);
        g.Outputs.Add(TensorInfo("y", UInt8Elem, 4));

        var fast = Import(WrapModel(g, 23));
        var results = ComputeContext.Default.Execute(fast,
            new IData[] { TensorData(DType.Float32, [4L], 1f, 2f, 3f, 4f) });
        var td = results[0].ToTensorData();
        Assert.Equal(DType.UInt8, td.DType);
    }

    [Fact]
    public void TestCastRoundModeImports()
    {
        var g = new GraphProto { Name = "c" };
        g.Inputs.Add(TensorInfo("x", FloatElem, 2));
        var n = new NodeProto { OpType = "Cast", Name = "c0" };
        n.Inputs.Add("x");
        n.Outputs.Add("y");
        n.Attributes.Add(IntAttr("to", FloatElem));
        n.Attributes.Add(StringAttr("round_mode", "up")); // default-valued; float8e8m0-only semantics
        g.Nodes.Add(n);
        g.Outputs.Add(TensorInfo("y", FloatElem, 2));

        var fast = Import(WrapModel(g, 24));
        var results = ComputeContext.Default.Execute(fast,
            new IData[] { TensorData(DType.Float32, [2L], 1.5f, -2.5f) });
        var td = (TensorData<float32>)results[0].ToTensorData();
        Assert.Equal(new[] { 1.5f, -2.5f }, td.AccessMemory().ToArray());
    }
}

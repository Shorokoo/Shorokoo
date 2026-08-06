using System.IO;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Core.Inference;
using Shorokoo.Core.Graph;
using Shorokoo.Runtime;
using static Shorokoo.Tests.OnnxProtoBuilders;

namespace Shorokoo.Tests;

/// <summary>
/// Import tolerance for the optional attributes opsets 22-26 added to existing ops:
/// DequantizeLinear-23 <c>output_dtype</c>, QuantizeLinear-23 <c>precision</c>,
/// Cast-24 <c>round_mode</c>.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class Opset26ImportAttrTests
{
    private const int FloatElem = 1;
    private const int UInt8Elem = 2;

    private static AttributeProto IntAttr(string name, long value)
        => new AttributeProto { Name = name, Type = AttributeProto.AttributeType.Int, I = value };

    private static AttributeProto StringAttr(string name, string value)
        => new AttributeProto { Name = name, Type = AttributeProto.AttributeType.String, S = System.Text.Encoding.UTF8.GetBytes(value) };

    private static ModelProto WrapModel(GraphProto graph, long opset)
    {
        var model = new ModelProto { IrVersion = 10, Graph = graph };
        model.OpsetImports.Add(new OperatorSetIdProto { Domain = "", Version = opset });
        return model;
    }

    [Fact]
    public void TestDequantizeOutputDtypeQuantizePrecisionAndCastRoundModeImport()
    {
        string[] xScaleInputs = ["x", "scale"];

        var dq = new GraphProto { Name = "dq" };
        dq.Inputs.Add(TensorInfo("x", UInt8Elem, 4));
        dq.Initializers.Add(Init("scale", FloatElem, [], System.BitConverter.GetBytes(0.5f)));
        var dqNode = new NodeProto { OpType = "DequantizeLinear", Name = "dq0" };
        dqNode.Inputs.AddRange(xScaleInputs);
        dqNode.Outputs.Add("y");
        dqNode.Attributes.Add(IntAttr("output_dtype", 1));
        dq.Nodes.Add(dqNode);
        dq.Outputs.Add(TensorInfo("y", FloatElem, 4));

        var dqGraph = Import(WrapModel(dq, 23));
        IData[] dqInputs = [TensorData(DType.UInt8, [4L], (byte)2, (byte)4, (byte)6, (byte)8)];
        float[] dqExpected = [1f, 2f, 3f, 4f];
        Assert.Equal(dqExpected,
            ((TensorData<float32>)ComputeContext.Default.Execute(dqGraph, dqInputs)[0].ToTensorData())
                .AccessMemory().ToArray());

        TensorData[] dqQeeInputs = [TensorData(DType.UInt8, [4L], (byte)2, (byte)4, (byte)6, (byte)8)];
        var dqStore = new QuickExecutionEngine().Run(dqGraph, dqQeeInputs);
        Assert.Equal(DType.Float32, dqStore[dqGraph.Outputs[0]].DType);

        var q = new GraphProto { Name = "q" };
        q.Inputs.Add(TensorInfo("x", FloatElem, 4));
        q.Initializers.Add(Init("scale", FloatElem, [], System.BitConverter.GetBytes(0.5f)));
        var qNode = new NodeProto { OpType = "QuantizeLinear", Name = "q0" };
        qNode.Inputs.AddRange(xScaleInputs);
        qNode.Outputs.Add("y");
        qNode.Attributes.Add(IntAttr("precision", 0));
        q.Nodes.Add(qNode);
        q.Outputs.Add(TensorInfo("y", UInt8Elem, 4));

        IData[] qInputs = [TensorData(DType.Float32, [4L], 1f, 2f, 3f, 4f)];
        Assert.Equal(DType.UInt8,
            ComputeContext.Default.Execute(Import(WrapModel(q, 23)), qInputs)[0].ToTensorData().DType);

        var c = new GraphProto { Name = "c" };
        c.Inputs.Add(TensorInfo("x", FloatElem, 2));
        var cNode = new NodeProto { OpType = "Cast", Name = "c0" };
        cNode.Inputs.Add("x");
        cNode.Outputs.Add("y");
        cNode.Attributes.Add(IntAttr("to", FloatElem));
        cNode.Attributes.Add(StringAttr("round_mode", "up"));
        c.Nodes.Add(cNode);
        c.Outputs.Add(TensorInfo("y", FloatElem, 2));

        IData[] cInputs = [TensorData(DType.Float32, [2L], 1.5f, -2.5f)];
        float[] cExpected = [1.5f, -2.5f];
        Assert.Equal(cExpected,
            ((TensorData<float32>)ComputeContext.Default.Execute(Import(WrapModel(c, 24)), cInputs)[0].ToTensorData())
                .AccessMemory().ToArray());
    }
}

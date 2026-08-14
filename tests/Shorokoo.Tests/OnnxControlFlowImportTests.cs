using Shorokoo.Core.Factory.IR;
using Shorokoo.Core.Inference;
using Shorokoo.Runtime;
using static Shorokoo.Tests.OnnxProtoBuilders;

namespace Shorokoo.Tests;

/// <summary>
/// Import-side coverage for the ONNX control-flow operators: <c>Loop</c>, and the
/// <c>SequenceMap</c> that is rejected as a documented limitation. Models are built from
/// Shorokoo's own proto classes, imported through <see cref="OnnxModelImporter"/>, then
/// executed via ComputeContext and the QuickExecutionEngine.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class OnnxControlFlowImportTests
{
    private const int FloatElem = 1;
    private const int Int64Elem = 7;
    private const int BoolElem = 9;

    private static ValueInfoProto SequenceInfo(string name, int elemType)
        => new ValueInfoProto
        {
            Name = name,
            Type = new TypeProto
            {
                SequenceType = new TypeProto.Sequence
                {
                    ElemType = new TypeProto
                    {
                        TensorType = new TypeProto.Tensor { ElemType = elemType, Shape = new TensorShapeProto() },
                    },
                },
            },
        };

    private static NodeProto Node(string opType, string name, string[] inputs, string[] outputs, params AttributeProto[] attrs)
    {
        var node = new NodeProto { OpType = opType, Name = name };
        node.Inputs.AddRange(inputs);
        node.Outputs.AddRange(outputs);
        node.Attributes.AddRange(attrs);
        return node;
    }

    private static void AssertTensorEquals(TensorData expected, TensorData actual)
    {
        Assert.Equal(expected.Shape.Dims, actual.Shape.Dims);
        Assert.Equal(expected.AccessRawMemory().ToArray(), actual.AccessRawMemory().ToArray());
    }

    /// <summary>Only reachable through import — a Shorokoo-authored loop passes no initial
    /// condition, so its zero-trip case is a trip count of 0.</summary>
    [Fact]
    public void TestLoopWithAFalseInitialConditionKeepsTheInitialStateOnBothEngines()
    {
        var graph = Import(BuildDoublingWhileLoopModel());
        var init = TensorData(DType.Float32, [4L], 1f, 2f, 3f, 4f);
        var no = TensorData(DType.Bool, [], false);

        AssertTensorEquals(init, new ComputeContext().Execute(graph, no, init)[0].ToTensorData());

        var sFinal = Assert.IsType<RuntimeTensor>(
            new QuickExecutionEngine().Run(graph, no, init)[graph.Outputs[0]]);
        Assert.Equal([4L], sFinal.Shape!.Dims);
        Assert.Equal([1f, 2f, 3f, 4f], sFinal.FloatData!.Value.ToArray());
    }

    /// <summary>A graph that supplies no iteration count leaves the loop unbounded, so the
    /// engine bounds the walk itself and reports shape without data.</summary>
    [Fact]
    public void TestLoopWithNoTripCountTerminatesWithShapeOnlyOutputs()
    {
        var graph = Import(BuildDoublingWhileLoopModel());

        var hinted = Assert.IsType<RuntimeTensor>(new QuickExecutionEngine().Run(graph,
            TensorData(DType.Bool, [], true),
            TensorData(DType.Float32, [4L], 1f, 2f, 3f, 4f))[graph.Outputs[0]]);
        Assert.Equal([4L], hinted.Shape!.Dims);
        Assert.Null(hinted.FloatData);

        var unhinted = Assert.IsType<RuntimeTensor>(
            new QuickExecutionEngine().Run(graph)[graph.Outputs[0]]);
        Assert.Null(unhinted.FloatData);
    }

    /// <summary>while (cond) { s = s + s; } over a [4] state, with no trip-count limit.</summary>
    private static ModelProto BuildDoublingWhileLoopModel()
    {
        var body = new GraphProto { Name = "loop_body" };
        body.Inputs.Add(TensorInfo("iter", Int64Elem));
        body.Inputs.Add(TensorInfo("cond_in", BoolElem));
        body.Inputs.Add(TensorInfo("s_in", FloatElem, 4));
        body.Nodes.Add(Node("Identity", "body_cond", ["cond_in"], ["cond_out"]));
        body.Nodes.Add(Node("Add", "body_double", ["s_in", "s_in"], ["s_out"]));
        body.Outputs.Add(TensorInfo("cond_out", BoolElem));
        body.Outputs.Add(TensorInfo("s_out", FloatElem, 4));

        var graph = new GraphProto { Name = "while_graph" };
        graph.Inputs.Add(TensorInfo("cond", BoolElem));
        graph.Inputs.Add(TensorInfo("init", FloatElem, 4));
        graph.Nodes.Add(Node("Loop", "the_loop", ["", "cond", "init"], ["s_final"],
            new AttributeProto { Name = "body", Type = AttributeProto.AttributeType.Graph, G = body }));
        graph.Outputs.Add(TensorInfo("s_final", FloatElem, 4));
        return WrapModel(graph);
    }

    [Fact]
    public void TestSequenceMapImportFailsWithActionableError()
    {
        var body = new GraphProto { Name = "seqmap_body" };
        body.Inputs.Add(TensorInfo("elem_in", FloatElem));
        body.Nodes.Add(Node("Identity", "body_id", ["elem_in"], ["elem_out"]));
        body.Outputs.Add(TensorInfo("elem_out", FloatElem));

        var graph = new GraphProto { Name = "seqmap_graph" };
        graph.Inputs.Add(SequenceInfo("seq", FloatElem));
        graph.Nodes.Add(Node("SequenceMap", "the_seqmap", ["seq"], ["out_seq"],
            new AttributeProto { Name = "body", Type = AttributeProto.AttributeType.Graph, G = body }));
        graph.Outputs.Add(SequenceInfo("out_seq", FloatElem));

        var ex = Assert.Throws<NotSupportedException>(() => Import(WrapModel(graph)));
        Assert.Contains("SequenceMap", ex.Message);
        Assert.Contains("the_seqmap", ex.Message);
        Assert.Contains("LoopAPI", ex.Message);
    }
}

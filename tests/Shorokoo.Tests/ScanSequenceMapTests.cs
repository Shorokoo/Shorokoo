using Shorokoo.Core.Factory.IR;
using Shorokoo.Core.Inference;
using Shorokoo.Runtime;
using static Shorokoo.Tests.OnnxProtoBuilders;

namespace Shorokoo.Tests;

/// <summary>
/// Import-side coverage for <c>Scan</c> (lowered to <c>Loop</c> by
/// <c>OnnxControlFlowLowering</c>), for the <c>Loop</c> it lowers to, and for
/// <c>SequenceMap</c> (a documented import limitation). Models are built from Shorokoo's
/// own proto classes, imported through <see cref="OnnxModelImporter"/>, then executed via
/// ComputeContext and the QuickExecutionEngine.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class ScanSequenceMapTests
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

    /// <summary>Running-sum Scan: s_out = s_in + x_t; y_t = Identity(s_out), over a [3,4] tensor.</summary>
    private static ModelProto BuildCumulativeSumScanModel(
        long stateLen,
        long[] outYShape,
        long[]? scanInputAxes = null,
        long[]? scanInputDirections = null,
        long[]? scanOutputAxes = null,
        long[]? scanOutputDirections = null,
        long xRows = 3)
    {
        var body = new GraphProto { Name = "scan_body" };
        body.Inputs.Add(TensorInfo("s_in", FloatElem, stateLen));
        body.Inputs.Add(TensorInfo("x_t", FloatElem, stateLen));
        body.Nodes.Add(Node("Add", "body_add", ["s_in", "x_t"], ["s_out"]));
        body.Nodes.Add(Node("Identity", "body_y", ["s_out"], ["y_t"]));
        body.Outputs.Add(TensorInfo("s_out", FloatElem, stateLen));
        body.Outputs.Add(TensorInfo("y_t", FloatElem, stateLen));

        var scanAttrs = new List<AttributeProto>
        {
            new AttributeProto { Name = "body", Type = AttributeProto.AttributeType.Graph, G = body },
            new AttributeProto { Name = "num_scan_inputs", Type = AttributeProto.AttributeType.Int, I = 1 },
        };
        void AddInts(string name, long[]? vals)
        {
            if (vals is not null)
                scanAttrs.Add(new AttributeProto { Name = name, Type = AttributeProto.AttributeType.Ints, Ints = vals });
        }
        AddInts("scan_input_axes", scanInputAxes);
        AddInts("scan_input_directions", scanInputDirections);
        AddInts("scan_output_axes", scanOutputAxes);
        AddInts("scan_output_directions", scanOutputDirections);

        var graph = new GraphProto { Name = "scan_graph" };
        graph.Inputs.Add(TensorInfo("init", FloatElem, stateLen));
        graph.Inputs.Add(TensorInfo("x", FloatElem, xRows, 4));
        graph.Nodes.Add(Node("Scan", "the_scan", ["init", "x"], ["s_final", "y"], scanAttrs.ToArray()));
        graph.Outputs.Add(TensorInfo("s_final", FloatElem, stateLen));
        graph.Outputs.Add(TensorInfo("y", FloatElem, outYShape));
        return WrapModel(graph);
    }

    private static readonly TensorData X3x4 = TensorData(DType.Float32, [3L, 4L],
        1f, 2f, 3f, 4f,
        5f, 6f, 7f, 8f,
        9f, 10f, 11f, 12f);

    private static void AssertTensorEquals(TensorData expected, TensorData actual)
    {
        Assert.Equal(expected.Shape.Dims, actual.Shape.Dims);
        Assert.Equal(expected.AccessRawMemory().ToArray(), actual.AccessRawMemory().ToArray());
    }

    [Fact]
    public void TestScanImportLoweringForwardAxis0ReverseAxis1AndQuickExecutionEngine()
    {
        var graph = Import(BuildCumulativeSumScanModel(stateLen: 4, outYShape: [3L, 4L]));

        Assert.DoesNotContain(graph.Nodes, n => n.OpCode.StartsWith(OpCodes.SCAN));
        Assert.Contains(graph.Nodes, n => n.OpCode == OpCodes.LOOP_OPEN);
        Assert.Contains(graph.Nodes, n => n.OpCode == OpCodes.LOOP_CLOSE);

        var results = new ComputeContext().Execute(graph, TensorData(DType.Float32, [4L], 0f, 0f, 0f, 0f), X3x4);
        Assert.Equal(2, results.Length);
        AssertTensorEquals(
            TensorData(DType.Float32, [4L], 15f, 18f, 21f, 24f),
            results[0].ToTensorData());
        AssertTensorEquals(
            TensorData(DType.Float32, [3L, 4L],
                1f, 2f, 3f, 4f,
                6f, 8f, 10f, 12f,
                15f, 18f, 21f, 24f),
            results[1].ToTensorData());

        var reverseGraph = Import(BuildCumulativeSumScanModel(
            stateLen: 3, outYShape: [4L, 3L],
            scanInputAxes: [1L], scanInputDirections: [1L]));
        var reverseResults = new ComputeContext().Execute(
            reverseGraph, TensorData(DType.Float32, [3L], 0f, 0f, 0f), X3x4);
        Assert.Equal(2, reverseResults.Length);
        AssertTensorEquals(
            TensorData(DType.Float32, [3L], 10f, 26f, 42f),
            reverseResults[0].ToTensorData());
        AssertTensorEquals(
            TensorData(DType.Float32, [4L, 3L],
                4f, 8f, 12f,
                7f, 15f, 23f,
                9f, 21f, 33f,
                10f, 26f, 42f),
            reverseResults[1].ToTensorData());

        var store = new QuickExecutionEngine().Run(graph,
            TensorData(DType.Float32, [4L], 0f, 0f, 0f, 0f),
            X3x4);
        var sFinal = store[graph.Outputs[0]];
        Assert.Equal(DType.Float32, sFinal.DType);
        Assert.Equal([4L], Assert.IsType<RuntimeTensor>(sFinal).Shape!.Dims);
        var y = store[graph.Outputs[1]];
        Assert.Equal(DType.Float32, y.DType);
        Assert.Equal([3L, 4L], Assert.IsType<RuntimeTensor>(y).Shape!.Dims);
    }

    /// <summary>A zero-row Scan lowers to a zero-trip Loop. Only the scan output's row count
    /// is pinned: its inner dims are engine-dependent with no iteration to infer them from.</summary>
    [Fact]
    public void TestZeroRowScanKeepsTheInitialStateAndEmptiesTheScanOutputOnBothEngines()
    {
        var graph = Import(BuildCumulativeSumScanModel(stateLen: 4, outYShape: [0L, 4L], xRows: 0));
        var init = TensorData(DType.Float32, [4L], 1f, 2f, 3f, 4f);
        var x = TensorData(DType.Float32, [0L, 4L]);

        var results = new ComputeContext().Execute(graph, init, x);
        AssertTensorEquals(init, results[0].ToTensorData());
        Assert.Equal(0L, results[1].ToTensorData().Shape.Dims[0]);

        var store = new QuickExecutionEngine().Run(graph, init, x);
        var sFinal = Assert.IsType<RuntimeTensor>(store[graph.Outputs[0]]);
        Assert.Equal([4L], sFinal.Shape!.Dims);
        Assert.Equal([1f, 2f, 3f, 4f], sFinal.FloatData!.Value.ToArray());
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

    // A no-trip-count Loop whose condition QEE cannot resolve to false never terminates:
    // LoopOpenOp's vestigial "true" makes continueWhen look resolved, so anyUnknown stays
    // false and the MaxIterationsForUnknownBounds cap never fires (Shorokoo#169). Skipped
    // rather than left failing because it hangs and exhausts memory instead of returning.
    [Fact(Skip = "Shorokoo#169: QEE never terminates a no-trip-count Loop with a true condition")]
    public void TestLoopWithNoTripCountAndATrueConditionStillTerminates()
    {
        var graph = Import(BuildDoublingWhileLoopModel());
        var store = new QuickExecutionEngine().Run(graph,
            TensorData(DType.Bool, [], true), TensorData(DType.Float32, [4L], 1f, 2f, 3f, 4f));
        Assert.Equal([4L], Assert.IsType<RuntimeTensor>(store[graph.Outputs[0]]).Shape!.Dims);
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
    public void TestScanImportRejectsNonZeroScanOutputAxesAndReverseScanOutputDirections()
    {
        var axesEx = Assert.Throws<NotSupportedException>(
            () => Import(BuildCumulativeSumScanModel(stateLen: 4, outYShape: [4L, 3L], scanOutputAxes: [1L])));
        Assert.Contains("scan_output_axes", axesEx.Message);
        Assert.Contains("the_scan", axesEx.Message);

        var dirEx = Assert.Throws<NotSupportedException>(
            () => Import(BuildCumulativeSumScanModel(stateLen: 4, outYShape: [3L, 4L], scanOutputDirections: [1L])));
        Assert.Contains("scan_output_directions", dirEx.Message);
        Assert.Contains("the_scan", dirEx.Message);
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

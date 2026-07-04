using Shorokoo.Core.Factory.Mlir;
using Shorokoo.Core.Graph;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage-purpose round-trip tests for the MLIR-flavored assembly printer/parser
/// (<see cref="MlirTextWriter"/> / <see cref="MlirTextReader"/>). Each case builds a small
/// <see cref="FastComputationGraph"/> with the standard <c>Globals</c> idiom, prints it, parses
/// it back, and asserts two independent equalities:
///   * <b>semantic</b> — the reconstructed graph serializes to the same ONNX JSON via the
///     existing <see cref="OnnxUtils.ToJson"/> path;
///   * <b>textual fixpoint</b> — re-printing the parsed graph reproduces the exact same text.
/// See <c>src/docs/design/mlir-assembly-parser.md</c>.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class MlirTextRoundTripCoverageTests
{
    private static void AssertRoundTrips(FastComputationGraph graph)
    {
        var text = MlirTextWriter.Write(graph);
        var parsed = MlirTextReader.Parse(text);

        // Structural sanity.
        Assert.True(parsed.IsLinearOrderValid());
        Assert.Equal(graph.Nodes.Count, parsed.Nodes.Count);
        Assert.Equal(graph.Inputs.Count, parsed.Inputs.Count);
        Assert.Equal(graph.Outputs.Count, parsed.Outputs.Count);

        // Textual fixpoint: parsing then reprinting yields identical text.
        Assert.Equal(text, MlirTextWriter.Write(parsed));

        // Semantic equality through the existing ONNX serializer.
        Assert.Equal(graph.ToJson(), parsed.ToJson());
    }

    [Fact]
    public void ElementwiseAdd_RoundTrips()
    {
        var x = InputTensor<float32>("x");
        var y = InputTensor<float32>("y");
        var graph = new FastComputationGraph([x, y], [x + y]);
        AssertRoundTrips(graph);

        // The input nodes carry a DType attribute, so that printer/parser path is exercised here.
        Assert.Contains("dtype<", MlirTextWriter.Write(graph));
    }

    [Fact]
    public void MaxPool_ExercisesEnumLongsAndBoolAttributes()
    {
        var x = InputTensor<float32>("x", rank: 4);
        var pooled = NN.MaxPool(
            x,
            ceilMode: true,
            dilations: [1L, 1L],
            kernelShape: [2L, 2L],
            pads: [0L, 0L, 0L, 0L],
            storageOrder: 0L,
            strides: [2L, 2L],
            autoPad: AutoPad.SameUpper);
        var graph = new FastComputationGraph([x], [pooled]);
        AssertRoundTrips(graph);

        // Forces the Enum (autoPad), Longs (kernelShape/strides/...), and Bool (ceilMode) paths.
        var text = MlirTextWriter.Write(graph);
        Assert.Contains("enum<", text);
        Assert.Contains(": i64", text);
        Assert.Contains("true", text);
    }

    [Fact]
    public void ChainedAdds_RoundTrip()
    {
        var x = InputTensor<float32>("x");
        var y = InputTensor<float32>("y");
        var z = InputTensor<float32>("z");
        AssertRoundTrips(new FastComputationGraph([x, y, z], [(x + y) + z]));
    }

    [Fact]
    public void MultipleOutputs_RoundTrip()
    {
        var x = InputTensor<float32>("x");
        var y = InputTensor<float32>("y");
        AssertRoundTrips(new FastComputationGraph([x, y], [x + y, x - y]));
    }

    [Fact]
    public void ScalarConstant_TensorAttribute_RoundTrips()
    {
        var x = InputTensor<float32>("x");
        var graph = new FastComputationGraph([x], [x + Scalar(1.0f)]);
        AssertRoundTrips(graph);

        // The float32 Constant node forces the Tensor (dense<…>) attribute path.
        Assert.Contains("dense<", MlirTextWriter.Write(graph));
    }

    [Fact]
    public void Int64ShapeConstant_TensorAttribute_RoundTrips()
    {
        var x = InputTensor<float32>("x", rank: 2);
        var reshaped = x.Reshape([Scalar(4L), Scalar(4L)]);
        var graph = new FastComputationGraph([x], [reshaped]);
        AssertRoundTrips(graph);

        Assert.Contains("dtype<7>", MlirTextWriter.Write(graph)); // int64 shape constant(s)
    }

    [Fact]
    public void LoopWithTrainableParams_RoundTrips()
    {
        // LoopLayer exercises everything at once: a Loop#OPEN/#CLOSE scope with named 'body'
        // input/output groups, the close node's Graph (body) attribute, #TrainableParamRef#
        // nodes carrying a TargetFunction (InitSimple), and int64 shape constants.
        var graph = Shorokoo.Tests.Modules.LoopLayer.ComputationGraph;
        AssertRoundTrips(graph);

        var text = MlirTextWriter.Write(graph);
        Assert.Contains("Loop#OPEN", text);
        Assert.Contains("out \"body\"", text);      // named output group
        Assert.Contains("graphattr<", text);         // Graph (body) attribute
        Assert.Contains("func @fn", text);           // function symbol table
        Assert.Contains("tgtfn @fn", text);          // TargetFunction reference
    }

    [Fact]
    public void IfElse_RoundTrips()
    {
        // If#OPEN/#CLOSE with then/else branch groups plus trainable params on both branches.
        var graph = Shorokoo.Tests.Modules.ConditionalTrainableParamLayer.ComputationGraph;
        AssertRoundTrips(graph);
        Assert.Contains("If#OPEN", MlirTextWriter.Write(graph));
    }

    [Fact]
    public void IfInsideLoop_RoundTrips()
    {
        // Nested control flow: an If scope inside a Loop scope, both with trainable params.
        AssertRoundTrips(Shorokoo.Tests.Modules.ConditionalTrainableParamInLoopLayer.ComputationGraph);
    }

    [Fact]
    public void Parse_UnknownOpCode_Throws()
    {
        const string text = """
            graph {
              inputs = []
              outputs = [%N1_T0]
              input_names = []
              output_names = [_]
              %N1 = "TotallyNotAnOp"() -> (%N1_T0)
            }
            """;
        Assert.Throws<NotSupportedException>(() => MlirTextReader.Parse(text));
    }
}

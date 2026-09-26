using Shorokoo.Core.Factory.IR;
using Shorokoo.Runtime;
using static Shorokoo.Tests.OnnxProtoBuilders;

namespace Shorokoo.Tests;

/// <summary>
/// Models written at an opset before an operator's operands moved from attributes to inputs
/// (<c>axes</c> of Squeeze, Unsqueeze and ReduceSum at 13, of the other reductions at 18;
/// <c>split</c> of Split at 13) import with the operand as an input, and run.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class LegacyOpsetImportTests
{
    private const int FloatElem = 1;

    private static float[] Run(long opset, string opType, long[] inputDims, long[] outputDims, params AttributeProto[] attributes)
    {
        var graph = new GraphProto { Name = "legacy" };
        graph.Inputs.Add(TensorInfo("x", FloatElem, inputDims));
        var node = new NodeProto { OpType = opType, Name = "n0" };
        node.Inputs.Add("x");
        node.Outputs.Add("y");
        if (opType == "Split")
            node.Outputs.Add("rest");
        node.Attributes.AddRange(attributes);
        graph.Nodes.Add(node);
        graph.Outputs.Add(TensorInfo("y", FloatElem, outputDims));
        var model = new ModelProto { IrVersion = 7, Graph = graph };
        model.OpsetImports.Add(new OperatorSetIdProto { Domain = "", Version = opset });

        long count = inputDims.Aggregate(1L, (a, b) => a * b);
        float[] values = [.. Enumerable.Range(1, (int)count).Select(i => (float)i)];
        IData[] inputs = [TensorData(inputDims, values)];
        return ((TensorData<float32>)ComputeContext.Default.Execute(Import(model), inputs)[0].ToTensorData())
            .AccessMemory().ToArray();
    }

    private static AttributeProto Ints(string name, params long[] values)
        => new() { Name = name, Type = AttributeProto.AttributeType.Ints, Ints = values };

    private static AttributeProto Int(string name, long value)
        => new() { Name = name, Type = AttributeProto.AttributeType.Int, I = value };

    [Fact]
    public void OperandsGivenAsAttributesBeforeTheyBecameInputsImportAndRun()
    {
        Assert.Equal([1f, 2f, 3f], Run(11, "Unsqueeze", [3], [1, 3], Ints("axes", 0)));
        Assert.Equal([1f, 2f, 3f], Run(11, "Squeeze", [1, 3], [3], Ints("axes", 0)));
        Assert.Equal([6f, 15f], Run(11, "ReduceSum", [2, 3], [2], Ints("axes", 1), Int("keepdims", 0)));
        Assert.Equal([2f, 5f], Run(11, "ReduceMean", [2, 3], [2, 1], Ints("axes", -1)));
        Assert.Equal([3f, 6f], Run(17, "ReduceMax", [2, 3], [2, 1], Ints("axes", 1)));
        Assert.Equal([1f, 4f], Run(11, "Split", [2, 3], [2, 1], Int("axis", 1), Ints("split", 1, 2)));
    }
}

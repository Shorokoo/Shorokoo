using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage for the <c>CallCustomOperator&lt;T...&gt;</c> /
/// <c>CallCustomOperatorArrayOut&lt;T&gt;</c> overloads on <see cref="NodeBuilder"/>,
/// which <c>CSharpModelBuilder.MakeCustomCodeTemplate</c> emits for custom ops without
/// a built-in <c>CodeTemplate</c>.
/// </summary>
[Trait("Domain", "Framework")]
[Trait("Purpose", "Coverage")]
public class NodeBuilderCoverageTests
{
    [Fact]
    public void TestCallCustomOperatorArity1To4AndArrayOutOverloads()
    {
        object?[] noAttrs = [];
        object?[] topKAttrs = [AttrAxis, 0L, AttrLargest, true, AttrSorted, true];
        object?[] uniqueAttrs = [AttrSorted, true];
        object?[] splitAttrs = [AttrAxis, 0L, AttrNumOutputs, 2L];

        var a = InputScalar<float32>("a");
        var b = InputScalar<float32>("b");
        Assert.NotNull(NodeBuilder.CallCustomOperator<Scalar<float32>>(ADD, [a, b], noAttrs));

        var x = InputTensor<float32>("x", rank: 1);
        var k = InputScalar<int64>("k");
        var (values, indices) = NodeBuilder.CallCustomOperator<Tensor<float32>, Tensor<int64>>(
            TOPK, [x, k], topKAttrs);
        Assert.NotNull(values);
        Assert.NotNull(indices);

        var dx = InputTensor<float32>("dx", rank: 1);
        var (y, yScale, yZeroPoint) =
            NodeBuilder.CallCustomOperator<Tensor<uint8>, Scalar<float32>, Scalar<uint8>>(
                DYNAMIC_QUANTIZE_LINEAR, [dx], noAttrs);
        Assert.NotNull(y);
        Assert.NotNull(yScale);
        Assert.NotNull(yZeroPoint);

        var ux = InputTensor<float32>("ux", rank: 1);
        var (uy, uIdx, uInv, uCnt) =
            NodeBuilder.CallCustomOperator<
                Tensor<float32>, Tensor<int64>, Tensor<int64>, Tensor<int64>>(
                UNIQUE, [ux], uniqueAttrs);
        Assert.NotNull(uy);
        Assert.NotNull(uIdx);
        Assert.NotNull(uInv);
        Assert.NotNull(uCnt);

        var sData = InputTensor<float32>("sd", rank: 1);
        var pieces = NodeBuilder.CallCustomOperatorArrayOut<Tensor<float32>>(
            SPLIT, [sData, null], splitAttrs);
        Assert.Equal(2, pieces.Length);
        Assert.All(pieces, p => Assert.NotNull(p));
    }
}

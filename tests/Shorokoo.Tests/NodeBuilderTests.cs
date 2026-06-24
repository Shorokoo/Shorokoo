using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage-purpose tests that drive the <c>CallCustomOperator&lt;T...&gt;</c> /
/// <c>CallCustomOperatorArrayOut&lt;T&gt;</c> overloads on
/// <see cref="NodeBuilder"/>. These overloads are emitted by
/// <c>CSharpModelBuilder.MakeCustomCodeTemplate</c> as the call template for
/// custom ops without a built-in <c>CodeTemplate</c>, but no test in the
/// existing suite actually invokes them at runtime.
/// </summary>
[Trait("Domain", "Framework")]
[Trait("Purpose", "Coverage")]
public class NodeBuilderCoverageTests
{
    [Fact]
    public void TestCallCustomOperatorOverloads()
    {
        // Arity-1: ADD has one output. Drives CallCustomOperator<T1>.
        var a = InputScalar<float32>("a");
        var b = InputScalar<float32>("b");
        var sum = NodeBuilder.CallCustomOperator<Scalar<float32>>(
            ADD, [a, b], new object?[] { });
        Assert.NotNull(sum);

        // Arity-2: TOPK returns (values, indices). Drives CallCustomOperator<T1, T2>.
        var x = InputTensor<float32>("x", rank: 1);
        var k = InputScalar<int64>("k");
        var (values, indices) = NodeBuilder.CallCustomOperator<Tensor<float32>, Tensor<int64>>(
            TOPK, [x, k], new object?[] { AttrAxis, 0L, AttrLargest, true, AttrSorted, true });
        Assert.NotNull(values);
        Assert.NotNull(indices);

        // Arity-3: DYNAMIC_QUANTIZE_LINEAR returns (y, yScale, yZeroPoint).
        // Drives CallCustomOperator<T1, T2, T3>.
        var dx = InputTensor<float32>("dx", rank: 1);
        var (y, yScale, yZeroPoint) =
            NodeBuilder.CallCustomOperator<Tensor<uint8>, Scalar<float32>, Scalar<uint8>>(
                DYNAMIC_QUANTIZE_LINEAR, [dx], new object?[] { });
        Assert.NotNull(y);
        Assert.NotNull(yScale);
        Assert.NotNull(yZeroPoint);

        // Arity-4: UNIQUE returns (y, indices, inverseIndices, counts).
        // Drives CallCustomOperator<T1, T2, T3, T4>.
        var ux = InputTensor<float32>("ux", rank: 1);
        var (uy, uIdx, uInv, uCnt) =
            NodeBuilder.CallCustomOperator<
                Tensor<float32>, Tensor<int64>, Tensor<int64>, Tensor<int64>>(
                UNIQUE, [ux], new object?[] { AttrSorted, true });
        Assert.NotNull(uy);
        Assert.NotNull(uIdx);
        Assert.NotNull(uInv);
        Assert.NotNull(uCnt);

        // Variadic-output: SPLIT with num_outputs=2 returns a 2-element array.
        // Drives CallCustomOperatorArrayOut<T>.
        var sData = InputTensor<float32>("sd", rank: 1);
        var pieces = NodeBuilder.CallCustomOperatorArrayOut<Tensor<float32>>(
            SPLIT, [sData, null], new object?[] { AttrAxis, 0L, AttrNumOutputs, 2L });
        Assert.Equal(2, pieces.Length);
        Assert.All(pieces, p => Assert.NotNull(p));
    }
}

using System;
using System.Linq;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Rng;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>Keyed uniform draw at the input's shape under a literal key (123, 456), drawBase 0.</summary>
[Module]
public partial class RngKeyedUniformDraw
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        Vector<int64> key = [Scalar(123L), Scalar(456L)];
        return (Tensor<float32>)InternalOp.RngUniform(
            key, Scalar(0L), x.ShapeTensor(), Scalar(0f), Scalar(1f), RngAlgorithms.Default);
    }
}

/// <summary>Splits key (7, 9) at index 5, then draws uniform under the child key.</summary>
[Module]
public partial class RngSplitThenDraw
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        Vector<int64> parent = [Scalar(7L), Scalar(9L)];
        var child = InternalOp.RngSplit(parent, Scalar(5L), RngAlgorithms.Default);
        return (Tensor<float32>)InternalOp.RngUniform(
            child, Scalar(0L), x.ShapeTensor(), Scalar(0f), Scalar(1f), RngAlgorithms.Default);
    }
}

/// <summary>Keyed normal draw at the input's shape under a literal key.</summary>
[Module]
public partial class RngKeyedNormalDraw
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        Vector<int64> key = [Scalar(11L), Scalar(13L)];
        return (Tensor<float32>)InternalOp.RngNormal(
            key, Scalar(0L), x.ShapeTensor(), Scalar(0f), Scalar(1f), RngAlgorithms.Default);
    }
}

/// <summary>
/// Coverage for the named-algorithm keyed RNG operators (SHRK_RNG_SPLIT / UNIFORM / NORMAL)
/// and their ONNX lowering: each op lowers at export to a call of the algorithm's
/// <b>non-inlined</b> function (an ONNX local FunctionProto tagged with
/// RngAlgorithm / RngFunctionKind metadata), and the executed values reproduce the host
/// Threefry generator bit-for-bit — including through an index-based key split.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngAlgorithmTests
{
    private static float[] RunDraw<TModule>(long rows, long cols)
    {
        var g = (FastComputationGraph)typeof(TModule)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var input = TensorData([rows, cols], Enumerable.Repeat(0f, (int)(rows * cols)).ToArray());
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([input])).ToConcreteModel();
        var outputs = ComputeContext.Default.Execute(concrete, input);
        return outputs[0].ToTensorData().As<float32>().AccessMemory().ToArray();
    }

    // Host reference: element i -> counter (i, drawBase); uniform = low 24 bits of x0 * 2^-24.
    private static float HostUniform(long i, uint k0, uint k1, uint drawBase = 0)
    {
        var (x0, _) = Threefry2x32.Bijection((uint)i, drawBase, k0, k1);
        return (x0 & 0x00FFFFFFu) * (1.0f / 16777216.0f);
    }

    [Fact]
    public void TestKeyedUniformMatchesHostBitExact()
    {
        var vals = RunDraw<RngKeyedUniformDraw>(4, 4);
        Assert.Equal(16, vals.Length);
        for (long i = 0; i < 16; i++)
            Assert.Equal(HostUniform(i, 123, 456), vals[i]);
    }

    [Fact]
    public void TestSplitThenDrawMatchesHostFold()
    {
        // Child key = Bijection(counter: (5, 0), key: (7, 9)) — the split — then the draw
        // under the child key must match the host generator keyed by that child.
        var (ck0, ck1) = Threefry2x32.Bijection(5u, 0u, 7u, 9u);
        var vals = RunDraw<RngSplitThenDraw>(4, 4);
        Assert.Equal(16, vals.Length);
        for (long i = 0; i < 16; i++)
            Assert.Equal(HostUniform(i, ck0, ck1), vals[i]);
    }

    [Fact]
    public void TestKeyedNormalHasStandardMoments()
    {
        var vals = RunDraw<RngKeyedNormalDraw>(40, 40);
        double mean = vals.Average();
        double variance = vals.Select(v => (v - mean) * (v - mean)).Average();
        Assert.InRange(mean, -0.1, 0.1);
        Assert.InRange(variance, 0.85, 1.15);
    }

    [Fact]
    public void TestGetFunctionRejectsUnknownAlgorithmForEveryKind()
    {
        // An unknown algorithm name must fail loudly for every kind. The split kind remaps
        // the name to the default (the key tree is algorithm-independent), and that remap
        // must never launder an unrecognized name into a valid one.
        foreach (var kind in (string[])[RngAlgorithms.KindSplit, RngAlgorithms.KindUniform, RngAlgorithms.KindNormal])
        {
            var ex = Assert.Throws<NotSupportedException>(
                () => RngAlgorithms.GetFunction("Threefry4x64-Ziggurat.v9", kind));
            Assert.Contains("Threefry4x64-Ziggurat.v9", ex.Message);
        }
    }

    [Fact]
    public void TestRngFunctionsExportNonInlinedWithMetadata()
    {
        var g = (FastComputationGraph)typeof(RngSplitThenDraw)
            .GetProperty("ComputationGraph")!.GetValue(null)!;
        var input = TensorData([2L, 2L], 0f, 0f, 0f, 0f);
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([input])).ToConcreteModel();

        var proto = FastOnnxModelBuilder.BuildOnnxModel(concrete);

        // The SPLIT constant-folds away by design: its QEE op computes real values, so a
        // split chain over constant key/index collapses to a literal child-key constant
        // at concretization — key derivation costs nothing in the exported graph. The DRAW
        // (whose values QEE deliberately does not compute) must survive as a call of the
        // algorithm's NON-INLINED function: a local FunctionProto tagged with the algorithm
        // name and function kind.
        var rngFns = proto.Functions.Where(f => f.Name.Contains("ShrkRng_")).ToArray();
        Assert.True(rngFns.Length >= 1,
            $"expected the uniform algorithm FunctionProto; functions=[{string.Join(",", proto.Functions.Select(f => f.Name))}]");
        foreach (var fn in rngFns)
        {
            var algo = fn.MetadataProps.FirstOrDefault(p => p.Key == Function.IRRngAlgorithmParamName)?.Value;
            var kind = fn.MetadataProps.FirstOrDefault(p => p.Key == Function.IRRngFunctionKindParamName)?.Value;
            Assert.Equal(RngAlgorithms.Threefry2x32BoxMullerV1, algo);
            Assert.Contains(kind, (string[])[RngAlgorithms.KindSplit, RngAlgorithms.KindUniform, RngAlgorithms.KindNormal]);
        }

        // The main graph must CALL the draw (a Functions-domain call node), not contain
        // its spliced body; and the folded split must NOT appear as a node.
        var callOps = proto.Graph.Nodes.Where(n => n.Domain == "Functions").Select(n => n.OpType).ToArray();
        Assert.Contains(callOps, op => op.Contains("uniform"));
        Assert.DoesNotContain(proto.Graph.Nodes, n => n.OpType.Contains("RngSplit"));
    }
}

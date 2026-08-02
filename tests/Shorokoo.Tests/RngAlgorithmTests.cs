using System;
using System.Linq;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Rng;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>Keyed uniform draw at the input's shape under a literal key, drawBase 0.</summary>
[Module]
public partial class RngKeyedUniformDraw
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var key = Scalar(123UL | (456UL << 32));
        return (Tensor<float32>)InternalOp.RngUniform(
            key, Scalar(0UL), x.ShapeTensor(), Scalar(0f), Scalar(1f), RngAlgorithms.Default);
    }
}

/// <summary>Splits a literal key at index 5, then draws uniform under the child key.</summary>
[Module]
public partial class RngSplitThenDraw
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var parent = Scalar(7UL | (9UL << 32));
        var child = InternalOp.RngSplit(parent, Scalar(5UL), RngAlgorithms.Default);
        return (Tensor<float32>)InternalOp.RngUniform(
            child, Scalar(0UL), x.ShapeTensor(), Scalar(0f), Scalar(1f), RngAlgorithms.Default);
    }
}

/// <summary>Keyed normal draw at the input's shape under a literal key.</summary>
[Module]
public partial class RngKeyedNormalDraw
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var key = Scalar(11UL | (13UL << 32));
        return (Tensor<float32>)InternalOp.RngNormal(
            key, Scalar(0UL), x.ShapeTensor(), Scalar(0f), Scalar(1f), RngAlgorithms.Default);
    }
}

/// <summary>Keyed raw-bits draw (U32) at the input's shape under a literal key, drawBase 0.</summary>
[Module]
public partial class RngKeyedBitsDraw
{
    public static Tensor<uint32> Inline(Tensor<float32> x)
    {
        var key = Scalar(123UL | (456UL << 32));
        return (Tensor<uint32>)InternalOp.RngBits(
            key, Scalar(0UL), x.ShapeTensor(), DType.UInt32, RngAlgorithms.Default);
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
        var g = ((ComputationGraph)typeof(TModule)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var input = TensorData([rows, cols], Enumerable.Repeat(0f, (int)(rows * cols)).ToArray());
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([input])).ToConcreteModel();
        var outputs = ComputeContext.Default.Execute(concrete, input);
        return outputs[0].ToTensorData().As<float32>().AccessMemory().ToArray();
    }

    // Host reference: element i -> counter (i, drawBase); uniform = low 24 bits of x0 * 2^-24.
    private static float HostUniform(long i, ulong key, uint drawBase = 0)
    {
        var (x0, _) = Threefry2x32.Bijection((uint)i, drawBase, (uint)key, (uint)(key >> 32));
        return (x0 & 0x00FFFFFFu) * (1.0f / 16777216.0f);
    }

    private static uint[] RunDrawU32<TModule>(long rows, long cols)
    {
        var g = ((ComputationGraph)typeof(TModule)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var input = TensorData([rows, cols], Enumerable.Repeat(0f, (int)(rows * cols)).ToArray());
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([input])).ToConcreteModel();
        return ComputeContext.Default.Execute(concrete, input)[0]
            .ToTensorData().As<uint32>().AccessMemory().ToArray();
    }

    [Fact]
    public void TestKeyedUniformMatchesHostBitExact()
    {
        var vals = RunDraw<RngKeyedUniformDraw>(4, 4);
        Assert.Equal(16, vals.Length);
        for (long i = 0; i < 16; i++)
            Assert.Equal(HostUniform(i, 123UL | (456UL << 32)), vals[i]);
    }

    [Fact]
    public void TestKeyedBitsMatchesHostBitExact()
    {
        // The keyed raw-bits draw (InternalOp.RngBits) executes to the whole generator word x0
        // under the literal key, bit-for-bit — the raw-bits analogue of the uniform keyed draw.
        var vals = RunDrawU32<RngKeyedBitsDraw>(4, 4);
        Assert.Equal(16, vals.Length);
        for (long i = 0; i < 16; i++)
        {
            var (x0, _) = Threefry2x32.Bijection((uint)i, 0u, 123u, 456u);
            Assert.Equal(x0, vals[i]);
        }
    }

    [Fact]
    public void TestGetFunctionBitsValidatesWidth()
    {
        // bits requires a supported uint width...
        Assert.Throws<NotSupportedException>(
            () => RngAlgorithms.GetFunction(RngAlgorithms.Default, RngAlgorithms.KindBits, DType.Float32));
        Assert.Throws<NotSupportedException>(
            () => RngAlgorithms.GetFunction(RngAlgorithms.Default, RngAlgorithms.KindBits, null));
        // ...and a bitsDtype must not be supplied for a non-bits kind.
        Assert.Throws<ArgumentException>(
            () => RngAlgorithms.GetFunction(RngAlgorithms.Default, RngAlgorithms.KindUniform, DType.UInt32));
    }

    [Fact]
    public void TestSplitThenDrawMatchesHostFold()
    {
        // Child key = Bijection(counter: 5, key) — the split — then the draw
        // under the child key must match the host generator keyed by that child.
        var (ck0, ck1) = Threefry2x32.Bijection(5u, 0u, 7u, 9u);
        var childKey = ck0 | ((ulong)ck1 << 32);
        var vals = RunDraw<RngSplitThenDraw>(4, 4);
        Assert.Equal(16, vals.Length);
        for (long i = 0; i < 16; i++)
            Assert.Equal(HostUniform(i, childKey), vals[i]);
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
        foreach (var kind in (string[])[
            RngAlgorithms.KindSplit, RngAlgorithms.KindSplitBatch,
            RngAlgorithms.KindUniform, RngAlgorithms.KindNormal])
        {
            var ex = Assert.Throws<NotSupportedException>(
                () => RngAlgorithms.GetFunction("Threefry4x64-Ziggurat.v9", kind));
            Assert.Contains("Threefry4x64-Ziggurat.v9", ex.Message);
        }
    }

    [Fact]
    public void TestRngFunctionsExportNonInlinedWithMetadata()
    {
        var g = ((ComputationGraph)typeof(RngSplitThenDraw)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var input = TensorData([2L, 2L], 0f, 0f, 0f, 0f);
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([input])).ToConcreteModel();

        var proto = FastOnnxModelBuilder.BuildOnnxModel(concrete);

        // RNG is graph-only (#136): the split no longer constant-folds host-side. Both the
        // split AND the draw survive as calls of the algorithm's NON-INLINED functions — local
        // FunctionProtos tagged with the algorithm name and function kind. (The split resolves
        // to a literal key only later, at ORT session build, not in this exported proto.)
        var rngFns = proto.Functions.Where(f => f.Name.Contains("ShrkRng_")).ToArray();
        Assert.True(rngFns.Length == 2,
            $"expected the split + uniform algorithm FunctionProtos; functions=[{string.Join(",", proto.Functions.Select(f => f.Name))}]");
        foreach (var fn in rngFns)
        {
            var algo = fn.MetadataProps.FirstOrDefault(p => p.Key == Function.IRRngAlgorithmParamName)?.Value;
            var kind = fn.MetadataProps.FirstOrDefault(p => p.Key == Function.IRRngFunctionKindParamName)?.Value;
            Assert.Equal(RngAlgorithms.Threefry2x32BoxMullerV1, algo);
            Assert.Contains(kind, (string[])[RngAlgorithms.KindSplit, RngAlgorithms.KindUniform, RngAlgorithms.KindNormal]);
        }

        // The main graph CALLS both the split and the draw (Functions-domain call nodes), not
        // their spliced bodies; and no raw SHRK_RNG_SPLIT opcode survives (it was lowered).
        var callOps = proto.Graph.Nodes.Where(n => n.Domain == "Functions").Select(n => n.OpType).ToArray();
        Assert.Contains(callOps, op => op.Contains("uniform"));
        Assert.Contains(callOps, op => op.Contains("split"));   // #136: split now exported as a call, not folded away
        Assert.DoesNotContain(proto.Graph.Nodes, n => n.OpType.Contains("RngSplit"));
    }
}

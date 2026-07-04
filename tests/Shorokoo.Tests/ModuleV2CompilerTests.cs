using System.Linq;
using Shorokoo.Core;
using Shorokoo.Core.Factory.Mlir;
using Shorokoo.Core.Graph;
using Shorokoo.ModuleV2;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage tests for the prototype <see cref="ModuleV2Compiler"/> (Phase 2 frontend slice):
/// statically lowering a straight-line C# method to MLIR text. Each case compiles source,
/// parses the result through <see cref="MlirTextReader"/> to prove it is well-formed, and
/// where applicable checks it structurally against the graph the tracer produces for the
/// equivalent delegate — the free differential oracle. See
/// <c>src/docs/design/mlir-assembly-parser.md</c>.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class ModuleV2CompilerCoverageTests
{
    private static string[] SortedOpCodes(FastComputationGraph g)
        => g.Nodes.Select(n => n.OpCode).OrderBy(x => x, System.StringComparer.Ordinal).ToArray();

    [Fact]
    public void ElementwiseAdd_MatchesTracer()
    {
        const string source = """
            static Tensor<float32> F(Tensor<float32> x, Tensor<float32> y)
            {
                return x + y;
            }
            """;
        var compiled = MlirTextReader.Parse(ModuleV2Compiler.CompileToMlir(source));

        Assert.Equal(2, compiled.Inputs.Count);
        Assert.Single(compiled.Outputs);
        Assert.True(compiled.IsLinearOrderValid());

        // Differential check against the tracer for the same computation.
        var traced = GraphBuilder.BuildFastComputationGraphFromDelegate(
            (System.Func<Tensor<float32>, Tensor<float32>, Tensor<float32>>)(static (x, y) => x + y));
        Assert.Equal(SortedOpCodes(traced), SortedOpCodes(compiled));
    }

    [Fact]
    public void LocalsChainWithMatMul_MatchesTracer()
    {
        const string source = """
            static Tensor<float32> F(Tensor<float32> x, Tensor<float32> y)
            {
                var a = x.MatMul(y);
                var b = a + x;
                return b;
            }
            """;
        var compiled = MlirTextReader.Parse(ModuleV2Compiler.CompileToMlir(source));

        Assert.Equal(2, compiled.Inputs.Count);
        Assert.Contains("MatMul", SortedOpCodes(compiled));

        var traced = GraphBuilder.BuildFastComputationGraphFromDelegate(
            (System.Func<Tensor<float32>, Tensor<float32>, Tensor<float32>>)(static (x, y) =>
            {
                var a = x.MatMul(y);
                var b = a + x;
                return b;
            }));
        Assert.Equal(SortedOpCodes(traced), SortedOpCodes(compiled));
    }

    [Fact]
    public void UnaryReluChain_Parses()
    {
        const string source = """
            static Tensor<float32> F(Tensor<float32> x)
            {
                var a = x + x;
                return a.Relu();
            }
            """;
        var g = MlirTextReader.Parse(ModuleV2Compiler.CompileToMlir(source));
        Assert.Contains("Relu", SortedOpCodes(g));
        Assert.Single(g.Inputs);
    }

    [Fact]
    public void UnsupportedConstruct_Throws()
    {
        // A numeric literal is not lowerable in this slice (constants come later).
        const string source = """
            static Tensor<float32> F(Tensor<float32> x)
            {
                return x + 1;
            }
            """;
        Assert.Throws<System.NotSupportedException>(() => ModuleV2Compiler.CompileToMlir(source));
    }
}

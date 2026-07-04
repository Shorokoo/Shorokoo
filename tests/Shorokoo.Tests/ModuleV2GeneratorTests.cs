using System.Linq;
using Shorokoo.Graph;

namespace Shorokoo.Tests;

/// <summary>
/// A <c>[ModuleV2]</c> class whose body is compiled statically (never executed) to MLIR text by
/// the <c>ModuleV2SourceGenerator</c> at build time; the generated <c>ComputationGraph</c> property
/// parses that embedded text at runtime. This exercises the full shipping path end to end.
/// </summary>
[ModuleV2]
public partial class V2GeneratedDense
{
    public static Tensor<float32> Inline(Tensor<float32> x, Tensor<float32> w)
    {
        var y = x.MatMul(w);
        return y + x;
    }
}

[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class ModuleV2GeneratorCoverageTests
{
    [Fact]
    public void GeneratedComputationGraph_IsBuiltFromEmbeddedMlir()
    {
        // The property exists only because the source generator statically compiled Inline to MLIR
        // and emitted the parser call — no tracing, no reflection over the method body.
        FastComputationGraph g = V2GeneratedDense.ComputationGraph;

        Assert.True(g.IsLinearOrderValid());
        Assert.Equal(2, g.Inputs.Count);
        Assert.Single(g.Outputs);

        var opCodes = g.Nodes.Select(n => n.OpCode).ToList();
        Assert.Contains("MatMul", opCodes);
        Assert.Contains("Add", opCodes);

        // Each access returns an independent clone (mirrors the [Module] ComputationGraph contract).
        Assert.NotSame(g, V2GeneratedDense.ComputationGraph);
    }
}

using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;

namespace Shorokoo.Tests;

/// <summary>
/// MSG007: a plain C# loop in a <c>[Module]</c>, or in a helper it calls, that creates unnamed
/// parameters or sub-models on every pass.
/// </summary>
[Trait("Domain", "Modules")]
[Trait("Purpose", "Coverage")]
public class ModulesCodeGeneratorPlainLoopTests
{
    private static readonly Lazy<MetadataReference[]> References = new(() =>
    {
        _ = typeof(Shorokoo.Modules.Layers.Linear);
        return
        [
            .. AppDomain.CurrentDomain.GetAssemblies()
                .Where(a => !a.IsDynamic && a.Location != "")
                .Select(a => MetadataReference.CreateFromFile(a.Location)),
        ];
    });

    private const string Usings = """
        using System.Collections.Generic;
        using Shorokoo;
        using Shorokoo.Modules;
        using Shorokoo.Modules.Initializers;
        using Shorokoo.Modules.Layers;
        using static Shorokoo.Globals;
        namespace N;

        """;

    private static int Found(string body, string helpers = "") => FoundAcrossEdits(body, helpers)[0];

    private static int[] FoundAcrossEdits(string body, params string[] helperEdits)
    {
        var module = CSharpSyntaxTree.ParseText(Usings + $$"""
            [TrainableParamInitializer]
            public static partial class Half
            {
                public static Tensor<float32> Inline(Vector<int64> shape) => TensorFill(shape, 0.5f);
            }

            [Module]
            public partial class Leaf
            {
                public static Tensor<float32> Inline(Tensor<float32> x) => x;
            }

            [Module]
            public partial class Host
            {
                public static Tensor<float32> Inline(Tensor<float32> x, Scalar<int64> n)
                {
                    {{body}}
                    return x;
                }
            }
            """);
        GeneratorDriver driver = CSharpGeneratorDriver.Create(new ModuleSourceGenerator());
        Compilation? compilation = null;
        SyntaxTree? helperTree = null;
        List<int> counts = [];
        foreach (var helpers in helperEdits)
        {
            var edited = CSharpSyntaxTree.ParseText(Usings + $$"""
                public static class Helpers
                {
                    public static Tensor<float32> Layer(Tensor<float32> x) => x * Half.Init([Scalar(2L)]);
                    public static Tensor<float32> Pure(Tensor<float32> x) => x + x;
                    public static Tensor<float32> Recursive(Tensor<float32> x, int n) => n == 0 ? x : Recursive(x, n - 1);
                    {{helpers}}
                }
                """);
            compilation = compilation is null
                ? CSharpCompilation.Create("PlainLoops", [module, edited], References.Value, new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary))
                : compilation.ReplaceSyntaxTree(helperTree!, edited);
            helperTree = edited;
            driver = driver.RunGenerators(compilation);
            counts.Add(driver.GetRunResult().Diagnostics.Count(d => d.Id == "MSG007" && d.Severity == DiagnosticSeverity.Warning));
        }
        return [.. counts];
    }

    [Fact]
    public void TestPlainLoopsThatStackLayersAreReported()
    {
        Assert.Equal(1, Found("for (int i = 0; i < 3; i++) x = x * Half.Init([Scalar(2L)]);"));
        Assert.Equal(1, Found("for (int i = 0; i < 3; i++) { var w = Ones.Init([Scalar(2L)]); x = x * w; }"));
        Assert.Equal(1, Found("foreach (var w in (long[])[2L, 3L]) x = Linear.Call(Scalar(w), Scalar(true), x);"));
        Assert.Equal(1, Found("int i = 0; while (i++ < 3) x = Leaf.Model().Call(x);"));
        Assert.Equal(1, Found("int i = 0; do { x = Leaf.Call(x); } while (++i < 3);"));
        Assert.Equal(1, Found("for (int i = 0; i < 3; i++) for (int j = 0; j < 2; j++) x = x * Half.Init([Scalar(2L)]);"));
        Assert.Equal(1, Found("for (int i = 0; i < 3; i++) foreach (var ctx in LoopAPI.Iterate(n)) x = x * Half.Init([Scalar(2L)]);"));
        Assert.Equal(1, Found("for (int i = 0; i < 3; i++) x = Leaf.Call(x).Named($\"l{i}\");"));
    }

    [Fact]
    public void TestLoopsReachedThroughHelpersAndLocalFunctionsAreReported()
    {
        Assert.Equal(1, Found("for (int i = 0; i < 3; i++) x = Helpers.Layer(x);"));
        Assert.Equal(1, Found("x = Helpers.Stack(x);", "public static Tensor<float32> Stack(Tensor<float32> x) { for (int i = 0; i < 3; i++) x = Layer(x); return x; }"));
        Assert.Equal(1, Found("x = Helpers.Stack(x); x = Helpers.Stack(x);", "public static Tensor<float32> Stack(Tensor<float32> x) { for (int i = 0; i < 3; i++) x = Layer(x); return x; }"));
        Assert.Equal(1, Found("Tensor<float32> Layer(Tensor<float32> t) => t * Half.Init([Scalar(2L)]); for (int i = 0; i < 3; i++) x = Layer(x);"));
        Assert.Equal(1, Found("Tensor<float32> Stack(Tensor<float32> t) { for (int i = 0; i < 3; i++) t = t * Half.Init([Scalar(2L)]); return t; } x = Stack(x);"));
    }

    [Fact]
    public void TestAnEditToAHelperInAnotherFileUpdatesTheReport()
    {
        const string stacks = "public static Tensor<float32> Block(Tensor<float32> x) => Layer(x);";
        const string pure = "public static Tensor<float32> Block(Tensor<float32> x) => Pure(x);";
        Assert.Equal([1, 0, 1], FoundAcrossEdits("for (int i = 0; i < 3; i++) x = Helpers.Block(x);", stacks, pure, stacks));
    }

    [Fact]
    public void TestLoopsThatDoNotStackLayersAreNotReported()
    {
        Assert.Equal(0, Found("foreach (var ctx in LoopAPI.Iterate(n)) x = x * Half.Init([Scalar(2L)]);"));
        Assert.Equal(0, Found("for (int i = 0; i < 3; i++) x = x * Half.Init([Scalar(2L)]).Named($\"w{i}\");"));
        Assert.Equal(0, Found("for (int i = 0; i < 3; i++) { var w = Half.Init([Scalar(2L)]); w = w.Named($\"w{i}\"); x = x * w; }"));
        Assert.Equal(0, Found("for (int i = 0; i < 3; i++) { var m = Leaf.Model(); m = m.Named($\"l{i}\"); x = m.Call(x); }"));
        Assert.Equal(0, Found("var lin = Linear.Model(Scalar(2L), Scalar(true)); for (int i = 0; i < 3; i++) x = lin.Call(x);"));
        Assert.Equal(0, Found("for (int i = 0; i < 3; i++) x = Helpers.Pure(x) + Helpers.Recursive(x, 2);"));
        Assert.Equal(0, Found("var layers = new List<System.Func<Tensor<float32>, Tensor<float32>>>(); for (int i = 0; i < 3; i++) layers.Add(t => t * Half.Init([Scalar(2L)]));"));
        Assert.Equal(0, Found("for (int i = 0; i < 3; i++) LoopAPI.Init(x);"));
        Assert.Equal(0, Found("x = x * Half.Init([Scalar(2L)]); x = Helpers.Layer(x);"));
        Assert.Equal(0, Found("x = Helpers.Named(x);", "public static Tensor<float32> Named(Tensor<float32> x) { for (int i = 0; i < 3; i++) x = x * Half.Init([Scalar(2L)]).Named($\"h{i}\"); return x; }"));
    }
}

using System.Text.RegularExpressions;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Shorokoo.CodeGen;

namespace Shorokoo.Tests;

[Trait("Domain", "Modules")]
[Trait("Purpose", "Coverage")]
public class ModulesCodeGeneratorLocalNameTests
{
    private const string Source = """
        using Shorokoo;
        using Shorokoo.Modules;
        using Shorokoo.Modules.Initializers;
        using Shorokoo.Modules.Layers;
        using static Shorokoo.Globals;
        namespace N;

        [TrainableParamInitializer]
        public static partial class Half
        {
            public static Tensor<float32> Inline(Vector<int64> shape) => TensorFill(shape, 0.5f);
        }

        [Module]
        public partial class Host
        {
            public static Tensor<float32> Inline(Tensor<float32> x)
            {
                var gain = Half.Init([Scalar(2L)]);
                var proj = Linear.Model(Scalar(2L), Scalar(true));
                var inner = Leaf.Model();
                var shift = Ones.Init([Scalar(2L)]).Vec();
                var _ = Ones.Init([Scalar(2L)]);
                return inner.Call(proj.Call(x)) * gain + shift;
            }
        }

        [Module]
        public partial class Leaf
        {
            public static Tensor<float32> Inline(Tensor<float32> x) => x;
        }
        """;

    private static (string[] Names, string[] GeneratorDiagnostics, string[] Errors) Generate(bool interceptorsEnabled)
    {
        _ = typeof(Shorokoo.Modules.Layers.Linear);
        var options = CSharpParseOptions.Default.WithLanguageVersion(LanguageVersion.Preview);
        if (interceptorsEnabled)
            options = options.WithFeatures([new("InterceptorsNamespaces", LocalParamNameGenerator.InterceptorNamespace)]);
        var compilation = CSharpCompilation.Create("LocalNames",
            [CSharpSyntaxTree.ParseText(Source, options)],
            AppDomain.CurrentDomain.GetAssemblies()
                .Where(a => !a.IsDynamic && a.Location != "")
                .Select(a => MetadataReference.CreateFromFile(a.Location)),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary, nullableContextOptions: NullableContextOptions.Enable));
        CSharpGeneratorDriver
            .Create([new ModuleSourceGenerator().AsSourceGenerator(), new LocalParamNameGenerator().AsSourceGenerator()],
                parseOptions: options)
            .RunGeneratorsAndUpdateCompilation(compilation, out var updated, out var diagnostics);
        var generated = updated.SyntaxTrees.FirstOrDefault(t => t.FilePath.EndsWith("ShorokooLocalParamNames.g.cs"));
        return (
            generated is null ? [] : [.. Regex.Matches(generated.ToString(), "NamedByLocal\\(.*, \"(\\w+)\"\\)").Select(m => m.Groups[1].Value)],
            [.. diagnostics.Where(d => d.Id.StartsWith("MSG")).Select(d => d.Id)],
            [.. updated.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error).Select(d => d.ToString())]);
    }

    [Fact]
    public void TestALocalAssignedAnInitOrModelCallNamesItThroughAnInterceptor()
    {
        var (names, generatorDiagnostics, errors) = Generate(interceptorsEnabled: true);
        Assert.Equal(["gain", "proj", "inner"], names);
        Assert.Empty(generatorDiagnostics.Where(id => id == "MSG006"));
        Assert.Empty(errors);

        (names, generatorDiagnostics, errors) = Generate(interceptorsEnabled: false);
        Assert.Empty(names);
        Assert.Contains("MSG006", generatorDiagnostics);
        Assert.Empty(errors);
    }
}

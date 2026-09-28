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

    private const string CoreNamespaceSource = """
        using Shorokoo;
        namespace Core;

        public interface Pair : IStruct
        {
            Scalar<float32> First { get; }
            Scalar<float32> Second { get; }
        }

        [Module]
        public partial class Swap
        {
            public static TensorStruct<Pair> Inline(TensorStruct<Pair> p) => p;
        }

        [Module]
        public partial class Wrap
        {
            public static TensorStruct<Pair> Inline(TensorStruct<Pair> p, [Hyper] Model<TensorStruct<Pair>, TensorStruct<Pair>> inner)
                => inner.Call(p);
        }

        [Module]
        public partial class Outer
        {
            public static TensorStruct<Pair> Inline(TensorStruct<Pair> p)
            {
                var swap = Swap.Model();
                var wrap = Wrap.Model(swap);
                return wrap.Call(p);
            }
        }
        """;

    private const string SpelledSource = """
        using Shorokoo;
        using Shorokoo.Modules.Initializers;
        using static Shorokoo.Globals;
        namespace N;

        [Module]
        public partial class Spelled
        {
            public static Tensor<float32> Inline(Tensor<float32> x)
            {
                var @class = Ones.Init([Scalar(2L)]);
                var paren = (Ones.Init([Scalar(2L)]));
                return x * @class * paren;
            }
        }
        """;

    private static (string Names, string GeneratorDiagnostics, string Errors) Generate(
        string source, string? interceptorsNamespaces = LocalParamNameGenerator.InterceptorNamespace)
    {
        _ = typeof(Shorokoo.Modules.Layers.Linear);
        var options = CSharpParseOptions.Default.WithLanguageVersion(LanguageVersion.Preview);
        if (interceptorsNamespaces is not null)
            options = options.WithFeatures([new("InterceptorsNamespaces", interceptorsNamespaces)]);
        var compilation = CSharpCompilation.Create("LocalNames",
            [CSharpSyntaxTree.ParseText(source, options)],
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
            generated is null ? "" : string.Join(" ", Regex.Matches(generated.ToString(), "NamedByLocal\\(.*, \"(\\w+)\"\\)").Select(m => m.Groups[1].Value)),
            string.Join(" ", diagnostics.Where(d => d.Id.StartsWith("MSG")).Select(d => d.Id)),
            string.Join("\n", updated.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error)));
    }

    [Fact]
    public void TestALocalAssignedAnInitOrModelCallNamesItThroughAnInterceptor()
    {
        Assert.Equal(("gain proj inner", "", ""), Generate(Source));
        Assert.Equal(("gain proj inner", "", ""), Generate(Source, "Shorokoo.Generated"));
        Assert.Equal(("swap wrap", "", ""), Generate(CoreNamespaceSource));
        Assert.Equal(("class paren", "", ""), Generate(SpelledSource));
        Assert.Equal(("", "MSG006", ""), Generate(Source, interceptorsNamespaces: null));
    }
}

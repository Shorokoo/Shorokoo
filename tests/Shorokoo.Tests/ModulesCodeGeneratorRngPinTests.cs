using System.Linq;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Shorokoo.Tests;

/// <summary>
/// The MSG004 <c>Rng.Pin</c> suggestion builder
/// (<see cref="ModuleSourceGenerator.TryBuildRngPinSuggestion"/>), driven directly on parsed
/// module snippets. It suggests one compilable pin per scope when every RNG consumer (Model /
/// Init / feed capture, nested Iterate loop) is provably nameable, and refuses (null) when
/// anything is not — a wrong pin silently re-keys.
/// </summary>
[Trait("Domain", "Modules")]
[Trait("Purpose", "Coverage")]
public class ModulesCodeGeneratorRngPinTests
{
    private static string? Suggest(string classBody)
    {
        var tree = CSharpSyntaxTree.ParseText(classBody);
        var classDecl = tree.GetRoot().DescendantNodes().OfType<ClassDeclarationSyntax>().First();
        return ModuleSourceGenerator.TryBuildRngPinSuggestion(classDecl);
    }

    private static void AssertSuggests(string classBody, params string[] fragments)
    {
        var s = Suggest(classBody);
        Assert.NotNull(s);
        foreach (var f in fragments) Assert.Contains(f, s);
    }

    [Fact]
    public void TestPinnableSites()
    {
        // Captured Model + Init at module scope.
        AssertSuggests("""
            public partial class M
            {
                public static Tensor<float32> Inline(Tensor<float32> x)
                {
                    var a = Linear.Model(Scalar(4L), Scalar(false));
                    var w = InitSimple.Init([Scalar(2L)]);
                    return a.Call(x) + w.Reduce(ReduceKind.Sum);
                }
            }
            """, "Rng.Pin(a, w);");

        // A captured feed is a consumer exactly like an Init.
        AssertSuggests("""
            public partial class M
            {
                public static Tensor<float32> Inline(Tensor<float32> x)
                {
                    var w = InitSimple.Init([Scalar(2L)]);
                    var u = RandomUniform(x.ShapeTensor(), 0f, 1f);
                    return x + u + w.Reduce(ReduceKind.Sum);
                }
            }
            """, "Rng.Pin(w, u);");

        // Globals-qualified feed.
        AssertSuggests("""
            public partial class M
            {
                public static Tensor<float32> Inline(Tensor<float32> x)
                {
                    var u = Globals.RandomUniform(x.ShapeTensor(), 0f, 1f);
                    return x + u;
                }
            }
            """, "Rng.Pin(u);");

        // A captured feed inside a loop gets a loop-scoped pin.
        AssertSuggests("""
            public partial class M
            {
                public static Tensor<float32> Inline(Tensor<float32> x, Scalar<int64> steps)
                {
                    var acc = x;
                    foreach (var ctx in LoopAPI.Iterate(steps))
                    {
                        var w = InitSimple.Init([Scalar(2L)]);
                        var u = RandomUniform(x.ShapeTensor(), 0f, 1f);
                        acc = acc + u + w.Reduce(ReduceKind.Sum);
                        ctx.ContinueWhile(Scalar(true));
                    }
                    return acc;
                }
            }
            """, "Rng.Pin(w, u);", "inside `foreach");

        // A counted capture re-invoked inside a loop is trusted to be stream-free.
        AssertSuggests("""
            public partial class M
            {
                public static Tensor<float32> Inline(Tensor<float32> x, Scalar<int64> steps)
                {
                    var a = Linear.Model(Scalar(4L), Scalar(false));
                    var acc = a.Call(x);
                    foreach (var ctx in LoopAPI.Iterate(steps))
                    {
                        var u = RandomUniform(x.ShapeTensor(), 0f, 1f);
                        acc = a.Call(acc) + u;
                        ctx.ContinueWhile(Scalar(true));
                    }
                    return acc;
                }
            }
            """, "Rng.Pin(u);", "inside `foreach");
    }

    [Fact]
    public void TestRefusedSites()
    {
        // .Call on an uncounted receiver; an uncaptured (unnameable) feed; an opaque
        // uppercase helper call; and a body that already carries a pin.
        string[] refused =
        [
            """
            public partial class M
            {
                public static Tensor<float32> Inline(Tensor<float32> x)
                {
                    var m = MakeModel();
                    var w = InitSimple.Init([Scalar(2L)]);
                    return m.Call(x) + w.Reduce(ReduceKind.Sum);
                }
            }
            """,
            """
            public partial class M
            {
                public static Tensor<float32> Inline(Tensor<float32> x)
                {
                    var w = InitSimple.Init([Scalar(2L)]);
                    return x + RandomUniform(x.ShapeTensor(), 0f, 1f) + w.Reduce(ReduceKind.Sum);
                }
            }
            """,
            """
            public partial class M
            {
                public static Tensor<float32> Inline(Tensor<float32> x)
                {
                    var w = InitSimple.Init([Scalar(2L)]);
                    var y = DropoutMasking.Mask(x, Scalar(0.5f));
                    return y + w.Reduce(ReduceKind.Sum);
                }
            }
            """,
            """
            public partial class M
            {
                public static Tensor<float32> Inline(Tensor<float32> x)
                {
                    var u = RandomUniform(x.ShapeTensor(), 0f, 1f);
                    Rng.Pin(u);
                    return x + u;
                }
            }
            """,
        ];
        foreach (var body in refused) Assert.Null(Suggest(body));
    }
}

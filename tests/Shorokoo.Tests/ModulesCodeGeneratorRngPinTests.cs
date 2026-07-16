using System.Linq;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Shorokoo.Tests;

/// <summary>
/// Unit tests for the MSG004 <c>Rng.Pin</c> suggestion builder
/// (<see cref="ModuleSourceGenerator.TryBuildRngPinSuggestion"/>), driven directly on parsed
/// module snippets — the builder is syntax-only, so no compilation is needed. The contract:
/// suggest one compilable pin per scope when every RNG consumer (Model / Init / feed capture,
/// nested Iterate loop) is provably nameable; refuse (null) when anything is not, because a
/// wrong pin silently re-keys.
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

    [Fact]
    public void TestCapturedModelAndInitGetPositionalPin()
    {
        var s = Suggest("""
            public partial class M
            {
                public static Tensor<float32> Inline(Tensor<float32> x)
                {
                    var a = Linear.Model(Scalar(4L), Scalar(false));
                    var w = InitSimple.Init([Scalar(2L)]);
                    return a.Call(x) + w.Reduce(ReduceKind.Sum);
                }
            }
            """);
        Assert.NotNull(s);
        Assert.Contains("Rng.Pin(a, w);", s);
    }

    [Fact]
    public void TestCapturedFeedIsPinnableLikeAnInit()
    {
        // A feed is a ModelId-based consumer exactly like a param: captured in a local, it
        // takes a slot and is named in the pin — it must not disqualify the module.
        var s = Suggest("""
            public partial class M
            {
                public static Tensor<float32> Inline(Tensor<float32> x)
                {
                    var w = InitSimple.Init([Scalar(2L)]);
                    var u = RandomUniform(x.ShapeTensor(), 0f, 1f);
                    return x + u + w.Reduce(ReduceKind.Sum);
                }
            }
            """);
        Assert.NotNull(s);
        Assert.Contains("Rng.Pin(w, u);", s);
    }

    [Fact]
    public void TestGlobalsQualifiedFeedIsPinnable()
    {
        var s = Suggest("""
            public partial class M
            {
                public static Tensor<float32> Inline(Tensor<float32> x)
                {
                    var u = Globals.RandomUniform(x.ShapeTensor(), 0f, 1f);
                    return x + u;
                }
            }
            """);
        Assert.NotNull(s);
        Assert.Contains("Rng.Pin(u);", s);
    }

    [Fact]
    public void TestCapturedFeedInsideLoopGetsLoopScopedPin()
    {
        var s = Suggest("""
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
            """);
        Assert.NotNull(s);
        Assert.Contains("Rng.Pin(w, u);", s);
        Assert.Contains("inside `foreach", s);
    }

    [Fact]
    public void TestCallOnUncountedReceiverRefuses()
    {
        // m comes from an opaque bare helper call, not a counted Recv.Model(...) capture —
        // its .Call may create streams the pin would silently omit. A lowercase receiver
        // alone proves nothing, so the suggestion is withheld.
        var s = Suggest("""
            public partial class M
            {
                public static Tensor<float32> Inline(Tensor<float32> x)
                {
                    var m = MakeModel();
                    var w = InitSimple.Init([Scalar(2L)]);
                    return m.Call(x) + w.Reduce(ReduceKind.Sum);
                }
            }
            """);
        Assert.Null(s);
    }

    [Fact]
    public void TestCallOnCountedModelInsideLoopIsTrusted()
    {
        // A model captured at module scope and re-invoked inside a loop body: the counted
        // capture flows into the nested scope, so the .Call is provably stream-free and
        // both scopes get their pins.
        var s = Suggest("""
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
            """);
        Assert.NotNull(s);
        Assert.Contains("Rng.Pin(u);", s);
        Assert.Contains("inside `foreach", s);
    }

    [Fact]
    public void TestUncapturedFeedStillRefuses()
    {
        // An inline (uncaptured) feed has no name a pin could use — the whole suggestion is
        // withheld rather than emitting a pin that silently omits a consumer.
        var s = Suggest("""
            public partial class M
            {
                public static Tensor<float32> Inline(Tensor<float32> x)
                {
                    var w = InitSimple.Init([Scalar(2L)]);
                    return x + RandomUniform(x.ShapeTensor(), 0f, 1f) + w.Reduce(ReduceKind.Sum);
                }
            }
            """);
        Assert.Null(s);
    }

    [Fact]
    public void TestExistingPinSuppressesTheSuggestion()
    {
        var s = Suggest("""
            public partial class M
            {
                public static Tensor<float32> Inline(Tensor<float32> x)
                {
                    var u = RandomUniform(x.ShapeTensor(), 0f, 1f);
                    Rng.Pin(u);
                    return x + u;
                }
            }
            """);
        Assert.Null(s);
    }

    [Fact]
    public void TestOpaqueHelperCallRefuses()
    {
        // An uppercase static-helper call may create streams internally — nothing an
        // end-of-body pin could name, so the suggestion is withheld.
        var s = Suggest("""
            public partial class M
            {
                public static Tensor<float32> Inline(Tensor<float32> x)
                {
                    var w = InitSimple.Init([Scalar(2L)]);
                    var y = DropoutMasking.Mask(x, Scalar(0.5f));
                    return y + w.Reduce(ReduceKind.Sum);
                }
            }
            """);
        Assert.Null(s);
    }
}

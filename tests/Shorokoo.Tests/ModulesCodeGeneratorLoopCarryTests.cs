using System.Linq;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Shorokoo.Tests;

/// <summary>
/// The MSG005 scanner (<see cref="ModuleSourceGenerator.FindLoopCarriesAssignedFromOutside"/>),
/// driven directly on parsed module snippets. It reports the assignment the trace refuses with
/// FW023 — a carry declared by <c>LoopAPI.Init</c> in a loop body, assigned a bare name bound
/// outside that body — and nothing else, since it is a warning on every user build.
/// </summary>
[Trait("Domain", "Modules")]
[Trait("Purpose", "Coverage")]
public class ModulesCodeGeneratorLoopCarryTests
{
    private static int Found(string body)
    {
        var tree = CSharpSyntaxTree.ParseText("public partial class M { public static Scalar<int64> Inline(Scalar<int64> n) {" + body + "} }");
        var method = tree.GetRoot().DescendantNodes().OfType<MethodDeclarationSyntax>().First();
        return ModuleSourceGenerator.FindLoopCarriesAssignedFromOutside(method).Count;
    }

    [Fact]
    public void TestTheRefusedShapeIsReported()
    {
        Assert.Equal(1, Found("var c = n; foreach (var ctx in LoopAPI.Iterate(n)) { LoopAPI.Init(c); c = n; } return c;"));
        // Nested inside other C# statements, where the Iterate body is not a direct child.
        Assert.Equal(1, Found("var c = n; if (true) { foreach (var ctx in LoopAPI.Iterate(n)) { LoopAPI.Init(c); c = n; } } return c;"));
        Assert.Equal(1, Found("var c = n; for (int i = 0; i < 2; i++) { foreach (var ctx in LoopAPI.Iterate(n)) { LoopAPI.Init(c); c = n; } } return c;"));
        // Two carries, one offending.
        Assert.Equal(1, Found("var c = n; var d = n; foreach (var ctx in LoopAPI.Iterate(n)) { LoopAPI.Init(c); LoopAPI.Init(d); c = n; d = d + n; } return c;"));
        // An inner loop assigned a value declared in the outer body — outside the inner loop.
        Assert.Equal(1, Found("foreach (var o in LoopAPI.Iterate(n)) { var mid = o.IterationIndex; var c = n; foreach (var i in LoopAPI.Iterate(n)) { LoopAPI.Init(c); c = mid; } } return n;"));
        // Nested one level deeper than the body's own statement list.
        Assert.Equal(1, Found("var c = n; foreach (var ctx in LoopAPI.Iterate(n)) { LoopAPI.Init(c); { c = n; } } return c;"));
    }

    [Fact]
    public void TestShapesThatAreNotRefusedAreNotReported()
    {
        // No loop at all.
        Assert.Equal(0, Found("Scalar<int64> c; c = n; return c;"));
        // In a loop, but never declared a carry.
        Assert.Equal(0, Found("var c = n; foreach (var ctx in LoopAPI.Iterate(n)) { c = n; } return c;"));
        // The remedy, and any value the body computes.
        Assert.Equal(0, Found("var c = n; foreach (var ctx in LoopAPI.Iterate(n)) { LoopAPI.Init(c); c = LoopAPI.Carry(n); } return c;"));
        Assert.Equal(0, Found("var c = n; foreach (var ctx in LoopAPI.Iterate(n)) { LoopAPI.Init(c); c = n + n; } return c;"));
        // Assigned from a value the body itself declared.
        Assert.Equal(0, Found("var c = n; foreach (var ctx in LoopAPI.Iterate(n)) { LoopAPI.Init(c); var t = n + n; c = t; } return c;"));
        // Assigned from a name declared outside but written inside, so it holds a body value.
        Assert.Equal(0, Found("var c = n; var t = n; foreach (var ctx in LoopAPI.Iterate(n)) { LoopAPI.Init(c); t = n + n; c = t; } return c;"));
    }
}

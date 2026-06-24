using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Graph;
using System.Collections.Immutable;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage tests for the two parameter-name DSLs in
/// <c>Utils/ModelParamNameResolver.cs</c>: <see cref="ModelIdFormat"/> (the
/// ModelId-to-Name DSL driven by <see cref="ModelIdNamingScheme"/>) and
/// <see cref="SimplePatternScheme"/> / <see cref="SimplePatternNamingScheme"/>
/// (the Shorokoo-ID-to-Name DSL).
///
/// <para>
/// Both DSLs are exercised end-to-end against real checkpoints (ResNet18 / ViT /
/// RetinaNet) only during manual release validation (release check E-6). The
/// tests here drive the same DSLs through
/// <see cref="LoopLayer"/>'s real <see cref="ConcreteModelParamInfo"/> / Shorokoo-ID
/// catalog (1 outer + 3 loop-body params, with ModelIds <c>[1,1]</c> and
/// <c>[1,2,iter,1]</c>) so the advanced DSL branches (range maps, OR
/// alternatives, range-with-step, lower transform, capture wildcards, named-map
/// lookups, …) get covered without LFS data.
/// </para>
/// </summary>
[Trait("Domain", "Utils")]
[Trait("Purpose", "Coverage")]
public class ParamNameDslCoverageTests
{
    [Fact]
    public void TestModelIdFormatDslThroughLoopLayer()
    {
        // Build a real concrete architecture: LoopLayer(numOutFeatures=10, numIterations=3)
        // → 1 outer trainable param (ModelId [1,1]) + 3 loop-body params
        // (ModelIds [1,2,0,1] / [1,2,1,1] / [1,2,2,1] — val[2] is the iter index).
        var (paramInfos, _) = BuildLoopLayerParams();
        var candidates = paramInfos.ParamInfos.Select(p => p.ModelId).ToImmutableArray();

        // Build a ModelIdNamingScheme using ModelIdFormat patterns that exercise the
        // advanced DSL features on these real ModelIds:
        //   - OR alternative in match (`1|2`)
        //   - wildcard in match (`*`)
        //   - arithmetic in format (addition + subtraction)
        //   - named map lookup
        //   - inline map
        //   - range map with `start::step`, `start:end`, `start:`
        //   - recursive substitution through map outputs
        var maps = new Dictionary<string, Dictionary<int, string>>
        {
            ["kindMap"] = new() { [1] = "primary" }
        };
        var scheme = new ModelIdNamingScheme(new[]
        {
            // Outer param: ModelId [1,1] — exact-literal match.
            new ModelIdFormat(match: "[1, 1]", format: "outer.weight"),
            // Loop bodies: ModelId [1, 2, iter, 1] — OR alternative + wildcard +
            // arithmetic + named map.
            new ModelIdFormat(
                match: "[1, 1|2, *, 1]",
                format: "block{2 + 1}.kind{3|kindMap}",
                maps: maps),
        }, ModuleParamSetNamingScheme.PyTorchFrameworkId);

        // ToName on each param exercises Matches + EvaluateFormat + EvaluatePlaceholder
        // + EvaluateIndexExpression (addition) + named-map branch.
        var names = paramInfos.ParamInfos.Select(p => scheme.ToName(p.ModelId)).ToArray();
        Assert.Equal("outer.weight", names[0]);
        Assert.Equal("block1.kindprimary", names[1]); // val[2]=0 → 0+1=1
        Assert.Equal("block2.kindprimary", names[2]); // val[2]=1 → 1+1=2
        Assert.Equal("block3.kindprimary", names[3]); // val[2]=2 → 2+1=3

        // ToModelId round-trip — first call builds the reverse cache, second hits it.
        for (int i = 0; i < names.Length; i++)
            Assert.Equal(paramInfos.ParamInfos[i].ModelId, scheme.ToModelId(names[i], candidates));
        Assert.Equal(paramInfos.ParamInfos[0].ModelId, scheme.ToModelId("outer.weight", candidates));
        // Cache miss on an unknown name returns null.
        Assert.Null(scheme.ToModelId("does.not.exist", candidates));

        // Same module, different patterns — exercises:
        //   - subtraction arithmetic
        //   - inline map (with recursive substitution)
        //   - range map with `start::step` and `start:`
        //   - range map with `start:end`
        //   - literal escapes \\o (→ '{') and \\c (→ '}')
        var schemeWithRange = new ModelIdNamingScheme(new[]
        {
            new ModelIdFormat(match: "[1, 1]", format: @"\oouter\c"),  // → "{outer}"
            new ModelIdFormat(
                match: "[1, 2, *, *]",
                format: "{2|0::2,1::2|even-iter{2 - 0},odd-iter{2 - 0}}"),
        }, ModuleParamSetNamingScheme.PyTorchFrameworkId);
        Assert.Equal("{outer}", schemeWithRange.ToName(paramInfos.ParamInfos[0].ModelId));
        Assert.Equal("even-iter0", schemeWithRange.ToName(paramInfos.ParamInfos[1].ModelId)); // val[2]=0 → even
        Assert.Equal("odd-iter1", schemeWithRange.ToName(paramInfos.ParamInfos[2].ModelId));  // val[2]=1 → odd
        Assert.Equal("even-iter2", schemeWithRange.ToName(paramInfos.ParamInfos[3].ModelId)); // val[2]=2 → even

        // Inline-map + range-map with bounded `start:end`.
        var schemeWithBounded = new ModelIdNamingScheme(new[]
        {
            new ModelIdFormat(match: "[1, 1]", format: "{0|outer,inner,either}"), // val[0]=1 → "inner"
            new ModelIdFormat(
                match: "[1, 2, *, *]",
                format: "{2|0:0,1:|first,rest}"),
        }, ModuleParamSetNamingScheme.PyTorchFrameworkId);
        Assert.Equal("inner", schemeWithBounded.ToName(paramInfos.ParamInfos[0].ModelId));
        Assert.Equal("first", schemeWithBounded.ToName(paramInfos.ParamInfos[1].ModelId)); // val[2]=0
        Assert.Equal("rest", schemeWithBounded.ToName(paramInfos.ParamInfos[2].ModelId));  // val[2]=1
        Assert.Equal("rest", schemeWithBounded.ToName(paramInfos.ParamInfos[3].ModelId));  // val[2]=2

        // Negative paths — these throw on specific malformed / out-of-range inputs.
        // None of them are reachable through a well-formed real-module flow, so they
        // need targeted inputs; the LoopLayer ModelIds above provide a realistic anchor.
        var outerId = paramInfos.ParamInfos[0].ModelId;       // [1, 1]
        var loopId  = paramInfos.ParamInfos[1].ModelId;       // [1, 2, 0, 1]
        // No range matches — val[0]=1, ranges 2,3,4.
        Assert.Throws<KeyNotFoundException>(() =>
            new ModelIdFormat(match: "[1, 1]", format: "{0|2,3,4|a,b,c}").ToName(outerId));
        // Range count ≠ output count.
        Assert.Throws<System.FormatException>(() =>
            new ModelIdFormat(match: "[1, 1]", format: "{0|1,2,3|a,b}").ToName(outerId));
        // Named map missing the key.
        Assert.Throws<KeyNotFoundException>(() =>
            new ModelIdFormat(match: "[1, 1]", format: "{0|kindMap}",
                maps: new Dictionary<string, Dictionary<int, string>> { ["kindMap"] = new() { [99] = "x" } })
                .ToName(outerId));
        // Inline map index out of range — val[0]=1, map has only index 0.
        Assert.Throws<System.IndexOutOfRangeException>(() =>
            new ModelIdFormat(match: "[1, 1]", format: "{0|onlyZero}").ToName(outerId));
        // No pattern in the scheme matches → ToName throws.
        Assert.Throws<System.InvalidOperationException>(() => scheme.ToName(new ModelId(99, 99)));

        // Matches() non-match branches — each driven against a real LoopLayer ModelId
        // by a pattern shaped to demonstrate the corresponding DSL rule:
        //   - length mismatch:    pattern length ≠ ModelId length
        //   - literal mismatch:   pattern position has a literal that doesn't match
        //   - OR-set non-member:  pattern OR-alternative excludes the actual value
        //   - missing brackets:   malformed pattern
        //   - exact-literal pos:  '-1' as wildcard (synonym for '*')
        Assert.False(new ModelIdFormat("x", match: "[1, 2, 3]").Matches(outerId));         // len 3 vs 2
        Assert.False(new ModelIdFormat("x", match: "[1, 9]").Matches(outerId));            // val[1]=1, not 9
        Assert.False(new ModelIdFormat("x", match: "[1, 9, *, *]").Matches(loopId));       // val[1]=2, not 9
        Assert.False(new ModelIdFormat("x", match: "[1, 3|4|5, *, *]").Matches(loopId));   // val[1]=2 not in {3,4,5}
        Assert.False(new ModelIdFormat("x", match: "no_brackets_here").Matches(outerId));  // malformed
        Assert.True(new ModelIdFormat("x", match: "[1, -1]").Matches(outerId));            // '-1' acts as wildcard
    }

    [Fact]
    public void TestSimplePatternSchemeDslThroughLoopLayer()
    {
        // LoopLayer's real Shorokoo-ID catalog:
        //   outer: "TrainableParam#0.LoopLayer#0.InitSimple#0"
        //   iter0: "TrainableParam#0.LoopLayer#0.Loop#0:0.InitSimple#1"
        //   iter1: "TrainableParam#0.LoopLayer#0.Loop#0:1.InitSimple#1"
        //   iter2: "TrainableParam#0.LoopLayer#0.Loop#0:2.InitSimple#1"
        var (paramInfos, shorokooIdScheme) = BuildLoopLayerParams();
        var shorokooIds = paramInfos.ParamInfos.Select(p => shorokooIdScheme.ToName(p)).ToArray();

        // ---- (1) Outer + loop-body with arithmetic + named-map ----
        var kindMap = new Dictionary<string, string> { ["0"] = "weight", ["1"] = "weight" };
        var maps = new Dictionary<string, Dictionary<string, string>> { ["kindMap"] = kindMap };
        var pyTorchScheme = new SimplePatternNamingScheme(new[]
        {
            new SimplePatternScheme(
                pattern: "TrainableParam#0.LoopLayer#0.InitSimple#{p}",
                format:  "outer.{p|kindMap}",
                maps:    maps),
            new SimplePatternScheme(
                pattern: "TrainableParam#0.LoopLayer#0.Loop#0:{idx}.InitSimple#{p}",
                format:  "blocks.{idx + 1}.{p|kindMap}",
                maps:    maps),
        }, shorokooIdScheme, ModuleParamSetNamingScheme.PyTorchFrameworkId);
        var pyTorchNames = paramInfos.ParamInfos.Select(p => pyTorchScheme.ToName(p)).ToArray();
        Assert.Equal("outer.weight", pyTorchNames[0]);
        Assert.Equal("blocks.1.weight", pyTorchNames[1]);
        Assert.Equal("blocks.2.weight", pyTorchNames[2]);
        Assert.Equal("blocks.3.weight", pyTorchNames[3]);
        // ToName(ModelId) overload (routes through ModelIdNamingScheme).
        Assert.Equal("outer.weight", pyTorchScheme.ToName(paramInfos.ParamInfos[0].ModelId));
        // Round-trip — first call builds the reverse cache, second hits the fast-path.
        var candidates = paramInfos.ParamInfos.Select(p => p.ModelId).ToImmutableArray();
        for (int i = 0; i < pyTorchNames.Length; i++)
            Assert.Equal(paramInfos.ParamInfos[i].ModelId, pyTorchScheme.ToModelId(pyTorchNames[i]!, candidates));
        Assert.Equal(paramInfos.ParamInfos[0].ModelId, pyTorchScheme.ToModelId(pyTorchNames[0]!, candidates));
        Assert.Null(pyTorchScheme.ToModelId("nope", candidates));

        // ---- (2) |lower transform + capture-with-subtraction ----
        // Applied to the real outer Shorokoo ID: "...LoopLayer#0.InitSimple#0" — capture
        // the module name and lowercase it; subtract from the captured init index.
        var lowerScheme = new SimplePatternScheme(
            pattern: "TrainableParam#0.{Mod}#0.InitSimple#{p}",
            format:  "{Mod|lower}.init{p - 0}");
        Assert.Equal("looplayer.init0", lowerScheme.ToName(shorokooIds[0]));

        // ---- (3) Range constraints (start:end, start:, :end, start::step) on the
        // captured iter index. LoopLayer's real Shorokoo IDs have iter ∈ {0, 1, 2}.
        // All four forms accept iter=2 (the last iter); the 5:9 form rejects it.
        foreach (var rangeExpr in new[] { "0:9", ":9", "0:", "0::1" })
        {
            var ranged = new SimplePatternScheme(
                pattern: "TrainableParam#0.LoopLayer#0.Loop#0:{idx|" + rangeExpr + "}.InitSimple#1",
                format:  "ok.{idx}");
            Assert.True(ranged.Matches(shorokooIds[3]));        // iter=2 → in range
            Assert.Equal("ok.2", ranged.ToName(shorokooIds[3]));
        }
        var rejecting = new SimplePatternScheme(
            pattern: "TrainableParam#0.LoopLayer#0.Loop#0:{idx|5:9}.InitSimple#1",
            format:  "rejected");
        Assert.False(rejecting.Matches(shorokooIds[3]));        // iter=2 not in 5:9

        // ---- (4) Wildcard {*} eats the middle of the Shorokoo ID ----
        var wildcard = new SimplePatternScheme(
            pattern: "TrainableParam#0.{*}.InitSimple#{p}",
            format:  "wild.{p}");
        Assert.True(wildcard.Matches(shorokooIds[1]));
        Assert.Equal("wild.1", wildcard.ToName(shorokooIds[1])); // p=1 (loop-body InitSimple#1)
    }

    /// <summary>Common LoopLayer concretization used by both DSL tests.</summary>
    private static (ConcreteModelParamInfos infos, ModelIdNamingScheme shorokooIdScheme) BuildLoopLayerParams()
    {
        var model = LoopLayer.Model(Scalar(10L), Scalar(3L));
        var output = model.Call(Vector(1f, 2f, 3f, 4f, 5f));
        var arch = new FastComputationGraph([], [output]).ToConcreteArchitecture(new ModelParamList());
        var infos = arch.GetConcreteModelParamInfos();
        // 1 outer + 3 loop-body params.
        Assert.Equal(4, infos.ParamInfos.Length);
        return (infos, arch.GetShorokooIdNamingScheme());
    }
}

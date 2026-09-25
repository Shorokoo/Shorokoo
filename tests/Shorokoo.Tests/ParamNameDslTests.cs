using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Graph;
using System.Collections.Immutable;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage for the two parameter-name DSLs in <c>Utils/ModelParamNameResolver.cs</c>:
/// <see cref="ModelIdFormat"/> / <see cref="ModelIdNamingScheme"/> and
/// <see cref="SimplePatternScheme"/> / <see cref="SimplePatternNamingScheme"/>, driven
/// against <see cref="LoopLayer"/>'s real param catalog (1 outer param with ModelId
/// <c>[1,1]</c> plus 3 loop-body params with ModelIds <c>[1,2,iter,1]</c>).
/// </summary>
[Trait("Domain", "Utils")]
[Trait("Purpose", "Coverage")]
public class ParamNameDslCoverageTests
{
    private static readonly Lazy<InternalComputationGraph> LoopLayerArch = new(() =>
    {
        var model = LoopLayer.Model(Scalar(10L), Scalar(3L));
        var output = model.Call(Vector(1f, 2f, 3f, 4f, 5f));
        return new InternalComputationGraph([], [output]).ToConcreteArchitecture(new ModelParamList());
    });

    private static readonly Lazy<(ConcreteModelParamInfos Infos, ModelIdNamingScheme ShorokooIdScheme)> LoopLayerParams =
        new(() => (LoopLayerArch.Value.GetConcreteModelParamInfos(), LoopLayerArch.Value.GetShorokooIdNamingScheme()));

    private static readonly ModelIdFormat OuterFormat = new(match: "[1, 1]", format: "outer.weight");
    private static readonly ModelIdFormat BlockFormat = new(match: "[1, 2, *, 1]", format: "block{2}.weight");

    private static ModelIdNamingScheme SchemeOf(params ModelIdFormat[] formats)
        => new(formats, ModuleParamSetNamingScheme.PyTorchFrameworkId);

    [Fact]
    public void TestModelIdFormatOrAlternativeWildcardArithmeticAndNamedMap()
    {
        var infos = LoopLayerParams.Value.Infos;
        Assert.Equal(4, infos.ParamInfos.Length);
        var candidates = infos.ParamInfos.Select(p => p.ModelId).ToImmutableArray();

        var maps = new Dictionary<string, Dictionary<int, string>> { ["kindMap"] = new() { [1] = "primary" } };
        var scheme = SchemeOf(
            new ModelIdFormat(match: "[1, 1]", format: "outer.weight"),
            new ModelIdFormat(match: "[1, 1|2, *, 1]", format: "block{2 + 1}.kind{3|kindMap}", maps: maps));

        var names = infos.ParamInfos.Select(p => scheme.ToName(p.ModelId)).ToArray();
        Assert.Equal("outer.weight", names[0]);
        Assert.Equal("block1.kindprimary", names[1]);
        Assert.Equal("block2.kindprimary", names[2]);
        Assert.Equal("block3.kindprimary", names[3]);

        for (int i = 0; i < names.Length; i++)
            Assert.Equal(infos.ParamInfos[i].ModelId, scheme.ToModelId(names[i], candidates));
        Assert.Equal(infos.ParamInfos[0].ModelId, scheme.ToModelId("outer.weight", candidates));
        Assert.Null(scheme.ToModelId("does.not.exist", candidates));
    }

    [Fact]
    public void TestModelIdFormatRangeMapsInlineMapsAndLiteralEscapes()
    {
        var infos = LoopLayerParams.Value.Infos;

        var schemeWithRange = SchemeOf(
            new ModelIdFormat(match: "[1, 1]", format: @"\oouter\c"),
            new ModelIdFormat(match: "[1, 2, *, *]", format: "{2|0::2,1::2|even-iter{2 - 0},odd-iter{2 - 0}}"));
        Assert.Equal("{outer}", schemeWithRange.ToName(infos.ParamInfos[0].ModelId));
        Assert.Equal("even-iter0", schemeWithRange.ToName(infos.ParamInfos[1].ModelId));
        Assert.Equal("odd-iter1", schemeWithRange.ToName(infos.ParamInfos[2].ModelId));
        Assert.Equal("even-iter2", schemeWithRange.ToName(infos.ParamInfos[3].ModelId));

        var schemeWithBounded = SchemeOf(
            new ModelIdFormat(match: "[1, 1]", format: "{0|outer,inner,either}"),
            new ModelIdFormat(match: "[1, 2, *, *]", format: "{2|0:0,1:|first,rest}"));
        Assert.Equal("inner", schemeWithBounded.ToName(infos.ParamInfos[0].ModelId));
        Assert.Equal("first", schemeWithBounded.ToName(infos.ParamInfos[1].ModelId));
        Assert.Equal("rest", schemeWithBounded.ToName(infos.ParamInfos[2].ModelId));
        Assert.Equal("rest", schemeWithBounded.ToName(infos.ParamInfos[3].ModelId));
    }

    [Fact]
    public void TestModelIdFormatMalformedFormatsAndNonMatchingPatterns()
    {
        var infos = LoopLayerParams.Value.Infos;
        var outerId = infos.ParamInfos[0].ModelId;   // [1, 1]
        var loopId = infos.ParamInfos[1].ModelId;    // [1, 2, 0, 1]

        Assert.Throws<KeyNotFoundException>(() =>
            new ModelIdFormat(match: "[1, 1]", format: "{0|2,3,4|a,b,c}").ToName(outerId));
        Assert.Throws<System.FormatException>(() =>
            new ModelIdFormat(match: "[1, 1]", format: "{0|1,2,3|a,b}").ToName(outerId));
        Assert.Throws<KeyNotFoundException>(() =>
            new ModelIdFormat(match: "[1, 1]", format: "{0|kindMap}",
                maps: new Dictionary<string, Dictionary<int, string>> { ["kindMap"] = new() { [99] = "x" } })
                .ToName(outerId));
        Assert.Throws<System.IndexOutOfRangeException>(() =>
            new ModelIdFormat(match: "[1, 1]", format: "{0|onlyZero}").ToName(outerId));
        Assert.Throws<System.InvalidOperationException>(() =>
            SchemeOf(new ModelIdFormat(match: "[1, 1]", format: "outer.weight")).ToName(new ModelId(99, 99)));

        Assert.False(new ModelIdFormat("x", match: "[1, 2, 3]").Matches(outerId));
        Assert.False(new ModelIdFormat("x", match: "[1, 9]").Matches(outerId));
        Assert.False(new ModelIdFormat("x", match: "[1, 9, *, *]").Matches(loopId));
        Assert.False(new ModelIdFormat("x", match: "[1, 3|4|5, *, *]").Matches(loopId));
        Assert.False(new ModelIdFormat("x", match: "no_brackets_here").Matches(outerId));
        Assert.True(new ModelIdFormat("x", match: "[1, -1]").Matches(outerId));
    }

    [Fact]
    public void TestSimplePatternSchemeNamesArithmeticNamedMapAndModelIdRoundTrip()
    {
        var (infos, shorokooIdScheme) = LoopLayerParams.Value;

        var maps = new Dictionary<string, Dictionary<string, string>>
        {
            ["kindMap"] = new() { ["0"] = "weight", ["1"] = "weight" },
        };
        SimplePatternScheme[] patterns =
        [
            new SimplePatternScheme(
                pattern: "TrainableParam#0.LoopLayer#0.InitSimple#{p}",
                format:  "outer.{p|kindMap}",
                maps:    maps),
            new SimplePatternScheme(
                pattern: "TrainableParam#0.LoopLayer#0.Loop#0:{idx}.InitSimple#{p}",
                format:  "blocks.{idx + 1}.{p|kindMap}",
                maps:    maps),
        ];
        var scheme = new SimplePatternNamingScheme(
            patterns, shorokooIdScheme, ModuleParamSetNamingScheme.PyTorchFrameworkId);

        var names = infos.ParamInfos.Select(p => scheme.ToName(p)).ToArray();
        Assert.Equal("outer.weight", names[0]);
        Assert.Equal("blocks.1.weight", names[1]);
        Assert.Equal("blocks.2.weight", names[2]);
        Assert.Equal("blocks.3.weight", names[3]);
        Assert.Equal("outer.weight", scheme.ToName(infos.ParamInfos[0].ModelId));

        var candidates = infos.ParamInfos.Select(p => p.ModelId).ToImmutableArray();
        for (int i = 0; i < names.Length; i++)
            Assert.Equal(infos.ParamInfos[i].ModelId, scheme.ToModelId(names[i]!, candidates));
        Assert.Equal(infos.ParamInfos[0].ModelId, scheme.ToModelId(names[0]!, candidates));
        Assert.Null(scheme.ToModelId("nope", candidates));
    }

    [Fact]
    public void TestSimplePatternSchemeLowerTransformRangeConstraintsAndWildcard()
    {
        var (infos, shorokooIdScheme) = LoopLayerParams.Value;
        var shorokooIds = infos.ParamInfos.Select(p => shorokooIdScheme.ToName(p)).ToArray();

        var lowerScheme = new SimplePatternScheme(
            pattern: "TrainableParam#0.{Mod}#0.InitSimple#{p}",
            format:  "{Mod|lower}.init{p - 0}");
        Assert.Equal("looplayer.init0", lowerScheme.ToName(shorokooIds[0]));

        string[] acceptingRanges = ["0:9", ":9", "0:", "0::1"];
        foreach (var rangeExpr in acceptingRanges)
        {
            var ranged = new SimplePatternScheme(
                pattern: "TrainableParam#0.LoopLayer#0.Loop#0:{idx|" + rangeExpr + "}.InitSimple#1",
                format:  "ok.{idx}");
            Assert.True(ranged.Matches(shorokooIds[3]));
            Assert.Equal("ok.2", ranged.ToName(shorokooIds[3]));
        }

        var rejecting = new SimplePatternScheme(
            pattern: "TrainableParam#0.LoopLayer#0.Loop#0:{idx|5:9}.InitSimple#1",
            format:  "rejected");
        Assert.False(rejecting.Matches(shorokooIds[3]));

        var wildcard = new SimplePatternScheme(
            pattern: "TrainableParam#0.{*}.InitSimple#{p}",
            format:  "wild.{p}");
        Assert.True(wildcard.Matches(shorokooIds[1]));
        Assert.Equal("wild.1", wildcard.ToName(shorokooIds[1]));
    }

    [Fact]
    public void TestSimplePatternSchemeToModelIdSkipsCandidatesNoPatternCovers()
    {
        var (infos, shorokooIdScheme) = LoopLayerParams.Value;
        var candidates = infos.ParamInfos.Select(p => p.ModelId).ToImmutableArray();
        var scheme = new SimplePatternNamingScheme(
            [new SimplePatternScheme("TrainableParam#0.LoopLayer#0.InitSimple#{p}", "outer.weight")],
            shorokooIdScheme, ModuleParamSetNamingScheme.PyTorchFrameworkId);

        Assert.Equal(infos.ParamInfos[0].ModelId, scheme.ToModelId("outer.weight", candidates));
        Assert.Null(scheme.ToModelId("not.in.the.scheme", candidates));
    }

    [Fact]
    public void TestModelIdSchemeToModelIdSkipsCandidatesNoFormatCovers()
    {
        var infos = LoopLayerParams.Value.Infos;
        var candidates = infos.ParamInfos.Select(p => p.ModelId).ToImmutableArray();
        var scheme = SchemeOf(new ModelIdFormat(match: "[1, 1]", format: "outer.weight"));

        Assert.Equal(infos.ParamInfos[0].ModelId, scheme.ToModelId("outer.weight", candidates));
        Assert.Null(scheme.ToModelId("not.in.the.scheme", candidates));
        Assert.Equal(infos.ParamInfos[0].ModelId, scheme.ToModelId("outer.weight", candidates.Add(candidates[0])));
    }

    [Fact]
    public void TestToModelIdOverCandidatesSharingANameNamesBothModelIdsAndTheName()
    {
        var (infos, shorokooIdScheme) = LoopLayerParams.Value;
        var candidates = infos.ParamInfos.Select(p => p.ModelId).ToImmutableArray();
        var (first, second) = (infos.ParamInfos[1].ModelId, infos.ParamInfos[2].ModelId);
        var modelIdScheme = SchemeOf(
            new ModelIdFormat(match: "[1, 1]", format: "outer.weight"),
            new ModelIdFormat(match: "[1, 2, *, 1]", format: "block.weight"));
        var patternScheme = new SimplePatternNamingScheme(
            [new SimplePatternScheme("TrainableParam#0.LoopLayer#0.Loop#0:{idx}.InitSimple#{p}", "block.weight")],
            shorokooIdScheme, ModuleParamSetNamingScheme.PyTorchFrameworkId);

        Assert.True(NamesTheCollision(() => modelIdScheme.ToModelId("outer.weight", candidates), first, second, "block.weight"));
        Assert.True(NamesTheCollision(() => patternScheme.ToModelId("block.weight", candidates), first, second, "block.weight"));
    }

    [Fact]
    public void TestToModelIdResolvesAgainstTheCandidatesOfEachCallNotOfTheFirst()
    {
        var (infos, shorokooIdScheme) = LoopLayerParams.Value;
        var candidates = infos.ParamInfos.Select(p => p.ModelId).ToImmutableArray();
        var outerId = infos.ParamInfos[0].ModelId;
        var modelIdScheme = SchemeOf(
            new ModelIdFormat(match: "[1, 1]", format: "outer.weight"),
            new ModelIdFormat(match: "[1, 2, *, 1]", format: "block{2}.weight"));
        var patternScheme = new SimplePatternNamingScheme(
            [
                new SimplePatternScheme("TrainableParam#0.LoopLayer#0.InitSimple#{p}", "outer.weight"),
                new SimplePatternScheme("TrainableParam#0.LoopLayer#0.Loop#0:{idx}.InitSimple#{p}", "block{idx}.weight"),
            ],
            shorokooIdScheme, ModuleParamSetNamingScheme.PyTorchFrameworkId);

        Assert.Equal(outerId, modelIdScheme.ToModelId("outer.weight", candidates));
        Assert.Null(modelIdScheme.ToModelId("outer.weight", candidates.RemoveAt(0)));
        Assert.Equal(outerId, modelIdScheme.ToModelId("outer.weight", candidates));
        Assert.Equal(outerId, patternScheme.ToModelId("outer.weight", candidates));
        Assert.Null(patternScheme.ToModelId("outer.weight", candidates.RemoveAt(0)));
        Assert.Equal(outerId, patternScheme.ToModelId("outer.weight", candidates));
    }

    [Fact]
    public void TestToModelIdUnderConcurrentCallsResolvesEachCallAgainstItsOwnCandidates()
    {
        var infos = LoopLayerParams.Value.Infos;
        var candidates = infos.ParamInfos.Select(p => p.ModelId).ToImmutableArray();
        var withoutOuter = candidates.RemoveAt(0);
        var outerId = infos.ParamInfos[0].ModelId;
        var scheme = SchemeOf(OuterFormat, BlockFormat);
        int wrong = 0;

        Thread[] threads =
        [
            new(() => { for (int i = 0; i < 400_000; i++) if (scheme.ToModelId("outer.weight", candidates) != outerId) Interlocked.Increment(ref wrong); }),
            new(() => { for (int i = 0; i < 400_000; i++) if (scheme.ToModelId("outer.weight", withoutOuter) is not null) Interlocked.Increment(ref wrong); }),
        ];
        foreach (var t in threads) t.Start();
        foreach (var t in threads) t.Join();

        Assert.Equal(0, wrong);
    }

    [Fact]
    public void TestToConcreteModelWithASchemeLeavingAParameterUncoveredIsRefusedAsFW059NamingThatParameter()
    {
        var arch = LoopLayerArch.Value;
        var values = arch.InitializeTrainableParams(SchemeOf(OuterFormat, BlockFormat));
        var ex = Assert.Throws<ModelException>(() => arch.ToConcreteModel(values, SchemeOf(BlockFormat)));
        Assert.Equal(ErrorCodes.FW059, ex.ErrorCode);
        Assert.Contains(LoopLayerParams.Value.Infos.ParamInfos[0].ToShorokooIdString(), ex.Message);
    }

    private static bool NamesTheCollision(Func<ModelId?> toModelId, ModelId first, ModelId second, string name)
        => Record.Exception(() => toModelId()) is InvalidOperationException ex
            && ex.Message.Contains(string.Join(",", first.Vals))
            && ex.Message.Contains(string.Join(",", second.Vals))
            && ex.Message.Contains(name);
}

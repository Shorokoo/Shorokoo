using System;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace Shorokoo.Tests;

/// <summary>
/// The graph-only-RNG guard (#136): <b>no production code may run the RNG algorithm
/// host-side</b>. Every RNG operation — key splits/folds and draws alike — is computed by the
/// in-graph tagged functions; a host consumer that needs a concrete key resolves it by
/// <em>executing</em> that derivation (<c>RngKeyResolver</c>), never by recomputing it.
///
/// <para>This is the <b>precondition</b> for letting a custom algorithm own its <c>split</c>
/// (issue #122): a host path that recomputed the key tree assuming built-in Threefry would
/// silently diverge from a custom algorithm the moment the two differ. (The key tree is still
/// algorithm-independent today — <c>RngAlgorithms.GetFunction</c> pins <c>split</c> to the
/// default — so removing the host copy does not by itself make split customizable; it removes
/// the obstacle.) The C# <see cref="Core.Rng.Threefry2x32"/> generator survives only as a
/// <b>test oracle</b> (see <see cref="RngTestOracle"/>). The oracle is an independent
/// transcription of the fold rather than a call into the graph, so value assertions do not
/// merely compare the graph against itself — though, being transcribed from the removed host
/// fold, it shares that fold's <em>structure</em> and so cannot catch a pre-existing
/// structural error.</para>
///
/// <para>Source-level guard rather than a behavioural one: a reintroduced host fold would
/// otherwise pass every value test (it computes the same numbers for the built-ins) and only
/// break once a custom algorithm exists.</para>
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngNoHostRngTests
{
    private static string ProductionSourceRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !Directory.Exists(Path.Combine(dir.FullName, "src", "Shorokoo")))
            dir = dir.Parent;
        Assert.NotNull(dir);
        return Path.Combine(dir!.FullName, "src", "Shorokoo");
    }

    // Host-RNG use, matched against comment/string-stripped source (so a split across lines is
    // still caught). `Rounds`/`Rounds13` are round-count constants parameterising the in-graph
    // function builders, not computation, so they are deliberately exempt — but only as an exact
    // member name, so a `RoundsFold(...)` cannot hide behind the prefix.
    private static readonly Regex[] HostRngUse =
    [
        // Threefry2x32.<member> — including fully qualified and across newlines.
        new(@"Threefry2x32\s*\.\s*(?!Rounds13\b|Rounds\b)\w+", RegexOptions.Compiled),
        // `using static ...Threefry2x32;` — makes its members callable bare, evading the above.
        new(@"using\s+static\s+[\w\.]*\bThreefry2x32\s*;", RegexOptions.Compiled),
        // `using Alias = ...Threefry2x32;` — same, via an alias.
        new(@"using\s+\w+\s*=\s*[\w\.]*\bThreefry2x32\s*;", RegexOptions.Compiled),
    ];

    /// <summary>Strips comments and string literals so prose mentioning the generator does not
    /// trip the guard, and so a call split across lines is still seen as one text.</summary>
    private static string StripCommentsAndStrings(string source)
    {
        source = Regex.Replace(source, @"/\*.*?\*/", " ", RegexOptions.Singleline);   // block comments
        source = Regex.Replace(source, @"//[^\n]*", " ");                             // line + doc comments
        source = Regex.Replace(source, @"@""(?:[^""]|"""")*""", " ");                 // verbatim strings
        source = Regex.Replace(source, @"""(?:\\.|[^""\\])*""", " ");                 // regular strings
        return source;
    }

    private static string[] ProductionSourceFiles() => Directory
        .EnumerateFiles(ProductionSourceRoot(), "*.cs", SearchOption.AllDirectories)
        .Where(f => !f.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}") &&
                    !f.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}"))
        .ToArray();

    [Fact]
    public void TestNoProductionCodeRunsTheRngAlgorithmHostSide()
    {
        var files = ProductionSourceFiles();

        // A guard that silently scans nothing is worse than no guard: assert it really swept the
        // product before trusting a clean result.
        Assert.True(files.Length > 100,
            $"Guard swept only {files.Length} production files — the source root " +
            $"('{ProductionSourceRoot()}') looks wrong, so a clean result would be meaningless.");

        var offenders = files
            // Threefry2x32.cs DEFINES the generator; defining it is not calling it. Everything
            // else in that file is still swept (a host fold hidden there is caught below).
            .SelectMany(f => HostRngUse
                .SelectMany(rx => rx.Matches(StripCommentsAndStrings(File.ReadAllText(f)))
                    .Where(_ => Path.GetFileName(f) != "Threefry2x32.cs" || !rx.ToString().StartsWith("Threefry2x32"))
                    .Select(m => $"{Path.GetFileName(f)}: {m.Value.Trim()}")))
            .Distinct()
            .ToArray();

        Assert.True(offenders.Length == 0,
            "Production code must not run the RNG algorithm host-side (#136) — the key tree and " +
            "draws are computed in-graph, and concrete keys are resolved by executing that " +
            "derivation (RngKeyResolver). Offending uses:\n  " + string.Join("\n  ", offenders));
    }

    [Fact]
    public void TestGuardDetectsEveryKnownHostRngEvasion()
    {
        // Positive control: the guard above is only meaningful if its patterns still fire. Pin
        // each evasion the line-based first version missed, so a future edit that weakens the
        // patterns (or renames the generator out from under them) fails HERE rather than going
        // quietly green.
        string[] mustFlag =
        [
            "var k = Threefry2x32.Bijection(a, b, c, d);",
            "var k = Shorokoo.Core.Rng.Threefry2x32.Bijection(a, b, c, d);",
            "var k = Threefry2x32\n    .Bijection(a, b, c, d);",         // split across lines
            "using static Shorokoo.Core.Rng.Threefry2x32;",
            "using TF = Shorokoo.Core.Rng.Threefry2x32;",
            "var k = Threefry2x32.RoundsFold(key, i);",                  // not the exempt constants
        ];
        foreach (var sample in mustFlag)
            Assert.True(HostRngUse.Any(rx => rx.IsMatch(StripCommentsAndStrings(sample))),
                $"guard no longer detects host RNG use: {sample}");

        // Negative control: the exempt round-count constants and prose must NOT trip it.
        string[] mustNotFlag =
        [
            "int rounds = Threefry2x32.Rounds;",
            "int rounds = Threefry2x32.Rounds13;",
            "// see Threefry2x32.Bijection for the host oracle",
            "/* Threefry2x32.Bijection */",
            "var name = \"Threefry2x32.Bijection\";",
        ];
        foreach (var sample in mustNotFlag)
            Assert.False(HostRngUse.Any(rx => rx.IsMatch(StripCommentsAndStrings(sample))),
                $"guard falsely flags a benign use: {sample}");
    }

    [Fact]
    public void TestProductionExposesNoHostKeyFoldMethod()
    {
        // The test oracle supplies FoldInitKey/FoldRunKey as EXTENSION methods. C# prefers an
        // instance method over an extension, so if production ever re-added a member of that
        // name, every oracle call site would silently rebind to it and the assertions would
        // compare production against itself. Fail loudly instead.
        var flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Static |
                    System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic;
        string[] forbidden = ["FoldKey", "FoldInitKey", "FoldRunKey"];
        foreach (var name in forbidden)
        {
            Assert.Null(typeof(RngConfig).GetMethod(name, flags));
            Assert.Null(typeof(Core.Rng.RngRuntimeIdentity).GetMethod(name, flags));
        }
    }
}

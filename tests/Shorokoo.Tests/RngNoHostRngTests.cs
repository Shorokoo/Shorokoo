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
/// <para>This is what allows a custom algorithm (issue #122) to own its own <c>split</c>: if
/// any host path recomputed the key tree assuming built-in Threefry, it would silently diverge
/// from a custom algorithm the moment the two differ. The C# <see cref="Core.Rng.Threefry2x32"/>
/// generator therefore survives only as a <b>test oracle</b> (see <see cref="RngTestOracle"/>),
/// which is exactly why these assertions are meaningful — the oracle does not share an
/// implementation with the graph it validates.</para>
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

    [Fact]
    public void TestNoProductionCodeRunsTheRngAlgorithmHostSide()
    {
        // The host generator's *entry points* (the ones that actually compute RNG). The
        // `Rounds`/`Rounds13` round-count constants are not computation — they parameterise the
        // in-graph function builders — so they are deliberately not matched here.
        var hostRngCall = new Regex(@"Threefry2x32\s*\.\s*(?!Rounds)\w+", RegexOptions.Compiled);

        var offenders = Directory
            .EnumerateFiles(ProductionSourceRoot(), "*.cs", SearchOption.AllDirectories)
            .Where(f => !f.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}") &&
                        !f.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}"))
            // Threefry2x32.cs itself only DEFINES the generator; defining it is not calling it.
            .Where(f => Path.GetFileName(f) != "Threefry2x32.cs")
            .SelectMany(f => File.ReadAllLines(f)
                .Select((line, i) => (file: Path.GetFileName(f), no: i + 1, line))
                // Ignore doc comments: an XML <see cref> naming the generator is prose.
                .Where(x => !x.line.TrimStart().StartsWith("//") && !x.line.TrimStart().StartsWith("///"))
                .Where(x => hostRngCall.IsMatch(x.line)))
            .Select(x => $"{x.file}:{x.no}: {x.line.Trim()}")
            .ToArray();

        Assert.True(offenders.Length == 0,
            "Production code must not run the RNG algorithm host-side (#136) — the key tree and " +
            "draws are computed in-graph, and concrete keys are resolved by executing that " +
            "derivation (RngKeyResolver). Offending lines:\n  " + string.Join("\n  ", offenders));
    }
}

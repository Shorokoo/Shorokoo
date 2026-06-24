using System.Diagnostics;
using System.Text;

namespace Shorokoo.Tests;

/// <summary>
/// Code-pinned hygiene gate: the shipping product must build warning-free.
///
/// <para>
/// This guards the regression caught by <c>release-test-plan</c> <c>H-2</c>,
/// where v0.1.8-dev shipped with 7 compiler warnings (4× CS8321 dead local
/// functions in the module source generator, 3× CS1573 missing
/// <c>&lt;param&gt;</c> docs on a public API). Those
/// warnings were invisible to the automated suite — only manual release
/// validation caught them. This test makes any such regression fail the
/// <c>Purpose=Coverage</c> suite instead.
/// </para>
///
/// <para>
/// It shells out to a fresh <c>dotnet build -c Release</c> of
/// <c>src/Shorokoo.Modules/Shorokoo.Modules.csproj</c> with <c>-warnaserror</c>.
/// Building Modules transitively recompiles the two other projects that carry
/// real product C# — <c>Shorokoo</c> (Core) and the <c>Shorokoo.CodeGen</c>
/// analyzer — so a single build covers every project where the 7 warnings lived
/// and where new warnings would realistically arise. The backend wrappers and
/// the deps-only meta-package carry no compiled C#, so they are out of scope; if
/// that ever changes, add their csprojs to <see cref="ProductProjects"/>.
/// </para>
///
/// <para>
/// The build is redirected to an isolated temp output directory with
/// <c>--no-incremental</c>, so it forces a real recompile (warnings re-emit) and
/// never overwrites the loaded test-host assemblies (which would lock files on
/// Windows).
/// </para>
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class BuildWarningsGateTests
{
    /// <summary>
    /// Product projects whose compilation must be warning-free, relative to the repo
    /// root. Building Modules pulls in Core + the CodeGen analyzer, so this one entry
    /// covers all three code-bearing product projects.
    /// </summary>
    private static readonly string[] ProductProjects =
    [
        Path.Combine("src", "Shorokoo.Modules", "Shorokoo.Modules.csproj"),
    ];

    private static readonly TimeSpan BuildTimeout = TimeSpan.FromMinutes(5);

    [Fact]
    public void ShippingProductBuildsWarningFree()
    {
        var repoRoot = FindRepoRoot();
        var tempOut = Path.Combine(Path.GetTempPath(), "shorokoo-warngate-" + Guid.NewGuid().ToString("N"));

        try
        {
            foreach (var relProject in ProductProjects)
            {
                var project = Path.Combine(repoRoot, relProject);
                Assert.True(File.Exists(project), $"product project not found at {project}");

                var (exitCode, output) = RunBuild(project, tempOut);

                Assert.True(exitCode == 0,
                    $"`dotnet build -c Release -warnaserror` of {relProject} failed (exit {exitCode}) — " +
                    $"the shipping build is not warning-clean. Offending diagnostics:\n{ExtractDiagnostics(output)}");
            }
        }
        finally
        {
            TryDeleteDirectory(tempOut);
        }
    }

    private static (int ExitCode, string Output) RunBuild(string projectPath, string outputDir)
    {
        var psi = new ProcessStartInfo
        {
            FileName = "dotnet",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            WorkingDirectory = Path.GetDirectoryName(projectPath)!,
        };
        // Treat warnings as errors, force a real recompile, and isolate the output so we
        // never touch the assemblies the test host has loaded.
        psi.ArgumentList.Add("build");
        psi.ArgumentList.Add(projectPath);
        psi.ArgumentList.Add("-c");
        psi.ArgumentList.Add("Release");
        psi.ArgumentList.Add("-warnaserror");
        psi.ArgumentList.Add("--no-incremental");
        psi.ArgumentList.Add("-o");
        psi.ArgumentList.Add(outputDir);
        psi.ArgumentList.Add("-nodereuse:false");
        psi.ArgumentList.Add("-clp:NoSummary");

        using var process = new Process { StartInfo = psi };
        var sb = new StringBuilder();
        process.OutputDataReceived += (_, e) => { if (e.Data is not null) lock (sb) sb.AppendLine(e.Data); };
        process.ErrorDataReceived += (_, e) => { if (e.Data is not null) lock (sb) sb.AppendLine(e.Data); };

        process.Start();
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();

        if (!process.WaitForExit((int)BuildTimeout.TotalMilliseconds))
        {
            try { process.Kill(entireProcessTree: true); } catch { /* best effort */ }
            Assert.Fail($"`dotnet build` of {projectPath} did not finish within {BuildTimeout.TotalMinutes:F0} min.");
        }
        process.WaitForExit(); // flush async readers

        lock (sb) return (process.ExitCode, sb.ToString());
    }

    /// <summary>Keeps only the warning/error lines so the failure message is readable.</summary>
    private static string ExtractDiagnostics(string buildOutput)
    {
        var lines = buildOutput
            .Split('\n')
            .Select(l => l.TrimEnd('\r'))
            .Where(l => l.Contains(": warning ", StringComparison.Ordinal)
                     || l.Contains(": error ", StringComparison.Ordinal))
            .Distinct()
            .ToArray();
        return lines.Length > 0 ? string.Join('\n', lines) : buildOutput;
    }

    /// <summary>Walks up from the test output directory to the repo root (the dir holding Shorokoo.sln).</summary>
    private static string FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            if (File.Exists(Path.Combine(dir.FullName, "Shorokoo.sln")))
                return dir.FullName;
            dir = dir.Parent;
        }
        throw new InvalidOperationException(
            $"could not locate the repo root (Shorokoo.sln) above {AppContext.BaseDirectory}.");
    }

    private static void TryDeleteDirectory(string path)
    {
        try { if (Directory.Exists(path)) Directory.Delete(path, recursive: true); }
        catch { /* best effort cleanup */ }
    }
}

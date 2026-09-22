using System.Diagnostics;
using System.Text;

namespace Shorokoo.Tests;

/// <summary>
/// Code-pinned hygiene gate: the shipping product must build warning-free.
///
/// <para>
/// This guards a regression caught at release time,
/// where v0.1.8-dev shipped with 7 compiler warnings (4× CS8321 dead local
/// functions in the module source generator, 3× CS1573 missing
/// <c>&lt;param&gt;</c> docs on a public API). Those
/// warnings were invisible to the automated suite — only manual release
/// validation caught them. This test makes any such regression fail the
/// <c>Purpose=Coverage</c> suite instead.
/// </para>
///
/// <para>
/// It shells out to a fresh <c>dotnet build -c Release</c> with <c>-warnaserror</c>
/// for each project in <see cref="ProductProjects"/>, which between them recompile
/// every line of product C# that ships.
/// </para>
///
/// <para>
/// That list used to be <c>Shorokoo.Modules</c> alone, on the reasoning that
/// building it pulls in <c>Shorokoo</c> (Core) and the <c>Shorokoo.CodeGen</c>
/// analyzer, and that the backends carried no compiled C#. The second half stopped
/// being true: <c>Shorokoo.OnnxRuntime</c> holds <c>OrtBackend</c>,
/// <c>OrtSession</c>, <c>OrtTensorValue</c> and the glue around them, and
/// each platform package holds its <c>[ShorokooBackend]</c> manifest and a factory
/// of its own. A <c>CS1734</c> lived there unnoticed for exactly as long as the gate
/// looked away, which is the argument for naming projects here rather than relying on
/// one of them to reach the rest. The deps-only meta-package still ships no assembly,
/// so it stays out.
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
[Trait("Purpose", "Gate")]
public class BuildWarningsGateTests
{
    /// <summary>
    /// Product projects whose compilation must be warning-free, relative to the repo
    /// root. Modules pulls in Core and the CodeGen analyzer; each platform backend
    /// pulls in the ONNX Runtime glue, which is named anyway because it carries the
    /// most of it. The four platform packages are all listed rather than one standing
    /// for the rest: each compiles a manifest and a factory that only it has, and all
    /// four build on either operating system, since only their natives are
    /// platform-bound.
    /// </summary>
    private static readonly string[] ProductProjects =
    [
        Path.Combine("src", "Shorokoo.Modules", "Shorokoo.Modules.csproj"),
        Path.Combine("src", "Backend", "OnnxRuntime", "Shorokoo.OnnxRuntime", "Shorokoo.OnnxRuntime.csproj"),
        Path.Combine("src", "Backend", "OnnxRuntime", "Shorokoo.WinCPU", "Shorokoo.WinCPU.csproj"),
        Path.Combine("src", "Backend", "OnnxRuntime", "Shorokoo.WinGPU", "Shorokoo.WinGPU.csproj"),
        Path.Combine("src", "Backend", "OnnxRuntime", "Shorokoo.LinuxCPU", "Shorokoo.LinuxCPU.csproj"),
        Path.Combine("src", "Backend", "OnnxRuntime", "Shorokoo.LinuxGPU", "Shorokoo.LinuxGPU.csproj"),
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
                Assert.True(File.Exists(project));

                var (exitCode, output) = RunBuild(project, tempOut);

                Assert.Equal(0, exitCode);
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

        var finished = process.WaitForExit((int)BuildTimeout.TotalMilliseconds);
        if (!finished)
            try { process.Kill(entireProcessTree: true); } catch { /* best effort */ }
        Assert.True(finished);
        process.WaitForExit(); // flush async readers

        lock (sb) return (process.ExitCode, sb.ToString());
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

using System.Diagnostics;
using System.Text.RegularExpressions;

// Interactive release helper for the Shorokoo NuGet packages.
//
// The app asks for the few inputs only a human can provide (version, API key,
// go/no-go) and automates everything else: version stamping, test run, packing,
// validation, and pushing to nuget.org. See README.md next to this file for the
// one-time setup steps (nuget.org account, API key) and the manual equivalent
// of every step.
//
// Usage:  dotnet run --project tools/NuGetRelease            (interactive)
//         dotnet run --project tools/NuGetRelease -- --dry-run

const string PushSource = "https://api.nuget.org/v3/index.json";
string[] packageIds =
[
    "Shorokoo",
    "Shorokoo.Modules",
    "Shorokoo.CodeGen",
    "Shorokoo.OnnxRuntime",
    "Shorokoo.LinuxCPU",
    "Shorokoo.LinuxGPU",
    "Shorokoo.WinCPU",
    "Shorokoo.WinGPU",
];

bool dryRun = args.Contains("--dry-run");

var repoRoot = FindRepoRoot();
Console.WriteLine($"Repository root: {repoRoot}");
if (dryRun) Console.WriteLine("DRY RUN: everything except the final push.");
Console.WriteLine();

// ---------------------------------------------------------------- 1. Version
var propsPath = Path.Combine(repoRoot, "Directory.Build.props");
var versionCsPath = Path.Combine(repoRoot, "src", "Shorokoo", "Version.cs");
var currentVersion = ExtractSingle(File.ReadAllText(propsPath), @"<Version>([^<]+)</Version>", "Directory.Build.props <Version>");

Console.WriteLine($"Current version: {currentVersion}");
var newVersion = Prompt($"Version to release [{currentVersion}]: ");
if (string.IsNullOrWhiteSpace(newVersion)) newVersion = currentVersion;
if (!Regex.IsMatch(newVersion, @"^\d+\.\d+\.\d+(-[0-9A-Za-z\.\-]+)?$"))
    Fail($"'{newVersion}' is not a valid SemVer version (expected e.g. 0.2.0 or 1.0.0-preview.1).");

if (newVersion != currentVersion)
{
    StampVersion(propsPath, versionCsPath, currentVersion, newVersion);
    Console.WriteLine($"Stamped {newVersion} into Directory.Build.props and src/Shorokoo/Version.cs.");
    Console.WriteLine("NOTE: commit this version bump after the release succeeds.");
}
Console.WriteLine();

// ------------------------------------------------------------------ 2. Tests
if (AskYesNo("Run the Quick+Standard test suite first? [Y/n]: ", defaultYes: true))
{
    Step("dotnet", "test tests/Shorokoo.Tests/Shorokoo.Tests.csproj --filter \"Purpose=Coverage\" -c Release --logger \"console;verbosity=minimal\"",
        repoRoot, "Test run failed — aborting the release.");
}

// ------------------------------------------------------------------- 3. Pack
var outDir = Path.Combine(repoRoot, "artifacts", "nupkgs");
if (Directory.Exists(outDir)) Directory.Delete(outDir, recursive: true);
Step("dotnet", $"pack Shorokoo.sln -c Release -o \"{outDir}\"", repoRoot, "dotnet pack failed.");

// -------------------------------------------------------------- 4. Validate
Console.WriteLine();
Console.WriteLine("Packed packages:");
var missing = new List<string>();
foreach (var id in packageIds)
{
    var nupkg = Path.Combine(outDir, $"{id}.{newVersion}.nupkg");
    if (File.Exists(nupkg))
        Console.WriteLine($"  OK  {Path.GetFileName(nupkg)}  ({new FileInfo(nupkg).Length / 1024} KB)");
    else
        missing.Add(id);
}
if (missing.Count > 0)
    Fail($"Missing expected package(s): {string.Join(", ", missing)}");

// ------------------------------------------------------------------ 5. Push
Console.WriteLine();
if (dryRun)
{
    Console.WriteLine($"Dry run complete. Packages are in {outDir}.");
    Console.WriteLine("Re-run without --dry-run to push to nuget.org.");
    return;
}

if (!AskYesNo($"Push {packageIds.Length} packages (+symbols) v{newVersion} to nuget.org? [y/N]: ", defaultYes: false))
{
    Console.WriteLine($"Not pushing. Packages remain in {outDir}.");
    return;
}

var apiKey = Environment.GetEnvironmentVariable("NUGET_API_KEY");
if (string.IsNullOrWhiteSpace(apiKey))
{
    apiKey = Prompt("nuget.org API key (or set NUGET_API_KEY): ", secret: true);
    if (string.IsNullOrWhiteSpace(apiKey)) Fail("An API key is required to push.");
}

// Push in dependency order so consumers never see a package whose dependencies
// don't resolve yet. --skip-duplicate makes the whole run safely re-runnable.
foreach (var id in packageIds)
{
    var nupkg = Path.Combine(outDir, $"{id}.{newVersion}.nupkg");
    Console.WriteLine($"Pushing {Path.GetFileName(nupkg)} ...");
    Step("dotnet", $"nuget push \"{nupkg}\" --api-key {apiKey} --source {PushSource} --skip-duplicate",
        repoRoot, $"Push failed for {id}. Fix the issue and re-run; already-pushed packages are skipped.");
}

Console.WriteLine();
Console.WriteLine($"Done. {packageIds.Length} packages v{newVersion} pushed to nuget.org.");
Console.WriteLine("Remaining manual steps:");
Console.WriteLine("  1. Commit + tag the version bump:  git tag v" + newVersion);
Console.WriteLine("  2. Verify the packages render correctly on nuget.org (readme, license, deps).");
Console.WriteLine("  3. Create a GitHub release for the tag.");

// ------------------------------------------------------------------ helpers

static string FindRepoRoot()
{
    var dir = new DirectoryInfo(AppContext.BaseDirectory);
    while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "Shorokoo.sln")))
        dir = dir.Parent;
    if (dir is null)
        Fail("Could not locate Shorokoo.sln above the executable. Run via 'dotnet run --project tools/NuGetRelease' from inside the repository.");
    return dir!.FullName;
}

static string ExtractSingle(string text, string pattern, string what)
{
    var matches = Regex.Matches(text, pattern);
    if (matches.Count != 1)
        Fail($"Expected exactly one match for {what}, found {matches.Count}.");
    return matches[0].Groups[1].Value;
}

static void StampVersion(string propsPath, string versionCsPath, string oldVersion, string newVersion)
{
    File.WriteAllText(propsPath,
        File.ReadAllText(propsPath).Replace($"<Version>{oldVersion}</Version>", $"<Version>{newVersion}</Version>"));

    var versionCs = File.ReadAllText(versionCsPath);
    var parts = newVersion.Split('-')[0].Split('.');
    versionCs = Regex.Replace(versionCs, @"new Version\(\d+,\s*\d+,\s*\d+\)", $"new Version({parts[0]}, {parts[1]}, {parts[2]})");
    versionCs = Regex.Replace(versionCs, "VersionString = \"[^\"]+\"", $"VersionString = \"{newVersion}\"");
    File.WriteAllText(versionCsPath, versionCs);
}

static string Prompt(string message, bool secret = false)
{
    Console.Write(message);
    if (!secret) return Console.ReadLine()?.Trim() ?? "";

    // Don't echo secrets to the terminal.
    var buffer = new Stack<char>();
    while (true)
    {
        var key = Console.ReadKey(intercept: true);
        if (key.Key == ConsoleKey.Enter) { Console.WriteLine(); break; }
        if (key.Key == ConsoleKey.Backspace) { if (buffer.Count > 0) buffer.Pop(); continue; }
        if (!char.IsControl(key.KeyChar)) buffer.Push(key.KeyChar);
    }
    return new string(buffer.Reverse().ToArray()).Trim();
}

static bool AskYesNo(string message, bool defaultYes)
{
    var answer = Prompt(message).ToLowerInvariant();
    if (answer.Length == 0) return defaultYes;
    return answer is "y" or "yes";
}

static void Step(string fileName, string arguments, string workingDirectory, string failureMessage)
{
    Console.WriteLine($"> {fileName} {Redact(arguments)}");
    var psi = new ProcessStartInfo(fileName, arguments) { WorkingDirectory = workingDirectory };
    using var process = Process.Start(psi)!;
    process.WaitForExit();
    if (process.ExitCode != 0) Fail(failureMessage);
}

static string Redact(string arguments)
    => Regex.Replace(arguments, @"(--api-key)\s+\S+", "$1 ********");

static void Fail(string message)
{
    Console.Error.WriteLine($"ERROR: {message}");
    Environment.Exit(1);
}

using System.Runtime.CompilerServices;

namespace Shorokoo.Tests.Utils;

/// <summary>
/// Resolves the on-disk <c>tests/test-data</c> directory so tests can locate
/// fixtures independent of the build-output layout. The primary strategy walks
/// up from the running assembly's location until it finds <c>tests/test-data</c>;
/// it falls back to a path derived from this source file when that fails.
/// </summary>
public static class TestDataPaths
{
    /// <summary>Absolute path to the repo's <c>tests/test-data</c> directory.</summary>
    public static readonly string Root = ResolveRoot();

    /// <summary>Combine <see cref="Root"/> with the given relative parts.</summary>
    public static string Of(params string[] relativeParts)
        => Path.Combine(new[] { Root }.Concat(relativeParts).ToArray());

    private static string ResolveRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            var candidate = Path.Combine(dir.FullName, "tests", "test-data");
            if (Directory.Exists(candidate))
                return candidate;
        }

        // Fallback: this file lives at tests/Shorokoo.Tests/Utils/TestDataPaths.cs,
        // so tests/test-data is two directories up.
        return Path.GetFullPath(Path.Combine(SourceFileDir(), "..", "..", "test-data"));
    }

    private static string SourceFileDir([CallerFilePath] string path = "")
        => Path.GetDirectoryName(path)!;
}

namespace Shorokoo.Tests.Utils;

/// <summary>
/// Marks a test that needs a model checkpoint a developer downloads manually.
/// These files are intentionally not committed (see <c>tests/test-data/README.md</c>
/// for the download commands).
///
/// If the checkpoint is missing — or implausibly small, e.g. a truncated download —
/// the test is <see cref="FactAttribute.Skip"/>ped with a message that includes the
/// download command, rather than failing.
///
/// Tests using this attribute are tagged <c>[Trait("Purpose", "Manual")]</c> so the
/// automated Coverage suite (<c>Purpose=Coverage</c>) never selects them; they run
/// only when a developer opts in after downloading the data.
/// </summary>
public sealed class RequiresDownloadedModelFactAttribute : FactAttribute
{
    /// <param name="relativePath">
    /// Path to the checkpoint relative to <c>tests/test-data</c>, using '/' separators.
    /// </param>
    /// <param name="downloadHint">
    /// A copy-pasteable command (or short instruction) shown when the file is absent.
    /// </param>
    public RequiresDownloadedModelFactAttribute(string relativePath, string downloadHint)
    {
        Skip = SkipReason(downloadHint, [relativePath]);
    }

    /// <param name="downloadHint">
    /// A copy-pasteable command (or short instruction) shown when a file is absent.
    /// </param>
    /// <param name="relativePaths">
    /// One or more paths relative to <c>tests/test-data</c> (using '/' separators) that must
    /// all be present; the test is skipped if any is missing or implausibly small.
    /// </param>
    public RequiresDownloadedModelFactAttribute(string downloadHint, params string[] relativePaths)
    {
        Skip = SkipReason(downloadHint, relativePaths);
    }

    private static string? SkipReason(string downloadHint, string[] relativePaths)
    {
        foreach (var relativePath in relativePaths)
        {
            var full = TestDataPaths.Of(relativePath.Split('/'));

            if (!File.Exists(full))
                return $"Manual download required: '{relativePath}' not found under tests/test-data. " +
                       $"{downloadHint} (see tests/test-data/README.md).";

            if (new FileInfo(full).Length < 4096)
                return $"Manual download incomplete: '{relativePath}' is only {new FileInfo(full).Length} bytes " +
                       $"(expected a real checkpoint). {downloadHint} (see tests/test-data/README.md).";
        }
        return null;
    }
}

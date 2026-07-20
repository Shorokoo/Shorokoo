namespace Shorokoo.Core.Utils
{
    /// <summary>
    /// Atomic write-and-commit for files: content is staged under a
    /// <c>.tmp-</c>-prefixed sibling name in the target's directory (same filesystem, so the
    /// commit rename is atomic), flushed to disk, then renamed onto the target. On any failure
    /// the staged copy is deleted and the previous target is left untouched — a crash mid-save
    /// never corrupts the existing file. This is the designated write path for every
    /// save/export API (see <see cref="TrainingCheckpoint.Save"/>); it carries no assumptions
    /// about what is being written.
    /// </summary>
    /// <remarks>
    /// A successful commit also removes stale <c>.tmp-</c> siblings left behind by earlier
    /// failed saves of the same target. This cleanup never fails the save: the new content is
    /// already committed, so a cleanup failure is surfaced only through the optional
    /// <c>onWarning</c> callback (silent if none is given).
    /// </remarks>
    internal static class AtomicFileWriter
    {
        /// <summary>
        /// Prefix of staged (not yet committed) sibling names. A crashed save can leave one
        /// beside the committed target until a later successful save sweeps it, so anything
        /// enumerating the target's directory must ignore entries for which
        /// <see cref="IsTempName"/> is true.
        /// </summary>
        internal const string TempPrefix = ".tmp-";

        /// <summary>
        /// Test hook: invoked with the staged temp path after the content is written and
        /// flushed but before the commit rename. Throwing here simulates a crash in the
        /// commit window. Hooks must filter on the path they receive (tests can run in
        /// parallel) and be reset in a <c>finally</c>.
        /// </summary>
        internal static Action<string>? CommitFaultInjection;

        /// <summary>True if <paramref name="name"/> is a staged (uncommitted) sibling name.</summary>
        internal static bool IsTempName(string name) =>
            name.StartsWith(TempPrefix, StringComparison.Ordinal);

        /// <summary>
        /// Atomically writes a single file: <paramref name="writeContent"/> streams the full
        /// content into a staged temp file next to <paramref name="targetPath"/>, which is
        /// fsynced and then renamed onto the target (replacing any previous file). The
        /// target's directory must already exist — the temp file lives there precisely so the
        /// rename cannot cross filesystems. The callback must leave the stream open: the
        /// writer flushes it to disk (and disposes it) itself.
        /// </summary>
        internal static void WriteFile(
            string targetPath,
            Action<Stream> writeContent,
            Action<string>? onWarning = null)
        {
            if (writeContent is null) throw new ArgumentNullException(nameof(writeContent));
            var (directory, name, fullTarget) = ValidateTarget(targetPath);

            string tempPath = Path.Combine(directory, StageName(name));
            try
            {
                using (var fs = new FileStream(tempPath, FileMode.CreateNew, FileAccess.Write, FileShare.None))
                {
                    writeContent(fs);
                    fs.Flush(flushToDisk: true);
                }
                CommitFaultInjection?.Invoke(tempPath);
                File.Move(tempPath, fullTarget, overwrite: true);
            }
            catch
            {
                try { File.Delete(tempPath); } catch { /* the stale temp is swept on the next successful save */ }
                throw;
            }

            AfterCommit(directory, name, onWarning);
        }

        /// <summary>
        /// Validates the target up front — before any data is written — and resolves it to
        /// (directory, file name, full path). The directory must already exist: staging happens
        /// inside it, which is what guarantees the commit rename never crosses filesystems.
        /// </summary>
        private static (string Directory, string Name, string FullTarget) ValidateTarget(string targetPath)
        {
            if (string.IsNullOrWhiteSpace(targetPath))
                throw new ArgumentException("Target path cannot be null or empty.", nameof(targetPath));

            string fullTarget = Path.GetFullPath(targetPath);
            string name = Path.GetFileName(fullTarget);
            string? directory = Path.GetDirectoryName(fullTarget);
            if (string.IsNullOrEmpty(name) || string.IsNullOrEmpty(directory))
                throw new ArgumentException($"Target path '{targetPath}' does not name a file or directory.", nameof(targetPath));
            if (IsTempName(name))
                throw new ArgumentException(
                    $"Target name '{name}' starts with the reserved staging prefix '{TempPrefix}'.", nameof(targetPath));
            if (!Directory.Exists(directory))
                throw new DirectoryNotFoundException(
                    $"Cannot save '{targetPath}': directory '{directory}' does not exist. Atomic writes stage a " +
                    "temp copy inside the target's directory (so the commit rename stays on one filesystem); " +
                    "create the directory first.");
            return (directory, name, fullTarget);
        }

        /// <summary>Staged sibling name for a target name; unique so concurrent savers never collide.</summary>
        private static string StageName(string targetName) =>
            $"{TempPrefix}{targetName}-{Guid.NewGuid():N}";

        /// <summary>
        /// Post-commit housekeeping: sweep stale temps from earlier failed saves of this target.
        /// Best-effort — the commit has already succeeded.
        /// </summary>
        private static void AfterCommit(string directory, string targetName, Action<string>? onWarning)
        {
            onWarning ??= static _ => { }; // best-effort housekeeping; observe it by passing a callback

            // Our own temp was renamed away, so every remaining sibling shaped exactly like
            // ".tmp-<targetName>-<32-hex-guid>" is a leftover from a failed save of *this* target.
            // The strict GUID-suffix match avoids touching a different target whose name shares
            // this one as a prefix (saving "run" must not sweep "run-2"'s ".tmp-run-2-<guid>").
            foreach (var stale in EnumerateSafely(directory, $"{TempPrefix}{targetName}-*", onWarning))
            {
                if (!IsStagedTempFor(Path.GetFileName(stale), targetName)) continue;
                try { DeleteAbandonedTemp(stale); }
                catch (Exception e) { onWarning($"Shorokoo: failed to remove stale temp '{stale}': {e.Message}"); }
            }
        }

        private static IEnumerable<string> EnumerateSafely(string directory, string pattern, Action<string> onWarning)
        {
            try { return Directory.EnumerateFileSystemEntries(directory, pattern).ToList(); }
            catch (Exception e)
            {
                onWarning($"Shorokoo: failed to scan '{directory}' for '{pattern}': {e.Message}");
                return [];
            }
        }

        /// <summary>
        /// True when <paramref name="entryName"/> is exactly a staged sibling for
        /// <paramref name="targetName"/>: the reserved prefix, the target name, a '-', then a
        /// 32-character hex GUID (the <see cref="StageName"/> shape). The strict suffix keeps a
        /// target from matching a differently named one that merely shares its name as a prefix.
        /// </summary>
        private static bool IsStagedTempFor(string entryName, string targetName)
        {
            string prefix = $"{TempPrefix}{targetName}-";
            if (!entryName.StartsWith(prefix, StringComparison.Ordinal)) return false;
            ReadOnlySpan<char> suffix = entryName.AsSpan(prefix.Length);
            if (suffix.Length != 32) return false;
            foreach (char c in suffix)
                if (!char.IsAsciiHexDigit(c)) return false;
            return true;
        }

        /// <summary>
        /// Removes an abandoned staged temp without clobbering one a concurrent writer still
        /// holds. The file is opened with a deny-all share before deletion (delete-on-close): that
        /// open succeeds only when no live writer owns it — on Windows, and on Unix where .NET
        /// maps <see cref="FileShare.None"/> to an advisory <c>flock</c> — so an in-flight save to
        /// the same target throws here and is skipped rather than losing its data.
        /// </summary>
        private static void DeleteAbandonedTemp(string path)
        {
            using (new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None, 1, FileOptions.DeleteOnClose)) { }
        }
    }
}

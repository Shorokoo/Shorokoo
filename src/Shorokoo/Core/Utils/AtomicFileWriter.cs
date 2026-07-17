namespace Shorokoo.Core.Utils
{
    /// <summary>
    /// Atomic write-and-commit for files and directories: content is staged under a
    /// <c>.tmp-</c>-prefixed sibling name in the target's directory (same filesystem, so the
    /// commit rename is atomic), flushed to disk, then renamed onto the target. On any failure
    /// the staged copy is deleted and the previous target is left untouched — a crash mid-save
    /// never corrupts the existing file. This is the designated write path for every
    /// save/export API (see <see cref="TrainingCheckpoint.Save"/>); it carries no assumptions
    /// about what is being written.
    /// </summary>
    /// <remarks>
    /// A successful commit also removes stale <c>.tmp-</c> siblings left behind by earlier
    /// failed saves of the same target, and — when a <see cref="RetainPolicy"/> is supplied —
    /// prunes older siblings matching the policy's pattern (retain-last-N rotation). Rotation
    /// and cleanup never fail the save: the new content is already committed, so their failures
    /// are surfaced only through the optional <c>onWarning</c> callback (silent if none is given).
    /// </remarks>
    internal static class AtomicFileWriter
    {
        /// <summary>
        /// Prefix of staged (not yet committed) sibling names. Readers of container formats
        /// must ignore entries whose name matches <see cref="IsTempName"/>.
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
            RetainPolicy? retain = null,
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

            AfterCommit(directory, name, fullTarget, retain, onWarning);
        }

        /// <summary>
        /// Atomically writes a directory (for container formats): <paramref name="populate"/>
        /// fills a staged <c>.tmp-</c> directory next to <paramref name="targetPath"/>, whose
        /// files are then flushed to disk and the whole directory renamed onto the target.
        /// Because a rename cannot atomically replace a non-empty directory, the target must
        /// not exist yet — write each version under a fresh name and use a
        /// <see cref="RetainPolicy"/> to prune old ones. Readers must ignore siblings for
        /// which <see cref="IsTempName"/> is true.
        /// </summary>
        internal static void WriteDirectory(
            string targetPath,
            Action<string> populate,
            RetainPolicy? retain = null,
            Action<string>? onWarning = null)
        {
            if (populate is null) throw new ArgumentNullException(nameof(populate));
            var (directory, name, fullTarget) = ValidateTarget(targetPath);
            if (Directory.Exists(fullTarget) || File.Exists(fullTarget))
                throw new IOException(
                    $"Cannot atomically write directory '{targetPath}': the target already exists, and a rename " +
                    "cannot atomically replace a non-empty directory. Write each version under a fresh name and " +
                    "prune old ones with a retain policy.");

            string tempPath = Path.Combine(directory, StageName(name));
            try
            {
                Directory.CreateDirectory(tempPath);
                populate(tempPath);
                foreach (var file in Directory.EnumerateFiles(tempPath, "*", SearchOption.AllDirectories))
                    FlushFileToDisk(file);
                CommitFaultInjection?.Invoke(tempPath);
                Directory.Move(tempPath, fullTarget);
            }
            catch
            {
                try { Directory.Delete(tempPath, recursive: true); } catch { /* swept on the next successful save */ }
                throw;
            }

            AfterCommit(directory, name, fullTarget, retain, onWarning);
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
        /// Post-commit housekeeping: sweep stale temps from earlier failed saves of this target,
        /// and apply rotation. All best-effort — the commit has already succeeded.
        /// </summary>
        private static void AfterCommit(
            string directory, string targetName, string fullTarget, RetainPolicy? retain, Action<string>? onWarning)
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

            if (retain is null) return;
            try
            {
                // Keep the N most recent siblings matching the pattern; staged temps and files
                // not matching the pattern are never touched, nor is the entry just committed.
                var survivors = Directory.EnumerateFileSystemEntries(directory, retain.SearchPattern)
                    .Where(p => !IsTempName(Path.GetFileName(p)))
                    .OrderByDescending(File.GetLastWriteTimeUtc)
                    .ThenByDescending(p => p, StringComparer.Ordinal)
                    .Skip(retain.Count);
                foreach (var old in survivors)
                {
                    if (string.Equals(Path.GetFullPath(old), fullTarget, StringComparison.Ordinal)) continue;
                    try { DeleteEntry(old); }
                    catch (Exception e) { onWarning($"Shorokoo: rotation failed to remove '{old}': {e.Message}"); }
                }
            }
            catch (Exception e)
            {
                onWarning($"Shorokoo: rotation failed for pattern '{retain.SearchPattern}' in '{directory}': {e.Message}");
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

        private static void DeleteEntry(string path)
        {
            if (Directory.Exists(path)) Directory.Delete(path, recursive: true);
            else File.Delete(path);
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
        /// holds. A file is opened with a deny-all share before deletion (delete-on-close): that
        /// open succeeds only when no live writer owns it — on Windows, and on Unix where .NET
        /// maps <see cref="FileShare.None"/> to an advisory <c>flock</c> — so an in-flight save to
        /// the same target throws here and is skipped rather than losing its data. A directory
        /// temp can't be probed this way and is removed directly (concurrent directory writes to a
        /// single target are unsupported).
        /// </summary>
        private static void DeleteAbandonedTemp(string path)
        {
            if (Directory.Exists(path)) { Directory.Delete(path, recursive: true); return; }
            using (new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None, 1, FileOptions.DeleteOnClose)) { }
        }

        private static void FlushFileToDisk(string path)
        {
            try
            {
                using var handle = File.OpenHandle(path, FileMode.Open, FileAccess.ReadWrite);
                RandomAccess.FlushToDisk(handle);
            }
            catch { /* durability is best-effort; the content itself is already fully written */ }
        }
    }

    /// <summary>
    /// Opt-in retain-last-N rotation for <see cref="AtomicFileWriter"/>: after a successful
    /// commit, siblings of the target matching <see cref="SearchPattern"/> are pruned down to
    /// the <see cref="Count"/> most recent (by last-write time, name as tie-break). Entries
    /// that do not match the pattern are never touched.
    /// </summary>
    internal sealed class RetainPolicy
    {
        /// <summary>How many matching siblings survive a commit (including the one just written).</summary>
        internal int Count { get; }

        /// <summary>
        /// Wildcard pattern (as accepted by <see cref="Directory.EnumerateFileSystemEntries(string, string)"/>)
        /// selecting the sibling names that participate in rotation, e.g. <c>"ckpt-*.safetensors"</c>.
        /// </summary>
        internal string SearchPattern { get; }

        private RetainPolicy(int count, string searchPattern)
        {
            Count = count;
            SearchPattern = searchPattern;
        }

        /// <summary>Keeps the <paramref name="count"/> most recent entries matching <paramref name="searchPattern"/>.</summary>
        internal static RetainPolicy KeepLast(int count, string searchPattern)
        {
            if (count < 1)
                throw new ArgumentOutOfRangeException(nameof(count), count,
                    "KeepLast must retain at least the checkpoint being written.");
            if (string.IsNullOrWhiteSpace(searchPattern))
                throw new ArgumentException("Rotation needs a sibling-name pattern.", nameof(searchPattern));
            if (searchPattern.Contains(Path.DirectorySeparatorChar) || searchPattern.Contains(Path.AltDirectorySeparatorChar))
                throw new ArgumentException("Rotation patterns match sibling names only, not paths.", nameof(searchPattern));
            return new RetainPolicy(count, searchPattern);
        }
    }
}

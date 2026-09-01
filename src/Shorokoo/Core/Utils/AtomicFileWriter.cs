namespace Shorokoo.Core.Utils
{
    /// <summary>
    /// Atomic write-and-commit for files: content is staged under a
    /// <c>.tmp-</c>-prefixed sibling name in the target's directory (same filesystem, so the
    /// commit rename is atomic), flushed to disk, then renamed onto the target. On any failure
    /// the staged copy is deleted and the previous target is left untouched — a crash mid-save
    /// never corrupts the existing file. This is the designated write path for every
    /// save/export API (see <see cref="TrainingCheckpoint.Save(string, CheckpointComponents?)"/>); it carries no assumptions
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
        /// commit window. Thread-scoped, so a hook installed by one parallel test is invisible
        /// to every other thread; still reset it in a <c>finally</c>.
        /// </summary>
        [ThreadStatic]
        internal static Action<string>? CommitFaultInjection;

        /// <summary>
        /// Test hook: invoked at the start of rotation — i.e. after the new file has already been
        /// committed — with the committed path. Throwing here simulates a rotation failure, used to
        /// verify rotation never fails the save. Thread-scoped, so a hook installed by one parallel
        /// test is invisible to every other thread; still reset it in a <c>finally</c>.
        /// </summary>
        [ThreadStatic]
        internal static Action<string>? RotationFaultInjection;

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
        /// Test hook for the directory replace window: invoked after an existing target has been
        /// renamed aside but before the staged tree is renamed onto the target. Throwing here
        /// simulates a failure mid-replace, verifying the rollback that puts the previous
        /// checkpoint back in place. Thread-scoped like the other hooks; reset in a <c>finally</c>.
        /// </summary>
        [ThreadStatic]
        internal static Action<string>? ReplaceFaultInjection;

        /// <summary>
        /// Atomically writes a directory tree: <paramref name="writeContent"/> populates a staged
        /// <c>.tmp-</c>-prefixed sibling directory next to <paramref name="targetPath"/>, which is
        /// then committed by a directory rename — so the target path never names a half-written
        /// tree. Replacing an existing target is supported: a directory rename cannot overwrite,
        /// so the previous target is first renamed aside (to another staged name), and removed
        /// best-effort after the commit; a failure between the two renames moves it straight
        /// back, so on any in-process failure the staged copy is deleted and the previous target
        /// is left in place untouched. Only a hard crash inside that two-rename window can leave
        /// the target absent — the previous tree then still exists, complete, under its aside
        /// name until the sweep after the next successful save. The target's parent directory
        /// must already exist. Content durability is the callback's concern: flush files it
        /// writes to disk (the directory-format writers do).
        /// </summary>
        internal static void WriteDirectory(
            string targetPath,
            Action<string> writeContent,
            Action<string>? onWarning = null)
        {
            if (writeContent is null) throw new ArgumentNullException(nameof(writeContent));
            // A directory target is commonly written with trailing separators ("run.skpt/");
            // normalize so the name/parent split below sees the directory's own name. Trimming
            // everything down to nothing (an all-separator path) falls through to ValidateTarget's
            // own does-not-name-a-directory rejection.
            string trimmedTarget = string.IsNullOrWhiteSpace(targetPath)
                ? targetPath
                : targetPath.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            var (directory, name, fullTarget) = ValidateTarget(
                trimmedTarget.Length == 0 ? targetPath : trimmedTarget);
            if (File.Exists(fullTarget))
                throw new IOException(
                    $"Cannot save directory '{targetPath}': the target already exists as a file. " +
                    "Delete it first, or save to another path.");

            string tempPath = Path.Combine(directory, StageName(name));
            try
            {
                Directory.CreateDirectory(tempPath);
                writeContent(tempPath);
                CommitFaultInjection?.Invoke(tempPath);
                if (Directory.Exists(fullTarget))
                {
                    // A staged-shaped aside name, so a crash between the two renames leaves it
                    // recognizable as uncommitted debris for the post-commit sweep.
                    string asidePath = Path.Combine(directory, StageName(name));
                    Directory.Move(fullTarget, asidePath);
                    try
                    {
                        ReplaceFaultInjection?.Invoke(tempPath);
                        Directory.Move(tempPath, fullTarget);
                    }
                    catch
                    {
                        // Roll the previous checkpoint back into place, so a failed replace
                        // leaves the target exactly as it was. If even the rollback fails, the
                        // previous tree stays recoverable under the aside name — say where.
                        try { Directory.Move(asidePath, fullTarget); }
                        catch (Exception rollbackFailure)
                        {
                            (onWarning ?? (static _ => { }))(
                                "Shorokoo: a failed replace could not restore the previous checkpoint " +
                                $"to '{fullTarget}'; it is preserved, complete, at '{asidePath}' " +
                                $"({rollbackFailure.Message}).");
                        }
                        throw;
                    }
                    try { Directory.Delete(asidePath, recursive: true); }
                    catch (Exception e)
                    {
                        (onWarning ?? (static _ => { }))(
                            $"Shorokoo: failed to remove the replaced checkpoint copy '{asidePath}': {e.Message}");
                    }
                }
                else
                {
                    Directory.Move(tempPath, fullTarget);
                }
            }
            catch
            {
                try { if (Directory.Exists(tempPath)) Directory.Delete(tempPath, recursive: true); }
                catch { /* the stale temp is swept on the next successful save */ }
                throw;
            }

            AfterCommit(directory, name, onWarning);
        }

        /// <summary>Suffix a sweep renames a doomed staged directory to before deleting it, so a
        /// half-finished sweep leaves a recognizable non-staged name (readers already ignore the
        /// <see cref="TempPrefix"/>) that the next sweep finishes off.</summary>
        private const string SweepSuffix = "-sweep";

        /// <summary>
        /// The newest creation/write timestamp anywhere in a staged tree — the liveness signal
        /// the sweep checks, since a concurrent saver flushing content into the tree keeps some
        /// timestamp fresh. When the tree cannot be examined, now is returned, so the sweep never
        /// deletes what it cannot rule out as live.
        /// </summary>
        private static DateTime LastActivityUtc(string root)
        {
            try
            {
                DateTime newest = Directory.GetCreationTimeUtc(root);
                DateTime rootWrite = Directory.GetLastWriteTimeUtc(root);
                if (rootWrite > newest) newest = rootWrite;
                foreach (var entry in Directory.EnumerateFileSystemEntries(root, "*", SearchOption.AllDirectories))
                {
                    var t = File.GetLastWriteTimeUtc(entry);
                    if (t > newest) newest = t;
                }
                return newest;
            }
            catch
            {
                return DateTime.UtcNow;
            }
        }

        /// <summary>How long a staged sibling directory must show no write activity before the
        /// post-commit sweep treats it as abandoned rather than a concurrent saver's live
        /// staging area.</summary>
        private static readonly TimeSpan StaleDirectoryGracePeriod = TimeSpan.FromHours(1);

        /// <summary>
        /// Writes a file exactly as <see cref="WriteFile(string, Action{Stream}, Action{string})"/>
        /// and then, <b>after the commit has succeeded</b>, prunes older members of the same
        /// checkpoint series so only the <see cref="RetainPolicy.Keep"/> most recent survive.
        ///
        /// <para>
        /// Ordering is <b>explicit and producer-owned</b>: rotation orders the series by the
        /// integer token each name carries between <see cref="RetainPolicy.Prefix"/> and
        /// <see cref="RetainPolicy.Suffix"/> (the caller encodes a monotonic key there — e.g. the
        /// training step). Integer compare of that token is correct regardless of filesystem
        /// timestamp resolution and of zero-padding, so <c>ckpt-9</c> sorts before <c>ckpt-10</c>.
        /// Ordering is never inferred from file mtime.
        /// </para>
        ///
        /// <para>
        /// Rotation only ever deletes members of the series. Non-matching siblings (different
        /// prefix/suffix, or a non-numeric token), staged <c>.tmp-</c> temps, and the entry just
        /// committed are left untouched. Rotation is best-effort: because the new file is already
        /// committed when it runs, a rotation failure <b>never</b> fails the save — it surfaces only
        /// through <paramref name="onWarning"/> (silent if none is given).
        /// </para>
        /// </summary>
        internal static void WriteFile(
            string targetPath,
            Action<Stream> writeContent,
            RetainPolicy retain,
            Action<string>? onWarning = null)
        {
            // Commit first — this throws on write failure, which correctly fails the save.
            WriteFile(targetPath, writeContent, onWarning);

            // The checkpoint is now committed; rotation is pure housekeeping and must not throw
            // out of a successful save.
            try { Rotate(targetPath, retain, onWarning); }
            catch (Exception e)
            {
                (onWarning ?? (static _ => { }))($"Shorokoo: checkpoint rotation failed: {e.Message}");
            }
        }

        /// <summary>
        /// Prunes older members of the retain series in the committed file's directory, keeping the
        /// <see cref="RetainPolicy.Keep"/> highest-indexed members. See
        /// <see cref="WriteFile(string, Action{Stream}, RetainPolicy, Action{string})"/> for the
        /// ordering and never-delete-outside-the-series guarantees. Best-effort; the commit already
        /// succeeded.
        /// </summary>
        private static void Rotate(string committedPath, RetainPolicy retain, Action<string>? onWarning)
        {
            onWarning ??= static _ => { };

            string fullTarget = Path.GetFullPath(committedPath);
            string directory = Path.GetDirectoryName(fullTarget)!;
            string committedName = Path.GetFileName(fullTarget);

            RotationFaultInjection?.Invoke(fullTarget);

            // Collect series members with their parsed integer index. The glob is only a coarse
            // pre-filter; TryParseSeriesIndex below is authoritative, so an over-broad glob can't
            // cause a wrong deletion.
            var members = new List<(long Index, string Path)>();
            foreach (var entry in EnumerateSafely(directory, $"{retain.Prefix}*{retain.Suffix}", onWarning))
            {
                if (!File.Exists(entry)) continue;                       // a directory, or vanished under us
                string name = Path.GetFileName(entry);
                if (IsTempName(name)) continue;                         // never touch a staged temp
                if (!TryParseSeriesIndex(name, retain, out long index)) continue; // non-matching sibling
                members.Add((index, entry));
            }

            if (members.Count <= retain.Keep) return;

            // Newest-first by the producer-owned integer key; delete everything past the first Keep.
            members.Sort(static (a, b) => b.Index.CompareTo(a.Index));
            for (int i = retain.Keep; i < members.Count; i++)
            {
                // Defence in depth: the file we just committed is by construction among the newest
                // and so never reaches here, but never delete it regardless.
                if (string.Equals(Path.GetFileName(members[i].Path), committedName, StringComparison.Ordinal))
                    continue;
                try { File.Delete(members[i].Path); }
                catch (Exception e)
                {
                    onWarning($"Shorokoo: failed to rotate out old checkpoint '{members[i].Path}': {e.Message}");
                }
            }
        }

        /// <summary>
        /// True when <paramref name="name"/> is <c>{Prefix}{token}{Suffix}</c> with a non-empty
        /// base-10 <paramref name="index"/> token (digits only — no sign, whitespace, or grouping).
        /// A name whose token is empty or non-numeric is <b>not</b> a series member and is never
        /// rotated out.
        /// </summary>
        private static bool TryParseSeriesIndex(string name, RetainPolicy retain, out long index)
        {
            index = 0;
            if (name.Length <= retain.Prefix.Length + retain.Suffix.Length) return false; // no room for a token
            if (!name.StartsWith(retain.Prefix, StringComparison.Ordinal)) return false;
            if (!name.EndsWith(retain.Suffix, StringComparison.Ordinal)) return false;

            ReadOnlySpan<char> token = name.AsSpan(
                retain.Prefix.Length, name.Length - retain.Prefix.Length - retain.Suffix.Length);
            foreach (char c in token)
                if (!char.IsAsciiDigit(c)) return false;
            // Token is non-empty ASCII digits, so the culture-sensitive default parse is safe here.
            return long.TryParse(token, out index);
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
        /// Post-commit housekeeping shared by <see cref="WriteFile(string, Action{Stream}, Action{string})"/>
        /// and <see cref="WriteDirectory"/>: sweep stale temps — file- and directory-shaped alike,
        /// so a crashed save of either form is cleaned up by the next successful save of the same
        /// target in either form. Best-effort — the commit has already succeeded.
        ///
        /// <para>Our own temp was renamed away, so every remaining sibling shaped exactly like
        /// <c>.tmp-&lt;targetName&gt;-&lt;32-hex-guid&gt;</c> is a leftover from a failed save of
        /// <i>this</i> target (the strict GUID-suffix match keeps a target from sweeping a
        /// differently named one that merely shares its name as a prefix). A file temp is guarded
        /// against a live concurrent writer by <see cref="DeleteAbandonedTemp"/>'s share-mode
        /// open. A directory temp has no share-mode guard, so two defenses stand in: only a tree
        /// with no write activity anywhere in it for the whole grace period is treated as
        /// abandoned, and a doomed tree is renamed aside before deletion — the rename is atomic
        /// and the staging writers never recreate directories mid-write, so a saver that was
        /// somehow still using the tree fails loudly on its next write or its commit rename
        /// rather than ever committing a partially swept checkpoint.</para>
        /// </summary>
        private static void AfterCommit(string directory, string targetName, Action<string>? onWarning)
        {
            onWarning ??= static _ => { }; // best-effort housekeeping; observe it by passing a callback

            foreach (var stale in EnumerateSafely(directory, $"{TempPrefix}{targetName}-*", onWarning))
            {
                string staleName = Path.GetFileName(stale);
                if (Directory.Exists(stale) && staleName.EndsWith(SweepSuffix, StringComparison.Ordinal)
                    && IsStagedTempFor(staleName[..^SweepSuffix.Length], targetName))
                {
                    // A doomed tree an earlier sweep renamed but failed to finish deleting.
                    try { Directory.Delete(stale, recursive: true); }
                    catch (Exception e) { onWarning($"Shorokoo: failed to remove stale temp '{stale}': {e.Message}"); }
                    continue;
                }
                if (!IsStagedTempFor(staleName, targetName)) continue;
                try
                {
                    if (Directory.Exists(stale))
                    {
                        if (DateTime.UtcNow - LastActivityUtc(stale) < StaleDirectoryGracePeriod)
                            continue;
                        string doomed = stale + SweepSuffix;
                        Directory.Move(stale, doomed);
                        Directory.Delete(doomed, recursive: true);
                    }
                    else
                    {
                        DeleteAbandonedTemp(stale);
                    }
                }
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

        /// <summary>
        /// Opt-in retain-last-N rotation policy for
        /// <see cref="WriteFile(string, Action{Stream}, RetainPolicy, Action{string})"/>. A series is
        /// the set of files in one directory named <c>{Prefix}{index}{Suffix}</c> with a base-10
        /// integer <c>index</c>; the producer chooses a monotonic key for that index (for training
        /// checkpoints, the step). Rotation keeps the <see cref="Keep"/> highest-indexed members and
        /// orders strictly by the parsed integer, never by file mtime — so it is correct even with
        /// coarse timestamp resolution and non-zero-padded indices.
        /// </summary>
        internal readonly struct RetainPolicy
        {
            /// <summary>Literal text before the integer index in every series member's name.</summary>
            internal string Prefix { get; }

            /// <summary>Literal text after the integer index (e.g. a file extension); may be empty.</summary>
            internal string Suffix { get; }

            /// <summary>Number of most-recent members to keep; at least 1.</summary>
            internal int Keep { get; }

            private RetainPolicy(string prefix, string suffix, int keep)
            {
                Prefix = prefix;
                Suffix = suffix;
                Keep = keep;
            }

            /// <summary>
            /// Keep the <paramref name="keep"/> highest-indexed members of the series whose names are
            /// <c>{seriesPrefix}{index}{seriesSuffix}</c>. <paramref name="keep"/> must be at least 1
            /// (rotation never deletes down to nothing — the entry just committed is always kept).
            /// </summary>
            internal static RetainPolicy KeepLast(int keep, string seriesPrefix, string seriesSuffix)
            {
                if (keep < 1)
                    throw new ArgumentOutOfRangeException(nameof(keep), keep, "Retain count must be at least 1.");
                ArgumentNullException.ThrowIfNull(seriesPrefix);
                ArgumentNullException.ThrowIfNull(seriesSuffix);
                return new RetainPolicy(seriesPrefix, seriesSuffix, keep);
            }
        }
    }
}

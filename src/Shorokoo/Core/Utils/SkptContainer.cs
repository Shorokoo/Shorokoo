using System;
using System.IO;
using System.IO.Compression;

namespace Shorokoo.Core.Utils
{
    /// <summary>
    /// Read-side access to one .skpt checkpoint in either of its two on-disk shapes: the
    /// single-file STORED zip, or the directory form (issue #183) whose entries are real files
    /// under the checkpoint root. The manifest is identical in both shapes — entries are
    /// addressed by the same archive-relative paths — so every loader reads through this one
    /// surface and works on either shape unchanged. <see cref="Open"/> picks the shape from the
    /// filesystem (a directory path is the directory form; a file path is the zip), never from
    /// content sniffing.
    /// </summary>
    internal abstract class SkptContainer : IDisposable
    {
        /// <summary>The checkpoint's path as given by the caller, used in error messages.</summary>
        internal string CheckpointPath { get; }

        private protected SkptContainer(string checkpointPath) => CheckpointPath = checkpointPath;

        /// <summary>
        /// Opens the checkpoint at <paramref name="path"/>: a directory opens as the directory
        /// form, a file as the zip form; a path naming neither throws
        /// <see cref="FileNotFoundException"/>. A file that is not a zip archive fails loudly
        /// here (the zip form's container gate).
        /// </summary>
        internal static SkptContainer Open(string path)
        {
            if (Directory.Exists(path)) return new SkptDirectoryContainer(path);
            if (File.Exists(path)) return SkptZipContainer.OpenFile(path);
            throw new FileNotFoundException($"Checkpoint not found: {path}", path);
        }

        /// <summary>
        /// The bytes of one entry by its manifest (archive-relative, forward-slash) path, or
        /// <c>null</c> when the checkpoint has no such entry. Fails loudly on an entry larger
        /// than this .skpt version reads (2 GiB), and — in the directory form — on an entry path
        /// that does not resolve inside the checkpoint root (absolute path or <c>..</c>
        /// traversal), the same rule the ONNX external-data reader applies to its
        /// <c>location</c> field.
        /// </summary>
        internal abstract byte[]? TryReadEntry(string entryPath);

        /// <summary>Reads an entry the manifest requires, failing loudly (naming
        /// <paramref name="role"/>) when the checkpoint lacks it.</summary>
        internal byte[] ReadRequiredEntry(string entryPath, string role)
            => TryReadEntry(entryPath) ?? throw new InvalidDataException(
                $"'{CheckpointPath}': the manifest references entry '{entryPath}' (for {role}), " +
                $"but {MissingEntryWhere}.");

        /// <summary>Reads the config.json manifest bytes, failing loudly when absent — a
        /// checkpoint without its manifest is not a .skpt at all.</summary>
        internal byte[] ReadManifestBytes()
            => TryReadEntry(SkptFileFormat.ConfigEntryName) ?? throw new InvalidDataException(NotACheckpointMessage);

        private protected abstract string MissingEntryWhere { get; }

        private protected abstract string NotACheckpointMessage { get; }

        public abstract void Dispose();
    }

    /// <summary>The zip shape: the whole file is read into memory and entries come out of the
    /// BCL <see cref="ZipArchive"/> (an implementation independent of the .skpt writer, which
    /// doubles as a standardness check).</summary>
    internal sealed class SkptZipContainer : SkptContainer
    {
        private readonly MemoryStream _stream;
        private readonly ZipArchive _archive;

        private SkptZipContainer(string path, MemoryStream stream, ZipArchive archive)
            : base(path)
        {
            _stream = stream;
            _archive = archive;
        }

        internal static SkptZipContainer OpenFile(string filePath)
        {
            var stream = new MemoryStream(File.ReadAllBytes(filePath), writable: false);
            try
            {
                return new SkptZipContainer(
                    filePath, stream, new ZipArchive(stream, ZipArchiveMode.Read, leaveOpen: true));
            }
            catch (InvalidDataException e)
            {
                stream.Dispose();
                throw new InvalidDataException(
                    $"'{filePath}' is not a .skpt checkpoint — it does not open as a zip archive. ({e.Message})", e);
            }
        }

        internal override byte[]? TryReadEntry(string entryPath)
        {
            var entry = _archive.GetEntry(entryPath);
            if (entry is null) return null;
            // entry.Length is the uncompressed size declared in the archive's directory; a
            // corrupt or hostile file can declare up to ~4 GiB. Reject oversize entries with
            // the loader's usual named error rather than letting the (int) cast below throw a
            // context-free OverflowException. This .skpt version reads in-memory entries only.
            if (entry.Length > int.MaxValue)
                throw new InvalidDataException(
                    $"'{CheckpointPath}': entry '{entry.FullName}' declares an uncompressed size of {entry.Length} " +
                    "bytes, which exceeds the maximum this .skpt version reads.");
            using var entryStream = entry.Open();
            using var buffer = new MemoryStream((int)entry.Length);
            entryStream.CopyTo(buffer);
            return buffer.ToArray();
        }

        private protected override string MissingEntryWhere => "the archive contains no such entry";

        private protected override string NotACheckpointMessage =>
            $"'{CheckpointPath}' is not a .skpt checkpoint — the archive contains no " +
            $"'{SkptFileFormat.ConfigEntryName}' manifest.";

        public override void Dispose()
        {
            _archive.Dispose();
            _stream.Dispose();
        }
    }

    /// <summary>The directory shape (issue #183): entries are real files under the checkpoint
    /// root, addressed by the same archive-relative paths the manifest records for the zip.</summary>
    internal sealed class SkptDirectoryContainer : SkptContainer
    {
        private readonly string _rootFull;

        internal SkptDirectoryContainer(string directoryPath)
            : base(directoryPath)
            => _rootFull = Path.GetFullPath(directoryPath);

        internal override byte[]? TryReadEntry(string entryPath)
        {
            var resolved = ResolveEntryPath(_rootFull, entryPath, CheckpointPath);
            if (!File.Exists(resolved)) return null;
            long length = new FileInfo(resolved).Length;
            if (length > int.MaxValue)
                throw new InvalidDataException(
                    $"'{CheckpointPath}': entry '{entryPath}' is {length} bytes, which exceeds the " +
                    "maximum this .skpt version reads.");
            return File.ReadAllBytes(resolved);
        }

        /// <summary>
        /// Resolves a manifest entry path against the checkpoint root, failing loudly on a path
        /// that does not stay inside it: an absolute path, or a relative one whose
        /// <c>..</c> traversal escapes the root. Same rule (and failure shape) as the ONNX
        /// external-data reader's <c>location</c> handling — a hostile manifest must not be able
        /// to read (or, on the write paths that share this helper, write) outside the checkpoint.
        /// </summary>
        internal static string ResolveEntryPath(string rootFullPath, string entryPath, string checkpointPath)
        {
            if (string.IsNullOrEmpty(entryPath))
                throw new InvalidDataException(
                    $"'{checkpointPath}': the manifest references an empty entry path.");
            if (Path.IsPathRooted(entryPath))
                throw new InvalidDataException(
                    $"'{checkpointPath}': the manifest references entry '{entryPath}', which is an " +
                    "absolute path; checkpoint entries must live inside the checkpoint directory.");
            string resolved;
            try
            {
                resolved = Path.GetFullPath(Path.Combine(rootFullPath, entryPath));
            }
            catch (ArgumentException e)   // e.g. an embedded NUL — the loader's named error, not a bare throw
            {
                throw new InvalidDataException(
                    $"'{checkpointPath}': the manifest references entry '{entryPath}', which is not " +
                    $"a valid path. ({e.Message})", e);
            }
            var rootPrefix = Path.EndsInDirectorySeparator(rootFullPath)
                ? rootFullPath
                : rootFullPath + Path.DirectorySeparatorChar;
            if (!resolved.StartsWith(rootPrefix, StringComparison.Ordinal))
                throw new InvalidDataException(
                    $"'{checkpointPath}': the manifest references entry '{entryPath}', which escapes " +
                    "the checkpoint directory; entries must resolve inside it.");
            return resolved;
        }

        /// <summary>Non-throwing form of <see cref="ResolveEntryPath"/>'s containment rule, for
        /// <see cref="Shorokoo.Persistence.Inspect"/>'s never-throw-on-content path: true iff the
        /// entry path is relative and resolves inside the root.</summary>
        internal static bool EntryPathStaysInside(string rootFullPath, string entryPath)
        {
            if (string.IsNullOrEmpty(entryPath) || Path.IsPathRooted(entryPath)) return false;
            string resolved;
            try
            {
                resolved = Path.GetFullPath(Path.Combine(rootFullPath, entryPath));
            }
            catch (ArgumentException)   // an invalid path (e.g. an embedded NUL) stays "outside"
            {
                return false;
            }
            var rootPrefix = Path.EndsInDirectorySeparator(rootFullPath)
                ? rootFullPath
                : rootFullPath + Path.DirectorySeparatorChar;
            return resolved.StartsWith(rootPrefix, StringComparison.Ordinal);
        }

        private protected override string MissingEntryWhere => "the checkpoint directory contains no such file";

        private protected override string NotACheckpointMessage =>
            $"'{CheckpointPath}' is not a .skpt checkpoint directory — it contains no " +
            $"'{SkptFileFormat.ConfigEntryName}' manifest.";

        public override void Dispose() { }
    }
}

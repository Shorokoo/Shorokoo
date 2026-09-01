using System;
using System.Collections.Generic;
using System.IO;
using Shorokoo.Core.Utils;

namespace Shorokoo
{
    // The zip ↔ directory conversion half of the Persistence facade (issue #183). A .skpt has two
    // on-disk shapes with byte-identical entry content — the single-file STORED zip and the
    // directory of real files — described by the same config.json manifest, so converting is a
    // matter of moving the manifest and every entry it references between the two layouts. The
    // manifest drives the conversion (it is the single source of wiring), each entry's recorded
    // SHA-256 is verified in transit, and both directions commit atomically, so a conversion can
    // never silently produce a corrupt checkpoint or carry stray content a hostile archive smuggled
    // in beside the manifest.
    public static partial class Persistence
    {
        /// <summary>
        /// Converts a single-file <c>.skpt</c> checkpoint into its <b>directory form</b> (issue
        /// #183): <paramref name="directoryPath"/> gets the same <c>config.json</c> at its root
        /// and the same models/ and data/ entries as real files, byte-identical to the zip's —
        /// so the result loads (<see cref="Load(string)"/>,
        /// <see cref="TrainingRig.Load(string, Runtime.ComputeContext?, Runtime.ComputeContext?)"/>)
        /// and inspects exactly like the source. Every entry's recorded SHA-256 is verified in
        /// transit, entry paths must resolve inside the target (a hostile manifest cannot write
        /// elsewhere), and only the manifest and the entries it references are carried. The
        /// write is atomic (staged to a temp directory beside the target and committed by
        /// rename); the target's parent directory must already exist, and an existing directory
        /// at the target — whatever it holds — is replaced by a completed conversion.
        /// <see cref="PackSkpt"/> converts back.
        /// </summary>
        /// <param name="skptFilePath">The single-file <c>.skpt</c> to convert.</param>
        /// <param name="directoryPath">Target directory path for the directory form.</param>
        public static void ExtractSkpt(string skptFilePath, string directoryPath)
        {
            if (string.IsNullOrWhiteSpace(skptFilePath))
                throw new ArgumentException("Checkpoint path cannot be null or empty.", nameof(skptFilePath));
            if (string.IsNullOrWhiteSpace(directoryPath))
                throw new ArgumentException("Checkpoint path cannot be null or empty.", nameof(directoryPath));
            if (Directory.Exists(skptFilePath))
                throw new ArgumentException(
                    $"'{skptFilePath}' is already a .skpt checkpoint directory; ExtractSkpt converts the " +
                    "single-file form. To produce the single-file form from it, use Persistence.PackSkpt.",
                    nameof(skptFilePath));

            using var container = SkptContainer.Open(skptFilePath);
            var entries = CollectManifestEntries(container);
            AtomicFileWriter.WriteDirectory(directoryPath,
                stagingRoot => SkptFileFormat.WriteDirectoryEntries(stagingRoot, entries, skptFilePath));
        }

        /// <summary>
        /// Converts a <c>.skpt</c> checkpoint <b>directory</b> back into the single-file form
        /// (issue #183): the same <c>config.json</c> and the same entries, byte-identical,
        /// written as the STORED zip <see cref="CheckpointBuilder.Save"/> produces (uncompressed
        /// safetensors data entries keep their 64-byte payload alignment). Every entry's
        /// recorded SHA-256 is verified in transit, and only the manifest and the entries it
        /// references are carried. The write is atomic (staged to a temp file and committed by
        /// rename); the target's directory must already exist. <see cref="ExtractSkpt"/>
        /// converts the other way.
        /// </summary>
        /// <param name="directoryPath">The <c>.skpt</c> checkpoint directory to convert.</param>
        /// <param name="skptFilePath">Target path of the single-file <c>.skpt</c>.</param>
        public static void PackSkpt(string directoryPath, string skptFilePath)
        {
            if (string.IsNullOrWhiteSpace(directoryPath))
                throw new ArgumentException("Checkpoint path cannot be null or empty.", nameof(directoryPath));
            if (string.IsNullOrWhiteSpace(skptFilePath))
                throw new ArgumentException("Checkpoint path cannot be null or empty.", nameof(skptFilePath));
            if (!Directory.Exists(directoryPath) && File.Exists(directoryPath))
                throw new ArgumentException(
                    $"'{directoryPath}' is a file, not a .skpt checkpoint directory; PackSkpt converts the " +
                    "directory form. To produce the directory form from a single-file .skpt, use " +
                    "Persistence.ExtractSkpt.", nameof(directoryPath));

            using var container = SkptContainer.Open(directoryPath);
            var entries = CollectManifestEntries(container);
            AtomicFileWriter.WriteFile(skptFilePath,
                stream => SkptFileFormat.WriteStoredZip(stream, entries, DateTime.UtcNow));
        }

        /// <summary>
        /// Reads a checkpoint's manifest and every entry it references out of either container
        /// shape, as the entry list the writers consume: <c>config.json</c> (verbatim) first,
        /// then the model entries and the data entries in manifest order. Each model/data
        /// entry's recorded SHA-256 is verified (over the stored bytes, so a compressed entry
        /// needs no decompression), and an uncompressed safetensors data entry is flagged for
        /// the zip form's payload alignment — reproducing the writers' rule, so a round-tripped
        /// zip keeps the writers' STORED-and-aligned payload layout (entry order may differ
        /// from a direct save; content and per-entry properties do not).
        /// </summary>
        private static List<SkptFileFormat.ZipEntrySpec> CollectManifestEntries(SkptContainer container)
        {
            var path = container.CheckpointPath;
            var configBytes = container.ReadManifestBytes();
            var manifest = SkptFileFormat.ParseManifest(configBytes, path);
            ValidateManifestIdentity(manifest, path);

            var entries = new List<SkptFileFormat.ZipEntrySpec>
            {
                new(SkptFileFormat.ConfigEntryName, configBytes, Align: false),
            };
            var carried = new Dictionary<string, byte[]>(StringComparer.Ordinal)
            {
                [SkptFileFormat.ConfigEntryName] = configBytes,
            };

            void Carry(string entryPath, string role, string? sha256, bool align)
            {
                // Two registry keys may share one stored entry; it is carried once, but every
                // key's recorded hash must match it, or the source is internally inconsistent.
                if (carried.TryGetValue(entryPath, out var already))
                {
                    VerifySha256(already, sha256, entryPath, path);
                    return;
                }
                var bytes = container.ReadRequiredEntry(entryPath, role);
                VerifySha256(bytes, sha256, entryPath, path);
                carried[entryPath] = bytes;
                entries.Add(new(entryPath, bytes, align));
            }

            foreach (var (key, model) in manifest.Models ?? new())
            {
                if (string.IsNullOrEmpty(model?.Entry))
                    throw new InvalidDataException(
                        $"'{path}': the manifest's model '{key}' names no archive entry.");
                Carry(model.Entry, $"model '{key}'", model.Sha256, align: false);
            }
            foreach (var (key, data) in manifest.Data ?? new())
            {
                if (string.IsNullOrEmpty(data?.Entry))
                    throw new InvalidDataException(
                        $"'{path}': the manifest's data entry '{key}' names no archive entry.");
                bool align = data.Format == SkptFileFormat.DataFormatSafeTensors
                    && (data.Compression ?? SkptFileFormat.CompressionNone) == SkptFileFormat.CompressionNone;
                Carry(data.Entry, $"data entry '{key}'", data.Sha256, align);
            }
            return entries;
        }
    }
}

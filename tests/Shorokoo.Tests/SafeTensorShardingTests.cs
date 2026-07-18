using System.IO.Compression;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage-purpose tests for the sharded (Hugging Face multi-file convention)
/// safetensors support in <see cref="SafeTensorLoader"/>: sharded save with an
/// opt-in maximum shard size producing a single atomically written zip
/// container, content-sniffed sharded load (zip container, index-manifest path,
/// or directory holding one), subset loads that leave unneeded shards
/// untouched, save-failure atomicity, and the loud validation failures
/// (missing shard, missing tensor, duplicate name, corrupt shard).
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class SafeTensorShardingCoverageTests
{
    /// <summary>Creates a fresh working directory unique to one test run.</summary>
    private static string CreateTempDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), "ShorokooSafeTensorShardingTests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    /// <summary>A 2x3 F32 tensor (24 raw bytes) with values derived from <paramref name="seed"/>.</summary>
    private static SafeTensor MakeTensor(string name, float seed)
    {
        var data = TensorData([2, 3], seed, seed + 1, seed + 2, seed + 3, seed + 4, seed + 5);
        return new SafeTensor(name, data, "F32", data.Shape.Dims);
    }

    /// <summary>
    /// List order deliberately differs from sorted-name order so that names that
    /// are adjacent when sorted (e.g. "w.b"/"w.c") land in different shards.
    /// Each tensor is 24 bytes; a 48-byte shard limit packs them [2, 2, 1].
    /// </summary>
    private static List<SafeTensor> MakeFixtureTensors() =>
    [
        MakeTensor("w.b", 0f),
        MakeTensor("w.a", 10f),
        MakeTensor("w.d", 20f),
        MakeTensor("w.c", 30f),
        MakeTensor("w.e", 40f),
    ];

    /// <summary>Serializes tensors to single-file safetensors bytes in memory.</summary>
    private static byte[] SafeTensorBytes(params SafeTensor[] tensors)
    {
        using var ms = new MemoryStream();
        SafeTensorLoader.SaveSafeTensorsToStream(ms, [.. tensors]);
        return ms.ToArray();
    }

    private static byte[] ReadEntryBytes(ZipArchiveEntry entry)
    {
        using var stream = entry.Open();
        var bytes = new byte[entry.Length];
        stream.ReadExactly(bytes);
        return bytes;
    }

    [Fact]
    public void TestShardedSaveLoadRoundtrip()
    {
        var dir = CreateTempDir();
        try
        {
            var tensors = MakeFixtureTensors();
            var globalMetadata = new Dictionary<string, object> { ["format"] = "pt" };

            // Reference: plain single-file save of the same tensors.
            var singlePath = Path.Combine(dir, "merged.safetensors");
            SafeTensorLoader.SaveSafeTensors(singlePath, tensors, globalMetadata);
            var expected = SafeTensorLoader.LoadSafeTensors(singlePath)
                .ToDictionary(t => t.Name);

            // A threshold the total size (120 bytes) does not exceed keeps the
            // single-file output byte-for-byte identical to a save without one.
            var unshardedPath = Path.Combine(dir, "unsharded.safetensors");
            SafeTensorLoader.SaveSafeTensors(unshardedPath, tensors, globalMetadata, maxShardSizeBytes: 1024);
            Assert.Equal(File.ReadAllBytes(singlePath), File.ReadAllBytes(unshardedPath));

            // Sharded save: 48-byte limit forces three shards of [2, 2, 1]
            // tensors, packaged as ONE zip container at the target path — no
            // loose shard or index files appear beside it.
            var shardedPath = Path.Combine(dir, "model.safetensors");
            SafeTensorLoader.SaveSafeTensors(shardedPath, tensors, globalMetadata, maxShardSizeBytes: 48);

            Assert.True(File.Exists(shardedPath));
            Assert.Equal(3, Directory.GetFiles(dir).Length); // merged, unsharded, model — nothing else
            byte[] expectedMagic = [(byte)'P', (byte)'K', 3, 4];
            Assert.Equal(expectedMagic, File.ReadAllBytes(shardedPath)[..4]);

            string[] shardNames =
            [
                "model-00001-of-00003.safetensors",
                "model-00002-of-00003.safetensors",
                "model-00003-of-00003.safetensors",
            ];
            using (var zip = ZipFile.OpenRead(shardedPath))
            {
                string[] entryNames = [.. zip.Entries.Select(e => e.FullName).OrderBy(n => n, StringComparer.Ordinal)];
                string[] expectedEntries = [.. shardNames, "model.safetensors.index.json"];
                Assert.Equal(expectedEntries, entryNames);

                var indexJson = System.Text.Encoding.UTF8.GetString(
                    ReadEntryBytes(zip.GetEntry("model.safetensors.index.json")!));
                Assert.Contains("\"weight_map\"", indexJson);
                Assert.Contains("\"total_size\":120", indexJson);

                // Every shard entry is an individually valid safetensors file
                // (stored uncompressed) that still carries the global metadata.
                var standaloneNames = new List<string>();
                foreach (var shardName in shardNames)
                {
                    var entry = zip.GetEntry(shardName)!;
                    Assert.Equal(entry.Length, entry.CompressedLength);
                    var shardBytes = ReadEntryBytes(entry);
                    var standalone = SafeTensorLoader.ParseSafeTensorBytes(shardBytes);
                    Assert.InRange(standalone.Count, 1, 2);
                    standaloneNames.AddRange(standalone.Select(t => t.Name));
                    Assert.Contains("__metadata__", System.Text.Encoding.UTF8.GetString(shardBytes));
                }
                Assert.Equal(tensors.Count, standaloneNames.Distinct().Count());
            }

            // The zip checkpoint is auto-detected by content and reproduces
            // every tensor bit-exactly.
            var loaded = SafeTensorLoader.LoadSafeTensors(shardedPath);
            Assert.Equal(tensors.Count, loaded.Count);
            foreach (var tensor in loaded)
            {
                var reference = expected[tensor.Name];
                Assert.Equal(reference.DataType, tensor.DataType);
                Assert.Equal(reference.Shape, tensor.Shape);
                Assert.Equal(reference.Data.AccessRawMemory().ToArray(), tensor.Data.AccessRawMemory().ToArray());
            }

            // The flow-through APIs auto-detect the container too.
            Assert.Equal(tensors.Count, SafeTensorLoader.LoadTensorDictionary(shardedPath).Count);
            Assert.Equal(tensors.Count, SafeTensorLoader.LoadModelParamSet(shardedPath).ModelParams.Length);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void TestShardedSubsetLoad()
    {
        var dir = CreateTempDir();
        try
        {
            // Zip container with one valid shard entry, one corrupt entry, and a
            // weight_map that also references an entry missing from the archive.
            var zipPath = Path.Combine(dir, "custom.safetensors");
            using (var fs = File.Create(zipPath))
            using (var zip = new ZipArchive(fs, ZipArchiveMode.Create))
            {
                void AddEntry(string name, byte[] bytes)
                {
                    var entry = zip.CreateEntry(name, CompressionLevel.NoCompression);
                    using var stream = entry.Open();
                    stream.Write(bytes, 0, bytes.Length);
                }
                AddEntry("s1.safetensors", SafeTensorBytes(MakeTensor("w.a", 10f), MakeTensor("w.b", 0f)));
                AddEntry("corrupt.safetensors", System.Text.Encoding.UTF8.GetBytes("not a safetensors payload"));
                AddEntry("custom.safetensors.index.json", System.Text.Encoding.UTF8.GetBytes(
                    "{\"weight_map\":{\"w.a\":\"s1.safetensors\",\"w.b\":\"s1.safetensors\"," +
                    "\"w.x\":\"corrupt.safetensors\",\"w.z\":\"missing.safetensors\"}}"));
            }

            // A subset living entirely in s1 loads without the corrupt or missing
            // entries ever being touched.
            var subset = SafeTensorLoader.LoadTensorDictionary(zipPath, ["w.a", "w.b"]);
            Assert.Equal(2, subset.Count);
            Assert.Equal(
                MakeTensor("w.a", 10f).Data.AccessRawMemory().ToArray(),
                subset["w.a"].AccessRawMemory().ToArray());

            // Requesting the corrupt entry's tensor fails parsing it; requesting a
            // tensor whose entry is absent names both tensor and entry.
            Assert.Throws<InvalidOperationException>(() => SafeTensorLoader.LoadSafeTensors(zipPath, ["w.x"]));
            var missing = Assert.Throws<FileNotFoundException>(() => SafeTensorLoader.LoadSafeTensors(zipPath, ["w.z"]));
            Assert.Contains("missing.safetensors", missing.Message);
            Assert.Contains("w.z", missing.Message);

            // Requesting a name absent from the weight_map names the tensor.
            var unknown = Assert.Throws<KeyNotFoundException>(() => SafeTensorLoader.LoadSafeTensors(zipPath, ["w.ghost"]));
            Assert.Contains("w.ghost", unknown.Message);

            // Hugging Face directory layout (hand-laid loose files): deleting the
            // shard that holds none of the requested tensors proves a subset load
            // never touches it.
            var hfDir = Path.Combine(dir, "hf");
            Directory.CreateDirectory(hfDir);
            File.WriteAllBytes(Path.Combine(hfDir, "s1.safetensors"),
                SafeTensorBytes(MakeTensor("w.a", 10f), MakeTensor("w.b", 0f)));
            File.WriteAllBytes(Path.Combine(hfDir, "s2.safetensors"),
                SafeTensorBytes(MakeTensor("w.c", 30f)));
            File.WriteAllText(Path.Combine(hfDir, "model.safetensors.index.json"),
                "{\"weight_map\":{\"w.a\":\"s1.safetensors\",\"w.b\":\"s1.safetensors\",\"w.c\":\"s2.safetensors\"}}");

            Assert.Equal(3, SafeTensorLoader.LoadSafeTensors(hfDir).Count);
            File.Delete(Path.Combine(hfDir, "s2.safetensors"));
            var hfSubset = SafeTensorLoader.LoadTensorDictionary(hfDir, ["w.a", "w.b"]);
            Assert.Equal(2, hfSubset.Count);
            var hfMissing = Assert.Throws<FileNotFoundException>(() => SafeTensorLoader.LoadSafeTensors(hfDir, ["w.c"]));
            Assert.Contains("s2.safetensors", hfMissing.Message);
            Assert.Contains("w.c", hfMissing.Message);

            // Subset loads work on plain single files as well, with the same
            // loud failure for unknown names.
            var singlePath = Path.Combine(dir, "single.safetensors");
            SafeTensorLoader.SaveSafeTensors(singlePath, MakeFixtureTensors());
            var singleSubset = SafeTensorLoader.LoadSafeTensors(singlePath, ["w.c"]);
            Assert.Equal("w.c", Assert.Single(singleSubset).Name);
            var singleUnknown = Assert.Throws<KeyNotFoundException>(
                () => SafeTensorLoader.LoadSafeTensors(singlePath, ["w.ghost"]));
            Assert.Contains("w.ghost", singleUnknown.Message);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void TestFailedSaveLeavesExistingFileIntact()
    {
        var dir = CreateTempDir();
        try
        {
            var path = Path.Combine(dir, "ckpt.safetensors");
            SafeTensorLoader.SaveSafeTensors(path, [MakeTensor("a", 0f)]);
            var original = File.ReadAllBytes(path);

            // Invalid input fails before any byte reaches the target: the
            // existing checkpoint survives a null list, an empty list, and a
            // null entry, with no staged temp files left behind.
            Assert.Throws<ArgumentNullException>(() => SafeTensorLoader.SaveSafeTensors(path, null!));
            Assert.Throws<ArgumentException>(() => SafeTensorLoader.SaveSafeTensors(path, []));
            Assert.Throws<InvalidOperationException>(
                () => SafeTensorLoader.SaveSafeTensors(path, [MakeTensor("b", 10f), null!]));
            Assert.Equal(original, File.ReadAllBytes(path));
            Assert.Equal([path], Directory.GetFiles(dir));

            // A sharded re-save atomically replaces the single file with the zip
            // container at the same path — one artifact, no stale siblings.
            SafeTensorLoader.SaveSafeTensors(path, MakeFixtureTensors(), maxShardSizeBytes: 48);
            Assert.Equal([path], Directory.GetFiles(dir));
            Assert.Equal(5, SafeTensorLoader.LoadSafeTensors(path).Count);

            // A failing save over the zip checkpoint leaves it untouched too.
            var zipOriginal = File.ReadAllBytes(path);
            Assert.Throws<ArgumentException>(() => SafeTensorLoader.SaveSafeTensors(path, [], maxShardSizeBytes: 48));
            Assert.Equal(zipOriginal, File.ReadAllBytes(path));
            Assert.Equal([path], Directory.GetFiles(dir));

            // And a single-file re-save replaces the zip container in place.
            SafeTensorLoader.SaveSafeTensors(path, [MakeTensor("c", 20f)]);
            Assert.Equal([path], Directory.GetFiles(dir));
            Assert.Equal("c", Assert.Single(SafeTensorLoader.LoadSafeTensors(path)).Name);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void TestShardedValidationFailures()
    {
        var dir = CreateTempDir();
        try
        {
            // Two hand-laid shards: s1 holds "a" and "b", s2 holds "c". Index
            // manifests are written by hand so each failure mode can be staged.
            File.WriteAllBytes(Path.Combine(dir, "s1.safetensors"),
                SafeTensorBytes(MakeTensor("a", 0f), MakeTensor("b", 10f)));
            File.WriteAllBytes(Path.Combine(dir, "s2.safetensors"),
                SafeTensorBytes(MakeTensor("c", 20f)));

            var indexPath = Path.Combine(dir, "model.safetensors.index.json");
            void WriteIndex(string weightMapJson) =>
                File.WriteAllText(indexPath, $"{{\"weight_map\":{weightMapJson}}}");

            // Baseline: a consistent hand-written index loads (shard file names
            // need not follow the -0000x-of-000NN convention), via the index path
            // and via the directory holding it.
            WriteIndex("{\"a\":\"s1.safetensors\",\"b\":\"s1.safetensors\",\"c\":\"s2.safetensors\"}");
            Assert.Equal(3, SafeTensorLoader.LoadSafeTensors(indexPath).Count);
            Assert.Equal(3, SafeTensorLoader.LoadSafeTensors(dir).Count);

            // Tensor listed in the weight_map but missing from its shard.
            WriteIndex("{\"a\":\"s1.safetensors\",\"b\":\"s1.safetensors\",\"c\":\"s2.safetensors\",\"ghost\":\"s2.safetensors\"}");
            var missingTensor = Assert.Throws<InvalidOperationException>(
                () => SafeTensorLoader.LoadSafeTensors(indexPath));
            Assert.Contains("ghost", missingTensor.Message);
            Assert.Contains("s2.safetensors", missingTensor.Message);

            // Tensor present in a shard but assigned to a different one (a
            // duplicate across shards).
            WriteIndex("{\"a\":\"s1.safetensors\",\"b\":\"s2.safetensors\",\"c\":\"s2.safetensors\"}");
            var duplicate = Assert.Throws<InvalidOperationException>(
                () => SafeTensorLoader.LoadSafeTensors(indexPath));
            Assert.Contains("Duplicate tensor 'b'", duplicate.Message);

            // The same tensor mapped twice inside the weight_map itself.
            WriteIndex("{\"a\":\"s1.safetensors\",\"a\":\"s2.safetensors\"}");
            var duplicateKey = Assert.Throws<InvalidOperationException>(
                () => SafeTensorLoader.LoadSafeTensors(indexPath));
            Assert.Contains("Duplicate tensor 'a'", duplicateKey.Message);

            // Tensor present in a shard but absent from the weight_map.
            WriteIndex("{\"a\":\"s1.safetensors\",\"c\":\"s2.safetensors\"}");
            var unmapped = Assert.Throws<InvalidOperationException>(
                () => SafeTensorLoader.LoadSafeTensors(indexPath));
            Assert.Contains("'b'", unmapped.Message);
            Assert.Contains("weight_map", unmapped.Message);

            // Missing shard file, named together with a tensor it should hold.
            WriteIndex("{\"a\":\"s1.safetensors\",\"b\":\"s1.safetensors\",\"z\":\"nope.safetensors\"}");
            var missingShard = Assert.Throws<FileNotFoundException>(
                () => SafeTensorLoader.LoadSafeTensors(indexPath));
            Assert.Contains("nope.safetensors", missingShard.Message);
            Assert.Contains("z", missingShard.Message);

            // A shard whose bytes are not a safetensors payload (e.g. a stale
            // Git-LFS pointer file) fails as ordinary corrupt data.
            File.WriteAllText(Path.Combine(dir, "corrupt.safetensors"),
                "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n");
            WriteIndex("{\"l\":\"corrupt.safetensors\"}");
            Assert.Throws<InvalidOperationException>(() => SafeTensorLoader.LoadSafeTensors(indexPath));

            // An index without a weight_map object.
            File.WriteAllText(indexPath, "{\"metadata\":{\"total_size\":1}}");
            var noMap = Assert.Throws<InvalidOperationException>(
                () => SafeTensorLoader.LoadSafeTensors(indexPath));
            Assert.Contains("weight_map", noMap.Message);

            // Zip containers without exactly one index entry fail loudly.
            void WriteZip(string path, params (string Name, byte[] Bytes)[] entries)
            {
                using var fs = File.Create(path);
                using var zip = new ZipArchive(fs, ZipArchiveMode.Create);
                foreach (var (name, bytes) in entries)
                {
                    var entry = zip.CreateEntry(name, CompressionLevel.NoCompression);
                    using var stream = entry.Open();
                    stream.Write(bytes, 0, bytes.Length);
                }
            }
            var shardBytes = SafeTensorBytes(MakeTensor("a", 0f));
            var noIndexZip = Path.Combine(dir, "noindex.safetensors");
            WriteZip(noIndexZip, ("s1.safetensors", shardBytes));
            var noIndex = Assert.Throws<InvalidOperationException>(() => SafeTensorLoader.LoadSafeTensors(noIndexZip));
            Assert.Contains("index", noIndex.Message);

            var twoIndexZip = Path.Combine(dir, "twoindex.safetensors");
            var indexBytes = System.Text.Encoding.UTF8.GetBytes("{\"weight_map\":{\"a\":\"s1.safetensors\"}}");
            WriteZip(twoIndexZip,
                ("s1.safetensors", shardBytes),
                ("one.safetensors.index.json", indexBytes),
                ("two.safetensors.index.json", indexBytes));
            Assert.Throws<InvalidOperationException>(() => SafeTensorLoader.LoadSafeTensors(twoIndexZip));

            // Directory resolution: none or several indexes fail loudly.
            var emptyDir = Path.Combine(dir, "empty");
            Directory.CreateDirectory(emptyDir);
            Assert.Throws<FileNotFoundException>(() => SafeTensorLoader.LoadSafeTensors(emptyDir));
            File.WriteAllText(Path.Combine(dir, "other.safetensors.index.json"), "{}");
            Assert.Throws<InvalidOperationException>(() => SafeTensorLoader.LoadSafeTensors(dir));

            // Sharded saving rejects non-positive limits and duplicate names.
            Assert.Throws<ArgumentOutOfRangeException>(() => SafeTensorLoader.SaveSafeTensors(
                Path.Combine(dir, "bad.safetensors"), [MakeTensor("a", 0f)], maxShardSizeBytes: 0));
            Assert.Throws<InvalidOperationException>(() => SafeTensorLoader.SaveSafeTensors(
                Path.Combine(dir, "dup.safetensors"),
                [MakeTensor("a", 0f), MakeTensor("b", 10f), MakeTensor("a", 20f)],
                maxShardSizeBytes: 48));
            Assert.False(File.Exists(Path.Combine(dir, "dup.safetensors")));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}

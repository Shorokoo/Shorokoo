namespace Shorokoo.Tests;

/// <summary>
/// Coverage-purpose tests for the sharded (Hugging Face multi-file) safetensors
/// support in <see cref="SafeTensorLoader"/>: sharded save with an opt-in
/// maximum shard size, auto-detected sharded load via the
/// <c>*.safetensors.index.json</c> manifest (index path or directory), subset
/// loads that leave unneeded shards untouched, and the loud validation failures
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
            Assert.False(File.Exists(unshardedPath + SafeTensorLoader.ShardIndexSuffix));

            // Sharded save: 48-byte limit forces three shards of [2, 2, 1] tensors.
            var shardedPath = Path.Combine(dir, "model.safetensors");
            SafeTensorLoader.SaveSafeTensors(shardedPath, tensors, globalMetadata, maxShardSizeBytes: 48);

            string[] shardNames =
            [
                "model-00001-of-00003.safetensors",
                "model-00002-of-00003.safetensors",
                "model-00003-of-00003.safetensors",
            ];
            var indexPath = shardedPath + SafeTensorLoader.ShardIndexSuffix;
            Assert.True(File.Exists(indexPath));
            Assert.False(File.Exists(shardedPath)); // no monolithic file alongside the shards
            foreach (var shardName in shardNames)
                Assert.True(File.Exists(Path.Combine(dir, shardName)));

            var indexJson = File.ReadAllText(indexPath);
            Assert.Contains("\"weight_map\"", indexJson);
            Assert.Contains("\"total_size\":120", indexJson);

            // Every shard is an individually valid safetensors file that still
            // carries the global metadata, and respects the size limit.
            var standaloneNames = new List<string>();
            foreach (var shardName in shardNames)
            {
                var shardPath = Path.Combine(dir, shardName);
                var standalone = SafeTensorLoader.LoadSafeTensors(shardPath);
                Assert.InRange(standalone.Count, 1, 2);
                standaloneNames.AddRange(standalone.Select(t => t.Name));
                Assert.Contains("__metadata__", File.ReadAllText(shardPath));
            }
            Assert.Equal(tensors.Count, standaloneNames.Distinct().Count());

            // Sharded load — via the index path, the directory, and the dictionary
            // helper — reproduces every tensor bit-exactly.
            foreach (var loadPath in new[] { indexPath, dir })
            {
                var loaded = SafeTensorLoader.LoadSafeTensors(loadPath);
                Assert.Equal(tensors.Count, loaded.Count);
                foreach (var tensor in loaded)
                {
                    var reference = expected[tensor.Name];
                    Assert.Equal(reference.DataType, tensor.DataType);
                    Assert.Equal(reference.Shape, tensor.Shape);
                    Assert.Equal(reference.Data.AccessRawMemory().ToArray(), tensor.Data.AccessRawMemory().ToArray());
                }
            }

            var dict = SafeTensorLoader.LoadTensorDictionary(dir);
            Assert.Equal(tensors.Count, dict.Count);

            // The flow-through APIs auto-detect sharded checkpoints too.
            var paramSet = SafeTensorLoader.LoadModelParamSet(dir);
            Assert.Equal(tensors.Count, paramSet.ModelParams.Length);
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
            var tensors = MakeFixtureTensors();
            var shardedPath = Path.Combine(dir, "model.safetensors");
            SafeTensorLoader.SaveSafeTensors(shardedPath, tensors, maxShardSizeBytes: 48);
            var indexPath = shardedPath + SafeTensorLoader.ShardIndexSuffix;

            // Deleting the shards that hold none of the requested tensors proves a
            // subset load never touches them ("w.b"/"w.a" live in shard 1 only).
            File.Delete(Path.Combine(dir, "model-00002-of-00003.safetensors"));
            File.Delete(Path.Combine(dir, "model-00003-of-00003.safetensors"));

            var subset = SafeTensorLoader.LoadTensorDictionary(indexPath, ["w.a", "w.b"]);
            Assert.Equal(2, subset.Count);
            Assert.Equal(
                MakeTensor("w.a", 10f).Data.AccessRawMemory().ToArray(),
                subset["w.a"].AccessRawMemory().ToArray());

            // Requesting a tensor whose shard is gone names both tensor and file.
            var missing = Assert.Throws<FileNotFoundException>(
                () => SafeTensorLoader.LoadSafeTensors(indexPath, ["w.d"]));
            Assert.Contains("model-00002-of-00003.safetensors", missing.Message);
            Assert.Contains("w.d", missing.Message);

            // Requesting a name absent from the weight_map names the tensor.
            var unknown = Assert.Throws<KeyNotFoundException>(
                () => SafeTensorLoader.LoadSafeTensors(indexPath, ["w.ghost"]));
            Assert.Contains("w.ghost", unknown.Message);

            // Subset loads work on plain single files as well, with the same
            // loud failure for unknown names.
            var singlePath = Path.Combine(dir, "single.safetensors");
            SafeTensorLoader.SaveSafeTensors(singlePath, tensors);
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

            // A valid re-save still replaces the content in place.
            SafeTensorLoader.SaveSafeTensors(path, [MakeTensor("b", 10f)]);
            Assert.Equal("b", Assert.Single(SafeTensorLoader.LoadSafeTensors(path)).Name);
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
            SafeTensorLoader.SaveSafeTensors(Path.Combine(dir, "s1.safetensors"),
                [MakeTensor("a", 0f), MakeTensor("b", 10f)]);
            SafeTensorLoader.SaveSafeTensors(Path.Combine(dir, "s2.safetensors"),
                [MakeTensor("c", 20f)]);

            var indexPath = Path.Combine(dir, "model.safetensors.index.json");
            void WriteIndex(string weightMapJson) =>
                File.WriteAllText(indexPath, $"{{\"weight_map\":{weightMapJson}}}");

            // Baseline: a consistent hand-written index loads (shard file names
            // need not follow the -0000x-of-000NN convention).
            WriteIndex("{\"a\":\"s1.safetensors\",\"b\":\"s1.safetensors\",\"c\":\"s2.safetensors\"}");
            Assert.Equal(3, SafeTensorLoader.LoadSafeTensors(indexPath).Count);

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
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}

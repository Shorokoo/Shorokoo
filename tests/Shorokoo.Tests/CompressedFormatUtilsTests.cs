using Shorokoo.Core.Factory;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage-purpose tests for <see cref="CompressedFormatUtils"/> covering the
/// three format families it supports: raw zstd byte streams, .zsrk compressed
/// architecture, and .zsafetensor compressed tensor archives. Mirrors the most
/// coverage-rich roundtrip tests in <c>CompressedFormatUtilsTests</c> so the
/// curated Coverage scan picks the file up.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class CompressedFormatUtilsCoverageTests
{
    private static readonly string TempDir = Path.Combine(Path.GetTempPath(), "ShorokooCompressedCoverageTests");

    public CompressedFormatUtilsCoverageTests()
    {
        Directory.CreateDirectory(TempDir);
    }

    [Fact]
    public void TestCompressedFormatUtilsCoverage()
    {
        // ──────────────────────────────────────────────────────────────────
        // Raw zstd: Compress / Decompress / CompressToFile / DecompressFile
        // ──────────────────────────────────────────────────────────────────
        var originalBytes = new byte[10000];
        for (int i = 0; i < originalBytes.Length; i++)
            originalBytes[i] = (byte)(i % 10);
        var compressed = CompressedFormatUtils.Compress(originalBytes);
        Assert.True(compressed.Length < originalBytes.Length);
        Assert.Equal(originalBytes, CompressedFormatUtils.Decompress(compressed));

        var zstPath = Path.Combine(TempDir, "test_roundtrip.zst");
        try
        {
            CompressedFormatUtils.CompressToFile(zstPath, originalBytes);
            Assert.Equal(originalBytes, CompressedFormatUtils.DecompressFile(zstPath));
        }
        finally { if (File.Exists(zstPath)) File.Delete(zstPath); }

        Assert.Throws<FileNotFoundException>(
            () => CompressedFormatUtils.DecompressFile(Path.Combine(TempDir, "nope.zst")));

        // Stream variants: CompressToStream + DecompressStream
        using (var memStream = new MemoryStream())
        {
            CompressedFormatUtils.CompressToStream(memStream, originalBytes);
            memStream.Position = 0;
            Assert.Equal(originalBytes, CompressedFormatUtils.DecompressStream(memStream));
        }

        // ──────────────────────────────────────────────────────────────────
        // Architecture: SaveFastGraphTo{File,Binary} / LoadFastGraphFrom{File,Binary}
        // ──────────────────────────────────────────────────────────────────
        var input = InputTensor<float32>("input");
        var output = input + Scalar(1.0f);
        var graph = new InternalComputationGraph([input], [output]);
        var fastGraph = (graph);

        // SaveFastGraphToFile + matching LoadFastGraphFromFile (v2 container).
        // .zsrk written this way is also what ToJson / GetNodeAndTensorNameListing
        // read — they extract the ONNX payload from any .srk layout by content.
        var zsrkPath = Path.Combine(TempDir, "test_arch.zsrk");
        var zsrkPath2 = Path.Combine(TempDir, "test_arch2.zsrk");
        var binPath = Path.Combine(TempDir, "test_arch.bin");
        try
        {
            CompressedFormatUtils.SaveFastGraphToFile(zsrkPath, fastGraph);
            var loaded = CompressedFormatUtils.LoadFastGraphFromFile(zsrkPath);
            Assert.Equal(graph.InputTensors.Count(), loaded.ToInternal().InputTensors.Count());
            Assert.Equal(graph.OutputTensors.Count(), loaded.ToInternal().OutputTensors.Count());

            var uncompressedBytes = CompressedFormatUtils.SaveFastGraphToBinary(fastGraph);
            File.WriteAllBytes(binPath, uncompressedBytes);

            // ──────────────────────────────────────────────────────────────
            // JSON helpers on SaveFastGraphToFile output (single-wrap).
            // ToJson / SaveAsJson / CompareJson / FindFirstJsonDiff /
            // GetNodeAndTensorNameListing.
            // ──────────────────────────────────────────────────────────────
            var json = CompressedFormatUtils.ToJson(zsrkPath);
            Assert.False(string.IsNullOrEmpty(json));
            Assert.Contains("Graph", json);

            // SaveAsJson with explicit targetPath.
            var jsonPath = Path.Combine(TempDir, "test_arch.json");
            try
            {
                var savedPath = CompressedFormatUtils.SaveAsJson(zsrkPath, jsonPath);
                Assert.Equal(jsonPath, savedPath);
                Assert.True(File.Exists(jsonPath));
                // SaveAsJson with null targetPath derives from sourcePath.
                var derivedPath = CompressedFormatUtils.SaveAsJson(zsrkPath);
                Assert.True(File.Exists(derivedPath));
                Assert.EndsWith(".json", derivedPath);
                if (File.Exists(derivedPath)) File.Delete(derivedPath);
            }
            finally { if (File.Exists(jsonPath)) File.Delete(jsonPath); }

            // Save a second graph that differs in structure → CompareJson false
            // and FindFirstJsonDiff returns the first differing line.
            var input2 = InputTensor<float32>("input");
            var output2 = input2 * Scalar(2.0f);
            var graph2 = (new InternalComputationGraph([input2], [output2]));
            CompressedFormatUtils.SaveFastGraphToFile(zsrkPath2, graph2);

            Assert.True(CompressedFormatUtils.CompareJson(zsrkPath, zsrkPath));
            Assert.False(CompressedFormatUtils.CompareJson(zsrkPath, zsrkPath2));

            Assert.Null(CompressedFormatUtils.FindFirstJsonDiff(zsrkPath, zsrkPath));
            var diff = CompressedFormatUtils.FindFirstJsonDiff(zsrkPath, zsrkPath2);
            Assert.NotNull(diff);
            Assert.True(diff!.Value.LineNumber > 0);

            // GetNodeAndTensorNameListing.
            var listing = CompressedFormatUtils.GetNodeAndTensorNameListing(zsrkPath);
            Assert.False(string.IsNullOrEmpty(listing));
            Assert.Contains("\n", listing);
        }
        finally
        {
            if (File.Exists(zsrkPath)) File.Delete(zsrkPath);
            if (File.Exists(zsrkPath2)) File.Delete(zsrkPath2);
            if (File.Exists(binPath)) File.Delete(binPath);
        }

        Assert.Throws<FileNotFoundException>(
            () => CompressedFormatUtils.LoadFastGraphFromFile(Path.Combine(TempDir, "nope.zsrk")));

        // ──────────────────────────────────────────────────────────────────
        // SafeTensors: SaveCompressedSafeTensors / Load* family +
        // SaveCompressedModelParamSet / LoadCompressedModelParamSet.
        // ──────────────────────────────────────────────────────────────────
        var t1 = TensorData([2, 3], 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);
        var t2 = TensorData([3], 7.0f, 8.0f, 9.0f);
        var tensors = new List<SafeTensor>
        {
            new SafeTensor("tensor1", t1, "F32", t1.Shape.Dims),
            new SafeTensor("tensor2", t2, "F32", t2.Shape.Dims),
        };
        var zsafePath = Path.Combine(TempDir, "test_tensors.zsafetensor");
        var zsafeSinglePath = Path.Combine(TempDir, "test_single.zsafetensor");
        var paramSetPath = Path.Combine(TempDir, "test_params.zsafetensor");
        try
        {
            CompressedFormatUtils.SaveCompressedSafeTensors(zsafePath, tensors);
            Assert.Equal(2, CompressedFormatUtils.LoadCompressedSafeTensors(zsafePath).Count);
            var dict = CompressedFormatUtils.LoadCompressedTensorDictionary(zsafePath);
            Assert.True(dict.ContainsKey("tensor1") && dict.ContainsKey("tensor2"));
            Assert.Throws<InvalidOperationException>(
                () => CompressedFormatUtils.LoadCompressedSingleTensor(zsafePath));

            var singleTensor = new List<SafeTensor>
            {
                new SafeTensor("only", t1, "F32", t1.Shape.Dims),
            };
            CompressedFormatUtils.SaveCompressedSafeTensors(zsafeSinglePath, singleTensor);
            var loadedSingle = CompressedFormatUtils.LoadCompressedSingleTensor(zsafeSinglePath);
            Assert.Equal(t1.Shape.Dims, loadedSingle.Shape.Dims);

            // SaveCompressedModelParamSet + LoadCompressedModelParamSet.
            var paramList = new ModelParamList(
                new (string name, TensorData data)[] { ("p1", t1), ("p2", t2) },
                ModelParamType.TrainableParam);
            CompressedFormatUtils.SaveCompressedModelParamSet(paramSetPath, paramList);
            var loadedParams = CompressedFormatUtils.LoadCompressedModelParamSet(paramSetPath);
            Assert.Equal(2, loadedParams.ModelParams.Length);
            Assert.NotNull(loadedParams.Find("p1"));
            Assert.NotNull(loadedParams.Find("p2"));
        }
        finally
        {
            if (File.Exists(zsafePath)) File.Delete(zsafePath);
            if (File.Exists(zsafeSinglePath)) File.Delete(zsafeSinglePath);
            if (File.Exists(paramSetPath)) File.Delete(paramSetPath);
        }

        // ──────────────────────────────────────────────────────────────────
        // Format-detection helpers.
        // ──────────────────────────────────────────────────────────────────
        Assert.True(CompressedFormatUtils.IsCompressedSafeTensor("foo.zsafetensor"));
        Assert.False(CompressedFormatUtils.IsCompressedSafeTensor("foo.safetensors"));
    }

    // ──────────────────────────────────────────────────────────────────────
    // .srk v2 container (issue #34): self-describing header, stage marker,
    // single header-declared compression layer, v1 content-sniffing shim.
    // ──────────────────────────────────────────────────────────────────────

    /// <summary>Builds the same small graph at all three lifecycle stages.</summary>
    private static (ComputationGraph Module, ComputationGraph Arch, ComputationGraph Model)
        BuildStageGraphs()
    {
        var moduleGraph = ScalarMultiplyModel.ComputationGraph;
        var arch = moduleGraph.ToConcreteArchitecture(
            moduleGraph.FromOrderedInputs([TensorData([2], 1.0f, 2.0f)]));
        var model = arch.ToConcreteModel();
        return (moduleGraph, arch, model);
    }

    /// <summary>
    /// A v2 file round-trips (save → load → identical graph) with and without
    /// compression for all three stages, and its header records srkVersion, stage,
    /// compression and producer info. Also covers stage detection itself.
    /// </summary>
    [Fact]
    public void TestSrkV2RoundtripAllStagesAndHeader()
    {
        var (moduleGraph, arch, model) = BuildStageGraphs();

        // Op-scan detection agrees with the stamped kind at every stage.
        Assert.Equal(GraphKind.Module, SrkFileFormat.DetectStage(moduleGraph.ToInternal()));
        Assert.Equal(GraphKind.ConcreteArchitecture, SrkFileFormat.DetectStage(arch.ToInternal()));
        Assert.Equal(GraphKind.ConcreteModel, SrkFileFormat.DetectStage(model.ToInternal()));

        (ComputationGraph Graph, GraphKind Stage)[] stages =
            [(moduleGraph, GraphKind.Module),
             (arch, GraphKind.ConcreteArchitecture),
             (model, GraphKind.ConcreteModel)];
        bool[] compressionModes = [true, false];

        foreach (var (graph, stage) in stages)
        foreach (var compressed in compressionModes)
        {
            var bytes = CompressedFormatUtils.SaveFastGraphToBinary(graph, compressed);
            Assert.True(SrkFileFormat.IsSrkV2(bytes));

            var header = SrkFileFormat.TryReadHeader(bytes);
            Assert.NotNull(header);
            Assert.Equal(SrkFileFormat.CurrentVersion, header!.SrkVersion);
            Assert.Equal(SrkFileFormat.StageName(stage), header.Stage);
            Assert.Equal(stage, header.TryGetStage());
            Assert.Equal(compressed ? "zstd" : "none", header.Compression);
            Assert.False(string.IsNullOrEmpty(header.PayloadSha256));
            Assert.NotNull(header.Producer);
            // The header records the same framework version the ONNX exporter stamps as
            // producer_version — one source of truth, no "+build-metadata".
            Assert.Equal(Shorokoo.ShorokooVersion.VersionString, header.Producer!.Shorokoo);
            Assert.True(header.Producer.IrVersion > 0);
            Assert.NotNull(header.Producer.Opsets);
            Assert.NotEmpty(header.Producer.Opsets!);

            // The loaded graph comes back at the same lifecycle stage. (Structural
            // JSON identity does not survive a reload — the importer assigns fresh
            // node/tensor keys — so graph identity is asserted below by bit-identical
            // execution of the concrete model instead.)
            var reloaded = CompressedFormatUtils.LoadFastGraphFromBinary(bytes);
            Assert.NotEmpty(reloaded.ToInternal().Nodes);
            Assert.Equal(stage, reloaded.Kind);
            Assert.Equal(stage, SrkFileFormat.DetectStage(reloaded.ToInternal()));

            if (stage == GraphKind.ConcreteModel && compressed)
            {
                var input = TensorData([2], 1.0f, 2.0f);
                var direct = ComputeContext.Default.Execute(graph, input)[0]
                    .ToTensorData().AccessRawMemory().ToArray();
                var roundtrip = ComputeContext.Default.Execute(reloaded, input)[0]
                    .ToTensorData().AccessRawMemory().ToArray();
                Assert.Equal(direct, roundtrip);
            }
        }

        // A reloaded module graph continues through the normal lowering pipeline.
        var reloadedModule = CompressedFormatUtils.LoadFastGraphFromBinary(
            CompressedFormatUtils.SaveFastGraphToBinary(moduleGraph));
        var rearch = reloadedModule.ToConcreteArchitecture(
            reloadedModule.FromOrderedInputs([TensorData([2], 1.0f, 2.0f)]));
        Assert.Equal(GraphKind.ConcreteArchitecture, rearch.Kind);
        Assert.Equal(GraphKind.ConcreteArchitecture, SrkFileFormat.DetectStage(rearch.ToInternal()));
    }

    /// <summary>
    /// The three historical v1 layouts — bare protobuf, single-Zstd
    /// (v1 SaveFastGraphToFile) and double-Zstd (the retired architecture writer) —
    /// still load through the content-sniffing shim, from bytes and from files,
    /// regardless of extension.
    /// </summary>
    [Fact]
    public void TestSrkV1LegacyLayoutsLoadThroughShim()
    {
        var (_, arch, _) = BuildStageGraphs();

        // Reconstruct the exact v1 byte layouts from the v2 payload: the payload of an
        // uncompressed v2 container IS the bare-protobuf v1 layout the old writers
        // produced; the old single/double-Zstd layouts wrapped it in 1 / 2 Zstd frames.
        var v2Bytes = CompressedFormatUtils.SaveFastGraphToBinary(arch, compressed: false);
        var bareProtobuf = SrkFileFormat.Read(v2Bytes).OnnxBytes;
        var singleZstd = CompressedFormatUtils.Compress(bareProtobuf);
        var doubleZstd = CompressedFormatUtils.Compress(singleZstd);

        var referenceNodeCount = CompressedFormatUtils.LoadFastGraphFromBinary(v2Bytes).ToInternal().Nodes.Count;

        (string Name, byte[] Bytes)[] layouts =
            [("bare", bareProtobuf), ("single-zstd", singleZstd), ("double-zstd", doubleZstd)];
        foreach (var (name, bytes) in layouts)
        {
            Assert.Null(SrkFileFormat.TryReadHeader(bytes));

            var fromBinary = CompressedFormatUtils.LoadFastGraphFromBinary(bytes);
            Assert.Equal(referenceNodeCount, fromBinary.ToInternal().Nodes.Count);
            Assert.Equal(GraphKind.ConcreteArchitecture, SrkFileFormat.DetectStage(fromBinary.ToInternal()));

            // File load with a deliberately "wrong" extension: content decides.
            var path = Path.Combine(TempDir, $"v1_{name}.zsrk");
            try
            {
                File.WriteAllBytes(path, bytes);
                var fromFile = CompressedFormatUtils.LoadFastGraphFromFile(path);
                Assert.Equal(referenceNodeCount, fromFile.ToInternal().Nodes.Count);

                // The JSON introspection helpers accept every layout too.
                Assert.Contains("Graph", CompressedFormatUtils.ToJson(path));
            }
            finally { if (File.Exists(path)) File.Delete(path); }
        }
    }

    /// <summary>
    /// A renamed v2 file (wrong or no extension) loads identically — content, not
    /// extension, decides how the file parses.
    /// </summary>
    [Fact]
    public void TestSrkV2RenamedFileLoadsIdentically()
    {
        var (_, arch, _) = BuildStageGraphs();

        var zsrkPath = Path.Combine(TempDir, "renamed_src.zsrk");
        string[] renamedPaths =
            [Path.Combine(TempDir, "renamed_copy.srk"),
             Path.Combine(TempDir, "renamed_copy.bin"),
             Path.Combine(TempDir, "renamed_copy")];
        try
        {
            CompressedFormatUtils.SaveFastGraphToFile(zsrkPath, arch, compressed: true, overrideExtension: false);
            var referenceJson = CompressedFormatUtils.ToJson(zsrkPath);

            foreach (var renamed in renamedPaths)
            {
                File.Copy(zsrkPath, renamed, overwrite: true);
                var loaded = CompressedFormatUtils.LoadFastGraphFromFile(renamed);
                Assert.NotEmpty(loaded.ToInternal().Nodes);
                Assert.Equal(referenceJson, CompressedFormatUtils.ToJson(renamed));
            }
        }
        finally
        {
            if (File.Exists(zsrkPath)) File.Delete(zsrkPath);
            foreach (var renamed in renamedPaths)
                if (File.Exists(renamed)) File.Delete(renamed);
        }
    }

    /// <summary>Hand-assembles a v2 container around an arbitrary header, for fault injection.</summary>
    private static byte[] BuildRawSrkContainer(string headerJson, byte[] payload)
    {
        var headerBytes = System.Text.Encoding.UTF8.GetBytes(headerJson);
        var result = new byte[4 + 2 + headerBytes.Length + payload.Length];
        result[0] = (byte)'S'; result[1] = (byte)'R'; result[2] = (byte)'K'; result[3] = 2;
        result[4] = (byte)(headerBytes.Length & 0xFF);
        result[5] = (byte)(headerBytes.Length >> 8);
        headerBytes.CopyTo(result, 6);
        payload.CopyTo(result, 6 + headerBytes.Length);
        return result;
    }

    private static string Sha256Hex(byte[] payload)
        => Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(payload)).ToLowerInvariant();

    /// <summary>
    /// Corrupt, truncated and mismatching v2 files fail loudly with a message naming
    /// the file (or origin) and the failure: payload corruption → SHA-256 mismatch;
    /// truncation inside the header → truncation error; malformed header JSON,
    /// unsupported future container version and unknown compression each name the
    /// offending value.
    /// </summary>
    [Fact]
    public void TestSrkV2CorruptionFailsLoudly()
    {
        var (_, arch, _) = BuildStageGraphs();
        var bytes = CompressedFormatUtils.SaveFastGraphToBinary(arch, compressed: true);

        // Payload corruption: flip the last byte → SHA-256 mismatch naming the origin.
        var corrupt = (byte[])bytes.Clone();
        corrupt[^1] ^= 0xFF;
        var corruptPath = Path.Combine(TempDir, "corrupt.zsrk");
        try
        {
            File.WriteAllBytes(corruptPath, corrupt);
            var ex = Assert.Throws<InvalidDataException>(
                () => CompressedFormatUtils.LoadFastGraphFromFile(corruptPath));
            Assert.Contains("SHA-256 mismatch", ex.Message);
            Assert.Contains(corruptPath, ex.Message);
        }
        finally { if (File.Exists(corruptPath)) File.Delete(corruptPath); }

        // Truncated payload → SHA-256 mismatch (truncation detection).
        var truncated = bytes[..^16];
        var exTrunc = Assert.Throws<InvalidDataException>(
            () => CompressedFormatUtils.LoadFastGraphFromBinary(truncated));
        Assert.Contains("corrupt or truncated", exTrunc.Message);

        // Truncated inside the header → explicit truncation error.
        var exHeader = Assert.Throws<InvalidDataException>(
            () => CompressedFormatUtils.LoadFastGraphFromBinary(bytes[..8]));
        Assert.Contains("truncated", exHeader.Message);

        // Malformed header JSON.
        byte[] payload = [.. bytes.Skip(6 + (bytes[4] | (bytes[5] << 8)))];
        var exJson = Assert.Throws<InvalidDataException>(() =>
            CompressedFormatUtils.LoadFastGraphFromBinary(BuildRawSrkContainer("{not json", payload)));
        Assert.Contains("header", exJson.Message);

        // Future container version is refused up front.
        var exVersion = Assert.Throws<InvalidDataException>(() =>
            CompressedFormatUtils.LoadFastGraphFromBinary(BuildRawSrkContainer(
                $"{{\"srkVersion\":3,\"stage\":\"concrete-architecture\",\"compression\":\"zstd\",\"payloadSha256\":\"{Sha256Hex(payload)}\"}}",
                payload)));
        Assert.Contains("version 3", exVersion.Message);

        // Unknown compression scheme is named in the error.
        var exCompression = Assert.Throws<InvalidDataException>(() =>
            CompressedFormatUtils.LoadFastGraphFromBinary(BuildRawSrkContainer(
                $"{{\"srkVersion\":2,\"stage\":\"concrete-architecture\",\"compression\":\"lz4\",\"payloadSha256\":\"{Sha256Hex(payload)}\"}}",
                payload)));
        Assert.Contains("lz4", exCompression.Message);

        // Header missing srkVersion → a missing-field diagnostic, not a bogus "version 0
        // ... newer framework version" message.
        var exMissing = Assert.Throws<InvalidDataException>(() =>
            CompressedFormatUtils.LoadFastGraphFromBinary(BuildRawSrkContainer(
                $"{{\"stage\":\"concrete-architecture\",\"compression\":\"none\",\"payloadSha256\":\"{Sha256Hex(payload)}\"}}",
                payload)));
        Assert.Contains("'srkVersion'", exMissing.Message);
        Assert.Contains("missing or zero", exMissing.Message);

        // An older (but positive) container version reads as "older, unsupported", not "newer".
        var exOlder = Assert.Throws<InvalidDataException>(() =>
            CompressedFormatUtils.LoadFastGraphFromBinary(BuildRawSrkContainer(
                $"{{\"srkVersion\":1,\"stage\":\"concrete-architecture\",\"compression\":\"none\",\"payloadSha256\":\"{Sha256Hex(payload)}\"}}",
                payload)));
        Assert.Contains("version 1", exOlder.Message);
        Assert.Contains("older", exOlder.Message);
    }

    /// <summary>
    /// The header-peek API reads a real v2 header (identifying stage/compression/producer)
    /// and returns null for legacy v1 data that carries no container header.
    /// </summary>
    [Fact]
    public void TestSrkHeaderPeekIdentifiesContainer()
    {
        var (_, arch, _) = BuildStageGraphs();

        var v2Path = Path.Combine(TempDir, "peek.zsrk");
        var v1Path = Path.Combine(TempDir, "peek_v1.zsrk");
        try
        {
            CompressedFormatUtils.SaveFastGraphToFile(v2Path, arch, compressed: true, overrideExtension: false);
            var header = SrkFileFormat.TryReadHeaderFromFile(v2Path);
            Assert.NotNull(header);
            Assert.Equal(SrkFileFormat.CurrentVersion, header!.SrkVersion);
            Assert.Equal(GraphKind.ConcreteArchitecture, header.TryGetStage());
            Assert.Equal("zstd", header.Compression);

            // A legacy v1 file (single-Zstd bare protobuf) has no container header → null.
            var bare = SrkFileFormat.Read(
                CompressedFormatUtils.SaveFastGraphToBinary(arch, compressed: false)).OnnxBytes;
            File.WriteAllBytes(v1Path, CompressedFormatUtils.Compress(bare));
            Assert.Null(SrkFileFormat.TryReadHeaderFromFile(v1Path));
        }
        finally
        {
            if (File.Exists(v2Path)) File.Delete(v2Path);
            if (File.Exists(v1Path)) File.Delete(v1Path);
        }
    }

    /// <summary>
    /// Loading a module-stage graph through an API that requires a concrete graph
    /// produces a clear stage-mismatch error at load time — from the header for v2
    /// files (before the payload is parsed) and from the detected stage for legacy v1
    /// data. Matching stages load normally.
    /// </summary>
    [Fact]
    public void TestSrkStageMismatchIsRejectedAtLoadTime()
    {
        var (moduleGraph, arch, model) = BuildStageGraphs();

        // v2: header-based refusal, error names both stages.
        var moduleBytes = CompressedFormatUtils.SaveFastGraphToBinary(moduleGraph);
        var ex = Assert.Throws<InvalidOperationException>(() =>
            CompressedFormatUtils.LoadFastGraphFromBinary(moduleBytes, requiredStage: GraphKind.ConcreteModel));
        Assert.Contains("'module'", ex.Message);
        Assert.Contains("'concrete-model'", ex.Message);

        // File variant names the file.
        var modulePath = Path.Combine(TempDir, "stage_module.zsrk");
        try
        {
            CompressedFormatUtils.SaveFastGraphToFile(modulePath, moduleGraph, compressed: true, overrideExtension: false);
            var exFile = Assert.Throws<InvalidOperationException>(() =>
                CompressedFormatUtils.LoadFastGraphFromFile(modulePath, requiredStage: GraphKind.ConcreteModel));
            Assert.Contains(modulePath, exFile.Message);
        }
        finally { if (File.Exists(modulePath)) File.Delete(modulePath); }

        // v1 shim: no header, stage detected from the loaded graph.
        var v1ModuleBytes = CompressedFormatUtils.Compress(
            SrkFileFormat.Read(CompressedFormatUtils.SaveFastGraphToBinary(moduleGraph, compressed: false)).OnnxBytes);
        Assert.Throws<InvalidOperationException>(() =>
            CompressedFormatUtils.LoadFastGraphFromBinary(v1ModuleBytes, requiredStage: GraphKind.ConcreteModel));

        // Matching required stages load fine, for all three stages.
        Assert.NotEmpty(CompressedFormatUtils.LoadFastGraphFromBinary(
            moduleBytes, requiredStage: GraphKind.Module).ToInternal().Nodes);
        Assert.NotEmpty(CompressedFormatUtils.LoadFastGraphFromBinary(
            CompressedFormatUtils.SaveFastGraphToBinary(arch), requiredStage: GraphKind.ConcreteArchitecture).ToInternal().Nodes);
        Assert.NotEmpty(CompressedFormatUtils.LoadFastGraphFromBinary(
            CompressedFormatUtils.SaveFastGraphToBinary(model), requiredStage: GraphKind.ConcreteModel).ToInternal().Nodes);
    }

    /// <summary>
    /// Saving with the default extension normalization must not delete an unrelated file
    /// sitting at the caller's original path: SaveFastGraphToFile("x.onnx", …) writes
    /// x.zsrk and leaves x.onnx untouched.
    /// </summary>
    [Fact]
    public void TestSaveFastGraphToFileDoesNotDeleteUnrelatedFile()
    {
        var (_, arch, _) = BuildStageGraphs();
        var onnxPath = Path.Combine(TempDir, "sentinel_model.onnx");
        var zsrkPath = Path.ChangeExtension(onnxPath, ".zsrk");
        try
        {
            byte[] sentinel = [1, 2, 3, 4];
            File.WriteAllBytes(onnxPath, sentinel);

            // Default overrideExtension:true normalizes .onnx → .zsrk. The original
            // .onnx is a different file and must survive.
            var written = CompressedFormatUtils.SaveFastGraphToFile(onnxPath, arch, compressed: true);

            Assert.Equal(zsrkPath, written);
            Assert.True(File.Exists(onnxPath), "SaveFastGraphToFile deleted the caller's unrelated .onnx file");
            Assert.Equal(sentinel, File.ReadAllBytes(onnxPath));
            Assert.True(File.Exists(zsrkPath));
            Assert.NotEmpty(CompressedFormatUtils.LoadFastGraphFromFile(zsrkPath).ToInternal().Nodes);
        }
        finally
        {
            if (File.Exists(onnxPath)) File.Delete(onnxPath);
            if (File.Exists(zsrkPath)) File.Delete(zsrkPath);
        }
    }

    /// <summary>
    /// A container whose magic version byte is newer than this build understands fails with a
    /// clear "unsupported major version" error — before header parsing — rather than falling
    /// through to the legacy content shim and dying as unparseable protobuf. TryReadHeader and
    /// its file variant reject it the same way instead of misreporting it as legacy (null).
    /// </summary>
    [Fact]
    public void TestSrkFutureContainerVersionFailsClearly()
    {
        var (_, arch, _) = BuildStageGraphs();

        // A real v2 container with the magic's major-version byte bumped 2 → 3.
        var v3 = (byte[])CompressedFormatUtils.SaveFastGraphToBinary(arch, compressed: true).Clone();
        v3[3] = 3;

        Assert.False(SrkFileFormat.IsSrkV2(v3));

        var exLoad = Assert.Throws<InvalidDataException>(
            () => CompressedFormatUtils.LoadFastGraphFromBinary(v3));
        Assert.Contains("major version 3", exLoad.Message);

        var exHeader = Assert.Throws<InvalidDataException>(() => SrkFileFormat.TryReadHeader(v3));
        Assert.Contains("major version 3", exHeader.Message);

        var path = Path.Combine(TempDir, "future.srk");
        try
        {
            File.WriteAllBytes(path, v3);
            var exFile = Assert.Throws<InvalidDataException>(() => SrkFileFormat.TryReadHeaderFromFile(path));
            Assert.Contains(path, exFile.Message);
            Assert.Contains("major version 3", exFile.Message);
            Assert.Throws<InvalidDataException>(() => CompressedFormatUtils.LoadFastGraphFromFile(path));
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    /// <summary>
    /// Corrupt legacy (v1) data fails loudly with an InvalidDataException naming the origin:
    /// an empty file, non-.srk garbage, and bytes still Zstd-framed after the maximum legacy
    /// layer count each produce a clear message instead of a bare protobuf/index exception.
    /// </summary>
    [Fact]
    public void TestSrkV1CorruptDataFailsLoudly()
    {
        // Empty input.
        var exEmpty = Assert.Throws<InvalidDataException>(
            () => CompressedFormatUtils.LoadFastGraphFromBinary([]));
        Assert.Contains("empty", exEmpty.Message);

        // Non-.srk garbage (not SRK-prefixed, not Zstd, not valid protobuf): the importer
        // failure is wrapped with the origin named.
        var garbage = new byte[64];
        Array.Fill(garbage, (byte)0x77);
        var garbagePath = Path.Combine(TempDir, "garbage.srk");
        try
        {
            File.WriteAllBytes(garbagePath, garbage);
            var exGarbage = Assert.Throws<InvalidDataException>(
                () => CompressedFormatUtils.LoadFastGraphFromFile(garbagePath));
            Assert.Contains(garbagePath, exGarbage.Message);
        }
        finally { if (File.Exists(garbagePath)) File.Delete(garbagePath); }

        // Triple-Zstd-wrapped: the shim unwraps at most two layers, so the remainder is
        // still Zstd-framed → a clear "still Zstd-compressed" error, not a cryptic crash.
        var (_, arch, _) = BuildStageGraphs();
        var bare = SrkFileFormat.Read(
            CompressedFormatUtils.SaveFastGraphToBinary(arch, compressed: false)).OnnxBytes;
        var triple = CompressedFormatUtils.Compress(
            CompressedFormatUtils.Compress(CompressedFormatUtils.Compress(bare)));
        var exTriple = Assert.Throws<InvalidDataException>(
            () => CompressedFormatUtils.LoadFastGraphFromBinary(triple));
        Assert.Contains("still Zstd-compressed", exTriple.Message);
    }

    // ──────────────────────────────────────────────────────────────────────
    // Checkpoint.Inspect (issue #57): identify and summarize artifacts from
    // headers/prefixes only, never loading tensor payloads.
    // ──────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Inspect identifies .srk graph files of every layout. v2 containers (compressed and
    /// uncompressed, i.e. SaveFastGraphToFile's two modes) report the header metadata that
    /// was written — stage, compression, producer, payload hash. Legacy v1 layouts are
    /// sniffed with bounded reads. A corrupt payload does not disturb inspection (payload
    /// bytes are never read — the same file fails a full load on its hash check), and
    /// corrupt headers / future versions yield structured results instead of exceptions.
    /// </summary>
    [Fact]
    public void TestCheckpointInspectSrkArtifacts()
    {
        var (_, arch, _) = BuildStageGraphs();

        // v2 round-trip, compressed and uncompressed.
        bool[] compressionModes = [true, false];
        foreach (var compressed in compressionModes)
        {
            var path = Path.Combine(TempDir, $"inspect_{compressed}.zsrk");
            try
            {
                CompressedFormatUtils.SaveFastGraphToFile(path, arch, compressed, overrideExtension: false);
                var result = Checkpoint.Inspect(path);

                Assert.Equal(ArtifactKind.SrkGraph, result.Kind);
                Assert.Equal(path, result.FilePath);
                Assert.Equal(new FileInfo(path).Length, result.FileSizeBytes);
                Assert.NotNull(result.Srk);
                Assert.Null(result.SafeTensors);
                Assert.Null(result.TrainingCheckpoint);
                Assert.Empty(result.Observations);

                var header = result.Srk!.Header;
                Assert.NotNull(header);
                Assert.Equal(SrkFileFormat.CurrentVersion, header!.SrkVersion);
                Assert.Equal(GraphKind.ConcreteArchitecture, header.TryGetStage());
                Assert.Equal(compressed ? "zstd" : "none", header.Compression);
                Assert.False(string.IsNullOrEmpty(header.PayloadSha256));
                Assert.Equal(Shorokoo.ShorokooVersion.VersionString, header.Producer!.Shorokoo);
                Assert.Null(result.Srk.LegacyLayout);
                Assert.True(result.Srk.PayloadSizeBytes > 0);

                var text = result.ToString();
                Assert.Contains("concrete-architecture", text);
                Assert.Contains(compressed ? "zstd" : "none", text);

                // Corrupt the payload's last byte: a full load fails on the SHA-256 check,
                // but Inspect — which never touches payload bytes — still reads the header.
                var corrupt = File.ReadAllBytes(path);
                corrupt[^1] ^= 0xFF;
                var corruptPath = Path.Combine(TempDir, $"inspect_corrupt_{compressed}.zsrk");
                try
                {
                    File.WriteAllBytes(corruptPath, corrupt);
                    Assert.Throws<InvalidDataException>(
                        () => CompressedFormatUtils.LoadFastGraphFromFile(corruptPath));
                    var corruptResult = Checkpoint.Inspect(corruptPath);
                    Assert.Equal(ArtifactKind.SrkGraph, corruptResult.Kind);
                    Assert.Equal(header.PayloadSha256, corruptResult.Srk!.Header!.PayloadSha256);
                }
                finally { if (File.Exists(corruptPath)) File.Delete(corruptPath); }
            }
            finally { if (File.Exists(path)) File.Delete(path); }
        }

        // Legacy v1 layouts: bare protobuf, single-Zstd, double-Zstd.
        var v2Bytes = CompressedFormatUtils.SaveFastGraphToBinary(arch, compressed: false);
        var bare = SrkFileFormat.Read(v2Bytes).OnnxBytes;
        (string ExpectedLayout, byte[] Bytes)[] legacy =
            [("bare ONNX protobuf", bare),
             ("single-Zstd", CompressedFormatUtils.Compress(bare)),
             ("double-Zstd", CompressedFormatUtils.Compress(CompressedFormatUtils.Compress(bare)))];
        foreach (var (expectedLayout, bytes) in legacy)
        {
            var path = Path.Combine(TempDir, "inspect_v1.srk");
            try
            {
                File.WriteAllBytes(path, bytes);
                var result = Checkpoint.Inspect(path);
                Assert.Equal(ArtifactKind.SrkGraph, result.Kind);
                Assert.Null(result.Srk!.Header);
                Assert.Equal(expectedLayout, result.Srk.LegacyLayout);
                Assert.Equal(bytes.Length, result.Srk.PayloadSizeBytes);
                Assert.Contains("legacy", result.ToString());
            }
            finally { if (File.Exists(path)) File.Delete(path); }
        }

        // A future container version and a truncated container yield structured results
        // with an observation naming the problem — never an exception.
        var future = (byte[])v2Bytes.Clone();
        future[3] = 3;
        (string Name, byte[] Bytes)[] damaged =
            [("future", future), ("truncated", v2Bytes[..5])];
        foreach (var (name, bytes) in damaged)
        {
            var path = Path.Combine(TempDir, $"inspect_{name}.srk");
            try
            {
                File.WriteAllBytes(path, bytes);
                var result = Checkpoint.Inspect(path);
                Assert.Equal(ArtifactKind.SrkGraph, result.Kind);
                Assert.Null(result.Srk!.Header);
                Assert.Null(result.Srk.LegacyLayout);
                Assert.Contains(result.Observations, o => o.Contains("header is not readable"));
            }
            finally { if (File.Exists(path)) File.Delete(path); }
        }

        // Garbage and empty files: the structured "not recognized" outcome, no exception.
        var garbagePath = Path.Combine(TempDir, "inspect_garbage.bin");
        var emptyPath = Path.Combine(TempDir, "inspect_empty.bin");
        try
        {
            var garbage = new byte[64];
            Array.Fill(garbage, (byte)0x77);
            File.WriteAllBytes(garbagePath, garbage);
            var garbageResult = Checkpoint.Inspect(garbagePath);
            Assert.Equal(ArtifactKind.NotRecognized, garbageResult.Kind);
            Assert.NotEmpty(garbageResult.Observations);
            Assert.Contains("not recognized", garbageResult.ToString());

            File.WriteAllBytes(emptyPath, []);
            var emptyResult = Checkpoint.Inspect(emptyPath);
            Assert.Equal(ArtifactKind.NotRecognized, emptyResult.Kind);
            Assert.Contains(emptyResult.Observations, o => o.Contains("empty"));
        }
        finally
        {
            if (File.Exists(garbagePath)) File.Delete(garbagePath);
            if (File.Exists(emptyPath)) File.Delete(emptyPath);
        }

        // A missing file is the one thing that still throws.
        Assert.Throws<FileNotFoundException>(
            () => Checkpoint.Inspect(Path.Combine(TempDir, "inspect_nope.srk")));
    }

    /// <summary>
    /// Inspect identifies SaveSafeTensors output from the 8-byte length prefix + JSON
    /// header alone: the tensor listing (name, dtype, shape, byte size) and total payload
    /// size match what was written, a rank-0 scalar reports its empty shape, and the cheap
    /// sanity observations fire — declared extents past the end of a truncated file, and
    /// trailing bytes beyond the declared data.
    /// </summary>
    [Fact]
    public void TestCheckpointInspectSafeTensorsArtifacts()
    {
        var t1 = TensorData([2, 3], 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);
        var t2 = TensorData([3], 7.0f, 8.0f, 9.0f);
        var scalar = TensorData([], 42.0f);
        var tensors = new List<SafeTensor>
        {
            new SafeTensor("tensor1", t1, "F32", t1.Shape.Dims),
            new SafeTensor("tensor2", t2, "F32", t2.Shape.Dims),
            new SafeTensor("scalar", scalar, "F32", scalar.Shape.Dims),
        };

        var path = Path.Combine(TempDir, "inspect_weights.safetensors");
        var truncatedPath = Path.Combine(TempDir, "inspect_weights_truncated.safetensors");
        var trailingPath = Path.Combine(TempDir, "inspect_weights_trailing.safetensors");
        try
        {
            SafeTensorLoader.SaveSafeTensors(path, tensors);
            var result = Checkpoint.Inspect(path);

            Assert.Equal(ArtifactKind.SafeTensors, result.Kind);
            Assert.Null(result.Srk);
            Assert.Null(result.TrainingCheckpoint);
            Assert.Empty(result.Observations);

            var st = result.SafeTensors!;
            Assert.True(st.HeaderSizeBytes > 0);
            Assert.Equal(3, st.Tensors.Count);
            Assert.Equal(6 * 4 + 3 * 4 + 4, st.TotalTensorBytes);

            var byName = st.Tensors.ToDictionary(t => t.Name);
            long[] expectedShape1 = [2, 3];
            long[] expectedShape2 = [3];
            Assert.Equal("F32", byName["tensor1"].DType);
            Assert.Equal(expectedShape1, byName["tensor1"].Shape);
            Assert.Equal(24, byName["tensor1"].ByteSize);
            Assert.Equal(expectedShape2, byName["tensor2"].Shape);
            Assert.Empty(byName["scalar"].Shape);   // rank-0 scalar: empty shape, 4 bytes
            Assert.Equal(4, byName["scalar"].ByteSize);

            var text = result.ToString();
            Assert.Contains("SafeTensors", text);
            Assert.Contains("tensor1: F32[2, 3], 24 bytes", text);

            // Truncation: declared extents point past the end of the file → observation,
            // still recognized, no exception.
            var bytes = File.ReadAllBytes(path);
            File.WriteAllBytes(truncatedPath, bytes[..^8]);
            var truncated = Checkpoint.Inspect(truncatedPath);
            Assert.Equal(ArtifactKind.SafeTensors, truncated.Kind);
            Assert.Contains(truncated.Observations, o => o.Contains("past the end"));

            // Trailing bytes beyond the declared data → observation.
            File.WriteAllBytes(trailingPath, [.. bytes, 0, 0, 0, 0]);
            var trailing = Checkpoint.Inspect(trailingPath);
            Assert.Equal(ArtifactKind.SafeTensors, trailing.Kind);
            Assert.Contains(trailing.Observations, o => o.Contains("trailing"));
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
            if (File.Exists(truncatedPath)) File.Delete(truncatedPath);
            if (File.Exists(trailingPath)) File.Delete(trailingPath);
        }
    }

    /// <summary>Hand-assembles a SafeTensors file (8-byte length prefix + JSON header + payload)
    /// around an arbitrary header, for fault injection.</summary>
    private static byte[] BuildRawSafeTensors(string headerJson, byte[] payload)
    {
        var headerBytes = System.Text.Encoding.UTF8.GetBytes(headerJson);
        var result = new byte[8 + headerBytes.Length + payload.Length];
        BitConverter.GetBytes((long)headerBytes.Length).CopyTo(result, 0);
        headerBytes.CopyTo(result, 8);
        payload.CopyTo(result, 8 + headerBytes.Length);
        return result;
    }

    /// <summary>
    /// Inspect stays structured — no exception — on hostile and edge inputs, and reports
    /// them honestly. Regressions pinned: a marker whose declared offset is near
    /// long.MaxValue used to wrap past the bounds guard and crash on the seek; a huge
    /// declared end offset used to wrap the extent arithmetic and misreport truncation as
    /// negative "trailing bytes"; a Zstd file whose decompressed prefix merely starts 0x08
    /// (e.g. a .zsafetensor with header length ≡ 8 mod 256) used to be mislabeled a legacy
    /// .srk graph. Also covers the paths the first round of tests missed: a malformed
    /// marker degrades to plain SafeTensors, a future checkpoint version is observed, and
    /// __metadata__ is surfaced.
    /// </summary>
    [Fact]
    public void TestCheckpointInspectHostileAndEdgeInputs()
    {
        var paths = new List<string>();
        string NextPath(string name)
        {
            var p = Path.Combine(TempDir, name);
            paths.Add(p);
            return p;
        }

        try
        {
            // Marker offset near long.MaxValue: markerStart + 16 wraps, which used to
            // bypass the bounds guard and crash in the seek/read. Iterate because the
            // offset's digits feed back into the header length.
            static string MarkerJson(long start) =>
                $"{{\"__shorokoo_checkpoint__\":{{\"dtype\":\"I64\",\"shape\":[2],\"data_offsets\":[{start},{start + 16}]}}}}";
            var markerHeader = MarkerJson(long.MaxValue / 2);
            for (int i = 0; i < 4; i++)
            {
                long dataStart = 8 + System.Text.Encoding.UTF8.GetByteCount(markerHeader);
                markerHeader = MarkerJson(long.MaxValue - dataStart - 8);
            }
            var overflowMarkerPath = NextPath("hostile_marker_offset.safetensors");
            File.WriteAllBytes(overflowMarkerPath, BuildRawSafeTensors(markerHeader, new byte[32]));
            var overflowMarker = Checkpoint.Inspect(overflowMarkerPath);
            Assert.Equal(ArtifactKind.SafeTensors, overflowMarker.Kind);
            Assert.Null(overflowMarker.TrainingCheckpoint);
            Assert.Contains(overflowMarker.Observations, o => o.Contains("malformed"));

            // Huge declared end offset: dataStart + maxEnd wraps, which used to report
            // nonsense negative "trailing bytes" instead of the truncation warning.
            var hugeEndPath = NextPath("hostile_huge_end.safetensors");
            File.WriteAllBytes(hugeEndPath, BuildRawSafeTensors(
                "{\"t\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,9223372036854775800]}}",
                new byte[8]));
            var hugeEnd = Checkpoint.Inspect(hugeEndPath);
            Assert.Equal(ArtifactKind.SafeTensors, hugeEnd.Kind);
            Assert.Contains(hugeEnd.Observations, o => o.Contains("past the end"));
            Assert.DoesNotContain(hugeEnd.Observations, o => o.Contains("trailing"));

            // Reversed and wrapping data_offsets pairs: flagged as invalid extents with the
            // reported size clamped to zero.
            var badExtentPath = NextPath("hostile_bad_extent.safetensors");
            File.WriteAllBytes(badExtentPath, BuildRawSafeTensors(
                "{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[10,2]}," +
                "\"b\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[-9223372036854775808,8]}}",
                new byte[16]));
            var badExtent = Checkpoint.Inspect(badExtentPath);
            Assert.Equal(ArtifactKind.SafeTensors, badExtent.Kind);
            Assert.Equal(2, badExtent.Observations.Count(o => o.Contains("invalid extent")));
            Assert.All(badExtent.SafeTensors!.Tensors, t => Assert.Equal(0, t.ByteSize));

            // A marker with the wrong dtype degrades to plain SafeTensors + observation.
            var wrongDtypePath = NextPath("hostile_marker_dtype.safetensors");
            File.WriteAllBytes(wrongDtypePath, BuildRawSafeTensors(
                "{\"__shorokoo_checkpoint__\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,16]}}",
                new byte[16]));
            var wrongDtype = Checkpoint.Inspect(wrongDtypePath);
            Assert.Equal(ArtifactKind.SafeTensors, wrongDtype.Kind);
            Assert.Null(wrongDtype.TrainingCheckpoint);
            Assert.Contains(wrongDtype.Observations, o => o.Contains("malformed"));

            // A future checkpoint format version still inspects as a checkpoint (version and
            // step read from the marker) with an observation; a tensor outside the known
            // sections is observed too.
            var w = TensorData([2L], 1.0f, 2.0f);
            var futureCkptPath = NextPath("future_checkpoint.safetensors");
            SafeTensorLoader.SaveSafeTensors(futureCkptPath, new List<SafeTensor>
            {
                new SafeTensor("trainable/w", w, "F32", w.Shape.Dims),
                new SafeTensor("stray", w, "F32", w.Shape.Dims),
                new SafeTensor("__shorokoo_checkpoint__", TensorData([2L], 99L, 3L), "I64", [2L]),
            });
            var futureCkpt = Checkpoint.Inspect(futureCkptPath);
            Assert.Equal(ArtifactKind.TrainingCheckpoint, futureCkpt.Kind);
            Assert.Equal(99, futureCkpt.TrainingCheckpoint!.FormatVersion);
            Assert.Equal(3, futureCkpt.TrainingCheckpoint.Step);
            Assert.Single(futureCkpt.TrainingCheckpoint.Sections["trainable"]);
            Assert.Contains(futureCkpt.Observations, o => o.Contains("format version 99"));
            Assert.Contains(futureCkpt.Observations, o => o.Contains("'stray'"));

            // Zstd-compressed non-ONNX data is NotRecognized — including the near-miss
            // whose decompressed prefix starts 0x08 (a .zsafetensor header-length prefix
            // with headerLen ≡ 8 mod 256), which used to be mislabeled "single-Zstd" .srk.
            byte[] textBytes = System.Text.Encoding.UTF8.GetBytes("clearly not a model, just some text.");
            byte[] nearMiss = [0x08, 0x01, 0, 0, 0, 0, 0, 0, 0x7B, 0x22];
            (string Name, byte[] Inner)[] zstdCases =
                [("zstd_text.bin", textBytes), ("zstd_nearmiss.zsafetensor", nearMiss)];
            foreach (var (name, inner) in zstdCases)
            {
                var p = NextPath(name);
                File.WriteAllBytes(p, CompressedFormatUtils.Compress(inner));
                var r = Checkpoint.Inspect(p);
                Assert.Equal(ArtifactKind.NotRecognized, r.Kind);
                Assert.Contains(r.Observations, o => o.Contains("Zstd frame"));
            }

            // __metadata__ entries are surfaced.
            var metaPath = NextPath("with_metadata.safetensors");
            SafeTensorLoader.SaveSafeTensors(metaPath,
                new List<SafeTensor> { new SafeTensor("w", w, "F32", w.Shape.Dims) },
                new Dictionary<string, object> { ["format"] = "shorokoo-test" });
            var meta = Checkpoint.Inspect(metaPath);
            Assert.Equal(ArtifactKind.SafeTensors, meta.Kind);
            Assert.Equal("shorokoo-test", meta.SafeTensors!.GlobalMetadata!["format"]);
        }
        finally
        {
            foreach (var p in paths)
                if (File.Exists(p)) File.Delete(p);
        }
    }

    private static Tensor<float32> ParamlessDouble(Tensor<float32> x) => x + x;

    /// <summary>
    /// Issue #54: a parameterless module lowers to a concrete architecture whose
    /// op-scan classification says "concrete-model" (there are no MODEL_PARAM nodes
    /// to see), so the stamped kind is the only reliable answer. The writer records
    /// the stamp in the header and the loader stamps Kind from the header, so the
    /// kind survives the .srk round-trip even with no op-scan evidence.
    /// </summary>
    [Fact]
    public void TestStampedKindSurvivesSrkRoundtripWithoutOpScanEvidence()
    {
        var moduleGraph = ModuleFactory.ComputationGraph(
            (Func<Tensor<float32>, Tensor<float32>>)ParamlessDouble);
        Assert.Equal(GraphKind.Module, moduleGraph.Kind);

        var arch = moduleGraph.ToConcreteArchitecture(
            moduleGraph.FromOrderedInputs([TensorData([2L], 1.0f, 2.0f)]));
        Assert.Equal(GraphKind.ConcreteArchitecture, arch.Kind);
        // No trainable params -> op-scanning misclassifies this architecture.
        Assert.Equal(GraphKind.ConcreteModel, SrkFileFormat.DetectStage(arch.ToInternal()));

        var bytes = CompressedFormatUtils.SaveFastGraphToBinary(arch, compressed: false);
        Assert.Equal(GraphKind.ConcreteArchitecture, SrkFileFormat.TryReadHeader(bytes)!.TryGetStage());

        var reloaded = CompressedFormatUtils.LoadFastGraphFromBinary(bytes);
        Assert.Equal(GraphKind.ConcreteArchitecture, reloaded.Kind);

        // The stamped kind is authoritative on the reloaded graph too: it may
        // continue the pipeline exactly like the in-memory architecture.
        Assert.Equal(GraphKind.ConcreteModel, reloaded.ToConcreteModel().Kind);
    }


    /// <summary>
    /// The graph kind rides ONNX serialization as a model metadata tag, so a graph
    /// reloads as the kind it was saved with even as a bare ONNX payload — including
    /// exactly the graphs op-scanning misclassifies (machinery-free module bodies and
    /// parameterless architectures both scan as concrete-model). A tag that is
    /// structurally impossible for the content fails loudly instead of stamping a lie.
    /// </summary>
    [Fact]
    public void TestGraphKindMetadataTagRoundtripsThroughOnnx()
    {
        var moduleGraph = ModuleFactory.ComputationGraph(
            (Func<Tensor<float32>, Tensor<float32>>)ParamlessDouble);
        var arch = moduleGraph.ToConcreteArchitecture(
            moduleGraph.FromOrderedInputs([TensorData([2L], 1.0f, 2.0f)]));

        (ComputationGraph Graph, GraphKind Kind)[] cases =
            [(moduleGraph, GraphKind.Module), (arch, GraphKind.ConcreteArchitecture)];
        foreach (var (graph, kind) in cases)
        {
            var proto = FastOnnxModelBuilder.BuildInternalOnnxModel(graph.ToInternal(), stage: graph.Kind);
            using var ms = new MemoryStream();
            ProtoBuf.Serializer.Serialize(ms, proto);
            var reloaded = OnnxModelImporter.FromOnnxModel(ms.ToArray());
            Assert.Equal(kind, reloaded.Kind);
            // The tag is doing the work: op-scanning the same content misclassifies it.
            Assert.Equal(GraphKind.ConcreteModel, SrkFileFormat.DetectStage(reloaded.ToInternal()));
        }

        // Impossible tag: module machinery tagged concrete-model is refused at import.
        var lyingProto = FastOnnxModelBuilder.BuildInternalOnnxModel(
            ScalarMultiplyModel.ComputationGraph.ToInternal(), stage: GraphKind.ConcreteModel);
        using var lyingMs = new MemoryStream();
        ProtoBuf.Serializer.Serialize(lyingMs, lyingProto);
        var ex = Assert.Throws<InvalidDataException>(
            () => OnnxModelImporter.FromOnnxModel(lyingMs.ToArray()));
        Assert.Contains("shrk_graph_kind", ex.Message);
        Assert.Contains("module-stage op", ex.Message);
    }

}

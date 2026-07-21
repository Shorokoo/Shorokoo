using System.IO.Compression;
using System.Text.Json.Nodes;
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

    // ──────────────────────────────────────────────────────────────────────
    // .skpt single-file checkpoint container (issue #58): STORED zip +
    // config.json manifest, concrete-model save/load with execution parity.
    // ──────────────────────────────────────────────────────────────────────

    /// <summary>Builds a small concrete FCLayer model plus the sample inputs to execute it.</summary>
    private static (ComputationGraph Model, TensorData NumOut, TensorData Input) BuildSkptModel()
    {
        var numOut = TensorData(DType.Int64, [], 4L);
        var input = TensorDataWithSmallVals(DType.Float32, [4L, 4L]);
        var g = FCLayer.ComputationGraph;   // two trainable params: weights [4,4], bias [4]
        var model = g.ToConcreteArchitecture(g.FromOrderedInputs([numOut, input])).ToConcreteModel();
        return (model, numOut, input);
    }

    private static byte[] ExecuteToBytes(ComputationGraph model, TensorData numOut, TensorData input)
        => ComputeContext.Default.Execute(model, numOut, input)[0]
            .ToTensorData().AccessRawMemory().ToArray();

    /// <summary>The model's weight tensors (raw bytes) keyed by parameter identifier,
    /// excluding the RNG identity parameter — the set a .skpt stores in its data tree.</summary>
    private static Dictionary<string, byte[]> WeightBytesByParam(ComputationGraph model)
        => model.ToInternal().Nodes
            .Where(n => n.OpCode == InternalOpCodes.MODEL_PARAM_DATA
                && n.IdentifierTemplate !=
                    Shorokoo.Core.Nodes.Processors.Fast.FastWireRngKeyDerivation.RngSeedIdentifierTemplate)
            .ToDictionary(
                n => n.IdentifierTemplate!,
                n => n.GetTensorData()!.AccessRawMemory().ToArray(),
                StringComparer.Ordinal);

    /// <summary>Extracts every archive entry through the BCL zip reader — an implementation
    /// independent of the .skpt writer, so success doubles as a standard-zip check.</summary>
    private static Dictionary<string, byte[]> ReadZipEntries(string path)
    {
        using var archive = ZipFile.OpenRead(path);
        var result = new Dictionary<string, byte[]>(StringComparer.Ordinal);
        foreach (var entry in archive.Entries)
        {
            using var entryStream = entry.Open();
            using var buffer = new MemoryStream();
            entryStream.CopyTo(buffer);
            result[entry.FullName] = buffer.ToArray();
        }
        return result;
    }

    /// <summary>Rebuilds a .skpt archive from raw entries (for tamper/corruption cases),
    /// re-aligning data-tree entries the way the real writer does.</summary>
    private static void RewriteSkpt(string path, IReadOnlyList<(string Name, byte[] Data)> entries)
    {
        using var stream = File.Create(path);
        SkptFileFormat.WriteStoredZip(
            stream,
            entries.Select(e => new SkptFileFormat.ZipEntrySpec(
                e.Name, e.Data, e.Name.StartsWith("data/", StringComparison.Ordinal))).ToList(),
            DateTime.UtcNow);
    }

    /// <summary>Walks the raw local file headers of a zip (no library involved), returning
    /// each entry's name, compression method, absolute payload offset and stored size.</summary>
    private static List<(string Name, ushort Method, long DataOffset, uint Size)> ParseLocalZipHeaders(byte[] zip)
    {
        var headers = new List<(string, ushort, long, uint)>();
        int offset = 0;
        while (offset + 30 <= zip.Length && BitConverter.ToUInt32(zip, offset) == 0x04034b50)
        {
            ushort method = BitConverter.ToUInt16(zip, offset + 8);
            uint size = BitConverter.ToUInt32(zip, offset + 18);
            ushort nameLength = BitConverter.ToUInt16(zip, offset + 26);
            ushort extraLength = BitConverter.ToUInt16(zip, offset + 28);
            string name = System.Text.Encoding.ASCII.GetString(zip, offset + 30, nameLength);
            long dataOffset = offset + 30 + nameLength + extraLength;
            headers.Add((name, method, dataOffset, size));
            offset = (int)(dataOffset + size);
        }
        return headers;
    }

    /// <summary>
    /// The .skpt acceptance round-trip: a concrete model saves to a single file that is a
    /// standard zip of STORED-only entries (data payload 64-byte aligned), whose manifest
    /// wires model → format/stage/hash and parameters → data tensors; the weights entry is
    /// byte-identical to the model's parameters while the model entry carries only zero
    /// placeholders (no duplicated weight bytes); and loading rebinds the weights into a
    /// concrete model that executes bit-identically to the original.
    /// </summary>
    [Fact]
    public void TestSkptRoundTripConcreteModel()
    {
        var (model, numOut, input) = BuildSkptModel();
        var path = Path.Combine(TempDir, "roundtrip.skpt");
        try
        {
            Checkpoint.From(model).WithModel().WithWeights().Save(path);

            var originalWeights = WeightBytesByParam(model);
            Assert.Equal(2, originalWeights.Count);
            // Default-initialized FCLayer weights are non-zero, so the byte-identity and
            // placeholder assertions below cannot pass vacuously.
            Assert.Contains(originalWeights.Values, bytes => bytes.Any(b => b != 0));

            // Standard zip, exactly the documented entries (read via the BCL, not our writer).
            var entries = ReadZipEntries(path);
            string[] expectedEntries =
                [SkptFileFormat.ConfigEntryName, SkptFileFormat.WeightsEntryPath, SkptFileFormat.ModelEntryPath];
            Assert.Equal(expectedEntries.OrderBy(n => n, StringComparer.Ordinal),
                entries.Keys.OrderBy(n => n, StringComparer.Ordinal));

            // All entries STORED; the data payload starts 64-byte aligned and verbatim.
            var fileBytes = File.ReadAllBytes(path);
            var localHeaders = ParseLocalZipHeaders(fileBytes);
            Assert.Equal(entries.Count, localHeaders.Count);
            Assert.All(localHeaders, h => Assert.Equal(0, h.Method));
            var weightsHeader = localHeaders.Single(h => h.Name == SkptFileFormat.WeightsEntryPath);
            Assert.Equal(0L, weightsHeader.DataOffset % SkptFileFormat.DataAlignment);
            Assert.Equal(entries[SkptFileFormat.WeightsEntryPath],
                fileBytes.AsSpan((int)weightsHeader.DataOffset, (int)weightsHeader.Size).ToArray());

            // The manifest is the wiring: format/version identity, model registry entry
            // (srk2, concrete-model, entry hash), data registry entry (safetensors,
            // uncompressed, entry hash), and the default mapping set covering every parameter.
            var manifest = SkptFileFormat.ParseManifest(entries[SkptFileFormat.ConfigEntryName], path);
            Assert.Equal(SkptFileFormat.FormatName, manifest.Format);
            Assert.Equal(SkptFileFormat.CurrentVersion, manifest.SkptVersion);
            Assert.False(string.IsNullOrEmpty(manifest.CreatedUtc));
            Assert.Equal(Shorokoo.ShorokooVersion.VersionString, manifest.Producer?.Shorokoo);
            var modelEntry = Assert.Single(manifest.Models!).Value;
            Assert.Equal(SkptFileFormat.ModelEntryPath, modelEntry.Entry);
            Assert.Equal(SkptFileFormat.ModelFormatSrk2, modelEntry.Format);
            Assert.Equal(SrkFileFormat.StageName(GraphKind.ConcreteModel), modelEntry.Stage);
            Assert.Equal(SkptFileFormat.Sha256Hex(entries[SkptFileFormat.ModelEntryPath]), modelEntry.Sha256);
            var dataEntry = Assert.Single(manifest.Data!).Value;
            Assert.Equal(SkptFileFormat.WeightsEntryPath, dataEntry.Entry);
            Assert.Equal(SkptFileFormat.DataFormatSafeTensors, dataEntry.Format);
            Assert.Equal(SkptFileFormat.CompressionNone, dataEntry.Compression);
            Assert.Equal(SkptFileFormat.Sha256Hex(entries[SkptFileFormat.WeightsEntryPath]), dataEntry.Sha256);
            var mapping = manifest.TensorMappings!["model"]["default"].Tensors!;
            Assert.Equal(originalWeights.Keys.OrderBy(k => k, StringComparer.Ordinal),
                mapping.Keys.OrderBy(k => k, StringComparer.Ordinal));
            Assert.All(mapping.Values, r => Assert.Equal("weights", r.Data));

            // The weights entry is plain safetensors holding byte-identical tensors.
            var storedTensors = SafeTensorLoader.ParseSafeTensorBytes(entries[SkptFileFormat.WeightsEntryPath])
                .ToDictionary(t => t.Name, t => t.Data.AccessRawMemory().ToArray(), StringComparer.Ordinal);
            Assert.Equal(originalWeights.Count, storedTensors.Count);
            foreach (var (paramId, bytes) in originalWeights)
                Assert.Equal(bytes, storedTensors[mapping[paramId].Tensor!]);

            // The model entry is definition-only: a loadable concrete model whose weight
            // parameters are zero placeholders — the real bytes live once, in the data tree.
            var strippedDefinition = CompressedFormatUtils.LoadFastGraphFromBinary(
                entries[SkptFileFormat.ModelEntryPath], GraphKind.ConcreteModel);
            Assert.All(WeightBytesByParam(strippedDefinition).Values,
                bytes => Assert.All(bytes, b => Assert.Equal(0, b)));

            // Load: a runnable concrete model, weights bound byte-identically, and
            // bit-identical execution on the sample input.
            var loaded = Checkpoint.Load(path);
            Assert.Equal(GraphKind.ConcreteModel, loaded.Kind);
            var loadedWeights = WeightBytesByParam(loaded);
            Assert.Equal(originalWeights.Count, loadedWeights.Count);
            foreach (var (paramId, bytes) in originalWeights)
                Assert.Equal(bytes, loadedWeights[paramId]);
            Assert.Equal(ExecuteToBytes(model, numOut, input), ExecuteToBytes(loaded, numOut, input));
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    /// <summary>
    /// The builder admits exactly the supported checkpoint shape (concrete model in,
    /// .WithModel().WithWeights() selected), and Save commits atomically: a crash in the
    /// commit window leaves the previous checkpoint bytes untouched and loadable.
    /// </summary>
    [Fact]
    public void TestSkptBuilderGatesAndAtomicSave()
    {
        var (model, numOut, input) = BuildSkptModel();

        // Only a concrete model can start a checkpoint.
        var exKind = Assert.Throws<InvalidOperationException>(() => Checkpoint.From(FCLayer.ComputationGraph));
        Assert.Contains("concrete-model", exKind.Message);

        // This version writes exactly one shape: model + weights, both selected.
        var incompletePath = Path.Combine(TempDir, "incomplete.skpt");
        var exNone = Assert.Throws<InvalidOperationException>(() => Checkpoint.From(model).Save(incompletePath));
        Assert.Contains("WithModel", exNone.Message);
        Assert.Throws<InvalidOperationException>(() => Checkpoint.From(model).WithModel().Save(incompletePath));
        Assert.Throws<InvalidOperationException>(() => Checkpoint.From(model).WithWeights().Save(incompletePath));
        Assert.False(File.Exists(incompletePath));

        // The atomic writer stages in the target's directory, so it must exist up front.
        Assert.Throws<DirectoryNotFoundException>(() => Checkpoint.From(model).WithModel().WithWeights()
            .Save(Path.Combine(TempDir, "no-such-dir", "model.skpt")));

        // A simulated crash between staging and commit leaves the existing checkpoint intact.
        var path = Path.Combine(TempDir, "atomic.skpt");
        try
        {
            Checkpoint.From(model).WithModel().WithWeights().Save(path);
            var committed = File.ReadAllBytes(path);

            AtomicFileWriter.CommitFaultInjection = tempPath =>
            {
                if (tempPath.Contains("atomic.skpt")) throw new IOException("simulated commit crash");
            };
            try
            {
                Assert.Throws<IOException>(() => Checkpoint.From(model).WithModel().WithWeights().Save(path));
            }
            finally { AtomicFileWriter.CommitFaultInjection = null; }

            Assert.Equal(committed, File.ReadAllBytes(path));
            var loaded = Checkpoint.Load(path);
            Assert.Equal(ExecuteToBytes(model, numOut, input), ExecuteToBytes(loaded, numOut, input));
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    /// <summary>
    /// Load-side contract of the manifest: unknown keys anywhere are ignored (keys are
    /// add-only), while real faults fail loudly naming the offender — a non-zip file, a
    /// missing manifest, a manifest referencing a missing entry, an entry failing its
    /// SHA-256, an unsupported future version, and a tensor mapping that does not match
    /// the model's parameters.
    /// </summary>
    [Fact]
    public void TestSkptLoadValidationAndUnknownKeyTolerance()
    {
        var (model, numOut, input) = BuildSkptModel();
        var path = Path.Combine(TempDir, "validation.skpt");
        var tamperedPath = Path.Combine(TempDir, "tampered.skpt");
        try
        {
            Checkpoint.From(model).WithModel().WithWeights().Save(path);
            var entries = ReadZipEntries(path);
            var direct = ExecuteToBytes(model, numOut, input);
            List<(string Name, byte[] Data)> Without(string name) =>
                entries.Where(e => e.Key != name).Select(e => (e.Key, e.Value)).ToList();
            List<(string Name, byte[] Data)> WithConfig(string configJson) =>
                entries.Select(e => (e.Key, e.Key == SkptFileFormat.ConfigEntryName
                    ? System.Text.Encoding.UTF8.GetBytes(configJson) : e.Value)).ToList();

            // Not a zip at all.
            File.WriteAllBytes(tamperedPath, [1, 2, 3, 4]);
            var exNotZip = Assert.Throws<InvalidDataException>(() => Checkpoint.Load(tamperedPath));
            Assert.Contains("zip", exNotZip.Message);

            // A zip without the manifest is not a checkpoint.
            RewriteSkpt(tamperedPath, Without(SkptFileFormat.ConfigEntryName));
            var exNoConfig = Assert.Throws<InvalidDataException>(() => Checkpoint.Load(tamperedPath));
            Assert.Contains(SkptFileFormat.ConfigEntryName, exNoConfig.Message);

            // A manifest referencing a missing entry names the entry.
            RewriteSkpt(tamperedPath, Without(SkptFileFormat.WeightsEntryPath));
            var exMissing = Assert.Throws<InvalidDataException>(() => Checkpoint.Load(tamperedPath));
            Assert.Contains(SkptFileFormat.WeightsEntryPath, exMissing.Message);

            // A tampered entry fails its SHA-256 check, naming the entry.
            var flipped = entries.Select(e =>
            {
                if (e.Key != SkptFileFormat.WeightsEntryPath) return (e.Key, e.Value);
                var copy = e.Value.ToArray();
                copy[^1] ^= 0xFF;
                return (e.Key, copy);
            }).ToList();
            RewriteSkpt(tamperedPath, flipped);
            var exSha = Assert.Throws<InvalidDataException>(() => Checkpoint.Load(tamperedPath));
            Assert.Contains("SHA-256", exSha.Message);
            Assert.Contains(SkptFileFormat.WeightsEntryPath, exSha.Message);

            // Unknown keys at every level are ignored: the manifest's keys are add-only.
            var config = JsonNode.Parse(entries[SkptFileFormat.ConfigEntryName])!;
            config["futureTopLevelKey"] = "ignored";
            config["models"]!["model"]!["futureModelKey"] = 42;
            config["data"]!["weights"]!["futureDataKey"] = true;
            config["tensorMappings"]!["model"]!["default"]!["futureSetKey"] = "ignored";
            var firstParam = ((JsonObject)config["tensorMappings"]!["model"]!["default"]!["tensors"]!)
                .First().Key;
            config["tensorMappings"]!["model"]!["default"]!["tensors"]![firstParam]!["futureRefKey"] = 1;
            RewriteSkpt(tamperedPath, WithConfig(config.ToJsonString()));
            Assert.Equal(direct, ExecuteToBytes(Checkpoint.Load(tamperedPath), numOut, input));

            // A future major version is refused with a clear message.
            var futureConfig = JsonNode.Parse(entries[SkptFileFormat.ConfigEntryName])!;
            futureConfig["skptVersion"] = SkptFileFormat.CurrentVersion + 1;
            RewriteSkpt(tamperedPath, WithConfig(futureConfig.ToJsonString()));
            var exVersion = Assert.Throws<InvalidDataException>(() => Checkpoint.Load(tamperedPath));
            Assert.Contains("newer framework version", exVersion.Message);

            // A mapping missing one of the model's parameters names the parameter.
            var missingParamConfig = JsonNode.Parse(entries[SkptFileFormat.ConfigEntryName])!;
            ((JsonObject)missingParamConfig["tensorMappings"]!["model"]!["default"]!["tensors"]!)
                .Remove(firstParam);
            RewriteSkpt(tamperedPath, WithConfig(missingParamConfig.ToJsonString()));
            var exUnmapped = Assert.Throws<InvalidDataException>(() => Checkpoint.Load(tamperedPath));
            Assert.Contains(firstParam, exUnmapped.Message);

            // A mapping entry for a parameter the model does not declare names the stray.
            var strayConfig = JsonNode.Parse(entries[SkptFileFormat.ConfigEntryName])!;
            strayConfig["tensorMappings"]!["model"]!["default"]!["tensors"]!["not_a_real_param"] =
                new JsonObject { ["data"] = "weights", ["tensor"] = "not_a_real_param" };
            RewriteSkpt(tamperedPath, WithConfig(strayConfig.ToJsonString()));
            var exStray = Assert.Throws<InvalidDataException>(() => Checkpoint.Load(tamperedPath));
            Assert.Contains("not_a_real_param", exStray.Message);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
            if (File.Exists(tamperedPath)) File.Delete(tamperedPath);
        }
    }

}

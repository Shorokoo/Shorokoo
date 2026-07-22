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
    /// A truncated .safetensors file fails loudly with a diagnostic naming truncation and
    /// the declared vs. actual byte counts, at every cut point: inside the tensor data
    /// (ST003), inside the JSON header (ST002), and before the 8-byte length field (ST001).
    /// File-path loads name the offending file in the message; the untruncated bytes still
    /// parse (the checks are prefix/length-only and cost the valid path nothing).
    /// </summary>
    [Fact]
    public void TestSafeTensorTruncationFailsLoudly()
    {
        var tensors = new List<SafeTensor>
        {
            new("w", TensorData([4L], new float[] { 1f, 2f, 3f, 4f }), "F32", [4L]),
            new("b", TensorData([2L], new float[] { 5f, 6f }), "F32", [2L]),
        };
        using var stream = new MemoryStream();
        SafeTensorLoader.SaveSafeTensorsToStream(stream, tensors);
        var bytes = stream.ToArray();

        // The intact buffer parses; truncation checks must not disturb the valid path.
        Assert.Equal(2, SafeTensorLoader.ParseSafeTensorBytes(bytes).Count);

        // Cut inside the tensor data → the affected tensor's declared range vs. the actual length.
        var exData = Assert.Throws<ModelException>(() => SafeTensorLoader.ParseSafeTensorBytes(bytes[..^4]));
        Assert.Equal(ErrorCodes.ST003, exData.ErrorCode);
        Assert.Contains("truncated", exData.Message);
        Assert.Contains("'b'", exData.Message);
        Assert.Contains($"{bytes.Length} bytes", exData.Message);       // declared (required) size
        Assert.Contains($"{bytes.Length - 4} bytes", exData.Message);   // actual size

        // Cut inside the JSON header → declared header length vs. the bytes that follow.
        long headerLen = BitConverter.ToInt64(bytes, 0);
        var exHeader = Assert.Throws<ModelException>(() => SafeTensorLoader.ParseSafeTensorBytes(bytes[..10]));
        Assert.Equal(ErrorCodes.ST002, exHeader.ErrorCode);
        Assert.Contains("truncated", exHeader.Message);
        Assert.Contains($"declares {headerLen} bytes", exHeader.Message);
        Assert.Contains("only 2 byte(s)", exHeader.Message);

        // Cut before the length field even completes.
        var exTiny = Assert.Throws<ModelException>(() => SafeTensorLoader.ParseSafeTensorBytes(bytes[..5]));
        Assert.Equal(ErrorCodes.ST001, exTiny.ErrorCode);
        Assert.Contains("truncated", exTiny.Message);

        // File-path load: the error names the offending file.
        var truncPath = Path.Combine(TempDir, "truncated.safetensors");
        try
        {
            File.WriteAllBytes(truncPath, bytes[..^4]);
            var exFile = Assert.Throws<ModelException>(() => SafeTensorLoader.LoadSafeTensors(truncPath));
            Assert.Equal(ErrorCodes.ST003, exFile.ErrorCode);
            Assert.Contains(truncPath, exFile.Message);
            Assert.Contains("truncated", exFile.Message);
        }
        finally { if (File.Exists(truncPath)) File.Delete(truncPath); }

        // Compressed (.zsafetensor) load of truncated content: the error still names the
        // file the caller passed, not an in-memory placeholder.
        var zPath = Path.Combine(TempDir, "truncated.zsafetensor");
        try
        {
            CompressedFormatUtils.CompressToFile(zPath, bytes[..^4]);
            var exZ = Assert.Throws<ModelException>(() => CompressedFormatUtils.LoadCompressedSafeTensors(zPath));
            Assert.Equal(ErrorCodes.ST003, exZ.ErrorCode);
            Assert.Contains(zPath, exZ.Message);
            Assert.Contains("truncated", exZ.Message);
        }
        finally { if (File.Exists(zPath)) File.Delete(zPath); }
    }

    /// <summary>
    /// A tensor entry missing a required metadata field (shape / data_offsets / dtype) is
    /// refused loudly. The loader previously fabricated defaults ([1] / [0, 4) / F32), which
    /// let a corrupt header validate against invented offsets and load garbage silently.
    /// </summary>
    [Fact]
    public void TestSafeTensorMissingMetadataFailsLoudly()
    {
        static byte[] Build(string headerJson, int payloadBytes)
        {
            var headerBytes = System.Text.Encoding.UTF8.GetBytes(headerJson);
            return [.. BitConverter.GetBytes((long)headerBytes.Length), .. headerBytes, .. new byte[payloadBytes]];
        }

        var noOffsets = Build("{\"w\":{\"dtype\":\"F32\",\"shape\":[1]}}", 4);
        var exOffsets = Assert.Throws<InvalidOperationException>(
            () => SafeTensorLoader.ParseSafeTensorBytes(noOffsets));
        Assert.Contains("data_offsets", exOffsets.Message);
        Assert.Contains("'w'", exOffsets.Message);

        var noDtype = Build("{\"w\":{\"shape\":[1],\"data_offsets\":[0,4]}}", 4);
        var exDtype = Assert.Throws<InvalidOperationException>(
            () => SafeTensorLoader.ParseSafeTensorBytes(noDtype));
        Assert.Contains("dtype", exDtype.Message);

        var noShape = Build("{\"w\":{\"dtype\":\"F32\",\"data_offsets\":[0,4]}}", 4);
        var exShape = Assert.Throws<InvalidOperationException>(
            () => SafeTensorLoader.ParseSafeTensorBytes(noShape));
        Assert.Contains("shape", exShape.Message);

        // A rank-0 scalar's empty shape ("shape": []) is valid, not "missing" — it must load.
        var scalar = Build("{\"s\":{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[0,4]}}", 4);
        var loaded = SafeTensorLoader.ParseSafeTensorBytes(scalar);
        Assert.Empty(loaded.Single().Data.Shape.Dims);
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
                Assert.Equal((long?)bytes.Length, result.Srk.PayloadSizeBytes);
                // Every sniffed legacy layout carries the same no-header observation,
                // however it was detected (bare protobuf or Zstd-wrapped).
                Assert.Contains(result.Observations, o => o.Contains("record no stage"));
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
                Assert.Null(result.Srk.PayloadSizeBytes);   // unknown, no longer a 0 sentinel
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

    /// <summary>
    /// Inspect positively identifies .zsafetensor archives (issue #70): the frame's inner
    /// 8-byte length prefix and JSON header are stream-decompressed — bounded reads, never
    /// the tensor payload — and parsed with the shared SafeTensors header logic. Covers:
    /// a real SaveCompressedSafeTensors round-trip recognized with the correct tensor
    /// listing and metadata; proof the payload is untouched (an archive whose compressed
    /// tail is chopped off still inspects, while a full load fails); a compressed training
    /// checkpoint reporting the archive kind plus an observation instead of version/step
    /// (the marker payload is not boundedly reachable through the non-seekable stream);
    /// and corrupt/truncated compressed files yielding structured non-throwing results.
    /// </summary>
    [Fact]
    public void TestCheckpointInspectCompressedSafeTensorsArtifacts()
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
            // Real round-trip: written by the actual saver, recognized with the full
            // tensor listing and the __metadata__ entries, no observations.
            var t1 = TensorData([2, 3], 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);
            var t2 = TensorData([3], 7.0f, 8.0f, 9.0f);
            var zPath = NextPath("inspect_weights.zsafetensor");
            CompressedFormatUtils.SaveCompressedSafeTensors(zPath, new List<SafeTensor>
            {
                new SafeTensor("tensor1", t1, "F32", t1.Shape.Dims),
                new SafeTensor("tensor2", t2, "F32", t2.Shape.Dims),
            }, new Dictionary<string, object> { ["format"] = "shorokoo-test" });

            var result = Checkpoint.Inspect(zPath);
            Assert.Equal(ArtifactKind.CompressedSafeTensors, result.Kind);
            Assert.Equal(zPath, result.FilePath);
            Assert.Equal(new FileInfo(zPath).Length, result.FileSizeBytes);
            Assert.Null(result.Srk);
            Assert.Null(result.TrainingCheckpoint);
            Assert.Empty(result.Observations);

            var st = result.SafeTensors!;
            Assert.True(st.HeaderSizeBytes > 0);
            Assert.Equal(2, st.Tensors.Count);
            Assert.Equal(6 * 4 + 3 * 4, st.TotalTensorBytes);
            var byName = st.Tensors.ToDictionary(t => t.Name);
            long[] expectedShape1 = [2, 3];
            Assert.Equal("F32", byName["tensor1"].DType);
            Assert.Equal(expectedShape1, byName["tensor1"].Shape);
            Assert.Equal(24, byName["tensor1"].ByteSize);
            Assert.Equal("shorokoo-test", st.GlobalMetadata!["format"]);

            var text = result.ToString();
            Assert.Contains("Zstd-compressed SafeTensors archive", text);
            Assert.Contains("tensor1: F32[2, 3], 24 bytes", text);

            // Payload untouched: an incompressible multi-block payload whose compressed
            // tail is chopped off. A full load fails on the broken frame, but Inspect —
            // which decompresses only the prefix + header from the intact leading blocks —
            // still recognizes the archive. (Zstd blocks hold at most 128 KB decompressed,
            // so a ~1.2 MB payload guarantees the header and the chopped tail sit in
            // different blocks.)
            var bigValues = new float[300_000];
            uint seed = 1;
            for (int i = 0; i < bigValues.Length; i++)
            {
                seed = seed * 747796405u + 2891336453u;   // cheap PCG-ish, incompressible
                bigValues[i] = BitConverter.UInt32BitsToSingle((seed >> 9) | 0x3F800000u);
            }
            var big = TensorData([300_000L], bigValues);
            var bigPath = NextPath("inspect_big.zsafetensor");
            CompressedFormatUtils.SaveCompressedSafeTensors(bigPath, new List<SafeTensor>
            {
                new SafeTensor("big", big, "F32", big.Shape.Dims),
            });
            var bigBytes = File.ReadAllBytes(bigPath);
            Assert.True(bigBytes.Length > 256 * 1024);   // really multi-block
            var choppedPath = NextPath("inspect_big_chopped.zsafetensor");
            File.WriteAllBytes(choppedPath, bigBytes[..^64]);
            Assert.ThrowsAny<Exception>(
                () => CompressedFormatUtils.LoadCompressedSafeTensors(choppedPath));
            var chopped = Checkpoint.Inspect(choppedPath);
            Assert.Equal(ArtifactKind.CompressedSafeTensors, chopped.Kind);
            Assert.Equal("big", Assert.Single(chopped.SafeTensors!.Tensors).Name);
            Assert.Equal(300_000L * 4, chopped.SafeTensors.TotalTensorBytes);

            // A checkpoint saved compressed: the marker is visible in the header, but its
            // [version, step] payload sits inside the compressed tensor data, beyond the
            // bounded header read — the archive kind is reported, TrainingCheckpoint stays
            // null, and an observation says why.
            var w = TensorData([2L], 1.0f, 2.0f);
            var ckptPath = NextPath("inspect_ckpt.zsafetensor");
            CompressedFormatUtils.SaveCompressedSafeTensors(ckptPath, new List<SafeTensor>
            {
                new SafeTensor("trainable/w", w, "F32", w.Shape.Dims),
                new SafeTensor("__shorokoo_checkpoint__", TensorData([2L], 1L, 7L), "I64", [2L]),
            });
            var ckpt = Checkpoint.Inspect(ckptPath);
            Assert.Equal(ArtifactKind.CompressedSafeTensors, ckpt.Kind);
            Assert.Null(ckpt.TrainingCheckpoint);
            Assert.Contains(ckpt.SafeTensors!.Tensors, t => t.Name == "__shorokoo_checkpoint__");
            Assert.Contains(ckpt.Observations,
                o => o.Contains("__shorokoo_checkpoint__") && o.Contains("bounded"));

            // A Zstd frame truncated to its first bytes fails to decompress → structured
            // NotRecognized, no exception.
            var stubPath = NextPath("inspect_stub.zsafetensor");
            File.WriteAllBytes(stubPath, bigBytes[..5]);
            var stub = Checkpoint.Inspect(stubPath);
            Assert.Equal(ArtifactKind.NotRecognized, stub.Kind);
            Assert.Contains(stub.Observations, o => o.Contains("Zstd frame"));

            // A frame that decompresses cleanly but ends inside its declared SafeTensors
            // header (the compressed analogue of a truncated header) → structured
            // NotRecognized naming the truncation.
            byte[] shortDecl = [.. BitConverter.GetBytes(1000L), 0x7B, 0x22, 0x74];
            var shortPath = NextPath("inspect_short.zsafetensor");
            File.WriteAllBytes(shortPath, CompressedFormatUtils.Compress(shortDecl));
            var shortResult = Checkpoint.Inspect(shortPath);
            Assert.Equal(ArtifactKind.NotRecognized, shortResult.Kind);
            Assert.Contains(shortResult.Observations, o => o.Contains("ends after"));

            // Amplification guard: a small file declaring a near-cap (99 MB) header must
            // yield the same structured result without the declaration costing 99 MB of
            // allocation — the header buffer grows with what the stream delivers (here
            // 200 KB), which also exercises the growth path across block boundaries.
            var hugeDecl = new byte[8 + 200_000];
            BitConverter.GetBytes(99_000_000L).CopyTo(hugeDecl, 0);
            var hugePath = NextPath("inspect_huge_decl.zsafetensor");
            File.WriteAllBytes(hugePath, CompressedFormatUtils.Compress(hugeDecl));
            var hugeResult = Checkpoint.Inspect(hugePath);
            Assert.Equal(ArtifactKind.NotRecognized, hugeResult.Kind);
            Assert.Contains(hugeResult.Observations, o => o.Contains("ends after 200000"));

            // The .zsafetensor probe must not disturb legacy-.srk recognition: a
            // single-Zstd legacy graph still sniffs as such (its decompressed prefix
            // never declares a plausible header length).
            var (_, arch, _) = BuildStageGraphs();
            var bare = SrkFileFormat.Read(
                CompressedFormatUtils.SaveFastGraphToBinary(arch, compressed: false)).OnnxBytes;
            var legacyPath = NextPath("inspect_legacy_single.srk");
            File.WriteAllBytes(legacyPath, CompressedFormatUtils.Compress(bare));
            var legacy = Checkpoint.Inspect(legacyPath);
            Assert.Equal(ArtifactKind.SrkGraph, legacy.Kind);
            Assert.Equal("single-Zstd", legacy.Srk!.LegacyLayout);
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


    // ──────────────────────────────────────────────────────────────────────
    // Module-stage .srk round-trip fidelity audit (issue #59).
    //
    // The construct inventory, and where each construct's save → load coverage
    // lives:
    //
    //  1. Optional/absent tensor arguments on module-stage ops — the absent
    //     optional model slot of MODEL_PARAM_MODEL_REF (every top-level
    //     initializer call, e.g. ScalarMultiplyModel), the absent optional
    //     "key" input of the SHRK_RANDOM_* runtime feeds (pre-concretization),
    //     and an OptionalTensor model input (MODEL_OPTIONAL_INPUT,
    //     NullableBiasLayer): TestModuleStageSrkRoundTripStructuralFidelity
    //     (the structure descriptor records the null-slot pattern of every
    //     node) + TestModuleStageSrkRoundTripLoweredExecutionMatches.
    //  2. State-initializer ownership tags (StateOwnership.ModuleOwned /
    //     OptimizerOwned on StateParamInitializer functions):
    //     TestModuleStageSrkRoundTripPreservesStateInitializerOwnership.
    //  3. Struct definitions and struct-typed values — a TensorStruct model
    //     input (MODEL_TENSORSTRUCT_INPUT + TensorStructDef metadata,
    //     SimplePairSum) and TENSOR_STRUCT_CREATE / TENSOR_STRUCT_GETFIELD
    //     values (TensorStructLoopCarry): structural fidelity + lowered
    //     execution below; the load-side arch pipeline is additionally covered
    //     by ModulesTests.TestModuleGraphSaveLoadOnlyCoverage.
    //  4. Loop/scope structure with module-stage ops inside loop bodies —
    //     MODEL_PARAM_MODEL_REF in nested LOOP bands (TrainablesInBothLoopLevels)
    //     and TENSOR_STRUCT_CREATE/GETFIELD in a loop band (TensorStructLoopCarry):
    //     structural fidelity + lowered execution below.
    //  5. RNG-related module-stage constructs — a runtime feed inside a loop
    //     body (RngRuntimeLoopFeed; at module stage the feed has no key-derivation
    //     chain yet, so the optional key slot is absent) and random trainable-param
    //     initializers (RngInitTwoLinears): lowered execution equality below
    //     proves the reloaded module derives the same keyed streams (RngSeed at
    //     ModelId [0], split chains) and draws identical values.
    //  6. Generic-typed module graphs (GENERIC_TYPE_INPUT placeholders,
    //     GenericRecordSumCaller pre-specialization): structural fidelity below.
    //  7. Sub-module invocation machinery (MODEL_INVOKE / SUBMODEL# /
    //     CREATE_MODULE / MODULE_SET_HYPERPARAMS / MODEL_HYPERPARAM /
    //     FUNCTION_INVOKE): structural fidelity + lowered execution below;
    //     end-to-end also ModulesTests.TestModuleGraphOnnxRoundtripCoverage.
    //  8. Hyperparameter defaults ([Hyper(v)] → ShrkAttrDefaultValue):
    //     NullableParamTests.DefaultedHyper_DefaultValue_SurvivesOnnxBinaryRoundtrip
    //     (pre-existing pin); DefaultedHyperLayer also rides the structural set.
    // ──────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Order-independent structure descriptor of a module graph: per-node
    /// opcode + per-input-slot present/absent pattern (multiset), plus the graph
    /// I/O signature names. Node/tensor keys are freshly assigned on load, so the
    /// descriptor deliberately excludes them; execution equality (below) covers
    /// value-level semantics the descriptor abstracts away.
    /// </summary>
    private static string DescribeModuleGraphStructure(ComputationGraph graph)
    {
        var g = graph.ToInternal();
        var nodeLines = g.Nodes
            .Select(n => $"{n.OpCode}({string.Join(",", n.Inputs.Select(k => k is null ? "-" : "x"))})")
            .OrderBy(x => x, StringComparer.Ordinal);
        return $"inputs=[{string.Join(",", g.InputUniqueNames)}] outputs=[{string.Join(",", g.OutputUniqueNames)}]\n"
             + string.Join("\n", nodeLines);
    }

    /// <summary>
    /// Saves a module graph to .srk, asserts the header stamps the module stage,
    /// reloads, and asserts (a) the reloaded graph is stamped Module, (b) its
    /// structure descriptor is unchanged, and (c) a second save → load → save is a
    /// byte-level fixed point — so nothing the first load produced is lost or
    /// mutated by another cycle. Returns the reloaded graph.
    /// </summary>
    private static ComputationGraph AssertModuleStageSrkRoundTrip(ComputationGraph moduleGraph)
    {
        Assert.Equal(GraphKind.Module, moduleGraph.Kind);
        var bytes = CompressedFormatUtils.SaveFastGraphToBinary(moduleGraph, compressed: false);
        Assert.Equal(GraphKind.Module, SrkFileFormat.TryReadHeader(bytes)!.TryGetStage());

        var reloaded = CompressedFormatUtils.LoadFastGraphFromBinary(bytes);
        Assert.Equal(GraphKind.Module, reloaded.Kind);
        Assert.Equal(DescribeModuleGraphStructure(moduleGraph), DescribeModuleGraphStructure(reloaded));

        var bytes2 = CompressedFormatUtils.SaveFastGraphToBinary(reloaded, compressed: false);
        var bytes3 = CompressedFormatUtils.SaveFastGraphToBinary(
            CompressedFormatUtils.LoadFastGraphFromBinary(bytes2), compressed: false);
        Assert.True(bytes2.SequenceEqual(bytes3),
            "save → load → save is not a fixed point: a load/save cycle keeps changing the serialized module graph.");
        return reloaded;
    }

    /// <summary>
    /// Issue #59 construct inventory, structural leg: every inventoried module-level
    /// construct survives save → load with an unchanged structure descriptor (see the
    /// inventory comment above for the construct ↔ fixture mapping).
    /// </summary>
    [Fact]
    public void TestModuleStageSrkRoundTripStructuralFidelity()
    {
        ComputationGraph[] moduleGraphs =
        [
            ScalarMultiplyModel.ComputationGraph,            // absent optional model slot on MODEL_PARAM_MODEL_REF
            ScalarMultiplyWithBatchNormModel.ComputationGraph, // module-owned state initializers
            StepCountingSgdOptimizer.ComputationGraph,       // optimizer-owned state initializer + defaulted hyper
            NullableBiasLayer.ComputationGraph,              // MODEL_OPTIONAL_INPUT (optional model input)
            DefaultedHyperLayer.ComputationGraph,            // [Hyper(3f)] default on MODEL_TENSOR_INPUT
            SimplePairSum.ComputationGraph,                  // MODEL_TENSORSTRUCT_INPUT + TENSOR_STRUCT_GETFIELD
            TensorStructLoopCarry.ComputationGraph,          // TENSOR_STRUCT_CREATE/GETFIELD inside a LOOP band
            TrainablesInBothLoopLevels.ComputationGraph,     // MODEL_PARAM_MODEL_REF in nested LOOP bodies
            RngRuntimeLoopFeed.ComputationGraph,             // SHRK_RANDOM_UNIFORM feed (absent key) in a LOOP body
            RngInitTwoLinears.ComputationGraph,              // sub-module invokes + random param initializers
            GenericRecordSumCaller.ComputationGraph,         // GENERIC_TYPE_INPUT + struct-typed sub-module call
        ];
        foreach (var moduleGraph in moduleGraphs)
            AssertModuleStageSrkRoundTrip(moduleGraph);
    }

    /// <summary>
    /// Issue #59 construct inventory, execution leg: for every lowerable fixture,
    /// the original and the reloaded module graph lower to concrete models that
    /// execute bit-identically (same inputs, same RngConfig — covering keyed init
    /// draws, runtime feeds and loop unrolling on both sides).
    /// </summary>
    [Fact]
    public void TestModuleStageSrkRoundTripLoweredExecutionMatches()
    {
        var x23 = TensorData([2L, 3L], 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);
        var x44 = TensorData([4L, 4L],
            [.. Enumerable.Range(0, 16).Select(i => 0.25f * i - 2.0f)]);

        (ComputationGraph Module, TensorData[] Inputs)[] cases =
        [
            (ScalarMultiplyModel.ComputationGraph, [TensorData([2L], 1.0f, 2.0f)]),
            (TrainablesInBothLoopLevels.ComputationGraph, [x23]),
            (TensorStructLoopCarry.ComputationGraph,
                [TensorData(DType.Float32, [], 1.5f), TensorData(DType.Float32, [], -0.5f)]),
            (NullableBiasLayer.ComputationGraph, [x23, x23]),
            (RngRuntimeLoopFeed.ComputationGraph, [x23, TensorData(DType.Int64, [], 3L)]),
            (RngInitTwoLinears.ComputationGraph, [x44]),
        ];

        foreach (var (moduleGraph, inputs) in cases)
        {
            var reloaded = AssertModuleStageSrkRoundTrip(moduleGraph);

            byte[] Run(ComputationGraph module)
            {
                var model = module
                    .ToConcreteArchitecture(module.FromOrderedInputs([.. inputs]))
                    .ToConcreteModel(RngConfig.Default);
                return ComputeContext.Default.Execute(model, inputs)[0]
                    .ToTensorData().AccessRawMemory().ToArray();
            }

            Assert.Equal(Run(moduleGraph), Run(reloaded));
        }
    }

    /// <summary>
    /// Issue #59, construct 2: the StateOwnership tag of a state-initializer function
    /// survives save → load at module stage — an OptimizerOwned initializer must not
    /// silently reload as the ModuleOwned default (the TrainingRig's ownership checks
    /// branch on it), and ModuleOwned must stay ModuleOwned.
    /// </summary>
    [Fact]
    public void TestModuleStageSrkRoundTripPreservesStateInitializerOwnership()
    {
        (ComputationGraph Module, StateOwnership Expected)[] cases =
        [
            (StepCountingSgdOptimizer.ComputationGraph, StateOwnership.OptimizerOwned),
            (ScalarMultiplyWithBatchNormModel.ComputationGraph, StateOwnership.ModuleOwned),
        ];
        foreach (var (moduleGraph, expected) in cases)
        {
            var reloaded = AssertModuleStageSrkRoundTrip(moduleGraph);
            var stateInits = reloaded.ToInternal().Nodes
                .Select(n => n.TargetFunction)
                .Where(fn => fn is { FunctionType: FunctionType.StateParamInitializer })
                .ToArray();
            Assert.NotEmpty(stateInits);
            Assert.All(stateInits, fn => Assert.Equal(expected, fn!.StateOwnership));
        }
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
    /// byte-identical to the model's parameters while the model entry carries only
    /// metadata-only placeholders (no duplicated weight bytes, no materialized zero
    /// buffers); and loading rebinds the weights into a concrete model that executes
    /// bit-identically to the original.
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
            // parameters are dtype/shape-true metadata-only placeholders — the real bytes
            // live once, in the data tree, and neither save nor load materializes a
            // weight-sized buffer for a placeholder (issue #69): its storage is elided
            // entirely, so accessing its values fails loudly.
            var strippedDefinition = CompressedFormatUtils.LoadFastGraphFromBinary(
                entries[SkptFileFormat.ModelEntryPath], GraphKind.ConcreteModel);
            var originalParams = model.ToInternal().Nodes
                .Where(n => n.OpCode == InternalOpCodes.MODEL_PARAM_DATA)
                .ToDictionary(n => n.IdentifierTemplate!, n => n.GetTensorData()!, StringComparer.Ordinal);
            var strippedWeightParams = strippedDefinition.ToInternal().Nodes
                .Where(n => n.OpCode == InternalOpCodes.MODEL_PARAM_DATA
                    && n.IdentifierTemplate !=
                        Shorokoo.Core.Nodes.Processors.Fast.FastWireRngKeyDerivation.RngSeedIdentifierTemplate)
                .ToList();
            Assert.Equal(originalWeights.Count, strippedWeightParams.Count);
            foreach (var param in strippedWeightParams)
            {
                var placeholder = Assert.IsType<WeightPlaceholderTensorData>(param.GetTensorData());
                var original = originalParams[param.IdentifierTemplate!];
                Assert.Equal(original.DType.ToIVarType(), placeholder.DType.ToIVarType());
                Assert.Equal(original.Shape.Dims, placeholder.Shape.Dims);
                var exElided = Assert.Throws<InvalidOperationException>(
                    () => { placeholder.AccessRawMemory(); });
                Assert.Contains("placeholder", exElided.Message);
            }
            // Stripping touches exactly the weight parameters: the stripped definition
            // declares the same parameter set as the original, and any non-weight
            // parameter (e.g. an RNG identity, when the model carries one) stays
            // embedded with real values, never a placeholder.
            var strippedAllParams = strippedDefinition.ToInternal().Nodes
                .Where(n => n.OpCode == InternalOpCodes.MODEL_PARAM_DATA)
                .ToList();
            Assert.Equal(originalParams.Count, strippedAllParams.Count);
            Assert.All(
                strippedAllParams.Where(n => !originalWeights.ContainsKey(n.IdentifierTemplate!)),
                n => Assert.IsNotType<WeightPlaceholderTensorData>(n.GetTensorData()));

            // Load: a runnable concrete model, weights bound byte-identically, and
            // bit-identical execution on the sample input.
            var loaded = Checkpoint.Load(path);
            Assert.Equal(GraphKind.ConcreteModel, loaded.Kind);
            var loadedWeights = WeightBytesByParam(loaded);
            Assert.Equal(originalWeights.Count, loadedWeights.Count);
            foreach (var (paramId, bytes) in originalWeights)
                Assert.Equal(bytes, loadedWeights[paramId]);
            Assert.Equal(ExecuteToBytes(model, numOut, input), ExecuteToBytes(loaded, numOut, input));

            // Back-compat: a checkpoint whose model definition carries materialized
            // zero placeholders without the values-elided marker — the shape every
            // .skpt written before the marker existed has — still loads and binds
            // identically. (Synthesized by re-stripping with full zero tensors and
            // splicing the entry + its manifest hash into the archive.)
            var legacyPath = Path.Combine(TempDir, "legacy-zero-placeholders.skpt");
            try
            {
                var legacyGraph = model.ToInternal().Clone();
                foreach (var node in legacyGraph.Nodes)
                {
                    if (node.OpCode != InternalOpCodes.MODEL_PARAM_DATA
                        || !originalWeights.ContainsKey(node.IdentifierTemplate ?? "")) continue;
                    var data = node.GetTensorData()!;
                    node.Attributes = node.Attributes.SetAttributes(
                        (OnnxOpAttributeNames.ShrkAttrTensorData,
                         (object?)TensorDataWithDefaultVals(data.DType, data.Shape.Dims)));
                }
                var legacyModelBytes = CompressedFormatUtils.SaveFastGraphToBinary(
                    legacyGraph, GraphKind.ConcreteModel, compressed: true);
                var legacyConfig = JsonNode.Parse(entries[SkptFileFormat.ConfigEntryName])!;
                legacyConfig["models"]!["model"]!["sha256"] = SkptFileFormat.Sha256Hex(legacyModelBytes);
                RewriteSkpt(legacyPath, entries.Select(e => (e.Key, e.Key switch
                {
                    SkptFileFormat.ConfigEntryName =>
                        System.Text.Encoding.UTF8.GetBytes(legacyConfig.ToJsonString()),
                    SkptFileFormat.ModelEntryPath => legacyModelBytes,
                    _ => e.Value,
                })).ToList());
                var legacyLoaded = Checkpoint.Load(legacyPath);
                var legacyWeights = WeightBytesByParam(legacyLoaded);
                Assert.Equal(originalWeights.Count, legacyWeights.Count);
                foreach (var (paramId, bytes) in originalWeights)
                    Assert.Equal(bytes, legacyWeights[paramId]);
                Assert.Equal(ExecuteToBytes(model, numOut, input),
                    ExecuteToBytes(legacyLoaded, numOut, input));
            }
            finally { if (File.Exists(legacyPath)) File.Delete(legacyPath); }
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

    /// <summary>Builds a concrete FCLayer model (32 output features, [32,32] input) whose
    /// weight tensors are overwritten with a deterministic repeating non-zero pattern —
    /// large enough and compressible enough that Zstd reliably shrinks the data entry,
    /// and distinct from the zero placeholders the model entry stores.</summary>
    private static (ComputationGraph Model, TensorData NumOut, TensorData Input) BuildCompressibleSkptModel()
    {
        var numOut = TensorData(DType.Int64, [], 32L);
        var input = TensorDataWithSmallVals(DType.Float32, [32L, 32L]);
        var g = FCLayer.ComputationGraph;   // two trainable params: weights [32,32], bias [32]
        var model = g.ToConcreteArchitecture(g.FromOrderedInputs([numOut, input])).ToConcreteModel();
        foreach (var node in model.ToInternal().Nodes)
        {
            if (node.OpCode != InternalOpCodes.MODEL_PARAM_DATA) continue;
            if (node.IdentifierTemplate ==
                    Shorokoo.Core.Nodes.Processors.Fast.FastWireRngKeyDerivation.RngSeedIdentifierTemplate)
                continue;
            var dims = node.GetTensorData()!.Shape.Dims;
            var vals = new float[dims.Aggregate(1L, (a, d) => a * d)];
            for (int i = 0; i < vals.Length; i++) vals[i] = 1.0f + i % 8 * 0.25f;
            node.Attributes = node.Attributes.SetAttributes(
                (OnnxOpAttributeNames.ShrkAttrTensorData, (object?)TensorData(dims, vals)));
        }
        return (model, numOut, input);
    }

    /// <summary>
    /// Opt-in per-entry Zstd compression (issue #75): .WithZstdCompressedData() shrinks the
    /// weights data entry, records compression "zstd" and a stored-bytes (compressed) sha256
    /// in the manifest, leaves config.json / models/*.srk and the STORED zip framing
    /// untouched (the file still reads through the BCL zip reader), keeps the default save
    /// byte-equivalent to the feature-less output, and round-trips bit-identically.
    /// </summary>
    [Fact]
    public void TestSkptZstdCompressedDataRoundTrip()
    {
        var (model, numOut, input) = BuildCompressibleSkptModel();
        var plainPath = Path.Combine(TempDir, "zstd-plain.skpt");
        var zstdPath = Path.Combine(TempDir, "zstd-on.skpt");
        try
        {
            Checkpoint.From(model).WithModel().WithWeights().Save(plainPath);
            Checkpoint.From(model).WithModel().WithWeights().WithZstdCompressedData().Save(zstdPath);

            // Both files stay standard zips (read via the BCL, not our writer) with the same
            // entry set, and every entry remains method-0 STORED — the Zstd layer lives
            // inside the data entry's bytes, not in the zip framing.
            var plainEntries = ReadZipEntries(plainPath);
            var zstdEntries = ReadZipEntries(zstdPath);
            Assert.Equal(plainEntries.Keys.OrderBy(n => n, StringComparer.Ordinal),
                zstdEntries.Keys.OrderBy(n => n, StringComparer.Ordinal));
            var zstdFileBytes = File.ReadAllBytes(zstdPath);
            Assert.All(ParseLocalZipHeaders(zstdFileBytes), h => Assert.Equal(0, h.Method));

            // Compression touches only the weights data entry: models/*.srk is byte-identical
            // across the two saves, and decompressing the compressed entry yields exactly the
            // default save's uncompressed entry — the default output is byte-unchanged by the
            // feature (its own layout is pinned by TestSkptRoundTripConcreteModel).
            Assert.Equal(plainEntries[SkptFileFormat.ModelEntryPath],
                zstdEntries[SkptFileFormat.ModelEntryPath]);
            var storedWeights = zstdEntries[SkptFileFormat.WeightsEntryPath];
            Assert.True(SkptFileFormat.LooksLikeZstdFrame(storedWeights));
            Assert.Equal(plainEntries[SkptFileFormat.WeightsEntryPath],
                CompressedFormatUtils.Decompress(storedWeights));

            // Compressible data: the compressed entry — and the whole file — is smaller.
            Assert.True(storedWeights.Length < plainEntries[SkptFileFormat.WeightsEntryPath].Length);
            Assert.True(zstdFileBytes.Length < new FileInfo(plainPath).Length);

            // Manifest: compression is recorded per entry ("none" by default, "zstd" when
            // opted in — never inferred from the entry name), and the compressed entry's
            // sha256 covers the stored (compressed) bytes, so integrity checking does not
            // require decompression.
            var plainData = Assert.Single(
                SkptFileFormat.ParseManifest(plainEntries[SkptFileFormat.ConfigEntryName], plainPath).Data!).Value;
            Assert.Equal(SkptFileFormat.CompressionNone, plainData.Compression);
            var zstdData = Assert.Single(
                SkptFileFormat.ParseManifest(zstdEntries[SkptFileFormat.ConfigEntryName], zstdPath).Data!).Value;
            Assert.Equal(SkptFileFormat.CompressionZstd, zstdData.Compression);
            Assert.Equal(SkptFileFormat.Sha256Hex(storedWeights), zstdData.Sha256);

            // Round-trip: weights bound byte-identically and bit-identical execution. The
            // pattern weights are non-zero, so byte-identity cannot pass via the model
            // entry's zero placeholders.
            var originalWeights = WeightBytesByParam(model);
            Assert.All(originalWeights.Values, bytes => Assert.Contains(bytes, b => b != 0));
            var loaded = Checkpoint.Load(zstdPath);
            var loadedWeights = WeightBytesByParam(loaded);
            Assert.Equal(originalWeights.Count, loadedWeights.Count);
            foreach (var (paramId, bytes) in originalWeights)
                Assert.Equal(bytes, loadedWeights[paramId]);
            Assert.Equal(ExecuteToBytes(model, numOut, input), ExecuteToBytes(loaded, numOut, input));

            // An out-of-range compression level is rejected up front.
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                Checkpoint.From(model).WithZstdCompressedData(compressionLevel: 0));
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                Checkpoint.From(model).WithZstdCompressedData(compressionLevel: 23));
        }
        finally
        {
            if (File.Exists(plainPath)) File.Delete(plainPath);
            if (File.Exists(zstdPath)) File.Delete(zstdPath);
        }
    }

    /// <summary>
    /// A manifest/stored compression mismatch fails loud naming the entry, in both
    /// directions: an entry declared "zstd" whose stored bytes are raw, an entry declared
    /// "none" whose stored bytes are a Zstd frame, an unknown compression name, and a
    /// declared-zstd entry whose frame is corrupt past its magic (with a matching sha256,
    /// so the failure comes from decompression, not the integrity check).
    /// </summary>
    [Fact]
    public void TestSkptCompressionMismatchFailsLoud()
    {
        var (model, _, _) = BuildCompressibleSkptModel();
        var plainPath = Path.Combine(TempDir, "mismatch-plain.skpt");
        var zstdPath = Path.Combine(TempDir, "mismatch-zstd.skpt");
        var tamperedPath = Path.Combine(TempDir, "mismatch-tampered.skpt");
        try
        {
            Checkpoint.From(model).WithModel().WithWeights().Save(plainPath);
            Checkpoint.From(model).WithModel().WithWeights().WithZstdCompressedData().Save(zstdPath);
            var plainEntries = ReadZipEntries(plainPath);
            var zstdEntries = ReadZipEntries(zstdPath);

            void RewriteWith(Dictionary<string, byte[]> source, string configJson, byte[]? weights = null)
                => RewriteSkpt(tamperedPath, source.Select(e => (e.Key,
                    e.Key == SkptFileFormat.ConfigEntryName ? System.Text.Encoding.UTF8.GetBytes(configJson)
                    : e.Key == SkptFileFormat.WeightsEntryPath && weights is not null ? weights
                    : e.Value)).ToList());

            // Declared "zstd", stored raw: the sha256 still matches the raw stored bytes, so
            // the mismatch is caught by the framing cross-check, naming the entry.
            var rawAsZstd = JsonNode.Parse(plainEntries[SkptFileFormat.ConfigEntryName])!;
            rawAsZstd["data"]!["weights"]!["compression"] = SkptFileFormat.CompressionZstd;
            RewriteWith(plainEntries, rawAsZstd.ToJsonString());
            var exRaw = Assert.Throws<InvalidDataException>(() => Checkpoint.Load(tamperedPath));
            Assert.Contains(SkptFileFormat.WeightsEntryPath, exRaw.Message);
            Assert.Contains("not a Zstd frame", exRaw.Message);

            // Declared "none", stored compressed: fails loud naming the entry instead of
            // feeding a Zstd frame to the safetensors parser.
            var zstdAsRaw = JsonNode.Parse(zstdEntries[SkptFileFormat.ConfigEntryName])!;
            zstdAsRaw["data"]!["weights"]!["compression"] = SkptFileFormat.CompressionNone;
            RewriteWith(zstdEntries, zstdAsRaw.ToJsonString());
            var exZstd = Assert.Throws<InvalidDataException>(() => Checkpoint.Load(tamperedPath));
            Assert.Contains(SkptFileFormat.WeightsEntryPath, exZstd.Message);
            Assert.Contains("Zstd frame", exZstd.Message);

            // An unknown compression name is refused as unsupported (future-format skew).
            var unknown = JsonNode.Parse(plainEntries[SkptFileFormat.ConfigEntryName])!;
            unknown["data"]!["weights"]!["compression"] = "lz4";
            RewriteWith(plainEntries, unknown.ToJsonString());
            var exUnknown = Assert.Throws<InvalidDataException>(() => Checkpoint.Load(tamperedPath));
            Assert.Contains("lz4", exUnknown.Message);
            Assert.Contains("unsupported compression", exUnknown.Message);

            // A corrupt Zstd frame (truncated past the magic) with a recomputed, matching
            // sha256: the integrity check passes and decompression fails loud, naming the
            // entry — never returning garbage bytes.
            var truncated = zstdEntries[SkptFileFormat.WeightsEntryPath]
                .Take(zstdEntries[SkptFileFormat.WeightsEntryPath].Length / 2).ToArray();
            Assert.True(SkptFileFormat.LooksLikeZstdFrame(truncated));
            var corrupt = JsonNode.Parse(zstdEntries[SkptFileFormat.ConfigEntryName])!;
            corrupt["data"]!["weights"]!["sha256"] = SkptFileFormat.Sha256Hex(truncated);
            RewriteWith(zstdEntries, corrupt.ToJsonString(), truncated);
            var exCorrupt = Assert.Throws<InvalidDataException>(() => Checkpoint.Load(tamperedPath));
            Assert.Contains(SkptFileFormat.WeightsEntryPath, exCorrupt.Message);
            Assert.Contains("Zstd-decompress", exCorrupt.Message);
        }
        finally
        {
            if (File.Exists(plainPath)) File.Delete(plainPath);
            if (File.Exists(zstdPath)) File.Delete(zstdPath);
            if (File.Exists(tamperedPath)) File.Delete(tamperedPath);
        }
    }

    /// <summary>
    /// Checkpoint.Inspect recognizes the .skpt container (issue #73): a checkpoint written
    /// by the Save path inspects to SkptCheckpoint with the manifest's whole-archive
    /// metadata, model and data registries (sha256 reported as recorded, never verified)
    /// and mapping-set names — reading only the zip central directory plus config.json, so
    /// a corrupt tensor payload does not disturb inspection (the same payload-untouched
    /// technique as the .srk corruption case). Cheap sanity observations fire on
    /// manifest/archive mismatches in both directions, unknown keys, a future version,
    /// STORED-expectation violations and empty trees; and a non-.skpt zip, a foreign
    /// config.json, a garbage manifest and a truncated archive all yield structured
    /// NotRecognized results — never an exception.
    /// </summary>
    [Fact]
    public void TestCheckpointInspectSkptArtifacts()
    {
        var (model, _, _) = BuildSkptModel();
        var path = Path.Combine(TempDir, "inspect.skpt");
        var variantPath = Path.Combine(TempDir, "inspect_variant.skpt");
        try
        {
            Checkpoint.From(model).WithModel().WithWeights().Save(path);
            var entries = ReadZipEntries(path);
            var manifest = SkptFileFormat.ParseManifest(entries[SkptFileFormat.ConfigEntryName], path);
            List<(string Name, byte[] Data)> WithConfig(string configJson) =>
                entries.Select(e => (e.Key, e.Key == SkptFileFormat.ConfigEntryName
                    ? System.Text.Encoding.UTF8.GetBytes(configJson) : e.Value)).ToList();

            // The clean checkpoint: new kind, whole-archive metadata, both registries and
            // the mapping-set names all match what Save wrote; no observations.
            var result = Checkpoint.Inspect(path);
            Assert.Equal(ArtifactKind.SkptCheckpoint, result.Kind);
            Assert.Equal(path, result.FilePath);
            Assert.Equal(new FileInfo(path).Length, result.FileSizeBytes);
            Assert.Null(result.Srk);
            Assert.Null(result.SafeTensors);
            Assert.Null(result.TrainingCheckpoint);
            Assert.Empty(result.Observations);

            var skpt = result.Skpt!;
            Assert.NotNull(skpt);
            Assert.Equal(SkptFileFormat.FormatName, skpt.FormatName);
            Assert.Equal(SkptFileFormat.CurrentVersion, skpt.SkptVersion);
            Assert.Equal(manifest.CreatedUtc, skpt.CreatedUtc);
            Assert.Equal(Shorokoo.ShorokooVersion.VersionString, skpt.Producer);

            var modelSummary = Assert.Single(skpt.Models);
            Assert.Equal("model", modelSummary.Key);
            Assert.Equal(SkptFileFormat.ModelEntryPath, modelSummary.EntryPath);
            Assert.Equal(SkptFileFormat.ModelFormatSrk2, modelSummary.Format);
            Assert.Equal(SrkFileFormat.StageName(GraphKind.ConcreteModel), modelSummary.Stage);
            Assert.Equal(SkptFileFormat.Sha256Hex(entries[SkptFileFormat.ModelEntryPath]),
                modelSummary.GraphHash);

            var dataSummary = Assert.Single(skpt.DataEntries);
            Assert.Equal("weights", dataSummary.Key);
            Assert.Equal(SkptFileFormat.WeightsEntryPath, dataSummary.EntryPath);
            Assert.Equal(SkptFileFormat.DataFormatSafeTensors, dataSummary.Format);
            Assert.Equal(SkptFileFormat.CompressionNone, dataSummary.Compression);
            Assert.Equal(entries[SkptFileFormat.WeightsEntryPath].LongLength,
                dataSummary.DeclaredSizeBytes);
            Assert.Equal(SkptFileFormat.Sha256Hex(entries[SkptFileFormat.WeightsEntryPath]),
                dataSummary.Sha256);

            string[] expectedSets = ["default"];
            Assert.Equal(expectedSets, skpt.MappingSetNames);

            // ToString renders the model + data inventory.
            var text = result.ToString();
            Assert.Contains(".skpt", text);
            Assert.Contains(SkptFileFormat.ModelEntryPath, text);
            Assert.Contains(SkptFileFormat.WeightsEntryPath, text);
            Assert.Contains("unverified", text);
            Assert.Contains("mapping sets: default", text);

            // Payload untouched: corrupt one byte inside the weights payload — a full load
            // fails its SHA-256 check, but Inspect (which never reads payload bytes) returns
            // the same clean summary, recorded hash included.
            var fileBytes = File.ReadAllBytes(path);
            var weightsHeader = ParseLocalZipHeaders(fileBytes)
                .Single(h => h.Name == SkptFileFormat.WeightsEntryPath);
            fileBytes[weightsHeader.DataOffset + weightsHeader.Size - 1] ^= 0xFF;
            File.WriteAllBytes(variantPath, fileBytes);
            Assert.Throws<InvalidDataException>(() => Checkpoint.Load(variantPath));
            var corrupt = Checkpoint.Inspect(variantPath);
            Assert.Equal(ArtifactKind.SkptCheckpoint, corrupt.Kind);
            Assert.Empty(corrupt.Observations);
            Assert.Equal(dataSummary.Sha256, Assert.Single(corrupt.Skpt!.DataEntries).Sha256);

            // Manifest/archive mismatches in both directions are observed; the file still
            // inspects as a .skpt.
            var mismatched = entries.Where(e => e.Key != SkptFileFormat.WeightsEntryPath)
                .Select(e => (e.Key, e.Value)).Append(("data/stray.bin", new byte[16])).ToList();
            RewriteSkpt(variantPath, mismatched);
            var mismatch = Checkpoint.Inspect(variantPath);
            Assert.Equal(ArtifactKind.SkptCheckpoint, mismatch.Kind);
            Assert.Contains(mismatch.Observations,
                o => o.Contains(SkptFileFormat.WeightsEntryPath) && o.Contains("no such entry"));
            Assert.Contains(mismatch.Observations,
                o => o.Contains("data/stray.bin") && o.Contains("not referenced"));
            Assert.Null(Assert.Single(mismatch.Skpt!.DataEntries).DeclaredSizeBytes);

            // Unknown manifest keys and a future version are observations, not failures.
            var futureConfig = JsonNode.Parse(entries[SkptFileFormat.ConfigEntryName])!;
            futureConfig["futureTopLevelKey"] = "??";
            futureConfig["skptVersion"] = SkptFileFormat.CurrentVersion + 1;
            RewriteSkpt(variantPath, WithConfig(futureConfig.ToJsonString()));
            var future = Checkpoint.Inspect(variantPath);
            Assert.Equal(ArtifactKind.SkptCheckpoint, future.Kind);
            Assert.Equal(SkptFileFormat.CurrentVersion + 1, future.Skpt!.SkptVersion);
            Assert.Contains(future.Observations, o => o.Contains("futureTopLevelKey"));
            Assert.Contains(future.Observations,
                o => o.Contains($"version {SkptFileFormat.CurrentVersion + 1}"));

            // STORED-expectation violation: the same entries written deflated (via the BCL
            // writer) still inspect as a .skpt, with observations naming compressed entries.
            using (var deflated = new ZipArchive(File.Create(variantPath), ZipArchiveMode.Create))
            {
                foreach (var (name, data) in entries)
                {
                    using var s = deflated.CreateEntry(name, CompressionLevel.SmallestSize).Open();
                    s.Write(data);
                }
            }
            var storedViolation = Checkpoint.Inspect(variantPath);
            Assert.Equal(ArtifactKind.SkptCheckpoint, storedViolation.Kind);
            Assert.Contains(storedViolation.Observations, o => o.Contains("expected STORED"));

            // Empty trees: a bare identity-only manifest inspects with observations for the
            // missing models / data / mapping sets.
            RewriteSkpt(variantPath, [(SkptFileFormat.ConfigEntryName,
                System.Text.Encoding.UTF8.GetBytes("{\"format\":\"skpt\",\"skptVersion\":1}"))]);
            var empty = Checkpoint.Inspect(variantPath);
            Assert.Equal(ArtifactKind.SkptCheckpoint, empty.Kind);
            Assert.Contains(empty.Observations, o => o.Contains("no models"));
            Assert.Contains(empty.Observations, o => o.Contains("no data entries"));
            Assert.Contains(empty.Observations, o => o.Contains("no tensor mapping sets"));

            // Non-.skpt zips: no config.json, a foreign tool's config.json, and a manifest
            // that is not JSON — structured NotRecognized every time, never an exception.
            RewriteSkpt(variantPath,
                [("readme.txt", System.Text.Encoding.UTF8.GetBytes("just a zip"))]);
            var noConfig = Checkpoint.Inspect(variantPath);
            Assert.Equal(ArtifactKind.NotRecognized, noConfig.Kind);
            Assert.Contains(noConfig.Observations, o => o.Contains(SkptFileFormat.ConfigEntryName));

            RewriteSkpt(variantPath, [(SkptFileFormat.ConfigEntryName,
                System.Text.Encoding.UTF8.GetBytes("{\"name\":\"some-other-tool\"}"))]);
            var foreign = Checkpoint.Inspect(variantPath);
            Assert.Equal(ArtifactKind.NotRecognized, foreign.Kind);
            Assert.Contains(foreign.Observations, o => o.Contains("format"));

            RewriteSkpt(variantPath, [(SkptFileFormat.ConfigEntryName,
                System.Text.Encoding.UTF8.GetBytes("not json at all"))]);
            var badJson = Checkpoint.Inspect(variantPath);
            Assert.Equal(ArtifactKind.NotRecognized, badJson.Kind);
            Assert.Contains(badJson.Observations, o => o.Contains("not a readable"));

            // Truncated and garbage files with a zip signature: structured NotRecognized.
            File.WriteAllBytes(variantPath, File.ReadAllBytes(path)[..40]);
            var truncated = Checkpoint.Inspect(variantPath);
            Assert.Equal(ArtifactKind.NotRecognized, truncated.Kind);
            Assert.Contains(truncated.Observations, o => o.Contains("not readable"));

            byte[] garbageZip = [0x50, 0x4B, 0x03, 0x04, 0xDE, 0xAD, 0xBE, 0xEF];
            File.WriteAllBytes(variantPath, garbageZip);
            var garbage = Checkpoint.Inspect(variantPath);
            Assert.Equal(ArtifactKind.NotRecognized, garbage.Kind);
            Assert.NotEmpty(garbage.Observations);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
            if (File.Exists(variantPath)) File.Delete(variantPath);
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // SafeTensors weight exchange (issue #74): ExportSafeTensors writes a
    // model's weights to a standard .safetensors file (canonical names or a
    // naming scheme); ImportSafeTensors binds a foreign .safetensors onto a
    // concrete architecture with strict fail-loud mapping checks; the
    // one-call .skpt landing reuses the container writer.
    // ──────────────────────────────────────────────────────────────────────

    /// <summary>The two canonical FCLayer parameter ids (weights [4,4] and bias [4]) —
    /// what the canonical naming scheme produces for the architecture's two params.</summary>
    private const string FcWeightsId = "TrainableParam#0.InitSimple#0";
    private const string FcBiasId = "TrainableParam#0.InitSimple#1";

    /// <summary>Builds the FCLayer architecture + concrete model + sample inputs the
    /// safetensors exchange tests share.</summary>
    private static (ComputationGraph Arch, ComputationGraph Model, TensorData NumOut, TensorData Input)
        BuildSafeTensorsExchangeModel()
    {
        var numOut = TensorData(DType.Int64, [], 4L);
        var input = TensorDataWithSmallVals(DType.Float32, [4L, 4L]);
        var g = FCLayer.ComputationGraph;
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([numOut, input]));
        return (arch, arch.ToConcreteModel(), numOut, input);
    }

    /// <summary>The model's weight bytes keyed by <b>canonical</b> parameter name — the
    /// graph nodes carry the serialized "[ModelId]:parts" identifier; the canonical name
    /// (what export writes and naming schemes match) is the parts portion.</summary>
    private static Dictionary<string, byte[]> CanonicalWeightBytes(ComputationGraph model)
        => WeightBytesByParam(model).ToDictionary(
            kv => kv.Key[(kv.Key.IndexOf("]:", StringComparison.Ordinal) + 2)..],
            kv => kv.Value, StringComparer.Ordinal);

    /// <summary>A PyTorch-style naming scheme for the FCLayer architecture: the two
    /// canonical parameter ids map to torch-conventional <c>fc.weight</c> / <c>fc.bias</c>.</summary>
    private static SimplePatternNamingScheme BuildFcTorchScheme(ComputationGraph arch)
    {
        SimplePatternScheme[] patterns =
        [
            new SimplePatternScheme(FcWeightsId, "fc.weight"),
            new SimplePatternScheme(FcBiasId, "fc.bias"),
        ];
        return new SimplePatternNamingScheme(
            patterns, arch.GetShorokooIdNamingScheme(), ModuleParamSetNamingScheme.PyTorchFrameworkId);
    }

    /// <summary>
    /// The export → import acceptance round-trip, in both naming modes. Canonical:
    /// ExportSafeTensors writes one tensor per weight parameter under its canonical
    /// Shorokoo id, byte-identical to the model's weights, into a plain safetensors
    /// file (read back with the independent loader); ImportSafeTensors binds it onto
    /// the architecture and the imported model executes bit-identically. Scheme: the
    /// same round-trip through PyTorch-style names (fc.weight / fc.bias) with the
    /// scheme applied at both boundaries.
    /// </summary>
    [Fact]
    public void TestSafeTensorsExportImportRoundTrip()
    {
        var (arch, model, numOut, input) = BuildSafeTensorsExchangeModel();
        var canonicalPath = Path.Combine(TempDir, "exchange_canonical.safetensors");
        var torchPath = Path.Combine(TempDir, "exchange_torch.safetensors");
        try
        {
            var originalWeights = CanonicalWeightBytes(model);
            var direct = ExecuteToBytes(model, numOut, input);
            Assert.Equal([FcWeightsId, FcBiasId],
                originalWeights.Keys.OrderBy(k => k, StringComparer.Ordinal).ToArray());

            // ── Canonical names (no scheme) ────────────────────────────────
            Checkpoint.ExportSafeTensors(model, canonicalPath);
            var storedCanonical = SafeTensorLoader.LoadSafeTensors(canonicalPath)
                .ToDictionary(t => t.Name, t => t.Data.AccessRawMemory().ToArray(), StringComparer.Ordinal);
            Assert.Equal(originalWeights.Keys.OrderBy(k => k, StringComparer.Ordinal),
                storedCanonical.Keys.OrderBy(k => k, StringComparer.Ordinal));
            foreach (var (paramId, bytes) in originalWeights)
                Assert.Equal(bytes, storedCanonical[paramId]);

            var importedCanonical = Checkpoint.ImportSafeTensors(arch, canonicalPath);
            Assert.Equal(GraphKind.ConcreteModel, importedCanonical.Kind);
            Assert.Equal(originalWeights, CanonicalWeightBytes(importedCanonical));
            Assert.Equal(direct, ExecuteToBytes(importedCanonical, numOut, input));

            // A __metadata__ block is metadata, not a tensor — a file carrying one imports
            // unchanged (it must not trip the unmapped-source-tensor check).
            var withMetadata = SafeTensorLoader.LoadSafeTensors(canonicalPath);
            SafeTensorLoader.SaveSafeTensors(canonicalPath, withMetadata,
                new Dictionary<string, object> { ["format"] = "pt", ["producer"] = "unit-test" });
            var importedWithMetadata = Checkpoint.ImportSafeTensors(arch, canonicalPath);
            Assert.Equal(direct, ExecuteToBytes(importedWithMetadata, numOut, input));

            // ── PyTorch-style names via a naming scheme ────────────────────
            var scheme = BuildFcTorchScheme(arch);
            Checkpoint.ExportSafeTensors(model, torchPath, scheme);
            var storedTorch = SafeTensorLoader.LoadSafeTensors(torchPath)
                .ToDictionary(t => t.Name, t => t.Data.AccessRawMemory().ToArray(), StringComparer.Ordinal);
            Assert.Equal(["fc.bias", "fc.weight"],
                storedTorch.Keys.OrderBy(k => k, StringComparer.Ordinal).ToArray());
            Assert.Equal(originalWeights[FcWeightsId], storedTorch["fc.weight"]);
            Assert.Equal(originalWeights[FcBiasId], storedTorch["fc.bias"]);

            var importedTorch = Checkpoint.ImportSafeTensors(arch, torchPath, scheme);
            Assert.Equal(originalWeights, CanonicalWeightBytes(importedTorch));
            Assert.Equal(direct, ExecuteToBytes(importedTorch, numOut, input));

            // The ModelId format DSL binds at the import boundary too: the same torch-named
            // file imports under a ModelIdNamingScheme keyed on the params' ModelIds ([1]/[2]).
            ModelIdFormat[] formats =
            [
                new ModelIdFormat(match: "[1]", format: "fc.weight"),
                new ModelIdFormat(match: "[2]", format: "fc.bias"),
            ];
            var formatScheme = new ModelIdNamingScheme(
                formats, ModuleParamSetNamingScheme.PyTorchFrameworkId);
            var importedFormat = Checkpoint.ImportSafeTensors(arch, torchPath, formatScheme);
            Assert.Equal(originalWeights, CanonicalWeightBytes(importedFormat));
            Assert.Equal(direct, ExecuteToBytes(importedFormat, numOut, input));
        }
        finally
        {
            if (File.Exists(canonicalPath)) File.Delete(canonicalPath);
            if (File.Exists(torchPath)) File.Delete(torchPath);
        }
    }

    /// <summary>
    /// Import fails loudly, naming the offending tensor, on every mapping-mismatch
    /// class: a source tensor that maps to no parameter (with a dedicated hint when the
    /// file is a training checkpoint), a required parameter with no source tensor, a
    /// dtype mismatch, a shape mismatch, an ambiguous scheme mapping two parameters to
    /// one name, and a scheme that fails to name a required parameter. Kind gates
    /// refuse the wrong graph stage up front, and validation always precedes binding.
    /// </summary>
    [Fact]
    public void TestImportSafeTensorsFailsLoudOnMappingMismatches()
    {
        var (arch, model, _, _) = BuildSafeTensorsExchangeModel();
        var path = Path.Combine(TempDir, "exchange_good.safetensors");
        var badPath = Path.Combine(TempDir, "exchange_bad.safetensors");
        try
        {
            Checkpoint.ExportSafeTensors(model, path);
            var good = SafeTensorLoader.LoadSafeTensors(path);
            const string weightsId = FcWeightsId;
            const string biasId = FcBiasId;

            // A source tensor mapping to nothing names the tensor (and the file).
            var withStray = good.ToList();
            withStray.Add(new SafeTensor("not.a.param", TensorData([2L], 1f, 2f), "F32", [2L]));
            SafeTensorLoader.SaveSafeTensors(badPath, withStray);
            var exStray = Assert.Throws<InvalidDataException>(() => Checkpoint.ImportSafeTensors(arch, badPath));
            Assert.Contains("not.a.param", exStray.Message);
            Assert.Contains(badPath, exStray.Message);

            // A required parameter with no source tensor names the parameter.
            SafeTensorLoader.SaveSafeTensors(badPath, good.Where(t => t.Name != biasId).ToList());
            var exMissing = Assert.Throws<InvalidDataException>(() => Checkpoint.ImportSafeTensors(arch, badPath));
            Assert.Contains(biasId, exMissing.Message);

            // A dtype mismatch after mapping names the tensor and both dtypes.
            var wrongDtype = good.Select(t => t.Name != weightsId ? t
                : new SafeTensor(t.Name, TensorDataWithSmallVals(DType.Int64, [4L, 4L]), "I64", [4L, 4L])).ToList();
            SafeTensorLoader.SaveSafeTensors(badPath, wrongDtype);
            var exDtype = Assert.Throws<InvalidDataException>(() => Checkpoint.ImportSafeTensors(arch, badPath));
            Assert.Contains(weightsId, exDtype.Message);
            Assert.Contains("dtype", exDtype.Message);

            // A shape mismatch after mapping names the tensor and both shapes.
            var wrongShape = good.Select(t => t.Name != weightsId ? t
                : new SafeTensor(t.Name, TensorDataWithSmallVals(DType.Float32, [2L, 8L]), "F32", [2L, 8L])).ToList();
            SafeTensorLoader.SaveSafeTensors(badPath, wrongShape);
            var exShape = Assert.Throws<InvalidDataException>(() => Checkpoint.ImportSafeTensors(arch, badPath));
            Assert.Contains(weightsId, exShape.Message);
            Assert.Contains("[2,8]", exShape.Message);
            Assert.Contains("[4,4]", exShape.Message);

            // A scheme mapping two parameters onto one source name is ambiguous —
            // refused naming both parameters, before any tensor lookup can pick one.
            SimplePatternScheme[] colliding =
            [
                new SimplePatternScheme("TrainableParam#0.InitSimple#{p}", "fc.same"),
            ];
            var ambiguous = new SimplePatternNamingScheme(
                colliding, arch.GetShorokooIdNamingScheme(), ModuleParamSetNamingScheme.PyTorchFrameworkId);
            var exAmbiguous = Assert.Throws<InvalidDataException>(
                () => Checkpoint.ImportSafeTensors(arch, path, ambiguous));
            Assert.Contains(weightsId, exAmbiguous.Message);
            Assert.Contains(biasId, exAmbiguous.Message);
            Assert.Contains("fc.same", exAmbiguous.Message);

            // A scheme covering only some parameters names the uncovered one — even when
            // every tensor the file does carry maps cleanly.
            SimplePatternScheme[] partial =
            [
                new SimplePatternScheme(weightsId, "fc.weight"),
            ];
            var partialScheme = new SimplePatternNamingScheme(
                partial, arch.GetShorokooIdNamingScheme(), ModuleParamSetNamingScheme.PyTorchFrameworkId);
            var weightsOnly = good.Single(t => t.Name == weightsId);
            SafeTensorLoader.SaveSafeTensors(badPath,
                [new SafeTensor("fc.weight", weightsOnly.Data, weightsOnly.DataType, weightsOnly.Shape)]);
            var exUncovered = Assert.Throws<InvalidDataException>(
                () => Checkpoint.ImportSafeTensors(arch, badPath, partialScheme));
            Assert.Contains(biasId, exUncovered.Message);
            Assert.Contains("naming scheme", exUncovered.Message);

            // A training checkpoint is recognized by its marker and redirected.
            SafeTensorLoader.SaveSafeTensors(badPath,
                [new SafeTensor("__shorokoo_checkpoint__", TensorData([2L], 1L, 0L), "I64", [2L])]);
            var exCheckpoint = Assert.Throws<InvalidDataException>(() => Checkpoint.ImportSafeTensors(arch, badPath));
            Assert.Contains("training checkpoint", exCheckpoint.Message);

            // Kind gates: import takes a concrete architecture, export a concrete model.
            var exImportKind = Assert.Throws<InvalidOperationException>(
                () => Checkpoint.ImportSafeTensors(FCLayer.ComputationGraph, path));
            Assert.Contains("concrete-architecture", exImportKind.Message);
            var exImportModel = Assert.Throws<InvalidOperationException>(
                () => Checkpoint.ImportSafeTensors(model, path));
            Assert.Contains("concrete-architecture", exImportModel.Message);
            var exExportKind = Assert.Throws<InvalidOperationException>(
                () => Checkpoint.ExportSafeTensors(arch, badPath));
            Assert.Contains("concrete-model", exExportKind.Message);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
            if (File.Exists(badPath)) File.Delete(badPath);
        }
    }

    /// <summary>
    /// Export-side fail-loud checks: a scheme that leaves a parameter unnamed refuses
    /// the export naming the parameter (weights are never silently dropped); two
    /// parameters colliding on one exported name are refused naming both; and a
    /// ModelId-keyed scheme — which cannot translate the canonical id strings a bound
    /// model carries — is refused with the supported alternative named. A failed
    /// export writes nothing (the atomic writer never commits).
    /// </summary>
    [Fact]
    public void TestExportSafeTensorsFailsLoudOnSchemeGaps()
    {
        var (arch, model, _, _) = BuildSafeTensorsExchangeModel();
        var path = Path.Combine(TempDir, "exchange_export_fail.safetensors");
        const string weightsId = FcWeightsId;
        const string biasId = FcBiasId;

        // Scheme gap: the bias has no rule → refused naming the parameter.
        SimplePatternScheme[] partial = [new SimplePatternScheme(weightsId, "fc.weight")];
        var partialScheme = new SimplePatternNamingScheme(
            partial, arch.GetShorokooIdNamingScheme(), ModuleParamSetNamingScheme.PyTorchFrameworkId);
        var exGap = Assert.Throws<InvalidOperationException>(
            () => Checkpoint.ExportSafeTensors(model, path, partialScheme));
        Assert.Contains(biasId, exGap.Message);

        // Name collision after remapping → refused naming both parameters and the name.
        SimplePatternScheme[] colliding =
        [
            new SimplePatternScheme("TrainableParam#0.InitSimple#{p}", "fc.same"),
        ];
        var collidingScheme = new SimplePatternNamingScheme(
            colliding, arch.GetShorokooIdNamingScheme(), ModuleParamSetNamingScheme.PyTorchFrameworkId);
        var exCollision = Assert.Throws<InvalidOperationException>(
            () => Checkpoint.ExportSafeTensors(model, path, collidingScheme));
        Assert.Contains(weightsId, exCollision.Message);
        Assert.Contains(biasId, exCollision.Message);
        Assert.Contains("fc.same", exCollision.Message);

        // A ModelId-keyed scheme cannot run in the export direction: a bound model
        // carries canonical id strings, not ModelIds.
        var modelIdScheme = new ModelIdNamingScheme(
            [new ModelIdFormat(format: "x")], ModuleParamSetNamingScheme.PyTorchFrameworkId);
        Assert.Throws<NotSupportedException>(
            () => Checkpoint.ExportSafeTensors(model, path, modelIdScheme));

        // No failed attempt above committed a file.
        Assert.False(File.Exists(path));
    }

    /// <summary>
    /// The one-call native landing: ImportSafeTensorsToCheckpoint imports a foreign
    /// (PyTorch-named) safetensors file under a scheme and writes the bound result
    /// straight to a .skpt via the standard container writer. The returned model, the
    /// reloaded checkpoint, and the original model all execute bit-identically, and
    /// the checkpoint's weights are byte-identical to the source tensors.
    /// </summary>
    [Fact]
    public void TestImportSafeTensorsToCheckpointRoundTrip()
    {
        var (arch, model, numOut, input) = BuildSafeTensorsExchangeModel();
        var torchPath = Path.Combine(TempDir, "exchange_landing.safetensors");
        var skptPath = Path.Combine(TempDir, "exchange_landing.skpt");
        try
        {
            var scheme = BuildFcTorchScheme(arch);
            Checkpoint.ExportSafeTensors(model, torchPath, scheme);

            var imported = Checkpoint.ImportSafeTensorsToCheckpoint(arch, torchPath, skptPath, scheme);
            Assert.Equal(GraphKind.ConcreteModel, imported.Kind);

            var direct = ExecuteToBytes(model, numOut, input);
            Assert.Equal(direct, ExecuteToBytes(imported, numOut, input));

            var reloaded = Checkpoint.Load(skptPath);
            Assert.Equal(GraphKind.ConcreteModel, reloaded.Kind);
            Assert.Equal(WeightBytesByParam(model), WeightBytesByParam(reloaded));
            Assert.Equal(direct, ExecuteToBytes(reloaded, numOut, input));

            // A failed import lands nothing: reusing the paths with a scheme gap leaves
            // the previously written checkpoint bytes untouched.
            var committed = File.ReadAllBytes(skptPath);
            SimplePatternScheme[] partial =
            [
                new SimplePatternScheme(FcWeightsId, "fc.weight"),
            ];
            var partialScheme = new SimplePatternNamingScheme(
                partial, arch.GetShorokooIdNamingScheme(), ModuleParamSetNamingScheme.PyTorchFrameworkId);
            Assert.Throws<InvalidDataException>(
                () => Checkpoint.ImportSafeTensorsToCheckpoint(arch, torchPath, skptPath, partialScheme));
            Assert.Equal(committed, File.ReadAllBytes(skptPath));
        }
        finally
        {
            if (File.Exists(torchPath)) File.Delete(torchPath);
            if (File.Exists(skptPath)) File.Delete(skptPath);
        }
    }

}

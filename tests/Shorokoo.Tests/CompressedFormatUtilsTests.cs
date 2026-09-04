using System.IO.Compression;
using System.Text.Json.Nodes;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// The persisted artifact formats: raw zstd streams, the .srk graph container, .safetensors /
/// .zsafetensor archives, the .skpt checkpoint container, the SafeTensors and ONNX exchange
/// boundaries, and <c>Persistence.Inspect</c> over all of them.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class CompressedFormatUtilsCoverageTests : IDisposable
{
    private readonly string _tempDir =
        Path.Combine(Path.GetTempPath(), $"ShorokooCompressedCoverage_{Guid.NewGuid():N}");

    public CompressedFormatUtilsCoverageTests() => Directory.CreateDirectory(_tempDir);

    public void Dispose() => Directory.Delete(_tempDir, recursive: true);

    private string P(string name) => Path.Combine(_tempDir, name);

    private static void AssertInspection(ArtifactInspection result, ArtifactKind kind,
        params string[][] observationsContainingAll)
    {
        Assert.Equal(kind, result.Kind);
        foreach (var group in observationsContainingAll)
            Assert.Contains(result.Observations, o => group.All(f => o.Contains(f)));
    }

    [Fact]
    public void TestRawZstdArchitectureJsonAndSafeTensorsRoundTrips()
    {
        var originalBytes = new byte[10000];
        for (int i = 0; i < originalBytes.Length; i++)
            originalBytes[i] = (byte)(i % 10);
        var compressed = CompressedFormatUtils.Compress(originalBytes);
        Assert.True(compressed.Length < originalBytes.Length);
        Assert.Equal(originalBytes, CompressedFormatUtils.Decompress(compressed));

        var zstPath = P("test_roundtrip.zst");
        CompressedFormatUtils.CompressToFile(zstPath, originalBytes);
        Assert.Equal(originalBytes, CompressedFormatUtils.DecompressFile(zstPath));
        Assert.Throws<FileNotFoundException>(
            () => CompressedFormatUtils.DecompressFile(P("nope.zst")));

        using (var memStream = new MemoryStream())
        {
            CompressedFormatUtils.CompressToStream(memStream, originalBytes);
            memStream.Position = 0;
            Assert.Equal(originalBytes, CompressedFormatUtils.DecompressStream(memStream));
        }

        var input = InputTensor<float32>("input");
        var output = input + Scalar(1.0f);
        var graph = new InternalComputationGraph([input], [output]);
        var fastGraph = (graph);

        var zsrkPath = P("test_arch.zsrk");
        var zsrkPath2 = P("test_arch2.zsrk");
        CompressedFormatUtils.SaveFastGraphToFile(zsrkPath, fastGraph);
        var loaded = CompressedFormatUtils.LoadFastGraphFromFile(zsrkPath);
        Assert.Equal(graph.InputTensors.Count(), loaded.ToInternal().InputTensors.Count());
        Assert.Equal(graph.OutputTensors.Count(), loaded.ToInternal().OutputTensors.Count());
        File.WriteAllBytes(P("test_arch.bin"), CompressedFormatUtils.SaveFastGraphToBinary(fastGraph));

        var json = CompressedFormatUtils.ToJson(zsrkPath);
        Assert.False(string.IsNullOrEmpty(json));
        Assert.Contains("Graph", json);

        var jsonPath = P("test_arch.json");
        Assert.Equal(jsonPath, CompressedFormatUtils.SaveAsJson(zsrkPath, jsonPath));
        Assert.True(File.Exists(jsonPath));
        var derivedPath = CompressedFormatUtils.SaveAsJson(zsrkPath);
        Assert.True(File.Exists(derivedPath));
        Assert.EndsWith(".json", derivedPath);

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

        var listing = CompressedFormatUtils.GetNodeAndTensorNameListing(zsrkPath);
        Assert.False(string.IsNullOrEmpty(listing));
        Assert.Contains("\n", listing);

        Assert.Throws<FileNotFoundException>(
            () => CompressedFormatUtils.LoadFastGraphFromFile(P("nope.zsrk")));

        var t1 = TensorData([2, 3], 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);
        var t2 = TensorData([3], 7.0f, 8.0f, 9.0f);
        var tensors = new List<SafeTensor>
        {
            new SafeTensor("tensor1", t1, "F32", t1.Shape.Dims),
            new SafeTensor("tensor2", t2, "F32", t2.Shape.Dims),
        };
        var zsafePath = P("test_tensors.zsafetensor");
        CompressedFormatUtils.SaveCompressedSafeTensors(zsafePath, tensors);
        Assert.Equal(2, CompressedFormatUtils.LoadCompressedSafeTensors(zsafePath).Count);
        var dict = CompressedFormatUtils.LoadCompressedTensorDictionary(zsafePath);
        Assert.True(dict.ContainsKey("tensor1") && dict.ContainsKey("tensor2"));
        Assert.Throws<InvalidOperationException>(
            () => CompressedFormatUtils.LoadCompressedSingleTensor(zsafePath));

        var zsafeSinglePath = P("test_single.zsafetensor");
        CompressedFormatUtils.SaveCompressedSafeTensors(zsafeSinglePath,
            new List<SafeTensor> { new SafeTensor("only", t1, "F32", t1.Shape.Dims) });
        Assert.Equal(t1.Shape.Dims,
            CompressedFormatUtils.LoadCompressedSingleTensor(zsafeSinglePath).Shape.Dims);

        var paramSetPath = P("test_params.zsafetensor");
        (string name, TensorData data)[] paramPairs = [("p1", t1), ("p2", t2)];
        var paramList = new ModelParamList(paramPairs, ModelParamType.TrainableParam);
        CompressedFormatUtils.SaveCompressedModelParamSet(paramSetPath, paramList);
        var loadedParams = CompressedFormatUtils.LoadCompressedModelParamSet(paramSetPath);
        Assert.Equal(2, loadedParams.ModelParams.Length);
        Assert.NotNull(loadedParams.Find("p1"));
        Assert.NotNull(loadedParams.Find("p2"));

        Assert.True(CompressedFormatUtils.IsCompressedSafeTensor("foo.zsafetensor"));
        Assert.False(CompressedFormatUtils.IsCompressedSafeTensor("foo.safetensors"));
    }

    /// <summary>
    /// The raw format writers below the <c>Persistence.*</c> facade stage through
    /// <see cref="AtomicFileWriter"/> as well, so a crash in the commit window of any of them
    /// leaves the previously written file byte-for-byte as it was.
    /// </summary>
    [Fact]
    public void TestTheRawFormatWritersLeaveThePreviousFileIntactWhenACommitCrashes()
    {
        var graphA = new InternalComputationGraph([InputTensor<float32>("in")],
            [InputTensor<float32>("in") + Scalar(1.0f)]);
        var input = InputTensor<float32>("in");
        var graphB = new InternalComputationGraph([input], [input * Scalar(2.0f)]);
        var srcA = P("src-a.zsrk");
        var srcB = P("src-b.zsrk");
        CompressedFormatUtils.SaveFastGraphToFile(srcA, graphA, overrideExtension: false);
        CompressedFormatUtils.SaveFastGraphToFile(srcB, graphB, overrideExtension: false);
        var t1 = TensorData([2, 3], 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);
        var t2 = TensorData([3], 7.0f, 8.0f, 9.0f);
        List<SafeTensor> tensorsA = [new SafeTensor("a", t1, "F32", t1.Shape.Dims)];
        List<SafeTensor> tensorsB = [new SafeTensor("b", t2, "F32", t2.Shape.Dims)];

        (string Name, Action<string> Seed, Action<string> Rewrite)[] writers =
        [
            ("zst", p => CompressedFormatUtils.CompressToFile(p, [1, 2, 3]),
                    p => CompressedFormatUtils.CompressToFile(p, [4, 5, 6, 7])),
            ("zsrk", p => CompressedFormatUtils.SaveFastGraphToFile(p, graphA, overrideExtension: false),
                     p => CompressedFormatUtils.SaveFastGraphToFile(p, graphB, overrideExtension: false)),
            ("json", p => CompressedFormatUtils.SaveAsJson(srcA, p),
                     p => CompressedFormatUtils.SaveAsJson(srcB, p)),
            ("safetensors", p => SafeTensorLoader.SaveSafeTensors(p, tensorsA),
                            p => SafeTensorLoader.SaveSafeTensors(p, tensorsB)),
        ];
        string[] expected = ["kept", "kept", "kept", "kept"];
        string[] actual = [.. writers.Select(AfterACommitCrash)];
        Assert.Equal(expected, actual);
    }

    /// <summary>Seeds a fresh path with the writer's first form, then rewrites it with the
    /// second under a commit-window crash, and reports whether the seeded bytes survived.</summary>
    private string AfterACommitCrash((string Name, Action<string> Seed, Action<string> Rewrite) writer)
    {
        var path = P($"atomic-{writer.Name}");
        writer.Seed(path);
        var committed = File.ReadAllBytes(path);
        AtomicFileWriter.CommitFaultInjection = _ => throw new IOException("simulated commit crash");
        try { Assert.Throws<IOException>(() => writer.Rewrite(path)); }
        finally { AtomicFileWriter.CommitFaultInjection = null; }
        return committed.SequenceEqual(File.ReadAllBytes(path)) ? "kept" : "lost";
    }

    private static (ComputationGraph Module, ComputationGraph Arch, ComputationGraph Model)
        BuildStageGraphs()
    {
        var moduleGraph = ScalarMultiplyModel.ComputationGraph;
        var arch = moduleGraph.ToConcreteArchitecture(
            moduleGraph.FromOrderedInputs([TensorData([2], 1.0f, 2.0f)]));
        var model = arch.ToConcreteModel();
        return (moduleGraph, arch, model);
    }

    [Fact]
    public void TestSrkRoundtripAllStagesHeaderPeekAndRenamedFiles()
    {
        var (moduleGraph, arch, model) = BuildStageGraphs();

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
            Assert.True(SrkFileFormat.IsSrkContainer(bytes));

            var header = SrkFileFormat.TryReadHeader(bytes);
            Assert.NotNull(header);
            Assert.Equal(SrkFileFormat.CurrentVersion, header!.SrkVersion);
            Assert.Equal(SrkFileFormat.StageName(stage), header.Stage);
            Assert.Equal(stage, header.TryGetStage());
            Assert.Equal(compressed ? "zstd" : "none", header.Compression);
            Assert.False(string.IsNullOrEmpty(header.PayloadSha256));
            Assert.NotNull(header.Producer);
            Assert.Equal(Shorokoo.ShorokooVersion.VersionString, header.Producer!.Shorokoo);
            Assert.True(header.Producer.IrVersion > 0);
            Assert.NotNull(header.Producer.Opsets);
            Assert.NotEmpty(header.Producer.Opsets!);

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

        var reloadedModule = CompressedFormatUtils.LoadFastGraphFromBinary(
            CompressedFormatUtils.SaveFastGraphToBinary(moduleGraph));
        var rearch = reloadedModule.ToConcreteArchitecture(
            reloadedModule.FromOrderedInputs([TensorData([2], 1.0f, 2.0f)]));
        Assert.Equal(GraphKind.ConcreteArchitecture, rearch.Kind);
        Assert.Equal(GraphKind.ConcreteArchitecture, SrkFileFormat.DetectStage(rearch.ToInternal()));

        var srcPath = P("renamed_src.zsrk");
        string[] renamedPaths = [P("renamed_copy.srk"), P("renamed_copy.bin"), P("renamed_copy")];
        CompressedFormatUtils.SaveFastGraphToFile(srcPath, arch, compressed: true, overrideExtension: false);
        var referenceJson = CompressedFormatUtils.ToJson(srcPath);
        foreach (var renamed in renamedPaths)
        {
            File.Copy(srcPath, renamed, overwrite: true);
            Assert.NotEmpty(CompressedFormatUtils.LoadFastGraphFromFile(renamed).ToInternal().Nodes);
            Assert.Equal(referenceJson, CompressedFormatUtils.ToJson(renamed));
        }

        var peeked = SrkFileFormat.TryReadHeaderFromFile(srcPath);
        Assert.NotNull(peeked);
        Assert.Equal(SrkFileFormat.CurrentVersion, peeked!.SrkVersion);
        Assert.Equal(GraphKind.ConcreteArchitecture, peeked.TryGetStage());
        Assert.Equal("zstd", peeked.Compression);

        var barePath = P("peek_bare.zsrk");
        File.WriteAllBytes(barePath, CompressedFormatUtils.Compress(SrkFileFormat.Read(
            CompressedFormatUtils.SaveFastGraphToBinary(arch, compressed: false)).OnnxBytes));
        Assert.Null(SrkFileFormat.TryReadHeaderFromFile(barePath));
    }

    [Fact]
    public void TestSrkStageGateAtLoadTimeAndExtensionNormalizationKeepsUnrelatedFile()
    {
        var (moduleGraph, arch, model) = BuildStageGraphs();

        var moduleBytes = CompressedFormatUtils.SaveFastGraphToBinary(moduleGraph);
        var ex = Assert.Throws<InvalidOperationException>(() =>
            CompressedFormatUtils.LoadFastGraphFromBinary(moduleBytes, requiredStage: GraphKind.ConcreteModel));
        Assert.Contains("'module'", ex.Message);
        Assert.Contains("'concrete-model'", ex.Message);

        var modulePath = P("stage_module.zsrk");
        CompressedFormatUtils.SaveFastGraphToFile(modulePath, moduleGraph, compressed: true, overrideExtension: false);
        var exFile = Assert.Throws<InvalidOperationException>(() =>
            CompressedFormatUtils.LoadFastGraphFromFile(modulePath, requiredStage: GraphKind.ConcreteModel));
        Assert.Contains(modulePath, exFile.Message);

        Assert.NotEmpty(CompressedFormatUtils.LoadFastGraphFromBinary(
            moduleBytes, requiredStage: GraphKind.Module).ToInternal().Nodes);
        Assert.NotEmpty(CompressedFormatUtils.LoadFastGraphFromBinary(
            CompressedFormatUtils.SaveFastGraphToBinary(arch),
            requiredStage: GraphKind.ConcreteArchitecture).ToInternal().Nodes);
        Assert.NotEmpty(CompressedFormatUtils.LoadFastGraphFromBinary(
            CompressedFormatUtils.SaveFastGraphToBinary(model),
            requiredStage: GraphKind.ConcreteModel).ToInternal().Nodes);

        // Default overrideExtension:true normalizes .onnx → .zsrk; the caller's .onnx survives.
        var onnxPath = P("sentinel_model.onnx");
        var zsrkPath = Path.ChangeExtension(onnxPath, ".zsrk");
        byte[] sentinel = [1, 2, 3, 4];
        File.WriteAllBytes(onnxPath, sentinel);
        Assert.Equal(zsrkPath, CompressedFormatUtils.SaveFastGraphToFile(onnxPath, arch, compressed: true));
        Assert.True(File.Exists(onnxPath));
        Assert.Equal(sentinel, File.ReadAllBytes(onnxPath));
        Assert.True(File.Exists(zsrkPath));
        Assert.NotEmpty(CompressedFormatUtils.LoadFastGraphFromFile(zsrkPath).ToInternal().Nodes);
    }

    private static byte[] BuildRawSrkContainer(string headerJson, byte[] payload)
    {
        var headerBytes = System.Text.Encoding.UTF8.GetBytes(headerJson);
        var result = new byte[4 + 2 + headerBytes.Length + payload.Length];
        result[0] = (byte)'S'; result[1] = (byte)'R'; result[2] = (byte)'K'; result[3] = 1;
        result[4] = (byte)(headerBytes.Length & 0xFF);
        result[5] = (byte)(headerBytes.Length >> 8);
        headerBytes.CopyTo(result, 6);
        payload.CopyTo(result, 6 + headerBytes.Length);
        return result;
    }

    private static string Sha256Hex(byte[] payload)
        => Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(payload)).ToLowerInvariant();

    [Fact]
    public void TestSrkCorruptFutureAndNonContainerDataFailLoudly()
    {
        var (_, arch, _) = BuildStageGraphs();
        var bytes = CompressedFormatUtils.SaveFastGraphToBinary(arch, compressed: true);
        byte[] payload = [.. bytes.Skip(6 + (bytes[4] | (bytes[5] << 8)))];

        static void FromBinary(byte[] data, params string[] fragments)
        {
            var ex = Assert.Throws<InvalidDataException>(
                () => CompressedFormatUtils.LoadFastGraphFromBinary(data));
            foreach (var fragment in fragments)
                Assert.Contains(fragment, ex.Message);
        }

        void FromFile(string name, byte[] data, params string[] fragments)
        {
            var path = P(name);
            File.WriteAllBytes(path, data);
            var ex = Assert.Throws<InvalidDataException>(
                () => CompressedFormatUtils.LoadFastGraphFromFile(path));
            Assert.Contains(path, ex.Message);
            foreach (var fragment in fragments)
                Assert.Contains(fragment, ex.Message);
        }

        var corrupt = (byte[])bytes.Clone();
        corrupt[^1] ^= 0xFF;
        var garbage = new byte[64];
        Array.Fill(garbage, (byte)0x77);
        var bareZstd = CompressedFormatUtils.Compress(SrkFileFormat.Read(
            CompressedFormatUtils.SaveFastGraphToBinary(arch, compressed: false)).OnnxBytes);

        FromFile("corrupt.zsrk", corrupt, "SHA-256 mismatch");
        FromFile("garbage.srk", garbage);
        FromBinary(bytes[..^16], "corrupt or truncated");
        FromBinary(bytes[..8], "truncated");
        FromBinary([], "empty");
        FromBinary(bareZstd, "not a Shorokoo .srk container");
        FromBinary(BuildRawSrkContainer("{not json", payload), "header");
        FromBinary(BuildRawSrkContainer(
            $"{{\"srkVersion\":3,\"stage\":\"concrete-architecture\",\"compression\":\"zstd\",\"payloadSha256\":\"{Sha256Hex(payload)}\"}}",
            payload), "version 3");
        FromBinary(BuildRawSrkContainer(
            $"{{\"srkVersion\":1,\"stage\":\"concrete-architecture\",\"compression\":\"lz4\",\"payloadSha256\":\"{Sha256Hex(payload)}\"}}",
            payload), "lz4");
        FromBinary(BuildRawSrkContainer(
            $"{{\"stage\":\"concrete-architecture\",\"compression\":\"none\",\"payloadSha256\":\"{Sha256Hex(payload)}\"}}",
            payload), "'srkVersion'", "missing or zero");

        // A magic version byte this build does not read is refused up front, never as a null.
        var v3 = (byte[])bytes.Clone();
        v3[3] = 3;
        Assert.False(SrkFileFormat.IsSrkContainer(v3));
        FromBinary(v3, "major version 3");
        Assert.Contains("major version 3",
            Assert.Throws<InvalidDataException>(() => SrkFileFormat.TryReadHeader(v3)).Message);
        FromFile("future.srk", v3, "major version 3");
        var exPeek = Assert.Throws<InvalidDataException>(
            () => SrkFileFormat.TryReadHeaderFromFile(P("future.srk")));
        Assert.Contains(P("future.srk"), exPeek.Message);
        Assert.Contains("major version 3", exPeek.Message);
    }

    [Fact]
    public void TestSafeTensorTruncationAndMissingMetadataFailLoudly()
    {
        var tensors = new List<SafeTensor>
        {
            new("w", TensorData([4L], [1f, 2f, 3f, 4f]), "F32", [4L]),
            new("b", TensorData([2L], [5f, 6f]), "F32", [2L]),
        };
        using var stream = new MemoryStream();
        SafeTensorLoader.SaveSafeTensorsToStream(stream, tensors);
        var bytes = stream.ToArray();
        long headerLen = BitConverter.ToInt64(bytes, 0);

        Assert.Equal(2, SafeTensorLoader.ParseSafeTensorBytes(bytes).Count);

        static void Truncated(byte[] data, string code, params string[] fragments)
        {
            var ex = Assert.Throws<ModelException>(() => SafeTensorLoader.ParseSafeTensorBytes(data));
            Assert.Equal(code, ex.ErrorCode);
            foreach (var fragment in fragments)
                Assert.Contains(fragment, ex.Message);
        }

        Truncated(bytes[..^4], ErrorCodes.ST003, "truncated", "'b'",
            $"{bytes.Length} bytes", $"{bytes.Length - 4} bytes");
        Truncated(bytes[..10], ErrorCodes.ST002, "truncated",
            $"declares {headerLen} bytes", "only 2 byte(s)");
        Truncated(bytes[..5], ErrorCodes.ST001, "truncated");

        var truncPath = P("truncated.safetensors");
        File.WriteAllBytes(truncPath, bytes[..^4]);
        var exFile = Assert.Throws<ModelException>(() => SafeTensorLoader.LoadSafeTensors(truncPath));
        Assert.Equal(ErrorCodes.ST003, exFile.ErrorCode);
        Assert.Contains(truncPath, exFile.Message);
        Assert.Contains("truncated", exFile.Message);

        var zPath = P("truncated.zsafetensor");
        CompressedFormatUtils.CompressToFile(zPath, bytes[..^4]);
        var exZ = Assert.Throws<ModelException>(() => CompressedFormatUtils.LoadCompressedSafeTensors(zPath));
        Assert.Equal(ErrorCodes.ST003, exZ.ErrorCode);
        Assert.Contains(zPath, exZ.Message);
        Assert.Contains("truncated", exZ.Message);

        static byte[] Build(string headerJson, int payloadBytes)
        {
            var headerBytes = System.Text.Encoding.UTF8.GetBytes(headerJson);
            return [.. BitConverter.GetBytes((long)headerBytes.Length), .. headerBytes, .. new byte[payloadBytes]];
        }

        static void MissingField(string headerJson, params string[] fragments)
        {
            var ex = Assert.Throws<InvalidOperationException>(
                () => SafeTensorLoader.ParseSafeTensorBytes(Build(headerJson, 4)));
            foreach (var fragment in fragments)
                Assert.Contains(fragment, ex.Message);
        }

        MissingField("{\"w\":{\"dtype\":\"F32\",\"shape\":[1]}}", "data_offsets", "'w'");
        MissingField("{\"w\":{\"shape\":[1],\"data_offsets\":[0,4]}}", "dtype");
        MissingField("{\"w\":{\"dtype\":\"F32\",\"data_offsets\":[0,4]}}", "shape");

        // A rank-0 scalar's empty shape ("shape": []) is valid, not "missing".
        var scalar = SafeTensorLoader.ParseSafeTensorBytes(
            Build("{\"s\":{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[0,4]}}", 4));
        Assert.Empty(scalar.Single().Data.Shape.Dims);
    }

    private static byte[] BuildRawSafeTensors(string headerJson, byte[] payload)
    {
        var headerBytes = System.Text.Encoding.UTF8.GetBytes(headerJson);
        var result = new byte[8 + headerBytes.Length + payload.Length];
        BitConverter.GetBytes((long)headerBytes.Length).CopyTo(result, 0);
        headerBytes.CopyTo(result, 8);
        payload.CopyTo(result, 8 + headerBytes.Length);
        return result;
    }

    [Fact]
    public void TestInspectSrkAndSafeTensorsArtifacts()
    {
        var (_, arch, _) = BuildStageGraphs();

        bool[] compressionModes = [true, false];
        foreach (var compressed in compressionModes)
        {
            var path = P($"inspect_{compressed}.zsrk");
            CompressedFormatUtils.SaveFastGraphToFile(path, arch, compressed, overrideExtension: false);
            var result = Persistence.Inspect(path);

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
            Assert.True(result.Srk.PayloadSizeBytes > 0);

            var text = result.ToString();
            Assert.Contains("concrete-architecture", text);
            Assert.Contains(compressed ? "zstd" : "none", text);

            // Inspect never touches payload bytes: a corrupt payload still reads its header.
            var corrupt = File.ReadAllBytes(path);
            corrupt[^1] ^= 0xFF;
            var corruptPath = P($"inspect_corrupt_{compressed}.zsrk");
            File.WriteAllBytes(corruptPath, corrupt);
            Assert.Throws<InvalidDataException>(
                () => CompressedFormatUtils.LoadFastGraphFromFile(corruptPath));
            var corruptResult = Persistence.Inspect(corruptPath);
            Assert.Equal(ArtifactKind.SrkGraph, corruptResult.Kind);
            Assert.Equal(header.PayloadSha256, corruptResult.Srk!.Header!.PayloadSha256);
        }

        var srkBytes = CompressedFormatUtils.SaveFastGraphToBinary(arch, compressed: false);
        var future = (byte[])srkBytes.Clone();
        future[3] = 3;
        (string Name, byte[] Bytes)[] damaged = [("future", future), ("truncated", srkBytes[..5])];
        foreach (var (name, bytes) in damaged)
        {
            var path = P($"inspect_{name}.srk");
            File.WriteAllBytes(path, bytes);
            var result = Persistence.Inspect(path);
            AssertInspection(result, ArtifactKind.SrkGraph, ["header is not readable"]);
            Assert.Null(result.Srk!.Header);
            Assert.Null(result.Srk.PayloadSizeBytes);
        }

        var garbagePath = P("inspect_garbage.bin");
        var garbage = new byte[64];
        Array.Fill(garbage, (byte)0x77);
        File.WriteAllBytes(garbagePath, garbage);
        var garbageResult = Persistence.Inspect(garbagePath);
        Assert.Equal(ArtifactKind.NotRecognized, garbageResult.Kind);
        Assert.NotEmpty(garbageResult.Observations);
        Assert.Contains("not recognized", garbageResult.ToString());

        var emptyPath = P("inspect_empty.bin");
        File.WriteAllBytes(emptyPath, []);
        AssertInspection(Persistence.Inspect(emptyPath), ArtifactKind.NotRecognized, ["empty"]);

        Assert.Throws<FileNotFoundException>(() => Persistence.Inspect(P("inspect_nope.srk")));

        var t1 = TensorData([2, 3], 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);
        var t2 = TensorData([3], 7.0f, 8.0f, 9.0f);
        var scalar = TensorData([], 42.0f);
        var stPath = P("inspect_weights.safetensors");
        SafeTensorLoader.SaveSafeTensors(stPath, new List<SafeTensor>
        {
            new SafeTensor("tensor1", t1, "F32", t1.Shape.Dims),
            new SafeTensor("tensor2", t2, "F32", t2.Shape.Dims),
            new SafeTensor("scalar", scalar, "F32", scalar.Shape.Dims),
        });
        var stResult = Persistence.Inspect(stPath);

        Assert.Equal(ArtifactKind.SafeTensors, stResult.Kind);
        Assert.Null(stResult.Srk);
        Assert.Null(stResult.TrainingCheckpoint);
        Assert.Empty(stResult.Observations);

        var st = stResult.SafeTensors!;
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
        Assert.Empty(byName["scalar"].Shape);
        Assert.Equal(4, byName["scalar"].ByteSize);

        var stText = stResult.ToString();
        Assert.Contains("SafeTensors", stText);
        Assert.Contains("tensor1: F32[2, 3], 24 bytes", stText);

        var stBytes = File.ReadAllBytes(stPath);
        var truncatedPath = P("inspect_weights_truncated.safetensors");
        File.WriteAllBytes(truncatedPath, stBytes[..^8]);
        AssertInspection(Persistence.Inspect(truncatedPath), ArtifactKind.SafeTensors, ["past the end"]);

        var trailingPath = P("inspect_weights_trailing.safetensors");
        File.WriteAllBytes(trailingPath, [.. stBytes, 0, 0, 0, 0]);
        AssertInspection(Persistence.Inspect(trailingPath), ArtifactKind.SafeTensors, ["trailing"]);
    }

    [Fact]
    public void TestInspectCompressedSafeTensorsAndHostileInputs()
    {
        var t1 = TensorData([2, 3], 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);
        var t2 = TensorData([3], 7.0f, 8.0f, 9.0f);
        var zPath = P("inspect_weights.zsafetensor");
        CompressedFormatUtils.SaveCompressedSafeTensors(zPath, new List<SafeTensor>
        {
            new SafeTensor("tensor1", t1, "F32", t1.Shape.Dims),
            new SafeTensor("tensor2", t2, "F32", t2.Shape.Dims),
        }, new Dictionary<string, object> { ["format"] = "shorokoo-test" });

        var result = Persistence.Inspect(zPath);
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

        // Payload untouched: this ~1.2 MB incompressible payload spans several Zstd blocks, so a
        // chopped tail breaks a full load while the header still inspects.
        var bigValues = new float[300_000];
        uint seed = 1;
        for (int i = 0; i < bigValues.Length; i++)
        {
            seed = seed * 747796405u + 2891336453u;
            bigValues[i] = BitConverter.UInt32BitsToSingle((seed >> 9) | 0x3F800000u);
        }
        var big = TensorData([300_000L], bigValues);
        var bigPath = P("inspect_big.zsafetensor");
        CompressedFormatUtils.SaveCompressedSafeTensors(bigPath, new List<SafeTensor>
        {
            new SafeTensor("big", big, "F32", big.Shape.Dims),
        });
        var bigBytes = File.ReadAllBytes(bigPath);
        Assert.True(bigBytes.Length > 256 * 1024);
        var choppedPath = P("inspect_big_chopped.zsafetensor");
        File.WriteAllBytes(choppedPath, bigBytes[..^64]);
        Assert.ThrowsAny<Exception>(() => CompressedFormatUtils.LoadCompressedSafeTensors(choppedPath));
        var chopped = Persistence.Inspect(choppedPath);
        Assert.Equal(ArtifactKind.CompressedSafeTensors, chopped.Kind);
        Assert.Equal("big", Assert.Single(chopped.SafeTensors!.Tensors).Name);
        Assert.Equal(300_000L * 4, chopped.SafeTensors.TotalTensorBytes);

        // Compressed checkpoint: the marker shows in the header, its [version, step] payload
        // sits beyond the bounded header read.
        var w = TensorData([2L], 1.0f, 2.0f);
        var ckptPath = P("inspect_ckpt.zsafetensor");
        CompressedFormatUtils.SaveCompressedSafeTensors(ckptPath, new List<SafeTensor>
        {
            new SafeTensor("trainable/w", w, "F32", w.Shape.Dims),
            new SafeTensor("__shorokoo_checkpoint__", TensorData([2L], 1L, 7L), "I64", [2L]),
        });
        var ckpt = Persistence.Inspect(ckptPath);
        AssertInspection(ckpt, ArtifactKind.CompressedSafeTensors, ["__shorokoo_checkpoint__", "bounded"]);
        Assert.Null(ckpt.TrainingCheckpoint);
        Assert.Contains(ckpt.SafeTensors!.Tensors, t => t.Name == "__shorokoo_checkpoint__");

        var stubPath = P("inspect_stub.zsafetensor");
        File.WriteAllBytes(stubPath, bigBytes[..5]);
        AssertInspection(Persistence.Inspect(stubPath), ArtifactKind.NotRecognized, ["Zstd frame"]);

        byte[] shortDecl = [.. BitConverter.GetBytes(1000L), 0x7B, 0x22, 0x74];
        var shortPath = P("inspect_short.zsafetensor");
        File.WriteAllBytes(shortPath, CompressedFormatUtils.Compress(shortDecl));
        AssertInspection(Persistence.Inspect(shortPath), ArtifactKind.NotRecognized, ["ends after"]);

        // Amplification guard: a declared near-cap (99 MB) header must not cost 99 MB of
        // allocation — the buffer grows with what the stream delivers (200 KB here).
        var hugeDecl = new byte[8 + 200_000];
        BitConverter.GetBytes(99_000_000L).CopyTo(hugeDecl, 0);
        var hugePath = P("inspect_huge_decl.zsafetensor");
        File.WriteAllBytes(hugePath, CompressedFormatUtils.Compress(hugeDecl));
        AssertInspection(Persistence.Inspect(hugePath), ArtifactKind.NotRecognized, ["ends after 200000"]);

        var (_, arch, _) = BuildStageGraphs();
        var framedPath = P("inspect_zstd_onnx.srk");
        File.WriteAllBytes(framedPath, CompressedFormatUtils.Compress(SrkFileFormat.Read(
            CompressedFormatUtils.SaveFastGraphToBinary(arch, compressed: false)).OnnxBytes));
        AssertInspection(Persistence.Inspect(framedPath), ArtifactKind.NotRecognized,
            ["not a SafeTensors archive"]);

        // Marker offset near long.MaxValue: markerStart + 32 wraps past the bounds guard.
        // Iterate — the offset's digits feed back into the header length.
        static string MarkerJson(long start) =>
            $"{{\"__shorokoo_checkpoint__\":{{\"dtype\":\"I64\",\"shape\":[4],\"data_offsets\":[{start},{start + 32}]}}}}";
        var markerHeader = MarkerJson(long.MaxValue / 2);
        for (int i = 0; i < 4; i++)
        {
            long dataStart = 8 + System.Text.Encoding.UTF8.GetByteCount(markerHeader);
            markerHeader = MarkerJson(long.MaxValue - dataStart - 8);
        }
        var overflowMarkerPath = P("hostile_marker_offset.safetensors");
        File.WriteAllBytes(overflowMarkerPath, BuildRawSafeTensors(markerHeader, new byte[32]));
        var overflowMarker = Persistence.Inspect(overflowMarkerPath);
        AssertInspection(overflowMarker, ArtifactKind.SafeTensors, ["malformed"]);
        Assert.Null(overflowMarker.TrainingCheckpoint);

        // Huge declared end offset: dataStart + maxEnd wraps — truncation, not negative trailing.
        var hugeEndPath = P("hostile_huge_end.safetensors");
        File.WriteAllBytes(hugeEndPath, BuildRawSafeTensors(
            "{\"t\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,9223372036854775800]}}",
            new byte[8]));
        var hugeEnd = Persistence.Inspect(hugeEndPath);
        AssertInspection(hugeEnd, ArtifactKind.SafeTensors, ["past the end"]);
        Assert.DoesNotContain(hugeEnd.Observations, o => o.Contains("trailing"));

        var badExtentPath = P("hostile_bad_extent.safetensors");
        File.WriteAllBytes(badExtentPath, BuildRawSafeTensors(
            "{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[10,2]}," +
            "\"b\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[-9223372036854775808,8]}}",
            new byte[16]));
        var badExtent = Persistence.Inspect(badExtentPath);
        Assert.Equal(ArtifactKind.SafeTensors, badExtent.Kind);
        Assert.Equal(2, badExtent.Observations.Count(o => o.Contains("invalid extent")));
        Assert.All(badExtent.SafeTensors!.Tensors, t => Assert.Equal(0, t.ByteSize));

        var wrongDtypePath = P("hostile_marker_dtype.safetensors");
        File.WriteAllBytes(wrongDtypePath, BuildRawSafeTensors(
            "{\"__shorokoo_checkpoint__\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,32]}}",
            new byte[32]));
        var wrongDtype = Persistence.Inspect(wrongDtypePath);
        AssertInspection(wrongDtype, ArtifactKind.SafeTensors, ["malformed"]);
        Assert.Null(wrongDtype.TrainingCheckpoint);

        // An unreadable format version still inspects as a checkpoint; a stray tensor is observed.
        var futureCkptPath = P("future_checkpoint.safetensors");
        SafeTensorLoader.SaveSafeTensors(futureCkptPath, new List<SafeTensor>
        {
            new SafeTensor("trainable/w", w, "F32", w.Shape.Dims),
            new SafeTensor("stray", w, "F32", w.Shape.Dims),
            new SafeTensor("__shorokoo_checkpoint__", TensorData([2L], 99L, 3L), "I64", [2L]),
        });
        var futureCkpt = Persistence.Inspect(futureCkptPath);
        AssertInspection(futureCkpt, ArtifactKind.TrainingCheckpoint, ["format version 99"], ["'stray'"]);
        Assert.Equal(99, futureCkpt.TrainingCheckpoint!.FormatVersion);
        Assert.Equal(3, futureCkpt.TrainingCheckpoint.Step);
        Assert.Single(futureCkpt.TrainingCheckpoint.Sections["trainable"]);

        // Zstd-compressed non-SafeTensors data, including a near-miss prefix starting 0x08.
        byte[] textBytes = System.Text.Encoding.UTF8.GetBytes("clearly not a model, just some text.");
        byte[] nearMiss = [0x08, 0x01, 0, 0, 0, 0, 0, 0, 0x7B, 0x22];
        (string Name, byte[] Inner)[] zstdCases =
            [("zstd_text.bin", textBytes), ("zstd_nearmiss.zsafetensor", nearMiss)];
        foreach (var (name, inner) in zstdCases)
        {
            var p = P(name);
            File.WriteAllBytes(p, CompressedFormatUtils.Compress(inner));
            AssertInspection(Persistence.Inspect(p), ArtifactKind.NotRecognized, ["Zstd frame"]);
        }

        var metaPath = P("with_metadata.safetensors");
        SafeTensorLoader.SaveSafeTensors(metaPath,
            new List<SafeTensor> { new SafeTensor("w", w, "F32", w.Shape.Dims) },
            new Dictionary<string, object> { ["format"] = "shorokoo-test" });
        var meta = Persistence.Inspect(metaPath);
        Assert.Equal(ArtifactKind.SafeTensors, meta.Kind);
        Assert.Equal("shorokoo-test", meta.SafeTensors!.GlobalMetadata!["format"]);
    }

    private static Tensor<float32> ParamlessDouble(Tensor<float32> x) => x + x;

    [Fact]
    public void TestStampedGraphKindSurvivesSrkAndOnnxWithoutOpScanEvidence()
    {
        var moduleGraph = ModuleFactory.ComputationGraph(
            (Func<Tensor<float32>, Tensor<float32>>)ParamlessDouble);
        Assert.Equal(GraphKind.Module, moduleGraph.Kind);

        var arch = moduleGraph.ToConcreteArchitecture(
            moduleGraph.FromOrderedInputs([TensorData([2L], 1.0f, 2.0f)]));
        Assert.Equal(GraphKind.ConcreteArchitecture, arch.Kind);
        // No trainable params → op-scanning misclassifies this architecture.
        Assert.Equal(GraphKind.ConcreteModel, SrkFileFormat.DetectStage(arch.ToInternal()));

        var bytes = CompressedFormatUtils.SaveFastGraphToBinary(arch, compressed: false);
        Assert.Equal(GraphKind.ConcreteArchitecture, SrkFileFormat.TryReadHeader(bytes)!.TryGetStage());
        var reloaded = CompressedFormatUtils.LoadFastGraphFromBinary(bytes);
        Assert.Equal(GraphKind.ConcreteArchitecture, reloaded.Kind);
        Assert.Equal(GraphKind.ConcreteModel, reloaded.ToConcreteModel().Kind);

        (ComputationGraph Graph, GraphKind Kind)[] cases =
            [(moduleGraph, GraphKind.Module), (arch, GraphKind.ConcreteArchitecture)];
        foreach (var (graph, kind) in cases)
        {
            var proto = FastOnnxModelBuilder.BuildInternalOnnxModel(graph.ToInternal(), stage: graph.Kind);
            using var ms = new MemoryStream();
            ProtoBuf.Serializer.Serialize(ms, proto);
            var viaOnnx = OnnxModelImporter.FromOnnxModel(ms.ToArray());
            Assert.Equal(kind, viaOnnx.Kind);
            Assert.Equal(GraphKind.ConcreteModel, SrkFileFormat.DetectStage(viaOnnx.ToInternal()));
        }

        // Module machinery tagged concrete-model is structurally impossible: refused at import.
        var lyingProto = FastOnnxModelBuilder.BuildInternalOnnxModel(
            ScalarMultiplyModel.ComputationGraph.ToInternal(), stage: GraphKind.ConcreteModel);
        using var lyingMs = new MemoryStream();
        ProtoBuf.Serializer.Serialize(lyingMs, lyingProto);
        var ex = Assert.Throws<InvalidDataException>(
            () => OnnxModelImporter.FromOnnxModel(lyingMs.ToArray()));
        Assert.Contains("shrk_graph_kind", ex.Message);
        Assert.Contains("module-stage op", ex.Message);
    }

    /// <summary>Order-independent structure descriptor: per-node opcode + per-input-slot
    /// present/absent pattern, plus the graph I/O signature names. Node/tensor keys are
    /// freshly assigned on load, so they are deliberately excluded.</summary>
    private static string DescribeModuleGraphStructure(ComputationGraph graph)
    {
        var g = graph.ToInternal();
        var nodeLines = g.Nodes
            .Select(n => $"{n.OpCode}({string.Join(",", n.Inputs.Select(k => k is null ? "-" : "x"))})")
            .OrderBy(x => x, StringComparer.Ordinal);
        return $"inputs=[{string.Join(",", g.InputUniqueNames)}] outputs=[{string.Join(",", g.OutputUniqueNames)}]\n"
             + string.Join("\n", nodeLines);
    }

    /// <summary>Saves to .srk, checks the header stamps the module stage, reloads, checks the
    /// structure descriptor is unchanged and that save → load → save is a byte-level fixed
    /// point. Returns the reloaded graph.</summary>
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
        Assert.True(bytes2.SequenceEqual(bytes3));
        return reloaded;
    }

    /// <summary>Op-code + dtype of each top-level graph input, in graph-input order.</summary>
    private static List<(string OpCode, DType? DType)> DescribeTopLevelInputs(InternalComputationGraph g)
        => g.Inputs
            .Select(key =>
            {
                var node = g.Nodes.First(n => n.Outputs.Any(o => o.HasValue && o.Value.Equals(key)));
                return (node.OpCode, node.Attributes.GetDTypeVal(OnnxOpAttributeNames.AttrDtype));
            })
            .ToList();

    [Fact]
    public void TestModuleStageSrkRoundTripStructureInputKindsAndStateOwnership()
    {
        ComputationGraph[] moduleGraphs =
        [
            ScalarMultiplyModel.ComputationGraph,               // absent optional model slot on MODEL_PARAM_MODEL_REF
            ScalarMultiplyWithBatchNormModel.ComputationGraph,  // module-owned state initializers
            StepCountingSgdOptimizer.ComputationGraph,          // optimizer-owned state initializer + defaulted hyper
            NullableBiasLayer.ComputationGraph,                 // MODEL_OPTIONAL_INPUT
            DefaultedHyperLayer.ComputationGraph,               // [Hyper(3f)] default on MODEL_TENSOR_INPUT
            SimplePairSum.ComputationGraph,                     // MODEL_TENSORSTRUCT_INPUT + TENSOR_STRUCT_GETFIELD
            TensorStructLoopCarry.ComputationGraph,             // TENSOR_STRUCT_CREATE/GETFIELD inside a LOOP band
            TrainablesInBothLoopLevels.ComputationGraph,        // MODEL_PARAM_MODEL_REF in nested LOOP bodies
            RngRuntimeLoopFeed.ComputationGraph,                // SHRK_RANDOM_UNIFORM feed (absent key) in a LOOP body
            RngInitTwoLinears.ComputationGraph,                 // sub-module invokes + random param initializers
            GenericRecordSumCaller.ComputationGraph,            // GENERIC_TYPE_INPUT + struct-typed sub-module call
        ];
        foreach (var moduleGraph in moduleGraphs)
            AssertModuleStageSrkRoundTrip(moduleGraph);

        (ComputationGraph Graph, string ExpectedKind)[] inputKinds =
        [
            (NullableBiasLayer.ComputationGraph, InternalOpCodes.MODEL_OPTIONAL_INPUT),
            (SeqHypersLayer.ComputationGraph, InternalOpCodes.MODEL_SEQUENCE_INPUT),
            (SimplePairSum.ComputationGraph, InternalOpCodes.MODEL_TENSORSTRUCT_INPUT),
            (GenericRecordSumCaller.ComputationGraph, InternalOpCodes.GENERIC_TYPE_INPUT),
        ];
        foreach (var (graph, expectedKind) in inputKinds)
        {
            var before = DescribeTopLevelInputs(graph.ToInternal());
            Assert.Contains(expectedKind, before.Select(d => d.OpCode));
            var after = DescribeTopLevelInputs(AssertModuleStageSrkRoundTrip(graph).ToInternal());
            Assert.Equal(before, after);
            Assert.Contains(expectedKind, after.Select(d => d.OpCode));
        }

        (ComputationGraph Module, StateOwnership Expected)[] ownership =
        [
            (StepCountingSgdOptimizer.ComputationGraph, StateOwnership.OptimizerOwned),
            (ScalarMultiplyWithBatchNormModel.ComputationGraph, StateOwnership.ModuleOwned),
        ];
        foreach (var (moduleGraph, expected) in ownership)
        {
            var stateInits = AssertModuleStageSrkRoundTrip(moduleGraph).ToInternal().Nodes
                .Select(n => n.TargetFunction)
                .Where(fn => fn is { FunctionType: FunctionType.StateParamInitializer })
                .ToArray();
            Assert.NotEmpty(stateInits);
            Assert.All(stateInits, fn => Assert.Equal(expected, fn!.StateOwnership));
        }
    }

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

    // ──────────────────────────────────────────────────────────────────────
    // .skpt single-file checkpoint container: STORED zip + config.json manifest.
    // ──────────────────────────────────────────────────────────────────────

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

    /// <summary>The model's weight tensors (raw bytes) keyed by parameter identifier, excluding
    /// the RNG identity parameter — the set a .skpt stores in its data tree.</summary>
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

    /// <summary>Rebuilds a .skpt archive from raw entries, re-aligning data-tree entries the
    /// way the real writer does.</summary>
    private static void RewriteSkpt(string path, IReadOnlyList<(string Name, byte[] Data)> entries)
    {
        using var stream = File.Create(path);
        SkptFileFormat.WriteStoredZip(
            stream,
            entries.Select(e => new SkptFileFormat.ZipEntrySpec(
                e.Name, e.Data, e.Name.StartsWith("data/", StringComparison.Ordinal))).ToList(),
            DateTime.UtcNow);
    }

    /// <summary>Walks the raw local file headers of a zip (no library involved), returning each
    /// entry's name, compression method, absolute payload offset and stored size.</summary>
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

    [Fact]
    public void TestSkptRoundTripConcreteModelBuilderGatesAndAtomicSave()
    {
        var (model, numOut, input) = BuildSkptModel();
        var path = P("roundtrip.skpt");
        Persistence.From(model).WithModel().WithWeights().Save(path);

        var originalWeights = WeightBytesByParam(model);
        Assert.Equal(2, originalWeights.Count);
        // Non-zero default weights: the byte-identity checks below cannot pass vacuously.
        Assert.Contains(originalWeights.Values, bytes => bytes.Any(b => b != 0));

        var entries = ReadZipEntries(path);
        string[] expectedEntries =
            [SkptFileFormat.ConfigEntryName, SkptFileFormat.WeightsEntryPath, SkptFileFormat.ModelEntryPath];
        Assert.Equal(expectedEntries.OrderBy(n => n, StringComparer.Ordinal),
            entries.Keys.OrderBy(n => n, StringComparer.Ordinal));

        var fileBytes = File.ReadAllBytes(path);
        var localHeaders = ParseLocalZipHeaders(fileBytes);
        Assert.Equal(entries.Count, localHeaders.Count);
        Assert.All(localHeaders, h => Assert.Equal(0, h.Method));
        var weightsHeader = localHeaders.Single(h => h.Name == SkptFileFormat.WeightsEntryPath);
        Assert.Equal(0L, weightsHeader.DataOffset % SkptFileFormat.DataAlignment);
        Assert.Equal(entries[SkptFileFormat.WeightsEntryPath],
            fileBytes.AsSpan((int)weightsHeader.DataOffset, (int)weightsHeader.Size).ToArray());

        var manifest = SkptFileFormat.ParseManifest(entries[SkptFileFormat.ConfigEntryName], path);
        Assert.Equal(SkptFileFormat.FormatName, manifest.Format);
        Assert.Equal(SkptFileFormat.CurrentVersion, manifest.SkptVersion);
        Assert.False(string.IsNullOrEmpty(manifest.CreatedUtc));
        Assert.Equal(Shorokoo.ShorokooVersion.VersionString, manifest.Producer?.Shorokoo);
        var modelEntry = Assert.Single(manifest.Models!).Value;
        Assert.Equal(SkptFileFormat.ModelEntryPath, modelEntry.Entry);
        Assert.Equal(SkptFileFormat.ModelFormatSrk1, modelEntry.Format);
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

        var storedTensors = SafeTensorLoader.ParseSafeTensorBytes(entries[SkptFileFormat.WeightsEntryPath])
            .ToDictionary(t => t.Name, t => t.Data.AccessRawMemory().ToArray(), StringComparer.Ordinal);
        Assert.Equal(originalWeights.Count, storedTensors.Count);
        foreach (var (paramId, bytes) in originalWeights)
            Assert.Equal(bytes, storedTensors[mapping[paramId].Tensor!]);

        // The model entry is definition-only: dtype/shape-true placeholders with elided storage.
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
        var strippedAllParams = strippedDefinition.ToInternal().Nodes
            .Where(n => n.OpCode == InternalOpCodes.MODEL_PARAM_DATA)
            .ToList();
        Assert.Equal(originalParams.Count, strippedAllParams.Count);
        Assert.All(
            strippedAllParams.Where(n => !originalWeights.ContainsKey(n.IdentifierTemplate!)),
            n => Assert.IsNotType<WeightPlaceholderTensorData>(n.GetTensorData()));

        var loaded = Persistence.Load(path);
        Assert.Equal(GraphKind.ConcreteModel, loaded.Kind);
        var loadedWeights = WeightBytesByParam(loaded);
        Assert.Equal(originalWeights.Count, loadedWeights.Count);
        foreach (var (paramId, bytes) in originalWeights)
            Assert.Equal(bytes, loadedWeights[paramId]);
        var direct = ExecuteToBytes(model, numOut, input);
        Assert.Equal(direct, ExecuteToBytes(loaded, numOut, input));

        // Binding is authoritative: a definition holding full-size zeros binds identically —
        // the elided marker is a size optimization, not part of the load contract.
        var zerosPath = P("materialized-zero-placeholders.skpt");
        var zerosGraph = model.ToInternal().Clone();
        foreach (var node in zerosGraph.Nodes)
        {
            if (node.OpCode != InternalOpCodes.MODEL_PARAM_DATA
                || !originalWeights.ContainsKey(node.IdentifierTemplate ?? "")) continue;
            var data = node.GetTensorData()!;
            node.Attributes = node.Attributes.SetAttributes(
                (OnnxOpAttributeNames.ShrkAttrTensorData,
                 (object?)TensorDataWithDefaultVals(data.DType, data.Shape.Dims)));
        }
        var zerosModelBytes = CompressedFormatUtils.SaveFastGraphToBinary(
            zerosGraph, GraphKind.ConcreteModel, compressed: true);
        var zerosConfig = JsonNode.Parse(entries[SkptFileFormat.ConfigEntryName])!;
        zerosConfig["models"]!["model"]!["sha256"] = SkptFileFormat.Sha256Hex(zerosModelBytes);
        RewriteSkpt(zerosPath, entries.Select(e => (e.Key, e.Key switch
        {
            SkptFileFormat.ConfigEntryName =>
                System.Text.Encoding.UTF8.GetBytes(zerosConfig.ToJsonString()),
            SkptFileFormat.ModelEntryPath => zerosModelBytes,
            _ => e.Value,
        })).ToList());
        var zerosLoaded = Persistence.Load(zerosPath);
        var zerosWeights = WeightBytesByParam(zerosLoaded);
        Assert.Equal(originalWeights.Count, zerosWeights.Count);
        foreach (var (paramId, bytes) in originalWeights)
            Assert.Equal(bytes, zerosWeights[paramId]);
        Assert.Equal(direct, ExecuteToBytes(zerosLoaded, numOut, input));

        // Builder gates: only a concrete model starts a checkpoint, and only model + weights saves.
        var exKind = Assert.Throws<InvalidOperationException>(() => Persistence.From(FCLayer.ComputationGraph));
        Assert.Contains("concrete-model", exKind.Message);
        var incompletePath = P("incomplete.skpt");
        var exNone = Assert.Throws<InvalidOperationException>(() => Persistence.From(model).Save(incompletePath));
        Assert.Contains("WithModel", exNone.Message);
        Assert.Throws<InvalidOperationException>(() => Persistence.From(model).WithModel().Save(incompletePath));
        Assert.Throws<InvalidOperationException>(() => Persistence.From(model).WithWeights().Save(incompletePath));
        Assert.False(File.Exists(incompletePath));

        // The atomic writer stages in the target's directory, so it must exist up front.
        Assert.Throws<DirectoryNotFoundException>(() => Persistence.From(model).WithModel().WithWeights()
            .Save(P(Path.Combine("no-such-dir", "model.skpt"))));

        // A simulated crash between staging and commit leaves the existing checkpoint intact.
        var atomicPath = P("atomic.skpt");
        Persistence.From(model).WithModel().WithWeights().Save(atomicPath);
        var committed = File.ReadAllBytes(atomicPath);
        AtomicFileWriter.CommitFaultInjection = tempPath =>
        {
            if (tempPath.Contains("atomic.skpt")) throw new IOException("simulated commit crash");
        };
        try
        {
            Assert.Throws<IOException>(
                () => Persistence.From(model).WithModel().WithWeights().Save(atomicPath));
        }
        finally { AtomicFileWriter.CommitFaultInjection = null; }
        Assert.Equal(committed, File.ReadAllBytes(atomicPath));
        Assert.Equal(direct, ExecuteToBytes(Persistence.Load(atomicPath), numOut, input));
    }

    /// <summary>An FCLayer model (32 output features, [32,32] input) whose weights carry a
    /// deterministic repeating non-zero pattern — compressible enough that Zstd reliably
    /// shrinks the data entry, and distinct from the zero placeholders the model entry
    /// stores.</summary>
    private static (ComputationGraph Model, TensorData NumOut, TensorData Input) BuildCompressibleSkptModel()
    {
        var numOut = TensorData(DType.Int64, [], 32L);
        var input = TensorDataWithSmallVals(DType.Float32, [32L, 32L]);
        var g = FCLayer.ComputationGraph;
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

    [Fact]
    public void TestSkptLoadValidationZstdDataAndCompressionFaults()
    {
        var (model, numOut, input) = BuildSkptModel();
        var path = P("validation.skpt");
        var tamperedPath = P("tampered.skpt");
        Persistence.From(model).WithModel().WithWeights().Save(path);
        var entries = ReadZipEntries(path);
        var direct = ExecuteToBytes(model, numOut, input);

        List<(string Name, byte[] Data)> Without(string name) =>
            entries.Where(e => e.Key != name).Select(e => (e.Key, e.Value)).ToList();
        List<(string Name, byte[] Data)> WithConfig(string configJson) =>
            entries.Select(e => (e.Key, e.Key == SkptFileFormat.ConfigEntryName
                ? System.Text.Encoding.UTF8.GetBytes(configJson) : e.Value)).ToList();
        void Refused(List<(string Name, byte[] Data)> tampered, params string[] fragments)
        {
            RewriteSkpt(tamperedPath, tampered);
            var ex = Assert.Throws<InvalidDataException>(() => Persistence.Load(tamperedPath));
            foreach (var fragment in fragments)
                Assert.Contains(fragment, ex.Message);
        }

        File.WriteAllBytes(tamperedPath, [1, 2, 3, 4]);
        Assert.Contains("zip",
            Assert.Throws<InvalidDataException>(() => Persistence.Load(tamperedPath)).Message);

        Refused(Without(SkptFileFormat.ConfigEntryName), SkptFileFormat.ConfigEntryName);
        Refused(Without(SkptFileFormat.WeightsEntryPath), SkptFileFormat.WeightsEntryPath);
        Refused(entries.Select(e =>
        {
            if (e.Key != SkptFileFormat.WeightsEntryPath) return (e.Key, e.Value);
            var copy = e.Value.ToArray();
            copy[^1] ^= 0xFF;
            return (e.Key, copy);
        }).ToList(), "SHA-256", SkptFileFormat.WeightsEntryPath);

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
        Assert.Equal(direct, ExecuteToBytes(Persistence.Load(tamperedPath), numOut, input));

        var otherVersionConfig = JsonNode.Parse(entries[SkptFileFormat.ConfigEntryName])!;
        otherVersionConfig["skptVersion"] = SkptFileFormat.CurrentVersion + 1;
        Refused(WithConfig(otherVersionConfig.ToJsonString()),
            $"reads version {SkptFileFormat.CurrentVersion} only");

        var missingParamConfig = JsonNode.Parse(entries[SkptFileFormat.ConfigEntryName])!;
        ((JsonObject)missingParamConfig["tensorMappings"]!["model"]!["default"]!["tensors"]!)
            .Remove(firstParam);
        Refused(WithConfig(missingParamConfig.ToJsonString()), firstParam);

        var strayConfig = JsonNode.Parse(entries[SkptFileFormat.ConfigEntryName])!;
        strayConfig["tensorMappings"]!["model"]!["default"]!["tensors"]!["not_a_real_param"] =
            new JsonObject { ["data"] = "weights", ["tensor"] = "not_a_real_param" };
        Refused(WithConfig(strayConfig.ToJsonString()), "not_a_real_param");

        // ── Opt-in per-entry Zstd compression of the data tree.
        var (zModel, zNumOut, zInput) = BuildCompressibleSkptModel();
        var plainPath = P("zstd-plain.skpt");
        var zstdPath = P("zstd-on.skpt");
        Persistence.From(zModel).WithModel().WithWeights().Save(plainPath);
        Persistence.From(zModel).WithModel().WithWeights().WithZstdCompressedData().Save(zstdPath);

        var plainEntries = ReadZipEntries(plainPath);
        var zstdEntries = ReadZipEntries(zstdPath);
        Assert.Equal(plainEntries.Keys.OrderBy(n => n, StringComparer.Ordinal),
            zstdEntries.Keys.OrderBy(n => n, StringComparer.Ordinal));
        var zstdFileBytes = File.ReadAllBytes(zstdPath);
        Assert.All(ParseLocalZipHeaders(zstdFileBytes), h => Assert.Equal(0, h.Method));

        // Compression touches only the weights data entry; the default output is unchanged.
        Assert.Equal(plainEntries[SkptFileFormat.ModelEntryPath],
            zstdEntries[SkptFileFormat.ModelEntryPath]);
        var storedWeights = zstdEntries[SkptFileFormat.WeightsEntryPath];
        Assert.True(SkptFileFormat.LooksLikeZstdFrame(storedWeights));
        Assert.Equal(plainEntries[SkptFileFormat.WeightsEntryPath],
            CompressedFormatUtils.Decompress(storedWeights));
        Assert.True(storedWeights.Length < plainEntries[SkptFileFormat.WeightsEntryPath].Length);
        Assert.True(zstdFileBytes.Length < new FileInfo(plainPath).Length);

        var plainData = Assert.Single(
            SkptFileFormat.ParseManifest(plainEntries[SkptFileFormat.ConfigEntryName], plainPath).Data!).Value;
        Assert.Equal(SkptFileFormat.CompressionNone, plainData.Compression);
        var zstdData = Assert.Single(
            SkptFileFormat.ParseManifest(zstdEntries[SkptFileFormat.ConfigEntryName], zstdPath).Data!).Value;
        Assert.Equal(SkptFileFormat.CompressionZstd, zstdData.Compression);
        Assert.Equal(SkptFileFormat.Sha256Hex(storedWeights), zstdData.Sha256);

        var zOriginalWeights = WeightBytesByParam(zModel);
        Assert.All(zOriginalWeights.Values, bytes => Assert.Contains(bytes, b => b != 0));
        var zLoadedWeights = WeightBytesByParam(Persistence.Load(zstdPath));
        Assert.Equal(zOriginalWeights.Count, zLoadedWeights.Count);
        foreach (var (paramId, bytes) in zOriginalWeights)
            Assert.Equal(bytes, zLoadedWeights[paramId]);
        Assert.Equal(ExecuteToBytes(zModel, zNumOut, zInput),
            ExecuteToBytes(Persistence.Load(zstdPath), zNumOut, zInput));

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            Persistence.From(zModel).WithZstdCompressedData(compressionLevel: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            Persistence.From(zModel).WithZstdCompressedData(compressionLevel: 23));

        // ── Manifest/stored compression mismatches, in both directions.
        void RewriteWith(Dictionary<string, byte[]> source, string configJson, byte[]? weights = null)
            => RewriteSkpt(tamperedPath, source.Select(e => (e.Key,
                e.Key == SkptFileFormat.ConfigEntryName ? System.Text.Encoding.UTF8.GetBytes(configJson)
                : e.Key == SkptFileFormat.WeightsEntryPath && weights is not null ? weights
                : e.Value)).ToList());
        void RefusedLoad(params string[] fragments)
        {
            var ex = Assert.Throws<InvalidDataException>(() => Persistence.Load(tamperedPath));
            foreach (var fragment in fragments)
                Assert.Contains(fragment, ex.Message);
        }

        var rawAsZstd = JsonNode.Parse(plainEntries[SkptFileFormat.ConfigEntryName])!;
        rawAsZstd["data"]!["weights"]!["compression"] = SkptFileFormat.CompressionZstd;
        RewriteWith(plainEntries, rawAsZstd.ToJsonString());
        RefusedLoad(SkptFileFormat.WeightsEntryPath, "not a Zstd frame");

        var zstdAsRaw = JsonNode.Parse(zstdEntries[SkptFileFormat.ConfigEntryName])!;
        zstdAsRaw["data"]!["weights"]!["compression"] = SkptFileFormat.CompressionNone;
        RewriteWith(zstdEntries, zstdAsRaw.ToJsonString());
        RefusedLoad(SkptFileFormat.WeightsEntryPath, "Zstd frame");

        var unknown = JsonNode.Parse(plainEntries[SkptFileFormat.ConfigEntryName])!;
        unknown["data"]!["weights"]!["compression"] = "lz4";
        RewriteWith(plainEntries, unknown.ToJsonString());
        RefusedLoad("lz4", "unsupported compression");

        // A corrupt Zstd frame with a matching sha256: integrity passes, decompression fails loud.
        var truncated = storedWeights.Take(storedWeights.Length / 2).ToArray();
        Assert.True(SkptFileFormat.LooksLikeZstdFrame(truncated));
        var corrupt = JsonNode.Parse(zstdEntries[SkptFileFormat.ConfigEntryName])!;
        corrupt["data"]!["weights"]!["sha256"] = SkptFileFormat.Sha256Hex(truncated);
        RewriteWith(zstdEntries, corrupt.ToJsonString(), truncated);
        RefusedLoad(SkptFileFormat.WeightsEntryPath, "Zstd-decompress");
    }

    /// <summary>Every file of a .skpt checkpoint directory keyed by its manifest-style relative
    /// path (forward slashes) — the directory analogue of <see cref="ReadZipEntries"/>.</summary>
    private static Dictionary<string, byte[]> ReadDirectoryEntries(string dirPath)
        => Directory.EnumerateFiles(dirPath, "*", SearchOption.AllDirectories)
            .ToDictionary(
                f => Path.GetRelativePath(dirPath, f).Replace(Path.DirectorySeparatorChar, '/'),
                File.ReadAllBytes,
                StringComparer.Ordinal);

    [Fact]
    public void TestSkptDirectoryFormRoundTripAndConversions()
    {
        var (model, numOut, input) = BuildSkptModel();
        var direct = ExecuteToBytes(model, numOut, input);
        var originalWeights = WeightBytesByParam(model);
        var zipPath = P("dirform.skpt");
        var dirPath = P("dirform-dir.skpt");
        Persistence.From(model).WithModel().WithWeights().Save(zipPath);
        Persistence.From(model).WithModel().WithWeights().SaveAsDirectory(dirPath);

        var zipEntries = ReadZipEntries(zipPath);
        var dirEntries = ReadDirectoryEntries(dirPath);
        Assert.Equal(zipEntries.Keys.OrderBy(n => n, StringComparer.Ordinal),
            dirEntries.Keys.OrderBy(n => n, StringComparer.Ordinal));

        var fromDir = Persistence.Load(dirPath);
        Assert.Equal(GraphKind.ConcreteModel, fromDir.Kind);
        var dirWeights = WeightBytesByParam(fromDir);
        Assert.Equal(originalWeights.Count, dirWeights.Count);
        foreach (var (paramId, bytes) in originalWeights)
            Assert.Equal(bytes, dirWeights[paramId]);
        Assert.Equal(direct, ExecuteToBytes(fromDir, numOut, input));
        Assert.Equal(direct, ExecuteToBytes(Persistence.Load(zipPath), numOut, input));

        var dirManifest = SkptFileFormat.ParseManifest(dirEntries[SkptFileFormat.ConfigEntryName], dirPath);
        Assert.Equal(SkptFileFormat.Sha256Hex(dirEntries[SkptFileFormat.WeightsEntryPath]),
            Assert.Single(dirManifest.Data!).Value.Sha256);
        Assert.Contains("default",
            Assert.Throws<InvalidDataException>(() => Persistence.Load(dirPath, "nope")).Message);

        var extractedPath = P("extracted.skpt");
        Persistence.ExtractSkpt(zipPath, extractedPath);
        var extractedEntries = ReadDirectoryEntries(extractedPath);
        Assert.Equal(zipEntries.Keys.OrderBy(n => n, StringComparer.Ordinal),
            extractedEntries.Keys.OrderBy(n => n, StringComparer.Ordinal));
        foreach (var (name, bytes) in zipEntries)
            Assert.Equal(bytes, extractedEntries[name]);
        Assert.Equal(direct, ExecuteToBytes(Persistence.Load(extractedPath), numOut, input));

        var packedPath = P("packed.skpt");
        Persistence.PackSkpt(extractedPath, packedPath);
        var packedEntries = ReadZipEntries(packedPath);
        Assert.Equal(zipEntries.Keys.OrderBy(n => n, StringComparer.Ordinal),
            packedEntries.Keys.OrderBy(n => n, StringComparer.Ordinal));
        foreach (var (name, bytes) in zipEntries)
            Assert.Equal(bytes, packedEntries[name]);
        var packedHeaders = ParseLocalZipHeaders(File.ReadAllBytes(packedPath));
        Assert.All(packedHeaders, h => Assert.Equal(0, h.Method));
        Assert.Equal(0L, packedHeaders.Single(h => h.Name == SkptFileFormat.WeightsEntryPath)
            .DataOffset % SkptFileFormat.DataAlignment);
        Assert.Equal(direct, ExecuteToBytes(Persistence.Load(packedPath), numOut, input));

        Assert.Contains("PackSkpt",
            Assert.Throws<ArgumentException>(() => Persistence.ExtractSkpt(dirPath, P("x.skpt"))).Message);
        Assert.Contains("ExtractSkpt",
            Assert.Throws<ArgumentException>(() => Persistence.PackSkpt(zipPath, P("y.skpt"))).Message);

        var (zModel, zNumOut, zInput) = BuildCompressibleSkptModel();
        var zstdZip = P("zstd-dirform.skpt");
        var zstdDir = P("zstd-dirform-dir.skpt");
        Persistence.From(zModel).WithModel().WithWeights().WithZstdCompressedData().Save(zstdZip);
        Persistence.ExtractSkpt(zstdZip, zstdDir);
        Assert.True(SkptFileFormat.LooksLikeZstdFrame(
            File.ReadAllBytes(Path.Combine(zstdDir, SkptFileFormat.WeightsEntryPath))));
        Assert.Equal(ExecuteToBytes(zModel, zNumOut, zInput),
            ExecuteToBytes(Persistence.Load(zstdDir), zNumOut, zInput));
    }

    [Fact]
    public void TestInspectSkptDirectoryForm()
    {
        var (model, _, _) = BuildSkptModel();
        var dirPath = P("inspect-dir.skpt");
        Persistence.From(model).WithModel().WithWeights().SaveAsDirectory(dirPath);
        var dirEntries = ReadDirectoryEntries(dirPath);

        var result = Persistence.Inspect(dirPath);
        Assert.Equal(ArtifactKind.SkptCheckpoint, result.Kind);
        Assert.Equal(dirPath, result.FilePath);
        Assert.Equal(dirEntries.Values.Sum(b => b.LongLength), result.FileSizeBytes);
        Assert.Empty(result.Observations);
        var skpt = result.Skpt!;
        Assert.Equal(SkptFileFormat.CurrentVersion, skpt.SkptVersion);
        Assert.Equal(SkptFileFormat.ModelEntryPath, Assert.Single(skpt.Models).EntryPath);
        var dataSummary = Assert.Single(skpt.DataEntries);
        Assert.Equal(SkptFileFormat.WeightsEntryPath, dataSummary.EntryPath);
        Assert.Equal(dirEntries[SkptFileFormat.WeightsEntryPath].LongLength, dataSummary.DeclaredSizeBytes);
        string[] expectedSets = ["default"];
        Assert.Equal(expectedSets, skpt.MappingSetNames);

        File.WriteAllBytes(Path.Combine(dirPath, "data", "stray.bin"), new byte[16]);
        File.Delete(Path.Combine(dirPath, SkptFileFormat.ModelEntryPath));
        AssertInspection(Persistence.Inspect(dirPath), ArtifactKind.SkptCheckpoint,
            [SkptFileFormat.ModelEntryPath, "no such entry"], ["data/stray.bin", "not referenced"]);

        var plainDir = P("plain-dir");
        Directory.CreateDirectory(plainDir);
        File.WriteAllText(Path.Combine(plainDir, "readme.txt"), "just a directory");
        AssertInspection(Persistence.Inspect(plainDir), ArtifactKind.NotRecognized,
            [SkptFileFormat.ConfigEntryName]);
        File.WriteAllText(Path.Combine(plainDir, SkptFileFormat.ConfigEntryName),
            "{\"name\":\"other-tool\"}");
        AssertInspection(Persistence.Inspect(plainDir), ArtifactKind.NotRecognized, ["format"]);
        File.WriteAllText(Path.Combine(plainDir, SkptFileFormat.ConfigEntryName), "not json");
        AssertInspection(Persistence.Inspect(plainDir), ArtifactKind.NotRecognized, ["not a readable"]);
    }

    [Fact]
    public void TestSkptDirectorySaveAtomicityInterruptionAndPathEscapes()
    {
        var (model, numOut, input) = BuildSkptModel();
        var direct = ExecuteToBytes(model, numOut, input);
        var target = P("atomic-dir.skpt");

        Assert.Throws<DirectoryNotFoundException>(() => Persistence.From(model).WithModel().WithWeights()
            .SaveAsDirectory(P(Path.Combine("no-such-dir", "model.skpt"))));
        var filePath = P("already-a-file.skpt");
        File.WriteAllBytes(filePath, [1]);
        Assert.Throws<IOException>(() => Persistence.From(model).WithModel().WithWeights()
            .SaveAsDirectory(filePath));

        // A simulated crash in the commit window leaves the existing checkpoint fully intact —
        // an interrupted save is never visible at the target path.
        Persistence.From(model).WithModel().WithWeights().SaveAsDirectory(target);
        var committed = ReadDirectoryEntries(target);
        AtomicFileWriter.CommitFaultInjection = tempPath =>
        {
            if (tempPath.Contains("atomic-dir.skpt")) throw new IOException("simulated commit crash");
        };
        try
        {
            Assert.Throws<IOException>(
                () => Persistence.From(model).WithModel().WithWeights().SaveAsDirectory(target));
        }
        finally { AtomicFileWriter.CommitFaultInjection = null; }
        var afterCrash = ReadDirectoryEntries(target);
        Assert.Equal(committed.Keys.OrderBy(n => n, StringComparer.Ordinal),
            afterCrash.Keys.OrderBy(n => n, StringComparer.Ordinal));
        foreach (var (name, bytes) in committed)
            Assert.Equal(bytes, afterCrash[name]);
        Assert.Equal(direct, ExecuteToBytes(Persistence.Load(target), numOut, input));

        var (zModel, zNumOut, zInput) = BuildCompressibleSkptModel();
        Persistence.From(zModel).WithModel().WithWeights().SaveAsDirectory(target);
        var zDirect = ExecuteToBytes(zModel, zNumOut, zInput);
        Assert.Equal(zDirect, ExecuteToBytes(Persistence.Load(target), zNumOut, zInput));

        // A failure after the previous checkpoint was renamed aside (mid-replace) rolls it back.
        AtomicFileWriter.ReplaceFaultInjection = tempPath =>
        {
            if (tempPath.Contains("atomic-dir.skpt")) throw new IOException("simulated replace crash");
        };
        try
        {
            Assert.Throws<IOException>(
                () => Persistence.From(model).WithModel().WithWeights().SaveAsDirectory(target));
        }
        finally { AtomicFileWriter.ReplaceFaultInjection = null; }
        Assert.Equal(zDirect, ExecuteToBytes(Persistence.Load(target), zNumOut, zInput));

        // A '-sweep' leftover from an interrupted earlier sweep is finished off by the next save.
        var doomed = P($".tmp-atomic-dir.skpt-{Guid.NewGuid():N}-sweep");
        Directory.CreateDirectory(doomed);
        File.WriteAllBytes(Path.Combine(doomed, "junk.bin"), [1]);
        Persistence.From(model).WithModel().WithWeights().SaveAsDirectory(target);
        Assert.False(Directory.Exists(doomed));
        Assert.Equal(direct, ExecuteToBytes(Persistence.Load(target), numOut, input));

        var partial = P("partial.skpt");
        Directory.CreateDirectory(Path.Combine(partial, "data"));
        File.WriteAllBytes(Path.Combine(partial, SkptFileFormat.WeightsEntryPath),
            committed[SkptFileFormat.WeightsEntryPath]);
        Assert.Contains(SkptFileFormat.ConfigEntryName,
            Assert.Throws<InvalidDataException>(() => Persistence.Load(partial)).Message);
        Assert.Throws<FileNotFoundException>(() => Persistence.Load(P("nope.skpt")));

        // ── A hostile manifest's entry path may never resolve outside the checkpoint.
        var hostile = P("hostile.skpt");
        Persistence.From(model).WithModel().WithWeights().SaveAsDirectory(hostile);
        var configPath = Path.Combine(hostile, SkptFileFormat.ConfigEntryName);
        var goodConfig = File.ReadAllBytes(configPath);
        File.WriteAllBytes(P("outside-secret.bin"), [0xAA, 0xBB]);

        void Refused(string registry, string entryValue, params string[] fragments)
        {
            var config = JsonNode.Parse(goodConfig)!;
            (registry == "data" ? config["data"]!["weights"]! : config["models"]!["model"]!)
                ["entry"] = entryValue;
            File.WriteAllBytes(configPath, System.Text.Encoding.UTF8.GetBytes(config.ToJsonString()));
            var ex = Assert.Throws<InvalidDataException>(() => Persistence.Load(hostile));
            foreach (var fragment in fragments)
                Assert.Contains(fragment, ex.Message);
        }
        Refused("data", "../outside-secret.bin", "escapes");
        Refused("data", "data/../../outside-secret.bin", "escapes");
        Refused("data", P("outside-secret.bin"), "absolute");
        Refused("models", "../outside-secret.bin", "escapes");

        var inspection = Persistence.Inspect(hostile);
        Assert.Equal(ArtifactKind.SkptCheckpoint, inspection.Kind);
        Assert.Contains(inspection.Observations, o => o.Contains("escapes"));
        Assert.Contains("escapes", Assert.Throws<InvalidDataException>(
            () => Persistence.PackSkpt(hostile, P("packed-hostile.skpt"))).Message);
        File.WriteAllBytes(configPath, goodConfig);
        Assert.Equal(direct, ExecuteToBytes(Persistence.Load(hostile), numOut, input));

        // Extraction of a hostile zip cannot write outside the target either.
        var zipPath = P("hostile-zip.skpt");
        Persistence.From(model).WithModel().WithWeights().Save(zipPath);
        var entries = ReadZipEntries(zipPath);
        var config2 = JsonNode.Parse(entries[SkptFileFormat.ConfigEntryName])!;
        config2["data"]!["weights"]!["entry"] = "../evil.bin";
        RewriteSkpt(zipPath, entries.Select(e => (
            e.Key == SkptFileFormat.WeightsEntryPath ? "../evil.bin" : e.Key,
            e.Key == SkptFileFormat.ConfigEntryName
                ? System.Text.Encoding.UTF8.GetBytes(config2.ToJsonString()) : e.Value)).ToList());
        var extractTarget = P("hostile-extract.skpt");
        Assert.Contains("escapes", Assert.Throws<InvalidDataException>(
            () => Persistence.ExtractSkpt(zipPath, extractTarget)).Message);
        Assert.False(Directory.Exists(extractTarget));
        Assert.False(File.Exists(P("evil.bin")));
    }

    /// <summary>The model's weight tensors (TensorData) keyed by parameter identifier, excluding
    /// the RNG identity parameter — the values an additional mapping set is built over.</summary>
    private static Dictionary<string, TensorData> WeightDataByParam(ComputationGraph model)
        => model.ToInternal().Nodes
            .Where(n => n.OpCode == InternalOpCodes.MODEL_PARAM_DATA
                && n.IdentifierTemplate !=
                    Shorokoo.Core.Nodes.Processors.Fast.FastWireRngKeyDerivation.RngSeedIdentifierTemplate)
            .ToDictionary(n => n.IdentifierTemplate!, n => n.GetTensorData()!, StringComparer.Ordinal);

    [Fact]
    public void TestInspectSkptArtifactsAndNamedMappingSets()
    {
        var (model, numOut, input) = BuildSkptModel();
        var path = P("inspect.skpt");
        var variantPath = P("inspect_variant.skpt");
        Persistence.From(model).WithModel().WithWeights().Save(path);
        var entries = ReadZipEntries(path);
        var manifest = SkptFileFormat.ParseManifest(entries[SkptFileFormat.ConfigEntryName], path);

        var result = Persistence.Inspect(path);
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
        Assert.Equal(SkptFileFormat.ModelFormatSrk1, modelSummary.Format);
        Assert.Equal(SrkFileFormat.StageName(GraphKind.ConcreteModel), modelSummary.Stage);
        Assert.Equal(SkptFileFormat.Sha256Hex(entries[SkptFileFormat.ModelEntryPath]), modelSummary.GraphHash);

        var dataSummary = Assert.Single(skpt.DataEntries);
        Assert.Equal("weights", dataSummary.Key);
        Assert.Equal(SkptFileFormat.WeightsEntryPath, dataSummary.EntryPath);
        Assert.Equal(SkptFileFormat.DataFormatSafeTensors, dataSummary.Format);
        Assert.Equal(SkptFileFormat.CompressionNone, dataSummary.Compression);
        Assert.Equal(entries[SkptFileFormat.WeightsEntryPath].LongLength, dataSummary.DeclaredSizeBytes);
        Assert.Equal(SkptFileFormat.Sha256Hex(entries[SkptFileFormat.WeightsEntryPath]), dataSummary.Sha256);

        string[] expectedSets = ["default"];
        Assert.Equal(expectedSets, skpt.MappingSetNames);

        var text = result.ToString();
        Assert.Contains(".skpt", text);
        Assert.Contains(SkptFileFormat.ModelEntryPath, text);
        Assert.Contains(SkptFileFormat.WeightsEntryPath, text);
        Assert.Contains("unverified", text);
        Assert.Contains("mapping sets: default", text);

        // Payload untouched: a corrupt weights payload fails a full load but inspects the same.
        var fileBytes = File.ReadAllBytes(path);
        var weightsHeader = ParseLocalZipHeaders(fileBytes)
            .Single(h => h.Name == SkptFileFormat.WeightsEntryPath);
        fileBytes[weightsHeader.DataOffset + weightsHeader.Size - 1] ^= 0xFF;
        File.WriteAllBytes(variantPath, fileBytes);
        Assert.Throws<InvalidDataException>(() => Persistence.Load(variantPath));
        var corrupt = Persistence.Inspect(variantPath);
        Assert.Equal(ArtifactKind.SkptCheckpoint, corrupt.Kind);
        Assert.Empty(corrupt.Observations);
        Assert.Equal(dataSummary.Sha256, Assert.Single(corrupt.Skpt!.DataEntries).Sha256);

        ArtifactInspection Rewritten(List<(string Name, byte[] Data)> variantEntries)
        {
            RewriteSkpt(variantPath, variantEntries);
            return Persistence.Inspect(variantPath);
        }
        List<(string Name, byte[] Data)> WithConfig(string configJson) =>
            entries.Select(e => (e.Key, e.Key == SkptFileFormat.ConfigEntryName
                ? System.Text.Encoding.UTF8.GetBytes(configJson) : e.Value)).ToList();
        static List<(string Name, byte[] Data)> Only(string name, string content) =>
            [(name, System.Text.Encoding.UTF8.GetBytes(content))];

        // Manifest/archive mismatches in both directions are observed, both ways round.
        var mismatch = Rewritten(entries.Where(e => e.Key != SkptFileFormat.WeightsEntryPath)
            .Select(e => (e.Key, e.Value)).Append(("data/stray.bin", new byte[16])).ToList());
        AssertInspection(mismatch, ArtifactKind.SkptCheckpoint,
            [SkptFileFormat.WeightsEntryPath, "no such entry"], ["data/stray.bin", "not referenced"]);
        Assert.Null(Assert.Single(mismatch.Skpt!.DataEntries).DeclaredSizeBytes);

        var futureConfig = JsonNode.Parse(entries[SkptFileFormat.ConfigEntryName])!;
        futureConfig["futureTopLevelKey"] = "??";
        futureConfig["skptVersion"] = SkptFileFormat.CurrentVersion + 1;
        var future = Rewritten(WithConfig(futureConfig.ToJsonString()));
        AssertInspection(future, ArtifactKind.SkptCheckpoint,
            ["futureTopLevelKey"], [$"version {SkptFileFormat.CurrentVersion + 1}"]);
        Assert.Equal(SkptFileFormat.CurrentVersion + 1, future.Skpt!.SkptVersion);

        // STORED-expectation violation: the same entries written deflated by the BCL writer.
        using (var deflated = new ZipArchive(File.Create(variantPath), ZipArchiveMode.Create))
        {
            foreach (var (name, data) in entries)
            {
                using var s = deflated.CreateEntry(name, CompressionLevel.SmallestSize).Open();
                s.Write(data);
            }
        }
        AssertInspection(Persistence.Inspect(variantPath), ArtifactKind.SkptCheckpoint, ["expected STORED"]);

        AssertInspection(
            Rewritten(Only(SkptFileFormat.ConfigEntryName, "{\"format\":\"skpt\",\"skptVersion\":1}")),
            ArtifactKind.SkptCheckpoint, ["no models"], ["no data entries"], ["no tensor mapping sets"]);
        AssertInspection(Rewritten(Only("readme.txt", "just a zip")),
            ArtifactKind.NotRecognized, [SkptFileFormat.ConfigEntryName]);
        AssertInspection(Rewritten(Only(SkptFileFormat.ConfigEntryName, "{\"name\":\"some-other-tool\"}")),
            ArtifactKind.NotRecognized, ["format"]);
        AssertInspection(Rewritten(Only(SkptFileFormat.ConfigEntryName, "not json at all")),
            ArtifactKind.NotRecognized, ["not a readable"]);

        File.WriteAllBytes(variantPath, File.ReadAllBytes(path)[..40]);
        AssertInspection(Persistence.Inspect(variantPath), ArtifactKind.NotRecognized, ["not readable"]);

        byte[] garbageZip = [0x50, 0x4B, 0x03, 0x04, 0xDE, 0xAD, 0xBE, 0xEF];
        File.WriteAllBytes(variantPath, garbageZip);
        var garbage = Persistence.Inspect(variantPath);
        Assert.Equal(ArtifactKind.NotRecognized, garbage.Kind);
        Assert.NotEmpty(garbage.Observations);

        // ── Named mapping sets over shared data: an "ema" set with one distinct tensor (its
        // own data entry) and one byte-identical to the default (shared, stored once).
        var setsPath = P("named-sets.skpt");
        var defaultOnlyPath = P("named-sets-default-only.skpt");
        var modelData = WeightDataByParam(model);
        Assert.Equal(2, modelData.Count);
        var distinctId = modelData
            .OrderByDescending(kv => kv.Value.Shape.Dims.Aggregate(1L, (a, d) => a * d))
            .First().Key;
        var sharedId = modelData.Keys.Single(k => k != distinctId);

        var emaValues = new Dictionary<string, TensorData>(StringComparer.Ordinal);
        foreach (var (id, data) in modelData)
        {
            if (id == distinctId)
            {
                var dims = data.Shape.Dims;
                var vals = new float[dims.Aggregate(1L, (a, d) => a * d)];
                for (int i = 0; i < vals.Length; i++) vals[i] = 2.0f + i * 0.5f;
                emaValues[id] = TensorData(dims, vals);
            }
            else
            {
                emaValues[id] = data;
            }
        }
        Assert.NotEqual(modelData[distinctId].AccessRawMemory().ToArray(),
            emaValues[distinctId].AccessRawMemory().ToArray());

        Persistence.From(model).WithModel().WithWeights().WithWeights("ema", emaValues).Save(setsPath);
        Persistence.From(model).WithModel().WithWeights().Save(defaultOnlyPath);

        var setEntries = ReadZipEntries(setsPath);
        var defaultOnlyEntries = ReadZipEntries(defaultOnlyPath);
        const string emaEntryPath = "data/ema.safetensors";
        Assert.Equal(
            ((string[])[SkptFileFormat.ConfigEntryName, SkptFileFormat.ModelEntryPath,
                        SkptFileFormat.WeightsEntryPath, emaEntryPath])
                .OrderBy(n => n, StringComparer.Ordinal),
            setEntries.Keys.OrderBy(n => n, StringComparer.Ordinal));
        Assert.Equal(
            ((string[])[SkptFileFormat.ConfigEntryName, SkptFileFormat.ModelEntryPath,
                        SkptFileFormat.WeightsEntryPath])
                .OrderBy(n => n, StringComparer.Ordinal),
            defaultOnlyEntries.Keys.OrderBy(n => n, StringComparer.Ordinal));

        Assert.Equal(defaultOnlyEntries[SkptFileFormat.ModelEntryPath],
            setEntries[SkptFileFormat.ModelEntryPath]);
        Assert.Equal(defaultOnlyEntries[SkptFileFormat.WeightsEntryPath],
            setEntries[SkptFileFormat.WeightsEntryPath]);

        var emaStored = SafeTensorLoader.ParseSafeTensorBytes(setEntries[emaEntryPath])
            .ToDictionary(t => t.Name, t => t.Data.AccessRawMemory().ToArray(), StringComparer.Ordinal);
        Assert.Equal((string[])[distinctId], emaStored.Keys.ToArray());
        Assert.DoesNotContain(sharedId, emaStored.Keys);

        var setManifest = SkptFileFormat.ParseManifest(setEntries[SkptFileFormat.ConfigEntryName], setsPath);
        var sets = setManifest.TensorMappings!["model"];
        Assert.Equal((string[])["default", "ema"], sets.Keys.ToArray());
        Assert.Equal(2, setManifest.Data!.Count);
        Assert.Equal(emaEntryPath, setManifest.Data["ema"].Entry);
        var emaRefs = sets["ema"].Tensors!;
        Assert.Equal(SkptFileFormat.DefaultDataKey, emaRefs[sharedId].Data);
        Assert.Equal(sharedId, emaRefs[sharedId].Tensor);
        Assert.Equal("ema", emaRefs[distinctId].Data);
        Assert.Equal(distinctId, emaRefs[distinctId].Tensor);

        var originalBytes = WeightBytesByParam(model);
        var loadedDefault = WeightBytesByParam(Persistence.Load(setsPath));
        Assert.Equal(originalBytes.Count, loadedDefault.Count);
        foreach (var (id, bytes) in originalBytes)
            Assert.Equal(bytes, loadedDefault[id]);

        var loadedEma = WeightBytesByParam(Persistence.Load(setsPath, "ema"));
        Assert.Equal(emaValues[distinctId].AccessRawMemory().ToArray(), loadedEma[distinctId]);
        Assert.Equal(originalBytes[sharedId], loadedEma[sharedId]);
        Assert.NotEqual(loadedDefault[distinctId], loadedEma[distinctId]);
        Assert.Equal(ExecuteToBytes(model, numOut, input),
            ExecuteToBytes(Persistence.Load(setsPath, "default"), numOut, input));

        var exAbsent = Assert.Throws<InvalidDataException>(() => Persistence.Load(setsPath, "nope"));
        Assert.Contains("nope", exAbsent.Message);
        Assert.Contains("default", exAbsent.Message);
        Assert.Contains("ema", exAbsent.Message);

        var inspected = Persistence.Inspect(setsPath);
        Assert.Equal(ArtifactKind.SkptCheckpoint, inspected.Kind);
        Assert.Equal((string[])["default", "ema"], inspected.Skpt!.MappingSetNames.ToArray());
        Assert.Contains("mapping sets: default, ema", inspected.ToString());

        // A set fully shared with the default weights adds no data entry, yet still loads.
        var sharedOnlyPath = P("named-sets-shared-only.skpt");
        Persistence.From(model).WithModel().WithWeights()
            .WithWeights("shadow", modelData).Save(sharedOnlyPath);
        var sharedEntries = ReadZipEntries(sharedOnlyPath);
        Assert.DoesNotContain("data/shadow.safetensors", sharedEntries.Keys);
        Assert.Equal(defaultOnlyEntries.Keys.OrderBy(n => n, StringComparer.Ordinal),
            sharedEntries.Keys.OrderBy(n => n, StringComparer.Ordinal));
        var sharedManifest = SkptFileFormat.ParseManifest(
            sharedEntries[SkptFileFormat.ConfigEntryName], sharedOnlyPath);
        Assert.Equal((string[])["default", "shadow"],
            sharedManifest.TensorMappings!["model"].Keys.ToArray());
        Assert.Equal((string[])[SkptFileFormat.DefaultDataKey], sharedManifest.Data!.Keys.ToArray());
        Assert.All(sharedManifest.TensorMappings["model"]["shadow"].Tensors!.Values,
            r => Assert.Equal(SkptFileFormat.DefaultDataKey, r.Data));
        var shadowBytes = WeightBytesByParam(Persistence.Load(sharedOnlyPath, "shadow"));
        foreach (var (id, bytes) in originalBytes)
            Assert.Equal(bytes, shadowBytes[id]);

        // Builder validation: reserved names, an incomplete set and a stray parameter fail up front.
        Assert.Throws<ArgumentException>(() => Persistence.From(model).WithWeights("default", emaValues));
        Assert.Throws<ArgumentException>(() => Persistence.From(model).WithWeights("weights", emaValues));
        Assert.Throws<ArgumentException>(() => Persistence.From(model).WithWeights("bad/name", emaValues));
        var incompletePath = P("named-sets-incomplete.skpt");
        var missing = new Dictionary<string, TensorData>(StringComparer.Ordinal)
            { [distinctId] = emaValues[distinctId] };
        var exMissing = Assert.Throws<InvalidOperationException>(() =>
            Persistence.From(model).WithModel().WithWeights().WithWeights("ema", missing).Save(incompletePath));
        Assert.Contains(sharedId, exMissing.Message);
        var stray = new Dictionary<string, TensorData>(emaValues, StringComparer.Ordinal)
            { ["not_a_real_param"] = emaValues[distinctId] };
        var exStray = Assert.Throws<InvalidOperationException>(() =>
            Persistence.From(model).WithModel().WithWeights().WithWeights("ema", stray).Save(incompletePath));
        Assert.Contains("not_a_real_param", exStray.Message);
        Assert.False(File.Exists(incompletePath));
    }

    /// <summary>A host pipeline-state POCO for the generic
    /// <c>WithUserData&lt;T&gt;</c> / <c>GetUserData&lt;T&gt;</c> round-trip.</summary>
    private sealed class PipelineState
    {
        public string? Corpus { get; set; }
        public int ShuffleSeed { get; set; }
        public int Epoch { get; set; }
        public string[] Shards { get; set; } = [];
    }

    [Fact]
    public void TestSkptUserProvenanceMetadataAndHostUserDataBag()
    {
        var (model, numOut, input) = BuildSkptModel();
        var plainPath = P("provenance_plain.skpt");
        var metaPath = P("provenance_meta.skpt");
        var reference = ExecuteToBytes(model, numOut, input);

        Persistence.From(model).WithModel().WithWeights().Save(plainPath);
        var plainEntries = ReadZipEntries(plainPath);
        var plainConfig = plainEntries[SkptFileFormat.ConfigEntryName];
        Assert.DoesNotContain("userMetadata", System.Text.Encoding.UTF8.GetString(plainConfig));
        Assert.False(plainEntries.ContainsKey(SkptFileFormat.UserDataEntryPath));
        Assert.DoesNotContain("user-data.json", System.Text.Encoding.UTF8.GetString(plainConfig));

        // Byte-identity is proven at the serialization boundary (the file's createdUtc/zip
        // timestamps are not): the serializer omits nulls, so null and absent are the same bytes.
        SkptManifest Template() => new()
        {
            Format = SkptFileFormat.FormatName,
            SkptVersion = SkptFileFormat.CurrentVersion,
            CreatedUtc = "2026-07-22T00:00:00Z",
            Producer = new SkptProducerInfo { Shorokoo = "x" },
        };
        var absentBytes = SkptFileFormat.SerializeManifest(Template());
        var withNull = Template();
        withNull.UserMetadata = null;
        Assert.Equal(absentBytes, SkptFileFormat.SerializeManifest(withNull));
        Assert.DoesNotContain("userMetadata", System.Text.Encoding.UTF8.GetString(absentBytes));

        var plainInspect = Persistence.Inspect(plainPath);
        Assert.Equal(ArtifactKind.SkptCheckpoint, plainInspect.Kind);
        Assert.Null(plainInspect.Skpt!.UserMetadata);
        Assert.Null(plainInspect.Skpt.UserData);
        Assert.Null(plainInspect.Skpt.GetUserData<PipelineState>());
        Assert.DoesNotContain("user metadata", plainInspect.ToString());
        Assert.DoesNotContain("user-data:", plainInspect.ToString());

        // Well-known keys as named parameters + extra pairs; a named parameter wins over the map.
        var extra = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["experiment"] = "ablation-7",
            [SkptFileFormat.MetadataLicenseKey] = "OVERRIDDEN",
        };
        Persistence.From(model).WithModel().WithWeights()
            .WithMetadata(extra,
                gitCommit: "9f3c1ba",
                datasetId: "imagenet-1k@v2",
                runName: "nightly-run-42",
                license: "Apache-2.0")
            .Save(metaPath);

        var metaEntries = ReadZipEntries(metaPath);
        var metaManifest = SkptFileFormat.ParseManifest(
            metaEntries[SkptFileFormat.ConfigEntryName], metaPath);
        var recorded = metaManifest.UserMetadata!;
        Assert.Equal("9f3c1ba", recorded[SkptFileFormat.MetadataGitCommitKey]);
        Assert.Equal("imagenet-1k@v2", recorded[SkptFileFormat.MetadataDatasetIdKey]);
        Assert.Equal("nightly-run-42", recorded[SkptFileFormat.MetadataRunNameKey]);
        Assert.Equal("Apache-2.0", recorded[SkptFileFormat.MetadataLicenseKey]);
        Assert.Equal("ablation-7", recorded["experiment"]);

        var metaInspect = Persistence.Inspect(metaPath);
        Assert.Equal(ArtifactKind.SkptCheckpoint, metaInspect.Kind);
        var userMeta = metaInspect.Skpt!.UserMetadata!;
        Assert.NotNull(userMeta);
        Assert.Equal(5, userMeta.Count);
        foreach (var (k, v) in recorded)
            Assert.Equal(v, userMeta[k]);
        Assert.Empty(metaInspect.Observations);
        Assert.Equal(Shorokoo.ShorokooVersion.VersionString, metaInspect.Skpt.Producer);
        Assert.False(userMeta.ContainsKey("shorokoo"));

        var metaText = metaInspect.ToString();
        Assert.Contains("user metadata", metaText);
        Assert.Contains($"{SkptFileFormat.MetadataGitCommitKey}: 9f3c1ba", metaText);
        Assert.Contains($"{SkptFileFormat.MetadataDatasetIdKey}: imagenet-1k@v2", metaText);
        Assert.Contains($"{SkptFileFormat.MetadataRunNameKey}: nightly-run-42", metaText);
        Assert.Contains($"{SkptFileFormat.MetadataLicenseKey}: Apache-2.0", metaText);
        Assert.Contains("experiment: ablation-7", metaText);

        var plainLoaded = Persistence.Load(plainPath);
        var metaLoaded = Persistence.Load(metaPath);
        var plainWeights = WeightBytesByParam(plainLoaded);
        var metaWeights = WeightBytesByParam(metaLoaded);
        Assert.Equal(plainWeights.Count, metaWeights.Count);
        foreach (var (paramId, bytes) in plainWeights)
            Assert.Equal(bytes, metaWeights[paramId]);
        Assert.Equal(reference, ExecuteToBytes(plainLoaded, numOut, input));
        Assert.Equal(reference, ExecuteToBytes(metaLoaded, numOut, input));

        Assert.Throws<ArgumentException>(() =>
            Persistence.From(model).WithMetadata(new Dictionary<string, string> { [""] = "v" }));
        Assert.Throws<ArgumentNullException>(() =>
            Persistence.From(model).WithMetadata(new Dictionary<string, string> { ["k"] = null! }));

        // A large metadata map renders bounded: capped like the other registries, then elided.
        var bigPath = P("provenance_big.skpt");
        var big = new Dictionary<string, string>(StringComparer.Ordinal);
        for (int i = 0; i < 200; i++) big[$"k{i:D3}"] = $"v{i}";
        Persistence.From(model).WithModel().WithWeights().WithMetadata(big).Save(bigPath);
        var bigInspect = Persistence.Inspect(bigPath);
        Assert.Equal(200, bigInspect.Skpt!.UserMetadata!.Count);
        var bigText = bigInspect.ToString();
        Assert.Contains("user metadata (200", bigText);
        Assert.Contains("more", bigText);
        Assert.True(bigText.Split('\n').Count(l => l.TrimStart().StartsWith("k0")) <= 50);

        // Control characters stay raw in the structured property, sanitized in ToString.
        var hostilePath = P("provenance_hostile.skpt");
        const string hostileValue = "clean\n  note: forged\ttab\u0000nul\u0085nel\u009Fc1\u007Fdel";
        const string hostileKey = "e\nvil";
        Persistence.From(model).WithModel().WithWeights()
            .WithMetadata(new Dictionary<string, string>(StringComparer.Ordinal)
            {
                [SkptFileFormat.MetadataRunNameKey] = hostileValue,
                [hostileKey] = "hk",
            }).Save(hostilePath);

        var hostileInspect = Persistence.Inspect(hostilePath);
        var hostileMeta = hostileInspect.Skpt!.UserMetadata!;
        Assert.Equal(hostileValue, hostileMeta[SkptFileFormat.MetadataRunNameKey]);
        Assert.True(hostileMeta.ContainsKey(hostileKey));
        var hostileText = hostileInspect.ToString();
        Assert.DoesNotContain("\n  note: forged", hostileText);
        Assert.DoesNotContain("forged\ttab", hostileText);
        Assert.DoesNotContain("e\nvil", hostileText);
        Assert.DoesNotContain('\u0000', hostileText);   // C0 (NUL)
        Assert.DoesNotContain('\u0085', hostileText);   // C1 (NEL)
        Assert.DoesNotContain('\u009F', hostileText);   // C1
        Assert.DoesNotContain('\u007F', hostileText);   // DEL
        Assert.Contains('�', hostileText);
        Assert.Equal(reference, ExecuteToBytes(Persistence.Load(hostilePath), numOut, input));

        // ── Host user-data bag: an arbitrary JSON object stored as data/user-data.json and
        // wired into the manifest's data registry.
        var dataPath = P("userdata_bag.skpt");
        var bag = new JsonObject
        {
            ["corpus"] = "imagenet-1k",
            ["shuffleSeed"] = 12345,
            ["epoch"] = 3,
            ["done"] = false,
            ["cursor"] = null,
            ["shards"] = new JsonArray("a.tar", "b.tar", "c.tar"),
            ["augment"] = new JsonObject { ["flip"] = true, ["crop"] = 224 },
        };
        Persistence.From(model).WithModel().WithWeights().WithUserData(bag).Save(dataPath);

        var dataEntries = ReadZipEntries(dataPath);
        Assert.True(dataEntries.ContainsKey(SkptFileFormat.UserDataEntryPath));
        var dataManifest = SkptFileFormat.ParseManifest(
            dataEntries[SkptFileFormat.ConfigEntryName], dataPath);
        var udEntry = dataManifest.Data![SkptFileFormat.UserDataDataKey];
        Assert.Equal(SkptFileFormat.UserDataEntryPath, udEntry.Entry);
        Assert.Equal(SkptFileFormat.DataFormatJson, udEntry.Format);
        Assert.Equal(SkptFileFormat.CompressionNone, udEntry.Compression);
        Assert.Equal(SkptFileFormat.Sha256Hex(dataEntries[SkptFileFormat.UserDataEntryPath]), udEntry.Sha256);

        var inspect = Persistence.Inspect(dataPath);
        Assert.Empty(inspect.Observations);
        var read = inspect.Skpt!.UserData!;
        Assert.NotNull(read);
        Assert.Equal(7, read.Count);
        Assert.Equal("imagenet-1k", (string?)read["corpus"]);
        Assert.Equal(12345, (int?)read["shuffleSeed"]);
        Assert.Equal(3, (int?)read["epoch"]);
        Assert.False((bool?)read["done"]);
        Assert.Null(read["cursor"]);
        Assert.Equal(3, read["shards"]!.AsArray().Count);
        Assert.Equal("b.tar", (string?)read["shards"]![1]);
        Assert.True((bool?)read["augment"]!["flip"]);
        Assert.Equal(224, (int?)read["augment"]!["crop"]);

        // Inspect summarizes the bag as a one-line key count — never a nested dump.
        var dataText = inspect.ToString();
        Assert.Contains("user-data: 7 keys", dataText);
        Assert.DoesNotContain("imagenet-1k", dataText);
        Assert.DoesNotContain("shuffleSeed", dataText);

        var dataWeights = WeightBytesByParam(Persistence.Load(dataPath));
        Assert.Equal(plainWeights.Count, dataWeights.Count);
        foreach (var (paramId, bytes) in plainWeights)
            Assert.Equal(bytes, dataWeights[paramId]);
        Assert.Equal(reference, ExecuteToBytes(Persistence.Load(dataPath), numOut, input));

        var pocoPath = P("userdata_poco.skpt");
        Persistence.From(model).WithModel().WithWeights().WithUserData(new PipelineState
        {
            Corpus = "c4",
            ShuffleSeed = 7,
            Epoch = 1,
            Shards = ["x", "y"],
        }).Save(pocoPath);
        var pocoBack = Persistence.Inspect(pocoPath).Skpt!.GetUserData<PipelineState>()!;
        Assert.Equal("c4", pocoBack.Corpus);
        Assert.Equal(7, pocoBack.ShuffleSeed);
        Assert.Equal(["x", "y"], pocoBack.Shards);

        Assert.Throws<ArgumentException>(() => Persistence.From(model).WithUserData((int[])[1, 2, 3]));
        Assert.Throws<ArgumentException>(() => Persistence.From(model).WithUserData("just a string"));
        Assert.Throws<ArgumentException>(() => Persistence.From(model).WithUserData(42));
        Assert.Throws<ArgumentException>(() => Persistence.From(model).WithUserData<object?>(null));
        Assert.Throws<ArgumentNullException>(() => Persistence.From(model).WithUserData((JsonObject)null!));

        // Only the root's $-prefixed keys are reserved; nested ones are ordinary data.
        Assert.Throws<ArgumentException>(() =>
            Persistence.From(model).WithUserData(new JsonObject { ["$reserved"] = 1 }));
        Assert.Throws<ArgumentException>(() =>
            Persistence.From(model).WithUserData(new Dictionary<string, int> { ["$nope"] = 1 }));
        Persistence.From(model).WithUserData(
            new JsonObject { ["ok"] = new JsonObject { ["$inner"] = 1 } });

        // The JsonObject overload snapshots: later mutation does not change what was stored.
        var mutPath = P("userdata_mut.skpt");
        var live = new JsonObject { ["v"] = 1 };
        var builder = Persistence.From(model).WithModel().WithWeights().WithUserData(live);
        live["v"] = 999;
        live["added"] = "late";
        builder.Save(mutPath);
        var snap = Persistence.Inspect(mutPath).Skpt!.UserData!;
        Assert.Equal(1, (int?)snap["v"]);
        Assert.False(snap.ContainsKey("added"));
    }

    // ──────────────────────────────────────────────────────────────────────
    // SafeTensors weight exchange: ExportSafeTensors / ImportSafeTensors, with
    // canonical names or a naming scheme, plus the one-call .skpt landing.
    // ──────────────────────────────────────────────────────────────────────

    private const string FcWeightsId = "TrainableParam#0.InitSimple#0";
    private const string FcBiasId = "TrainableParam#0.InitSimple#1";

    private static (ComputationGraph Arch, ComputationGraph Model, TensorData NumOut, TensorData Input)
        BuildSafeTensorsExchangeModel()
    {
        var numOut = TensorData(DType.Int64, [], 4L);
        var input = TensorDataWithSmallVals(DType.Float32, [4L, 4L]);
        var g = FCLayer.ComputationGraph;
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([numOut, input]));
        return (arch, arch.ToConcreteModel(), numOut, input);
    }

    /// <summary>The model's weight bytes keyed by canonical parameter name — graph nodes carry
    /// the serialized "[ModelId]:parts" identifier; the canonical name is the parts portion.</summary>
    private static Dictionary<string, byte[]> CanonicalWeightBytes(ComputationGraph model)
        => WeightBytesByParam(model).ToDictionary(
            kv => kv.Key[(kv.Key.IndexOf("]:", StringComparison.Ordinal) + 2)..],
            kv => kv.Value, StringComparer.Ordinal);

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

    [Fact]
    public void TestSafeTensorsExportImportRoundTripSchemesAndCheckpointLanding()
    {
        var (arch, model, numOut, input) = BuildSafeTensorsExchangeModel();
        var canonicalPath = P("exchange_canonical.safetensors");
        var torchPath = P("exchange_torch.safetensors");
        var originalWeights = CanonicalWeightBytes(model);
        var direct = ExecuteToBytes(model, numOut, input);
        Assert.Equal([FcWeightsId, FcBiasId],
            originalWeights.Keys.OrderBy(k => k, StringComparer.Ordinal).ToArray());

        Persistence.ExportSafeTensors(model, canonicalPath);
        var storedCanonical = SafeTensorLoader.LoadSafeTensors(canonicalPath)
            .ToDictionary(t => t.Name, t => t.Data.AccessRawMemory().ToArray(), StringComparer.Ordinal);
        Assert.Equal(originalWeights.Keys.OrderBy(k => k, StringComparer.Ordinal),
            storedCanonical.Keys.OrderBy(k => k, StringComparer.Ordinal));
        foreach (var (paramId, bytes) in originalWeights)
            Assert.Equal(bytes, storedCanonical[paramId]);

        var importedCanonical = Persistence.ImportSafeTensors(arch, canonicalPath);
        Assert.Equal(GraphKind.ConcreteModel, importedCanonical.Kind);
        Assert.Equal(originalWeights, CanonicalWeightBytes(importedCanonical));
        Assert.Equal(direct, ExecuteToBytes(importedCanonical, numOut, input));

        // A __metadata__ block is metadata, not a tensor: it must not trip the unmapped check.
        var withMetadata = SafeTensorLoader.LoadSafeTensors(canonicalPath);
        SafeTensorLoader.SaveSafeTensors(canonicalPath, withMetadata,
            new Dictionary<string, object> { ["format"] = "pt", ["producer"] = "unit-test" });
        Assert.Equal(direct,
            ExecuteToBytes(Persistence.ImportSafeTensors(arch, canonicalPath), numOut, input));

        var scheme = BuildFcTorchScheme(arch);
        Persistence.ExportSafeTensors(model, torchPath, scheme);
        var storedTorch = SafeTensorLoader.LoadSafeTensors(torchPath)
            .ToDictionary(t => t.Name, t => t.Data.AccessRawMemory().ToArray(), StringComparer.Ordinal);
        Assert.Equal(["fc.bias", "fc.weight"],
            storedTorch.Keys.OrderBy(k => k, StringComparer.Ordinal).ToArray());
        Assert.Equal(originalWeights[FcWeightsId], storedTorch["fc.weight"]);
        Assert.Equal(originalWeights[FcBiasId], storedTorch["fc.bias"]);

        var importedTorch = Persistence.ImportSafeTensors(arch, torchPath, scheme);
        Assert.Equal(originalWeights, CanonicalWeightBytes(importedTorch));
        Assert.Equal(direct, ExecuteToBytes(importedTorch, numOut, input));

        // The ModelId format DSL binds at the import boundary too.
        ModelIdFormat[] formats =
        [
            new ModelIdFormat(match: "[1]", format: "fc.weight"),
            new ModelIdFormat(match: "[2]", format: "fc.bias"),
        ];
        var importedFormat = Persistence.ImportSafeTensors(arch, torchPath,
            new ModelIdNamingScheme(formats, ModuleParamSetNamingScheme.PyTorchFrameworkId));
        Assert.Equal(originalWeights, CanonicalWeightBytes(importedFormat));
        Assert.Equal(direct, ExecuteToBytes(importedFormat, numOut, input));

        // The one-call native landing writes the bound result straight to a .skpt.
        var skptPath = P("exchange_landing.skpt");
        var landed = Persistence.ImportSafeTensorsToCheckpoint(arch, torchPath, skptPath, scheme);
        Assert.Equal(GraphKind.ConcreteModel, landed.Kind);
        Assert.Equal(direct, ExecuteToBytes(landed, numOut, input));
        var reloaded = Persistence.Load(skptPath);
        Assert.Equal(GraphKind.ConcreteModel, reloaded.Kind);
        Assert.Equal(WeightBytesByParam(model), WeightBytesByParam(reloaded));
        Assert.Equal(direct, ExecuteToBytes(reloaded, numOut, input));

        // A failed import lands nothing: the previously written checkpoint is untouched.
        var committed = File.ReadAllBytes(skptPath);
        SimplePatternScheme[] partial = [new SimplePatternScheme(FcWeightsId, "fc.weight")];
        var partialScheme = new SimplePatternNamingScheme(
            partial, arch.GetShorokooIdNamingScheme(), ModuleParamSetNamingScheme.PyTorchFrameworkId);
        Assert.Throws<InvalidDataException>(
            () => Persistence.ImportSafeTensorsToCheckpoint(arch, torchPath, skptPath, partialScheme));
        Assert.Equal(committed, File.ReadAllBytes(skptPath));

        // The injected RngExecutionCounter is bookkeeping, not a weight: export excludes it and
        // the import fills it from its initializer default.
        var rngNumOut = TensorData(DType.Int64, [], 4L);
        var rngInput = TensorDataWithSmallVals(DType.Float32, [4L, 4L]);
        var rngG = RtFcWithRngFeed.ComputationGraph;
        var rngArch = rngG.ToConcreteArchitecture(rngG.FromOrderedInputs([rngNumOut, rngInput]));
        var rngModel = rngArch.ToConcreteModel();
        var rngPath = P("rng_feed_exchange.safetensors");
        var rngDirect = ExecuteToBytes(rngModel, rngNumOut, rngInput);
        Persistence.ExportSafeTensors(rngModel, rngPath);
        var names = SafeTensorLoader.LoadSafeTensors(rngPath).Select(t => t.Name).ToList();
        Assert.NotEmpty(names);
        Assert.DoesNotContain(names, n => n.Contains(
            Shorokoo.Core.Nodes.Processors.Fast.FastInjectRngDrawCounter.CounterName));
        var rngImported = Persistence.ImportSafeTensors(rngArch, rngPath);
        Assert.Equal(GraphKind.ConcreteModel, rngImported.Kind);
        Assert.Equal(rngDirect, ExecuteToBytes(rngImported, rngNumOut, rngInput));
    }

    [Fact]
    public void TestSafeTensorsExchangeFailsLoudOnMappingMismatchesAndSchemeGaps()
    {
        var (arch, model, _, _) = BuildSafeTensorsExchangeModel();
        var path = P("exchange_good.safetensors");
        var badPath = P("exchange_bad.safetensors");
        Persistence.ExportSafeTensors(model, path);
        var good = SafeTensorLoader.LoadSafeTensors(path);

        void ImportRefused(IReadOnlyList<SafeTensor> tensors, params string[] fragments)
        {
            SafeTensorLoader.SaveSafeTensors(badPath, [.. tensors]);
            var ex = Assert.Throws<InvalidDataException>(
                () => Persistence.ImportSafeTensors(arch, badPath));
            foreach (var fragment in fragments)
                Assert.Contains(fragment, ex.Message);
        }

        var withStray = good.ToList();
        withStray.Add(new SafeTensor("not.a.param", TensorData([2L], 1f, 2f), "F32", [2L]));
        ImportRefused(withStray, "not.a.param", badPath);
        ImportRefused(good.Where(t => t.Name != FcBiasId).ToList(), FcBiasId);
        ImportRefused(good.Select(t => t.Name != FcWeightsId ? t
            : new SafeTensor(t.Name, TensorDataWithSmallVals(DType.Int64, [4L, 4L]), "I64", [4L, 4L])).ToList(),
            FcWeightsId, "dtype");
        ImportRefused(good.Select(t => t.Name != FcWeightsId ? t
            : new SafeTensor(t.Name, TensorDataWithSmallVals(DType.Float32, [2L, 8L]), "F32", [2L, 8L])).ToList(),
            FcWeightsId, "[2,8]", "[4,4]");
        // A training checkpoint is recognized by its marker and redirected.
        ImportRefused([new SafeTensor("__shorokoo_checkpoint__", TensorData([2L], 1L, 0L), "I64", [2L])],
            "training checkpoint");

        // Two parameters onto one source name is ambiguous — refused before any tensor lookup.
        SimplePatternScheme[] colliding =
        [
            new SimplePatternScheme("TrainableParam#0.InitSimple#{p}", "fc.same"),
        ];
        var collidingScheme = new SimplePatternNamingScheme(
            colliding, arch.GetShorokooIdNamingScheme(), ModuleParamSetNamingScheme.PyTorchFrameworkId);
        var exAmbiguous = Assert.Throws<InvalidDataException>(
            () => Persistence.ImportSafeTensors(arch, path, collidingScheme));
        Assert.Contains(FcWeightsId, exAmbiguous.Message);
        Assert.Contains(FcBiasId, exAmbiguous.Message);
        Assert.Contains("fc.same", exAmbiguous.Message);

        // A partial scheme names the uncovered parameter, even when every tensor maps cleanly.
        SimplePatternScheme[] partial = [new SimplePatternScheme(FcWeightsId, "fc.weight")];
        var partialScheme = new SimplePatternNamingScheme(
            partial, arch.GetShorokooIdNamingScheme(), ModuleParamSetNamingScheme.PyTorchFrameworkId);
        var weightsOnly = good.Single(t => t.Name == FcWeightsId);
        SafeTensorLoader.SaveSafeTensors(badPath,
            [new SafeTensor("fc.weight", weightsOnly.Data, weightsOnly.DataType, weightsOnly.Shape)]);
        var exUncovered = Assert.Throws<InvalidDataException>(
            () => Persistence.ImportSafeTensors(arch, badPath, partialScheme));
        Assert.Contains(FcBiasId, exUncovered.Message);
        Assert.Contains("naming scheme", exUncovered.Message);

        // Kind gates: import takes a concrete architecture, export a concrete model.
        Assert.Contains("concrete-architecture", Assert.Throws<InvalidOperationException>(
            () => Persistence.ImportSafeTensors(FCLayer.ComputationGraph, path)).Message);
        Assert.Contains("concrete-architecture", Assert.Throws<InvalidOperationException>(
            () => Persistence.ImportSafeTensors(model, path)).Message);
        Assert.Contains("concrete-model", Assert.Throws<InvalidOperationException>(
            () => Persistence.ExportSafeTensors(arch, badPath)).Message);

        // Export-side gaps: unnamed parameter, name collision, and a ModelId-keyed scheme (which
        // cannot translate the canonical id strings a bound model carries). None commits.
        var exportPath = P("exchange_export_fail.safetensors");
        Assert.Contains(FcBiasId, Assert.Throws<InvalidOperationException>(
            () => Persistence.ExportSafeTensors(model, exportPath, partialScheme)).Message);
        var exCollision = Assert.Throws<InvalidOperationException>(
            () => Persistence.ExportSafeTensors(model, exportPath, collidingScheme));
        Assert.Contains(FcWeightsId, exCollision.Message);
        Assert.Contains(FcBiasId, exCollision.Message);
        Assert.Contains("fc.same", exCollision.Message);
        Assert.Throws<NotSupportedException>(() => Persistence.ExportSafeTensors(model, exportPath,
            new ModelIdNamingScheme([new ModelIdFormat(format: "x")],
                ModuleParamSetNamingScheme.PyTorchFrameworkId)));
        Assert.False(File.Exists(exportPath));
    }

    // ──────────────────────────────────────────────────────────────────────
    // ONNX exchange boundary: ExportOnnx writes a standard vanilla .onnx;
    // ImportOnnx turns a foreign vanilla .onnx into a native runnable graph.
    // ──────────────────────────────────────────────────────────────────────

    private static ValueInfoProto OnnxFloatVec(string name, long len)
    {
        var shape = new TensorShapeProto();
        shape.Dims.Add(new TensorShapeProto.Dimension { DimValue = len });
        return new ValueInfoProto
        {
            Name = name,
            Type = new TypeProto { TensorType = new TypeProto.Tensor { ElemType = 1, Shape = shape } },
        };
    }

    private static byte[] OnnxFloatBytes(params float[] vals)
    {
        var bytes = new byte[vals.Length * sizeof(float)];
        System.Buffer.BlockCopy(vals, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    /// <summary>A minimal third-party-style vanilla ONNX model: y = x + w, with w a plain named
    /// initializer and no Shorokoo metadata.</summary>
    private static ModelProto BuildForeignAddModel(string initName, float[] wVals)
    {
        var g = new GraphProto { Name = "foreign" };
        g.Inputs.Add(OnnxFloatVec("x", wVals.Length));
        g.Initializers.Add(new TensorProto
        {
            Name = initName, data_type = 1, Dims = [wVals.Length], RawData = OnnxFloatBytes(wVals),
        });
        var add = new NodeProto { OpType = "Add", Name = "add0" };
        add.Inputs.AddRange(["x", initName]);
        add.Outputs.Add("y");
        g.Nodes.Add(add);
        g.Outputs.Add(OnnxFloatVec("y", wVals.Length));
        var model = new ModelProto { IrVersion = 10, Graph = g };
        model.OpsetImports.Add(new OperatorSetIdProto { Domain = "", Version = 21 });
        return model;
    }

    /// <summary>A vanilla ONNX model whose only node is an op the reader cannot ingest.</summary>
    private static ModelProto BuildUnsupportedOpModel()
    {
        var g = new GraphProto { Name = "bogus" };
        g.Inputs.Add(OnnxFloatVec("x", 4));
        var node = new NodeProto { OpType = "TotallyMadeUpOp", Name = "n0" };
        node.Inputs.Add("x");
        node.Outputs.Add("y");
        g.Nodes.Add(node);
        g.Outputs.Add(OnnxFloatVec("y", 4));
        var model = new ModelProto { IrVersion = 10, Graph = g };
        model.OpsetImports.Add(new OperatorSetIdProto { Domain = "", Version = 21 });
        return model;
    }

    private static string WriteOnnx(string path, ModelProto model)
    {
        using var fs = File.Create(path);
        ProtoBuf.Serializer.Serialize(fs, model);
        return path;
    }

    private static float[] RunFloatVecModel(ComputationGraph graph, params float[] x)
    {
        IData[] inputs = [TensorData(DType.Float32, [(long)x.Length], x.Cast<object>().ToArray())];
        return ((TensorData<float32>)ComputeContext.Default.Execute(graph, inputs)[0].ToTensorData())
            .AccessMemory().ToArray();
    }

    /// <summary>
    /// The 2 GB protobuf ceiling is the whole reason the external-data layout exists, so the
    /// graph-level export must refuse an over-ceiling model with the framework's own XD007 —
    /// naming the remedy — rather than let protobuf fail on its own terms. Driven with a tiny
    /// injected ceiling rather than by allocating gigabytes.
    /// </summary>
    [Fact]
    public void TestExportOnnxRefusesAModelOverTheProtobufCeilingNamingTheExternalDataRemedy()
    {
        var (model, _, _) = BuildSkptModel();
        var ex = Assert.Throws<ModelException>(
            () => Persistence.ExportOnnx(model, P("over-ceiling.onnx"), OpSetVersion.OPS_21, 8));
        Assert.Equal(ErrorCodes.XD007, ex.ErrorCode);
        Assert.Contains("SaveWithExternalData", ex.Message);
        Assert.False(File.Exists(P("over-ceiling.onnx")));
    }

    [Fact]
    public void TestOnnxExportImportRoundTripThirdPartyModelsAndCheckpointLanding()
    {
        var (model, numOut, input) = BuildSkptModel();
        var onnxPath = P("roundtrip.onnx");
        var direct = ExecuteToBytes(model, numOut, input);

        Persistence.ExportOnnx(model, onnxPath);
        Assert.True(File.Exists(onnxPath));
        var imported = Persistence.ImportOnnx(onnxPath);
        Assert.Equal(GraphKind.ConcreteModel, imported.Kind);
        Assert.Equal(direct, ExecuteToBytes(imported, numOut, input));

        // The same file loads through the low-level importer — it is a plain vanilla .onnx.
        Assert.Equal(direct, ExecuteToBytes(OnnxModelImporter.FromOnnxModel(onnxPath), numOut, input));

        // A third-party y = x + w model: the foreign initializer's ONNX name becomes the param id.
        var foreignPath = P("foreign.onnx");
        float[] w = [10f, 20f, 30f, 40f];
        WriteOnnx(foreignPath, BuildForeignAddModel("w", w));
        var foreign = Persistence.ImportOnnx(foreignPath);
        Assert.Equal([11f, 22f, 33f, 44f], RunFloatVecModel(foreign, 1f, 2f, 3f, 4f));
        Assert.Equal(["[1]:TrainableParam#0.w#0"], WeightBytesByParam(foreign).Keys.ToArray());

        SimplePatternScheme[] patterns = [new SimplePatternScheme("w", "TrainableParam#0.MyWeight#0")];
        var scheme = new SimplePatternNamingScheme(
            patterns, new ModelIdNamingScheme([], ModuleParamSetNamingScheme.PyTorchFrameworkId),
            ModuleParamSetNamingScheme.PyTorchFrameworkId);
        var renamed = Persistence.ImportOnnx(foreignPath, scheme);
        Assert.Equal(["[1]:TrainableParam#0.MyWeight#0"], WeightBytesByParam(renamed).Keys.ToArray());
        Assert.Equal([11f, 22f, 33f, 44f], RunFloatVecModel(renamed, 1f, 2f, 3f, 4f));

        // The one-call native landing.
        var skptPath = P("landing.skpt");
        var landed = Persistence.ImportOnnxToCheckpoint(foreignPath, skptPath);
        Assert.Equal(GraphKind.ConcreteModel, landed.Kind);
        var expected = RunFloatVecModel(Persistence.ImportOnnx(foreignPath), 1f, 2f, 3f, 4f);
        Assert.Equal(expected, RunFloatVecModel(landed, 1f, 2f, 3f, 4f));
        var reloaded = Persistence.Load(skptPath);
        Assert.Equal(GraphKind.ConcreteModel, reloaded.Kind);
        Assert.Equal(WeightBytesByParam(landed), WeightBytesByParam(reloaded));
        Assert.Equal(expected, RunFloatVecModel(reloaded, 1f, 2f, 3f, 4f));

        // A failed import lands nothing: the committed checkpoint is untouched.
        var committed = File.ReadAllBytes(skptPath);
        var badPath = WriteOnnx(P("landing_bad.onnx"), BuildUnsupportedOpModel());
        Assert.Throws<InvalidDataException>(() => Persistence.ImportOnnxToCheckpoint(badPath, skptPath));
        Assert.Equal(committed, File.ReadAllBytes(skptPath));

        // Composition with ONNX external data: w's bytes live in a .data side file.
        var inlinePath = P("xd_inline.onnx");
        var extPath = P("xd_external.onnx");
        var xdSkptPath = P("xd.skpt");
        float[] xw = [0.5f, -1.5f, 2.5f, 3.5f];
        WriteOnnx(inlinePath, BuildForeignAddModel("w", xw));
        var extModel = BuildForeignAddModel("w", xw);
        File.WriteAllBytes(P("xd_external.onnx.data"), OnnxFloatBytes(xw));
        var wInit = extModel.Graph.Initializers.Single();
        wInit.RawData = null!;
        wInit.data_location = TensorProto.DataLocation.External;
        wInit.ExternalDatas.Add(new StringStringEntryProto { Key = "location", Value = "xd_external.onnx.data" });
        wInit.ExternalDatas.Add(new StringStringEntryProto { Key = "offset", Value = "0" });
        wInit.ExternalDatas.Add(new StringStringEntryProto
        { Key = "length", Value = (xw.Length * sizeof(float)).ToString() });
        WriteOnnx(extPath, extModel);

        var inlineOut = RunFloatVecModel(Persistence.ImportOnnx(inlinePath), 1f, 1f, 1f, 1f);
        Assert.Equal(inlineOut, RunFloatVecModel(Persistence.ImportOnnx(extPath), 1f, 1f, 1f, 1f));
        var xdLanded = Persistence.ImportOnnxToCheckpoint(extPath, xdSkptPath);
        Assert.Equal(inlineOut, RunFloatVecModel(Persistence.Load(xdSkptPath), 1f, 1f, 1f, 1f));
        Assert.Equal(GraphKind.ConcreteModel, xdLanded.Kind);
    }

    [Fact]
    public void TestImportOnnxFailsLoud()
    {
        var badOpPath = WriteOnnx(P("badop.onnx"), BuildUnsupportedOpModel());
        var exOp = Assert.Throws<InvalidDataException>(() => Persistence.ImportOnnx(badOpPath));
        Assert.Contains("TotallyMadeUpOp", exOp.Message);
        Assert.Contains(badOpPath, exOp.Message);

        var garbagePath = P("garbage.onnx");
        File.WriteAllBytes(garbagePath, Enumerable.Range(0, 256).Select(i => (byte)(i * 7 + 1)).ToArray());
        Assert.Contains(garbagePath,
            Assert.Throws<InvalidDataException>(() => Persistence.ImportOnnx(garbagePath)).Message);

        var truncPath = P("trunc.onnx");
        var whole = File.ReadAllBytes(WriteOnnx(truncPath, BuildForeignAddModel("w", [1f, 2f, 3f, 4f])));
        File.WriteAllBytes(truncPath, whole[..(whole.Length / 2)]);
        Assert.Contains(truncPath,
            Assert.Throws<InvalidDataException>(() => Persistence.ImportOnnx(truncPath)).Message);
    }
}

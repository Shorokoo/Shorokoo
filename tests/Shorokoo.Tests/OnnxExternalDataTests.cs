using System.IO;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage for the standard ONNX external-data mechanism (issue #38): initializer
/// bytes stored in a side file, referenced from <c>TensorProto.external_data</c>
/// (location/offset/length) with <c>data_location=EXTERNAL</c>.
///
/// Read side: external tensors import with values identical to their inline
/// equivalents (including offset/length slicing into a shared side file and non-float
/// dtypes), and every failure mode — missing side file, path traversal, bad
/// offset/length, length/shape mismatch, directoryless stream — fails loudly naming
/// the tensor and the problem.
///
/// Write side: <c>OnnxModelExporter.SaveWithExternalData</c> moves initializers at or
/// above the size threshold into a single aligned <c>.onnx.data</c> side file with a
/// deterministic layout, keeps small initializers inline, leaves the caller's proto
/// unmodified, round-trips bit-exactly through the importer, and produces a pair that
/// stock onnxruntime loads; below the threshold the output is identical to the
/// self-contained <c>Save</c>.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class OnnxExternalDataTests
{
    private const int FloatElem = 1;    // TensorProto.DataType.FLOAT
    private const int Int64Elem = 7;    // INT64
    private const int Float16Elem = 10; // FLOAT16

    // ------------------------- proto-building helpers -------------------------

    private static ValueInfoProto TensorInfo(string name, int elemType, params long[] dims)
    {
        var shape = new TensorShapeProto();
        foreach (var d in dims)
            shape.Dims.Add(new TensorShapeProto.Dimension { DimValue = d });
        return new ValueInfoProto
        {
            Name = name,
            Type = new TypeProto
            {
                TensorType = new TypeProto.Tensor { ElemType = elemType, Shape = shape },
            },
        };
    }

    private static TensorProto Init(string name, int elemType, long[] dims, byte[] raw)
        => new TensorProto { Name = name, data_type = elemType, Dims = dims, RawData = raw };

    private static NodeProto Node(string opType, string name, string[] inputs, string[] outputs)
    {
        var n = new NodeProto { OpType = opType, Name = name };
        n.Inputs.AddRange(inputs);
        n.Outputs.AddRange(outputs);
        return n;
    }

    private static ModelProto WrapModel(GraphProto graph)
    {
        var model = new ModelProto { IrVersion = 10, Graph = graph };
        model.OpsetImports.Add(new OperatorSetIdProto { Domain = "", Version = 21 });
        return model;
    }

    /// <summary>x:float[4] plus initializers w and (optionally) b, chained Adds into y.</summary>
    private static ModelProto BuildAddModel(TensorProto w, TensorProto? b = null)
    {
        var g = new GraphProto { Name = "addmodel" };
        g.Inputs.Add(TensorInfo("x", FloatElem, 4));
        g.Initializers.Add(w);
        if (b is null)
        {
            g.Nodes.Add(Node("Add", "add0", ["x", "w"], ["y"]));
        }
        else
        {
            g.Initializers.Add(b);
            g.Nodes.Add(Node("Add", "add0", ["x", "w"], ["t"]));
            g.Nodes.Add(Node("Add", "add1", ["t", "b"], ["y"]));
        }
        g.Outputs.Add(TensorInfo("y", FloatElem, 4));
        return WrapModel(g);
    }

    private static byte[] FloatBytes(params float[] vals)
    {
        var bytes = new byte[vals.Length * sizeof(float)];
        System.Buffer.BlockCopy(vals, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static byte[] LongBytes(params long[] vals)
    {
        var bytes = new byte[vals.Length * sizeof(long)];
        System.Buffer.BlockCopy(vals, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static void MarkExternal(TensorProto t, string location, long? offset = null, long? length = null)
    {
        t.RawData = null!;
        t.data_location = TensorProto.DataLocation.External;
        t.ExternalDatas.Add(new StringStringEntryProto
        { Key = "location", Value = location });
        if (offset is long o)
            t.ExternalDatas.Add(new StringStringEntryProto
            { Key = "offset", Value = o.ToString(System.Globalization.CultureInfo.InvariantCulture) });
        if (length is long l)
            t.ExternalDatas.Add(new StringStringEntryProto
            { Key = "length", Value = l.ToString(System.Globalization.CultureInfo.InvariantCulture) });
    }

    private static string WriteModel(string dir, string fileName, ModelProto model)
    {
        var path = Path.Combine(dir, fileName);
        using var fs = File.Create(path);
        ProtoBuf.Serializer.Serialize(fs, model);
        return path;
    }

    private static void WithTempDir(Action<string> body)
    {
        var dir = Path.Combine(Path.GetTempPath(), "shrk-xd-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try { body(dir); }
        finally { Directory.Delete(dir, recursive: true); }
    }

    private static float[] RunAddModel(ComputationGraph fast)
    {
        IData[] inputs = [TensorData(DType.Float32, [4L], 1f, 2f, 3f, 4f)];
        var results = ComputeContext.Default.Execute(fast, inputs);
        return ((TensorData<float32>)results[0].ToTensorData()).AccessMemory().ToArray();
    }

    // ------------------------- read path -------------------------

    [Fact]
    public void TestExternalInitializerImportsIdenticalToInline()
    {
        float[] wVals = [0.1f, -2.5f, 3.25f, 1e-7f];
        WithTempDir(dir =>
        {
            var inlinePath = WriteModel(dir, "inline.onnx", BuildAddModel(
                Init("w", FloatElem, [4], FloatBytes(wVals))));

            File.WriteAllBytes(Path.Combine(dir, "weights.bin"), FloatBytes(wVals));
            var wExt = Init("w", FloatElem, [4], null!);
            MarkExternal(wExt, "weights.bin", offset: 0, length: 16);
            var externalPath = WriteModel(dir, "external.onnx", BuildAddModel(wExt));

            var inlineOut = RunAddModel(OnnxModelImporter.FromOnnxModel(inlinePath));
            var externalOut = RunAddModel(OnnxModelImporter.FromOnnxModel(externalPath));
            Assert.Equal(inlineOut, externalOut);
        });
    }

    [Fact]
    public void TestExternalDataSharedFileOffsetSlicing()
    {
        float[] wVals = [1f, 2f, 3f, 4f];
        float[] bVals = [-0.5f, 0.5f, -1.5f, 2.5f];
        WithTempDir(dir =>
        {
            // Shared side file: 8 junk bytes, w's 16 bytes, 4 junk bytes, b's 16 bytes.
            // b carries no 'length' entry — the byte count implied by shape/dtype applies.
            var shared = new byte[8 + 16 + 4 + 16];
            System.Buffer.BlockCopy(FloatBytes(wVals), 0, shared, 8, 16);
            System.Buffer.BlockCopy(FloatBytes(bVals), 0, shared, 28, 16);
            File.WriteAllBytes(Path.Combine(dir, "shared.bin"), shared);

            var wExt = Init("w", FloatElem, [4], null!);
            MarkExternal(wExt, "shared.bin", offset: 8, length: 16);
            var bExt = Init("b", FloatElem, [4], null!);
            MarkExternal(bExt, "shared.bin", offset: 28);
            var externalPath = WriteModel(dir, "external.onnx", BuildAddModel(wExt, bExt));

            var inlinePath = WriteModel(dir, "inline.onnx", BuildAddModel(
                Init("w", FloatElem, [4], FloatBytes(wVals)),
                Init("b", FloatElem, [4], FloatBytes(bVals))));

            var inlineOut = RunAddModel(OnnxModelImporter.FromOnnxModel(inlinePath));
            var externalOut = RunAddModel(OnnxModelImporter.FromOnnxModel(externalPath));
            Assert.Equal(inlineOut, externalOut);
        });
    }

    [Fact]
    public void TestExternalDataNonFloatDtypesImport()
    {
        // The side file carries raw little-endian bytes for any element type; cover a
        // 64-bit integer and a sub-word float (float16, stored as ushort bit patterns).
        long[] iVals = [long.MaxValue - 1, -42];
        ushort[] hBits = [0x3C00 /* 1.0 */, 0xC000 /* -2.0 */];
        var hBytes = new byte[4];
        System.Buffer.BlockCopy(hBits, 0, hBytes, 0, 4);

        WithTempDir(dir =>
        {
            var side = new byte[16 + 4];
            System.Buffer.BlockCopy(LongBytes(iVals), 0, side, 0, 16);
            System.Buffer.BlockCopy(hBytes, 0, side, 16, 4);
            File.WriteAllBytes(Path.Combine(dir, "mixed.bin"), side);

            var g = new GraphProto { Name = "dtypes" };
            var wi = Init("wi", Int64Elem, [2], null!);
            MarkExternal(wi, "mixed.bin", offset: 0, length: 16);
            var wh = Init("wh", Float16Elem, [2], null!);
            MarkExternal(wh, "mixed.bin", offset: 16, length: 4);
            g.Initializers.Add(wi);
            g.Initializers.Add(wh);
            g.Nodes.Add(Node("Identity", "id0", ["wi"], ["y1"]));
            g.Nodes.Add(Node("Identity", "id1", ["wh"], ["y2"]));
            g.Outputs.Add(TensorInfo("y1", Int64Elem, 2));
            g.Outputs.Add(TensorInfo("y2", Float16Elem, 2));
            var path = WriteModel(dir, "dtypes.onnx", WrapModel(g));

            var fast = OnnxModelImporter.FromOnnxModel(path);
            var results = ComputeContext.Default.Execute(fast);
            Assert.Equal(LongBytes(iVals), results[0].ToTensorData().AccessRawMemory().ToArray());
            Assert.Equal(hBytes, results[1].ToTensorData().AccessRawMemory().ToArray());
        });
    }

    [Fact]
    public void TestExternalDataMissingFileFails()
    {
        WithTempDir(dir =>
        {
            var wExt = Init("w", FloatElem, [4], null!);
            MarkExternal(wExt, "missing.bin", offset: 0, length: 16);
            var path = WriteModel(dir, "external.onnx", BuildAddModel(wExt));

            var ex = Assert.Throws<ModelException>(() => OnnxModelImporter.FromOnnxModel(path));
            Assert.Equal(ErrorCodes.XD004, ex.ErrorCode);
            Assert.Contains("'w'", ex.Message);
            Assert.Contains("missing.bin", ex.Message);
        });
    }

    [Fact]
    public void TestExternalDataPathTraversalFails()
    {
        WithTempDir(dir =>
        {
            var wExt = Init("w", FloatElem, [4], null!);
            MarkExternal(wExt, "../escape.bin", offset: 0, length: 16);
            var path = WriteModel(dir, "traversal.onnx", BuildAddModel(wExt));

            var ex = Assert.Throws<ModelException>(() => OnnxModelImporter.FromOnnxModel(path));
            Assert.Equal(ErrorCodes.XD003, ex.ErrorCode);
            Assert.Contains("'w'", ex.Message);

            var wAbs = Init("w", FloatElem, [4], null!);
            MarkExternal(wAbs, Path.Combine(Path.GetTempPath(), "abs.bin"), offset: 0, length: 16);
            var absPath = WriteModel(dir, "absolute.onnx", BuildAddModel(wAbs));

            var exAbs = Assert.Throws<ModelException>(() => OnnxModelImporter.FromOnnxModel(absPath));
            Assert.Equal(ErrorCodes.XD003, exAbs.ErrorCode);
        });
    }

    [Fact]
    public void TestExternalDataBadOffsetLengthFails()
    {
        WithTempDir(dir =>
        {
            File.WriteAllBytes(Path.Combine(dir, "short.bin"), FloatBytes(1f, 2f, 3f, 4f));

            // Range past the end of the side file.
            var wRange = Init("w", FloatElem, [4], null!);
            MarkExternal(wRange, "short.bin", offset: 8, length: 16);
            var rangePath = WriteModel(dir, "range.onnx", BuildAddModel(wRange));
            var exRange = Assert.Throws<ModelException>(() => OnnxModelImporter.FromOnnxModel(rangePath));
            Assert.Equal(ErrorCodes.XD005, exRange.ErrorCode);
            Assert.Contains("'w'", exRange.Message);
            Assert.Contains("short.bin", exRange.Message);

            // Unparsable offset.
            var wParse = Init("w", FloatElem, [4], null!);
            wParse.data_location = TensorProto.DataLocation.External;
            wParse.ExternalDatas.Add(new StringStringEntryProto { Key = "location", Value = "short.bin" });
            wParse.ExternalDatas.Add(new StringStringEntryProto { Key = "offset", Value = "not-a-number" });
            var parsePath = WriteModel(dir, "parse.onnx", BuildAddModel(wParse));
            var exParse = Assert.Throws<ModelException>(() => OnnxModelImporter.FromOnnxModel(parsePath));
            Assert.Equal(ErrorCodes.XD005, exParse.ErrorCode);

            // Near-long.MaxValue offset: offset + length would overflow a naive
            // additive range check; must still fail as a loud out-of-range error,
            // not an opaque allocation/end-of-stream failure.
            var wHuge = Init("w", FloatElem, [4], null!);
            MarkExternal(wHuge, "short.bin", offset: long.MaxValue - 8, length: 16);
            var hugePath = WriteModel(dir, "huge.onnx", BuildAddModel(wHuge));
            var exHuge = Assert.Throws<ModelException>(() => OnnxModelImporter.FromOnnxModel(hugePath));
            Assert.Equal(ErrorCodes.XD005, exHuge.ErrorCode);
            Assert.Contains("'w'", exHuge.Message);
        });
    }

    [Fact]
    public void TestExternalDataLengthShapeMismatchFails()
    {
        WithTempDir(dir =>
        {
            File.WriteAllBytes(Path.Combine(dir, "w.bin"), FloatBytes(1f, 2f, 3f, 4f));
            // 'length' says 12 bytes but float[4] implies 16.
            var wExt = Init("w", FloatElem, [4], null!);
            MarkExternal(wExt, "w.bin", offset: 0, length: 12);
            var path = WriteModel(dir, "mismatch.onnx", BuildAddModel(wExt));

            var ex = Assert.Throws<ModelException>(() => OnnxModelImporter.FromOnnxModel(path));
            Assert.Equal(ErrorCodes.XD006, ex.ErrorCode);
            Assert.Contains("'w'", ex.Message);
            Assert.Contains("16", ex.Message);
        });
    }

    [Fact]
    public void TestExternalDataFromBytesRequiresDirectory()
    {
        WithTempDir(dir =>
        {
            File.WriteAllBytes(Path.Combine(dir, "w.bin"), FloatBytes(1f, 2f, 3f, 4f));
            var wExt = Init("w", FloatElem, [4], null!);
            MarkExternal(wExt, "w.bin", offset: 0, length: 16);
            using var ms = new MemoryStream();
            ProtoBuf.Serializer.Serialize(ms, BuildAddModel(wExt));
            var bytes = ms.ToArray();

            // No base directory: loud failure, no silent zero-fill.
            var ex = Assert.Throws<ModelException>(() => OnnxModelImporter.FromOnnxModel(bytes));
            Assert.Equal(ErrorCodes.XD001, ex.ErrorCode);
            Assert.Contains("'w'", ex.Message);

            // Same bytes with the directory supplied: loads and executes.
            var fast = OnnxModelImporter.FromOnnxModel(bytes, externalDataDirectory: dir);
            float[] expected = [2f, 4f, 6f, 8f];
            Assert.Equal(expected, RunAddModel(fast));
        });
    }

    // ------------------------- write path -------------------------

    [Fact]
    public void TestSaveWithExternalDataRoundTripsBitExact()
    {
        var numOut = TensorData(DType.Int64, [], 4L);
        var input = TensorDataWithSmallVals(DType.Float32, [4L, 4L]);
        var g = FCLayer.ComputationGraph; // weights [4,4] (64 B) + bias [4] (16 B)
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([numOut, input])).ToConcreteModel();
        var proto = FastOnnxModelBuilder.BuildOnnxModel(concrete);
        var direct = ComputeContext.Default.Execute(concrete, numOut, input)[0]
            .ToTensorData().AccessRawMemory().ToArray();

        WithTempDir(dir =>
        {
            var path = Path.Combine(dir, "model.onnx");
            OnnxModelExporter.SaveWithExternalData(proto, path,
                new OnnxExternalDataOptions { SizeThreshold = 32 });

            // The 64-byte weights went external into model.onnx.data; the 16-byte bias
            // stayed inline.
            Assert.True(File.Exists(path + ".data"));
            using (var fs = File.OpenRead(path))
            {
                var saved = ProtoBuf.Serializer.Deserialize<ModelProto>(fs);
                var external = saved.Graph.Initializers
                    .Where(t => t.data_location == TensorProto.DataLocation.External).ToArray();
                var ext = Assert.Single(external);
                Assert.Equal("model.onnx.data",
                    ext.ExternalDatas.Single(e => e.Key == "location").Value);
                Assert.Equal("64", ext.ExternalDatas.Single(e => e.Key == "length").Value);
            }

            // The caller's proto was restored to its self-contained form.
            Assert.All(proto.Graph.Initializers, t =>
            {
                Assert.NotEqual(TensorProto.DataLocation.External, t.data_location);
                Assert.Empty(t.ExternalDatas);
            });

            // Export-with-external-data → import round-trips bit-exactly.
            var reimported = OnnxModelImporter.FromOnnxModel(path);
            var roundtrip = ComputeContext.Default.Execute(reimported, numOut, input)[0]
                .ToTensorData().AccessRawMemory().ToArray();
            Assert.Equal(direct, roundtrip);
        });
    }

    [Fact]
    public void TestSaveWithExternalDataAlignedAndDeterministic()
    {
        float[] wVals = [1f, 2f, 3f, 4f];
        float[] bVals = [5f, 6f, 7f, 8f];
        var proto = BuildAddModel(
            Init("w", FloatElem, [4], FloatBytes(wVals)),
            Init("b", FloatElem, [4], FloatBytes(bVals)));

        WithTempDir(dir =>
        {
            var path1 = Path.Combine(dir, "one.onnx");
            var path2 = Path.Combine(dir, "two.onnx");
            var opts = new OnnxExternalDataOptions { SizeThreshold = 0 };
            OnnxModelExporter.SaveWithExternalData(proto, path1, opts);
            OnnxModelExporter.SaveWithExternalData(proto, path2, opts);

            // Deterministic layout: byte-identical pairs across runs (the side files
            // trivially so, the .onnx files modulo their differing 'location' file name —
            // so compare a re-save under the SAME name instead).
            Assert.Equal(File.ReadAllBytes(path1 + ".data"), File.ReadAllBytes(path2 + ".data"));
            var firstBytes = File.ReadAllBytes(path1);
            OnnxModelExporter.SaveWithExternalData(proto, path1, opts);
            Assert.Equal(firstBytes, File.ReadAllBytes(path1));

            // Each tensor's data starts on an alignment boundary: w at 0, b at 4096.
            using (var fs = File.OpenRead(path1))
            {
                var saved = ProtoBuf.Serializer.Deserialize<ModelProto>(fs);
                var offsets = saved.Graph.Initializers
                    .ToDictionary(t => t.Name, t => t.ExternalDatas.Single(e => e.Key == "offset").Value);
                Assert.Equal("0", offsets["w"]);
                Assert.Equal("4096", offsets["b"]);
            }
            Assert.Equal(4096 + 16, new FileInfo(path1 + ".data").Length);

            // The externalized pair still evaluates like the inline proto.
            var inlinePath = WriteModel(dir, "inline.onnx", proto);
            var inlineOut = RunAddModel(OnnxModelImporter.FromOnnxModel(inlinePath));
            var externalOut = RunAddModel(OnnxModelImporter.FromOnnxModel(path1));
            Assert.Equal(inlineOut, externalOut);
        });
    }

    [Fact]
    public void TestSaveWithExternalDataBelowThresholdMatchesInlineSave()
    {
        var proto = BuildAddModel(Init("w", FloatElem, [4], FloatBytes(1f, 2f, 3f, 4f)));
        WithTempDir(dir =>
        {
            var inlinePath = Path.Combine(dir, "inline.onnx");
            var extPath = Path.Combine(dir, "ext.onnx");
            OnnxModelExporter.Save(proto, inlinePath);
            // First an external save (threshold 0) so a side file exists, then a
            // below-threshold re-save of the same path: the now-stale side file must
            // be removed, and the .onnx must be byte-equal to the inline save.
            OnnxModelExporter.SaveWithExternalData(proto, extPath,
                new OnnxExternalDataOptions { SizeThreshold = 0 });
            Assert.True(File.Exists(extPath + ".data"));
            OnnxModelExporter.SaveWithExternalData(proto, extPath,
                new OnnxExternalDataOptions { SizeThreshold = 1024 });

            Assert.False(File.Exists(extPath + ".data"));
            Assert.Equal(File.ReadAllBytes(inlinePath), File.ReadAllBytes(extPath));
        });
    }

    [Fact]
    public void TestSelfContainedSaveOverLimitThrows()
    {
        // The real ceiling is protobuf's 2 GB message limit; drive the same check with a
        // tiny injected ceiling rather than allocating gigabytes.
        var proto = BuildAddModel(Init("w", FloatElem, [4], FloatBytes(1f, 2f, 3f, 4f)));
        WithTempDir(dir =>
        {
            var path = Path.Combine(dir, "big.onnx");
            var ex = Assert.Throws<ModelException>(() => OnnxModelExporter.Save(proto, path, maxTensorBytes: 8));
            Assert.Equal(ErrorCodes.XD007, ex.ErrorCode);
            Assert.Contains("SaveWithExternalData", ex.Message);

            // Payload stored in a typed data field (int64_data — like the RNG key
            // vector initializer) instead of raw_data counts against the ceiling too.
            var typed = new TensorProto
            { Name = "keys", data_type = Int64Elem, Dims = [4], Int64Datas = [1L, 2L, 3L, 4L] };
            var g = new GraphProto { Name = "typed" };
            g.Initializers.Add(typed);
            g.Nodes.Add(Node("Identity", "id0", ["keys"], ["y"]));
            g.Outputs.Add(TensorInfo("y", Int64Elem, 4));
            var exTyped = Assert.Throws<ModelException>(() => OnnxModelExporter.Save(
                WrapModel(g), Path.Combine(dir, "typed.onnx"), maxTensorBytes: 16));
            Assert.Equal(ErrorCodes.XD007, exTyped.ErrorCode);
        });
    }

    [Fact]
    public void TestExportedPairLoadsInStockOnnxRuntime()
    {
        var proto = BuildAddModel(Init("w", FloatElem, [4], FloatBytes(10f, 20f, 30f, 40f)));
        WithTempDir(dir =>
        {
            var path = Path.Combine(dir, "model.onnx");
            OnnxModelExporter.SaveWithExternalData(proto, path,
                new OnnxExternalDataOptions { SizeThreshold = 0 });
            Assert.True(File.Exists(path + ".data"));

            // Stock onnxruntime resolves the side file itself, from the model's path.
            using var session = new Microsoft.ML.OnnxRuntime.InferenceSession(path);
            float[] xVals = [1f, 2f, 3f, 4f];
            var x = new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(xVals, [4]);
            using var results = session.Run(
                [Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor("x", x)]);
            float[] expected = [11f, 22f, 33f, 44f];
            Assert.Equal(expected, results.First().AsEnumerable<float>().ToArray());
        });
    }

    /// <summary>
    /// Issue #54: SaveWithExternalData applies to concrete models only — it
    /// externalizes top-level graph initializers, which is complete exactly when
    /// every weight lives there. A module-stage proto (internal dialect) is
    /// refused up front with XD008 naming the actual vs required kind.
    /// </summary>
    [Fact]
    public void TestSaveWithExternalDataRequiresConcreteModel()
    {
        var moduleProto = FastOnnxModelBuilder.BuildInternalOnnxModel(
            Shorokoo.Tests.Modules.ScalarMultiplyModel.ComputationGraph.Internal);

        var path = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".onnx");
        var ex = Assert.Throws<ModelException>(
            () => OnnxModelExporter.SaveWithExternalData(moduleProto, path));
        Assert.Contains("XD008", ex.Message);
        Assert.Contains("'concrete-model'", ex.Message);
        Assert.Contains("'module'", ex.Message);
        Assert.False(File.Exists(path));
        Assert.False(File.Exists(path + ".data"));
    }

}

using System.IO;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Core.Inference;
using Shorokoo.Runtime;
using static Shorokoo.Tests.OnnxProtoBuilders;

namespace Shorokoo.Tests;

internal static class OnnxProtoBuilders
{
    internal static ValueInfoProto TensorInfo(string name, int elemType, params long[] dims)
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

    internal static TensorProto Init(string name, int elemType, long[] dims, byte[] raw)
        => new TensorProto { Name = name, data_type = elemType, Dims = dims, RawData = raw };

    internal static ModelProto WrapModel(GraphProto graph)
    {
        var model = new ModelProto { IrVersion = 10, Graph = graph };
        model.OpsetImports.Add(new OperatorSetIdProto { Domain = "", Version = 21 });
        return model;
    }

    internal static InternalComputationGraph Import(ModelProto model)
    {
        using var ms = new MemoryStream();
        ProtoBuf.Serializer.Serialize(ms, model);
        return OnnxModelImporter.FromOnnxModelToInternalGraph(ms.ToArray());
    }
}

/// <summary>
/// The standard ONNX external-data mechanism (issue #38): initializer bytes stored in a side
/// file, referenced from <c>TensorProto.external_data</c> (location/offset/length) with
/// <c>data_location=EXTERNAL</c> — both the import side and
/// <c>OnnxModelExporter.SaveWithExternalData</c>.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class OnnxExternalDataTests
{
    private const int FloatElem = 1;    // TensorProto.DataType.FLOAT
    private const int Int64Elem = 7;    // INT64
    private const int Float16Elem = 10; // FLOAT16

    private static NodeProto Node(string opType, string name, string[] inputs, string[] outputs)
    {
        var n = new NodeProto { OpType = opType, Name = name };
        n.Inputs.AddRange(inputs);
        n.Outputs.AddRange(outputs);
        return n;
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

    private static TensorProto ExternalW(string location, long? offset = null, long? length = null)
    {
        var w = Init("w", FloatElem, [4], null!);
        MarkExternal(w, location, offset, length);
        return w;
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

    [Fact]
    public void TestExternalDataImportsIdenticalToInlineIncludingSharedFileSlicingAndNonFloatDtypes()
    {
        WithTempDir(dir =>
        {
            float[] wVals = [0.1f, -2.5f, 3.25f, 1e-7f];
            File.WriteAllBytes(Path.Combine(dir, "weights.bin"), FloatBytes(wVals));
            var inlinePath = WriteModel(dir, "inline.onnx",
                BuildAddModel(Init("w", FloatElem, [4], FloatBytes(wVals))));
            var externalPath = WriteModel(dir, "external.onnx",
                BuildAddModel(ExternalW("weights.bin", offset: 0, length: 16)));
            Assert.Equal(
                RunAddModel(OnnxModelImporter.FromOnnxModel(inlinePath)),
                RunAddModel(OnnxModelImporter.FromOnnxModel(externalPath)));

            // Shared side file: 8 junk bytes, w's 16 bytes, 4 junk bytes, b's 16 bytes. b
            // carries no 'length' entry — the byte count implied by shape/dtype applies.
            float[] wShared = [1f, 2f, 3f, 4f];
            float[] bShared = [-0.5f, 0.5f, -1.5f, 2.5f];
            var shared = new byte[8 + 16 + 4 + 16];
            System.Buffer.BlockCopy(FloatBytes(wShared), 0, shared, 8, 16);
            System.Buffer.BlockCopy(FloatBytes(bShared), 0, shared, 28, 16);
            File.WriteAllBytes(Path.Combine(dir, "shared.bin"), shared);
            var bExt = Init("b", FloatElem, [4], null!);
            MarkExternal(bExt, "shared.bin", offset: 28);
            var sharedInlinePath = WriteModel(dir, "shared_inline.onnx", BuildAddModel(
                Init("w", FloatElem, [4], FloatBytes(wShared)),
                Init("b", FloatElem, [4], FloatBytes(bShared))));
            var sharedPath = WriteModel(dir, "shared.onnx", BuildAddModel(
                ExternalW("shared.bin", offset: 8, length: 16), bExt));
            Assert.Equal(
                RunAddModel(OnnxModelImporter.FromOnnxModel(sharedInlinePath)),
                RunAddModel(OnnxModelImporter.FromOnnxModel(sharedPath)));

            // Raw little-endian bytes for any element type: a 64-bit integer and a sub-word
            // float (float16, stored as ushort bit patterns).
            long[] iVals = [long.MaxValue - 1, -42];
            ushort[] hBits = [0x3C00, 0xC000];
            var hBytes = new byte[4];
            System.Buffer.BlockCopy(hBits, 0, hBytes, 0, 4);
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

            var results = ComputeContext.Default.Execute(
                OnnxModelImporter.FromOnnxModel(WriteModel(dir, "dtypes.onnx", WrapModel(g))));
            Assert.Equal(LongBytes(iVals), results[0].ToTensorData().AccessRawMemory().ToArray());
            Assert.Equal(hBytes, results[1].ToTensorData().AccessRawMemory().ToArray());
        });
    }

    [Fact]
    public void TestExternalDataFaultsFailLoudlyNamingTensorAndProblem()
    {
        WithTempDir(dir =>
        {
            File.WriteAllBytes(Path.Combine(dir, "short.bin"), FloatBytes(1f, 2f, 3f, 4f));

            void Case(string file, TensorProto w, string code, params string[] fragments)
            {
                var path = WriteModel(dir, file, BuildAddModel(w));
                var ex = Assert.Throws<ModelException>(() => OnnxModelImporter.FromOnnxModel(path));
                Assert.Equal(code, ex.ErrorCode);
                foreach (var fragment in fragments)
                    Assert.Contains(fragment, ex.Message);
            }

            Case("missing.onnx", ExternalW("missing.bin", 0, 16), ErrorCodes.XD004, "'w'", "missing.bin");
            Case("traversal.onnx", ExternalW("../escape.bin", 0, 16), ErrorCodes.XD003, "'w'");
            Case("absolute.onnx", ExternalW(Path.Combine(Path.GetTempPath(), "abs.bin"), 0, 16),
                ErrorCodes.XD003);
            Case("range.onnx", ExternalW("short.bin", 8, 16), ErrorCodes.XD005, "'w'", "short.bin");
            // offset + length would overflow a naive additive range check.
            Case("huge.onnx", ExternalW("short.bin", long.MaxValue - 8, 16), ErrorCodes.XD005, "'w'");
            // 'length' says 12 bytes but float[4] implies 16.
            Case("mismatch.onnx", ExternalW("short.bin", 0, 12), ErrorCodes.XD006, "'w'", "16");

            var wParse = Init("w", FloatElem, [4], null!);
            wParse.data_location = TensorProto.DataLocation.External;
            wParse.ExternalDatas.Add(new StringStringEntryProto { Key = "location", Value = "short.bin" });
            wParse.ExternalDatas.Add(new StringStringEntryProto { Key = "offset", Value = "not-a-number" });
            Case("parse.onnx", wParse, ErrorCodes.XD005);

            using var ms = new MemoryStream();
            ProtoBuf.Serializer.Serialize(ms, BuildAddModel(ExternalW("short.bin", 0, 16)));
            var bytes = ms.ToArray();
            var exNoDir = Assert.Throws<ModelException>(() => OnnxModelImporter.FromOnnxModel(bytes));
            Assert.Equal(ErrorCodes.XD001, exNoDir.ErrorCode);
            Assert.Contains("'w'", exNoDir.Message);

            float[] expected = [2f, 4f, 6f, 8f];
            Assert.Equal(expected,
                RunAddModel(OnnxModelImporter.FromOnnxModel(bytes, externalDataDirectory: dir)));
        });
    }

    [Fact]
    public void TestSaveWithExternalDataRoundTripsBitExactAlignedDeterministicAndOnnxRuntimeReadable()
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

            Assert.True(File.Exists(path + ".data"));
            using (var fs = File.OpenRead(path))
            {
                var saved = ProtoBuf.Serializer.Deserialize<ModelProto>(fs);
                var ext = Assert.Single(saved.Graph.Initializers
                    .Where(t => t.data_location == TensorProto.DataLocation.External));
                Assert.Equal("model.onnx.data",
                    ext.ExternalDatas.Single(e => e.Key == "location").Value);
                Assert.Equal("64", ext.ExternalDatas.Single(e => e.Key == "length").Value);
            }

            Assert.All(proto.Graph.Initializers, t =>
            {
                Assert.NotEqual(TensorProto.DataLocation.External, t.data_location);
                Assert.Empty(t.ExternalDatas);
            });

            var reimported = OnnxModelImporter.FromOnnxModel(path);
            Assert.Equal(direct, ComputeContext.Default.Execute(reimported, numOut, input)[0]
                .ToTensorData().AccessRawMemory().ToArray());
        });

        float[] wVals = [1f, 2f, 3f, 4f];
        float[] bVals = [5f, 6f, 7f, 8f];
        var addProto = BuildAddModel(
            Init("w", FloatElem, [4], FloatBytes(wVals)),
            Init("b", FloatElem, [4], FloatBytes(bVals)));

        WithTempDir(dir =>
        {
            var path1 = Path.Combine(dir, "one.onnx");
            var path2 = Path.Combine(dir, "two.onnx");
            var opts = new OnnxExternalDataOptions { SizeThreshold = 0 };
            OnnxModelExporter.SaveWithExternalData(addProto, path1, opts);
            OnnxModelExporter.SaveWithExternalData(addProto, path2, opts);

            // Deterministic layout: the side files are byte-identical across saves; the .onnx
            // files differ only in their 'location', so compare a re-save under the same name.
            Assert.Equal(File.ReadAllBytes(path1 + ".data"), File.ReadAllBytes(path2 + ".data"));
            var firstBytes = File.ReadAllBytes(path1);
            OnnxModelExporter.SaveWithExternalData(addProto, path1, opts);
            Assert.Equal(firstBytes, File.ReadAllBytes(path1));

            using (var fs = File.OpenRead(path1))
            {
                var saved = ProtoBuf.Serializer.Deserialize<ModelProto>(fs);
                var offsets = saved.Graph.Initializers
                    .ToDictionary(t => t.Name, t => t.ExternalDatas.Single(e => e.Key == "offset").Value);
                Assert.Equal("0", offsets["w"]);
                Assert.Equal("4096", offsets["b"]);
            }
            Assert.Equal(4096 + 16, new FileInfo(path1 + ".data").Length);

            var inlinePath = WriteModel(dir, "inline.onnx", addProto);
            Assert.Equal(
                RunAddModel(OnnxModelImporter.FromOnnxModel(inlinePath)),
                RunAddModel(OnnxModelImporter.FromOnnxModel(path1)));

            // Stock onnxruntime resolves the side file itself, from the model's path.
            var ortPath = Path.Combine(dir, "ort.onnx");
            OnnxModelExporter.SaveWithExternalData(
                BuildAddModel(Init("w", FloatElem, [4], FloatBytes(10f, 20f, 30f, 40f))), ortPath, opts);
            Assert.True(File.Exists(ortPath + ".data"));
            using var session = new Microsoft.ML.OnnxRuntime.InferenceSession(ortPath);
            float[] xVals = [1f, 2f, 3f, 4f];
            var x = new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(xVals, [4]);
            using var results = session.Run(
                [Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor("x", x)]);
            float[] expected = [11f, 22f, 33f, 44f];
            Assert.Equal(expected, results.First().AsEnumerable<float>().ToArray());
        });
    }

    [Fact]
    public void TestSaveBelowThresholdMatchesInlineSaveAndSelfContainedOverLimitThrows()
    {
        var proto = BuildAddModel(Init("w", FloatElem, [4], FloatBytes(1f, 2f, 3f, 4f)));
        WithTempDir(dir =>
        {
            var inlinePath = Path.Combine(dir, "inline.onnx");
            var extPath = Path.Combine(dir, "ext.onnx");
            OnnxModelExporter.Save(proto, inlinePath);
            // An external save first, so a now-stale side file must be removed by the
            // below-threshold re-save, whose .onnx must be byte-equal to the inline save.
            OnnxModelExporter.SaveWithExternalData(proto, extPath,
                new OnnxExternalDataOptions { SizeThreshold = 0 });
            Assert.True(File.Exists(extPath + ".data"));
            OnnxModelExporter.SaveWithExternalData(proto, extPath,
                new OnnxExternalDataOptions { SizeThreshold = 1024 });
            Assert.False(File.Exists(extPath + ".data"));
            Assert.Equal(File.ReadAllBytes(inlinePath), File.ReadAllBytes(extPath));

            // The real ceiling is protobuf's 2 GB message limit; drive the same check with a
            // tiny injected ceiling rather than allocating gigabytes.
            var ex = Assert.Throws<ModelException>(
                () => OnnxModelExporter.Save(proto, Path.Combine(dir, "big.onnx"), maxTensorBytes: 8));
            Assert.Equal(ErrorCodes.XD007, ex.ErrorCode);
            Assert.Contains("SaveWithExternalData", ex.Message);

            // A payload in a typed data field (int64_data) counts against the ceiling too.
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
    public void TestSaveWithExternalDataRequiresConcreteModel()
    {
        var moduleGraph = Shorokoo.Tests.Modules.ScalarMultiplyModel.ComputationGraph;
        var arch = moduleGraph.ToConcreteArchitecture(
            moduleGraph.FromOrderedInputs([TensorData([2L], 1.0f, 2.0f)]));
        var path = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".onnx");

        // The concrete architecture is refused twice over: via its graph-kind metadata tag,
        // and — untagged — via the proto op-scan that recognizes serialized unmaterialized
        // parameters (calls to initializer-typed functions).
        (ModelProto Proto, string[] Fragments)[] cases =
        [
            (FastOnnxModelBuilder.BuildInternalOnnxModel(moduleGraph.ToInternal()),
                ["'concrete-model'", "'module'"]),
            (FastOnnxModelBuilder.BuildInternalOnnxModel(arch.ToInternal(), stage: arch.Kind),
                ["'concrete-architecture'"]),
            (FastOnnxModelBuilder.BuildInternalOnnxModel(arch.ToInternal()),
                ["'concrete-architecture'"]),
        ];
        foreach (var (proto, fragments) in cases)
        {
            var ex = Assert.Throws<ModelException>(
                () => OnnxModelExporter.SaveWithExternalData(proto, path));
            Assert.Contains("XD008", ex.Message);
            foreach (var fragment in fragments)
                Assert.Contains(fragment, ex.Message);
            Assert.False(File.Exists(path));
            Assert.False(File.Exists(path + ".data"));
        }
    }

    private static ModelProto AddModelThatFailsMidSerialization()
    {
        var model = BuildAddModel(Init("w", FloatElem, [4], FloatBytes(1f, 2f, 3f, 4f)));
        model.OpsetImports.Add(null!);
        return model;
    }

    /// <summary>The saved model and its external-data side file, as hex; a missing file reads "absent".</summary>
    private static string[] SavedPair(string path)
    {
        string[] paths = [path, path + ".data"];
        return [.. paths.Select(p => File.Exists(p)
            ? Convert.ToHexString(File.ReadAllBytes(p))
            : "absent")];
    }

    /// <summary>
    /// Both exporter entry points truncate the target before the model is serialized into it, so
    /// a save that fails partway destroys the previously saved model — and, with the external-data
    /// layout, deletes its side file as well. Every <c>Persistence.*</c> save gets this right by
    /// staging through <see cref="Shorokoo.Core.Utils.AtomicFileWriter"/>; these do not.
    /// </summary>
    [Fact]
    public void TestAnExportFailingMidSerializationLeavesThePreviouslySavedModelAndSideFileIntact()
    {
        WithTempDir(dir =>
        {
            (string Name, Action<ModelProto, string> Save)[] savers =
            [
                ("inline", (m, p) => OnnxModelExporter.Save(m, p)),
                ("external", (m, p) => OnnxModelExporter.SaveWithExternalData(
                    m, p, new OnnxExternalDataOptions { SizeThreshold = 0 })),
                ("external-below-threshold", (m, p) => OnnxModelExporter.SaveWithExternalData(m, p)),
            ];
            foreach (var (name, save) in savers)
            {
                var path = Path.Combine(dir, name + ".onnx");
                save(BuildAddModel(Init("w", FloatElem, [4], FloatBytes(1f, 2f, 3f, 4f))), path);
                var committed = SavedPair(path);
                Assert.ThrowsAny<Exception>(() => save(AddModelThatFailsMidSerialization(), path));
                Assert.Equal(committed, SavedPair(path));
            }
        });
    }
}

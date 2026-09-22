using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Core.Utils;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Backends;
using Shorokoo.Core.Nodes;
using Shorokoo.Onnx;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage for <see cref="DType.Utf8"/> — the lone DType for ONNX
/// <c>TensorProto.DataType.STRING</c> (proto num 8): its conversion arms and the
/// ORT-backed variable-length string tensor construct/read path.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class DTypeStringCoverageTests
{
    [Fact]
    public void TestDTypeStringConversionArmsAndOrtStringTensorRoundtrip()
    {
        Assert.Equal(8, DType.Utf8.ProtoTypeNum);
        Assert.Equal("Utf8", DType.Utf8.ToString());
        Assert.Same(DType.Utf8, DType.FromProtoTypeNum(8));
        Assert.Same(DType.Utf8, (DType)8);

        Assert.Equal(typeof(utf8), DType.Utf8.ToIVarType());
        Assert.Equal(typeof(string), DType.Utf8.ToPrimitiveType());

        var bitCountEx = Assert.Throws<UnsupportedDTypeException>(() => DType.Utf8.EncodingBitCount);
        Assert.Equal(ErrorCodes.DT020, bitCountEx.ErrorCode);

        Assert.Same(DType.Utf8, OnnxUtils.GetDType<utf8>());
        Assert.Same(DType.Utf8, OnnxUtils.GetDType<string>());
        Assert.Same(DType.Utf8, OnnxUtils.GetDType(typeof(utf8)));
        Assert.Same(DType.Utf8, OnnxUtils.GetDType(typeof(string)));

        string[] values = ["hello", "", "shoroko̅o", "with\nnewline", "🦀"];
        long[] shape = [values.Length];

        using (var tensor = DefaultBackend.Instance.CreateStringTensor(values, shape))
        {
            Assert.Equal(ShorokooOnnxValueType.Tensor, tensor.ValueType);
            Assert.Equal(ShorokooTensorElementType.String, tensor.ElementType);
            Assert.Equal(shape, tensor.Shape);
            Assert.Equal(values, tensor.GetStringTensorData());
        }

        var rawBytesEx = Assert.Throws<NotSupportedException>(() =>
            DefaultBackend.Instance.CreateTensorFromRawBytes(
                ShorokooTensorElementType.String, [], [0L]));
        Assert.Contains("CreateStringTensor", rawBytesEx.Message);

        var hostRawEx = Assert.Throws<NotSupportedException>(() =>
            TensorData.CreateFromRawBytes(new Shape(0L), DType.Utf8, []));
        Assert.Contains("variable-length", hostRawEx.Message);
    }

    [Fact]
    public void TestAStringLiteralIsBuiltWithNoBackendAndRoundTripsThroughOne()
    {
        string[] values = ["hello", "", "with\nnewline", "shorokoo"];
        long[] dims = [2L, 2L];

        TensorData literal = null!;
        // The AsyncLocal seam rather than a before/after read of the process-wide slot: this suite
        // runs four tests at once, so any of them may settle that slot inside the window.
        Assert.Equal(0, DefaultBackend.CountInstanceReads(() => literal = TensorData(dims, values)));

        Assert.IsType<HostStringTensorData>(literal);
        // Nothing of a runtime's is in it -- which is the whole of "no backend was needed".
        Assert.False(literal is IOnnxData);
        Assert.Same(DType.Utf8, literal.DType);
        Assert.Equal(new Shape(2L, 2L), literal.Shape);
        Assert.Equal(values, ((HostStringTensorData)literal).Strings);

        var value = literal.ToTensorValue();
        Assert.Equal(ShorokooTensorElementType.String, value.ElementType);
        Assert.Equal(dims, value.Shape);
        Assert.Equal(values, value.GetStringTensorData());

        // The tensor's own, kept per backend rather than rebuilt per ask: two feeds of one
        // backend must not be handed two values, nor one the other has released.
        Assert.Same(value, literal.ToTensorValue());
        Assert.Same(value, ((IData)literal).ToTensorValue());

        // There is no flat buffer under a string tensor here or in ONNX Runtime, and the refusal
        // names the reads that do work rather than handing back a span of nothing.
        var spanEx = Assert.Throws<InvalidOperationException>(() => { literal.AccessRawMemory(); });
        Assert.Contains("Strings", spanEx.Message);

        literal.Dispose();
        Assert.Throws<ObjectDisposedException>(() => _ = ((HostStringTensorData)literal).Strings);
        Assert.Throws<ObjectDisposedException>(() => literal.ToTensorValue());
    }

    // A string initializer whose string_data does not cover its dims is refused, as the numeric
    // path refuses a short raw_data -- not padded out with empty strings that read as model data.
    [Fact]
    public void TestAStringInitializerIsRefusedWhenItsDataDoesNotCoverItsDims()
    {
        static TensorProto Proto(long[] dims, params string[] values)
        {
            var t = new TensorProto { Dims = dims, data_type = 8 };
            foreach (var v in values) t.StringDatas.Add(System.Text.Encoding.UTF8.GetBytes(v));
            return t;
        }

        static InternalComputationGraph Read(TensorProto initializer)
        {
            var g = new GraphProto { Name = "g" };
            g.Initializers.Add(initializer);
            var model = new ModelProto { IrVersion = 10, Graph = g };
            model.OpsetImports.Add(new OperatorSetIdProto { Domain = "", Version = 21 });
            using var ms = new MemoryStream();
            ProtoBuf.Serializer.Serialize(ms, model);
            return OnnxModelImporter.FromOnnxModelToInternalGraph(ms.ToArray());
        }

        var init = Proto([2L, 2L], "a", "b");
        init.Name = "w";
        Assert.Throws<ArgumentException>(() => Read(init));

        var none = Proto([2L]);
        none.Name = "w";
        Assert.Throws<ArgumentException>(() => Read(none));

        // ONNX forbids raw_data for STRING, so a tensor carrying it is refused by name rather
        // than silently read as holding nothing.
        var raw = new TensorProto { Dims = [2L], data_type = 8, RawData = [0x61, 0x62], Name = "w" };
        Assert.Throws<NotSupportedException>(() => Read(raw));

        // A zero-element string tensor carries no elements and is not a shortfall.
        var empty = Proto([0L]);
        empty.Name = "w";
        Read(empty);
    }

    // A literal standing for a generic type parameter describes the same storage as the dtype it
    // stands for, so every read of its elements has to agree about what it holds.
    [Fact]
    public void TestAStringTensorTaggedWithAGenericParameterReadsLikeAnUntaggedOne()
    {
        static TensorData<utf8> Tagged() => (TensorData<utf8>)TensorDataConversion.ConvertTensorDataType(
            TensorData([2L], (string[])["a", "b"]), DType.CreateWithGenericParam(DType.Utf8, "T"));

        Assert.Equal((object[])["a", "b"], Tagged().DebugData);
        Assert.Equal((object[])["a", "b"], Tagged().Data);
        Assert.Equal((string[])["a", "b"], [.. Tagged().MoveToAttribute().Values]);

        var attribute = TensorData([2L], (string[])["a", "b"]).MoveToAttribute()
            .WithDType(DType.CreateWithGenericParam(DType.Utf8, "T"));
        var proto = OnnxIRFactory.CreateTensor(
            [2L], "w", attribute.DType, identifierTemplate: null, isTrainable: true, attribute);
        Assert.Equal((string[])["a", "b"],
            [.. proto.StringDatas.Select(System.Text.Encoding.UTF8.GetString)]);
    }

    // Export writes string_data and import reads it. Both directions in one test on purpose:
    // they were absent together, and either alone would have gone unnoticed the same way.
    [Fact]
    public void TestAStringConstantSurvivesAnOnnxRoundTrip()
    {
        string[] values = ["hello", "", "with\nnewline", "日本語"];
        var graph = new InternalComputationGraph(
            [], [Globals.Tensor(TensorData([2L, 2L], values).MoveToAttribute())]);

        var model = FastOnnxModelBuilder.BuildInternalOnnxModel(graph, prepForOnnx: true);
        var written = model.Graph.Nodes.Single(n => n.OpType == OpCodes.CONSTANT)
            .Attributes.Single(a => a.Name == OnnxOpAttributeNames.AttrValue).T;

        Assert.Equal(8, written.data_type);
        Assert.Null(written.RawData);
        Assert.Equal(values, written.StringDatas.Select(System.Text.Encoding.UTF8.GetString));

        using var stream = new MemoryStream();
        ProtoBuf.Serializer.Serialize(stream, model);
        var reread = OnnxModelImporter.FromOnnxModelToInternalGraph(stream.ToArray());

        var bound = reread.Nodes.Single(n => n.OpCode == OpCodes.CONSTANT)
            .Attributes.GetAttributeVal(OnnxOpAttributeNames.AttrValue)!;
        Assert.Same(DType.Utf8, bound.DType);
        Assert.Equal(new Shape(2L, 2L), bound.Shape);
        Assert.Equal(values, bound.Values);
    }
}

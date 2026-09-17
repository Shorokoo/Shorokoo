using Shorokoo.Core.Inference.Abstractions;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage for <see cref="DType.String"/> — the lone DType for ONNX
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
        Assert.Equal(8, DType.String.ProtoTypeNum);
        Assert.Equal("String", DType.String.ToString());
        Assert.Same(DType.String, DType.FromProtoTypeNum(8));
        Assert.Same(DType.String, (DType)8);

        Assert.Equal(typeof(@string), DType.String.ToIVarType());
        Assert.Equal(typeof(string), DType.String.ToPrimitiveType());

        var bitCountEx = Assert.Throws<UnsupportedDTypeException>(() => DType.String.EncodingBitCount);
        Assert.Equal(ErrorCodes.DT020, bitCountEx.ErrorCode);

        Assert.Same(DType.String, OnnxUtils.GetDType<@string>());
        Assert.Same(DType.String, OnnxUtils.GetDType<string>());
        Assert.Same(DType.String, OnnxUtils.GetDType(typeof(@string)));
        Assert.Same(DType.String, OnnxUtils.GetDType(typeof(string)));

        string[] values = ["hello", "", "shoroko̅o", "with\nnewline", "🦀"];
        long[] shape = [values.Length];

        using (var tensor = InferenceBackend.Default.CreateStringTensor(values, shape))
        {
            Assert.Equal(ShorokooOnnxValueType.Tensor, tensor.ValueType);
            Assert.Equal(ShorokooTensorElementType.String, tensor.ElementType);
            Assert.Equal(shape, tensor.Shape);
            Assert.Equal(values, tensor.GetStringTensorData());
        }

        var rawBytesEx = Assert.Throws<NotSupportedException>(() =>
            InferenceBackend.Default.CreateTensorFromRawBytes(
                ShorokooTensorElementType.String, [], [0L]));
        Assert.Contains("CreateStringTensor", rawBytesEx.Message);

        var hostRawEx = Assert.Throws<NotSupportedException>(() =>
            TensorData.CreateFromRawBytes(new Shape(0L), DType.String, []));
        Assert.Contains("variable-length", hostRawEx.Message);
    }

    [Fact]
    public void TestAStringLiteralIsBuiltWithNoBackendAndRoundTripsThroughOne()
    {
        string[] values = ["hello", "", "with\nnewline", "shorokoo"];
        long[] dims = [2L, 2L];

        var backendBefore = InferenceBackend.Current;
        var literal = TensorData(dims, values);
        Assert.Same(backendBefore, InferenceBackend.Current);

        Assert.IsType<HostStringTensorData>(literal);
        // Nothing of a runtime's is in it -- which is the whole of "no backend was needed".
        Assert.False(literal is IOnnxData);
        Assert.Same(DType.String, literal.DType);
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
}

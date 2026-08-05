using Shorokoo.Core.Inference.Abstractions;

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

        using (var tensor = InferenceBackend.Factory.CreateStringTensor(values, shape))
        {
            Assert.Equal(ShorokooOnnxValueType.Tensor, tensor.ValueType);
            Assert.Equal(ShorokooTensorElementType.String, tensor.ElementType);
            Assert.Equal(shape, tensor.Shape);
            Assert.Equal(values, tensor.GetStringTensorData());
        }

        var rawBytesEx = Assert.Throws<NotSupportedException>(() =>
            InferenceBackend.Factory.CreateTensorFromRawBytes(
                ShorokooTensorElementType.String, [], [0L]));
        Assert.Contains("CreateStringTensor", rawBytesEx.Message);
    }
}

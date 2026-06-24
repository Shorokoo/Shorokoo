using Shorokoo.Core.Inference.Abstractions;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage-purpose tests for <see cref="DType.String"/> — the lone DType for
/// ONNX <c>TensorProto.DataType.STRING</c> (proto num 8) that replaced the four
/// legacy fixed-width encoding DTypes (UTF32/UCS2/AsCII/UTF8String). Bundles the
/// DType conversion arms (<c>ProtoTypeNum</c>, <c>ToIVarType</c>,
/// <c>ToPrimitiveType</c>, <c>EncodingBitCount</c>, the generic/non-generic
/// <c>OnnxUtils.GetDType</c> dispatchers) together with the ORT-backed
/// variable-length string tensor construct/read path
/// (<c>CreateStringTensor</c> / <c>GetStringTensorData</c>) into a single
/// Coverage fact so the curated Coverage scan picks the file up.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class DTypeStringCoverageTests
{
    [Fact]
    public void TestDTypeStringCoverage()
    {
        // ──────────────────────────────────────────────────────────────────
        // DType identity + the Mapping/FromProtoTypeNum round-trip at 8.
        // ──────────────────────────────────────────────────────────────────
        Assert.Equal(8, DType.String.ProtoTypeNum);
        Assert.Equal("String", DType.String.ToString());
        Assert.Same(DType.String, DType.FromProtoTypeNum(8));
        Assert.Same(DType.String, (DType)8);

        // ──────────────────────────────────────────────────────────────────
        // Type-conversion arms: @string IVarType + System.String primitive,
        // and the variable-length EncodingBitCount guard.
        // ──────────────────────────────────────────────────────────────────
        Assert.Equal(typeof(@string), DType.String.ToIVarType());
        Assert.Equal(typeof(string), DType.String.ToPrimitiveType());

        var bitCountEx = Assert.Throws<UnsupportedDTypeException>(() => DType.String.EncodingBitCount);
        Assert.Equal(ErrorCodes.DT020, bitCountEx.ErrorCode);

        // ──────────────────────────────────────────────────────────────────
        // OnnxUtils.GetDType — both the generic and non-generic dispatchers,
        // mapping the @string IVarType and the System.String CLR type.
        // ──────────────────────────────────────────────────────────────────
        Assert.Same(DType.String, OnnxUtils.GetDType<@string>());
        Assert.Same(DType.String, OnnxUtils.GetDType<string>());
        Assert.Same(DType.String, OnnxUtils.GetDType(typeof(@string)));
        Assert.Same(DType.String, OnnxUtils.GetDType(typeof(string)));

        // ──────────────────────────────────────────────────────────────────
        // ORT-backed string tensor: CreateStringTensor builds a variable-length
        // UTF-8 tensor, and GetStringTensorData reads it back element-for-element
        // (covering empty, multi-byte, control-char, and emoji elements).
        // ──────────────────────────────────────────────────────────────────
        string[] values = ["hello", "", "shoroko̅o", "with\nnewline", "🦀"];
        long[] shape = [values.Length];

        using (var tensor = InferenceBackend.Factory.CreateStringTensor(values, shape))
        {
            Assert.Equal(ShorokooOnnxValueType.Tensor, tensor.ValueType);
            Assert.Equal(ShorokooTensorElementType.String, tensor.ElementType);
            Assert.Equal(shape, tensor.Shape);
            Assert.Equal(values, tensor.GetStringTensorData());
        }

        // The raw-bytes path rejects String and points at CreateStringTensor.
        var rawBytesEx = Assert.Throws<NotSupportedException>(() =>
            InferenceBackend.Factory.CreateTensorFromRawBytes(
                ShorokooTensorElementType.String, [], [0L]));
        Assert.Contains("CreateStringTensor", rawBytesEx.Message);
    }
}

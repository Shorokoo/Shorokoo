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

        // And the refusal survives one level up, where it used to come from the backend: raw
        // bytes do not describe a string tensor, so asking for one is an error rather than a
        // HostTensorData<@string> over bytes that mean nothing.
        var hostRawEx = Assert.Throws<NotSupportedException>(() =>
            TensorData.CreateFromRawBytes(new Shape(0L), DType.String, []));
        Assert.Contains("variable-length", hostRawEx.Message);
    }

    /// <summary>
    /// A string literal is managed strings and nothing else until a backend asks for one: it
    /// carries no runtime value, and the value it builds when finally asked reads back the
    /// elements it was given.
    ///
    /// <para>What says that building one costs no backend is that nothing resolves one while it
    /// happens. <see cref="InferenceBackend.Current"/> answers without resolving a backend, so
    /// comparing it across the construction states exactly that; in a process that has bound none
    /// yet — this class run on its own — it is the stronger statement that there is still none
    /// afterwards. Asserting null outright would instead pass for the wrong reason in a shared
    /// test host, where some earlier test has already bound one.</para>
    /// </summary>
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

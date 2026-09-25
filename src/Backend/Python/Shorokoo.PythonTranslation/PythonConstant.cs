using System.Runtime.InteropServices;
using System.Text;
using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory.IR;

namespace Shorokoo.PythonTranslation;

/// <summary>
/// A tensor the translated model holds as a constant — an initializer, a <c>Constant</c> node's
/// value, a tensor attribute — as host bytes (or strings) the session turns into a tensor of
/// its backend when it is built.
/// </summary>
internal sealed record PythonConstant(
    ShorokooTensorElementType ElementType, long[] Shape, byte[]? Bytes, string[]? Strings)
{
    public static PythonConstant Scalar(float value) => new(
        ShorokooTensorElementType.Float, [], BitConverter.GetBytes(value), null);

    public static PythonConstant Vector(float[] values) => new(
        ShorokooTensorElementType.Float, [values.Length], MemoryMarshal.AsBytes(values.AsSpan()).ToArray(), null);

    public static PythonConstant Scalar(long value) => new(
        ShorokooTensorElementType.Int64, [], BitConverter.GetBytes(value), null);

    public static PythonConstant Vector(long[] values) => new(
        ShorokooTensorElementType.Int64, [values.Length], MemoryMarshal.AsBytes(values.AsSpan()).ToArray(), null);

    /// <summary>A string constant of <paramref name="utf8"/>, which <paramref name="what"/> holds.</summary>
    /// <exception cref="NotSupportedException">One of them is not UTF-8.</exception>
    public static PythonConstant StringsOf(
        PythonDialect dialect, long[] shape, IEnumerable<byte[]> utf8, string what, string? operatorType)
        => new(ShorokooTensorElementType.String, shape, null, [.. utf8.Select(b => Text(dialect, b, what, operatorType))]);

    private static readonly UTF8Encoding StrictUtf8 = new(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: true);

    /// <summary>
    /// <paramref name="utf8"/> as text. ONNX strings are bytes, which ONNX Runtime carries as they
    /// are; a Python-based backend holds them as text, so bytes that are not UTF-8 are refused rather
    /// than read with a replacement character where they stood.
    /// </summary>
    /// <exception cref="NotSupportedException">The bytes are not UTF-8.</exception>
    public static string Text(PythonDialect dialect, byte[] utf8, string what, string? operatorType)
    {
        try
        {
            return StrictUtf8.GetString(utf8);
        }
        catch (DecoderFallbackException)
        {
            throw dialect.Unsupported(UnsupportedReason.UnsupportedModel, null, operatorType,
                $"{what} holds a string that is not UTF-8 text, and the {dialect.BackendName} backend holds strings as text.");
        }
    }

    /// <summary>
    /// The constant a <see cref="TensorProto"/> holds, read from its raw bytes or from whichever
    /// typed field ONNX puts that element type in.
    /// </summary>
    /// <exception cref="NotSupportedException">The tensor's data is external, is not the
    /// size its shape says, or its element type has no fixed stride the backend holds.</exception>
    public static PythonConstant FromTensor(PythonDialect dialect, TensorProto tensor, string? operatorType)
    {
        var shape = tensor.Dims ?? [];
        var elementType = (ShorokooTensorElementType)tensor.data_type;
        if (tensor.data_location == TensorProto.DataLocation.External || tensor.ExternalDatas.Count > 0)
            throw dialect.Unsupported(UnsupportedReason.UnsupportedModel, null, operatorType,
                $"The tensor '{tensor.Name}' keeps its data in an external file, which the {dialect.BackendName} backend "
                + "does not read. Save the model with its tensors inline.");
        if (elementType == ShorokooTensorElementType.String)
            return StringsOf(dialect, shape, tensor.StringDatas, $"The tensor '{tensor.Name}'", operatorType);

        int byteCount;
        try
        {
            byteCount = PythonElementTypes.ByteCount(elementType, shape);
        }
        catch (NotSupportedException ex)
        {
            throw dialect.Unsupported(UnsupportedReason.UnsupportedModel, null, operatorType,
                $"The tensor '{tensor.Name}' is of element type {elementType}, which the {dialect.BackendName} backend cannot "
                + $"hold: {ex.Message}");
        }
        var bytes = tensor.RawData ?? elementType switch
        {
            ShorokooTensorElementType.Float or ShorokooTensorElementType.Complex64
                => MemoryMarshal.AsBytes((tensor.FloatDatas ?? []).AsSpan()).ToArray(),
            ShorokooTensorElementType.Double or ShorokooTensorElementType.Complex128
                => MemoryMarshal.AsBytes((tensor.DoubleDatas ?? []).AsSpan()).ToArray(),
            ShorokooTensorElementType.Int64
                => MemoryMarshal.AsBytes((tensor.Int64Datas ?? []).AsSpan()).ToArray(),
            ShorokooTensorElementType.UInt64
                => MemoryMarshal.AsBytes((tensor.Uint64Datas ?? []).AsSpan()).ToArray(),
            ShorokooTensorElementType.UInt32
                => MemoryMarshal.AsBytes((tensor.Uint64Datas ?? []).Select(v => (uint)v).ToArray().AsSpan()).ToArray(),
            ShorokooTensorElementType.Int32
                => MemoryMarshal.AsBytes((tensor.Int32Datas ?? []).AsSpan()).ToArray(),
            // Everything narrower than 32 bits -- the small integers, bool, the 16-bit floats' bit
            // patterns and the 8-bit floats' -- is carried one element per int32, in its low bytes.
            _ => Narrowed(tensor.Int32Datas ?? [], PythonElementTypes.ElementSize(elementType)),
        };
        // Exactly, as ONNX Runtime has it: the session copies the constant's bytes by its shape, and
        // data of another size is a tensor written wrong -- or, empty, one written without its data.
        if (bytes.Length != byteCount)
            throw dialect.Unsupported(UnsupportedReason.UnsupportedModel, null, operatorType,
                $"The tensor '{tensor.Name}' holds {bytes.Length} bytes of {(tensor.RawData is null ? "typed" : "raw")} data where its "
                + $"shape [{string.Join(", ", shape)}] of {elementType} needs {byteCount}.");
        return new PythonConstant(elementType, shape, bytes, null);
    }

    private static byte[] Narrowed(int[] values, int size)
    {
        var bytes = new byte[values.Length * size];
        for (int i = 0; i < values.Length; i++)
            for (int b = 0; b < size; b++)
                bytes[i * size + b] = (byte)(values[i] >> (8 * b));
        return bytes;
    }
}

using System.Runtime.InteropServices;
using System.Text;
using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory.IR;

namespace Shorokoo.PyTorch.Translation;

/// <summary>
/// A tensor the translated model holds as a constant — an initializer, a <c>Constant</c> node's
/// value, a tensor attribute — as host bytes (or strings) the session turns into a torch tensor
/// on its device when it is built.
/// </summary>
internal sealed record TorchConstant(
    ShorokooTensorElementType ElementType, long[] Shape, byte[]? Bytes, string[]? Strings)
{
    public static TorchConstant Scalar(float value) => new(
        ShorokooTensorElementType.Float, [], BitConverter.GetBytes(value), null);

    public static TorchConstant Vector(float[] values) => new(
        ShorokooTensorElementType.Float, [values.Length], MemoryMarshal.AsBytes(values.AsSpan()).ToArray(), null);

    public static TorchConstant Scalar(long value) => new(
        ShorokooTensorElementType.Int64, [], BitConverter.GetBytes(value), null);

    public static TorchConstant Vector(long[] values) => new(
        ShorokooTensorElementType.Int64, [values.Length], MemoryMarshal.AsBytes(values.AsSpan()).ToArray(), null);

    public static TorchConstant StringsOf(long[] shape, IEnumerable<byte[]> utf8)
        => new(ShorokooTensorElementType.String, shape, null, [.. utf8.Select(b => Encoding.UTF8.GetString(b))]);

    /// <summary>
    /// The constant a <see cref="TensorProto"/> holds, read from its raw bytes or from whichever
    /// typed field ONNX puts that element type in.
    /// </summary>
    /// <exception cref="TorchUnsupportedModelException">The tensor's data is external, or its
    /// element type has no torch dtype.</exception>
    public static TorchConstant FromTensor(TensorProto tensor, string? operatorType)
    {
        var shape = tensor.Dims ?? [];
        var elementType = (ShorokooTensorElementType)tensor.data_type;
        if (tensor.data_location == TensorProto.DataLocation.External || tensor.ExternalDatas.Count > 0)
            throw new TorchUnsupportedModelException(TorchUnsupportedReason.UnsupportedModel, null, operatorType,
                $"The tensor '{tensor.Name}' keeps its data in an external file, which the PyTorch backend "
                + "does not read. Save the model with its tensors inline.");
        if (elementType == ShorokooTensorElementType.String)
            return StringsOf(shape, tensor.StringDatas);

        int byteCount;
        try
        {
            byteCount = TorchElementTypes.ByteCount(elementType, shape);
        }
        catch (NotSupportedException ex)
        {
            throw new TorchUnsupportedModelException(TorchUnsupportedReason.UnsupportedModel, null, operatorType,
                $"The tensor '{tensor.Name}' is of element type {elementType}, which the PyTorch backend cannot "
                + $"hold: {ex.Message}");
        }
        if (tensor.RawData is { } raw)
            return new TorchConstant(elementType, shape, raw, null);

        var bytes = elementType switch
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
            _ => Narrowed(tensor.Int32Datas ?? [], TorchElementTypes.ElementSize(elementType)),
        };
        if (bytes.Length < byteCount)
            throw new TorchUnsupportedModelException(TorchUnsupportedReason.UnsupportedModel, null, operatorType,
                $"The tensor '{tensor.Name}' holds {bytes.Length} bytes of data where its shape "
                + $"[{string.Join(", ", shape)}] of {elementType} needs {byteCount}.");
        return new TorchConstant(elementType, shape, bytes, null);
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

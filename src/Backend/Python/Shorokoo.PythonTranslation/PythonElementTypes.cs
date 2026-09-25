using System.Numerics;
using Shorokoo.Core.Backends;

namespace Shorokoo.PythonTranslation;

/// <summary>
/// The element types a tensor of a Python-based backend can hold, and how many bytes each lays down.
///
/// <para>PyTorch and JAX have every fixed-stride ONNX type, including the four 8-bit float kinds and
/// both complex ones, which ONNX Runtime does not build from bytes. So the byte table is
/// <see cref="TensorElementLayout"/>'s — the one every backend shares — extended by exactly those;
/// anything it turns away that is not one of them (a string, the 4-bit types) is turned away here
/// in its words.</para>
/// </summary>
internal static class PythonElementTypes
{
    /// <summary>The bytes one element occupies.</summary>
    /// <exception cref="NotSupportedException">The type has no fixed stride, or no dtype in PyTorch or JAX.</exception>
    public static int ElementSize(ShorokooTensorElementType elementType) => elementType switch
    {
        ShorokooTensorElementType.Complex64 => 8,
        ShorokooTensorElementType.Complex128 => 16,
        ShorokooTensorElementType.Float8E4M3FN or ShorokooTensorElementType.Float8E4M3FNUZ
            or ShorokooTensorElementType.Float8E5M2 or ShorokooTensorElementType.Float8E5M2FNUZ => 1,
        _ => TensorElementLayout.ElementSizeInBytes(elementType),
    };

    /// <summary>The bytes a tensor of this type and shape covers.</summary>
    public static int ByteCount(ShorokooTensorElementType elementType, long[] shape)
    {
        ArgumentNullException.ThrowIfNull(shape);
        var size = ElementSize(elementType);
        // The shape checks are TensorElementLayout's, asked through a type it knows the stride of.
        var elements = TensorElementLayout.ByteCount(ShorokooTensorElementType.UInt8, shape);
        return checked(elements * size);
    }

    /// <summary>The element type of the storage primitive <typeparamref name="T"/>.</summary>
    public static ShorokooTensorElementType Of<T>() where T : unmanaged
    {
        if (typeof(T) == typeof(float)) return ShorokooTensorElementType.Float;
        if (typeof(T) == typeof(double)) return ShorokooTensorElementType.Double;
        if (typeof(T) == typeof(bool)) return ShorokooTensorElementType.Bool;
        if (typeof(T) == typeof(sbyte)) return ShorokooTensorElementType.Int8;
        if (typeof(T) == typeof(byte)) return ShorokooTensorElementType.UInt8;
        if (typeof(T) == typeof(short)) return ShorokooTensorElementType.Int16;
        if (typeof(T) == typeof(ushort)) return ShorokooTensorElementType.UInt16;
        if (typeof(T) == typeof(int)) return ShorokooTensorElementType.Int32;
        if (typeof(T) == typeof(uint)) return ShorokooTensorElementType.UInt32;
        if (typeof(T) == typeof(long)) return ShorokooTensorElementType.Int64;
        if (typeof(T) == typeof(ulong)) return ShorokooTensorElementType.UInt64;
        if (typeof(T) == typeof(Float16) || typeof(T) == typeof(Half)) return ShorokooTensorElementType.Float16;
        if (typeof(T) == typeof(BFloat16)) return ShorokooTensorElementType.BFloat16;
        if (typeof(T) == typeof(Complex)) return ShorokooTensorElementType.Complex128;
        throw new NotSupportedException($"CreateTensor does not support element type {typeof(T).Name}.");
    }
}

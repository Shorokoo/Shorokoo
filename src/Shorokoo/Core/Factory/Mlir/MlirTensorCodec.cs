using System;
using System.Runtime.InteropServices;

namespace Shorokoo.Core.Factory.Mlir
{
    /// <summary>
    /// Lossless textual codec for <c>AttributeType.Tensor</c> attribute values (the
    /// <c>Constant</c> op's <c>value</c>, model-parameter data, etc.). A tensor is spelled
    /// <c>dense&lt;[d0, d1], dtype&lt;N&gt;, "base64"&gt;</c> — dimensions, proto element type, and the
    /// raw little-endian storage bytes.
    ///
    /// <para>Phase 1 covers the integer, bool, and IEEE float element types that appear in module
    /// graphs (shape constants are int64; scalar constants are float32/float64/int32/int64; a loop's
    /// continue flag is bool). float16, bfloat16, string and complex tensors throw
    /// <see cref="NotSupportedException"/> until a later phase adds them.</para>
    /// </summary>
    internal static class MlirTensorCodec
    {
        public static byte[] ToRawBytes(TensorData td)
        {
            var tv = td.ToTensorValue();
            return td.DType.ProtoTypeNum switch
            {
                1 => MemoryMarshal.AsBytes(tv.GetTensorDataAsSpan<float>()).ToArray(),   // Float32
                11 => MemoryMarshal.AsBytes(tv.GetTensorDataAsSpan<double>()).ToArray(), // Float64
                6 => MemoryMarshal.AsBytes(tv.GetTensorDataAsSpan<int>()).ToArray(),     // Int32
                7 => MemoryMarshal.AsBytes(tv.GetTensorDataAsSpan<long>()).ToArray(),    // Int64
                3 => MemoryMarshal.AsBytes(tv.GetTensorDataAsSpan<sbyte>()).ToArray(),   // Int8
                2 => MemoryMarshal.AsBytes(tv.GetTensorDataAsSpan<byte>()).ToArray(),    // UInt8
                9 => MemoryMarshal.AsBytes(tv.GetTensorDataAsSpan<byte>()).ToArray(),    // Bool (1 byte/element)
                5 => MemoryMarshal.AsBytes(tv.GetTensorDataAsSpan<short>()).ToArray(),   // Int16
                4 => MemoryMarshal.AsBytes(tv.GetTensorDataAsSpan<ushort>()).ToArray(),  // UInt16
                12 => MemoryMarshal.AsBytes(tv.GetTensorDataAsSpan<uint>()).ToArray(),   // UInt32
                13 => MemoryMarshal.AsBytes(tv.GetTensorDataAsSpan<ulong>()).ToArray(),  // UInt64
                _ => throw new NotSupportedException(
                    $"MlirTensorCodec: tensor element type {td.DType} (proto {td.DType.ProtoTypeNum}) is not serialized in Phase 1.")
            };
        }

        public static TensorData FromRawBytes(long[] dims, DType dtype, byte[] data)
        {
            // Reject dtypes the writer would refuse, so the read side fails symmetrically rather
            // than producing a tensor that could not have been written.
            _ = dtype.ProtoTypeNum switch
            {
                1 or 11 or 6 or 7 or 3 or 2 or 9 or 5 or 4 or 12 or 13 => true,
                _ => throw new NotSupportedException(
                    $"MlirTensorCodec: tensor element type {dtype} (proto {dtype.ProtoTypeNum}) is not parsed in Phase 1.")
            };
            return TensorData.CreateFromRawBytes(new Shape(dims), dtype, data);
        }
    }
}

using System;
using System.Buffers;
using System.IO;

namespace Shorokoo.Core.Backends
{
    /// <summary>
    /// Builds a value from a stream through one bounded host buffer, a piece at a time, so the
    /// value's whole contents are never in host memory twice, nor ever in one managed array: in an
    /// execution provider's own memory, the shape a load onto the device takes
    /// (Shorokoo/Shorokoo#436), and in host memory of a backend's runtime, the shape a load of a
    /// tensor too large for any managed array takes. The mirror of <see cref="StagedReadBack"/>.
    ///
    /// <para>The pieces go through <see cref="IShorokooBackend.TryCopyHostToTensorRange"/> into a
    /// value allocated uninitialized. A backend that cannot write part of a value declines on the
    /// first piece, and the value is then built from its whole contents
    /// (<see cref="IShorokooBackend.CreateTensorInBackendMemory"/>, or
    /// <see cref="IShorokooBackend.CreateTensorFromRawBytes"/> in host memory): host memory holds one
    /// tensor at a time rather than one staging buffer, which is still one tensor rather than the
    /// whole state.</para>
    /// </summary>
    internal static class StagedUpload
    {
        /// <summary>
        /// A value of <paramref name="backend"/> of this element type and shape in its own memory,
        /// holding the next <paramref name="length"/> bytes of <paramref name="source"/>. Throws
        /// <see cref="EndOfStreamException"/> where the stream ends first; the value is released
        /// then, as on every failure.
        /// </summary>
        internal static IShorokooTensorValue Read(
            IShorokooBackend backend, ShorokooTensorElementType elementType, long[] shape, long length, Stream source)
            => Fill(backend, backend.CreateUninitializedTensorInBackendMemory(elementType, shape), length, source,
                whole => backend.CreateTensorInBackendMemory(elementType, whole, shape));

        /// <summary>
        /// <see cref="Read"/>, into host memory of <paramref name="backend"/>'s runtime
        /// (<see cref="IShorokooBackend.CreateUninitializedHostTensor"/>) whatever device it
        /// computes on.
        /// </summary>
        internal static IShorokooTensorValue ReadIntoHostMemory(
            IShorokooBackend backend, ShorokooTensorElementType elementType, long[] shape, long length, Stream source)
            => Fill(backend, backend.CreateUninitializedHostTensor(elementType, shape), length, source,
                whole => backend.CreateTensorFromRawBytes(elementType, whole, shape));

        /// <summary>
        /// Writes the next <paramref name="length"/> bytes of <paramref name="source"/> into
        /// <paramref name="value"/>, a value of <paramref name="backend"/> of at least that many bytes,
        /// a piece at a time, and hands it back — or, where the backend declines the first piece, the
        /// value <paramref name="whole"/> builds from the whole contents in its place. Takes the value
        /// over: it is released on every failure, and where it is replaced.
        /// </summary>
        internal static IShorokooTensorValue Fill(
            IShorokooBackend backend, IShorokooTensorValue value, long length, Stream source,
            Func<byte[], IShorokooTensorValue> whole)
        {
            if (length == 0) return value;
            byte[]? buffer = null;
            try
            {
                buffer = ArrayPool<byte>.Shared.Rent((int)Math.Min(length, StagedReadBack.StagingBytes));
                for (long offset = 0; offset < length;)
                {
                    var piece = buffer.AsSpan(0, (int)Math.Min(length - offset, Math.Min(buffer.Length, StagedReadBack.StagingBytes)));
                    source.ReadExactly(piece);
                    if (!backend.TryCopyHostToTensorRange(value, offset, piece))
                    {
                        // A backend answers for a value, not for a piece of it: one that wrote the
                        // first pieces and declines a later one has failed, not declined.
                        if (offset != 0)
                            throw new InvalidOperationException(
                                $"{backend.Description} wrote the first {offset} bytes of a tensor into "
                                + "its memory and then declined the rest.");
                        var contents = new byte[length];
                        piece.CopyTo(contents);
                        source.ReadExactly(contents.AsSpan(piece.Length));
                        var made = whole(contents);
                        value.Dispose();
                        return made;
                    }
                    offset += piece.Length;
                }
                return value;
            }
            catch
            {
                value.Dispose();
                throw;
            }
            finally
            {
                if (buffer is not null) ArrayPool<byte>.Shared.Return(buffer);
            }
        }
    }
}

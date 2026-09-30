using System;
using System.Buffers;
using System.IO;

namespace Shorokoo.Core.Backends
{
    /// <summary>
    /// Builds a value in an execution provider's own memory from a stream through one bounded host
    /// buffer, a piece at a time, so the value's whole contents are never in host memory at once —
    /// the shape a load onto the device takes (Shorokoo/Shorokoo#436). The mirror of
    /// <see cref="StagedReadBack"/>.
    ///
    /// <para>The pieces go through <see cref="IShorokooBackend.TryCopyHostToTensorRange"/> into a
    /// value allocated uninitialized. A backend that cannot write part of a value declines on the
    /// first piece, and the value is then built by
    /// <see cref="IShorokooBackend.CreateTensorInBackendMemory"/> from its whole contents: host
    /// memory holds one tensor at a time rather than one staging buffer, which is still one tensor
    /// rather than the whole state.</para>
    /// </summary>
    internal static class StagedUpload
    {
        /// <summary>
        /// A value of <paramref name="backend"/> of this element type and shape, holding the next
        /// <paramref name="length"/> bytes of <paramref name="source"/>. Throws
        /// <see cref="EndOfStreamException"/> where the stream ends first; the value is released
        /// then, as on every failure.
        /// </summary>
        internal static IShorokooTensorValue Read(
            IShorokooBackend backend, ShorokooTensorElementType elementType, long[] shape, long length, Stream source)
        {
            var value = backend.CreateUninitializedTensorInBackendMemory(elementType, shape);
            if (length == 0) return value;
            var buffer = ArrayPool<byte>.Shared.Rent((int)Math.Min(length, StagedReadBack.StagingBytes));
            try
            {
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
                                + "its execution provider's memory and then declined the rest.");
                        var whole = new byte[length];
                        piece.CopyTo(whole);
                        source.ReadExactly(whole.AsSpan(piece.Length));
                        var made = backend.CreateTensorInBackendMemory(elementType, whole, shape);
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
                ArrayPool<byte>.Shared.Return(buffer);
            }
        }
    }
}

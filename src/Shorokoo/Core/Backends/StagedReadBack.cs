using System;
using System.Buffers;
using System.IO;

namespace Shorokoo.Core.Backends
{
    /// <summary>
    /// Writes a value out of an execution provider's own memory to a stream through one bounded host
    /// buffer, a piece at a time, so the value's whole contents are never in host memory at once —
    /// the shape a save of a device-resident checkpoint takes (Shorokoo/Shorokoo#436).
    ///
    /// <para>The pieces come from <see cref="IShorokooBackend.TryCopyTensorRangeToHost"/>. A backend
    /// that cannot copy part of a value declines on the first piece, and the value then goes through
    /// <see cref="IShorokooBackend.CopyTensorToHost"/> whole: host memory holds one tensor at a time
    /// rather than one staging buffer, which is still one tensor rather than the whole state.</para>
    /// </summary>
    internal static class StagedReadBack
    {
        /// <summary>The most host memory one write stages at a time. Large enough that a copy's
        /// fixed cost is small beside its bytes; small enough to be negligible beside any state
        /// worth streaming.</summary>
        internal const int StagingBytes = 8 << 20;

        /// <summary>
        /// Writes <paramref name="length"/> bytes of <paramref name="value"/>, made by
        /// <paramref name="backend"/>, to <paramref name="destination"/>. The caller keeps the value
        /// alive and unconsumed for the length of the call.
        /// </summary>
        internal static void Write(
            IShorokooBackend backend, IShorokooTensorValue value, long length, Stream destination)
        {
            if (length == 0) return;
            var buffer = ArrayPool<byte>.Shared.Rent((int)Math.Min(length, StagingBytes));
            try
            {
                for (long offset = 0; offset < length;)
                {
                    var piece = buffer.AsSpan(0, (int)Math.Min(length - offset, Math.Min(buffer.Length, StagingBytes)));
                    if (!backend.TryCopyTensorRangeToHost(value, offset, piece))
                    {
                        // A backend answers for a value, not for a piece of it: one that copied the
                        // first pieces and declines a later one has failed, not declined.
                        if (offset != 0)
                            throw new InvalidOperationException(
                                $"{backend.Description} copied the first {offset} bytes of a tensor out of "
                                + "its execution provider's memory and then declined the rest.");
                        destination.Write(backend.CopyTensorToHost(value));
                        return;
                    }
                    destination.Write(piece);
                    offset += piece.Length;
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(buffer);
            }
        }
    }
}

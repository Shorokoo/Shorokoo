using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using Shorokoo.Core.Factory.IR;

namespace Shorokoo.Onnx
{
    /// <summary>
    /// Writes a <see cref="ModelProto"/> as protobuf to a stream with no bound on its size. The
    /// protobuf wire format gives every message a varint length, which holds any size, but
    /// protobuf-net builds a message whole in one buffer before it writes it, so it cannot write a
    /// model whose messages reach 2 GiB. Here each tensor's <c>raw_data</c> of at least
    /// <see cref="MinStreamedBytes"/> is set aside, a placeholder in its place, and protobuf-net
    /// serializes the model besides those payloads; the model is then written from that, each
    /// placeholder replaced by the payload it stands for and every message holding one given the
    /// length it has with the payload in it. The bytes written are the bytes protobuf-net writes
    /// for the model wherever it can write it at all. A tensor whose elements are carried beside
    /// its message (<see cref="TensorProto.Carried"/>) — a weight past what one array holds — is
    /// written with them as its <c>raw_data</c>, streamed from the attribute holding them.
    ///
    /// <para>Payloads are found where the importer reads tensors (as
    /// <see cref="OnnxExternalData.EnumerateAllTensors"/> walks them): graph initializers, sparse
    /// initializers, the tensors of node attributes, through nested graphs and function bodies.
    /// Each payload is written from where it lives, by the piece its source writes, so no buffer
    /// ever holds more than one message besides its payloads.</para>
    /// </summary>
    internal sealed class OnnxStreamingWriter
    {
        /// <summary>The smallest <c>raw_data</c> set aside rather than serialized with the rest of
        /// the model: below it, a placeholder saves nothing worth its walk.</summary>
        internal const int MinStreamedBytes = 1024;

        private const int WireVarint = 0, WireFixed64 = 1, WireLengthDelimited = 2, WireFixed32 = 5;
        private const int RawDataField = 9;
        private const int NonceLength = 16;
        private const int PlaceholderLength = NonceLength + sizeof(long);

        private enum Kind { Model, Graph, Node, Attribute, Function, Tensor, SparseTensor, Opaque }

        /// <summary>A tensor payload set aside: its length, and how its bytes are written.</summary>
        internal readonly record struct Payload(long Length, Action<Stream> WriteTo)
        {
            /// <summary>A payload held in one array.</summary>
            public static Payload Of(byte[] bytes) => new(bytes.LongLength, s => s.Write(bytes));
        }

        private readonly byte[] _skeleton;
        private readonly byte[] _nonce;
        private readonly List<Payload> _payloads;

        /// <summary>The length of each rewritten message, in the order the messages open: the
        /// measuring walk records it, the writing walk writes it as the message's length prefix.</summary>
        private readonly List<long> _sizes = [];

        /// <summary>The length of the model as written.</summary>
        internal long Length { get; }

        private OnnxStreamingWriter(byte[] skeleton, byte[] nonce, List<Payload> payloads)
        {
            _skeleton = skeleton;
            _nonce = nonce;
            _payloads = payloads;
            var found = new bool[payloads.Count];
            Length = Measure(Kind.Model, 0, skeleton.Length, found);
            if (Array.IndexOf(found, false) >= 0)
                throw new InvalidOperationException(
                    "A tensor payload set aside to be streamed was not found where the model's tensors are written.");
        }

        /// <summary>
        /// Prepares <paramref name="model"/> to be written. The model is left as it was given; the
        /// payloads set aside are held by reference, so they must not change before
        /// <see cref="WriteTo"/> writes them. Refuses, with <see cref="NotSupportedException"/>, a
        /// model whose protobuf besides its streamed payloads exceeds
        /// <paramref name="maxSkeletonBytes"/> (by default what one array holds) before any of it is written.
        /// </summary>
        internal static OnnxStreamingWriter Prepare(ModelProto model, long? maxSkeletonBytes = null)
        {
            if (model is null) throw new ArgumentNullException(nameof(model));
            long ceiling = maxSkeletonBytes ?? Array.MaxLength;

            var nonce = RandomNumberGenerator.GetBytes(NonceLength);
            var payloads = new List<Payload>();
            var setAside = new List<(TensorProto Tensor, byte[]? RawData)>();
            var seen = new HashSet<TensorProto>(ReferenceEqualityComparer.Instance);
            try
            {
                foreach (var tensor in OnnxExternalData.EnumerateAllTensors(model))
                {
                    Payload payload;
                    if (tensor.RawData is { Length: >= MinStreamedBytes } raw)
                        payload = Payload.Of(raw);
                    else if (CarriedPayload(tensor) is { } carried)
                        payload = new Payload(carried.ByteLength, carried.WriteTo);
                    else
                        continue;
                    if (!seen.Add(tensor))
                        continue;
                    var placeholder = new byte[PlaceholderLength];
                    nonce.CopyTo(placeholder, 0);
                    BitConverter.TryWriteBytes(placeholder.AsSpan(NonceLength), (long)payloads.Count);
                    setAside.Add((tensor, tensor.RawData));
                    payloads.Add(payload);
                    tensor.RawData = placeholder;
                }

                using var measured = ProtoBuf.Serializer.Measure(model);
                if (measured.Length > ceiling)
                    throw new NotSupportedException(
                        $"The ONNX model holds {measured.Length:N0} bytes besides its tensors' raw data, more than " +
                        $"the {ceiling:N0} bytes one protobuf message is built in. Only a tensor's raw_data " +
                        "is streamed: string tensors and typed data fields are held with the rest of the model.");
                var skeleton = new byte[measured.Length];
                using (var buffer = new MemoryStream(skeleton))
                    measured.Serialize(buffer);
                return new OnnxStreamingWriter(skeleton, nonce, payloads);
            }
            finally
            {
                foreach (var (tensor, raw) in setAside)
                    tensor.RawData = raw!;
            }
        }

        /// <summary>The attribute a tensor carries beside its message in place of its
        /// <c>raw_data</c> (<see cref="TensorProto.Carried"/>) — a weight past what one array
        /// holds — whose elements are written as that <c>raw_data</c>, or null. One declared
        /// external is the side file's, and strings have no flat layout.</summary>
        private static TensorAttribute? CarriedPayload(TensorProto tensor)
            => tensor is { RawData: null, Carried: { HasValues: true } carried }
                && tensor.data_location != TensorProto.DataLocation.External
                && !carried.DType.IsSameElementTypeAs(DType.Utf8)
                ? carried : null;

        /// <summary>Writes the model to <paramref name="destination"/>: <see cref="Length"/> bytes,
        /// each payload streamed from where it lives.</summary>
        internal void WriteTo(Stream destination)
        {
            var counted = new CountingWriteStream(destination);
            using (var output = new BufferedStream(counted, 1 << 16))
            {
                int slot = 0;
                Write(Kind.Model, 0, _skeleton.Length, output, ref slot);
                output.Flush();
            }
            if (counted.Count != Length)
                throw new InvalidOperationException(
                    $"The ONNX model was measured at {Length:N0} bytes but {counted.Count:N0} were written: " +
                    "a tensor payload changed while the model was written.");
        }

        private static Kind ChildOf(Kind parent, int field) => (parent, field) switch
        {
            (Kind.Model, 7) => Kind.Graph,
            (Kind.Model, 25) => Kind.Function,
            (Kind.Graph, 1) => Kind.Node,
            (Kind.Graph, 5) => Kind.Tensor,
            (Kind.Graph, 15) => Kind.SparseTensor,
            (Kind.Node, 5) => Kind.Attribute,
            (Kind.Attribute, 5) or (Kind.Attribute, 10) => Kind.Tensor,
            (Kind.Attribute, 6) or (Kind.Attribute, 11) => Kind.Graph,
            (Kind.Attribute, 22) or (Kind.Attribute, 23) => Kind.SparseTensor,
            (Kind.Function, 7) => Kind.Node,
            (Kind.SparseTensor, 1) or (Kind.SparseTensor, 2) => Kind.Tensor,
            _ => Kind.Opaque,
        };

        /// <summary>The length the message of <paramref name="kind"/> at
        /// [<paramref name="start"/>, <paramref name="end"/>) of the skeleton has as written,
        /// recording each child message's in <see cref="_sizes"/> and each payload met in
        /// <paramref name="found"/>.</summary>
        private long Measure(Kind kind, int start, int end, bool[] found)
        {
            long size = 0;
            int at = start;
            while (at < end)
            {
                int fieldStart = at;
                var (key, field, wire) = ReadKey(ref at);
                if (wire == WireLengthDelimited)
                {
                    int length = ReadLength(ref at, end);
                    int body = at;
                    at += length;
                    if (kind == Kind.Tensor && field == RawDataField && TryReadPlaceholder(body, length, out int index))
                    {
                        found[index] = true;
                        long payload = _payloads[index].Length;
                        size += VarintLength(key) + VarintLength((ulong)payload) + payload;
                        continue;
                    }
                    var child = ChildOf(kind, field);
                    if (child != Kind.Opaque)
                    {
                        int slot = _sizes.Count;
                        _sizes.Add(0);
                        long childSize = Measure(child, body, at, found);
                        _sizes[slot] = childSize;
                        size += VarintLength(key) + VarintLength((ulong)childSize) + childSize;
                        continue;
                    }
                }
                else
                    SkipValue(wire, ref at, end);
                size += at - fieldStart;
            }
            return size;
        }

        /// <summary>Writes the message <see cref="Measure"/> measured, in the same order.</summary>
        private void Write(Kind kind, int start, int end, Stream output, ref int slot)
        {
            int at = start;
            while (at < end)
            {
                int fieldStart = at;
                var (key, field, wire) = ReadKey(ref at);
                if (wire == WireLengthDelimited)
                {
                    int length = ReadLength(ref at, end);
                    int body = at;
                    at += length;
                    if (kind == Kind.Tensor && field == RawDataField && TryReadPlaceholder(body, length, out int index))
                    {
                        var payload = _payloads[index];
                        WriteVarint(output, key);
                        WriteVarint(output, (ulong)payload.Length);
                        payload.WriteTo(output);
                        continue;
                    }
                    var child = ChildOf(kind, field);
                    if (child != Kind.Opaque)
                    {
                        WriteVarint(output, key);
                        WriteVarint(output, (ulong)_sizes[slot++]);
                        Write(child, body, at, output, ref slot);
                        continue;
                    }
                }
                else
                    SkipValue(wire, ref at, end);
                output.Write(_skeleton, fieldStart, at - fieldStart);
            }
        }

        private bool TryReadPlaceholder(int at, int length, out int index)
        {
            index = -1;
            if (length != PlaceholderLength || !_skeleton.AsSpan(at, NonceLength).SequenceEqual(_nonce))
                return false;
            long value = BitConverter.ToInt64(_skeleton, at + NonceLength);
            if (value < 0 || value >= _payloads.Count)
                return false;
            index = (int)value;
            return true;
        }

        private (ulong Key, int Field, int Wire) ReadKey(ref int at)
        {
            var key = ReadVarint(ref at, _skeleton.Length);
            return (key, (int)(key >> 3), (int)(key & 7));
        }

        private int ReadLength(ref int at, int end)
        {
            var length = ReadVarint(ref at, end);
            if (length > (ulong)(end - at))
                throw new InvalidOperationException("The serialized ONNX model holds a field that runs past its message.");
            return (int)length;
        }

        private void SkipValue(int wire, ref int at, int end)
        {
            switch (wire)
            {
                case WireVarint:
                    ReadVarint(ref at, end);
                    return;
                case WireFixed64:
                    at += 8;
                    return;
                case WireFixed32:
                    at += 4;
                    return;
                default:
                    throw new InvalidOperationException($"The serialized ONNX model holds a field of wire type {wire}.");
            }
        }

        private ulong ReadVarint(ref int at, int end)
        {
            ulong value = 0;
            for (int shift = 0; shift < 64 && at < end; shift += 7)
            {
                byte b = _skeleton[at++];
                value |= (ulong)(b & 0x7F) << shift;
                if ((b & 0x80) == 0) return value;
            }
            throw new InvalidOperationException("The serialized ONNX model holds a malformed varint.");
        }

        private static void WriteVarint(Stream output, ulong value)
        {
            Span<byte> bytes = stackalloc byte[10];
            int length = 0;
            while (value >= 0x80)
            {
                bytes[length++] = (byte)(value | 0x80);
                value >>= 7;
            }
            bytes[length++] = (byte)value;
            output.Write(bytes[..length]);
        }

        private static int VarintLength(ulong value)
        {
            int length = 1;
            for (; value >= 0x80; value >>= 7) length++;
            return length;
        }

        /// <summary>A forward write-only stream passing every byte on and counting them.</summary>
        private sealed class CountingWriteStream(Stream inner) : Stream
        {
            public long Count { get; private set; }

            public override void Write(ReadOnlySpan<byte> buffer)
            {
                inner.Write(buffer);
                Count += buffer.Length;
            }

            public override void Write(byte[] buffer, int offset, int count) => Write(buffer.AsSpan(offset, count));
            public override void Flush() => inner.Flush();
            public override bool CanRead => false;
            public override bool CanSeek => false;
            public override bool CanWrite => true;
            public override long Length => throw new NotSupportedException();
            public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
            public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
            public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
            public override void SetLength(long value) => throw new NotSupportedException();
        }
    }
}

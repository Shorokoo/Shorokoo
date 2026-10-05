using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Runtime;

namespace Shorokoo.Onnx
{
    /// <summary>
    /// Reads a <see cref="ModelProto"/> from a stream read forward once, with no bound on the size of
    /// a tensor's <c>raw_data</c>: the reading counterpart of <see cref="OnnxStreamingWriter"/>. The
    /// protobuf wire format gives a field a varint length, which holds any size, but protobuf-net
    /// reads a bytes field into one array, so it cannot read a <c>raw_data</c> past what one array
    /// holds. Here the stream is walked at the wire level as it is read, and each tensor's
    /// <c>raw_data</c> of at least <see cref="MinSetAsideBytes"/> is read out of it into storage of
    /// its own, a placeholder in its place: an array, or — past what one array holds — a tensor in
    /// host memory read a piece at a time (<see cref="ComputeContext.ReadTensor"/>). protobuf-net
    /// parses what remains, each message holding a placeholder given the length it has with it, and
    /// each tensor then takes its payload back: the array as its <c>raw_data</c>, the tensor as the
    /// attribute it carries (<see cref="TensorProto.Carried"/>).
    ///
    /// <para>So each weight is held once, where the model keeps it, and nothing holds the model
    /// whole: what remains besides the payloads set aside is held once more while protobuf-net
    /// parses it. Tensors are found where the writer streams them (graph initializers, sparse
    /// initializers, the tensors and sparse tensors of node attributes, through nested graphs and
    /// function bodies). A <c>raw_data</c> past what one array holds is read where its tensor's <c>dims</c> and
    /// <c>data_type</c> come first and account for exactly its bytes, as every writer of a
    /// <c>.srk</c> payload lays a tensor out; any other is refused as malformed.</para>
    /// </summary>
    internal sealed class OnnxStreamingReader
    {
        /// <summary>The smallest <c>raw_data</c> set aside rather than parsed with the rest of the
        /// model: below it, a placeholder saves nothing worth its walk.</summary>
        internal const int MinSetAsideBytes = OnnxStreamingWriter.MinStreamedBytes;

        /// <summary>Test hook: the largest <c>raw_data</c> read into an array, in place of what one
        /// array holds; a larger one is read into host memory and carried as an attribute.
        /// Thread-scoped, so a hook installed by one parallel test is invisible to every other
        /// thread; still reset it in a <c>finally</c>.</summary>
        [ThreadStatic]
        internal static long? HeldPayloadThresholdInjection;

        private const int WireVarint = 0, WireFixed64 = 1, WireLengthDelimited = 2, WireStartGroup = 3, WireEndGroup = 4, WireFixed32 = 5;
        private const int DimsField = 1, DataTypeField = 2, RawDataField = 9;
        private const int NonceLength = 16;
        private const int PlaceholderLength = NonceLength + sizeof(long);

        private enum Kind { Model, Graph, Node, Attribute, Function, Tensor, SparseTensor, Opaque }

        /// <summary>What a tensor's fields read so far say of its layout.</summary>
        private sealed class Layout
        {
            public int DataType;
            public readonly List<long> Dims = [];
        }

        private readonly Forward _source;
        private readonly string _origin;
        private readonly long _heldPast;
        private readonly byte[] _nonce = RandomNumberGenerator.GetBytes(NonceLength);

        /// <summary>The model as it is read, each payload set aside replaced by its placeholder and
        /// each rewritten message's length prefix left out: those are written into it at
        /// <see cref="_lengthsAt"/> once known.</summary>
        private readonly MemoryStream _skeleton = new();
        private readonly List<(long At, ulong Length)> _lengthsAt = [];
        private long _lengthBytes;

        /// <summary>The payloads set aside, in the order their placeholders number them: an array,
        /// or a host tensor to be moved into the attribute its tensor carries.</summary>
        private readonly List<object> _payloads = [];

        private OnnxStreamingReader(Stream source, string origin)
        {
            _source = new Forward(source);
            _origin = origin;
            _heldPast = Math.Min(HeldPayloadThresholdInjection ?? Array.MaxLength, Array.MaxLength);
        }

        /// <summary>
        /// The model <paramref name="source"/> holds from where it stands to its end, read forward
        /// once: at most <paramref name="maxLength"/> bytes, which bounds every length a field of it
        /// declares, so a field claiming more than the source can hold is refused before anything
        /// is allocated for it. A truncated or malformed model throws <see cref="EndOfStreamException"/> or
        /// <see cref="ProtoBuf.ProtoException"/>, as parsing it whole does; a model holding more
        /// than one array besides its payloads, or a <c>raw_data</c> past one array that its
        /// tensor's layout does not account for, throws <see cref="InvalidDataException"/> naming
        /// <paramref name="origin"/>.
        /// </summary>
        internal static ModelProto ReadModel(Stream source, long maxLength, string origin)
        {
            ArgumentNullException.ThrowIfNull(source);
            var reader = new OnnxStreamingReader(source, origin);
            try
            {
                reader.Message(Kind.Model, maxLength, 0, null);
                var model = OnnxProtobuf.ReadModel(new MemoryStream(reader.Assemble(), writable: false));
                reader.Restore(model);
                return model;
            }
            finally
            {
                // Each host tensor a tensor took is spent by its move; any other — a parse that
                // failed, or a payload a later field of its tensor replaced — is released here.
                foreach (var payload in reader._payloads)
                    if (payload is TensorData held && !held.IsDisposed)
                        held.Delete();
            }
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

        /// <summary>Reads the message of <paramref name="kind"/> running to <paramref name="end"/>
        /// (the model's own runs to the end of the stream, at most there), <paramref name="depth"/> levels below
        /// the model, into the skeleton; <paramref name="layout"/> gathers a tensor's.</summary>
        private void Message(Kind kind, long end, int depth, Layout? layout)
        {
            while (kind == Kind.Model ? !_source.AtEnd() : _source.Consumed < end)
            {
                var key = ReadVarint(end);
                if (key >> 3 is 0 or > int.MaxValue)
                    throw new ProtoBuf.ProtoException($"Invalid field number {key >> 3} at byte {_source.Consumed}.");
                int field = (int)(key >> 3), wire = (int)(key & 7);
                if (wire != WireLengthDelimited)
                {
                    WriteVarint(key);
                    if (layout is not null && wire == WireVarint && field is DimsField or DataTypeField)
                    {
                        var value = ReadVarint(end);
                        WriteVarint(value);
                        if (field == DimsField) layout.Dims.Add(unchecked((long)value));
                        else layout.DataType = unchecked((int)value);
                    }
                    else
                        CopyValue(key, wire, end, depth);
                    continue;
                }

                long length = ReadLength(end);
                long bodyEnd = _source.Consumed + length;
                if (layout is not null && field == RawDataField && length >= MinSetAsideBytes)
                {
                    SetAside(key, length, layout);
                    continue;
                }
                var child = ChildOf(kind, field);
                WriteVarint(key);
                if (child == Kind.Opaque)
                {
                    WriteVarint((ulong)length);
                    if (layout is not null && field == DimsField)
                        ReadPackedDims(length, layout);
                    else
                        Copy(length);
                    continue;
                }
                if (depth + 1 > OnnxProtobuf.MaxDepth)
                    throw new ProtoBuf.ProtoException(
                        $"'{_origin}': the ONNX model nests a message {depth + 1} levels deep at byte {_source.Consumed}; " +
                        $"a model is read to a depth of {OnnxProtobuf.MaxDepth}.");
                int slot = _lengthsAt.Count;
                _lengthsAt.Add((_skeleton.Length, 0));
                long bodyStart = _skeleton.Length, lengthBytesBefore = _lengthBytes;
                Message(child, bodyEnd, depth + 1, child == Kind.Tensor ? new Layout() : null);
                var bodyLength = (ulong)(_skeleton.Length - bodyStart + _lengthBytes - lengthBytesBefore);
                _lengthsAt[slot] = (bodyStart, bodyLength);
                _lengthBytes += VarintLength(bodyLength);
            }
        }

        /// <summary>Reads the <paramref name="length"/> bytes of a tensor's <c>raw_data</c> into
        /// storage of their own, and writes the placeholder numbering them in their place.</summary>
        private void SetAside(ulong key, long length, Layout layout)
        {
            if (length > _heldPast)
            {
                var dtype = (DType)layout.DataType;
                if (OnnxExternalData.TryGetExpectedByteLength(new TensorProto { data_type = layout.DataType, Dims = [.. layout.Dims] }) != length)
                    throw new InvalidDataException(
                        $"'{_origin}': a tensor's raw_data of {length:N0} bytes, more than one array holds, does not " +
                        "follow dims and a data_type that account for exactly its bytes, so it cannot be read.");
                _payloads.Add(ComputeContext.Host.ReadTensor(new Shape([.. layout.Dims]), dtype, _source));
            }
            else
            {
                var bytes = GC.AllocateUninitializedArray<byte>((int)length);
                _source.ReadExactly(bytes);
                _payloads.Add(bytes);
            }

            Span<byte> placeholder = stackalloc byte[PlaceholderLength];
            _nonce.CopyTo(placeholder);
            BitConverter.TryWriteBytes(placeholder[NonceLength..], (long)(_payloads.Count - 1));
            WriteVarint(key);
            WriteVarint(PlaceholderLength);
            Write(placeholder);
        }

        /// <summary>The skeleton with every length prefix written into its place.</summary>
        private byte[] Assemble()
        {
            long total = _skeleton.Length + _lengthBytes;
            if (total > Array.MaxLength)
                throw new InvalidDataException(
                    $"'{_origin}': the ONNX model holds {total:N0} bytes besides the raw_data of at least " +
                    $"{MinSetAsideBytes:N0} bytes it sets aside, more than one protobuf message is read into.");
            var output = GC.AllocateUninitializedArray<byte>((int)total);
            var skeleton = _skeleton.GetBuffer();
            int from = 0, to = 0;
            foreach (var (at, length) in _lengthsAt)
            {
                skeleton.AsSpan(from, (int)at - from).CopyTo(output.AsSpan(to));
                to += (int)at - from;
                from = (int)at;
                for (var value = length; ; value >>= 7)
                {
                    if (value < 0x80)
                    {
                        output[to++] = (byte)value;
                        break;
                    }
                    output[to++] = (byte)(value | 0x80);
                }
            }
            skeleton.AsSpan(from, (int)_skeleton.Length - from).CopyTo(output.AsSpan(to));
            return output;
        }

        /// <summary>
        /// Hands each tensor of <paramref name="model"/> whose <c>raw_data</c> of at least
        /// <see cref="MinSetAsideBytes"/> is exactly its dims' worth of a flat data type that
        /// <c>raw_data</c> as the attribute it carries (<see cref="TensorProto.Carried"/>), over
        /// the array itself, so an import takes each weight where it lies rather than copying it.
        /// Only for a model the caller owns outright, as <see cref="ReadModel"/> returns it:
        /// nothing else may hold its arrays. Any other tensor keeps its <c>raw_data</c>, for the
        /// import to read or refuse.
        /// </summary>
        internal static void CarryRawData(ModelProto model)
        {
            foreach (var tensor in OnnxExternalData.EnumerateAllTensors(model))
            {
                if (tensor is not { RawData: { Length: >= MinSetAsideBytes } raw, Carried: null }
                    || tensor.data_location == TensorProto.DataLocation.External
                    || OnnxExternalData.TryGetExpectedByteLength(tensor) != raw.Length
                    || !OnnxExternalData.HasFlatBuffer((DType)tensor.data_type))
                    continue;
                Shape shape = tensor.Dims is { Length: > 0 } dims ? dims : (long[])[];
                tensor.Carried = TensorData.NewHostTensor(shape, (DType)tensor.data_type, raw).MoveToAttribute();
                tensor.RawData = null!;
            }
        }

        /// <summary>Gives each tensor holding a placeholder the payload it stands for.</summary>
        private void Restore(ModelProto model)
        {
            foreach (var tensor in OnnxExternalData.EnumerateAllTensors(model))
            {
                if (tensor.RawData is not { Length: PlaceholderLength } raw || !raw.AsSpan(0, NonceLength).SequenceEqual(_nonce))
                    continue;
                long index = BitConverter.ToInt64(raw, NonceLength);
                if (index < 0 || index >= _payloads.Count)
                    continue;
                switch (_payloads[(int)index])
                {
                    case byte[] bytes:
                        tensor.RawData = bytes;
                        break;
                    case TensorData held:
                        tensor.RawData = null!;
                        tensor.Carried = held.MoveToAttribute();
                        break;
                }
            }
        }

        /// <summary>Copies a packed <c>dims</c> field's <paramref name="length"/> bytes as they
        /// stand, each varint however it is spelled, gathering the dims they hold.</summary>
        private void ReadPackedDims(long length, Layout layout)
        {
            long end = _source.Consumed + length;
            while (_source.Consumed < end)
                layout.Dims.Add(unchecked((long)ReadVarint(end, copy: true)));
        }

        /// <summary>Copies the value of a field of wire type <paramref name="wire"/> as it stands: a
        /// group whole, through its end.</summary>
        private void CopyValue(ulong key, int wire, long end, int depth)
        {
            switch (wire)
            {
                case WireVarint:
                    WriteVarint(ReadVarint(end));
                    return;
                case WireFixed64:
                    RequireWithin(end, 8);
                    Copy(8);
                    return;
                case WireFixed32:
                    RequireWithin(end, 4);
                    Copy(4);
                    return;
                case WireLengthDelimited:
                    long length = ReadLength(end);
                    WriteVarint((ulong)length);
                    Copy(length);
                    return;
                case WireStartGroup:
                    if (depth + 1 > OnnxProtobuf.MaxDepth)
                        throw new ProtoBuf.ProtoException(
                            $"'{_origin}': the ONNX model nests a group {depth + 1} levels deep at byte {_source.Consumed}.");
                    while (true)
                    {
                        var inner = ReadVarint(end);
                        int innerWire = (int)(inner & 7);
                        WriteVarint(inner);
                        if (innerWire == WireEndGroup)
                        {
                            if (inner >> 3 != key >> 3)
                                throw new ProtoBuf.ProtoException(
                                    $"A group of field {key >> 3} ends as field {inner >> 3} at byte {_source.Consumed}.");
                            return;
                        }
                        CopyValue(inner, innerWire, end, depth + 1);
                    }
                default:
                    throw new ProtoBuf.ProtoException($"Unexpected wire type {wire} at byte {_source.Consumed}.");
            }
        }

        private ulong ReadVarint(long end, bool copy = false)
        {
            ulong value = 0;
            for (int shift = 0; shift < 64; shift += 7)
            {
                RequireWithin(end, 1);
                int b = _source.ReadByte();
                if (b < 0) throw new EndOfStreamException();
                if (copy) Write([(byte)b]);
                value |= (ulong)(b & 0x7F) << shift;
                if ((b & 0x80) == 0) return value;
            }
            throw new ProtoBuf.ProtoException($"Malformed varint at byte {_source.Consumed}.");
        }

        private long ReadLength(long end)
        {
            var length = ReadVarint(end);
            if (length > (ulong)(end - _source.Consumed))
                throw new EndOfStreamException(
                    $"A field of {length} bytes at byte {_source.Consumed} runs past the end of its message.");
            return (long)length;
        }

        private void RequireWithin(long end, long count)
        {
            if (end - _source.Consumed < count) throw new EndOfStreamException();
        }

        private void Copy(long length)
        {
            Span<byte> chunk = stackalloc byte[4096];
            while (length > 0)
            {
                var piece = chunk[..(int)Math.Min(length, chunk.Length)];
                _source.ReadExactly(piece);
                Write(piece);
                length -= piece.Length;
            }
        }

        private void WriteVarint(ulong value)
        {
            Span<byte> bytes = stackalloc byte[10];
            int length = 0;
            while (value >= 0x80)
            {
                bytes[length++] = (byte)(value | 0x80);
                value >>= 7;
            }
            bytes[length++] = (byte)value;
            Write(bytes[..length]);
        }

        private void Write(ReadOnlySpan<byte> bytes)
        {
            if (_skeleton.Length + _lengthBytes + bytes.Length > Array.MaxLength)
                throw new InvalidDataException(
                    $"'{_origin}': the ONNX model holds more than {Array.MaxLength:N0} bytes besides the raw_data " +
                    $"of at least {MinSetAsideBytes:N0} bytes it sets aside, more than one protobuf message is read into.");
            _skeleton.Write(bytes);
        }

        private static int VarintLength(ulong value)
        {
            int length = 1;
            for (; value >= 0x80; value >>= 7) length++;
            return length;
        }

        /// <summary>A forward read-only stream over <paramref name="inner"/>, read through a buffer of
        /// its own and counting the bytes it hands out.</summary>
        private sealed class Forward(Stream inner) : Stream
        {
            private readonly byte[] _buffer = new byte[1 << 16];
            private int _at, _count;

            /// <summary>The bytes handed out so far.</summary>
            public long Consumed { get; private set; }

            /// <summary>Whether the stream read has nothing more to give.</summary>
            public bool AtEnd() => _at == _count && !Fill();

            private bool Fill()
            {
                _at = 0;
                _count = inner.Read(_buffer);
                return _count > 0;
            }

            public override int ReadByte()
            {
                if (_at == _count && !Fill()) return -1;
                Consumed++;
                return _buffer[_at++];
            }

            public override int Read(Span<byte> destination)
            {
                if (destination.IsEmpty) return 0;
                if (_at == _count)
                {
                    if (destination.Length >= _buffer.Length)
                    {
                        int direct = inner.Read(destination);
                        Consumed += direct;
                        return direct;
                    }
                    if (!Fill()) return 0;
                }
                int count = Math.Min(destination.Length, _count - _at);
                _buffer.AsSpan(_at, count).CopyTo(destination);
                _at += count;
                Consumed += count;
                return count;
            }

            public override int Read(byte[] buffer, int offset, int count) => Read(buffer.AsSpan(offset, count));
            public override bool CanRead => true;
            public override bool CanSeek => false;
            public override bool CanWrite => false;
            public override long Length => throw new NotSupportedException();
            public override long Position { get => Consumed; set => throw new NotSupportedException(); }
            public override void Flush() { }
            public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
            public override void SetLength(long value) => throw new NotSupportedException();
            public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        }
    }
}

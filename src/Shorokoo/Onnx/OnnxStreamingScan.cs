using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Shorokoo.Core.Factory.IR;

namespace Shorokoo.Onnx
{
    /// <summary>
    /// Reads the <see cref="ModelProto"/> of an <c>.onnx</c> file without holding its weights: the
    /// file is walked at the protobuf wire level, and each tensor whose payload is at least
    /// <see cref="MinReferencedBytes"/> gets, in place of the payload, an external-data reference
    /// to where that payload lies in the same file — ONNX's external-data mechanism names any file
    /// in the model's directory, the model's own included. The scan holds the rest of the model;
    /// the weights are read later, each straight into its tensor (<see cref="OnnxExternalData"/>),
    /// so host memory is bounded by what the model is besides its weights, not by the file.
    ///
    /// <para>Tensors are found where the importer reads them: graph initializers and the tensors of
    /// node attributes, through nested graphs and function bodies. A payload is referenced when it
    /// is <c>raw_data</c>, or packed <c>float_data</c> of a float tensor or <c>double_data</c> of a
    /// double one (the same little-endian bytes), exactly the shape's worth of bytes in one piece,
    /// the tensor holds no other piece of a payload, and it carries no external data of its own —
    /// no <c>external_data</c> entry, and no <c>data_location</c> but <c>DEFAULT</c>. Any other
    /// payload stays inline, each field where it stands, as it is read without
    /// the scan: the varint-coded ones (<c>int32_data</c>, <c>int64_data</c>, <c>uint64_data</c>),
    /// strings, and a tensor's bytes that disagree with its shape. Everything else in the file is
    /// kept as it is. A sparse initializer, which the importer does not read, is refused.</para>
    ///
    /// <para>The file is walked twice. The first walk writes nothing and measures each message it
    /// rewrites; the second writes the rewritten model, each length prefix from the first walk,
    /// into one buffer of exactly its size. What stays inline is so held once, however deeply it is
    /// nested.</para>
    /// </summary>
    internal sealed class OnnxStreamingScan
    {
        /// <summary>The smallest payload referenced rather than kept inline: below it, a read of
        /// its own costs more than the bytes it saves.</summary>
        internal const int MinReferencedBytes = 1024;

        private const int WireVarint = 0, WireFixed64 = 1, WireLengthDelimited = 2, WireFixed32 = 5;
        private const int RawDataField = 9, FloatDataField = 4, DoubleDataField = 10;
        private const int ExternalDataField = 13, DataLocationField = 14;
        private const int FloatType = 1, StringType = 8, DoubleType = 11;

        private enum Kind { Model, Graph, Node, Attribute, Function, Tensor, Opaque }

        private readonly Stream _file;
        private readonly string _path;
        private readonly string _location;

        /// <summary>The size of each rewritten message, in the order the messages open: the first
        /// walk records it, the second writes it as the message's length prefix.</summary>
        private readonly List<long> _sizes = [];
        private int _nextSize;

        /// <summary>The rewritten model, null during the first walk.</summary>
        private byte[]? _output;
        private int _written;

        private OnnxStreamingScan(Stream file, string path)
        {
            _file = file;
            _path = path;
            _location = Path.GetFileName(path);
        }

        /// <summary>Test hook: invoked with the model's path once its scan is done, before any of
        /// its weights is read. Thread-scoped, so a hook installed by one parallel test is invisible
        /// to every other thread; still reset it in a <c>finally</c>.</summary>
        [ThreadStatic]
        internal static Action<string>? ScannedInjection;

        /// <summary>The model file at <paramref name="filePath"/>, opened for its scan
        /// (<see cref="ReadModel"/>) and for the read of the weights the scan references in it
        /// (<see cref="OnnxExternalData.LoadIntoModel"/>): the one handle serves both, so the
        /// weights read are those of the file scanned, whatever replaces it on disk meanwhile.</summary>
        internal static FileStream Open(string filePath)
            => new(filePath, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16);

        /// <summary>The model in <paramref name="file"/>, its large tensor payloads referenced in
        /// place rather than read. A truncated or malformed file throws
        /// <see cref="EndOfStreamException"/> or <see cref="ProtoBuf.ProtoException"/>, and one
        /// nested <see cref="OnnxProtobuf.MaxDepth"/> levels deep or more throws
        /// <see cref="InvalidOperationException"/>, as parsing it whole does.</summary>
        internal static ModelProto ReadModel(FileStream file)
        {
            var model = new OnnxStreamingScan(file, file.Name).Scan();
            ScannedInjection?.Invoke(file.Name);
            using var stream = new MemoryStream(model, writable: false);
            return OnnxProtobuf.ReadModel(stream);
        }

        /// <summary>The refusal of a model with a sparse initializer, which the importer does not
        /// read: it would otherwise leave every node reading it without an input.</summary>
        internal static InvalidDataException SparseInitializerRefusal(string origin)
            => new($"{origin}: the ONNX graph carries a sparse initializer, which Shorokoo's importer " +
                "does not read. Store it as a dense initializer.");

        private static Kind ChildOf(Kind parent, int field) => (parent, field) switch
        {
            (Kind.Model, 7) => Kind.Graph,
            (Kind.Model, 25) => Kind.Function,
            (Kind.Graph, 1) => Kind.Node,
            (Kind.Graph, 5) => Kind.Tensor,
            (Kind.Node, 5) => Kind.Attribute,
            (Kind.Attribute, 5) or (Kind.Attribute, 10) => Kind.Tensor,
            (Kind.Attribute, 6) or (Kind.Attribute, 11) => Kind.Graph,
            (Kind.Function, 7) => Kind.Node,
            _ => Kind.Opaque,
        };

        private byte[] Scan()
        {
            long end = _file.Length;
            _file.Position = 0;
            long size = Message(Kind.Model, end, 0);
            if (size > Array.MaxLength)
                throw new InvalidDataException(
                    $"'{_path}': the ONNX model holds {size} bytes besides the weights it references, " +
                    "more than one protobuf message can.");

            _output = GC.AllocateUninitializedArray<byte>((int)size);
            _nextSize = 0;
            _file.Position = 0;
            Message(Kind.Model, end, 0);
            if (_written != size)
                throw new IOException($"'{_path}' changed while it was read.");
            return _output;
        }

        private long Message(Kind kind, long end, int depth)
        {
            long size = 0;
            while (_file.Position < end)
            {
                var (key, field, wire) = ReadKey(end);
                if (kind == Kind.Graph && field == 15)
                    throw SparseInitializerRefusal($"'{_path}'");
                if (wire != WireLengthDelimited || ChildOf(kind, field) is Kind.Opaque)
                {
                    size += CopyField(key, wire, end);
                    continue;
                }
                long length = ReadLength(end);
                size += Child(key, ChildOf(kind, field), _file.Position + length, depth + 1);
            }
            return size;
        }

        /// <summary>The message of <paramref name="kind"/> running to <paramref name="end"/>,
        /// <paramref name="depth"/> levels below the model, with its key and length prefix. Refused
        /// where the parse of the rewritten model would refuse it (<see cref="OnnxProtobuf.MaxDepth"/>),
        /// before the walk descends any further: the walk recurses as the messages nest, so a file
        /// nested deep enough would otherwise overflow the stack.</summary>
        private long Child(ulong key, Kind kind, long end, int depth)
        {
            if (depth >= OnnxProtobuf.MaxDepth)
                throw new InvalidOperationException(
                    $"'{_path}': the ONNX model nests a message {depth} levels deep at byte {_file.Position}; " +
                    $"a model is read to a depth of {OnnxProtobuf.MaxDepth - 1}.");
            int slot = _nextSize++;
            if (_output is null) _sizes.Add(0);
            long size = WriteVarint(key);
            if (_output is not null) size += WriteVarint((ulong)_sizes[slot]);
            long body = kind == Kind.Tensor ? Tensor(end) : Message(kind, end, depth);
            if (_output is null) size += WriteVarint((ulong)(_sizes[slot] = body));
            return size + body;
        }

        /// <summary>
        /// A tensor, rewritten: each field written where it stands, but for a payload referenced in
        /// place of being copied (see the class). The fields the decision rests on may stand
        /// anywhere in the message, so it is walked once to read them and again to write it. What is
        /// kept so keeps its order, which matters for a payload written in pieces: a repeated field
        /// may be written partly packed and partly unpacked, and its elements are read in the order
        /// its pieces stand.
        /// </summary>
        private long Tensor(long end)
        {
            long start = _file.Position;
            int dataType = 0;
            var dims = new List<long>();
            bool external = false;
            int payloads = 0;
            (int Field, long Offset, long Length)? flat = null;
            while (_file.Position < end)
            {
                var (_, field, wire) = ReadKey(end);
                if (field is RawDataField or FloatDataField or DoubleDataField)
                {
                    payloads++;
                    if (wire != WireLengthDelimited)
                    {
                        Skip(wire, end);
                        continue;
                    }
                    long length = ReadLength(end);
                    flat = (field, _file.Position, length);
                    _file.Seek(length, SeekOrigin.Current);
                    continue;
                }
                external |= field == ExternalDataField;
                if (field == DataLocationField && wire == WireVarint)
                    external |= ReadVarint(end) == (ulong)TensorProto.DataLocation.External;
                else if (field == 2 && wire == WireVarint)
                    dataType = unchecked((int)ReadVarint(end));
                else if (field == 1 && wire == WireVarint)
                    dims.Add(unchecked((long)ReadVarint(end)));
                else if (field == 1 && wire == WireLengthDelimited)
                {
                    long packedEnd = ReadLength(end) + _file.Position;
                    while (_file.Position < packedEnd)
                        dims.Add(unchecked((long)ReadVarint(packedEnd)));
                }
                else
                    Skip(wire, end);
            }

            bool referenced = payloads == 1 && !external && flat is { } only
                && only.Length >= MinReferencedBytes
                && (only.Field == RawDataField ? dataType != StringType
                    : only.Field == FloatDataField ? dataType == FloatType : dataType == DoubleType)
                && only.Length == OnnxExternalData.TryGetExpectedByteLength(
                    new TensorProto { data_type = dataType, Dims = [.. dims] });

            _file.Position = start;
            long size = 0;
            while (_file.Position < end)
            {
                var (key, field, wire) = ReadKey(end);
                if (referenced && field is RawDataField or FloatDataField or DoubleDataField)
                    Skip(wire, end);
                else
                    size += CopyField(key, wire, end);
            }
            if (!referenced) return size;

            var (_, offset, bytes) = flat!.Value;
            size += WriteExternalEntry(OnnxExternalData.LocationKey, _location);
            size += WriteExternalEntry(OnnxExternalData.OffsetKey, offset.ToString(CultureInfo.InvariantCulture));
            size += WriteExternalEntry(OnnxExternalData.LengthKey, bytes.ToString(CultureInfo.InvariantCulture));
            size += WriteVarint(DataLocationField << 3 | WireVarint);
            size += WriteVarint((ulong)TensorProto.DataLocation.External);
            return size;
        }

        private long WriteExternalEntry(string key, string value)
        {
            long entry = StringLength(1, key) + StringLength(2, value);
            return WriteVarint(ExternalDataField << 3 | WireLengthDelimited) + WriteVarint((ulong)entry)
                + WriteString(1, key) + WriteString(2, value);
        }

        private static long StringLength(int field, string value)
        {
            int count = Encoding.UTF8.GetByteCount(value);
            return VarintLength((ulong)(field << 3 | WireLengthDelimited)) + VarintLength((ulong)count) + count;
        }

        private long WriteString(int field, string value)
        {
            int count = Encoding.UTF8.GetByteCount(value);
            long size = WriteVarint((ulong)(field << 3 | WireLengthDelimited)) + WriteVarint((ulong)count) + count;
            if (_output is not null)
                _written += Encoding.UTF8.GetBytes(value, _output.AsSpan(_written));
            return size;
        }

        private (ulong Key, int Field, int Wire) ReadKey(long end)
        {
            var key = ReadVarint(end);
            if (key >> 3 is 0 or > int.MaxValue)
                throw new ProtoBuf.ProtoException($"Invalid field number {key >> 3} at byte {_file.Position}.");
            return (key, (int)(key >> 3), (int)(key & 7));
        }

        private long CopyField(ulong key, int wire, long end)
        {
            long size = WriteVarint(key);
            switch (wire)
            {
                case WireVarint:
                    return size + WriteVarint(ReadVarint(end));
                case WireFixed64:
                    RequireWithin(end, 8);
                    return size + Copy(8);
                case WireFixed32:
                    RequireWithin(end, 4);
                    return size + Copy(4);
                case WireLengthDelimited:
                    long length = ReadLength(end);
                    return size + WriteVarint((ulong)length) + Copy(length);
                default:
                    throw new ProtoBuf.ProtoException($"Unsupported wire type {wire} at byte {_file.Position}.");
            }
        }

        /// <summary>Reads past a field's value, writing nothing.</summary>
        private void Skip(int wire, long end)
        {
            switch (wire)
            {
                case WireVarint:
                    ReadVarint(end);
                    return;
                case WireFixed64:
                    RequireWithin(end, 8);
                    _file.Seek(8, SeekOrigin.Current);
                    return;
                case WireFixed32:
                    RequireWithin(end, 4);
                    _file.Seek(4, SeekOrigin.Current);
                    return;
                case WireLengthDelimited:
                    long length = ReadLength(end);
                    _file.Seek(length, SeekOrigin.Current);
                    return;
                default:
                    throw new ProtoBuf.ProtoException($"Unsupported wire type {wire} at byte {_file.Position}.");
            }
        }

        private ulong ReadVarint(long end)
        {
            ulong value = 0;
            for (int shift = 0; shift < 64; shift += 7)
            {
                RequireWithin(end, 1);
                int b = _file.ReadByte();
                if (b < 0) throw new EndOfStreamException();
                value |= (ulong)(b & 0x7F) << shift;
                if ((b & 0x80) == 0) return value;
            }
            throw new ProtoBuf.ProtoException($"Malformed varint at byte {_file.Position}.");
        }

        private long ReadLength(long end)
        {
            var length = ReadVarint(end);
            if (length > (ulong)(end - _file.Position))
                throw new EndOfStreamException(
                    $"A field of {length} bytes at byte {_file.Position} runs past the end of its message.");
            return (long)length;
        }

        private void RequireWithin(long end, long count)
        {
            if (end - _file.Position < count) throw new EndOfStreamException();
        }

        /// <summary>The next <paramref name="length"/> bytes of the file, written as they are —
        /// skipped by the first walk, which only measures.</summary>
        private long Copy(long length)
        {
            if (_output is null)
            {
                _file.Seek(length, SeekOrigin.Current);
                return length;
            }
            _file.ReadExactly(_output.AsSpan(_written, (int)length));
            _written += (int)length;
            return length;
        }

        /// <summary>Writes <paramref name="value"/> as a varint — the first walk only measures
        /// it — and returns its length.</summary>
        private long WriteVarint(ulong value)
        {
            if (_output is null)
                return VarintLength(value);
            int start = _written;
            while (value >= 0x80)
            {
                _output[_written++] = (byte)(value | 0x80);
                value >>= 7;
            }
            _output[_written++] = (byte)value;
            return _written - start;
        }

        private static int VarintLength(ulong value)
        {
            int length = 1;
            for (; value >= 0x80; value >>= 7) length++;
            return length;
        }
    }
}

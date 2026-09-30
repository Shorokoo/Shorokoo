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
    /// double one (the same little-endian bytes), exactly the shape's worth of bytes, and the tensor
    /// carries no external data of its own. Any other payload stays inline, as it is read without
    /// the scan: the varint-coded ones (<c>int32_data</c>, <c>int64_data</c>, <c>uint64_data</c>),
    /// strings, and a tensor's bytes that disagree with its shape. Everything else in the file is
    /// kept as it is. A sparse initializer, which the importer does not read, is refused.</para>
    /// </summary>
    internal sealed class OnnxStreamingScan
    {
        /// <summary>The smallest payload referenced rather than kept inline: below it, a read of
        /// its own costs more than the bytes it saves.</summary>
        internal const int MinReferencedBytes = 1024;

        private const int WireVarint = 0, WireFixed64 = 1, WireLengthDelimited = 2, WireFixed32 = 5;
        private const int RawDataField = 9, FloatDataField = 4, DoubleDataField = 10;
        private const int FloatType = 1, StringType = 8, DoubleType = 11;

        private enum Kind { Model, Graph, Node, Attribute, Function, Tensor, Opaque }

        private readonly Stream _file;
        private readonly string _path;
        private readonly string _location;
        private readonly byte[] _copyBuffer = new byte[81920];

        private OnnxStreamingScan(Stream file, string path)
        {
            _file = file;
            _path = path;
            _location = Path.GetFileName(path);
        }

        /// <summary>The model at <paramref name="filePath"/>, its large tensor payloads referenced
        /// in place rather than read. A truncated or malformed file throws
        /// <see cref="EndOfStreamException"/> or <see cref="ProtoBuf.ProtoException"/>, as parsing
        /// it whole does.</summary>
        internal static ModelProto ReadModel(string filePath)
        {
            byte[] model;
            using (var file = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16))
                model = new OnnxStreamingScan(file, filePath).Message(Kind.Model, file.Length);
            using var stream = new MemoryStream(model, writable: false);
            return ProtoBuf.Serializer.Deserialize<ModelProto>(stream);
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

        private byte[] Message(Kind kind, long end)
        {
            var output = new MemoryStream();
            while (_file.Position < end)
            {
                var (key, field, wire) = ReadKey(end);
                if (kind == Kind.Graph && field == 15)
                    throw SparseInitializerRefusal($"'{_path}'");
                if (wire != WireLengthDelimited || ChildOf(kind, field) is Kind.Opaque)
                {
                    CopyField(key, wire, end, output);
                    continue;
                }
                long length = ReadLength(end);
                long childEnd = _file.Position + length;
                var child = ChildOf(kind, field) == Kind.Tensor ? Tensor(childEnd) : Message(ChildOf(kind, field), childEnd);
                WriteVarint(output, key);
                WriteVarint(output, (ulong)child.Length);
                output.Write(child);
            }
            return output.ToArray();
        }

        private byte[] Tensor(long end)
        {
            var output = new MemoryStream();
            int dataType = 0;
            var dims = new List<long>();
            bool external = false;
            var payloads = new List<(ulong Key, int Field, long Offset, long Length)>();
            while (_file.Position < end)
            {
                var (key, field, wire) = ReadKey(end);
                if (wire == WireLengthDelimited && field is RawDataField or FloatDataField or DoubleDataField)
                {
                    long length = ReadLength(end);
                    payloads.Add((key, field, _file.Position, length));
                    _file.Seek(length, SeekOrigin.Current);
                    continue;
                }
                external |= field is 13 or 14;
                if (field == 2 && wire == WireVarint)
                {
                    var value = ReadVarint(end);
                    dataType = unchecked((int)value);
                    WriteVarint(output, key);
                    WriteVarint(output, value);
                }
                else if (field == 1 && wire == WireVarint)
                {
                    var value = ReadVarint(end);
                    dims.Add(unchecked((long)value));
                    WriteVarint(output, key);
                    WriteVarint(output, value);
                }
                else if (field == 1 && wire == WireLengthDelimited)
                {
                    long length = ReadLength(end);
                    long packedEnd = _file.Position + length;
                    var packed = new MemoryStream();
                    while (_file.Position < packedEnd)
                    {
                        var value = ReadVarint(packedEnd);
                        dims.Add(unchecked((long)value));
                        WriteVarint(packed, value);
                    }
                    WriteVarint(output, key);
                    WriteVarint(output, (ulong)packed.Length);
                    packed.WriteTo(output);
                }
                else
                {
                    CopyField(key, wire, end, output);
                }
            }

            if (payloads.Count == 1 && !external && payloads[0] is var (_, only, offset, bytes)
                && bytes >= MinReferencedBytes
                && (only == RawDataField ? dataType != StringType
                    : only == FloatDataField ? dataType == FloatType : dataType == DoubleType)
                && bytes == OnnxExternalData.TryGetExpectedByteLength(
                    new TensorProto { data_type = dataType, Dims = [.. dims] }))
            {
                WriteExternalEntry(output, OnnxExternalData.LocationKey, _location);
                WriteExternalEntry(output, OnnxExternalData.OffsetKey, offset.ToString(CultureInfo.InvariantCulture));
                WriteExternalEntry(output, OnnxExternalData.LengthKey, bytes.ToString(CultureInfo.InvariantCulture));
                WriteVarint(output, 14 << 3 | WireVarint);
                WriteVarint(output, (ulong)TensorProto.DataLocation.External);
                return output.ToArray();
            }

            foreach (var (key, _, start, length) in payloads)
            {
                _file.Position = start;
                WriteVarint(output, key);
                WriteVarint(output, (ulong)length);
                Copy(length, output);
            }
            _file.Position = end;
            return output.ToArray();
        }

        private static void WriteExternalEntry(MemoryStream output, string key, string value)
        {
            var entry = new MemoryStream();
            WriteString(entry, 1, key);
            WriteString(entry, 2, value);
            WriteVarint(output, 13 << 3 | WireLengthDelimited);
            WriteVarint(output, (ulong)entry.Length);
            entry.WriteTo(output);
        }

        private static void WriteString(MemoryStream output, int field, string value)
        {
            var bytes = Encoding.UTF8.GetBytes(value);
            WriteVarint(output, (ulong)(field << 3 | WireLengthDelimited));
            WriteVarint(output, (ulong)bytes.Length);
            output.Write(bytes);
        }

        private (ulong Key, int Field, int Wire) ReadKey(long end)
        {
            var key = ReadVarint(end);
            if (key >> 3 is 0 or > int.MaxValue)
                throw new ProtoBuf.ProtoException($"Invalid field number {key >> 3} at byte {_file.Position}.");
            return (key, (int)(key >> 3), (int)(key & 7));
        }

        private void CopyField(ulong key, int wire, long end, MemoryStream output)
        {
            WriteVarint(output, key);
            switch (wire)
            {
                case WireVarint:
                    WriteVarint(output, ReadVarint(end));
                    break;
                case WireFixed64:
                    RequireWithin(end, 8);
                    Copy(8, output);
                    break;
                case WireFixed32:
                    RequireWithin(end, 4);
                    Copy(4, output);
                    break;
                case WireLengthDelimited:
                    long length = ReadLength(end);
                    WriteVarint(output, (ulong)length);
                    Copy(length, output);
                    break;
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

        private void Copy(long length, Stream output)
        {
            while (length > 0)
            {
                int chunk = (int)Math.Min(length, _copyBuffer.Length);
                _file.ReadExactly(_copyBuffer, 0, chunk);
                output.Write(_copyBuffer, 0, chunk);
                length -= chunk;
            }
        }

        private static void WriteVarint(Stream output, ulong value)
        {
            while (value >= 0x80)
            {
                output.WriteByte((byte)(value | 0x80));
                value >>= 7;
            }
            output.WriteByte((byte)value);
        }
    }
}

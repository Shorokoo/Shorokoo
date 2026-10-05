using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Shorokoo.Core.Utils;
using Shorokoo.Runtime;

namespace Shorokoo
{
    /// <summary>
    /// A tensor in a graph's <i>description</i>: an operator's tensor-valued attribute.
    ///
    /// <para>A shape, a dtype and the elements, and nothing else. It is not
    /// <see cref="IDisposable"/>, belongs to no <see cref="ComputeContext"/>, holds no runtime
    /// value and has no storage to release — because a description has no lifetime. The same
    /// description is the same description on every machine, and a graph that captured one can be
    /// built, serialized and read anywhere, whether or not a backend is deployed.</para>
    ///
    /// <para>Immutable, and shared by every graph that captured it. That is what makes the two
    /// conversions asymmetric: <see cref="TensorData.MoveToAttribute"/> <b>moves</b> — the tensor
    /// surrenders its bytes and is spent, so a 165 M-parameter checkpoint binds without copying a
    /// byte — while <see cref="CopyToTensorData()"/> <b>copies</b>, because a mutable tensor over an
    /// attribute's bytes would be a way to edit a description through the back door.</para>
    ///
    /// <para>Of any size. Elements one managed array holds are kept in one; more than that — a
    /// weight past 2 GiB — are kept in host memory of the backend <see cref="ComputeContext.Default"/>
    /// runs on, in a tensor the attribute alone holds and never hands out, and are read and written
    /// a piece at a time (<see cref="CopyToTensorData()"/>, and every save and export of a model
    /// holding it). Such an attribute has no byte view: <see cref="Bytes"/> and
    /// <see cref="Elements{V}"/> refuse it, since no span reaches that far.</para>
    /// </summary>
    public sealed class TensorAttribute
    {
        private readonly byte[]? _bytes;
        private readonly string[]? _values;
        // The elements where one managed array cannot hold them: a host tensor nobody else names,
        // which lives exactly as long as this attribute and is released by its value's finalizer
        // once the attribute is collected.
        private readonly TensorData? _held;

        private TensorAttribute(
            Shape shape, DType dtype, byte[]? bytes, string[]? values, DType? storageDType = null,
            TensorData? held = null)
        {
            Shape = shape;
            DType = dtype;
            StorageDType = storageDType ?? dtype;
            _bytes = bytes;
            _values = values;
            _held = held;
        }

        /// <summary>The attribute's shape.</summary>
        public Shape Shape { get; }

        /// <summary>The element data type.</summary>
        public DType DType { get; }

        /// <summary>
        /// The dtype the bytes are actually laid out at. The same as <see cref="DType"/> except for
        /// a generic placeholder (<c>DType.GenericType1</c>..<c>8</c>), which names the type
        /// parameter this literal stands for and says nothing about what it holds — a
        /// <see cref="TensorData"/> carried that in its runtime value, and an attribute has no
        /// runtime value to carry it in.
        /// </summary>
        internal DType StorageDType { get; }

        /// <summary>
        /// Whether the elements are here. False for a weights-elided attribute — a parameter whose
        /// model definition was saved without its weights, which carries dtype and shape and no
        /// values at all until a checkpoint is bound back onto it.
        /// </summary>
        public bool HasValues => _bytes is not null || _values is not null || _held is not null;

        /// <summary>
        /// Whether the elements are more bytes than one managed array holds, so that they have no
        /// byte view and are read and written a piece at a time.
        /// </summary>
        internal bool PastOneArray => _held is not null;

        /// <summary>
        /// The bytes the elements take up laid out flat — what <see cref="WriteTo"/> writes —
        /// measured, never read. Zero for a string attribute, whose elements are variable-length.
        /// </summary>
        /// <exception cref="InvalidOperationException">The values were elided.</exception>
        internal long ByteLength
            => _held?.ByteCount ?? _bytes?.LongLength ?? (_values is not null ? 0 : throw ValuesElided());

        /// <summary>
        /// The elements as raw bytes. The span is a window onto the attribute's own array, which
        /// is immutable and lives as long as the attribute, so nothing has to be copied out of it
        /// to be read safely.
        /// </summary>
        /// <exception cref="InvalidOperationException">The values were elided, or the dtype is
        /// <see cref="DType.Utf8"/>, whose elements are variable-length.</exception>
        /// <exception cref="NotSupportedException">The elements are more bytes than one span
        /// reaches; <see cref="CopyToTensorData()"/> reads them.</exception>
        public ReadOnlySpan<byte> Bytes => BytesArray;

        /// <summary>
        /// The elements read as <typeparamref name="V"/>, which must be the dtype's storage type.
        /// Same window, same lifetime rule, same refusals as <see cref="Bytes"/>.
        /// </summary>
        public ReadOnlySpan<V> Elements<V>() where V : unmanaged
            => MemoryMarshal.Cast<byte, V>(BytesArray);

        /// <summary>
        /// The elements of a <see cref="DType.Utf8"/> attribute, in row-major order. Strings are
        /// variable-length and reference-typed, so they have no byte view — this is their
        /// <see cref="Bytes"/>.
        /// </summary>
        /// <exception cref="InvalidOperationException">The values were elided, or this is not a
        /// string attribute.</exception>
        public IReadOnlyList<string> Values => _values ?? throw (
            HasValues
                ? new InvalidOperationException(
                    $"Attribute {this} holds bytes, not strings. Read them with {nameof(Bytes)} or "
                    + $"{nameof(Elements)}<V>().")
                : ValuesElided());

        /// <summary>
        /// An attribute of <paramref name="shape"/> and <paramref name="dtype"/> over a copy of
        /// <paramref name="bytes"/>. The copy is what keeps the attribute immutable; a caller with
        /// a tensor to give away wants <see cref="TensorData.MoveToAttribute"/>, which moves.
        /// </summary>
        public static TensorAttribute Create(Shape shape, DType dtype, ReadOnlySpan<byte> bytes)
            => new(shape, dtype, bytes.ToArray(), null);

        /// <summary>
        /// An attribute of <paramref name="shape"/> over a copy of <paramref name="values"/>, at
        /// the dtype <typeparamref name="V"/> stands for — the description counterpart of
        /// <c>Globals.TensorData(dims, vals)</c>, and what every typed graph literal is built with.
        ///
        /// <para>A shape, a dtype and the bytes is all an attribute is, so this goes straight to
        /// the bytes: no tensor is built on the way, and so no backend is resolved. A
        /// program that only describes a model and exports it therefore needs no deployed
        /// runtime.</para>
        ///
        /// <para>Too few values is an error; a surplus is not, and is ignored — the same asymmetry
        /// <see cref="HostTensorData{T}.From{V}"/> keeps, because the node-definition tables hand
        /// over a buffer longer than the shape covers.</para>
        /// </summary>
        /// <exception cref="ArgumentException"><paramref name="values"/> does not cover
        /// <paramref name="shape"/>.</exception>
        public static TensorAttribute Create<V>(Shape shape, params V[] values) where V : unmanaged
            => new(shape, OnnxUtils.GetDType<V>(), PackBytes(shape, values), null);

        /// <summary>An attribute of <paramref name="shape"/> over a copy of
        /// <paramref name="values"/>, at <see cref="DType.Utf8"/>. Same coverage rule as
        /// <see cref="Create{V}"/>.</summary>
        /// <exception cref="ArgumentException"><paramref name="values"/> does not cover
        /// <paramref name="shape"/>.</exception>
        public static TensorAttribute Create(Shape shape, params string[] values)
        {
            ArgumentNullException.ThrowIfNull(values);
            var required = checked((int)shape.Count);
            if (values.Length < required)
                throw new ArgumentException(
                    $"Supplied data of {values.Length} strings is less than shape size {required} "
                    + "strings.", nameof(values));
            return new TensorAttribute(shape, DType.Utf8, null, values[..required]);
        }

        /// <summary>
        /// A literal standing for a generic type parameter: declared at <paramref name="dtype"/>,
        /// which is a placeholder (<c>DType.GenericType1</c>..<c>8</c>), while its elements are
        /// laid out at <typeparamref name="V"/>'s width and recorded in <see cref="StorageDType"/>.
        ///
        /// <para>The placeholder describes no byte layout, so nothing could be read through it
        /// alone — the width the elements were written at has to travel with them until type
        /// inference replaces the placeholder with a real dtype.</para>
        /// </summary>
        /// <exception cref="ArgumentException"><paramref name="values"/> does not cover
        /// <paramref name="shape"/>.</exception>
        internal static TensorAttribute CreateStandIn<V>(Shape shape, DType dtype, params V[] values)
            where V : unmanaged
            => new(shape, dtype, PackBytes(shape, values), null, OnnxUtils.GetDType<V>());

        private static byte[] PackBytes<V>(Shape shape, V[] values) where V : unmanaged
        {
            ArgumentNullException.ThrowIfNull(values);
            var required = checked((int)shape.Count * Unsafe.SizeOf<V>());
            var supplied = MemoryMarshal.AsBytes(values.AsSpan());
            if (supplied.Length < required)
                throw new ArgumentException(
                    $"Supplied data of {supplied.Length} bytes is less than shape size {required} "
                    + "bytes.", nameof(values));
            return supplied[..required].ToArray();
        }

        /// <summary>
        /// A dtype-and-shape-true attribute carrying no values — what a parameter's slot holds in a
        /// model definition saved without its weights. Every read of its elements throws until a
        /// checkpoint is bound back onto it.
        /// </summary>
        public static TensorAttribute WithoutValues(Shape shape, DType dtype)
            => new(shape, dtype, null, null);

        /// <summary>The attribute over <paramref name="bytes"/> itself, which it takes as its own
        /// storage rather than copying. Internal because the caller has to be one that will never
        /// write to the array again.</summary>
        internal static TensorAttribute OverBytes(
            Shape shape, DType dtype, byte[] bytes, DType? storageDType = null)
            => new(shape, dtype, bytes ?? throw new ArgumentNullException(nameof(bytes)), null,
                   storageDType);

        /// <summary>The attribute over <paramref name="values"/> itself, not a copy of them; see
        /// <see cref="OverBytes"/>.</summary>
        internal static TensorAttribute OverStrings(Shape shape, string[] values)
            => new(shape, DType.Utf8, null, values ?? throw new ArgumentNullException(nameof(values)));

        /// <summary>
        /// The attribute over <paramref name="held"/> — a tensor in host memory of more bytes than
        /// one managed array holds, which it takes as its own storage. Internal because the caller
        /// has to be one that hands the tensor over: nothing else may name it again, since the
        /// attribute is immutable and never releases it but by being collected.
        /// </summary>
        internal static TensorAttribute OverHeld(Shape shape, DType dtype, TensorData held, DType? storageDType = null)
            => new(shape, dtype, null, null, storageDType, held ?? throw new ArgumentNullException(nameof(held)));

        /// <summary>The same elements at <paramref name="dtype"/>. The bytes are shared, which
        /// costs nothing and is safe: both attributes are immutable.</summary>
        internal TensorAttribute WithDType(DType dtype)
            => new(Shape, dtype, _bytes, _values, StorageDType, _held);

        /// <summary>The array itself, for the one test that can see a move did not copy.</summary>
        internal byte[] BytesArray => _bytes ?? throw (
            _values is not null
                ? new InvalidOperationException(
                    $"Attribute {this} holds strings. Their elements are variable-length and "
                    + $"reference-typed, so there is no flat buffer to span over. Read them with "
                    + $"{nameof(Values)}.")
            : _held is not null
                ? new NotSupportedException(
                    $"Attribute {this} holds {ByteLength} bytes, more than one managed array or span "
                    + $"reaches, so it has no byte view. {nameof(CopyToTensorData)}() reads it into a "
                    + "tensor a piece at a time.")
                : ValuesElided());

        /// <summary>
        /// The host tensor holding the elements of an attribute <see cref="PastOneArray"/> — for a
        /// compiled session to read them where they are — or null. The attribute's own: a caller
        /// reads it and keeps the attribute reachable while anything does, and never ends it.
        /// </summary>
        internal TensorData? Held => _held;

        /// <summary>
        /// Writes the elements, laid out flat, to <paramref name="destination"/>: the attribute's
        /// own array in one write, or a held tensor a piece at a time
        /// (<see cref="TensorData.WriteContentTo"/>), never whole in a managed array.
        /// </summary>
        /// <exception cref="InvalidOperationException">The values were elided, or the dtype is
        /// <see cref="DType.Utf8"/>.</exception>
        internal void WriteTo(Stream destination)
        {
            ArgumentNullException.ThrowIfNull(destination);
            if (_held is not null) _held.WriteContentTo(destination);
            else destination.Write(BytesArray);
        }

        /// <summary>
        /// The elements, laid out flat, as a stream read forward: over the attribute's own array,
        /// or a piece at a time out of a held tensor.
        /// </summary>
        /// <exception cref="InvalidOperationException">The values were elided, or the dtype is
        /// <see cref="DType.Utf8"/>.</exception>
        internal Stream OpenRead()
            => _held is not null ? _held.OpenContentStream() : new MemoryStream(BytesArray, writable: false);

        /// <summary>
        /// Whether <paramref name="other"/> holds the same elements: the same strings, or the same
        /// bytes — compared a piece at a time where either is past one array. Shape and dtype are
        /// the caller's to compare.
        /// </summary>
        internal bool SameElements(TensorAttribute other)
        {
            ArgumentNullException.ThrowIfNull(other);
            if (ReferenceEquals(this, other) || (_held is not null && ReferenceEquals(_held, other._held))) return true;
            if (_values is not null || other._values is not null)
                return _values is not null && other._values is not null && _values.AsSpan().SequenceEqual(other._values);
            if (_held is null && other._held is null) return BytesArray.AsSpan().SequenceEqual(other.BytesArray);
            if (ByteLength != other.ByteLength) return false;

            using var mine = OpenRead();
            using var theirs = other.OpenRead();
            var (a, b) = (new byte[Core.Backends.StagedReadBack.StagingBytes], new byte[Core.Backends.StagedReadBack.StagingBytes]);
            for (long left = ByteLength; left > 0;)
            {
                var count = (int)Math.Min(left, a.Length);
                mine.ReadExactly(a, 0, count);
                theirs.ReadExactly(b, 0, count);
                if (!a.AsSpan(0, count).SequenceEqual(b.AsSpan(0, count))) return false;
                left -= count;
            }
            return true;
        }

        /// <summary>
        /// A <see cref="TensorData"/> holding a copy of these elements, in the framework's own host
        /// memory — or, for elements past one managed array, in host memory of the backend that
        /// holds them (of the one <see cref="ComputeContext.Default"/> runs on, where the runtime
        /// that made them was not recorded), copied a piece at a time.
        ///
        /// <para>A copy, always. This attribute is immutable and every graph that captured it holds
        /// the same one, so a tensor sharing its bytes would be a way to edit a description through
        /// the back door. This is the cold direction — exporting serializes into a protobuf, which
        /// copies regardless, and a checkpoint save reads the run's outputs rather than the
        /// attribute.</para>
        /// </summary>
        /// <exception cref="InvalidOperationException">The values were elided.</exception>
        public TensorData CopyToTensorData() => CopyToTensorData(atStorageDType: false);

        /// <summary>
        /// The copy, read at <see cref="StorageDType"/> rather than <see cref="DType"/> — the form
        /// the conversions need, since a generic placeholder dtype describes no byte layout and
        /// nothing can be read through it.
        /// </summary>
        internal TensorData CopyToTensorData(bool atStorageDType)
        {
            if (_values is not null)
                return TensorData.NewHostStringTensor(Shape, [.. _values]);
            if (_held is not null)
            {
                // Through one bounded buffer into host memory of the backend the held tensor is in,
                // since no managed array holds the copy either -- or, for a tensor whose producer was
                // not recorded, which allocates nothing, of the one ComputeContext.Default runs on.
                var backend = _held.AllocatingBackend is Core.Backends.UnrecordedBackend
                    ? ComputeContext.Default.ResolvedBackend
                    : _held.AllocatingBackend;
                using var source = _held.OpenContentStream();
                return TensorData.Create(Shape, atStorageDType ? StorageDType : DType,
                    Core.Backends.StagedUpload.ReadIntoHostMemory(
                        backend, (Core.Backends.ShorokooTensorElementType)(int)StorageDType, (long[])Shape, ByteLength, source),
                    backend);
            }
            return OnnxUtils.CreateHostTensorData(
                Shape, atStorageDType ? StorageDType : DType, BytesArray.AsSpan().ToArray());
        }

        /// <summary>"shape:dtype" diagnostic string, as <see cref="TensorData.ToString"/> gives.</summary>
        public override string ToString() => $"{Shape}:{DType}";

        private InvalidOperationException ValuesElided() => new(
            $"Attribute {this} carries dtype and shape only — its values were elided when the "
            + "model definition was saved without its weights. Bind the checkpoint's weights "
            + "(Persistence.Load) before reading parameter values.");
    }
}

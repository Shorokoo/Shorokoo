using System;
using System.Collections.Generic;
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
    /// </summary>
    public sealed class TensorAttribute
    {
        private readonly byte[]? _bytes;
        private readonly string[]? _values;

        private TensorAttribute(
            Shape shape, DType dtype, byte[]? bytes, string[]? values, DType? storageDType = null)
        {
            Shape = shape;
            DType = dtype;
            StorageDType = storageDType ?? dtype;
            _bytes = bytes;
            _values = values;
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
        public bool HasValues => _bytes is not null || _values is not null;

        /// <summary>
        /// The elements as raw bytes. The span is a window onto the attribute's own array, which
        /// is immutable and lives as long as the attribute, so nothing has to be copied out of it
        /// to be read safely.
        /// </summary>
        /// <exception cref="InvalidOperationException">The values were elided, or the dtype is
        /// <see cref="DType.String"/>, whose elements are variable-length.</exception>
        public ReadOnlySpan<byte> Bytes => BytesArray;

        /// <summary>
        /// The elements read as <typeparamref name="V"/>, which must be the dtype's storage type.
        /// Same window, same lifetime rule as <see cref="Bytes"/>.
        /// </summary>
        public ReadOnlySpan<V> Elements<V>() where V : unmanaged
            => MemoryMarshal.Cast<byte, V>(BytesArray);

        /// <summary>
        /// The elements of a <see cref="DType.String"/> attribute, in row-major order. Strings are
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

        /// <summary>An attribute of <paramref name="shape"/> over a copy of
        /// <paramref name="values"/>, at <see cref="DType.String"/>.</summary>
        public static TensorAttribute Create(Shape shape, string[] values)
        {
            ArgumentNullException.ThrowIfNull(values);
            return new TensorAttribute(shape, DType.String, null, [.. values]);
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
            => new(shape, DType.String, null, values ?? throw new ArgumentNullException(nameof(values)));

        /// <summary>The same elements at <paramref name="dtype"/>. The bytes are shared, which
        /// costs nothing and is safe: both attributes are immutable.</summary>
        internal TensorAttribute WithDType(DType dtype)
            => new(Shape, dtype, _bytes, _values, StorageDType);

        /// <summary>The array itself, for the one test that can see a move did not copy.</summary>
        internal byte[] BytesArray => _bytes ?? throw (
            _values is not null
                ? new InvalidOperationException(
                    $"Attribute {this} holds strings. Their elements are variable-length and "
                    + $"reference-typed, so there is no flat buffer to span over. Read them with "
                    + $"{nameof(Values)}.")
                : ValuesElided());

        /// <summary>
        /// A <see cref="TensorData"/> holding a copy of these elements, in the framework's own host
        /// memory and belonging to <see cref="ComputeContext.Host"/>.
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
                return TensorData.NewHostStringTensor(Shape, [.. _values], ComputeContext.Host);
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

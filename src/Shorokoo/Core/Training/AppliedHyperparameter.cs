using System;
using System.Runtime.InteropServices;
using Shorokoo.Core.Backends;

namespace Shorokoo
{
    /// <summary>
    /// The value one optimizer hyperparameter had in one training step — what the optimizer update of
    /// that step actually read — as an immutable host value. A checkpoint carries one per
    /// hyperparameter in <see cref="TrainingCheckpoint.AppliedHyperparameters"/>, and every entry of a
    /// <see cref="TrainingHistory"/> one per hyperparameter in
    /// <see cref="TrainingHistoryEntry.AppliedHyperparameters"/>.
    ///
    /// <para>The value is copied into managed memory when the step records it, so it belongs to no
    /// backend, needs no disposal and outlives every tensor the step used. It has the
    /// hyperparameter's declared dtype (<see cref="TrainingRig.HyperparameterDTypes"/>) and the shape
    /// the rig was built at (<see cref="TrainingRig.HyperparameterShapes"/>): any supported
    /// hyperparameter dtype, and any shape. <see cref="ToDouble"/> and <see cref="ToSingle"/> read a
    /// single-element value; <see cref="ToArray{T}"/> and <see cref="ToTensorData"/> give every element
    /// of any value. Two values are equal when their dtype, shape and element bytes are.</para>
    /// </summary>
    public sealed class AppliedHyperparameter : IEquatable<AppliedHyperparameter>
    {
        // The element bytes as the host lays them out, read and written with MemoryMarshal: little-
        // endian, like every platform .NET runs on and like the safetensors files they are saved to.
        private readonly byte[] _bytes;
        private readonly long[] _dims;

        private AppliedHyperparameter(DType dtype, long[] dims, byte[] bytes)
        {
            DType = dtype;
            _dims = dims;
            _bytes = bytes;
        }

        /// <summary>
        /// A copy of <paramref name="value"/>'s elements, read back to the host first where it is not
        /// host-readable. <paramref name="value"/> itself is left as it was.
        /// </summary>
        internal static AppliedHyperparameter Of(TensorData value)
        {
            ArgumentNullException.ThrowIfNull(value);
            var host = value.ToHost();
            try
            {
                return new AppliedHyperparameter(value.DType, [.. value.Shape.Dims], host.CopyRawMemory());
            }
            finally
            {
                if (!ReferenceEquals(host, value)) host.Dispose();
            }
        }

        /// <summary>
        /// A value over <paramref name="bytes"/>, which it takes as its own: the caller hands over a
        /// buffer nothing else holds. The bytes must cover <paramref name="dims"/> at
        /// <paramref name="dtype"/>'s whole-byte element size.
        /// </summary>
        internal static AppliedHyperparameter FromRawBytes(DType dtype, long[] dims, byte[] bytes)
        {
            long count = 1;
            foreach (var d in dims) count *= d;
            if (dtype.EncodingBitCount <= 0 || dtype.EncodingBitCount % 8 != 0
                || bytes.LongLength != count * (dtype.EncodingBitCount / 8))
                throw new ArgumentException(
                    $"{bytes.Length} byte(s) do not hold a {dtype} value of shape [{string.Join(", ", dims)}].",
                    nameof(bytes));
            return new AppliedHyperparameter(dtype, dims, bytes);
        }

        /// <summary>The value's element bytes, row-major, for a writer to copy from.</summary>
        internal ReadOnlySpan<byte> RawBytes => _bytes;

        /// <summary>The value's dims, for a comparison that allocates nothing.</summary>
        internal ReadOnlySpan<long> Dims => _dims;

        /// <summary>The value's element type: the hyperparameter's declared dtype.</summary>
        public DType DType { get; }

        /// <summary>The value's shape — empty for a scalar. A new <see cref="Shorokoo.Shape"/> on every
        /// read, so changing it changes nothing here.</summary>
        public Shape Shape => new([.. _dims]);

        /// <summary>The number of elements: <c>1</c> for a scalar.</summary>
        public long ElementCount => _bytes.Length / (DType.EncodingBitCount / 8);

        /// <summary>
        /// The single element as a <see cref="double"/>, converted from whatever dtype the value has; a
        /// <c>bool</c> reads as <c>1</c> or <c>0</c>. A <c>float32</c> learning rate reads back exactly.
        /// </summary>
        /// <exception cref="InvalidOperationException">The value has more or fewer than one
        /// element.</exception>
        public double ToDouble()
        {
            if (ElementCount != 1)
                throw new InvalidOperationException(
                    $"ToDouble() reads a single-element value; this {DType} value has shape [{string.Join(", ", _dims)}]. " +
                    "Read its elements with ToArray<T>() or ToTensorData().");
            ReadOnlySpan<byte> b = _bytes;
            if (DType == DType.Float32) return MemoryMarshal.Read<float>(b);
            if (DType == DType.Float64) return MemoryMarshal.Read<double>(b);
            if (DType == DType.Float16) return (float)MemoryMarshal.Read<Float16>(b);
            if (DType == DType.BFloat16) return (float)MemoryMarshal.Read<BFloat16>(b);
            if (DType == DType.Int8) return MemoryMarshal.Read<sbyte>(b);
            if (DType == DType.Int16) return MemoryMarshal.Read<short>(b);
            if (DType == DType.Int32) return MemoryMarshal.Read<int>(b);
            if (DType == DType.Int64) return MemoryMarshal.Read<long>(b);
            if (DType == DType.UInt8) return b[0];
            if (DType == DType.UInt16) return MemoryMarshal.Read<ushort>(b);
            if (DType == DType.UInt32) return MemoryMarshal.Read<uint>(b);
            if (DType == DType.UInt64) return MemoryMarshal.Read<ulong>(b);
            if (DType == DType.Bool) return b[0] != 0 ? 1d : 0d;
            throw new InvalidOperationException($"'{DType}' is not a hyperparameter dtype.");
        }

        /// <summary>The single element as a <see cref="float"/>: <see cref="ToDouble"/>, narrowed.</summary>
        /// <exception cref="InvalidOperationException">The value has more or fewer than one
        /// element.</exception>
        public float ToSingle() => (float)ToDouble();

        /// <summary>
        /// Every element, in row-major order, in a new array the caller owns. <typeparamref name="T"/>
        /// is the dtype's own storage type (<see cref="DType.ToPrimitiveType"/>): <see cref="float"/>
        /// for <c>float32</c>, <see cref="long"/> for <c>int64</c>, <see cref="bool"/> for <c>bool</c>,
        /// <see cref="Float16"/> for <c>float16</c>, and so on.
        /// </summary>
        /// <exception cref="InvalidOperationException"><typeparamref name="T"/> is not the value's
        /// storage type.</exception>
        public T[] ToArray<T>() where T : unmanaged
        {
            if (typeof(T) != DType.ToPrimitiveType())
                throw new InvalidOperationException(
                    $"This hyperparameter value is {DType}, stored as {DType.ToPrimitiveType().Name}; " +
                    $"it cannot be read as {typeof(T).Name}.");
            return MemoryMarshal.Cast<byte, T>(_bytes).ToArray();
        }

        /// <summary>
        /// A new host <see cref="TensorData"/> holding a copy of the value, at its dtype and shape. The
        /// tensor is the caller's: feeding it to a run, or disposing it, leaves this value unchanged.
        /// </summary>
        public TensorData ToTensorData() => TensorData.CreateFromRawBytes(Shape, DType, _bytes);

        /// <inheritdoc />
        public bool Equals(AppliedHyperparameter? other)
            => other is not null
               && DType == other.DType
               && _dims.AsSpan().SequenceEqual(other._dims)
               && _bytes.AsSpan().SequenceEqual(other._bytes);

        /// <inheritdoc />
        public override bool Equals(object? obj) => Equals(obj as AppliedHyperparameter);

        /// <inheritdoc />
        public override int GetHashCode()
        {
            var hash = new HashCode();
            hash.Add(DType);
            foreach (var d in _dims) hash.Add(d);
            hash.AddBytes(_bytes);
            return hash.ToHashCode();
        }

        /// <summary>The value — a scalar as its number, at its own precision, anything else as its
        /// dtype and shape.</summary>
        public override string ToString()
        {
            if (ElementCount != 1 || _dims.Length != 0) return $"{DType}[{string.Join(", ", _dims)}]";
            var culture = System.Globalization.CultureInfo.InvariantCulture;
            if (DType == DType.Bool) return (_bytes[0] != 0).ToString();
            if (DType == DType.Float32 || DType == DType.Float16 || DType == DType.BFloat16)
                return ToSingle().ToString(culture);
            if (DType == DType.Int64) return MemoryMarshal.Read<long>(_bytes).ToString(culture);
            if (DType == DType.UInt64) return MemoryMarshal.Read<ulong>(_bytes).ToString(culture);
            return ToDouble().ToString(culture);
        }
    }
}

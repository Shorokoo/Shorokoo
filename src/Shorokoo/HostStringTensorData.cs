using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using Shorokoo.Core.Inference.Abstractions;
using Shorokoo.Runtime;

namespace Shorokoo
{
    /// <summary>
    /// A string tensor held as an ordinary managed <c>string[]</c>, owned by this object and
    /// belonging to no backend.
    ///
    /// <para><see cref="HostTensorData{T}"/>'s counterpart for <c>@string</c>, and here for
    /// the same reason. <c>TensorData([2], "a", "b")</c> and <c>Scalar("hello")</c> describe a
    /// graph; they do not run one, so building one should need no execution provider, no native
    /// runtime and no deployed backend. Each of them went through
    /// <c>InferenceBackend.Factory.CreateStringTensor</c> instead, which resolved the process-wide
    /// backend the moment a model mentioned a string literal.</para>
    ///
    /// <para>Strings stayed out of <see cref="HostTensorData{T}"/> because its storage is a flat
    /// byte buffer and a string element is variable-length and reference-typed. That is an
    /// argument about the storage and not about when the value is built: the storage differs here,
    /// the deferral does not. The runtime value is built the first time a backend asks for one, in
    /// <see cref="ToTensorValue(IShorokooInferenceSessionFactory)"/>, and kept per backend from
    /// then on.</para>
    ///
    /// <para>There is no byte view of these elements, and there was none before: an ONNX Runtime
    /// string tensor has no flat buffer to span over either, so <see cref="AccessRawMemory"/> and
    /// its siblings refuse here exactly as they refused through <see cref="OnnxTensorData{T}"/>.
    /// <see cref="Strings"/> is the read that needs no backend at all;
    /// <c>ToTensorValue(...).GetStringTensorData()</c> is the same answer by way of a runtime.</para>
    /// </summary>
    public sealed class HostStringTensorData : TensorData<@string>, IDisposable
    {
        private readonly string[] _values;

        // One materialized value per backend this tensor has been fed to, owned here and released
        // on Dispose -- HostTensorData<T>'s arrangement, for its reasons. Built on demand, because
        // most literals are only ever read by the graph builder; keyed by backend under reference
        // equality, because a value belongs to the runtime that made it and two factories are the
        // same backend exactly when they are the same object.
        private Dictionary<IShorokooInferenceSessionFactory, IShorokooTensorValue>? _materialized;
        private readonly object _gate = new();

        /// <summary>Creates a string tensor of <paramref name="shape"/> over
        /// <paramref name="values"/>, which it takes as its own storage rather than copying.</summary>
        public HostStringTensorData(Shape shape, string[] values)
            : this(shape, values, context: null, ownsMemory: true, storage: null)
        {
        }

        private HostStringTensorData(
            Shape shape, string[] values, ComputeContext? context, bool ownsMemory, TensorStorage? storage)
            : base(shape, storage ?? HostStorage(), context, ownsMemory)
        {
            _values = values ?? throw new ArgumentNullException(nameof(values));
        }

        /// <summary>
        /// Creates a string tensor of <paramref name="shape"/> holding <paramref name="values"/>,
        /// checked against the shape.
        ///
        /// <para>Too few is an error; a surplus is not, and is trimmed. That is the same asymmetry
        /// the numeric literals keep, and it used to be the backend's: <c>CreateStringTensor</c>
        /// set an element per supplied value into a shape-sized tensor, so a surplus threw at the
        /// construction site. Holding the array instead would let a tensor outrun its own dims and
        /// only fail if some session ever materialized it -- which for a program that just
        /// describes a graph and exports it never happens.</para>
        /// </summary>
        /// <exception cref="ArgumentException"><paramref name="values"/> does not cover
        /// <paramref name="shape"/>.</exception>
        public static HostStringTensorData From(Shape shape, string[] values)
        {
            ArgumentNullException.ThrowIfNull(values);
            var required = checked((int)shape.Count);
            if (values.Length < required)
                throw new ArgumentException(
                    $"Supplied data of {values.Length} strings is less than shape size {required} "
                    + "strings.", nameof(values));
            return new HostStringTensorData(shape, values.Length == required ? values : values[..required]);
        }

        /// <summary>A host string tensor over <paramref name="values"/> belonging to
        /// <paramref name="context"/>, which must be a host-memory context or null.</summary>
        internal static HostStringTensorData Bound(Shape shape, string[] values, ComputeContext? context)
            => new(shape, values, context, ownsMemory: true, storage: null);

        // The strings are the garbage collector's to reclaim, so releasing this storage frees
        // nothing directly. It still matters: it is what tells a tensor that was given access to
        // these elements that the owner has let go of them.
        private static TensorStorage HostStorage() => new(MemorySpace.Host, static () => { });

        /// <inheritdoc/>
        internal override TensorData CloneSharing(ComputeContext? context, bool ownsMemory)
            => new HostStringTensorData(Shape, _values, context, ownsMemory, Storage);

        /// <summary>
        /// The elements as they were given, in row-major order. This is the one read of a string
        /// tensor that costs no backend.
        ///
        /// <para>It is the literal, not the laid-out tensor: a runtime value of this tensor covers
        /// <see cref="TensorData.Shape"/> and pads a short literal with empty strings, which is the
        /// backend's rule and still applies where it always did — when the value is built.</para>
        /// </summary>
        public IReadOnlyList<string> Strings
        {
            get
            {
                ThrowIfDisposed();
                return _values;
            }
        }

        /// <summary>The elements boxed as objects, for debugging/diagnostics. The strings
        /// themselves, because they are the storage: there are no raw bytes underneath to box
        /// instead.</summary>
        public override object[] Data
        {
            get
            {
                ThrowIfDisposed();
                return _values.Cast<object>().ToArray();
            }
        }

        /// <summary>Always true: this tensor is managed memory and nothing else.</summary>
        public override bool IsHostResident => true;

        /// <inheritdoc/>
        public override Span<V> AccessModifiableMemory<V>()
        {
            ThrowIfDisposed();
            throw NoFlatBuffer();
        }

        /// <inheritdoc/>
        public override ReadOnlySpan<V> AccessMemory<V>()
        {
            ThrowIfDisposed();
            throw NoFlatBuffer();
        }

        /// <inheritdoc/>
        public override Span<byte> AccessModifiableRawMemory()
        {
            ThrowIfDisposed();
            throw NoFlatBuffer();
        }

        /// <inheritdoc/>
        public override ReadOnlySpan<byte> AccessRawMemory()
        {
            ThrowIfDisposed();
            throw NoFlatBuffer();
        }

        /// <summary>
        /// These strings as a tensor of <paramref name="factory"/>'s runtime, built the first time
        /// that backend asks and kept for the next time. The value is this tensor's, like
        /// <see cref="OnnxTensorData{T}"/>'s is: the caller reads it and does not dispose it.
        /// </summary>
        internal override IShorokooTensorValue ToTensorValue(IShorokooInferenceSessionFactory factory)
        {
            ArgumentNullException.ThrowIfNull(factory);
            ThrowIfDisposed();

            lock (_gate)
            {
                _materialized ??= new Dictionary<IShorokooInferenceSessionFactory, IShorokooTensorValue>(
                    ReferenceEqualityComparer.Instance);

                if (_materialized.TryGetValue(factory, out var existing)) return existing;

                var value = factory.CreateStringTensor(_values, (long[])this.Shape);
                _materialized[factory] = value;
                return value;
            }
        }

        /// <summary>
        /// Releases the values this tensor had built on backends. The strings themselves are
        /// managed and need no release; what needs one is each runtime's copy of them.
        ///
        /// <para>No finalizer, for the reason <see cref="OnnxTensorData{T}"/> has none: a
        /// finalizer must not touch another managed object that may already have been finalized,
        /// and each materialized value has its own.</para>
        /// </summary>
        public override void Dispose()
        {
            if (IsDisposed) return;
            IsDisposed = true;
            if (OwnsMemory) Storage.Release();
            lock (_gate)
            {
                if (_materialized is null) return;
                foreach (var value in _materialized.Values) value.Dispose();
                _materialized = null;
            }
        }

        // Every byte-wise accessor lands here rather than on a cast that cannot work. The message
        // names the two reads that do, because the caller reaching for a span of a string tensor
        // is asking a question with an answer -- just not that one.
        private InvalidOperationException NoFlatBuffer() => new(
            $"Tensor {this} holds strings. Their elements are variable-length and reference-typed, "
            + "so there is no flat buffer to span over -- an ONNX Runtime string tensor has none "
            + $"either. Read them with {nameof(Strings)}, which needs no backend, or build a "
            + "runtime value with ToTensorValue(...) and read that with GetStringTensorData().");
    }
}

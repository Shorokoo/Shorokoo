using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Shorokoo.Core.Inference.Abstractions;
using Shorokoo.Core.Utils;
using Shorokoo.Runtime;

namespace Shorokoo
{
    /// <summary>
    /// <see cref="TensorData{T}"/> held in ordinary managed memory, owned by this object and
    /// belonging to no backend.
    ///
    /// <para>This is what the convenience constructors build — <c>TensorData([4], 1f, 2f, 3f, 4f)</c>
    /// and its thirty-odd siblings — and so it is what a model's literals, a node definition's
    /// test values and an operator's tensor attributes are made of. None of those are inference:
    /// they are how a graph is <i>described</i>. Describing one therefore needs no execution
    /// provider, no native runtime, and no deployed backend at all.</para>
    ///
    /// <para>It did before. Every literal went through
    /// <c>InferenceBackend.Factory.CreateTensor</c>, so the node definition table's own tensors
    /// resolved the process-wide backend the first time anything touched a graph — which meant a
    /// program that only wanted to build a model and export it as ONNX still had to deploy a
    /// runtime to do it. The tensor arrives on a backend when it is fed to a session, in
    /// <see cref="TensorData.ToTensorValue(IShorokooInferenceSessionFactory)"/>, and not before.</para>
    ///
    /// <para>String tensors are not held here: their elements are variable-length and
    /// reference-typed, so they do not fit a flat byte buffer and keep the backend-backed path.</para>
    /// </summary>
    public sealed class HostTensorData<T> : TensorData<T>, IDisposable
        where T : IVarType
    {
        private readonly byte[] _bytes;

        // What these bytes have been built into, per backend. Shared with every clone over the
        // same bytes, because the materializations name the bytes rather than this wrapper.
        private readonly MaterializedValues _materialized;

        /// <summary>Creates a tensor of <paramref name="shape"/> over <paramref name="bytes"/>,
        /// which it takes as its own storage rather than copying.</summary>
        public HostTensorData(Shape shape, byte[] bytes)
            : this(shape, bytes, context: null, ownsMemory: true, storage: null, materialized: null)
        {
        }

        internal HostTensorData(Shape shape, byte[] bytes, DType actualDType)
            : base(shape, actualDType, HostStorage(), null, true)
        {
            _bytes = bytes ?? throw new ArgumentNullException(nameof(bytes));
            _materialized = new MaterializedValues();
        }

        private HostTensorData(
            Shape shape, byte[] bytes, ComputeContext? context, bool ownsMemory, TensorStorage? storage,
            MaterializedValues? materialized)
            : base(shape, storage ?? HostStorage(), context, ownsMemory)
        {
            _bytes = bytes ?? throw new ArgumentNullException(nameof(bytes));
            _materialized = materialized ?? new MaterializedValues();
        }

        /// <summary>A host tensor over <paramref name="bytes"/> belonging to
        /// <paramref name="context"/>, which must be a host-memory context or null.</summary>
        internal static HostTensorData<T> Bound(Shape shape, byte[] bytes, ComputeContext? context)
            => new(shape, bytes, context, ownsMemory: true, storage: null, materialized: null);

        // Managed bytes are the garbage collector's to reclaim, so releasing this storage frees
        // nothing directly. It still matters: it is what tells a tensor that was given access to
        // these bytes that the owner has let go of them.
        private static TensorStorage HostStorage() => new(MemorySpace.Host, static () => { });

        /// <inheritdoc/>
        internal override TensorData CloneSharing(ComputeContext? context, bool ownsMemory)
            => new HostTensorData<T>(Shape, _bytes, context, ownsMemory, Storage, _materialized);

        /// <summary>
        /// Creates a tensor of <paramref name="shape"/> holding a copy of
        /// <paramref name="values"/>' bytes.
        ///
        /// <para>Too few values is an error; a surplus is not, and is ignored. That asymmetry is
        /// the backend allocator's, kept because the node-definition tables rely on it — they hand
        /// over a buffer longer than the shape covers, and the surplus was never part of the
        /// tensor.</para>
        /// </summary>
        /// <exception cref="ArgumentException"><paramref name="values"/> does not cover
        /// <paramref name="shape"/>.</exception>
        public static HostTensorData<T> From<V>(Shape shape, V[] values) where V : unmanaged
        {
            ArgumentNullException.ThrowIfNull(values);
            var required = checked((int)shape.Count * Unsafe.SizeOf<V>());
            var supplied = MemoryMarshal.AsBytes(values.AsSpan());
            if (supplied.Length < required)
                throw new ArgumentException(
                    $"Supplied data of {supplied.Length} bytes is less than shape size {required} bytes.",
                    nameof(values));
            return new HostTensorData<T>(shape, supplied[..required].ToArray());
        }

        /// <summary>The raw storage bytes boxed as objects, for debugging/diagnostics.</summary>
        public override object[] Data
        {
            get
            {
                ThrowIfDisposed();
                return _bytes.Cast<object>().ToArray();
            }
        }

        /// <summary>Always true: this tensor is managed memory and nothing else.</summary>
        public override bool IsHostResident => true;

        /// <inheritdoc/>
        public override Span<V> AccessModifiableMemory<V>()
        {
            ThrowIfDisposed();
            _materialized.Invalidate();
            return MemoryMarshal.Cast<byte, V>(_bytes.AsSpan());
        }

        /// <inheritdoc/>
        public override ReadOnlySpan<V> AccessMemory<V>()
        {
            ThrowIfDisposed();
            return MemoryMarshal.Cast<byte, V>(_bytes);
        }

        /// <inheritdoc/>
        public override Span<byte> AccessModifiableRawMemory()
        {
            ThrowIfDisposed();
            _materialized.Invalidate();
            return _bytes;
        }

        /// <inheritdoc/>
        public override ReadOnlySpan<byte> AccessRawMemory()
        {
            ThrowIfDisposed();
            return _bytes;
        }

        /// <summary>
        /// This tensor's contents as a value of <paramref name="factory"/>'s runtime, built the
        /// first time that backend asks and kept for the next time. The value is this tensor's,
        /// like <see cref="OnnxTensorData{T}"/>'s is: the caller reads it and does not dispose it.
        /// </summary>
        internal override IShorokooTensorValue ToTensorValue(IShorokooInferenceSessionFactory factory)
        {
            ArgumentNullException.ThrowIfNull(factory);
            ThrowIfDisposed();

            return _materialized.Get(factory, f => f.CreateTensorFromRawBytes(
                (ShorokooTensorElementType)(int)this.DType, _bytes, (long[])this.Shape));
        }

        /// <summary>
        /// Releases the values this tensor had built on backends. The bytes themselves are managed
        /// and need no release; what needs one is each runtime's copy of them.
        ///
        /// <para>No finalizer, for the reason <see cref="OnnxTensorData{T}"/> has none: a
        /// finalizer must not touch another managed object that may already have been finalized,
        /// and each materialized value has its own.</para>
        /// </summary>
        public override void Dispose()
        {
            if (IsDisposed) return;
            IsDisposed = true;
            // Only the owner tears the materializations down: they are shared with every clone
            // over these bytes, and a reader letting go of its name for them frees nothing.
            if (OwnsMemory)
            {
                Storage.Release();
                _materialized.Invalidate();
            }
        }
    }
}

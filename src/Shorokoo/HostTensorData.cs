using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Shorokoo.Core.Backends;
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
    /// <c>DefaultBackend.Instance.CreateTensor</c>, so the node definition table's own tensors
    /// resolved the process-wide backend the first time anything touched a graph — which meant a
    /// program that only wanted to build a model and export it as ONNX still had to deploy a
    /// runtime to do it. The tensor arrives on a backend when it is fed to a session, in
    /// <see cref="TensorData.ToTensorValue(IShorokooBackend)"/>, and not before.</para>
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
            : this(shape, bytes, ComputeContext.Host, storage: null, new MaterializedValues())
        {
        }

        internal HostTensorData(Shape shape, byte[] bytes, DType actualDType)
            : this(shape, bytes, actualDType, new MaterializedValues())
        {
        }

        private HostTensorData(Shape shape, byte[] bytes, DType actualDType, MaterializedValues materialized)
            : base(shape, actualDType, HostStorage(materialized), ComputeContext.Host)
        {
            _bytes = bytes ?? throw new ArgumentNullException(nameof(bytes));
            _materialized = materialized;
        }

        // The materializations are built by the caller rather than defaulted here, because the
        // storage's release action closes over them and so needs them before the base call.
        private HostTensorData(
            Shape shape, byte[] bytes, ComputeContext context, TensorStorage? storage,
            MaterializedValues materialized)
            : base(shape, storage ?? HostStorage(materialized), context)
        {
            _bytes = bytes ?? throw new ArgumentNullException(nameof(bytes));
            _materialized = materialized;
        }

        /// <summary>A host tensor over <paramref name="bytes"/> belonging to
        /// <paramref name="context"/>, which must be a host-memory context.</summary>
        internal static HostTensorData<T> Bound(Shape shape, byte[] bytes, ComputeContext context)
            => new(shape, bytes, context, storage: null, new MaterializedValues());

        // Managed bytes are the garbage collector's to reclaim, so freeing this allocation frees no
        // host memory. What it does free is each runtime's copy of them, which is native and can be
        // a device allocation. It happens when the last handle and the last lock let go, which is
        // what keeps a run reading a tensor its caller has just disposed.
        private static TensorStorage HostStorage(MaterializedValues materialized)
            => new(MemorySpace.Host, materialized.Invalidate);

        /// <inheritdoc/>
        internal override byte[]? OwnBytes => _bytes;

        /// <summary>Whether no runtime holds a copy of these bytes -- the seam a test needs to see
        /// that a release freed the materializations rather than merely forgetting the tensor.
        /// </summary>
        internal bool MaterializationsAreEmpty => _materialized.IsEmpty;

        /// <inheritdoc/>
        internal override TensorData CloneSharing(ComputeContext context)
            => new HostTensorData<T>(Shape, _bytes, context, Storage, _materialized);

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
            RetireMaterializations();
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
            RetireMaterializations();
            return _bytes;
        }

        /// <summary>
        /// Drops every runtime's copy of these contents, because they are about to be written to.
        /// The copies are taken off the cache at once, so the next feed rebuilds them from what
        /// was written; freeing them waits for the last run reading them to return, because a
        /// value handed to a session is a bare pointer from that moment on and freeing one under
        /// a running read is the use-after-free the reference count exists to stop
        /// (Shorokoo/Shorokoo#366).
        /// </summary>
        private void RetireMaterializations()
        {
            if (_materialized.Retire() is { } free) Storage.FreeWhenUnlocked(free);
        }

        /// <inheritdoc/>
        public override ReadOnlySpan<byte> AccessRawMemory()
        {
            ThrowIfDisposed();
            return _bytes;
        }

        /// <summary>
        /// This tensor's contents as a value of <paramref name="backend"/>'s runtime, built the
        /// first time that backend asks and kept for the next time. The value is this tensor's,
        /// like <see cref="OnnxTensorData{T}"/>'s is: the caller reads it and does not dispose it.
        /// </summary>
        internal override IShorokooTensorValue ToTensorValue(IShorokooBackend backend)
        {
            ArgumentNullException.ThrowIfNull(backend);
            ThrowIfDisposed();

            return _materialized.Get(backend, f => f.CreateTensorFromRawBytes(
                (ShorokooTensorElementType)(int)this.DType, _bytes, (long[])this.Shape));
        }

        // Disposal is the base class's: drop this handle's reference, and the allocation tears the
        // materializations down when the last reference goes. They are shared with every clone
        // over these bytes, so one handle letting go of its name for them frees nothing.
        //
        // No finalizer, for the reason OnnxTensorData<T> has none: a finalizer must not touch
        // another managed object that may already have been finalized, and each materialized value
        // has its own.
    }
}

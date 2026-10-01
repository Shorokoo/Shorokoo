using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using Shorokoo.Core.Backends;
using Shorokoo.Core.Utils;
using Shorokoo.Runtime;

namespace Shorokoo
{
    /// <summary>
    /// <see cref="TensorData{T}"/> held in ordinary managed memory, owned by this object and
    /// belonging to no backend — its <see cref="TensorData.AllocatingBackend"/> is
    /// <see cref="HostBackend.Instance"/>, the framework's own host memory.
    ///
    /// <para>This is what the convenience constructors build — <c>TensorData([4], 1f, 2f, 3f, 4f)</c>
    /// and its thirty-odd siblings — and so it is what a model's literals, a node definition's
    /// test values and an operator's tensor attributes are made of. None of those are inference:
    /// they are how a graph is <i>described</i>. Describing one therefore needs no execution
    /// provider, no native runtime, and no deployed backend at all.</para>
    ///
    /// <para>Building one resolves no backend, so a program that only builds a model and exports
    /// it as ONNX needs no runtime deployed. The tensor arrives on a backend when it is fed to a
    /// session, and not before: a session is handed a runtime value, never a managed array, so a
    /// run reads this tensor through a copy its backend builds — held by this tensor for the next
    /// run, a tensor in its own right (<see cref="TensorData.CopyAt"/>).</para>
    ///
    /// <para>String tensors are not held here: their elements are variable-length and
    /// reference-typed, so they do not fit a flat byte buffer — see
    /// <see cref="HostStringTensorData"/>.</para>
    /// </summary>
    public sealed class HostTensorData<T> : TensorData<T>, IDisposable
        where T : IVarType
    {
        // Let go of when the tensor dies, so a dead tensor still referenced does not keep the bytes
        // it no longer has: a batch a run consumed is garbage from the moment the run takes it.
        private byte[]? _bytes;

        /// <summary>
        /// Creates a tensor of <paramref name="shape"/> holding a copy of <paramref name="bytes"/>.
        ///
        /// <para>A copy, so that the tensor is the only name for its memory: holding the caller's
        /// array would leave a second one, through which a write would change contents that are
        /// fixed when the tensor is built — and a run that had copied them would go on reading the
        /// old ones.</para>
        /// </summary>
        public HostTensorData(Shape shape, byte[] bytes)
            : this(shape, [.. bytes ?? throw new ArgumentNullException(nameof(bytes))], OnnxUtils.GetDType<T>())
        {
        }

        /// <summary>A tensor that takes <paramref name="bytes"/> — an array the framework has just
        /// made and nothing else names — as its own storage, carrying <paramref name="actualDType"/>
        /// exactly as given, a specialized dtype's generic parameter name included.</summary>
        internal HostTensorData(Shape shape, byte[] bytes, DType actualDType)
            : base(shape, actualDType, HostBackend.Instance, MemorySpace.Host)
        {
            _bytes = bytes ?? throw new ArgumentNullException(nameof(bytes));
        }

        /// <inheritdoc/>
        internal override byte[]? OwnBytes => Volatile.Read(ref _bytes);

        /// <summary>The storage, refused in the tensor's own words once it is gone.</summary>
        private byte[] Bytes
        {
            get
            {
                if (Volatile.Read(ref _bytes) is { } bytes) return bytes;
                ThrowIfDisposed();
                throw new ObjectDisposedException(nameof(HostTensorData<T>));
            }
        }

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
            return new HostTensorData<T>(shape, supplied[..required].ToArray(), OnnxUtils.GetDType<T>());
        }

        /// <summary>The raw storage bytes boxed as objects, for debugging/diagnostics.</summary>
        public override object[] Data
        {
            get
            {
                ThrowIfDisposed();
                return Bytes.Cast<object>().ToArray();
            }
        }

        /// <inheritdoc/>
        private protected override ReadOnlySpan<V> AccessElements<V>()
        {
            ThrowIfDisposed();
            return MemoryMarshal.Cast<byte, V>(Bytes);
        }

        /// <inheritdoc/>
        private protected override ReadOnlySpan<byte> AccessRawStorage()
        {
            ThrowIfDisposed();
            return Bytes;
        }

        /// <summary>
        /// These bytes as a value of <paramref name="backend"/>'s runtime: the copy in host memory of
        /// that runtime, built the first time that backend asks and kept for the next time — the same
        /// copy a run on a host backend reads. The value is the copy's: the caller reads it and does
        /// not dispose it.
        /// </summary>
        private protected override IShorokooTensorValue ValueFor(IShorokooBackend backend)
            => CopyAt(
                    new MemoryLocation(MemorySpace.Host, backend.RuntimeIdentity),
                    () => BuiltBy(backend, backend.CreateTensorFromRawBytes(
                        (ShorokooTensorElementType)(int)this.DType, Bytes, (long[])this.Shape)))
                .UncheckedValue(backend);

        /// <summary>
        /// Managed bytes are the garbage collector's to reclaim, so releasing this tensor frees no
        /// host memory of its own: it lets go of the array, which is then garbage however long the
        /// tensor itself is kept. The copies runs made of the bytes — native, and possibly on a
        /// card — are released by the base class, each through the backend that built it.
        /// </summary>
        private protected override void ReleaseMemory() => Volatile.Write(ref _bytes, null);

        /// <inheritdoc/>
        private protected override byte[] CopyContentBytes() => Bytes.AsSpan().ToArray();

        /// <summary>The array itself: a backend building a copy copies out of it and keeps
        /// nothing.</summary>
        private protected override byte[] ContentBytesForCopy() => Bytes;

        /// <inheritdoc/>
        private protected override IReadOnlyList<string> CopyContentStrings()
            => throw new InvalidOperationException(
                $"Tensor {this} holds {DType}, not strings, so it has no string elements to read.");

        // No finalizer, for the reason OnnxTensorData<T> has none: a finalizer must not touch
        // another managed object that may already have been finalized, and each copy has its own.
    }
}

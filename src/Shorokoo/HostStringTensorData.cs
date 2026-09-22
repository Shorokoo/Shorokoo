using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using Shorokoo.Core.Backends;
using Shorokoo.Runtime;

namespace Shorokoo
{
    /// <summary>
    /// A string tensor held as an ordinary managed <c>string[]</c>, owned by this object and
    /// belonging to no backend.
    ///
    /// <para><see cref="HostTensorData{T}"/>'s counterpart for <c>utf8</c>, and here for
    /// the same reason. <c>TensorData([2], "a", "b")</c> and <c>Scalar("hello")</c> describe a
    /// graph; they do not run one, so building one should need no execution provider, no native
    /// runtime and no deployed backend. Each of them went through
    /// <c>DefaultBackend.Instance.CreateStringTensor</c> instead, which resolved the process-wide
    /// backend the moment a model mentioned a string literal.</para>
    ///
    /// <para>Strings stayed out of <see cref="HostTensorData{T}"/> because its storage is a flat
    /// byte buffer and a string element is variable-length and reference-typed. That is an
    /// argument about the storage and not about when the value is built: the storage differs here,
    /// the deferral does not. The runtime value is built the first time a backend asks for one, in
    /// <see cref="ToTensorValue(IShorokooBackend)"/>, and kept per backend from
    /// then on.</para>
    ///
    /// <para>There is no byte view of these elements, and there was none before: an ONNX Runtime
    /// string tensor has no flat buffer to span over either, so <see cref="AccessRawMemory"/> and
    /// its siblings refuse here exactly as they refused through <see cref="OnnxTensorData{T}"/>.
    /// <see cref="Strings"/> is the read that needs no backend at all;
    /// <c>ToTensorValue(...).GetStringTensorData()</c> is the same answer by way of a runtime.</para>
    /// </summary>
    public sealed class HostStringTensorData : TensorData<utf8>, IDisposable
    {
        private readonly string[] _values;

        // What these strings have been built into, per backend. Shared with every clone over the
        // same strings, because the materializations name the strings rather than this wrapper.
        private readonly MaterializedValues _materialized;

        /// <summary>Creates a string tensor of <paramref name="shape"/> over
        /// <paramref name="values"/>, which it takes as its own storage rather than copying.</summary>
        public HostStringTensorData(Shape shape, string[] values)
            : this(shape, values, ComputeContext.Host, storage: null, new MaterializedValues())
        {
        }

        // The materializations are built by the caller rather than defaulted here, because the
        // allocation's release action closes over them and so needs them before the base call.
        private HostStringTensorData(
            Shape shape, string[] values, ComputeContext context, TensorStorage? storage,
            MaterializedValues materialized)
            : base(shape, storage ?? HostStorage(materialized), context)
        {
            _values = values ?? throw new ArgumentNullException(nameof(values));
            _materialized = materialized;
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
        /// <paramref name="context"/>, which must be a host-memory context.</summary>
        internal static HostStringTensorData Bound(Shape shape, string[] values, ComputeContext context)
            => new(shape, values, context, storage: null, new MaterializedValues());

        // The strings are the garbage collector's to reclaim, so freeing this allocation frees no
        // host memory. What it frees is each runtime's copy of them, once the last handle and the
        // last lock have let go -- and it is also what tells a second handle on these elements
        // that they are gone.
        private static TensorStorage HostStorage(MaterializedValues materialized)
            => new(MemorySpace.Host, materialized.Invalidate);

        /// <inheritdoc/>
        internal override TensorData CloneSharing(ComputeContext context)
            => new HostStringTensorData(Shape, _values, context, Storage, _materialized);

        /// <summary>
        /// The elements as they were given, in row-major order. This is the one read of a string
        /// tensor that costs no backend.
        ///
        /// <para>Exactly <see cref="TensorData.Shape"/>'s worth of them. The backend used to pad a
        /// short literal with empty strings when it built the value, and that is no longer how a
        /// short one ends: <see cref="From"/> refuses it, at the construction site, where the
        /// mistake is.</para>
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
        /// These strings as a tensor of <paramref name="backend"/>'s runtime, built the first time
        /// that backend asks and kept for the next time. The value is this tensor's, like
        /// <see cref="OnnxTensorData{T}"/>'s is: the caller reads it and does not dispose it.
        /// </summary>
        internal override IShorokooTensorValue ToTensorValue(IShorokooBackend backend)
        {
            ArgumentNullException.ThrowIfNull(backend);
            ThrowIfDisposed();

            return _materialized.Get(
                backend, f => f.CreateStringTensor(_values, (long[])this.Shape));
        }

        // Disposal is the base class's: drop this handle's reference, and the allocation tears the
        // materializations down when the last reference goes. They are shared with every clone
        // over these strings, so one handle letting go of its name for them frees nothing.
        //
        // No finalizer, for the reason OnnxTensorData<T> has none: a finalizer must not touch
        // another managed object that may already have been finalized, and each materialized value
        // has its own.

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

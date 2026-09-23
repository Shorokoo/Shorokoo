using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using Shorokoo.Core.Backends;
using static Shorokoo.Globals;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using Shorokoo.Runtime;

namespace Shorokoo
{
    /// <summary>
    /// Tensor value storage typed by IVarType element type T, with typed span access
    /// to the underlying buffer.
    /// </summary>
    public abstract class TensorData<T> : TensorData, IData<T>
        where T : IVarType
    {
        internal TensorData(Shape shape, IShorokooBackend allocatingBackend, MemorySpace space)
            : base(shape, OnnxUtils.GetDType<T>(), allocatingBackend, space)
        {
        }

        internal TensorData(
            Shape shape, DType dtype, IShorokooBackend allocatingBackend, MemorySpace space)
            : base(shape, dtype, allocatingBackend, space)
        {
        }

        /// <summary>Exposes the underlying buffer as a writable span of V (V must match T's storage
        /// type). The span points straight into the tensor's storage, so it is valid only while the
        /// tensor is — see <see cref="TensorData.AccessRawMemory"/>.</summary>
        public abstract Span<V> AccessModifiableMemory<V>() where V : unmanaged;

        /// <summary>
        /// Fills the buffer through <paramref name="write"/>, with the tensor kept alive for the
        /// length of the call. This is <see cref="AccessModifiableMemory{V}"/> done safely, and it
        /// is the write counterpart of <see cref="CopyMemory{V}"/>.
        ///
        /// <para>Taking the span is the tensor's last read, so a buffer filled through a bare
        /// <c>AccessModifiableMemory</c> races the collection that frees what the span points at:
        /// the tensor is unreachable from that call onwards, and on a backend-allocated buffer the
        /// runtime value's finalizer hands the block back while the caller is still writing into
        /// it (Shorokoo/Shorokoo#178). Scope is not reachability. Whatever fills a tensor should
        /// fill it here.</para>
        /// </summary>
        /// <exception cref="ArgumentNullException"><paramref name="write"/> is null.</exception>
        public void WriteMemory<V>(SpanWriter<V> write) where V : unmanaged
        {
            ArgumentNullException.ThrowIfNull(write);
            write(AccessModifiableMemory<V>());
            GC.KeepAlive(this);
        }

        /// <summary>
        /// The elements copied into an array of V the caller owns, valid however long the caller
        /// keeps it. This is <see cref="AccessMemory{V}"/> plus the copy, done safely: taking a
        /// span is the tensor's last read, so copying out of one by hand races the collection that
        /// frees what it points at (Shorokoo/Shorokoo#178). Prefer this wherever the whole buffer
        /// is being copied anyway.
        /// </summary>
        public V[] CopyMemory<V>() where V : unmanaged
        {
            var copy = AccessMemory<V>().ToArray();
            GC.KeepAlive(this);
            return copy;
        }

        /// <summary>
        /// One element, read safely — the single-value counterpart of <see cref="CopyMemory{V}"/>,
        /// for the very common case of a scalar or a leading element. Reading
        /// <c>AccessMemory&lt;V&gt;()[i]</c> by hand indexes a span whose tensor the JIT may already
        /// have retired (Shorokoo/Shorokoo#178).
        /// </summary>
        public V ValueAt<V>(int index) where V : unmanaged
        {
            var value = AccessMemory<V>()[index];
            GC.KeepAlive(this);
            return value;
        }
        /// <summary>Exposes the underlying buffer as a read-only span of V (V must match T's storage
        /// type). The span points straight into the tensor's storage, so it is valid only while the
        /// tensor is — see <see cref="TensorData.AccessRawMemory"/>.</summary>
        public abstract ReadOnlySpan<V> AccessMemory<V>() where V : unmanaged;

        /// <summary>The element values boxed as objects, for debugging/diagnostics.</summary>
        public object[] DebugData
        {
            get
            {
                switch(typeof(T))
                {
                    case Type t when t == typeof(utf8):
                        return [.. StringElements()];
                    case Type t when t == typeof(bit):
                        return this.CopyMemory<bool>().Cast<object>().ToArray();
                    case Type t when t == typeof(int8):
                        return this.CopyMemory<sbyte>().Cast<object>().ToArray();
                    case Type t when t == typeof(int16):
                        return this.CopyMemory<short>().Cast<object>().ToArray();
                    case Type t when t == typeof(int32):
                        return this.CopyMemory<int>().Cast<object>().ToArray();
                    case Type t when t == typeof(int64):
                        return this.CopyMemory<long>().Cast<object>().ToArray();
                    case Type t when t == typeof(uint8):
                        return this.CopyMemory<byte>().Cast<object>().ToArray();
                    case Type t when t == typeof(uint16):
                        return this.CopyMemory<ushort>().Cast<object>().ToArray();
                    case Type t when t == typeof(uint32):
                        return this.CopyMemory<uint>().Cast<object>().ToArray();
                    case Type t when t == typeof(uint64):
                        return this.CopyMemory<ulong>().Cast<object>().ToArray();
                    case Type t when t == typeof(float16):
                        return this.CopyMemory<Float16>().Cast<object>().ToArray();
                    case Type t when t == typeof(bfloat16):
                        return this.CopyMemory<BFloat16>().Cast<object>().ToArray();
                    case Type t when t == typeof(float32):
                        return this.CopyMemory<float>().Cast<object>().ToArray();
                    case Type t when t == typeof(float64):
                        return this.CopyMemory<double>().Cast<object>().ToArray();
                    default:
                        return this.CopyMemory<byte>().Cast<object>().ToArray();
                }
            }
        }
    }

    /// <summary>
    /// Typed AccessMemory / AccessModifiableMemory shortcuts mapping each IVarType
    /// to its storage primitive (e.g. <see cref="TensorData{T}"/> of bit to bool spans).
    /// </summary>
    public static class TensorDataExtensions
    {
        /// <summary>Read-only span over the elements of a <c>bit</c> tensor as <c>bool</c>.</summary>
        public static ReadOnlySpan<bool> AccessMemory(this TensorData<bit> data) => data.AccessMemory<bool>();
        /// <summary>Read-only span over the elements of a <c>int8</c> tensor as <c>sbyte</c>.</summary>
        public static ReadOnlySpan<sbyte> AccessMemory(this TensorData<int8> data) => data.AccessMemory<sbyte>();
        /// <summary>Read-only span over the elements of a <c>int16</c> tensor as <c>short</c>.</summary>
        public static ReadOnlySpan<short> AccessMemory(this TensorData<int16> data) => data.AccessMemory<short>();
        /// <summary>Read-only span over the elements of a <c>int32</c> tensor as <c>int</c>.</summary>
        public static ReadOnlySpan<int> AccessMemory(this TensorData<int32> data) => data.AccessMemory<int>();
        /// <summary>Read-only span over the elements of a <c>int64</c> tensor as <c>long</c>.</summary>
        public static ReadOnlySpan<long> AccessMemory(this TensorData<int64> data) => data.AccessMemory<long>();
        /// <summary>Read-only span over the elements of a <c>uint8</c> tensor as <c>byte</c>.</summary>
        public static ReadOnlySpan<byte> AccessMemory(this TensorData<uint8> data) => data.AccessMemory<byte>();
        /// <summary>Read-only span over the elements of a <c>uint16</c> tensor as <c>ushort</c>.</summary>
        public static ReadOnlySpan<ushort> AccessMemory(this TensorData<uint16> data) => data.AccessMemory<ushort>();
        /// <summary>Read-only span over the elements of a <c>uint32</c> tensor as <c>uint</c>.</summary>
        public static ReadOnlySpan<uint> AccessMemory(this TensorData<uint32> data) => data.AccessMemory<uint>();
        /// <summary>Read-only span over the elements of a <c>uint64</c> tensor as <c>ulong</c>.</summary>
        public static ReadOnlySpan<ulong> AccessMemory(this TensorData<uint64> data) => data.AccessMemory<ulong>();
        /// <summary>Read-only span over the elements of a <c>float16</c> tensor as <c>Float16</c>.</summary>
        public static ReadOnlySpan<Float16> AccessMemory(this TensorData<float16> data) => data.AccessMemory<Float16>();
        /// <summary>Read-only span over the elements of a <c>bfloat16</c> tensor as <c>BFloat16</c>.</summary>
        public static ReadOnlySpan<BFloat16> AccessMemory(this TensorData<bfloat16> data) => data.AccessMemory<BFloat16>();
        /// <summary>Read-only span over the elements of a <c>float32</c> tensor as <c>float</c>.</summary>
        public static ReadOnlySpan<float> AccessMemory(this TensorData<float32> data) => data.AccessMemory<float>();
        /// <summary>Read-only span over the elements of a <c>float64</c> tensor as <c>double</c>.</summary>
        public static ReadOnlySpan<double> AccessMemory(this TensorData<float64> data) => data.AccessMemory<double>();

        /// <summary>Writable span over the elements of a <c>int8</c> tensor as <c>sbyte</c>.</summary>
        public static Span<sbyte> AccessModifiableMemory(this TensorData<int8> data) => data.AccessModifiableMemory<sbyte>();
        /// <summary>Writable span over the elements of a <c>int16</c> tensor as <c>short</c>.</summary>
        public static Span<short> AccessModifiableMemory(this TensorData<int16> data) => data.AccessModifiableMemory<short>();
        /// <summary>Writable span over the elements of a <c>int32</c> tensor as <c>int</c>.</summary>
        public static Span<int> AccessModifiableMemory(this TensorData<int32> data) => data.AccessModifiableMemory<int>();
        /// <summary>Writable span over the elements of a <c>int64</c> tensor as <c>long</c>.</summary>
        public static Span<long> AccessModifiableMemory(this TensorData<int64> data) => data.AccessModifiableMemory<long>();
        /// <summary>Writable span over the elements of a <c>uint8</c> tensor as <c>byte</c>.</summary>
        public static Span<byte> AccessModifiableMemory(this TensorData<uint8> data) => data.AccessModifiableMemory<byte>();
        /// <summary>Writable span over the elements of a <c>uint16</c> tensor as <c>ushort</c>.</summary>
        public static Span<ushort> AccessModifiableMemory(this TensorData<uint16> data) => data.AccessModifiableMemory<ushort>();
        /// <summary>Writable span over the elements of a <c>uint32</c> tensor as <c>uint</c>.</summary>
        public static Span<uint> AccessModifiableMemory(this TensorData<uint32> data) => data.AccessModifiableMemory<uint>();
        /// <summary>Writable span over the elements of a <c>uint64</c> tensor as <c>ulong</c>.</summary>
        public static Span<ulong> AccessModifiableMemory(this TensorData<uint64> data) => data.AccessModifiableMemory<ulong>();
        /// <summary>Writable span over the elements of a <c>float16</c> tensor as <c>Float16</c>.</summary>
        public static Span<Float16> AccessModifiableMemory(this TensorData<float16> data) => data.AccessModifiableMemory<Float16>();
        /// <summary>Writable span over the elements of a <c>bfloat16</c> tensor as <c>BFloat16</c>.</summary>
        public static Span<BFloat16> AccessModifiableMemory(this TensorData<bfloat16> data) => data.AccessModifiableMemory<BFloat16>();
        /// <summary>Writable span over the elements of a <c>float32</c> tensor as <c>float</c>.</summary>
        public static Span<float> AccessModifiableMemory(this TensorData<float32> data) => data.AccessModifiableMemory<float>();
        /// <summary>Writable span over the elements of a <c>float64</c> tensor as <c>double</c>.</summary>
        public static Span<double> AccessModifiableMemory(this TensorData<float64> data) => data.AccessModifiableMemory<double>();
    }

    /// <summary>
    /// Fills a tensor's buffer in place. Used by <see cref="TensorData{T}.WriteMemory{V}"/>, which
    /// keeps the tensor reachable for the length of the call — which a bare span does not.
    /// </summary>
    public delegate void SpanWriter<V>(Span<V> destination) where V : unmanaged;

    /// <summary>A data value with an associated <see cref="DType"/>.</summary>
    public interface IData
    {
        /// <summary>The value's data type.</summary>
        public DType DType { get; }
    }

    /// <summary>An <see cref="IData"/> whose element type is the IVarType T.</summary>
    public interface IData<T> : IData where T : IVarType { }

    /// <summary>
    /// Concrete tensor value: a shape, a dtype, and raw element storage.
    /// Base of the typed <see cref="TensorData{T}"/> hierarchy.
    ///
    /// <para><b>One tensor, one allocation.</b> A <c>TensorData</c> is its memory: no two of them
    /// ever name the same bytes, and there is no second name to hand out. It records the backend
    /// that allocated that memory (<see cref="AllocatingBackend"/>), which is the one that releases
    /// it, and where the memory is (<see cref="Location"/>) — and it does not know which compute
    /// contexts it is attached to. A context keeps its own weak list of those, for its own
    /// purposes; attachment never keeps a tensor alive and never ends its life.</para>
    ///
    /// <para><b>A tensor dies in exactly three ways</b>: it is deleted (<see cref="Delete"/>,
    /// <see cref="Dispose"/>, <see cref="TryDelete"/>, <see cref="DeleteAsync"/>), it is consumed
    /// by a run — fed to it as it is, which is the default, or through <see cref="TryConsume"/> —
    /// or it is moved into an attribute (<see cref="MoveToAttribute"/>). Nothing else ends its life
    /// — disposing a context it is attached to does not — and a tensor nothing references is
    /// reclaimed like any other object, its memory released through its backend's ordinary path. A
    /// dead tensor's shape, dtype and <see cref="ToString"/> stay readable; every other access
    /// throws an <see cref="ObjectDisposedException"/> that says why it died — for a consumed one,
    /// which run took it and how to keep it next time.</para>
    ///
    /// <para><b>Fed as it is, a tensor is consumed.</b> A run given a tensor takes it when it
    /// starts: the tensor is dead from then on, even if the run fails, and its memory belongs to
    /// the run's backend, which releases it as soon as the run no longer needs it. Feed
    /// <see cref="Shared"/> to have the run read it and leave it alive.</para>
    ///
    /// <para><b>A run holds what it is reading.</b> A run takes a reader lock on every tensor it
    /// reads and holds it, and a reference to the tensor, for as long as it runs. A locked tensor
    /// cannot be deleted — <see cref="Delete"/> throws and <see cref="TryDelete"/> declines — or
    /// consumed by another run, so nothing frees a buffer a run is reading
    /// (Shorokoo/Shorokoo#366); <see cref="DeleteAsync"/> is the one call that negotiates with
    /// the readers instead.</para>
    /// </summary>
    public abstract partial class TensorData : IData, IDisposable
    {
        /// <summary>The tensor's shape.</summary>
        public Shape Shape { get; }
        /// <summary>The element data type.</summary>
        public DType DType { get; }

        /// <summary>The raw storage bytes boxed as objects, for debugging/diagnostics.</summary>
        public virtual object[] Data
        {
            get
            {
                // Strings are the one dtype with no flat buffer to box out of; their elements are
                // the storage, so they are read as themselves.
                if (DType.IsSameElementTypeAs(DType.Utf8)) return [.. StringElements()];
                return this.CopyRawMemory().Cast<object>().ToArray();
            }
        }

        internal TensorData(
            Shape shape, DType dtype, IShorokooBackend allocatingBackend, MemorySpace space)
        {
            ArgumentNullException.ThrowIfNull(allocatingBackend);
            this.Shape = shape;
            this.DType = dtype;
            this.AllocatingBackend = allocatingBackend;
            this.Space = space;
        }

        /// <summary>
        /// The backend that created this tensor's memory, and the one that releases it — whichever
        /// compute contexts the tensor is attached to, or none. <see cref="HostBackend.Instance"/>
        /// for a tensor built from a C# array, whose memory is the framework's own; the backend a
        /// session ran on for that session's outputs; the target context's backend for a copy made
        /// for it.
        /// </summary>
        public IShorokooBackend AllocatingBackend { get; }

        /// <summary>
        /// Where this tensor's bytes are: host memory, or a particular device's. Fixed when the
        /// tensor is made — a tensor never moves; <see cref="To"/> and <see cref="CopyTo"/> make
        /// another one where it has to be somewhere else.
        /// </summary>
        public MemorySpace Space { get; }

        /// <summary>The memory device this tensor's bytes are in, shared with every other tensor in
        /// the same <see cref="Space"/>.</summary>
        public MemoryDevice Device => MemoryDevice.For(Space);

        /// <summary>
        /// Where this tensor's memory is, completely enough for a backend to say whether it can read
        /// it as it stands: the <see cref="Space"/>, and the runtime of the backend that allocated
        /// it. <see cref="IShorokooBackend.CanAddress"/> is asked this, by <see cref="To"/>.
        /// </summary>
        public MemoryLocation Location => new(Space, AllocatingBackend.RuntimeIdentity);

        /// <summary>
        /// This tensor's own byte array where it has one, so <see cref="MoveToAttribute"/> can take
        /// it rather than copy it. Null when the elements are a runtime value's or are strings, in
        /// which case only a copy can get them out.
        /// </summary>
        internal virtual byte[]? OwnBytes => null;

        /// <summary>"shape:dtype" diagnostic string.</summary>
        public override string ToString()
        {
            var shapeStr = this.Shape.ToString();
            return $"{shapeStr}:{this.DType.ToString()}";
        }

        /// <summary>Exposes the underlying storage as a writable byte span. Same lifetime rule as
        /// <see cref="AccessRawMemory"/>.</summary>
        public abstract Span<byte> AccessModifiableRawMemory();

        /// <summary>
        /// Exposes the underlying storage as a read-only byte span.
        ///
        /// <para>The span is a window onto the tensor's own storage, not a copy, and nothing ties
        /// its lifetime to the tensor's. It is valid only while the tensor is alive AND still
        /// reachable: deleting the tensor frees what the span points at (later reads through the
        /// tensor itself throw, but the span has no such guard), and so does letting the tensor
        /// become unreachable, since its storage is released when the runtime value behind it is
        /// finalized. Being in scope is not being reachable — a local is retired at its last read,
        /// which is the call that produced the span. Copy out of the span before the tensor's last
        /// use, or keep the tensor alive across it (Shorokoo/Shorokoo#178).</para>
        /// </summary>
        public abstract ReadOnlySpan<byte> AccessRawMemory();

        /// <summary>
        /// The storage bytes copied into an array the caller owns, valid however long the caller
        /// keeps it — <see cref="AccessRawMemory"/> plus the copy, with the tensor kept alive
        /// across it. Prefer this wherever the whole buffer is being copied anyway.
        /// </summary>
        public byte[] CopyRawMemory()
        {
            var copy = AccessRawMemory().ToArray();
            GC.KeepAlive(this);
            return copy;
        }

        /// <summary>
        /// Whether this tensor's storage is host memory, so the <c>Access…Memory</c> accessors
        /// may be called. Like every path to the elements it throws once the tensor is dead, rather
        /// than answering about storage that is gone — ask <see cref="IsDisposed"/> first if a
        /// tensor may have died. It is <c>false</c> for a tensor an execution provider produced in
        /// its own memory, or one put there by <see cref="To"/> or <see cref="CopyTo"/> on a device
        /// context; reading such a tensor throws, and <see cref="ToHost"/> is what brings one back
        /// to the host.
        /// </summary>
        public virtual bool IsHostResident
        {
            get
            {
                ThrowIfDisposed();
                return true;
            }
        }

        /// <summary>Downcasts to the typed <see cref="TensorData{T}"/>; T must match the actual element type.</summary>
        public TensorData<T> As<T>() where T : IVarType => (TensorData<T>)this;

        /// <summary>
        /// Creates TensorData backed by an existing backend-runtime tensor value, without saying
        /// which backend made it. The tensor takes the value over: it is released with the tensor,
        /// by disposing it.
        ///
        /// <para>Prefer <see cref="Create(Shape, DType, IShorokooTensorValue, IShorokooBackend)"/>
        /// wherever the backend is known. A tensor whose producer was not named is host memory if
        /// the value says it is and somewhere unnamed otherwise, no backend can read it in place,
        /// and one that is not host-readable can be read back by nothing at all.</para>
        /// </summary>
        public static TensorData Create(Shape shape, DType dtype, IShorokooTensorValue data)
            => OnnxUtils.CreateTensorDataFromValue(shape, dtype, data);

        /// <summary>
        /// Creates TensorData backed by <paramref name="data"/>, a value
        /// <paramref name="allocatingBackend"/> made. The tensor takes the value over, and releases
        /// it through that backend.
        /// </summary>
        /// <exception cref="ArgumentNullException"><paramref name="data"/> or
        /// <paramref name="allocatingBackend"/> is null.</exception>
        public static TensorData Create(
            Shape shape, DType dtype, IShorokooTensorValue data, IShorokooBackend allocatingBackend)
            => OnnxUtils.CreateTensorDataFromValue(shape, dtype, data, allocatingBackend);

        /// <summary>A tensor over the given bytes, in the framework's own host memory.</summary>
        internal static TensorData NewHostTensor(Shape shape, DType dtype, byte[] bytes)
            => OnnxUtils.CreateManagedTensorData(shape, dtype, bytes);

        /// <summary>A string tensor over the given elements, in the framework's own host
        /// memory.</summary>
        internal static TensorData NewHostStringTensor(Shape shape, string[] values)
            => new HostStringTensorData(shape, values);

        /// <summary>
        /// Creates TensorData of the given shape and dtype over <paramref name="data"/> — plain
        /// host memory belonging to no backend, the same thing <see cref="NewHostTensor"/> makes.
        ///
        /// <para>Raw bytes are what a tensor read out of a model file, or zeroed for a gradient
        /// buffer, already is; wrapping them describes data rather than running anything. This
        /// went through <c>DefaultBackend.Instance</c> instead, so reading an <c>.onnx</c> file
        /// resolved the process-wide backend and put a native allocation behind every initializer
        /// in it. The value is built when a session is fed this tensor, in
        /// <see cref="ToTensorValue(IShorokooBackend)"/>, and not before.</para>
        ///
        /// <para>Exactly <paramref name="shape"/>'s worth of <paramref name="data"/> becomes the
        /// tensor: too few bytes is an error, a surplus is not and is dropped. That asymmetry is
        /// the backend allocator's — it allocated to the shape and filled what it could reach, so
        /// the surplus was never part of the tensor — and it has to be kept, because a tensor whose
        /// buffer outruns its own dims serializes to an ONNX initializer no runtime will
        /// deserialize.</para>
        /// </summary>
        /// <exception cref="ArgumentException"><paramref name="data"/> does not cover
        /// <paramref name="shape"/>.</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> is
        /// <see cref="DType.Utf8"/>, or <paramref name="shape"/> has no known element
        /// count.</exception>
        /// <exception cref="UnsupportedDTypeException"><paramref name="dtype"/> has no whole-byte
        /// element stride, so no flat buffer can describe it.</exception>
        public static TensorData CreateFromRawBytes(Shape shape, DType dtype, byte[] data)
        {
            ArgumentNullException.ThrowIfNull(data);
            // The refusal the backend used to give, kept where it can still be given eagerly: a
            // string element is variable-length, so a flat byte buffer does not describe one and
            // a HostTensorData<utf8> over these bytes would be a tensor of nothing.
            if (dtype == DType.Utf8)
                throw new NotSupportedException(
                    "String tensors are variable-length and not byte-stride, so raw bytes cannot "
                    + "describe one. Build it from its elements with TensorData(dims, string[]).");

            // Throws for the element types with no whole-byte stride at all -- the sub-byte
            // integers and the complex pairs -- which the byte-wise backend constructor refused
            // just as flatly, and which the framework refuses in its own words here.
            var bits = dtype.EncodingBitCount;
            // A shape carrying an unknown dimension reports a negative count, and so would the
            // slice length below; refuse it while it can still be said what is wrong.
            if (bits < 8 || shape.Count < 0)
                throw new NotSupportedException(
                    $"A tensor of {shape}:{dtype} cannot be described by a flat byte buffer: its "
                    + "elements have no whole-byte stride, or its shape has no known element count.");

            var required = checked(shape.Count * (bits / 8));
            if (data.Length < required)
                throw new ArgumentException(
                    $"Supplied data of {data.Length} bytes is less than shape size {required} bytes.",
                    nameof(data));

            // Always a copy, and exactly one. Taking the caller's array made every tensor share
            // mutable state with whatever produced the buffer -- a model's initializers aliased
            // the parsed protobuf, and a loader reusing one scratch array got tensors that all
            // held the contents of its last read. Through a span rather than the range indexer,
            // which allocates one array of its own and then hands it to LINQ for a second.
            return NewHostTensor(shape, dtype, data.AsSpan(0, (int)required).ToArray());
        }

        /// <summary>
        /// Returns the backing backend-runtime tensor value, on the process-wide backend;
        /// throws if this instance has none and none can be built.
        /// </summary>
        public IShorokooTensorValue ToTensorValue() => ToTensorValue(DefaultBackend.Instance);

        /// <summary>
        /// This tensor as a value of <paramref name="backend"/>'s runtime. A tensor that already
        /// holds one hands it over and ignores the argument, since a value belongs to the runtime
        /// that made it; one held in plain host memory is copied into host memory of that runtime
        /// here — the copy runs on that backend read it through — which is the first moment a
        /// backend is needed at all.
        ///
        /// <para>The value returned is the tensor's, or its copy's: read it, do not dispose it.</para>
        /// </summary>
        internal IShorokooTensorValue ToTensorValue(IShorokooBackend backend)
        {
            ArgumentNullException.ThrowIfNull(backend);
            ThrowIfDisposed();
            return ValueFor(backend);
        }

        /// <summary>
        /// <see cref="ToTensorValue(IShorokooBackend)"/> without the liveness check. Only a caller
        /// that has vouched for the memory may call it: a run holding this tensor's lock, or one that
        /// has taken it, whose memory is the taker's alone.
        /// </summary>
        internal IShorokooTensorValue UncheckedValue(IShorokooBackend backend)
        {
            ArgumentNullException.ThrowIfNull(backend);
            return ValueFor(backend);
        }

        /// <summary>See <see cref="UncheckedValue"/>.</summary>
        private protected abstract IShorokooTensorValue ValueFor(IShorokooBackend backend);

        /// <summary>
        /// Whether a run on <paramref name="backend"/> is handed this tensor's own value: it holds
        /// one, and the backend can address it where it is. False for the framework's own managed
        /// memory, which is a <c>byte[]</c> no session can be handed — a run reads it through a copy
        /// its backend builds.
        /// </summary>
        internal virtual bool FeedsInPlace(IShorokooBackend backend) => false;

        /// <summary>
        /// This tensor's contents as host bytes for a copy to be built from: its own array where it
        /// has one — the backend copies out of it and keeps nothing — and a copy otherwise. Without
        /// the liveness check, like <see cref="CopyContentBytes"/>.
        /// </summary>
        private protected virtual byte[] ContentBytesForCopy() => CopyContentBytes();

        /// <summary>Creates int32 TensorData of the given shape holding 0, 1, ..., Count-1 in row-major order.</summary>
        public static TensorData<int32> BuildRange(Shape shape)
        {
            var vals = Enumerable.Range(0, (int)shape.Count).ToArray();
            return (TensorData<int32>)TensorData(shape.Dims, vals);
        }
    }

    /// <summary>TensorData backed by a backend-runtime tensor value.</summary>
    public interface IOnnxData
    {
        /// <summary>The backing backend-runtime tensor value.</summary>
        public IShorokooTensorValue Value { get; }
    }

    /// <summary>
    /// <see cref="TensorData{T}"/> implementation backed by a backend-runtime
    /// (ONNX) tensor value; span access reads the runtime tensor's buffer directly.
    /// </summary>
    public sealed class OnnxTensorData<T> : TensorData<T>, IOnnxData, IDisposable
        where T : IVarType
    {
        private readonly IShorokooTensorValue backing;

        /// <summary>
        /// The backing backend-runtime tensor value, which this tensor owns: it is released through
        /// <see cref="TensorData.AllocatingBackend"/> when the tensor dies, and nothing else may hold
        /// or free it (Shorokoo/Shorokoo#180).
        /// </summary>
        public IShorokooTensorValue Value
        {
            get
            {
                ThrowIfDisposed();
                return backing;
            }
        }

        /// <summary>The raw storage bytes boxed as objects, for debugging/diagnostics.</summary>
        public override object[] Data
        {
            get
            {
                if (DType.IsSameElementTypeAs(DType.Utf8)) return [.. StringElements()];
                return this.CopyMemory<byte>().Cast<object>().ToArray();
            }
        }

        /// <summary>
        /// Creates TensorData of the given shape around an existing runtime tensor value, without
        /// saying which backend made it; the dtype is derived from T. The tensor takes the value
        /// over. See <see cref="TensorData.Create(Shape, DType, IShorokooTensorValue)"/> for what
        /// leaving the producer unnamed costs.
        /// </summary>
        public OnnxTensorData(Shape shape, IShorokooTensorValue value)
            : this(shape, value, UnrecordedBackend.Instance)
        {
        }

        /// <summary>A tensor over a value <paramref name="allocatingBackend"/> made, taking it
        /// over.</summary>
        internal OnnxTensorData(Shape shape, IShorokooTensorValue value, IShorokooBackend allocatingBackend)
            : base(shape, allocatingBackend, SpaceOf(value, allocatingBackend))
        {
            this.backing = value;
        }

        /// <summary>The same, carrying <paramref name="actualDType"/> exactly as given — a
        /// specialized dtype's generic parameter name included.</summary>
        internal OnnxTensorData(
            Shape shape, IShorokooTensorValue value, DType actualDType, IShorokooBackend allocatingBackend)
            : base(shape, actualDType, allocatingBackend, SpaceOf(value, allocatingBackend))
        {
            this.backing = value;
        }

        /// <summary>
        /// Where a runtime value's bytes are. The value is asked first, because it knows: ONNX
        /// Runtime names the allocator a buffer came from, so a value a session produced on the host
        /// says so. Only when the answer is "not the host" does the allocating backend say
        /// <i>which</i> device, which is the one thing the value cannot — and a backend nobody
        /// named answers "somewhere unrecorded", which is the honest answer where there is no
        /// producer to ask.
        ///
        /// <para>That order matters. Letting the backend answer outright would label a genuinely
        /// host-readable value — a device session's ordinary output, which ONNX Runtime fetches to
        /// the host — as living on the card it came from, and every later hand-off of it would copy
        /// bytes that were already where they were wanted.</para>
        /// </summary>
        private static MemorySpace SpaceOf(IShorokooTensorValue value, IShorokooBackend allocatingBackend)
        {
            ArgumentNullException.ThrowIfNull(value);
            ArgumentNullException.ThrowIfNull(allocatingBackend);
            return value.IsHostAccessible ? MemorySpace.Host : allocatingBackend.MemorySpace;
        }

        /// <inheritdoc/>
        private protected override void ReleaseMemory() => AllocatingBackend.Release(backing);

        /// <inheritdoc/>
        private protected override IShorokooTensorValue ValueFor(IShorokooBackend backend) => backing;

        /// <summary>
        /// Whether a run on <paramref name="backend"/> can be handed this value as it stands: the
        /// backend can address the memory it is in — the same device and the same runtime — or, for
        /// a string tensor, the value is its own runtime's. ONNX Runtime keeps every string tensor
        /// in host memory whatever the provider, so a string is fed where it is to a session on a
        /// card too, and a copy could be put nowhere else.
        /// </summary>
        internal override bool FeedsInPlace(IShorokooBackend backend)
            => backend.CanAddress(Location)
               || (DType.IsSameElementTypeAs(DType.Utf8) && Space.IsHost
                   && ReferenceEquals(AllocatingBackend.RuntimeIdentity, backend.RuntimeIdentity));

        /// <inheritdoc/>
        private protected override byte[] CopyContentBytes()
        {
            // The allocating backend's copy only where the host cannot read the value itself: that
            // one is a native round trip, and a host value needs nothing but the span.
            if (!backing.IsHostAccessible) return AllocatingBackend.CopyTensorToHost(backing);
            var bytes = backing.GetTensorDataAsSpan<byte>().ToArray();
            // Taking the span is the value's last read, so without this the JIT may retire it
            // before ToArray has copied out of the buffer it points at (Shorokoo/Shorokoo#178).
            GC.KeepAlive(backing);
            return bytes;
        }

        /// <inheritdoc/>
        private protected override IReadOnlyList<string> CopyContentStrings()
            => backing.IsHostAccessible
                ? backing.GetStringTensorData()
                // Guarded like every other read of a backend-held tensor: a value in the
                // provider's own memory is not the host's to read, and saying so beats handing
                // back whatever the elements happen to collide with.
                : throw new InvalidOperationException(
                    $"This tensor ({Shape}:{DType}) lives in the execution provider's own memory, "
                    + "not host memory, so its elements cannot be read here. ToHost() takes a copy "
                    + "in host memory.");

        /// <inheritdoc/>
        public override bool IsHostResident => this.Value.IsHostAccessible;

        /// <summary>
        /// The backing value, checked to be readable from the host first. The span accessors below
        /// hand out a raw pointer with no idea what it points at, so a device-resident value would
        /// not fail on them — it would read whatever host address the device pointer happens to
        /// collide with. This turns that into an exception naming what to do instead.
        /// </summary>
        private IShorokooTensorValue HostValue => this.Value.IsHostAccessible ? this.Value
            : throw new InvalidOperationException(
                $"This tensor ({this.Shape}:{this.DType}) lives in the execution provider's own " +
                "memory, not host memory, so its contents cannot be read here. ToHost() takes a " +
                "copy in host memory; a ResidentTrainingRun, which keeps training state on the " +
                "device between steps, brings its state home with StepToCheckpoint(...) on the " +
                "step you want to read or save.");

        /// <summary>
        /// A writable span over the elements. Taking it retires every copy a run made of this
        /// tensor, since the contents they were copied from are about to change: the next run makes
        /// a fresh one, and a run still reading an old copy finishes on it.
        /// </summary>
        public override Span<V> AccessModifiableMemory<V>()
        {
            var value = this.HostValue;
            Written();
            return value.GetTensorMutableDataAsSpan<V>();
        }

        /// <inheritdoc/>
        public override ReadOnlySpan<V> AccessMemory<V>()
        {
            return this.HostValue.GetTensorDataAsSpan<V>();
        }

        /// <summary>A writable byte span over the storage, retiring the copies runs made of this
        /// tensor as <see cref="AccessModifiableMemory{V}"/> does.</summary>
        public override Span<byte> AccessModifiableRawMemory()
        {
            var value = this.HostValue;
            Written();
            return value.GetTensorMutableDataAsSpan<byte>();
        }
        /// <inheritdoc/>
        public override ReadOnlySpan<byte> AccessRawMemory()
        {
            return this.HostValue.GetTensorDataAsSpan<byte>();
        }

        // There is deliberately no finalizer -- one here could only release the backing value,
        // and a finalizer must not touch another managed object that may already have been
        // finalized itself. The backing value has its own finalizer, which is what reclaims a
        // tensor nobody deletes (Shorokoo/Shorokoo#180).
    }
}

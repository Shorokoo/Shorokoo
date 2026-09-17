using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using Shorokoo.Core.Inference.Abstractions;
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
        internal TensorData(Shape shape) : base(shape, OnnxUtils.GetDType<T>())
        {
        }

        internal TensorData(Shape shape, TensorStorage storage, ComputeContext? context, bool ownsMemory)
            : base(shape, OnnxUtils.GetDType<T>(), storage, context, ownsMemory)
        {
        }

        internal TensorData(
            Shape shape, DType dtype, TensorStorage storage, ComputeContext? context, bool ownsMemory)
            : base(shape, dtype, storage, context, ownsMemory)
        {
        }

        internal TensorData(Shape shape, DType dtype) : base(shape, dtype)
        {
        }

        /// <summary>Exposes the underlying buffer as a writable span of V (V must match T's storage
        /// type). The span points straight into the tensor's storage, so it is valid only while the
        /// tensor is — see <see cref="TensorData.AccessRawMemory"/>.</summary>
        public abstract Span<V> AccessModifiableMemory<V>() where V : unmanaged;

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
    /// </summary>
    public abstract partial class TensorData : IData, IDisposable
    {
        /// <summary>The tensor's shape.</summary>
        public Shape Shape { get; private set; }
        /// <summary>The element data type.</summary>
        public DType DType { get; private set; }

        /// <summary>The raw storage bytes boxed as objects, for debugging/diagnostics.</summary>
        public virtual object[] Data
        {
            get
            {
                return this.CopyRawMemory().Cast<object>().ToArray();
            }
        }

        internal TensorData(Shape shape, DType dtype)
            : this(shape, dtype, TensorStorage.None, context: null, ownsMemory: true) { }

        internal TensorData(
            Shape shape, DType dtype, TensorStorage storage, ComputeContext? context, bool ownsMemory)
        {
            this.Shape = shape;
            this.DType = dtype;
            this.Storage = storage;
            this.Context = context;
            this.OwnsMemory = ownsMemory;

            // Invariant: a tensor with no context is the framework's own -- plain host memory that
            // it owns. There is no other kind, and a context-free tensor in device memory would
            // have no way to say which device or to reach it.
            if (context is null && !storage.Space.IsHost && storage.Space.IsKnown)
                throw new ArgumentException(
                    $"A tensor with no compute context holds host memory, but this storage is in "
                    + $"{storage.Space}. Give it the context whose memory that is.", nameof(storage));
            if (context is null && !ownsMemory)
                throw new ArgumentException(
                    "A tensor with no compute context owns its memory: there is no other tensor or "
                    + "backend that could own it instead.", nameof(ownsMemory));

            // Taking ownership puts the bytes on this context's books, so disposing the context
            // releases them -- and takes them off whoever had them before, so disposing that one
            // does not.
            if (ownsMemory) storage.TransferOwnershipTo(context);
        }

        /// <summary>
        /// The bytes, shared with any other tensor naming the same memory. Internal because
        /// ownership is expressed through <see cref="OwnsMemory"/> and the transfer operations;
        /// nothing outside needs the handle itself.
        /// </summary>
        internal TensorStorage Storage { get; private set; }

        /// <summary>
        /// The compute context whose memory this tensor's bytes are in, or null when they are in
        /// ordinary host memory belonging to no backend — which is what every tensor built by the
        /// convenience constructors is, and what every tensor used as an operator attribute must be.
        /// </summary>
        public ComputeContext? Context { get; private set; }

        /// <summary>
        /// Whether this tensor is the one responsible for releasing its bytes. False for a tensor
        /// that was only given access to memory another owns; disposing such a tensor frees
        /// nothing.
        /// </summary>
        public bool OwnsMemory { get; private set; }

        /// <summary>Where this tensor's bytes are. Derived from the storage, never set.</summary>
        public MemorySpace Space => Storage.Space;

        /// <summary>
        /// Gives up ownership without releasing anything — the other half of a transfer. The
        /// storage's owner has already moved on by the time this runs, so there is nothing to
        /// deregister here.
        /// </summary>
        internal void SurrenderOwnership() => OwnsMemory = false;

        /// <summary>
        /// True once <see cref="Dispose"/> has released this tensor's storage. Its shape, dtype and
        /// <see cref="ToString"/> stay readable as metadata; every path to the elements throws.
        /// </summary>
        public bool IsDisposed { get; protected set; }

        /// <summary>Guards every path to the tensor's elements. Call it before touching storage.</summary>
        protected void ThrowIfDisposed()
        {
            if (IsDisposed)
                throw new ObjectDisposedException(GetType().Name,
                    $"Tensor {this} has been disposed; its storage is gone and reading it would " +
                    "read freed memory.");

            // Not the same check. This tensor may be perfectly undisposed and still be pointing at
            // memory whose owner released it -- that is exactly what a tensor holding borrowed
            // storage is exposed to, and the whole reason liveness lives on the storage.
            if (!Storage.IsLive)
                throw new ObjectDisposedException(GetType().Name,
                    $"Tensor {this} reads memory owned by something that has since released it -- "
                    + "the compute context it belonged to was disposed, or the tensor that owned "
                    + "the memory was. Reading it would read freed memory. Take a CopyTo(...) "
                    + "while the owner is alive if the data has to outlive it.");
        }

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
        /// its lifetime to the tensor's. It is valid only while the tensor is undisposed AND still
        /// reachable: disposing the tensor frees what the span points at (later reads through the
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
        /// may be called. Like every path to the elements it throws once the tensor is disposed,
        /// rather than answering about storage that is gone — ask <see cref="IsDisposed"/> first if
        /// a tensor may have been released. It is <c>false</c> only for a tensor an execution provider produced in
        /// its own memory and a <see cref="ResidentTrainingRun"/> deliberately left there; reading
        /// such a tensor throws, and <see cref="ResidentTrainingRun.StepToCheckpoint(TensorDataStruct, TensorDataStruct)"/>
        /// is what brings one back to the host.
        /// </summary>
        public virtual bool IsHostResident => true;

        /// <summary>Downcasts to the typed <see cref="TensorData{T}"/>; T must match the actual element type.</summary>
        public TensorData<T> As<T>() where T : IVarType => (TensorData<T>)this;

        /// <summary>
        /// Creates TensorData backed by an existing inference-runtime tensor value, belonging to no
        /// compute context — the framework's own host memory, which is where a value it built
        /// itself is. A value a session produced comes with the context that produced it instead,
        /// so that it can say where it is; that is the internal overload below, and every path
        /// through <c>ComputeContext</c> takes it.
        /// </summary>
        public static TensorData Create(Shape shape, DType dtype, IShorokooTensorValue data)
        {
            return OnnxUtils.CreateTensorDataFromValue(shape, dtype, data);
        }

        /// <summary>A backend-backed tensor bound to the context whose memory it is in.</summary>
        internal static TensorData Create(
            Shape shape, DType dtype, IShorokooTensorValue data, ComputeContext? context)
            => OnnxUtils.CreateTensorDataFromValue(shape, dtype, data, context);

        /// <summary>A host tensor over the given bytes, bound to the given host context.</summary>
        internal static TensorData NewHostTensor(
            Shape shape, DType dtype, byte[] bytes, ComputeContext? context)
            => OnnxUtils.CreateHostTensorData(shape, dtype, bytes, context);

        /// <summary>A host string tensor over the given elements, bound to the given host
        /// context.</summary>
        internal static TensorData NewHostStringTensor(
            Shape shape, string[] values, ComputeContext? context)
            => HostStringTensorData.Bound(shape, values, context);

        /// <summary>
        /// Creates TensorData of the given shape and dtype over <paramref name="data"/> — plain
        /// host memory belonging to no backend, the same thing <see cref="NewHostTensor"/> makes.
        ///
        /// <para>Raw bytes are what a tensor read out of a model file, or zeroed for a gradient
        /// buffer, already is; wrapping them describes data rather than running anything. This
        /// went through <c>InferenceBackend.Factory</c> instead, so reading an <c>.onnx</c> file
        /// resolved the process-wide backend and put a native allocation behind every initializer
        /// in it. The value is built when a session is fed this tensor, in
        /// <see cref="ToTensorValue(IShorokooInferenceSessionFactory)"/>, and not before.</para>
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
        /// <see cref="DType.String"/>, or <paramref name="shape"/> has no known element
        /// count.</exception>
        /// <exception cref="UnsupportedDTypeException"><paramref name="dtype"/> has no whole-byte
        /// element stride, so no flat buffer can describe it.</exception>
        public static TensorData CreateFromRawBytes(Shape shape, DType dtype, byte[] data)
        {
            ArgumentNullException.ThrowIfNull(data);
            // The refusal the backend used to give, kept where it can still be given eagerly: a
            // string element is variable-length, so a flat byte buffer does not describe one and
            // a HostTensorData<@string> over these bytes would be a tensor of nothing.
            if (dtype == DType.String)
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
            return NewHostTensor(
                shape, dtype, data.AsSpan(0, (int)required).ToArray(), context: null);
        }

        /// <summary>
        /// Returns the backing inference-runtime tensor value, on the process-wide backend;
        /// throws if this instance has none and none can be built.
        /// </summary>
        public IShorokooTensorValue ToTensorValue() => ToTensorValue(InferenceBackend.Factory);

        /// <summary>
        /// This tensor as a value of <paramref name="factory"/>'s runtime. A tensor that already
        /// holds one hands it over and ignores the argument, since a value belongs to the runtime
        /// that made it; one held in plain host memory builds it here, which is the first moment a
        /// backend is needed at all.
        ///
        /// <para>The value returned is the tensor's own: read it, do not dispose it.</para>
        /// </summary>
        internal virtual IShorokooTensorValue ToTensorValue(IShorokooInferenceSessionFactory factory)
        {
            ThrowIfDisposed();
            if (this is IOnnxData od) return od.Value;
            throw new InvalidOperationException(
                $"TensorData of type {this.GetType().Name} does not expose an inference-runtime tensor value.");
        }

        /// <summary>Creates int32 TensorData of the given shape holding 0, 1, ..., Count-1 in row-major order.</summary>
        public static TensorData<int32> BuildRange(Shape shape)
        {
            var vals = Enumerable.Range(0, (int)shape.Count).ToArray();
            return (TensorData<int32>)TensorData(shape.Dims, vals);
        }

        /// <summary>Releases the underlying storage.</summary>
        public abstract void Dispose();

        /// <summary>
        /// A second tensor over the very same bytes, with the context and ownership given. The
        /// storage handle is shared, not copied, so releasing it through one of them is visible
        /// to the other -- which is what makes a reader's access check work.
        /// </summary>
        internal abstract TensorData CloneSharing(ComputeContext? context, bool ownsMemory);
    }

    /// <summary>TensorData backed by an inference-runtime tensor value.</summary>
    public interface IOnnxData
    {
        /// <summary>The backing inference-runtime tensor value.</summary>
        public IShorokooTensorValue Value { get; }
    }

    /// <summary>
    /// <see cref="TensorData{T}"/> implementation backed by an inference-runtime
    /// (ONNX) tensor value; span access reads the runtime tensor's buffer directly.
    /// </summary>
    public sealed class OnnxTensorData<T> : TensorData<T>, IOnnxData, IDisposable
        where T : IVarType
    {
        private readonly IShorokooTensorValue backing;

        /// <summary>
        /// The backing inference-runtime tensor value, which this tensor owns: disposing the
        /// tensor releases it, and nothing else may hold or free it (Shorokoo/Shorokoo#180).
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
                return this.CopyMemory<byte>().Cast<object>().ToArray();
            }
        }

        /// <summary>
        /// Creates TensorData of the given shape around an existing runtime tensor value; the dtype
        /// is derived from T. The tensor belongs to no compute context, so its value must be one
        /// the host can read — every path that wraps a session's output hands over the context that
        /// produced it, and a value in a provider's own memory needs that context to say which
        /// memory it is (see <see cref="StorageFor"/>).
        /// </summary>
        public OnnxTensorData(Shape shape, IShorokooTensorValue value)
            : this(shape, value, context: null, ownsMemory: true, storage: null)
        {
        }

        internal OnnxTensorData(Shape shape, IShorokooTensorValue value, DType actualDType)
            : base(shape, actualDType, StorageFor(value, null), null, true)
        {
            this.backing = value;
        }

        internal OnnxTensorData(
            Shape shape, IShorokooTensorValue value, ComputeContext? context, bool ownsMemory,
            TensorStorage? storage)
            : base(shape, storage ?? StorageFor(value, context), context, ownsMemory)
        {
            this.backing = value;
        }

        /// <summary>
        /// Where a runtime value's bytes are: the space of the context this tensor belongs to. A
        /// tensor of a context holds that context's memory — host memory on a host backend, the
        /// card's own on a CUDA one — and that is what decides whether handing it to another
        /// context has to copy anything. Releasing the storage disposes the value, which is what
        /// owning it means.
        ///
        /// <para>The value is asked first, because it knows: ONNX Runtime names the allocator a
        /// buffer came from, so a tensor the backend allocated on the card says so and one it
        /// allocated on the host says that. Only when the answer is "not the host" does the
        /// context decide <i>which</i> device, which is the one thing the value cannot say.</para>
        ///
        /// <para>That order matters. Letting the context answer outright would label a genuinely
        /// host-readable value — a resident run's published state, say, which was deliberately
        /// brought home — as living on the card it came from, and every later hand-off of it would
        /// copy bytes that were already where they were wanted.</para>
        /// </summary>
        private static TensorStorage StorageFor(IShorokooTensorValue value, ComputeContext? context)
        {
            if (value.IsHostAccessible) return new TensorStorage(MemorySpace.Host, value.Dispose);
            // A value the provider kept, wrapped without the context that produced it, is somewhere
            // this cannot name. Recorded as unknown rather than guessed at: a wrong device id would
            // make two unrelated allocations look like one space and invite a transfer between them.
            // Nothing the framework runs arrives here without one -- a session's outputs, a
            // sequence's elements and a transfer's results all carry theirs -- so this is reached
            // only by a caller wrapping a value of its own.
            return new TensorStorage(
                context?.MemorySpace ?? MemorySpace.UnknownDevice, value.Dispose);
        }

        /// <inheritdoc/>
        internal override TensorData CloneSharing(ComputeContext? context, bool ownsMemory)
            => new OnnxTensorData<T>(Shape, backing, context, ownsMemory, Storage);

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
                "memory, not host memory, so its contents cannot be read here. It belongs to a " +
                "ResidentTrainingRun, which keeps training state on the device between steps; take " +
                "a host copy of the state with StepToCheckpoint(...) on the step you want to read " +
                "or save.");

        /// <inheritdoc/>
        public override Span<V> AccessModifiableMemory<V>()
        {
            return this.HostValue.GetTensorMutableDataAsSpan<V>();
        }

        /// <inheritdoc/>
        public override ReadOnlySpan<V> AccessMemory<V>()
        {
            return this.HostValue.GetTensorDataAsSpan<V>();
        }

        /// <inheritdoc/>
        public override Span<byte> AccessModifiableRawMemory()
        {
            return this.HostValue.GetTensorMutableDataAsSpan<byte>();
        }
        /// <inheritdoc/>
        public override ReadOnlySpan<byte> AccessRawMemory()
        {
            return this.HostValue.GetTensorDataAsSpan<byte>();
        }

        #region IDisposable

        /// <summary>
        /// Releases the backing value's buffer. Idempotent; every read afterwards throws
        /// <see cref="ObjectDisposedException"/> rather than reading freed memory.
        ///
        /// <para>There is deliberately no finalizer. One here could only release the backing
        /// value, and a finalizer must not touch another managed object that may already have
        /// been finalized itself. The backing value has its own finalizer, which is what reclaims
        /// a tensor nobody disposes; adding a second one would put every tensor in the framework
        /// on the finalization queue to duplicate it (Shorokoo/Shorokoo#180).</para>
        /// </summary>
        public override void Dispose()
        {
            if (IsDisposed) return;
            IsDisposed = true;
            // Only the owner releases. A tensor that was merely given access to these bytes leaves
            // them alone; that is the whole content of not owning them.
            if (OwnsMemory) Storage.Release();
        }

        #endregion
    }
}

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

        internal TensorData(Shape shape, DType dtype) : base(shape, dtype)
        {
        }

        /// <summary>Exposes the underlying buffer as a writable span of V (V must match T's storage type).</summary>
        public abstract Span<V> AccessModifiableMemory<V>() where V : unmanaged;
        /// <summary>Exposes the underlying buffer as a read-only span of V (V must match T's storage type).</summary>
        public abstract ReadOnlySpan<V> AccessMemory<V>() where V : unmanaged;

        /// <summary>The element values boxed as objects, for debugging/diagnostics.</summary>
        public object[] DebugData
        {
            get
            {
                switch(typeof(T))
                {
                    case Type t when t == typeof(bit):
                        return this.AccessMemory<bool>().ToArray().Cast<object>().ToArray();
                    case Type t when t == typeof(int8):
                        return this.AccessMemory<sbyte>().ToArray().Cast<object>().ToArray();
                    case Type t when t == typeof(int16):
                        return this.AccessMemory<short>().ToArray().Cast<object>().ToArray();
                    case Type t when t == typeof(int32):
                        return this.AccessMemory<int>().ToArray().Cast<object>().ToArray();
                    case Type t when t == typeof(int64):
                        return this.AccessMemory<long>().ToArray().Cast<object>().ToArray();
                    case Type t when t == typeof(uint8):
                        return this.AccessMemory<byte>().ToArray().Cast<object>().ToArray();
                    case Type t when t == typeof(uint16):
                        return this.AccessMemory<ushort>().ToArray().Cast<object>().ToArray();
                    case Type t when t == typeof(uint32):
                        return this.AccessMemory<uint>().ToArray().Cast<object>().ToArray();
                    case Type t when t == typeof(uint64):
                        return this.AccessMemory<ulong>().ToArray().Cast<object>().ToArray();
                    case Type t when t == typeof(float16):
                        return this.AccessMemory<Float16>().ToArray().Cast<object>().ToArray();
                    case Type t when t == typeof(bfloat16):
                        return this.AccessMemory<BFloat16>().ToArray().Cast<object>().ToArray();
                    case Type t when t == typeof(float32):
                        return this.AccessMemory<float>().ToArray().Cast<object>().ToArray();
                    case Type t when t == typeof(float64):
                        return this.AccessMemory<double>().ToArray().Cast<object>().ToArray();
                    default:
                        return this.AccessMemory<byte>().ToArray().Cast<object>().ToArray();
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
    public abstract class TensorData : IData, IDisposable
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
                return this.AccessRawMemory().ToArray().Cast<object>().ToArray();
            }
        }

        internal TensorData(Shape shape, DType dtype)
        {
            this.Shape = shape;
            this.DType = dtype;
        }

        /// <summary>"shape:dtype" diagnostic string.</summary>
        public override string ToString()
        {
            var shapeStr = this.Shape.ToString();
            return $"{shapeStr}:{this.DType.ToString()}";
        }

        /// <summary>Exposes the underlying storage as a writable byte span.</summary>
        public abstract Span<byte> AccessModifiableRawMemory();
        /// <summary>Exposes the underlying storage as a read-only byte span.</summary>
        public abstract ReadOnlySpan<byte> AccessRawMemory();

        /// <summary>Downcasts to the typed <see cref="TensorData{T}"/>; T must match the actual element type.</summary>
        public TensorData<T> As<T>() where T : IVarType => (TensorData<T>)this;

        /// <summary>Creates TensorData backed by an existing inference-runtime tensor value.</summary>
        public static TensorData Create(Shape shape, DType dtype, IShorokooTensorValue data)
        {
            return OnnxUtils.CreateTensorDataFromValue(shape, dtype, data);
        }

        /// <summary>Creates TensorData of the given shape and dtype from raw storage bytes.</summary>
        public static TensorData CreateFromRawBytes(Shape shape, DType dtype, byte[] data)
        {
            var value = OnnxUtils.CreateTensorValueFromRawData(shape, dtype, data);
            return Create(shape, dtype, value);
        }

        /// <summary>Returns the backing inference-runtime tensor value; throws if this instance has none.</summary>
        public IShorokooTensorValue ToTensorValue()
        {
            if (this is IOnnxData od) return od.Value;
            throw new InvalidOperationException(
                $"TensorData of type {this.GetType().Name} does not expose an inference-runtime tensor value.");
        }

        /// <summary>Creates int32 TensorData of the given shape holding 0, 1, ..., Count-1 in row-major order.</summary>
        public static OnnxTensorData<int32> BuildRange(Shape shape)
        {
            var vals = Enumerable.Range(0, (int)shape.Count).ToArray();
            return (OnnxTensorData<int32>)TensorData(shape.Dims, vals);
        }

        /// <summary>Releases the underlying storage.</summary>
        public abstract void Dispose();
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
    public class OnnxTensorData<T> : TensorData<T>, IOnnxData, IDisposable
        where T : IVarType
    {
        private bool disposedValue = false;

        /// <summary>The backing inference-runtime tensor value.</summary>
        public IShorokooTensorValue Value { get; private set; }

        /// <summary>The raw storage bytes boxed as objects, for debugging/diagnostics.</summary>
        public override object[] Data
        {
            get
            {
                return this.AccessMemory<byte>().ToArray().Cast<object>().ToArray();
            }
        }

        /// <summary>Creates TensorData of the given shape around an existing runtime tensor value; the dtype is derived from T.</summary>
        public OnnxTensorData(Shape shape, IShorokooTensorValue value) : base(shape)
        {
            this.Value = value;
        }

        internal OnnxTensorData(Shape shape, IShorokooTensorValue value, DType actualDType) : base(shape, actualDType)
        {
            this.Value = value;
        }

        /// <inheritdoc/>
        public override Span<V> AccessModifiableMemory<V>()
        {
            return this.Value.GetTensorMutableDataAsSpan<V>();
        }

        /// <inheritdoc/>
        public override ReadOnlySpan<V> AccessMemory<V>()
        {
            return this.Value.GetTensorDataAsSpan<V>();
        }

        /// <inheritdoc/>
        public override Span<byte> AccessModifiableRawMemory()
        {
            return this.Value.GetTensorMutableDataAsSpan<byte>();
        }
        /// <inheritdoc/>
        public override ReadOnlySpan<byte> AccessRawMemory()
        {
            return this.Value.GetTensorDataAsSpan<byte>();
        }

        #region IDisposable

        /// <summary>Standard dispose pattern hook; the runtime value owns the native buffer.</summary>
        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                }
                disposedValue = true;
            }
        }

        /// <summary>Finalizer running the dispose pattern.</summary>
        ~OnnxTensorData()
        {
            Dispose(disposing: false);
        }

        /// <inheritdoc/>
        public override void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }

        #endregion
    }
}

using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Backends;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Utils;
using Shorokoo.Core.Nodes;

namespace Shorokoo.Core.Nodes.Processors.Helpers
{
    /// <summary>
    /// Shared helper methods for TensorData type conversion used by multiple processors.
    /// </summary>
    internal static class TensorDataConversion
    {
        /// <summary>
        /// The same conversion for a graph literal — an operator's <see cref="TensorAttribute"/>.
        ///
        /// <para>A dtype that differs only in its generic parameter name is answered here outright:
        /// an attribute is immutable, so the one at the new dtype shares its bytes and costs
        /// nothing — where the tensor form has to copy, and has to resolve a backend to do it. A
        /// real conversion goes through the tensor path, on a copy, which is what a description
        /// changing its element type is.</para>
        /// </summary>
        internal static TensorAttribute ConvertAttributeType(TensorAttribute original, DType targetDType)
        {
            if (original.DType.ProtoTypeNum == targetDType.ProtoTypeNum)
                return original.DType.GenericTypeParamName == targetDType.GenericTypeParamName
                    ? original : original.WithDType(targetDType);

            // At the dtype the bytes are laid out at: a generic placeholder describes no layout,
            // so reading through it is what the tensor form did by way of its runtime value.
            var source = original.CopyToTensorData(atStorageDType: true);
            // Every remaining branch of ConvertTensorDataType builds a tensor of its own, so this
            // copy is nobody's once it has been read.
            try { return ConvertTensorDataType(source, targetDType).MoveToAttribute(); }
            finally { source.Dispose(); }
        }

        /// <summary>
        /// Converts TensorData from one data type to another by reading values and creating new TensorData.
        /// Used when processing generic constants that need type conversion during specialization.
        /// </summary>
        /// <param name="originalData">The source TensorData to convert</param>
        /// <param name="targetDType">The target DType to convert to</param>
        /// <returns>New TensorData with converted values</returns>
        internal static TensorData ConvertTensorDataType(TensorData originalData, DType targetDType)
        {
            var shape = originalData.Shape;
            var sourceDType = originalData.DType;
            
            // Check if types match AND generic metadata matches
            bool typesMatch = sourceDType.ProtoTypeNum == targetDType.ProtoTypeNum;
            bool metadataMatches = sourceDType.GenericTypeParamName == targetDType.GenericTypeParamName;
            
            // If everything matches, return as-is
            if (typesMatch && metadataMatches)
                return originalData;
            
            // If types match but metadata differs, we need to recreate with the new metadata
            // without converting the data
            if (typesMatch && !metadataMatches)
            {
                // Same bytes, different metadata — but on storage of its own, like every branch
                // that converts. (The branch above, where nothing needs converting at all, hands
                // the source straight back rather than copying it.) Wrapping the source's own
                // value handed the caller back a second
                // tensor over one runtime value, so whichever was disposed first took the other
                // one's storage with it (Shorokoo/Shorokoo#180). The source graph may well still
                // be in use: these passes rebuild attributes into a new graph and leave the
                // original standing.
                //
                // Asked of the tensor rather than cast out of it, so that a literal held in plain
                // managed memory answers too: a HostTensorData builds the runtime value here, and
                // a backend-backed one hands over the one it already has. The cast would have
                // thrown on the first of those, and this path already needs a backend to copy on.
                // Under the tensor's reader lock, as every copy out of a tensor made outside a run is.
                var copy = originalData.Reading(() => OnnxUtils.CopyTensorValue(originalData.ToTensorValue()));
                return OnnxUtils.CreateTensorDataFromValue(
                    shape, targetDType, copy, targetDType, DefaultBackend.Instance);
            }
            
            // Extract element count for conversion
            int elementCount = (int)shape.Count;
            
            // Convert based on target type (compare ProtoTypeNum to handle DTTypes with generic metadata)
            // This uses a similar approach to Global.Constructors.Scalar<T>(object)
            if (targetDType.ProtoTypeNum == DType.Bool.ProtoTypeNum) return ConvertToType<bit>(originalData, sourceDType, elementCount, targetDType);
            else if (targetDType.ProtoTypeNum == DType.Int8.ProtoTypeNum) return ConvertToType<int8>(originalData, sourceDType, elementCount, targetDType);
            else if (targetDType.ProtoTypeNum == DType.Int16.ProtoTypeNum) return ConvertToType<int16>(originalData, sourceDType, elementCount, targetDType);
            else if (targetDType.ProtoTypeNum == DType.Int32.ProtoTypeNum) return ConvertToType<int32>(originalData, sourceDType, elementCount, targetDType);
            else if (targetDType.ProtoTypeNum == DType.Int64.ProtoTypeNum) return ConvertToType<int64>(originalData, sourceDType, elementCount, targetDType);
            else if (targetDType.ProtoTypeNum == DType.UInt8.ProtoTypeNum) return ConvertToType<uint8>(originalData, sourceDType, elementCount, targetDType);
            else if (targetDType.ProtoTypeNum == DType.UInt16.ProtoTypeNum) return ConvertToType<uint16>(originalData, sourceDType, elementCount, targetDType);
            else if (targetDType.ProtoTypeNum == DType.UInt32.ProtoTypeNum) return ConvertToType<uint32>(originalData, sourceDType, elementCount, targetDType);
            else if (targetDType.ProtoTypeNum == DType.UInt64.ProtoTypeNum) return ConvertToType<uint64>(originalData, sourceDType, elementCount, targetDType);
            else if (targetDType.ProtoTypeNum == DType.Float16.ProtoTypeNum) return ConvertToType<float16>(originalData, sourceDType, elementCount, targetDType);
            else if (targetDType.ProtoTypeNum == DType.BFloat16.ProtoTypeNum) return ConvertToType<bfloat16>(originalData, sourceDType, elementCount, targetDType);
            else if (targetDType.ProtoTypeNum == DType.Float32.ProtoTypeNum) return ConvertToType<float32>(originalData, sourceDType, elementCount, targetDType);
            else if (targetDType.ProtoTypeNum == DType.Float64.ProtoTypeNum) return ConvertToType<float64>(originalData, sourceDType, elementCount, targetDType);
            else throw new NotSupportedException($"Conversion to {targetDType} is not supported");
        }

        private static TensorData ConvertToType<TTarget>(TensorData originalData, DType sourceDType, int elementCount, DType targetDType) where TTarget : IVarType
        {
            // Extract values from source based on source type
            object[] sourceValues = ExtractValues(originalData, sourceDType, elementCount);
            
            // Convert to target type and create OrtValue
            IShorokooTensorValue ortValue;
            if (typeof(TTarget) == typeof(bit))
            {
                var converted = sourceValues.Select(v => Convert.ToBoolean(v)).ToArray();
                ortValue = OnnxUtils.CreateTensorValue(originalData.Shape, converted);
            }
            else if (typeof(TTarget) == typeof(int8))
            {
                var converted = sourceValues.Select(v => Convert.ToSByte(v)).ToArray();
                ortValue = OnnxUtils.CreateTensorValue(originalData.Shape, converted);
            }
            else if (typeof(TTarget) == typeof(int16))
            {
                var converted = sourceValues.Select(v => Convert.ToInt16(v)).ToArray();
                ortValue = OnnxUtils.CreateTensorValue(originalData.Shape, converted);
            }
            else if (typeof(TTarget) == typeof(int32))
            {
                var converted = sourceValues.Select(v => Convert.ToInt32(v)).ToArray();
                ortValue = OnnxUtils.CreateTensorValue(originalData.Shape, converted);
            }
            else if (typeof(TTarget) == typeof(int64))
            {
                var converted = sourceValues.Select(v => Convert.ToInt64(v)).ToArray();
                ortValue = OnnxUtils.CreateTensorValue(originalData.Shape, converted);
            }
            else if (typeof(TTarget) == typeof(uint8))
            {
                var converted = sourceValues.Select(v => Convert.ToByte(v)).ToArray();
                ortValue = OnnxUtils.CreateTensorValue(originalData.Shape, converted);
            }
            else if (typeof(TTarget) == typeof(uint16))
            {
                var converted = sourceValues.Select(v => Convert.ToUInt16(v)).ToArray();
                ortValue = OnnxUtils.CreateTensorValue(originalData.Shape, converted);
            }
            else if (typeof(TTarget) == typeof(uint32))
            {
                var converted = sourceValues.Select(v => Convert.ToUInt32(v)).ToArray();
                ortValue = OnnxUtils.CreateTensorValue(originalData.Shape, converted);
            }
            else if (typeof(TTarget) == typeof(uint64))
            {
                var converted = sourceValues.Select(v => Convert.ToUInt64(v)).ToArray();
                ortValue = OnnxUtils.CreateTensorValue(originalData.Shape, converted);
            }
            else if (typeof(TTarget) == typeof(float16))
            {
                // IEEE binary16 via the ushort-backed Float16 struct: the float32→half
                // narrowing (System.Half) rounds to nearest-even.
                var converted = sourceValues.Select(v => (Float16)Convert.ToSingle(v)).ToArray();
                ortValue = OnnxUtils.CreateTensorValue(originalData.Shape, converted);
            }
            else if (typeof(TTarget) == typeof(bfloat16))
            {
                // bfloat16 = top 16 bits of float32; BFloat16's float→bf16 cast rounds
                // to nearest-even before truncating.
                var converted = sourceValues.Select(v => (BFloat16)Convert.ToSingle(v)).ToArray();
                ortValue = OnnxUtils.CreateTensorValue(originalData.Shape, converted);
            }
            else if (typeof(TTarget) == typeof(float32))
            {
                var converted = sourceValues.Select(v => Convert.ToSingle(v)).ToArray();
                ortValue = OnnxUtils.CreateTensorValue(originalData.Shape, converted);
            }
            else if (typeof(TTarget) == typeof(float64))
            {
                var converted = sourceValues.Select(v => Convert.ToDouble(v)).ToArray();
                ortValue = OnnxUtils.CreateTensorValue(originalData.Shape, converted);
            }
            else
            {
                throw new NotSupportedException($"Conversion to {typeof(TTarget).Name} is not supported");
            }
            
            // Create TensorData with the targetDType which may include generic metadata
            // Use the internal constructor that accepts explicit DType
            return new OnnxTensorData<TTarget>(originalData.Shape, ortValue, targetDType, DefaultBackend.Instance);
        }

        private static object[] ExtractValues(TensorData data, DType dtype, int count)
        {
            // Use ProtoTypeNum for comparison to handle DTTypes with generic metadata
            // For generic types (IGenericType1-8), get the actual data type from the OrtValue
            if (dtype.IsGenericType)
            {
                // For generic types, we need to determine the actual data type from the OrtValue.
                // Asked of the tensor, not cast out of it: a literal in plain managed memory has no
                // runtime value to cast to and builds one on demand, and either way the value
                // stays the tensor's to dispose.
                var ortValue = data.ToTensorValue();
                var actualDType = (DType)(int)ortValue.ElementType;
                
                // Extract directly from OrtValue based on actual element type
                return ExtractValuesFromOrtValue(ortValue, actualDType);
            }
            
            object[] values;
            if (dtype.ProtoTypeNum == DType.Bool.ProtoTypeNum) values = data.As<bit>().CopyMemory<bool>().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.Int8.ProtoTypeNum) values = data.As<int8>().CopyMemory<sbyte>().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.Int16.ProtoTypeNum) values = data.As<int16>().CopyMemory<short>().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.Int32.ProtoTypeNum) values = data.As<int32>().CopyMemory<int>().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.Int64.ProtoTypeNum) values = data.As<int64>().CopyMemory<long>().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.UInt8.ProtoTypeNum) values = data.As<uint8>().CopyMemory<byte>().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.UInt16.ProtoTypeNum) values = data.As<uint16>().CopyMemory<ushort>().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.UInt32.ProtoTypeNum) values = data.As<uint32>().CopyMemory<uint>().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.UInt64.ProtoTypeNum) values = data.As<uint64>().CopyMemory<ulong>().Cast<object>().ToArray();
            // F16/BF16 widen exactly to float32, so extract as floats — keeps the
            // downstream Convert.To* calls working for every target type.
            else if (dtype.ProtoTypeNum == DType.Float16.ProtoTypeNum) values = data.As<float16>().CopyMemory<Float16>().Select(v => (object)(float)v).ToArray();
            else if (dtype.ProtoTypeNum == DType.BFloat16.ProtoTypeNum) values = data.As<bfloat16>().CopyMemory<BFloat16>().Select(v => (object)(float)v).ToArray();
            else if (dtype.ProtoTypeNum == DType.Float32.ProtoTypeNum) values = data.As<float32>().CopyMemory<float>().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.Float64.ProtoTypeNum) values = data.As<float64>().CopyMemory<double>().Cast<object>().ToArray();
            else throw new NotSupportedException($"Extraction from {dtype} is not supported");
            return values;
        }
        
        private static object[] ExtractValuesFromOrtValue(IShorokooTensorValue ortValue, DType dtype)
        {
            // Extract values directly from OrtValue without going through TensorData.As<T>()
            object[] values;
            if (dtype.ProtoTypeNum == DType.Bool.ProtoTypeNum) values = ortValue.GetTensorDataAsSpan<bool>().ToArray().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.Int8.ProtoTypeNum) values = ortValue.GetTensorDataAsSpan<sbyte>().ToArray().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.Int16.ProtoTypeNum) values = ortValue.GetTensorDataAsSpan<short>().ToArray().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.Int32.ProtoTypeNum) values = ortValue.GetTensorDataAsSpan<int>().ToArray().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.Int64.ProtoTypeNum) values = ortValue.GetTensorDataAsSpan<long>().ToArray().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.UInt8.ProtoTypeNum) values = ortValue.GetTensorDataAsSpan<byte>().ToArray().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.UInt16.ProtoTypeNum) values = ortValue.GetTensorDataAsSpan<ushort>().ToArray().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.UInt32.ProtoTypeNum) values = ortValue.GetTensorDataAsSpan<uint>().ToArray().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.UInt64.ProtoTypeNum) values = ortValue.GetTensorDataAsSpan<ulong>().ToArray().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.Float16.ProtoTypeNum) values = ortValue.GetTensorDataAsSpan<Float16>().ToArray().Select(v => (object)(float)v).ToArray();
            else if (dtype.ProtoTypeNum == DType.BFloat16.ProtoTypeNum) values = ortValue.GetTensorDataAsSpan<BFloat16>().ToArray().Select(v => (object)(float)v).ToArray();
            else if (dtype.ProtoTypeNum == DType.Float32.ProtoTypeNum) values = ortValue.GetTensorDataAsSpan<float>().ToArray().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.Float64.ProtoTypeNum) values = ortValue.GetTensorDataAsSpan<double>().ToArray().Cast<object>().ToArray();
            else throw new NotSupportedException($"Extraction from {dtype} is not supported");
            // Every branch reads through a span pointing into ortValue's own buffer, and
            // taking that span is ortValue's last read — so without this the JIT may retire it
            // and a collection free the buffer mid-copy (Shorokoo/Shorokoo#178).
            GC.KeepAlive(ortValue);
            return values;
        }
    }
}
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Inference.Abstractions;
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
                // Get the OrtValue from the original data
                var ortValue = ((IOnnxData)originalData).Value;
                // Create new TensorData with same data but different metadata
                return OnnxUtils.CreateTensorDataFromValue(shape, targetDType, ortValue, targetDType);
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
            return new OnnxTensorData<TTarget>(originalData.Shape, ortValue, targetDType);
        }

        private static object[] ExtractValues(TensorData data, DType dtype, int count)
        {
            // Use ProtoTypeNum for comparison to handle DTTypes with generic metadata
            // For generic types (IGenericType1-8), get the actual data type from the OrtValue
            if (dtype.IsGenericType)
            {
                // For generic types, we need to determine the actual data type from the OrtValue
                var ortValue = ((IOnnxData)data).Value;
                var actualDType = (DType)(int)ortValue.ElementType;
                
                // Extract directly from OrtValue based on actual element type
                return ExtractValuesFromOrtValue(ortValue, actualDType);
            }
            
            if (dtype.ProtoTypeNum == DType.Bool.ProtoTypeNum) return data.As<bit>().AccessMemory<bool>().ToArray().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.Int8.ProtoTypeNum) return data.As<int8>().AccessMemory<sbyte>().ToArray().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.Int16.ProtoTypeNum) return data.As<int16>().AccessMemory<short>().ToArray().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.Int32.ProtoTypeNum) return data.As<int32>().AccessMemory<int>().ToArray().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.Int64.ProtoTypeNum) return data.As<int64>().AccessMemory<long>().ToArray().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.UInt8.ProtoTypeNum) return data.As<uint8>().AccessMemory<byte>().ToArray().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.UInt16.ProtoTypeNum) return data.As<uint16>().AccessMemory<ushort>().ToArray().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.UInt32.ProtoTypeNum) return data.As<uint32>().AccessMemory<uint>().ToArray().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.UInt64.ProtoTypeNum) return data.As<uint64>().AccessMemory<ulong>().ToArray().Cast<object>().ToArray();
            // F16/BF16 widen exactly to float32, so extract as floats — keeps the
            // downstream Convert.To* calls working for every target type.
            else if (dtype.ProtoTypeNum == DType.Float16.ProtoTypeNum) return data.As<float16>().AccessMemory<Float16>().ToArray().Select(v => (object)(float)v).ToArray();
            else if (dtype.ProtoTypeNum == DType.BFloat16.ProtoTypeNum) return data.As<bfloat16>().AccessMemory<BFloat16>().ToArray().Select(v => (object)(float)v).ToArray();
            else if (dtype.ProtoTypeNum == DType.Float32.ProtoTypeNum) return data.As<float32>().AccessMemory<float>().ToArray().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.Float64.ProtoTypeNum) return data.As<float64>().AccessMemory<double>().ToArray().Cast<object>().ToArray();
            else throw new NotSupportedException($"Extraction from {dtype} is not supported");
        }
        
        private static object[] ExtractValuesFromOrtValue(IShorokooTensorValue ortValue, DType dtype)
        {
            // Extract values directly from OrtValue without going through TensorData.As<T>()
            if (dtype.ProtoTypeNum == DType.Bool.ProtoTypeNum) return ortValue.GetTensorDataAsSpan<bool>().ToArray().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.Int8.ProtoTypeNum) return ortValue.GetTensorDataAsSpan<sbyte>().ToArray().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.Int16.ProtoTypeNum) return ortValue.GetTensorDataAsSpan<short>().ToArray().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.Int32.ProtoTypeNum) return ortValue.GetTensorDataAsSpan<int>().ToArray().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.Int64.ProtoTypeNum) return ortValue.GetTensorDataAsSpan<long>().ToArray().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.UInt8.ProtoTypeNum) return ortValue.GetTensorDataAsSpan<byte>().ToArray().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.UInt16.ProtoTypeNum) return ortValue.GetTensorDataAsSpan<ushort>().ToArray().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.UInt32.ProtoTypeNum) return ortValue.GetTensorDataAsSpan<uint>().ToArray().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.UInt64.ProtoTypeNum) return ortValue.GetTensorDataAsSpan<ulong>().ToArray().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.Float16.ProtoTypeNum) return ortValue.GetTensorDataAsSpan<Float16>().ToArray().Select(v => (object)(float)v).ToArray();
            else if (dtype.ProtoTypeNum == DType.BFloat16.ProtoTypeNum) return ortValue.GetTensorDataAsSpan<BFloat16>().ToArray().Select(v => (object)(float)v).ToArray();
            else if (dtype.ProtoTypeNum == DType.Float32.ProtoTypeNum) return ortValue.GetTensorDataAsSpan<float>().ToArray().Cast<object>().ToArray();
            else if (dtype.ProtoTypeNum == DType.Float64.ProtoTypeNum) return ortValue.GetTensorDataAsSpan<double>().ToArray().Cast<object>().ToArray();
            else throw new NotSupportedException($"Extraction from {dtype} is not supported");
        }
    }
}

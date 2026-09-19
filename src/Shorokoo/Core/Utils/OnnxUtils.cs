using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Inference.Abstractions;
using ProtoBuf;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Xml.Linq;

namespace Shorokoo.Core.Utils
{
    internal static class OnnxUtils
    {
        public static long FromIndex(Index val)
        {
            if (val.IsFromEnd && val.Value == 0)
                return long.MaxValue;

            if (val.IsFromEnd)
                return -val.Value;

            return val.Value;
        }

        public static byte[] GetRawBytesZero(DType type)
        {
            if (type == DType.UInt16)
                return Enumerable.Repeat((byte)0, 2).ToArray();
            else if (type == DType.UInt32)
                return Enumerable.Repeat((byte)0, 4).ToArray();
            else if (type == DType.UInt64)
                return Enumerable.Repeat((byte)0, 8).ToArray();
            else if (type == DType.Int8)
                return Enumerable.Repeat((byte)0, 1).ToArray();
            else if (type == DType.UInt8)
                return Enumerable.Repeat((byte)0, 1).ToArray();
            else if (type == DType.Int16)
                return Enumerable.Repeat((byte)0, 2).ToArray();
            else if (type == DType.Int32)
                return Enumerable.Repeat((byte)0, 4).ToArray();
            else if (type == DType.Int64)
                return Enumerable.Repeat((byte)0, 8).ToArray();
            else if (type == DType.Float16)
                return Enumerable.Repeat((byte)0, 2).ToArray();
            else if (type == DType.BFloat16)
                return Enumerable.Repeat((byte)0, 2).ToArray();
            else if (type == DType.Float32)
                return Enumerable.Repeat((byte)0, 4).ToArray();
            else if (type == DType.Float64)
                return Enumerable.Repeat((byte)0, 8).ToArray();
            else if (type == DType.Bool)
                return Enumerable.Repeat((byte)0, 1).ToArray();
            else
                throw new UnsupportedDTypeException(ErrorCodes.OU005, type.ToString(), "GetDataOfType",
                    "DType not supported for default data array generation");
        }

        public static DType? GetDType(Type type)
        {
            if (type == typeof(bit)) return DType.Bool;
            else if (type == typeof(int4)) return DType.Int4;
            else if (type == typeof(int8)) return DType.Int8;
            else if (type == typeof(int16)) return DType.Int16;
            else if (type == typeof(int32)) return DType.Int32;
            else if (type == typeof(int64)) return DType.Int64;
            else if (type == typeof(uint4)) return DType.UInt4;
            else if (type == typeof(uint8)) return DType.UInt8;
            else if (type == typeof(uint16)) return DType.UInt16;
            else if (type == typeof(uint32)) return DType.UInt32;
            else if (type == typeof(uint64)) return DType.UInt64;
            else if (type == typeof(float16)) return DType.Float16;
            else if (type == typeof(bfloat16)) return DType.BFloat16;
            else if (type == typeof(float32)) return DType.Float32;
            else if (type == typeof(float64)) return DType.Float64;
            else if (type == typeof(@string) || type == typeof(string)) return DType.String;
            else if (type == typeof(IModuleVarType)) return DType.Module;
            else if (type == typeof(IModelVarType)) return DType.Model;
            else if (type == typeof(invalid)) return DType.Invalid;
            // Handle generic type placeholders
            else if (type == typeof(IGenericType1)) return DType.GenericType1;
            else if (type == typeof(IGenericType2)) return DType.GenericType2;
            else if (type == typeof(IGenericType3)) return DType.GenericType3;
            else if (type == typeof(IGenericType4)) return DType.GenericType4;
            else if (type == typeof(IGenericType5)) return DType.GenericType5;
            else if (type == typeof(IGenericType6)) return DType.GenericType6;
            else if (type == typeof(IGenericType7)) return DType.GenericType7;
            else if (type == typeof(IGenericType8)) return DType.GenericType8;

            return null;
        }

        public static DType GetDType<T>()
        {
            if (typeof(T) == typeof(bool) || typeof(T) == typeof(bit)) return DType.Bool;
            else if (typeof(T) == typeof(int4)) return DType.Int4;
            else if (typeof(T) == typeof(sbyte) || typeof(T) == typeof(int8)) return DType.Int8;
            else if (typeof(T) == typeof(short) || typeof(T) == typeof(int16)) return DType.Int16;
            else if (typeof(T) == typeof(int) || typeof(T) == typeof(int32)) return DType.Int32;
            else if (typeof(T) == typeof(long) || typeof(T) == typeof(int64)) return DType.Int64;
            else if (typeof(T) == typeof(uint4)) return DType.UInt4;
            else if (typeof(T) == typeof(byte) || typeof(T) == typeof(uint8)) return DType.UInt8;
            else if (typeof(T) == typeof(ushort) || typeof(T) == typeof(uint16)) return DType.UInt16;
            else if (typeof(T) == typeof(uint) || typeof(T) == typeof(uint32)) return DType.UInt32;
            else if (typeof(T) == typeof(ulong) || typeof(T) == typeof(uint64)) return DType.UInt64;
            else if (typeof(T) == typeof(Float16) || typeof(T) == typeof(float16)) return DType.Float16;
            else if (typeof(T) == typeof(BFloat16) || typeof(T) == typeof(bfloat16)) return DType.BFloat16;
            else if (typeof(T) == typeof(float) || typeof(T) == typeof(float32)) return DType.Float32;
            else if (typeof(T) == typeof(double) || typeof(T) == typeof(float64)) return DType.Float64;
            else if (typeof(T) == typeof(string) || typeof(T) == typeof(@string)) return DType.String;
            else if (typeof(T) == typeof(IModuleVarType)) return DType.Module;
            else if (typeof(T) == typeof(IModelVarType)) return DType.Model;
            else if (typeof(T) == typeof(invalid)) return DType.Invalid;
            // Handle generic type placeholders
            else if (typeof(T) == typeof(IGenericType1)) return DType.GenericType1;
            else if (typeof(T) == typeof(IGenericType2)) return DType.GenericType2;
            else if (typeof(T) == typeof(IGenericType3)) return DType.GenericType3;
            else if (typeof(T) == typeof(IGenericType4)) return DType.GenericType4;
            else if (typeof(T) == typeof(IGenericType5)) return DType.GenericType5;
            else if (typeof(T) == typeof(IGenericType6)) return DType.GenericType6;
            else if (typeof(T) == typeof(IGenericType7)) return DType.GenericType7;
            else if (typeof(T) == typeof(IGenericType8)) return DType.GenericType8;

            throw new UnsupportedDTypeException(ErrorCodes.OU004, typeof(T).Name, "GetDType",
                $"Type '{typeof(T).FullName}' cannot be mapped to a DType. Supported types include primitive types, IVarType implementations, and ONNX tensor types");
        }

        public static object CallGeneric(Type genericType, Type declaringType, string methodName, params object?[] parameters)
        {
            MethodInfo? method = null;

            // Case 1: Check for method with generic parameters
            method = declaringType
                .GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)
                .FirstOrDefault(m =>
                    m.IsGenericMethodDefinition &&
                    m.Name == methodName &&
                    m.GetGenericArguments().Length == 1 &&
                    m.GetParameters().Length == parameters.Length);

            // If no method was found in Case 1, check for method in a generic declaring type
            if (method == null && declaringType.IsGenericTypeDefinition)
            {
                // Construct the generic declaring type
                var constructedDeclaringType = declaringType.MakeGenericType(genericType);

                // Search for a non-generic method in the constructed generic type
                method = constructedDeclaringType
                    .GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)
                    .FirstOrDefault(m =>
                        m.Name == methodName &&
                        !m.IsGenericMethod &&
                        m.GetParameters().Length == parameters.Length);
            }

            if (method == null)
            {
                var paramTypes = string.Join(", ", parameters.Select((p, i) => $"[{i}]={p?.GetType().Name ?? "null"}"));
                throw new ReflectionException(ErrorCodes.OU001, methodName, declaringType.Name,
                    $"No suitable method found with {parameters.Length} parameters. Parameter types: {paramTypes}. " +
                    $"Generic type: {genericType.Name}");
            }

            // If the method is generic, make it a generic method with the provided type
            if (method.IsGenericMethodDefinition)
            {
                method = method.MakeGenericMethod(genericType);
            }

            // DoNotWrapExceptions, because everything reached this way is ordinary framework code
            // whose failures are part of its contract: a disposed compute context refusing to take
            // ownership, an argument that does not cover its shape. Wrapped in a
            // TargetInvocationException, none of them could be caught as the type they are
            // documented to be, and the call being generic is the caller's business rather than
            // theirs.
            var result = method.Invoke(
                null, BindingFlags.DoNotWrapExceptions, binder: null, parameters, culture: null);

            // Assert that the return value is not null
            Debug.Assert(result != null, "The method's return value should not be null.");

            return result!;
        }

        public static byte[] ConvertToByteArray<T>(T[] data)
            where T : struct
        {
            var span = data.AsSpan<T>();
            return MemoryMarshal.AsBytes<T>(span).ToArray();
        }

        public static byte[] ConvertToByteArray(List<string> data)
        {
            using (var ms = new MemoryStream())
            {
                foreach (var value in data)
                {
                    var bytes = System.Text.Encoding.UTF8.GetBytes(value);
                    ms.Write(bytes, 0, bytes.Length);
                }
                return ms.ToArray();
            }
        }

        public static byte[] ConvertToComplexByteArray(List<System.Numerics.Complex> data)
        {
            using (var ms = new MemoryStream())
            {
                using (var bw = new BinaryWriter(ms))
                {
                    foreach (var value in data)
                    {
                        bw.Write(value.Real);
                        bw.Write(value.Imaginary);
                    }
                }
                return ms.ToArray();
            }
        }

        public static byte[] ConvertToByteArray(List<bool> data)
        {
            var bytes = new byte[data.Count];
            for (int i = 0; i < data.Count; i++)
            {
                bytes[i] = (byte)(data[i] ? 1 : 0);
            }
            return bytes;
        }

        public static IShorokooTensorValue CreateTensorValue<T>(Shape shape, T[] data)
            where T : unmanaged
            => InferenceBackend.Default.CreateTensor<T>(data, (long[])shape);

        public static IShorokooTensorValue CreateTensorValue(Shape shape, byte[] data)
            => InferenceBackend.Default.CreateTensor<byte>(data, (long[])shape);

        public static IShorokooTensorValue CreateTensorValue(Shape shape, string[] data)
            => InferenceBackend.Default.CreateStringTensor(data, (long[])shape);

        /// <summary>
        /// A tensor over <paramref name="value"/> with no context: the framework's own host memory,
        /// which is what a value it built itself is in. The overload taking a
        /// <c>ComputeContext</c> is the one for a value a session produced.
        /// </summary>
        public static TensorData CreateTensorDataFromValue(IShorokooTensorValue value)
        {
            var shape = new Shape(value.Shape);
            var dtype = (DType)(int)value.ElementType;
            return CreateTensorDataFromValue(shape, dtype, value);
        }

        public static TensorData CreateTensorDataFromValue(Shape shape, DType dtype, IShorokooTensorValue value)
            => CreateTensorDataFromValue(shape, dtype, value, dtype);

        public static TensorData CreateTensorDataFromValue(Shape shape, DType dtype, IShorokooTensorValue value, DType typeGenericParam)
            => (TensorData)CallGeneric(typeGenericParam.ToIVarType(), typeof(OnnxUtils), nameof(OnnxUtils.internalCreateTensorDataFromValue), shape, value, dtype);

        public static TensorDataSequence CreateTensorDataSequenceFromValue(IShorokooTensorValue value)
        {
            var dtype = (DType)(int)value.GetSequenceElementType();
            return CreateTensorDataSequenceFromValue(dtype, value);
        }

        public static TensorDataSequence CreateTensorDataSequenceFromValue(DType type, IShorokooTensorValue value)
            => (TensorDataSequence)CallGeneric(type.ToIVarType(), typeof(OnnxUtils), nameof(OnnxUtils.internalCreateTensorDataSequenceFromValue), value);

        internal static TensorDataSequence CreateTensorDataSequence(DType dtype, List<TensorData> data)
        {
            // Building a sequence hands its elements to the runtime, which takes them over: the
            // sequence owns them from then on and releases them with itself. Copy first, so the
            // caller's tensors keep the storage they own and go on working. Handing the live
            // values over instead left every source tensor pointing at memory the sequence would
            // free under it — reading one after the sequence was disposed was a hard crash, from
            // four lines of public API (Shorokoo/Shorokoo#180).
            var inner = new List<IShorokooTensorValue>(data.Count);
            try
            {
                foreach (var d in data) inner.Add(CopyTensorValue(d.ToTensorValue()));
                var sequence = InferenceBackend.Default.CreateSequence(inner);
                return CreateTensorDataSequenceFromValue(dtype, sequence);
            }
            catch
            {
                // These copies belong to nobody yet; on failure nothing else will release them.
                foreach (var v in inner) v.Dispose();
                throw;
            }
        }

        /// <summary>
        /// An independent copy of <paramref name="value"/>, on storage of its own, for the places
        /// that would otherwise leave two owners pointing at one runtime value.
        /// </summary>
        internal static IShorokooTensorValue CopyTensorValue(IShorokooTensorValue value)
            => BackendTransfer.CopyTo(InferenceBackend.Default, value);

        /// <summary>
        /// Wraps a runtime value whose producer is not known — the framework's own paths all know
        /// theirs and call the overload below.
        ///
        /// <para>A value an execution provider kept in its own memory has no way to say <i>which</i>
        /// memory when it arrives here, so it lands in <see cref="MemoryKind.Unknown"/> and can be
        /// read but not moved. That is the honest answer rather than a useful one: pass the
        /// context that produced it and it names a real place.</para>
        /// </summary>
        public static IData CreateData(IShorokooTensorValue value)
            => CreateData(value, Shorokoo.Runtime.ComputeContext.Host);

        /// <summary>
        /// Wraps a runtime value as the data it holds, belonging to <paramref name="context"/> —
        /// the context whose session produced it, and so whose memory it is in.
        /// <see cref="Shorokoo.Runtime.ComputeContext.Host"/> means it came from outside any
        /// context, which can only be the framework's own host memory.
        ///
        /// <para>A sequence is bound the same way a tensor is. Its elements are copied out of it
        /// one at a time, on demand, by the runtime that holds it, so they are in that context's
        /// memory too and <see cref="OnnxTensorDataSequence{T}"/> hands each of them this same
        /// context.</para>
        /// </summary>
        public static IData CreateData(IShorokooTensorValue value, Shorokoo.Runtime.ComputeContext context)
        {
            ArgumentNullException.ThrowIfNull(context);
            if (value.ValueType == ShorokooOnnxValueType.Tensor)
                return ReferenceEquals(context, Shorokoo.Runtime.ComputeContext.Host)
                    ? CreateTensorDataFromValue(value)
                    : CreateTensorDataFromValue(
                        new Shape(value.Shape), (DType)(int)value.ElementType, value, context);
            else if (value.ValueType == ShorokooOnnxValueType.Sequence)
            {
                var sequence = CreateTensorDataSequenceFromValue(value);
                sequence.BindTo(context);
                return sequence;
            }

            throw new UnsupportedDTypeException(ErrorCodes.OU002, value.ValueType.ToString(), "CreateData",
                $"ONNX value type '{value.ValueType}' is not supported. Only TENSOR and SEQUENCE types are supported");
        }

        /// <summary>A named parameter over a runtime value with no known producer; see
        /// <see cref="CreateData(IShorokooTensorValue)"/> for what that costs.</summary>
        public static NamedModelParam CreateNamedModelParam(IShorokooTensorValue value, ModelParamType paramType, string name)
            => CreateNamedModelParam(value, paramType, name, Shorokoo.Runtime.ComputeContext.Host);

        /// <summary>A named parameter over a runtime value produced by <paramref name="context"/>.
        /// This is the shape every session output takes, so that an output can say where it is and
        /// be moved from there.</summary>
        public static NamedModelParam CreateNamedModelParam(
            IShorokooTensorValue value, ModelParamType paramType, string name,
            Shorokoo.Runtime.ComputeContext context)
        {
            var data = CreateData(value, context);
            if (data is TensorDataSequence sequenceData)
                return new TensorDataSequenceModelParam(name, paramType, sequenceData);
            if (data is TensorData tensorData)
                return new TensorDataModelParam(name, paramType, tensorData);
            else
                throw new ModelException(ErrorCodes.OU003, $"Tensor value type {data.GetType().Name}",
                    $"Unsupported data type for named model parameter creation. Expected TensorData or TensorDataSequence but got {data.GetType().Name}");
        }

        internal static TensorData internalCreateTensorDataFromValue<T>(Shape shape, IShorokooTensorValue value, DType? dtype) where T : IVarType
            => dtype is null ? new OnnxTensorData<T>(shape, value) : new OnnxTensorData<T>(shape, value, dtype);

        /// <summary>A backend-backed tensor bound to the context whose memory it is in.</summary>
        public static TensorData CreateTensorDataFromValue(
            Shape shape, DType dtype, IShorokooTensorValue value, Shorokoo.Runtime.ComputeContext context)
            => (TensorData)CallGeneric(dtype.ToIVarType(), typeof(OnnxUtils),
                nameof(internalCreateBoundTensorData), shape, value, context);

        internal static TensorData internalCreateBoundTensorData<T>(
            Shape shape, IShorokooTensorValue value, Shorokoo.Runtime.ComputeContext context)
            where T : IVarType
            => new OnnxTensorData<T>(shape, value, context, storage: null);

        /// <summary>A host tensor over <paramref name="bytes"/>, bound to <paramref name="context"/>
        /// (whose memory must be host memory).</summary>
        public static TensorData CreateHostTensorData(
            Shape shape, DType dtype, byte[] bytes, Shorokoo.Runtime.ComputeContext context)
            => (TensorData)CallGeneric(dtype.ToIVarType(), typeof(OnnxUtils),
                nameof(internalCreateHostTensorData), shape, bytes, context);

        internal static TensorData internalCreateHostTensorData<T>(
            Shape shape, byte[] bytes, Shorokoo.Runtime.ComputeContext context) where T : IVarType
            => HostTensorData<T>.Bound(shape, bytes, context);

        /// <summary>
        /// A host tensor over <paramref name="bytes"/> on <see cref="Shorokoo.Runtime.ComputeContext.Host"/>,
        /// keeping <paramref name="dtype"/> exactly as given. The overload above derives the dtype
        /// from the element type instead, which drops the generic parameter name a specialized
        /// dtype carries -- the one thing a graph literal's dtype is for.
        /// </summary>
        internal static TensorData CreateHostTensorData(Shape shape, DType dtype, byte[] bytes)
            => (TensorData)CallGeneric(dtype.ToIVarType(), typeof(OnnxUtils),
                nameof(internalCreateHostTensorDataAt), shape, bytes, dtype);

        internal static TensorData internalCreateHostTensorDataAt<T>(
            Shape shape, byte[] bytes, DType dtype) where T : IVarType
            => new HostTensorData<T>(shape, bytes, dtype);

        internal static TensorDataSequence internalCreateTensorDataSequenceFromValue<T>(IShorokooTensorValue value) where T : IVarType
            => new OnnxTensorDataSequence<T>(value);

        public static IShorokooTensorValue CreateTensorValueFromRawData(Shape shape, DType type, byte[] data)
            => InferenceBackend.Default.CreateTensorFromRawBytes(
                (ShorokooTensorElementType)(int)type.ProtoTypeNum, data, (long[])shape);

        public static string ToJson(this InternalComputationGraph fastGraph)
        {
            var obj = FastOnnxModelBuilder.BuildInternalOnnxModel(fastGraph);
            var graph = obj.Graph;
            foreach (var initializer in graph.Initializers)
                initializer.ResetRawData();

            string jsonString = JsonSerializer.Serialize(obj, new JsonSerializerOptions { WriteIndented = true });
            return jsonString;
        }
    }
}

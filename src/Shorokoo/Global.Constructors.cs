using Shorokoo.Core.Inference.Abstractions;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Modules;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection.Metadata;
using System.Runtime.CompilerServices;
using static Shorokoo.Core.Nodes.Ops;
using static Shorokoo.Core.Nodes.AutoDiff.Ops;
using static RandN.Distributions.Uniform;
using static Shorokoo.Core.InternalGlobals;
using Shorokoo.Core.Nodes.NodeDefinitions;
using System.Buffers.Text;
using System.Reflection;

namespace Shorokoo
{
    public static partial class Globals
    {

        #region Scalar constants constructors

        /// <summary>Creates a constant scalar from a boxed value, dispatching on its runtime type.</summary>
        public static Variable Scalar(object val)
        {
            switch (val)
            {
                case bool boolVal: 
                    return Scalar(boolVal);
                case sbyte sbyteVal: 
                    return Scalar(sbyteVal);
                case short shortVal: 
                    return Scalar(shortVal);
                case int intVal: 
                    return Scalar(intVal);
                case long longVal: 
                    return Scalar(longVal);
                case byte byteVal: 
                    return Scalar(byteVal);
                case ushort ushortVal: 
                    return Scalar(ushortVal);
                case uint uintVal: 
                    return Scalar(uintVal);
                case ulong ulongVal: 
                    return Scalar(ulongVal);
                case BFloat16 bfloat16Val: 
                    return Scalar(bfloat16Val);
                case Float16 float16Val: 
                    return Scalar(float16Val);
                case float floatVal: 
                    return Scalar(floatVal);
                case double doubleVal: 
                    return Scalar(doubleVal);
                default: 
                    throw new UnsupportedDTypeException(ErrorCodes.GC001, val?.GetType()?.Name ?? "null", "Scalar", 
                        $"Unsupported type for scalar creation. Supported types: bool, sbyte, short, int, long, byte, ushort, uint, ulong, BFloat16, Float16, float, double. Received: {val?.GetType()?.FullName ?? "null"}");
            }
        }

        /// <summary>Creates a constant <see cref="Scalar{T}"/> by converting a boxed value to T; supports IGenericType stand-ins.</summary>
        public static Scalar<T> Scalar<T>(object val) where T : IVarType
        {
            var typeofT = typeof(T);
            
            // Check if T is a generic type (IGenericType1, IGenericType2, etc.)
            if (typeofT.IsAssignableTo(typeof(IGenericType)))
                return CreateGenericScalar<T>(val);
            
            // Original non-generic logic
            if (typeofT == typeof(bit)) return (Scalar<T>)(object)Scalar(Convert.ToBoolean(val));
            else if (typeofT == typeof(int8)) return (Scalar<T>)(object)Scalar(Convert.ToSByte(val));
            else if (typeofT == typeof(int16)) return (Scalar<T>)(object)Scalar(Convert.ToInt16(val));
            else if (typeofT == typeof(int32)) return (Scalar<T>)(object)Scalar(Convert.ToInt32(val));
            else if (typeofT == typeof(int64)) return (Scalar<T>)(object)Scalar(Convert.ToInt64(val));
            else if (typeofT == typeof(uint8)) return (Scalar<T>)(object)Scalar(Convert.ToByte(val));
            else if (typeofT == typeof(uint16)) return (Scalar<T>)(object)Scalar(Convert.ToUInt16(val));
            else if (typeofT == typeof(uint32)) return (Scalar<T>)(object)Scalar(Convert.ToUInt32(val));
            else if (typeofT == typeof(uint64)) return (Scalar<T>)(object)Scalar(Convert.ToUInt64(val));
            // Use an already-half value directly; otherwise route any other numeric value
            // through float (every supported numeric primitive is IConvertible). A bare
            // (BFloat16)/(Float16) cast off `object` is an unbox that only succeeds when the
            // boxed value is already that exact half type, so it would throw for e.g. an int —
            // the path reached by `Scalar<bfloat16> x = 5;` via the implicit conversions.
            else if (typeofT == typeof(bfloat16)) return (Scalar<T>)(object)Scalar(val is BFloat16 bf ? bf : (BFloat16)Convert.ToSingle(val));
            else if (typeofT == typeof(float16)) return (Scalar<T>)(object)Scalar(val is Float16 f16 ? f16 : (Float16)Convert.ToSingle(val));
            else if (typeofT == typeof(float32)) return (Scalar<T>)(object)Scalar(Convert.ToSingle(val));
            else if (typeofT == typeof(float64)) return (Scalar<T>)(object)Scalar(Convert.ToDouble(val));
            else 
                throw new UnsupportedDTypeException(ErrorCodes.GC002, typeof(T).Name, "Scalar<T>", 
                    $"Type '{typeof(T).FullName}' is not supported for generic scalar creation. Supported IVarTypes: bit, int8, int16, int32, int64, uint8, uint16, uint32, uint64, bfloat16, float16, float32, float64");
        }

        private static Scalar<T> CreateGenericScalar<T>(object val) where T : IVarType
        {
            // For generic types (IGenericType1, etc.), create TensorData with:
            // - Type parameter T = IGenericTypeX (e.g., IGenericType1)
            // - DType = IGenericTypeX (e.g., DType.GenericType1), not the actual data type
            // - Data is stored in the actual type (e.g., uint16)
            // The data will be converted later by ProcessGenericStandInTypeInference
            
            var genericDType = OnnxUtils.GetDType<T>() ?? throw new InvalidOperationException($"Cannot get DType for {typeof(T).Name}");
            
            if (val is bool boolVal) return OnnxOp.Constant(new OnnxTensorData<T>(new Shape(), OnnxUtils.CreateTensorValue(Array.Empty<long>(), new[] { boolVal }), genericDType));
            else if (val is sbyte sbyteVal) return OnnxOp.Constant(new OnnxTensorData<T>(new Shape(), OnnxUtils.CreateTensorValue(Array.Empty<long>(), new[] { sbyteVal }), genericDType));
            else if (val is short shortVal) return OnnxOp.Constant(new OnnxTensorData<T>(new Shape(), OnnxUtils.CreateTensorValue(Array.Empty<long>(), new[] { shortVal }), genericDType));
            else if (val is int intVal) return OnnxOp.Constant(new OnnxTensorData<T>(new Shape(), OnnxUtils.CreateTensorValue(Array.Empty<long>(), new[] { intVal }), genericDType));
            else if (val is long longVal) return OnnxOp.Constant(new OnnxTensorData<T>(new Shape(), OnnxUtils.CreateTensorValue(Array.Empty<long>(), new[] { longVal }), genericDType));
            else if (val is byte byteVal) return OnnxOp.Constant(new OnnxTensorData<T>(new Shape(), OnnxUtils.CreateTensorValue(Array.Empty<long>(), new[] { byteVal }), genericDType));
            else if (val is ushort ushortVal) return OnnxOp.Constant(new OnnxTensorData<T>(new Shape(), OnnxUtils.CreateTensorValue(Array.Empty<long>(), new[] { ushortVal }), genericDType));
            else if (val is uint uintVal) return OnnxOp.Constant(new OnnxTensorData<T>(new Shape(), OnnxUtils.CreateTensorValue(Array.Empty<long>(), new[] { uintVal }), genericDType));
            else if (val is ulong ulongVal) return OnnxOp.Constant(new OnnxTensorData<T>(new Shape(), OnnxUtils.CreateTensorValue(Array.Empty<long>(), new[] { ulongVal }), genericDType));
            else if (val is BFloat16 bfloat16Val) return OnnxOp.Constant(new OnnxTensorData<T>(new Shape(), OnnxUtils.CreateTensorValue(Array.Empty<long>(), new[] { bfloat16Val }), genericDType));
            else if (val is Float16 float16Val) return OnnxOp.Constant(new OnnxTensorData<T>(new Shape(), OnnxUtils.CreateTensorValue(Array.Empty<long>(), new[] { float16Val }), genericDType));
            else if (val is float floatVal) return OnnxOp.Constant(new OnnxTensorData<T>(new Shape(), OnnxUtils.CreateTensorValue(Array.Empty<long>(), new[] { floatVal }), genericDType));
            else if (val is double doubleVal) return OnnxOp.Constant(new OnnxTensorData<T>(new Shape(), OnnxUtils.CreateTensorValue(Array.Empty<long>(), new[] { doubleVal }), genericDType));
            else throw new UnsupportedDTypeException(ErrorCodes.GC002, val?.GetType()?.Name ?? "null", "Scalar<T>",
                $"Type '{val?.GetType()?.FullName ?? "null"}' is not supported for generic scalar creation with IGenericType. Supported types: bool, sbyte, short, int, long, byte, ushort, uint, ulong, BFloat16, Float16, float, double");
        }

        /// <summary>Creates a constant scalar holding the given value.</summary>
        public static Scalar<bit> Scalar(bool val) => OnnxOp.Constant(TensorData([], val));
        /// <summary>Creates a constant scalar holding the given value.</summary>
        public static Scalar<int8> Scalar(sbyte val) => OnnxOp.Constant(TensorData([], val));
        /// <summary>Creates a constant scalar holding the given value.</summary>
        public static Scalar<int16> Scalar(short val) => OnnxOp.Constant(TensorData([], val));
        /// <summary>Creates a constant scalar holding the given value.</summary>
        public static Scalar<int32> Scalar(int val) => OnnxOp.Constant(TensorData([], val));
        /// <summary>Creates a constant scalar holding the given value.</summary>
        public static Scalar<int64> Scalar(long val) => OnnxOp.Constant(TensorData([], val));
        /// <summary>Creates a constant scalar holding the given value.</summary>
        public static Scalar<uint8> Scalar(byte val) => OnnxOp.Constant(TensorData([], val));
        /// <summary>Creates a constant scalar holding the given value.</summary>
        public static Scalar<uint16> Scalar(ushort val) => OnnxOp.Constant(TensorData([], val));
        /// <summary>Creates a constant scalar holding the given value.</summary>
        public static Scalar<uint32> Scalar(uint val) => OnnxOp.Constant(TensorData([], val));
        /// <summary>Creates a constant scalar holding the given value.</summary>
        public static Scalar<uint64> Scalar(ulong val) => OnnxOp.Constant(TensorData([], val));
        /// <summary>Creates a constant scalar holding the given value.</summary>
        public static Scalar<bfloat16> Scalar(BFloat16 val) => OnnxOp.Constant(TensorData([], val));
        /// <summary>Creates a constant scalar holding the given value.</summary>
        public static Scalar<float16> Scalar(Float16 val) => OnnxOp.Constant(TensorData([], val));
        /// <summary>Creates a constant scalar holding the given value.</summary>
        public static Scalar<float32> Scalar(float val) => OnnxOp.Constant(TensorData([], val));
        /// <summary>Creates a constant scalar holding the given value.</summary>
        public static Scalar<float64> Scalar(double val) => OnnxOp.Constant(TensorData([], val));

        /// <summary>Creates a constant scalar holding the default value of T (zero / false).</summary>
        public static Scalar<T> DefaultScalar<T>() where T : IVarType => OnnxOp.Constant(TensorDataWithDefaultVals(OnnxUtils.GetDType<T>(), []));

        #endregion

        #region Vector constants constructors

        /// <summary>Creates an empty (length-0) constant vector of element type T.</summary>
        public static Vector<T> EmptyVector<T>() where T : IVarType
        {
            if (typeof(T) == typeof(bit)) return (Vector<T>)(object)Vector(new bool[0]);
            if (typeof(T) == typeof(int8)) return (Vector<T>)(object)Vector(new sbyte[0]);
            if (typeof(T) == typeof(int16)) return (Vector<T>)(object)Vector(new short[0]);
            if (typeof(T) == typeof(int32)) return (Vector<T>)(object)Vector(new int[0]);
            if (typeof(T) == typeof(int64)) return (Vector<T>)(object)Vector(new long[0]);
            if (typeof(T) == typeof(uint8)) return (Vector<T>)(object)Vector(new byte[0]);
            if (typeof(T) == typeof(uint16)) return (Vector<T>)(object)Vector(new ushort[0]);
            if (typeof(T) == typeof(uint32)) return (Vector<T>)(object)Vector(new uint[0]);
            if (typeof(T) == typeof(uint64)) return (Vector<T>)(object)Vector(new ulong[0]);
            if (typeof(T) == typeof(float16)) return (Vector<T>)(object)Vector(new Float16[0]);
            if (typeof(T) == typeof(bfloat16)) return (Vector<T>)(object)Vector(new BFloat16[0]);
            if (typeof(T) == typeof(float32)) return (Vector<T>)(object)Vector(new float[0]);
            throw new UnsupportedDTypeException(ErrorCodes.GC006, typeof(T).Name, "Vector<T>", 
                $"Type '{typeof(T).FullName}' is not supported for generic empty vector creation");
        }

        /// <summary>Creates a constant vector from the given values.</summary>
        public static Vector<bit> Vector(params bool[] val) => OnnxOp.Constant(OnnxTensorData(val.Length, val));
        /// <summary>Creates a constant vector from the given values.</summary>
        public static Vector<int8> Vector(params sbyte[] val) => OnnxOp.Constant(OnnxTensorData(val.Length, val));
        /// <summary>Creates a constant vector from the given values.</summary>
        public static Vector<int16> Vector(params short[] val) => OnnxOp.Constant(OnnxTensorData(val.Length, val));
        /// <summary>Creates a constant vector from the given values.</summary>
        public static Vector<int32> Vector(params int[] val) => OnnxOp.Constant(OnnxTensorData(val.Length, val));
        /// <summary>Creates a constant vector from the given values.</summary>
        public static Vector<int64> Vector(params long[] val) => OnnxOp.Constant(OnnxTensorData(val.Length, val));
        /// <summary>Creates a constant vector from the given values.</summary>
        public static Vector<uint8> Vector(params byte[] val) => OnnxOp.Constant(OnnxTensorData(val.Length, val));
        /// <summary>Creates a constant vector from the given values.</summary>
        public static Vector<uint16> Vector(params ushort[] val) => OnnxOp.Constant(OnnxTensorData(val.Length, val));
        /// <summary>Creates a constant vector from the given values.</summary>
        public static Vector<uint32> Vector(params uint[] val) => OnnxOp.Constant(OnnxTensorData(val.Length, val));
        /// <summary>Creates a constant vector from the given values.</summary>
        public static Vector<uint64> Vector(params ulong[] val) => OnnxOp.Constant(OnnxTensorData(val.Length, val));
        /// <summary>Creates a constant vector from the given values.</summary>
        public static Vector<bfloat16> Vector(params BFloat16[] val) => OnnxOp.Constant(OnnxTensorData(val.Length, val));
        /// <summary>Creates a constant vector from the given values.</summary>
        public static Vector<float16> Vector(params Float16[] val) => OnnxOp.Constant(OnnxTensorData(val.Length, val));
        /// <summary>Creates a constant vector from the given values.</summary>
        public static Vector<float32> Vector(params float[] val) => OnnxOp.Constant(OnnxTensorData(val.Length, val));
        /// <summary>Creates a constant vector from the given values.</summary>
        public static Vector<float64> Vector(params double[] val) => OnnxOp.Constant(OnnxTensorData(val.Length, val));

        /// <summary>Creates a constant-filled vector of the given length.</summary>
        public static Vector<bit> VectorFill(long length, bool val) => OnnxOp.ConstantOfShape(Vector(length), OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector of the given length.</summary>
        public static Vector<int8> VectorFill(long length, sbyte val) => OnnxOp.ConstantOfShape(Vector(length), OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector of the given length.</summary>
        public static Vector<int16> VectorFill(long length, short val) => OnnxOp.ConstantOfShape(Vector(length), OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector of the given length.</summary>
        public static Vector<int32> VectorFill(long length, int val) => OnnxOp.ConstantOfShape(Vector(length), OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector of the given length.</summary>
        public static Vector<int64> VectorFill(long length, long val) => OnnxOp.ConstantOfShape(Vector(length), OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector of the given length.</summary>
        public static Vector<uint8> VectorFill(long length, byte val) => OnnxOp.ConstantOfShape(Vector(length), OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector of the given length.</summary>
        public static Vector<uint16> VectorFill(long length, ushort val) => OnnxOp.ConstantOfShape(Vector(length), OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector of the given length.</summary>
        public static Vector<uint32> VectorFill(long length, uint val) => OnnxOp.ConstantOfShape(Vector(length), OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector of the given length.</summary>
        public static Vector<uint64> VectorFill(long length, ulong val) => OnnxOp.ConstantOfShape(Vector(length), OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector of the given length.</summary>
        public static Vector<bfloat16> VectorFill(long length, BFloat16 val) => OnnxOp.ConstantOfShape(Vector(length), OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector of the given length.</summary>
        public static Vector<float16> VectorFill(long length, Float16 val) => OnnxOp.ConstantOfShape(Vector(length), OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector of the given length.</summary>
        public static Vector<float32> VectorFill(long length, float val) => OnnxOp.ConstantOfShape(Vector(length), OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector of the given length.</summary>
        public static Vector<float64> VectorFill(long length, double val) => OnnxOp.ConstantOfShape(Vector(length), OnnxTensorData(1, val), rank: 1);

        /// <summary>Creates a constant-filled vector whose length is given by a runtime scalar.</summary>
        public static Vector<bit> VectorFill(Scalar<int64> shape, bool val) => OnnxOp.ConstantOfShape(shape.Unsqueeze(), OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector whose length is given by a runtime scalar.</summary>
        public static Vector<int8> VectorFill(Scalar<int64> shape, sbyte val) => OnnxOp.ConstantOfShape(shape.Unsqueeze(), OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector whose length is given by a runtime scalar.</summary>
        public static Vector<int16> VectorFill(Scalar<int64> shape, short val) => OnnxOp.ConstantOfShape(shape.Unsqueeze(), OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector whose length is given by a runtime scalar.</summary>
        public static Vector<int32> VectorFill(Scalar<int64> shape, int val) => OnnxOp.ConstantOfShape(shape.Unsqueeze(), OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector whose length is given by a runtime scalar.</summary>
        public static Vector<int64> VectorFill(Scalar<int64> shape, long val) => OnnxOp.ConstantOfShape(shape.Unsqueeze(), OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector whose length is given by a runtime scalar.</summary>
        public static Vector<uint8> VectorFill(Scalar<int64> shape, byte val) => OnnxOp.ConstantOfShape(shape.Unsqueeze(), OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector whose length is given by a runtime scalar.</summary>
        public static Vector<uint16> VectorFill(Scalar<int64> shape, ushort val) => OnnxOp.ConstantOfShape(shape.Unsqueeze(), OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector whose length is given by a runtime scalar.</summary>
        public static Vector<uint32> VectorFill(Scalar<int64> shape, uint val) => OnnxOp.ConstantOfShape(shape.Unsqueeze(), OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector whose length is given by a runtime scalar.</summary>
        public static Vector<uint64> VectorFill(Scalar<int64> shape, ulong val) => OnnxOp.ConstantOfShape(shape.Unsqueeze(), OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector whose length is given by a runtime scalar.</summary>
        public static Vector<bfloat16> VectorFill(Scalar<int64> shape, BFloat16 val) => OnnxOp.ConstantOfShape(shape.Unsqueeze(), OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector whose length is given by a runtime scalar.</summary>
        public static Vector<float16> VectorFill(Scalar<int64> shape, Float16 val) => OnnxOp.ConstantOfShape(shape.Unsqueeze(), OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector whose length is given by a runtime scalar.</summary>
        public static Vector<float32> VectorFill(Scalar<int64> shape, float val) => OnnxOp.ConstantOfShape(shape.Unsqueeze(), OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector whose length is given by a runtime scalar.</summary>
        public static Vector<float64> VectorFill(Scalar<int64> shape, double val) => OnnxOp.ConstantOfShape(shape.Unsqueeze(), OnnxTensorData(1, val), rank: 1);

        /// <summary>Creates a constant-filled vector whose length is given by a runtime one-element shape vector.</summary>
        public static Vector<bit> VectorFill(Vector<int64> shape, bool val) => OnnxOp.ConstantOfShape(shape, OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector whose length is given by a runtime one-element shape vector.</summary>
        public static Vector<int8> VectorFill(Vector<int64> shape, sbyte val) => OnnxOp.ConstantOfShape(shape, OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector whose length is given by a runtime one-element shape vector.</summary>
        public static Vector<int16> VectorFill(Vector<int64> shape, short val) => OnnxOp.ConstantOfShape(shape, OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector whose length is given by a runtime one-element shape vector.</summary>
        public static Vector<int32> VectorFill(Vector<int64> shape, int val) => OnnxOp.ConstantOfShape(shape, OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector whose length is given by a runtime one-element shape vector.</summary>
        public static Vector<int64> VectorFill(Vector<int64> shape, long val) => OnnxOp.ConstantOfShape(shape, OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector whose length is given by a runtime one-element shape vector.</summary>
        public static Vector<uint8> VectorFill(Vector<int64> shape, byte val) => OnnxOp.ConstantOfShape(shape, OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector whose length is given by a runtime one-element shape vector.</summary>
        public static Vector<uint16> VectorFill(Vector<int64> shape, ushort val) => OnnxOp.ConstantOfShape(shape, OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector whose length is given by a runtime one-element shape vector.</summary>
        public static Vector<uint32> VectorFill(Vector<int64> shape, uint val) => OnnxOp.ConstantOfShape(shape, OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector whose length is given by a runtime one-element shape vector.</summary>
        public static Vector<uint64> VectorFill(Vector<int64> shape, ulong val) => OnnxOp.ConstantOfShape(shape, OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector whose length is given by a runtime one-element shape vector.</summary>
        public static Vector<bfloat16> VectorFill(Vector<int64> shape, BFloat16 val) => OnnxOp.ConstantOfShape(shape, OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector whose length is given by a runtime one-element shape vector.</summary>
        public static Vector<float16> VectorFill(Vector<int64> shape, Float16 val) => OnnxOp.ConstantOfShape(shape, OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector whose length is given by a runtime one-element shape vector.</summary>
        public static Vector<float32> VectorFill(Vector<int64> shape, float val) => OnnxOp.ConstantOfShape(shape, OnnxTensorData(1, val), rank: 1);
        /// <summary>Creates a constant-filled vector whose length is given by a runtime one-element shape vector.</summary>
        public static Vector<float64> VectorFill(Vector<int64> shape, double val) => OnnxOp.ConstantOfShape(shape, OnnxTensorData(1, val), rank: 1);

        /// <summary>Creates a single-element constant vector.</summary>
        public static Vector<bit> Vector(bool val) => Vector([val]);
        /// <summary>Creates a single-element constant vector.</summary>
        public static Vector<int8> Vector(sbyte val) => Vector((sbyte[])[val]);
        /// <summary>Creates a single-element constant vector.</summary>
        public static Vector<int16> Vector(short val) => Vector((short[])[val]);
        /// <summary>Creates a single-element constant vector.</summary>
        public static Vector<int32> Vector(int val) => Vector((int[])[val]);
        /// <summary>Creates a single-element constant vector.</summary>
        public static Vector<int64> Vector(long val) => Vector((long[])[val]);
        /// <summary>Creates a single-element constant vector.</summary>
        public static Vector<uint8> Vector(byte val) => Vector((byte[])[val]);
        /// <summary>Creates a single-element constant vector.</summary>
        public static Vector<uint16> Vector(ushort val) => Vector((ushort[])[val]);
        /// <summary>Creates a single-element constant vector.</summary>
        public static Vector<uint32> Vector(uint val) => Vector((uint[])[val]);
        /// <summary>Creates a single-element constant vector.</summary>
        public static Vector<uint64> Vector(ulong val) => Vector((ulong[])[val]);
        /// <summary>Creates a single-element constant vector.</summary>
        public static Vector<bfloat16> Vector(BFloat16 val) => Vector([val]);
        /// <summary>Creates a single-element constant vector.</summary>
        public static Vector<float16> Vector(Float16 val) => Vector([val]);
        /// <summary>Creates a single-element constant vector.</summary>
        public static Vector<float32> Vector(float val) => Vector((float[])[val]);
        /// <summary>Creates a single-element constant vector.</summary>
        public static Vector<float64> Vector(double val) => Vector([val]);

        /// <summary>Creates a vector of values from start (inclusive) to limit (exclusive), stepping by delta.</summary>
        public static Vector<T> VectorRange<T>(Scalar<T> start, Scalar<T> limit, Scalar<T> delta) where T : SimpleNumLike =>
            ((Tensor<T>)OnnxOp.Range(start, limit, delta)).Vec();

        /// <summary>Creates a vector of values from start (inclusive) to limit (exclusive), stepping by delta.</summary>
        public static Vector<int8> VectorRange(sbyte start, sbyte limit, sbyte delta) => OnnxOp.Range(Scalar(start), Scalar(limit), Scalar(delta)).int8().Vec();
        /// <summary>Creates a vector of values from start (inclusive) to limit (exclusive), stepping by delta.</summary>
        public static Vector<int16> VectorRange(short start, short limit, short delta) => OnnxOp.Range(Scalar(start), Scalar(limit), Scalar(delta)).int16().Vec();
        /// <summary>Creates a vector of values from start (inclusive) to limit (exclusive), stepping by delta.</summary>
        public static Vector<int32> VectorRange(int start, int limit, int delta) => OnnxOp.Range(Scalar(start), Scalar(limit), Scalar(delta)).int32().Vec();
        /// <summary>Creates a vector of values from start (inclusive) to limit (exclusive), stepping by delta.</summary>
        public static Vector<int64> VectorRange(long start, long limit, long delta) => OnnxOp.Range(Scalar(start), Scalar(limit), Scalar(delta)).int64().Vec();
        /// <summary>Creates a vector of values from start (inclusive) to limit (exclusive), stepping by delta.</summary>
        public static Vector<uint8> VectorRange(byte start, byte limit, byte delta) => OnnxOp.Range(Scalar(start).Cast<int8>(), Scalar(limit).Cast<int8>(), Scalar(delta).Cast<int8>()).Cast<uint8>().Vec();
        /// <summary>Creates a vector of values from start (inclusive) to limit (exclusive), stepping by delta.</summary>
        public static Vector<uint16> VectorRange(ushort start, ushort limit, ushort delta) => OnnxOp.Range(Scalar(start).Cast<int16>(), Scalar(limit).Cast<int16>(), Scalar(delta).Cast<int16>()).Cast<uint16>().Vec();
        /// <summary>Creates a vector of values from start (inclusive) to limit (exclusive), stepping by delta.</summary>
        public static Vector<uint32> VectorRange(uint start, uint limit, uint delta) => OnnxOp.Range(Scalar(start).Cast<int32>(), Scalar(limit).Cast<int32>(), Scalar(delta).Cast<int32>()).Cast<uint32>().Vec();
        /// <summary>Creates a vector of values from start (inclusive) to limit (exclusive), stepping by delta.</summary>
        public static Vector<uint64> VectorRange(ulong start, ulong limit, ulong delta) => OnnxOp.Range(Scalar(start).Cast<int64>(), Scalar(limit).Cast<int64>(), Scalar(delta).Cast<int64>()).uint64().Vec();

        /// <summary>Creates a constant vector of the given length filled with the default value of T.</summary>
        public static Vector<T> DefaultVector<T>(long length) where T : IVarType => OnnxOp.Constant(TensorDataWithDefaultVals(OnnxUtils.GetDType<T>(), [length]));

        #endregion

        #region Tensor constants constructors

        /// <summary>Creates a constant tensor with the given dims from the flat values.</summary>
        public static Tensor<bit> Tensor(long[] dims, params bool[] val) => OnnxOp.Constant(TensorData(dims, val));
        /// <summary>Creates a constant tensor with the given dims from the flat values.</summary>
        public static Tensor<int8> Tensor(long[] dims, params sbyte[] val) => OnnxOp.Constant(TensorData(dims, val));
        /// <summary>Creates a constant tensor with the given dims from the flat values.</summary>
        public static Tensor<int16> Tensor(long[] dims, params short[] val) => OnnxOp.Constant(TensorData(dims, val));
        /// <summary>Creates a constant tensor with the given dims from the flat values.</summary>
        public static Tensor<int32> Tensor(long[] dims, params int[] val) => OnnxOp.Constant(TensorData(dims, val));
        /// <summary>Creates a constant tensor with the given dims from the flat values.</summary>
        public static Tensor<int64> Tensor(long[] dims, params long[] val) => OnnxOp.Constant(TensorData(dims, val));
        /// <summary>Creates a constant tensor with the given dims from the flat values.</summary>
        public static Tensor<uint8> Tensor(long[] dims, params byte[] val) => OnnxOp.Constant(TensorData(dims, val));
        /// <summary>Creates a constant tensor with the given dims from the flat values.</summary>
        public static Tensor<uint16> Tensor(long[] dims, params ushort[] val) => OnnxOp.Constant(TensorData(dims, val));
        /// <summary>Creates a constant tensor with the given dims from the flat values.</summary>
        public static Tensor<uint32> Tensor(long[] dims, params uint[] val) => OnnxOp.Constant(TensorData(dims, val));
        /// <summary>Creates a constant tensor with the given dims from the flat values.</summary>
        public static Tensor<uint64> Tensor(long[] dims, params ulong[] val) => OnnxOp.Constant(TensorData(dims, val));
        /// <summary>Creates a constant tensor with the given dims from the flat values.</summary>
        public static Tensor<bfloat16> Tensor(long[] dims, params BFloat16[] val) => OnnxOp.Constant(TensorData(dims, val));
        /// <summary>Creates a constant tensor with the given dims from the flat values.</summary>
        public static Tensor<float16> Tensor(long[] dims, params Float16[] val) => OnnxOp.Constant(TensorData(dims, val));
        /// <summary>Creates a constant tensor with the given dims from the flat values.</summary>
        public static Tensor<float32> Tensor(long[] dims, params float[] val) => OnnxOp.Constant(TensorData(dims, val));
        /// <summary>Creates a constant tensor with the given dims from the flat values.</summary>
        public static Tensor<float64> Tensor(long[] dims, params double[] val) => OnnxOp.Constant(TensorData(dims, val));


        /// <summary>Creates a constant <see cref="Tensor{T}"/> from base64-encoded raw IR element data.</summary>
        public static Tensor<T> MakeTensor<T>(long[] dims, string base64IREncodedData) where T : IVarType
            => (Variable)Tensor(OnnxUtils.GetDType<T>(), dims, base64IREncodedData);

        /// <summary>Creates a constant tensor of the given dtype from base64-encoded raw IR element data.</summary>
        public static Variable Tensor(DType dtype, long[] dims, string base64IREncodedData) => OnnxOp.Constant(TensorData(dtype, dims, base64IREncodedData));

        /// <summary>Wraps existing typed TensorData in a constant tensor node.</summary>
        public static Tensor<T> MakeTensor<T>(TensorData<T> data) where T : IVarType
            => OnnxOp.Constant(data);

        /// <summary>Wraps existing TensorData in a constant tensor node.</summary>
        public static Variable Tensor(TensorData data)
            => OnnxOp.Constant(data);

        /// <summary>Creates a constant tensor with the given dims filled with the element type's default value.</summary>
        public static Tensor<T> DefaultTensor<T>(long[] dims) where T : IVarType => (Variable)DefaultTensor(OnnxUtils.GetDType<T>(), dims);
        /// <summary>Creates a constant tensor with the given dims filled with the dtype's default value.</summary>
        public static Variable DefaultTensor(DType dtype, long[] dims) => OnnxOp.Constant(TensorDataWithDefaultVals(dtype, dims));

        #endregion

        #region TensorSequence constructors

        /// <summary>Creates a tensor sequence from the given tensors; an empty sequence when none are supplied.</summary>
        public static TensorSequence<T> TensorSequence<T>(params Tensor<T>[] tensors)
            where T : IVarType
            => tensors.Length == 0 ?
                    OnnxOp.SequenceEmpty(OnnxUtils.GetDType<T>()) :
                    OnnxOp.SequenceConstruct([.. tensors.Select(x => (Variable)x)]);

        /// <summary>Creates a tensor sequence from the given tensors; an empty sequence when none are supplied.</summary>
        public static Variable TensorSequence(DType dtype, params Variable[] tensors)
            => tensors.Length == 0 ?
                    OnnxOp.SequenceEmpty(dtype) :
                    OnnxOp.SequenceConstruct(tensors);
        #endregion

        #region Optional Tensor constructors

        /// <summary>Creates an optional tensor wrapping the given tensor, or an empty optional when null.</summary>
        public static OptionalTensor<T> OptionalTensor<T>(Tensor<T>? tensor = null)
            where T : IVarType
            => OnnxOp.Optional(tensor, DataStructure.Tensor, OnnxUtils.GetDType<T>());

        /// <summary>Creates an optional tensor wrapping the given tensor, or an empty optional when null.</summary>
        public static Variable OptionalTensor(DType dtype, Variable? tensor = null)
            => OnnxOp.Optional(tensor, DataStructure.Tensor, dtype);

        #endregion

        #region TensorData constants constructors

        /// <summary>Creates TensorData with the given dims from the flat values.</summary>
        public static TensorData<bit> TensorData(long[] dims, params bool[] val) => new OnnxTensorData<bit>(dims, OnnxUtils.CreateTensorValue(dims, val));
        /// <summary>Creates TensorData with the given dims from the flat values.</summary>
        public static TensorData<int8> TensorData(long[] dims, params sbyte[] val) => new OnnxTensorData<int8>(dims, OnnxUtils.CreateTensorValue(dims, val));
        /// <summary>Creates TensorData with the given dims from the flat values.</summary>
        public static TensorData<int16> TensorData(long[] dims, params short[] val) => new OnnxTensorData<int16>(dims, OnnxUtils.CreateTensorValue(dims, val));
        /// <summary>Creates TensorData with the given dims from the flat values.</summary>
        public static TensorData<int32> TensorData(long[] dims, params int[] val) => new OnnxTensorData<int32>(dims, OnnxUtils.CreateTensorValue(dims, val));
        /// <summary>Creates TensorData with the given dims from the flat values.</summary>
        public static TensorData<int64> TensorData(long[] dims, params long[] val) => new OnnxTensorData<int64>(dims, OnnxUtils.CreateTensorValue(dims, val));
        /// <summary>Creates TensorData with the given dims from the flat values.</summary>
        public static TensorData<uint8> TensorData(long[] dims, params byte[] val) => new OnnxTensorData<uint8>(dims, OnnxUtils.CreateTensorValue(dims, val));
        /// <summary>Creates TensorData with the given dims from the flat values.</summary>
        public static TensorData<uint16> TensorData(long[] dims, params ushort[] val) => new OnnxTensorData<uint16>(dims, OnnxUtils.CreateTensorValue(dims, val));
        /// <summary>Creates TensorData with the given dims from the flat values.</summary>
        public static TensorData<uint32> TensorData(long[] dims, params uint[] val) => new OnnxTensorData<uint32>(dims, OnnxUtils.CreateTensorValue(dims, val));
        /// <summary>Creates TensorData with the given dims from the flat values.</summary>
        public static TensorData<uint64> TensorData(long[] dims, params ulong[] val) => new OnnxTensorData<uint64>(dims, OnnxUtils.CreateTensorValue(dims, val));
        /// <summary>Creates TensorData with the given dims from the flat values.</summary>
        public static TensorData<bfloat16> TensorData(long[] dims, params BFloat16[] val) => new OnnxTensorData<bfloat16>(dims, OnnxUtils.CreateTensorValue(dims, val));
        /// <summary>Creates TensorData with the given dims from the flat values.</summary>
        public static TensorData<float16> TensorData(long[] dims, params Float16[] val) => new OnnxTensorData<float16>(dims, OnnxUtils.CreateTensorValue(dims, val));
        /// <summary>Creates TensorData with the given dims from the flat values.</summary>
        public static TensorData<float32> TensorData(long[] dims, params float[] val) => new OnnxTensorData<float32>(dims, OnnxUtils.CreateTensorValue(dims, val));
        /// <summary>Creates TensorData with the given dims from the flat values.</summary>
        public static TensorData<float64> TensorData(long[] dims, params double[] val) => new OnnxTensorData<float64>(dims, OnnxUtils.CreateTensorValue(dims, val));
        /// <summary>Creates TensorData with the given dims from the flat values.</summary>
        public static TensorData<@string> TensorData(long[] dims, params string[] val) => new OnnxTensorData<@string>(dims, OnnxUtils.CreateTensorValue(dims, val));

        /// <summary>Creates rank-1 TensorData of the given length from the values.</summary>
        public static TensorData<bit> TensorData(long dims, params bool[] val) => new OnnxTensorData<bit>(dims, OnnxUtils.CreateTensorValue(dims, val));
        /// <summary>Creates rank-1 TensorData of the given length from the values.</summary>
        public static TensorData<int8> TensorData(long dims, params sbyte[] val) => new OnnxTensorData<int8>(dims, OnnxUtils.CreateTensorValue(dims, val));
        /// <summary>Creates rank-1 TensorData of the given length from the values.</summary>
        public static TensorData<int16> TensorData(long dims, params short[] val) => new OnnxTensorData<int16>(dims, OnnxUtils.CreateTensorValue(dims, val));
        /// <summary>Creates rank-1 TensorData of the given length from the values.</summary>
        public static TensorData<int32> TensorData(long dims, params int[] val) => new OnnxTensorData<int32>(dims, OnnxUtils.CreateTensorValue(dims, val));
        /// <summary>Creates rank-1 TensorData of the given length from the values.</summary>
        public static TensorData<int64> TensorData(long dims, params long[] val) => new OnnxTensorData<int64>(dims, OnnxUtils.CreateTensorValue(dims, val));
        /// <summary>Creates rank-1 TensorData of the given length from the values.</summary>
        public static TensorData<uint8> TensorData(long dims, params byte[] val) => new OnnxTensorData<uint8>(dims, OnnxUtils.CreateTensorValue(dims, val));
        /// <summary>Creates rank-1 TensorData of the given length from the values.</summary>
        public static TensorData<uint16> TensorData(long dims, params ushort[] val) => new OnnxTensorData<uint16>(dims, OnnxUtils.CreateTensorValue(dims, val));
        /// <summary>Creates rank-1 TensorData of the given length from the values.</summary>
        public static TensorData<uint32> TensorData(long dims, params uint[] val) => new OnnxTensorData<uint32>(dims, OnnxUtils.CreateTensorValue(dims, val));
        /// <summary>Creates rank-1 TensorData of the given length from the values.</summary>
        public static TensorData<uint64> TensorData(long dims, params ulong[] val) => new OnnxTensorData<uint64>(dims, OnnxUtils.CreateTensorValue(dims, val));
        /// <summary>Creates rank-1 TensorData of the given length from the values.</summary>
        public static TensorData<bfloat16> TensorData(long dims, params BFloat16[] val) => new OnnxTensorData<bfloat16>(dims, OnnxUtils.CreateTensorValue(dims, val));
        /// <summary>Creates rank-1 TensorData of the given length from the values.</summary>
        public static TensorData<float16> TensorData(long dims, params Float16[] val) => new OnnxTensorData<float16>(dims, OnnxUtils.CreateTensorValue(dims, val));
        /// <summary>Creates rank-1 TensorData of the given length from the values.</summary>
        public static TensorData<float32> TensorData(long dims, params float[] val) => new OnnxTensorData<float32>(dims, OnnxUtils.CreateTensorValue(dims, val));
        /// <summary>Creates rank-1 TensorData of the given length from the values.</summary>
        public static TensorData<float64> TensorData(long dims, params double[] val) => new OnnxTensorData<float64>(dims, OnnxUtils.CreateTensorValue(dims, val));
        /// <summary>Creates rank-1 TensorData of the given length from the values.</summary>
        public static TensorData<@string> TensorData(long dims, params string[] val) => new OnnxTensorData<@string>(dims, OnnxUtils.CreateTensorValue(dims, val));

        /// <summary>Creates a tensor of the given runtime shape filled with the value (ONNX ConstantOfShape).</summary>
        public static Tensor<bit> TensorFill(Vector<int64> shape, bool val) => OnnxOp.ConstantOfShape(shape, OnnxTensorData(1, val));
        /// <summary>Creates a tensor of the given runtime shape filled with the value (ONNX ConstantOfShape).</summary>
        public static Tensor<int8> TensorFill(Vector<int64> shape, sbyte val) => OnnxOp.ConstantOfShape(shape, OnnxTensorData(1, val));
        /// <summary>Creates a tensor of the given runtime shape filled with the value (ONNX ConstantOfShape).</summary>
        public static Tensor<int16> TensorFill(Vector<int64> shape, short val) => OnnxOp.ConstantOfShape(shape, OnnxTensorData(1, val));
        /// <summary>Creates a tensor of the given runtime shape filled with the value (ONNX ConstantOfShape).</summary>
        public static Tensor<int32> TensorFill(Vector<int64> shape, int val) => OnnxOp.ConstantOfShape(shape, OnnxTensorData(1, val));
        /// <summary>Creates a tensor of the given runtime shape filled with the value (ONNX ConstantOfShape).</summary>
        public static Tensor<int64> TensorFill(Vector<int64> shape, long val) => OnnxOp.ConstantOfShape(shape, OnnxTensorData(1, val));
        /// <summary>Creates a tensor of the given runtime shape filled with the value (ONNX ConstantOfShape).</summary>
        public static Tensor<uint8> TensorFill(Vector<int64> shape, byte val) => OnnxOp.ConstantOfShape(shape, OnnxTensorData(1, val));
        /// <summary>Creates a tensor of the given runtime shape filled with the value (ONNX ConstantOfShape).</summary>
        public static Tensor<uint16> TensorFill(Vector<int64> shape, ushort val) => OnnxOp.ConstantOfShape(shape, OnnxTensorData(1, val));
        /// <summary>Creates a tensor of the given runtime shape filled with the value (ONNX ConstantOfShape).</summary>
        public static Tensor<uint32> TensorFill(Vector<int64> shape, uint val) => OnnxOp.ConstantOfShape(shape, OnnxTensorData(1, val));
        /// <summary>Creates a tensor of the given runtime shape filled with the value (ONNX ConstantOfShape).</summary>
        public static Tensor<uint64> TensorFill(Vector<int64> shape, ulong val) => OnnxOp.ConstantOfShape(shape, OnnxTensorData(1, val));
        /// <summary>Creates a tensor of the given runtime shape filled with the value (ONNX ConstantOfShape).</summary>
        public static Tensor<bfloat16> TensorFill(Vector<int64> shape, BFloat16 val) => OnnxOp.ConstantOfShape(shape, OnnxTensorData(1, val));
        /// <summary>Creates a tensor of the given runtime shape filled with the value (ONNX ConstantOfShape).</summary>
        public static Tensor<float16> TensorFill(Vector<int64> shape, Float16 val) => OnnxOp.ConstantOfShape(shape, OnnxTensorData(1, val));
        /// <summary>Creates a tensor of the given runtime shape filled with the value (ONNX ConstantOfShape).</summary>
        public static Tensor<float32> TensorFill(Vector<int64> shape, float val) => OnnxOp.ConstantOfShape(shape, OnnxTensorData(1, val));
        /// <summary>Creates a tensor of the given runtime shape filled with the value (ONNX ConstantOfShape).</summary>
        public static Tensor<float64> TensorFill(Vector<int64> shape, double val) => OnnxOp.ConstantOfShape(shape, OnnxTensorData(1, val));
        /// <summary>Creates a tensor of the given runtime shape filled with the value (ONNX ConstantOfShape).</summary>
        public static Tensor<T> TensorFill<T>(Vector<int64> shape, TensorData<T> val) where T : IVarType => OnnxOp.ConstantOfShape(shape, val);

        /// <summary>
        /// Creates a tensor filled with a constant value using generic type T.
        /// This method mirrors the behavior of Scalar&lt;T&gt;(object) for CONSTANT_OF_SHAPE operations.
        /// Supports both concrete types and generic types (IGenericType1, etc.).
        /// </summary>
        /// <typeparam name="T">The element type of the tensor</typeparam>
        /// <param name="shape">The shape of the output tensor</param>
        /// <param name="val">The fill value (will be converted to type T)</param>
        /// <returns>A tensor filled with the specified value</returns>
        public static Tensor<T> TensorFill<T>(Vector<int64> shape, object val) where T : IVarType
        {
            var typeofT = typeof(T);
            
            // Check if T is a generic type (IGenericType1, IGenericType2, etc.)
            if (typeofT.IsAssignableTo(typeof(IGenericType)))
                return CreateGenericTensorFill<T>(shape, val);
            
            // Original non-generic logic
            if (typeofT == typeof(bit)) return (Tensor<T>)(object)TensorFill(shape, Convert.ToBoolean(val));
            else if (typeofT == typeof(int8)) return (Tensor<T>)(object)TensorFill(shape, Convert.ToSByte(val));
            else if (typeofT == typeof(int16)) return (Tensor<T>)(object)TensorFill(shape, Convert.ToInt16(val));
            else if (typeofT == typeof(int32)) return (Tensor<T>)(object)TensorFill(shape, Convert.ToInt32(val));
            else if (typeofT == typeof(int64)) return (Tensor<T>)(object)TensorFill(shape, Convert.ToInt64(val));
            else if (typeofT == typeof(uint8)) return (Tensor<T>)(object)TensorFill(shape, Convert.ToByte(val));
            else if (typeofT == typeof(uint16)) return (Tensor<T>)(object)TensorFill(shape, Convert.ToUInt16(val));
            else if (typeofT == typeof(uint32)) return (Tensor<T>)(object)TensorFill(shape, Convert.ToUInt32(val));
            else if (typeofT == typeof(uint64)) return (Tensor<T>)(object)TensorFill(shape, Convert.ToUInt64(val));
            else if (typeofT == typeof(bfloat16)) return (Tensor<T>)(object)TensorFill(shape, (BFloat16)val);
            else if (typeofT == typeof(float16)) return (Tensor<T>)(object)TensorFill(shape, (Float16)val);
            else if (typeofT == typeof(float32)) return (Tensor<T>)(object)TensorFill(shape, Convert.ToSingle(val));
            else if (typeofT == typeof(float64)) return (Tensor<T>)(object)TensorFill(shape, Convert.ToDouble(val));
            else 
                throw new UnsupportedDTypeException(ErrorCodes.GC003, typeof(T).Name, "TensorFill<T>", 
                    $"Type '{typeof(T).FullName}' is not supported for generic tensor fill creation. Supported IVarTypes: bit, int8, int16, int32, int64, uint8, uint16, uint32, uint64, bfloat16, float16, float32, float64");
        }

        private static Tensor<T> CreateGenericTensorFill<T>(Vector<int64> shape, object val) where T : IVarType
        {
            // For generic types (IGenericType1, etc.), create TensorData with:
            // - Type parameter T = IGenericTypeX (e.g., IGenericType1)
            // - DType = IGenericTypeX (e.g., DType.GenericType1), not the actual data type
            // - Data is stored in the actual type (e.g., uint16)
            // The data will be converted later by ProcessGenericStandInTypeInference
            
            var genericDType = OnnxUtils.GetDType<T>() ?? throw new InvalidOperationException($"Cannot get DType for {typeof(T).Name}");
            
            // Create a scalar TensorData (1-dimensional with 1 element) for ConstantOfShape fill value
            TensorData<T> fillValueData;
            if (val is bool boolVal) fillValueData = new OnnxTensorData<T>(new Shape(1), OnnxUtils.CreateTensorValue(new long[] { 1 }, new[] { boolVal }), genericDType);
            else if (val is sbyte sbyteVal) fillValueData = new OnnxTensorData<T>(new Shape(1), OnnxUtils.CreateTensorValue(new long[] { 1 }, new[] { sbyteVal }), genericDType);
            else if (val is short shortVal) fillValueData = new OnnxTensorData<T>(new Shape(1), OnnxUtils.CreateTensorValue(new long[] { 1 }, new[] { shortVal }), genericDType);
            else if (val is int intVal) fillValueData = new OnnxTensorData<T>(new Shape(1), OnnxUtils.CreateTensorValue(new long[] { 1 }, new[] { intVal }), genericDType);
            else if (val is long longVal) fillValueData = new OnnxTensorData<T>(new Shape(1), OnnxUtils.CreateTensorValue(new long[] { 1 }, new[] { longVal }), genericDType);
            else if (val is byte byteVal) fillValueData = new OnnxTensorData<T>(new Shape(1), OnnxUtils.CreateTensorValue(new long[] { 1 }, new[] { byteVal }), genericDType);
            else if (val is ushort ushortVal) fillValueData = new OnnxTensorData<T>(new Shape(1), OnnxUtils.CreateTensorValue(new long[] { 1 }, new[] { ushortVal }), genericDType);
            else if (val is uint uintVal) fillValueData = new OnnxTensorData<T>(new Shape(1), OnnxUtils.CreateTensorValue(new long[] { 1 }, new[] { uintVal }), genericDType);
            else if (val is ulong ulongVal) fillValueData = new OnnxTensorData<T>(new Shape(1), OnnxUtils.CreateTensorValue(new long[] { 1 }, new[] { ulongVal }), genericDType);
            else if (val is BFloat16 bfloat16Val) fillValueData = new OnnxTensorData<T>(new Shape(1), OnnxUtils.CreateTensorValue(new long[] { 1 }, new[] { bfloat16Val }), genericDType);
            else if (val is Float16 float16Val) fillValueData = new OnnxTensorData<T>(new Shape(1), OnnxUtils.CreateTensorValue(new long[] { 1 }, new[] { float16Val }), genericDType);
            else if (val is float floatVal) fillValueData = new OnnxTensorData<T>(new Shape(1), OnnxUtils.CreateTensorValue(new long[] { 1 }, new[] { floatVal }), genericDType);
            else if (val is double doubleVal) fillValueData = new OnnxTensorData<T>(new Shape(1), OnnxUtils.CreateTensorValue(new long[] { 1 }, new[] { doubleVal }), genericDType);
            else throw new UnsupportedDTypeException(ErrorCodes.GC003, val?.GetType()?.Name ?? "null", "TensorFill<T>",
                $"Type '{val?.GetType()?.FullName ?? "null"}' is not supported for generic tensor fill creation with IGenericType. Supported types: bool, sbyte, short, int, long, byte, ushort, uint, ulong, BFloat16, Float16, float, double");
            
            return OnnxOp.ConstantOfShape(shape, fillValueData);
        }


        /// <summary>
        /// Creates a tensor filled with random values from a uniform distribution in [low, high).
        /// Takes shape as a dynamic tensor input. Lowered to ONNX RandomUniformLike before execution.
        /// </summary>
        public static Tensor<float32> RandomUniform(Vector<int64> shape, float low = 0.0f, float high = 1.0f, float? seed = null)
            => InternalOp.RandomUniform(shape, high: high, low: low, seed: seed);

        /// <summary>
        /// Creates a tensor filled with random values from a normal distribution with given mean and scale (std dev).
        /// Takes shape as a dynamic tensor input. Lowered to ONNX RandomNormalLike before execution.
        /// </summary>
        public static Tensor<float32> RandomNormal(Vector<int64> shape, float mean = 0.0f, float scale = 1.0f, float? seed = null)
            => InternalOp.RandomNormal(shape, mean: mean, scale: scale, seed: seed);

        /// <summary>Creates TensorData of the given dtype from boxed values.</summary>
        public static TensorData TensorData(DType type, long[] dims, params object[] vals)
        {
            if (type.ToIVarType() == typeof(bit)) return TensorData(dims, vals.Cast<bool>().ToArray());
            if (type.ToIVarType() == typeof(int8)) return TensorData(dims, vals.Cast<sbyte>().ToArray());
            if (type.ToIVarType() == typeof(int16)) return TensorData(dims, vals.Cast<short>().ToArray());
            if (type.ToIVarType() == typeof(int32)) return TensorData(dims, vals.Cast<int>().ToArray());
            if (type.ToIVarType() == typeof(int64)) return TensorData(dims, vals.Cast<long>().ToArray());
            if (type.ToIVarType() == typeof(uint8)) return TensorData(dims, vals.Cast<byte>().ToArray());
            if (type.ToIVarType() == typeof(uint16)) return TensorData(dims, vals.Cast<ushort>().ToArray());
            if (type.ToIVarType() == typeof(uint32)) return TensorData(dims, vals.Cast<uint>().ToArray());
            if (type.ToIVarType() == typeof(uint64)) return TensorData(dims, vals.Cast<ulong>().ToArray());
            if (type.ToIVarType() == typeof(float16)) return TensorData(dims, vals.Cast<Float16>().ToArray());
            if (type.ToIVarType() == typeof(bfloat16)) return TensorData(dims, vals.Cast<BFloat16>().ToArray());
            if (type.ToIVarType() == typeof(float32)) return TensorData(dims, vals.Cast<float>().ToArray());
            if (type.ToIVarType() == typeof(float64)) return TensorData(dims, vals.Cast<double>().ToArray());
            if (type.ToIVarType() == typeof(@string)) return TensorData(dims, vals.Cast<string>().ToArray());
            throw new UnsupportedDTypeException(ErrorCodes.GC006, type.ToString(), "TensorData",
                $"DType '{type}' is not supported for tensor data creation with values");
        }

        /// <summary>Creates TensorData with the given dims filled with the dtype's default value (zero / false).</summary>
        public static TensorData TensorDataWithDefaultVals(DType type, long[] dims)
        {
            var size = 1;
            foreach (var dim in dims) size *= (int)dim;

            if (type.ToIVarType() == typeof(bit)) return TensorData(dims, Enumerable.Repeat(false, size).ToArray());
            if (type.ToIVarType() == typeof(int8)) return TensorData(dims, Enumerable.Repeat((sbyte)0, size).ToArray());
            if (type.ToIVarType() == typeof(int16)) return TensorData(dims, Enumerable.Repeat((short)0, size).ToArray());
            if (type.ToIVarType() == typeof(int32)) return TensorData(dims, Enumerable.Repeat((int)0, size).ToArray());
            if (type.ToIVarType() == typeof(int64)) return TensorData(dims, Enumerable.Repeat((long)0, size).ToArray());
            if (type.ToIVarType() == typeof(uint8)) return TensorData(dims, Enumerable.Repeat((byte)0, size).ToArray());
            if (type.ToIVarType() == typeof(uint16)) return TensorData(dims, Enumerable.Repeat((ushort)0, size).ToArray());
            if (type.ToIVarType() == typeof(uint32)) return TensorData(dims, Enumerable.Repeat((uint)0, size).ToArray());
            if (type.ToIVarType() == typeof(uint64)) return TensorData(dims, Enumerable.Repeat((ulong)0, size).ToArray());
            if (type.ToIVarType() == typeof(float16)) return TensorData(dims, Enumerable.Repeat((Float16)0f, size).ToArray());
            if (type.ToIVarType() == typeof(bfloat16)) return TensorData(dims, Enumerable.Repeat((BFloat16)0f, size).ToArray());
            if (type.ToIVarType() == typeof(float32)) return TensorData(dims, Enumerable.Repeat((float)0f, size).ToArray());
            if (type.ToIVarType() == typeof(float64)) return TensorData(dims, Enumerable.Repeat((double)0f, size).ToArray());
            throw new UnsupportedDTypeException(ErrorCodes.GC006, type.ToString(), "TensorDataWithDefaultVals", 
                $"DType '{type}' is not supported for tensor data creation with default values");
        }

        /// <summary>Creates TensorData filled with small non-zero values: 1 for integer types, 0.1 for float types, false for bit.</summary>
        public static TensorData TensorDataWithSmallVals(DType type, long[] dims)
        {
            var size = 1;
            foreach (var dim in dims) size *= (int)dim;

            if (type.ToIVarType() == typeof(bit)) return TensorData(dims, Enumerable.Repeat(false, size).ToArray());
            if (type.ToIVarType() == typeof(int8)) return TensorData(dims, Enumerable.Repeat((sbyte)1, size).ToArray());
            if (type.ToIVarType() == typeof(int16)) return TensorData(dims, Enumerable.Repeat((short)1, size).ToArray());
            if (type.ToIVarType() == typeof(int32)) return TensorData(dims, Enumerable.Repeat((int)1, size).ToArray());
            if (type.ToIVarType() == typeof(int64)) return TensorData(dims, Enumerable.Repeat((long)1, size).ToArray());
            if (type.ToIVarType() == typeof(uint8)) return TensorData(dims, Enumerable.Repeat((byte)1, size).ToArray());
            if (type.ToIVarType() == typeof(uint16)) return TensorData(dims, Enumerable.Repeat((ushort)1, size).ToArray());
            if (type.ToIVarType() == typeof(uint32)) return TensorData(dims, Enumerable.Repeat((uint)1, size).ToArray());
            if (type.ToIVarType() == typeof(uint64)) return TensorData(dims, Enumerable.Repeat((ulong)1, size).ToArray());
            if (type.ToIVarType() == typeof(float16)) return TensorData(dims, Enumerable.Repeat((Float16)0.1f, size).ToArray());
            if (type.ToIVarType() == typeof(bfloat16)) return TensorData(dims, Enumerable.Repeat((BFloat16)0.1f, size).ToArray());
            if (type.ToIVarType() == typeof(float32)) return TensorData(dims, Enumerable.Repeat((float)0.1f, size).ToArray());
            if (type.ToIVarType() == typeof(float64)) return TensorData(dims, Enumerable.Repeat((double)0.1f, size).ToArray());
            throw new UnsupportedDTypeException(ErrorCodes.GC006, type.ToString(), "TensorDataWithSmallVals", 
                $"DType '{type}' is not supported for tensor data creation with small values");
        }

        /// <summary>
        /// Creates TensorData for ConstantOfShape value attribute - must be 1-dimensional with 1 element
        /// </summary>
        public static TensorData TensorDataForConstantOfShapeFill(DType type)
        {
            if (type.ToIVarType() == typeof(bit)) return OnnxTensorData(1, false);
            if (type.ToIVarType() == typeof(int8)) return OnnxTensorData(1, (sbyte)1);
            if (type.ToIVarType() == typeof(int16)) return OnnxTensorData(1, (short)1);
            if (type.ToIVarType() == typeof(int32)) return OnnxTensorData(1, (int)1);
            if (type.ToIVarType() == typeof(int64)) return OnnxTensorData(1, (long)1);
            if (type.ToIVarType() == typeof(uint8)) return OnnxTensorData(1, (byte)1);
            if (type.ToIVarType() == typeof(uint16)) return OnnxTensorData(1, (ushort)1);
            if (type.ToIVarType() == typeof(uint32)) return OnnxTensorData(1, (uint)1);
            if (type.ToIVarType() == typeof(uint64)) return OnnxTensorData(1, (ulong)1);
            if (type.ToIVarType() == typeof(float16)) return OnnxTensorData(1, (Float16)0.1f);
            if (type.ToIVarType() == typeof(bfloat16)) return OnnxTensorData(1, (BFloat16)0.1f);
            if (type.ToIVarType() == typeof(float32)) return OnnxTensorData(1, (float)0.1f);
            if (type.ToIVarType() == typeof(float64)) return OnnxTensorData(1, (double)0.1f);
            throw new UnsupportedDTypeException(ErrorCodes.GC006, type.ToString(), "TensorDataForConstantOfShapeFill", 
                $"DType '{type}' is not supported for ConstantOfShape fill value creation");
        }

        /// <summary>Creates a placeholder variable: a default-valued tensor constant, an empty optional, or an empty sequence, depending on structure.</summary>
        public static Variable CreateVariable(DType type, int rank, DataStructure structure)
        {
            switch (structure)
            {
                case DataStructure.Tensor:
                    return OnnxOp.Constant(TensorDataWithDefaultVals(type, Enumerable.Repeat<long>(1, rank).ToArray()));
                case DataStructure.Optional:
                    return OnnxOp.Optional(null, DataStructure.Tensor, type);
                case DataStructure.Sequence:
                    return OnnxOp.SequenceEmpty(type);
                case DataStructure.TensorStruct:
                    throw new InvalidOperationException($"CreateVariable does not support TensorStruct - use InternalOp.TensorStructCreate (or Globals.TensorStruct<T>) instead with appropriate field values");
            }

            throw new UnsupportedDTypeException(ErrorCodes.GC003, type.ToString(), "CreateVariable", 
                $"DataStructure type '{structure}' is not supported for CreateVariable operation. Supported types: Tensor, Optional, Sequence");
        }

        /// <summary>Encodes an array into its raw IR byte representation (in-memory layout; half types via their 16-bit patterns).</summary>
        public static byte[] Enc<T>(T[] data) where T : unmanaged
        {
            if (data == null)
                throw new InvalidTensorOperationException(ErrorCodes.GC004, "Enc", "data array", "Input data array cannot be null");

            // BFloat16/Float16 expose their underlying ushort via .Bits (the
            // legacy code reached for a non-existent ".value" property via
            // dynamic dispatch). Cast through object so the same code path
            // works for both struct types.
            if (typeof(T) == typeof(BFloat16))
                return Enc<ushort>(data.Select(x => ((BFloat16)(object)x!).Bits).ToArray());

            if (typeof(T) == typeof(Float16))
                return Enc<ushort>(data.Select(x => ((Float16)(object)x!).Bits).ToArray());

            // General case. Use:
            //  - Unsafe.SizeOf<T>() (in-memory size, e.g. 1 for bool) rather
            //    than Marshal.SizeOf<T>() (P/Invoke marshalled size, 4 for
            //    bool); the latter mismatched the in-memory layout that
            //    Buffer.BlockCopy actually moves and broke the assert for bool.
            //  - GetDType<T>() (generic overload), which maps both Shorokoo
            //    IVarTypes (bit/int32/...) and CLR primitives (bool/int/...);
            //    the (Type) overload only handles the former, so it returned
            //    null for CLR T's and crashed .NotNull(). Dec<T> below already
            //    uses the generic overload.
            var elementSize = System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
            Debug.Assert(OnnxUtils.GetDType<T>().EncodingBitCount == elementSize * 8);
            var byteArrayGeneral = new byte[data.Length * elementSize];
            Buffer.BlockCopy(data, 0, byteArrayGeneral, 0, byteArrayGeneral.Length);
            return byteArrayGeneral;
        }

        /// <summary>Decodes raw IR bytes back into a typed array; the inverse of <see cref="Enc{T}"/>.</summary>
        public static T[] Dec<T>(byte[] data) where T : unmanaged
        {
            var bitCount = OnnxUtils.GetDType<T>().NotNull().EncodingBitCount;
            if (data.Length % (bitCount / 8) != 0)
                throw new InvalidTensorOperationException(ErrorCodes.GC005, "Dec", $"data array of {data.Length} bytes", 
                    $"Data must be a multiple of {bitCount} bits ({bitCount/8} bytes), but received {data.Length} bytes which gives {data.Length * 8} bits");

            if (typeof(T) == typeof(BFloat16))
                return (T[])(object)Dec<ushort>(data).Select(x => new BFloat16(x)).ToArray();

            if (typeof(T) == typeof(Float16))
                return (T[])(object)Dec<ushort>(data).Select(x => new Float16(x)).ToArray();

            var decodedArray = new T[data.Length * 8 / bitCount];
            Buffer.BlockCopy(data, 0, decodedArray, 0, data.Length);
            return decodedArray;
        }

        /// <summary>Creates TensorData by decoding raw IR-encoded bytes as the given dtype.</summary>
        public static TensorData TensorData(DType type, long[] dims, byte[] irEncodedData)
        {
            if (type.ToIVarType() == typeof(bit)) return TensorData(dims, Dec<bool>(irEncodedData));
            if (type.ToIVarType() == typeof(int8)) return TensorData(dims, Dec<sbyte>(irEncodedData));
            if (type.ToIVarType() == typeof(int16)) return TensorData(dims, Dec<short>(irEncodedData));
            if (type.ToIVarType() == typeof(int32)) return TensorData(dims, Dec<int>(irEncodedData));
            if (type.ToIVarType() == typeof(int64)) return TensorData(dims, Dec<long>(irEncodedData));
            if (type.ToIVarType() == typeof(uint8)) return TensorData(dims, Dec<byte>(irEncodedData));
            if (type.ToIVarType() == typeof(uint16)) return TensorData(dims, Dec<ushort>(irEncodedData));
            if (type.ToIVarType() == typeof(uint32)) return TensorData(dims, Dec<uint>(irEncodedData));
            if (type.ToIVarType() == typeof(uint64)) return TensorData(dims, Dec<ulong>(irEncodedData));
            if (type.ToIVarType() == typeof(float16)) return TensorData(dims, Dec<Float16>(irEncodedData));
            if (type.ToIVarType() == typeof(bfloat16)) return TensorData(dims, Dec<BFloat16>(irEncodedData));
            if (type.ToIVarType() == typeof(float32)) return TensorData(dims, Dec<float>(irEncodedData));
            if (type.ToIVarType() == typeof(float64)) return TensorData(dims, Dec<double>(irEncodedData));
            throw new UnsupportedDTypeException(ErrorCodes.GC006, type.ToString(), "data type", 
                $"Method not implemented for data type '{type}'");
        }

        /// <summary>Creates TensorData by decoding base64 raw IR-encoded data.</summary>
        public static TensorData TensorData(DType type, long[] dims, string base64IREncodedData)
        {
            byte[] data = Convert.FromBase64String(base64IREncodedData);
            return TensorData(type, dims, data);
        }

        /// <summary>Creates empty (zero-length) TensorData of the given dtype.</summary>
        public static TensorData TensorData(DType type)
            => TensorData(type, (long[])[0], (byte[])[]);

        #endregion

        #region Input Tensors Constructors

        /// <summary>Creates a module input node matching the variable's dtype, rank, and data structure.</summary>
        public static Variable ToInput(this Variable var)
        {
            var structure = var.Structure();
            var dtype = var.Type;
            var rank = var.Rank;
            if (structure == DataStructure.Tensor)
                return InternalOp.ModuleTensorInput(dtype, rank, null, var.ModuleFn, var.UniqueName);

            if (structure == DataStructure.Optional)
                return InternalOp.ModuleOptionalInput(dtype, null, null, var.UniqueName);

            if (structure == DataStructure.Sequence)
                return InternalOp.ModuleSequenceInput(dtype, null, null, var.UniqueName);

            throw new UnsupportedDTypeException(ErrorCodes.GC009, structure.ToString(), "data structure",
                $"InputTensorSequence not implemented for data structure '{structure}'");
        }

        /// <summary>Creates a runtime (graph) input tensor with optional name and rank/dims.</summary>
        public static Tensor<T> InputTensor<T>(string? defaultName = null, TensorDim?[]? dims = null, int? rank = null, Function? moduleFn = null) where T : IVarType
        { 
            if (moduleFn is null && (typeof(T) == typeof(IModelVarType) || typeof(T) == typeof(IModel)))
                throw new InvalidTensorOperationException(ErrorCodes.GC008, "InputTensor", typeof(T).Name, 
                    "Module function cannot be null when T is IModelVarType or IModel");

            return InternalOp.RuntimeInput(OnnxUtils.GetDType<T>(), rank ?? dims?.Length, defaultName, moduleFn);
        }

        /// <summary>Creates a runtime (graph) input tensor with optional name and rank/dims. The
        /// runtime dtype is already known, so build the node directly (no generic round-trip).</summary>
        public static Variable InputTensor(DType type, string? defaultName = null, TensorDim?[]? dims = null, int? rank = null)
            => InternalOp.RuntimeInput(type, rank ?? dims?.Length, defaultName);

        /// <summary>Creates a rank-1 runtime (graph) input.</summary>
        public static Vector<T> InputVector<T>(string? defaultName = null) where T : IVarType
            => InputTensor<T>(defaultName, rank: 1).Vec();

        /// <summary>Creates a rank-1 runtime (graph) input.</summary>
        public static Variable InputVector(DType type, string? defaultName = null)
            => InternalOp.RuntimeInput(type, 1, defaultName);

        /// <summary>Creates a rank-0 runtime (graph) input.</summary>
        public static Scalar<T> InputScalar<T>(string? defaultName = null, Function? moduleFn = null) where T : IVarType
            => InputTensor<T>(defaultName, rank: 0, moduleFn: moduleFn).Scalar();

        /// <summary>Creates a rank-0 runtime (graph) input.</summary>
        public static Variable InputScalar(DType type, string? defaultName = null, Function? moduleFn = null)
            => InternalOp.RuntimeInput(type, 0, defaultName, moduleFn);

        /// <summary>Not yet implemented; always throws.</summary>
        public static OptionalTensor<T> InputOptionalTensor<T>(string? defaultName = null) where T : IVarType
            => throw new InvalidTensorOperationException(ErrorCodes.GC009, "InputOptionalTensor", typeof(T).Name,
                "InputOptionalTensor functionality is not yet implemented");

        /// <summary>Not yet implemented; always throws.</summary>
        public static Variable InputOptionalTensor(DType type, string? defaultName = null)
            => throw new InvalidTensorOperationException(ErrorCodes.GC009, "InputOptionalTensor", type.ToString(),
                "InputOptionalTensor functionality is not yet implemented");

        /// <summary>Not yet implemented; always throws.</summary>
        public static TensorSequence<T> InputTensorSequence<T>(string? defaultName = null) where T : IVarType
            => throw new InvalidTensorOperationException(ErrorCodes.GC010, "InputTensorSequence", typeof(T).Name,
                "InputTensorSequence functionality is not yet implemented");

        /// <summary>Not yet implemented; always throws.</summary>
        public static Variable InputTensorSequence(DType type, string? defaultName = null)
            => throw new InvalidTensorOperationException(ErrorCodes.GC010, "InputTensorSequence", type.ToString(),
                "InputTensorSequence functionality is not yet implemented");

        #endregion

        #region TensorStruct Factory Methods

        /// <summary>
        /// Creates a TensorStruct from positional field values and wraps it in a DispatchProxy
        /// implementing the given IStruct interface. Property access on the returned proxy is
        /// wired to TensorStructGetField graph operations rooted on the new TENSOR_STRUCT_CREATE
        /// node — never on the original input tensors. Fields must be supplied in the order
        /// declared by the IStruct interface.
        /// </summary>
        /// <typeparam name="T">The IStruct interface type defining the struct fields</typeparam>
        /// <param name="fields">Field values, positional in IStruct declaration order</param>
        /// <returns>A proxy implementing T with property access wired to graph operations</returns>
        public static T TensorStruct<T>(params Variable[] fields) where T : IStruct
        {
            if (typeof(T) == typeof(DTypeStruct))
                throw new InvalidOperationException("Globals.TensorStruct<DTypeStruct> requires a concrete IStruct interface type");

            var def = StructDefExtractor.ExtractFromType<T>();
            var dtype = DType.GetOrCreateForTensorStruct(def);
            var structVar = InternalOp.TensorStructCreate(dtype, fields);
            return (T)TensorStructProxyFactory.Create(typeof(T), structVar, def);
        }

        /// <summary>
        /// Wraps an existing struct-shaped Variable (e.g. the result of <c>IfElse</c> or
        /// <c>SequenceAt</c> over a struct sequence) in a DispatchProxy implementing the given
        /// IStruct interface, so callers can read fields via property access (e.g.
        /// <c>proxy.First</c>) instead of <c>InternalOp.TensorStructGetField</c> calls.
        /// </summary>
        /// <typeparam name="T">The IStruct interface to expose</typeparam>
        /// <param name="existingStruct">An Variable carrying struct-shaped data.</param>
        public static T AsTensorStruct<T>(Variable existingStruct) where T : IStruct
        {
            if (typeof(T) == typeof(DTypeStruct))
                throw new InvalidOperationException("Globals.AsTensorStruct<DTypeStruct> requires a concrete IStruct interface type");

            if (existingStruct is null)
                throw new ArgumentNullException(nameof(existingStruct));

            var def = StructDefExtractor.ExtractFromType<T>();
            return (T)TensorStructProxyFactory.Create(typeof(T), existingStruct, def);
        }

        #endregion

        #region Trainable Tensors

        /// <summary>Creates a trainable parameter tensor initialized from the given data.</summary>
        public static Variable TrainableTensor(TensorData data, string? defaultName = null)
            => InternalOp.ModelParamData(data, isTrainable: true, identifierTemplateString: null, defaultName);

        internal static Variable TrainableTensor(TensorData data, bool isTrainable, string? identifierTemplateString, string? defaultName = null)
            => InternalOp.ModelParamData(data, isTrainable: isTrainable, identifierTemplateString, defaultName);

        /// <summary>Creates a trainable parameter tensor initialized from the given data.</summary>
        public static Tensor<T> TrainableTensor<T>(TensorData<T> data, string? defaultName = null) where T : IVarType
            => (Variable)TrainableTensor((TensorData)data, defaultName);
        
        #endregion
    }
}


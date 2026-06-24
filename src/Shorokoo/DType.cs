
using Shorokoo.Core.Inference.Abstractions;
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
using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace Shorokoo
{
    /// <summary>
    /// Runtime element data-type descriptor: the standard numeric/String/Bool types,
    /// Module/Model markers, generic-type placeholders (GenericType1..8), and
    /// dynamically registered TensorStruct types. Compares by type id and generic
    /// parameter name.
    /// </summary>
    public class DType
    {
        private readonly string sType;
        private readonly int iType;
        private readonly object defaultVal;
        private readonly string? genericTypeParamName;

        private DType(string sType, int iType, object defaultVal, string? genericTypeParamName = null)
        {
            this.sType = sType;
            this.iType = iType;
            this.defaultVal = defaultVal;
            this.genericTypeParamName = genericTypeParamName;
        }

        /// <summary>False only for <see cref="Invalid"/>.</summary>
        public bool IsValid => this != DType.Invalid;
        /// <summary>The default (zero-like) value of this type, boxed.</summary>
        public object DefaultVal => defaultVal;
        /// <summary>The generic type parameter name (e.g. "T") when this DType references a generic parameter; otherwise null.</summary>
        public string? GenericTypeParamName => genericTypeParamName;
        /// <summary>True if this is one of the GenericType1..8 placeholder types.</summary>
        public bool IsGenericType => iType >= 1001 && iType <= 1008;
        /// <summary>True if this DType carries a generic type parameter name (see <see cref="GenericTypeParamName"/>).</summary>
        public bool IsGenericTypeReference => this.genericTypeParamName is not null;
        /// <summary>The 1-based generic placeholder index (1-8), or null for non-generic types.</summary>
        public int? GenericTypeIndex => IsGenericType ? (iType - 1000) : null;

        /// <summary>
        /// Returns true if this DType represents a TensorStruct type (iType in range 2000-2999).
        /// </summary>
        public bool IsTensorStructType => iType >= 2000 && iType <= 2999;

        /// <summary>
        /// Gets the TensorStructDef associated with this DType, if it is a TensorStruct type.
        /// Returns null for non-TensorStruct DTypes.
        /// </summary>
        public TensorStructDef? TensorStructDef => IsTensorStructType && _tensorStructDefRegistry.TryGetValue(this, out var def) ? def : null;


        /// <summary>The 16-bit brain floating point data type.</summary>
        public static DType BFloat16 { get; private set; } = new DType("BFloat16", 16, new BFloat16());
        /// <summary>The 16-bit IEEE floating point data type.</summary>
        public static DType Float16 { get; private set; } = new DType("Float16", 10, new Float16());
        /// <summary>The 32-bit IEEE floating point data type.</summary>
        public static DType Float32 { get; private set; } = new DType("Float32", 1, default(float));
        /// <summary>The 64-bit IEEE floating point data type.</summary>
        public static DType Float64 { get; private set; } = new DType("Float64", 11, default(double));
        /// <summary>The 4-bit signed integer data type.</summary>
        public static DType Int4 { get; private set; } = new DType("Int4", -8, default(sbyte));
        /// <summary>The 8-bit signed integer data type.</summary>
        public static DType Int8 { get; private set; } = new DType("Int8", 3, default(sbyte));
        /// <summary>The 16-bit signed integer data type.</summary>
        public static DType Int16 { get; private set; } = new DType("Int16", 5, default(short));
        /// <summary>The 32-bit signed integer data type.</summary>
        public static DType Int32 { get; private set; } = new DType("Int32", 6, default(int));
        /// <summary>The 64-bit signed integer data type.</summary>
        public static DType Int64 { get; private set; } = new DType("Int64", 7, default(long));
        /// <summary>The 4-bit unsigned integer data type.</summary>
        public static DType UInt4 { get; private set; } = new DType("UInt4", -2, default(byte));
        /// <summary>The 8-bit unsigned integer data type.</summary>
        public static DType UInt8 { get; private set; } = new DType("UInt8", 2, default(byte));
        /// <summary>The 16-bit unsigned integer data type.</summary>
        public static DType UInt16 { get; private set; } = new DType("UInt16", 4, default(ushort));
        /// <summary>The 32-bit unsigned integer data type.</summary>
        public static DType UInt32 { get; private set; } = new DType("UInt32", 12, default(uint));
        /// <summary>The 64-bit unsigned integer data type.</summary>
        public static DType UInt64 { get; private set; } = new DType("UInt64", 13, default(ulong));
        /// <summary>The UTF-8 string data type.</summary>
        public static DType String { get; private set; } = new DType("String", 8, string.Empty);
        /// <summary>The boolean data type.</summary>
        public static DType Bool { get; private set; } = new DType("Bool", 9, default(sbyte));
        /// <summary>The 64-bit complex data type.</summary>
        public static DType Complex64 { get; private set; } = new DType("Complex64", 14, default(double));
        /// <summary>The 128-bit complex data type.</summary>
        public static DType Complex128 { get; private set; } = new DType("Complex128", 15, default(double));
        /// <summary>Marker dtype for module-valued variables (not a tensor element type).</summary>
        public static DType Module { get; private set; } = new DType("Module", 997, default(int));
        /// <summary>Marker dtype for model-valued variables (not a tensor element type).</summary>
        public static DType Model { get; private set; } = new DType("Model", 999, default(int));
        /// <summary>Sentinel for an invalid/unknown dtype.</summary>
        public static DType Invalid { get; private set; } = new DType("Invalid", -1, default(int));

        // Generic type placeholders - used during VirtualGraph construction for unresolved generic type parameters
        /// <summary>Placeholder for unresolved generic type parameter #1 during VirtualGraph construction.</summary>
        public static DType GenericType1 { get; private set; } = new DType("GenericType1", 1001, default(int), null);
        /// <summary>Placeholder for unresolved generic type parameter #2 during VirtualGraph construction.</summary>
        public static DType GenericType2 { get; private set; } = new DType("GenericType2", 1002, default(int), null);
        /// <summary>Placeholder for unresolved generic type parameter #3 during VirtualGraph construction.</summary>
        public static DType GenericType3 { get; private set; } = new DType("GenericType3", 1003, default(int), null);
        /// <summary>Placeholder for unresolved generic type parameter #4 during VirtualGraph construction.</summary>
        public static DType GenericType4 { get; private set; } = new DType("GenericType4", 1004, default(int), null);
        /// <summary>Placeholder for unresolved generic type parameter #5 during VirtualGraph construction.</summary>
        public static DType GenericType5 { get; private set; } = new DType("GenericType5", 1005, default(int), null);
        /// <summary>Placeholder for unresolved generic type parameter #6 during VirtualGraph construction.</summary>
        public static DType GenericType6 { get; private set; } = new DType("GenericType6", 1006, default(int), null);
        /// <summary>Placeholder for unresolved generic type parameter #7 during VirtualGraph construction.</summary>
        public static DType GenericType7 { get; private set; } = new DType("GenericType7", 1007, default(int), null);
        /// <summary>Placeholder for unresolved generic type parameter #8 during VirtualGraph construction.</summary>
        public static DType GenericType8 { get; private set; } = new DType("GenericType8", 1008, default(int), null);

        // TensorStruct type registry - maps TensorStructDef to DType instances
        // iType range 2000-2999 is reserved for TensorStruct DTypes
        private static readonly object _tensorStructRegistryLock = new object();
        private static readonly Dictionary<TensorStructDef, DType> _tensorStructRegistry = new Dictionary<TensorStructDef, DType>();
        private static readonly Dictionary<DType, TensorStructDef> _tensorStructDefRegistry = new Dictionary<DType, TensorStructDef>();
        private static readonly Dictionary<int, DType> _tensorStructITypeToDType = new Dictionary<int, DType>();
        private static int _nextTensorStructIType = 2000;

        /// <summary>
        /// Gets or creates a DType for a TensorStruct with the given definition.
        /// Returns the same DType instance for structurally equal TensorStructDefs.
        /// </summary>
        /// <param name="def">The TensorStructDef describing the struct's fields</param>
        /// <returns>A DType representing the TensorStruct type</returns>
        public static DType GetOrCreateForTensorStruct(TensorStructDef def)
        {
            if (def == null) throw new ArgumentNullException(nameof(def));

            lock (_tensorStructRegistryLock)
            {
                // Check if we already have a DType for this struct definition
                if (_tensorStructRegistry.TryGetValue(def, out var existing))
                    return existing;

                // Create a new DType with a unique iType in the 2000-2999 range
                if (_nextTensorStructIType > 2999)
                    throw new InvalidOperationException("TensorStruct DType registry is full (max 1000 unique struct types)");

                var iType = _nextTensorStructIType++;
                var typeName = def.TypeName ?? $"TensorStruct_{iType}";
                var newDType = new DType(typeName, iType, default(int));

                _tensorStructRegistry[def] = newDType;
                _tensorStructDefRegistry[newDType] = def;
                _tensorStructITypeToDType[iType] = newDType;

                return newDType;
            }
        }

        /// <summary>
        /// Loader-side counterpart of <see cref="GetOrCreateForTensorStruct(TensorStructDef)"/>:
        /// registers <paramref name="def"/> at the specified <paramref name="protoTypeNum"/>
        /// so the saved attribute values (which carry the original protoTypeNum) resolve
        /// to the same DType on reload. If <paramref name="def"/> is already registered
        /// returns the existing DType (its iType may differ from <paramref name="protoTypeNum"/>
        /// if registration happened via the allocating overload before this call). If
        /// <paramref name="protoTypeNum"/> is already occupied by a different def, returns
        /// the occupant — the caller should treat structurally-different defs sharing an
        /// iType as a save-file corruption rather than try to merge them.
        /// </summary>
        public static DType GetOrCreateForTensorStructAtProtoTypeNum(TensorStructDef def, int protoTypeNum)
        {
            if (def == null) throw new ArgumentNullException(nameof(def));
            if (protoTypeNum < 2000 || protoTypeNum > 2999)
                throw new ArgumentException(
                    $"protoTypeNum must be in the TensorStruct range [2000, 2999], got {protoTypeNum}",
                    nameof(protoTypeNum));

            lock (_tensorStructRegistryLock)
            {
                if (_tensorStructRegistry.TryGetValue(def, out var existing))
                    return existing;

                if (_tensorStructITypeToDType.TryGetValue(protoTypeNum, out var occupant))
                    return occupant;

                var typeName = def.TypeName ?? $"TensorStruct_{protoTypeNum}";
                var newDType = new DType(typeName, protoTypeNum, default(int));

                _tensorStructRegistry[def] = newDType;
                _tensorStructDefRegistry[newDType] = def;
                _tensorStructITypeToDType[protoTypeNum] = newDType;
                if (protoTypeNum >= _nextTensorStructIType)
                    _nextTensorStructIType = protoTypeNum + 1;

                return newDType;
            }
        }

        /// <summary>
        /// Creates a DType with a generic type parameter name (e.g., "T", "Q", "R")
        /// </summary>
        public static DType CreateGenericType(int genericTypeIndex, string parameterName)
        {
            var baseDType = genericTypeIndex switch
            {
                1 => GenericType1,
                2 => GenericType2,
                3 => GenericType3,
                4 => GenericType4,
                5 => GenericType5,
                6 => GenericType6,
                7 => GenericType7,
                8 => GenericType8,
                _ => throw new ArgumentException($"Invalid generic type index: {genericTypeIndex}. Must be between 1 and 8.")
            };

            return new DType(baseDType.sType, baseDType.iType, baseDType.defaultVal, parameterName);
        }

        /// <summary>
        /// Creates a DType with a generic type parameter name from an existing DType
        /// </summary>
        public static DType CreateWithGenericParam(DType baseType, string parameterName)
        {
            return new DType(baseType.sType, baseType.iType, baseType.defaultVal, parameterName);
        }
        
        /// <summary>
        /// Gets the base DType from a ProtoTypeNum value.
        /// </summary>
        public static DType FromProtoTypeNum(int protoTypeNum)
        {
            if (Mapping.TryGetValue(protoTypeNum, out var dtype))
                return dtype;
            throw new ArgumentException($"Unknown ProtoTypeNum: {protoTypeNum}");
        }

        private static Dictionary<int, DType>? _dctMapping = null;
        private static Dictionary<int, DType> Mapping
        {
            get
            {
                if (_dctMapping == null)
                {
                    _dctMapping = new Dictionary<int, DType>
                    {
                        { 16, DType.BFloat16 },
                        { 10, DType.Float16 },
                        { 01, DType.Float32 },
                        { 11, DType.Float64 },
                        { -8, DType.Int4 },
                        { 03, DType.Int8 },
                        { 05, DType.Int16 },
                        { 06, DType.Int32 },
                        { 07, DType.Int64 },
                        { -2, DType.UInt4 },
                        { 02, DType.UInt8 },
                        { 04, DType.UInt16 },
                        { 12, DType.UInt32 },
                        { 13, DType.UInt64 },
                        { 08, DType.String },
                        { 09, DType.Bool },
                        { 14, DType.Complex64 },
                        { 15, DType.Complex128 },
                        { 997, DType.Module },
                        { 999, DType.Model },
                        { -1, DType.Invalid },
                        { 1001, DType.GenericType1 },
                        { 1002, DType.GenericType2 },
                        { 1003, DType.GenericType3 },
                        { 1004, DType.GenericType4 },
                        { 1005, DType.GenericType5 },
                        { 1006, DType.GenericType6 },
                        { 1007, DType.GenericType7 },
                        { 1008, DType.GenericType8 },
                    };
                }

                return _dctMapping;
            }
        }

        /// <summary>Resolves a DType from its ProtoTypeNum; null stays null.</summary>
        public static explicit operator DType?(int? dType) => dType == null ? null : (DType)dType.Value;
        /// <summary>Resolves a DType from its ProtoTypeNum, including dynamically registered TensorStruct dtypes (2000-2999).</summary>
        public static implicit operator DType(int iType)
        {
            if (Mapping.ContainsKey(iType)) return Mapping[iType];

            // TensorStruct DTypes are dynamically registered in the 2000-2999 range
            // and not present in the static Mapping. The reload path stamps each struct
            // DType with its original protoTypeNum (via GetOrCreateForTensorStructAtProtoTypeNum
            // in ParseTensorStructMetadata), so a node attribute carrying that protoTypeNum
            // must resolve through the registry rather than Mapping.
            if (iType >= 2000 && iType <= 2999)
            {
                lock (_tensorStructRegistryLock)
                {
                    if (_tensorStructITypeToDType.TryGetValue(iType, out var structDType))
                        return structDType;
                }
            }

            Debug.Assert(false);
            return DType.Invalid;
        }

        /// <summary>Returns the ProtoTypeNum; null stays null.</summary>
        public static explicit operator int?(DType? dType) => dType?.iType;
        /// <summary>Returns the ProtoTypeNum.</summary>
        public static explicit operator int(DType dType) => dType.iType;
        /// <summary>Returns the type name.</summary>
        public static explicit operator string(DType dType) => dType.sType;

        /// <summary>The numeric type id; matches the ONNX TensorProto.DataType value for standard types, Shorokoo-specific values otherwise.</summary>
        public int ProtoTypeNum => this.iType;

        /// <summary>Maps this DType to its Shorokoo IVarType marker type (e.g. Float32 to <see cref="float32"/>); throws for unsupported types.</summary>
        public Type ToIVarType()
        {
            // If this DType has a GenericTypeParamName, we need to convert based on the base iType
            // The GenericTypeParamName is metadata that doesn't affect the underlying type
            if (this.IsGenericTypeReference)
            {
                // Get the base DType without the GenericTypeParamName using FromProtoTypeNum
                var baseDType = DType.FromProtoTypeNum(this.iType);
                return baseDType.ToIVarType();
            }
            
            if (this == DType.BFloat16) return typeof(bfloat16);
            else if (this == DType.Float16) return typeof(float16);
            else if (this == DType.Float32) return typeof(float32);
            else if (this == DType.Float64) return typeof(float64);
            else if (this == DType.Int4) 
                throw new UnsupportedDTypeException(ErrorCodes.DT001, this.sType, "ToIVarType", "Int4 precision is not supported for variable type conversion");
            else if (this == DType.Int8) return typeof(int8);
            else if (this == DType.Int16) return typeof(int16);
            else if (this == DType.Int32) return typeof(int32);
            else if (this == DType.Int64) return typeof(int64);
            else if (this == DType.UInt4) 
                throw new UnsupportedDTypeException(ErrorCodes.DT002, this.sType, "ToIVarType", "UInt4 precision is not supported for variable type conversion");
            else if (this == DType.UInt8) return typeof(uint8);
            else if (this == DType.UInt16) return typeof(uint16);
            else if (this == DType.UInt32) return typeof(uint32);
            else if (this == DType.UInt64) return typeof(uint64);
            else if (this == DType.String) return typeof(@string);
            else if (this == DType.Bool) return typeof(bit);
            else if (this == DType.Complex64) 
                throw new UnsupportedDTypeException(ErrorCodes.DT007, this.sType, "ToIVarType", "Complex64 numbers are not supported for variable type conversion");
            else if (this == DType.Complex128) 
                throw new UnsupportedDTypeException(ErrorCodes.DT008, this.sType, "ToIVarType", "Complex128 numbers are not supported for variable type conversion");
            else if (this == DType.Module) return typeof(IModuleVarType);
            else if (this == DType.Model) return typeof(IModelVarType);
            else if (this == DType.Invalid) return typeof(invalid);
            
            // Handle generic types
            else if (this == DType.GenericType1) return typeof(IGenericType1);
            else if (this == DType.GenericType2) return typeof(IGenericType2);
            else if (this == DType.GenericType3) return typeof(IGenericType3);
            else if (this == DType.GenericType4) return typeof(IGenericType4);
            else if (this == DType.GenericType5) return typeof(IGenericType5);
            else if (this == DType.GenericType6) return typeof(IGenericType6);
            else if (this == DType.GenericType7) return typeof(IGenericType7);
            else if (this == DType.GenericType8) return typeof(IGenericType8);
            
            // Handle TensorStruct types - return DTypeStruct for dynamic struct definitions
            else if (this.IsTensorStructType) return typeof(DTypeStruct);

            throw new UnsupportedDTypeException(ErrorCodes.DT009, this.sType, "ToIVarType", $"Unknown DType with iType={this.iType}");
        }

        /// <summary>Maps this DType to its CLR storage type (e.g. Float32 to <see cref="float"/>); throws for unsupported types.</summary>
        public Type ToPrimitiveType()
        {
            if (this == DType.BFloat16) return typeof(BFloat16);
            else if (this == DType.Float16) return typeof(Float16);
            else if (this == DType.Float32) return typeof(float);
            else if (this == DType.Float64) return typeof(double);
            else if (this == DType.Int4) 
                throw new UnsupportedDTypeException(ErrorCodes.DT010, this.sType, "ToPrimitiveType", "Int4 precision is not supported for primitive type conversion");
            else if (this == DType.Int8) return typeof(sbyte);
            else if (this == DType.Int16) return typeof(short);
            else if (this == DType.Int32) return typeof(int);
            else if (this == DType.Int64) return typeof(long);
            else if (this == DType.UInt4) 
                throw new UnsupportedDTypeException(ErrorCodes.DT011, this.sType, "ToPrimitiveType", "UInt4 precision is not supported for primitive type conversion");
            else if (this == DType.UInt8) return typeof(byte);
            else if (this == DType.UInt16) return typeof(ushort);
            else if (this == DType.UInt32) return typeof(uint);
            else if (this == DType.UInt64) return typeof(ulong);
            else if (this == DType.String) return typeof(string);
            else if (this == DType.Bool) return typeof(bool);
            else if (this == DType.Complex64) 
                throw new UnsupportedDTypeException(ErrorCodes.DT013, this.sType, "ToPrimitiveType", "Complex64 numbers are not supported for primitive type conversion");
            else if (this == DType.Complex128) 
                throw new UnsupportedDTypeException(ErrorCodes.DT014, this.sType, "ToPrimitiveType", "Complex128 numbers are not supported for primitive type conversion");
            else if (this == DType.Module) 
                throw new UnsupportedDTypeException(ErrorCodes.DT015, this.sType, "ToPrimitiveType", "Module type cannot be converted to a primitive type");
            else if (this == DType.Model) 
                throw new UnsupportedDTypeException(ErrorCodes.DT016, this.sType, "ToPrimitiveType", "Model type cannot be converted to a primitive type");
            else if (this == DType.Invalid) return typeof(int);
            
            // Handle generic types - they can't be converted to primitive types (use new error code DT018)
            else if (this.IsGenericType) 
                throw new UnsupportedDTypeException(ErrorCodes.DT018, this.sType, "ToPrimitiveType", $"Generic type {this.sType} cannot be converted to a primitive type. Generic types must be specialized first.");

            throw new UnsupportedDTypeException(ErrorCodes.DT019, this.sType, "ToPrimitiveType", $"Unknown DType with iType={this.iType}");
        }

        /// <summary>Strips a generic type parameter reference, returning the base placeholder DType; returns this instance unchanged otherwise.</summary>
        public DType ToNonGenericType()
        {
            if (this.IsGenericTypeReference)
                return DType.FromProtoTypeNum(this.iType);

            return this;
        }

        /// <summary>Storage width in bits of one element (e.g. 16 for BFloat16, 8 for Bool); throws for types without a fixed width.</summary>
        public int EncodingBitCount
        {
            get
            {
                // BFloat16 elements occupy 16 storage bits (ushort bit pattern), same as
                // Float16 — EncodingBitCount is the storage width used for byte-buffer
                // sizing (Enc/Dec, zero-tensor creation), not the mantissa precision.
                if (this == DType.BFloat16) return 16;
                else if (this == DType.Float16) return 16;
                else if (this == DType.Float32) return 32;
                else if (this == DType.Float64) return 64;
                else if (this == DType.Int4) 
                    throw new UnsupportedDTypeException(ErrorCodes.DT018, this.sType, "EncodingBitCount", "Int4 precision is not supported for bit count encoding");
                else if (this == DType.Int8) return 8;
                else if (this == DType.Int16) return 16;
                else if (this == DType.Int32) return 32;
                else if (this == DType.Int64) return 64;
                else if (this == DType.UInt4) 
                    throw new UnsupportedDTypeException(ErrorCodes.DT019, this.sType, "EncodingBitCount", "UInt4 precision is not supported for bit count encoding");
                else if (this == DType.UInt8) return 8;
                else if (this == DType.UInt16) return 16;
                else if (this == DType.UInt32) return 32;
                else if (this == DType.UInt64) return 64;
                else if (this == DType.String)
                    throw new UnsupportedDTypeException(ErrorCodes.DT020, this.sType, "EncodingBitCount", "String DType has variable bit count and is not supported");
                else if (this == DType.Bool) return 8;
                else if (this == DType.Complex64) 
                    throw new UnsupportedDTypeException(ErrorCodes.DT021, this.sType, "EncodingBitCount", "Complex64 numbers are not supported for bit count encoding");
                else if (this == DType.Complex128) 
                    throw new UnsupportedDTypeException(ErrorCodes.DT022, this.sType, "EncodingBitCount", "Complex128 numbers are not supported for bit count encoding");
                else if (this == DType.Module) 
                    throw new UnsupportedDTypeException(ErrorCodes.DT023, this.sType, "EncodingBitCount", "Module type has no fixed bit count encoding");
                else if (this == DType.Model) 
                    throw new UnsupportedDTypeException(ErrorCodes.DT024, this.sType, "EncodingBitCount", "Model type has no fixed bit count encoding");
                else if (this == DType.Invalid) return -1;
                
                // Handle generic types - they don't have a fixed bit count
                else if (this.IsGenericType) 
                    throw new UnsupportedDTypeException(ErrorCodes.DT025, this.sType, "EncodingBitCount", $"Generic type {this.sType} has no fixed bit count. Generic types must be specialized first.");

                throw new UnsupportedDTypeException(ErrorCodes.DT025, this.sType, "EncodingBitCount", $"Unknown DType with iType={this.iType}");
            }
        }

        /// <summary>True if this is <see cref="Model"/>.</summary>
        public bool IsModelType => this == DType.Model;

        /// <summary>The dtype name (with the parameter name appended for tagged generic dtypes).</summary>
        public override string ToString()
        {
            // For generic types with parameter names, return a more descriptive string
            if (this.IsGenericType && !string.IsNullOrEmpty(this.genericTypeParamName))
            {
                return $"{this.sType}<{this.genericTypeParamName}>";
            }
            return this.sType;
        }

        /// <summary>True if this DType's IVarType is assignable to T (e.g. <c>DType.Float32.Is&lt;FloatLike&gt;()</c>).</summary>
        public bool Is<T>() where T : IVarType
        {
            return typeof(T).IsAssignableFrom(this.ToIVarType());
        }

        /// <summary>Value equality on the type id and the generic type parameter name.</summary>
        public override bool Equals(object? obj)
        {
            if (obj is null)
                return false;

            if (obj is DType dtype)
            {
                return this.iType == dtype.iType &&
                       this.genericTypeParamName == dtype.genericTypeParamName;
            }

            return false;
        }

        /// <summary>Hash code combining the underlying type id and generic parameter tag.</summary>
        public override int GetHashCode()
        {
            return HashCode.Combine(this.iType, this.genericTypeParamName);
        }
    }

}

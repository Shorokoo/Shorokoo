using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using Shorokoo;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Onnx;

namespace Shorokoo.Core
{
    /// <summary>
    /// Utility class for extracting TensorStructDef from user-defined IStruct types.
    /// </summary>
    internal static class StructDefExtractor
    {
        /// <summary>
        /// Extracts a TensorStructDef from a user-defined IStruct interface type.
        /// The interface's property declarations define the struct's fields.
        /// </summary>
        /// <typeparam name="T">The IStruct interface type to extract from</typeparam>
        /// <returns>A TensorStructDef describing the struct's fields</returns>
        /// <exception cref="InvalidOperationException">Thrown if T is DTypeStruct (requires explicit definition)</exception>
        public static TensorStructDef ExtractFromType<T>() where T : IStruct
        {
            return ExtractFromType(typeof(T));
        }

        /// <summary>
        /// Extracts a TensorStructDef from a user-defined IStruct interface, record or class.
        /// </summary>
        /// <param name="type">The IStruct type to extract from</param>
        /// <returns>A TensorStructDef describing the struct's fields</returns>
        /// <exception cref="InvalidOperationException">Thrown if type is DTypeStruct (requires explicit definition)</exception>
        /// <exception cref="InvalidOperationException">Thrown if type declares a member a struct field cannot be</exception>
        /// <exception cref="ArgumentException">Thrown if type doesn't implement IStruct</exception>
        public static TensorStructDef ExtractFromType(Type type)
        {
            if (type == typeof(DTypeStruct))
                throw new InvalidOperationException("DTypeStruct requires an explicit TensorStructDef; it cannot be extracted from type.");

            if (!typeof(IStruct).IsAssignableFrom(type))
                throw new ArgumentException($"Type {type.Name} does not implement IStruct", nameof(type));

            // A struct's fields are its public instance properties, inherited ones included. A member that
            // cannot be one — a C# field, a ref-returning property, a property of a type a field cannot
            // hold — is refused rather than left out of the layout.
            var csharpField = type.GetFields(BindingFlags.Public | BindingFlags.Instance).FirstOrDefault();
            if (csharpField is not null)
                throw new InvalidOperationException(
                    $"Struct type {type.Name} declares '{csharpField.Name}' as a C# field; declare struct fields as properties.");

            var fields = new List<TensorStructFieldDef>();
            foreach (var declaring in DeclaringTypes(type))
            {
                foreach (var prop in declaring.GetProperties(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly))
                {
                    if (fields.Any(f => f.Name == prop.Name))
                        continue;
                    var fieldDef = prop.PropertyType.IsByRef || prop.GetIndexParameters().Length > 0
                        ? null
                        : AnalyzePropertyType(prop.Name, prop.PropertyType);
                    fields.Add(fieldDef ?? throw new InvalidOperationException(
                        $"Struct type {type.Name} declares property '{prop.Name}' of type {prop.PropertyType.Name}, which a struct field cannot hold; " +
                        "a field is a Tensor, Vector, Scalar, TensorSequence, OptionalTensor or TensorStruct."));
                }
            }

            return new TensorStructDef(fields, type.FullName) { DeclaringType = type };
        }

        /// <summary>The types whose own properties make up <paramref name="type"/>'s fields, in field order:
        /// a class's base classes from the root down, then the type itself, then the IStruct interfaces it extends.</summary>
        private static IEnumerable<Type> DeclaringTypes(Type type)
        {
            var classes = new List<Type>();
            for (var t = type; t is not null && t != typeof(object) && t != typeof(ValueType); t = t.BaseType)
                classes.Insert(0, t);
            return classes.Concat(type.GetInterfaces()
                .Where(i => i != typeof(IStruct) && i != typeof(IVarType) && typeof(IStruct).IsAssignableFrom(i)));
        }

        /// <summary>
        /// Analyzes a property type to determine the field definition.
        /// </summary>
        private static TensorStructFieldDef? AnalyzePropertyType(string propertyName, Type propertyType)
        {
            DataStructure structure;
            int? rank = null;
            DType elementType;

            // Check if it's a generic type (Tensor<T>, TensorSequence<T>, etc.)
            if (propertyType.IsGenericType)
            {
                var genericDef = propertyType.GetGenericTypeDefinition();
                var genericArgs = propertyType.GetGenericArguments();

                if (genericDef == typeof(Tensor<>) || propertyType.Name.StartsWith("Tensor`"))
                {
                    structure = DataStructure.Tensor;
                    rank = null; // Unknown rank
                    elementType = GetDTypeFromIVarType(genericArgs[0]);
                }
                else if (genericDef == typeof(Vector<>) || propertyType.Name.StartsWith("Vector`"))
                {
                    structure = DataStructure.Tensor;
                    rank = 1;
                    elementType = GetDTypeFromIVarType(genericArgs[0]);
                }
                else if (genericDef == typeof(Scalar<>) || propertyType.Name.StartsWith("Scalar`"))
                {
                    structure = DataStructure.Tensor;
                    rank = 0;
                    elementType = GetDTypeFromIVarType(genericArgs[0]);
                }
                else if (genericDef == typeof(TensorSequence<>) || propertyType.Name.StartsWith("TensorSequence`"))
                {
                    structure = DataStructure.Sequence;
                    elementType = GetDTypeFromIVarType(genericArgs[0]);
                }
                else if (genericDef == typeof(OptionalTensor<>) || propertyType.Name.StartsWith("OptionalTensor`"))
                {
                    structure = DataStructure.Optional;
                    elementType = GetDTypeFromIVarType(genericArgs[0]);
                }
                else if (genericDef == typeof(TensorStruct<>) || propertyType.Name.StartsWith("TensorStruct`"))
                {
                    structure = DataStructure.TensorStruct;
                    // For nested TensorStruct, we need to extract or lookup the DType
                    var nestedStructType = genericArgs[0];
                    if (nestedStructType == typeof(DTypeStruct))
                    {
                        // Can't handle DTypeStruct in nested position - needs explicit definition
                        throw new InvalidOperationException($"Property {propertyName}: Nested TensorStruct<DTypeStruct> requires explicit TensorStructDef");
                    }
                    var nestedDef = ExtractFromType(nestedStructType);
                    elementType = DType.GetOrCreateForTensorStruct(nestedDef);
                }
                else
                {
                    // Unknown generic type
                    return null;
                }
            }
            else
            {
                // Non-generic types are not supported
                return null;
            }

            return new TensorStructFieldDef(propertyName, structure, rank, elementType);
        }

        /// <summary>
        /// Gets the DType corresponding to an IVarType.
        /// Handles both concrete types (float32, int64, etc.) and generic type placeholders (IGenericType1-8).
        /// </summary>
        private static DType GetDTypeFromIVarType(Type iVarType)
        {
            if (iVarType == typeof(float32)) return DType.Float32;
            if (iVarType == typeof(float64)) return DType.Float64;
            if (iVarType == typeof(float16)) return DType.Float16;
            if (iVarType == typeof(bfloat16)) return DType.BFloat16;
            if (iVarType == typeof(int8)) return DType.Int8;
            if (iVarType == typeof(int16)) return DType.Int16;
            if (iVarType == typeof(int32)) return DType.Int32;
            if (iVarType == typeof(int64)) return DType.Int64;
            if (iVarType == typeof(uint8)) return DType.UInt8;
            if (iVarType == typeof(uint16)) return DType.UInt16;
            if (iVarType == typeof(uint32)) return DType.UInt32;
            if (iVarType == typeof(uint64)) return DType.UInt64;
            if (iVarType == typeof(bit)) return DType.Bool;

            // Generic type placeholders — used during graph building when generic parameters
            // are replaced with IGenericType1-8 placeholders by GraphBuilder.ResolveGenericMethodWithPlaceholders
            if (iVarType == typeof(IGenericType1)) return DType.GenericType1;
            if (iVarType == typeof(IGenericType2)) return DType.GenericType2;
            if (iVarType == typeof(IGenericType3)) return DType.GenericType3;
            if (iVarType == typeof(IGenericType4)) return DType.GenericType4;
            if (iVarType == typeof(IGenericType5)) return DType.GenericType5;
            if (iVarType == typeof(IGenericType6)) return DType.GenericType6;
            if (iVarType == typeof(IGenericType7)) return DType.GenericType7;
            if (iVarType == typeof(IGenericType8)) return DType.GenericType8;

            throw new ArgumentException($"Unknown IVarType: {iVarType.Name}");
        }

        /// <summary>
        /// Extracts the TensorStructDef and corresponding DType from a TensorStruct&lt;T&gt; type.
        /// This is a convenience method that validates the type argument, extracts the struct definition,
        /// and creates/gets the DType in a single call.
        /// </summary>
        /// <param name="tensorStructType">The TensorStruct&lt;T&gt; type</param>
        /// <param name="context">Description of the operation for error messages</param>
        /// <returns>Tuple of (TensorStructDef, DType)</returns>
        /// <exception cref="InvalidOperationException">Thrown if the type argument is DTypeStruct</exception>
        public static (TensorStructDef def, DType dtype) ExtractFromTensorStructType(Type tensorStructType, string context)
        {
            var structTypeArg = tensorStructType.GetGenericArguments()[0];
            if (structTypeArg == typeof(DTypeStruct))
                throw new InvalidOperationException($"TensorStruct<DTypeStruct> requires an explicit TensorStructDef for {context}");
            
            var structDef = ExtractFromType(structTypeArg);
            var structDType = DType.GetOrCreateForTensorStruct(structDef);
            return (structDef, structDType);
        }
    }
}

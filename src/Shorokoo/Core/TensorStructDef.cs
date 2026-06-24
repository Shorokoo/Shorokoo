using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
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
    /// Defines a field within a TensorStruct: its name, data structure type, optional rank, and element type.
    /// </summary>
    public class TensorStructFieldDef
    {
        /// <summary>
        /// The name of the field.
        /// </summary>
        public string Name { get; }

        /// <summary>
        /// The data structure type of the field (Tensor, Sequence, Optional, TensorStruct).
        /// </summary>
        public DataStructure Structure { get; }

        /// <summary>
        /// The rank of the field (for Tensor types). Null if rank is unknown or not applicable.
        /// </summary>
        public int? Rank { get; }

        /// <summary>
        /// The element type of the field. For nested TensorStruct fields, this will be a TensorStruct DType.
        /// </summary>
        public DType ElementType { get; }

        public TensorStructFieldDef(string name, DataStructure structure, int? rank, DType elementType)
        {
            Name = name ?? throw new ArgumentNullException(nameof(name));
            Structure = structure;
            Rank = rank;
            ElementType = elementType ?? throw new ArgumentNullException(nameof(elementType));
        }

        /// <summary>
        /// Checks structural equality with another TensorStructFieldDef.
        /// </summary>
        public bool StructuralEquals(TensorStructFieldDef? other)
        {
            if (other is null) return false;
            if (ReferenceEquals(this, other)) return true;
            
            return Name == other.Name
                && Structure == other.Structure
                && Rank == other.Rank
                && ReferenceEquals(ElementType, other.ElementType);
        }

        /// <summary>
        /// Computes a structural hash code for this field definition.
        /// </summary>
        public int ComputeStructuralHash()
        {
            unchecked
            {
                int hash = 17;
                hash = hash * 31 + Name.GetHashCode();
                hash = hash * 31 + (int)Structure;
                hash = hash * 31 + (Rank?.GetHashCode() ?? 0);
                hash = hash * 31 + ElementType.ProtoTypeNum;
                return hash;
            }
        }

        public override string ToString() => $"{Name}: {Structure}<{ElementType}>{(Rank.HasValue ? $"[{Rank}]" : "")}";
    }

    /// <summary>
    /// Describes the structure of a TensorStruct: field names, field types, and field order.
    /// Two TensorStructDef instances are considered structurally equal if they have the same fields in the same order.
    /// </summary>
    public class TensorStructDef : IEquatable<TensorStructDef>
    {
        private readonly ImmutableArray<TensorStructFieldDef> _fields;
        private readonly int _structuralHash;

        /// <summary>
        /// The ordered list of field definitions.
        /// </summary>
        public ImmutableArray<TensorStructFieldDef> Fields => _fields;

        public ImmutableArray<ImmutableArray<TensorStructFieldDef>> FlattenedFields => 
            [..Fields.SelectMany(x => x.Structure == DataStructure.TensorStruct ? 
                                        ((TensorStructDef)x.ElementType.TensorStructDef!).FlattenedFields : 
                                        [ImmutableArray.Create(x)])];

        /// <summary>
        /// The type name for this struct (e.g., the full name of the IStruct interface).
        /// Null for dynamically created structs.
        /// </summary>
        public string? TypeName { get; }

        /// <summary>
        /// Creates a new TensorStructDef with the specified fields and optional type name.
        /// </summary>
        public TensorStructDef(IEnumerable<TensorStructFieldDef> fields, string? typeName = null)
        {
            _fields = [..fields];
            TypeName = typeName;
            _structuralHash = ComputeStructuralHash();
        }

        /// <summary>
        /// Gets a field definition by name.
        /// </summary>
        public TensorStructFieldDef? GetField(string name) => _fields.FirstOrDefault(f => f.Name == name);

        /// <summary>
        /// Gets the index of a field by name, or -1 if not found.
        /// </summary>
        public int GetFieldIndex(string name)
        {
            for (int i = 0; i < _fields.Length; i++)
            {
                if (_fields[i].Name == name) return i;
            }
            return -1;
        }

        /// <summary>
        /// Constructs a <see cref="TensorDataStruct"/> by pairing this definition's fields
        /// (in declaration order) with the supplied <paramref name="data"/> values.
        /// The count of values must equal the number of fields.
        /// </summary>
        /// <example>
        /// <code>
        /// // Single-input model — no need to name the field manually.
        /// var batch = rig.InputDef.FromOrderedData(TensorData([4L, 8L], myArray));
        /// </code>
        /// </example>
        public TensorDataStruct FromOrderedData(params TensorData[] data)
        {
            if (data is null) throw new ArgumentNullException(nameof(data));
            if (data.Length != _fields.Length)
                throw new ArgumentException(
                    $"Expected {_fields.Length} data value(s) to match the field count, " +
                    $"but received {data.Length}.", nameof(data));
            var fields = _fields.Zip(data,
                (field, tensor) => new KeyValuePair<string, IData>(field.Name, tensor));
            return new TensorDataStruct(this, fields);
        }

        /// <summary>
        /// Checks structural equality with another TensorStructDef.
        /// Two TensorStructDef instances are structurally equal if:
        /// 1. They have the same number of fields
        /// 2. Each field at position i has the same Name, Structure, Rank, and ElementType
        /// </summary>
        public bool StructuralEquals(TensorStructDef? other)
        {
            if (other is null) return false;
            if (ReferenceEquals(this, other)) return true;
            if (_fields.Length != other._fields.Length) return false;

            for (int i = 0; i < _fields.Length; i++)
            {
                if (!_fields[i].StructuralEquals(other._fields[i]))
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Computes a structural hash code for this struct definition.
        /// </summary>
        private int ComputeStructuralHash()
        {
            unchecked
            {
                int hash = 17;
                hash = hash * 31 + _fields.Length;
                foreach (var field in _fields)
                {
                    hash = hash * 31 + field.ComputeStructuralHash();
                }
                return hash;
            }
        }

        /// <summary>
        /// Gets the precomputed structural hash code.
        /// </summary>
        public int StructuralHash => _structuralHash;

        // IEquatable<TensorStructDef> implementation for use in dictionaries/sets
        public bool Equals(TensorStructDef? other) => StructuralEquals(other);

        public override bool Equals(object? obj) => obj is TensorStructDef other && StructuralEquals(other);

        public override int GetHashCode() => _structuralHash;

        public override string ToString() => TypeName ?? $"TensorStruct[{_fields.Length} fields]";

        #region Serialization

        /// <summary>
        /// Serializes this TensorStructDef to a JSON string for ONNX metadata storage.
        /// Format: {"typeName":"...", "fields":[{"name":"...", "structure":"...", "rank":..., "elemType":...}, ...]}
        /// </summary>
        public string ToJson()
        {
            var fieldsJson = string.Join(",", _fields.Select(f =>
                $"{{\"name\":\"{EscapeJson(f.Name)}\",\"structure\":\"{f.Structure}\",\"rank\":{(f.Rank.HasValue ? f.Rank.Value.ToString() : "null")},\"elemType\":{f.ElementType.ProtoTypeNum}}}"));
            
            var typeNameJson = TypeName != null ? $"\"{EscapeJson(TypeName)}\"" : "null";
            return $"{{\"typeName\":{typeNameJson},\"fields\":[{fieldsJson}]}}";
        }

        /// <summary>
        /// Deserializes a TensorStructDef from a JSON string stored in ONNX metadata.
        /// Note: Field names should not contain quotes - they are not properly escaped in this simple parser.
        /// </summary>
        public static TensorStructDef FromJson(string json)
        {
            // Simple JSON parsing without external dependencies
            // Expected format: {"typeName":"..." or null, "fields":[...]}
            
            if (string.IsNullOrEmpty(json))
                throw new ArgumentException("JSON string cannot be null or empty", nameof(json));

            // Extract typeName
            string? typeName = null;
            var typeNameMatch = System.Text.RegularExpressions.Regex.Match(json, @"""typeName""\s*:\s*(null|""([^""]*)"")");
            if (typeNameMatch.Success && typeNameMatch.Groups[2].Success)
            {
                typeName = UnescapeJson(typeNameMatch.Groups[2].Value);
            }

            // Extract fields array
            var fieldsMatch = System.Text.RegularExpressions.Regex.Match(json, @"""fields""\s*:\s*\[([^\]]*)\]");
            if (!fieldsMatch.Success)
                throw new ArgumentException("Invalid JSON format: missing 'fields' array", nameof(json));

            var fieldsContent = fieldsMatch.Groups[1].Value;
            var fields = new List<TensorStructFieldDef>();

            // Parse each field object
            // Note: This simple regex doesn't handle escaped quotes in field names - field names should be simple identifiers
            var fieldMatches = System.Text.RegularExpressions.Regex.Matches(fieldsContent, 
                @"\{""name""\s*:\s*""([^""]*)""\s*,\s*""structure""\s*:\s*""([^""]*)""\s*,\s*""rank""\s*:\s*(null|\d+)\s*,\s*""elemType""\s*:\s*(\d+)\}");

            foreach (System.Text.RegularExpressions.Match fieldMatch in fieldMatches)
            {
                var name = UnescapeJson(fieldMatch.Groups[1].Value);
                var structureStr = fieldMatch.Groups[2].Value;
                var rankStr = fieldMatch.Groups[3].Value;
                var elemTypeNum = int.Parse(fieldMatch.Groups[4].Value);

                var structure = Enum.Parse<DataStructure>(structureStr);
                int? rank = rankStr == "null" ? null : int.Parse(rankStr);
                
                // Look up DType from ProtoTypeNum
                DType elemType;
                
                // Check if this is a TensorStruct type by creating a temporary lookup
                // TensorStruct types are in the 2000-2999 range (same check as DType.IsTensorStructType)
                var tempDType = new TempProtoTypeNumHolder(elemTypeNum);
                if (tempDType.IsTensorStructRange)
                {
                    // This is a nested TensorStruct - we need to look it up from the registry
                    // For now, we'll throw - the caller should ensure nested structs are loaded first
                    throw new NotSupportedException("Nested TensorStruct deserialization not yet supported. Load nested structs in dependency order.");
                }
                else
                {
                    elemType = DType.FromProtoTypeNum(elemTypeNum);
                }

                fields.Add(new TensorStructFieldDef(name, structure, rank, elemType));
            }

            return new TensorStructDef(fields, typeName);
        }

        /// <summary>
        /// Helper struct to check if a ProtoTypeNum is in the TensorStruct range without creating a DType.
        /// </summary>
        private readonly struct TempProtoTypeNumHolder
        {
            public int ProtoTypeNum { get; }
            
            /// <summary>
            /// Returns true if ProtoTypeNum is in the TensorStruct range (2000-2999).
            /// This matches the logic in DType.IsTensorStructType.
            /// </summary>
            public bool IsTensorStructRange => ProtoTypeNum >= 2000 && ProtoTypeNum <= 2999;
            
            public TempProtoTypeNumHolder(int protoTypeNum) => ProtoTypeNum = protoTypeNum;
        }

        private static string EscapeJson(string s)
        {
            // Order matters: escape backslashes first, then other escape sequences
            return s.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\n").Replace("\r", "\\r").Replace("\t", "\\t");
        }

        private static string UnescapeJson(string s)
        {
            // Order matters: unescape other sequences first, then double-backslashes last
            return s.Replace("\\t", "\t").Replace("\\r", "\r").Replace("\\n", "\n").Replace("\\\"", "\"").Replace("\\\\", "\\");
        }

        #endregion
    }
}

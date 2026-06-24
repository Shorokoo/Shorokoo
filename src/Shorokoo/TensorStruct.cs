using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Modules;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Onnx;

namespace Shorokoo
{
    /// <summary>
    /// TensorStruct is a mechanism for grouping multiple IVariables together into a single composite IVariable.
    /// This follows the same pattern as Tensor&lt;T&gt;, TensorSequence&lt;T&gt;, and OptionalTensor&lt;T&gt;.
    /// </summary>
    /// <typeparam name="T">The IStruct type that defines the struct fields. Use DTypeStruct for dynamic struct definitions.</typeparam>
    public class TensorStruct<T> : Variable<T>, ITensorStruct where T : IStruct
    {
        private readonly ImmutableDictionary<string, IVariable> _fields;
        private readonly TensorStructDef _definition;

        /// <summary>
        /// Creates a new TensorStruct with the specified fields.
        /// </summary>
        /// <param name="dtype">The DType for this TensorStruct (contains the TensorStructDef)</param>
        /// <param name="owningNode">The node that produces this TensorStruct</param>
        /// <param name="moduleFn">Optional function context</param>
        /// <param name="name">Optional name for this variable</param>
        /// <param name="definition">The definition describing this struct's fields</param>
        /// <param name="fields">Dictionary of field name to field IVariable</param>
        internal TensorStruct(DType dtype, Node owningNode, Function? moduleFn, string? name, 
            TensorStructDef definition, ImmutableDictionary<string, IVariable>? fields = null)
            : base(dtype, owningNode, moduleFn, name)
        {
            _definition = definition ?? throw new ArgumentNullException(nameof(definition));
            _fields = fields ?? ImmutableDictionary<string, IVariable>.Empty;
        }

        /// <summary>
        /// Gets the definition describing the structure of this TensorStruct.
        /// </summary>
        public TensorStructDef Definition => _definition;

        /// <summary>
        /// Gets a field from this TensorStruct by name.
        /// </summary>
        /// <param name="name">The name of the field to retrieve</param>
        /// <returns>The IVariable for the specified field</returns>
        /// <exception cref="KeyNotFoundException">Thrown if the field name does not exist</exception>
        public IVariable GetField(string name)
        {
            if (_fields.TryGetValue(name, out var field))
                return field;
            
            throw new KeyNotFoundException($"Field '{name}' not found in TensorStruct. Available fields: {string.Join(", ", _fields.Keys)}");
        }

        /// <summary>
        /// Gets a field from this TensorStruct by name with specific type.
        /// </summary>
        /// <typeparam name="TField">The expected type of the field</typeparam>
        /// <param name="name">The name of the field to retrieve</param>
        /// <returns>The IVariable for the specified field, cast to the expected type</returns>
        public TField GetField<TField>(string name) where TField : IVariable
        {
            var field = GetField(name);
            if (field is TField typedField)
                return typedField;
            
            throw new InvalidCastException($"Field '{name}' is of type {field.GetType().Name}, not {typeof(TField).Name}");
        }

        /// <summary>
        /// Tries to get a field from this TensorStruct by name.
        /// </summary>
        /// <param name="name">The name of the field to retrieve</param>
        /// <param name="field">The field if found, otherwise null</param>
        /// <returns>True if the field was found, false otherwise</returns>
        public bool TryGetField(string name, out IVariable? field)
        {
            return _fields.TryGetValue(name, out field);
        }

        /// <summary>
        /// Gets all field names in this TensorStruct.
        /// </summary>
        public IEnumerable<string> FieldNames => _fields.Keys;

        /// <summary>
        /// Gets all fields as key-value pairs.
        /// </summary>
        public IEnumerable<KeyValuePair<string, IVariable>> AllFields => _fields;

        /// <summary>
        /// Creates a new TensorStruct with the same definition but updated fields.
        /// </summary>
        internal TensorStruct<T> WithFields(ImmutableDictionary<string, IVariable> newFields)
        {
            return new TensorStruct<T>(this.Type, this.OwningNode, this.ModuleFn, this.UniqueName, _definition, newFields);
        }

        public override string ToString()
        {
            var typeName = _definition.TypeName ?? "DTypeStruct";
            return $"TensorStruct<{typeName}>[{_fields.Count} fields]";
        }
    }
}

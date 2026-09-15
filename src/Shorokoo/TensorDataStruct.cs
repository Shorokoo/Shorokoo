using Microsoft.CodeAnalysis.CSharp.Syntax;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Onnx;

namespace Shorokoo
{
    public class TensorDataStruct : IData, IReadOnlyList<IData>
    {
        /// <summary>
        /// Gets the definition describing the structure of this TensorStruct.
        /// </summary>
        public TensorStructDef Definition { get; private set; }

        public DType DType => Shorokoo.DType.GetOrCreateForTensorStruct(Definition);

        public ImmutableDictionary<string, IData> Fields { get; private set; }

        public int Count => Fields.Count;

        public IData this[int index] => Fields[Definition.Fields[index].Name];

        /// <summary>
        /// Creates a new TensorStructData with the specified definition and field data.
        /// </summary>
        /// <param name="definition">The struct definition describing the fields</param>
        /// <param name="fields">Dictionary of field name to TensorData</param>
        public TensorDataStruct(TensorStructDef definition, IEnumerable<KeyValuePair<string, IData>> fields)
        {
            Definition = definition ?? throw new ArgumentNullException(nameof(definition));
            Fields = ImmutableDictionary.CreateRange(fields);

            // Every definition field must be present, and present as the kind the definition declares.
            // Nothing downstream can recover from a value that contradicts its own definition, and the
            // consumers do not even agree on what to do with one: the checkpoint writer refuses a
            // non-tensor field by name, while weight binding drops it and fails two layers down on a
            // lookup for a parameter that is no longer there. The definition is the contract; this is
            // the one place that holds a value to it.
            foreach (var fieldDef in definition.Fields)
            {
                if (!Fields.TryGetValue(fieldDef.Name, out var value))
                    throw new ArgumentException($"Missing data for field '{fieldDef.Name}'", nameof(fields));
                var actual = StructureOf(value);
                if (actual != fieldDef.Structure)
                    throw new ArgumentException(
                        $"Field '{fieldDef.Name}' is declared {fieldDef.Structure} by this struct's "
                        + $"definition, but the value given for it is a {actual}.", nameof(fields));
            }
        }

        /// <summary>The structural kind of a value, as a definition declares kinds.</summary>
        private static DataStructure StructureOf(IData value) => value switch
        {
            TensorDataStruct => DataStructure.TensorStruct,
            TensorDataSequence => DataStructure.Sequence,
            OptionalTensorData => DataStructure.Optional,
            TensorData => DataStructure.Tensor,
            _ => throw new ArgumentException(
                $"Unsupported field value type '{value?.GetType().Name ?? "null"}'.", nameof(value)),
        };

        public override string ToString()
        {
            var fieldNames = string.Join(", ", Definition.Fields.Select(x => x.Name));
            return $"TensorStructData[{Definition.TypeName ?? "anonymous"}]({fieldNames})";
        }

        public ImmutableArray<IData<T>> FlattenedFieldsOfType<T>() where T : IVarType
        {
            return [..this.SelectMany(field =>  field is TensorDataStruct fieldStruct ? [.. fieldStruct.FlattenedFieldsOfType<T>()] :
                                                field is IData<T> tensorData ? [tensorData] : 
                                                ImmutableArray<IData<T>>.Empty)];
        }

        public IEnumerator<IData> GetEnumerator()
        {
            return this.Definition.Fields.Select(x => (IData)Fields[x.Name]).GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}

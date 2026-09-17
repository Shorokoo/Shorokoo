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

        /// <summary>The number of fields the definition declares — the same fields the indexer and the
        /// enumerator walk. A key in <see cref="Fields"/> beyond the definition is not a field of this
        /// struct, and counting it made this disagree with both.</summary>
        public int Count => Definition.Fields.Length;

        public IData this[int index] => Fields[Definition.Fields[index].Name];

        /// <summary>
        /// Creates a new TensorStructData with the specified definition and field data. Every field the
        /// definition declares must be present, and must be the structural kind it is declared as — a
        /// field declared <c>Tensor</c> takes a <see cref="TensorData"/>, one declared
        /// <c>TensorStruct</c> takes a <see cref="TensorDataStruct"/>, and so on. A value that
        /// contradicts its definition throws <see cref="ArgumentException"/>.
        /// </summary>
        /// <param name="definition">The struct definition describing the fields</param>
        /// <param name="fields">Field name to value, one per definition field, each of the kind that
        /// field declares — a plain tensor also serves for a field declared <c>Optional</c>, meaning
        /// present. A key the definition does not declare is not a field of this struct: it is kept in
        /// <see cref="Fields"/> but ignored by everything that reads the struct, including
        /// <see cref="Count"/>, the indexer and the enumerator.</param>
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
                var actual = StructureOf(value, fieldDef.Name);
                // A plain tensor for a field declared Optional is the PRESENT case, and is accepted:
                // the runtime hands a struct's field values straight through, and ONNX Runtime takes a
                // tensor where an optional input is expected (see NamedModelParam.ToTensorValue). A
                // caller holding the tensor writes exactly that, so refusing it would break a working
                // call for a distinction the layer below does not draw.
                var fits = actual == fieldDef.Structure
                    || (fieldDef.Structure == DataStructure.Optional && actual == DataStructure.Tensor);
                if (!fits)
                    throw new ArgumentException(
                        $"Field '{fieldDef.Name}' is declared {fieldDef.Structure} by this struct's "
                        + $"definition, but the value given for it is a {actual}.", nameof(fields));
            }
        }

        /// <summary>The structural kind of a value, as a definition declares kinds.</summary>
        private static DataStructure StructureOf(IData value, string fieldName) => value switch
        {
            TensorDataStruct => DataStructure.TensorStruct,
            TensorDataSequence => DataStructure.Sequence,
            OptionalTensorData => DataStructure.Optional,
            TensorData => DataStructure.Tensor,
            // "fields" is the constructor's parameter; this method is only ever reached from there.
            _ => throw new ArgumentException(
                $"Field '{fieldName}' has an unsupported value type "
                + $"'{value?.GetType().Name ?? "null"}'.", "fields"),
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

        /// <summary>
        /// The compute context this struct's fields belong to, or null for the framework's own
        /// host memory. Set by the transfer operations below.
        /// </summary>
        public Shorokoo.Runtime.ComputeContext? Context { get; private set; }

        /// <summary>Moves this struct's owned tensors to <paramref name="target"/>, recursing
        /// through nested structs and sequences. A field it only has access to is left where it
        /// is, since moving what you do not own is what the non-owning case forbids.</summary>
        public TensorDataStruct TransferTo(Shorokoo.Runtime.ComputeContext? target)
            => Rebuild(target, static (d, c) => Move(d, c, static (t, x) => t.TransferTo(x),
                static (q, x) => q.TransferTo(x), static (u, x) => u.TransferTo(x)));

        /// <summary>Copies this struct's owned tensors into <paramref name="target"/>'s memory,
        /// leaving this struct untouched.</summary>
        public TensorDataStruct CopyTo(Shorokoo.Runtime.ComputeContext? target)
            => Rebuild(target, static (d, c) => Move(d, c, static (t, x) => t.CopyTo(x),
                static (q, x) => q.CopyTo(x), static (u, x) => u.CopyTo(x)));

        /// <summary>Hands <paramref name="target"/> readers for this struct's owned tensors,
        /// taking no ownership of any of them.</summary>
        public TensorDataStruct GiveAccessTo(Shorokoo.Runtime.ComputeContext? target)
            => Rebuild(target, static (d, c) => Move(d, c, static (t, x) => t.GiveAccessTo(x),
                static (q, x) => q.GiveAccessTo(x), static (u, x) => u.GiveAccessTo(x)));

        private TensorDataStruct Rebuild(
            Shorokoo.Runtime.ComputeContext? target,
            Func<IData, Shorokoo.Runtime.ComputeContext?, IData> operation)
        {
            var moved = Fields.Select(f =>
                new KeyValuePair<string, IData>(f.Key, operation(f.Value, target)));
            return new TensorDataStruct(Definition, moved) { Context = target };
        }

        /// <summary>Applies the right one of three operations to whichever kind of field this is,
        /// and leaves anything else -- a tensor that owns nothing included -- alone.</summary>
        private static IData Move(
            IData field, Shorokoo.Runtime.ComputeContext? target,
            Func<TensorData, Shorokoo.Runtime.ComputeContext?, TensorData> onTensor,
            Func<TensorDataSequence, Shorokoo.Runtime.ComputeContext?, TensorDataSequence> onSequence,
            Func<TensorDataStruct, Shorokoo.Runtime.ComputeContext?, TensorDataStruct> onStruct)
            => field switch
            {
                TensorData t => t.OwnsMemory ? onTensor(t, target) : t,
                TensorDataSequence q => onSequence(q, target),
                TensorDataStruct u => onStruct(u, target),
                _ => field,
            };
    }
}

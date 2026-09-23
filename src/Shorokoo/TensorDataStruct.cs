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
        /// This struct where <paramref name="target"/> can use it, field by field under each
        /// field's own <c>To</c>: a tensor the target's backend can read as it stands is handed over
        /// as the same object and attached to the target, anything else is copied into the target's
        /// memory. The very same struct when nothing had to be copied; this struct is untouched
        /// either way.
        /// </summary>
        /// <exception cref="ArgumentNullException"><paramref name="target"/> is null.</exception>
        public TensorDataStruct To(Shorokoo.Runtime.ComputeContext target)
        {
            ArgumentNullException.ThrowIfNull(target);
            return Rebuild(field => Apply(field,
                t => t.To(target), q => q.To(target), u => u.To(target)));
        }

        /// <summary>An independent copy of this struct, every tensor in it copied into
        /// <paramref name="target"/>'s memory and attached to it. This struct is untouched.</summary>
        /// <exception cref="ArgumentNullException"><paramref name="target"/> is null.</exception>
        public TensorDataStruct CopyTo(Shorokoo.Runtime.ComputeContext target)
        {
            ArgumentNullException.ThrowIfNull(target);
            return Rebuild(field => Apply(field,
                t => t.CopyTo(target), q => q.CopyTo(target), u => u.CopyTo(target)));
        }

        /// <summary>This struct where the host can read it, field by field under each field's own
        /// <c>ToHost</c>. The very same struct when every field already is host-readable.</summary>
        public TensorDataStruct ToHost()
            => Rebuild(field => Apply(field,
                static t => t.ToHost(), static q => q.ToHost(), static u => u.ToHost()));

        /// <summary>
        /// A struct of what <paramref name="operation"/> makes of each field — or this very struct,
        /// if it made nothing new. What it did make is released if a later field fails: the struct
        /// that would have held it is never constructed. The sources are never touched, so there is
        /// nothing of theirs to put back.
        /// </summary>
        private TensorDataStruct Rebuild(Func<IData, IData> operation)
        {
            // Materialized as it goes rather than left lazy, so a field that throws can be caught
            // here at all: Select would defer every operation into the constructor, past any
            // cleanup this method could do.
            List<KeyValuePair<string, IData>> rebuilt = new(Fields.Count);
            var changed = false;
            try
            {
                foreach (var field in Fields)
                {
                    var result = operation(field.Value);
                    changed |= !ReferenceEquals(result, field.Value);
                    rebuilt.Add(new KeyValuePair<string, IData>(field.Key, result));
                }
            }
            catch
            {
                foreach (var (key, result) in rebuilt) ReleaseNew(result, Fields[key]);
                throw;
            }
            return changed ? new TensorDataStruct(Definition, rebuilt) : this;
        }

        /// <summary>
        /// Releases whatever of <paramref name="result"/> is not <paramref name="source"/>'s own —
        /// the copies a failed rebuild made. A field handed over as the same object is the source's,
        /// and is left alone.
        /// </summary>
        private static void ReleaseNew(IData result, IData source)
        {
            if (ReferenceEquals(result, source)) return;
            switch (result)
            {
                case TensorData t:
                    t.TryDelete();
                    break;
                case TensorDataSequence q:
                    q.Dispose();
                    break;
                case TensorDataStruct u when source is TensorDataStruct original:
                    foreach (var (key, value) in u.Fields)
                        if (original.Fields.TryGetValue(key, out var was)) ReleaseNew(value, was);
                    break;
                case OptionalTensorData { Value: { } inner }
                    when source is OptionalTensorData { Value: var was } && !ReferenceEquals(inner, was):
                    inner.TryDelete();
                    break;
            }
        }

        /// <summary>Applies the right one of three operations to whichever kind of field this is,
        /// and leaves anything else alone.</summary>
        /// <param name="field">The field to rebuild.</param>
        /// <param name="onTensor">What to do with a tensor field.</param>
        /// <param name="onSequence">What to do with a sequence field.</param>
        /// <param name="onStruct">What to do with a nested struct field.</param>
        private static IData Apply(
            IData field,
            Func<TensorData, TensorData> onTensor,
            Func<TensorDataSequence, TensorDataSequence> onSequence,
            Func<TensorDataStruct, TensorDataStruct> onStruct)
            => field switch
            {
                TensorData t => onTensor(t),
                TensorDataSequence q => onSequence(q),
                TensorDataStruct u => onStruct(u),
                // A present optional holds a tensor like any other field does, and is rebuilt only
                // where its tensor was: an optional whose tensor came back as itself is the same
                // optional.
                OptionalTensorData { HasValue: true, Value: { } v }
                    => onTensor(v) is var r && ReferenceEquals(r, v) ? field : OptionalTensorData.Some(r),
                _ => field,
            };
    }
}

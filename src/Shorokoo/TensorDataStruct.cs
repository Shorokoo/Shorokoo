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
        /// The mode each field given through <c>.Shared()</c> or <c>.TryConsume()</c> was given in,
        /// by name; a field given as it is has none. What <see cref="FieldFeedMode"/> reads.
        /// </summary>
        private readonly ImmutableDictionary<string, SharedInputMode> _fieldModes;

        /// <summary>
        /// Creates a new TensorStructData with the specified definition and field data. Every field the
        /// definition declares must be present, and must be the structural kind it is declared as — a
        /// field declared <c>Tensor</c> takes a <see cref="TensorData"/>, one declared
        /// <c>TensorStruct</c> takes a <see cref="TensorDataStruct"/>, and so on. A value that
        /// contradicts its definition throws <see cref="ArgumentException"/>.
        ///
        /// <para>A field may be given through <c>.Shared()</c> or <c>.TryConsume()</c> — a
        /// <see cref="SharedInput"/> over a value of the kind it declares — to be fed that way
        /// whatever the struct is fed as: <c>rig.InputDef.FromOrderedData(tokens, mask.Shared())</c>
        /// keeps the mask while a step consumes the tokens. The struct holds the value itself, so
        /// <see cref="Fields"/>, the indexer and the enumerator read the tensor, not its wrapper; the
        /// mode is kept beside it, and <see cref="To"/>, <see cref="CopyTo"/> and
        /// <see cref="ToHost"/> carry it to the struct they build. Where the struct is fed to a run a
        /// field's mode combines with the struct's own: a field given <c>.Shared()</c> is read even
        /// when the struct is fed as it is, a struct fed <c>.Shared()</c> has every field read, and
        /// otherwise a field is fed as it was given, or as the struct is fed where it was given as it
        /// is.</para>
        /// </summary>
        /// <param name="definition">The struct definition describing the fields</param>
        /// <param name="fields">Field name to value, one per definition field, each of the kind that
        /// field declares — a plain tensor also serves for a field declared <c>Optional</c>, meaning
        /// present — as it is or through <c>.Shared()</c> or <c>.TryConsume()</c>. A key the
        /// definition does not declare is not a field of this struct: it is kept in
        /// <see cref="Fields"/> but ignored by everything that reads the struct, including
        /// <see cref="Count"/>, the indexer and the enumerator.</param>
        public TensorDataStruct(TensorStructDef definition, IEnumerable<KeyValuePair<string, IData>> fields)
            : this(definition, fields, ImmutableDictionary<string, SharedInputMode>.Empty)
        {
        }

        /// <summary>The public constructor, over fields some of which were given their modes
        /// already — <paramref name="modes"/>, by name — by a struct this one is rebuilt from.</summary>
        private TensorDataStruct(
            TensorStructDef definition, IEnumerable<KeyValuePair<string, IData>> fields,
            ImmutableDictionary<string, SharedInputMode> modes)
        {
            Definition = definition ?? throw new ArgumentNullException(nameof(definition));
            ArgumentNullException.ThrowIfNull(fields);
            var values = ImmutableDictionary.CreateBuilder<string, IData>();
            foreach (var (name, value) in fields)
            {
                // Unwrapped here, and nowhere else: everything that reads Fields sees the value.
                if (value is SharedInput shared) modes = modes.SetItem(name, shared.Mode);
                values.Add(name, value is SharedInput { Value: var held } ? held : value);
            }
            Fields = values.ToImmutable();
            _fieldModes = modes;

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

        /// <summary>
        /// This struct to be <b>read</b> by the run it is fed to — every member of it — rather than
        /// consumed. Fed as it is, a struct's members are consumed as a bare tensor is; see
        /// <see cref="TensorData.Shared"/>.
        /// </summary>
        public SharedInput Shared() => new(this, SharedInputMode.Shared);

        /// <summary>
        /// This struct's members to be consumed by the run it is fed to where nothing else is
        /// reading them when that run starts, and read otherwise — decided member by member; see
        /// <see cref="TensorData.TryConsume"/>.
        /// </summary>
        public SharedInput TryConsume() => new(this, SharedInputMode.TryConsume);

        /// <summary>
        /// How a run fed this struct as <paramref name="structMode"/> says (null: as it is) is fed
        /// field <paramref name="name"/>: read where the field was given <c>.Shared()</c> or the
        /// struct is fed that way, and otherwise as the field was given, or as the struct is fed
        /// where the field was given as it is. Null is as it is, consumed.
        ///
        /// <para>Shared wins either way because it is the one mode a caller asks for to keep
        /// something: a mask built <c>.Shared()</c> into a batch the step consumes, or a checkpoint
        /// fed <c>.Shared()</c> whose struct had one field built <c>.TryConsume()</c>, would
        /// otherwise lose what they were built to keep.</para>
        /// </summary>
        internal SharedInputMode? FieldFeedMode(string name, SharedInputMode? structMode)
        {
            SharedInputMode? given = _fieldModes.TryGetValue(name, out var mode) ? mode : null;
            return given == SharedInputMode.Shared || structMode == SharedInputMode.Shared
                ? SharedInputMode.Shared
                : given ?? structMode;
        }

        /// <summary>This struct's values under <paramref name="definition"/>, each field fed as it
        /// was given — for a caller whose definition the same fields are to be read against, in its
        /// order.</summary>
        internal TensorDataStruct WithDefinition(TensorStructDef definition) => new(definition, Fields, _fieldModes);

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
        /// <exception cref="InvalidOperationException"><paramref name="target"/>'s device-memory
        /// budget cannot take what the struct's fields would put on it. Nothing is placed.</exception>
        public TensorDataStruct To(Shorokoo.Runtime.ComputeContext target)
        {
            ArgumentNullException.ThrowIfNull(target);
            return target.PlaceAll(BytesPlacedOnto(target, copying: false), () => $"To(context) of {this}",
                () => Rebuild(field => Apply(field, t => t.To(target), q => q.To(target), u => u.To(target))));
        }

        /// <summary>An independent copy of this struct, every tensor in it copied into
        /// <paramref name="target"/>'s memory and attached to it. This struct is untouched.</summary>
        /// <exception cref="ArgumentNullException"><paramref name="target"/> is null.</exception>
        /// <exception cref="InvalidOperationException"><paramref name="target"/>'s device-memory
        /// budget cannot take the copies. Nothing is copied.</exception>
        public TensorDataStruct CopyTo(Shorokoo.Runtime.ComputeContext target)
        {
            ArgumentNullException.ThrowIfNull(target);
            return target.PlaceAll(BytesPlacedOnto(target, copying: true), () => $"CopyTo(context) of {this}",
                () => Rebuild(field => Apply(field, t => t.CopyTo(target), q => q.CopyTo(target), u => u.CopyTo(target))));
        }

        /// <summary>
        /// The bytes putting this struct's fields on <paramref name="target"/> adds to what the
        /// target's device-memory budget counts, field by field as each field's own <c>To</c> or
        /// <c>CopyTo</c> would place it, a tensor handed over as it stands counted once however many
        /// fields hold it — or null where a field cannot tell without making its copy.
        /// </summary>
        internal long? BytesPlacedOnto(
            Shorokoo.Runtime.ComputeContext target, bool copying, HashSet<TensorData>? handedOver = null)
        {
            handedOver ??= new HashSet<TensorData>(ReferenceEqualityComparer.Instance);
            long bytes = 0;
            foreach (var field in Fields.Values)
            {
                long? adding = field switch
                {
                    TensorData t => t.BytesPlacedOnto(target, copying, handedOver),
                    OptionalTensorData { HasValue: true, Value: { } present } => present.BytesPlacedOnto(target, copying, handedOver),
                    TensorDataSequence q => q.BytesPlacedOnto(target, copying, handedOver),
                    TensorDataStruct u => u.BytesPlacedOnto(target, copying, handedOver),
                    _ => 0,
                };
                if (adding is not { } known) return null;
                bytes += known;
            }
            return bytes;
        }

        /// <summary>This struct where the host can read it, field by field under each field's own
        /// <c>ToHost</c>. The very same struct when every field already is host-readable.</summary>
        public TensorDataStruct ToHost()
            => Rebuild(field => Apply(field,
                static t => t.ToHost(), static q => q.ToHost(), static u => u.ToHost()));

        /// <summary>
        /// A struct of what <paramref name="operation"/> makes of each field — or this very struct,
        /// if it made nothing new — each fed as the field it was made from was given. What it did
        /// make is released if a later field fails: the struct that would have held it is never
        /// constructed. The sources are never touched, so there is nothing of theirs to put back.
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
                // Every one of them, whatever releasing one does; the failure that got here is what
                // the caller hears, not a release failing on the way out.
                foreach (var (key, result) in rebuilt)
                {
                    try
                    {
                        ReleaseNew(result, Fields[key]);
                    }
                    catch (Exception)
                    {
                        // Released as far as it would go.
                    }
                }
                throw;
            }
            return changed ? new TensorDataStruct(Definition, rebuilt, _fieldModes) : this;
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

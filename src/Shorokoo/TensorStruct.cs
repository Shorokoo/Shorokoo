using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Modules;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;

namespace Shorokoo
{

    /// <summary>
    /// Value-type handle for a TensorStruct. The original <c>TensorStruct&lt;T&gt;</c> name now denotes
    /// this <see langword="struct"/>; the reference type was renamed <see cref="Variable"/>.
    /// This struct carries the full user-facing surface. It holds the immutable directly in a field
    /// (value-copy semantics for the Module DSL). This pass only makes mutation possible — behaviour
    /// is unchanged (de-facto immutable).
    /// <para>
    /// A defaulted handle (<c>default</c>, <c>inner == null</c>) materialises the established default on
    /// first use: default-filled fields when <typeparamref name="T"/>'s layout is recoverable, otherwise a
    /// zero-filled struct (via <c>InternalGlobals.DefaultVariable</c>).
    /// </para>
    /// </summary>
    public struct TensorStruct<T> : ITensorStruct where T : IStruct
    {
        private Variable? inner;

        Variable IValue.ToVariable() => Immutable;

        /// <summary>The backing graph node, materialising the established default (default-filled fields, or a
        /// zero-filled struct when the layout is unknown) for a defaulted handle.</summary>
        internal readonly Variable Immutable => inner ?? InternalGlobals.DefaultVariable(typeof(TensorStruct<T>));

        public static implicit operator TensorStruct<T>(Variable imm)
        {
            // A struct's dtype is its field layout, not T, so only the structural kind is checked here.
            IValue.RequireKind(imm, DataStructure.TensorStruct);
            return new TensorStruct<T> { inner = imm };
        }
        public static implicit operator Variable(TensorStruct<T> handle)
            => handle.Immutable;

        // ── User-facing API (the struct surface lives here, not on the immutable) ──
        public TensorStructDef Definition => Immutable.Def;

        public Variable GetField(string name) => Immutable.Field(name);

        public TField GetField<TField>(string name) where TField : IValue
            => Immutable.Field(name).ToValue<TField>();

        public bool TryGetField(string name, out Variable? field)
        {
            if (inner is null) { field = null; return false; }
            return inner.Fields.TryGetValue(name, out field);
        }

        public IEnumerable<string> FieldNames => inner?.Fields.Keys ?? [];

        public IEnumerable<KeyValuePair<string, Variable>> AllFields => inner?.Fields ?? [];

        internal TensorStruct<T> WithFields(ImmutableDictionary<string, Variable> newFields) => Immutable.WithFields(newFields);

        public override readonly string ToString() => Immutable.ToString();

        // ITensorStruct explicit members.
        TensorStructDef ITensorStruct.Definition => Immutable.Def;
        Variable ITensorStruct.GetField(string name) => Immutable.Field(name);

        // IValue surface — forward to the backing Variable.
        public Node OwningNode => Immutable.OwningNode;
        public DType Type => Immutable.Type;
        public Function? ModuleFn => Immutable.ModuleFn;
        public TensorKey Key => Immutable.Key;
        public string UniqueName => Immutable.UniqueName;
        public bool IsValid { get => Immutable.IsValid; set => Immutable.IsValid = value; }

#pragma warning disable CS0618 // forwarding the obsolete member is intentional
        string? IValue.FriendlyName => ((IValue)Immutable).FriendlyName;
#pragma warning restore CS0618
    }
}

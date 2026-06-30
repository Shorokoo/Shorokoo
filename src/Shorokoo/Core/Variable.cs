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
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection;
using System.Reflection.Metadata;
using System.Runtime.CompilerServices;
using static Shorokoo.Core.Nodes.Ops;
using static Shorokoo.Core.Nodes.AutoDiff.Ops;
using static RandN.Distributions.Uniform;
using Shorokoo.Core.Nodes.NodeDefinitions;
using System.Collections.Immutable;
using System.Linq;

namespace Shorokoo.Core
{
    /// <summary>
    /// Non-generic immutable graph-value node — the object the Node/Variable graph is built from.
    /// Holds only runtime node state: the owning <see cref="Node"/>, the runtime element
    /// <see cref="DType"/>, the <see cref="TensorKey"/>, the producing module function, validity,
    /// and the serialization name. The element type lives in <see cref="Type"/> at runtime, NOT in
    /// a C# type parameter, and the structural kind (tensor / sequence / optional / struct) lives on
    /// the producing node — so a node needs neither generics nor a per-kind subclass.
    /// <para>
    /// <see cref="Variable"/> is the graph's internal currency and deliberately does NOT implement
    /// <see cref="IValue"/>: a <c>Variable</c> is unambiguously a graph-side node, whereas an
    /// <see cref="IValue"/> is a user-side value-struct handle (such as <see cref="Tensor{T}"/>).
    /// An <see cref="IValue"/> and a <see cref="Variable"/> convert to each other through the handle's
    /// implicit operators, with reflective fallbacks — <see cref="ToValue{A}()"/> and
    /// <see cref="ToValue(System.Type)"/> — for converting between the two when the target type is known
    /// only at runtime rather than statically.
    /// </para>
    /// </summary>
    public class Variable
    {
        public Node OwningNode { get; private set; }
        public Node ParentNode => this.OwningNode;

        public DType Type { get; private set; }
        public DType DType => this.Type;

        public Function? ModuleFn { get; private set; }

        public bool IsValid { get; set; } = true;

        /// <summary>The structural kind of this graph value (tensor / optional / sequence / struct).
        /// Tensor/vector/scalar all share <see cref="DataStructure.Tensor"/> and are distinguished by
        /// <see cref="Rank"/>.</summary>
        public DataStructure Kind { get; }

        /// <summary>Statically known rank (number of dimensions), or null when not known at
        /// graph-construction time. Only meaningful for <see cref="DataStructure.Tensor"/> values.</summary>
        public int? Rank { get; }

        /// <summary>
        /// A globally unique identifier for this tensor, composed of the parent node's key and the output index.
        /// Set by the Node constructor after creating outputs.
        /// </summary>
        public TensorKey Key { get; private set; }

        private string? uniqueName;
        private readonly Func<Vector<int64>>? shapeInferer;
        private Vector<int64>? infShapeTensor;
        private readonly TensorStructDef? structDef;
        private readonly ImmutableDictionary<string, Variable> fields;

        internal Variable(DType type, Node owningNode, Function? moduleFn, string? name,
            DataStructure kind = DataStructure.Tensor, int? rank = null,
            Func<Vector<int64>>? shapeFn = null,
            TensorStructDef? structDef = null,
            ImmutableDictionary<string, Variable>? fields = null)
        {
            this.OwningNode = owningNode;
            this.Type = type;
            this.uniqueName = name;
            this.ModuleFn = moduleFn;
            this.Kind = kind;
            this.Rank = rank;
            this.shapeInferer = shapeFn;
            this.structDef = structDef;
            this.fields = fields ?? ImmutableDictionary<string, Variable>.Empty;
        }

        /// <summary>
        /// Sets the TensorKey for this variable. Called by the Node constructor after creating outputs.
        /// </summary>
        internal void SetKey(TensorKey key)
        {
            this.Key = key;
        }

        /// <summary>
        /// Override this variable's UniqueName. Used by
        /// <see cref="Shorokoo.Graph.FastComputationGraphConverter.ToComputationGraph"/>
        /// to restore original graph-input/output names after a Fast↔CG roundtrip.
        /// </summary>
        internal void SetUniqueName(string? name)
        {
            this.uniqueName = name;
        }

        /// <summary>
        /// The unique name for this tensor. Defaults to Key.ToString() but can be set to human-readable
        /// names like "N1_T0" by processors during construction. Used for ONNX serialization.
        /// </summary>
        public string UniqueName => this.uniqueName ?? this.Key.ToString();

        /// <summary>
        /// Obsolete: Use UniqueName instead. DefaultName now redirects to UniqueName for backwards compatibility.
        /// </summary>
        [Obsolete("Use UniqueName instead. DefaultName is deprecated and will be removed in a future version.")]
        public string DefaultName => this.UniqueName;

        /// <summary>
        /// Deprecated: FriendlyName is no longer used. Use UniqueName for ONNX names or Key for stable identifiers.
        /// </summary>
        [Obsolete("FriendlyName is deprecated. Use UniqueName for ONNX names or Key.ToString() for stable identifiers.")]
        public string? FriendlyName => this.uniqueName;

        /// <summary>
        /// Converts this <c>Variable</c> to the <see cref="IValue"/> type <typeparamref name="A"/> — the
        /// generic form of the <c>Variable</c>→<c>IValue</c> conversion, for call sites where the target
        /// <see cref="IValue"/> type is only known as a type parameter (so the compiler cannot apply the
        /// operator). Routes through the validating <c>op_Implicit</c>, so structure / dtype / rank are
        /// checked exactly as a direct cast would be.
        /// </summary>
        // A is a type parameter, so the compiler can't apply the Variable→IValue operator; ToValue(Type)
        // finds and invokes it at runtime. The result's runtime type is A, so the (A) cast just unboxes it.
        public A ToValue<A>() where A : IValue => (A)ToValue(typeof(A));

        /// <summary>
        /// Convert this <c>Variable</c> to its <em>natural</em> user-facing <see cref="IValue"/> — the
        /// inverse of <see cref="IValue.ToVariable"/> — chosen from the value's own structure, dtype and
        /// rank: a tensor becomes <see cref="Scalar{T}"/> (rank 0), <see cref="Vector{T}"/> (rank 1) or
        /// <see cref="Tensor{T}"/>; an optional / sequence / struct becomes <see cref="OptionalTensor{T}"/>
        /// / <see cref="TensorSequence{T}"/> / <see cref="TensorStruct{T}"/>.
        /// </summary>
        public IValue ToValue() => ToValue(this.Rank);

        /// <summary>
        /// As <see cref="ToValue()"/>, but selects the tensor handle (<see cref="Scalar{T}"/> = 0,
        /// <see cref="Vector{T}"/> = 1, <see cref="Tensor{T}"/> = otherwise) from the supplied
        /// <paramref name="rank"/> rather than this value's own — for when the caller knows the rank the
        /// handle should present (e.g. a rank-0 value wanted as a general <c>Tensor&lt;T&gt;</c> via
        /// <c>rank: null</c>). Ignored for optional / sequence / struct values, whose handle is fixed by
        /// their structural kind.
        /// </summary>
        public IValue ToValue(int? rank) => ToValue(NaturalHandleType(rank));

        /// <summary>
        /// Convert this <c>Variable</c> to a general <see cref="Tensor{T}"/> (rank-agnostic) over its own
        /// element dtype — unlike <see cref="ToValue()"/>, which narrows a rank-0 / rank-1 tensor to
        /// <see cref="Scalar{T}"/> / <see cref="Vector{T}"/>. Only valid for tensor-structured values.
        /// </summary>
        public ITensor ToTensor() => (ITensor)ToValue(typeof(Tensor<>).MakeGenericType(this.Type.ToIVarType()));

        /// <summary>
        /// Convert this <c>Variable</c> to the value-struct <see cref="IValue"/> of the given <paramref name="type"/> — e.g.
        /// <c>Tensor&lt;float32&gt;</c>, <c>Scalar&lt;int64&gt;</c>, or a nullable handle such as
        /// <c>Tensor&lt;float32&gt;?</c>. The handle's implicit <c>operator(Variable)</c> is found and
        /// invoked by reflection — the compiler can't apply it when the target is only a runtime
        /// <see cref="System.Type"/> — and it validates structure / dtype / rank exactly as a direct cast.
        /// </summary>
        public IValue ToValue(Type type)
        {
            // The IValue may be declared nullable (Tensor<T>?); the value-struct is the underlying type.
            var handleType = Nullable.GetUnderlyingType(type) ?? type;
            var conv = MatchingConverter(handleType, this.GetType())
                ?? throw new InvalidTensorOperationException(ErrorCodes.CR001, "ToValue", handleType.Name,
                    $"no implicit Variable conversion to handle '{handleType.Name}'");
            return (IValue)conv.Invoke(null, [this])!;
        }

        // The handle type that mirrors this value's structure / dtype, with the tensor handle chosen by
        // the given rank (for a Tensor-kind value).
        private Type NaturalHandleType(int? rank)
        {
            if (this.Kind == DataStructure.TensorStruct)
                // A struct's element type is its field layout, not an IVarType; the handle's T is phantom
                // (the implicit operator only checks the structural kind), so a generic carrier suffices.
                return typeof(TensorStruct<>).MakeGenericType(typeof(IStruct));

            var elem = this.Type.ToIVarType();
            return this.Kind switch
            {
                DataStructure.Optional => typeof(OptionalTensor<>).MakeGenericType(elem),
                DataStructure.Sequence => typeof(TensorSequence<>).MakeGenericType(elem),
                _ => rank switch                               // DataStructure.Tensor
                {
                    0 => typeof(Scalar<>).MakeGenericType(elem),
                    1 => typeof(Vector<>).MakeGenericType(elem),
                    _ => typeof(Tensor<>).MakeGenericType(elem),
                },
            };
        }

        // ── Reflective Variable→IValue conversion (relocated from the former VariableHandle) ──
        // Each handle's implicit operator(Variable) is found and invoked by reflection to convert a
        // Variable to an IValue for the cases the compiler can't resolve statically: a generic ToValue<A> or
        // a runtime-built IValue Type (ToValue).
        private static readonly ConcurrentDictionary<Type, MethodInfo[]> implicitCasts = new();

        // The handle type's implicit operator(Variable), if one exists, for a value of valueType.
        internal static MethodInfo? MatchingConverter(Type handleType, Type valueType)
        {
            var candidates = implicitCasts.GetOrAdd(handleType, FindImplicitCasts);
            foreach (var m in candidates)
            {
                if (m.GetParameters()[0].ParameterType.IsAssignableFrom(valueType))
                    return m;
            }
            return null;
        }

        // Implicit conversion operators that PRODUCE the handle type from a Variable value.
        private static MethodInfo[] FindImplicitCasts(Type handleType)
        {
            var result = new List<MethodInfo>();
            foreach (var m in handleType.GetMethods(BindingFlags.Public | BindingFlags.Static))
            {
                if (m.Name != "op_Implicit" || m.ReturnType != handleType)
                    continue;
                var p = m.GetParameters();
                if (p.Length != 1)
                    continue;
                var pt = p[0].ParameterType;
                if (!pt.IsValueType && typeof(Variable).IsAssignableFrom(pt))
                    result.Add(m);
            }
            return result.ToArray();
        }

        /// <summary>The structural kind of this graph value (graph-side mirror of <c>IValue.Structure()</c>).</summary>
        public DataStructure Structure() => this.Kind;

        // Element-type reinterprets — the typed tensor handle over this node (mirror of IValueExtensions
        // for graph-side Variable values; the runtime dtype is unchanged).
        public Tensor<uint4> uint4() => (Tensor<uint4>)this;
        public Tensor<uint8> uint8() => (Tensor<uint8>)this;
        public Tensor<uint16> uint16() => (Tensor<uint16>)this;
        public Tensor<uint32> uint32() => (Tensor<uint32>)this;
        public Tensor<uint64> uint64() => (Tensor<uint64>)this;
        public Tensor<int4> int4() => (Tensor<int4>)this;
        public Tensor<int8> int8() => (Tensor<int8>)this;
        public Tensor<int16> int16() => (Tensor<int16>)this;
        public Tensor<int32> int32() => (Tensor<int32>)this;
        public Tensor<int64> int64() => (Tensor<int64>)this;
        public Tensor<float16> float16() => (Tensor<float16>)this;
        public Tensor<bfloat16> bfloat16() => (Tensor<bfloat16>)this;
        public Tensor<float32> float32() => (Tensor<float32>)this;
        public Tensor<float64> float64() => (Tensor<float64>)this;

        // ── Graph-value introspection (the members the IValue handle interface exposes user-side) ──
        public bool IsConnectingTensor => OwningNode.IsOpenNode && OwningNode.ConnectingTensor == this;

        public InputType? InputType
        {
            get
            {
                if (!this.OwningNode.IsModelInput)
                    return null;
                if (this.OwningNode.OpCode == InternalOpCodes.GENERIC_TYPE_INPUT)
                    return Shorokoo.Core.Nodes.NodeDefinitions.InputType.GenericType;
                var inputType = this.OwningNode.Attributes.GetEnumVal<InputType>(OnnxOpAttributeNames.ShrkAttrInputType);
                return inputType ?? Shorokoo.Core.Nodes.NodeDefinitions.InputType.ReadyInput;
            }
        }

        public float? HyperDefaultValue
            => this.OwningNode.IsModelInput
                && this.OwningNode.Attributes.GetAttributeVals().TryGetValue(OnnxOpAttributeNames.ShrkAttrDefaultValue, out var dv)
                ? (float?)dv
                : null;

        public TensorDim[]? TensorDims
        {
            get
            {
                var tensorData = this.OwningNode.GetTensorData();
                if (tensorData is not null)
                    return tensorData.Shape.Dims.Select(x => new TensorDim(x)).ToArray();
                return this.Rank is int r ? Enumerable.Range(1, r).Select(_ => new TensorDim()).ToArray() : null;
            }
        }

        // ── Tensor surface (meaningful for DataStructure.Tensor values) ──
        public virtual Vector<int64> DShape => OnnxOp.Shape(this, null, null);

        public virtual Vector<int64>? InfShape => this.Kind == DataStructure.Tensor && this.Rank == 0
            ? Vector<int64>.Empty
            : (this.infShapeTensor ??= this.shapeInferer?.Invoke());

        public Vector<int64> TShape => this.DShape;
        public Scalar<int64> TRank => TShape.TShape[0];

        /// <summary>Casts the element type to <typeparamref name="V"/>; returns this tensor unchanged when the types already match.</summary>
        public Tensor<V> Cast<V>(bool saturate = true) where V : IVarType
            => OnnxUtils.GetDType<V>() == this.Type ?
                (Tensor<V>)this :
                OnnxOp.Cast(this, saturate ? null : saturate, OnnxUtils.GetDType<V>());

        // ── Sequence surface (meaningful for DataStructure.Sequence values) ──
        public Scalar<int64> Count => OnnxOp.SequenceLength(this);
        public Variable Concat(long axis, bool newAxis = false) => OnnxOp.ConcatFromSequence(this, axis, newAxis);
        public Variable At(Scalar<int64> index) => OnnxOp.SequenceAt(this, index);
        public Variable RemoveAt(Scalar<int64> index) => OnnxOp.SequenceErase(this, index);
        public Variable InsertAt(Variable tensor, Scalar<int64> index) => OnnxOp.SequenceInsert(this, tensor, index);

        // ── Struct surface (meaningful for DataStructure.TensorStruct values) ──
        public TensorStructDef Definition => this.structDef!;
        public Variable GetField(string name) => Field(name);
        internal TensorStructDef Def => this.structDef!;
        internal ImmutableDictionary<string, Variable> Fields => this.fields;

        internal Variable Field(string name)
        {
            if (this.fields.TryGetValue(name, out var field))
                return field;
            throw new KeyNotFoundException($"Field '{name}' not found in TensorStruct. Available fields: {string.Join(", ", this.fields.Keys)}");
        }

        internal Variable WithFields(ImmutableDictionary<string, Variable> newFields)
            => new Variable(this.Type, this.OwningNode, this.ModuleFn, this.UniqueName, DataStructure.TensorStruct, structDef: this.structDef, fields: newFields);

        public override string ToString()
            => this.Kind == DataStructure.TensorStruct
                ? $"TensorStruct<{this.structDef?.TypeName ?? "DTypeStruct"}>[{this.fields.Count} fields]"
                : (this.uniqueName ?? "") + ": " + this.GetType().Name
                    + (this.Kind == DataStructure.Tensor ? "[" + (this.Rank ?? -1) + "]" : "/" + this.Kind);
    }
}

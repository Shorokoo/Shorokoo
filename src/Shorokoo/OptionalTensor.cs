
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using Shorokoo.Core;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection;
using System.Reflection.Metadata;
using System.Runtime.CompilerServices;
using static Shorokoo.Core.Nodes.Ops;
using static Shorokoo.Core.Nodes.AutoDiff.Ops;
using Shorokoo;

namespace Shorokoo
{
    public interface IOptionalTensor : IValue
    {
    }

    /// <summary>
    /// Value-type handle for an optional tensor. The original <c>OptionalTensor&lt;T&gt;</c> name now
    /// denotes this <see langword="struct"/>; the reference type was renamed
    /// <see cref="Variable"/>. This struct carries the full user-facing surface.
    /// <para>
    /// The struct holds the immutable <b>directly</b> in a field (no shared box) so copying a handle
    /// copies the reference field by value — giving the Module DSL value-type semantics: a callee
    /// mutating its parameter does not affect the caller. This pass only makes mutation
    /// <i>possible</i>; nothing mutates yet, so behaviour is unchanged (de-facto immutable).
    /// </para>
    /// <para>
    /// A zero-initialised handle (<c>default</c>, <c>inner == null</c>) lazily materialises an
    /// <b>absent</b> optional on first use.
    /// </para>
    /// </summary>
    public struct OptionalTensor<T> : IOptionalTensor where T : IVarType
    {
        private Variable? inner;

        // Immutable materialises an absent optional for a defaulted handle, which is its established default.
        Variable IValue.ToVariable() => Immutable;

        /// <summary>The backing Variable, materialising an absent optional for a defaulted handle.</summary>
        internal Variable Immutable
            => inner ??= OnnxOp.Optional(null, DataStructure.Tensor, OnnxUtils.GetDType<T>());

        // Convert between the handle and its backing Variable.
        private static readonly DType? expectedDType = OnnxUtils.GetDType(typeof(T));
        public static implicit operator OptionalTensor<T>(Variable imm)
        {
            IValue.RequireKind(imm, DataStructure.Optional);
            IValue.RequireDType(imm, expectedDType);
            return new OptionalTensor<T> { inner = imm };
        }
        public static implicit operator Variable(OptionalTensor<T> handle)
            => handle.Immutable;

        /// <summary>
        /// Implicitly unwraps an optional to a nullable tensor (<c>Tensor&lt;T&gt;?</c>) by reading
        /// its element. An absent handle (defaulted, <c>inner == null</c>) maps to <c>null</c>;
        /// otherwise the element is taken via <see cref="TensorValue"/>.
        /// </summary>
        public static implicit operator Tensor<T>?(OptionalTensor<T> optional)
            => optional.inner is null ? default(Tensor<T>?) : optional.TensorValue();

        // ── User-facing API (the optional surface lives here, not on the immutable) ──
        public Variable Value() => OnnxOp.OptionalGetElement(Immutable);
        public Tensor<T> TensorValue() => (Variable)Value();
        public Scalar<bit> HasValue() => OnnxOp.OptionalHasElement(Immutable);

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

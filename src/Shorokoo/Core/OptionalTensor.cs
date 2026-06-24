
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

namespace Shorokoo.Core
{
    public interface IOptionalTensor : IVariable
    {
    }

    public class OptionalTensor<T> : Variable<T>, IOptionalTensor where T : IVarType
    {
        public OptionalTensor(DType type, Node owningNode, Function? moduleFn, string? name) : base(type, owningNode, moduleFn, name) {}

        public IVariable Value()
            => OnnxOp.OptionalGetElement(this);

        public Tensor<T> TensorValue()
            => (Tensor<T>)Value();

        public Tensor<T> SequenceValue()
            => (Tensor<T>)Value();

        public Scalar<bit> HasValue()
            => (Scalar<bit>)OnnxOp.OptionalHasElement(this);

        /// <summary>
        /// Implicitly unwraps an optional to a nullable tensor (<c>Tensor&lt;T&gt;?</c>) by reading
        /// its element. This lets an <c>OptionalTensor</c> be passed where the source-generated
        /// surface now expects a <c>Tensor&lt;T&gt;?</c> (a present optional forwards its value).
        /// A C#-null reference maps to null; otherwise the element is taken via
        /// <see cref="Value"/>/<c>OptionalGetElement</c> (the present case — known-absent optionals
        /// should be passed as <c>null</c> directly rather than via this conversion).
        /// </summary>
        public static implicit operator Tensor<T>?(OptionalTensor<T>? optional)
            => optional is null ? null : optional.TensorValue();
    }
}

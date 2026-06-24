
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
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection;
using System.Reflection.Metadata;
using System.Runtime.CompilerServices;
using static Shorokoo.Core.Nodes.Ops;
using static Shorokoo.Core.Nodes.AutoDiff.Ops;

namespace Shorokoo.Core
{
    public interface ITensorSequence : IVariable
    {
        public Scalar<int64> Count { get; }
        ITensor Concat(long axis, bool newAxis = false);
        ITensor this[Scalar<int64> index] { get; }
        ITensorSequence RemoveAt(Scalar<int64> index);
        ITensorSequence InsertAt(ITensor tensor, Scalar<int64> index);

        /// <summary>
        /// Inserts an arbitrary IVariable (e.g. an <see cref="ITensorStruct"/> for
        /// sequence-of-struct cases) at the given position, or appends when
        /// <paramref name="index"/> is null. Default implementation lowers directly to
        /// <c>SEQUENCE_INSERT</c>.
        /// </summary>
        public ITensorSequence Insert(IVariable variable, Scalar<int64>? index = null)
            => (ITensorSequence)OnnxOp.SequenceInsert(this, variable, index);

        public static ITensorSequence CreateEmpty(DType dtype)
                => (ITensorSequence)OnnxOp.SequenceEmpty(dtype);

        public static ITensorSequence Create(IVariable[] variables)
            => (ITensorSequence)OnnxOp.SequenceConstruct(variables);
    }

    public class TensorSequence<T> : Variable<T>, ITensorSequence
        where T : IVarType
    {
        internal TensorSequence(DType dtype, Node owningNode, Function? moduleFn, string? name) : base(dtype, owningNode, moduleFn, name) {}

        public Scalar<int64> Count => (Scalar<int64>)OnnxOp.SequenceLength(this);

        ITensor ITensorSequence.Concat(long axis, bool newAxis) => this.Concat(axis, newAxis);
        public Tensor<T> Concat(long axis, bool newAxis = false)
            => (Tensor<T>)OnnxOp.ConcatFromSequence(this, axis, newAxis);

        ITensor ITensorSequence.this[Scalar<int64> index] => this[index];
        public Tensor<T> this[Scalar<int64> index]
            => (Tensor<T>)OnnxOp.SequenceAt(this, index);

        ITensorSequence ITensorSequence.RemoveAt(Scalar<int64> index) => RemoveAt(index);
        /// <summary>Removes the element at <paramref name="index"/>, or the LAST element
        /// when called without an index (ONNX SequenceErase's optional-position default).</summary>
        public TensorSequence<T> RemoveAt(Scalar<int64>? index = null)
            => this.OwningNode.TargetFunction is null ?
                    (TensorSequence<T>)OnnxOp.SequenceErase(this, index) :
                    (TensorSequence<T>)OnnxOp.SequenceErase(this.OwningNode.TargetFunction, this, index);

        ITensorSequence ITensorSequence.InsertAt(ITensor tensor, Scalar<int64> index) => InsertAt((Tensor<T>)tensor, index);
        public TensorSequence<T> InsertAt(Tensor<T> tensor, Scalar<int64>? index)
            => this.OwningNode.TargetFunction is null ?
                    (TensorSequence<T>)OnnxOp.SequenceInsert(this, tensor, index) :
                    (TensorSequence<T>)OnnxOp.SequenceInsert(this.OwningNode.TargetFunction, this, tensor, index);

        public TensorSequence<T> Append(Tensor<T> tensor) => this.InsertAt(tensor, null);

        public static TensorSequence<T> CreateEmpty()
            => (TensorSequence<T>)OnnxOp.SequenceEmpty(OnnxUtils.GetDType<T>());

        internal static TensorSequence<T> CreateEmpty(Function targetFunction)
            => (TensorSequence<T>)OnnxOp.SequenceEmpty(OnnxUtils.GetDType<T>(), targetFunction);

        public static TensorSequence<T> Create(Tensor<T>[] tensors)
            => tensors.Length == 0 ? CreateEmpty() :
                    (TensorSequence<T>)OnnxOp.SequenceConstruct(tensors);

        internal static TensorSequence<T> Create(Tensor<T>[] tensors, Function targetFunction)
            => tensors.Length == 0 ? CreateEmpty() :
                    (TensorSequence<T>)OnnxOp.SequenceConstruct(targetFunction, tensors);

    }
}

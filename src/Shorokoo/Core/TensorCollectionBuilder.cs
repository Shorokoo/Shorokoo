using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using Shorokoo;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text;

namespace Shorokoo.Core
{
    public class TensorExpressionHelper<T>
        where T : IVarType
    {
        internal readonly Tensor<T> tensor;
        internal readonly bool isEnumerated;

        internal TensorExpressionHelper(Tensor<T> tensor, bool isEnumerated)
        {
            this.tensor = tensor;
            this.isEnumerated = isEnumerated;
        }

        public static implicit operator TensorExpressionHelper<T>(Tensor<T> element) { return new TensorExpressionHelper<T>(element, isEnumerated: false); }
        public static implicit operator TensorExpressionHelper<T>(Scalar<T> element) { return new TensorExpressionHelper<T>(element, isEnumerated: false); }
        public static implicit operator TensorExpressionHelper<T>(Vector<T> element) { return new TensorExpressionHelper<T>(element, isEnumerated: false); }
    }

    public class VectorExpressionHelper<T>
        where T : IVarType
    {
        internal readonly Scalar<T>? scalar;
        internal readonly Vector<T>? vector;

        internal VectorExpressionHelper(Scalar<T> scalar)
        {
            this.scalar = scalar;
        }

        internal VectorExpressionHelper(Vector<T> vector)
        {
            // A Vector<T> is always rank-1 (the Variable→Vector conversion enforces it), so it is
            // stored as the vector arm directly.
            this.vector = vector;
        }

        public static implicit operator VectorExpressionHelper<T>(Scalar<T> scalar) 
             => new VectorExpressionHelper<T>(scalar);
    }

    public static class TensorCollectionBuilder
    {
        public static Tensor<T> Create<T>(ReadOnlySpan<TensorExpressionHelper<T>> elements)
            where T : IVarType
        {
            if (elements.Length == 0)
            {
                // We know that we're creating a 0 scalar vector. The problem is we don't know its shape.
                // It could be a (0,) vector or a (0,0) tensor or a (3,5,1,0,5) tensor. We have no idea...

                // AND it won't get caught until runtime :(
                throw new InvalidTensorOperationException(ErrorCodes.FW008, "Create", "empty collection expression", 
                    "When using collection expression, e.g. OnnxObj<float> myTensor = [..listOfTensors1, ..listOfTensor2]. The contents must not evaluate to an empty list. If your lists of tensors contain tensors of shape (x,4,6), add a tensor of shape (0,4,6): [..listOfTensors1, ..listOfTensors2, OnnxObj<TT>.Create((0,4,6)]. This will make this exception go away");
            }

            var toConcat = elements.ToArray().Select(x => x.isEnumerated ? x.tensor : x.tensor.Unsqueeze(0)).ToArray();
            return toConcat[0].Concat(0, toConcat.Skip(1).ToArray());
        }

        public static Vector<T> CreateVector<T>(ReadOnlySpan<VectorExpressionHelper<T>> elements)
            where T : IVarType
        {
            if (elements.Length == 0)
            {
                // Unlike the general tensor case, here we know for a fact we want a OnnxObj of shape (0)
                return Globals.EmptyVector<T>();
            }

            if (elements.ToArray().Any(x => x.vector is null && x.scalar is null))
                throw new InvalidTensorOperationException(ErrorCodes.FW008, "CreateVector", "null elements in collection", 
                    "Cannot make vectors that contain null elements");

            var toConcat = elements.ToArray().Select(x => x.scalar is not null ? (Tensor<T>)x.scalar.Value.Unsqueeze() : (Tensor<T>)x.vector!.Value).ToArray();
            return toConcat[0].Concat(0L, toConcat.Skip(1).ToArray()).Vec();
        }

        public static Tensor<T> CreateTensor<T>(ReadOnlySpan<VectorExpressionHelper<T>> elements)
            where T : IVarType
            => CreateVector(elements);
    }
}

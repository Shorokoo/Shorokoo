using Shorokoo.Core;
using Shorokoo;
using Shorokoo.Runtime;
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
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection.Metadata;
using System.Runtime.CompilerServices;
using Shorokoo.Core.Nodes.NodeDefinitions;
using static Shorokoo.Core.Nodes.Ops;
using static Shorokoo.Core.Nodes.AutoDiff.Ops;
using static RandN.Distributions.Uniform;
using static Shorokoo.Globals;

namespace Shorokoo
{

    /// <summary>
    /// Non-generic view of a symbolic tensor: a graph variable with a queryable shape, an optional
    /// statically known rank, and conversions to the typed <see cref="Tensor{T}"/> family.
    /// </summary>
    public interface ITensor : IVariable
    {
        /// <summary>The shape of this tensor as a symbolic rank-1 tensor of dimension sizes.</summary>
        public Vector<int64> TShape { get; }

        /// <summary>Statically known rank (number of dimensions), or null when not known at graph-construction time.</summary>
        public int? Rank { get; }

        /// <summary>Casts the element type to <typeparamref name="T"/> and reinterprets the result as a rank-1 <see cref="Vector{T}"/>.</summary>
        public Vector<T> Vec<T>() where T : IVarType;
        /// <summary>Casts the element type to <typeparamref name="T"/>.</summary>
        public Tensor<T> Cast<T>(bool saturate = true) where T : IVarType;

        /// <summary>Reinterprets this tensor as a rank-1 vector.</summary>
        public IVector Vec();

        /// <summary>Reinterprets this tensor as a rank-0 scalar.</summary>
        public IScalar Scalar();

        /// <summary>Casts the element type to <typeparamref name="T"/> and reinterprets the result as a rank-0 <see cref="Scalar{T}"/>.</summary>
        public Scalar<T> Scalar<T>() where T : IVarType;
    }

    /// <summary>
    /// A symbolic tensor of element type <typeparamref name="T"/> - the main user-facing tensor type.
    /// Operations do not compute values; they add ONNX-style nodes to the computation graph, which is
    /// executed on demand via <see cref="Eval()"/>.
    /// </summary>
    [CollectionBuilder(typeof(Shorokoo.Core.TensorCollectionBuilder), nameof(Shorokoo.Core.TensorCollectionBuilder.Create))]
    public class Tensor<T> : Variable<T>, ITensor, IEnumerable<Shorokoo.Core.TensorExpressionHelper<T>>
         where T : IVarType
    {
        #region Data Member

        /// <summary>Statically known rank (number of dimensions), or null when not known at graph-construction time.</summary>
        public int? Rank { get; }

        /// <summary>The dynamic shape: a rank-1 tensor produced in-graph by an ONNX Shape node.</summary>
        public virtual Vector<int64> DShape =>
            (Vector<int64>)OnnxOp.Shape(this, null, null);

        private Vector<int64>? infShapeTensor = null;

        /// <summary>The statically inferred shape, or null when no shape inference is available for this tensor.</summary>
        public virtual Vector<int64>? InfShape => this.infShapeTensor ??
                    (this.infShapeTensor = shapeInferer?.Invoke());

        /// <summary>The shape of this tensor as a rank-1 tensor of dimension sizes (currently the dynamic <see cref="DShape"/>).</summary>
        public Vector<int64> TShape => // this.InfShape ??
            this.DShape;

        /// <summary>The rank of this tensor as a symbolic scalar, derived from <see cref="TShape"/>.</summary>
        public Scalar<int64> TRank => TShape.TShape[0].T;

        private Func<Vector<int64>>? shapeInferer;

        #endregion

        #region Constructor

        internal Tensor(Func<Vector<int64>>? shapeFn, DType dtype, Node owningNode, Function? moduleFn, string? name, int? rank) : base(dtype, owningNode, moduleFn, name)
        {
            this.shapeInferer = shapeFn;
            this.Rank = rank;
        }

        #endregion

        #region ITensor

        IVector ITensor.Vec() => this.Vec();
        Vector<V> ITensor.Vec<V>() => this.Cast<V>().Vec();
        IScalar ITensor.Scalar() => this.Scalar();
        Scalar<V> ITensor.Scalar<V>() => this.Cast<V>().Scalar();

        #endregion

        #region Compute

        /// <summary>
        /// Builds a computation graph with this tensor as its sole output, executes it with a fresh
        /// <see cref="ComputeContext"/>, and returns the materialized data.
        /// </summary>
        public TensorData Eval()
        {
            return this.Eval(new ComputeContext());
        }

        /// <summary>Executes the graph rooted at this tensor using the given context (a fresh one if null) and returns the data.</summary>
        public TensorData Eval(ComputeContext ctx)
        {
            if (ctx == null)
                ctx = new ComputeContext();

            var graph = new Shorokoo.Graph.FastComputationGraph([], [this]);

            return ctx.Execute(graph)[0].ToTensorData();
        }

        #endregion

        #region Casts

        /// <summary>Reinterprets this tensor as a rank-1 <see cref="Vector{T}"/>, inserting an Identity node when needed.</summary>
        public new Vector<T> Vec()
        {
            if (this is Vector<T> vec)
            {
                return vec;
            }
            else
            {
                return (Vector<T>)OnnxOp.Identity(this, rank: 1);
            }
        }

        /// <summary>Reinterprets this tensor as a rank-0 <see cref="Scalar{T}"/>, inserting an Identity node when needed.</summary>
        public new Scalar<T> Scalar()
        {
            if (this is Scalar<T> scalar)
            {
                return scalar;
            }
            else
            {
                return (Scalar<T>)OnnxOp.Identity(this, rank: 0);
            }
        }

        #endregion

        #region Onxx Cast Operators

        /// <summary>Casts the element type to int32.</summary>
        public static explicit operator Tensor<int32>(Tensor<T> tensor) => tensor.Cast<int32>();
        /// <summary>Casts the element type to uint32.</summary>
        public static explicit operator Tensor<uint32>(Tensor<T> tensor) => tensor.Cast<uint32>();
        /// <summary>Casts the element type to int64.</summary>
        public static explicit operator Tensor<int64>(Tensor<T> tensor) => tensor.Cast<int64>();
        /// <summary>Casts the element type to uint64.</summary>
        public static explicit operator Tensor<uint64>(Tensor<T> tensor) => tensor.Cast<uint64>();
        /// <summary>Casts the element type to float16.</summary>
        public static explicit operator Tensor<float16>(Tensor<T> tensor) => tensor.Cast<float16>();
        /// <summary>Casts the element type to bfloat16.</summary>
        public static explicit operator Tensor<bfloat16>(Tensor<T> tensor) => tensor.Cast<bfloat16>();
        /// <summary>Casts the element type to float32.</summary>
        public static explicit operator Tensor<float32>(Tensor<T> tensor) => tensor.Cast<float32>();
        /// <summary>Casts the element type to float64.</summary>
        public static explicit operator Tensor<float64>(Tensor<T> tensor) => tensor.Cast<float64>();
        /// <summary>Casts the element type to bit.</summary>
        public static explicit operator Tensor<bit>(Tensor<T> tensor) => tensor.Cast<bit>();

        #endregion

        #region Collection Expression Helper
        /// <summary>Supports C# collection-expression composition by yielding a single helper that wraps this tensor.</summary>
        public IEnumerator<Shorokoo.Core.TensorExpressionHelper<T>> GetEnumerator()
        {
            return new List<Shorokoo.Core.TensorExpressionHelper<T>> {
                new Shorokoo.Core.TensorExpressionHelper<T>(this, isEnumerated: true) }
                .GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator() => this.GetEnumerator();

        #endregion

        #region Operator Overloads

        /// <summary>Element-wise addition.</summary>
        public static Tensor<T> operator +(Tensor<T> left, Tensor<T> right)
        {
            left.ThrowIsNumLike();

            return (Tensor<T>)OnnxOp.Add(left, right);
        }

        /// <summary>Element-wise subtraction.</summary>
        public static Tensor<T> operator -(Tensor<T> left, Tensor<T> right)
        {
            return (Tensor<T>)OnnxOp.Sub(left, right);
        }

        /// <summary>Element-wise multiplication.</summary>
        public static Tensor<T> operator *(Tensor<T> left, Tensor<T> right)
        {
            return (Tensor<T>)OnnxOp.Mul(left, right);
        }

        /// <summary>Element-wise division.</summary>
        public static Tensor<T> operator /(Tensor<T> left, Tensor<T> right)
        {
            return (Tensor<T>)OnnxOp.Div(left, right);
        }

        /// <summary>Element-wise modulo.</summary>
        public static Tensor<T> operator %(Tensor<T> left, Tensor<T> right)
            => (Tensor<T>)OnnxOp.Mod(left, right);

        /// <summary>Element-wise XOR: logical for bit tensors, bitwise otherwise.</summary>
        public static Tensor<T> operator ^(Tensor<T> left, Tensor<T> right)
            => typeof(T) == typeof(bit) ?
                (Tensor<T>)OnnxOp.Xor(left, right) :
                (Tensor<T>)OnnxOp.BitwiseXor(left, right);

        /// <summary>Element-wise AND: logical for bit tensors, bitwise otherwise.</summary>
        public static Tensor<T> operator &(Tensor<T> left, Tensor<T> right)
            => typeof(T) == typeof(bit) ?
                (Tensor<T>)OnnxOp.And(left, right) :
                (Tensor<T>)OnnxOp.BitwiseAnd(left, right);

        /// <summary>Element-wise OR: logical for bit tensors, bitwise otherwise.</summary>
        public static Tensor<T> operator |(Tensor<T> left, Tensor<T> right)
            => typeof(T) == typeof(bit) ?
                (Tensor<T>)OnnxOp.Or(left, right) :
                (Tensor<T>)OnnxOp.BitwiseOr(left, right);

        /// <summary>Element-wise left bit-shift.</summary>
        public static Tensor<T> operator <<(Tensor<T> left, Tensor<T> right)
            => (Tensor<T>)OnnxOp.BitShift(left, right, BitShiftDirection.Left);

        /// <summary>Element-wise right bit-shift.</summary>
        public static Tensor<T> operator >>(Tensor<T> left, Tensor<T> right)
            => (Tensor<T>)OnnxOp.BitShift(left, right, BitShiftDirection.Right);

        /// <summary>Element-wise negation.</summary>
        public static Tensor<T> operator -(Tensor<T> input)
        {
            return (Tensor<T>)OnnxOp.Neg(input);
        }

        /// <summary>Element-wise NOT: logical for bit tensors, bitwise otherwise.</summary>
        public static Tensor<T> operator !(Tensor<T> input)
            => typeof(T) == typeof(bit) ?
                (Tensor<T>)OnnxOp.Not(input) :
                (Tensor<T>)OnnxOp.BitwiseNot(input);

        /// <summary>Element-wise greater-than, yielding a bit tensor.</summary>
        public static Tensor<bit> operator >(Tensor<T> left, Tensor<T> right) => (Tensor<bit>)OnnxOp.Greater(left, right);

        /// <summary>Element-wise greater-or-equal, yielding a bit tensor.</summary>
        public static Tensor<bit> operator >=(Tensor<T> left, Tensor<T> right) => (Tensor<bit>)OnnxOp.GreaterOrEqual(left, right);

        /// <summary>Element-wise less-than, yielding a bit tensor.</summary>
        public static Tensor<bit> operator <(Tensor<T> left, Tensor<T> right) => (Tensor<bit>)OnnxOp.Less(left, right);

        /// <summary>Element-wise less-or-equal, yielding a bit tensor.</summary>
        public static Tensor<bit> operator <=(Tensor<T> left, Tensor<T> right) => (Tensor<bit>)OnnxOp.LessOrEqual(left, right);

        /// <summary>Element-wise equality, yielding a bit tensor (not reference equality).</summary>
        public static Tensor<bit> operator ==(Tensor<T> left, Tensor<T> right) => (Tensor<bit>)OnnxOp.Equal(left, right);

        /// <summary>Element-wise inequality, yielding a bit tensor.</summary>
        public static Tensor<bit> operator !=(Tensor<T> left, Tensor<T> right) => !(left == right);


        #region primitive param support

        /// <summary>Element-wise addition with a primitive constant.</summary>
        public static Tensor<T> operator +(Tensor<T> left, PrimitiveParam right) => left + (Scalar<T>)right;
        /// <summary>Element-wise addition with a primitive constant.</summary>
        public static Tensor<T> operator +(PrimitiveParam left, Tensor<T> right) => (Scalar<T>)left + right;

        /// <summary>Element-wise subtraction with a primitive constant.</summary>
        public static Tensor<T> operator -(Tensor<T> left, PrimitiveParam right) => left - (Scalar<T>)right;
        /// <summary>Element-wise subtraction with a primitive constant.</summary>
        public static Tensor<T> operator -(PrimitiveParam left, Tensor<T> right) => (Scalar<T>)left - right;

        /// <summary>Element-wise multiplication with a primitive constant.</summary>
        public static Tensor<T> operator *(Tensor<T> left, PrimitiveParam right) => left * (Scalar<T>)right;
        /// <summary>Element-wise multiplication with a primitive constant.</summary>
        public static Tensor<T> operator *(PrimitiveParam left, Tensor<T> right) => (Scalar<T>)left * right;

        /// <summary>Element-wise division with a primitive constant.</summary>
        public static Tensor<T> operator /(Tensor<T> left, PrimitiveParam right) => left / (Scalar<T>)right;
        /// <summary>Element-wise division with a primitive constant.</summary>
        public static Tensor<T> operator /(PrimitiveParam left, Tensor<T> right) => (Scalar<T>)left / right;

        /// <summary>Element-wise modulo with a primitive constant.</summary>
        public static Tensor<T> operator %(Tensor<T> left, PrimitiveParam right) => left % (Scalar<T>)right;
        /// <summary>Element-wise modulo with a primitive constant.</summary>
        public static Tensor<T> operator %(PrimitiveParam left, Tensor<T> right) => (Scalar<T>)left % right;

        /// <summary>Element-wise XOR with a primitive constant.</summary>
        public static Tensor<T> operator ^(Tensor<T> left, PrimitiveParam right) => left ^ (Scalar<T>)right;
        /// <summary>Element-wise XOR with a primitive constant.</summary>
        public static Tensor<T> operator ^(PrimitiveParam left, Tensor<T> right) => (Scalar<T>)left ^ right;

        /// <summary>Element-wise AND with a primitive constant.</summary>
        public static Tensor<T> operator &(Tensor<T> left, PrimitiveParam right) => left & (Scalar<T>)right;
        /// <summary>Element-wise AND with a primitive constant.</summary>
        public static Tensor<T> operator &(PrimitiveParam left, Tensor<T> right) => (Scalar<T>)left & right;

        /// <summary>Element-wise OR with a primitive constant.</summary>
        public static Tensor<T> operator |(Tensor<T> left, PrimitiveParam right) => left | (Scalar<T>)right;
        /// <summary>Element-wise OR with a primitive constant.</summary>
        public static Tensor<T> operator |(PrimitiveParam left, Tensor<T> right) => (Scalar<T>)left | right;

        /// <summary>Element-wise left bit-shift by a primitive constant.</summary>
        public static Tensor<T> operator <<(Tensor<T> left, PrimitiveParam right) => left << (Scalar<T>)right;
        // public static Tensor<T> operator <<(PrimitiveParam left, Tensor<T> right) => (Scalar<T>)left + right;

        /// <summary>Element-wise right bit-shift by a primitive constant.</summary>
        public static Tensor<T> operator >>(Tensor<T> left, PrimitiveParam right) => left >> (Scalar<T>)right;
        // public static Tensor<T> operator >>(PrimitiveParam left, Tensor<T> right) => (Scalar<T>)left + right;

        /// <summary>Element-wise greater-than with a primitive constant, yielding a bit tensor.</summary>
        public static Tensor<bit> operator >(Tensor<T> left, PrimitiveParam right) => left > (Scalar<T>)right;
        /// <summary>Element-wise greater-than with a primitive constant, yielding a bit tensor.</summary>
        public static Tensor<bit> operator >(PrimitiveParam left, Tensor<T> right) => (Scalar<T>)left > right;
        /// <summary>Element-wise greater-or-equal with a primitive constant, yielding a bit tensor.</summary>
        public static Tensor<bit> operator >=(Tensor<T> left, PrimitiveParam right) => left >= (Scalar<T>)right;
        /// <summary>Element-wise greater-or-equal with a primitive constant, yielding a bit tensor.</summary>
        public static Tensor<bit> operator >=(PrimitiveParam left, Tensor<T> right) => (Scalar<T>)left >= right;
        /// <summary>Element-wise less-than with a primitive constant, yielding a bit tensor.</summary>
        public static Tensor<bit> operator <(Tensor<T> left, PrimitiveParam right) => left < (Scalar<T>)right;
        /// <summary>Element-wise less-than with a primitive constant, yielding a bit tensor.</summary>
        public static Tensor<bit> operator <(PrimitiveParam left, Tensor<T> right) => (Scalar<T>)left < right;
        /// <summary>Element-wise less-or-equal with a primitive constant, yielding a bit tensor.</summary>
        public static Tensor<bit> operator <=(Tensor<T> left, PrimitiveParam right) => left <= (Scalar<T>)right;
        /// <summary>Element-wise less-or-equal with a primitive constant, yielding a bit tensor.</summary>
        public static Tensor<bit> operator <=(PrimitiveParam left, Tensor<T> right) => (Scalar<T>)left <= right;
        /// <summary>Element-wise equality with a primitive constant, yielding a bit tensor.</summary>
        public static Tensor<bit> operator ==(Tensor<T> left, PrimitiveParam right) => left == (Scalar<T>)right;
        /// <summary>Element-wise equality with a primitive constant, yielding a bit tensor.</summary>
        public static Tensor<bit> operator ==(PrimitiveParam left, Tensor<T> right) => (Scalar<T>)left == right;
        /// <summary>Element-wise inequality with a primitive constant, yielding a bit tensor.</summary>
        public static Tensor<bit> operator !=(Tensor<T> left, PrimitiveParam right) => left != (Scalar<T>)right;
        /// <summary>Element-wise inequality with a primitive constant, yielding a bit tensor.</summary>
        public static Tensor<bit> operator !=(PrimitiveParam left, Tensor<T> right) => (Scalar<T>)left != right;

        #endregion

        #endregion

        #region Onnx Ops

        /// <summary>Splits into <paramref name="numOutputs"/> equal parts along <paramref name="axis"/>.</summary>
        public Tensor<T>[] Split(long numOutputs, long axis = 0)
        {
            return OnnxOp.Split(this, null, axis: axis, numOutputs: numOutputs, variadicOutputCount: numOutputs).Cast<Tensor<T>>().ToArray();
        }

        /// <summary>Splits along <paramref name="axis"/> into parts of the given sizes.</summary>
        public Tensor<T>[] Split(long[] splits, long axis = 0)
        {
            return OnnxOp.Split(this, Vector(splits), axis: axis, numOutputs: null, variadicOutputCount: splits.Length).Cast<Tensor<T>>().ToArray();
        }

        /// <summary>Splits along <paramref name="axis"/>: by <paramref name="splits"/> sizes when given, otherwise into <paramref name="numOutputs"/> equal parts.</summary>
        public Tensor<T>[] Split(Vector<int64>? splits, long axis, long numOutputs)
        {
            return OnnxOp.Split(this, splits, axis: axis, numOutputs: splits is null ? numOutputs : null, variadicOutputCount: numOutputs).Cast<Tensor<T>>().ToArray();
        }

        /// <summary>Resizes to the target <paramref name="sizes"/> (ONNX Resize).</summary>
        public Tensor<T> Resize(Vector<int64> sizes,
            KeepAspectRatioPolicy? aspectRatio = null,
            bool? antiaAlias = null,
            long[]? axes = null,
            CoordinateTransformationMode? transformMode = null,
            ResizeMode? mode = null,
            NearestMode? nearestMode = null,
            float? cubicCoefficient = null,
            bool? excludeOutside = null)
        {
            return (Tensor<T>)OnnxOp.Resize(
                this, null, null, sizes, antiaAlias, axes, transformMode, cubicCoefficient, excludeOutside, null, 
                aspectRatio, 
                mode, nearestMode);
        }

        /// <summary>Resizes by the given per-axis scale factors (ONNX Resize).</summary>
        public Tensor<T> Rescale(Vector<float32> scales,
            bool? antiaAlias = null,
            long[]? axes = null,
            CoordinateTransformationMode? transformMode = null,
            ResizeMode? mode = null,
            NearestMode? nearestMode = null,
            float? cubicCoefficient = null,
            bool? excludeOutside = null)
        {
            return (Tensor<T>)OnnxOp.Resize(
                this, null, scales, null, antiaAlias, axes, transformMode, cubicCoefficient, excludeOutside, null,
                null, mode, nearestMode);
        }

        /// <summary>Removes size-1 dimensions; when <paramref name="axes"/> is null, all size-1 dimensions are removed.</summary>
        public Tensor<T> Squeeze(Vector<int64>? axes = null)
        {
            // Absent axes propagate as an absent (optional) input: ONNX then squeezes ALL
            // size-1 dims. (Previously substituted axes=[-1] — squeeze only the LAST dim,
            // and an ORT shape-inference error when that dim isn't 1.)
            return (Tensor<T>)OnnxOp.Squeeze(this, axes);
        }

        /// <summary>Slices along the given axes using start/end indices and optional steps (ONNX Slice).</summary>
        public Tensor<T> Slice(Vector<int64> start, Vector<int64> end, Vector<int64>? axes = null, Vector<int64>? steps = null)
            => (Tensor<T>)OnnxOp.Slice(this, start, end, axes, steps);

        /// <summary>Softmax normalization along <paramref name="axis"/> (defaults to the last axis).</summary>
        public Tensor<T> Softmax(long? axis = null)
            => (Tensor<T>)OnnxOp.Softmax(this, axis);

        /// <summary>Inserts a size-1 dimension at <paramref name="axis"/>.</summary>
        public Tensor<T> Unsqueeze(long axis)
            => (Tensor<T>)OnnxOp.Unsqueeze(this, Vector(axis));

        /// <summary>Inserts size-1 dimensions at the given axes.</summary>
        public Tensor<T> Unsqueeze(Vector<int64> axes)
            => (Tensor<T>)OnnxOp.Unsqueeze(this, axes);

        /// <summary>Appends a trailing size-1 dimension (axis -1).</summary>
        public Tensor<T> Unsqueeze()
            => (Tensor<T>)OnnxOp.Unsqueeze(this, Vector(-1L));

        /// <summary>The shape - optionally the dims[start:end] slice of it - as an in-graph vector.</summary>
        public Vector<int64> ShapeTensor(long? start = null, long? end = null)
            // OnnxOp.Shape declares (data, end, start) — name the args; passing positionally
            // swapped start/end (e.g. ShapeTensor(1) sliced dims[:1] instead of dims[1:]).
            => ((Tensor<int64>)OnnxOp.Shape(this, end: end, start: start)).Vec();

        /// <summary>The element count: the product of the dimensions, optionally restricted to dims[start:end].</summary>
        public Scalar<int64> SizeTensor(long? start = null, long? end = null)
            => this.ShapeTensor(start, end).Reduce(ReduceKind.Prod);

        /// <summary>The size of dimension <paramref name="axis"/> as a symbolic scalar.</summary>
        public Scalar<int64> DimTensor(long axis)
            => this.ShapeTensor()[axis].T.Scalar();

        /// <summary>Gathers entries along <paramref name="axis"/> using the given indices (ONNX Gather).</summary>
        public Tensor<T> Gather(Tensor<int64> indices, long? axis)
            => (Tensor<T>)OnnxOp.Gather(this, indices, axis);

        /// <summary>Gathers slices using multi-dimensional indices (ONNX GatherND).</summary>
        public Tensor<T> GatherND(Tensor<int64> indices, long? batchDims)
            => (Tensor<T>)OnnxOp.GatherND(this, indices, batchDims);

        /// <summary>Casts the element type to <typeparamref name="V"/>; returns this tensor unchanged when the types already match.</summary>
        public Tensor<V> Cast<V>(bool saturate = true) where V : IVarType
            => typeof(V) == typeof(T) ? 
                (Tensor<V>)(ITensor)this :
                (Tensor<V>)OnnxOp.Cast(this, saturate ? null : saturate, OnnxUtils.GetDType<V>());

        /// <summary>Creates a tensor of the given shape filled with the scalar value <paramref name="val"/> (ONNX ConstantOfShape).</summary>
        public static Tensor<T> Fill(Vector<int64> shape, TensorData val)
        {
            // ConstantOfShape expects:
            // - shape: A 1D tensor indicating the shape of the output
            // - val: A scalar TensorData that will be broadcasted to fill the entire output
            // The issue was using pre-shaped TensorData instead of scalar fill value
            return (Tensor<T>)OnnxOp.ConstantOfShape(shape, val);
        }
        
        /// <summary>Reduction (e.g. sum, mean, max) over <paramref name="axes"/> - all axes when null - keeping reduced dimensions by default.</summary>
        public Tensor<T> Reduce(ReduceKind reduceKind, Vector<int64>? axes = null, bool keepDims = true)
            => NN.Reduce(reduceKind, this, axes, keepDims, false);

        /// <summary>Tiles the tensor by repeating it <paramref name="repeats"/> times along each axis.</summary>
        public Tensor<T> Tile(Tensor<int64> repeats)
            => (Tensor<T>)OnnxOp.Tile(this, repeats);

        /// <summary>Element-wise maximum of this tensor and <paramref name="others"/>.</summary>
        public Tensor<T> Max(params Tensor<T>[] others)
            => (Tensor<T>)OnnxOp.Max([this, .. others]);

        /// <summary>Element-wise minimum of this tensor and <paramref name="others"/>.</summary>
        public Tensor<T> Min(params Tensor<T>[] others)
            => (Tensor<T>)OnnxOp.Min([this, .. others]);

        /// <summary>Element-wise floor.</summary>
        public Tensor<T> Floor()
            => (Tensor<T>)OnnxOp.Floor(this);

        /// <summary>Matrix product with <paramref name="other"/> (ONNX MatMul).</summary>
        public Tensor<T> MatMul(Tensor<T> other)
            => (Tensor<T>)OnnxOp.MatMul(this, other);

        /// <summary>Element-wise absolute value.</summary>
        public Tensor<T> Abs()
            => (Tensor<T>)OnnxOp.Abs(this);

        /// <summary>Element-wise arccosine.</summary>
        public Tensor<T> Acos()
            => (Tensor<T>)OnnxOp.Acos(this);

        /// <summary>Element-wise inverse hyperbolic cosine.</summary>
        public Tensor<T> Acosh()
            => (Tensor<T>)OnnxOp.Acosh(this);

        /// <summary>Element-wise arcsine.</summary>
        public Tensor<T> Asin()
            => (Tensor<T>)OnnxOp.Asin(this);

        /// <summary>Element-wise inverse hyperbolic sine.</summary>
        public Tensor<T> Asinh()
            => (Tensor<T>)OnnxOp.Asinh(this);

        /// <summary>Element-wise arctangent.</summary>
        public Tensor<T> Atan()
            => (Tensor<T>)OnnxOp.Atan(this);

        /// <summary>Element-wise inverse hyperbolic tangent.</summary>
        public Tensor<T> Atanh()
            => (Tensor<T>)OnnxOp.Atanh(this);

        /// <summary>Indices of the maximum values along <paramref name="axis"/>.</summary>
        public Tensor<int64> ArgMax(long axis = 0, bool keepdims = false, bool selectLastIndex = false)
            => (Tensor<int64>)OnnxOp.ArgMax(this, axis == 0 ? null : axis, keepdims ? null : keepdims, selectLastIndex ? selectLastIndex : null);

        /// <summary>Indices of the minimum values along <paramref name="axis"/>.</summary>
        public Tensor<int64> ArgMin(long axis = 0, bool keepdims = false, bool selectLastIndex = false)
            => (Tensor<int64>)OnnxOp.ArgMin(this, axis == 0 ? null : axis, keepdims ? null : keepdims, selectLastIndex ? selectLastIndex : null);

        /// <summary>Average pooling with the given kernel shape (ONNX AveragePool).</summary>
        public Tensor<T> AveragePool(long[] kernelShape, RoundMode roundMode = RoundMode.Floor, bool countIncludePad = false, long[]? dilations = null, long[]? pads = null, long[]? strides = null)
            => (Tensor<T>)OnnxOp.AveragePool(this, null, roundMode == RoundMode.Floor ? null : true, countIncludePad, dilations, kernelShape, pads, strides);

        /// <summary>Batch normalization using the given scale, bias, mean, and variance (ONNX BatchNormalization).</summary>
        public Tensor<T> BatchNormalization<T1, T2>(Vector<T1> scale, Vector<T1> bias, Vector<T2> mean, Vector<T2> variance, float epsilon = 1e-05f, float momentum = 0.9f, bool trainingMode = false)
                where T1 : FloatLike where T2 : FloatLike
        {
            var retval = OnnxOp.BatchNormalization(this, scale, bias, mean, variance, epsilon == 1e-05f ? null : epsilon, momentum == 0.9f ? null : momentum, trainingMode == false ? null : trainingMode);
            return (Tensor<T>)retval;
        }

        /// <summary>Batch normalization that also returns the updated running mean and variance.</summary>
        public (Tensor<T> y, Vector<T2> runningMean, Vector<T2> runningVariance) BatchNormalizationFullOuputs<T1, T2>(Vector<T1> scale, Vector<T1> bias, Vector<T2> mean, Vector<T2> variance, float epsilon = 1e-05f, float momentum = 0.9f, bool trainingMode = false)
                where T1 : FloatLike where T2 : FloatLike
        {
            var retval = OnnxOp.BatchNormalizationFullOutputs(this, scale, bias, mean, variance, epsilon == 1e-05f ? null : epsilon, momentum == 0.9f ? null : momentum, trainingMode == false ? null : trainingMode);
            return ((Tensor<T>)retval.y, retval.runningMean.As<T2>().Vec(), retval.runningVariance.As<T2>().Vec());
        }

        /// <summary>Element-wise Bernoulli sampling, treating each element as a probability.</summary>
        public Tensor<T> Bernoulli(float? seed = null)
            => (Tensor<T>)OnnxOp.Bernoulli(this, null, seed);

        /// <summary>Element-wise Bernoulli sampling, treating each element as a probability, with result element type <typeparamref name="V"/>.</summary>
        public Tensor<V> Bernoulli<V>(float? seed = null) where V : CommonLike
            => (Tensor<V>)OnnxOp.Bernoulli(this, OnnxUtils.GetDType<V>(), seed);

        /// <summary>Element-wise ceiling.</summary>
        public Tensor<T> Ceiling()
            => (Tensor<T>)OnnxOp.Ceil(this);

        /// <summary>Element-wise CELU activation.</summary>
        public Tensor<T> Celu(float alpha = 1.0f)
            => (Tensor<T>)OnnxOp.Celu(this, alpha == 1.0f ? null : alpha);

        /// <summary>Center-crops or pads to the given dimensions (ONNX CenterCropPad).</summary>
        public Tensor<T> CenterCropPad(Vector<int64> newDims, long[]? axes = null)
            => (Tensor<T>)OnnxOp.CenterCropPad(this, newDims, axes);

        /// <summary>Center-crops or pads to the given dimensions (ONNX CenterCropPad).</summary>
        public Tensor<T> CenterCropPad(Vector<int32> newDims, long[]? axes = null)
            => (Tensor<T>)OnnxOp.CenterCropPad(this, newDims, axes);

        /// <summary>Element-wise cosine.</summary>
        public Tensor<T> Cos()
            => (Tensor<T>)OnnxOp.Cos(this);

        /// <summary>Element-wise hyperbolic cosine.</summary>
        public Tensor<T> Cosh()
            => (Tensor<T>)OnnxOp.Cosh(this);

        /// <summary>Cumulative sum along <paramref name="axis"/>.</summary>
        public Tensor<T> CumSum<V>(Scalar<V> axis, bool exclusive = false, bool reverse = false) where V : IndexLike
            => (Tensor<T>)OnnxOp.CumSum(this, axis, exclusive, reverse);

        /// <summary>Rearranges channel data into spatial blocks (ONNX DepthToSpace).</summary>
        public Tensor<T> DepthToSpace(long blockSize, DepthColumnRowMode mode = DepthColumnRowMode.DCR)
            => (Tensor<T>)OnnxOp.DepthToSpace(this, blockSize, mode);

        //public Tensor<T> Dropout<V>(Scalar<V> ratio, Scalar<bit> trainingMode, long? seed = null) where V : FloatLike
        //    => (Tensor<T>)OnnxOp.Dropout(this, ratio, trainingMode, seed);

        /// <summary>Element-wise ELU activation.</summary>
        public Tensor<T> Elu(float alpha = 1.0f)
            => (Tensor<T>)OnnxOp.Elu(this, alpha == 1.0f ? null : alpha);

        /// <summary>Element-wise GELU activation.</summary>
        public Tensor<T> Gelu(GeluApproximate approximate = GeluApproximate.None)
            => (Tensor<T>)OnnxOp.Gelu(this, approximate);

        /// <summary>Element-wise leaky ReLU activation.</summary>
        public Tensor<T> LeakyRelu(float alpha = 0.01f)
            => (Tensor<T>)OnnxOp.LeakyRelu(this, alpha);

        /// <summary>Element-wise ReLU activation.</summary>
        public Tensor<T> Relu()
            => (Tensor<T>)OnnxOp.Relu(this);

        /// <summary>Element-wise SELU activation.</summary>
        public Tensor<T> Selu(float alpha = 1.67326319217681884765625f, float gamma = 1.0507010221481323242187f)
            => (Tensor<T>)OnnxOp.Selu(this, alpha, gamma);

        /// <summary>Element-wise sine.</summary>
        public Tensor<T> Sin()
            => (Tensor<T>)OnnxOp.Sin(this);

        /// <summary>Element-wise hyperbolic sine.</summary>
        public Tensor<T> Sinh()
            => (Tensor<T>)OnnxOp.Sinh(this);

        /// <summary>Element-wise tangent.</summary>
        public Tensor<T> Tan()
            => (Tensor<T>)OnnxOp.Tan(this);

        /// <summary>Element-wise hyperbolic tangent.</summary>
        public Tensor<T> Tanh()
            => (Tensor<T>)OnnxOp.Tanh(this);

        /// <summary>Element-wise power.</summary>
        public Tensor<T> Pow<T1>(Tensor<T1> power) where T1 : IVarType
            => (Tensor<T>)OnnxOp.Pow(this, power);

        /// <summary>Element-wise natural logarithm.</summary>
        public Tensor<T> Ln()
            => (Tensor<T>)OnnxOp.Log(this);

        /// <summary>Element-wise square root.</summary>
        public Tensor<T> Sqrt()
            => (Tensor<T>)OnnxOp.Sqrt(this);

        /// <summary>Element-wise reciprocal.</summary>
        public Tensor<T> Reciprocal()
            => (Tensor<T>)OnnxOp.Reciprocal(this);

        /// <summary>Element-wise error function.</summary>
        public Tensor<T> Erf()
            => (Tensor<T>)OnnxOp.Erf(this);

        /// <summary>Element-wise sign.</summary>
        public Tensor<T> Sign()
            => (Tensor<T>)OnnxOp.Sign(this);
        /// <summary>Concatenates this tensor with <paramref name="others"/> along <paramref name="axis"/>.</summary>
        public Tensor<T> Concat(long axis, params Tensor<T>[] others)
            => (Tensor<T>)OnnxOp.Concat([this, .. others], axis);

        /// <summary>Reshapes to <paramref name="newShape"/>; a 0 entry copies the corresponding input dimension unless <paramref name="allowZero"/> is true.</summary>
        public Tensor<T> Reshape(Vector<int64> newShape, bool allowZero = false)
            => (Tensor<T>)OnnxOp.Reshape(this, newShape, allowZero);

        /// <summary>Permutes the dimensions; with no arguments, reverses them.</summary>
        public Tensor<T> Transpose(params long[] newDims)
            => (Tensor<T>)OnnxOp.Transpose(this, newDims.Length == 0 ? null : newDims);

        /// <summary>Pads using separate begin (<paramref name="outerPads"/>) and end (<paramref name="innerPads"/>) pad counts per axis.</summary>
        public Tensor<T> Pad(PadMode mode, Vector<int64> outerPads, Vector<int64> innerPads, Scalar<T>? val = null, Vector<int64>? axes = null)
            => (Tensor<T>)OnnxOp.Pad(this, (Vector<int64>)[.. outerPads, .. innerPads], val, axes, mode);

        /// <summary>Pads using an ONNX-style pads vector (begin counts for all axes, then end counts).</summary>
        public Tensor<T> Pad(PadMode mode, Vector<int64> pads, Scalar<T>? val = null, Vector<int64>? axes = null)
            => (Tensor<T>)OnnxOp.Pad(this, pads, val, axes, mode);

        /// <summary>Broadcasts to the given shape.</summary>
        public Tensor<T> Expand(params long[] shape)
            => this.Expand(Vector(shape));

        /// <summary>Broadcasts to the given shape (ONNX Expand).</summary>
        public Tensor<T> Expand(Vector<int64> shape)
            => (Tensor<T>)OnnxOp.Expand(this, shape);

        /// <summary>Element-wise sigmoid.</summary>
        public Tensor<T> Sigmoid()
            => (Tensor<T>)OnnxOp.Sigmoid(this);

        /// <summary>One-hot encoding of the maximum along <paramref name="axis"/> (ONNX Hardmax).</summary>
        public Tensor<T> Hardmax(long? axis = null)
            => (Tensor<T>)OnnxOp.Hardmax(this, axis);

        /// <summary>Element-wise hard sigmoid.</summary>
        public Tensor<T> HardSigmoid(float? alpha = null, float? beta = null)
            => (Tensor<T>)OnnxOp.HardSigmoid(this, alpha, beta);

        /// <summary>Element-wise hard swish.</summary>
        public Tensor<T> HardSwish()
            => (Tensor<T>)OnnxOp.HardSwish(this);

        /// <summary>Element-wise infinity test, yielding a bit tensor.</summary>
        public Tensor<bit> IsInf(bool detectNegative = true, bool detectPositive = true)
            => (Tensor<bit>)OnnxOp.IsInf(this,
                detectNegative ? null : detectNegative,
                detectPositive ? null : detectPositive);

        /// <summary>Element-wise NaN test, yielding a bit tensor.</summary>
        public Tensor<bit> IsNaN()
            => (Tensor<bit>)OnnxOp.IsNaN(this);

        /// <summary>Log-softmax along <paramref name="axis"/>.</summary>
        public Tensor<T> LogSoftmax(long? axis = null)
            => (Tensor<T>)OnnxOp.LogSoftmax(this, axis);

        /// <summary>Normalizes to zero mean and unit variance over <paramref name="axes"/>.</summary>
        public Tensor<T> MeanVarianceNormalization(long[]? axes = null)
            => (Tensor<T>)OnnxOp.MeanVarianceNormalization(this, axes);

        /// <summary>Element-wise Mish activation.</summary>
        public Tensor<T> Mish()
            => (Tensor<T>)OnnxOp.Mish(this);

        /// <summary>Element-wise rounding to the nearest integer (half to even).</summary>
        public Tensor<T> Round()
            => (Tensor<T>)OnnxOp.Round(this);

        /// <summary>Element-wise shrink thresholding (ONNX Shrink).</summary>
        public Tensor<T> Shrink(float? bias = null, float? lambd = null)
            => (Tensor<T>)OnnxOp.Shrink(this, bias, lambd);

        /// <summary>Element-wise softplus.</summary>
        public Tensor<T> Softplus()
            => (Tensor<T>)OnnxOp.Softplus(this);

        /// <summary>Element-wise softsign.</summary>
        public Tensor<T> Softsign()
            => (Tensor<T>)OnnxOp.Softsign(this);

        /// <summary>Element-wise thresholded ReLU.</summary>
        public Tensor<T> ThresholdedRelu(float? alpha = null)
            => (Tensor<T>)OnnxOp.ThresholdedRelu(this, alpha);

        /// <summary>Top <paramref name="k"/> values and their indices along <paramref name="axis"/>.</summary>
        public (Tensor<T> topK, Tensor<int64> indices) TopK(long k, long axis = -1, bool largest = true, bool sorted = true)
            => this.TopK(Globals.Scalar(k), axis, largest, sorted);

        /// <summary>Top <paramref name="k"/> values and their indices along <paramref name="axis"/>.</summary>
        public (Tensor<T> topK, Tensor<int64> indices) TopK(Scalar<int64> k, long axis = -1, bool largest = true, bool sorted = true)
         => NN.TopK(this, k, axis, largest ? null : largest, sorted ? null : sorted);

        /// <summary>Writes <paramref name="values"/> at <paramref name="indices"/> into a copy of this tensor (ONNX ScatterND).</summary>
        public Tensor<T> ScatterND(Tensor<int64> indices, Tensor<T> values, ScatterNDReduction? reduceMode = ScatterNDReduction.None)
        {
            return (Tensor<T>)OnnxOp.ScatterND(this, indices, values, reduceMode);
        }

        /// <summary>Element-wise clamping to [min, max] given as primitive constants.</summary>
        public Tensor<T> Clip(PrimitiveParam min, PrimitiveParam max)
            => (Tensor<T>)OnnxOp.Clip(this, (Scalar<T>)min, (Scalar<T>)max);

        /// <summary>Element-wise clamping to [min, max].</summary>
        public Tensor<T> Clip(Scalar<T> min, Scalar<T> max)
            => (Tensor<T>)OnnxOp.Clip(this, min, max);

        /// <summary>Selects elements where <paramref name="condition"/> is true, flattening to a rank-1 result.</summary>
        public Vector<T> Compress(Vector<bit> condition)
            => (Vector<T>)OnnxOp.Compress(this, condition, null);

        /// <summary>Selects slices along <paramref name="axis"/> where <paramref name="condition"/> is true.</summary>
        public Tensor<T> Compress(Vector<bit> condition, long axis)
            => (Tensor<T>)OnnxOp.Compress(this, condition, axis);

        /// <summary>Element-wise exponential.</summary>
        public Tensor<T> Exp()
            => (Tensor<T>)OnnxOp.Exp(this);

        /// <summary>Flattens to 2-D: dimensions before <paramref name="axis"/> collapse into the first dimension, the rest into the second.</summary>
        public Tensor<T> Flatten(long axis = 1)
            => (Tensor<T>)OnnxOp.Flatten(this, axis);

        #endregion

        #region Equal and HashCode

        /// <summary>Reference equality; element-wise comparison is provided by the equality operators.</summary>
        public override bool Equals(object? obj)
         => Object.ReferenceEquals(this, obj);

        /// <summary>Identity-based hash code, consistent with the reference-equality semantics of <see cref="Equals(object?)"/>.</summary>
        public override int GetHashCode()
            => base.GetHashCode();

        #endregion
    }


    /// <summary>
    /// Element-wise selection (Where) extensions available on bit-valued condition tensors.
    /// </summary>
    public static class TensorTypeSpecificExtensions
    {
        /// <summary>Element-wise selection: takes <paramref name="whenTrue"/> where <paramref name="cond"/> is true, otherwise <paramref name="whenFalse"/>.</summary>
        public static Tensor<V> Where<V>(this Tensor<bit> cond, Tensor<V> whenTrue, Tensor<V> whenFalse)
            where V : IVarType
            => (Tensor<V>)OnnxOp.Where(cond, whenTrue, whenFalse);

        /// <summary>Element-wise selection: takes <paramref name="whenTrue"/> where <paramref name="cond"/> is true, otherwise <paramref name="whenFalse"/>.</summary>
        public static Vector<V> Where<V>(this Vector<bit> cond, Vector<V> whenTrue, Vector<V> whenFalse)
            where V : IVarType
            => ((Tensor<V>)OnnxOp.Where(cond, whenTrue, whenFalse)).Vec();

        /// <summary>Element-wise selection: takes <paramref name="whenTrue"/> where <paramref name="cond"/> is true, otherwise <paramref name="whenFalse"/>.</summary>
        public static Scalar<V> Where<V>(this Scalar<bit> cond, Scalar<V> whenTrue, Scalar<V> whenFalse)
            where V : IVarType
            => ((Tensor<V>)OnnxOp.Where(cond, whenTrue, whenFalse)).Scalar();
    }
}

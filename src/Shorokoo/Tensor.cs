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
    public interface ITensor : IValue
    {
        /// <summary>The shape of this tensor as a symbolic rank-1 tensor of dimension sizes.</summary>
        public Vector<int64> TShape { get; }

        /// <summary>Statically known rank (number of dimensions), or null when not known at graph-construction time.</summary>
        public int? Rank { get; }

        /// <summary>Casts the element type to <typeparamref name="T"/>.</summary>
        public Tensor<T> Cast<T>(bool saturate = true) where T : IVarType;

        /// <summary>Reinterprets this tensor as a rank-1 vector.</summary>
        public IVector Vec();

        /// <summary>Reinterprets this tensor as a rank-0 scalar.</summary>
        public IScalar Scalar();

    }

    /// <summary>
    /// Value-type handle for a tensor — the user-facing <c>Tensor&lt;T&gt;</c>. Holds a
    /// <see cref="Variable"/> directly (value-copy semantics for the Module DSL) and
    /// carries the full op/operator surface. This pass only makes mutation possible — behaviour is
    /// unchanged (de-facto immutable). A defaulted handle materialises an empty rank-1 vector.
    /// </summary>
    [CollectionBuilder(typeof(Shorokoo.Core.TensorCollectionBuilder), nameof(Shorokoo.Core.TensorCollectionBuilder.Create))]
    public partial struct Tensor<T> : ITensor, System.Collections.Generic.IEnumerable<Shorokoo.Core.TensorExpressionHelper<T>> where T : IVarType
    {
        private Variable? inner;
        // The backing graph node, materialising the established default (per dtype/rank) for a defaulted handle.
        internal Variable Immutable => inner ?? InternalGlobals.DefaultVariable(typeof(Tensor<T>));

        private static readonly DType? expectedDType = OnnxUtils.GetDType(typeof(T));
        public static implicit operator Tensor<T>(Variable imm)
        {
            IValue.RequireKind(imm, DataStructure.Tensor);
            IValue.RequireDType(imm, expectedDType);
            return new Tensor<T> { inner = imm };
        }
        public static implicit operator Variable(Tensor<T> h) => h.Immutable;

        // Convert to the backing graph node, materialising the established default for a defaulted handle.
        Variable IValue.ToVariable() => Immutable;


        // ITensor contract — forward to the backing Variable.
        public int? Rank => Immutable.Rank;
        public Vector<int64> DShape => Immutable.DShape;
        public Vector<int64> TShape => Immutable.TShape;
        public Scalar<int64> TRank => Immutable.TRank;
        IVector ITensor.Vec() => Vec();
        IScalar ITensor.Scalar() => Scalar();

        // IValue surface — forward to the backing Variable.
        public Node OwningNode => Immutable.OwningNode;
        public DType Type => Immutable.Type;
        public Function? ModuleFn => Immutable.ModuleFn;
        public TensorKey Key => Immutable.Key;
        public string UniqueName => Immutable.UniqueName;
        public bool IsValid { get => Immutable.IsValid; set => Immutable.IsValid = value; }
#pragma warning disable CS0618
        string? IValue.FriendlyName => ((IValue)Immutable).FriendlyName;
#pragma warning restore CS0618

        // user-facing reinterpret casts: the Variable→handle operators validate rank
        // (pass-through on a match, refine an unknown rank, throw on a known mismatch —
        // e.g. a rank-0 tensor cannot be reinterpreted as a Vector).
        public Vector<T> Vec() => (Vector<T>)Immutable;
        public Scalar<T> Scalar() => (Scalar<T>)Immutable;

        // == builds an Equal graph node (see operators); Equals/GetHashCode use the wrapped node.
        public override bool Equals(object? obj) => obj is Tensor<T> t && Equals(inner, t.inner);
        public override int GetHashCode() => inner?.GetHashCode() ?? 0;

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
        #region Onnx Cast Operators

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

            return OnnxOp.Add(left, right);
        }

        /// <summary>Element-wise subtraction.</summary>
        public static Tensor<T> operator -(Tensor<T> left, Tensor<T> right)
        {
            return OnnxOp.Sub(left, right);
        }

        /// <summary>Element-wise multiplication.</summary>
        public static Tensor<T> operator *(Tensor<T> left, Tensor<T> right)
        {
            return OnnxOp.Mul(left, right);
        }

        /// <summary>Element-wise division.</summary>
        public static Tensor<T> operator /(Tensor<T> left, Tensor<T> right)
        {
            return OnnxOp.Div(left, right);
        }

        /// <summary>Element-wise modulo.</summary>
        public static Tensor<T> operator %(Tensor<T> left, Tensor<T> right)
            => OnnxOp.Mod(left, right);

        /// <summary>Element-wise XOR: logical for bit tensors, bitwise otherwise.</summary>
        public static Tensor<T> operator ^(Tensor<T> left, Tensor<T> right)
            => typeof(T) == typeof(bit) ?
                OnnxOp.Xor(left, right) :
                OnnxOp.BitwiseXor(left, right);

        /// <summary>Element-wise AND: logical for bit tensors, bitwise otherwise.</summary>
        public static Tensor<T> operator &(Tensor<T> left, Tensor<T> right)
            => typeof(T) == typeof(bit) ?
                OnnxOp.And(left, right) :
                OnnxOp.BitwiseAnd(left, right);

        /// <summary>Element-wise OR: logical for bit tensors, bitwise otherwise.</summary>
        public static Tensor<T> operator |(Tensor<T> left, Tensor<T> right)
            => typeof(T) == typeof(bit) ?
                OnnxOp.Or(left, right) :
                OnnxOp.BitwiseOr(left, right);

        /// <summary>Element-wise left bit-shift.</summary>
        public static Tensor<T> operator <<(Tensor<T> left, Tensor<T> right)
            => OnnxOp.BitShift(left, right, BitShiftDirection.Left);

        /// <summary>Element-wise right bit-shift.</summary>
        public static Tensor<T> operator >>(Tensor<T> left, Tensor<T> right)
            => OnnxOp.BitShift(left, right, BitShiftDirection.Right);

        /// <summary>Element-wise negation.</summary>
        public static Tensor<T> operator -(Tensor<T> input)
        {
            return OnnxOp.Neg(input);
        }

        /// <summary>Element-wise NOT: logical for bit tensors, bitwise otherwise.</summary>
        public static Tensor<T> operator !(Tensor<T> input)
            => typeof(T) == typeof(bit) ?
                OnnxOp.Not(input) :
                OnnxOp.BitwiseNot(input);

        /// <summary>Element-wise greater-than, yielding a bit tensor.</summary>
        public static Tensor<bit> operator >(Tensor<T> left, Tensor<T> right) => OnnxOp.Greater(left, right);

        /// <summary>Element-wise greater-or-equal, yielding a bit tensor.</summary>
        public static Tensor<bit> operator >=(Tensor<T> left, Tensor<T> right) => OnnxOp.GreaterOrEqual(left, right);

        /// <summary>Element-wise less-than, yielding a bit tensor.</summary>
        public static Tensor<bit> operator <(Tensor<T> left, Tensor<T> right) => OnnxOp.Less(left, right);

        /// <summary>Element-wise less-or-equal, yielding a bit tensor.</summary>
        public static Tensor<bit> operator <=(Tensor<T> left, Tensor<T> right) => OnnxOp.LessOrEqual(left, right);

        /// <summary>Element-wise equality, yielding a bit tensor (not reference equality).</summary>
        public static Tensor<bit> operator ==(Tensor<T> left, Tensor<T> right) => OnnxOp.Equal(left, right);

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
            return OnnxOp.Split(this, null, axis: axis, numOutputs: numOutputs, variadicOutputCount: numOutputs).Select(v => (Tensor<T>)v).ToArray();
        }

        /// <summary>Splits along <paramref name="axis"/> into parts of the given sizes.</summary>
        public Tensor<T>[] Split(long[] splits, long axis = 0)
        {
            return OnnxOp.Split(this, Vector(splits), axis: axis, numOutputs: null, variadicOutputCount: splits.Length).Select(v => (Tensor<T>)v).ToArray();
        }

        /// <summary>Splits along <paramref name="axis"/>: by <paramref name="splits"/> sizes when given, otherwise into <paramref name="numOutputs"/> equal parts.</summary>
        public Tensor<T>[] Split(Vector<int64>? splits, long axis, long numOutputs)
        {
            return OnnxOp.Split(this, splits, axis: axis, numOutputs: splits is null ? numOutputs : null, variadicOutputCount: numOutputs).Select(v => (Tensor<T>)v).ToArray();
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
            return OnnxOp.Resize(
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
            return OnnxOp.Resize(
                this, null, scales, null, antiaAlias, axes, transformMode, cubicCoefficient, excludeOutside, null,
                null, mode, nearestMode);
        }

        /// <summary>Removes size-1 dimensions; when <paramref name="axes"/> is null, all size-1 dimensions are removed.</summary>
        public Tensor<T> Squeeze(Vector<int64>? axes = null)
        {
            // Absent axes propagate as an absent (optional) input: ONNX then squeezes ALL
            // size-1 dims. (Previously substituted axes=[-1] — squeeze only the LAST dim,
            // and an ORT shape-inference error when that dim isn't 1.)
            return OnnxOp.Squeeze(this, axes);
        }

        /// <summary>Slices along the given axes using start/end indices and optional steps (ONNX Slice).</summary>
        public Tensor<T> Slice(Vector<int64> start, Vector<int64> end, Vector<int64>? axes = null, Vector<int64>? steps = null)
            => OnnxOp.Slice(this, start, end, axes, steps);

        /// <summary>Softmax normalization along <paramref name="axis"/> (defaults to the last axis).</summary>
        public Tensor<T> Softmax(long? axis = null)
            => OnnxOp.Softmax(this, axis);

        /// <summary>Inserts a size-1 dimension at <paramref name="axis"/>.</summary>
        public Tensor<T> Unsqueeze(long axis)
            => OnnxOp.Unsqueeze(this, Vector(axis));

        /// <summary>Inserts size-1 dimensions at the given axes.</summary>
        public Tensor<T> Unsqueeze(Vector<int64> axes)
            => OnnxOp.Unsqueeze(this, axes);

        /// <summary>Appends a trailing size-1 dimension (axis -1).</summary>
        public Tensor<T> Unsqueeze()
            => OnnxOp.Unsqueeze(this, Vector(-1L));

        /// <summary>The shape - optionally the dims[start:end] slice of it - as an in-graph vector.</summary>
        public Vector<int64> ShapeTensor(long? start = null, long? end = null)
            // OnnxOp.Shape declares (data, end, start) — name the args; passing positionally
            // swapped start/end (e.g. ShapeTensor(1) sliced dims[:1] instead of dims[1:]).
            // The Variable→Vector<int64> operator validates the rank-1 result.
            => OnnxOp.Shape(this, end: end, start: start);

        /// <summary>The element count: the product of the dimensions, optionally restricted to dims[start:end].</summary>
        public Scalar<int64> SizeTensor(long? start = null, long? end = null)
            => this.ShapeTensor(start, end).Reduce(ReduceKind.Prod);

        /// <summary>The size of dimension <paramref name="axis"/> as a symbolic scalar.</summary>
        public Scalar<int64> DimTensor(long axis)
            => this.ShapeTensor()[axis];

        /// <summary>Gathers entries along <paramref name="axis"/> using the given indices (ONNX Gather).</summary>
        public Tensor<T> Gather(Tensor<int64> indices, long? axis)
            => OnnxOp.Gather(this, indices, axis);

        /// <summary>Gathers slices using multi-dimensional indices (ONNX GatherND).</summary>
        public Tensor<T> GatherND(Tensor<int64> indices, long? batchDims)
            => OnnxOp.GatherND(this, indices, batchDims);

        /// <summary>Casts the element type to <typeparamref name="V"/>; returns this tensor unchanged when the types already match.</summary>
        public Tensor<V> Cast<V>(bool saturate = true) where V : IVarType
            => typeof(V) == typeof(T) ?
                (Tensor<V>)(object)this :
                OnnxOp.Cast(this, saturate ? null : saturate, OnnxUtils.GetDType<V>());

        /// <summary>Creates a tensor of the given shape filled with the scalar value <paramref name="val"/> (ONNX ConstantOfShape).</summary>
        public static Tensor<T> Fill(Vector<int64> shape, TensorData val)
        {
            // ConstantOfShape expects:
            // - shape: A 1D tensor indicating the shape of the output
            // - val: A scalar TensorData that will be broadcasted to fill the entire output
            // The issue was using pre-shaped TensorData instead of scalar fill value
            return OnnxOp.ConstantOfShape(shape, val);
        }
        
        /// <summary>Reduction (e.g. sum, mean, max) over <paramref name="axes"/> - all axes when null - keeping reduced dimensions by default.</summary>
        public Tensor<T> Reduce(ReduceKind reduceKind, Vector<int64>? axes = null, bool keepDims = true)
            => NN.Reduce(reduceKind, this, axes, keepDims, false);

        /// <summary>Tiles the tensor by repeating it <paramref name="repeats"/> times along each axis.</summary>
        public Tensor<T> Tile(Tensor<int64> repeats)
            => OnnxOp.Tile(this, repeats);

        /// <summary>Element-wise maximum of this tensor and <paramref name="others"/>.</summary>
        public Tensor<T> Max(params Tensor<T>[] others)
            => OnnxOp.Max([this, .. others]);

        /// <summary>Element-wise minimum of this tensor and <paramref name="others"/>.</summary>
        public Tensor<T> Min(params Tensor<T>[] others)
            => OnnxOp.Min([this, .. others]);

        /// <summary>Element-wise floor.</summary>
        public Tensor<T> Floor()
            => OnnxOp.Floor(this);

        /// <summary>Matrix product with <paramref name="other"/> (ONNX MatMul).</summary>
        public Tensor<T> MatMul(Tensor<T> other)
            => OnnxOp.MatMul(this, other);

        /// <summary>Element-wise absolute value.</summary>
        public Tensor<T> Abs()
            => OnnxOp.Abs(this);

        /// <summary>Element-wise arccosine.</summary>
        public Tensor<T> Acos()
            => OnnxOp.Acos(this);

        /// <summary>Element-wise inverse hyperbolic cosine.</summary>
        public Tensor<T> Acosh()
            => OnnxOp.Acosh(this);

        /// <summary>Element-wise arcsine.</summary>
        public Tensor<T> Asin()
            => OnnxOp.Asin(this);

        /// <summary>Element-wise inverse hyperbolic sine.</summary>
        public Tensor<T> Asinh()
            => OnnxOp.Asinh(this);

        /// <summary>Element-wise arctangent.</summary>
        public Tensor<T> Atan()
            => OnnxOp.Atan(this);

        /// <summary>Element-wise inverse hyperbolic tangent.</summary>
        public Tensor<T> Atanh()
            => OnnxOp.Atanh(this);

        /// <summary>Indices of the maximum values along <paramref name="axis"/>.</summary>
        public Tensor<int64> ArgMax(long axis = 0, bool keepdims = false, bool selectLastIndex = false)
            => OnnxOp.ArgMax(this, axis == 0 ? null : axis, keepdims ? null : keepdims, selectLastIndex ? selectLastIndex : null);

        /// <summary>Indices of the minimum values along <paramref name="axis"/>.</summary>
        public Tensor<int64> ArgMin(long axis = 0, bool keepdims = false, bool selectLastIndex = false)
            => OnnxOp.ArgMin(this, axis == 0 ? null : axis, keepdims ? null : keepdims, selectLastIndex ? selectLastIndex : null);

        /// <summary>Average pooling with the given kernel shape (ONNX AveragePool).</summary>
        public Tensor<T> AveragePool(long[] kernelShape, RoundMode roundMode = RoundMode.Floor, bool countIncludePad = false, long[]? dilations = null, long[]? pads = null, long[]? strides = null)
            => OnnxOp.AveragePool(this, null, roundMode == RoundMode.Floor ? null : true, countIncludePad, dilations, kernelShape, pads, strides);

        /// <summary>Batch normalization using the given scale, bias, mean, and variance (ONNX BatchNormalization).</summary>
        public Tensor<T> BatchNormalization<T1, T2>(Vector<T1> scale, Vector<T1> bias, Vector<T2> mean, Vector<T2> variance, float epsilon = 1e-05f, float momentum = 0.9f, bool trainingMode = false)
                where T1 : FloatLike where T2 : FloatLike
        {
            var retval = OnnxOp.BatchNormalization(this, scale, bias, mean, variance, epsilon == 1e-05f ? null : epsilon, momentum == 0.9f ? null : momentum, trainingMode == false ? null : trainingMode);
            return (Variable)retval;
        }

        /// <summary>Batch normalization that also returns the updated running mean and variance.</summary>
        public (Tensor<T> y, Vector<T2> runningMean, Vector<T2> runningVariance) BatchNormalizationFullOuputs<T1, T2>(Vector<T1> scale, Vector<T1> bias, Vector<T2> mean, Vector<T2> variance, float epsilon = 1e-05f, float momentum = 0.9f, bool trainingMode = false)
                where T1 : FloatLike where T2 : FloatLike
        {
            var retval = OnnxOp.BatchNormalizationFullOutputs(this, scale, bias, mean, variance, epsilon == 1e-05f ? null : epsilon, momentum == 0.9f ? null : momentum, trainingMode == false ? null : trainingMode);
            return ((Variable)retval.y, ((Tensor<T2>)retval.runningMean).Vec(), ((Tensor<T2>)retval.runningVariance).Vec());
        }

        /// <summary>Element-wise Bernoulli sampling, treating each element as a probability.</summary>
        public Tensor<T> Bernoulli(float? seed = null)
            => OnnxOp.Bernoulli(this, null, seed);

        /// <summary>Element-wise Bernoulli sampling, treating each element as a probability, with result element type <typeparamref name="V"/>.</summary>
        public Tensor<V> Bernoulli<V>(float? seed = null) where V : CommonLike
            => OnnxOp.Bernoulli(this, OnnxUtils.GetDType<V>(), seed);

        /// <summary>Element-wise ceiling.</summary>
        public Tensor<T> Ceiling()
            => OnnxOp.Ceil(this);

        /// <summary>Element-wise CELU activation.</summary>
        public Tensor<T> Celu(float alpha = 1.0f)
            => OnnxOp.Celu(this, alpha == 1.0f ? null : alpha);

        /// <summary>Center-crops or pads to the given dimensions (ONNX CenterCropPad).</summary>
        public Tensor<T> CenterCropPad(Vector<int64> newDims, long[]? axes = null)
            => OnnxOp.CenterCropPad(this, newDims, axes);

        /// <summary>Center-crops or pads to the given dimensions (ONNX CenterCropPad).</summary>
        public Tensor<T> CenterCropPad(Vector<int32> newDims, long[]? axes = null)
            => OnnxOp.CenterCropPad(this, newDims, axes);

        /// <summary>Element-wise cosine.</summary>
        public Tensor<T> Cos()
            => OnnxOp.Cos(this);

        /// <summary>Element-wise hyperbolic cosine.</summary>
        public Tensor<T> Cosh()
            => OnnxOp.Cosh(this);

        /// <summary>Cumulative sum along <paramref name="axis"/>.</summary>
        public Tensor<T> CumSum<V>(Scalar<V> axis, bool exclusive = false, bool reverse = false) where V : IndexLike
            => OnnxOp.CumSum(this, axis, exclusive, reverse);

        /// <summary>Rearranges channel data into spatial blocks (ONNX DepthToSpace).</summary>
        public Tensor<T> DepthToSpace(long blockSize, DepthColumnRowMode mode = DepthColumnRowMode.DCR)
            => OnnxOp.DepthToSpace(this, blockSize, mode);

        //public Tensor<T> Dropout<V>(Scalar<V> ratio, Scalar<bit> trainingMode, long? seed = null) where V : FloatLike
        //    => OnnxOp.Dropout(this, ratio, trainingMode, seed);

        /// <summary>Element-wise ELU activation.</summary>
        public Tensor<T> Elu(float alpha = 1.0f)
            => OnnxOp.Elu(this, alpha == 1.0f ? null : alpha);

        /// <summary>Element-wise GELU activation.</summary>
        public Tensor<T> Gelu(GeluApproximate approximate = GeluApproximate.None)
            => OnnxOp.Gelu(this, approximate);

        /// <summary>Element-wise leaky ReLU activation.</summary>
        public Tensor<T> LeakyRelu(float alpha = 0.01f)
            => OnnxOp.LeakyRelu(this, alpha);

        /// <summary>Element-wise ReLU activation.</summary>
        public Tensor<T> Relu()
            => OnnxOp.Relu(this);

        /// <summary>Element-wise SELU activation.</summary>
        public Tensor<T> Selu(float alpha = 1.67326319217681884765625f, float gamma = 1.0507010221481323242187f)
            => OnnxOp.Selu(this, alpha, gamma);

        /// <summary>Element-wise sine.</summary>
        public Tensor<T> Sin()
            => OnnxOp.Sin(this);

        /// <summary>Element-wise hyperbolic sine.</summary>
        public Tensor<T> Sinh()
            => OnnxOp.Sinh(this);

        /// <summary>Element-wise tangent.</summary>
        public Tensor<T> Tan()
            => OnnxOp.Tan(this);

        /// <summary>Element-wise hyperbolic tangent.</summary>
        public Tensor<T> Tanh()
            => OnnxOp.Tanh(this);

        /// <summary>Element-wise power.</summary>
        public Tensor<T> Pow<T1>(Tensor<T1> power) where T1 : IVarType
            => OnnxOp.Pow(this, power);

        /// <summary>Element-wise natural logarithm.</summary>
        public Tensor<T> Ln()
            => OnnxOp.Log(this);

        /// <summary>Element-wise square root.</summary>
        public Tensor<T> Sqrt()
            => OnnxOp.Sqrt(this);

        /// <summary>Element-wise reciprocal.</summary>
        public Tensor<T> Reciprocal()
            => OnnxOp.Reciprocal(this);

        /// <summary>Element-wise error function.</summary>
        public Tensor<T> Erf()
            => OnnxOp.Erf(this);

        /// <summary>Element-wise sign.</summary>
        public Tensor<T> Sign()
            => OnnxOp.Sign(this);
        /// <summary>Concatenates this tensor with <paramref name="others"/> along <paramref name="axis"/>.</summary>
        public Tensor<T> Concat(long axis, params Tensor<T>[] others)
            => OnnxOp.Concat([this, .. others], axis);

        /// <summary>Reshapes to <paramref name="newShape"/>; a 0 entry copies the corresponding input dimension unless <paramref name="allowZero"/> is true.</summary>
        public Tensor<T> Reshape(Vector<int64> newShape, bool allowZero = false)
            => OnnxOp.Reshape(this, newShape, allowZero);

        /// <summary>Permutes the dimensions; with no arguments, reverses them.</summary>
        public Tensor<T> Transpose(params long[] newDims)
            => OnnxOp.Transpose(this, newDims.Length == 0 ? null : newDims);

        /// <summary>Pads using separate begin (<paramref name="outerPads"/>) and end (<paramref name="innerPads"/>) pad counts per axis.</summary>
        public Tensor<T> Pad(PadMode mode, Vector<int64> outerPads, Vector<int64> innerPads, Scalar<T>? val = null, Vector<int64>? axes = null)
            => OnnxOp.Pad(this, (Vector<int64>)[.. outerPads, .. innerPads], val, axes, mode);

        /// <summary>Pads using an ONNX-style pads vector (begin counts for all axes, then end counts).</summary>
        public Tensor<T> Pad(PadMode mode, Vector<int64> pads, Scalar<T>? val = null, Vector<int64>? axes = null)
            => OnnxOp.Pad(this, pads, val, axes, mode);

        /// <summary>Broadcasts to the given shape.</summary>
        public Tensor<T> Expand(params long[] shape)
            => this.Expand(Vector(shape));

        /// <summary>Broadcasts to the given shape (ONNX Expand).</summary>
        public Tensor<T> Expand(Vector<int64> shape)
            => OnnxOp.Expand(this, shape);

        /// <summary>Element-wise sigmoid.</summary>
        public Tensor<T> Sigmoid()
            => OnnxOp.Sigmoid(this);

        /// <summary>One-hot encoding of the maximum along <paramref name="axis"/> (ONNX Hardmax).</summary>
        public Tensor<T> Hardmax(long? axis = null)
            => OnnxOp.Hardmax(this, axis);

        /// <summary>Element-wise hard sigmoid.</summary>
        public Tensor<T> HardSigmoid(float? alpha = null, float? beta = null)
            => OnnxOp.HardSigmoid(this, alpha, beta);

        /// <summary>Element-wise hard swish.</summary>
        public Tensor<T> HardSwish()
            => OnnxOp.HardSwish(this);

        /// <summary>Element-wise infinity test, yielding a bit tensor.</summary>
        public Tensor<bit> IsInf(bool detectNegative = true, bool detectPositive = true)
            => OnnxOp.IsInf(this,
                detectNegative ? null : detectNegative,
                detectPositive ? null : detectPositive);

        /// <summary>Element-wise NaN test, yielding a bit tensor.</summary>
        public Tensor<bit> IsNaN()
            => OnnxOp.IsNaN(this);

        /// <summary>Log-softmax along <paramref name="axis"/>.</summary>
        public Tensor<T> LogSoftmax(long? axis = null)
            => OnnxOp.LogSoftmax(this, axis);

        /// <summary>Normalizes to zero mean and unit variance over <paramref name="axes"/>.</summary>
        public Tensor<T> MeanVarianceNormalization(long[]? axes = null)
            => OnnxOp.MeanVarianceNormalization(this, axes);

        /// <summary>Element-wise Mish activation.</summary>
        public Tensor<T> Mish()
            => OnnxOp.Mish(this);

        /// <summary>Element-wise rounding to the nearest integer (half to even).</summary>
        public Tensor<T> Round()
            => OnnxOp.Round(this);

        /// <summary>Element-wise shrink thresholding (ONNX Shrink).</summary>
        public Tensor<T> Shrink(float? bias = null, float? lambd = null)
            => OnnxOp.Shrink(this, bias, lambd);

        /// <summary>Element-wise softplus.</summary>
        public Tensor<T> Softplus()
            => OnnxOp.Softplus(this);

        /// <summary>Element-wise softsign.</summary>
        public Tensor<T> Softsign()
            => OnnxOp.Softsign(this);

        /// <summary>Element-wise thresholded ReLU.</summary>
        public Tensor<T> ThresholdedRelu(float? alpha = null)
            => OnnxOp.ThresholdedRelu(this, alpha);

        /// <summary>Top <paramref name="k"/> values and their indices along <paramref name="axis"/>.</summary>
        public (Tensor<T> topK, Tensor<int64> indices) TopK(long k, long axis = -1, bool largest = true, bool sorted = true)
            => this.TopK(Globals.Scalar(k), axis, largest, sorted);

        /// <summary>Top <paramref name="k"/> values and their indices along <paramref name="axis"/>.</summary>
        public (Tensor<T> topK, Tensor<int64> indices) TopK(Scalar<int64> k, long axis = -1, bool largest = true, bool sorted = true)
         => NN.TopK(this, k, axis, largest ? null : largest, sorted ? null : sorted);

        /// <summary>Writes <paramref name="values"/> at <paramref name="indices"/> into a copy of this tensor (ONNX ScatterND).</summary>
        public Tensor<T> ScatterND(Tensor<int64> indices, Tensor<T> values, ScatterNDReduction? reduceMode = ScatterNDReduction.None)
        {
            return OnnxOp.ScatterND(this, indices, values, reduceMode);
        }

        /// <summary>Element-wise clamping to [min, max] given as primitive constants.</summary>
        public Tensor<T> Clip(PrimitiveParam min, PrimitiveParam max)
            => OnnxOp.Clip(this, (Scalar<T>)min, (Scalar<T>)max);

        /// <summary>Element-wise clamping to [min, max].</summary>
        public Tensor<T> Clip(Scalar<T> min, Scalar<T> max)
            => OnnxOp.Clip(this, min, max);

        /// <summary>Selects elements where <paramref name="condition"/> is true, flattening to a rank-1 result.</summary>
        public Vector<T> Compress(Vector<bit> condition)
            => OnnxOp.Compress(this, condition, null);

        /// <summary>Selects slices along <paramref name="axis"/> where <paramref name="condition"/> is true.</summary>
        public Tensor<T> Compress(Vector<bit> condition, long axis)
            => OnnxOp.Compress(this, condition, axis);

        /// <summary>Element-wise exponential.</summary>
        public Tensor<T> Exp()
            => OnnxOp.Exp(this);

        /// <summary>Flattens to 2-D: dimensions before <paramref name="axis"/> collapse into the first dimension, the rest into the second.</summary>
        public Tensor<T> Flatten(long axis = 1)
            => OnnxOp.Flatten(this, axis);

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
            => OnnxOp.Where(cond, whenTrue, whenFalse);

        /// <summary>Element-wise selection: takes <paramref name="whenTrue"/> where <paramref name="cond"/> is true, otherwise <paramref name="whenFalse"/>.</summary>
        public static Vector<V> Where<V>(this Vector<bit> cond, Vector<V> whenTrue, Vector<V> whenFalse)
            where V : IVarType
            // The Variable→Vector<V> operator validates the rank-1 result.
            => OnnxOp.Where(cond, whenTrue, whenFalse);

        /// <summary>Element-wise selection: takes <paramref name="whenTrue"/> where <paramref name="cond"/> is true, otherwise <paramref name="whenFalse"/>.</summary>
        public static Scalar<V> Where<V>(this Scalar<bit> cond, Scalar<V> whenTrue, Scalar<V> whenFalse)
            where V : IVarType
            // The Variable→Scalar<V> operator validates the rank-0 result.
            => OnnxOp.Where(cond, whenTrue, whenFalse);
    }
}

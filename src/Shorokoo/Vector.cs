using Shorokoo.Core.Inference.Abstractions;
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
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.Net.Mail;
using System.Runtime.CompilerServices;
using static Shorokoo.Core.Nodes.Ops;
using static Shorokoo.Core.Nodes.AutoDiff.Ops;
using static Shorokoo.Globals;
using static Shorokoo.Core.InternalGlobals;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo
{
    /// <summary>Non-generic marker for rank-1 (vector) symbolic tensors.</summary>
    public interface IVector : ITensor
    {
    }

    [CollectionBuilder(typeof(Shorokoo.Core.TensorCollectionBuilder), nameof(Shorokoo.Core.TensorCollectionBuilder.CreateVector))]
    public partial struct Vector<T> : IVector, System.Collections.Generic.IEnumerable<VectorExpressionHelper<T>> where T : IVarType
    {
        private Variable? inner;
        // The backing graph node, materialising the established default (per dtype/rank) for a defaulted handle.
        internal Variable Immutable => inner ?? InternalGlobals.DefaultVariable(typeof(Vector<T>));

        private static readonly DType? expectedDType = OnnxUtils.GetDType(typeof(T));
        public static implicit operator Vector<T>(Variable imm)
        {
            IValue.RequireKind(imm, DataStructure.Tensor);
            IValue.RequireDType(imm, expectedDType);
            return new Vector<T> { inner = IValue.RequireRank(imm, 1) };
        }
        public static implicit operator Variable(Vector<T> h) => h.Immutable;
        public static implicit operator Tensor<T>(Vector<T> h) => h.Immutable;

        // Convert to the backing graph node, materialising the established default for a defaulted handle.
        Variable IValue.ToVariable() => Immutable;

        // ITensor contract — forward to the backing Variable.
        public int? Rank => Immutable.Rank;
        public Vector<int64> DShape => Immutable.DShape;
        public Vector<int64> TShape => Immutable.TShape;
        public Scalar<int64> TRank => Immutable.TRank;
        public Vector<T> Vec() => (Vector<T>)Immutable;     // already rank-1; validates
        public Scalar<T> Scalar() => (Scalar<T>)Immutable;  // throws: a rank-1 vector is not a scalar
        IVector ITensor.Vec() => Vec();
        IScalar ITensor.Scalar() => Scalar();
        Tensor<V> ITensor.Cast<V>(bool saturate) => Immutable.Cast<V>(saturate);

        public Node OwningNode => Immutable.OwningNode;
        public DType Type => Immutable.Type;
        public Function? ModuleFn => Immutable.ModuleFn;
        public TensorKey Key => Immutable.Key;
        public string UniqueName => Immutable.UniqueName;
        public bool IsValid { get => Immutable.IsValid; set => Immutable.IsValid = value; }
#pragma warning disable CS0618
        string? IValue.FriendlyName => ((IValue)Immutable).FriendlyName;
#pragma warning restore CS0618
        public override bool Equals(object? obj) => obj is Vector<T> t && Equals(inner, t.inner);
        public override int GetHashCode() => inner?.GetHashCode() ?? 0;

        #region Unit and Empty vectors

        private static Vector<T>? unit;

        /// <summary>
        /// A tensor of shape (1,) containing a single element of value 1 (or true).
        /// </summary>
        public static Vector<T> Unit
        {
            get
            {
                if (Vector<T>.unit is null)
                {
                    var type = OnnxUtils.GetDType<T>();
                    if (type == DType.BFloat16) unit = (Vector<T>)(object)Vector(BFloat16.One);
                    else if (type == DType.Float16) unit = (Vector<T>)(object)Vector(Float16.One);
                    else if (type == DType.Float32) unit = (Vector<T>)(object)Vector(1f);
                    else if (type == DType.Float64) unit = (Vector<T>)(object)Vector(1d);
                    else if (type == DType.Int4) 
                        throw new UnsupportedDTypeException(ErrorCodes.VT001, type.ToString(), "Unit Vector", "Int4 precision is not supported for unit vector creation");
                    else if (type == DType.Int8) unit = (Vector<T>)(object)Vector((sbyte)1);
                    else if (type == DType.Int16) unit = (Vector<T>)(object)Vector((Int16)1);
                    else if (type == DType.Int32) unit = (Vector<T>)(object)Vector((int)1);
                    else if (type == DType.Int64) unit = (Vector<T>)(object)Vector(1L);
                    else if (type == DType.UInt4) 
                        throw new UnsupportedDTypeException(ErrorCodes.VT002, type.ToString(), "Unit Vector", "UInt4 precision is not supported for unit vector creation");
                    else if (type == DType.UInt8) unit = (Vector<T>)(object)Vector((byte)1);
                    else if (type == DType.UInt16) unit = (Vector<T>)(object)Vector((ushort)1);
                    else if (type == DType.UInt32) unit = (Vector<T>)(object)Vector((uint)1);
                    else if (type == DType.UInt64) unit = (Vector<T>)(object)Vector((ulong)1);
                    else if (type == DType.String)
                        throw new UnsupportedDTypeException(ErrorCodes.VT003, type.ToString(), "Unit Vector", "String type is not supported for unit vector creation");
                    else if (type == DType.Bool) unit = (Vector<T>)(object)Vector(true);
                    else if (type == DType.Complex64) 
                        throw new UnsupportedDTypeException(ErrorCodes.VT004, type.ToString(), "Unit Vector", "Complex64 numbers are not supported for unit vector creation");
                    else if (type == DType.Complex128) 
                        throw new UnsupportedDTypeException(ErrorCodes.VT005, type.ToString(), "Unit Vector", "Complex128 numbers are not supported for unit vector creation");
                    else if (type == DType.Invalid) unit = (Vector<T>)(object)Vector(1);
                }

                Debug.Assert(Vector<T>.unit is not null);
                return Vector<T>.unit!.Value;
            }
        }

        private static Vector<T>? empty;
        /// <summary>A cached vector of shape (0,) containing no elements.</summary>
        public static Vector<T> Empty
        {
            get
            {
                if (Vector<T>.empty is null)
                {
                    var type = OnnxUtils.GetDType<T>();
                    if (type == DType.BFloat16) empty = (Vector<T>)(object)Vector(new BFloat16[] { });
                    else if (type == DType.Float16) empty = (Vector<T>)(object)Vector(new Float16[] {});
                    else if (type == DType.Float32) empty = (Vector<T>)(object)Vector(new float[] {});
                    else if (type == DType.Float64) empty = (Vector<T>)(object)Vector(new double[] {});
                    else if (type == DType.Int4) 
                        throw new UnsupportedDTypeException(ErrorCodes.VT006, type.ToString(), "Empty Vector", "Int4 precision is not supported for empty vector creation");
                    else if (type == DType.Int8) empty = (Vector<T>)(object)Vector(new sbyte[] {});
                    else if (type == DType.Int16) empty = (Vector<T>)(object)Vector(new short[] {});
                    else if (type == DType.Int32) empty = (Vector<T>)(object)Vector(new int[] {});
                    else if (type == DType.Int64) empty = (Vector<T>)(object)Vector(new long[] {});
                    else if (type == DType.UInt4) 
                        throw new UnsupportedDTypeException(ErrorCodes.VT007, type.ToString(), "Empty Vector", "UInt4 precision is not supported for empty vector creation");
                    else if (type == DType.UInt8) empty = (Vector<T>)(object)Vector(new byte[] {});
                    else if (type == DType.UInt16) empty = (Vector<T>)(object)Vector(new ushort[] {});
                    else if (type == DType.UInt32) empty = (Vector<T>)(object)Vector(new uint[] {});
                    else if (type == DType.UInt64) empty = (Vector<T>)(object)Vector(new ulong[] {});
                    else if (type == DType.String)
                        throw new UnsupportedDTypeException(ErrorCodes.VT008, type.ToString(), "Empty Vector", "String type is not supported for empty vector creation");
                    else if (type == DType.Bool) empty = (Vector<T>)(object)Vector(new bool[] {});
                    else if (type == DType.Complex64) 
                        throw new UnsupportedDTypeException(ErrorCodes.VT009, type.ToString(), "Empty Vector", "Complex64 numbers are not supported for empty vector creation");
                    else if (type == DType.Complex128) 
                        throw new UnsupportedDTypeException(ErrorCodes.VT010, type.ToString(), "Empty Vector", "Complex128 numbers are not supported for empty vector creation");
                    else if (type == DType.Invalid) empty = (Vector<T>)(object)Vector(new int[] {});
                }

                Debug.Assert(Vector<T>.empty is not null);
                return Vector<T>.empty!.Value;
            }
        }

        #endregion
        #region Collection Expression Helpers

        IEnumerator IEnumerable.GetEnumerator() => this.GetEnumerator();

        /// <summary>Supports C# collection-expression composition by yielding a single helper that wraps this vector.</summary>
        public IEnumerator<VectorExpressionHelper<T>> GetEnumerator()
        {
            var asList = new List<VectorExpressionHelper<T>> { new VectorExpressionHelper<T>(this) };
            return ((IEnumerable<VectorExpressionHelper<T>>)asList).GetEnumerator();
        }

        #endregion
        #region Operator Overloads

        /// <summary>Element-wise addition.</summary>
        public static Vector<T> operator +(Vector<T> left, Vector<T> right) => ((Tensor<T>)left + (Tensor<T>)right).Vec();
        /// <summary>Element-wise addition with a scalar constant.</summary>
        public static Vector<T> operator +(Scalar<T> left, Vector<T> right) => ((Tensor<T>)left + (Tensor<T>)right).Vec();
        /// <summary>Element-wise addition with a scalar constant.</summary>
        public static Vector<T> operator +(Vector<T> left, Scalar<T> right) => ((Tensor<T>)left + (Tensor<T>)right).Vec();

        /// <summary>Element-wise subtraction.</summary>
        public static Vector<T> operator -(Vector<T> left, Vector<T> right) => ((Tensor<T>)left - (Tensor<T>)right).Vec();
        /// <summary>Element-wise subtraction with a scalar constant.</summary>
        public static Vector<T> operator -(Scalar<T> left, Vector<T> right) => ((Tensor<T>)left - (Tensor<T>)right).Vec();
        /// <summary>Element-wise subtraction with a scalar constant.</summary>
        public static Vector<T> operator -(Vector<T> left, Scalar<T> right) => ((Tensor<T>)left - (Tensor<T>)right).Vec();

        /// <summary>Element-wise multiplication.</summary>
        public static Vector<T> operator *(Vector<T> left, Vector<T> right) => ((Tensor<T>)left * (Tensor<T>)right).Vec();
        /// <summary>Element-wise multiplication with a scalar constant.</summary>
        public static Vector<T> operator *(Scalar<T> left, Vector<T> right) => ((Tensor<T>)left * (Tensor<T>)right).Vec();
        /// <summary>Element-wise multiplication with a scalar constant.</summary>
        public static Vector<T> operator *(Vector<T> left, Scalar<T> right) => ((Tensor<T>)left * (Tensor<T>)right).Vec();

        /// <summary>Element-wise division.</summary>
        public static Vector<T> operator /(Vector<T> left, Vector<T> right) => ((Tensor<T>)left / (Tensor<T>)right).Vec();
        /// <summary>Element-wise division with a scalar constant.</summary>
        public static Vector<T> operator /(Scalar<T> left, Vector<T> right) => ((Tensor<T>)left / (Tensor<T>)right).Vec();
        /// <summary>Element-wise division with a scalar constant.</summary>
        public static Vector<T> operator /(Vector<T> left, Scalar<T> right) => ((Tensor<T>)left / (Tensor<T>)right).Vec();

        /// <summary>Element-wise modulo.</summary>
        public static Vector<T> operator %(Vector<T> left, Vector<T> right) => ((Tensor<T>)left % (Tensor<T>)right).Vec();
        /// <summary>Element-wise modulo with a scalar constant.</summary>
        public static Vector<T> operator %(Scalar<T> left, Vector<T> right) => ((Tensor<T>)left % (Tensor<T>)right).Vec();
        /// <summary>Element-wise modulo with a scalar constant.</summary>
        public static Vector<T> operator %(Vector<T> left, Scalar<T> right) => ((Tensor<T>)left % (Tensor<T>)right).Vec();

        /// <summary>Element-wise XOR: logical for bit vectors, bitwise otherwise.</summary>
        public static Vector<T> operator ^(Vector<T> left, Vector<T> right) => ((Tensor<T>)left ^ (Tensor<T>)right).Vec();
        /// <summary>Element-wise XOR with a scalar constant.</summary>
        public static Vector<T> operator ^(Scalar<T> left, Vector<T> right) => ((Tensor<T>)left ^ (Tensor<T>)right).Vec();
        /// <summary>Element-wise XOR with a scalar constant.</summary>
        public static Vector<T> operator ^(Vector<T> left, Scalar<T> right) => ((Tensor<T>)left ^ (Tensor<T>)right).Vec();

        /// <summary>Element-wise AND: logical for bit vectors, bitwise otherwise.</summary>
        public static Vector<T> operator &(Vector<T> left, Vector<T> right) => ((Tensor<T>)left & (Tensor<T>)right).Vec();
        /// <summary>Element-wise AND with a scalar constant.</summary>
        public static Vector<T> operator &(Scalar<T> left, Vector<T> right) => ((Tensor<T>)left & (Tensor<T>)right).Vec();
        /// <summary>Element-wise AND with a scalar constant.</summary>
        public static Vector<T> operator &(Vector<T> left, Scalar<T> right) => ((Tensor<T>)left & (Tensor<T>)right).Vec();

        /// <summary>Element-wise OR: logical for bit vectors, bitwise otherwise.</summary>
        public static Vector<T> operator |(Vector<T> left, Vector<T> right) => ((Tensor<T>)left | (Tensor<T>)right).Vec();
        /// <summary>Element-wise OR with a scalar constant.</summary>
        public static Vector<T> operator |(Scalar<T> left, Vector<T> right) => ((Tensor<T>)left | (Tensor<T>)right).Vec();
        /// <summary>Element-wise OR with a scalar constant.</summary>
        public static Vector<T> operator |(Vector<T> left, Scalar<T> right) => ((Tensor<T>)left | (Tensor<T>)right).Vec();

        /// <summary>Element-wise left bit-shift.</summary>
        public static Vector<T> operator <<(Vector<T> left, Vector<T> right) => ((Tensor<T>)left << (Tensor<T>)right).Vec();
        /// <summary>Element-wise left bit-shift by a scalar constant.</summary>
        public static Vector<T> operator <<(Vector<T> left, Scalar<T> right) => ((Tensor<T>)left << (Tensor<T>)right).Vec();

        /// <summary>Element-wise right bit-shift.</summary>
        public static Vector<T> operator >>(Vector<T> left, Vector<T> right) => ((Tensor<T>)left >> (Tensor<T>)right).Vec();
        /// <summary>Element-wise right bit-shift by a scalar constant.</summary>
        public static Vector<T> operator >>(Vector<T> left, Scalar<T> right) => ((Tensor<T>)left >> (Tensor<T>)right).Vec();

        /// <summary>Element-wise negation.</summary>
        public static Vector<T> operator -(Vector<T> input) => (-((Tensor<T>)input)).Vec();

        /// <summary>Element-wise NOT: logical for bit vectors, bitwise otherwise.</summary>
        public static Vector<T> operator !(Vector<T> input) => (!((Tensor<T>)input)).Vec();

        /// <summary>Element-wise greater-than, yielding a bit vector.</summary>
        public static Vector<bit> operator >(Vector<T> left, Vector<T> right) => ((Tensor<T>)left > (Tensor<T>)right).Vec();
        /// <summary>Element-wise greater-than with a scalar constant, yielding a bit vector.</summary>
        public static Vector<bit> operator >(Scalar<T> left, Vector<T> right) => ((Tensor<T>)left > (Tensor<T>)right).Vec();
        /// <summary>Element-wise greater-than with a scalar constant, yielding a bit vector.</summary>
        public static Vector<bit> operator >(Vector<T> left, Scalar<T> right) => ((Tensor<T>)left > (Tensor<T>)right).Vec();

        /// <summary>Element-wise greater-or-equal, yielding a bit vector.</summary>
        public static Vector<bit> operator >=(Vector<T> left, Vector<T> right) => ((Tensor<T>)left >= (Tensor<T>)right).Vec();
        /// <summary>Element-wise greater-or-equal with a scalar constant, yielding a bit vector.</summary>
        public static Vector<bit> operator >=(Scalar<T> left, Vector<T> right) => ((Tensor<T>)left >= (Tensor<T>)right).Vec();
        /// <summary>Element-wise greater-or-equal with a scalar constant, yielding a bit vector.</summary>
        public static Vector<bit> operator >=(Vector<T> left, Scalar<T> right) => ((Tensor<T>)left >= (Tensor<T>)right).Vec();

        /// <summary>Element-wise less-than, yielding a bit vector.</summary>
        public static Vector<bit> operator <(Vector<T> left, Vector<T> right) => ((Tensor<T>)left < (Tensor<T>)right).Vec();
        /// <summary>Element-wise less-than with a scalar constant, yielding a bit vector.</summary>
        public static Vector<bit> operator <(Scalar<T> left, Vector<T> right) => ((Tensor<T>)left < (Tensor<T>)right).Vec();
        /// <summary>Element-wise less-than with a scalar constant, yielding a bit vector.</summary>
        public static Vector<bit> operator <(Vector<T> left, Scalar<T> right) => ((Tensor<T>)left < (Tensor<T>)right).Vec();

        /// <summary>Element-wise less-or-equal, yielding a bit vector.</summary>
        public static Vector<bit> operator <=(Vector<T> left, Vector<T> right) => ((Tensor<T>)left <= (Tensor<T>)right).Vec();
        /// <summary>Element-wise less-or-equal with a scalar constant, yielding a bit vector.</summary>
        public static Vector<bit> operator <=(Scalar<T> left, Vector<T> right) => ((Tensor<T>)left <= (Tensor<T>)right).Vec();
        /// <summary>Element-wise less-or-equal with a scalar constant, yielding a bit vector.</summary>
        public static Vector<bit> operator <=(Vector<T> left, Scalar<T> right) => ((Tensor<T>)left <= (Tensor<T>)right).Vec();

        /// <summary>Element-wise equality, yielding a bit vector (not reference equality).</summary>
        public static Vector<bit> operator ==(Vector<T> left, Vector<T> right) => ((Tensor<T>)left == (Tensor<T>)right).Vec();
        /// <summary>Element-wise equality with a scalar constant, yielding a bit vector.</summary>
        public static Vector<bit> operator ==(Scalar<T> left, Vector<T> right) => ((Tensor<T>)left == (Tensor<T>)right).Vec();
        /// <summary>Element-wise equality with a scalar constant, yielding a bit vector.</summary>
        public static Vector<bit> operator ==(Vector<T> left, Scalar<T> right) => ((Tensor<T>)left == (Tensor<T>)right).Vec();

        /// <summary>Element-wise inequality, yielding a bit vector.</summary>
        public static Vector<bit> operator !=(Vector<T> left, Vector<T> right) => ((Tensor<T>)left != (Tensor<T>)right).Vec();
        /// <summary>Element-wise inequality with a scalar constant, yielding a bit vector.</summary>
        public static Vector<bit> operator !=(Scalar<T> left, Vector<T> right) => ((Tensor<T>)left != (Tensor<T>)right).Vec();
        /// <summary>Element-wise inequality with a scalar constant, yielding a bit vector.</summary>
        public static Vector<bit> operator !=(Vector<T> left, Scalar<T> right) => ((Tensor<T>)left != (Tensor<T>)right).Vec();


        #endregion
        #region Onnx Operators

        /// <summary>Splits into <paramref name="numOutputs"/> equal parts.</summary>
        public Vector<T>[] Split(int numOutputs)
            => ((Tensor<T>)this).Split(numOutputs).Select(x => x.Vec()).ToArray();

        /// <summary>Splits into parts of the given sizes.</summary>
        public Vector<T>[] Split(long[] split)
            => ((Tensor<T>)this).Split(split).Select(x => x.Vec()).ToArray();

        /// <summary>Splits into <paramref name="numOutputs"/> parts of the sizes given by <paramref name="split"/>.</summary>
        public Vector<T>[] Split(Vector<int64> split, long numOutputs)
            => ((Tensor<T>)this).Split(split, axis: 0L, numOutputs: numOutputs).Select(x => x.Vec()).ToArray();

        /// <summary>Resizes to the target <paramref name="sizes"/> (ONNX Resize).</summary>
        public Vector<T> Resize(Vector<int64> sizes,
            KeepAspectRatioPolicy? aspectRatio = null,
            bool? antiaAlias = null,
            long[]? axes = null,
            CoordinateTransformationMode? transformMode = null,
            ResizeMode? mode = null,
            NearestMode? nearestMode = null,
            float? cubicCoefficient = null,
            bool? excludeOutside = null)
            => ((Tensor<T>)this).Resize(sizes, aspectRatio, antiaAlias, axes, transformMode, mode, nearestMode, cubicCoefficient, excludeOutside).Vec();

        /// <summary>Resizes by the given per-axis scale factors (ONNX Resize).</summary>
        public Vector<T> Rescale(Vector<float32> scales,
            bool? antiaAlias = null,
            long[]? axes = null,
            CoordinateTransformationMode? transformMode = null,
            ResizeMode? mode = null,
            NearestMode? nearestMode = null,
            float? cubicCoefficient = null,
            bool? excludeOutside = null)
            => ((Tensor<T>)this).Rescale(scales, antiaAlias, axes, transformMode, mode, nearestMode, cubicCoefficient, excludeOutside).Vec();

        /// <summary>Removes the single dimension, yielding a scalar.</summary>
        public Scalar<T> Squeeze()
            => ((Tensor<T>)this).Squeeze().Scalar();

        /// <summary>Slices by start/end indices with optional steps (ONNX Slice).</summary>
        public Vector<T> Slice(Vector<int64> start, Vector<int64> end, Vector<int64>? steps = null)
            => ((Tensor<T>)this).Slice(start, end, null, steps).Vec();

        /// <summary>Softmax normalization over the vector.</summary>
        public Vector<T> Softmax()
            => ((Tensor<T>)this).Softmax().Vec();

        /// <summary>One-hot encoding of the maximum along <paramref name="axis"/> (ONNX Hardmax).</summary>
        public Vector<T> Hardmax(long? axis = null)
            => ((Tensor<T>)this).Hardmax(axis).Vec();

        /// <summary>Element-wise hard sigmoid.</summary>
        public Vector<T> HardSigmoid(float? alpha = null, float? beta = null)
            => ((Tensor<T>)this).HardSigmoid(alpha, beta).Vec();

        /// <summary>Element-wise hard swish.</summary>
        public Vector<T> HardSwish()
            => ((Tensor<T>)this).HardSwish().Vec();

        /// <summary>Element-wise infinity test, yielding a bit vector.</summary>
        public Vector<bit> IsInf(bool detectNegative = true, bool detectPositive = true)
            => ((Tensor<T>)this).IsInf(detectNegative, detectPositive).Vec();

        /// <summary>Element-wise NaN test, yielding a bit vector.</summary>
        public Vector<bit> IsNaN()
            => ((Tensor<T>)this).IsNaN().Vec();

        /// <summary>Log-softmax along <paramref name="axis"/>.</summary>
        public Vector<T> LogSoftmax(long? axis = null)
            => ((Tensor<T>)this).LogSoftmax(axis).Vec();

        /// <summary>Normalizes to zero mean and unit variance over <paramref name="axes"/>.</summary>
        public Vector<T> MeanVarianceNormalization(long[]? axes = null)
            => ((Tensor<T>)this).MeanVarianceNormalization(axes).Vec();

        /// <summary>Element-wise Mish activation.</summary>
        public Vector<T> Mish()
            => ((Tensor<T>)this).Mish().Vec();

        /// <summary>Element-wise rounding to the nearest integer (half to even).</summary>
        public Vector<T> Round()
            => ((Tensor<T>)this).Round().Vec();

        /// <summary>Element-wise shrink thresholding (ONNX Shrink).</summary>
        public Vector<T> Shrink(float? bias = null, float? lambd = null)
            => ((Tensor<T>)this).Shrink(bias, lambd).Vec();

        /// <summary>Element-wise softplus.</summary>
        public Vector<T> Softplus()
            => ((Tensor<T>)this).Softplus().Vec();

        /// <summary>Element-wise softsign.</summary>
        public Vector<T> Softsign()
            => ((Tensor<T>)this).Softsign().Vec();

        /// <summary>Element-wise thresholded ReLU.</summary>
        public Vector<T> ThresholdedRelu(float? alpha = null)
            => ((Tensor<T>)this).ThresholdedRelu(alpha).Vec();

        /// <summary>Slices using scalar start/end indices with an optional step.</summary>
        public Vector<T> Slice(Scalar<int64> start, Scalar<int64> end, Scalar<int64>? steps = null)
            => ((Tensor<T>)this).Slice(start.Unsqueeze(), end.Unsqueeze(), null, steps?.Unsqueeze()).Vec();

        /// <summary>Gathers slices using multi-dimensional indices (ONNX GatherND).</summary>
        public Vector<T> GatherND(Vector<int64> indices, long? batchDims)
            => ((Tensor<T>)this).GatherND(indices,batchDims).Vec();

        /// <summary>Casts the element type to <typeparamref name="V"/>, preserving rank 1.</summary>
        public Vector<V> Cast<V>(bool saturate = true) where V : IVarType
            => ((Tensor<T>)this).Cast<V>(saturate).Vec();

        /// <summary>Reduces the whole vector to a scalar (e.g. sum, mean, max).</summary>
        public Scalar<T> Reduce(ReduceKind reduceKind)
            => ((Tensor<T>)this).Reduce(reduceKind, null, keepDims: false).Scalar();

        /// <summary>Reduces the whole vector, keeping the result as a length-1 vector.</summary>
        public Vector<T> ReduceKeepDims(ReduceKind reduceKind)
            => ((Tensor<T>)this).Reduce(reduceKind, null, keepDims: true).Vec();

        /// <summary>Tiles the vector by repeating it the given number of times.</summary>
        public Vector<T> Tile(Vector<int64> repeats)
            => ((Tensor<T>)this).Tile(repeats).Vec();

        /// <summary>Element-wise minimum of this vector and <paramref name="others"/>.</summary>
        public Vector<T> Min(params Tensor<T>[] others)
            => ((Tensor<T>)this).Min(others).Vec();

        /// <summary>Element-wise maximum of this vector and <paramref name="others"/>.</summary>
        public Vector<T> Max(params Tensor<T>[] others)
            => ((Tensor<T>)this).Max(others).Vec();

        /// <summary>Element-wise floor.</summary>
        public Vector<T> Floor()
            => ((Tensor<T>)this).Floor().Vec();

        /// <summary>Element-wise absolute value.</summary>
        public Vector<T> Abs()
            => ((Tensor<T>)this).Abs().Vec();

        /// <summary>Element-wise reciprocal.</summary>
        public Vector<T> Reciprocal()
            => ((Tensor<T>)this).Reciprocal().Vec();

        /// <summary>Element-wise error function.</summary>
        public Vector<T> Erf()
            => ((Tensor<T>)this).Erf().Vec();

        /// <summary>Element-wise arccosine.</summary>
        public Vector<T> Acos()
            => ((Tensor<T>)this).Acos().Vec();

        /// <summary>Element-wise inverse hyperbolic cosine.</summary>
        public Vector<T> Acosh()
            => ((Tensor<T>)this).Acosh().Vec();

        /// <summary>Element-wise arcsine.</summary>
        public Vector<T> Asin()
            => ((Tensor<T>)this).Asin().Vec();

        /// <summary>Element-wise inverse hyperbolic sine.</summary>
        public Vector<T> Asinh()
            => ((Tensor<T>)this).Asinh().Vec();

        /// <summary>Element-wise arctangent.</summary>
        public Vector<T> Atan()
            => ((Tensor<T>)this).Atan().Vec();

        /// <summary>Element-wise inverse hyperbolic tangent.</summary>
        public Vector<T> Atanh()
            => ((Tensor<T>)this).Atanh().Vec();

        /// <summary>Index of the maximum element, as a scalar.</summary>
        public Scalar<int64> ArgMaxReduce(bool selectLastIndex = false)
            => ((Tensor<T>)this).ArgMax(0, false, selectLastIndex).Scalar();

        /// <summary>Index of the maximum element, as a length-1 vector.</summary>
        public Vector<int64> ArgMaxKeepdims(bool selectLastIndex = false)
            => ((Tensor<T>)this).ArgMax(0, true, selectLastIndex).Vec();

        /// <summary>Index of the minimum element, as a scalar.</summary>
        public Scalar<int64> ArgMinReduce(bool selectLastIndex = false)
            => ((Tensor<T>)this).ArgMin(0, false, selectLastIndex).Scalar();

        /// <summary>Index of the minimum element, as a length-1 vector.</summary>
        public Vector<int64> ArgMinKeepdims(bool selectLastIndex = false)
            => ((Tensor<T>)this).ArgMin(0, true, selectLastIndex).Vec();

        /// <summary>Average pooling with the given kernel shape (ONNX AveragePool).</summary>
        public Vector<T> AveragePool(long[] kernelShape, RoundMode roundMode = RoundMode.Floor, bool countIncludePad = false, long[]? dilations = null, long[]? pads = null, long[]? strides = null)
            => ((Tensor<T>)this).AveragePool(kernelShape, roundMode, countIncludePad, dilations, pads, strides).Vec();

        /// <summary>Batch normalization using the given scale, bias, mean, and variance (ONNX BatchNormalization).</summary>
        public Vector<T> BatchNormalization<T1, T2>(Vector<T1> scale, Vector<T1> bias, Vector<T2> mean, Vector<T2> variance, float epsilon = 1e-05f, float momentum = 0.9f, bool trainingMode = false)
                where T1 : FloatLike where T2 : FloatLike
            => ((Tensor<T>)this).BatchNormalization(scale, bias, mean, variance, epsilon, momentum, trainingMode).Vec();

        /// <summary>Element-wise Bernoulli sampling, treating each element as a probability.</summary>
        public Vector<T> Bernoulli(float? seed = null)
            => ((Tensor<T>)this).Bernoulli(seed).Vec();

        /// <summary>Element-wise Bernoulli sampling, treating each element as a probability, with result element type <typeparamref name="V"/>.</summary>
        public Vector<V> Bernoulli<V>(float? seed = null) where V : CommonLike
            => ((Tensor<T>)this).Bernoulli<V>(seed).Vec();

        /// <summary>Element-wise CELU activation.</summary>
        public Vector<T> Celu(float alpha = 1.0f)
            => ((Tensor<T>)this).Celu(alpha).Vec();

        /// <summary>Element-wise ceiling.</summary>
        public Vector<T> Ceiling()
            => ((Tensor<T>)this).Ceiling().Vec();

        /// <summary>Element-wise cosine.</summary>
        public Vector<T> Cos()
            => ((Tensor<T>)this).Cos().Vec();

        /// <summary>Element-wise hyperbolic cosine.</summary>
        public Vector<T> Cosh()
            => ((Tensor<T>)this).Cosh().Vec();

        /// <summary>Element-wise sine.</summary>
        public Vector<T> Sin()
            => ((Tensor<T>)this).Sin().Vec();

        /// <summary>Element-wise hyperbolic sine.</summary>
        public Vector<T> Sinh()
            => ((Tensor<T>)this).Sinh().Vec();

        /// <summary>Element-wise tangent.</summary>
        public Vector<T> Tan()
            => ((Tensor<T>)this).Tan().Vec();

        /// <summary>Element-wise hyperbolic tangent.</summary>
        public Vector<T> Tanh()
            => ((Tensor<T>)this).Tanh().Vec();

        /// <summary>Element-wise power.</summary>
        public Vector<T> Pow<T1>(Tensor<T1> power) where T1 : IVarType
            => ((Tensor<T>)this).Pow(power).Vec();

        /// <summary>Element-wise natural logarithm.</summary>
        public Vector<T> Ln()
            => ((Tensor<T>)this).Ln().Vec();

        /// <summary>Element-wise square root.</summary>
        public Vector<T> Sqrt()
            => ((Tensor<T>)this).Sqrt().Vec();

        /// <summary>Element-wise sign.</summary>
        public Vector<T> Sign()
            => ((Tensor<T>)this).Sign().Vec();

        /// <summary>Concatenates this vector with <paramref name="others"/>.</summary>
        public Vector<T> Concat(params Vector<T>[] others)
            // The Variable→Vector<T> operator validates the rank-1 result.
            => OnnxOp.Concat([this, .. others], 0);

        /// <summary>Pads with <paramref name="padLeft"/> elements before and <paramref name="padRight"/> elements after the vector.</summary>
        public Vector<T> Pad(PadMode mode, Scalar<int64> padLeft, Scalar<int64> padRight, Scalar<T> val)
            => ((Tensor<T>)this).Pad(mode, pads: (Vector<int64>)[padLeft, padRight], val: val, axes: null).Vec();

        /// <summary>Pads using begin (<paramref name="outerPads"/>) and end (<paramref name="innerPads"/>) pad counts.</summary>
        public Vector<T> Pad(PadMode mode, Vector<int64> outerPads, Vector<int64> innerPads, Scalar<T> val)
            => ((Tensor<T>)this).Pad(mode, outerPads, innerPads, val, axes: null).Vec();

        /// <summary>Pads using begin (<paramref name="outerPads"/>) and end (<paramref name="innerPads"/>) pad counts.</summary>
        public Vector<T> Pad(PadMode mode, Vector<int64> outerPads, Vector<int64> innerPads, Scalar<T>? val = null, Vector<int64>? axes = null)
            => ((Tensor<T>)this).Pad(mode, outerPads, innerPads, val, axes).Vec();

        /// <summary>Pads using an ONNX-style pads vector (begin counts, then end counts).</summary>
        public Vector<T> Pad(PadMode mode, Vector<int64> pads, Scalar<T> val)
            => ((Tensor<T>)this).Pad(mode, pads, val, axes: null).Vec();
        /// <summary>Pads using an ONNX-style pads vector (begin counts, then end counts).</summary>
        public Vector<T> Pad(PadMode mode, Vector<int64> pads, Scalar<T>? val = null, Vector<int64>? axes = null)
            => ((Tensor<T>)this).Pad(mode, pads, val, axes).Vec();

        /// <summary>Element-wise ELU activation.</summary>
        public Vector<T> Elu(float alpha = 1.0f)
            => ((Tensor<T>)this).Elu(alpha).Vec();

        /// <summary>Element-wise GELU activation.</summary>
        public Vector<T> Gelu(GeluApproximate approximate = GeluApproximate.None)
            => ((Tensor<T>)this).Gelu(approximate).Vec();

        /// <summary>Element-wise leaky ReLU activation.</summary>
        public Vector<T> LeakyRelu(float alpha = 0.01f)
            => ((Tensor<T>)this).LeakyRelu(alpha).Vec();

        /// <summary>Element-wise ReLU activation.</summary>
        public Vector<T> Relu()
            => ((Tensor<T>)this).Relu().Vec();

        /// <summary>Element-wise SELU activation.</summary>
        public Vector<T> Selu(float alpha = 1.67326319217681884765625f, float gamma = 1.0507010221481323242187f)
            => ((Tensor<T>)this).Selu(alpha, gamma).Vec();

        /// <summary>Element-wise sigmoid.</summary>
        public Vector<T> Sigmoid()
            => ((Tensor<T>)this).Sigmoid().Vec();

        /// <summary>Top <paramref name="k"/> values and their indices.</summary>
        public (Vector<T> topK, Vector<int64> indices) TopK(long k, long axis = -1, bool largest = true, bool sorted = true)
        {
            var results = ((Tensor<T>)this).TopK(k, axis, largest, sorted);
            return (results.topK.Vec(), results.indices.Vec());
        }

        /// <summary>Writes <paramref name="values"/> at <paramref name="indices"/> into a copy of this vector (ONNX ScatterND).</summary>
        public Vector<T> ScatterND(Tensor<int64> indices, Tensor<T> values, ScatterNDReduction? reduceMode = ScatterNDReduction.None)
            => ((Tensor<T>)this).ScatterND(indices, values, reduceMode).Vec();

        /// <summary>Element-wise clamping to [min, max].</summary>
        public Vector<T> Clip(Scalar<T> min, Scalar<T> max)
            => ((Tensor<T>)this).Clip(min, max).Vec();

        /// <summary>Selects elements where <paramref name="condition"/> is true.</summary>
        public Vector<T> Compress(Vector<bit> condition, long axis)
            => OnnxOp.Compress(this, condition, axis);

        /// <summary>Element-wise exponential.</summary>
        public Vector<T> Exp()
            => ((Tensor<T>)this).Exp().Vec();

        #endregion
    }
}

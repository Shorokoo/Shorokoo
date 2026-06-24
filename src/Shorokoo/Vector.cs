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

    /// <summary>
    /// A rank-1 symbolic tensor of element type <typeparamref name="T"/>.
    /// Operations mirror <see cref="Tensor{T}"/> but statically preserve the rank-1 typing; instances can
    /// also be built from C# collection expressions.
    /// </summary>
    [CollectionBuilder(typeof(Shorokoo.Core.TensorCollectionBuilder), nameof(Shorokoo.Core.TensorCollectionBuilder.CreateVector))]
    public partial class Vector<T> : Tensor<T>, IVector, IEnumerable<VectorExpressionHelper<T>>
        where T : IVarType
    {
        /// <summary>Shape inferred at graph-build time, as a constant <c>int64</c> vector (null when unknown).</summary>
        public override Vector<int64>? InfShape => base.InfShape;

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
                return Vector<T>.unit;
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
                return Vector<T>.empty;
            }
        }

        #endregion

        #region Constructors

        internal Vector(Func<Vector<int64>>? shapeFn, DType dtype, Node owningNode, Function? moduleFn, string? name = null) : base(shapeFn, dtype, owningNode, moduleFn, name, rank: 1)
        {
        }

        #endregion

        #region Collection Expression Helpers

        IEnumerator IEnumerable.GetEnumerator() => this.GetEnumerator();

        /// <summary>Supports C# collection-expression composition by yielding a single helper that wraps this vector.</summary>
        public new IEnumerator<VectorExpressionHelper<T>> GetEnumerator()
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
            => base.Split(numOutputs).Select(x => x.Vec()).ToArray();

        /// <summary>Splits into parts of the given sizes.</summary>
        public Vector<T>[] Split(long[] split)
            => base.Split(split).Select(x => x.Vec()).ToArray();

        /// <summary>Splits into <paramref name="numOutputs"/> parts of the sizes given by <paramref name="split"/>.</summary>
        public Vector<T>[] Split(Vector<int64> split, long numOutputs)
            => base.Split(split, axis: 0L, numOutputs: numOutputs).Select(x => x.Vec()).ToArray();

        /// <summary>Resizes to the target <paramref name="sizes"/> (ONNX Resize).</summary>
        public new Vector<T> Resize(Vector<int64> sizes,
            KeepAspectRatioPolicy? aspectRatio = null,
            bool? antiaAlias = null,
            long[]? axes = null,
            CoordinateTransformationMode? transformMode = null,
            ResizeMode? mode = null,
            NearestMode? nearestMode = null,
            float? cubicCoefficient = null,
            bool? excludeOutside = null)
            => base.Resize(sizes, aspectRatio, antiaAlias, axes, transformMode, mode, nearestMode, cubicCoefficient, excludeOutside).Vec();

        /// <summary>Resizes by the given per-axis scale factors (ONNX Resize).</summary>
        public new Vector<T> Rescale(Vector<float32> scales,
            bool? antiaAlias = null,
            long[]? axes = null,
            CoordinateTransformationMode? transformMode = null,
            ResizeMode? mode = null,
            NearestMode? nearestMode = null,
            float? cubicCoefficient = null,
            bool? excludeOutside = null)
            => base.Rescale(scales, antiaAlias, axes, transformMode, mode, nearestMode, cubicCoefficient, excludeOutside).Vec();

        /// <summary>Removes the single dimension, yielding a scalar.</summary>
        public Scalar<T> Squeeze()
            => base.Squeeze().Scalar();

        /// <summary>Slices by start/end indices with optional steps (ONNX Slice).</summary>
        public Vector<T> Slice(Vector<int64> start, Vector<int64> end, Vector<int64>? steps = null)
            => base.Slice(start, end, null, steps).Vec();

        /// <summary>Softmax normalization over the vector.</summary>
        public Vector<T> Softmax()
            => base.Softmax().Vec();

        /// <summary>One-hot encoding of the maximum along <paramref name="axis"/> (ONNX Hardmax).</summary>
        public new Vector<T> Hardmax(long? axis = null)
            => base.Hardmax(axis).Vec();

        /// <summary>Element-wise hard sigmoid.</summary>
        public new Vector<T> HardSigmoid(float? alpha = null, float? beta = null)
            => base.HardSigmoid(alpha, beta).Vec();

        /// <summary>Element-wise hard swish.</summary>
        public new Vector<T> HardSwish()
            => base.HardSwish().Vec();

        /// <summary>Element-wise infinity test, yielding a bit vector.</summary>
        public new Vector<bit> IsInf(bool detectNegative = true, bool detectPositive = true)
            => base.IsInf(detectNegative, detectPositive).Vec();

        /// <summary>Element-wise NaN test, yielding a bit vector.</summary>
        public new Vector<bit> IsNaN()
            => base.IsNaN().Vec();

        /// <summary>Log-softmax along <paramref name="axis"/>.</summary>
        public new Vector<T> LogSoftmax(long? axis = null)
            => base.LogSoftmax(axis).Vec();

        /// <summary>Normalizes to zero mean and unit variance over <paramref name="axes"/>.</summary>
        public new Vector<T> MeanVarianceNormalization(long[]? axes = null)
            => base.MeanVarianceNormalization(axes).Vec();

        /// <summary>Element-wise Mish activation.</summary>
        public new Vector<T> Mish()
            => base.Mish().Vec();

        /// <summary>Element-wise rounding to the nearest integer (half to even).</summary>
        public new Vector<T> Round()
            => base.Round().Vec();

        /// <summary>Element-wise shrink thresholding (ONNX Shrink).</summary>
        public new Vector<T> Shrink(float? bias = null, float? lambd = null)
            => base.Shrink(bias, lambd).Vec();

        /// <summary>Element-wise softplus.</summary>
        public new Vector<T> Softplus()
            => base.Softplus().Vec();

        /// <summary>Element-wise softsign.</summary>
        public new Vector<T> Softsign()
            => base.Softsign().Vec();

        /// <summary>Element-wise thresholded ReLU.</summary>
        public new Vector<T> ThresholdedRelu(float? alpha = null)
            => base.ThresholdedRelu(alpha).Vec();

        /// <summary>Slices using scalar start/end indices with an optional step.</summary>
        public Vector<T> Slice(Scalar<int64> start, Scalar<int64> end, Scalar<int64>? steps = null)
            => base.Slice(start.Unsqueeze(), end.Unsqueeze(), null, steps?.Unsqueeze()).Vec();

        /// <summary>Gathers slices using multi-dimensional indices (ONNX GatherND).</summary>
        public Vector<T> GatherND(Vector<int64> indices, long? batchDims)
            => base.GatherND(indices,batchDims).Vec();

        /// <summary>Casts the element type to <typeparamref name="V"/>, preserving rank 1.</summary>
        public new Vector<V> Cast<V>(bool saturate = true) where V : IVarType
            => base.Cast<V>(saturate).Vec();

        /// <summary>Reduces the whole vector to a scalar (e.g. sum, mean, max).</summary>
        public Scalar<T> Reduce(ReduceKind reduceKind)
            => base.Reduce(reduceKind, null, keepDims: false).Scalar();

        /// <summary>Reduces the whole vector, keeping the result as a length-1 vector.</summary>
        public Vector<T> ReduceKeepDims(ReduceKind reduceKind)
            => base.Reduce(reduceKind, null, keepDims: true).Vec();

        /// <summary>Tiles the vector by repeating it the given number of times.</summary>
        public Vector<T> Tile(Vector<int64> repeats)
            => base.Tile(repeats).Vec();

        /// <summary>Element-wise minimum of this vector and <paramref name="others"/>.</summary>
        public new Vector<T> Min(params Tensor<T>[] others)
            => base.Min(others).Vec();

        /// <summary>Element-wise maximum of this vector and <paramref name="others"/>.</summary>
        public new Vector<T> Max(params Tensor<T>[] others)
            => base.Max(others).Vec();

        /// <summary>Element-wise floor.</summary>
        public new Vector<T> Floor()
            => base.Floor().Vec();

        /// <summary>Element-wise absolute value.</summary>
        public new Vector<T> Abs()
            => base.Abs().Vec();

        /// <summary>Element-wise reciprocal.</summary>
        public new Vector<T> Reciprocal()
            => base.Reciprocal().Vec();

        /// <summary>Element-wise error function.</summary>
        public new Vector<T> Erf()
            => base.Erf().Vec();

        /// <summary>Element-wise arccosine.</summary>
        public new Vector<T> Acos()
            => base.Acos().Vec();

        /// <summary>Element-wise inverse hyperbolic cosine.</summary>
        public new Vector<T> Acosh()
            => base.Acosh().Vec();

        /// <summary>Element-wise arcsine.</summary>
        public new Vector<T> Asin()
            => base.Asin().Vec();

        /// <summary>Element-wise inverse hyperbolic sine.</summary>
        public new Vector<T> Asinh()
            => base.Asinh().Vec();

        /// <summary>Element-wise arctangent.</summary>
        public new Vector<T> Atan()
            => base.Atan().Vec();

        /// <summary>Element-wise inverse hyperbolic tangent.</summary>
        public new Vector<T> Atanh()
            => base.Atanh().Vec();

        /// <summary>Index of the maximum element, as a scalar.</summary>
        public Scalar<int64> ArgMaxReduce(bool selectLastIndex = false)
            => base.ArgMax(0, false, selectLastIndex).Scalar();

        /// <summary>Index of the maximum element, as a length-1 vector.</summary>
        public Vector<int64> ArgMaxKeepdims(bool selectLastIndex = false)
            => base.ArgMax(0, true, selectLastIndex).Vec();

        /// <summary>Index of the minimum element, as a scalar.</summary>
        public Scalar<int64> ArgMinReduce(bool selectLastIndex = false)
            => base.ArgMin(0, false, selectLastIndex).Scalar();

        /// <summary>Index of the minimum element, as a length-1 vector.</summary>
        public Vector<int64> ArgMinKeepdims(bool selectLastIndex = false)
            => base.ArgMin(0, true, selectLastIndex).Vec();

        /// <summary>Average pooling with the given kernel shape (ONNX AveragePool).</summary>
        public new Vector<T> AveragePool(long[] kernelShape, RoundMode roundMode = RoundMode.Floor, bool countIncludePad = false, long[]? dilations = null, long[]? pads = null, long[]? strides = null)
            => base.AveragePool(kernelShape, roundMode, countIncludePad, dilations, pads, strides).Vec();

        /// <summary>Batch normalization using the given scale, bias, mean, and variance (ONNX BatchNormalization).</summary>
        public new Vector<T> BatchNormalization<T1, T2>(Vector<T1> scale, Vector<T1> bias, Vector<T2> mean, Vector<T2> variance, float epsilon = 1e-05f, float momentum = 0.9f, bool trainingMode = false)
                where T1 : FloatLike where T2 : FloatLike
            => base.BatchNormalization(scale, bias, mean, variance, epsilon, momentum, trainingMode).Vec();

        /// <summary>Element-wise Bernoulli sampling, treating each element as a probability.</summary>
        public new Vector<T> Bernoulli(float? seed = null)
            => base.Bernoulli(seed).Vec();

        /// <summary>Element-wise Bernoulli sampling, treating each element as a probability, with result element type <typeparamref name="V"/>.</summary>
        public new Vector<V> Bernoulli<V>(float? seed = null) where V : CommonLike
            => base.Bernoulli<V>(seed).Vec();

        /// <summary>Element-wise CELU activation.</summary>
        public new Vector<T> Celu(float alpha = 1.0f)
            => base.Celu(alpha).Vec();

        /// <summary>Element-wise ceiling.</summary>
        public new Vector<T> Ceiling()
            => base.Ceiling().Vec();

        /// <summary>Element-wise cosine.</summary>
        public new Vector<T> Cos()
            => base.Cos().Vec();

        /// <summary>Element-wise hyperbolic cosine.</summary>
        public new Vector<T> Cosh()
            => base.Cosh().Vec();

        /// <summary>Element-wise sine.</summary>
        public new Vector<T> Sin()
            => base.Sin().Vec();

        /// <summary>Element-wise hyperbolic sine.</summary>
        public new Vector<T> Sinh()
            => base.Sinh().Vec();

        /// <summary>Element-wise tangent.</summary>
        public new Vector<T> Tan()
            => base.Tan().Vec();

        /// <summary>Element-wise hyperbolic tangent.</summary>
        public new Vector<T> Tanh()
            => base.Tanh().Vec();

        /// <summary>Element-wise power.</summary>
        public new Vector<T> Pow<T1>(Tensor<T1> power) where T1 : IVarType
            => base.Pow(power).Vec();

        /// <summary>Element-wise natural logarithm.</summary>
        public new Vector<T> Ln()
            => base.Ln().Vec();

        /// <summary>Element-wise square root.</summary>
        public new Vector<T> Sqrt()
            => base.Sqrt().Vec();

        /// <summary>Element-wise sign.</summary>
        public new Vector<T> Sign()
            => base.Sign().Vec();

        /// <summary>Concatenates this vector with <paramref name="others"/>.</summary>
        public Vector<T> Concat(params Vector<T>[] others)
            =>  ((Tensor<T>)OnnxOp.Concat([this, .. others], 0)).Vec();

        /// <summary>Pads with <paramref name="padLeft"/> elements before and <paramref name="padRight"/> elements after the vector.</summary>
        public Vector<T> Pad(PadMode mode, Scalar<int64> padLeft, Scalar<int64> padRight, Scalar<T> val)
            => base.Pad(mode, pads: (Vector<int64>)[padLeft, padRight], val: val, axes: null).Vec();

        /// <summary>Pads using begin (<paramref name="outerPads"/>) and end (<paramref name="innerPads"/>) pad counts.</summary>
        public Vector<T> Pad(PadMode mode, Vector<int64> outerPads, Vector<int64> innerPads, Scalar<T> val)
            => base.Pad(mode, outerPads, innerPads, val, axes: null).Vec();

        /// <summary>Pads using begin (<paramref name="outerPads"/>) and end (<paramref name="innerPads"/>) pad counts.</summary>
        public new Vector<T> Pad(PadMode mode, Vector<int64> outerPads, Vector<int64> innerPads, Scalar<T>? val = null, Vector<int64>? axes = null)
            => base.Pad(mode, outerPads, innerPads, val, axes).Vec();

        /// <summary>Pads using an ONNX-style pads vector (begin counts, then end counts).</summary>
        public Vector<T> Pad(PadMode mode, Vector<int64> pads, Scalar<T> val)
            => base.Pad(mode, pads, val, axes: null).Vec();
        /// <summary>Pads using an ONNX-style pads vector (begin counts, then end counts).</summary>
        public new Vector<T> Pad(PadMode mode, Vector<int64> pads, Scalar<T>? val = null, Vector<int64>? axes = null)
            => base.Pad(mode, pads, val, axes).Vec();

        /// <summary>Element-wise ELU activation.</summary>
        public new Vector<T> Elu(float alpha = 1.0f)
            => base.Elu(alpha).Vec();

        /// <summary>Element-wise GELU activation.</summary>
        public new Vector<T> Gelu(GeluApproximate approximate = GeluApproximate.None)
            => base.Gelu(approximate).Vec();

        /// <summary>Element-wise leaky ReLU activation.</summary>
        public new Vector<T> LeakyRelu(float alpha = 0.01f)
            => base.LeakyRelu(alpha).Vec();

        /// <summary>Element-wise ReLU activation.</summary>
        public new Vector<T> Relu()
            => base.Relu().Vec();

        /// <summary>Element-wise SELU activation.</summary>
        public new Vector<T> Selu(float alpha = 1.67326319217681884765625f, float gamma = 1.0507010221481323242187f)
            => base.Selu(alpha, gamma).Vec();

        /// <summary>Element-wise sigmoid.</summary>
        public new Vector<T> Sigmoid()
            => base.Sigmoid().Vec();

        /// <summary>Top <paramref name="k"/> values and their indices.</summary>
        public new (Vector<T> topK, Vector<int64> indices) TopK(long k, long axis = -1, bool largest = true, bool sorted = true)
        {
            var results = base.TopK(k, axis, largest, sorted);
            return (results.topK.Vec(), results.indices.Vec());
        }

        /// <summary>Writes <paramref name="values"/> at <paramref name="indices"/> into a copy of this vector (ONNX ScatterND).</summary>
        public new Vector<T> ScatterND(Tensor<int64> indices, Tensor<T> values, ScatterNDReduction? reduceMode = ScatterNDReduction.None)
            => base.ScatterND(indices, values, reduceMode).Vec();

        /// <summary>Element-wise clamping to [min, max].</summary>
        public new Vector<T> Clip(Scalar<T> min, Scalar<T> max)
            => base.Clip(min, max).Vec();

        /// <summary>Selects elements where <paramref name="condition"/> is true.</summary>
        public new Vector<T> Compress(Vector<bit> condition, long axis)
            => (Vector<T>)OnnxOp.Compress(this, condition, axis);

        /// <summary>Element-wise exponential.</summary>
        public new Vector<T> Exp()
            => base.Exp().Vec();

        #endregion

        #region Equal and HashCode

        /// <summary>Reference equality; element-wise comparison is provided by the equality operators.</summary>
        public override bool Equals(object? obj) => base.Equals(obj);

        /// <summary>Identity-based hash code, consistent with reference equality.</summary>
        public override int GetHashCode() => base.GetHashCode();

        #endregion
    }
}

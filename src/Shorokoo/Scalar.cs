using Shorokoo.Core.Inference.Abstractions;
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
using System.Runtime.CompilerServices;
using static Shorokoo.Core.Nodes.Ops;
using static Shorokoo.Core.Nodes.AutoDiff.Ops;
using static Shorokoo.Globals;

namespace Shorokoo
{
    /// <summary>Non-generic marker for rank-0 (scalar) symbolic tensors.</summary>
    public interface IScalar : ITensor
    {
    }

    public partial struct Scalar<T> : IScalar where T : IVarType
    {
        private Variable? inner;
        // The backing graph node, materialising the established default (per dtype/rank) for a defaulted handle.
        internal Variable Immutable => inner ?? InternalGlobals.DefaultVariable(typeof(Scalar<T>));

        private static readonly DType? expectedDType = OnnxUtils.GetDType(typeof(T));
        public static implicit operator Scalar<T>(Variable imm)
        {
            IValue.RequireKind(imm, DataStructure.Tensor);
            IValue.RequireDType(imm, expectedDType);
            return new Scalar<T> { inner = IValue.RequireRank(imm, 0) };
        }
        public static implicit operator Variable(Scalar<T> h) => h.Immutable;
        public static implicit operator Tensor<T>(Scalar<T> h) => h.Immutable;

        // Convert to the backing graph node, materialising the established default for a defaulted handle.
        Variable IValue.ToVariable() => Immutable;

        // ITensor contract — forward to the backing Variable.
        public int? Rank => Immutable.Rank;
        public Vector<int64> DShape => Immutable.DShape;
        public Vector<int64> TShape => Immutable.TShape;
        public Scalar<int64> TRank => Immutable.TRank;
        public Vector<T> Vec() => (Vector<T>)Immutable;     // throws: a rank-0 scalar is not a vector
        IVector ITensor.Vec() => Vec();
        IScalar ITensor.Scalar() => (Scalar<T>)Immutable;   // a Scalar<T> has no public Scalar(); already rank-0
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
        public override bool Equals(object? obj) => obj is Scalar<T> t && Equals(inner, t.inner);
        public override int GetHashCode() => inner?.GetHashCode() ?? 0;

        private static Scalar<T>? unit; 
        /// <summary>A cached constant scalar of value 1 (true for bit).</summary>
        public static Scalar<T> Unit
        {
            get
            {
                if (Shorokoo.Scalar<T>.unit is null)
                {
                    var type = OnnxUtils.GetDType<T>();
                    if (type == DType.BFloat16) unit = (Scalar<T>)(object)Shorokoo.Globals.Scalar(BFloat16.One);
                    else if (type == DType.Float16) unit = (Scalar<T>)(object)Shorokoo.Globals.Scalar(Float16.One);
                    else if (type == DType.Float32) unit = (Scalar<T>)(object)Shorokoo.Globals.Scalar(1f);
                    else if (type == DType.Float64) unit = (Scalar<T>)(object)Shorokoo.Globals.Scalar(1d);
                    else if (type == DType.Int4) throw new UnsupportedDTypeException(ErrorCodes.CR005, type.ToString(), "Unit Scalar", "Int4 precision is not supported for unit scalar creation");
                    else if (type == DType.Int8) unit = (Scalar<T>)(object)Shorokoo.Globals.Scalar((sbyte)1);
                    else if (type == DType.Int16) unit = (Scalar<T>)(object)Shorokoo.Globals.Scalar((short)1);
                    else if (type == DType.Int32) unit = (Scalar<T>)(object)Shorokoo.Globals.Scalar(1);
                    else if (type == DType.Int64) unit = (Scalar<T>)(object)Shorokoo.Globals.Scalar(1L);
                    else if (type == DType.UInt4) throw new UnsupportedDTypeException(ErrorCodes.CR005, type.ToString(), "Unit Scalar", "UInt4 precision is not supported for unit scalar creation");
                    else if (type == DType.UInt8) unit = (Scalar<T>)(object)Shorokoo.Globals.Scalar((byte)1);
                    else if (type == DType.UInt16) unit = (Scalar<T>)(object)Shorokoo.Globals.Scalar((ushort)1);
                    else if (type == DType.UInt32) unit = (Scalar<T>)(object)Shorokoo.Globals.Scalar(1u);
                    else if (type == DType.UInt64) unit = (Scalar<T>)(object)Shorokoo.Globals.Scalar(1ul);
                    else if (type == DType.String) throw new UnsupportedDTypeException(ErrorCodes.CR005, type.ToString(), "Unit Scalar", "String type is not supported for unit scalar creation");
                    else if (type == DType.Bool) unit = (Scalar<T>)(object)Shorokoo.Globals.Scalar(true);
                    else if (type == DType.Complex64) throw new UnsupportedDTypeException(ErrorCodes.CR005, type.ToString(), "Unit Scalar", "Complex64 numbers are not supported for unit scalar creation");
                    else if (type == DType.Complex128) throw new UnsupportedDTypeException(ErrorCodes.CR005, type.ToString(), "Unit Scalar", "Complex128 numbers are not supported for unit scalar creation");
                    else if (type == DType.Invalid) unit = (Scalar<T>)(object)Shorokoo.Globals.Scalar(1);
                }

                Debug.Assert(Shorokoo.Scalar<T>.unit is not null);
                return Shorokoo.Scalar<T>.unit!.Value;
            }
        }

        public static implicit operator Scalar<T>(PrimitiveParam param)
            => Globals.Scalar<T>(param.ParamVal);
        #region Primitive value conversions

        // A bare primitive cannot reach Scalar<T> through the PrimitiveParam conversion above:
        // C# applies at most one user-defined conversion, so primitive -> PrimitiveParam ->
        // Scalar<T> (two hops) is rejected. These direct conversions close that gap, letting a
        // primitive literal stand in wherever a Scalar<T> is expected — e.g. `Scalar<int64> n = 32;`
        // instead of `Scalar(32L)`. The element type comes from the contextually-required T, not
        // from the literal's C# type, so the same literal targets whatever scalar the surrounding
        // code asks for: `Scalar<int32> a = 5;`, `Scalar<int64> b = 5;`, and `Scalar<float32> c = 5;`
        // are all valid. The value is converted to T via Globals.Scalar<T>.

        /// <summary>Wraps a primitive value as a scalar of the contextually-required element type T.</summary>
        public static implicit operator Scalar<T>(bool value) => Globals.Scalar<T>(value);
        /// <summary>Wraps a primitive value as a scalar of the contextually-required element type T.</summary>
        public static implicit operator Scalar<T>(sbyte value) => Globals.Scalar<T>(value);
        /// <summary>Wraps a primitive value as a scalar of the contextually-required element type T.</summary>
        public static implicit operator Scalar<T>(short value) => Globals.Scalar<T>(value);
        /// <summary>Wraps a primitive value as a scalar of the contextually-required element type T.</summary>
        public static implicit operator Scalar<T>(int value) => Globals.Scalar<T>(value);
        /// <summary>Wraps a primitive value as a scalar of the contextually-required element type T.</summary>
        public static implicit operator Scalar<T>(long value) => Globals.Scalar<T>(value);
        /// <summary>Wraps a primitive value as a scalar of the contextually-required element type T.</summary>
        public static implicit operator Scalar<T>(byte value) => Globals.Scalar<T>(value);
        /// <summary>Wraps a primitive value as a scalar of the contextually-required element type T.</summary>
        public static implicit operator Scalar<T>(ushort value) => Globals.Scalar<T>(value);
        /// <summary>Wraps a primitive value as a scalar of the contextually-required element type T.</summary>
        public static implicit operator Scalar<T>(uint value) => Globals.Scalar<T>(value);
        /// <summary>Wraps a primitive value as a scalar of the contextually-required element type T.</summary>
        public static implicit operator Scalar<T>(ulong value) => Globals.Scalar<T>(value);
        /// <summary>Wraps a primitive value as a scalar of the contextually-required element type T.</summary>
        public static implicit operator Scalar<T>(float value) => Globals.Scalar<T>(value);
        /// <summary>Wraps a primitive value as a scalar of the contextually-required element type T.</summary>
        public static implicit operator Scalar<T>(double value) => Globals.Scalar<T>(value);
        /// <summary>Wraps a half-precision value as a scalar of the contextually-required element type T.</summary>
        public static implicit operator Scalar<T>(Float16 value) => Globals.Scalar<T>(value);
        /// <summary>Wraps a half-precision value as a scalar of the contextually-required element type T.</summary>
        public static implicit operator Scalar<T>(BFloat16 value) => Globals.Scalar<T>(value);

        #endregion
        #region ONNX Operators

        /// <summary>Scalar addition.</summary>
        public static Scalar<T> operator +(Scalar<T> left, Scalar<T> right)
            => ((Tensor<T>)left + (Tensor<T>)right).Scalar();

        /// <summary>Scalar subtraction.</summary>
        public static Scalar<T> operator -(Scalar<T> left, Scalar<T> right)
            => ((Tensor<T>)left - (Tensor<T>)right).Scalar();

        /// <summary>Scalar multiplication.</summary>
        public static Scalar<T> operator *(Scalar<T> left, Scalar<T> right)
            => ((Tensor<T>)left * (Tensor<T>)right).Scalar();

        /// <summary>Scalar division.</summary>
        public static Scalar<T> operator /(Scalar<T> left, Scalar<T> right)
            => ((Tensor<T>)left / (Tensor<T>)right).Scalar();

        /// <summary>Scalar modulo.</summary>
        public static Scalar<T> operator %(Scalar<T> left, Scalar<T> right)
            => ((Tensor<T>)left % (Tensor<T>)right).Scalar();

        /// <summary>Scalar XOR: logical for bit, bitwise otherwise.</summary>
        public static Scalar<T> operator ^(Scalar<T> left, Scalar<T> right)
            => ((Tensor<T>)left ^ (Tensor<T>)right).Scalar();

        /// <summary>Scalar AND: logical for bit, bitwise otherwise.</summary>
        public static Scalar<T> operator &(Scalar<T> left, Scalar<T> right)
            => ((Tensor<T>)left & (Tensor<T>)right).Scalar();

        /// <summary>Scalar OR: logical for bit, bitwise otherwise.</summary>
        public static Scalar<T> operator |(Scalar<T> left, Scalar<T> right)
            => ((Tensor<T>)left | (Tensor<T>)right).Scalar();

        /// <summary>Scalar left bit-shift.</summary>
        public static Scalar<T> operator <<(Scalar<T> left, Scalar<T> right)
            => ((Tensor<T>)left << (Tensor<T>)right).Scalar();

        /// <summary>Scalar right bit-shift.</summary>
        public static Scalar<T> operator >>(Scalar<T> left, Scalar<T> right)
            => ((Tensor<T>)left >> (Tensor<T>)right).Scalar();

        /// <summary>Scalar negation.</summary>
        public static Scalar<T> operator -(Scalar<T> input)
            => (-((Tensor<T>)input)).Scalar();

        /// <summary>Scalar NOT: logical for bit, bitwise otherwise.</summary>
        public static Scalar<T> operator !(Scalar<T> input)
            => (!((Tensor<T>)input)).Scalar();

        /// <summary>Scalar greater-than, yielding a bit scalar.</summary>
        public static Scalar<bit> operator >(Scalar<T> left, Scalar<T> right)
            => ((Tensor<T>)left > (Tensor<T>)right).Scalar();

        /// <summary>Scalar greater-or-equal, yielding a bit scalar.</summary>
        public static Scalar<bit> operator >=(Scalar<T> left, Scalar<T> right)
            => ((Tensor<T>)left >= (Tensor<T>)right).Scalar();

        /// <summary>Scalar less-than, yielding a bit scalar.</summary>
        public static Scalar<bit> operator <(Scalar<T> left, Scalar<T> right)
            => ((Tensor<T>)left < (Tensor<T>)right).Scalar();

        /// <summary>Scalar less-or-equal, yielding a bit scalar.</summary>
        public static Scalar<bit> operator <=(Scalar<T> left, Scalar<T> right)
            => ((Tensor<T>)left <= (Tensor<T>)right).Scalar();

        /// <summary>Scalar equality, yielding a bit scalar (not reference equality).</summary>
        public static Scalar<bit> operator ==(Scalar<T> left, Scalar<T> right)
            => ((Tensor<T>)left == (Tensor<T>)right).Scalar();

        /// <summary>Scalar inequality, yielding a bit scalar.</summary>
        public static Scalar<bit> operator !=(Scalar<T> left, Scalar<T> right)
            => ((Tensor<T>)left != (Tensor<T>)right).Scalar();

        /// <summary>Casts the element type to <typeparamref name="V"/>, preserving rank 0.</summary>
        public Scalar<V> Cast<V>(bool saturate = true) where V : IVarType
            => ((Tensor<T>)this).Cast<V>(saturate).Scalar();

        /// <summary>Minimum of this scalar and <paramref name="others"/>.</summary>
        public Scalar<T> Min(params Scalar<T>[] others)
            => ((Tensor<T>)this).Min([.. others.Select(o => (Tensor<T>)o)]).Scalar();

        /// <summary>Maximum of this scalar and <paramref name="others"/>.</summary>
        public Scalar<T> Max(params Scalar<T>[] others)
            => ((Tensor<T>)this).Max([.. others.Select(o => (Tensor<T>)o)]).Scalar();

        /// <summary>Scalar floor.</summary>
        public Scalar<T> Floor()
            => ((Tensor<T>)this).Floor().Scalar();

        /// <summary>Scalar absolute value.</summary>
        public Scalar<T> Abs()
            => ((Tensor<T>)this).Abs().Scalar();

        /// <summary>Scalar arccosine.</summary>
        public Scalar<T> Acos()
            => ((Tensor<T>)this).Acos().Scalar();

        /// <summary>Scalar inverse hyperbolic cosine.</summary>
        public Scalar<T> Acosh()
            => ((Tensor<T>)this).Acosh().Scalar();

        /// <summary>Scalar arcsine.</summary>
        public Scalar<T> Asin()
            => ((Tensor<T>)this).Asin().Scalar();

        /// <summary>Scalar inverse hyperbolic sine.</summary>
        public Scalar<T> Asinh()
            => ((Tensor<T>)this).Asinh().Scalar();

        /// <summary>Scalar arctangent.</summary>
        public Scalar<T> Atan()
            => ((Tensor<T>)this).Atan().Scalar();

        /// <summary>Scalar inverse hyperbolic tangent.</summary>
        public Scalar<T> Atanh()
            => ((Tensor<T>)this).Atanh().Scalar();

        /// <summary>Bernoulli sample, treating this scalar as a probability.</summary>
        public Scalar<T> Bernoulli(float? seed = null)
            => ((Tensor<T>)this).Bernoulli(seed).Scalar();

        /// <summary>Bernoulli sample, treating this scalar as a probability, with result element type <typeparamref name="V"/>.</summary>
        public Scalar<V> Bernoulli<V>(float? seed = null) where V : CommonLike
            => ((Tensor<T>)this).Bernoulli<V>(seed).Scalar();

        /// <summary>Scalar CELU activation.</summary>
        public Scalar<T> Celu(float alpha = 1.0f)
            => ((Tensor<T>)this).Celu(alpha).Scalar();

        /// <summary>Scalar ceiling.</summary>
        public Scalar<T> Ceiling()
            => ((Tensor<T>)this).Ceiling().Scalar();

        /// <summary>Scalar cosine.</summary>
        public Scalar<T> Cos()
            => ((Tensor<T>)this).Cos().Scalar();

        /// <summary>Scalar hyperbolic cosine.</summary>
        public Scalar<T> Cosh()
            => ((Tensor<T>)this).Cosh().Scalar();

        /// <summary>Scalar sine.</summary>
        public Scalar<T> Sin()
            => ((Tensor<T>)this).Sin().Scalar();

        /// <summary>Scalar hyperbolic sine.</summary>
        public Scalar<T> Sinh()
            => ((Tensor<T>)this).Sinh().Scalar();

        /// <summary>Scalar tangent.</summary>
        public Scalar<T> Tan()
            => ((Tensor<T>)this).Tan().Scalar();

        /// <summary>Scalar hyperbolic tangent.</summary>
        public Scalar<T> Tanh()
            => ((Tensor<T>)this).Tanh().Scalar();

        /// <summary>Scalar power.</summary>
        public Scalar<T> Pow<T1>(Scalar<T1> power) where T1 : IVarType
            => ((Tensor<T>)this).Pow<T1>(power).Scalar();

        /// <summary>Scalar natural logarithm.</summary>
        public Scalar<T> Ln()
            => ((Tensor<T>)this).Ln().Scalar();

        /// <summary>Scalar square root.</summary>
        public Scalar<T> Sqrt()
            => ((Tensor<T>)this).Sqrt().Scalar();

        /// <summary>Scalar reciprocal.</summary>
        public Scalar<T> Reciprocal()
            => ((Tensor<T>)this).Reciprocal().Scalar();

        /// <summary>Scalar error function.</summary>
        public Scalar<T> Erf()
            => ((Tensor<T>)this).Erf().Scalar();

        /// <summary>Scalar sign.</summary>
        public Scalar<T> Sign()
            => ((Tensor<T>)this).Sign().Scalar();

        /// <summary>Scalar ELU activation.</summary>
        public Scalar<T> Elu(float alpha = 1.0f)
            => ((Tensor<T>)this).Elu(alpha).Scalar();

        /// <summary>Scalar GELU activation.</summary>
        public Scalar<T> Gelu(GeluApproximate approximate = GeluApproximate.None)
            => ((Tensor<T>)this).Gelu(approximate).Scalar();

        /// <summary>Scalar leaky ReLU activation.</summary>
        public Scalar<T> LeakyRelu(float alpha = 0.01f)
            => ((Tensor<T>)this).LeakyRelu(alpha).Scalar();

        /// <summary>Scalar ReLU activation.</summary>
        public Scalar<T> Relu()
            => ((Tensor<T>)this).Relu().Scalar();

        /// <summary>Scalar SELU activation.</summary>
        public Scalar<T> Selu(float alpha = 1.67326319217681884765625f, float gamma = 1.0507010221481323242187f)
            => ((Tensor<T>)this).Selu(alpha, gamma).Scalar();

        /// <summary>Scalar sigmoid.</summary>
        public Scalar<T> Sigmoid()
            => ((Tensor<T>)this).Sigmoid().Scalar();

        /// <summary>Scalar hard sigmoid.</summary>
        public Scalar<T> HardSigmoid(float? alpha = null, float? beta = null)
            => ((Tensor<T>)this).HardSigmoid(alpha, beta).Scalar();

        /// <summary>Scalar hard swish.</summary>
        public Scalar<T> HardSwish()
            => ((Tensor<T>)this).HardSwish().Scalar();

        /// <summary>Tests whether the value is infinite, yielding a bit scalar.</summary>
        public Scalar<bit> IsInf(bool detectNegative = true, bool detectPositive = true)
            => ((Tensor<T>)this).IsInf(detectNegative, detectPositive).Scalar();

        /// <summary>Tests whether the value is NaN, yielding a bit scalar.</summary>
        public Scalar<bit> IsNaN()
            => ((Tensor<T>)this).IsNaN().Scalar();

        /// <summary>Scalar Mish activation.</summary>
        public Scalar<T> Mish()
            => ((Tensor<T>)this).Mish().Scalar();

        /// <summary>Rounds to the nearest integer (half to even).</summary>
        public Scalar<T> Round()
            => ((Tensor<T>)this).Round().Scalar();

        /// <summary>Scalar shrink thresholding (ONNX Shrink).</summary>
        public Scalar<T> Shrink(float? bias = null, float? lambd = null)
            => ((Tensor<T>)this).Shrink(bias, lambd).Scalar();

        /// <summary>Scalar softplus.</summary>
        public Scalar<T> Softplus()
            => ((Tensor<T>)this).Softplus().Scalar();

        /// <summary>Scalar softsign.</summary>
        public Scalar<T> Softsign()
            => ((Tensor<T>)this).Softsign().Scalar();

        /// <summary>Scalar thresholded ReLU.</summary>
        public Scalar<T> ThresholdedRelu(float? alpha = null)
            => ((Tensor<T>)this).ThresholdedRelu(alpha).Scalar();

        /// <summary>Clamps the value to [min, max].</summary>
        public Scalar<T> Clip(Scalar<T> min, Scalar<T> max)
            => ((Tensor<T>)this).Clip(min, max).Scalar();

        /// <summary>Scalar exponential.</summary>
        public Scalar<T> Exp()
            => ((Tensor<T>)this).Exp().Scalar();

        /// <summary>Promotes this scalar to a rank-1 vector of length 1.</summary>
        public Vector<T> Unsqueeze()
            => ((Tensor<T>)this).Unsqueeze().Vec();

        /// <summary>Promotes this scalar to a rank-1 vector of length 1; <paramref name="axis"/> must be 0 or -1.</summary>
        // Unsqueezing a rank-0 scalar at its single axis (0 / -1) is always rank-1, so narrow the
        // return to Vector<T> (mirrors the parameterless Unsqueeze() above). Lets the result bind
        // directly to APIs that require a Vector<int64> — e.g. a Reduce/Reshape axes argument —
        // without a manual .Vec().
        public Vector<T> Unsqueeze(long axis)
            => ((Tensor<T>)this).Unsqueeze(axis).Vec();

        #region primitive param support

        /// <summary>Scalar addition with a primitive constant.</summary>
        public static Scalar<T> operator +(Scalar<T> left, PrimitiveParam right) => left + (Scalar<T>)right;
        /// <summary>Scalar addition with a primitive constant.</summary>
        public static Scalar<T> operator +(PrimitiveParam left, Scalar<T> right) => (Scalar<T>)left + right;

        /// <summary>Scalar subtraction with a primitive constant.</summary>
        public static Scalar<T> operator -(Scalar<T> left, PrimitiveParam right) => left - (Scalar<T>)right;
        /// <summary>Scalar subtraction with a primitive constant.</summary>
        public static Scalar<T> operator -(PrimitiveParam left, Scalar<T> right) => (Scalar<T>)left - right;

        /// <summary>Scalar multiplication with a primitive constant.</summary>
        public static Scalar<T> operator *(Scalar<T> left, PrimitiveParam right) => left * (Scalar<T>)right;
        /// <summary>Scalar multiplication with a primitive constant.</summary>
        public static Scalar<T> operator *(PrimitiveParam left, Scalar<T> right) => (Scalar<T>)left * right;

        /// <summary>Scalar division with a primitive constant.</summary>
        public static Scalar<T> operator /(Scalar<T> left, PrimitiveParam right) => left / (Scalar<T>)right;
        /// <summary>Scalar division with a primitive constant.</summary>
        public static Scalar<T> operator /(PrimitiveParam left, Scalar<T> right) => (Scalar<T>)left / right;

        /// <summary>Scalar modulo with a primitive constant.</summary>
        public static Scalar<T> operator %(Scalar<T> left, PrimitiveParam right) => left % (Scalar<T>)right;
        /// <summary>Scalar modulo with a primitive constant.</summary>
        public static Scalar<T> operator %(PrimitiveParam left, Scalar<T> right) => (Scalar<T>)left % right;

        /// <summary>Scalar XOR with a primitive constant.</summary>
        public static Scalar<T> operator ^(Scalar<T> left, PrimitiveParam right) => left ^ (Scalar<T>)right;
        /// <summary>Scalar XOR with a primitive constant.</summary>
        public static Scalar<T> operator ^(PrimitiveParam left, Scalar<T> right) => (Scalar<T>)left ^ right;

        /// <summary>Scalar AND with a primitive constant.</summary>
        public static Scalar<T> operator &(Scalar<T> left, PrimitiveParam right) => left & (Scalar<T>)right;
        /// <summary>Scalar AND with a primitive constant.</summary>
        public static Scalar<T> operator &(PrimitiveParam left, Scalar<T> right) => (Scalar<T>)left & right;

        /// <summary>Scalar OR with a primitive constant.</summary>
        public static Scalar<T> operator |(Scalar<T> left, PrimitiveParam right) => left | (Scalar<T>)right;
        /// <summary>Scalar OR with a primitive constant.</summary>
        public static Scalar<T> operator |(PrimitiveParam left, Scalar<T> right) => (Scalar<T>)left | right;

        /// <summary>Scalar left bit-shift by a primitive constant.</summary>
        public static Scalar<T> operator <<(Scalar<T> left, PrimitiveParam right) => left << (Scalar<T>)right;
        // public static Scalar<T> operator <<(PrimitiveParam left, Scalar<T> right) => (Scalar<T>)left + right;

        /// <summary>Scalar right bit-shift by a primitive constant.</summary>
        public static Scalar<T> operator >>(Scalar<T> left, PrimitiveParam right) => left >> (Scalar<T>)right;
        // public static Scalar<T> operator >>(PrimitiveParam left, Scalar<T> right) => (Scalar<T>)left + right;

        /// <summary>Scalar greater-than with a primitive constant, yielding a bit scalar.</summary>
        public static Scalar<bit> operator >(Scalar<T> left, PrimitiveParam right) => left > (Scalar<T>)right;
        /// <summary>Scalar greater-than with a primitive constant, yielding a bit scalar.</summary>
        public static Scalar<bit> operator >(PrimitiveParam left, Scalar<T> right) => (Scalar<T>)left > right;
        /// <summary>Scalar greater-or-equal with a primitive constant, yielding a bit scalar.</summary>
        public static Scalar<bit> operator >=(Scalar<T> left, PrimitiveParam right) => left >= (Scalar<T>)right;
        /// <summary>Scalar greater-or-equal with a primitive constant, yielding a bit scalar.</summary>
        public static Scalar<bit> operator >=(PrimitiveParam left, Scalar<T> right) => (Scalar<T>)left >= right;
        /// <summary>Scalar less-than with a primitive constant, yielding a bit scalar.</summary>
        public static Scalar<bit> operator <(Scalar<T> left, PrimitiveParam right) => left < (Scalar<T>)right;
        /// <summary>Scalar less-than with a primitive constant, yielding a bit scalar.</summary>
        public static Scalar<bit> operator <(PrimitiveParam left, Scalar<T> right) => (Scalar<T>)left < right;
        /// <summary>Scalar less-or-equal with a primitive constant, yielding a bit scalar.</summary>
        public static Scalar<bit> operator <=(Scalar<T> left, PrimitiveParam right) => left <= (Scalar<T>)right;
        /// <summary>Scalar less-or-equal with a primitive constant, yielding a bit scalar.</summary>
        public static Scalar<bit> operator <=(PrimitiveParam left, Scalar<T> right) => (Scalar<T>)left <= right;
        /// <summary>Scalar equality with a primitive constant, yielding a bit scalar.</summary>
        public static Scalar<bit> operator ==(Scalar<T> left, PrimitiveParam right) => left == (Scalar<T>)right;
        /// <summary>Scalar equality with a primitive constant, yielding a bit scalar.</summary>
        public static Scalar<bit> operator ==(PrimitiveParam left, Scalar<T> right) => (Scalar<T>)left == right;
        /// <summary>Scalar inequality with a primitive constant, yielding a bit scalar.</summary>
        public static Scalar<bit> operator !=(Scalar<T> left, PrimitiveParam right) => left != (Scalar<T>)right;
        /// <summary>Scalar inequality with a primitive constant, yielding a bit scalar.</summary>
        public static Scalar<bit> operator !=(PrimitiveParam left, Scalar<T> right) => (Scalar<T>)left != right;

        #endregion

        #endregion
        #region Onxx Cast Operators

        //public static explicit operator Scalar<int>(Scalar<T> tensor) => tensor.Cast<int>();
        //public static explicit operator Scalar<uint32>(Scalar<T> tensor) => tensor.Cast<uint>();
        //public static explicit operator Scalar<int64>(Scalar<T> tensor) => tensor.Cast<long>();
        //public static explicit operator Scalar<ulong>(Scalar<T> tensor) => tensor.Cast<ulong>();
        //public static explicit operator Scalar<Half>(Scalar<T> tensor) => tensor.Cast<Half>();
        //public static explicit operator Scalar<Float16>(Scalar<T> tensor) => tensor.Cast<Float16>();
        //public static explicit operator Scalar<BFloat16>(Scalar<T> tensor) => tensor.Cast<BFloat16>();
        //public static explicit operator Scalar<float>(Scalar<T> tensor) => tensor.Cast<float>();
        //public static explicit operator Scalar<double>(Scalar<T> tensor) => tensor.Cast<double>();
        //public static explicit operator Scalar<bit>(Scalar<T> tensor) => tensor.Cast<bool>();

        #endregion
    }
}

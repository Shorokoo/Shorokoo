namespace Shorokoo.Tests.Modules
{
    // ===================================================================
    //  Modules that exercise QuickExecutionEngine op handlers which the
    //  AutoGrad-focused Coverage suite never touches. Each module chains
    //  several related op invocations so a single Coverage test (driving
    //  one Module class) widens QEE coverage across multiple branches.
    // ===================================================================

    /// <summary>ArgMax/ArgMin in their four shape variants (axis 0 vs -1; keepdims true vs false).</summary>
    [Module]
    public partial class QeeArgOpsCheck
    {
        public static (Tensor<int64>, Tensor<int64>, Tensor<int64>, Tensor<int64>) Inline(Tensor<float32> x)
        {
            var amax0 = (Tensor<int64>)OnnxOp.ArgMax(x, axis: 0, keepdims: false, selectLastIndex: false);
            var amaxK = (Tensor<int64>)OnnxOp.ArgMax(x, axis: -1, keepdims: true, selectLastIndex: true);
            var amin0 = (Tensor<int64>)OnnxOp.ArgMin(x, axis: 0, keepdims: false, selectLastIndex: false);
            var aminK = (Tensor<int64>)OnnxOp.ArgMin(x, axis: -1, keepdims: true, selectLastIndex: false);
            return (amax0, amaxK, amin0, aminK);
        }
    }

    /// <summary>EyeLike with default dtype + EyeLike with explicit Int64 dtype and a non-zero diagonal offset.</summary>
    [Module]
    public partial class QeeEyeLikeOpsCheck
    {
        public static (Tensor<float32>, Tensor<int64>) Inline(Tensor<float32> x)
        {
            var e1 = (Tensor<float32>)OnnxOp.EyeLike(x, dtype: null, k: 0);
            var e2 = (Tensor<int64>)OnnxOp.EyeLike(x, dtype: DType.Int64, k: 1);
            return (e1, e2);
        }
    }

    /// <summary>
    /// RandomNormal / RandomUniform (attribute-based shape) + RandomNormalLike / RandomUniformLike
    /// (input-shape) — covers every Random* op in one module.
    /// </summary>
    [Module]
    public partial class QeeRandomOpsCheck
    {
        public static (Tensor<float32>, Tensor<float32>, Tensor<float32>, Tensor<float32>) Inline(Tensor<float32> x)
        {
            var rn = (Tensor<float32>)OnnxOp.RandomNormal(new long[] { 2L, 3L }, mean: 0f, scale: 1f, dtype: DType.Float32, seed: 42f);
            var ru = (Tensor<float32>)OnnxOp.RandomUniform(new long[] { 2L, 3L }, high: 1f, low: 0f, dtype: DType.Float32, seed: 7f);
            var rnl = (Tensor<float32>)OnnxOp.RandomNormalLike(x, mean: 0f, scale: 1f, dtype: null, seed: 1f);
            var rul = (Tensor<float32>)OnnxOp.RandomUniformLike(x, high: 1f, low: 0f, dtype: null, seed: 1f);
            return (rn, ru, rnl, rul);
        }
    }

    /// <summary>shrk_RandomNormal + shrk_RandomUniform (dynamic-shape Shorokoo-internal variants).</summary>
    [Module]
    public partial class QeeShrkRandomOpsCheck
    {
        public static (Tensor<float32>, Tensor<float32>) Inline(Vector<int64> shape)
        {
            var srn = (Tensor<float32>)InternalOp.RandomNormal(shape, mean: 0f, scale: 1f, seed: 2f);
            var sru = (Tensor<float32>)InternalOp.RandomUniform(shape, high: 1f, low: 0f, seed: 3f);
            return (srn, sru);
        }
    }

    /// <summary>OptionalHasElement on an Optional-wrapped runtime input.</summary>
    [Module]
    public partial class QeeOptionalHasElementCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var opt = OnnxOp.Optional(x, DataStructure.Tensor, DType.Float32);
            return (Scalar<bit>)OnnxOp.OptionalHasElement(opt);
        }
    }

    /// <summary>
    /// Shorokoo-internal ops that never reach ORT: SequenceSlice + SequenceConcat
    /// (runtime int64 bounds keep them alive through FastFoldSequences), plus the Loop
    /// placeholder ops (LoopScanZombie, LoopFakeInput, LoopIndexVariable). Routed via
    /// the QeeOnly helper since AdvancedTestGraph's leading ComputeContext.Execute
    /// can't run these op codes.
    /// </summary>
    [Module]
    public partial class QeeInternalSequenceLoopOpsCheck
    {
        public static (Tensor<float32>, Tensor<float32>, Tensor<float32>, Tensor<float32>, Tensor<int64>) Inline(
            Tensor<float32> x, Scalar<float32> a, Scalar<float32> b, Scalar<float32> c)
        {
            var aV = (Tensor<float32>)OnnxOp.Reshape(a, Vector(1L), allowZero: false);
            var bV = (Tensor<float32>)OnnxOp.Reshape(b, Vector(1L), allowZero: false);
            var cV = (Tensor<float32>)OnnxOp.Reshape(c, Vector(1L), allowZero: false);

            // SequenceSlice with runtime int64 bounds.
            var seq = OnnxOp.SequenceConstruct(aV, bV, cV);
            var startL = Scalar(0f).Cast<int64>();
            var endL = Scalar(2f).Cast<int64>();
            var sliced = InternalOp.SequenceSlice(seq, startL, endL);
            var sliceElem = (Tensor<float32>)OnnxOp.SequenceAt(sliced, Scalar(0f).Cast<int64>());

            // SequenceConcat of two single-element sequences.
            var seq1 = OnnxOp.SequenceConstruct(aV);
            var seq2 = OnnxOp.SequenceConstruct(bV);
            var merged = InternalOp.SequenceConcat(new[] { seq1, seq2 });
            var concatElem = (Tensor<float32>)OnnxOp.SequenceAt(merged, Scalar(0f).Cast<int64>());

            // Standalone Loop placeholder ops.
            var scanZombie = (Tensor<float32>)OnnxOp.LoopScanZombie(x);
            var fakeInput = (Tensor<float32>)OnnxOp.LoopFakeInput(DType.Float32, rank: 2, DataStructure.Tensor);
            var loopIdx = (Tensor<int64>)OnnxOp.LoopIndexVariable();

            return (sliceElem, concatElem, scanZombie, fakeInput, loopIdx);
        }
    }

    /// <summary>Constant op with every typed attribute branch: value_int, value_ints, value_float, value_floats.</summary>
    [Module]
    public partial class QeeConstantOpsCheck
    {
        public static (Tensor<int64>, Tensor<int64>, Tensor<float32>, Tensor<float32>) Inline()
        {
            var ci = (Tensor<int64>)OnnxOp.Constant(42L);
            var cis = (Tensor<int64>)OnnxOp.Constant(new long[] { 1L, 2L, 3L });
            var cf = (Tensor<float32>)OnnxOp.Constant(2.5f);
            var cfs = (Tensor<float32>)OnnxOp.Constant(new float[] { 1.5f, 2.5f, 3.5f });
            return (ci, cis, cf, cfs);
        }
    }

    /// <summary>Integer Abs / Neg / Sign — hit UnaryNumericOp.IntData / ApplyInt.</summary>
    [Module]
    public partial class QeeIntUnaryOpsCheck
    {
        public static (Tensor<int64>, Tensor<int64>, Tensor<int64>) Inline(Tensor<int64> x)
        {
            var a = (Tensor<int64>)OnnxOp.Abs(x);
            var n = (Tensor<int64>)OnnxOp.Neg(x);
            var s = (Tensor<int64>)OnnxOp.Sign(x);
            return (a, n, s);
        }
    }

    /// <summary>Integer Less / LessOrEqual — hit CompareOp.CompareInt.</summary>
    [Module]
    public partial class QeeIntCompareOpsCheck
    {
        public static (Tensor<bit>, Tensor<bit>) Inline(Tensor<int64> a, Tensor<int64> b)
        {
            var lt = (Tensor<bit>)OnnxOp.Less(a, b);
            var le = (Tensor<bit>)OnnxOp.LessOrEqual(a, b);
            return (lt, le);
        }
    }

    /// <summary>Integer Pow / Mod — hit BinaryNumericOp.ApplyInt.</summary>
    [Module]
    public partial class QeeIntBinaryOpsCheck
    {
        public static (Tensor<int64>, Tensor<int64>) Inline(Tensor<int64> a, Tensor<int64> b)
        {
            var p = (Tensor<int64>)OnnxOp.Pow(a, b);
            var m = (Tensor<int64>)OnnxOp.Mod(a, b, fmod: false);
            return (p, m);
        }
    }

    /// <summary>BitwiseAnd / Or / Xor / Not on int64.</summary>
    [Module]
    public partial class QeeBitwiseOpsCheck
    {
        public static (Tensor<int64>, Tensor<int64>, Tensor<int64>, Tensor<int64>) Inline(Tensor<int64> a, Tensor<int64> b)
        {
            var bAnd = (Tensor<int64>)OnnxOp.BitwiseAnd(a, b);
            var bOr = (Tensor<int64>)OnnxOp.BitwiseOr(a, b);
            var bXor = (Tensor<int64>)OnnxOp.BitwiseXor(a, b);
            var bNot = (Tensor<int64>)OnnxOp.BitwiseNot(a);
            return (bAnd, bOr, bXor, bNot);
        }
    }

    /// <summary>
    /// Float Ceil + bool Xor + Range over floats — three small ops that share no input dtype,
    /// so they coexist in one module via separate parameters.
    /// </summary>
    [Module]
    public partial class QeeMiscFloatBoolOpsCheck
    {
        public static (Tensor<float32>, Tensor<bit>, Tensor<float32>) Inline(Tensor<float32> f, Tensor<bit> b1, Tensor<bit> b2)
        {
            var c = (Tensor<float32>)OnnxOp.Ceil(f);
            var x = (Tensor<bit>)OnnxOp.Xor(b1, b2);
            var r = (Tensor<float32>)OnnxOp.Range(Scalar(0f), Scalar(5f), Scalar(0.5f));
            return (c, x, r);
        }
    }

    /// <summary>Celu / Elu / Selu / LeakyRelu — every FloatData inner loop in the activations group.</summary>
    [Module]
    public partial class QeeActivationsCheck
    {
        public static (Tensor<float32>, Tensor<float32>, Tensor<float32>, Tensor<float32>) Inline(Tensor<float32> x)
        {
            var celu = (Tensor<float32>)OnnxOp.Celu(x, alpha: 1.0f);
            var elu = (Tensor<float32>)OnnxOp.Elu(x, alpha: 1.5f);
            var selu = (Tensor<float32>)OnnxOp.Selu(x, alpha: 1.0507f, gamma: 1.6732f);
            var lr = (Tensor<float32>)OnnxOp.LeakyRelu(x, alpha: 0.01f);
            return (celu, elu, selu, lr);
        }
    }

    /// <summary>Cast to Bool from each source dtype (float / int / bool) — Cast.cs Bool-target branch.</summary>
    [Module]
    public partial class QeeCastToBoolOpsCheck
    {
        public static (Tensor<bit>, Tensor<bit>, Tensor<bit>) Inline(Tensor<float32> f, Tensor<int64> i, Tensor<bit> b)
        {
            var cf = (Tensor<bit>)OnnxOp.Cast(f, saturate: null, to: DType.Bool);
            var ci = (Tensor<bit>)OnnxOp.Cast(i, saturate: null, to: DType.Bool);
            var cb = (Tensor<bit>)OnnxOp.Cast(b, saturate: null, to: DType.Bool);
            return (cf, ci, cb);
        }
    }

    /// <summary>
    /// Self-checking f32→f16→f32 and f32→bf16→f32 Cast roundtrips. Driven with values
    /// exactly representable in BOTH half formats, so the roundtripped tensor must equal
    /// the input bit-for-bit when executed for real (ComputeContext/ORT). Also exercises
    /// the QEE Cast float16/bfloat16 dtype propagation and the f16/bf16 ONNX
    /// export/import roundtrip inside AdvancedTestGraph. Used by DTypeF16Tests.
    /// </summary>
    [Module]
    public partial class DTypeF16CastRoundtripCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var viaF16 = x.Cast<float16>().Cast<float32>();
            var viaBf16 = x.Cast<bfloat16>().Cast<float32>();
            var diff =
                (viaF16 - x).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar() +
                (viaBf16 - x).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            return diff < Scalar(1e-6f);
        }
    }

    /// <summary>
    /// Routes inputs of every signed non-{float32,int64} dtype through QEE so the
    /// Float64 / Int32 / Int16 / Int8 branches in TensorDataConverter.ToRuntimeTensor fire.
    /// </summary>
    [Module]
    public partial class QeeDtypeIdentitySignedOpsCheck
    {
        public static (Tensor<float64>, Tensor<int32>, Tensor<int16>, Tensor<int8>) Inline(
            Tensor<float64> f64, Tensor<int32> i32, Tensor<int16> i16, Tensor<int8> i8)
            => (
                (Tensor<float64>)OnnxOp.Identity(f64, rank: null),
                (Tensor<int32>)OnnxOp.Identity(i32, rank: null),
                (Tensor<int16>)OnnxOp.Identity(i16, rank: null),
                (Tensor<int8>)OnnxOp.Identity(i8, rank: null)
            );
    }

    /// <summary>
    /// Routes inputs of every unsigned dtype plus Bool through QEE so the
    /// UInt8 / UInt16 / UInt32 / UInt64 / Bool branches in TensorDataConverter.ToRuntimeTensor fire.
    /// </summary>
    [Module]
    public partial class QeeDtypeIdentityUnsignedOpsCheck
    {
        public static (Tensor<uint8>, Tensor<uint16>, Tensor<uint32>, Tensor<uint64>, Tensor<bit>) Inline(
            Tensor<uint8> u8, Tensor<uint16> u16, Tensor<uint32> u32, Tensor<uint64> u64, Tensor<bit> b)
            => (
                (Tensor<uint8>)OnnxOp.Identity(u8, rank: null),
                (Tensor<uint16>)OnnxOp.Identity(u16, rank: null),
                (Tensor<uint32>)OnnxOp.Identity(u32, rank: null),
                (Tensor<uint64>)OnnxOp.Identity(u64, rank: null),
                (Tensor<bit>)OnnxOp.Identity(b, rank: null)
            );
    }
}

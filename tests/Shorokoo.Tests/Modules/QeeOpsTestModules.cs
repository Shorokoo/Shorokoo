namespace Shorokoo.Tests.Modules
{
    // ===================================================================
    //  Modules for the QuickExecutionEngine op handlers that no
    //  Qee*AuditModules module reaches. Everything else these files once
    //  covered is subsumed by the self-checking audit modules, which drive
    //  the same handlers with in-graph value/shape assertions.
    // ===================================================================

    /// <summary>InternalOp.SequenceSlice + InternalOp.SequenceConcat (runtime int64 bounds
    /// keep them alive through FastFoldSequences). No ONNX op-set registration, so this runs
    /// QEE-only.</summary>
    [Module]
    public partial class QeeInternalSequenceOpsCheck
    {
        public static (Tensor<float32>, Tensor<float32>) Inline(
            Scalar<float32> a, Scalar<float32> b, Scalar<float32> c)
        {
            var aV = (Tensor<float32>)OnnxOp.Reshape(a, Vector(1L), allowZero: false);
            var bV = (Tensor<float32>)OnnxOp.Reshape(b, Vector(1L), allowZero: false);
            var cV = (Tensor<float32>)OnnxOp.Reshape(c, Vector(1L), allowZero: false);

            var seq = OnnxOp.SequenceConstruct(aV, bV, cV);
            var sliced = InternalOp.SequenceSlice(seq, Scalar(0f).Cast<int64>(), Scalar(2f).Cast<int64>());
            var sliceElem = (Tensor<float32>)OnnxOp.SequenceAt(sliced, Scalar(0f).Cast<int64>());

            var merged = InternalOp.SequenceConcat([OnnxOp.SequenceConstruct(aV), OnnxOp.SequenceConstruct(bV)]);
            var concatElem = (Tensor<float32>)OnnxOp.SequenceAt(merged, Scalar(0f).Cast<int64>());

            return (sliceElem, concatElem);
        }
    }

    /// <summary>Constant value_int / value_ints / value_float / value_floats (the audit
    /// modules cover only value_string / value_strings).</summary>
    [Module]
    public partial class QeeConstantOpsCheck
    {
        public static (Tensor<int64>, Tensor<int64>, Tensor<float32>, Tensor<float32>) Inline()
            => (
                (Tensor<int64>)OnnxOp.Constant(42L),
                (Tensor<int64>)OnnxOp.Constant((long[])[1L, 2L, 3L]),
                (Tensor<float32>)OnnxOp.Constant(2.5f),
                (Tensor<float32>)OnnxOp.Constant((float[])[1.5f, 2.5f, 3.5f])
            );
    }

    /// <summary>NegOp.ApplyInt + SignOp.ApplyInt + PowOp.ApplyInt — the audit modules only
    /// reach the float arms of these three.</summary>
    [Module]
    public partial class QeeIntArithmeticOpsCheck
    {
        public static (Tensor<int64>, Tensor<int64>, Tensor<int64>) Inline(
            Tensor<int64> x, Tensor<int64> a, Tensor<int64> b)
            => (
                (Tensor<int64>)OnnxOp.Neg(x),
                (Tensor<int64>)OnnxOp.Sign(x),
                (Tensor<int64>)OnnxOp.Pow(a, b)
            );
    }

    /// <summary>BitwiseNot on int64 — the unmasked arm (the audit module uses uint32, which
    /// takes the width-masked arm).</summary>
    [Module]
    public partial class QeeBitwiseNotInt64Check
    {
        public static Tensor<int64> Inline(Tensor<int64> a) => (Tensor<int64>)OnnxOp.BitwiseNot(a);
    }

    /// <summary>Cast int→bool and bool→bool (the audit module covers only float→bool).</summary>
    [Module]
    public partial class QeeCastToBoolOpsCheck
    {
        public static (Tensor<bit>, Tensor<bit>) Inline(Tensor<int64> i, Tensor<bit> b)
            => (
                (Tensor<bit>)OnnxOp.Cast(i, saturate: null, to: DType.Bool),
                (Tensor<bit>)OnnxOp.Cast(b, saturate: null, to: DType.Bool)
            );
    }

    /// <summary>EyeLike with the k=0 main diagonal (the audit module uses k=1 and k=-1).</summary>
    [Module]
    public partial class QeeEyeLikeOpsCheck
    {
        public static Tensor<float32> Inline(Tensor<float32> x)
            => (Tensor<float32>)OnnxOp.EyeLike(x, dtype: null, k: 0);
    }

    /// <summary>Einsum trace form "ij,ji-&gt;" — the only equation with EMPTY output labels.</summary>
    [Module]
    public partial class QeeEinsumTraceCheck
    {
        public static Tensor<float32> Inline(Tensor<float32> mat)
            => (Tensor<float32>)OnnxOp.Einsum((Variable[])[mat, mat], equation: "ij,ji->");
    }

    /// <summary>OnnxOp.Size — the audit modules use SizeTensor(), which lowers to
    /// Shape + ReduceProd rather than a SIZE node.</summary>
    [Module]
    public partial class QeeSizeCheck
    {
        public static Tensor<int64> Inline(Tensor<float32> x) => (Tensor<int64>)OnnxOp.Size(x);
    }

    /// <summary>Multinomial with an EXPLICIT dtype (the audit module leaves it unset, taking
    /// the int32-default arm).</summary>
    [Module]
    public partial class QeeMultinomialCheck
    {
        public static Tensor<int64> Inline(Tensor<float32> input)
            => (Tensor<int64>)OnnxOp.Multinomial(input, dtype: DType.Int64, sampleSize: 5L, seed: 42f);
    }

    /// <summary>ImageDecoder shape inference. Driven by QeeImageRandomRnnAuditTests.</summary>
    [Module]
    public partial class QeeImageDecoderCheck
    {
        public static Tensor<uint8> Inline(Vector<uint8> encoded)
            => (Tensor<uint8>)OnnxOp.ImageDecoder(encoded, pixelFormat: "RGB");
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

    /// <summary>The Float64 / Int32 / Int16 / Int8 branches of
    /// TensorDataConverter.ToRuntimeTensor.</summary>
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

    /// <summary>The UInt8 / UInt16 / UInt32 / UInt64 / Bool branches of
    /// TensorDataConverter.ToRuntimeTensor.</summary>
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

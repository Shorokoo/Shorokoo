using Shorokoo.Core.Inference.Abstractions;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage-purpose tests for <see cref="Globals"/> constructor methods in
/// <c>Core/Global.Constructors.cs</c>. The bulk of that file is per-DType
/// constructor catalog (13 arms per concept) that the broader Modules
/// coverage suite only exercises for Float32 and Int64. This file packs the
/// remaining DType arms into a few <c>[Module]</c> modules whose Inline
/// bodies invoke the constructors directly and discard the results.
///
/// <para>
/// The constructors run during the source-generated <c>ComputationGraph</c>
/// build (calling Inline once at type-init), so coverage is captured even
/// when the produced nodes are orphans and get pruned during concretization.
/// </para>
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class GlobalConstructorsCoverageTests
{
    [Fact]
    public void TestGlobalConstructorsCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<ScalarAndFillDispatcherModel>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])]));
        Assert.True(AutoTest.AdvancedTestGraph<PureDataConstructorModel>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])]));
        Assert.True(AutoTest.AdvancedTestGraph<TensorStructFactoryModel>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])]));
        Assert.True(AutoTest.AdvancedTestGraph<VariableAndInputConstructorsModel>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])]));
    }
}

/// <summary>
/// Exercises the per-DType dispatch arms of <c>Scalar(object)</c>,
/// <c>Scalar&lt;T&gt;(object)</c>, <c>TensorFill</c> (typed + generic),
/// <c>VectorFill</c> (long + Scalar&lt;int64&gt; shape variants),
/// <c>EmptyVector&lt;T&gt;()</c>, and <c>DefaultScalar&lt;T&gt;()</c>.
/// Each call produces an orphan graph constant that is pruned during
/// concretization; the constructors themselves run at module-build time
/// so coverage is captured.
/// </summary>
[Module]
public partial class ScalarAndFillDispatcherModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        // Scalar(object) — runtime-type switch.
        _ = Scalar((object)true);
        _ = Scalar((object)(sbyte)1);
        _ = Scalar((object)(short)2);
        _ = Scalar((object)3);
        _ = Scalar((object)4L);
        _ = Scalar((object)(byte)5);
        _ = Scalar((object)(ushort)6);
        _ = Scalar((object)7u);
        _ = Scalar((object)8UL);
        _ = Scalar((object)(BFloat16)0.5f);
        _ = Scalar((object)(Float16)0.5f);
        _ = Scalar((object)0.5f);
        _ = Scalar((object)0.5);

        // Scalar<T>(object) — generic-type switch.
        _ = Scalar<bit>(true);
        _ = Scalar<int8>((sbyte)1);
        _ = Scalar<int16>((short)2);
        _ = Scalar<int32>(3);
        _ = Scalar<int64>(4L);
        _ = Scalar<uint8>((byte)5);
        _ = Scalar<uint16>((ushort)6);
        _ = Scalar<uint32>(7u);
        _ = Scalar<uint64>(8UL);
        _ = Scalar<bfloat16>((BFloat16)0.5f);
        _ = Scalar<float16>((Float16)0.5f);
        // Non-half numeric values into a half target exercise the float-conversion branch
        // (a bare unbox would throw); this is the path the implicit primitive -> Scalar<T>
        // conversions reach for `Scalar<bfloat16> x = 5;`.
        _ = Scalar<bfloat16>(5);
        _ = Scalar<float16>(5);
        _ = Scalar<bfloat16>(0.5);
        _ = Scalar<float16>(0.5);
        _ = Scalar<float32>(9f);
        _ = Scalar<float64>(10.0);

        // Scalar<IGenericTypeN>(object) — exercises the CreateGenericScalar<T>
        // branch of Scalar<T> (routed to via IsAssignableTo(typeof(IGenericType))).
        _ = Scalar<IGenericType1>((object)true);
        _ = Scalar<IGenericType2>((object)(sbyte)1);
        _ = Scalar<IGenericType3>((object)(short)2);
        _ = Scalar<IGenericType4>((object)3);
        _ = Scalar<IGenericType5>((object)4L);
        _ = Scalar<IGenericType6>((object)(byte)5);
        _ = Scalar<IGenericType7>((object)(ushort)6);
        _ = Scalar<IGenericType8>((object)7u);
        _ = Scalar<IGenericType1>((object)8UL);
        _ = Scalar<IGenericType2>((object)(BFloat16)0.5f);
        _ = Scalar<IGenericType3>((object)(Float16)0.5f);
        _ = Scalar<IGenericType4>((object)9f);
        _ = Scalar<IGenericType5>((object)10.0);

        // DefaultScalar<T>().
        _ = DefaultScalar<bit>();
        _ = DefaultScalar<int8>();
        _ = DefaultScalar<int16>();
        _ = DefaultScalar<int32>();
        _ = DefaultScalar<int64>();
        _ = DefaultScalar<uint8>();
        _ = DefaultScalar<uint16>();
        _ = DefaultScalar<uint32>();
        _ = DefaultScalar<uint64>();
        _ = DefaultScalar<bfloat16>();
        _ = DefaultScalar<float16>();
        _ = DefaultScalar<float32>();
        _ = DefaultScalar<float64>();

        // EmptyVector<T>().
        _ = EmptyVector<bit>();
        _ = EmptyVector<int8>();
        _ = EmptyVector<int16>();
        _ = EmptyVector<int32>();
        _ = EmptyVector<int64>();
        _ = EmptyVector<uint8>();
        _ = EmptyVector<uint16>();
        _ = EmptyVector<uint32>();
        _ = EmptyVector<uint64>();
        _ = EmptyVector<bfloat16>();
        _ = EmptyVector<float16>();
        _ = EmptyVector<float32>();

        // TensorFill(Vector<int64>, T) — 13 typed arms.
        var shape = Vector(2L, 3L);
        _ = TensorFill(shape, true);
        _ = TensorFill(shape, (sbyte)1);
        _ = TensorFill(shape, (short)2);
        _ = TensorFill(shape, 3);
        _ = TensorFill(shape, 4L);
        _ = TensorFill(shape, (byte)5);
        _ = TensorFill(shape, (ushort)6);
        _ = TensorFill(shape, 7u);
        _ = TensorFill(shape, 8UL);
        _ = TensorFill(shape, (BFloat16)0.5f);
        _ = TensorFill(shape, (Float16)0.5f);
        _ = TensorFill(shape, 9f);
        _ = TensorFill(shape, 10.0);

        // TensorFill<T>(shape, object) — generic dispatcher.
        _ = TensorFill<bit>(shape, true);
        _ = TensorFill<int8>(shape, (sbyte)1);
        _ = TensorFill<int16>(shape, (short)2);
        _ = TensorFill<int32>(shape, 3);
        _ = TensorFill<int64>(shape, 4L);
        _ = TensorFill<uint8>(shape, (byte)5);
        _ = TensorFill<uint16>(shape, (ushort)6);
        _ = TensorFill<uint32>(shape, 7u);
        _ = TensorFill<uint64>(shape, 8UL);
        _ = TensorFill<bfloat16>(shape, (BFloat16)0.5f);
        _ = TensorFill<float16>(shape, (Float16)0.5f);
        _ = TensorFill<float32>(shape, 9f);
        _ = TensorFill<float64>(shape, 10.0);

        // TensorFill<IGenericTypeN>(shape, object) — exercises the
        // CreateGenericTensorFill<T> branch.
        _ = TensorFill<IGenericType1>(shape, true);
        _ = TensorFill<IGenericType2>(shape, (sbyte)1);
        _ = TensorFill<IGenericType3>(shape, (short)2);
        _ = TensorFill<IGenericType4>(shape, 3);
        _ = TensorFill<IGenericType5>(shape, 4L);
        _ = TensorFill<IGenericType6>(shape, (byte)5);
        _ = TensorFill<IGenericType7>(shape, (ushort)6);
        _ = TensorFill<IGenericType8>(shape, 7u);
        _ = TensorFill<IGenericType1>(shape, 8UL);
        _ = TensorFill<IGenericType2>(shape, (BFloat16)0.5f);
        _ = TensorFill<IGenericType3>(shape, (Float16)0.5f);
        _ = TensorFill<IGenericType4>(shape, 9f);
        _ = TensorFill<IGenericType5>(shape, 10.0);

        // VectorFill(long, T) — 13 typed arms.
        _ = VectorFill(2L, true);
        _ = VectorFill(2L, (sbyte)1);
        _ = VectorFill(2L, (short)2);
        _ = VectorFill(2L, 3);
        _ = VectorFill(2L, 4L);
        _ = VectorFill(2L, (byte)5);
        _ = VectorFill(2L, (ushort)6);
        _ = VectorFill(2L, 7u);
        _ = VectorFill(2L, 8UL);
        _ = VectorFill(2L, (BFloat16)0.5f);
        _ = VectorFill(2L, (Float16)0.5f);
        _ = VectorFill(2L, 9f);
        _ = VectorFill(2L, 10.0);

        // VectorFill(Scalar<int64>, T) — same 13 with dynamic shape.
        var slen = Scalar(2L);
        _ = VectorFill(slen, true);
        _ = VectorFill(slen, (sbyte)1);
        _ = VectorFill(slen, (short)2);
        _ = VectorFill(slen, 3);
        _ = VectorFill(slen, 4L);
        _ = VectorFill(slen, (byte)5);
        _ = VectorFill(slen, (ushort)6);
        _ = VectorFill(slen, 7u);
        _ = VectorFill(slen, 8UL);
        _ = VectorFill(slen, (BFloat16)0.5f);
        _ = VectorFill(slen, (Float16)0.5f);
        _ = VectorFill(slen, 9f);
        _ = VectorFill(slen, 10.0);

        return input * Scalar(1.0f);
    }
}

/// <summary>
/// Exercises pure-data <see cref="Globals"/> methods that don't add graph
/// nodes: <c>TensorData</c> (every DType arm of the 3 overload families),
/// <c>TensorDataWithDefaultVals</c>, <c>TensorDataWithSmallVals</c>,
/// <c>TensorDataForConstantOfShapeFill</c>, and the <c>Enc</c>/<c>Dec</c>
/// round-trip including the BFloat16/Float16 special cases.
/// </summary>
[Module]
public partial class PureDataConstructorModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var dims = new long[] { 1L };

        // TensorData(DType, dims, params object[]).
        _ = TensorData(DType.Bool, dims, (object)true);
        _ = TensorData(DType.Int8, dims, (object)(sbyte)1);
        _ = TensorData(DType.Int16, dims, (object)(short)2);
        _ = TensorData(DType.Int32, dims, (object)3);
        _ = TensorData(DType.Int64, dims, (object)4L);
        _ = TensorData(DType.UInt8, dims, (object)(byte)5);
        _ = TensorData(DType.UInt16, dims, (object)(ushort)6);
        _ = TensorData(DType.UInt32, dims, (object)7u);
        _ = TensorData(DType.UInt64, dims, (object)8UL);
        _ = TensorData(DType.BFloat16, dims, (object)(BFloat16)0.5f);
        _ = TensorData(DType.Float16, dims, (object)(Float16)0.5f);
        _ = TensorData(DType.Float32, dims, (object)9f);
        _ = TensorData(DType.Float64, dims, (object)10.0);

        // TensorDataWithDefaultVals and TensorDataWithSmallVals — all DTypes.
        foreach (var dt in new[] { DType.Bool, DType.Int8, DType.Int16, DType.Int32, DType.Int64,
                                   DType.UInt8, DType.UInt16, DType.UInt32, DType.UInt64,
                                   DType.BFloat16, DType.Float16, DType.Float32, DType.Float64 })
        {
            _ = TensorDataWithDefaultVals(dt, dims);
            _ = TensorDataWithSmallVals(dt, dims);
            _ = TensorDataForConstantOfShapeFill(dt);
        }

        // Enc/Dec round-trip — 13 types, with BFloat16/Float16 special cases.
        _ = Dec<bool>(Enc(new bool[] { true }));
        _ = Dec<sbyte>(Enc(new sbyte[] { 1 }));
        _ = Dec<short>(Enc(new short[] { 2 }));
        _ = Dec<int>(Enc(new int[] { 3 }));
        _ = Dec<long>(Enc(new long[] { 4 }));
        _ = Dec<byte>(Enc(new byte[] { 5 }));
        _ = Dec<ushort>(Enc(new ushort[] { 6 }));
        _ = Dec<uint>(Enc(new uint[] { 7 }));
        _ = Dec<ulong>(Enc(new ulong[] { 8 }));
        _ = Dec<BFloat16>(Enc(new BFloat16[] { (BFloat16)0.5f }));
        _ = Dec<Float16>(Enc(new Float16[] { (Float16)0.5f }));
        _ = Dec<float>(Enc(new float[] { 9f }));
        _ = Dec<double>(Enc(new double[] { 10.0 }));

        // TensorData(DType, dims, byte[]) and TensorData(DType, dims, base64IR) —
        // both route through Dec<T> per DType. Cover every arm.
        foreach (var (dt, bytes) in new (DType, byte[])[]
        {
            (DType.Bool,     Enc(new bool[]    { true })),
            (DType.Int8,     Enc(new sbyte[]   { 1 })),
            (DType.Int16,    Enc(new short[]   { 2 })),
            (DType.Int32,    Enc(new int[]     { 3 })),
            (DType.Int64,    Enc(new long[]    { 4 })),
            (DType.UInt8,    Enc(new byte[]    { 5 })),
            (DType.UInt16,   Enc(new ushort[]  { 6 })),
            (DType.UInt32,   Enc(new uint[]    { 7 })),
            (DType.UInt64,   Enc(new ulong[]   { 8 })),
            (DType.BFloat16, Enc(new BFloat16[]{ (BFloat16)0.5f })),
            (DType.Float16,  Enc(new Float16[] { (Float16)0.5f })),
            (DType.Float32,  Enc(new float[]   { 9f })),
            (DType.Float64,  Enc(new double[]  { 10.0 })),
        })
        {
            _ = TensorData(dt, dims, bytes);
            _ = TensorData(dt, dims, Convert.ToBase64String(bytes));
        }

        // TensorData(DType) — empty 0-dim variant.
        _ = TensorData(DType.Float32);

        return input * Scalar(1.0f);
    }
}

/// <summary>
/// Exercises the surviving TensorStruct factory in Globals —
/// <see cref="Globals.TensorStruct{T}(IValue[])"/>, the DispatchProxy
/// variant that mirrors how Modules build typed TensorStructs.
/// </summary>
[Module]
public partial class TensorStructFactoryModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var first = Scalar(1.0f);
        var second = Scalar(2.0f);

        // DispatchProxy factory: TensorStruct<T>(val1, val2, ...). Returns a
        // proxy implementing T — works for any IStruct interface.
        _ = TensorStruct<GenericPairStruct>(first, second);

        return input * Scalar(1.0f);
    }
}

/// <summary>
/// Exercises the variable-construction and structure-conversion helpers:
/// <c>CreateVariable(DType, rank, structure)</c> (Tensor/Optional/Sequence
/// arms), <c>ToInput</c> (3 dispatch arms), the
/// <c>TensorSequence(DType, …)</c> / <c>OptionalTensor(DType, …)</c>
/// non-generic constructors, plus the DType-keyed <c>InputTensor</c> /
/// <c>InputScalar</c> / <c>InputVector</c> dispatchers and various
/// <c>DefaultTensor</c> / <c>DefaultVector</c> overloads.
/// </summary>
[Module]
public partial class VariableAndInputConstructorsModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        // CreateVariable — three valid DataStructure arms (TensorStruct throws).
        _ = CreateVariable(DType.Float32, 2, Shorokoo.Core.Nodes.NodeDefinitions.DataStructure.Tensor);
        _ = CreateVariable(DType.Float32, 1, Shorokoo.Core.Nodes.NodeDefinitions.DataStructure.Optional);
        _ = CreateVariable(DType.Float32, 1, Shorokoo.Core.Nodes.NodeDefinitions.DataStructure.Sequence);

        // ToInput — three dispatch arms (Tensor/Optional/Sequence).
        _ = ((Variable)Scalar(1.0f)).ToInput();
        _ = ((Variable)OptionalTensor<float32>(Vector(1.0f, 2.0f).Reshape(Vector(2L)))).ToInput();
        _ = ((Variable)TensorSequence(DType.Float32, (Variable)Vector(1.0f))).ToInput();

        // TensorSequence(DType, …) non-generic — empty + non-empty arms.
        _ = TensorSequence(DType.Float32);
        _ = TensorSequence(DType.Float32, (Variable)Vector(1.0f, 2.0f));

        // OptionalTensor(DType, …) non-generic — null + non-null arms.
        _ = OptionalTensor(DType.Float32);
        _ = OptionalTensor(DType.Float32, (Variable)Vector(1.0f));

        // DType-keyed Input* dispatchers (each routes through OnnxUtils.CallGeneric).
        _ = InputTensor(DType.Float32, "t1");
        _ = InputVector(DType.Float32, "v1");
        _ = InputScalar(DType.Float32, "s1");

        // DefaultTensor / DefaultVector overloads.
        _ = DefaultTensor<float32>(new long[] { 2L, 3L });
        _ = DefaultTensor(DType.Float32, new long[] { 2L, 3L });
        _ = DefaultVector<float32>(3L);

        // TrainableTensor overloads.
        _ = TrainableTensor(TensorData(new long[] { 2L }, 1.0f, 2.0f));
        _ = TrainableTensor<float32>(TensorData(new long[] { 2L }, 1.0f, 2.0f));

        return input * Scalar(1.0f);
    }
}

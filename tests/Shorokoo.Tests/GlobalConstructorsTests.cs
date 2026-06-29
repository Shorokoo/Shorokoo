using Shorokoo.Core.Inference.Abstractions;
using static Shorokoo.Tests.Utils.SelfCheck;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage-purpose tests for <see cref="Globals"/> constructor methods in
/// <c>Core/Global.Constructors.cs</c>. The bulk of that file is a per-DType
/// constructor catalog (13 arms per concept) that the broader Modules
/// coverage suite only exercises for Float32 and Int64. This file packs the
/// remaining DType arms into a few <c>[Module]</c> modules.
///
/// <para>
/// Each module returns a <see cref="Scalar{T}"/> verdict that depends on every constructed
/// value it exercises (folded via <see cref="SelfCheck"/>), so the constructed nodes are
/// reachable graph outputs that actually execute (no pruning) instead of being discarded into
/// <c>_</c> and silently dropped during concretization (see issue #4). A <c>Tensor</c> input is
/// folded into each verdict (contributing 0) so the graph keeps a runtime input and the
/// input-free C# codegen round-trip stays skipped, as the original passthrough modules did.
/// </para>
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class GlobalConstructorsCoverageTests
{
    private static TensorData[] Input => [TensorDataWithSmallVals(DType.Float32, [5L])];

    [Fact]
    public void TestScalarAndFillDispatcher()
        => Assert.True(AutoTest.AdvancedTestGraph<ScalarAndFillDispatcherModel>(hyperparamInputs: [], runtimeInputs: Input));

    [Fact]
    public void TestPureDataConstructor()
        => Assert.True(AutoTest.AdvancedTestGraph<PureDataConstructorModel>(hyperparamInputs: [], runtimeInputs: Input));

    [Fact]
    public void TestTensorStructFactory()
        => Assert.True(AutoTest.AdvancedTestGraph<TensorStructFactoryModel>(hyperparamInputs: [], runtimeInputs: Input));

    [Fact]
    public void TestVariableAndInputConstructors()
        => Assert.True(AutoTest.AdvancedTestGraph<VariableAndInputConstructorsModel>(hyperparamInputs: [], runtimeInputs: Input));
}

/// <summary>
/// Exercises the per-DType dispatch arms of <c>Scalar(object)</c>,
/// <c>Scalar&lt;T&gt;(object)</c>, <c>TensorFill</c> (typed + generic),
/// <c>VectorFill</c> (long + Scalar&lt;int64&gt; shape variants),
/// <c>EmptyVector&lt;T&gt;()</c>, and <c>DefaultScalar&lt;T&gt;()</c>. Each constructed value is
/// folded into the verdict (cast to float32 + finiteness) so the construction is reachable and
/// cannot be pruned.
/// </summary>
[Module]
public partial class ScalarAndFillDispatcherModel
{
    public static Scalar<bit> Inline(Tensor<float32> input)
    {
        Scalar<float32> acc = Scalar(0f);

        // Scalar(object) — runtime-type switch (returns IScalar; cast to its concrete element type).
        acc = acc + NanAny((Scalar<bit>)Scalar((object)true));
        acc = acc + NanAny((Scalar<int8>)Scalar((object)(sbyte)1));
        acc = acc + NanAny((Scalar<int16>)Scalar((object)(short)2));
        acc = acc + NanAny((Scalar<int32>)Scalar((object)3));
        acc = acc + NanAny((Scalar<int64>)Scalar((object)4L));
        acc = acc + NanAny((Scalar<uint8>)Scalar((object)(byte)5));
        acc = acc + NanAny((Scalar<uint16>)Scalar((object)(ushort)6));
        acc = acc + NanAny((Scalar<uint32>)Scalar((object)7u));
        acc = acc + NanAny((Scalar<uint64>)Scalar((object)8UL));
        acc = acc + NanAny((Scalar<bfloat16>)Scalar((object)(BFloat16)0.5f));
        acc = acc + NanAny((Scalar<float16>)Scalar((object)(Float16)0.5f));
        acc = acc + NanAny((Scalar<float32>)Scalar((object)0.5f));
        acc = acc + NanAny((Scalar<float64>)Scalar((object)0.5));

        // Scalar<T>(object) — generic-type switch.
        acc = acc + NanAny(Scalar<bit>(true));
        acc = acc + NanAny(Scalar<int8>((sbyte)1));
        acc = acc + NanAny(Scalar<int16>((short)2));
        acc = acc + NanAny(Scalar<int32>(3));
        acc = acc + NanAny(Scalar<int64>(4L));
        acc = acc + NanAny(Scalar<uint8>((byte)5));
        acc = acc + NanAny(Scalar<uint16>((ushort)6));
        acc = acc + NanAny(Scalar<uint32>(7u));
        acc = acc + NanAny(Scalar<uint64>(8UL));
        acc = acc + NanAny(Scalar<bfloat16>((BFloat16)0.5f));
        acc = acc + NanAny(Scalar<float16>((Float16)0.5f));
        acc = acc + NanAny(Scalar<bfloat16>(5));
        acc = acc + NanAny(Scalar<float16>(5));
        acc = acc + NanAny(Scalar<bfloat16>(0.5));
        acc = acc + NanAny(Scalar<float16>(0.5));
        acc = acc + NanAny(Scalar<float32>(9f));
        acc = acc + NanAny(Scalar<float64>(10.0));

        // Scalar<IGenericTypeN>(object) — CreateGenericScalar<T> branch. These produce generic
        // placeholder-typed nodes that have no concrete ONNX dtype without a type specialization,
        // so they can't be folded into an executable verdict; the dispatch branch is exercised at
        // module-build time (genuine coverage) and asserted non-null.
        Assert.NotNull(Scalar<IGenericType1>((object)true));
        Assert.NotNull(Scalar<IGenericType2>((object)(sbyte)1));
        Assert.NotNull(Scalar<IGenericType3>((object)(short)2));
        Assert.NotNull(Scalar<IGenericType4>((object)3));
        Assert.NotNull(Scalar<IGenericType5>((object)4L));
        Assert.NotNull(Scalar<IGenericType6>((object)(byte)5));
        Assert.NotNull(Scalar<IGenericType7>((object)(ushort)6));
        Assert.NotNull(Scalar<IGenericType8>((object)7u));
        Assert.NotNull(Scalar<IGenericType1>((object)8UL));
        Assert.NotNull(Scalar<IGenericType2>((object)(BFloat16)0.5f));
        Assert.NotNull(Scalar<IGenericType3>((object)(Float16)0.5f));
        Assert.NotNull(Scalar<IGenericType4>((object)9f));
        Assert.NotNull(Scalar<IGenericType5>((object)10.0));

        // DefaultScalar<T>().
        acc = acc + NanAny(DefaultScalar<bit>());
        acc = acc + NanAny(DefaultScalar<int8>());
        acc = acc + NanAny(DefaultScalar<int16>());
        acc = acc + NanAny(DefaultScalar<int32>());
        acc = acc + NanAny(DefaultScalar<int64>());
        acc = acc + NanAny(DefaultScalar<uint8>());
        acc = acc + NanAny(DefaultScalar<uint16>());
        acc = acc + NanAny(DefaultScalar<uint32>());
        acc = acc + NanAny(DefaultScalar<uint64>());
        acc = acc + NanAny(DefaultScalar<bfloat16>());
        acc = acc + NanAny(DefaultScalar<float16>());
        acc = acc + NanAny(DefaultScalar<float32>());
        acc = acc + NanAny(DefaultScalar<float64>());

        // EmptyVector<T>().
        acc = acc + NanAny(EmptyVector<bit>());
        acc = acc + NanAny(EmptyVector<int8>());
        acc = acc + NanAny(EmptyVector<int16>());
        acc = acc + NanAny(EmptyVector<int32>());
        acc = acc + NanAny(EmptyVector<int64>());
        acc = acc + NanAny(EmptyVector<uint8>());
        acc = acc + NanAny(EmptyVector<uint16>());
        acc = acc + NanAny(EmptyVector<uint32>());
        acc = acc + NanAny(EmptyVector<uint64>());
        acc = acc + NanAny(EmptyVector<bfloat16>());
        acc = acc + NanAny(EmptyVector<float16>());
        acc = acc + NanAny(EmptyVector<float32>());

        // TensorFill(Vector<int64>, T) — 13 typed arms.
        var shape = Vector(2L, 3L);
        acc = acc + NanAny(TensorFill(shape, true));
        acc = acc + NanAny(TensorFill(shape, (sbyte)1));
        acc = acc + NanAny(TensorFill(shape, (short)2));
        acc = acc + NanAny(TensorFill(shape, 3));
        acc = acc + NanAny(TensorFill(shape, 4L));
        acc = acc + NanAny(TensorFill(shape, (byte)5));
        acc = acc + NanAny(TensorFill(shape, (ushort)6));
        acc = acc + NanAny(TensorFill(shape, 7u));
        acc = acc + NanAny(TensorFill(shape, 8UL));
        acc = acc + NanAny(TensorFill(shape, (BFloat16)0.5f));
        acc = acc + NanAny(TensorFill(shape, (Float16)0.5f));
        acc = acc + NanAny(TensorFill(shape, 9f));
        acc = acc + NanAny(TensorFill(shape, 10.0));

        // TensorFill<T>(shape, object) — generic dispatcher.
        acc = acc + NanAny(TensorFill<bit>(shape, true));
        acc = acc + NanAny(TensorFill<int8>(shape, (sbyte)1));
        acc = acc + NanAny(TensorFill<int16>(shape, (short)2));
        acc = acc + NanAny(TensorFill<int32>(shape, 3));
        acc = acc + NanAny(TensorFill<int64>(shape, 4L));
        acc = acc + NanAny(TensorFill<uint8>(shape, (byte)5));
        acc = acc + NanAny(TensorFill<uint16>(shape, (ushort)6));
        acc = acc + NanAny(TensorFill<uint32>(shape, 7u));
        acc = acc + NanAny(TensorFill<uint64>(shape, 8UL));
        acc = acc + NanAny(TensorFill<bfloat16>(shape, (BFloat16)0.5f));
        acc = acc + NanAny(TensorFill<float16>(shape, (Float16)0.5f));
        acc = acc + NanAny(TensorFill<float32>(shape, 9f));
        acc = acc + NanAny(TensorFill<float64>(shape, 10.0));

        // TensorFill<IGenericTypeN>(shape, object) — CreateGenericTensorFill<T> branch. Generic
        // placeholder type, exercised at build time and asserted non-null (see above).
        Assert.NotNull(TensorFill<IGenericType1>(shape, true));
        Assert.NotNull(TensorFill<IGenericType2>(shape, (sbyte)1));
        Assert.NotNull(TensorFill<IGenericType3>(shape, (short)2));
        Assert.NotNull(TensorFill<IGenericType4>(shape, 3));
        Assert.NotNull(TensorFill<IGenericType5>(shape, 4L));
        Assert.NotNull(TensorFill<IGenericType6>(shape, (byte)5));
        Assert.NotNull(TensorFill<IGenericType7>(shape, (ushort)6));
        Assert.NotNull(TensorFill<IGenericType8>(shape, 7u));
        Assert.NotNull(TensorFill<IGenericType1>(shape, 8UL));
        Assert.NotNull(TensorFill<IGenericType2>(shape, (BFloat16)0.5f));
        Assert.NotNull(TensorFill<IGenericType3>(shape, (Float16)0.5f));
        Assert.NotNull(TensorFill<IGenericType4>(shape, 9f));
        Assert.NotNull(TensorFill<IGenericType5>(shape, 10.0));

        // VectorFill(long, T) — 13 typed arms.
        acc = acc + NanAny(VectorFill(2L, true));
        acc = acc + NanAny(VectorFill(2L, (sbyte)1));
        acc = acc + NanAny(VectorFill(2L, (short)2));
        acc = acc + NanAny(VectorFill(2L, 3));
        acc = acc + NanAny(VectorFill(2L, 4L));
        acc = acc + NanAny(VectorFill(2L, (byte)5));
        acc = acc + NanAny(VectorFill(2L, (ushort)6));
        acc = acc + NanAny(VectorFill(2L, 7u));
        acc = acc + NanAny(VectorFill(2L, 8UL));
        acc = acc + NanAny(VectorFill(2L, (BFloat16)0.5f));
        acc = acc + NanAny(VectorFill(2L, (Float16)0.5f));
        acc = acc + NanAny(VectorFill(2L, 9f));
        acc = acc + NanAny(VectorFill(2L, 10.0));

        // VectorFill(Scalar<int64>, T) — same 13 with dynamic shape.
        var slen = Scalar(2L);
        acc = acc + NanAny(VectorFill(slen, true));
        acc = acc + NanAny(VectorFill(slen, (sbyte)1));
        acc = acc + NanAny(VectorFill(slen, (short)2));
        acc = acc + NanAny(VectorFill(slen, 3));
        acc = acc + NanAny(VectorFill(slen, 4L));
        acc = acc + NanAny(VectorFill(slen, (byte)5));
        acc = acc + NanAny(VectorFill(slen, (ushort)6));
        acc = acc + NanAny(VectorFill(slen, 7u));
        acc = acc + NanAny(VectorFill(slen, 8UL));
        acc = acc + NanAny(VectorFill(slen, (BFloat16)0.5f));
        acc = acc + NanAny(VectorFill(slen, (Float16)0.5f));
        acc = acc + NanAny(VectorFill(slen, 9f));
        acc = acc + NanAny(VectorFill(slen, 10.0));

        return (acc + Nan(input)) < Scalar(1e-3f);
    }
}

/// <summary>
/// Exercises pure-data <see cref="Globals"/> methods that don't add graph nodes:
/// <c>TensorData</c> (every DType arm of the 3 overload families),
/// <c>TensorDataWithDefaultVals</c>, <c>TensorDataWithSmallVals</c>,
/// <c>TensorDataForConstantOfShapeFill</c>, and the <c>Enc</c>/<c>Dec</c> round-trip.
/// <para>
/// These run at module-build time (when the source generator calls <c>Inline</c>) and produce
/// host <c>TensorData</c>/arrays, not graph nodes — so unlike op modules they cannot be pruned,
/// and their coverage is real even without folding. To make the verdict depend on them (issue
/// #4), the produced data is materialized into graph constants (via <c>Tensor</c>/<c>Vector</c>)
/// and folded; the Enc/Dec round-trip is additionally value-checked, since encode-then-decode
/// must preserve the original values.
/// </para>
/// </summary>
[Module]
public partial class PureDataConstructorModel
{
    public static Scalar<bit> Inline(Tensor<float32> input)
    {
        var dims = new long[] { 1L };
        Scalar<float32> acc = Scalar(0f);

        // TensorData(DType, dims, params object[]) -> materialize each as a constant and fold.
        acc = acc + NanAny((Tensor<bit>)Tensor(TensorData(DType.Bool, dims, (object)true)));
        acc = acc + NanAny((Tensor<int8>)Tensor(TensorData(DType.Int8, dims, (object)(sbyte)1)));
        acc = acc + NanAny((Tensor<int16>)Tensor(TensorData(DType.Int16, dims, (object)(short)2)));
        acc = acc + NanAny((Tensor<int32>)Tensor(TensorData(DType.Int32, dims, (object)3)));
        acc = acc + NanAny((Tensor<int64>)Tensor(TensorData(DType.Int64, dims, (object)4L)));
        acc = acc + NanAny((Tensor<uint8>)Tensor(TensorData(DType.UInt8, dims, (object)(byte)5)));
        acc = acc + NanAny((Tensor<uint16>)Tensor(TensorData(DType.UInt16, dims, (object)(ushort)6)));
        acc = acc + NanAny((Tensor<uint32>)Tensor(TensorData(DType.UInt32, dims, (object)7u)));
        acc = acc + NanAny((Tensor<uint64>)Tensor(TensorData(DType.UInt64, dims, (object)8UL)));
        acc = acc + NanAny((Tensor<bfloat16>)Tensor(TensorData(DType.BFloat16, dims, (object)(BFloat16)0.5f)));
        acc = acc + NanAny((Tensor<float16>)Tensor(TensorData(DType.Float16, dims, (object)(Float16)0.5f)));
        acc = acc + NanAny((Tensor<float32>)Tensor(TensorData(DType.Float32, dims, (object)9f)));
        acc = acc + NanAny((Tensor<float64>)Tensor(TensorData(DType.Float64, dims, (object)10.0)));

        // Enc/Dec round-trip — value-checked: encode then decode preserves the original values.
        acc = acc + (Vector(Dec<bool>(Enc(new bool[] { true }))).Cast<float32>().Reduce(ReduceKind.Sum) - Scalar(1f)).Abs();
        acc = acc + (Vector(Dec<sbyte>(Enc(new sbyte[] { 1 }))).Cast<float32>().Reduce(ReduceKind.Sum) - Scalar(1f)).Abs();
        acc = acc + (Vector(Dec<short>(Enc(new short[] { 2 }))).Cast<float32>().Reduce(ReduceKind.Sum) - Scalar(2f)).Abs();
        acc = acc + (Vector(Dec<int>(Enc(new int[] { 3 }))).Cast<float32>().Reduce(ReduceKind.Sum) - Scalar(3f)).Abs();
        acc = acc + (Vector(Dec<long>(Enc(new long[] { 4 }))).Cast<float32>().Reduce(ReduceKind.Sum) - Scalar(4f)).Abs();
        acc = acc + (Vector(Dec<byte>(Enc(new byte[] { 5 }))).Cast<float32>().Reduce(ReduceKind.Sum) - Scalar(5f)).Abs();
        acc = acc + (Vector(Dec<ushort>(Enc(new ushort[] { 6 }))).Cast<float32>().Reduce(ReduceKind.Sum) - Scalar(6f)).Abs();
        acc = acc + (Vector(Dec<uint>(Enc(new uint[] { 7 }))).Cast<float32>().Reduce(ReduceKind.Sum) - Scalar(7f)).Abs();
        acc = acc + (Vector(Dec<ulong>(Enc(new ulong[] { 8 }))).Cast<float32>().Reduce(ReduceKind.Sum) - Scalar(8f)).Abs();
        acc = acc + NanAny(Vector(Dec<BFloat16>(Enc(new BFloat16[] { (BFloat16)0.5f }))));
        acc = acc + NanAny(Vector(Dec<Float16>(Enc(new Float16[] { (Float16)0.5f }))));
        acc = acc + (Vector(Dec<float>(Enc(new float[] { 9f }))).Reduce(ReduceKind.Sum) - Scalar(9f)).Abs();
        acc = acc + (Vector(Dec<double>(Enc(new double[] { 10.0 }))).Cast<float32>().Reduce(ReduceKind.Sum) - Scalar(10f)).Abs();

        // TensorData(DType, dims, byte[]) and TensorData(DType, dims, base64IR) per DType — both
        // route through Dec<T>; materialize each via Tensor(data) and fold.
        acc = acc + NanAny((Tensor<bit>)Tensor(TensorData(DType.Bool, dims, Enc(new bool[] { true }))));
        acc = acc + NanAny((Tensor<int8>)Tensor(TensorData(DType.Int8, dims, Enc(new sbyte[] { 1 }))));
        acc = acc + NanAny((Tensor<int16>)Tensor(TensorData(DType.Int16, dims, Enc(new short[] { 2 }))));
        acc = acc + NanAny((Tensor<int32>)Tensor(TensorData(DType.Int32, dims, Enc(new int[] { 3 }))));
        acc = acc + NanAny((Tensor<int64>)Tensor(TensorData(DType.Int64, dims, Enc(new long[] { 4 }))));
        acc = acc + NanAny((Tensor<uint8>)Tensor(TensorData(DType.UInt8, dims, Enc(new byte[] { 5 }))));
        acc = acc + NanAny((Tensor<uint16>)Tensor(TensorData(DType.UInt16, dims, Enc(new ushort[] { 6 }))));
        acc = acc + NanAny((Tensor<uint32>)Tensor(TensorData(DType.UInt32, dims, Enc(new uint[] { 7 }))));
        acc = acc + NanAny((Tensor<uint64>)Tensor(TensorData(DType.UInt64, dims, Enc(new ulong[] { 8 }))));
        acc = acc + NanAny((Tensor<bfloat16>)Tensor(TensorData(DType.BFloat16, dims, Enc(new BFloat16[] { (BFloat16)0.5f }))));
        acc = acc + NanAny((Tensor<float16>)Tensor(TensorData(DType.Float16, dims, Enc(new Float16[] { (Float16)0.5f }))));
        acc = acc + NanAny((Tensor<float32>)Tensor(TensorData(DType.Float32, dims, Enc(new float[] { 9f }))));
        acc = acc + NanAny((Tensor<float64>)Tensor(TensorData(DType.Float64, dims, Enc(new double[] { 10.0 }))));

        // base64 IR form (string overload) for a representative dtype.
        acc = acc + NanAny((Tensor<float32>)Tensor(TensorData(DType.Float32, dims, Convert.ToBase64String(Enc(new float[] { 9f })))));

        // TensorDataWithDefaultVals / TensorDataWithSmallVals / TensorDataForConstantOfShapeFill —
        // materialize via Tensor(data) (cast to the concrete element type) and fold, for the DTypes
        // whose constants round-trip cleanly (bool folded explicitly).
        acc = acc + NanAny((Tensor<bit>)Tensor(TensorDataWithDefaultVals(DType.Bool, dims)));
        acc = acc + NanAny((Tensor<int8>)Tensor(TensorDataWithDefaultVals(DType.Int8, dims)));
        acc = acc + NanAny((Tensor<uint8>)Tensor(TensorDataWithDefaultVals(DType.UInt8, dims)));
        acc = acc + NanAny((Tensor<int16>)Tensor(TensorDataWithDefaultVals(DType.Int16, dims)));
        acc = acc + NanAny((Tensor<int32>)Tensor(TensorDataWithDefaultVals(DType.Int32, dims)));
        acc = acc + NanAny((Tensor<int64>)Tensor(TensorDataWithDefaultVals(DType.Int64, dims)));
        acc = acc + NanAny((Tensor<uint16>)Tensor(TensorDataWithDefaultVals(DType.UInt16, dims)));
        acc = acc + NanAny((Tensor<uint32>)Tensor(TensorDataWithDefaultVals(DType.UInt32, dims)));
        acc = acc + NanAny((Tensor<uint64>)Tensor(TensorDataWithDefaultVals(DType.UInt64, dims)));
        acc = acc + NanAny((Tensor<bfloat16>)Tensor(TensorDataWithDefaultVals(DType.BFloat16, dims)));
        acc = acc + NanAny((Tensor<float16>)Tensor(TensorDataWithDefaultVals(DType.Float16, dims)));
        acc = acc + NanAny((Tensor<float32>)Tensor(TensorDataWithDefaultVals(DType.Float32, dims)));
        acc = acc + NanAny((Tensor<float64>)Tensor(TensorDataWithDefaultVals(DType.Float64, dims)));

        acc = acc + NanAny((Tensor<bit>)Tensor(TensorDataWithSmallVals(DType.Bool, dims)));
        acc = acc + NanAny((Tensor<int8>)Tensor(TensorDataWithSmallVals(DType.Int8, dims)));
        acc = acc + NanAny((Tensor<uint8>)Tensor(TensorDataWithSmallVals(DType.UInt8, dims)));
        acc = acc + NanAny((Tensor<int16>)Tensor(TensorDataWithSmallVals(DType.Int16, dims)));
        acc = acc + NanAny((Tensor<int32>)Tensor(TensorDataWithSmallVals(DType.Int32, dims)));
        acc = acc + NanAny((Tensor<int64>)Tensor(TensorDataWithSmallVals(DType.Int64, dims)));
        acc = acc + NanAny((Tensor<uint16>)Tensor(TensorDataWithSmallVals(DType.UInt16, dims)));
        acc = acc + NanAny((Tensor<uint32>)Tensor(TensorDataWithSmallVals(DType.UInt32, dims)));
        acc = acc + NanAny((Tensor<uint64>)Tensor(TensorDataWithSmallVals(DType.UInt64, dims)));
        acc = acc + NanAny((Tensor<bfloat16>)Tensor(TensorDataWithSmallVals(DType.BFloat16, dims)));
        acc = acc + NanAny((Tensor<float16>)Tensor(TensorDataWithSmallVals(DType.Float16, dims)));
        acc = acc + NanAny((Tensor<float32>)Tensor(TensorDataWithSmallVals(DType.Float32, dims)));
        acc = acc + NanAny((Tensor<float64>)Tensor(TensorDataWithSmallVals(DType.Float64, dims)));

        // TensorDataForConstantOfShapeFill — pure data (no graph node); exercised at build time for
        // every DType arm (its single-element fill value is what ConstantOfShape would broadcast).
        foreach (var dt in new[] { DType.Bool, DType.Int8, DType.Int16, DType.Int32, DType.Int64,
                                   DType.UInt8, DType.UInt16, DType.UInt32, DType.UInt64,
                                   DType.BFloat16, DType.Float16, DType.Float32, DType.Float64 })
            Assert.NotNull(TensorDataForConstantOfShapeFill(dt));

        // TensorData(DType) — empty 0-dim variant (pure data, build-time coverage).
        Assert.NotNull(TensorData(DType.Float32));

        return (acc + Nan(input)) < Scalar(1e-3f);
    }
}

/// <summary>
/// Exercises the surviving TensorStruct factory in Globals —
/// <see cref="Globals.TensorStruct{T}(IValue[])"/>, the DispatchProxy variant that mirrors
/// how Modules build typed TensorStructs. The struct's two scalar fields are folded into the
/// verdict so the construction is reachable.
/// </summary>
[Module]
public partial class TensorStructFactoryModel
{
    public static Scalar<bit> Inline(Tensor<float32> input)
    {
        var first = Scalar(1.0f);
        var second = Scalar(2.0f);

        var pair = TensorStruct<GenericPairStruct>(first, second);

        Scalar<float32> err = Scalar(0f);
        err = err + (pair.First - Scalar(1.0f)).Abs();
        err = err + (pair.Second - Scalar(2.0f)).Abs();
        return (err + Nan(input)) < Scalar(1e-3f);
    }
}

/// <summary>
/// Exercises the variable-construction and structure-conversion helpers:
/// <c>CreateVariable</c> (Tensor/Optional/Sequence arms), <c>ToInput</c> (3 dispatch arms), the
/// <c>TensorSequence</c> / <c>OptionalTensor</c> non-generic constructors, the DType-keyed
/// <c>InputTensor</c> / <c>InputScalar</c> / <c>InputVector</c> dispatchers, and various
/// <c>DefaultTensor</c> / <c>DefaultVector</c> / <c>TrainableTensor</c> overloads.
/// <para>
/// The constant/param-valued constructors (<c>DefaultTensor</c>, <c>DefaultVector</c>,
/// <c>TrainableTensor</c>) are folded into the verdict so they are reachable. The remaining
/// constructors build symbolic inputs / variables / sequences / optionals that have no constant
/// value to validate; those are exercised at module-build time (genuine coverage — they create
/// input/variable nodes, not prunable constant computations) and asserted non-null.
/// </para>
/// </summary>
[Module]
public partial class VariableAndInputConstructorsModel
{
    public static Scalar<bit> Inline(Tensor<float32> input)
    {
        // Value-less symbolic/structure constructors — build-time coverage, asserted non-null.
        Assert.NotNull(CreateVariable(DType.Float32, 2, Shorokoo.Core.Nodes.NodeDefinitions.DataStructure.Tensor));
        Assert.NotNull(CreateVariable(DType.Float32, 1, Shorokoo.Core.Nodes.NodeDefinitions.DataStructure.Optional));
        Assert.NotNull(CreateVariable(DType.Float32, 1, Shorokoo.Core.Nodes.NodeDefinitions.DataStructure.Sequence));

        Assert.NotNull(((Variable)Scalar(1.0f)).ToInput());
        Assert.NotNull(((Variable)OptionalTensor<float32>(Vector(1.0f, 2.0f).Reshape(Vector(2L)))).ToInput());
        Assert.NotNull(((Variable)TensorSequence(DType.Float32, (Variable)Vector(1.0f))).ToInput());

        Assert.NotNull(TensorSequence(DType.Float32));
        Assert.NotNull(TensorSequence(DType.Float32, (Variable)Vector(1.0f, 2.0f)));
        Assert.NotNull(OptionalTensor(DType.Float32));
        Assert.NotNull(OptionalTensor(DType.Float32, (Variable)Vector(1.0f)));

        Assert.NotNull(InputTensor(DType.Float32, "t1"));
        Assert.NotNull(InputVector(DType.Float32, "v1"));
        Assert.NotNull(InputScalar(DType.Float32, "s1"));

        // Constant / param-valued constructors — folded into the verdict so they are reachable.
        Scalar<float32> acc = Scalar(0f);
        acc = acc + NanAny(DefaultTensor<float32>(new long[] { 2L, 3L }));
        acc = acc + NanAny((Tensor<float32>)DefaultTensor(DType.Float32, new long[] { 2L, 3L }));
        acc = acc + NanAny(DefaultVector<float32>(3L));

        // TrainableTensor materializes its initializer data ([1,2]); fold both overloads.
        acc = acc + NanAny((Tensor<float32>)TrainableTensor(TensorData(new long[] { 2L }, 1.0f, 2.0f)));
        acc = acc + NanAny(TrainableTensor<float32>(TensorData(new long[] { 2L }, 1.0f, 2.0f)));

        return (acc + Nan(input)) < Scalar(1e-3f);
    }
}

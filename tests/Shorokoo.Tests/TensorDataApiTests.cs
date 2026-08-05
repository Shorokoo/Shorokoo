namespace Shorokoo.Tests;

/// <summary>
/// Coverage for the user-facing data-holder surface: <see cref="TensorData"/>,
/// <see cref="TensorDataSequence"/>, <see cref="NamedModelParam"/>, <see cref="Shape"/>
/// and the <see cref="OnnxEngine.Eval(IValue)"/> entry points.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class TensorDataApiCoverageTests
{
    private static readonly DType[] AllNumericDTypes =
    [
        DType.Bool,
        DType.Int8, DType.Int16, DType.Int32, DType.Int64,
        DType.UInt8, DType.UInt16, DType.UInt32, DType.UInt64,
        DType.Float16, DType.BFloat16, DType.Float32, DType.Float64,
    ];

    private static object[] DebugDataOf(TensorData td)
    {
        if (td.DType == DType.Bool) return td.As<bit>().DebugData;
        if (td.DType == DType.Int8) return td.As<int8>().DebugData;
        if (td.DType == DType.Int16) return td.As<int16>().DebugData;
        if (td.DType == DType.Int32) return td.As<int32>().DebugData;
        if (td.DType == DType.Int64) return td.As<int64>().DebugData;
        if (td.DType == DType.UInt8) return td.As<uint8>().DebugData;
        if (td.DType == DType.UInt16) return td.As<uint16>().DebugData;
        if (td.DType == DType.UInt32) return td.As<uint32>().DebugData;
        if (td.DType == DType.UInt64) return td.As<uint64>().DebugData;
        if (td.DType == DType.Float16) return td.As<float16>().DebugData;
        if (td.DType == DType.BFloat16) return td.As<bfloat16>().DebugData;
        if (td.DType == DType.Float32) return td.As<float32>().DebugData;
        if (td.DType == DType.Float64) return td.As<float64>().DebugData;
        throw new InvalidOperationException($"unexpected dtype {td.DType}");
    }

    [Fact]
    public void TestTensorDataDebugDataMemoryViewsFactoriesAndElementCountGuard()
    {
        foreach (var dtype in AllNumericDTypes)
        {
            var td = TensorDataWithSmallVals(dtype, [2L, 2L]);
            Assert.Equal(dtype, td.DType);
            Assert.Equal(4, DebugDataOf(td).Length);
            Assert.Contains(dtype.ToString(), td.ToString());
            Assert.NotNull(td.ToTensorValue());
            Assert.True(td.Data.Length > 0);
        }

        Assert.Equal(4, TensorDataWithSmallVals(DType.Bool, [4L]).As<bit>().AccessMemory().Length);
        Assert.Equal(4, TensorDataWithSmallVals(DType.Int8, [4L]).As<int8>().AccessMemory().Length);
        Assert.Equal(4, TensorDataWithSmallVals(DType.Int16, [4L]).As<int16>().AccessMemory().Length);
        Assert.Equal(4, TensorDataWithSmallVals(DType.Int32, [4L]).As<int32>().AccessMemory().Length);
        Assert.Equal(4, TensorDataWithSmallVals(DType.Int64, [4L]).As<int64>().AccessMemory().Length);
        Assert.Equal(4, TensorDataWithSmallVals(DType.UInt8, [4L]).As<uint8>().AccessMemory().Length);
        Assert.Equal(4, TensorDataWithSmallVals(DType.UInt16, [4L]).As<uint16>().AccessMemory().Length);
        Assert.Equal(4, TensorDataWithSmallVals(DType.UInt32, [4L]).As<uint32>().AccessMemory().Length);
        Assert.Equal(4, TensorDataWithSmallVals(DType.UInt64, [4L]).As<uint64>().AccessMemory().Length);
        Assert.Equal(4, TensorDataWithSmallVals(DType.Float16, [4L]).As<float16>().AccessMemory().Length);
        Assert.Equal(4, TensorDataWithSmallVals(DType.BFloat16, [4L]).As<bfloat16>().AccessMemory().Length);
        Assert.Equal(4, TensorDataWithSmallVals(DType.Float32, [4L]).As<float32>().AccessMemory().Length);
        Assert.Equal(4, TensorDataWithSmallVals(DType.Float64, [4L]).As<float64>().AccessMemory().Length);

        Assert.Equal(4, TensorDataWithSmallVals(DType.Int8, [4L]).As<int8>().AccessModifiableMemory().Length);
        Assert.Equal(4, TensorDataWithSmallVals(DType.Int16, [4L]).As<int16>().AccessModifiableMemory().Length);
        Assert.Equal(4, TensorDataWithSmallVals(DType.Int32, [4L]).As<int32>().AccessModifiableMemory().Length);
        Assert.Equal(4, TensorDataWithSmallVals(DType.Int64, [4L]).As<int64>().AccessModifiableMemory().Length);
        Assert.Equal(4, TensorDataWithSmallVals(DType.UInt8, [4L]).As<uint8>().AccessModifiableMemory().Length);
        Assert.Equal(4, TensorDataWithSmallVals(DType.UInt16, [4L]).As<uint16>().AccessModifiableMemory().Length);
        Assert.Equal(4, TensorDataWithSmallVals(DType.UInt32, [4L]).As<uint32>().AccessModifiableMemory().Length);
        Assert.Equal(4, TensorDataWithSmallVals(DType.UInt64, [4L]).As<uint64>().AccessModifiableMemory().Length);
        Assert.Equal(4, TensorDataWithSmallVals(DType.Float16, [4L]).As<float16>().AccessModifiableMemory().Length);
        Assert.Equal(4, TensorDataWithSmallVals(DType.BFloat16, [4L]).As<bfloat16>().AccessModifiableMemory().Length);
        Assert.Equal(4, TensorDataWithSmallVals(DType.Float64, [4L]).As<float64>().AccessModifiableMemory().Length);

        var f32 = TensorData(DType.Float32, [3L], 1f, 2f, 3f).As<float32>();
        f32.AccessModifiableMemory()[1] = 9f;
        Assert.Equal(9f, f32.AccessMemory()[1]);

        var raw = TensorData(DType.Float32, [2L], 1f, 2f);
        Assert.Equal(8, raw.AccessRawMemory().Length);
        Assert.Equal(8, raw.AccessModifiableRawMemory().Length);
        raw.Dispose();

        var bytes = new byte[8];
        BitConverter.GetBytes(1.5f).CopyTo(bytes, 0);
        BitConverter.GetBytes(-2.5f).CopyTo(bytes, 4);
        float[] expectedFromRaw = [1.5f, -2.5f];
        Assert.Equal(expectedFromRaw,
            TensorData.CreateFromRawBytes(new Shape(2L), DType.Float32, bytes).As<float32>().AccessMemory().ToArray());

        int[] expectedRange = [0, 1, 2, 3, 4, 5];
        Assert.Equal(expectedRange,
            TensorData.BuildRange(new Shape(2L, 3L)).As<int32>().AccessMemory().ToArray());

        var ex = Assert.ThrowsAny<Exception>(() => TensorData([2L, 2L], 1f, 2f, 3f));
        Assert.Contains("less than shape size", ex.Message);
    }

    [Fact]
    public void TestTensorDataSequenceCreateIndexEnumerateAndEmpty()
    {
        var a = TensorData(DType.Float32, [2L], 1f, 2f);
        var b = TensorData(DType.Float32, [2L], 3f, 4f);

        var seq = TensorDataSequence.Create([a, b], null);
        Assert.Equal(DType.Float32, seq.DType);
        Assert.Equal(2, seq.Count);
        Assert.Equal(2, ((IReadOnlyCollection<TensorData>)seq).Count);
        float[] first = [1f, 2f];
        Assert.Equal(first, seq[0].As<float32>().AccessMemory().ToArray());
        Assert.Contains("sequence", seq.ToString());
        Assert.Equal(2, seq.Count());
        Assert.Equal(2, ((System.Collections.IEnumerable)seq).Cast<object>().Count());

        var typed = seq.As<float32>();
        float[] second = [3f, 4f];
        Assert.Equal(second, typed[1].AccessMemory().ToArray());
        Assert.Equal(2, ((IEnumerable<TensorData<float32>>)typed).Count());
        Assert.Equal(2, ((System.Collections.IEnumerable)typed).Cast<object>().Count());
        Assert.Equal(2, typed.AsList.Count);

        Assert.Equal(1, TensorDataSequence.Create([a], DType.Float32).Count);

        var ex = Assert.Throws<InvalidTensorOperationException>(() => TensorDataSequence.Create([], null));
        Assert.Contains("Data cannot be empty", ex.Message);

        var empty = TensorDataSequence.Empty(DType.Int64);
        Assert.Equal(0, empty.Count);
        Assert.Equal(DType.Int64, empty.DType);

        seq.Dispose();
    }

    private sealed class UnsupportedData : IData
    {
        public DType DType => DType.Float32;
    }

    [Fact]
    public void TestNamedModelParamTensorDataSequenceAndUnsupportedIData()
    {
        var td = TensorData(DType.Float32, [2L], 1f, 2f);
        var tdp = Assert.IsType<TensorDataModelParam>(
            NamedModelParam.FromIData("weights", ModelParamType.TrainableParam, td));
        Assert.Equal("weights", tdp.ParamName);
        Assert.Equal(ModelParamType.TrainableParam, tdp.ParamType);
        Assert.Equal(DataStructure.Tensor, tdp.Structure);
        Assert.Same(td, tdp.ToTensorData());
        Assert.Same(td, tdp.ToTensorData<float32>());
        Assert.NotNull(tdp.ToTensorValue());
        Assert.Contains("weights", tdp.ToString());
        Assert.Throws<InvalidTensorOperationException>(() => tdp.ToTensorDataSequence());
        Assert.Throws<InvalidTensorOperationException>(() => tdp.ToTensorDataSequence<float32>());

        Assert.Throws<InvalidTensorOperationException>(
            () => NamedModelParam.FromIData("bad", ModelParamType.InputParam, new UnsupportedData()));

        var seq = TensorDataSequence.Create(
            [TensorData(DType.Float32, [2L], 1f, 2f), TensorData(DType.Float32, [2L], 3f, 4f)],
            DType.Float32);
        var sp = Assert.IsType<TensorDataSequenceModelParam>(
            NamedModelParam.FromIData("states", ModelParamType.InputParam, seq));
        Assert.Equal(2, sp.Count);
        Assert.Equal(DataStructure.Sequence, sp.Structure);
        Assert.Same(seq, sp.ToTensorDataSequence());
        Assert.Equal(2, sp.ToTensorDataSequence<float32>().Count);
        Assert.NotNull(sp.ToTensorValue());
        Assert.Contains("states", sp.ToString());
        Assert.Throws<InvalidTensorOperationException>(() => sp.ToTensorData());
        Assert.Throws<InvalidTensorOperationException>(() => sp.ToTensorData<float32>());
    }

    [Fact]
    public void TestShapeConstructorsConversionsEqualityAndOnnxEngineEval()
    {
        Assert.Empty(Shape.Scalar.Dims);
        Assert.Equal(1, Shape.Scalar.Count);
        Assert.Equal("()", new Shape().ToString());

        var s23 = new Shape(2L, 3L);
        Assert.Equal(6, s23.Count);
        Assert.Equal("(2,3)", s23.ToString());
        Assert.Equal("(3,)", new Shape(3L).ToString());
        ulong[] u23 = [2UL, 3UL];
        long[] l23 = [2L, 3L];
        Assert.Equal(6, new Shape(u23).Count);

        TensorDim[] knownDims = [new TensorDim { Size = 2 }, new TensorDim { Size = 4 }];
        TensorDim[] symbolicDims = [new TensorDim { Size = 2 }, new TensorDim("N")];
        Assert.Equal(8, new Shape(knownDims).Count);
        Assert.Equal(-1, new Shape(symbolicDims).Count);

        Assert.Equal(u23, (ulong[])s23);
        Assert.Equal(l23, (long[])s23);
        Assert.Equal(s23, (Shape)l23);
        Assert.Equal(s23, (Shape)u23);
        Assert.Equal(new Shape(5L), (Shape)5);
        Assert.Equal(new Shape(5L), (Shape)5u);
        Assert.Equal(new Shape(5L), (Shape)5L);
        Assert.Equal(new Shape(5L), (Shape)5UL);
        Assert.Equal(new Shape(1L, 2L), (Shape)(1UL, 2UL));
        Assert.Equal(new Shape(1L, 2L, 3L), (Shape)(1UL, 2UL, 3UL));
        Assert.Equal(new Shape(1L, 2L, 3L, 4L), (Shape)(1UL, 2UL, 3UL, 4UL));
        Assert.Equal(new Shape(1L, 2L, 3L, 4L, 5L), (Shape)(1UL, 2UL, 3UL, 4UL, 5UL));
        Assert.Equal(new Shape(1L, 2L, 3L, 4L, 5L, 6L), (Shape)(1UL, 2UL, 3UL, 4UL, 5UL, 6UL));
        Assert.Equal(new Shape(1L, 2L, 3L, 4L, 5L, 6L, 7L), (Shape)(1UL, 2UL, 3UL, 4UL, 5UL, 6UL, 7UL));

        var same = new Shape(2L, 3L);
        var sameReference = s23;
        Assert.True(s23 == same);
        Assert.False(s23 != same);
        Assert.True(s23 == sameReference);
        Assert.False(s23 == null);
        Assert.False(null == s23);
        Assert.True((Shape?)null == (Shape?)null);
        Assert.True(s23.Equals((object)same));
        Assert.False(s23.Equals((object)"nope"));
        Assert.False(s23.Equals(null));
        Assert.False(s23.Equals(new Shape(2L)));
        Assert.False(s23.Equals(new Shape(2L, 4L)));
        Assert.Equal(s23.GetHashCode(), same.GetHashCode());

        var sum = Scalar(2f) + Scalar(3f);
        var product = Scalar(2f) * Scalar(4f);
        var difference = Scalar(9f) - Scalar(1f);
        Assert.Equal(5f, OnnxEngine.Eval(sum).As<float32>().AccessMemory()[0]);

        var pair = OnnxEngine.Eval([(Variable)sum, product]);
        Assert.Equal(2, pair.Length);
        Assert.Equal(8f, pair[1].As<float32>().AccessMemory()[0]);

        var triple = OnnxEngine.Eval(sum, product, difference);
        Assert.Equal(3, triple.Length);
        Assert.Equal(8f, triple[2].As<float32>().AccessMemory()[0]);
    }
}

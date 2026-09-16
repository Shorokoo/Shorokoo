using System.Runtime.CompilerServices;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage for the user-facing data-holder surface: <see cref="TensorData"/>,
/// <see cref="TensorDataSequence"/>, <see cref="NamedModelParam"/>, <see cref="Shape"/>
/// and the <see cref="OnnxEngine.Eval(Variable)"/> entry points.
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

    [Fact]
    public void TestTensorDataDoesNotKeepTheCallerSValuesAliveOnceItIsGone()
    {
        Assert.False(SourceArrayStillReachable<float>(a => TensorData([4L], a)));
        Assert.False(SourceArrayStillReachable<byte>(
            a => TensorData.CreateFromRawBytes(new Shape(1L), DType.Float32, a)));
    }

    private static bool SourceArrayStillReachable<T>(Func<T[], TensorData> build) where T : unmanaged
    {
        var probe = BuildAndDrop(build);
        GC.Collect(2, GCCollectionMode.Forced, blocking: true);
        GC.WaitForPendingFinalizers();
        GC.Collect(2, GCCollectionMode.Forced, blocking: true);
        return probe.IsAlive;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference BuildAndDrop<T>(Func<T[], TensorData> build) where T : unmanaged
    {
        var source = new T[4];
        Assert.NotNull(build(source));
        return new WeakReference(source);
    }

    [Fact]
    public void TestDisposingATensorReleasesItsBackingValueExactlyOnce()
    {
        var spy = new SpyTensorValue();
        var td = new OnnxTensorData<float32>(new Shape(2L), spy);
        Assert.False(td.IsDisposed);
        Assert.Equal(0, spy.Disposals);

        td.Dispose();
        td.Dispose();
        Assert.True(td.IsDisposed);
        Assert.Equal(1, spy.Disposals);
    }

    [Fact]
    public void TestEveryReadOfADisposedTensorThrowsInsteadOfReadingFreedMemory()
    {
        var td = TensorData(DType.Float32, [2L], 1f, 2f);
        td.Dispose();

        Action[] reads =
        [
            () => td.AccessRawMemory(),
            () => td.AccessModifiableRawMemory(),
            () => td.As<float32>().AccessMemory<float>(),
            () => td.As<float32>().AccessModifiableMemory<float>(),
            () => td.As<float32>().AccessMemory(),
            () => td.As<float32>().AccessModifiableMemory(),
            () => _ = td.Data,
            () => _ = td.As<float32>().DebugData,
            () => td.ToTensorValue(),
            () => _ = ((IOnnxData)td).Value,
        ];
        Assert.All(reads, r => Assert.Throws<ObjectDisposedException>(r));

        // Metadata stays readable — a disposed tensor still says what it was.
        Assert.Equal(DType.Float32, td.DType);
        Assert.Equal(new Shape(2L), td.Shape);
        Assert.Contains("float", td.ToString(), StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>A disposed sequence says so on every path to its elements, including the two that
    /// used to defer: GetEnumerator validates before it hands back an iterator, and the empty
    /// sequence's indexer reports disposal ahead of the index it does not have.</summary>
    [Fact]
    public void TestEveryPathToADisposedSequencesElementsThrows()
    {
        var populated = TensorDataSequence.Create(
            [TensorData([2L], [1f, 2f]), TensorData([2L], [3f, 4f])], null);
        populated.Dispose();
        Assert.Throws<ObjectDisposedException>(() => populated.GetEnumerator());

        var empty = TensorDataSequence.internalCreateEmpty<float32>();
        empty.Dispose();
        Assert.Throws<ObjectDisposedException>(() => empty.GetEnumerator());
        Assert.Throws<ObjectDisposedException>(() => ((TensorDataSequence<float32>)empty)[0]);
    }

    [Fact]
    public void TestASequenceOwnsItsOwnElementsAndLeavesTheTensorsItWasBuiltFromAlone()
    {
        var a = TensorData(DType.Float32, [2L], 1f, 2f);
        var b = TensorData(DType.Float32, [2L], 3f, 4f);
        var seq = TensorDataSequence.Create([a, b], null);

        var element = seq[0];
        element.Dispose();
        float[] stillFirst = [1f, 2f];
        Assert.Equal(stillFirst, seq[0].As<float32>().AccessMemory().ToArray());

        seq.Dispose();
        Assert.True(seq.IsDisposed);
        Assert.Throws<ObjectDisposedException>(() => seq.Count);
        Assert.Throws<ObjectDisposedException>(() => seq[0]);

        // The sources were never the sequence's to free.
        Assert.False(a.IsDisposed);
        Assert.Equal(stillFirst, a.As<float32>().AccessMemory().ToArray());
        float[] stillSecond = [3f, 4f];
        Assert.Equal(stillSecond, b.As<float32>().AccessMemory().ToArray());
        Assert.Equal(2, TensorDataSequence.Create([a, b], null).Count);
    }

    [Fact]
    public void TestAnOutputOutlivesTheSessionThatProducedIt()
    {
        // ComputeContext.Run builds a session per call and disposes it before returning; the
        // values it hands back stay valid because an ORT tensor holds its allocator alive.
        var result = OnnxEngine.Eval(Scalar(2f) * Scalar(21f)).As<float32>();
        GC.Collect(2, GCCollectionMode.Forced, blocking: true);
        GC.WaitForPendingFinalizers();
        float[] values = result.AccessMemory().ToArray();
        GC.KeepAlive(result);
        Assert.Equal(42f, values[0]);
    }

    /// <summary>A tensor whose storage the provider kept refuses every read, naming the call that
    /// brings it home rather than dereferencing a device address as a host one. Reachable on a
    /// host-only machine only through a value that says it is not host-accessible.</summary>
    [Fact]
    public void TestATensorLeftInProviderMemoryRefusesEveryReadAndSaysWhatBringsItHome()
    {
        var resident = new OnnxTensorData<float32>(new Shape(2L), new SpyTensorValue { IsHostAccessible = false });
        Assert.False(resident.IsHostResident);

        var ex = Assert.Throws<InvalidOperationException>(() => resident.AccessMemory<float>().ToArray());
        Assert.Contains("StepToCheckpoint", ex.Message);
        Assert.Throws<InvalidOperationException>(() => resident.AccessRawMemory().ToArray());
        Assert.Throws<InvalidOperationException>(() => resident.CopyRawMemory());
        Assert.Throws<InvalidOperationException>(() => resident.ValueAt<float>(0));

        // Metadata stays readable; it is the elements that are elsewhere.
        Assert.Equal(2, resident.Shape.Dims[0]);
        Assert.False(resident.IsDisposed);

        var host = new OnnxTensorData<float32>(new Shape(2L), new SpyTensorValue());
        Assert.True(host.IsHostResident);
    }

    /// <summary>Records disposal; every other member is unreachable in these tests.</summary>
    private sealed class SpyTensorValue : Shorokoo.Core.Inference.Abstractions.IShorokooTensorValue
    {
        public int Disposals { get; private set; }
        public void Dispose() => Disposals++;

        // The one member the tensor wrapping this spy consults on its own. Settable so a test can
        // stand in for a value an execution provider left in its own memory, which is otherwise
        // only reachable with a card.
        public bool IsHostAccessible { get; init; } = true;

        public Shorokoo.Core.Inference.Abstractions.ShorokooOnnxValueType ValueType => throw new NotSupportedException();
        public Shorokoo.Core.Inference.Abstractions.ShorokooTensorElementType ElementType => throw new NotSupportedException();
        public long[] Shape => throw new NotSupportedException();
        public ReadOnlySpan<T> GetTensorDataAsSpan<T>() where T : unmanaged => throw new NotSupportedException();
        public Span<T> GetTensorMutableDataAsSpan<T>() where T : unmanaged => throw new NotSupportedException();
        public IReadOnlyList<string> GetStringTensorData() => throw new NotSupportedException();
        public int GetValueCount() => throw new NotSupportedException();
        public Shorokoo.Core.Inference.Abstractions.IShorokooTensorValue GetValue(int index) => throw new NotSupportedException();
        public Shorokoo.Core.Inference.Abstractions.ShorokooTensorElementType GetSequenceElementType() => throw new NotSupportedException();
    }
}

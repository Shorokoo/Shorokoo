namespace Shorokoo.Tests;

/// <summary>
/// Coverage-purpose tests for the user-facing data-holder surface of the main assembly:
/// <see cref="TensorData"/>/<see cref="TensorData{T}"/> (DebugData per dtype, the typed
/// AccessMemory/AccessModifiableMemory extension one-liners, CreateFromRawBytes/BuildRange),
/// <see cref="TensorDataSequence"/>/<see cref="OnnxTensorDataSequence{T}"/> (Create/Empty,
/// indexers, typed + untyped enumeration), the <see cref="NamedModelParam"/> hierarchy
/// (FromIData dispatch and the supported/unsupported conversion matrix), <see cref="Shape"/>
/// (constructors, conversions, equality, ToString), and the <see cref="OnnxEngine.Eval(IValue)"/>
/// convenience entry points. Plain xunit facts — no graph lowering is needed for these.
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

    /// <summary>Routes to the per-dtype <see cref="TensorData{T}.DebugData"/> switch arm.</summary>
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
    public void TestTensorDataDebugDataAllDTypesCoverage()
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
    }

    [Fact]
    public void TestTensorDataAccessMemoryExtensionsCoverage()
    {
        // Read-only typed views.
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

        // Mutable typed views — write through one element and read it back.
        var f32 = TensorData(DType.Float32, [3L], 1f, 2f, 3f).As<float32>();
        f32.AccessModifiableMemory()[1] = 9f;
        Assert.Equal(9f, f32.AccessMemory()[1]);
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

        // Raw views + dispose.
        var raw = TensorData(DType.Float32, [2L], 1f, 2f);
        Assert.Equal(8, raw.AccessRawMemory().Length);
        Assert.Equal(8, raw.AccessModifiableRawMemory().Length);
        raw.Dispose();
    }

    [Fact]
    public void TestTensorDataFactoriesCoverage()
    {
        // CreateFromRawBytes round-trips the payload.
        var bytes = new byte[8];
        BitConverter.GetBytes(1.5f).CopyTo(bytes, 0);
        BitConverter.GetBytes(-2.5f).CopyTo(bytes, 4);
        var fromRaw = TensorData.CreateFromRawBytes(new Shape(2L), DType.Float32, bytes);
        Assert.Equal(new[] { 1.5f, -2.5f }, fromRaw.As<float32>().AccessMemory().ToArray());

        // BuildRange produces 0..N-1 int32 values.
        var range = TensorData.BuildRange(new Shape(2L, 3L));
        Assert.Equal(new[] { 0, 1, 2, 3, 4, 5 }, range.As<int32>().AccessMemory().ToArray());
    }

    [Fact]
    public void TestTensorDataSequenceCoverage()
    {
        var a = TensorData(DType.Float32, [2L], 1f, 2f);
        var b = TensorData(DType.Float32, [2L], 3f, 4f);

        // Create with dtype inferred from the first element.
        var seq = TensorDataSequence.Create([a, b], null);
        Assert.Equal(DType.Float32, seq.DType);
        Assert.Equal(2, seq.Count);
        Assert.Equal(2, ((IReadOnlyCollection<TensorData>)seq).Count);
        Assert.Equal(new[] { 1f, 2f }, seq[0].As<float32>().AccessMemory().ToArray());
        Assert.Contains("sequence", seq.ToString());

        // Base (untyped) enumeration, generic + non-generic.
        Assert.Equal(2, seq.Count());
        Assert.Equal(2, ((System.Collections.IEnumerable)seq).Cast<object>().Count());

        // Typed view: indexer, enumeration (generic + non-generic), AsList.
        var typed = seq.As<float32>();
        Assert.Equal(new[] { 3f, 4f }, typed[1].AccessMemory().ToArray());
        Assert.Equal(2, ((IEnumerable<TensorData<float32>>)typed).Count());
        Assert.Equal(2, ((System.Collections.IEnumerable)typed).Cast<object>().Count());
        Assert.Equal(2, typed.AsList.Count);

        // Create with explicit dtype.
        var explicitDType = TensorDataSequence.Create([a], DType.Float32);
        Assert.Equal(1, explicitDType.Count);

        // Create with neither elements nor dtype is rejected.
        var ex = Assert.Throws<InvalidTensorOperationException>(() => TensorDataSequence.Create([], null));
        Assert.Contains("Data cannot be empty", ex.Message);

        seq.Dispose();
    }

    /// <summary>
    /// <see cref="TensorDataSequence.Empty"/> (and <see cref="TensorDataSequence.Create"/>
    /// with an empty list and a non-null dtype) returns a usable zero-element sequence.
    /// ORT's C# binding cannot represent zero-element sequence values, so the empty case
    /// is backed by a managed implementation (it cannot be fed to an ORT session input;
    /// use the in-graph SequenceEmpty op there).
    /// </summary>
    [Fact]
    public void TestTensorDataSequenceEmpty()
    {
        var empty = TensorDataSequence.Empty(DType.Int64);
        Assert.Equal(0, empty.Count);
        Assert.Equal(DType.Int64, empty.DType);
    }

    private sealed class UnsupportedData : IData
    {
        public DType DType => DType.Float32;
    }

    [Fact]
    public void TestNamedModelParamTensorDataCoverage()
    {
        var td = TensorData(DType.Float32, [2L], 1f, 2f);
        var p = NamedModelParam.FromIData("weights", ModelParamType.TrainableParam, td);

        var tdp = Assert.IsType<TensorDataModelParam>(p);
        Assert.Equal("weights", tdp.ParamName);
        Assert.Equal(ModelParamType.TrainableParam, tdp.ParamType);
        Assert.Equal(DataStructure.Tensor, tdp.Structure);
        Assert.Same(td, tdp.ToTensorData());
        Assert.Same(td, tdp.ToTensorData<float32>());
        Assert.NotNull(tdp.ToTensorValue());
        Assert.Contains("weights", tdp.ToString());
        Assert.Throws<InvalidTensorOperationException>(() => tdp.ToTensorDataSequence());
        Assert.Throws<InvalidTensorOperationException>(() => tdp.ToTensorDataSequence<float32>());

        // Unsupported IData implementations are rejected with a typed error.
        Assert.Throws<InvalidTensorOperationException>(
            () => NamedModelParam.FromIData("bad", ModelParamType.InputParam, new UnsupportedData()));
    }

    [Fact]
    public void TestNamedModelParamTensorDataSequenceCoverage()
    {
        var seq = TensorDataSequence.Create(
            [TensorData(DType.Float32, [2L], 1f, 2f), TensorData(DType.Float32, [2L], 3f, 4f)],
            DType.Float32);
        var p = NamedModelParam.FromIData("states", ModelParamType.InputParam, seq);

        var sp = Assert.IsType<TensorDataSequenceModelParam>(p);
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
    public void TestShapeCoverage()
    {
        // Scalar singleton + parameterless ctor.
        Assert.Empty(Shape.Scalar.Dims);
        Assert.Equal(1, Shape.Scalar.Count);
        Assert.Equal("()", new Shape().ToString());

        // long[]/ulong[] ctors + Count.
        var s23 = new Shape(2L, 3L);
        Assert.Equal(6, s23.Count);
        Assert.Equal("(2,3)", s23.ToString());
        Assert.Equal("(3,)", new Shape(3L).ToString());
        Assert.Equal(6, new Shape(new ulong[] { 2UL, 3UL }).Count);

        // TensorDim ctor: known dims multiply, symbolic dims poison Count to -1.
        Assert.Equal(8, new Shape(new[] { new TensorDim { Size = 2 }, new TensorDim { Size = 4 } }).Count);
        Assert.Equal(-1, new Shape(new[] { new TensorDim { Size = 2 }, new TensorDim("N") }).Count);

        // Conversion operators.
        Assert.Equal(new ulong[] { 2UL, 3UL }, (ulong[])s23);
        Assert.Equal(new long[] { 2L, 3L }, (long[])s23);
        Assert.Equal(s23, (Shape)new long[] { 2L, 3L });
        Assert.Equal(s23, (Shape)new ulong[] { 2UL, 3UL });
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

        // Equality surface: operators, Equals(object), Equals(Shape), GetHashCode.
        var same = new Shape(2L, 3L);
        var sameReference = s23;
        Assert.True(s23 == same);
        Assert.False(s23 != same);
        Assert.True(s23 == sameReference); // ReferenceEquals fast path
        Assert.False(s23 == null);
        Assert.False(null == s23);
        Assert.True((Shape?)null == (Shape?)null);
        Assert.True(s23.Equals((object)same));
        Assert.False(s23.Equals((object)"nope"));
        Assert.False(s23.Equals(null));
        Assert.False(s23.Equals(new Shape(2L)));
        Assert.False(s23.Equals(new Shape(2L, 4L)));
        Assert.Equal(s23.GetHashCode(), same.GetHashCode());
    }

    [Fact]
    public void TestOnnxEngineEvalCoverage()
    {
        var sum = Scalar(2f) + Scalar(3f);
        var product = Scalar(2f) * Scalar(4f);
        var difference = Scalar(9f) - Scalar(1f);

        // Single-output convenience overload.
        var single = OnnxEngine.Eval(sum);
        Assert.Equal(5f, single.As<float32>().AccessMemory()[0]);

        // Array overload.
        var pair = OnnxEngine.Eval([(Variable)sum, product]);
        Assert.Equal(2, pair.Length);
        Assert.Equal(8f, pair[1].As<float32>().AccessMemory()[0]);

        // Params (two-plus) overload.
        var triple = OnnxEngine.Eval(sum, product, difference);
        Assert.Equal(3, triple.Length);
        Assert.Equal(8f, triple[2].As<float32>().AccessMemory()[0]);
    }

    /// <summary>Analytic validation check (2026-06-12 campaign §3): TensorData with
    /// fewer values than the shape requires must throw, not accept silently.</summary>
    [Fact]
    public void TestTensorDataElementCountMismatchThrows()
    {
        var ex = Assert.ThrowsAny<Exception>(() => TensorData([2L, 2L], 1f, 2f, 3f));
        Assert.Contains("less than shape size", ex.Message);
    }
}

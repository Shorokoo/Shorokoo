using System.Text;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Core.Inference.Abstractions;
using Shorokoo.Core.Nodes.Processors.Helpers;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage for Float16/BFloat16 dtype completeness: SafeTensorLoader payload arms,
/// TensorDataConversion narrowing/widening (incl. BF16 round-to-nearest-even), the
/// ONNX int32-packed initializer narrowing and raw_data import/export, and
/// <c>EncodingBitCount</c> / IR Enc-Dec roundtrips.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class DTypeF16CoverageTests
{
    private static readonly ushort[] F16Bits =
    [
        BitConverter.HalfToUInt16Bits((Half)1.5f),
        BitConverter.HalfToUInt16Bits((Half)(-2.25f)),
        0x7BFF,
        0x0001,
    ];
    private static readonly float[] F16Floats =
        F16Bits.Select(b => (float)BitConverter.UInt16BitsToHalf(b)).ToArray();

    private static readonly ushort[] Bf16Bits = [0x3FC0, 0xC010, 0x7F7F, 0x0001];
    private static readonly float[] Bf16Floats =
        Bf16Bits.Select(b => BitConverter.UInt32BitsToSingle((uint)b << 16)).ToArray();

    private static byte[] BuildSafeTensorBuffer(params (string Name, string DType, ushort[] Bits)[] tensors)
    {
        var payload = new List<byte>();
        var headerEntries = new List<string>();
        foreach (var (name, dtype, bits) in tensors)
        {
            int start = payload.Count;
            foreach (var b in bits) payload.AddRange(BitConverter.GetBytes(b));
            headerEntries.Add(
                $"\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{bits.Length}],\"data_offsets\":[{start},{payload.Count}]}}");
        }
        var headerBytes = Encoding.UTF8.GetBytes("{" + string.Join(",", headerEntries) + "}");
        return [.. BitConverter.GetBytes((long)headerBytes.Length), .. headerBytes, .. payload];
    }

    [Fact]
    public void TestSafeTensorConversionEncodingAndOnnxProtoHalfFormats()
    {
        var buffer = BuildSafeTensorBuffer(("a", "F16", F16Bits), ("b", "BF16", Bf16Bits));
        var tensors = SafeTensorLoader.ParseSafeTensorBytes(buffer);
        Assert.Equal(2, tensors.Count);

        var a = tensors.Single(t => t.Name == "a");
        Assert.Equal("F16", a.DataType);
        Assert.True(a.Data.DType == DType.Float16);
        Assert.Equal([4L], a.Data.Shape.Dims);
        var aValues = a.Data.As<float16>().AccessMemory().ToArray();
        Assert.Equal(F16Bits, aValues.Select(x => x.Bits).ToArray());
        Assert.Equal(F16Floats, aValues.Select(x => (float)x).ToArray());

        var b = tensors.Single(t => t.Name == "b");
        Assert.Equal("BF16", b.DataType);
        Assert.True(b.Data.DType == DType.BFloat16);
        Assert.Equal([4L], b.Data.Shape.Dims);
        var bValues = b.Data.As<bfloat16>().AccessMemory().ToArray();
        Assert.Equal(Bf16Bits, bValues.Select(x => x.Bits).ToArray());
        Assert.Equal(Bf16Floats, bValues.Select(x => (float)x).ToArray());

        Assert.Equal("F16", SafeTensorLoader.DTypeToSafeTensorDType(DType.Float16));
        Assert.Equal("BF16", SafeTensorLoader.DTypeToSafeTensorDType(DType.BFloat16));

        using var stream = new MemoryStream();
        SafeTensorLoader.SaveSafeTensorsToStream(stream, tensors);
        var reloaded = SafeTensorLoader.ParseSafeTensorBytes(stream.ToArray());
        Assert.Equal(2, reloaded.Count);
        Assert.Equal(F16Bits, reloaded.Single(t => t.Name == "a")
            .Data.As<float16>().AccessMemory().ToArray().Select(x => x.Bits).ToArray());
        Assert.Equal(Bf16Bits, reloaded.Single(t => t.Name == "b")
            .Data.As<bfloat16>().AccessMemory().ToArray().Select(x => x.Bits).ToArray());

        var unsupported = BuildSafeTensorBuffer(("z", "F8_E4M3", [0x0000]));
        var unsupportedEx = Assert.Throws<InvalidOperationException>(
            () => SafeTensorLoader.ParseSafeTensorBytes(unsupported));
        Assert.Contains("F8_E4M3", unsupportedEx.Message);
        Assert.Contains("Supported formats", unsupportedEx.Message);

        long[] noDims = [];
        long[] rank1Dims = [2L];
        float[] vectorVals = [1f, 2f];
        List<SafeTensor> scalarAndVector =
        [
            new("s", TensorData(noDims, 7.5f), "F32", noDims),
            new("v", TensorData(rank1Dims, vectorVals), "F32", rank1Dims),
        ];
        using var scalarStream = new MemoryStream();
        SafeTensorLoader.SaveSafeTensorsToStream(scalarStream, scalarAndVector);
        var scalarReloaded = SafeTensorLoader.ParseSafeTensorBytes(scalarStream.ToArray());
        var s = scalarReloaded.Single(t => t.Name == "s");
        Assert.Empty(s.Data.Shape.Dims);
        Assert.Equal(7.5f, s.Data.As<float32>().AccessMemory()[0]);
        var v = scalarReloaded.Single(t => t.Name == "v");
        Assert.Equal([2L], v.Data.Shape.Dims);
        Assert.Equal(vectorVals, v.Data.As<float32>().AccessMemory().ToArray());

        var f32 = TensorData(DType.Float32, [4L],
            F16Floats[0], F16Floats[1], F16Floats[2], F16Floats[3]);
        var asF16 = TensorDataConversion.ConvertTensorDataType(f32, DType.Float16);
        Assert.True(asF16.DType == DType.Float16);
        Assert.Equal(F16Bits, asF16.As<float16>().AccessMemory().ToArray().Select(x => x.Bits).ToArray());

        var backF32 = TensorDataConversion.ConvertTensorDataType(asF16, DType.Float32);
        Assert.True(backF32.DType == DType.Float32);
        Assert.Equal(F16Floats, backF32.As<float32>().AccessMemory<float>().ToArray());

        float[] bf16Src =
        [
            Bf16Floats[0], Bf16Floats[1],
            BitConverter.UInt32BitsToSingle(0x3F808000u),
            BitConverter.UInt32BitsToSingle(0x3F818000u),
        ];
        ushort[] bf16ExpectedBits = [Bf16Bits[0], Bf16Bits[1], 0x3F80, 0x3F82];
        var f32b = TensorData(DType.Float32, [4L], bf16Src[0], bf16Src[1], bf16Src[2], bf16Src[3]);
        var asBf16 = TensorDataConversion.ConvertTensorDataType(f32b, DType.BFloat16);
        Assert.True(asBf16.DType == DType.BFloat16);
        Assert.Equal(bf16ExpectedBits, asBf16.As<bfloat16>().AccessMemory().ToArray().Select(x => x.Bits).ToArray());

        var backF32b = TensorDataConversion.ConvertTensorDataType(asBf16, DType.Float32);
        Assert.Equal(
            bf16ExpectedBits.Select(bits => BitConverter.UInt32BitsToSingle((uint)bits << 16)).ToArray(),
            backF32b.As<float32>().AccessMemory<float>().ToArray());

        var smallF16 = TensorDataConversion.ConvertTensorDataType(
            TensorData(DType.Float32, [2L], 1.5f, -2.25f), DType.Float16);
        var asInt = TensorDataConversion.ConvertTensorDataType(smallF16, DType.Int64);
        Assert.True(asInt.DType == DType.Int64);
        Assert.Equal([2L, -2L], asInt.As<int64>().AccessMemory<long>().ToArray());

        Assert.Equal(16, DType.Float16.EncodingBitCount);
        Assert.Equal(16, DType.BFloat16.EncodingBitCount);

        var f16Enc = Enc(F16Bits.Select(x => new Float16(x)).ToArray());
        Assert.Equal(F16Bits.Length * 2, f16Enc.Length);
        Assert.Equal(F16Bits, Dec<Float16>(f16Enc).Select(x => x.Bits).ToArray());

        var bf16Enc = Enc(Bf16Bits.Select(x => new BFloat16(x)).ToArray());
        Assert.Equal(Bf16Bits.Length * 2, bf16Enc.Length);
        Assert.Equal(Bf16Bits, Dec<BFloat16>(bf16Enc).Select(x => x.Bits).ToArray());

        long[] fourDims = [4L];
        var f16Packed = OnnxModelReader.ConvertInt32PackedData(DType.Float16, F16Bits.Select(x => (int)x).ToArray());
        Assert.Equal(F16Bits.Length * 2, f16Packed.Length);
        var f16FromPacked = TensorData.CreateFromRawBytes(fourDims, DType.Float16, f16Packed);
        Assert.True(f16FromPacked.DType == DType.Float16);
        Assert.Equal(F16Floats, f16FromPacked.As<float16>().AccessMemory().ToArray().Select(x => (float)x).ToArray());

        var bf16Packed = OnnxModelReader.ConvertInt32PackedData(DType.BFloat16, Bf16Bits.Select(x => (int)x).ToArray());
        Assert.Equal(Bf16Bits.Length * 2, bf16Packed.Length);
        var bf16FromPacked = TensorData.CreateFromRawBytes(fourDims, DType.BFloat16, bf16Packed);
        Assert.True(bf16FromPacked.DType == DType.BFloat16);
        Assert.Equal(Bf16Floats, bf16FromPacked.As<bfloat16>().AccessMemory().ToArray().Select(x => (float)x).ToArray());

        byte[] uint8Narrowed = [0xFF, 0x01];
        byte[] int16Narrowed = [0xFE, 0xFF];
        Assert.Equal(uint8Narrowed, OnnxModelReader.ConvertInt32PackedData(DType.UInt8, [255, 1]));
        Assert.Equal(int16Narrowed, OnnxModelReader.ConvertInt32PackedData(DType.Int16, [-2]));
        Assert.Equal(4, OnnxModelReader.ConvertInt32PackedData(DType.Int32, [42]).Length);

        var f16Proto = OnnxIRFactory.CreateTensor([4L], "w_f16", DType.Float16,
            identifierTemplate: null, isTrainable: true, f16FromPacked.AccessRawMemory().ToArray());
        Assert.Equal((int)TensorProto.DataType.Float16, f16Proto.data_type);
        Assert.Equal(f16Packed, f16Proto.RawData);

        var bf16Proto = OnnxIRFactory.CreateTensor([4L], "w_bf16", DType.BFloat16,
            identifierTemplate: null, isTrainable: true, bf16FromPacked.AccessRawMemory().ToArray());
        Assert.Equal((int)TensorProto.DataType.Bfloat16, bf16Proto.data_type);
        Assert.Equal(bf16Packed, bf16Proto.RawData);
    }

    [Fact]
    public void TestF16Bf16CastGraphExecution() =>
        Assert.True(AutoTest.AdvancedTestGraph<DTypeF16CastRoundtripCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [3L], 1.5f, -2.25f, 0.5f)]));
}

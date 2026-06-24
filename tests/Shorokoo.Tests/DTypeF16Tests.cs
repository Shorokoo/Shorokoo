using System.Text;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Core.Inference.Abstractions;
using Shorokoo.Core.Nodes.Processors.Helpers;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage-purpose tests for Float16/BFloat16 dtype completeness (Phase 4 D1):
/// the SafeTensorLoader F16/BF16 payload arms (load + save roundtrip + the
/// supported-formats error), the TensorDataConversion constant
/// conversion/extraction arms for both half formats (incl. BF16
/// round-to-nearest-even), the ONNX TensorProto int32-packed initializer
/// narrowing (OnnxModelReader.ConvertInt32PackedData) and raw_data
/// import/export, the DType.BFloat16.EncodingBitCount regression (was 8, must
/// be 16 — storage bits), and an f32→f16→f32 / f32→bf16→f32 Cast graph
/// executed for real through ComputeContext (ORT) + QEE via
/// <see cref="AutoTest.AdvancedTestGraph{TModule}"/>.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class DTypeF16CoverageTests
{
    // F16 test values: exactly representable, plus max-finite-half and the
    // smallest positive subnormal (2^-24).
    private static readonly ushort[] F16Bits =
    [
        BitConverter.HalfToUInt16Bits((Half)1.5f),    // 0x3E00
        BitConverter.HalfToUInt16Bits((Half)(-2.25f)), // 0xC080
        0x7BFF, // 65504, max finite half
        0x0001, // smallest positive subnormal: 2^-24
    ];
    private static readonly float[] F16Floats =
        F16Bits.Select(b => (float)BitConverter.UInt16BitsToHalf(b)).ToArray();

    // BF16 test values: exactly representable, plus max finite and the
    // smallest positive subnormal (2^-133, still a normal float32 subnormal).
    private static readonly ushort[] Bf16Bits =
    [
        0x3FC0, // 1.5
        0xC010, // -2.25
        0x7F7F, // max finite bfloat16
        0x0001, // smallest positive subnormal: 2^-133
    ];
    private static readonly float[] Bf16Floats =
        Bf16Bits.Select(b => BitConverter.UInt32BitsToSingle((uint)b << 16)).ToArray();

    /// <summary>
    /// Builds a minimal .safetensors byte buffer: 8-byte little-endian header
    /// length, UTF-8 JSON header, then the raw little-endian 2-byte payloads.
    /// </summary>
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
    public void TestSafeTensorF16Bf16Roundtrip()
    {
        var buffer = BuildSafeTensorBuffer(("a", "F16", F16Bits), ("b", "BF16", Bf16Bits));
        var tensors = SafeTensorLoader.ParseSafeTensorBytes(buffer);
        Assert.Equal(2, tensors.Count);

        var a = tensors.Single(t => t.Name == "a");
        Assert.Equal("F16", a.DataType);
        Assert.True(a.Data.DType == DType.Float16);
        Assert.Equal([4L], a.Data.Shape.Dims);
        var aValues = a.Data.As<float16>().AccessMemory().ToArray();
        Assert.Equal(F16Bits, aValues.Select(v => v.Bits).ToArray());
        Assert.Equal(F16Floats, aValues.Select(v => (float)v).ToArray()); // exact, incl. 65504 + denormal

        var b = tensors.Single(t => t.Name == "b");
        Assert.Equal("BF16", b.DataType);
        Assert.True(b.Data.DType == DType.BFloat16);
        Assert.Equal([4L], b.Data.Shape.Dims);
        var bValues = b.Data.As<bfloat16>().AccessMemory().ToArray();
        Assert.Equal(Bf16Bits, bValues.Select(v => v.Bits).ToArray());
        Assert.Equal(Bf16Floats, bValues.Select(v => (float)v).ToArray());

        // DType → SafeTensor dtype string arms for both half formats.
        Assert.Equal("F16", SafeTensorLoader.DTypeToSafeTensorDType(DType.Float16));
        Assert.Equal("BF16", SafeTensorLoader.DTypeToSafeTensorDType(DType.BFloat16));

        // Save → re-parse roundtrip preserves the exact bit patterns.
        using var stream = new MemoryStream();
        SafeTensorLoader.SaveSafeTensorsToStream(stream, tensors);
        var reloaded = SafeTensorLoader.ParseSafeTensorBytes(stream.ToArray());
        Assert.Equal(2, reloaded.Count);
        Assert.Equal(F16Bits, reloaded.Single(t => t.Name == "a")
            .Data.As<float16>().AccessMemory().ToArray().Select(v => v.Bits).ToArray());
        Assert.Equal(Bf16Bits, reloaded.Single(t => t.Name == "b")
            .Data.As<bfloat16>().AccessMemory().ToArray().Select(v => v.Bits).ToArray());

        // The unsupported-format arm names the supported formats (wrapped by the
        // per-tensor parse error handler).
        var unsupported = BuildSafeTensorBuffer(("z", "F8_E4M3", [0x0000]));
        var ex = Assert.Throws<InvalidOperationException>(() => SafeTensorLoader.ParseSafeTensorBytes(unsupported));
        Assert.Contains("F8_E4M3", ex.Message);
        Assert.Contains("Supported formats", ex.Message);
    }

    /// <summary>
    /// A rank-0 scalar (empty shape) round-trips through <c>SafeTensorLoader</c>. SafeTensors
    /// encodes a scalar as <c>"shape": []</c> (product of an empty shape = 1 element), which the
    /// format supports; the saver previously rejected it by conflating an empty shape with a
    /// missing one. Saving a scalar alongside a rank-1 tensor must preserve the empty shape and
    /// the value on reload (this is what lets Adam's scalar timestep checkpoint).
    /// </summary>
    [Fact]
    public void TestSafeTensorScalarRoundtrip()
    {
        var scalar = TensorData(System.Array.Empty<long>(), 7.5f);   // rank-0
        var vector = TensorData([2L], new float[] { 1f, 2f });       // rank-1 alongside

        var tensors = new List<SafeTensor>
        {
            new("s", scalar, "F32", System.Array.Empty<long>()),
            new("v", vector, "F32", new long[] { 2L }),
        };

        using var stream = new MemoryStream();
        SafeTensorLoader.SaveSafeTensorsToStream(stream, tensors);
        var reloaded = SafeTensorLoader.ParseSafeTensorBytes(stream.ToArray());

        var s = reloaded.Single(t => t.Name == "s");
        Assert.Empty(s.Data.Shape.Dims);                              // rank-0 preserved
        Assert.Equal(7.5f, s.Data.As<float32>().AccessMemory()[0]);
        var v = reloaded.Single(t => t.Name == "v");
        Assert.Equal([2L], v.Data.Shape.Dims);
        Assert.Equal(new float[] { 1f, 2f }, v.Data.As<float32>().AccessMemory().ToArray());
    }

    [Fact]
    public void TestTensorDataConversionF16Bf16Roundtrips()
    {
        // f32 → f16: System.Half narrowing; values chosen exactly representable.
        var f32 = TensorData(DType.Float32, [4L],
            F16Floats[0], F16Floats[1], F16Floats[2], F16Floats[3]);
        var asF16 = TensorDataConversion.ConvertTensorDataType(f32, DType.Float16);
        Assert.True(asF16.DType == DType.Float16);
        Assert.Equal(F16Bits, asF16.As<float16>().AccessMemory().ToArray().Select(v => v.Bits).ToArray());

        // f16 → f32 extraction widens exactly.
        var backF32 = TensorDataConversion.ConvertTensorDataType(asF16, DType.Float32);
        Assert.True(backF32.DType == DType.Float32);
        Assert.Equal(F16Floats, backF32.As<float32>().AccessMemory<float>().ToArray());

        // f32 → bf16: exact values plus the two round-to-nearest-even midpoints
        // (lower 16 bits == 0x8000): 0x3F80|0x3F81 → 0x3F80 (down to even),
        // 0x3F81|0x3F82 → 0x3F82 (up to even).
        var bf16Src = new float[]
        {
            Bf16Floats[0], Bf16Floats[1],
            BitConverter.UInt32BitsToSingle(0x3F808000u),
            BitConverter.UInt32BitsToSingle(0x3F818000u),
        };
        var bf16ExpectedBits = new ushort[] { Bf16Bits[0], Bf16Bits[1], 0x3F80, 0x3F82 };
        var f32b = TensorData(DType.Float32, [4L], bf16Src[0], bf16Src[1], bf16Src[2], bf16Src[3]);
        var asBf16 = TensorDataConversion.ConvertTensorDataType(f32b, DType.BFloat16);
        Assert.True(asBf16.DType == DType.BFloat16);
        Assert.Equal(bf16ExpectedBits, asBf16.As<bfloat16>().AccessMemory().ToArray().Select(v => v.Bits).ToArray());

        // bf16 → f32 extraction widens exactly to the rounded values.
        var backF32b = TensorDataConversion.ConvertTensorDataType(asBf16, DType.Float32);
        Assert.Equal(
            bf16ExpectedBits.Select(bits => BitConverter.UInt32BitsToSingle((uint)bits << 16)).ToArray(),
            backF32b.As<float32>().AccessMemory<float>().ToArray());

        // f16 → int64 drives the f16 extraction arm into a non-float target.
        var smallF16 = TensorDataConversion.ConvertTensorDataType(
            TensorData(DType.Float32, [2L], 1.5f, -2.25f), DType.Float16);
        var asInt = TensorDataConversion.ConvertTensorDataType(smallF16, DType.Int64);
        Assert.True(asInt.DType == DType.Int64);
        Assert.Equal([2L, -2L], asInt.As<int64>().AccessMemory<long>().ToArray());
    }

    [Fact]
    public void TestEncodingBitCountAndEncDecRegression()
    {
        // Regression: DType.BFloat16.EncodingBitCount returned 8 — storage is a
        // 16-bit (ushort) pattern, same as Float16. The wrong value half-sized
        // the zero-buffer allocations in TrainingModel/ShapeInferenceInterpreter.
        Assert.Equal(16, DType.Float16.EncodingBitCount);
        Assert.Equal(16, DType.BFloat16.EncodingBitCount);

        // Enc/Dec IR-encoding roundtrips for both half formats (2 bytes/element).
        var f16Arr = F16Bits.Select(b => new Float16(b)).ToArray();
        var f16Enc = Enc(f16Arr);
        Assert.Equal(F16Bits.Length * 2, f16Enc.Length);
        Assert.Equal(F16Bits, Dec<Float16>(f16Enc).Select(v => v.Bits).ToArray());

        var bf16Arr = Bf16Bits.Select(b => new BFloat16(b)).ToArray();
        var bf16Enc = Enc(bf16Arr);
        Assert.Equal(Bf16Bits.Length * 2, bf16Enc.Length);
        Assert.Equal(Bf16Bits, Dec<BFloat16>(bf16Enc).Select(v => v.Bits).ToArray());
    }

    [Fact]
    public void TestOnnxTensorProtoF16Bf16ImportExport()
    {
        // Per the ONNX spec, f16/bf16 TensorProto data may arrive in int32_data
        // (one uint16 bit pattern widened per int32 entry) — the reader must
        // narrow it back to 2 bytes per element.
        var f16Packed = OnnxModelReader.ConvertInt32PackedData(DType.Float16, F16Bits.Select(b => (int)b).ToArray());
        Assert.Equal(F16Bits.Length * 2, f16Packed.Length);
        var f16FromPacked = TensorData.CreateFromRawBytes(new long[] { 4L }, DType.Float16, f16Packed);
        Assert.True(f16FromPacked.DType == DType.Float16);
        Assert.Equal(F16Floats, f16FromPacked.As<float16>().AccessMemory().ToArray().Select(v => (float)v).ToArray());

        var bf16Packed = OnnxModelReader.ConvertInt32PackedData(DType.BFloat16, Bf16Bits.Select(b => (int)b).ToArray());
        Assert.Equal(Bf16Bits.Length * 2, bf16Packed.Length);
        var bf16FromPacked = TensorData.CreateFromRawBytes(new long[] { 4L }, DType.BFloat16, bf16Packed);
        Assert.True(bf16FromPacked.DType == DType.BFloat16);
        Assert.Equal(Bf16Floats, bf16FromPacked.As<bfloat16>().AccessMemory().ToArray().Select(v => (float)v).ToArray());

        // Other int32-packed narrow types narrow too; int32 itself passes through.
        Assert.Equal(new byte[] { 0xFF, 0x01 }, OnnxModelReader.ConvertInt32PackedData(DType.UInt8, [255, 1]));
        Assert.Equal(new byte[] { 0xFE, 0xFF }, OnnxModelReader.ConvertInt32PackedData(DType.Int16, [-2]));
        Assert.Equal(4, OnnxModelReader.ConvertInt32PackedData(DType.Int32, [42]).Length);

        // Export side: initializers are emitted as raw_data with the ONNX proto
        // type nums (Float16 = 10, Bfloat16 = 16) — byte-exact roundtrip.
        var f16Proto = OnnxIRFactory.CreateTensor([4L], "w_f16", DType.Float16,
            identifierTemplate: null, isTrainable: true, f16FromPacked.AccessRawMemory().ToArray());
        Assert.Equal((int)TensorProto.DataType.Float16, f16Proto.data_type);
        Assert.Equal(f16Packed, f16Proto.RawData);

        var bf16Proto = OnnxIRFactory.CreateTensor([4L], "w_bf16", DType.BFloat16,
            identifierTemplate: null, isTrainable: true, bf16FromPacked.AccessRawMemory().ToArray());
        Assert.Equal((int)TensorProto.DataType.Bfloat16, bf16Proto.data_type);
        Assert.Equal(bf16Packed, bf16Proto.RawData);
    }

    /// <summary>
    /// f32→f16→f32 and f32→bf16→f32 Cast chains executed for real through
    /// ComputeContext (ORT CPU EP supports both half-format Cast kernels), plus
    /// the ONNX save/load roundtrip and the QEE pass inside AdvancedTestGraph.
    /// The module self-checks: with exactly-representable inputs the roundtrip
    /// must be lossless.
    /// </summary>
    [Fact]
    public void TestF16Bf16CastGraphExecution() =>
        Assert.True(AutoTest.AdvancedTestGraph<DTypeF16CastRoundtripCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [3L], 1.5f, -2.25f, 0.5f)]));
}

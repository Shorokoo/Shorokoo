using Microsoft.ML.OnnxRuntime;
using Shorokoo.Core.Inference.Abstractions;

namespace Shorokoo.Tests.Benchmarks;

/// <summary>
/// The deterministic filler the profiling harnesses feed a training step with — the batch and
/// every weight tensor alike, since the step's pure graph takes its parameters as inputs.
///
/// <para><paramref name="tensor"/>, the tensor's position in the feed, is hashed along with the
/// element index, and that is the whole point. A filler keyed on length alone hands two weight
/// tensors of the same shape identical content, so <c>MultiHeadAttention</c>'s <c>Wq</c> and
/// <c>Wk</c> become one matrix, <c>Q·Kᵀ</c> becomes a Gram matrix of squared norms, and the
/// attention logits run two orders of magnitude above anything an initialization produces (row
/// spread of 132 nats against 9 under Xavier). The softmax tail then underflows into the
/// denormal range, where MLAS's GEMM runs about eight times slower — so every kernel time
/// recorded through such a filler measures the filler, and so do the constants fitted from it.
/// Values stay uniform over <c>[-0.5, 0.5)</c>, the range the length-keyed ramp covered.</para>
/// </summary>
internal static class SyntheticFeed
{
    public static float[] Floats(long count, int tensor)
    {
        var values = new float[count];
        for (var i = 0; i < values.Length; i++) values[i] = Uniform(tensor, i);
        return values;
    }

    public static OrtValue Tensor(Shape shape, DType dtype, int tensor)
    {
        var n = shape.Count;
        if (dtype == DType.Float32) return OrtValue.CreateTensorValueFromMemory(Floats(n, tensor), shape.Dims);
        if (dtype == DType.Int64) return OrtValue.CreateTensorValueFromMemory(new long[n], shape.Dims);
        if (dtype == DType.Int32) return OrtValue.CreateTensorValueFromMemory(new int[n], shape.Dims);
        if (dtype == DType.Bool) return OrtValue.CreateTensorValueFromMemory(new bool[n], shape.Dims);
        throw new NotSupportedException(dtype.ToString());
    }

    private static float Uniform(int tensor, int index)
    {
        var z = (ulong)(uint)tensor * 0x9E3779B97F4A7C15UL + (ulong)(uint)index + 0x165667B19E3779F9UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        z ^= z >> 31;
        return (z >> 40) / (float)(1 << 24) - 0.5f;
    }
}

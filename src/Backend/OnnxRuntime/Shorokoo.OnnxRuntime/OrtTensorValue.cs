using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Shorokoo.Core.Inference.Abstractions;
using OrtFloat16 = Microsoft.ML.OnnxRuntime.Float16;
using OrtBFloat16 = Microsoft.ML.OnnxRuntime.BFloat16;
using ShoFloat16 = Shorokoo.Core.Inference.Abstractions.Float16;
using ShoBFloat16 = Shorokoo.Core.Inference.Abstractions.BFloat16;

namespace Shorokoo.OnnxRuntime;

internal sealed class OrtTensorValue : IShorokooTensorValue
{
    internal OrtValue Inner { get; }

    public OrtTensorValue(OrtValue inner) { Inner = inner; }

    public ShorokooOnnxValueType ValueType => (ShorokooOnnxValueType)(int)Inner.OnnxType;

    public ShorokooTensorElementType ElementType =>
        (ShorokooTensorElementType)(int)Inner.GetTensorTypeAndShape().ElementDataType;

    public long[] Shape => Inner.GetTensorTypeAndShape().Shape;

    public ReadOnlySpan<T> GetTensorDataAsSpan<T>() where T : unmanaged
    {
        if (typeof(T) == typeof(ShoFloat16))
            return MemoryMarshal.Cast<OrtFloat16, T>(Inner.GetTensorDataAsSpan<OrtFloat16>());
        if (typeof(T) == typeof(ShoBFloat16))
            return MemoryMarshal.Cast<OrtBFloat16, T>(Inner.GetTensorDataAsSpan<OrtBFloat16>());
        return Inner.GetTensorDataAsSpan<T>();
    }

    public Span<T> GetTensorMutableDataAsSpan<T>() where T : unmanaged
    {
        if (typeof(T) == typeof(ShoFloat16))
            return MemoryMarshal.Cast<OrtFloat16, T>(Inner.GetTensorMutableDataAsSpan<OrtFloat16>());
        if (typeof(T) == typeof(ShoBFloat16))
            return MemoryMarshal.Cast<OrtBFloat16, T>(Inner.GetTensorMutableDataAsSpan<OrtBFloat16>());
        return Inner.GetTensorMutableDataAsSpan<T>();
    }

    public IReadOnlyList<string> GetStringTensorData() => Inner.GetStringTensorAsArray();

    public int GetValueCount() => Inner.GetValueCount();

    public IShorokooTensorValue GetValue(int index) =>
        new OrtTensorValue(Inner.GetValue(index, OrtAllocator.DefaultInstance));

    public ShorokooTensorElementType GetSequenceElementType()
    {
        var info = Inner.GetTypeInfo();
        return (ShorokooTensorElementType)(int)info.SequenceTypeInfo.ElementType.TensorTypeAndShapeInfo.ElementDataType;
    }

    public void Dispose() => Inner.Dispose();
}

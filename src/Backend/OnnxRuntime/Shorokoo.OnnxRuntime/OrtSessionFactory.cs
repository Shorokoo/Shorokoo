using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Shorokoo.Core.Inference.Abstractions;
using OrtFloat16 = Microsoft.ML.OnnxRuntime.Float16;
using OrtBFloat16 = Microsoft.ML.OnnxRuntime.BFloat16;
using ShoFloat16 = Shorokoo.Core.Inference.Abstractions.Float16;
using ShoBFloat16 = Shorokoo.Core.Inference.Abstractions.BFloat16;

namespace Shorokoo.OnnxRuntime;

// Concrete IShorokooInferenceSessionFactory backed by ORT. Each platform DLL
// (Shorokoo.WinCPU / Shorokoo.WinGPU / Shorokoo.LinuxCPU / Shorokoo.LinuxGPU)
// subclasses this and supplies its EP configuration via the constructor delegate.
public abstract class OrtSessionFactory : IShorokooInferenceSessionFactory
{
    private readonly Action<SessionOptions> _configureExecutionProvider;

    protected OrtSessionFactory(Action<SessionOptions> configureExecutionProvider)
    {
        _configureExecutionProvider = configureExecutionProvider;
    }

    public IShorokooInferenceSession CreateSession(
        ReadOnlyMemory<byte> modelBytes,
        ShorokooGraphOptimization graphOptimization,
        ShorokooLogSeverity logSeverity)
    {
        var options = new SessionOptions();
        options.LogSeverityLevel = (OrtLoggingLevel)(int)logSeverity;
        options.GraphOptimizationLevel = (GraphOptimizationLevel)(int)graphOptimization;
        _configureExecutionProvider(options);
        var session = new InferenceSession(modelBytes.ToArray(), options);
        return new OrtInferenceSession(session);
    }

    public IShorokooTensorValue CreateTensor<T>(T[] data, long[] shape) where T : unmanaged
    {
        if (typeof(T) == typeof(ShoFloat16))
        {
            var src = MemoryMarshal.Cast<T, OrtFloat16>(data.AsSpan()).ToArray();
            return new OrtTensorValue(OrtValue.CreateTensorValueFromMemory(src, shape));
        }
        if (typeof(T) == typeof(ShoBFloat16))
        {
            var src = MemoryMarshal.Cast<T, OrtBFloat16>(data.AsSpan()).ToArray();
            return new OrtTensorValue(OrtValue.CreateTensorValueFromMemory(src, shape));
        }
        return new OrtTensorValue(OrtValue.CreateTensorValueFromMemory(data, shape));
    }

    public IShorokooTensorValue CreateTensorFromRawBytes(
        ShorokooTensorElementType elementType,
        byte[] data,
        long[] shape)
    {
        return elementType switch
        {
            ShorokooTensorElementType.Float => MakeFromBytes<float>(data, shape),
            ShorokooTensorElementType.UInt8 => MakeFromBytes<byte>(data, shape),
            ShorokooTensorElementType.Int8 => MakeFromBytes<sbyte>(data, shape),
            ShorokooTensorElementType.UInt16 => MakeFromBytes<ushort>(data, shape),
            ShorokooTensorElementType.Int16 => MakeFromBytes<short>(data, shape),
            ShorokooTensorElementType.Int32 => MakeFromBytes<int>(data, shape),
            ShorokooTensorElementType.Int64 => MakeFromBytes<long>(data, shape),
            ShorokooTensorElementType.Bool => MakeFromBytes<bool>(data, shape),
            ShorokooTensorElementType.Float16 => MakeFromBytes<OrtFloat16>(data, shape),
            ShorokooTensorElementType.Double => MakeFromBytes<double>(data, shape),
            ShorokooTensorElementType.UInt32 => MakeFromBytes<uint>(data, shape),
            ShorokooTensorElementType.UInt64 => MakeFromBytes<ulong>(data, shape),
            ShorokooTensorElementType.BFloat16 => MakeFromBytes<OrtBFloat16>(data, shape),
            ShorokooTensorElementType.String => throw new NotSupportedException(
                "String tensors are variable-length and not byte-stride; use CreateStringTensor instead."),
            _ => throw new NotSupportedException(
                $"CreateTensorFromRawBytes does not support element type {elementType}."),
        };
    }

    public IShorokooTensorValue CreateStringTensor(IReadOnlyList<string> data, long[] shape)
    {
        var ortValue = OrtValue.CreateTensorWithEmptyStrings(OrtAllocator.DefaultInstance, shape);
        for (int i = 0; i < data.Count; i++)
            ortValue.StringTensorSetElementAt(data[i].AsSpan(), i);
        return new OrtTensorValue(ortValue);
    }

    public IShorokooTensorValue CreateSequence(IReadOnlyList<IShorokooTensorValue> values)
    {
        var inner = new List<OrtValue>(values.Count);
        foreach (var v in values) inner.Add(((OrtTensorValue)v).Inner);
        return new OrtTensorValue(OrtValue.CreateSequence(inner));
    }

    private static OrtTensorValue MakeFromBytes<T>(byte[] data, long[] shape) where T : unmanaged
    {
        var typed = MemoryMarshal.Cast<byte, T>(data.AsSpan()).ToArray();
        return new OrtTensorValue(OrtValue.CreateTensorValueFromMemory(typed, shape));
    }
}

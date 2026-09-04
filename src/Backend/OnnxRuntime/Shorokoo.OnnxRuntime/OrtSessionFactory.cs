using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Shorokoo.Core.Inference.Abstractions;
using OrtFloat16 = Microsoft.ML.OnnxRuntime.Float16;
using OrtBFloat16 = Microsoft.ML.OnnxRuntime.BFloat16;
using ShoFloat16 = Shorokoo.Core.Inference.Abstractions.Float16;
using ShoBFloat16 = Shorokoo.Core.Inference.Abstractions.BFloat16;

namespace Shorokoo.OnnxRuntime;

/// <summary>
/// The <see cref="IShorokooInferenceSessionFactory"/> implementation backed by ONNX
/// Runtime: it builds ORT sessions and ORT-backed tensor values for Shorokoo's inference
/// pipeline. It is platform-neutral and abstract — each platform package
/// (<c>Shorokoo.WinCPU</c>, <c>Shorokoo.WinGPU</c>, <c>Shorokoo.LinuxCPU</c>,
/// <c>Shorokoo.LinuxGPU</c>) subclasses it and supplies its execution-provider
/// configuration through the constructor delegate.
///
/// <para>You do not normally reference this type, or the <c>Shorokoo.OnnxRuntime</c>
/// package that carries it, directly: reference one platform package instead and let
/// <see cref="Shorokoo.Core.Inference.Abstractions.InferenceBackend"/> find its factory.
/// Subclass this only to drive a different ONNX Runtime execution provider than the four
/// shipped packages offer.</para>
/// </summary>
public abstract class OrtSessionFactory : IShorokooInferenceSessionFactory
{
    private readonly Action<SessionOptions> _configureExecutionProvider;

    /// <param name="configureExecutionProvider">
    /// Applied to the <see cref="SessionOptions"/> of every session this factory creates,
    /// after the log-severity and graph-optimization settings and before the session is
    /// constructed. This is where a subclass appends its execution provider (e.g.
    /// <c>opts =&gt; opts.AppendExecutionProvider_CUDA(0)</c>); a CPU backend leaves ORT on
    /// its default provider and does nothing here.
    /// </param>
    protected OrtSessionFactory(Action<SessionOptions> configureExecutionProvider)
    {
        _configureExecutionProvider = configureExecutionProvider;
    }

    /// <summary>
    /// Creates an ORT inference session over a serialized ONNX model, on this factory's
    /// execution provider.
    /// </summary>
    /// <param name="modelBytes">The serialized ONNX model.</param>
    /// <param name="graphOptimization">The ORT graph-optimization level to apply.</param>
    /// <param name="logSeverity">The minimum severity ORT logs at.</param>
    public IShorokooInferenceSession CreateSession(
        ReadOnlyMemory<byte> modelBytes,
        ShorokooGraphOptimization graphOptimization,
        ShorokooLogSeverity logSeverity)
    {
        // The `using` is load-bearing, not tidiness. SessionOptions is a SafeHandle, so it
        // carries a critical finalizer that calls OrtReleaseSessionOptions, and ORT takes its
        // handle as a bare IntPtr -- the P/Invoke does no SafeHandle ref-counting, and the
        // session does not retain the options the caller passes. Left as a plain local, the
        // options are unreachable the instant that IntPtr is read, so a GC landing inside
        // session creation frees them while ORT is still walking sess_options->provider_factories
        // (core/session/utils.cc, InitializeSession) -- a use-after-free that segfaults the
        // process. Disposing in a finally keeps them rooted across the constructor.
        using var options = new SessionOptions();
        options.LogSeverityLevel = (OrtLoggingLevel)(int)logSeverity;
        options.GraphOptimizationLevel = (GraphOptimizationLevel)(int)graphOptimization;
        _configureExecutionProvider(options);
        var session = new InferenceSession(modelBytes.ToArray(), options);
        return new OrtInferenceSession(session);
    }

    /// <summary>
    /// Wraps a flat managed array as an ORT tensor of the given shape. Shorokoo's
    /// <c>Float16</c>/<c>BFloat16</c> are reinterpreted as ORT's own half types; every
    /// other unmanaged element type is passed through as-is.
    /// </summary>
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

    /// <summary>
    /// Builds an ORT tensor of <paramref name="elementType"/> and
    /// <paramref name="shape"/> by reinterpreting a fixed-stride byte buffer.
    /// </summary>
    /// <exception cref="NotSupportedException">
    /// The element type has no fixed byte stride — <see cref="ShorokooTensorElementType.String"/>
    /// is variable-length, so use <see cref="CreateStringTensor"/> for it.
    /// </exception>
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

    /// <summary>
    /// Builds an ORT string tensor of the given shape, filling it element by element in
    /// row-major order from <paramref name="data"/>.
    /// </summary>
    public IShorokooTensorValue CreateStringTensor(IReadOnlyList<string> data, long[] shape)
    {
        var ortValue = OrtValue.CreateTensorWithEmptyStrings(OrtAllocator.DefaultInstance, shape);
        for (int i = 0; i < data.Count; i++)
            ortValue.StringTensorSetElementAt(data[i].AsSpan(), i);
        return new OrtTensorValue(ortValue);
    }

    /// <summary>
    /// Packs already-created tensor values into an ORT sequence value, in order.
    /// </summary>
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

using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Shorokoo.Core.Inference.Abstractions;
using OrtFloat16 = Microsoft.ML.OnnxRuntime.Float16;
using OrtBFloat16 = Microsoft.ML.OnnxRuntime.BFloat16;
using ShoFloat16 = Shorokoo.Core.Inference.Abstractions.Float16;
using ShoBFloat16 = Shorokoo.Core.Inference.Abstractions.BFloat16;
using TensorElementType = Microsoft.ML.OnnxRuntime.Tensors.TensorElementType;

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
        Configure(options, graphOptimization, logSeverity);
        _configureExecutionProvider(options);
        var session = new InferenceSession(modelBytes.ToArray(), options);
        return new OrtInferenceSession(session);
    }

    /// <summary>
    /// Applies the settings every session this factory creates runs with — the log severity
    /// and the graph-optimization level, plus the session configuration entry that
    /// <see cref="ShorokooGraphOptimization.TrainingStep"/> stands for — to
    /// <paramref name="options"/>. Public so a diagnostic can build an ORT session with exactly
    /// the product's configuration plus its own (profiling, an optimized-model dump).
    ///
    /// <para>For <see cref="ShorokooGraphOptimization.TrainingStep"/>:
    /// <c>optimization.disable_specified_optimizers</c> = CommonSubexpressionElimination;
    /// everything else in ORT_ENABLE_ALL — constant folding, the MatMul/Gelu/LayerNorm fusions,
    /// layout transforms — stays on.</para>
    ///
    /// <para>Deliberately absent: <c>session.set_denormal_as_zero</c>. ORT applies that entry to
    /// the constructing thread once per process (first session wins) by setting FTZ/DAZ in its
    /// MXCSR, which then flushes every later float and double operation on that thread — the
    /// caller's own managed code included, for the life of the thread. It was weighed for a
    /// measured speedup on denormal attention gradients; those gradients turned out to be an
    /// artefact of a profiling harness that fed two weight tensors identical values, so there is
    /// no speedup to set against the leak. Nothing asserts its absence: a guard on it was
    /// deleted deliberately, having needed a <c>dotnet test</c> invocation of its own to observe
    /// a process's first session.</para>
    /// </summary>
    public static void Configure(
        SessionOptions options,
        ShorokooGraphOptimization graphOptimization,
        ShorokooLogSeverity logSeverity)
    {
        options.LogSeverityLevel = (OrtLoggingLevel)(int)logSeverity;
        if (graphOptimization == ShorokooGraphOptimization.TrainingStep)
        {
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            options.AddSessionConfigEntry("optimization.disable_specified_optimizers", "CommonSubexpressionElimination");
        }
        else
            options.GraphOptimizationLevel = (GraphOptimizationLevel)(int)graphOptimization;
    }

    /// <summary>
    /// Copies a flat managed array into an ORT tensor of the given shape. Shorokoo's
    /// <c>Float16</c>/<c>BFloat16</c> are reinterpreted as ORT's own half types; every
    /// other unmanaged element type is copied through as-is.
    /// </summary>
    public IShorokooTensorValue CreateTensor<T>(T[] data, long[] shape) where T : unmanaged
    {
        if (typeof(T) == typeof(ShoFloat16))
            return Allocate(TensorElementType.Float16, MemoryMarshal.AsBytes(data.AsSpan()), shape);
        if (typeof(T) == typeof(ShoBFloat16))
            return Allocate(TensorElementType.BFloat16, MemoryMarshal.AsBytes(data.AsSpan()), shape);
        return Allocate(ElementTypeOf<T>(), MemoryMarshal.AsBytes(data.AsSpan()), shape);
    }

    /// <summary>
    /// Builds an ORT tensor of <paramref name="elementType"/> and
    /// <paramref name="shape"/> by reinterpreting a fixed-stride byte buffer.
    /// </summary>
    /// <exception cref="NotSupportedException">
    /// The element type has no fixed byte stride — <see cref="ShorokooTensorElementType.String"/>
    /// is variable-length, so use <see cref="CreateStringTensor"/> for it — or is not one this
    /// method handles at all.
    /// </exception>
    public IShorokooTensorValue CreateTensorFromRawBytes(
        ShorokooTensorElementType elementType,
        byte[] data,
        long[] shape)
    {
        return elementType switch
        {
            ShorokooTensorElementType.Float or ShorokooTensorElementType.UInt8
                or ShorokooTensorElementType.Int8 or ShorokooTensorElementType.UInt16
                or ShorokooTensorElementType.Int16 or ShorokooTensorElementType.Int32
                or ShorokooTensorElementType.Int64 or ShorokooTensorElementType.Bool
                or ShorokooTensorElementType.Float16 or ShorokooTensorElementType.Double
                or ShorokooTensorElementType.UInt32 or ShorokooTensorElementType.UInt64
                or ShorokooTensorElementType.BFloat16
                => Allocate((TensorElementType)(int)elementType, data.AsSpan(), shape),
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
    ///
    /// <para>This <b>takes ownership</b> of <paramref name="values"/>: ORT moves them into the
    /// sequence's own member list and the sequence frees them when it is disposed, so a caller
    /// that still needs them must pass copies. Shorokoo's <c>TensorDataSequence.Create</c> does
    /// exactly that.</para>
    /// </summary>
    public IShorokooTensorValue CreateSequence(IReadOnlyList<IShorokooTensorValue> values)
    {
        var inner = new List<OrtValue>(values.Count);
        foreach (var v in values) inner.Add(((OrtTensorValue)v).Inner);
        return new OrtTensorValue(OrtValue.CreateSequence(inner));
    }

    /// <summary>
    /// Builds an ORT tensor on a buffer ORT itself allocates and copies <paramref name="bytes"/>
    /// into it.
    ///
    /// <para>The obvious alternative — <c>OrtValue.CreateTensorValueFromMemory</c> over a managed
    /// array — is why this is a copy. That API pins the array for the value's lifetime and releases
    /// the pin only from <c>Dispose</c>: the release sits behind the <c>disposing</c> guard, so an
    /// <c>OrtValue</c> reclaimed by its finalizer never runs it. Nothing in Shorokoo disposes a
    /// tensor value, so every tensor built that way pinned its bytes for the life of the process —
    /// a training loop that fed a fresh batch each step leaked one batch per step, permanently, and
    /// no collection could ever get it back. An ORT-allocated buffer is released by the value's
    /// finalizer along with the value, so it behaves like every other tensor the runtime hands
    /// back.</para>
    /// </summary>
    private static OrtTensorValue Allocate(TensorElementType elementType, ReadOnlySpan<byte> bytes, long[] shape)
    {
        var wrapped = new OrtTensorValue(
            OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, elementType, shape));
        try
        {
            var destination = wrapped.Inner.GetTensorMutableRawData();
            if (bytes.Length < destination.Length)
                throw new ArgumentException(
                    $"Supplied data of {bytes.Length} bytes is less than shape size {destination.Length} bytes.",
                    nameof(bytes));
            // A caller may hand over a buffer longer than the shape covers — the node-definition
            // tables do — in which case the surplus was never part of the tensor and the
            // wrapped-memory path this replaced never read it either.
            bytes.Slice(0, destination.Length).CopyTo(destination);
            // The destination span is a bare pointer into the value's buffer: reading the value
            // for it is the value's last read, after which the JIT may retire the local and a
            // collection on any thread can run the value's finalizer and free the buffer out from
            // under the copy. Scope does not root anything (Shorokoo/Shorokoo#178); this does.
            GC.KeepAlive(wrapped);
        }
        catch
        {
            wrapped.Dispose();
            throw;
        }
        return wrapped;
    }

    /// <summary>The ORT element type of the storage primitive <typeparamref name="T"/>.</summary>
    private static TensorElementType ElementTypeOf<T>() where T : unmanaged
    {
        if (typeof(T) == typeof(float)) return TensorElementType.Float;
        if (typeof(T) == typeof(double)) return TensorElementType.Double;
        if (typeof(T) == typeof(bool)) return TensorElementType.Bool;
        if (typeof(T) == typeof(sbyte)) return TensorElementType.Int8;
        if (typeof(T) == typeof(byte)) return TensorElementType.UInt8;
        if (typeof(T) == typeof(short)) return TensorElementType.Int16;
        if (typeof(T) == typeof(ushort)) return TensorElementType.UInt16;
        if (typeof(T) == typeof(int)) return TensorElementType.Int32;
        if (typeof(T) == typeof(uint)) return TensorElementType.UInt32;
        if (typeof(T) == typeof(long)) return TensorElementType.Int64;
        if (typeof(T) == typeof(ulong)) return TensorElementType.UInt64;
        if (typeof(T) == typeof(OrtFloat16)) return TensorElementType.Float16;
        if (typeof(T) == typeof(OrtBFloat16)) return TensorElementType.BFloat16;
        throw new NotSupportedException($"CreateTensor does not support element type {typeof(T).Name}.");
    }
}

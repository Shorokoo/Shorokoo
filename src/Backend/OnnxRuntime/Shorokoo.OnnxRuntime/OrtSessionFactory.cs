using System.Globalization;
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
    private readonly Action<SessionOptions, DeviceMemorySettings> _configureExecutionProvider;
    private readonly int? _cudaDeviceId;

    /// <param name="configureExecutionProvider">
    /// Applied to the <see cref="SessionOptions"/> of every session this factory creates,
    /// after the log-severity and graph-optimization settings and before the session is
    /// constructed. This is where a subclass appends its execution provider; a CPU backend
    /// leaves ORT on its default provider and does nothing here. It is handed the
    /// <see cref="DeviceMemorySettings"/> of the session being built — the arena settings belong
    /// to that session, so they arrive with it rather than being read from anywhere else.
    /// </param>
    /// <param name="device">
    /// The kind of device those sessions run on. A subclass driving a provider that is neither
    /// the CPU nor CUDA — DirectML, ROCm, CoreML — passes <see cref="ComputeDevice.Other"/>.
    /// It is a parameter rather than something inferred from <paramref name="cudaDeviceId"/>
    /// precisely because such a subclass names no CUDA device: inferring would report it as the
    /// CPU, and <see cref="InferenceBackend.RequireDevice"/> would then wave work onto a card
    /// its author meant to stay off.
    /// </param>
    /// <param name="cudaDeviceId">
    /// The CUDA device the provider appended above allocates on, or <c>null</c> when it is
    /// not a CUDA provider. It names the arena that
    /// <see cref="RunSettings.ShrinkArenaAfterRun"/> shrinks, so a backend that does not
    /// allocate on a card passes <c>null</c> and its sessions ignore the setting.
    /// </param>
    /// <exception cref="ArgumentException"><paramref name="cudaDeviceId"/> disagrees with
    /// <paramref name="device"/>, or is negative.</exception>
    protected OrtSessionFactory(
        Action<SessionOptions, DeviceMemorySettings> configureExecutionProvider,
        ComputeDevice device,
        int? cudaDeviceId)
    {
        // Built here rather than on each read of Description, so a factory that could only
        // describe itself incoherently cannot be constructed at all.
        Description = new BackendDescription(GetType().Assembly.GetName().Name ?? GetType().Name, device, cudaDeviceId);
        _configureExecutionProvider = configureExecutionProvider;
        _cudaDeviceId = cudaDeviceId;
    }

    /// <summary>
    /// The CUDA-backend constructor: every session gets the CUDA execution provider on
    /// <paramref name="cudaDeviceId"/>, configured from the <see cref="DeviceMemorySettings"/>
    /// that session is built with, and honours <see cref="RunSettings.ShrinkArenaAfterRun"/> for
    /// that device's arena on each run.
    /// </summary>
    protected OrtSessionFactory(int cudaDeviceId)
        : this((opts, mem) => AppendCuda(opts, cudaDeviceId, mem), ComputeDevice.Cuda, cudaDeviceId) { }

    /// <summary>
    /// This backend: the assembly the concrete factory lives in, and the device the constructor
    /// named. Fixed at construction, so every read agrees and none can contradict the provider
    /// the subclass actually appended.
    /// </summary>
    public BackendDescription Description { get; }

    /// <summary>
    /// Creates an ORT inference session over a serialized ONNX model, on this factory's
    /// execution provider.
    /// </summary>
    /// <param name="modelBytes">The serialized ONNX model.</param>
    /// <param name="graphOptimization">The ORT graph-optimization level to apply.</param>
    /// <param name="logSeverity">The minimum severity ORT logs at.</param>
    /// <param name="deviceMemory">The arena settings this session is built with. ORT reads them
    /// during construction and the session keeps them for life, so they are settled here and
    /// nowhere else.</param>
    public IShorokooInferenceSession CreateSession(
        ReadOnlyMemory<byte> modelBytes,
        ShorokooGraphOptimization graphOptimization,
        ShorokooLogSeverity logSeverity,
        DeviceMemorySettings deviceMemory)
    {
        ArgumentNullException.ThrowIfNull(deviceMemory);
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
        _configureExecutionProvider(options, deviceMemory);
        var session = new InferenceSession(modelBytes.ToArray(), options);
        return new OrtInferenceSession(session, _cudaDeviceId);
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
    /// Appends the CUDA execution provider on <paramref name="deviceId"/>, configured with
    /// <paramref name="deviceMemory"/>. This is what the GPU backends pass as their
    /// execution-provider step, and the point at which
    /// <see cref="DeviceMemorySettings.LimitBytes"/> and
    /// <see cref="DeviceMemorySettings.ArenaExtend"/> reach ORT: the session being built keeps
    /// them for its life, and no other session is touched.
    /// </summary>
    public static void AppendCuda(SessionOptions options, int deviceId, DeviceMemorySettings deviceMemory)
    {
        ArgumentNullException.ThrowIfNull(deviceMemory);
        // OrtCUDAProviderOptions is a SafeHandle that ORT takes as a bare IntPtr, exactly like
        // the SessionOptions above, so it needs the same `using`: the options are read during
        // AppendExecutionProvider_CUDA, well after the JIT has retired the local at its .Handle
        // read, and a GC there would run the critical finalizer under the native call.
        using var cuda = new OrtCUDAProviderOptions();
        cuda.UpdateOptions(CudaProviderOptions(
            deviceId, deviceMemory.LimitBytes, deviceMemory.ArenaExtend));
        options.AppendExecutionProvider_CUDA(cuda);
    }

    /// <summary>
    /// The CUDA execution-provider options for a device and a device-memory configuration, in
    /// ORT's own <c>provider_options</c> spelling. Pure, and public alongside
    /// <see cref="Configure"/> so the mapping can be read and asserted without a CUDA machine to
    /// build a session on. An absent <paramref name="limitBytes"/> omits <c>gpu_mem_limit</c>
    /// altogether, which leaves ORT at its default of the whole card.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="arenaExtend"/> is not one of
    /// the two strategies ORT accepts — <see cref="ArenaExtendStrategy.Auto"/> is Shorokoo's choice
    /// between them and must be resolved first — or <paramref name="limitBytes"/> is not
    /// positive.</exception>
    public static Dictionary<string, string> CudaProviderOptions(
        int deviceId,
        long? limitBytes,
        ArenaExtendStrategy arenaExtend)
    {
        var options = new Dictionary<string, string>
        {
            ["device_id"] = deviceId.ToString(CultureInfo.InvariantCulture),
            ["arena_extend_strategy"] = arenaExtend switch
            {
                ArenaExtendStrategy.NextPowerOfTwo => "kNextPowerOfTwo",
                ArenaExtendStrategy.SameAsRequested => "kSameAsRequested",
                // Auto lands here too, and should: it is Shorokoo's choice between the two and
                // DeviceMemorySettings.Resolve settles it before a session is built, so one
                // reaching ORT means that step was skipped rather than that ORT gained a value.
                _ => throw new ArgumentOutOfRangeException(
                    nameof(arenaExtend), arenaExtend,
                    "Not an ONNX Runtime arena-extend strategy; resolve DeviceMemorySettings first."),
            },
        };
        if (limitBytes is { } limit)
        {
            // ORT parses this into a size_t, where a negative reads back as SIZE_MAX -- an
            // uncapped arena from a caller who asked for the opposite. Refuse it here, as
            // DeviceMemorySettings.LimitBytes refuses it at the assignment.
            if (limit <= 0)
                throw new ArgumentOutOfRangeException(
                    nameof(limitBytes), limit, "The device-memory limit must be positive.");
            options["gpu_mem_limit"] = limit.ToString(CultureInfo.InvariantCulture);
        }
        return options;
    }

    /// <summary>
    /// The arena ORT should shrink after a run — the value of its
    /// <c>memory.enable_memory_arena_shrinkage</c> run option — or <c>null</c> to leave the run
    /// option off. Only a GPU backend names one: the entry says <i>which</i> arena to shrink, and
    /// a CPU backend's device memory is not what <see cref="DeviceMemorySettings"/> is about.
    /// </summary>
    public static string? ArenaShrinkageRunConfig(int? cudaDeviceId, bool shrinkArenaAfterRun)
        => cudaDeviceId is { } device && shrinkArenaAfterRun
            ? $"gpu:{device.ToString(CultureInfo.InvariantCulture)}"
            : null;

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
        try
        {
            return new OrtTensorValue(OrtValue.CreateSequence(inner));
        }
        catch
        {
            // ORT hands the values back on failure — it empties the list only on success — so
            // without this they would sit undisposed until their finalizers ran.
            foreach (var v in inner) v.Dispose();
            throw;
        }
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

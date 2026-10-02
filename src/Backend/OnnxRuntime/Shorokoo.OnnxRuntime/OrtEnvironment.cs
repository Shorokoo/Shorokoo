using System.Reflection;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;

namespace Shorokoo.OnnxRuntime;

/// <summary>
/// The parts of ONNX Runtime's surface <see cref="RuntimeAllocator"/> needs that the managed package
/// does not expose: registering an allocator of the caller's own with the environment
/// (<c>RegisterAllocator</c> in the C API, which the managed <c>OrtEnv</c> has no call for), the
/// native handles of the environment and of a memory info, and a managed <see cref="OrtAllocator"/>
/// over a native one, to make tensors from.
///
/// <para>Bound through the same internal <c>OrtApi</c> instance <see cref="OrtArenaStats"/> reads,
/// and by the non-public names below. A surface that moved fails the first session built rather than
/// building one that allocates in an arena: an output in an arena keeps it alive, which is what the
/// allocator is there to prevent. <c>CoreUtilsCoverageTests.TestTheOrtEnvironmentBindingStillResolves</c>
/// asserts every name, so an ONNX Runtime upgrade that moves one fails there first.</para>
/// </summary>
internal static class OrtEnvironment
{
    // OrtStatus* RegisterAllocator(OrtEnv*, OrtAllocator*)
    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate IntPtr DRegisterAllocator(IntPtr env, IntPtr allocator);

    // const char* GetErrorMessage(const OrtStatus*)
    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate IntPtr DGetErrorMessage(IntPtr status);

    // void ReleaseStatus(OrtStatus*)
    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate void DReleaseStatus(IntPtr status);

    /// <summary>The <c>OrtApi</c> fields this reads.</summary>
    internal static IReadOnlyList<string> ApiEntryPointNames { get; } = ["RegisterAllocator", "GetErrorMessage", "ReleaseStatus"];

    /// <summary>The non-public property of <see cref="OrtEnv"/> holding its native handle, and of
    /// <see cref="OrtMemoryInfo"/> holding its own.</summary>
    internal const string EnvHandleProperty = "Handle";
    internal const string MemoryInfoPointerProperty = "Pointer";

    private sealed record Binding(
        DRegisterAllocator RegisterAllocator,
        DGetErrorMessage GetErrorMessage,
        DReleaseStatus ReleaseStatus,
        Func<OrtEnv, IntPtr> EnvHandle,
        Func<OrtMemoryInfo, IntPtr> MemoryInfoPointer,
        ConstructorInfo AllocatorOverNative);

    private static readonly Lazy<Binding?> _binding = new(Bind, LazyThreadSafetyMode.ExecutionAndPublication);

    /// <summary>Whether every entry point, handle and constructor resolved.</summary>
    internal static bool IsBound => _binding.Value is not null;

    private static Binding? Bind()
    {
        try
        {
            var holder = typeof(InferenceSession).Assembly.GetType(OrtArenaStats.ApiHolderTypeName);
            var field = holder?.GetField(
                OrtArenaStats.ApiFieldName, BindingFlags.Static | BindingFlags.NonPublic | BindingFlags.Public);
            if (field?.GetValue(null) is not { } api) return null;

            var pointers = new IntPtr[ApiEntryPointNames.Count];
            for (int i = 0; i < pointers.Length; i++)
            {
                var entry = api.GetType().GetField(ApiEntryPointNames[i]);
                if (entry?.FieldType != typeof(IntPtr)) return null;
                if (entry.GetValue(api) is not IntPtr pointer || pointer == IntPtr.Zero) return null;
                pointers[i] = pointer;
            }

            const BindingFlags Instance = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public;
            var envHandle = typeof(OrtEnv).GetProperty(EnvHandleProperty, Instance);
            var infoPointer = typeof(OrtMemoryInfo).GetProperty(MemoryInfoPointerProperty, Instance);
            var allocator = typeof(OrtAllocator).GetConstructor(Instance, [typeof(IntPtr), typeof(bool)]);
            if (envHandle?.PropertyType != typeof(IntPtr) || envHandle.GetMethod is not { } envGetter
                || infoPointer?.PropertyType != typeof(IntPtr) || infoPointer.GetMethod is not { } infoGetter
                || allocator is null)
                return null;

            return new Binding(
                Marshal.GetDelegateForFunctionPointer<DRegisterAllocator>(pointers[0]),
                Marshal.GetDelegateForFunctionPointer<DGetErrorMessage>(pointers[1]),
                Marshal.GetDelegateForFunctionPointer<DReleaseStatus>(pointers[2]),
                envGetter.CreateDelegate<Func<OrtEnv, IntPtr>>(),
                infoGetter.CreateDelegate<Func<OrtMemoryInfo, IntPtr>>(),
                allocator);
        }
        catch (Exception) { return null; }
    }

    private static Binding Required => _binding.Value ?? throw new InvalidOperationException(
        "This ONNX Runtime does not expose what Shorokoo registers its allocator through "
        + "(RegisterAllocator, and the native handles of its environment and memory infos), so no "
        + "session can be built to allocate through it.");

    /// <summary>
    /// Registers the native allocator <paramref name="allocator"/> with the process's ONNX Runtime
    /// environment, for every session built to use the environment's allocators
    /// (<c>session.use_env_allocators</c>) to allocate through on its device.
    /// </summary>
    /// <exception cref="InvalidOperationException">The runtime refused it.</exception>
    internal static void Register(IntPtr allocator)
    {
        var api = Required;
        var env = OrtEnv.Instance();
        var status = api.RegisterAllocator(api.EnvHandle(env), allocator);
        GC.KeepAlive(env);
        if (status == IntPtr.Zero) return;
        var message = Marshal.PtrToStringUTF8(api.GetErrorMessage(status));
        api.ReleaseStatus(status);
        throw new InvalidOperationException($"ONNX Runtime refused Shorokoo's allocator: {message}");
    }

    /// <summary>The native handle of the process's ONNX Runtime environment: one per native runtime,
    /// however many copies of the managed wrapper bind it.</summary>
    internal static IntPtr EnvironmentHandle()
    {
        var env = OrtEnv.Instance();
        var handle = Required.EnvHandle(env);
        GC.KeepAlive(env);
        return handle;
    }

    /// <summary>The native handle of <paramref name="info"/>, which the caller keeps alive for as
    /// long as the handle is used.</summary>
    internal static IntPtr PointerOf(OrtMemoryInfo info) => Required.MemoryInfoPointer(info);

    /// <summary>A managed <see cref="OrtAllocator"/> over the native <paramref name="allocator"/>,
    /// owning nothing.</summary>
    internal static OrtAllocator Wrap(IntPtr allocator)
        => (OrtAllocator)Required.AllocatorOverNative.Invoke([allocator, false]);
}

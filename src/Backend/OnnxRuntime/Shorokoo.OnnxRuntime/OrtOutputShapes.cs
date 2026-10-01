using System.Reflection;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;

namespace Shorokoo.OnnxRuntime;

/// <summary>
/// The shape ONNX Runtime gave each output of a session when it built it, read from its C API.
///
/// <para>The managed surface reports an output's shape as a list of dimensions, <c>-1</c> for one
/// it leaves open, and reports an output it has no shape for at all with no dimensions — which is
/// what a scalar has too. Measured: a session that took the empty list for a scalar and bound a
/// scalar to such an output failed the run as soon as the output came out <c>[1, 2]</c>, so the two
/// have to be told apart, and only the C API's <c>TensorTypeAndShape_HasShape</c> (ONNX Runtime
/// 1.24) does. The binding goes through the same internal <c>OrtApi</c> instance
/// <see cref="OrtArenaStats"/> reads, and to the session's native handle.</para>
///
/// <para>As there, a surface that moved answers <c>null</c> rather than a wrong pointer — a session
/// then settles no output, and copies every output out of its arena — and
/// <c>CoreUtilsCoverageTests.TestTheOrtOutputShapesBindingStillResolvesAndTellsAScalarFromAnUnknownShape</c>
/// asserts it by name so an ORT upgrade that moves it fails loudly.</para>
/// </summary>
internal static class OrtOutputShapes
{
    // OrtStatus* SessionGetOutputTypeInfo(const OrtSession*, size_t index, OrtTypeInfo** type_info)
    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate IntPtr DSessionGetOutputTypeInfo(IntPtr session, UIntPtr index, out IntPtr typeInfo);

    // OrtStatus* CastTypeInfoToTensorInfo(const OrtTypeInfo*, const OrtTensorTypeAndShapeInfo** out)
    // The info it answers with is the type info's own, released with it; null for a type that is not
    // a tensor.
    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate IntPtr DCastTypeInfoToTensorInfo(IntPtr typeInfo, out IntPtr tensorInfo);

    // bool TensorTypeAndShape_HasShape(const OrtTensorTypeAndShapeInfo*) -- a C++ bool, one byte.
    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate byte DHasShape(IntPtr tensorInfo);

    // OrtStatus* GetDimensionsCount(const OrtTensorTypeAndShapeInfo*, size_t* out)
    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate IntPtr DGetDimensionsCount(IntPtr tensorInfo, out UIntPtr count);

    // OrtStatus* GetDimensions(const OrtTensorTypeAndShapeInfo*, int64_t* dim_values, size_t length)
    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate IntPtr DGetDimensions(IntPtr tensorInfo, [Out] long[] dims, UIntPtr length);

    // void ReleaseTypeInfo(OrtTypeInfo*)
    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate void DReleaseTypeInfo(IntPtr typeInfo);

    // void ReleaseStatus(OrtStatus*)
    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate void DReleaseStatus(IntPtr status);

    private sealed record Binding(
        DSessionGetOutputTypeInfo SessionGetOutputTypeInfo,
        DCastTypeInfoToTensorInfo CastTypeInfoToTensorInfo,
        DHasShape HasShape,
        DGetDimensionsCount GetDimensionsCount,
        DGetDimensions GetDimensions,
        DReleaseTypeInfo ReleaseTypeInfo,
        DReleaseStatus ReleaseStatus,
        Func<InferenceSession, IntPtr> Handle);

    /// <summary>The <c>OrtApi</c> fields this reads, in the order the delegates above declare
    /// them.</summary>
    internal static IReadOnlyList<string> ApiEntryPointNames { get; } =
    [
        "SessionGetOutputTypeInfo", "CastTypeInfoToTensorInfo", "TensorTypeAndShape_HasShape",
        "GetDimensionsCount", "GetDimensions", "ReleaseTypeInfo", "ReleaseStatus",
    ];

    /// <summary>The non-public property of <see cref="InferenceSession"/> holding its native
    /// handle.</summary>
    internal const string HandlePropertyName = "Handle";

    private static readonly Lazy<Binding?> _binding = new(Bind, LazyThreadSafetyMode.ExecutionAndPublication);

    /// <summary>Whether the native entry points and the handle resolved.</summary>
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

            var handle = typeof(InferenceSession).GetProperty(
                HandlePropertyName, BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
            if (handle?.PropertyType != typeof(IntPtr) || handle.GetMethod is not { } getter) return null;

            return new Binding(
                Marshal.GetDelegateForFunctionPointer<DSessionGetOutputTypeInfo>(pointers[0]),
                Marshal.GetDelegateForFunctionPointer<DCastTypeInfoToTensorInfo>(pointers[1]),
                Marshal.GetDelegateForFunctionPointer<DHasShape>(pointers[2]),
                Marshal.GetDelegateForFunctionPointer<DGetDimensionsCount>(pointers[3]),
                Marshal.GetDelegateForFunctionPointer<DGetDimensions>(pointers[4]),
                Marshal.GetDelegateForFunctionPointer<DReleaseTypeInfo>(pointers[5]),
                Marshal.GetDelegateForFunctionPointer<DReleaseStatus>(pointers[6]),
                getter.CreateDelegate<Func<InferenceSession, IntPtr>>());
        }
        catch (Exception) { return null; }
    }

    /// <summary>
    /// The shape ONNX Runtime gives each of <paramref name="session"/>'s outputs, by position in its
    /// output names: its dimensions — none, for a scalar — each a known count, or <c>-1</c> for one
    /// the runtime leaves open, which its symbolic name may still tie to an input's; or null for an
    /// output that is no tensor, or that it has no shape for at all. Null as a whole when the binding
    /// did not resolve or a call failed.
    /// </summary>
    internal static long[]?[]? Read(InferenceSession session)
    {
        ArgumentNullException.ThrowIfNull(session);
        if (_binding.Value is not { } api) return null;

        var shapes = new long[]?[session.OutputNames.Count];
        var handle = api.Handle(session);
        try
        {
            for (int i = 0; i < shapes.Length; i++)
            {
                var status = api.SessionGetOutputTypeInfo(handle, (UIntPtr)i, out var typeInfo);
                if (status != IntPtr.Zero)
                {
                    api.ReleaseStatus(status);
                    return null;
                }
                try
                {
                    if (!TryShape(api, typeInfo, out shapes[i])) return null;
                }
                finally
                {
                    api.ReleaseTypeInfo(typeInfo);
                }
            }
        }
        finally
        {
            // The handle went over as a bare pointer, so nothing but this roots the session across
            // the calls; the JIT retires the argument at the handle read otherwise.
            GC.KeepAlive(session);
        }
        return shapes;
    }

    /// <summary>The shape <paramref name="typeInfo"/> gives, into <paramref name="shape"/>: null where
    /// it gives none. False where a call failed.</summary>
    private static bool TryShape(Binding api, IntPtr typeInfo, out long[]? shape)
    {
        shape = null;
        var status = api.CastTypeInfoToTensorInfo(typeInfo, out var tensorInfo);
        if (status != IntPtr.Zero)
        {
            api.ReleaseStatus(status);
            return false;
        }
        if (tensorInfo == IntPtr.Zero || api.HasShape(tensorInfo) == 0) return true;

        status = api.GetDimensionsCount(tensorInfo, out var count);
        if (status != IntPtr.Zero)
        {
            api.ReleaseStatus(status);
            return false;
        }
        var dims = new long[(int)count];
        status = api.GetDimensions(tensorInfo, dims, count);
        if (status != IntPtr.Zero)
        {
            api.ReleaseStatus(status);
            return false;
        }
        shape = dims;
        return true;
    }
}

using System.Reflection;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;

namespace Shorokoo.OnnxRuntime;

/// <summary>
/// The address of a tensor's buffer, read from ONNX Runtime's C API for a tensor whose byte length
/// no <c>int</c> holds.
///
/// <para>Every managed accessor ORT offers for a tensor's data — <c>GetTensorMutableRawData</c>,
/// <c>GetTensorMutableDataAsSpan&lt;T&gt;</c>, <c>GetTensorSpanMutableRawData&lt;T&gt;</c> — builds
/// its result from one <c>Span&lt;byte&gt;</c> over the whole buffer, whose length it casts to
/// <c>int</c> unchecked. Above 2 GiB that cast turns negative and the span constructor throws, so
/// none of them can answer with the address, although <c>GetTensorMutableData</c> itself returns
/// nothing else. The binding goes to that entry point, through the same internal <c>OrtApi</c>
/// instance <see cref="OrtArenaStats"/> reads, and to the value's native handle.</para>
///
/// <para>As there, a surface that moved answers <c>null</c> rather than a wrong pointer, and
/// <c>CoreUtilsCoverageTests.TestTheOrtTensorAddressBindingStillResolvesAndAgreesWithTheSpan</c>
/// asserts it by name so an ORT upgrade that moves it fails loudly.</para>
/// </summary>
internal static class OrtTensorAddress
{
    // OrtStatus* GetTensorMutableData(OrtValue* value, void** out)
    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate IntPtr DGetTensorMutableData(IntPtr value, out IntPtr data);

    // void ReleaseStatus(OrtStatus*)
    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate void DReleaseStatus(IntPtr status);

    private sealed record Binding(
        DGetTensorMutableData GetTensorMutableData,
        DReleaseStatus ReleaseStatus,
        Func<OrtValue, IntPtr> Handle);

    /// <summary>The <c>OrtApi</c> fields this reads, in the order the delegates above declare
    /// them.</summary>
    internal static IReadOnlyList<string> ApiEntryPointNames { get; } = ["GetTensorMutableData", "ReleaseStatus"];

    /// <summary>The non-public property of <see cref="OrtValue"/> holding its native handle.</summary>
    internal const string HandlePropertyName = "Handle";

    private static readonly Lazy<Binding?> _binding = new(Bind, LazyThreadSafetyMode.ExecutionAndPublication);

    /// <summary>Whether the native entry point and the handle resolved.</summary>
    internal static bool IsBound => _binding.Value is not null;

    private static Binding? Bind()
    {
        try
        {
            var holder = typeof(OrtValue).Assembly.GetType(OrtArenaStats.ApiHolderTypeName);
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

            var handle = typeof(OrtValue).GetProperty(
                HandlePropertyName, BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
            if (handle?.PropertyType != typeof(IntPtr) || handle.GetMethod is not { } getter) return null;

            return new Binding(
                Marshal.GetDelegateForFunctionPointer<DGetTensorMutableData>(pointers[0]),
                Marshal.GetDelegateForFunctionPointer<DReleaseStatus>(pointers[1]),
                getter.CreateDelegate<Func<OrtValue, IntPtr>>());
        }
        catch (Exception) { return null; }
    }

    /// <summary>
    /// The address of <paramref name="value"/>'s buffer, or <c>null</c> when the binding did not
    /// resolve or the call failed. The caller keeps <paramref name="value"/> alive for as long as
    /// it uses the address.
    /// </summary>
    internal static IntPtr? Read(OrtValue value)
    {
        ArgumentNullException.ThrowIfNull(value);
        if (_binding.Value is not { } api) return null;

        var status = api.GetTensorMutableData(api.Handle(value), out var data);
        // The handle went over as a bare pointer, so nothing but this local roots the value across
        // the call and the JIT retires it at the handle read.
        GC.KeepAlive(value);
        if (status != IntPtr.Zero)
        {
            api.ReleaseStatus(status);
            return null;
        }
        return data;
    }
}

using System.Globalization;
using System.Reflection;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Shorokoo.Core.Inference.Abstractions;

namespace Shorokoo.OnnxRuntime;

/// <summary>
/// Reads a session allocator's arena figures out of ONNX Runtime.
///
/// <para><b>This reaches the native entry points by reflection, and that is deliberate but not
/// comfortable.</b> ORT's C API has had <c>AllocatorGetStats</c> since 1.23 and the managed
/// package binds the function pointer — <c>OrtApi.AllocatorGetStats</c> is a public field of a
/// public struct — but the one instance of that struct lives on an <c>internal</c> type, and
/// <c>OrtAllocator</c> exposes no <c>GetStats</c> of its own. So the pointer is public and
/// unreachable at once, and the only route from here is the field on the internal type. A managed
/// <c>OrtAllocator.GetStats()</c> upstream would remove every line of this.
/// <see href="https://github.com/Shorokoo/Shorokoo/issues/375">Shorokoo/Shorokoo#375</see> tracks
/// a human review of the reflection below.</para>
///
/// <para>The hazard reflection brings here is not a missing feature but a <i>wrong</i> one: a
/// field that moved would hand back some other pointer, which this would then call as a function.
/// So every step is checked before it is used, anything unexpected answers <c>null</c>, and
/// <c>CoreUtilsCoverageTests.TestTheOrtArenaStatisticsBindingStillResolvesAndAnswers</c> asserts
/// the whole surface by name so an ORT upgrade that moves it fails loudly instead.</para>
/// </summary>
internal static class OrtArenaStats
{
    // OrtStatus* AllocatorGetStats(const OrtAllocator*, OrtKeyValuePairs** out)
    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate IntPtr DAllocatorGetStats(IntPtr allocator, out IntPtr keyValuePairs);

    // void GetKeyValuePairs(const OrtKeyValuePairs*, const char* const** keys,
    //                       const char* const** values, size_t* count)
    //
    // void, not a status: it returns nothing, so a managed delegate declared to return a status
    // would read whatever the call left in the return register and reject every successful read.
    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate void DGetKeyValuePairs(
        IntPtr keyValuePairs, out IntPtr keys, out IntPtr values, out UIntPtr count);

    // void ReleaseKeyValuePairs(OrtKeyValuePairs*)
    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate void DReleaseKeyValuePairs(IntPtr keyValuePairs);

    // void ReleaseStatus(OrtStatus*)
    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate void DReleaseStatus(IntPtr status);

    private sealed record Binding(
        DAllocatorGetStats AllocatorGetStats,
        DGetKeyValuePairs GetKeyValuePairs,
        DReleaseKeyValuePairs ReleaseKeyValuePairs,
        DReleaseStatus ReleaseStatus);

    /// <summary>The name of the internal type holding the one <c>OrtApi</c> instance, and of the
    /// field on it. Named here so the guard test can assert the same two strings the product
    /// depends on rather than a copy that could drift from them.</summary>
    internal const string ApiHolderTypeName = "Microsoft.ML.OnnxRuntime.NativeMethods";

    /// <inheritdoc cref="ApiHolderTypeName"/>
    internal const string ApiFieldName = "api_";

    private static readonly string[] _apiEntryPointNames =
        ["AllocatorGetStats", "GetKeyValuePairs", "ReleaseKeyValuePairs", "ReleaseStatus"];

    /// <summary>The <c>OrtApi</c> fields this reads, in the order the delegates above declare
    /// them.</summary>
    internal static IReadOnlyList<string> ApiEntryPointNames => _apiEntryPointNames;

    private static readonly string[] _statisticNames =
    [
        "InUse", "Limit", "MaxAllocSize", "MaxInUse", "NumAllocs",
        "NumArenaExtensions", "NumArenaShrinkages", "NumReserves", "TotalAllocated",
    ];

    /// <summary>The nine figures a session allocator answers with, which
    /// <see cref="Read"/> maps onto <see cref="ArenaStatistics"/>.</summary>
    internal static IReadOnlyList<string> StatisticNames => _statisticNames;

    // Bound once. The lazy is the whole of the thread safety here: the delegates are immutable and
    // the native pointers do not move for the life of the process.
    private static readonly Lazy<Binding?> _binding = new(Bind, LazyThreadSafetyMode.ExecutionAndPublication);

    /// <summary>Whether the native entry points resolved, so a read can be attempted at all.</summary>
    internal static bool IsBound => _binding.Value is not null;

    private static Binding? Bind()
    {
        try
        {
            var holder = typeof(OrtAllocator).Assembly.GetType(ApiHolderTypeName);
            var field = holder?.GetField(
                ApiFieldName, BindingFlags.Static | BindingFlags.NonPublic | BindingFlags.Public);
            // Boxed, which is what makes the struct's fields readable from here at all.
            if (field?.GetValue(null) is not { } api) return null;

            var apiType = api.GetType();
            var pointers = new IntPtr[ApiEntryPointNames.Count];
            for (int i = 0; i < pointers.Length; i++)
            {
                // Typed as well as named: a field that kept its name and changed its type is
                // exactly the silent wrong answer this is guarding against.
                var entry = apiType.GetField(ApiEntryPointNames[i]);
                if (entry?.FieldType != typeof(IntPtr)) return null;
                if (entry.GetValue(api) is not IntPtr pointer || pointer == IntPtr.Zero) return null;
                pointers[i] = pointer;
            }

            return new Binding(
                Marshal.GetDelegateForFunctionPointer<DAllocatorGetStats>(pointers[0]),
                Marshal.GetDelegateForFunctionPointer<DGetKeyValuePairs>(pointers[1]),
                Marshal.GetDelegateForFunctionPointer<DReleaseKeyValuePairs>(pointers[2]),
                Marshal.GetDelegateForFunctionPointer<DReleaseStatus>(pointers[3]));
        }
        // Broadly, and for the reason the memory-info probe next door catches broadly: a binding
        // that cannot be made costs the figures and nothing else, and a Lazy rethrows a cached
        // exception on every later access -- which would turn a missing diagnostic into a failure
        // of every run that asked for one.
        catch (Exception) { return null; }
    }

    /// <summary>
    /// Every name-and-value pair <paramref name="allocator"/> answers with, exactly as the runtime
    /// spells them, or <c>null</c> when the binding did not resolve or the call failed. A plain
    /// non-arena allocator answers with none, which is why this is asked of a session's allocator
    /// and not of <c>OrtAllocator.DefaultInstance</c>.
    /// </summary>
    internal static IReadOnlyDictionary<string, string>? ReadRaw(OrtAllocator allocator)
    {
        ArgumentNullException.ThrowIfNull(allocator);
        if (_binding.Value is not { } api) return null;

        var status = api.AllocatorGetStats(allocator.DangerousGetHandle(), out var keyValuePairs);
        // The handle went over as a bare pointer, so nothing but this local roots the allocator
        // across the call and the JIT retires it at the DangerousGetHandle read.
        GC.KeepAlive(allocator);
        if (status != IntPtr.Zero)
        {
            api.ReleaseStatus(status);
            return null;
        }
        if (keyValuePairs == IntPtr.Zero) return null;

        try
        {
            api.GetKeyValuePairs(keyValuePairs, out var keys, out var values, out var count);
            var pairs = new Dictionary<string, string>(StringComparer.Ordinal);
            // An allocator that implements no statistics answers with nothing at all, and nothing
            // is null arrays rather than empty ones -- which is an answer, not a failure, and is
            // what ORT's plain default allocator gives.
            if (count == UIntPtr.Zero || keys == IntPtr.Zero || values == IntPtr.Zero) return pairs;

            for (ulong i = 0; i < (ulong)count; i++)
            {
                var offset = checked((int)(i * (ulong)IntPtr.Size));
                var key = Marshal.PtrToStringUTF8(Marshal.ReadIntPtr(keys, offset));
                var value = Marshal.PtrToStringUTF8(Marshal.ReadIntPtr(values, offset));
                if (key is not null && value is not null) pairs[key] = value;
            }
            return pairs;
        }
        finally
        {
            api.ReleaseKeyValuePairs(keyValuePairs);
        }
    }

    /// <summary>
    /// <paramref name="allocator"/>'s arena figures, or <c>null</c> when
    /// <see cref="ReadRaw"/> answers with nothing or without one of the nine
    /// <see cref="ArenaStatistics"/> is. A name the runtime grows later is ignored rather than
    /// fatal; that one appeared is what the guard test says.
    /// </summary>
    internal static ArenaStatistics? Read(OrtAllocator allocator)
    {
        if (ReadRaw(allocator) is not { } pairs) return null;

        var figures = new Dictionary<string, long>(StringComparer.Ordinal);
        foreach (var name in StatisticNames)
        {
            if (!pairs.TryGetValue(name, out var value)
                || !long.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var figure))
                return null;
            figures[name] = figure;
        }

        return new ArenaStatistics(
            InUseBytes: figures["InUse"],
            LimitBytes: figures["Limit"],
            MaxAllocSizeBytes: figures["MaxAllocSize"],
            MaxInUseBytes: figures["MaxInUse"],
            AllocationCount: figures["NumAllocs"],
            ArenaExtensionCount: figures["NumArenaExtensions"],
            ArenaShrinkageCount: figures["NumArenaShrinkages"],
            ReserveCount: figures["NumReserves"],
            TotalAllocatedBytes: figures["TotalAllocated"]);
    }
}

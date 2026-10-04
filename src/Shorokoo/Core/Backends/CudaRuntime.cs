using System.Runtime.InteropServices;

namespace Shorokoo.Core.Backends;

/// <summary>
/// The thin bit of the CUDA runtime library Shorokoo calls itself, outside ONNX Runtime:
/// <c>cudaMemGetInfo</c>, which <see cref="DeviceMemory"/> reports, the two questions that
/// name the device it read — <c>cudaGetDevice</c> and <c>cudaDeviceGetPCIBusId</c> — which
/// <see cref="ProcessDeviceMemory"/> needs to find the same card elsewhere, and the card's own
/// allocation and release, which <see cref="CachingAllocator"/> takes its blocks from. The library
/// is resolved by name and its exports bound lazily, so nothing here requires a CUDA machine to
/// load — every entry point simply reports failure when the runtime is absent.
/// </summary>
internal static class CudaRuntime
{
    /// <summary>CUDA Toolkit 13.x ships its runtime under an OS-specific name.</summary>
    private static string LibraryName => RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
        ? "cudart64_13.dll"
        : "libcudart.so.13";

    // Cdecl on x64, where it is the only convention, which is the only architecture Shorokoo
    // builds for; cudart declares its entry points CUDARTAPI, i.e. __stdcall on 32-bit Windows.
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int MemGetInfo(out nuint free, out nuint total);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int GetDevice(out int device);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int GetPciBusId(byte[] pciBusId, int length, int device);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int Malloc(out IntPtr pointer, nuint count);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int Free(IntPtr pointer);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int SetDevice(int device);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int GetLastError();

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int DeviceSynchronize();

    /// <summary>The runtime's allocation entry points, bound together or not at all.</summary>
    private sealed record Allocation(Malloc Malloc, Free Free, GetDevice GetDevice, SetDevice SetDevice, GetLastError Clear, DeviceSynchronize Synchronize);

    /// <summary>Every entry point bound: <c>cudaMemGetInfo</c>, and those of the rest the library
    /// exports.</summary>
    private sealed record Entries(MemGetInfo MemGetInfo, GetDevice? GetDevice, GetPciBusId? GetPciBusId, Allocation? Allocation);

    private static readonly Binding<Entries> _binding = new(BindLibrary);

    /// <summary>
    /// A native library's entry points, bound on first use by <c>bind</c>, which answers them — null
    /// where they cannot be had — and whether that answer is final.
    /// </summary>
    internal sealed class Binding<T>(Func<(bool Final, T? Bound)> bind) where T : class
    {
        private readonly object _gate = new();
        private bool _final;
        private T? _bound;

        /// <summary>The entry points, or null.</summary>
        internal T? Value
        {
            get
            {
                lock (_gate)
                {
                    if (_final) return _bound;
                    // Binding is best effort -- a reading is worth nothing next to failing a run, and
                    // TryMemGetInfo promises to report rather than throw.
                    try
                    {
                        (_, _bound) = bind();
                    }
                    catch (Exception)
                    {
                        _bound = null;
                    }
                    _final = true;
                    return _bound;
                }
            }
        }
    }

    /// <summary>
    /// Reads the device-wide free and total memory of the current CUDA device. False when
    /// the CUDA runtime is not installed, exports no <c>cudaMemGetInfo</c>, or reports an
    /// error (no device, no driver) — in which case the outputs are zero.
    ///
    /// <para>Note that this initializes the calling process's CUDA context on the device if
    /// it has none yet, which itself costs device memory; the first reading of a run is
    /// therefore best taken after the backend is up rather than before.</para>
    /// </summary>
    internal static bool TryMemGetInfo(out long free, out long total)
    {
        free = total = 0;
        var memGetInfo = Bind();
        if (memGetInfo is null) return false;
        if (memGetInfo(out var freeBytes, out var totalBytes) != 0) return false;
        free = (long)freeBytes;
        total = (long)totalBytes;
        return true;
    }

    /// <summary>
    /// The CUDA device current for the calling thread — the one <see cref="TryMemGetInfo"/>
    /// reads. False, with <paramref name="device"/> zero, where the runtime cannot say.
    /// </summary>
    internal static bool TryGetDevice(out int device)
    {
        device = 0;
        if (_binding.Value?.GetDevice is not { } getDevice) return false;
        return getDevice(out device) == 0;
    }

    /// <summary>
    /// The PCI bus id of CUDA device <paramref name="device"/>, written
    /// <c>domain:bus:device.function</c> — the one name for a card that the CUDA runtime and
    /// NVML agree on whatever order each enumerates the cards in. Null where the runtime cannot
    /// say.
    /// </summary>
    internal static string? TryGetPciBusId(int device)
    {
        if (_binding.Value?.GetPciBusId is not { } getPciBusId) return null;
        // The form is 12 characters, domain:bus:device.function, and a terminator; the rest is spare.
        var buffer = new byte[64];
        if (getPciBusId(buffer, buffer.Length, device) != 0) return null;
        int end = Array.IndexOf(buffer, (byte)0);
        return System.Text.Encoding.ASCII.GetString(buffer, 0, end < 0 ? buffer.Length : end);
    }

    /// <summary>
    /// <paramref name="count"/> bytes of CUDA device <paramref name="deviceId"/>'s memory, from the
    /// CUDA runtime itself, or <see cref="IntPtr.Zero"/> where the card has not that much free or
    /// there is no runtime to ask. The thread's current device is left as it was found.
    /// </summary>
    internal static IntPtr Allocate(int deviceId, long count)
    {
        if (BindAllocation() is not { } cuda) return IntPtr.Zero;
        return OnDevice(cuda, deviceId, () => cuda.Malloc(out var pointer, (nuint)count) == 0 ? pointer : IntPtr.Zero);
    }

    /// <summary>
    /// Hands <paramref name="pointer"/>, which <see cref="Allocate"/> answered for CUDA device
    /// <paramref name="deviceId"/>, back to the card. The runtime waits for the work the card has in
    /// hand before it does, so nothing still reading the block is cut short.
    /// </summary>
    internal static void Release(int deviceId, IntPtr pointer)
    {
        if (BindAllocation() is not { } cuda) return;
        OnDevice(cuda, deviceId, () => cuda.Free(pointer) == 0 ? pointer : IntPtr.Zero);
    }

    /// <summary>
    /// Waits for the work CUDA device <paramref name="deviceId"/> has in hand, as handing back memory
    /// some of that work may still read must. False where there is no runtime to ask.
    /// </summary>
    internal static bool Synchronize(int deviceId)
    {
        if (BindAllocation() is not { } cuda) return false;
        return OnDevice(cuda, deviceId, () => cuda.Synchronize() == 0 ? (IntPtr)1 : IntPtr.Zero) != IntPtr.Zero;
    }

    /// <summary>
    /// <paramref name="call"/> with CUDA device <paramref name="deviceId"/> current on this thread —
    /// its primary context, which the runtime makes on first use — and the device that was current
    /// before it current again after: how the driver's own entry points are called for the device.
    /// <paramref name="call"/> runs regardless where there is no runtime to make it current.
    /// </summary>
    internal static T OnDevice<T>(int deviceId, Func<T> call)
    {
        if (BindAllocation() is not { } cuda) return call();
        var switched = cuda.GetDevice(out var current) == 0 && current != deviceId && cuda.SetDevice(deviceId) == 0;
        try
        {
            return call();
        }
        finally
        {
            if (switched) cuda.SetDevice(current);
        }
    }

    /// <summary>
    /// <paramref name="call"/> with CUDA device <paramref name="deviceId"/> current on this thread,
    /// and the device that was current before it current again after. A failure is cleared once
    /// answered: left as the thread's last CUDA error, it would be read by the execution provider
    /// after its own next launch on this thread, and reported as that run's failure.
    /// </summary>
    private static IntPtr OnDevice(Allocation cuda, int deviceId, Func<IntPtr> call)
    {
        var switched = cuda.GetDevice(out var current) == 0 && current != deviceId && cuda.SetDevice(deviceId) == 0;
        try
        {
            var answer = call();
            if (answer == IntPtr.Zero) cuda.Clear();
            return answer;
        }
        finally
        {
            if (switched) cuda.SetDevice(current);
        }
    }

    private static Allocation? BindAllocation() => _binding.Value?.Allocation;

    private static MemGetInfo? Bind() => _binding.Value?.MemGetInfo;

    /// <summary>The runtime library's entry points: none, and not final, where the library cannot be
    /// loaded; none, and final, where it lacks <c>cudaMemGetInfo</c>.</summary>
    private static (bool Final, Entries? Bound) BindLibrary()
    {
        if (!NativeLibrary.TryLoad(LibraryName, out var library)) return (false, null);
        if (!NativeLibrary.TryGetExport(library, "cudaMemGetInfo", out var export))
        {
            NativeLibrary.Free(library);
            return (true, null);
        }
        // The library stays loaded on purpose: the delegates point into it.
        var memGetInfo = Marshal.GetDelegateForFunctionPointer<MemGetInfo>(export);
        GetDevice? current = NativeLibrary.TryGetExport(library, "cudaGetDevice", out var getDevice)
            ? Marshal.GetDelegateForFunctionPointer<GetDevice>(getDevice)
            : null;
        GetPciBusId? busId = NativeLibrary.TryGetExport(library, "cudaDeviceGetPCIBusId", out var getPciBusId)
            ? Marshal.GetDelegateForFunctionPointer<GetPciBusId>(getPciBusId)
            : null;
        Allocation? allocation = null;
        if (current is not null
            && NativeLibrary.TryGetExport(library, "cudaMalloc", out var malloc)
            && NativeLibrary.TryGetExport(library, "cudaFree", out var free)
            && NativeLibrary.TryGetExport(library, "cudaSetDevice", out var setDevice)
            && NativeLibrary.TryGetExport(library, "cudaGetLastError", out var lastError)
            && NativeLibrary.TryGetExport(library, "cudaDeviceSynchronize", out var synchronize))
            allocation = new Allocation(
                Marshal.GetDelegateForFunctionPointer<Malloc>(malloc),
                Marshal.GetDelegateForFunctionPointer<Free>(free),
                current,
                Marshal.GetDelegateForFunctionPointer<SetDevice>(setDevice),
                Marshal.GetDelegateForFunctionPointer<GetLastError>(lastError),
                Marshal.GetDelegateForFunctionPointer<DeviceSynchronize>(synchronize));
        return (true, new Entries(memGetInfo, current, busId, allocation));
    }
}

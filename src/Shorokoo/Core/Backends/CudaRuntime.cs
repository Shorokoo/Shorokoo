using System.Runtime.InteropServices;

namespace Shorokoo.Core.Backends;

/// <summary>
/// The thin bit of the CUDA runtime library Shorokoo calls itself, outside ONNX Runtime:
/// <c>cudaMemGetInfo</c>, which <see cref="DeviceMemory"/> reports, and the two questions that
/// name the device it read — <c>cudaGetDevice</c> and <c>cudaDeviceGetPCIBusId</c> — which
/// <see cref="ProcessDeviceMemory"/> needs to find the same card elsewhere. The library is resolved
/// by name and its exports bound lazily, so nothing here requires a CUDA machine to load — every entry point
/// simply reports failure when the runtime is absent.
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

    private static readonly object _gate = new();
    private static MemGetInfo? _memGetInfo;
    private static GetDevice? _getDevice;
    private static GetPciBusId? _getPciBusId;
    private static bool _bound;

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
        if (Bind() is null || _getDevice is not { } getDevice) return false;
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
        if (Bind() is null || _getPciBusId is not { } getPciBusId) return null;
        // The form is 12 characters, domain:bus:device.function, and a terminator; the rest is spare.
        var buffer = new byte[64];
        if (getPciBusId(buffer, buffer.Length, device) != 0) return null;
        int end = Array.IndexOf(buffer, (byte)0);
        return System.Text.Encoding.ASCII.GetString(buffer, 0, end < 0 ? buffer.Length : end);
    }

    private static MemGetInfo? Bind()
    {
        lock (_gate)
        {
            if (_bound) return _memGetInfo;
            try
            {
                // By name, so the copy the process's CUDA backends share where they share one.
                CudaLibraries.Prepare();
                if (!NativeLibrary.TryLoad(LibraryName, out var library)) return null;
                if (NativeLibrary.TryGetExport(library, "cudaMemGetInfo", out var export))
                    // The library stays loaded on purpose: the delegate points into it.
                {
                    _memGetInfo = Marshal.GetDelegateForFunctionPointer<MemGetInfo>(export);
                    if (NativeLibrary.TryGetExport(library, "cudaGetDevice", out var getDevice))
                        _getDevice = Marshal.GetDelegateForFunctionPointer<GetDevice>(getDevice);
                    if (NativeLibrary.TryGetExport(library, "cudaDeviceGetPCIBusId", out var getPciBusId))
                        _getPciBusId = Marshal.GetDelegateForFunctionPointer<GetPciBusId>(getPciBusId);
                }
                else
                    NativeLibrary.Free(library);
            }
            // Binding is best effort -- a reading is worth nothing next to failing a run, and
            // TryMemGetInfo promises to report rather than throw.
            catch (Exception) { _memGetInfo = null; _getDevice = null; _getPciBusId = null; }
            finally { _bound = true; }
            return _memGetInfo;
        }
    }
}

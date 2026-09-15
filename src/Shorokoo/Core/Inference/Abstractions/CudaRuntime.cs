using System.Runtime.InteropServices;

namespace Shorokoo.Core.Inference.Abstractions;

/// <summary>
/// The thin bit of the CUDA runtime library Shorokoo calls itself, outside ONNX Runtime:
/// its presence (which decides the GPU/CPU backend choice in
/// <see cref="InferenceBackend"/>) and <c>cudaMemGetInfo</c>, which
/// <see cref="DeviceMemory"/> reports. The library is resolved by name and its exports
/// bound lazily, so nothing here requires a CUDA machine to load — every entry point
/// simply reports failure when the runtime is absent.
/// </summary>
internal static class CudaRuntime
{
    /// <summary>CUDA Toolkit 12.x ships its runtime under an OS-specific name.</summary>
    internal static string LibraryName => RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
        ? "cudart64_12.dll"
        : "libcudart.so.12";

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int MemGetInfo(out nuint free, out nuint total);

    private static readonly object _gate = new();
    private static MemGetInfo? _memGetInfo;
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

    private static MemGetInfo? Bind()
    {
        lock (_gate)
        {
            if (_bound) return _memGetInfo;
            _bound = true;
            if (NativeLibrary.TryLoad(LibraryName, out var library)
                && NativeLibrary.TryGetExport(library, "cudaMemGetInfo", out var export))
                _memGetInfo = Marshal.GetDelegateForFunctionPointer<MemGetInfo>(export);
            return _memGetInfo;
        }
    }
}

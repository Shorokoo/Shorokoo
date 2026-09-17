using System.Runtime.InteropServices;

namespace Shorokoo.OnnxRuntime;

/// <summary>
/// The one CUDA entry point this backend calls itself: a device-to-host copy, for reading back a
/// tensor an execution provider left on the card. ONNX Runtime's managed surface has no such call,
/// and a value it kept hands out a pointer with no way to read it.
///
/// <para>Bound lazily and by name, so nothing here requires a CUDA machine to load — the copy
/// simply reports failure when the runtime is absent, which is the right answer on a host-only
/// build where no value is device-resident in the first place.</para>
/// </summary>
internal static class CudaInterop
{
    private static string LibraryName => RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
        ? "cudart64_12.dll"
        : "libcudart.so.12";

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int Memcpy(IntPtr destination, IntPtr source, nuint count, int kind);

    private const int DeviceToHost = 2;

    private static readonly object _gate = new();
    private static Memcpy? _memcpy;
    private static bool _bound;

    public static bool CopyDeviceToHost(IntPtr source, byte[] destination)
    {
        var memcpy = Bind();
        if (memcpy is null) return false;

        var pinned = GCHandle.Alloc(destination, GCHandleType.Pinned);
        try
        {
            return memcpy(pinned.AddrOfPinnedObject(), source, (nuint)destination.Length, DeviceToHost) == 0;
        }
        finally
        {
            pinned.Free();
        }
    }

    private static Memcpy? Bind()
    {
        lock (_gate)
        {
            if (_bound) return _memcpy;
            _bound = true;
            try
            {
                if (NativeLibrary.TryLoad(LibraryName, out var handle)
                    && NativeLibrary.TryGetExport(handle, "cudaMemcpy", out var entry))
                    _memcpy = Marshal.GetDelegateForFunctionPointer<Memcpy>(entry);
            }
            catch (DllNotFoundException) { /* no CUDA here; the caller reports it */ }
            catch (BadImageFormatException) { /* likewise */ }
            return _memcpy;
        }
    }
}

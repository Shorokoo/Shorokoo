using System.Runtime.InteropServices;

namespace Shorokoo.OnnxRuntime;

/// <summary>
/// The one CUDA entry point this backend calls itself: a copy between host memory and the card,
/// in either direction. ONNX Runtime's managed surface has no such call, and a value living on
/// the card hands out a pointer with no way to read or fill it.
///
/// <para>Both directions serve one promise apiece. Device-to-host reads back a tensor an
/// execution provider left on the card; host-to-device is what puts a tensor into a CUDA
/// context's memory in the first place, rather than leaving host bytes for the provider to copy
/// over on every run.</para>
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

    private const int HostToDevice = 1;
    private const int DeviceToHost = 2;

    private static readonly object _gate = new();
    private static Memcpy? _memcpy;
    private static bool _bound;

    public static bool CopyDeviceToHost(IntPtr source, byte[] destination)
        => Copy(destination, (memcpy, pinned)
            => memcpy(pinned, source, (nuint)destination.Length, DeviceToHost));

    /// <summary>
    /// Fills <paramref name="count"/> bytes of the device allocation at
    /// <paramref name="destination"/> from the front of <paramref name="source"/>. The caller
    /// sizes the copy by the destination, because a buffer longer than the tensor's shape covers
    /// is something callers do hand over and the surplus was never part of the tensor.
    /// </summary>
    public static bool CopyHostToDevice(byte[] source, IntPtr destination, int count)
        => Copy(source, (memcpy, pinned) => memcpy(destination, pinned, (nuint)count, HostToDevice));

    /// <summary>
    /// Runs <paramref name="copy"/> over <paramref name="hostBuffer"/> pinned, once the runtime is
    /// bound. The pin is what makes the managed array's address meaningful to the CUDA runtime,
    /// which knows nothing of the GC and would otherwise be handed an address a collection may
    /// move out from under it mid-copy.
    /// </summary>
    private static bool Copy(byte[] hostBuffer, Func<Memcpy, IntPtr, int> copy)
    {
        var memcpy = Bind();
        if (memcpy is null) return false;

        var pinned = GCHandle.Alloc(hostBuffer, GCHandleType.Pinned);
        try
        {
            return copy(memcpy, pinned.AddrOfPinnedObject()) == 0;
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

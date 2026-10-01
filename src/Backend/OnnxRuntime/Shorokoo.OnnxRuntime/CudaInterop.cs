using System.Runtime.InteropServices;

namespace Shorokoo.OnnxRuntime;

/// <summary>
/// The one CUDA entry point this backend calls itself: a copy to, from or within the card. ONNX
/// Runtime's managed surface has no such call, and a value living on the card hands out a pointer
/// with no way to read or fill it.
///
/// <para>Each direction serves one promise. Device-to-host reads back a tensor on the card — an
/// output a run left there, or a tensor put there; host-to-device is what puts a tensor into the
/// card's memory in the first place, which is where a CUDA session reads every tensor it is fed;
/// and device-to-device takes a run's output out of the arena it was made in, into memory of its
/// own.</para>
///
/// <para>Bound lazily and by name, so nothing here requires a CUDA machine to load — the copy
/// simply reports failure when the runtime is absent, which is the right answer on a host-only
/// build where no value is device-resident in the first place.</para>
/// </summary>
internal static class CudaInterop
{
    private static string LibraryName => RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
        ? "cudart64_13.dll"
        : "libcudart.so.13";

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int Memcpy(IntPtr destination, IntPtr source, nuint count, int kind);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int GetLastError();

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int StreamSynchronize(IntPtr stream);

    private const int HostToDevice = 1;
    private const int DeviceToHost = 2;
    private const int DeviceToDevice = 3;

    private static readonly object _gate = new();
    private static Memcpy? _memcpy;
    private static Func<int>? _synchronize;
    private static bool _bound;

    /// <summary>
    /// Copies <paramref name="count"/> bytes from the device allocation at <paramref name="source"/>
    /// to the one at <paramref name="destination"/>, on the card and on the CUDA runtime's own stream
    /// (the legacy default one), without waiting for it: <see cref="Synchronize"/> is what does, and
    /// the source is not to be let go of before it has. Answers as
    /// <see cref="CopyDeviceToHost(IntPtr, Span{byte})"/> does.
    /// </summary>
    public static int? CopyDeviceToDevice(IntPtr destination, IntPtr source, long count)
    {
        var memcpy = Bind();
        if (memcpy is null) return null;
        if (count == 0) return 0;
        return memcpy(destination, source, (nuint)count, DeviceToDevice);
    }

    /// <summary>
    /// Waits for everything made on the CUDA runtime's own stream so far — the copies
    /// <see cref="CopyDeviceToDevice"/> made — to be done. Answers as
    /// <see cref="CopyDeviceToHost(IntPtr, Span{byte})"/> does.
    /// </summary>
    public static int? Synchronize()
    {
        Bind();
        return _synchronize?.Invoke();
    }

    public static bool CopyDeviceToHost(IntPtr source, byte[] destination)
        => CopyDeviceToHost(source, destination.AsSpan()) == 0;

    /// <summary>
    /// Fills <paramref name="destination"/> from the device allocation at
    /// <paramref name="source"/> — any range of one, since the caller offsets the address. What
    /// streams a tensor off the card through one reused buffer, a piece at a time. Answers the CUDA
    /// runtime's error code, <c>0</c> on success, or <c>null</c> where there is no runtime to ask.
    /// </summary>
    public static unsafe int? CopyDeviceToHost(IntPtr source, Span<byte> destination)
    {
        var memcpy = Bind();
        if (memcpy is null) return null;
        if (destination.IsEmpty) return 0;
        // Pinned for the length of the call, as Copy pins an array: the CUDA runtime knows nothing
        // of the GC, and a span over managed memory is a moveable address until fixed.
        fixed (byte* pinned = destination)
            return memcpy((IntPtr)pinned, source, (nuint)destination.Length, DeviceToHost);
    }

    /// <summary>
    /// Fills <paramref name="count"/> bytes of the device allocation at
    /// <paramref name="destination"/> from the front of <paramref name="source"/>. The caller
    /// sizes the copy by the destination, because a buffer longer than the tensor's shape covers
    /// is something callers do hand over and the surplus was never part of the tensor.
    /// </summary>
    public static bool CopyHostToDevice(byte[] source, IntPtr destination, int count)
        => Copy(source, (memcpy, pinned) => memcpy(destination, pinned, (nuint)count, HostToDevice));

    /// <summary>
    /// Fills the device allocation at <paramref name="destination"/> — any range of one, since the
    /// caller offsets the address — from <paramref name="source"/>. What streams a tensor onto the
    /// card through one reused buffer, a piece at a time. Answers as
    /// <see cref="CopyDeviceToHost(IntPtr, Span{byte})"/> does.
    /// </summary>
    public static unsafe int? CopyHostToDevice(ReadOnlySpan<byte> source, IntPtr destination)
    {
        var memcpy = Bind();
        if (memcpy is null) return null;
        if (source.IsEmpty) return 0;
        fixed (byte* pinned = source)
            return memcpy(destination, (IntPtr)pinned, (nuint)source.Length, HostToDevice);
    }

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
            // Try rather than Load: both report a missing or unloadable CUDA runtime by returning
            // false, so there is nothing here to catch -- the caller reports the absence.
            if (NativeLibrary.TryLoad(LibraryName, out var handle)
                && NativeLibrary.TryGetExport(handle, "cudaMemcpy", out var entry)
                && NativeLibrary.TryGetExport(handle, "cudaGetLastError", out var lastError)
                && NativeLibrary.TryGetExport(handle, "cudaStreamSynchronize", out var streamSync))
            {
                var memcpy = Marshal.GetDelegateForFunctionPointer<Memcpy>(entry);
                var clear = Marshal.GetDelegateForFunctionPointer<GetLastError>(lastError);
                var synchronize = Marshal.GetDelegateForFunctionPointer<StreamSynchronize>(streamSync);
                // A failed copy leaves its error as the thread's last CUDA error, which the execution
                // provider reads after its own next launch on this thread and reports as that run's
                // failure. The copy's error is answered here, so it is cleared here.
                _memcpy = (destination, source, count, kind) =>
                {
                    var status = memcpy(destination, source, count, kind);
                    // From pageable host memory, cudaMemcpy may return once the bytes are staged,
                    // before the DMA onto the card has landed, and the execution provider's
                    // kernels run on streams of their own that nothing orders after that DMA. So a
                    // copy onto the card waits on the stream it was made on (the legacy default
                    // one) before it counts as done, and one that does not complete is a failed
                    // copy.
                    if (status == 0 && kind == HostToDevice) status = synchronize(IntPtr.Zero);
                    if (status != 0) clear();
                    return status;
                };
                _synchronize = () =>
                {
                    var status = synchronize(IntPtr.Zero);
                    if (status != 0) clear();
                    return status;
                };
            }
            return _memcpy;
        }
    }
}

using System.Runtime.InteropServices;

namespace Shorokoo.OnnxRuntime;

/// <summary>
/// The CUDA runtime entry points this backend calls itself: a copy between host memory and the
/// card, in either direction, and the card's own allocation and release. ONNX Runtime's managed
/// surface has no such call, and a value living on the card hands out a pointer with no way to read
/// or fill it.
///
/// <para>Both directions serve one promise apiece. Device-to-host reads back a tensor on the
/// card — an output a run left there, or a tensor put there; host-to-device is what puts a tensor
/// into the card's memory in the first place, which is where a CUDA session reads every tensor it
/// is fed. The allocation is what the card's <see cref="CachingAllocator"/> gets its blocks
/// from.</para>
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

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int Malloc(out IntPtr pointer, nuint count);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int Free(IntPtr pointer);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int GetDevice(out int device);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int SetDevice(int device);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int HostAlloc(out IntPtr pointer, nuint count, uint flags);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int HostGetDevicePointer(out IntPtr device, IntPtr host, uint flags);

    /// <summary>The runtime's allocation entry points, bound together or not at all.</summary>
    private sealed record Allocation(
        Malloc Malloc, Free Free, GetDevice GetDevice, SetDevice SetDevice, GetLastError Clear,
        HostAlloc HostAlloc, HostGetDevicePointer HostGetDevicePointer, Free FreeHost);

    // cudaHostAllocPortable | cudaHostAllocMapped: pinned host memory every context may use, mapped
    // into the device's address space.
    private const uint PortableMapped = 1 | 2;

    private const int HostToDevice = 1;
    private const int DeviceToHost = 2;

    private static readonly object _gate = new();
    private static Memcpy? _memcpy;
    private static bool _bound;
    private static Allocation? _allocation;

    /// <summary>
    /// <paramref name="count"/> bytes of CUDA device <paramref name="deviceId"/>'s memory, from the
    /// CUDA runtime itself, or <see cref="IntPtr.Zero"/> where the card has not that much free or
    /// there is no runtime to ask. The thread's current device is left as it was found.
    /// </summary>
    public static IntPtr Allocate(int deviceId, long count)
    {
        if (BindAllocation() is not { } cuda) return IntPtr.Zero;
        return OnDevice(cuda, deviceId, () => cuda.Malloc(out var pointer, (nuint)count) == 0 ? pointer : IntPtr.Zero);
    }

    /// <summary>
    /// Hands <paramref name="pointer"/>, which <see cref="Allocate"/> answered for CUDA device
    /// <paramref name="deviceId"/>, back to the card. The runtime waits for the work the card has in
    /// hand before it does, so nothing still reading the block is cut short.
    /// </summary>
    public static void Release(int deviceId, IntPtr pointer)
    {
        if (BindAllocation() is not { } cuda) return;
        OnDevice(cuda, deviceId, () => cuda.Free(pointer) == 0 ? pointer : IntPtr.Zero);
    }

    /// <summary>
    /// <paramref name="call"/> with CUDA device <paramref name="deviceId"/> current on this thread,
    /// and the device that was current before it current again after. A failure is cleared once
    /// answered, as a failed copy's is (see <see cref="Bind"/>).
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

    /// <summary>
    /// <paramref name="count"/> bytes of pinned host memory mapped into CUDA device
    /// <paramref name="deviceId"/>'s address space, by the address a kernel on the device reads and
    /// writes it at, or <see cref="IntPtr.Zero"/> where there is not that much to pin. What a kernel
    /// is given in place of a block of the card's that could not be given it: it works, over the bus.
    /// </summary>
    public static IntPtr AllocateMappedHost(int deviceId, long count)
    {
        if (BindAllocation() is not { } cuda) return IntPtr.Zero;
        return OnDevice(cuda, deviceId, () =>
            cuda.HostAlloc(out var host, (nuint)count, PortableMapped) != 0 ? IntPtr.Zero
            : cuda.HostGetDevicePointer(out var device, host, 0) == 0 && device == host ? host
            : Unmapped(cuda, host));
    }

    /// <summary>Releases <paramref name="host"/>, pinned memory whose device address is not its own,
    /// which nothing here hands out.</summary>
    private static IntPtr Unmapped(Allocation cuda, IntPtr host)
    {
        cuda.FreeHost(host);
        return IntPtr.Zero;
    }

    /// <summary>Hands <paramref name="pointer"/>, which <see cref="AllocateMappedHost"/> answered,
    /// back.</summary>
    public static void ReleaseMappedHost(int deviceId, IntPtr pointer)
    {
        if (BindAllocation() is not { } cuda) return;
        OnDevice(cuda, deviceId, () => cuda.FreeHost(pointer) == 0 ? pointer : IntPtr.Zero);
    }

    private static Allocation? BindAllocation()
    {
        Bind();
        return _allocation;
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
                if (NativeLibrary.TryGetExport(handle, "cudaMalloc", out var malloc)
                    && NativeLibrary.TryGetExport(handle, "cudaFree", out var free)
                    && NativeLibrary.TryGetExport(handle, "cudaGetDevice", out var getDevice)
                    && NativeLibrary.TryGetExport(handle, "cudaSetDevice", out var setDevice)
                    && NativeLibrary.TryGetExport(handle, "cudaHostAlloc", out var hostAlloc)
                    && NativeLibrary.TryGetExport(handle, "cudaHostGetDevicePointer", out var hostDevicePointer)
                    && NativeLibrary.TryGetExport(handle, "cudaFreeHost", out var freeHost))
                    _allocation = new Allocation(
                        Marshal.GetDelegateForFunctionPointer<Malloc>(malloc),
                        Marshal.GetDelegateForFunctionPointer<Free>(free),
                        Marshal.GetDelegateForFunctionPointer<GetDevice>(getDevice),
                        Marshal.GetDelegateForFunctionPointer<SetDevice>(setDevice),
                        clear,
                        Marshal.GetDelegateForFunctionPointer<HostAlloc>(hostAlloc),
                        Marshal.GetDelegateForFunctionPointer<HostGetDevicePointer>(hostDevicePointer),
                        Marshal.GetDelegateForFunctionPointer<Free>(freeHost));
            }
            return _memcpy;
        }
    }
}

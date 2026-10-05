using System.Runtime.InteropServices;

namespace Shorokoo.Core.Backends;

/// <summary>
/// The CUDA driver's virtual memory management, which <see cref="CardMemory"/> builds a
/// card's arenas on: address ranges reserved apart from any memory (<c>cuMemAddressReserve</c>),
/// physical memory made a granule at a time (<c>cuMemCreate</c>), mapped into a range
/// (<c>cuMemMap</c>) and opened to the device (<c>cuMemSetAccess</c>), and each step undone on its
/// own. The library is resolved by name and its exports bound once; where it is absent, or the
/// device does not offer virtual memory management, <see cref="For"/> answers null and the card's
/// allocator takes its blocks from the CUDA runtime instead.
/// </summary>
internal static unsafe class CudaVirtualMemory
{
    private static string LibraryName => OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";

    // CUmemAllocationProp, for pinned device memory on one device and no exportable handle.
    [StructLayout(LayoutKind.Sequential)]
    internal struct AllocationProperties
    {
        internal int Type;
        internal int RequestedHandleTypes;
        internal int LocationType;
        internal int LocationId;
        internal IntPtr Win32HandleMetaData;
        internal ulong AllocationFlags;
    }

    // CUmemAccessDesc, read and write for one device.
    [StructLayout(LayoutKind.Sequential)]
    internal struct AccessDescription
    {
        internal int LocationType;
        internal int LocationId;
        internal int Flags;
    }

    private const int AllocationTypePinned = 1;
    private const int LocationTypeDevice = 1;
    private const int AccessReadWrite = 3;
    private const int VirtualMemoryManagementSupported = 102;

    /// <summary>The driver's entry points this needs, bound together or not at all.</summary>
    internal sealed class Api
    {
        internal delegate* unmanaged<int*, int, int, int> GetAttribute;
        internal delegate* unmanaged<nuint*, AllocationProperties*, int, int> Granularity;
        internal delegate* unmanaged<ulong*, nuint, nuint, ulong, ulong, int> AddressReserve;
        internal delegate* unmanaged<ulong, nuint, int> AddressFree;
        internal delegate* unmanaged<ulong*, nuint, AllocationProperties*, ulong, int> Create;
        internal delegate* unmanaged<ulong, int> ReleaseHandle;
        internal delegate* unmanaged<ulong, nuint, nuint, ulong, ulong, int> Map;
        internal delegate* unmanaged<ulong, nuint, int> Unmap;
        internal delegate* unmanaged<ulong, nuint, AccessDescription*, nuint, int> SetAccess;
    }

    private static readonly object _gate = new();
    private static Api? _api;
    private static bool _bound;

    /// <summary>
    /// The driver's virtual memory management for CUDA device <paramref name="device"/>, with the
    /// granule its physical memory comes in, or null where the driver or the device does not offer
    /// it.
    /// </summary>
    internal static (Api Api, long Granule)? For(int device)
    {
        if (Bind() is not { } api) return null;
        int supported;
        if (api.GetAttribute(&supported, VirtualMemoryManagementSupported, device) != 0 || supported == 0) return null;
        var properties = Properties(device);
        nuint granule;
        if (api.Granularity(&granule, &properties, 0) != 0 || granule == 0) return null;
        return (api, (long)granule);
    }

    /// <summary>The properties a granule of device <paramref name="device"/>'s memory is made
    /// with.</summary>
    internal static AllocationProperties Properties(int device)
        => new() { Type = AllocationTypePinned, LocationType = LocationTypeDevice, LocationId = device };

    /// <summary>Read and write access for device <paramref name="device"/>.</summary>
    internal static AccessDescription ReadWrite(int device)
        => new() { LocationType = LocationTypeDevice, LocationId = device, Flags = AccessReadWrite };

    private static Api? Bind()
    {
        lock (_gate)
        {
            if (_bound) return _api;
            _bound = true;
            try
            {
                if (!NativeLibrary.TryLoad(LibraryName, out var library)) return null;
                IntPtr Export(string name) => NativeLibrary.TryGetExport(library, name, out var export) ? export : IntPtr.Zero;
                var init = (delegate* unmanaged<uint, int>)Export("cuInit");
                string[] names =
                [
                    "cuDeviceGetAttribute", "cuMemGetAllocationGranularity", "cuMemAddressReserve", "cuMemAddressFree",
                    "cuMemCreate", "cuMemRelease", "cuMemMap", "cuMemUnmap", "cuMemSetAccess",
                ];
                var exports = names.Select(Export).ToArray();
                if (init == null || exports.Any(e => e == IntPtr.Zero) || init(0) != 0)
                {
                    // The library stays loaded where it bound, since the entry points point into it.
                    NativeLibrary.Free(library);
                    return null;
                }
                _api = new Api
                {
                    GetAttribute = (delegate* unmanaged<int*, int, int, int>)exports[0],
                    Granularity = (delegate* unmanaged<nuint*, AllocationProperties*, int, int>)exports[1],
                    AddressReserve = (delegate* unmanaged<ulong*, nuint, nuint, ulong, ulong, int>)exports[2],
                    AddressFree = (delegate* unmanaged<ulong, nuint, int>)exports[3],
                    Create = (delegate* unmanaged<ulong*, nuint, AllocationProperties*, ulong, int>)exports[4],
                    ReleaseHandle = (delegate* unmanaged<ulong, int>)exports[5],
                    Map = (delegate* unmanaged<ulong, nuint, nuint, ulong, ulong, int>)exports[6],
                    Unmap = (delegate* unmanaged<ulong, nuint, int>)exports[7],
                    SetAccess = (delegate* unmanaged<ulong, nuint, AccessDescription*, nuint, int>)exports[8],
                };
                return _api;
            }
            // Binding is best effort: where it fails, the card's blocks come from the runtime.
            catch (Exception)
            {
                _api = null;
                return null;
            }
        }
    }
}

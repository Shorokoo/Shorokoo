using System.Runtime.InteropServices;

namespace Shorokoo.Core.Backends;

/// <summary>
/// What this process holds on a CUDA device — its own allocations, the arenas, the resident state
/// and the CUDA context alike, and no other process's — which
/// <see cref="DeviceMemoryReading.ProcessBytes"/> reports.
///
/// <para>No one interface answers this everywhere, so it asks whichever one the device's driver
/// model answers through:</para>
/// <list type="bullet">
/// <item>On Windows, a card driven through WDDM — any card that also drives a display, and the
/// default for a GeForce — is a DXGI adapter, and <c>IDXGIAdapter3::QueryVideoMemoryInfo</c>
/// reports the calling process's usage of its own memory. It is the figure Task Manager shows per
/// process. NVML cannot give it there: under WDDM it lists the process with its usage not
/// available, which is what <c>nvidia-smi</c> prints as <c>[N/A]</c>. The card is matched to the
/// CUDA device by its LUID.</item>
/// <item>Under WSL, the Windows driver drives the card and the Linux side reaches it through
/// <c>libdxcore.so</c>, beside <c>libcuda.so.1</c> in <c>/usr/lib/wsl/lib</c>. Its
/// <c>D3DKMTQueryVideoMemoryInfo</c>, asked with no process, reports the calling process's usage of
/// the adapter's own memory: the figure DXGI gives on Windows. NVML cannot give it there, as it
/// lists no process on the card. The card is matched to the CUDA device by its PCI bus, device and
/// function, which the adapter reports as CUDA does; its LUID is not the one CUDA reports.</item>
/// <item>Everywhere else — native Linux, and a Windows card in TCC mode, which DXGI does not see —
/// NVML lists each process's usage of the device, and this takes the entries under this process's
/// id. The card is matched by its PCI bus id.</item>
/// </list>
///
/// <para>Bound lazily, per device, and best effort, as <see cref="CudaRuntime"/> is: a device no
/// interface will attribute memory on, or a machine that has none of them, reads <c>null</c>
/// rather than throwing. That includes a process NVML does not list under its own id, as in a
/// container with its own process-id namespace: an absent entry cannot be told from one this
/// process does not have, and a reading of zero would be a guess. So is a WSL card that no adapter,
/// or more than one, matches.</para>
/// </summary>
internal static unsafe class ProcessDeviceMemory
{
    /// <summary>The source that answers for one device, bound once and kept.</summary>
    private abstract class Source
    {
        internal abstract long? Read();
    }

    private static readonly object _gate = new();
    private static readonly Dictionary<int, Source?> _byDevice = [];

    /// <summary>
    /// The bytes this process holds on the CUDA device current for the calling thread — the one
    /// <see cref="CudaRuntime.TryMemGetInfo"/> reads — or <c>null</c> where nothing will say.
    /// </summary>
    internal static long? Read()
    {
        if (!CudaRuntime.TryGetDevice(out var device)) return null;
        Source? source;
        lock (_gate)
        {
            if (!_byDevice.TryGetValue(device, out source))
                _byDevice[device] = source = Bind(device);
        }
        try { return source?.Read(); }
        // A reading is worth nothing next to failing a run.
        catch (Exception) { return null; }
    }

    private static Source? Bind(int device)
    {
        try
        {
            Source? driverModel = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? Dxgi.Bind(device)
                : RuntimeInformation.IsOSPlatform(OSPlatform.Linux) ? DxCore.Bind(device)
                : null;
            return driverModel ?? Nvml.Bind(device);
        }
        catch (Exception) { return null; }
    }

    /// <summary>The WDDM route: the DXGI adapter with the CUDA device's LUID.</summary>
    private sealed class Dxgi : Source
    {
        private const int SegmentGroupLocal = 0;
        // IDXGIFactory4::EnumAdapterByLuid and IDXGIAdapter3::QueryVideoMemoryInfo, counted down
        // their interfaces' vtables from IUnknown.
        private const int EnumAdapterByLuidSlot = 26;
        private const int QueryVideoMemoryInfoSlot = 14;
        private const int ReleaseSlot = 2;

        private static readonly Guid IidFactory4 = new("1bc6ea02-ef36-464f-bf0c-21ca39e5168a");
        private static readonly Guid IidAdapter3 = new("645967a4-1392-4310-a798-8053ce3e93fd");

        /// <summary>
        /// The adapter, held for the life of the process: one reference per card, taken once, and
        /// the reading goes through it on every sample.
        /// </summary>
        private readonly nint _adapter;

        private Dxgi(nint adapter) => _adapter = adapter;

        /// <summary><c>DXGI_QUERY_VIDEO_MEMORY_INFO</c>.</summary>
        [StructLayout(LayoutKind.Sequential)]
        private struct VideoMemoryInfo
        {
            public ulong Budget;
            public ulong CurrentUsage;
            public ulong AvailableForReservation;
            public ulong CurrentReservation;
        }

        internal static Dxgi? Bind(int device)
        {
            if (Luid(device) is not { } luid) return null;
            if (!NativeLibrary.TryLoad("dxgi.dll", out var dxgi)
                || !NativeLibrary.TryGetExport(dxgi, "CreateDXGIFactory1", out var export))
                return null;
            var create = (delegate* unmanaged<Guid*, nint*, int>)export;

            Guid iidFactory = IidFactory4, iidAdapter = IidAdapter3;
            nint factory = 0;
            if (create(&iidFactory, &factory) != 0 || factory == 0) return null;
            try
            {
                var enumByLuid = (delegate* unmanaged<nint, long, Guid*, nint*, int>)Slot(factory, EnumAdapterByLuidSlot);
                nint adapter = 0;
                if (enumByLuid(factory, luid, &iidAdapter, &adapter) != 0 || adapter == 0) return null;
                var bound = new Dxgi(adapter);
                if (bound.Read() is null)
                {
                    Release(adapter);
                    return null;
                }
                return bound;
            }
            finally { Release(factory); }
        }

        internal override long? Read()
        {
            var query = (delegate* unmanaged<nint, uint, int, VideoMemoryInfo*, int>)Slot(_adapter, QueryVideoMemoryInfoSlot);
            VideoMemoryInfo info;
            return query(_adapter, 0, SegmentGroupLocal, &info) == 0 ? (long)info.CurrentUsage : null;
        }

        private static nint Slot(nint comObject, int slot) => (*(nint**)comObject)[slot];

        private static void Release(nint comObject)
            => ((delegate* unmanaged<nint, uint>)Slot(comObject, ReleaseSlot))(comObject);

        /// <summary>
        /// The LUID of CUDA device <paramref name="device"/>, from the driver API, which reports
        /// one only for a card WDDM drives — so a TCC card answers null here and goes to NVML.
        /// </summary>
        private static long? Luid(int device)
        {
            if (!NativeLibrary.TryLoad("nvcuda.dll", out var nvcuda)
                || !NativeLibrary.TryGetExport(nvcuda, "cuInit", out var initExport)
                || !NativeLibrary.TryGetExport(nvcuda, "cuDeviceGet", out var getExport)
                || !NativeLibrary.TryGetExport(nvcuda, "cuDeviceGetLuid", out var luidExport))
                return null;
            int handle;
            long luid;
            uint nodeMask;
            if (((delegate* unmanaged<uint, int>)initExport)(0) != 0
                || ((delegate* unmanaged<int*, int, int>)getExport)(&handle, device) != 0
                || ((delegate* unmanaged<long*, uint*, int, int>)luidExport)(&luid, &nodeMask, handle) != 0)
                return null;
            return luid;
        }
    }

    /// <summary>
    /// The WSL route: the DXCore adapter at the CUDA device's PCI address, through the kernel-mode
    /// thunks <c>libdxcore.so</c> exports. Bound only where that library is there, which is WSL.
    /// </summary>
    private sealed class DxCore : Source
    {
        private const string Library = "/usr/lib/wsl/lib/libdxcore.so";
        private const int SegmentGroupLocal = 0;
        /// <summary><c>KMTQAITYPE_ADAPTERADDRESS</c>.</summary>
        private const int AdapterAddress = 6;

        private readonly uint _adapter;
        private readonly delegate* unmanaged<VideoMemoryInfo*, int> _query;

        private DxCore(uint adapter, delegate* unmanaged<VideoMemoryInfo*, int> query)
        {
            _adapter = adapter;
            _query = query;
        }

        /// <summary><c>D3DKMT_ADAPTERINFO</c>.</summary>
        [StructLayout(LayoutKind.Sequential)]
        private struct AdapterInfo
        {
            public uint Adapter;
            public uint LuidLowPart;
            public int LuidHighPart;
            public uint NumOfSources;
            public int PrecisePresentRegionsPreferred;
        }

        /// <summary><c>D3DKMT_ENUMADAPTERS2</c>.</summary>
        [StructLayout(LayoutKind.Sequential)]
        private struct EnumAdapters
        {
            public uint NumAdapters;
            public AdapterInfo* Adapters;
        }

        /// <summary><c>D3DKMT_QUERYADAPTERINFO</c>.</summary>
        [StructLayout(LayoutKind.Sequential)]
        private struct QueryAdapterInfo
        {
            public uint Adapter;
            public int Type;
            public void* PrivateDriverData;
            public uint PrivateDriverDataSize;
        }

        /// <summary><c>D3DKMT_ADAPTERADDRESS</c>.</summary>
        [StructLayout(LayoutKind.Sequential)]
        private struct Address
        {
            public uint BusNumber;
            public uint DeviceNumber;
            public uint FunctionNumber;
        }

        /// <summary><c>D3DKMT_QUERYVIDEOMEMORYINFO</c>.</summary>
        [StructLayout(LayoutKind.Sequential)]
        private struct VideoMemoryInfo
        {
            public nint Process;
            public uint Adapter;
            public int MemorySegmentGroup;
            public ulong Budget;
            public ulong CurrentUsage;
            public ulong CurrentReservation;
            public ulong AvailableForReservation;
            public uint PhysicalAdapterIndex;
        }

        internal static DxCore? Bind(int device)
        {
            if (!File.Exists(Library) || PciAddress(device) is not { } wanted) return null;
            // The library stays loaded, and the matched adapter open, for as long as the process reads.
            if (!NativeLibrary.TryLoad(Library, out var dxcore)
                || !NativeLibrary.TryGetExport(dxcore, "D3DKMTEnumAdapters2", out var enumExport)
                || !NativeLibrary.TryGetExport(dxcore, "D3DKMTQueryAdapterInfo", out var infoExport)
                || !NativeLibrary.TryGetExport(dxcore, "D3DKMTQueryVideoMemoryInfo", out var queryExport)
                || !NativeLibrary.TryGetExport(dxcore, "D3DKMTCloseAdapter", out var closeExport))
                return null;
            var enumerate = (delegate* unmanaged<EnumAdapters*, int>)enumExport;
            var info = (delegate* unmanaged<QueryAdapterInfo*, int>)infoExport;
            var close = (delegate* unmanaged<uint*, int>)closeExport;

            // Asked with no array, the call counts the adapters; asked again, it opens each one.
            EnumAdapters list = default;
            if (enumerate(&list) != 0 || list.NumAdapters == 0) return null;
            var adapters = new AdapterInfo[list.NumAdapters];
            fixed (AdapterInfo* first = adapters)
            {
                list.Adapters = first;
                if (enumerate(&list) != 0) return null;
            }
            var opened = adapters.AsSpan(0, (int)Math.Min(list.NumAdapters, (uint)adapters.Length));

            uint? match = null;
            bool ambiguous = false;
            foreach (var adapter in opened)
            {
                Address address = default;
                var query = new QueryAdapterInfo
                {
                    Adapter = adapter.Adapter,
                    Type = AdapterAddress,
                    PrivateDriverData = &address,
                    PrivateDriverDataSize = (uint)sizeof(Address),
                };
                if (info(&query) != 0 || (address.BusNumber, address.DeviceNumber, address.FunctionNumber) != wanted) continue;
                ambiguous |= match is not null;
                match = adapter.Adapter;
            }
            foreach (var adapter in opened)
            {
                uint handle = adapter.Adapter;
                if (ambiguous || handle != match) close(&handle);
            }
            if (ambiguous || match is not { } matched) return null;

            var bound = new DxCore(matched, (delegate* unmanaged<VideoMemoryInfo*, int>)queryExport);
            if (bound.Read() is null)
            {
                close(&matched);
                return null;
            }
            return bound;
        }

        internal override long? Read()
        {
            // With no process named, the query answers for the calling one.
            var info = new VideoMemoryInfo { Adapter = _adapter, MemorySegmentGroup = SegmentGroupLocal };
            return _query(&info) == 0 ? (long)info.CurrentUsage : null;
        }

        /// <summary>
        /// The bus, device and function of CUDA device <paramref name="device"/>, from its PCI bus id
        /// (<c>domain:bus:device.function</c>, in hexadecimal).
        /// </summary>
        private static (uint Bus, uint Device, uint Function)? PciAddress(int device)
        {
            if (CudaRuntime.TryGetPciBusId(device) is not { } busId) return null;
            var parts = busId.Split(':', '.');
            if (parts.Length != 4) return null;
            uint[] numbers = [0, 0, 0];
            for (int i = 0; i < 3; i++)
                if (!uint.TryParse(parts[i + 1], System.Globalization.NumberStyles.HexNumber, null, out numbers[i])) return null;
            return (numbers[0], numbers[1], numbers[2]);
        }
    }

    /// <summary>The NVML route: the device with the CUDA device's PCI bus id.</summary>
    private sealed class Nvml : Source
    {
        private const int Success = 0;
        private const int InsufficientSize = 7;
        /// <summary><c>NVML_VALUE_NOT_AVAILABLE</c>: the process is listed, its usage is not.</summary>
        private const ulong NotAvailable = ulong.MaxValue;

        private readonly nint _device;
        private readonly delegate* unmanaged<nint, uint*, ProcessInfo*, int> _processes;

        private Nvml(nint device, delegate* unmanaged<nint, uint*, ProcessInfo*, int> processes)
        {
            _device = device;
            _processes = processes;
        }

        /// <summary><c>nvmlProcessInfo_t</c>, as <c>nvmlDeviceGetComputeRunningProcesses_v3</c>
        /// fills it.</summary>
        [StructLayout(LayoutKind.Sequential)]
        private struct ProcessInfo
        {
            public uint Pid;
            public ulong UsedGpuMemory;
            public uint GpuInstanceId;
            public uint ComputeInstanceId;
        }

        internal static Nvml? Bind(int device)
        {
            if (CudaRuntime.TryGetPciBusId(device) is not { } busId) return null;
            // On Windows the driver installs NVML into the system directory, and it is named in full
            // there: loaded by bare name, any nvml.dll found first — beside the application, say —
            // would be taken instead.
            var name = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                ? Path.Combine(Environment.SystemDirectory, "nvml.dll")
                : "libnvidia-ml.so.1";
            // The library stays loaded, and NVML initialized, for as long as the process reads.
            if (!NativeLibrary.TryLoad(name, out var nvml)
                || !NativeLibrary.TryGetExport(nvml, "nvmlInit_v2", out var initExport)
                || !NativeLibrary.TryGetExport(nvml, "nvmlDeviceGetHandleByPciBusId_v2", out var byBusExport)
                || !NativeLibrary.TryGetExport(nvml, "nvmlDeviceGetComputeRunningProcesses_v3", out var processesExport))
                return null;
            if (((delegate* unmanaged<int>)initExport)() != Success) return null;

            var bus = System.Text.Encoding.ASCII.GetBytes(busId + "\0");
            nint handle;
            fixed (byte* text = bus)
                if (((delegate* unmanaged<byte*, nint*, int>)byBusExport)(text, &handle) != Success) return null;
            var bound = new Nvml(handle, (delegate* unmanaged<nint, uint*, ProcessInfo*, int>)processesExport);
            return bound.Read() is null ? null : bound;
        }

        internal override long? Read()
        {
            // Sized for a busy card and grown if NVML says it holds more.
            var infos = new ProcessInfo[64];
            while (true)
            {
                uint count = (uint)infos.Length;
                int status;
                fixed (ProcessInfo* first = infos) status = _processes(_device, &count, first);
                if (status == InsufficientSize)
                {
                    infos = new ProcessInfo[Math.Max(count, (uint)infos.Length) * 2];
                    continue;
                }
                if (status != Success) return null;
                return Own(infos.AsSpan(0, (int)count), (uint)Environment.ProcessId);
            }
        }

        /// <summary>
        /// This process's usage among <paramref name="infos"/>: the sum of its entries — one per
        /// MIG instance it runs on — or null when it has none, or NVML will not say for one.
        /// </summary>
        private static long? Own(ReadOnlySpan<ProcessInfo> infos, uint pid)
        {
            long? total = null;
            foreach (var info in infos)
            {
                if (info.Pid != pid) continue;
                if (info.UsedGpuMemory == NotAvailable) return null;
                total = (total ?? 0) + (long)info.UsedGpuMemory;
            }
            return total;
        }
    }
}

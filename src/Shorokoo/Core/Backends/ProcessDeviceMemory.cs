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
/// <item>Everywhere else — Linux, and a Windows card in TCC mode, which DXGI does not see — NVML
/// lists each process's usage of the device, and this takes the entries under this process's id.
/// The card is matched by its PCI bus id.</item>
/// </list>
///
/// <para>Bound lazily, per device, and best effort, as <see cref="CudaRuntime"/> is: a device no
/// interface will attribute memory on, or a machine that has none of them, reads <c>null</c>
/// rather than throwing. That includes a process NVML does not list under its own id, as in a
/// container with its own process-id namespace: an absent entry cannot be told from one this
/// process does not have, and a reading of zero would be a guess.</para>
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
            Source? dxgi = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? Dxgi.Bind(device) : null;
            return dxgi ?? Nvml.Bind(device);
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

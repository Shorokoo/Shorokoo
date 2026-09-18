using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using Shorokoo.Core.Inference.Abstractions;

namespace Shorokoo.Core.Utils
{
    /// <summary>
    /// What is known about the accelerator side of a failing operation.
    /// <paramref name="HasDeviceMemory"/> is the authority on whether there is device memory to
    /// exhaust at all — a session's own answer, not a guess from the backend's name — and is the
    /// same value handed to <see cref="AllocationFailureReport.Classify"/>, so the classification
    /// and the wording built on it cannot disagree.
    /// </summary>
    /// <param name="HasDeviceMemory">Whether the failing session produces outputs in device memory.</param>
    /// <param name="Reading">The card's memory right now, or <c>null</c> where no CUDA runtime answers.</param>
    /// <param name="ArenaLimitBytes">This process's configured arena cap, or <c>null</c> when uncapped.</param>
    /// <param name="BackendAssemblyName">The loaded backend assembly, for display only.</param>
    internal readonly record struct DeviceFacts(
        bool HasDeviceMemory,
        DeviceMemoryReading? Reading,
        long? ArenaLimitBytes,
        string? BackendAssemblyName);

    /// <summary>Which allocator ran out, as far as the backend's own text allows it to be told apart.</summary>
    internal enum AllocationPool
    {
        /// <summary>A host (CPU) allocation — the C++ runtime's <c>new</c>, or a managed out-of-memory.</summary>
        Host,

        /// <summary>A device (accelerator) allocation — the CUDA execution provider's arena.</summary>
        Device,

        /// <summary>The text names no allocator, and the backend in use could be either.</summary>
        Unknown,
    }

    /// <summary>
    /// The facts a process can state about its own memory without asking the backend: the working
    /// set and commit charge, the managed heap, and the memory limit the runtime is running under —
    /// a container/cgroup or Job Object limit where one is in force, otherwise physical RAM.
    /// </summary>
    /// <param name="WorkingSetBytes">Resident set of this process, or <c>null</c> when unreadable.</param>
    /// <param name="CommitBytes">Private commit charge of this process, or <c>null</c> when unreadable.</param>
    /// <param name="ManagedHeapBytes">Bytes the GC considers allocated on the managed heap.</param>
    /// <param name="LimitBytes">The memory ceiling the runtime sees, or <c>null</c> when unreadable.</param>
    /// <param name="LimitIsExplicit">True when that ceiling is a configured limit (cgroup / Job Object /
    /// GC heap hard limit) rather than simply the machine's physical memory.</param>
    internal readonly record struct ProcessMemoryFacts(
        long? WorkingSetBytes,
        long? CommitBytes,
        long ManagedHeapBytes,
        long? LimitBytes,
        bool LimitIsExplicit);

    /// <summary>
    /// Turns an allocation failure raised from inside the backend into a message that says which
    /// memory ran out and what the process's own memory position was when it did.
    ///
    /// <para>The failures this exists for arrive as bare backend text — ONNX Runtime reports an
    /// arena expansion failure as a build-agent source path plus "Failed to allocate memory for
    /// requested buffer of size N", and a host <c>std::bad_alloc</c> as the two words "bad
    /// allocation". Neither says host or device, neither names the pool or its size, and the two
    /// arrive with identical managed stacks, so nothing in the exception separates "this
    /// accelerator is full" from "this process may not commit any more host memory"
    /// (Shorokoo/Shorokoo#330, Shorokoo/Shorokoo#332). Those two call for opposite responses —
    /// shrink the model, versus raise a limit that has nothing to do with the model — so the report
    /// states the process's commit against any limit in force, which is the one number that tells
    /// them apart.</para>
    ///
    /// <para><see cref="Render"/> takes every fact as a parameter and touches nothing ambient, so
    /// the wording can be driven from a test without the memory conditions that produce it; the
    /// gatherers (<see cref="Classify"/>, <see cref="ReadProcessMemory"/>) are the only parts that
    /// look at the live process.</para>
    /// </summary>
    internal static class AllocationFailureReport
    {
        // Text the backends actually produce for an allocation abort. ORT surfaces the native
        // failures as strings with no error code of their own, so the text is all there is. CUDA
        // spells its own out-of-memory with underscores, so both spacings are listed.
        private static readonly string[] AllocationMarkers =
        [
            "bad alloc", "bad_alloc", "out of memory", "out_of_memory", "outofmemory",
            "failed to allocate", "insufficient memory", "alloc_failed",
        ];

        // Fragments that name a CUDA allocator outright, so they mean device memory whatever the
        // session is.
        private static readonly string[] CudaMarkers =
        [
            "cuda_allocator", "cudaerrormemoryallocation", "cuda_error_out_of_memory",
            "cudamalloc", "cudnn_status_alloc_failed", "cublas_status_alloc_failed",
        ];

        // Fragments naming ONNX Runtime's BFC arena. These say NOTHING about which memory: the
        // arena in core/framework is execution-provider-agnostic and backs the CPU allocator too
        // (enable_cpu_mem_arena defaults on), so the identical text is produced by a CPU-only build.
        // Only a session that has device memory could have been allocating on a device at all.
        private static readonly string[] ArenaMarkers = ["bfc_arena", "bfcarena"];

        // A C++ `new` that could not be satisfied: host commit, whatever the session's device is.
        private static readonly string[] HostMarkers = ["bad alloc", "bad_alloc"];

        /// <summary>
        /// True for an allocation abort: a managed <see cref="OutOfMemoryException"/>, or a native
        /// failure the backend reports only as text. Walks the whole inner-exception chain, since
        /// the backend's own exception is usually wrapped by the time it reaches a step.
        /// </summary>
        internal static bool IsAllocationFailure(Exception ex)
        {
            for (var e = ex; e is not null; e = e.InnerException)
            {
                if (e is OutOfMemoryException) return true;
                // Device markers count as recognition too, or a failure Classify would name as
                // DEVICE would never reach it.
                if (ContainsAny(e.Message, AllocationMarkers)
                    || ContainsAny(e.Message, CudaMarkers)
                    || ContainsAny(e.Message, ArenaMarkers))
                    return true;
            }
            return false;
        }

        /// <summary>
        /// Which allocator the failure came from, as far as the text allows. An arena/CUDA frame
        /// names the device; a bare <c>bad allocation</c> is the C++ runtime failing to satisfy a
        /// host <c>new</c>, so it is host memory even on a GPU-backed session — that pair is
        /// exactly the one Shorokoo/Shorokoo#330 reports as indistinguishable. With neither marker
        /// present the pool is only known when the backend has no device to allocate on at all.
        /// </summary>
        internal static AllocationPool Classify(Exception ex, bool gpuBackend)
        {
            // Device evidence wins wherever it sits in the chain, so the WHOLE chain is scanned for
            // it before any host marker is considered. A CUDA frame names the allocator outright;
            // "bad allocation" is generic, and is exactly what the backend wraps such a frame in by
            // the time a step sees it — scanning frame by frame would let the generic outer text
            // mask the specific inner one, and report a device failure as a host one.
            for (var e = ex; e is not null; e = e.InnerException)
                if (ContainsAny(e.Message, CudaMarkers)) return AllocationPool.Device;

            // An arena frame is only device evidence on a session that HAS device memory: the same
            // text comes out of the CPU arena, so reading it as the accelerator's on a CPU-only
            // session would blame hardware that is not there.
            for (var e = ex; e is not null; e = e.InnerException)
                if (ContainsAny(e.Message, ArenaMarkers))
                    return gpuBackend ? AllocationPool.Device : AllocationPool.Host;

            for (var e = ex; e is not null; e = e.InnerException)
            {
                if (ContainsAny(e.Message, HostMarkers)) return AllocationPool.Host;
                if (e is OutOfMemoryException) return AllocationPool.Host;
            }
            return gpuBackend ? AllocationPool.Unknown : AllocationPool.Host;
        }

        /// <summary>
        /// Substring search over a marker table with no allocation. This runs inside an exception
        /// FILTER, which the CLR evaluates before unwinding: a filter that throws is treated as
        /// having returned false, so a LINQ chain allocating here under a genuine out-of-memory
        /// would silently skip the very wrap it is deciding on.
        /// </summary>
        private static bool ContainsAny(string text, string[] markers)
        {
            foreach (var m in markers)
                if (text.Contains(m, StringComparison.OrdinalIgnoreCase)) return true;
            return false;
        }

        /// <summary>True when the loaded backend drives an accelerator rather than the CPU.</summary>
        internal static bool IsGpuBackend(string? backendAssemblyName)
            => backendAssemblyName is not null
               && backendAssemblyName.Contains("GPU", StringComparison.OrdinalIgnoreCase);

        /// <summary>The assembly name of the loaded backend, or null when it cannot be determined.</summary>
        internal static string? BackendAssemblyName()
        {
            try { return InferenceBackend.Default.GetType().Assembly.GetName().Name; }
            catch { return null; }
        }

        /// <summary>
        /// Reads this process's memory position. Every reading is optional: a platform that does
        /// not expose one contributes <c>null</c> and the report simply omits it, because a
        /// diagnostic that throws while diagnosing is worse than one that is incomplete.
        /// </summary>
        internal static ProcessMemoryFacts ReadProcessMemory()
        {
            long? workingSet = null, commit = null;
            try
            {
                using var proc = Process.GetCurrentProcess();
                workingSet = proc.WorkingSet64;
                commit = proc.PrivateMemorySize64;
            }
            catch { /* a sandbox may deny /proc or the Win32 query; the report says so by omission */ }

            long managed = 0;
            try { managed = GC.GetTotalMemory(forceFullCollection: false); } catch { }

            long? limit = null;
            bool explicitLimit = false;
            try
            {
                var info = GC.GetGCMemoryInfo();
                if (info.TotalAvailableMemoryBytes > 0) limit = info.TotalAvailableMemoryBytes;
            }
            catch { }

            // A cgroup ceiling is the limit actually in force on Linux and is readable directly,
            // which also tells us the runtime's figure is a configured limit rather than RAM.
            if (ReadCGroupLimit() is long cgroup && cgroup > 0)
            {
                if (limit is null || cgroup < limit) limit = cgroup;
                explicitLimit = true;
            }
            else if (limit is long seen && PhysicalMemoryBytes() is long physical && seen < physical)
            {
                explicitLimit = true;
            }

            return new ProcessMemoryFacts(workingSet, commit, managed, limit, explicitLimit);
        }

        private static long? ReadCGroupLimit()
        {
            string[] paths =
            [
                "/sys/fs/cgroup/memory.max",                       // cgroup v2
                "/sys/fs/cgroup/memory/memory.limit_in_bytes",     // cgroup v1
            ];
            foreach (var path in paths)
            {
                try
                {
                    if (!File.Exists(path)) continue;
                    var text = File.ReadAllText(path).Trim();
                    if (text is "max") continue;
                    if (long.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out var v)
                        && v > 0 && v < long.MaxValue / 2)
                        return v;
                }
                catch { }
            }
            return null;
        }

        private static long? PhysicalMemoryBytes()
        {
            try
            {
                foreach (var line in File.ReadLines("/proc/meminfo"))
                {
                    if (!line.StartsWith("MemTotal:", StringComparison.Ordinal)) continue;
                    var kb = line.Split(':')[1].Trim().Split(' ')[0];
                    if (long.TryParse(kb, NumberStyles.Integer, CultureInfo.InvariantCulture, out var v))
                        return v * 1024;
                }
            }
            catch { }
            return null;
        }

        /// <summary>
        /// Renders the report. Pure: every fact is a parameter, so a test can state the memory
        /// conditions instead of having to create them.
        /// </summary>
        /// <param name="operation">What was being run, e.g. <c>"the training step at step 41"</c>.</param>
        /// <param name="pool">Which allocator failed, per <see cref="Classify"/>.</param>
        /// <param name="device">What is known about the accelerator side, per <see cref="DeviceFacts"/>.</param>
        /// <param name="inventory">The tensor state resident for the operation, by section.</param>
        /// <param name="memory">This process's memory position, per <see cref="ReadProcessMemory"/>.</param>
        /// <param name="underlying">The backend's own text, quoted verbatim at the end.</param>
        internal static string Render(
            string operation,
            AllocationPool pool,
            DeviceFacts device,
            IReadOnlyList<TensorInventorySection> inventory,
            ProcessMemoryFacts memory,
            string underlying)
        {
            var parts = new List<string>
            {
                $"allocating memory for {operation} failed.",
                DescribePool(pool, device),
            };

            if (inventory.Count > 0) parts.Add(DescribeInventory(inventory));
            if (DescribeDevice(device) is string deviceText) parts.Add(deviceText);
            parts.Add(DescribeMemory(memory));
            if (Advice(pool, device, memory) is string advice) parts.Add(advice);
            parts.Add("Underlying failure: " + underlying);
            return string.Join(" ", parts);
        }

        /// <summary>
        /// The card's own figures, where a CUDA runtime is there to ask. This is what turns "the
        /// accelerator is full" from an inference into a statement: Shorokoo/Shorokoo#332's failure
        /// is a 2.36 MB request refused with gigabytes still free on the device, which reads as
        /// absurd until the free figure is printed next to it.
        /// </summary>
        private static string? DescribeDevice(DeviceFacts device)
        {
            if (device.Reading is not DeviceMemoryReading d) return null;
            var text = $"Device: {Bytes(d.UsedBytes)} of {Bytes(d.TotalBytes)} in use across all "
                       + $"processes, {Bytes(d.FreeBytes)} free";
            if (device.ArenaLimitBytes is long limit)
                text += $"; this session's arena is capped at {Bytes(limit)} (its context's "
                      + "DeviceMemorySettings.LimitBytes), and any other live session can hold a "
                      + "budget of its own on top";
            return text + ".";
        }

        private static string DescribePool(AllocationPool pool, DeviceFacts device)
        {
            bool gpu = device.HasDeviceMemory;
            string where = device.BackendAssemblyName is not string backend ? "" : $" (backend '{backend}')";
            return pool switch
            {
                AllocationPool.Device =>
                    $"The failing allocation was for DEVICE memory — the accelerator's arena{where}.",
                AllocationPool.Host when gpu =>
                    "The failing allocation was for HOST memory — a C++ allocation the process could "
                    + $"not commit, not the accelerator's arena{where}.",
                AllocationPool.Host =>
                    $"The failing allocation was for HOST memory{where}.",
                _ =>
                    "The backend's text names no allocator, so HOST and DEVICE memory cannot be told "
                    + $"apart from the failure alone{where}; the figures below narrow it.",
            };
        }

        private static string DescribeInventory(IReadOnlyList<TensorInventorySection> inventory)
        {
            long totalBytes = 0;
            int totalTensors = 0;
            foreach (var section in inventory)
            {
                // A section of unknown size makes the whole total unknown. Folding its -1 sentinel
                // in arithmetically would saturate the sum and print an absurd figure — the one
                // thing a memory diagnostic must not do.
                totalBytes = totalBytes < 0 || section.TotalBytes < 0
                    ? -1 : AddSat(totalBytes, section.TotalBytes);
                totalTensors += section.Tensors.Count;
            }

            var perSection = inventory.Select(
                s => $"{s.Name} {Bytes(s.TotalBytes)} across {s.Tensors.Count} tensor(s)");

            var largest = inventory
                .SelectMany(s => s.Tensors.Select(t => (Section: s.Name, Tensor: t)))
                .OrderByDescending(x => x.Tensor.Bytes)
                .Take(5)
                .Select(x => $"'{x.Section}/{x.Tensor.Name}' {x.Tensor.DType} "
                             + $"[{string.Join(", ", x.Tensor.Shape)}] = {Bytes(x.Tensor.Bytes)}");

            var text = $"Training state held for this operation: {totalTensors} tensor(s), "
                       + $"{Bytes(totalBytes)} in total ({string.Join("; ", perSection)})";
            return totalTensors == 0
                ? text + "."
                : text + "; the largest: " + string.Join("; ", largest) + ".";
        }

        private static string DescribeMemory(ProcessMemoryFacts m)
        {
            var readings = new List<string>();
            if (m.WorkingSetBytes is long ws) readings.Add($"working set {Bytes(ws)}");
            if (m.CommitBytes is long commit) readings.Add($"commit {Bytes(commit)}");
            readings.Add($"managed heap {Bytes(m.ManagedHeapBytes)}");

            var text = "Host process: " + string.Join(", ", readings);
            if (m.LimitBytes is not long limit) return text + "; no memory limit could be read.";

            text += $"; against {(m.LimitIsExplicit ? "a configured memory limit of" : "a visible memory ceiling of")} {Bytes(limit)}";
            if (Utilisation(m) is double used)
                text += $" ({used * 100:F0}% used)";
            return text + ".";
        }

        /// <summary>
        /// The one sentence that separates the two failures Shorokoo/Shorokoo#332 reports as
        /// identical. It is stated whenever the process is close to a limit it did not have to be
        /// close to, because that is the case where the accelerator is the wrong thing to blame.
        /// </summary>
        private static string? Advice(AllocationPool pool, DeviceFacts device, ProcessMemoryFacts m)
        {
            bool gpu = device.HasDeviceMemory;
            bool nearLimit = Utilisation(m) is double u && u >= 0.85;
            // "Room left on the card" is only meaningful against the request that was refused; a
            // tenth of the device is far more than any single arena block, so free space above that
            // means the refusal did not come from the card being out of memory.
            bool deviceHasRoom = device.Reading is DeviceMemoryReading d
                                 && d.TotalBytes > 0 && d.FreeBytes > d.TotalBytes / 10;

            if (gpu && deviceHasRoom && nearLimit)
                return "The device has room, yet this process is close to its own memory limit — and "
                     + "on Windows/WDDM a device allocation is backed by system commit, so a limit "
                     + "meant to bound HOST memory bounds DEVICE memory too, and an arena expansion "
                     + "past it fails with the same message a genuinely full accelerator produces. "
                     + "This is the limit, not the model: re-run with it raised or removed.";

            if (gpu && nearLimit)
                return "This process is close to its memory limit, and on Windows/WDDM a device "
                     + "allocation is backed by system commit — so a limit meant to bound HOST memory "
                     + "bounds DEVICE memory too, and an arena expansion past it fails with the same "
                     + "message a genuinely full accelerator produces. Rule the limit out first: "
                     + "re-run with it raised or removed. If the failure moves, it was the limit, not "
                     + "the model.";

            if (nearLimit)
                return "This process is close to its memory limit, so the limit — not the model's "
                     + "size — is the first thing to rule out: re-run with it raised or removed.";

            if (pool == AllocationPool.Device && deviceHasRoom)
                return "The device still reports free memory, so the refusal is not the card being "
                     + "out of memory: the arena could not extend by the block it wanted. Turn on "
                     + "RunSettings.ShrinkArenaAfterRun so the arena stops ratcheting, or cap it "
                     + "with DeviceMemorySettings.LimitBytes so it asks for less at a time.";

            if (pool == AllocationPool.Device)
                return "The process is well inside its host memory limit, so this is the accelerator "
                     + "itself running out: reduce batch size, sequence length or model size, or free "
                     + "device memory held by other processes.";

            return null;
        }

        private static double? Utilisation(ProcessMemoryFacts m)
        {
            if (m.LimitBytes is not long limit || limit <= 0) return null;
            long used = Math.Max(m.CommitBytes ?? 0, m.WorkingSetBytes ?? 0);
            return used <= 0 ? null : (double)used / limit;
        }

        private static long AddSat(long a, long b) => a > long.MaxValue - b ? long.MaxValue : a + b;

        internal static string Bytes(long bytes)
        {
            if (bytes < 0) return "unknown size";
            string[] units = ["B", "KiB", "MiB", "GiB", "TiB"];
            double v = bytes;
            int unit = 0;
            while (v >= 1024 && unit < units.Length - 1) { v /= 1024; unit++; }
            return unit == 0
                ? $"{bytes} B"
                : string.Format(CultureInfo.InvariantCulture, "{0:0.##} {1}", v, units[unit]);
        }
    }

    /// <summary>One tensor of the inventory a failing operation had resident.</summary>
    internal readonly record struct TensorInventoryEntry(string Name, string DType, long[] Shape, long Bytes);

    /// <summary>A named group of the inventory — trainable parameters, model state, optimizer state.</summary>
    internal readonly record struct TensorInventorySection(string Name, IReadOnlyList<TensorInventoryEntry> Tensors)
    {
        /// <summary>Total bytes across the section, saturating rather than wrapping.</summary>
        internal long TotalBytes
        {
            get
            {
                long total = 0;
                foreach (var t in Tensors)
                {
                    if (t.Bytes < 0) return -1;
                    total = total > long.MaxValue - t.Bytes ? long.MaxValue : total + t.Bytes;
                }
                return total;
            }
        }
    }
}

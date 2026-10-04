using Microsoft.ML.OnnxRuntime;
using Shorokoo.Core.Backends;

namespace Shorokoo.OnnxRuntime;

/// <summary>
/// Shorokoo's allocator for one device (<see cref="CachingAllocator"/>), as the native runtime this
/// backend binds holds it: a native allocator of Shorokoo's (<see cref="NativeAllocator"/>) over the
/// process's allocator for the device, describing itself with a memory info of this runtime's, and
/// registered with the runtime's environment for every session built to use the environment's
/// allocators (<c>session.use_env_allocators</c>) in place of the arena it would otherwise make.
///
/// <para>The allocator itself is the process's, whichever copy of this backend asks for it, so two
/// backends that bind one runtime — the program's own and one loaded in isolation over the same
/// file — share it, and their sessions each charge their own accounts. The runtime is handed it once
/// (<see cref="CachingAllocator.HandTo"/>): it keeps one allocator per device, and a second
/// registration would take the device over from the first. Where the runtime refuses it — something
/// else registered an allocator for the device first, say — making this fails, and nothing is kept:
/// every later request asks again and fails again, rather than building sessions that would allocate
/// in memory Shorokoo does not manage.</para>
///
/// <para>Held for the life of the process, as everything registered with the runtime is.</para>
/// </summary>
internal sealed class RuntimeAllocator
{
    /// <summary>The key the host's allocator is held under, beside the CUDA devices'.</summary>
    private const int Host = -1;

    private static readonly object _gate = new();
    private static readonly Dictionary<int, RuntimeAllocator> _byDevice = [];

    /// <summary>The host's allocator, handed to this runtime on first use.</summary>
    internal static RuntimeAllocator ForHost() => For(Host);

    /// <summary>CUDA device <paramref name="deviceId"/>'s allocator, handed to this runtime on first
    /// use.</summary>
    internal static RuntimeAllocator ForCard(int deviceId) => For(deviceId);

    private static RuntimeAllocator For(int device)
    {
        lock (_gate)
        {
            if (_byDevice.TryGetValue(device, out var existing)) return existing;
            return _byDevice[device] = new RuntimeAllocator(device);
        }
    }

    private readonly OrtMemoryInfo _info;

    private RuntimeAllocator(int device)
    {
        Shared = device == Host ? CachingAllocator.ForHost() : CachingAllocator.ForCard(device);
        _info = InfoFor(device);
        var native = NativeAllocator.Create(
            Shared.State, CachingAllocator.AllocateEntry, CachingAllocator.FreeEntry, OrtEnvironment.PointerOf(_info));
        Shared.HandTo(OrtEnvironment.EnvironmentHandle(), () => OrtEnvironment.Register(native));
        Managed = OrtEnvironment.Wrap(native);
    }

    /// <summary>The memory info this runtime matches the allocator to a device by: the host's, or
    /// CUDA device <paramref name="device"/>'s.</summary>
    private static OrtMemoryInfo InfoFor(int device)
    {
        if (device == Host)
            return new OrtMemoryInfo(OrtMemoryInfo.allocatorCPU, OrtAllocatorType.DeviceAllocator, 0, OrtMemType.Default);
        return new OrtMemoryInfo(OrtMemoryInfo.allocatorCUDA, OrtAllocatorType.DeviceAllocator, device, OrtMemType.Default);
    }

    /// <summary>The process's allocator for the device, which keeps the accounts.</summary>
    internal CachingAllocator Shared { get; }

    /// <summary>The allocator as this runtime's managed surface takes one, to make a tensor
    /// from.</summary>
    internal OrtAllocator Managed { get; }
}

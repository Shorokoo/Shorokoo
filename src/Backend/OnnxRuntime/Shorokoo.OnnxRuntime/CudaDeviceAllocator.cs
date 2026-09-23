using System.Globalization;
using Microsoft.ML.OnnxRuntime;
using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory.IR;

namespace Shorokoo.OnnxRuntime;

/// <summary>
/// ONNX Runtime's allocator for a CUDA device under one device-memory configuration: what an
/// allocation in that card's own memory has to come from, and the one thing a backend cannot
/// simply ask the runtime for.
///
/// <para>The obstacle is that <see cref="OrtAllocator"/> is created <i>from a session</i> — ORT
/// looks the device's allocator up in the session's registered execution providers — and a
/// backend may be asked for a tensor before any session exists. So this opens one of its own, the
/// smallest model ORT will load, and keeps it. The session is never run: it exists to register the
/// CUDA execution provider's allocator, which a session does whether or not any node was placed on
/// the card.</para>
///
/// <para>Both are then held for the life of the process, and that is the point rather than an
/// oversight. ORT's tensor keeps a <i>raw</i> pointer to the allocator it was made from and
/// frees itself through it, so an allocator released while any of its tensors is still alive turns
/// every later collection of one into a use-after-free; and the arena the allocator draws on
/// belongs to the session, so the session has to outlive the allocator in turn. A tensor's lifetime
/// is the caller's business and can be arbitrarily long, so the only lifetime that is certainly
/// long enough is the process's. Holding it here rather than on the backend matters for the same
/// reason: a backend is an ordinary object a program may drop while its tensors live on.</para>
///
/// <para><b>The key is the device and the <see cref="DeviceMemorySettings"/> together</b>, and
/// that is what puts a tensor a transfer places on the card inside the budget of the context that
/// placed it. The session is built with that context's settings, so its arena carries that
/// context's <c>gpu_mem_limit</c> and extend strategy, and a copy onto the card that does not fit
/// the ceiling fails the way a step past it does instead of quietly eating the rest of the device.
/// Which budget governs a tensor settles when it is allocated and never moves: handing the tensor to
/// a second context that can read it where it is copies nothing, so re-homing what it is charged to
/// is not something that could do.</para>
///
/// <para>The cost is one session per (device, configuration), built on the first tensor actually
/// allocated that way: a program that puts none on a card pays nothing, and one that leaves the
/// settings alone pays it once per card as it always did. Each costs the execution provider's
/// per-session state on that device plus an arena of its own, and none of them is ever released,
/// so the count is capped at <see cref="MaxConfigurationsPerDevice"/> per device and a further
/// configuration is refused rather than opened. A program that trips the cap is configuring its
/// device memory per call rather than per component; the fix is to share one settings instance
/// between the contexts that mean the same thing by it.</para>
///
/// <para>An isolated backend gets its own set, because it gets its own copy of this assembly along
/// with the native runtime it binds, which is exactly right: its allocators belong to its
/// runtime.</para>
/// </summary>
internal static class CudaDeviceAllocator
{
    /// <summary>The ONNX IR version and opset the placeholder model declares. Nothing depends on
    /// either beyond ORT accepting them, so they track what the framework emits elsewhere.</summary>
    private const int OnnxIrVersion = 10;
    private const int OnnxOpset = 21;

    /// <summary>
    /// How many device-memory configurations one card may hold an allocator for. Each is a session
    /// that lives until the process ends, so this is the ceiling on what a program can be made to
    /// hold by asking for transfers under settings it keeps varying.
    /// </summary>
    internal const int MaxConfigurationsPerDevice = 8;

    /// <summary>A device's allocator and the session it was taken from, which has to outlive
    /// it.</summary>
    private sealed record Binding(InferenceSession Session, OrtAllocator Allocator);

    /// <summary>A card and the arena settings tensors are placed on it under. The settings are a
    /// record, so two contexts asking for the same budget ask for the same allocator.</summary>
    private readonly record struct Placement(int DeviceId, DeviceMemorySettings Settings);

    private static readonly object _gate = new();
    private static readonly Dictionary<Placement, Binding> _byPlacement = [];

    /// <summary>
    /// The allocator for CUDA device <paramref name="deviceId"/> under
    /// <paramref name="deviceMemory"/>, built on first use through a session configured by
    /// <paramref name="configureExecutionProvider"/> — the very step the asking backend puts its
    /// own sessions on that card with.
    ///
    /// <para>Keyed on the device <i>and</i> the settings. A card's memory is the card's memory, so
    /// two backends on one card under one configuration are asking for the same allocator; two
    /// configurations are not, because the budget is a property of the arena and an arena belongs
    /// to the session that built it.</para>
    ///
    /// <para>The lock is held across the session build, which is slow. Deliberately: two threads
    /// racing here would otherwise each build a session, and the loser's would be dropped after
    /// having initialized the execution provider — the expensive half — for nothing.</para>
    /// </summary>
    /// <exception cref="InvalidOperationException">This card already holds
    /// <see cref="MaxConfigurationsPerDevice"/> configurations.</exception>
    /// <exception cref="OnnxRuntimeException">There is no such CUDA device, or this build of ONNX
    /// Runtime cannot load its CUDA execution provider.</exception>
    internal static OrtAllocator For(
        int deviceId,
        DeviceMemorySettings deviceMemory,
        Action<SessionOptions, DeviceMemorySettings> configureExecutionProvider)
    {
        var placement = PlacementOf(deviceId, deviceMemory);
        lock (_gate)
        {
            if (_byPlacement.TryGetValue(placement, out var existing)) return existing.Allocator;
            RefuseBeyondTheCap(placement);
            var bound = Bind(placement, configureExecutionProvider);
            _byPlacement[placement] = bound;
            return bound.Allocator;
        }
    }

    /// <summary>
    /// The allocator tensors placed on <paramref name="deviceId"/> under
    /// <paramref name="deviceMemory"/> come from, or <c>null</c> when nothing has been placed that
    /// way yet.
    ///
    /// <para>It never builds one, which is what makes it safe to ask from a reporting path: a
    /// context that has moved nothing onto the card has no arena to report, and opening a session
    /// to say so would cost the very thing the figures are being read to watch.</para>
    /// </summary>
    internal static OrtAllocator? Existing(int deviceId, DeviceMemorySettings deviceMemory)
    {
        var placement = PlacementOf(deviceId, deviceMemory);
        lock (_gate)
            return _byPlacement.TryGetValue(placement, out var existing) ? existing.Allocator : null;
    }

    /// <summary>
    /// The cache key. The settings are resolved first, because <see cref="ArenaExtendStrategy.Auto"/>
    /// is Shorokoo's choice between ORT's two strategies rather than one of them: keying on the
    /// unresolved value would open a second session for a configuration identical to one already
    /// held, and <c>AppendCuda</c> refuses <c>Auto</c> anyway.
    /// </summary>
    private static Placement PlacementOf(int deviceId, DeviceMemorySettings deviceMemory)
    {
        ArgumentNullException.ThrowIfNull(deviceMemory);
        return new Placement(deviceId, deviceMemory.Resolve(reusedAcrossShapes: false));
    }

    /// <summary>
    /// Refuses a configuration past the cap, under <c>_gate</c>. Counted rather than tracked,
    /// because the count is at most <see cref="MaxConfigurationsPerDevice"/> per device and this
    /// runs once per configuration ever opened.
    /// </summary>
    private static void RefuseBeyondTheCap(Placement placement)
    {
        var held = 0;
        foreach (var key in _byPlacement.Keys)
            if (key.DeviceId == placement.DeviceId) held++;
        if (CapacityRefusal(placement.DeviceId, held, placement.Settings) is { } refusal)
            throw refusal;
    }

    /// <summary>
    /// The refusal a card holding <paramref name="held"/> configurations answers a further one
    /// with, or <c>null</c> while there is room. Pure, and separated from the cache for the reason
    /// <see cref="OrtBackend.CudaProviderOptions"/> is: the rule can then be read and asserted
    /// without a CUDA machine to fill a cache on.
    /// </summary>
    internal static InvalidOperationException? CapacityRefusal(
        int deviceId, int held, DeviceMemorySettings settings)
    {
        if (held < MaxConfigurationsPerDevice) return null;
        return new InvalidOperationException(
            $"Placing a tensor on CUDA device {deviceId.ToString(CultureInfo.InvariantCulture)} "
            + "under these device-memory settings (limit "
            + $"{settings.LimitBytes?.ToString(CultureInfo.InvariantCulture) ?? "none"}, "
            + $"{settings.ArenaExtend}) needs an ONNX Runtime session of its own to allocate from, "
            + $"and this device already holds {held.ToString(CultureInfo.InvariantCulture)} such "
            + "sessions -- the most it may. A session that has served a tensor can never be "
            + "released, because the tensor frees itself through it, so each one is held until the "
            + "process ends. What costs a session is a distinct pair of values, not a distinct "
            + "object: settings equal by value already share one, so building a fresh "
            + "DeviceMemorySettings per call is free and only varying the limit is not. Round the "
            + "budget to a few fixed sizes, or size it once per component rather than per model.");
    }

    private static Binding Bind(
        Placement placement, Action<SessionOptions, DeviceMemorySettings> configureExecutionProvider)
    {
        // The `using` is the same load-bearing one as in OrtBackend.CreateSession: ORT takes
        // the options as a bare IntPtr and does no ref-counting, so a plain local is collectible --
        // and its critical finalizer free-able -- while the session constructor is still reading it.
        using var options = new SessionOptions();
        // The settings of the context whose tensors this session will allocate, already resolved
        // by PlacementOf because AppendCuda refuses ArenaExtendStrategy.Auto. They are what the
        // arena is built with, and so what bounds every tensor taken out of it.
        configureExecutionProvider(options, placement.Settings);
        var session = new InferenceSession(MinimalModel(), options);
        try
        {
            using var info = new OrtMemoryInfo(
                OrtMemoryInfo.allocatorCUDA, OrtAllocatorType.DeviceAllocator, placement.DeviceId,
                OrtMemType.Default);
            // ORT reads the memory info to find the allocator and keeps nothing pointing at it --
            // the allocator reports its own -- so disposing it here is safe and a session's worth
            // of native handle is what it saves.
            return new Binding(session, new OrtAllocator(session, info));
        }
        catch
        {
            // Nothing has the session yet, and an undisposed one holds the execution provider's
            // whole context on the card.
            session.Dispose();
            throw;
        }
    }

    /// <summary>
    /// The smallest model ORT will open: one <c>Identity</c> over a one-element float vector.
    /// Neither its input nor its output is ever bound, because the session is never run — it is
    /// the registration of the execution provider's allocators that is wanted, and that happens
    /// when the session is built.
    /// </summary>
    private static byte[] MinimalModel()
    {
        static TypeProto FloatVector() => new()
        {
            TensorType = new TypeProto.Tensor
            {
                ElemType = (int)TensorProto.DataType.Float,
                Shape = new TensorShapeProto
                {
                    Dims = { new TensorShapeProto.Dimension { DimValue = 1 } },
                },
            },
        };

        var graph = new GraphProto { Name = "shorokoo_device_allocator" };
        graph.Nodes.Add(new NodeProto { OpType = "Identity", Inputs = { "x" }, Outputs = { "y" } });
        graph.Inputs.Add(new ValueInfoProto { Name = "x", Type = FloatVector() });
        graph.Outputs.Add(new ValueInfoProto { Name = "y", Type = FloatVector() });

        var model = new ModelProto { IrVersion = OnnxIrVersion, Graph = graph };
        model.OpsetImports.Add(new OperatorSetIdProto { Domain = "", Version = OnnxOpset });

        using var serialized = new MemoryStream();
        ProtoBuf.Serializer.Serialize(serialized, model);
        return serialized.ToArray();
    }
}

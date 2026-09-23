using Microsoft.ML.OnnxRuntime;
using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory.IR;

namespace Shorokoo.OnnxRuntime;

/// <summary>
/// ONNX Runtime's allocator for a CUDA device: what an allocation in that card's own memory has
/// to come from, and the one thing a backend cannot simply ask the runtime for.
///
/// <para>The obstacle is that <see cref="OrtAllocator"/> is created <i>from a session</i> — ORT
/// looks the device's allocator up in the session's registered execution providers — and a
/// backend may be asked for a tensor before any session exists. So this opens one of its own, the
/// smallest model ORT will load, and keeps it. The session is never run: it exists to register the
/// CUDA execution provider's allocator, which a session does whether or not any node was placed on
/// the card.</para>
///
/// <para><b>The allocator is held for the life of the process, and that is the point rather than
/// an oversight.</b> An ORT tensor frees itself through the allocator that made it, so the
/// allocator has to outlive every tensor it ever served: measured, releasing a tensor after its
/// allocator has been released takes the process down. A tensor's lifetime is the caller's
/// business and can be arbitrarily long, so the only lifetime that is certainly long enough is the
/// process's. Holding it here rather than on the backend matters for the same reason: a backend is
/// an ordinary object a program may drop while its tensors live on.</para>
///
/// <para>The session it was taken from does <i>not</i> have to outlive it — measured too: a tensor
/// allocated through it survives the session's disposal, the allocator keeping the arena it draws
/// on alive by itself. The session is held beside it all the same. Releasing it would give back one
/// idle session per card and buy nothing else, and holding both is the arrangement every release on
/// this path has been observed under.</para>
///
/// <para><b>One per CUDA device</b>, built on the first tensor actually allocated there, so a
/// program that never puts one on a card never pays for it. Every compute context on the card
/// shares it whatever its <see cref="DeviceMemorySettings"/>: its arena has no ceiling of its own,
/// and a context's device-memory budget is not enforced here but by the context's own accounting of
/// the tensors attached to it, which refuses a transfer that would take it past its budget before
/// this is ever asked for the memory. So there is one arena per card to hold, and nothing to
/// multiply. Nothing ever runs this session, so nothing asks its arena to shrink: it keeps the
/// blocks it has extended by, and holds the most it has ever been asked for at once.</para>
///
/// <para>An isolated backend gets its own, because it gets its own copy of this assembly along
/// with the native runtime it binds, which is exactly right: its allocator belongs to its
/// runtime.</para>
/// </summary>
internal static class CudaDeviceAllocator
{
    /// <summary>The ONNX IR version and opset the placeholder model declares. Nothing depends on
    /// either beyond ORT accepting them, so they track what the framework emits elsewhere.</summary>
    private const int OnnxIrVersion = 10;
    private const int OnnxOpset = 21;

    /// <summary>A device's allocator and the session it was taken from, which is held beside
    /// it.</summary>
    private sealed record Binding(InferenceSession Session, OrtAllocator Allocator);

    private static readonly object _gate = new();
    private static readonly Dictionary<int, Binding> _byDevice = [];

    /// <summary>
    /// The allocator for CUDA device <paramref name="deviceId"/>, built on first use through a
    /// session configured by <paramref name="configureExecutionProvider"/> — the very step the
    /// asking backend puts its own sessions on that card with.
    ///
    /// <para>Keyed on the device alone: a card's memory is the card's memory, so a second backend
    /// on the same one is asking for the same allocator, and the provider options that differ
    /// between them would only have configured a second arena over the same device.</para>
    ///
    /// <para>The lock is held across the session build, which is slow. Deliberately: two threads
    /// racing here would otherwise each build a session, and the loser's would be dropped after
    /// having initialized the execution provider — the expensive half — for nothing.</para>
    /// </summary>
    /// <exception cref="OnnxRuntimeException">There is no such CUDA device, or this build of ONNX
    /// Runtime cannot load its CUDA execution provider.</exception>
    internal static OrtAllocator For(
        int deviceId, Action<SessionOptions, DeviceMemorySettings> configureExecutionProvider)
    {
        lock (_gate)
        {
            if (_byDevice.TryGetValue(deviceId, out var existing)) return existing.Allocator;
            var bound = Bind(deviceId, configureExecutionProvider);
            _byDevice[deviceId] = bound;
            return bound.Allocator;
        }
    }

    private static Binding Bind(
        int deviceId, Action<SessionOptions, DeviceMemorySettings> configureExecutionProvider)
    {
        // The `using` is the same load-bearing one as in OrtBackend.CreateSession: ORT takes
        // the options as a bare IntPtr and does no ref-counting, so a plain local is collectible --
        // and its critical finalizer free-able -- while the session constructor is still reading it.
        using var options = new SessionOptions();
        // The shipped settings, and no caller's: this session is never run, and the allocator it
        // registers is the device's rather than any one context's -- the same reason the cache is
        // keyed on the device alone. Resolved because AppendCuda refuses ArenaExtendStrategy.Auto.
        configureExecutionProvider(options, DeviceMemorySettings.Default.Resolve(reusedAcrossShapes: false));
        var session = new InferenceSession(MinimalModel(), options);
        try
        {
            using var info = new OrtMemoryInfo(
                OrtMemoryInfo.allocatorCUDA, OrtAllocatorType.DeviceAllocator, deviceId,
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

using Microsoft.ML.OnnxRuntime;
using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory.IR;

namespace Shorokoo.OnnxRuntime;

/// <summary>
/// ONNX Runtime's allocators for memory of its own outside every session, one per device: a CUDA
/// card's, which every tensor placed in that card's memory and every output a run hands back there
/// comes from, and the host's, which every output a run hands back in host memory comes from.
///
/// <para>A card's is the one thing a backend cannot simply ask the runtime for. The obstacle is that
/// <see cref="OrtAllocator"/> is created <i>from a session</i> — ORT looks the device's allocator up
/// in the session's registered execution providers — and a backend may be asked for a tensor before
/// any session exists. So this opens a session of its own per device, over the smallest model ORT
/// will load, and keeps it. It exists to register the device's allocator — the CUDA execution
/// provider's on a card, which a session does whether or not any node was placed there, and the CPU
/// provider's arena on the host — and is run only to have that allocator hand back what it holds
/// unused (<see cref="Shrink"/>).</para>
///
/// <para>The host has its own for a run's outputs rather than ONNX Runtime's default allocator, which
/// hands each tensor fresh memory from the operating system and gives it back as the tensor goes.
/// Measured on Windows: making, filling and releasing a 1 MiB output took 278 microseconds from the
/// default allocator and 42 from an arena, whose blocks stay warm from one run to the next — an
/// inference loop's every run, on the CPU, would otherwise pay for its outputs' pages again.</para>
///
/// <para><b>Each allocator is held for the life of the process, and that is the point rather than
/// an oversight.</b> An ORT tensor frees itself through the allocator that made it, so the
/// allocator has to outlive every tensor it ever served: measured, releasing a tensor after its
/// allocator has been released takes the process down. A tensor's lifetime is the caller's
/// business and can be arbitrarily long, so the only lifetime that is certainly long enough is the
/// process's. Holding it here rather than on the backend matters for the same reason: a backend is
/// an ordinary object a program may drop while its tensors live on.</para>
///
/// <para>The session it was taken from does <i>not</i> have to outlive it — measured too: a tensor
/// allocated through it survives the session's disposal, the allocator keeping the arena it draws
/// on alive by itself. The session is held beside it all the same, since it is what shrinks
/// it.</para>
///
/// <para><b>One per device</b>, built on the first tensor actually allocated there, so a program
/// that never puts one on a card never pays for it. Every compute context on the device shares it
/// whatever its <see cref="DeviceMemorySettings"/>: its arena has no ceiling of its own, and a
/// context's device-memory budget is not enforced here but by the context's own accounting of the
/// tensors attached to it, which refuses a transfer that would take it past its budget before this
/// is ever asked for the memory. So there is one arena per device to hold, and nothing to multiply.
/// It grows to the most of its tensors ever alive at once, and keeps the blocks it grew by until it
/// is asked to hand back those no tensor is using (<see cref="Shrink"/>), which a run asked to
/// shrink its own arena does as it starts.</para>
///
/// <para>An isolated backend gets its own, because it gets its own copy of this assembly along
/// with the native runtime it binds, which is exactly right: its allocators belong to its
/// runtime.</para>
/// </summary>
internal static class RuntimeAllocators
{
    /// <summary>The ONNX IR version and opset the placeholder model declares. Nothing depends on
    /// either beyond ORT accepting them, so they track what the framework emits elsewhere.</summary>
    private const int OnnxIrVersion = 10;
    private const int OnnxOpset = 21;

    /// <summary>The placeholder model's one input, which is its one output too.</summary>
    private const string MinimalInput = "x";

    /// <summary>The key the host's allocator is held under, beside the CUDA devices'.</summary>
    private const int Host = -1;

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
    internal static OrtAllocator ForCard(
        int deviceId, Action<SessionOptions, DeviceMemorySettings> configureExecutionProvider)
        => For(deviceId, options =>
        {
            // The shipped settings, and no caller's: this session runs nothing but its own
            // shrinking, and the allocator it registers is the device's rather than any one
            // context's -- the same reason the cache is keyed on the device alone. Resolved because
            // AppendCuda refuses ArenaExtendStrategy.Auto.
            configureExecutionProvider(options, DeviceMemorySettings.Default.Resolve(reusedAcrossShapes: false));
        });

    /// <summary>The allocator for the host, an arena of the CPU provider's, built on first
    /// use.</summary>
    internal static OrtAllocator ForHost() => For(Host, static _ => { });

    private static OrtAllocator For(int device, Action<SessionOptions> configure)
    {
        lock (_gate)
        {
            if (_byDevice.TryGetValue(device, out var existing)) return existing.Allocator;
            var bound = Bind(device, configure);
            _byDevice[device] = bound;
            return bound.Allocator;
        }
    }

    /// <summary>
    /// Has the allocator of CUDA device <paramref name="cudaDeviceId"/>, or the host's where it is
    /// null, hand back every block of its arena no tensor is using, where the allocator has been
    /// built; nothing otherwise. <paramref name="cancellation"/> is the asking run's: one stopped
    /// as it shrinks leaves the blocks for the next.
    ///
    /// <para>ONNX Runtime hands an arena's blocks back only at the end of a run of the session that
    /// owns it, when the run asks (<c>memory.enable_memory_arena_shrinkage</c>), and the allocator it
    /// hands out has no way to be asked directly — its <c>Shrink</c> entry, ONNX Runtime 1.25's, is
    /// empty for a provider's arena. So the session the allocator was taken from is run, once, over a
    /// model with no node, which hands its input straight back: the run allocates nothing, so nothing
    /// of its own is in the arena as the arena shrinks. Measured on a card: eight released 32 MiB
    /// tensors had grown the arena by 256 MiB, and the run left it holding nothing; a four-byte
    /// tensor carved out of one of their blocks keeps that whole block until it goes. The CPU
    /// provider's arena keeps the first block it grew by: grown the same way, it went back to its
    /// first 32 MiB. A run costs some 8 microseconds on a card and 6 on the host, so this is done
    /// when asked for and not as each tensor goes.</para>
    /// </summary>
    internal static void Shrink(int? cudaDeviceId, CancellationToken cancellation)
    {
        Binding? bound;
        lock (_gate) _byDevice.TryGetValue(cudaDeviceId ?? Host, out bound);
        if (bound is null) return;
        using var input = OrtValue.CreateTensorValueFromMemory(new float[1], [1L]);
        using var runOptions = new RunOptions();
        runOptions.AddRunConfigEntry(
            "memory.enable_memory_arena_shrinkage",
            OrtBackend.ArenaShrinkageRunConfig(cudaDeviceId, shrinkArenaAfterRun: true)!);
        using var abort = OrtSession.AbortWhenCancelled(runOptions, cancellation);
        try
        {
            using var outputs = bound.Session.Run(
                runOptions, new Dictionary<string, OrtValue> { [MinimalInput] = input }, [MinimalInput]);
        }
        // Stopped with the run that asked for it, which is stopped before it starts: the blocks wait
        // for the next run that asks.
        catch (OnnxRuntimeException) when (cancellation.IsCancellationRequested) { }
    }

    private static Binding Bind(int device, Action<SessionOptions> configure)
    {
        // The `using` is the same load-bearing one as in OrtBackend.CreateSession: ORT takes
        // the options as a bare IntPtr and does no ref-counting, so a plain local is collectible --
        // and its critical finalizer free-able -- while the session constructor is still reading it.
        using var options = new SessionOptions();
        configure(options);
        var session = new InferenceSession(MinimalModel(), options);
        try
        {
            var host = device == Host;
            using var info = new OrtMemoryInfo(
                host ? OrtMemoryInfo.allocatorCPU : OrtMemoryInfo.allocatorCUDA,
                host ? OrtAllocatorType.ArenaAllocator : OrtAllocatorType.DeviceAllocator,
                host ? 0 : device, OrtMemType.Default);
            // ORT reads the memory info to find the allocator and keeps nothing pointing at it --
            // the allocator reports its own -- so disposing it here is safe and a session's worth
            // of native handle is what it saves.
            return new Binding(session, new OrtAllocator(session, info));
        }
        catch
        {
            // Nothing has the session yet, and an undisposed one holds the execution provider's
            // whole context on its device.
            session.Dispose();
            throw;
        }
    }

    /// <summary>
    /// The smallest model ORT will open: no node, and a one-element float vector that is both its
    /// input and its output. It is the registration of the execution provider's allocators that is
    /// wanted, which happens when the session is built whether or not any node is there; the session
    /// is run only to have its arena shrink (<see cref="Shrink"/>), which a run that computes nothing
    /// does without putting anything in it.
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
        graph.Inputs.Add(new ValueInfoProto { Name = MinimalInput, Type = FloatVector() });
        graph.Outputs.Add(new ValueInfoProto { Name = MinimalInput, Type = FloatVector() });

        var model = new ModelProto { IrVersion = OnnxIrVersion, Graph = graph };
        model.OpsetImports.Add(new OperatorSetIdProto { Domain = "", Version = OnnxOpset });

        using var serialized = new MemoryStream();
        ProtoBuf.Serializer.Serialize(serialized, model);
        return serialized.ToArray();
    }
}

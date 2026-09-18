using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using Shorokoo.Core.Utils;
using Shorokoo.Runtime;

namespace Shorokoo.Core.Inference.Abstractions;

/// <summary>
/// The memory a set of backends allocates in — one object per <see cref="MemorySpace"/>, and the
/// top of the four levels a tensor hangs from:
///
/// <code>
/// MemoryDevice        one per memory space
///   └─ backend        a runtime; allocates in exactly one MemoryDevice
///        └─ ComputeContext   compiles and runs
///             └─ TensorData  a handle on an allocation, attached to one context
/// </code>
///
/// <para><b>Interned by space.</b> <see cref="For"/> returns the same object for the same
/// <see cref="MemorySpace"/>, so the two CPU backends of a side-by-side test share one device, and
/// so do two isolated CUDA backends over device 0. That is what the <i>Memory</i> in the name is
/// for: it is the memory, not the processor. Two execution providers with no memory of their own
/// both land on the host device and can hand each other buffers without copying, which is already
/// what equal memory spaces mean.</para>
///
/// <para><b>The bookkeeping is the framework's.</b> A backend is implemented once per platform
/// package, so every member added to <see cref="IShorokooInferenceBackend"/> is a break for
/// out-of-tree backends. The device is therefore not something a backend reports: it is derived
/// from the <see cref="IShorokooInferenceBackend.MemorySpace"/> the interface already carries, and
/// the list of backends on a device is kept here. It is weak, so a backend the program has dropped
/// is not held alive by having been asked which device it allocates in.</para>
/// </summary>
public sealed class MemoryDevice
{
    // Interning table. Never pruned, and does not need to be: there is one entry per memory space
    // a process has touched -- the host, a card, the unnamed one -- not one per backend.
    //
    // Concurrent rather than a Dictionary behind a monitor, because every allocation asks which
    // device it is in as it is built: a process-wide lock here would serialize tensor construction
    // across the whole program, on the very two-device workload this layering is for. GetAdd may
    // build a device twice under a race and keeps only one, which is all reference identity needs.
    private static readonly ConcurrentDictionary<MemorySpace, MemoryDevice> Interned = new();

    private readonly WeakSet<IShorokooInferenceBackend> _backends = new();

    private MemoryDevice(MemorySpace space) => Space = space;

    /// <summary>The memory this device is. Two backends reporting this space allocate
    /// here.</summary>
    public MemorySpace Space { get; }

    /// <summary>
    /// The one device for <paramref name="space"/>. Calling it twice with the same space returns
    /// the same object, which is what makes reference equality a usable test for "the same
    /// memory".
    /// </summary>
    public static MemoryDevice For(MemorySpace space)
        => Interned.GetOrAdd(space, static s => new MemoryDevice(s));

    /// <summary>
    /// The device <paramref name="backend"/> allocates in, recording it there as one of that
    /// device's backends.
    /// </summary>
    /// <exception cref="ArgumentNullException"><paramref name="backend"/> is null.</exception>
    public static MemoryDevice Of(IShorokooInferenceBackend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);
        var device = For(backend.MemorySpace);
        device._backends.Add(backend);
        return device;
    }

    /// <summary>
    /// The backends known to allocate here and not yet collected — every one that has been asked
    /// for its device. A snapshot: a backend dropped after this returns stays in the list.
    /// </summary>
    public IReadOnlyList<IShorokooInferenceBackend> Backends => _backends.Snapshot();

    /// <inheritdoc/>
    public override string ToString() => Space.ToString();
}

/// <summary>
/// The contexts a backend compiles and runs for — the level between
/// <see cref="MemoryDevice"/> and <see cref="ComputeContext"/>, kept here rather than on the
/// backend.
///
/// <para>An extension method rather than an interface member, and deliberately: a backend is
/// implemented once per platform package, so a member added to
/// <see cref="IShorokooInferenceBackend"/> breaks every out-of-tree implementation. The register
/// lives on this side of the boundary instead, and a backend that knows nothing about it is
/// listed all the same.</para>
/// </summary>
public static class BackendRegistry
{
    // Backend -> the contexts compiling and running on it. The outer table is keyed weakly on the
    // backend, so a dropped backend takes its context list with it; the inner set is weak too, so
    // a context the program has let go of does not live on because a backend was once asked for
    // its contexts.
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<
        IShorokooInferenceBackend, WeakSet<ComputeContext>> Contexts = new();

    /// <summary>
    /// The compute contexts running on <paramref name="backend"/> and not yet collected. A
    /// snapshot, for the reason <see cref="MemoryDevice.Backends"/> is one.
    /// </summary>
    /// <exception cref="ArgumentNullException"><paramref name="backend"/> is null.</exception>
    public static IReadOnlyList<ComputeContext> ContextsOn(this IShorokooInferenceBackend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);
        return Contexts.TryGetValue(backend, out var live) ? live.Snapshot() : [];
    }

    /// <summary>Records that <paramref name="context"/> runs on
    /// <paramref name="backend"/>.</summary>
    internal static void Attach(IShorokooInferenceBackend backend, ComputeContext context)
        => Contexts.GetValue(backend, static _ => new WeakSet<ComputeContext>()).Add(context);
}

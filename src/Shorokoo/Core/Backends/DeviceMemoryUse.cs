namespace Shorokoo.Core.Backends;

/// <summary>
/// What a compute context has attached to it in its own memory, against its device-memory budget —
/// <see cref="Shorokoo.Runtime.ComputeContext.ReadDeviceMemoryUse"/>.
///
/// <para>The budget counts tensors, not arenas. <see cref="AttachedBytes"/> is the bytes of the
/// live tensors on the context's books that are in its memory: what <c>To</c>, <c>CopyTo</c> and
/// <c>AllocateUninitialized</c> placed for it, what its runs read there or copied there to read,
/// and the outputs its runs left there. A tensor attached to two contexts counts on both books; one
/// that dies, is collected, or is detached with <c>Detach</c> leaves them. What the arenas holding
/// those tensors have taken from the device besides — the blocks they keep spare — is not in it:
/// read <see cref="Shorokoo.Runtime.CompiledGraph.ReadArenaStatistics"/> for a session's, and
/// <see cref="DeviceMemory"/> for the card.</para>
/// </summary>
/// <param name="AttachedBytes">The bytes of the live tensors attached to the context that are in its
/// memory — the card's, on a GPU backend.</param>
/// <param name="AttachedTensors">How many tensors those are.</param>
/// <param name="LimitBytes">The context's budget: its
/// <see cref="DeviceMemorySettings.LimitBytes"/> where its memory is a device's, and <c>null</c>
/// where no budget is in force — none was set, or the context's memory is the host's, which a
/// device-memory budget does not govern.</param>
public readonly record struct DeviceMemoryUse(long AttachedBytes, int AttachedTensors, long? LimitBytes)
{
    /// <summary>
    /// What the budget has left: <see cref="LimitBytes"/> less <see cref="AttachedBytes"/>, or
    /// <c>null</c> where there is no budget. This is the most a transfer onto the context can
    /// place, and — less what a run holds there itself — what the arena a run computes in may take.
    /// </summary>
    public long? AvailableBytes => LimitBytes is { } limit ? limit - AttachedBytes : null;
}

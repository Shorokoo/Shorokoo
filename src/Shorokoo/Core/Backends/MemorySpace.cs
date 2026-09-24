namespace Shorokoo.Core.Backends;

/// <summary>What kind of memory a tensor's bytes are in.</summary>
public enum MemoryKind
{
    /// <summary>Ordinary host memory, readable by the CPU.</summary>
    Host = 0,

    /// <summary>An NVIDIA device's own memory, reachable only through CUDA.</summary>
    Cuda = 1,

    /// <summary>
    /// Somewhere an execution provider kept it, and no record of where. It is certainly not host
    /// memory, and nothing knows which device it is on, so no transfer can reason about it — two
    /// such tensors compare equal as spaces without being in the same place.
    ///
    /// <para>Nothing the framework runs produces one on an execution provider it knows: every
    /// session output, every element read out of a sequence and every copy records the backend that
    /// allocated it, and that backend names the device. It is what remains for a backend on a
    /// provider Shorokoo has no name for, and for a caller that wraps a runtime value of its own
    /// without saying which backend made it (<c>TensorData.Create(shape, dtype, value)</c>) when
    /// that value turns out not to be host-readable — the honest answer where there is no producer
    /// to ask.</para>
    /// </summary>
    Unknown = 2,
}

/// <summary>
/// Where a tensor's bytes live, precisely enough to say whether two of them are in the same
/// place: a kind, and the device within that kind.
///
/// <para>It is the first of the two questions a transfer asks: bytes in different spaces have to
/// be copied, and a transfer between a host context and a device one is a real copy across the
/// bus. Two CUDA contexts on device 0 are the same space, and the device pointer really is valid
/// across them even when they are two isolated backends over two separate native ONNX Runtimes —
/// both reach the device through its primary context.</para>
///
/// <para>The same space is not by itself enough to read a tensor in place, though, because what a
/// session is handed is a <i>runtime value</i> rather than an address, and a session recognises its
/// own by type. So the other half of where a tensor is is which runtime allocated it — the two
/// together are a <see cref="MemoryLocation"/> — and whether a backend can address one is asked of
/// that backend (<see cref="IShorokooBackend.CanAddress"/>).</para>
/// </summary>
public readonly record struct MemorySpace(MemoryKind Kind, int DeviceId)
{
    /// <summary>Ordinary host memory — where every tensor built from a C# array lives.</summary>
    public static MemorySpace Host { get; } = new(MemoryKind.Host, 0);

    /// <summary>An NVIDIA device's memory.</summary>
    public static MemorySpace Cuda(int deviceId) => new(MemoryKind.Cuda, deviceId);

    /// <summary>A device allocation whose device was never recorded.</summary>
    public static MemorySpace UnknownDevice { get; } = new(MemoryKind.Unknown, -1);

    /// <summary>Whether the CPU can read these bytes.</summary>
    public bool IsHost => Kind == MemoryKind.Host;

    /// <summary>Whether this names a place at all.</summary>
    public bool IsKnown => Kind != MemoryKind.Unknown;

    /// <inheritdoc/>
    public override string ToString() => Kind switch
    {
        MemoryKind.Host => "host memory",
        MemoryKind.Cuda => $"CUDA device {DeviceId} memory",
        MemoryKind.Unknown => "an execution provider's own memory, on an unrecorded device",
        _ => $"{Kind} memory on device {DeviceId}",
    };
}

/// <summary>
/// Where a tensor's memory is, completely enough to say whether a backend can read it as it
/// stands: the memory itself, and the runtime whose allocation it is.
///
/// <para>Both halves are needed. Two CUDA backends on one card share a <see cref="MemorySpace"/>,
/// but an allocation is meaningful only to the runtime that made it: two backends over one loaded
/// ONNX Runtime can hand each other a card allocation, and two isolated runtimes on the same card
/// cannot, and copy through the host. <see cref="Runtime"/> is what tells those apart — it is the
/// allocating backend's <see cref="IShorokooBackend.RuntimeIdentity"/>, compared by reference.</para>
///
/// <para>The framework's own managed host memory — the arrays behind every literal — is the one
/// allocation every backend whose memory is the host's can read, and its runtime is
/// <see cref="HostBackend.Instance"/>'s.</para>
/// </summary>
/// <param name="Space">The memory the bytes are in.</param>
/// <param name="Runtime">The runtime whose allocation they are: the allocating backend's
/// <see cref="IShorokooBackend.RuntimeIdentity"/>.</param>
public readonly record struct MemoryLocation(MemorySpace Space, object Runtime)
{
    /// <summary>Whether this is the framework's own managed host memory — a <c>byte[]</c> or
    /// <c>string[]</c> the garbage collector owns, which any host-memory backend can read.</summary>
    public bool IsManaged => ReferenceEquals(Runtime, HostBackend.Instance.RuntimeIdentity);

    /// <summary>Whether <paramref name="other"/> is the same memory of the same runtime — the
    /// runtime compared by reference, so two runtime identities that happen to be equal by value
    /// are two runtimes still.</summary>
    public bool Equals(MemoryLocation other) => Space == other.Space && ReferenceEquals(Runtime, other.Runtime);

    /// <inheritdoc/>
    public override int GetHashCode()
        => HashCode.Combine(Space, System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(Runtime));

    /// <inheritdoc/>
    public override string ToString() => IsManaged ? "managed host memory" : Space.ToString();
}

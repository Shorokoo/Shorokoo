namespace Shorokoo.Core.Inference.Abstractions;

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
    /// <para>Nothing the framework runs produces one: every session output, every element read out
    /// of a sequence and every transfer result carries the context whose memory it is in, and that
    /// context names the device. It is what remains for a caller that wraps a runtime value of its
    /// own (<c>TensorData.Create(shape, dtype, value)</c> and the other context-free factories)
    /// and that value turns out not to be host-readable — the honest answer where there is no
    /// producer to ask.</para>
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
/// <para>The same space is not by itself enough to re-wrap, though, because a re-wrap hands over
/// a <i>runtime value</i> rather than an address, and a session recognises its own by type. So a
/// transfer also asks whether the two contexts share a backend, and two isolated ones over the
/// same card do not; <c>TensorData.CanShareWith</c> is where the two questions meet.</para>
/// </summary>
public readonly record struct MemorySpace(MemoryKind Kind, int DeviceId)
{
    /// <summary>Ordinary host memory — where a tensor with no compute context always lives.</summary>
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

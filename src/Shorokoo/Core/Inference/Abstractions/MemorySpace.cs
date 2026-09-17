namespace Shorokoo.Core.Inference.Abstractions;

/// <summary>What kind of memory a tensor's bytes are in.</summary>
public enum MemoryKind
{
    /// <summary>Ordinary host memory, readable by the CPU.</summary>
    Host = 0,

    /// <summary>An NVIDIA device's own memory, reachable only through CUDA.</summary>
    Cuda = 1,

    /// <summary>
    /// Somewhere an execution provider kept it, and no record of where. A value that came back
    /// from a session without the context that produced it is in this state: it is certainly not
    /// host memory, and nothing knows which device it is on, so no transfer can reason about it.
    /// Transitional — binding session outputs to their producing context is what removes it.
    /// </summary>
    Unknown = 2,
}

/// <summary>
/// Where a tensor's bytes live, precisely enough to say whether two of them are in the same
/// place: a kind, and the device within that kind.
///
/// <para>This — and not which backend allocated it — is what decides whether moving a tensor
/// between two <c>ComputeContext</c>s has to copy anything. Two CUDA contexts on device 0 share
/// a space even when they are two isolated backends over two separate native ONNX Runtimes,
/// because a CUDA device pointer is valid across them: both reach the device through its primary
/// context. So a transfer between them re-wraps and moves no bytes, while a transfer between a
/// host context and either of them is a real copy across the bus.</para>
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

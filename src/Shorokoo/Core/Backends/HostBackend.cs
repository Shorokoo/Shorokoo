using System;
using System.Collections.Generic;

namespace Shorokoo.Core.Backends;

/// <summary>
/// The framework's own host memory, as a backend: the allocating backend of every tensor built
/// from a C# array, and what <see cref="Shorokoo.Runtime.ComputeContext.Host"/> names.
///
/// <para>It holds tensors and builds nothing. Every member that would build something throws, and
/// that is the whole of it: there is no runtime here to build a session with, and no runtime value
/// to build either, because the framework's host memory is a managed <c>byte[]</c> that becomes a
/// runtime value only when a real backend is fed it. Manufacturing values here would put back the
/// thing the backend-free literal exists to remove — a native allocation behind every tensor a
/// graph is described with — so this backend is a home for tensors, not a factory for values.</para>
///
/// <para>Attached to <see cref="MemoryDevice.For"/> of <see cref="MemorySpace.Host"/>, alongside
/// every CPU backend the process loads: they share the memory, which is what lets <c>To</c> hand a
/// tensor built from a C# array to a CPU context as it stands. A run on that context still reads it
/// through a copy its runtime builds, since a session is handed runtime values only.</para>
/// </summary>
public sealed class HostBackend : IShorokooBackend
{
    /// <summary>The one host backend. There is only ever one, because there is only one
    /// framework-owned host memory.</summary>
    public static HostBackend Instance { get; } = Attached();

    private HostBackend()
    {
    }

    private static HostBackend Attached()
    {
        var backend = new HostBackend();
        MemoryDevice.Of(backend);
        return backend;
    }

    /// <inheritdoc/>
    public BackendDescription Description { get; }
        = new("Shorokoo.HostMemory", ComputeDevice.Cpu, null);

    /// <inheritdoc/>
    public MemorySpace MemorySpace => MemorySpace.Host;

    /// <summary>
    /// This backend: the runtime of the framework's own managed memory, which is the runtime
    /// <see cref="MemoryLocation.IsManaged"/> recognises and every host-memory backend can read.
    /// </summary>
    public object RuntimeIdentity => this;

    /// <summary>
    /// Any host memory, whichever runtime allocated it. This backend reads nothing itself — it runs
    /// nothing — so what the question asks of it is whether a tensor already is plain host memory,
    /// and one a CPU session produced is: <c>To(ComputeContext.Host)</c> hands such a tensor back as
    /// it is, exactly as <c>ToHost()</c> does, and copies only what the host cannot read.
    /// </summary>
    public bool CanAddress(MemoryLocation location) => location.Space.IsHost;

    /// <summary>Always throws: host memory holds tensors and computes nothing.</summary>
    /// <exception cref="NotSupportedException">Always.</exception>
    public IShorokooSession CreateSession(
        ReadOnlyMemory<byte> modelBytes,
        ShorokooGraphOptimization graphOptimization,
        ShorokooLogSeverity logSeverity,
        DeviceMemorySettings deviceMemory)
        => throw new NotSupportedException(
            "The framework's host memory holds tensors and runs nothing, so it cannot build a "
            + "session. Compile and run on a real backend -- a platform package "
            + "(Shorokoo.LinuxCPU, Shorokoo.LinuxGPU, Shorokoo.WinCPU, Shorokoo.WinGPU), or one "
            + "from IsolatedBackend.Load -- and give a ComputeContext that backend.");

    /// <summary>Always throws: there is no runtime here to build a value with.</summary>
    /// <exception cref="NotSupportedException">Always.</exception>
    public IShorokooTensorValue CreateTensor<T>(T[] data, long[] shape) where T : unmanaged
        => throw NoValues();

    /// <summary>Always throws: there is no runtime here to build a value with.</summary>
    /// <exception cref="NotSupportedException">Always.</exception>
    public IShorokooTensorValue CreateTensorFromRawBytes(
        ShorokooTensorElementType elementType, byte[] data, long[] shape)
        => throw NoValues();

    /// <summary>Always throws: there is no runtime here to build a value with.</summary>
    /// <exception cref="NotSupportedException">Always.</exception>
    public IShorokooTensorValue CreateStringTensor(IReadOnlyList<string> data, long[] shape)
        => throw NoValues();

    /// <summary>Always throws: there is no runtime here to build a value with. Whatever it is
    /// handed is disposed first, as a failure to build a sequence must.</summary>
    /// <exception cref="NotSupportedException">Always.</exception>
    public IShorokooTensorValue CreateSequence(IReadOnlyList<IShorokooTensorValue> values)
    {
        foreach (var value in values) value.Dispose();
        throw NoValues();
    }

    /// <summary>Always throws, for the reason the others do; overridden rather than left to the
    /// interface default, which would allocate through a member that throws anyway and say so in
    /// worse words.</summary>
    /// <exception cref="NotSupportedException">Always.</exception>
    public IShorokooTensorValue CreateUninitializedTensorInBackendMemory(
        ShorokooTensorElementType elementType, long[] shape)
        => throw NoValues();

    // One refusal, because there is one reason. A tensor of the framework's own host memory is
    // managed bytes and nothing else; the runtime value is built when a backend is fed it, by
    // that backend, and there is no sense in which this one could build a value another runtime
    // would accept.
    private static NotSupportedException NoValues()
        => new("The framework's host memory holds tensors as managed bytes and builds no runtime "
            + "values: one belongs to the runtime that made it, and there is no runtime here. A "
            + "tensor becomes a value when a real backend is fed it. If you meant to read this "
            + "tensor's contents, the accessors on TensorData do that without any backend.");
}

/// <summary>
/// The allocating backend of a tensor wrapped around a runtime value whose producer was not named —
/// <c>TensorData.Create(shape, dtype, value)</c>, <c>new OnnxTensorData&lt;T&gt;(shape, value)</c>
/// and their siblings.
///
/// <para>Such a tensor still has to be released somehow and asked where it is, so it gets a
/// backend that answers honestly: its memory is released by disposing the value, which is the one
/// release anything can ask of a value; it is host memory if the value says so and somewhere
/// unnamed otherwise; and no runtime is handed it as it stands, since nothing knows which runtime
/// made it. A host-readable one is read where it is by the host — <c>ToHost()</c> and
/// <c>To(ComputeContext.Host)</c> hand it back as it is — and through a copy by a run, or by a
/// context whose backend runs. One that is not host-readable cannot be copied at all: moving it
/// is refused, and so is a run fed it, in words that name the fix.</para>
/// </summary>
internal sealed class UnrecordedBackend : IShorokooBackend
{
    /// <summary>The one such backend: there is nothing to tell two unrecorded producers
    /// apart by.</summary>
    internal static UnrecordedBackend Instance { get; } = new();

    private UnrecordedBackend()
    {
    }

    /// <inheritdoc/>
    public BackendDescription Description { get; }
        = new("an unrecorded backend", ComputeDevice.Other, null);

    /// <summary>
    /// The contents of a host-readable value, and a refusal for any other: the copy out of a
    /// provider's own memory is the producing backend's to make, and which backend that was is the
    /// one thing this does not know.
    /// </summary>
    /// <exception cref="InvalidOperationException">The value is not host-readable.</exception>
    public byte[] CopyTensorToHost(IShorokooTensorValue value)
    {
        ArgumentNullException.ThrowIfNull(value);
        if (!value.IsHostAccessible)
            throw new InvalidOperationException(
                "This tensor's value is in an execution provider's own memory, and the backend that "
                + "made it was not recorded, so there is no backend to ask for a copy of it. Wrap a "
                + "runtime value with TensorData.Create(shape, dtype, value, backend) to say which "
                + "backend made it.");
        var bytes = value.GetTensorDataAsSpan<byte>().ToArray();
        GC.KeepAlive(value);
        return bytes;
    }

    /// <summary>Always throws: a producer nobody named builds nothing.</summary>
    /// <exception cref="NotSupportedException">Always.</exception>
    public IShorokooSession CreateSession(
        ReadOnlyMemory<byte> modelBytes,
        ShorokooGraphOptimization graphOptimization,
        ShorokooLogSeverity logSeverity,
        DeviceMemorySettings deviceMemory)
        => throw Unnamed();

    /// <summary>Always throws: a producer nobody named builds nothing.</summary>
    /// <exception cref="NotSupportedException">Always.</exception>
    public IShorokooTensorValue CreateTensor<T>(T[] data, long[] shape) where T : unmanaged
        => throw Unnamed();

    /// <summary>Always throws: a producer nobody named builds nothing.</summary>
    /// <exception cref="NotSupportedException">Always.</exception>
    public IShorokooTensorValue CreateTensorFromRawBytes(
        ShorokooTensorElementType elementType, byte[] data, long[] shape)
        => throw Unnamed();

    /// <summary>Always throws: a producer nobody named builds nothing.</summary>
    /// <exception cref="NotSupportedException">Always.</exception>
    public IShorokooTensorValue CreateStringTensor(IReadOnlyList<string> data, long[] shape)
        => throw Unnamed();

    /// <summary>Always throws: a producer nobody named builds nothing. Whatever it is handed is
    /// disposed first, as a failure to build a sequence must.</summary>
    /// <exception cref="NotSupportedException">Always.</exception>
    public IShorokooTensorValue CreateSequence(IReadOnlyList<IShorokooTensorValue> values)
    {
        foreach (var value in values) value.Dispose();
        throw Unnamed();
    }

    private static NotSupportedException Unnamed()
        => new("This stands for the backend that made a runtime value which was wrapped without "
            + "saying which backend that was, so it has nothing to build with. Name the backend.");
}

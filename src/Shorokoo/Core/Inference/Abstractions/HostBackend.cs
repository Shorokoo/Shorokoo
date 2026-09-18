using System;
using System.Collections.Generic;

namespace Shorokoo.Core.Inference.Abstractions;

/// <summary>
/// The framework's own host memory, as a backend: what
/// <see cref="Shorokoo.Runtime.ComputeContext.Host"/> allocates in, and what a tensor belonging to
/// nobody in particular has always been made of.
///
/// <para>It holds tensors and builds nothing. Every member throws, and that is the whole of it:
/// there is no runtime here to build a session with, and no runtime value to build either,
/// because the framework's host memory is a managed <c>byte[]</c> that becomes a runtime value
/// only when a real backend is fed it. Manufacturing values here would put back the thing the
/// backend-free literal exists to remove — a native allocation behind every tensor a graph is
/// described with — so this backend is a home for tensors, not a factory for values.</para>
///
/// <para>Attached to <see cref="MemoryDevice.For"/> of <see cref="MemorySpace.Host"/>, alongside
/// every CPU backend the process loads: they share the memory, which is what lets a tensor pass
/// between a host context and a CPU one without a copy.</para>
/// </summary>
public sealed class HostBackend : IShorokooInferenceBackend
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

    /// <summary>Always throws: host memory holds tensors and computes nothing.</summary>
    /// <exception cref="InvalidOperationException">Always.</exception>
    public IShorokooInferenceSession CreateSession(
        ReadOnlyMemory<byte> modelBytes,
        ShorokooGraphOptimization graphOptimization,
        ShorokooLogSeverity logSeverity,
        DeviceMemorySettings deviceMemory)
        => throw new InvalidOperationException(
            "The framework's host memory holds tensors and runs nothing, so it cannot build an "
            + "inference session. Compile and run on a real backend -- a platform package "
            + "(Shorokoo.LinuxCPU, Shorokoo.LinuxGPU, Shorokoo.WinCPU, Shorokoo.WinGPU), or one "
            + "from IsolatedBackend.Load -- and give a ComputeContext that backend.");

    /// <summary>Always throws: there is no runtime here to build a value with.</summary>
    /// <exception cref="InvalidOperationException">Always.</exception>
    public IShorokooTensorValue CreateTensor<T>(T[] data, long[] shape) where T : unmanaged
        => throw NoValues();

    /// <summary>Always throws: there is no runtime here to build a value with.</summary>
    /// <exception cref="InvalidOperationException">Always.</exception>
    public IShorokooTensorValue CreateTensorFromRawBytes(
        ShorokooTensorElementType elementType, byte[] data, long[] shape)
        => throw NoValues();

    /// <summary>Always throws: there is no runtime here to build a value with.</summary>
    /// <exception cref="InvalidOperationException">Always.</exception>
    public IShorokooTensorValue CreateStringTensor(IReadOnlyList<string> data, long[] shape)
        => throw NoValues();

    /// <summary>Always throws: there is no runtime here to build a value with.</summary>
    /// <exception cref="InvalidOperationException">Always.</exception>
    public IShorokooTensorValue CreateSequence(IReadOnlyList<IShorokooTensorValue> values)
        => throw NoValues();

    /// <summary>Always throws, for the reason the others do; overridden rather than left to the
    /// interface default, which would allocate through a member that throws anyway and say so in
    /// worse words.</summary>
    /// <exception cref="InvalidOperationException">Always.</exception>
    public IShorokooTensorValue CreateUninitializedTensorInBackendMemory(
        ShorokooTensorElementType elementType, long[] shape)
        => throw NoValues();

    // One refusal, because there is one reason. A tensor of the framework's own host memory is
    // managed bytes and nothing else; the runtime value is built when a backend is fed it, by
    // that backend, and there is no sense in which this one could build a value another runtime
    // would accept.
    private static InvalidOperationException NoValues()
        => new("The framework's host memory holds tensors as managed bytes and builds no runtime "
            + "values: one belongs to the runtime that made it, and there is no runtime here. A "
            + "tensor becomes a value when a real backend is fed it. If you meant to read this "
            + "tensor's contents, the accessors on TensorData do that without any backend.");
}

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace Shorokoo.Core.Inference.Abstractions;

/// <summary>
/// The framework's own host memory, as a backend: what
/// <see cref="Shorokoo.Runtime.ComputeContext.Host"/> allocates in, and what a tensor belonging to
/// nobody in particular has always been made of.
///
/// <para>It holds tensors and runs nothing. <see cref="CreateSession"/> throws, because there is
/// no runtime here to build one — plain managed bytes are what a graph's literals are described
/// with, and describing a graph is precisely the thing that must not need a deployed inference
/// runtime. Everything else builds a value in ordinary managed memory, which is what the
/// framework's host tensors have always been.</para>
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

    /// <inheritdoc/>
    public IShorokooTensorValue CreateTensor<T>(T[] data, long[] shape) where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(shape);
        return new HostTensorValue(
            ElementTypeOf<T>(), MemoryMarshal.AsBytes(data.AsSpan()).ToArray(), shape);
    }

    /// <inheritdoc/>
    public IShorokooTensorValue CreateTensorFromRawBytes(
        ShorokooTensorElementType elementType, byte[] data, long[] shape)
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(shape);
        if (elementType == ShorokooTensorElementType.String)
            throw new NotSupportedException(
                "String elements are variable-length, so a flat byte buffer does not describe "
                + "them. Use CreateStringTensor.");
        return new HostTensorValue(elementType, data, shape);
    }

    /// <inheritdoc/>
    public IShorokooTensorValue CreateStringTensor(IReadOnlyList<string> data, long[] shape)
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(shape);
        return new HostTensorValue([.. data], shape);
    }

    /// <inheritdoc/>
    public IShorokooTensorValue CreateSequence(IReadOnlyList<IShorokooTensorValue> values)
    {
        ArgumentNullException.ThrowIfNull(values);
        return new HostTensorValue([.. values]);
    }

    private static ShorokooTensorElementType ElementTypeOf<T>() where T : unmanaged
    {
        if (typeof(T) == typeof(float)) return ShorokooTensorElementType.Float;
        if (typeof(T) == typeof(double)) return ShorokooTensorElementType.Double;
        if (typeof(T) == typeof(bool)) return ShorokooTensorElementType.Bool;
        if (typeof(T) == typeof(sbyte)) return ShorokooTensorElementType.Int8;
        if (typeof(T) == typeof(byte)) return ShorokooTensorElementType.UInt8;
        if (typeof(T) == typeof(short)) return ShorokooTensorElementType.Int16;
        if (typeof(T) == typeof(ushort)) return ShorokooTensorElementType.UInt16;
        if (typeof(T) == typeof(int)) return ShorokooTensorElementType.Int32;
        if (typeof(T) == typeof(uint)) return ShorokooTensorElementType.UInt32;
        if (typeof(T) == typeof(long)) return ShorokooTensorElementType.Int64;
        if (typeof(T) == typeof(ulong)) return ShorokooTensorElementType.UInt64;
        if (typeof(T) == typeof(Float16)) return ShorokooTensorElementType.Float16;
        if (typeof(T) == typeof(BFloat16)) return ShorokooTensorElementType.BFloat16;
        throw new NotSupportedException(
            $"Host memory has no element type for {typeof(T).Name}.");
    }
}

/// <summary>
/// A tensor value in the framework's own managed memory — no native allocation, no runtime, and
/// nothing to release. It is what <see cref="HostBackend"/> builds, and the one implementation of
/// <see cref="IShorokooTensorValue"/> that needs no inference runtime deployed.
/// </summary>
internal sealed class HostTensorValue : IShorokooTensorValue
{
    private readonly byte[]? _bytes;
    private readonly string[]? _strings;
    private readonly IShorokooTensorValue[]? _elements;

    internal HostTensorValue(ShorokooTensorElementType elementType, byte[] bytes, long[] shape)
    {
        ElementType = elementType;
        Shape = shape;
        _bytes = bytes;
    }

    internal HostTensorValue(string[] strings, long[] shape)
    {
        ElementType = ShorokooTensorElementType.String;
        Shape = shape;
        _strings = strings;
    }

    internal HostTensorValue(IShorokooTensorValue[] elements)
    {
        ElementType = elements.Length > 0
            ? elements[0].ElementType : ShorokooTensorElementType.Float;
        Shape = [];
        _elements = elements;
    }

    /// <inheritdoc/>
    public ShorokooOnnxValueType ValueType
        => _elements is null ? ShorokooOnnxValueType.Tensor : ShorokooOnnxValueType.Sequence;

    /// <inheritdoc/>
    public ShorokooTensorElementType ElementType { get; }

    /// <inheritdoc/>
    public long[] Shape { get; }

    /// <inheritdoc/>
    public ReadOnlySpan<T> GetTensorDataAsSpan<T>() where T : unmanaged
        => MemoryMarshal.Cast<byte, T>(Bytes);

    /// <inheritdoc/>
    public Span<T> GetTensorMutableDataAsSpan<T>() where T : unmanaged
        => MemoryMarshal.Cast<byte, T>(Bytes.AsSpan());

    /// <inheritdoc/>
    public IReadOnlyList<string> GetStringTensorData()
        => _strings ?? throw new InvalidOperationException(
            "This host value does not hold strings.");

    /// <inheritdoc/>
    public int GetValueCount() => Elements.Length;

    /// <inheritdoc/>
    public IShorokooTensorValue GetValue(int index) => Elements[index];

    /// <inheritdoc/>
    public ShorokooTensorElementType GetSequenceElementType() => ElementType;

    /// <summary>Releases the elements a sequence was handed, and nothing else: a tensor here is
    /// managed bytes, which the collector reclaims.</summary>
    public void Dispose()
    {
        if (_elements is null) return;
        foreach (var element in _elements) element.Dispose();
    }

    private byte[] Bytes
        => _bytes ?? throw new InvalidOperationException(
            _strings is not null
                ? "A string tensor's elements are variable-length and reference-typed, so there "
                  + "is no flat buffer to span over. Read them with GetStringTensorData()."
                : "A sequence has no flat buffer to span over. Read its elements with "
                  + "GetValue(index).");

    private IShorokooTensorValue[] Elements
        => _elements ?? throw new InvalidOperationException("This host value is not a sequence.");
}

using Python.Runtime;
using Shorokoo.Core.Backends;
using Shorokoo.PythonHost;

namespace Shorokoo.Jax;

/// <summary>
/// A value of the JAX backend: a tensor, held as a numpy array in host memory or as a JAX array in
/// a device's memory. JAX has no string tensors and no sequences, so a JAX value is never either.
///
/// <para>What a value is — its element type, shape, and for a host tensor the address and size of
/// its buffer — is read once, when it is made, and kept: none of it can change, and reading it
/// again would take the interpreter lock for every question.</para>
///
/// <para><b>The span accessors read the array's own memory.</b> A host value is always a numpy array
/// of its own — contiguous, writable, no other value's memory — and it lives for as long as this
/// value holds its reference, so a span stays valid until the value is released, exactly as an
/// ONNX Runtime value's does. No lock is needed to read it: the memory is numpy's, not the
/// interpreter's.</para>
/// </summary>
public sealed class JaxTensorValue : IShorokooTensorValue
{
    private readonly PyObject _value;
    private readonly long[] _shape;
    private readonly IntPtr _address;
    private readonly long _byteCount;
    private readonly bool _isHost;
    private int _released;

    private JaxTensorValue(PyObject value, ShorokooTensorElementType elementType, long[] shape, bool isHost, IntPtr address, long byteCount)
    {
        _value = value;
        ElementType = elementType;
        _shape = shape;
        _isHost = isHost;
        _address = address;
        _byteCount = byteCount;
    }

    /// <summary>Wraps <paramref name="value"/>, taking over the reference, and reads what it is.
    /// Called holding the interpreter lock.</summary>
    internal static JaxTensorValue Wrap(JaxRuntime runtime, PyObject value)
    {
        using var description = runtime.Describe.Invoke(value);
        return Wrap(value, description);
    }

    /// <summary>Wraps <paramref name="value"/> by a description <c>runtime.describe</c> already made
    /// of it. Called holding the interpreter lock.</summary>
    internal static JaxTensorValue Wrap(PyObject value, PyObject description)
    {
        var code = Item<int>(description, 1);
        using var dims = description[2];
        var shape = new long[(int)dims.Length()];
        for (int i = 0; i < shape.Length; i++) shape[i] = Item<long>(dims, i);
        return new JaxTensorValue(value, (ShorokooTensorElementType)code, shape,
            Item<bool>(description, 3), new IntPtr(Item<long>(description, 4)), Item<long>(description, 5));
    }

    private static T Item<T>(PyObject sequence, int index)
    {
        using var item = sequence[index];
        return item.As<T>();
    }

    /// <summary>The Python object this wraps, refused once released.</summary>
    internal PyObject Value => Volatile.Read(ref _released) == 0 ? _value : throw Released();

    private static ObjectDisposedException Released() => new(
        nameof(JaxTensorValue),
        "This JAX value has been released -- the tensor it belonged to was deleted, or consumed by a run "
        + "whose backend released it, or it was disposed -- so nothing may read it through this handle.");

    public ShorokooOnnxValueType ValueType => ShorokooOnnxValueType.Tensor;

    public ShorokooTensorElementType ElementType { get; }

    public long[] Shape => [.. _shape];

    public bool IsHostAccessible
    {
        get
        {
            if (Volatile.Read(ref _released) != 0) throw Released();
            return _isHost;
        }
    }

    private void ThrowIfNotReadable<T>() where T : unmanaged
    {
        if (!IsHostAccessible)
            throw new InvalidOperationException(
                "This tensor is in a device's memory, not host memory, so it cannot be read directly. "
                + "CopyTensorToHost brings it home.");
        if (_byteCount % System.Runtime.CompilerServices.Unsafe.SizeOf<T>() != 0)
            throw new InvalidOperationException(
                $"A {ElementType} tensor of {_byteCount} bytes cannot be read as {typeof(T).Name}.");
    }

    public unsafe ReadOnlySpan<T> GetTensorDataAsSpan<T>() where T : unmanaged
    {
        ThrowIfNotReadable<T>();
        return new ReadOnlySpan<T>((void*)_address, checked((int)(_byteCount / sizeof(T))));
    }

    public unsafe Span<T> GetTensorMutableDataAsSpan<T>() where T : unmanaged
    {
        ThrowIfNotReadable<T>();
        return new Span<T>((void*)_address, checked((int)(_byteCount / sizeof(T))));
    }

    public IReadOnlyList<string> GetStringTensorData()
        => throw new InvalidOperationException($"This is a {ElementType} tensor; JAX has no string tensors.");

    public int GetValueCount()
        => throw new InvalidOperationException("This value holds a Tensor, not a sequence; JAX has no sequences.");

    public IShorokooTensorValue GetValue(int index)
        => throw new InvalidOperationException("This value holds a Tensor, not a sequence; JAX has no sequences.");

    public ShorokooTensorElementType GetSequenceElementType()
        => throw new InvalidOperationException("This value holds a Tensor, not a sequence; JAX has no sequences.");

    /// <summary>Drops this value's reference, once: a second call does nothing, and every read
    /// afterwards is refused.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _released, 1) != 0) return;
        PythonRuntime.Release(_value);
    }
}
